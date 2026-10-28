#!/usr/bin/env python3
"""SF-Net adapted single-frame supervision for sensor TAL.

This baseline keeps the core SF-Net training recipe but adapts the expansion
step to long sensor sequences while keeping our paper-clean TAL protocol:

1. sample one Gaussian timestamp per action segment with the same seed as PSEE;
2. train a sensor localizer with SF-Net losses: video top-k MIL, point-frame
   classification, and actionness hard-negative mining;
3. expand each point on class-action support inside neighboring point bounds;
4. convert each expanded point component to one pseudo segment; and
5. train/evaluate the same SlimSTAD-1D detector config used by PSEE.

It intentionally does not use PSEE candidate ranking, transition scoring,
duration priors, or reliability-weighted detector loss. The detector stage is
unchanged so this is a point-supervision module baseline rather than a
detector/backbone confound.
"""

from __future__ import annotations

import os

# Runtime-only speed guard for multi-fold queues on CPU-limited servers.
# It does not change data, losses, model, seed, or evaluation protocol.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("SLIMSTAD_TORCH_NUM_THREADS", "1")
os.environ.setdefault("SLIMSTAD_DETECTOR_NUM_WORKERS", "2")
os.environ.setdefault("SLIMSTAD_EVAL_N_JOBS", "1")

import argparse
import copy
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from run_point_only_classifier_baseline import (
    PointWindowDataset,
    collate,
    import_wear_second_encoder,
    load_annotations,
    load_feature,
    load_info,
    make_gaussian_points,
    normalize_feature,
    segment_iou,
    set_seed,
)


PROJECT_ROOT = Path("/home/yinjiaxi/SlimSTAD/STAD-main")
DETR_ROOT = Path("/home/yinjiaxi/detr_e2e_tad")
PYTHON = Path("/home/yinjiaxi/miniconda3/envs/tal_for_har/bin/python")
TAG = "sfnet_adapt_imu_slimstad1d_clean_20260902"
METHOD_NAME = "SFNet-Adapt-2S"

FEATURE_ROOTS = {
    "wear": Path("/data1/yinjiaxi/wear/processed/inertial_features/50_samples_50_overlap"),
    "sbhar": Path("/data2/yinjiaxi/imwut24_tal4har_data/sbhar/processed/inertial_features/50_samples_50_overlap"),
    "wetlab": Path("/data2/yinjiaxi/imwut24_tal4har_data/wetlab/processed/inertial_features/50_samples_50_overlap"),
    "rwhar": Path("/data2/yinjiaxi/imwut24_tal4har_data/rwhar/processed/inertial_features/50_samples_50_overlap"),
    "xrfv2": Path("/data2/yinjiaxi/xrfv2-multimodal-tal-caption-qa-no-rgb/processed/slimstad_imu30"),
}

ANNOTATION_ROOTS = {
    "wear": PROJECT_ROOT / "wear_annotations_strict_map",
    "sbhar": PROJECT_ROOT / "tal4har_annotations_psee_slimstad1d",
    "wetlab": PROJECT_ROOT / "tal4har_annotations_psee_slimstad1d",
    "rwhar": PROJECT_ROOT / "tal4har_annotations_psee_slimstad1d",
    "xrfv2": PROJECT_ROOT / "xrfv2_annotations_weak_v2",
}

SPLIT_COUNTS = {"wear": 18, "sbhar": 30, "wetlab": 22, "rwhar": 15, "xrfv2": 16}


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    mkdir(path.parent)
    fields = ["video", "type", "type_idx", "start", "end", "startFrame", "endFrame"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def class_count_from_annotations(*annotation_lists: Sequence[dict]) -> int:
    max_label = 0
    for annotations in annotation_lists:
        for ann in annotations:
            max_label = max(max_label, int(ann["label"]))
    return max_label + 1


def group_annotations(annotations: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for ann in annotations:
        grouped[ann["video"]].append(ann)
    return grouped


def first_existing(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Missing expected file; tried: " + ", ".join(str(path) for path in candidates))


def split_paths(annotation_root: Path, dataset: str, split: int, split_dir: Path = None) -> Dict[str, Path]:
    if split_dir is None:
        if dataset == "wear":
            split_dir = annotation_root / f"outer_{split}_val_{(split + 1) % 18}_{(split + 2) % 18}"
        elif dataset == "xrfv2":
            split_dir = annotation_root / "imu30" / f"loso{split}"
        else:
            split_dir = annotation_root / dataset / f"loso{split}"
    prefix = f"_{dataset}" if dataset == "wear" else ""
    return {
        "dir": split_dir,
        "train_info": first_existing([split_dir / f"train{prefix}_info_C.csv", split_dir / "train_info_C.csv"]),
        "val_info": first_existing([split_dir / f"validation{prefix}_info_C.csv", split_dir / "validation_info_C.csv"]),
        "test_info": first_existing([split_dir / f"test{prefix}_info_C.csv", split_dir / "test_info_C.csv"]),
        "train_ann": first_existing([split_dir / f"train{prefix}_annotation.csv", split_dir / "train_annotation.csv"]),
        "val_ann": first_existing([split_dir / f"validation{prefix}_annotation.csv", split_dir / "validation_annotation.csv"]),
        "test_ann": first_existing([split_dir / f"test{prefix}_annotation.csv", split_dir / "test_annotation.csv"]),
    }


def frame_from_coordinate(value: float, video_info: dict) -> float:
    """Inverse of the STAD loader's startFrame -> sampled-coordinate mapping."""
    count = float(video_info.get("count", 0.0))
    sample_count = float(video_info.get("sample_count", 0.0))
    if count <= 0 or sample_count <= 0:
        return float(value)
    return float(value) * count / sample_count


def align_info_to_detector_coordinate(info: Dict[str, dict], feature_root: Path) -> Dict[str, dict]:
    """Use sampled feature coordinates, matching the annotation CSV start/end."""
    aligned = {}
    for video, row in info.items():
        item = dict(row)
        sample_count = float(item.get("sample_count") or 0.0)
        if sample_count <= 0:
            sample_count = float(load_feature(feature_root, video).shape[0])
        if sample_count > 0:
            item["duration"] = sample_count
        aligned[video] = item
    return aligned


def time_to_index(value: float, duration: float, out_len: int) -> int:
    idx = int(round(float(value) / max(float(duration), 1e-6) * max(out_len - 1, 1)))
    return max(0, min(out_len - 1, idx))


def index_to_time(index: int, duration: float, out_len: int) -> float:
    return float(index + 0.5) / max(float(out_len), 1.0) * float(duration)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class SensorSFNetLocalizer(nn.Module):
    """Single-stream sensor analogue of SF-Net's class/actionness network."""

    def __init__(
        self,
        wear_encoder_cls,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        downsample: int,
    ) -> None:
        super().__init__()
        self.wear_encoder = wear_encoder_cls(hidden_dim) if input_dim == 600 else None
        stem_dim = hidden_dim if self.wear_encoder is not None else input_dim
        layers: List[nn.Module] = [
            nn.Conv1d(stem_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        current_downsample = 1
        while current_downsample < downsample:
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            current_downsample *= 2
        layers.extend([ResidualBlock(hidden_dim, 1), ResidualBlock(hidden_dim, 2), ResidualBlock(hidden_dim, 4)])
        self.encoder = nn.Sequential(*layers)
        self.class_head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.action_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.wear_encoder is not None:
            feat = self.wear_encoder(feat)
        x = self.encoder(feat.transpose(1, 2))
        return self.class_head(x), self.action_head(x).squeeze(1)


def point_output_indices(points: Sequence[dict], input_len: int, out_len: int) -> List[Tuple[int, int]]:
    result = []
    for point in points:
        label = int(point["label"])
        out_idx = int(round(float(point["local_step"]) / max(float(input_len), 1.0) * out_len))
        result.append((max(0, min(out_len - 1, out_idx)), label))
    return result


def frame_loss(logits: torch.Tensor, points: Sequence[List[dict]], input_len: int, radius: int) -> torch.Tensor:
    losses = []
    _, _, out_len = logits.shape
    for batch_idx, batch_points in enumerate(points):
        for out_idx, label in point_output_indices(batch_points, input_len, out_len):
            lo = max(0, out_idx - radius)
            hi = min(out_len, out_idx + radius + 1)
            target = torch.full((hi - lo,), label, dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits[batch_idx, :, lo:hi].transpose(0, 1), target, reduction="mean"))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def video_topk_loss(logits: torch.Tensor, points: Sequence[List[dict]], num_classes: int) -> torch.Tensor:
    labels = torch.zeros((logits.shape[0], num_classes), dtype=logits.dtype, device=logits.device)
    for batch_idx, batch_points in enumerate(points):
        for point in batch_points:
            labels[batch_idx, int(point["label"])] = 1.0
    valid = labels.sum(dim=1) > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    k = max(1, int(np.ceil(logits.shape[-1] / 8.0)))
    topk = torch.topk(logits, k=k, dim=-1).values.mean(dim=-1)
    targets = labels[valid]
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(targets * F.log_softmax(topk[valid], dim=1)).sum(dim=1).mean()


def actionness_loss(action_logits: torch.Tensor, points: Sequence[List[dict]], input_len: int, tm: float) -> torch.Tensor:
    losses = []
    out_len = action_logits.shape[-1]
    for batch_idx, batch_points in enumerate(points):
        positives = sorted({idx for idx, _ in point_output_indices(batch_points, input_len, out_len)})
        if not positives:
            continue
        pos_logits = action_logits[batch_idx, positives]
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_mask = torch.ones(out_len, dtype=torch.bool, device=action_logits.device)
        neg_mask[positives] = False
        neg_logits = action_logits[batch_idx, neg_mask]
        if neg_logits.numel() > 0:
            k = min(max(1, int(round(len(positives) * tm))), int(neg_logits.numel()))
            hard_neg = torch.topk(neg_logits, k=k).values
            neg_loss = F.binary_cross_entropy_with_logits(hard_neg, torch.zeros_like(hard_neg))
            losses.append(pos_loss + neg_loss)
        else:
            losses.append(pos_loss)
    return torch.stack(losses).mean() if losses else action_logits.sum() * 0.0


def train_sfnet_localizer(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    num_classes: int,
    epochs: int,
    phase: str,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        sums = defaultdict(float)
        steps = 0
        for batch in loader:
            feat = batch["feat"].to(args.device)
            class_logits, action_logits = model(feat)
            fl = frame_loss(class_logits, batch["points"], args.classifier_window, args.point_radius)
            vl = video_topk_loss(class_logits, batch["points"], num_classes)
            al = actionness_loss(action_logits, batch["points"], args.classifier_window, args.tm)
            loss = fl + args.alpha * vl + args.beta * al
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            sums["frame"] += float(fl.detach().cpu())
            sums["video"] += float(vl.detach().cpu())
            sums["action"] += float(al.detach().cpu())
            sums["total"] += float(loss.detach().cpu())
            steps += 1
        print(
            f"[{phase} epoch {epoch:03d}/{epochs:03d}] total={sums['total']/max(steps,1):.4f} "
            f"frame={sums['frame']/max(steps,1):.4f} video={sums['video']/max(steps,1):.4f} "
            f"action={sums['action']/max(steps,1):.4f}",
            flush=True,
        )


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    feat: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    tensor = torch.from_numpy(normalize_feature(feat)).unsqueeze(0).to(args.device)
    class_logits, action_logits = model(tensor)
    if args.sf_score_mode == "raw":
        class_scores = class_logits[0].detach().cpu().numpy()
    elif args.sf_score_mode == "sigmoid":
        class_scores = torch.sigmoid(class_logits[0]).detach().cpu().numpy()
    else:
        class_scores = torch.softmax(class_logits[0], dim=0).detach().cpu().numpy()
    action_scores = torch.sigmoid(action_logits[0]).detach().cpu().numpy()
    return class_scores.astype(np.float32), action_scores.astype(np.float32)


def initial_frame_labels(video_points: Sequence[dict], duration: float, out_len: int) -> Tuple[List[List[int]], Dict[int, int]]:
    labels: List[List[int]] = [[] for _ in range(out_len)]
    anchor_for_annotation: Dict[int, int] = {}
    for point in video_points:
        idx = time_to_index(float(point["time"]), duration, out_len)
        label = int(point["label"])
        if label not in labels[idx]:
            labels[idx].append(label)
        anchor_for_annotation[int(point["annotation_index"])] = idx
    return labels, anchor_for_annotation


def smooth_signal(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size <= 1:
        return values.astype(np.float32)
    radius = min(radius, max((values.size - 1) // 2, 0))
    if radius <= 0:
        return values.astype(np.float32)
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= float(kernel.sum())
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def point_search_limits(points: Sequence[dict], point_index: int, duration: float) -> Tuple[float, float]:
    """Use only neighboring weak points to avoid crossing into adjacent events."""
    point_time = float(points[point_index]["time"])
    if point_index > 0:
        left = 0.5 * (float(points[point_index - 1]["time"]) + point_time)
    elif len(points) > 1:
        gap = max(float(points[1]["time"]) - point_time, 1e-6)
        left = max(0.0, point_time - 0.5 * gap)
    else:
        left = 0.0
    if point_index + 1 < len(points):
        right = 0.5 * (point_time + float(points[point_index + 1]["time"]))
    elif len(points) > 1:
        gap = max(point_time - float(points[point_index - 1]["time"]), 1e-6)
        right = min(duration, point_time + 0.5 * gap)
    else:
        right = duration
    return max(0.0, min(left, point_time)), min(duration, max(right, point_time))


def label_support(class_scores: np.ndarray, action_scores: np.ndarray, label: int, smooth_radius: int) -> np.ndarray:
    cls = np.clip(class_scores[label].astype(np.float32), 0.0, 1.0)
    act = np.clip(action_scores.astype(np.float32), 0.0, 1.0)
    support = np.sqrt(np.maximum(cls * act, 0.0))
    return smooth_signal(support, smooth_radius)


def otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return 1.0
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-5:
        return vmax + 1e-6
    hist, edges = np.histogram(finite, bins=bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = float(hist.sum())
    sum_total = float((hist * centers).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_thr = float(np.median(finite))
    for count, center in zip(hist[:-1], centers[:-1]):
        weight_bg += float(count)
        if weight_bg <= 0.0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0.0:
            break
        sum_bg += float(count) * float(center)
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_thr = float(center)
    return best_thr


def expand_one_point(
    support: np.ndarray,
    anchor_idx: int,
    left_bound: int,
    right_bound: int,
    point_radius: int,
    pv: float,
    max_gap: int,
) -> Tuple[int, int, float]:
    center_lo = max(left_bound, anchor_idx - max(1, point_radius))
    center_hi = min(right_bound + 1, anchor_idx + max(1, point_radius) + 1)
    anchor_score = float(np.max(support[center_lo:center_hi])) if center_hi > center_lo else float(support[anchor_idx])
    local_threshold = otsu_threshold(support[left_bound : right_bound + 1])
    threshold = max(anchor_score * pv, local_threshold, 1e-6)

    def walk(direction: int) -> int:
        edge = anchor_idx
        missed = 0
        current = anchor_idx + direction
        while left_bound <= current <= right_bound:
            if float(support[current]) >= threshold:
                edge = current
                missed = 0
            else:
                missed += 1
                if missed <= max_gap:
                    edge = current
                else:
                    break
            current += direction
        while edge != anchor_idx and float(support[edge]) < threshold:
            edge -= direction
        return int(edge)

    left = walk(-1)
    right = walk(1)
    if left > anchor_idx:
        left = anchor_idx
    if right < anchor_idx:
        right = anchor_idx
    return left, right, threshold


def sensor_adapt_expand(
    class_scores: np.ndarray,
    action_scores: np.ndarray,
    video_points: Sequence[dict],
    duration: float,
    args: argparse.Namespace,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Sensor-adapted SF-Net expansion over a full record.

    Compared with the original fixed-radius expansion, this follows the same
    class/actionness evidence but allows variable-length contiguous support.
    Search is capped by neighboring weak points, not by ground-truth duration.
    """
    out_len = int(class_scores.shape[1])
    frame_labels: List[List[int]] = [[] for _ in range(out_len)]
    anchor_for_annotation: Dict[int, int] = {}
    sorted_points = sorted(video_points, key=lambda item: float(item["time"]))
    for point_index, point in enumerate(sorted_points):
        label = int(point["label"])
        anchor_idx = time_to_index(float(point["time"]), duration, out_len)
        left_time, right_time = point_search_limits(sorted_points, point_index, duration)
        left_bound = min(anchor_idx, time_to_index(left_time, duration, out_len))
        right_bound = max(anchor_idx, time_to_index(right_time, duration, out_len))
        support = label_support(class_scores, action_scores, label, args.smooth_radius)
        left, right, _ = expand_one_point(
            support,
            anchor_idx,
            left_bound,
            right_bound,
            args.point_radius,
            args.sf_pv,
            args.max_gap,
        )
        for idx in range(left, right + 1):
            if label not in frame_labels[idx]:
                frame_labels[idx].append(label)
        anchor_for_annotation[int(point["annotation_index"])] = anchor_idx
    return frame_labels, anchor_for_annotation


def labels_to_training_points(
    labels_by_video: Dict[str, List[List[int]]],
    info: Dict[str, dict],
) -> Dict[str, List[dict]]:
    points: Dict[str, List[dict]] = defaultdict(list)
    for video, frame_labels in labels_by_video.items():
        duration = max(float(info[video]["duration"]), 1e-6)
        out_len = len(frame_labels)
        for idx, labels in enumerate(frame_labels):
            point_time = index_to_time(idx, duration, out_len)
            for label in labels:
                points[video].append(
                    {
                        "time": point_time,
                        "position": point_time / duration,
                        "label": int(label),
                        "annotation_index": -1,
                        "point_ratio": 0.5,
                        "point_mode": "sfnet_expanded_frame",
                    }
                )
    return points


def connected_component(labels: Sequence[List[int]], anchor_idx: int, label: int) -> Tuple[int, int]:
    if label not in labels[anchor_idx]:
        return anchor_idx, anchor_idx
    lo = anchor_idx
    while lo - 1 >= 0 and label in labels[lo - 1]:
        lo -= 1
    hi = anchor_idx
    while hi + 1 < len(labels) and label in labels[hi + 1]:
        hi += 1
    return lo, hi


@torch.no_grad()
def expand_all_records(
    model: nn.Module,
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    feature_root: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[List[int]]], Dict[str, Dict[int, int]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    labels_by_video: Dict[str, List[List[int]]] = {}
    anchor_by_video: Dict[str, Dict[int, int]] = {}
    class_scores_by_video: Dict[str, np.ndarray] = {}
    action_scores_by_video: Dict[str, np.ndarray] = {}
    for video in sorted(points):
        feat = load_feature(feature_root, video)
        duration = max(float(info[video]["duration"]), 1e-6)
        class_scores, action_scores = predict_scores(model, feat, args)
        labels, anchor_for_annotation = sensor_adapt_expand(class_scores, action_scores, points[video], duration, args)
        labels_by_video[video] = labels
        anchor_by_video[video] = anchor_for_annotation
        class_scores_by_video[video] = class_scores
        action_scores_by_video[video] = action_scores
    return labels_by_video, anchor_by_video, class_scores_by_video, action_scores_by_video


def make_pseudo_segments(
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    labels_by_video: Dict[str, List[List[int]]],
    anchor_by_video: Dict[str, Dict[int, int]],
    class_scores_by_video: Dict[str, np.ndarray],
    action_scores_by_video: Dict[str, np.ndarray],
) -> List[dict]:
    pseudo: List[dict] = []
    for video in sorted(points):
        duration = max(float(info[video]["duration"]), 1e-6)
        frame_labels = labels_by_video[video]
        class_scores = class_scores_by_video[video]
        action_scores = action_scores_by_video[video]
        out_len = len(frame_labels)
        for point in sorted(points[video], key=lambda item: float(item["time"])):
            label = int(point["label"])
            ann_idx = int(point["annotation_index"])
            anchor_idx = anchor_by_video[video].get(ann_idx, time_to_index(float(point["time"]), duration, out_len))
            lo, hi = connected_component(frame_labels, anchor_idx, label)
            start = float(lo) / max(float(out_len), 1.0) * duration
            end = float(hi + 1) / max(float(out_len), 1.0) * duration
            cls_part = class_scores[label, lo : hi + 1]
            act_part = action_scores[lo : hi + 1]
            score = float(np.mean(cls_part) + np.mean(act_part)) if cls_part.size else 0.0
            pseudo.append(
                {
                    "video": video,
                    "label": label,
                    "start": max(0.0, min(start, float(point["time"]) - 1e-3)),
                    "end": min(duration, max(end, float(point["time"]) + 1e-3)),
                    "score": score,
                    "point_time": float(point["time"]),
                    "annotation_index": ann_idx,
                    "point_ratio": float(point.get("point_ratio", 0.5)),
                }
            )
    return pseudo


def pseudo_quality(pseudo: Sequence[dict], train_annotations: Sequence[dict]) -> Dict[str, float]:
    gt_by_video = group_annotations(train_annotations)
    ious, start_err, end_err, ratio_err = [], [], [], []
    for row in pseudo:
        gt_list = gt_by_video.get(row["video"], [])
        ann_idx = int(row.get("annotation_index", -1))
        if ann_idx < 0 or ann_idx >= len(gt_list):
            continue
        gt = gt_list[ann_idx]
        pred = (float(row["start"]), float(row["end"]))
        target = (float(gt["start"]), float(gt["end"]))
        gt_len = max(target[1] - target[0], 1e-6)
        pred_len = max(pred[1] - pred[0], 1e-6)
        ious.append(segment_iou(pred, target))
        start_err.append(abs(pred[0] - target[0]) / gt_len)
        end_err.append(abs(pred[1] - target[1]) / gt_len)
        ratio_err.append(abs(pred_len / gt_len - 1.0))
    if not ious:
        return {
            "pseudo_mean_iou": 0.0,
            "pseudo_median_iou": 0.0,
            "pseudo_r@0.3": 0.0,
            "pseudo_r@0.5": 0.0,
            "pseudo_r@0.7": 0.0,
            "start_error_over_duration": 0.0,
            "end_error_over_duration": 0.0,
            "duration_ratio_error": 0.0,
        }
    arr = np.asarray(ious, dtype=np.float32)
    return {
        "pseudo_mean_iou": float(np.mean(arr)),
        "pseudo_median_iou": float(np.median(arr)),
        "pseudo_r@0.3": float(np.mean(arr >= 0.3)),
        "pseudo_r@0.5": float(np.mean(arr >= 0.5)),
        "pseudo_r@0.7": float(np.mean(arr >= 0.7)),
        "start_error_over_duration": float(np.mean(start_err)),
        "end_error_over_duration": float(np.mean(end_err)),
        "duration_ratio_error": float(np.mean(ratio_err)),
    }


def export_pseudo_csv(
    dataset: str,
    split: int,
    pseudo: Sequence[dict],
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    train_annotations: Sequence[dict],
    pseudo_root: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    gt_by_video = group_annotations(train_annotations)
    out_dir = mkdir(pseudo_root / dataset / f"loso{split}")
    csv_path = out_dir / "train_annotations_sfnet_adapt_000.csv"
    rows = []
    for item in pseudo:
        video = item["video"]
        gt = gt_by_video[video][int(item["annotation_index"])]
        start = max(0.0, float(item["start"]))
        end = max(start + 1e-3, float(item["end"]))
        rows.append(
            {
                "video": video,
                "type": gt.get("type", str(int(item["label"]))),
                "type_idx": int(item["label"]),
                "start": start,
                "end": end,
                "startFrame": frame_from_coordinate(start, info[video]),
                "endFrame": frame_from_coordinate(end, info[video]),
            }
        )
    write_csv(csv_path, rows)
    quality = {
        "dataset": dataset,
        "split": split,
        "method": METHOD_NAME,
        "stage": "sfnet_adapt_class_actionness_expansion_pseudo_boundary_generation",
        "point_sampling": "gaussian",
        "point_seed": args.point_seed,
        "pseudo_count": len(rows),
        "train_point_count": int(sum(len(v) for v in points.values())),
        "score_mode": args.sf_score_mode,
        "expansion": "variable_length_class_actionness_support",
        "search_limit": "midpoint_between_neighboring_weak_points",
        "threshold": "max(anchor_score * pv, local_otsu_threshold)",
        "smooth_radius": args.smooth_radius,
        "max_gap": args.max_gap,
        "sf_pv": args.sf_pv,
        "tm": args.tm,
        "no_gt_duration_prior": True,
        "reliability_weighted_detector": False,
        **pseudo_quality(pseudo, train_annotations),
    }
    meta_path = out_dir / "sfnet_adapt_pseudo_quality.json"
    write_json(meta_path, quality)
    return csv_path, meta_path


def find_template_config(dataset: str, split: int, project_root: Path) -> Path:
    if dataset == "wear":
        candidates = [
            project_root
            / "configs"
            / "psee_clean_20260715_point_ablation"
            / f"psee_clean_20260715_point_gaussian_wear_loso{split}_b000.yaml",
            project_root
            / "configs"
            / "psee_gaussian_slimstad1d_clean_yjx164_20260720"
            / f"psee_gaussian_slimstad1d_clean_yjx164_20260720_wear_loso{split}_b000.yaml",
        ]
    elif dataset == "xrfv2":
        candidates = [
            project_root
            / "configs"
            / "psee_clean_20260715_point_ablation"
            / f"psee_clean_20260715_point_gaussian_xrfv2_imu30_loso{split}_b000.yaml",
            project_root
            / "configs"
            / "psee_v2_slimstad1d"
            / "slimstad1d_psee_v2_gaussian_2020"
            / f"slimstad1d_psee_v2_gaussian_2020__xrfv2_imu30_loso{split}_b000.yaml",
        ]
    else:
        candidates = []
    for path in candidates:
        if path.exists():
            return path
    pattern = f"**/*{dataset}_loso{split}_b000.yaml"
    matches = [
        p
        for p in (project_root / "configs").glob(pattern)
        if "gaussian" in p.name and ("slimstad1d" in str(p) or "psee_clean" in str(p))
    ]
    if not matches:
        raise FileNotFoundError(f"No SlimSTAD1D template config for {dataset} loso{split}")
    return sorted(matches)[0]


def build_detector_config(dataset: str, split: int, pseudo_csv: Path, args: argparse.Namespace) -> Path:
    template = find_template_config(dataset, split, args.project_root)
    cfg = copy.deepcopy(yaml.safe_load(template.read_text(encoding="utf-8")))
    run_name = f"{TAG}_{dataset}_loso{split}_b000"
    if args.smoke:
        run_name += "_smoke"
    cfg["dataset"]["training"]["video_anno_path"] = str(pseudo_csv)
    cfg["dataset"]["training"].pop("psee_reliability_path", None)
    cfg["training"]["checkpoint_path"] = str(args.project_root / "models" / run_name)
    cfg["training"]["batch_size"] = 2
    cfg["training"]["learning_rate"] = 5e-5
    cfg["training"]["weight_decay"] = 1e-4
    cfg["training"]["max_epoch"] = int(args.detector_epochs)
    cfg["training"]["random_seed"] = 2020
    cfg["training"]["focal_loss"] = True
    cfg["training"]["grad_clip_norm"] = 0.5
    cfg["training"]["save_model_num"] = []
    cfg["training"]["resume"] = None
    cfg["training"].pop("init_checkpoint", None)
    cfg["training"].pop("psee_reliability_path", None)
    cfg["testing"]["checkpoint_path"] = str(args.project_root / "models" / run_name)
    cfg["testing"]["output_path"] = str(args.project_root / "output" / run_name)
    cfg["testing"]["output_json"] = f"{run_name}.json"
    cfg["protocol"] = {
        **(cfg.get("protocol") or {}),
        "method": METHOD_NAME,
        "detector": "SlimSTAD-1D",
        "two_stage": True,
        "point_sampling": "gaussian",
        "point_seed": args.point_seed,
        "sfnet_losses": ["video_topk_mil", "single_frame_ce", "actionness_hard_negative"],
        "sfnet_adapt_expansion": {
            "evidence": "class_activation_times_actionness",
            "search_limit": "midpoint_between_neighboring_weak_points",
            "duration_prior": "none",
            "psee_reliability": "not_used",
            "threshold": "max(anchor_score * pv, local_otsu_threshold)",
            "smooth_radius": args.smooth_radius,
            "max_gap": args.max_gap,
            "pv": args.sf_pv,
            "score_mode": args.sf_score_mode,
        },
        "pseudo_boundary_source": "sfnet_adapt_expanded_pseudo_action_frames",
        "reliability_weighted_detector": False,
        "no_gt_duration_prior": True,
        "fair_template_config": str(template),
        "detector_batch_size": 2,
        "detector_learning_rate": 5e-5,
        "detector_weight_decay": 1e-4,
        "detector_max_epoch": int(args.detector_epochs),
        "detector_seed": 2020,
    }
    config_path = mkdir(args.project_root / "configs" / TAG) / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def build_loader(train_videos: Sequence[str], points: Dict[str, List[dict]], train_info: Dict[str, dict], args: argparse.Namespace) -> DataLoader:
    dataset = PointWindowDataset(
        train_videos,
        points,
        train_info,
        FEATURE_ROOTS[args.dataset],
        args.classifier_window,
        args.classifier_stride,
        max_windows=args.max_train_windows,
    )
    if len(dataset) == 0:
        raise RuntimeError("No point-supervised train windows were found.")
    return DataLoader(
        dataset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def release_torch_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def generate_pseudo_and_config(dataset: str, split: int, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    paths = split_paths(ANNOTATION_ROOTS[dataset], dataset, split)
    train_info = align_info_to_detector_coordinate(load_info(paths["train_info"]), FEATURE_ROOTS[dataset])
    train_annotations = load_annotations(paths["train_ann"])
    val_annotations = load_annotations(paths["val_ann"])
    test_annotations = load_annotations(paths["test_ann"])
    num_classes = class_count_from_annotations(train_annotations, val_annotations, test_annotations)
    points = make_gaussian_points(train_annotations, train_info, args.point_seed)
    train_videos = sorted(train_info)
    input_dim = int(load_feature(FEATURE_ROOTS[dataset], train_videos[0]).shape[1])
    WearSecondEncoder = import_wear_second_encoder(args.detr_root)
    model = SensorSFNetLocalizer(
        WearSecondEncoder,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        downsample=args.downsample,
    ).to(args.device)

    warmup_epochs = max(0, min(args.expand_epoch, args.classifier_epochs))
    if warmup_epochs:
        print(
            f"[stage1a] {METHOD_NAME} {dataset} loso{split}: "
            f"warmup_windows={len(build_loader(train_videos, points, train_info, args).dataset)} "
            f"points={sum(len(v) for v in points.values())}",
            flush=True,
        )
        train_sfnet_localizer(model, build_loader(train_videos, points, train_info, args), args, num_classes, warmup_epochs, "warmup")

    labels_by_video, _, _, _ = expand_all_records(model, points, train_info, FEATURE_ROOTS[dataset], args)
    expanded_points = labels_to_training_points(labels_by_video, train_info)
    expanded_count = int(sum(len(v) for v in expanded_points.values()))
    finetune_epochs = max(0, args.classifier_epochs - warmup_epochs)
    if finetune_epochs:
        print(
            f"[stage1b] {METHOD_NAME} {dataset} loso{split}: "
            f"expanded_frames={expanded_count} finetune_epochs={finetune_epochs}",
            flush=True,
        )
        train_sfnet_localizer(
            model,
            build_loader(train_videos, expanded_points, train_info, args),
            args,
            num_classes,
            finetune_epochs,
            "expanded",
        )

    labels_by_video, anchor_by_video, class_scores_by_video, action_scores_by_video = expand_all_records(
        model, points, train_info, FEATURE_ROOTS[dataset], args
    )
    pseudo = make_pseudo_segments(
        points,
        train_info,
        labels_by_video,
        anchor_by_video,
        class_scores_by_video,
        action_scores_by_video,
    )
    pseudo_root = args.project_root / f"{TAG}_pseudo_point_gaussian_seed{args.point_seed}"
    pseudo_csv, quality_json = export_pseudo_csv(
        dataset,
        split,
        pseudo,
        points,
        train_info,
        train_annotations,
        pseudo_root,
        args,
    )
    config_path = build_detector_config(dataset, split, pseudo_csv, args)
    del model, labels_by_video, anchor_by_video, class_scores_by_video, action_scores_by_video
    release_torch_memory(args.device)
    return pseudo_csv, quality_json, config_path


def run_detector(config_path: Path, args: argparse.Namespace) -> dict:
    detector_python = os.environ.get("SLIMSTAD_DETECTOR_PYTHON", str(args.python))
    cmd = [detector_python, "all_strict_map_psee.py", str(config_path), "--ngpu", "1"]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[stage2]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(args.project_root), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"SlimSTAD1D detector failed with code {proc.returncode}: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_path = Path(cfg["testing"]["output_path"]) / "strict_result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def task_summary_path(args: argparse.Namespace, dataset: str, split: int) -> Path:
    suffix = "_smoke" if args.smoke else ""
    return args.project_root / "logs" / TAG / f"sfnet_adapt_{dataset}_loso{split}{suffix}_task_summary.json"


def run_task(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pseudo_csv, quality_json, config_path = generate_pseudo_and_config(args.dataset, args.split, args)
    release_torch_memory(args.device)
    strict = run_detector(config_path, args)
    out = {
        "method": METHOD_NAME,
        "method_key": "sfnet_adapt",
        "dataset": args.dataset,
        "split": args.split,
        "smoke": bool(args.smoke),
        "pseudo_csv": str(pseudo_csv),
        "pseudo_quality": json.loads(quality_json.read_text(encoding="utf-8")),
        "detector_config": str(config_path),
        "strict_result": strict,
    }
    write_json(task_summary_path(args, args.dataset, args.split), out)
    print("TASK_RESULT", json.dumps(out, sort_keys=True), flush=True)


def active_run_tasks(project_root: Path) -> set:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,cmd="],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return set()
    active = set()
    pattern = re.compile(r"run_sfnet_adapt_twostage_slimstad1d_queue\.py\s+run-task.*?--dataset\s+(\S+).*?--split\s+(\d+)")
    current_pid = os.getpid()
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        match = pattern.search(parts[1])
        if match:
            active.add((match.group(1), int(match.group(2))))
    return active


def load_existing_results(args: argparse.Namespace, tasks: Sequence[Tuple[str, int]]) -> List[dict]:
    results = []
    for dataset, split in tasks:
        path = task_summary_path(args, dataset, split)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("strict_result"):
            results.append(data)
    return results


def summarize(args: argparse.Namespace, results: Sequence[dict], failures: Sequence[dict], total: int, finished: bool) -> dict:
    rows = []
    by_dataset = defaultdict(list)
    for item in results:
        strict = item["strict_result"]
        maps = [float(x) for x in strict.get("test_mAP", [])]
        avg = float(strict.get("test_average_mAP", 0.0))
        by_dataset[item["dataset"]].append(avg)
        rows.append(
            {
                "method": item["method"],
                "dataset": item["dataset"],
                "split": int(item["split"]),
                "mAP@0.3": maps[0] if len(maps) > 0 else None,
                "mAP@0.4": maps[1] if len(maps) > 1 else None,
                "mAP@0.5": maps[2] if len(maps) > 2 else None,
                "mAP@0.6": maps[3] if len(maps) > 3 else None,
                "mAP@0.7": maps[4] if len(maps) > 4 else None,
                "avg": avg,
                "pseudo_quality": item.get("pseudo_quality", {}),
            }
        )
    groups = {
        dataset: {
            "n": len(vals),
            "avg_mean": float(np.mean(vals)) if vals else None,
            "avg_std": float(np.std(vals)) if vals else None,
        }
        for dataset, vals in by_dataset.items()
    }
    return {
        "tag": TAG,
        "method": METHOD_NAME,
        "finished": bool(finished),
        "total_tasks": total,
        "completed_tasks": len(results),
        "failed_tasks": len(failures),
        "rows": sorted(rows, key=lambda x: (x["dataset"], x["split"])),
        "group_summary": groups,
        "failures": list(failures),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def launch_queue(args: argparse.Namespace) -> None:
    tasks = [(dataset, split) for dataset in args.datasets for split in (args.splits or ([0] if args.smoke else list(range(SPLIT_COUNTS[dataset]))))]
    done = {(r["dataset"], int(r["split"])) for r in load_existing_results(args, tasks)}
    active = active_run_tasks(args.project_root)
    queue = deque([task for task in tasks if task not in done and task not in active])
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()] or ["0"]
    max_parallel = max(1, len(gpus) * max(1, args.workers_per_gpu))
    running, failures = [], []
    log_root = mkdir(args.project_root / "logs" / TAG)
    print(f"[queue] total={len(tasks)} done={len(done)} active={len(active)} queued={len(queue)} gpus={gpus}", flush=True)
    while queue or running:
        while queue and len(running) < max_parallel:
            dataset, split = queue.popleft()
            gpu_load = {gpu: 0 for gpu in gpus}
            for item in running:
                gpu_load[item["gpu"]] = gpu_load.get(item["gpu"], 0) + 1
            gpu = min(gpus, key=lambda value: (gpu_load[value], gpus.index(value)))
            task_name = f"sfnet_adapt_{dataset}_loso{split}{'_smoke' if args.smoke else ''}"
            log_path = log_root / f"{task_name}.log"
            cmd = [
                str(args.python),
                str(Path(__file__).resolve()),
                "run-task",
                "--dataset",
                dataset,
                "--split",
                str(split),
                "--project-root",
                str(args.project_root),
                "--detr-root",
                str(args.detr_root),
                "--python",
                str(args.python),
                "--classifier-epochs",
                str(args.classifier_epochs),
                "--detector-epochs",
                str(args.detector_epochs),
                "--num-workers",
                str(args.num_workers),
                "--expand-radius",
                str(args.expand_radius),
                "--smooth-radius",
                str(args.smooth_radius),
                "--max-gap",
                str(args.max_gap),
                "--sf-pv",
                str(args.sf_pv),
                "--sf-score-mode",
                args.sf_score_mode,
            ]
            if args.smoke:
                cmd.append("--smoke")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w") as log_file:
                proc = subprocess.Popen(cmd, cwd=str(args.project_root), env=env, stdout=log_file, stderr=subprocess.STDOUT)
            running.append({"proc": proc, "task": (dataset, split), "log": log_path, "gpu": gpu})
            print(f"[queue] started {task_name} gpu={gpu} log={log_path}", flush=True)
        time.sleep(args.poll_interval)
        keep = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                keep.append(item)
                continue
            dataset, split = item["task"]
            summary = task_summary_path(args, dataset, split)
            if code == 0 and summary.exists():
                data = json.loads(summary.read_text(encoding="utf-8"))
                avg = float(data["strict_result"].get("test_average_mAP", 0.0))
                print(f"[queue] done sfnet_adapt {dataset} loso{split} avg={avg:.6f}", flush=True)
            else:
                failure = {"dataset": dataset, "split": split, "returncode": code, "log": str(item["log"])}
                failures.append(failure)
                print(f"[queue] failed {failure}", flush=True)
        running = keep
        results = load_existing_results(args, tasks)
        write_json(args.summary_path, summarize(args, results, failures, len(tasks), finished=not queue and not running))
        if failures and args.stop_on_failure:
            raise RuntimeError(failures[-1])


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--detr-root", type=Path, default=DETR_ROOT)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--point-seed", type=int, default=3407)
    parser.add_argument("--classifier-window", type=int, default=1024)
    parser.add_argument("--classifier-stride", type=int, default=256)
    parser.add_argument("--classifier-epochs", type=int, default=25)
    parser.add_argument("--classifier-batch-size", type=int, default=2)
    parser.add_argument("--classifier-lr", type=float, default=2e-4)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--detector-epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tm", type=float, default=7.0)
    parser.add_argument("--expand-epoch", type=int, default=6)
    parser.add_argument("--expand-radius", type=int, default=3)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--sf-pv", type=float, default=0.5)
    parser.add_argument("--sf-score-mode", choices=["softmax", "sigmoid", "raw"], default="sigmoid")
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    task = sub.add_parser("run-task")
    add_common_args(task)
    task.add_argument("--dataset", choices=sorted(FEATURE_ROOTS), required=True)
    task.add_argument("--split", type=int, required=True)
    queue = sub.add_parser("queue")
    add_common_args(queue)
    queue.add_argument("--datasets", nargs="+", choices=sorted(FEATURE_ROOTS), default=["wear"])
    queue.add_argument("--splits", nargs="*", type=int, default=[])
    queue.add_argument("--gpus", default="0")
    queue.add_argument("--workers-per-gpu", type=int, default=1)
    queue.add_argument("--poll-interval", type=float, default=20.0)
    queue.add_argument("--stop-on-failure", action="store_true")
    queue.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "logs" / f"{TAG}_summary.json")
    args = parser.parse_args()
    if args.smoke:
        args.classifier_epochs = min(args.classifier_epochs, 4)
        args.detector_epochs = min(args.detector_epochs, 2)
        args.expand_epoch = min(args.expand_epoch, 2)
        args.max_train_windows = args.max_train_windows or 64
        args.num_workers = 0
    if args.command == "run-task":
        run_task(args)
    else:
        launch_queue(args)


if __name__ == "__main__":
    main()
