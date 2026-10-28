#!/usr/bin/env python3
"""Order-preserving timestamp baseline adapted for TAL + SlimSTAD-1D.

This is a TAL-compatible baseline inspired by the timestamp-supervised
wearable segmentation paper.  Stage 1 trains a timestamp localizer from the
same Gaussian point labels used by PSEE.  Stage 2 converts localizer scores to
local background-action-background pseudo segments with an order-preserving
point-gap constraint, then trains the same SlimSTAD-1D detector protocol.

The implementation deliberately avoids GT duration/boundary priors.  Ground
truth boundaries are used only for diagnostics and final mAP evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from run_point_only_classifier_baseline import (
    PointOnlyClassifier,
    PointWindowDataset,
    collate,
    import_wear_second_encoder,
    load_annotations,
    load_feature,
    load_info,
    make_gaussian_points,
    normalize_feature,
    point_neighborhood_loss,
    segment_iou,
    set_seed,
    split_paths,
)
from run_pointcls_twostage_slimstad1d_queue import (
    class_count_from_annotations,
    find_template_config,
    frame_from_coordinate,
    group_annotations,
    mkdir,
    write_json,
)


PROJECT_ROOT = Path("/home/yinjiaxi/SlimSTAD/STAD-main")
DETR_ROOT = Path("/home/yinjiaxi/detr_e2e_tad")
PYTHON = Path("/home/yinjiaxi/miniconda3/envs/slimstad/bin/python")
TAG = "opot_tal_gaussian_slimstad1d_coordfix_20260902"

FEATURE_ROOTS = {
    "wear": Path("/data2/yinjiaxi/wear/processed/inertial_features/50_samples_50_overlap"),
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
METHOD_NAME = "OPOT-TAL"


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


def find_template_config(dataset: str, split: int, project_root: Path) -> Path:
    if dataset == "wear":
        candidates = [
            project_root / "configs" / "psee_clean_20260715_point_ablation" / f"psee_clean_20260715_point_gaussian_wear_loso{split}_b000.yaml",
            project_root / "configs" / "psee_gaussian_slimstad1d_clean_yjx164_20260720" / f"psee_gaussian_slimstad1d_clean_yjx164_20260720_wear_loso{split}_b000.yaml",
        ]
    elif dataset == "xrfv2":
        candidates = [
            project_root / "configs" / "psee_clean_20260715_point_ablation" / f"psee_clean_20260715_point_gaussian_xrfv2_imu30_loso{split}_b000.yaml",
            project_root / "configs" / "psee_v2_slimstad1d" / "slimstad1d_psee_v2_gaussian_2020" / f"slimstad1d_psee_v2_gaussian_2020__xrfv2_imu30_loso{split}_b000.yaml",
        ]
    else:
        candidates = []
    for path in candidates:
        if path.exists():
            return path
    matches = [
        p
        for p in (project_root / "configs").glob(f"**/*{dataset}_loso{split}_b000.yaml")
        if "gaussian" in p.name and ("slimstad1d" in str(p) or "psee_clean" in str(p))
    ]
    if not matches:
        raise FileNotFoundError(f"No SlimSTAD-1D template config for {dataset} loso{split}")
    return sorted(matches)[0]


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    mkdir(path.parent)
    fields = ["video", "type", "type_idx", "start", "end", "startFrame", "endFrame"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def time_to_index(value: float, duration: float, out_len: int) -> int:
    idx = int(round(value / max(duration, 1e-6) * max(out_len - 1, 1)))
    return max(0, min(out_len - 1, idx))


def index_to_time(index: int, duration: float, out_len: int) -> float:
    return float(index) / max(out_len - 1, 1) * duration


def align_info_to_detector_coordinate(info: Dict[str, dict], feature_root: Path) -> Dict[str, dict]:
    """Use the same temporal coordinate as STAD annotations and features.

    Some Wear/TAL4HAR info files contain both raw 50Hz count/fps and sampled
    feature count/sample_fps.  The CSV start/end annotations are in sampled
    detector coordinates, while frame_from_coordinate keeps raw-frame fields
    consistent for the SlimSTAD loader.  Stage-1 timestamp learning must
    therefore use sample_count/feature length as its duration coordinate.
    """
    aligned = {}
    for video, row in info.items():
        item = dict(row)
        sample_count = float(item.get("sample_count") or 0.0)
        if sample_count <= 0:
            try:
                sample_count = float(load_feature(feature_root, video).shape[0])
            except Exception:
                sample_count = float(item.get("count") or item.get("duration") or 0.0)
        if sample_count > 0:
            item["duration"] = sample_count
        aligned[video] = item
    return aligned


def multilabel_window_loss(logits: torch.Tensor, points: Sequence[List[dict]], num_classes: int) -> torch.Tensor:
    targets = torch.zeros((logits.shape[0], num_classes), dtype=logits.dtype, device=logits.device)
    for batch_idx, batch_points in enumerate(points):
        for point in batch_points:
            label = int(point["label"])
            if 0 <= label < num_classes:
                targets[batch_idx, label] = 1.0
    pooled = logits.amax(dim=-1)
    return F.binary_cross_entropy_with_logits(pooled, targets)


def temporal_smooth_loss(logits: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    delta = torch.abs(logp[:, :, 1:] - logp[:, :, :-1])
    return torch.clamp(delta, max=tau).mean()


def train_opot_localizer(model: torch.nn.Module, loader: DataLoader, args: argparse.Namespace, num_classes: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    model.train()
    for epoch in range(1, args.classifier_epochs + 1):
        sums = defaultdict(float)
        steps = 0
        for batch in loader:
            feat = batch["feat"].to(args.device)
            logits = model(feat)
            point_loss = point_neighborhood_loss(logits, batch["points"], args.classifier_window, args.point_radius)
            ml_loss = multilabel_window_loss(logits, batch["points"], num_classes)
            smooth = temporal_smooth_loss(logits)
            loss = point_loss + args.lambda_mil * ml_loss + args.lambda_smooth * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            sums["point"] += float(point_loss.detach().cpu())
            sums["mil"] += float(ml_loss.detach().cpu())
            sums["smooth"] += float(smooth.detach().cpu())
            sums["total"] += float(loss.detach().cpu())
            steps += 1
        print(
            f"[epoch {epoch:03d}] total={sums['total']/max(steps,1):.4f} "
            f"point={sums['point']/max(steps,1):.4f} "
            f"mil={sums['mil']/max(steps,1):.4f} smooth={sums['smooth']/max(steps,1):.4f}",
            flush=True,
        )


def ordered_boundary_between(
    scores: np.ndarray,
    left_label: int,
    right_label: int,
    left_time: float,
    right_time: float,
    duration: float,
    rho: float,
    sigma: float,
) -> float:
    out_len = scores.shape[1]
    lo = time_to_index(left_time, duration, out_len)
    hi = time_to_index(right_time, duration, out_len)
    if hi <= lo + 1:
        return 0.5 * (left_time + right_time)
    prev = np.log(np.clip(scores[left_label, lo : hi + 1], 1e-6, 1.0))
    nxt = np.log(np.clip(scores[right_label, lo : hi + 1], 1e-6, 1.0))
    pos = np.linspace(0.0, 1.0, len(prev), dtype=np.float32)
    # A two-class order prior: previous timestamp owns the left side, next
    # timestamp owns the right side.  This is the TAL segment analogue of the
    # paper's diagonal order-preserving OT prior.
    prior_prev = np.exp(-0.5 * (pos / max(sigma, 1e-3)) ** 2)
    prior_next = np.exp(-0.5 * ((1.0 - pos) / max(sigma, 1e-3)) ** 2)
    prev_score = prev + rho * np.log(np.clip(prior_prev, 1e-6, 1.0))
    next_score = nxt + rho * np.log(np.clip(prior_next, 1e-6, 1.0))
    prefix_prev = np.concatenate([[0.0], np.cumsum(prev_score)])
    prefix_next = np.concatenate([[0.0], np.cumsum(next_score)])
    total_next = prefix_next[-1]
    best = -1e18
    best_k = len(prev) // 2
    for k in range(1, len(prev)):
        value = prefix_prev[k] + (total_next - prefix_next[k])
        if value > best:
            best = float(value)
            best_k = k
    boundary_idx = lo + best_k
    return index_to_time(boundary_idx, duration, out_len)


def ordered_action_background_action(
    scores: np.ndarray,
    left_label: int,
    right_label: int,
    left_time: float,
    right_time: float,
    duration: float,
    rho: float,
    sigma: float,
) -> Tuple[float, float]:
    """Split an interval as left-action -> implicit-background -> right-action.

    The timestamp paper assumes dense segmentation labels, so adjacent
    timestamps can explain the whole interval.  In TAL, only action instances
    have point labels and the gap can contain long background.  We therefore add
    an implicit background state derived from low action confidence, without
    using GT background boundaries.
    """
    out_len = scores.shape[1]
    lo = time_to_index(left_time, duration, out_len)
    hi = time_to_index(right_time, duration, out_len)
    if hi <= lo + 2:
        mid = 0.5 * (left_time + right_time)
        return mid, mid
    local = scores[:, lo : hi + 1]
    left = np.log(np.clip(local[left_label], 1e-6, 1.0))
    right = np.log(np.clip(local[right_label], 1e-6, 1.0))
    bg_raw = 1.0 - np.max(local, axis=0)
    bg = np.log(np.clip(bg_raw, 1e-6, 1.0))
    pos = np.linspace(0.0, 1.0, len(left), dtype=np.float32)
    prior_left = np.exp(-0.5 * (pos / max(sigma, 1e-3)) ** 2)
    prior_right = np.exp(-0.5 * ((1.0 - pos) / max(sigma, 1e-3)) ** 2)
    prior_bg = np.clip(1.0 - np.maximum(prior_left, prior_right), 1e-3, 1.0)
    left_score = left + rho * np.log(np.clip(prior_left, 1e-6, 1.0))
    right_score = right + rho * np.log(np.clip(prior_right, 1e-6, 1.0))
    bg_score = bg + rho * np.log(prior_bg)
    pre_left = np.concatenate([[0.0], np.cumsum(left_score)])
    pre_bg = np.concatenate([[0.0], np.cumsum(bg_score)])
    pre_right = np.concatenate([[0.0], np.cumsum(right_score)])
    total_right = pre_right[-1]
    best = -1e18
    best_a, best_b = 1, len(left) - 1
    for a in range(1, len(left) - 1):
        # Keep the background band legal but allow it to become short when two
        # annotated actions are genuinely adjacent.
        for b in range(a, len(left)):
            value = pre_left[a] + (pre_bg[b] - pre_bg[a]) + (total_right - pre_right[b])
            if value > best:
                best = float(value)
                best_a, best_b = a, b
    return index_to_time(lo + best_a, duration, out_len), index_to_time(lo + best_b, duration, out_len)


def ordered_background_action_boundary(
    scores: np.ndarray,
    label: int,
    point_time: float,
    duration: float,
    rho: float,
    sigma: float,
    side: str,
) -> float:
    out_len = scores.shape[1]
    if side == "left":
        lo, hi = 0, time_to_index(point_time, duration, out_len)
    else:
        lo, hi = time_to_index(point_time, duration, out_len), out_len - 1
    if hi <= lo + 1:
        return point_time
    local = scores[:, lo : hi + 1]
    act = np.log(np.clip(local[label], 1e-6, 1.0))
    bg = np.log(np.clip(1.0 - np.max(local, axis=0), 1e-6, 1.0))
    pos = np.linspace(0.0, 1.0, len(act), dtype=np.float32)
    if side == "left":
        prior_act = np.exp(-0.5 * ((1.0 - pos) / max(sigma, 1e-3)) ** 2)
        prior_bg = np.clip(1.0 - prior_act, 1e-3, 1.0)
        score_a = bg + rho * np.log(prior_bg)
        score_b = act + rho * np.log(np.clip(prior_act, 1e-6, 1.0))
    else:
        prior_act = np.exp(-0.5 * (pos / max(sigma, 1e-3)) ** 2)
        prior_bg = np.clip(1.0 - prior_act, 1e-3, 1.0)
        score_a = act + rho * np.log(np.clip(prior_act, 1e-6, 1.0))
        score_b = bg + rho * np.log(prior_bg)
    pre_a = np.concatenate([[0.0], np.cumsum(score_a)])
    pre_b = np.concatenate([[0.0], np.cumsum(score_b)])
    total_b = pre_b[-1]
    best, best_k = -1e18, len(act) // 2
    for k in range(1, len(act)):
        value = pre_a[k] + (total_b - pre_b[k])
        if value > best:
            best = float(value)
            best_k = k
    return index_to_time(lo + best_k, duration, out_len)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo + 1e-6:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def transition_curve(feat: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Generic boundary cue from sensor change and classifier-score change."""
    out_len = scores.shape[1]
    if out_len <= 1:
        return np.zeros((out_len,), dtype=np.float32)

    sensor = np.zeros((out_len,), dtype=np.float32)
    if feat.shape[0] > 1:
        diff = np.linalg.norm(np.diff(feat.astype(np.float32), axis=0), axis=1)
        if diff.size:
            src_x = np.linspace(0, diff.size - 1, diff.size)
            dst_x = np.linspace(0, diff.size - 1, out_len)
            sensor = np.interp(dst_x, src_x, normalize01(diff)).astype(np.float32)

    actionness = np.max(scores, axis=0)
    class_change = np.zeros((out_len,), dtype=np.float32)
    if out_len > 1:
        change = np.abs(np.diff(actionness))
        src_x = np.linspace(0, change.size - 1, change.size)
        dst_x = np.linspace(0, change.size - 1, out_len)
        class_change = np.interp(dst_x, src_x, normalize01(change)).astype(np.float32)

    return normalize01(0.5 * sensor + 0.5 * class_change)


def local_point_bounds(ordered: Sequence[dict], idx: int, duration: float, global_gap: float) -> Tuple[float, float]:
    point_time = float(ordered[idx]["time"])
    if idx > 0:
        left_limit = 0.5 * (float(ordered[idx - 1]["time"]) + point_time)
    else:
        right_gap = float(ordered[idx + 1]["time"]) - point_time if idx + 1 < len(ordered) else global_gap
        left_limit = point_time - 0.5 * max(right_gap, 1e-3)
    if idx + 1 < len(ordered):
        right_limit = 0.5 * (point_time + float(ordered[idx + 1]["time"]))
    else:
        left_gap = point_time - float(ordered[idx - 1]["time"]) if idx > 0 else global_gap
        right_limit = point_time + 0.5 * max(left_gap, 1e-3)
    return max(0.0, left_limit), min(duration, right_limit)


def snap_boundary_index(
    boundary_idx: int,
    transition: np.ndarray,
    lo: int,
    hi: int,
    args: argparse.Namespace,
) -> int:
    if not args.transition_snap or hi <= lo or transition.size == 0:
        return int(np.clip(boundary_idx, lo, hi))
    radius = max(1, int(round(args.snap_fraction * max(hi - lo + 1, 1))))
    left = max(lo, boundary_idx - radius)
    right = min(hi, boundary_idx + radius)
    if right <= left:
        return int(np.clip(boundary_idx, lo, hi))
    local = transition[left : right + 1]
    return int(left + int(np.argmax(local)))


def local_order_preserving_segment(
    scores: np.ndarray,
    transition: np.ndarray,
    label: int,
    point_time: float,
    left_limit: float,
    right_limit: float,
    duration: float,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    """OPOT-style local background-action-background assignment around one point."""
    out_len = scores.shape[1]
    lo = time_to_index(left_limit, duration, out_len)
    hi = time_to_index(right_limit, duration, out_len)
    point_idx = time_to_index(point_time, duration, out_len)
    lo = min(lo, max(0, point_idx - 1))
    hi = max(hi, min(out_len - 1, point_idx + 1))
    if hi <= lo + 1:
        eps = max(duration / max(out_len, 1), 1e-3)
        return max(0.0, point_time - eps), min(duration, point_time + eps), float(scores[label, point_idx])

    local = scores[:, lo : hi + 1]
    action_prob = np.clip(local[label], 1e-6, 1.0)
    bg_prob = np.clip(1.0 - np.max(local, axis=0), 1e-6, 1.0)
    n = len(action_prob)
    p = int(np.clip(point_idx - lo, 0, n - 1))

    pos = np.arange(n, dtype=np.float32)
    sigma = max(1.0, args.ot_sigma * max(p, n - 1 - p, 1))
    point_prior = np.exp(-0.5 * ((pos - p) / sigma) ** 2)
    action_score = np.log(action_prob) + args.ot_rho * np.log(np.clip(point_prior, 1e-6, 1.0))
    bg_score = np.log(bg_prob)

    pre_action = np.concatenate([[0.0], np.cumsum(action_score)])
    pre_bg = np.concatenate([[0.0], np.cumsum(bg_score)])
    total_bg = pre_bg[-1]
    best_value = -1e18
    best_s, best_e = p, p + 1
    for s in range(0, p + 1):
        for e in range(p + 1, n + 1):
            value = pre_bg[s] + (pre_action[e] - pre_action[s]) + (total_bg - pre_bg[e])
            if value > best_value:
                best_value = float(value)
                best_s, best_e = s, e

    start_idx = lo + best_s
    end_idx = min(lo + best_e, out_len - 1)
    start_idx = snap_boundary_index(start_idx, transition, lo, max(lo, point_idx - 1), args)
    end_idx = snap_boundary_index(end_idx, transition, min(out_len - 1, point_idx + 1), hi, args)
    if start_idx >= point_idx:
        start_idx = max(lo, point_idx - 1)
    if end_idx <= point_idx:
        end_idx = min(hi, point_idx + 1)

    start = max(0.0, min(index_to_time(start_idx, duration, out_len), point_time - 1e-3))
    end = min(duration, max(index_to_time(end_idx, duration, out_len), point_time + 1e-3))
    inside = action_prob[max(0, start_idx - lo) : max(0, end_idx - lo) + 1]
    confidence = float(np.mean(inside)) if inside.size else float(scores[label, point_idx])
    return start, end, confidence


@torch.no_grad()
def generate_opot_pseudo(
    model: torch.nn.Module,
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    feature_root: Path,
    args: argparse.Namespace,
) -> List[dict]:
    model.eval()
    pseudo: List[dict] = []
    all_gaps = []
    for video_points in points.values():
        ordered_points = sorted(video_points, key=lambda item: float(item["time"]))
        for left, right in zip(ordered_points[:-1], ordered_points[1:]):
            gap = float(right["time"]) - float(left["time"])
            if gap > 1e-3:
                all_gaps.append(gap)
    global_gap = float(np.median(all_gaps)) if all_gaps else 10.0
    for video in sorted(points):
        feat = normalize_feature(load_feature(feature_root, video))
        duration = max(float(info[video]["duration"]), 1e-6)
        tensor = torch.from_numpy(feat).unsqueeze(0).to(args.device)
        scores = torch.sigmoid(model(tensor))[0].detach().cpu().numpy()
        transition = transition_curve(feat, scores)
        ordered = sorted(points[video], key=lambda item: float(item["time"]))
        if not ordered:
            continue
        for idx, point in enumerate(ordered):
            label = int(point["label"])
            left_limit, right_limit = local_point_bounds(ordered, idx, duration, global_gap)
            start, end, score = local_order_preserving_segment(
                scores,
                transition,
                label,
                float(point["time"]),
                left_limit,
                right_limit,
                duration,
                args,
            )
            pseudo.append(
                {
                    "video": video,
                    "label": label,
                    "start": start,
                    "end": end,
                    "score": score,
                    "point_time": float(point["time"]),
                    "annotation_index": int(point["annotation_index"]),
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
        "pseudo_mean_iou": float(arr.mean()),
        "pseudo_median_iou": float(np.median(arr)),
        "pseudo_r@0.3": float((arr >= 0.3).mean()),
        "pseudo_r@0.5": float((arr >= 0.5).mean()),
        "pseudo_r@0.7": float((arr >= 0.7).mean()),
        "start_error_over_duration": float(np.mean(start_err)),
        "end_error_over_duration": float(np.mean(end_err)),
        "duration_ratio_error": float(np.mean(ratio_err)),
    }


def export_pseudo_csv(
    dataset: str,
    split: int,
    pseudo: Sequence[dict],
    info: Dict[str, dict],
    train_annotations: Sequence[dict],
    root: Path,
) -> Tuple[Path, Path]:
    gt_by_video = group_annotations(train_annotations)
    out_dir = mkdir(root / dataset / f"loso{split}")
    csv_path = out_dir / "train_annotations_opot_000.csv"
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
        "stage": "tal_adapted_order_preserving_pseudo_boundary_generation",
        "point_sampling": "gaussian",
        "point_seed": 3407,
        "pseudo_count": len(rows),
        "train_point_count": len(pseudo),
        "no_gt_duration_prior": True,
        "tal_adaptation": "local background-action-background assignment around each timestamp",
        "point_gap_constraint": "pseudo segment cannot cross neighboring point midpoints",
        "connected_component": "single action interval constrained to contain the annotated point",
        "transition_snap": True,
        "ot_rho": None,
        "ot_sigma": None,
        **pseudo_quality(pseudo, train_annotations),
    }
    meta_path = out_dir / "opot_pseudo_quality.json"
    write_json(meta_path, quality)
    return csv_path, meta_path


def build_detector_config(dataset: str, split: int, pseudo_csv: Path, args: argparse.Namespace) -> Path:
    template = find_template_config(dataset, split, args.project_root)
    cfg = copy.deepcopy(yaml.safe_load(template.read_text()))
    run_name = f"{TAG}_opot_{dataset}_loso{split}_b000"
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
    cfg["testing"]["checkpoint_path"] = str(args.project_root / "models" / run_name)
    cfg["testing"]["output_path"] = str(args.project_root / "output" / run_name)
    cfg["testing"]["output_json"] = f"{run_name}.json"
    cfg["protocol"] = {
        **(cfg.get("protocol") or {}),
        "method": METHOD_NAME,
        "two_stage": True,
        "detector": "SlimSTAD-1D",
        "point_sampling": "gaussian",
        "point_seed": 3407,
        "pseudo_boundary_source": "tal_adapted_order_preserving_timestamp_assignment",
        "reliability_weighted_detector": False,
        "no_gt_duration_prior": True,
        "tal_adaptation": "local background-action-background assignment plus lightweight transition snapping",
        "point_gap_constraint": "neighboring point midpoints only",
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
    model = PointOnlyClassifier(
        WearSecondEncoder,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        downsample=args.downsample,
    ).to(args.device)
    dataset_obj = PointWindowDataset(
        train_videos,
        points,
        train_info,
        FEATURE_ROOTS[dataset],
        args.classifier_window,
        args.classifier_stride,
        max_windows=args.max_train_windows,
    )
    loader = DataLoader(
        dataset_obj,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    print(
        f"[stage1] {METHOD_NAME} {dataset} loso{split}: "
        f"windows={len(dataset_obj)} points={sum(len(v) for v in points.values())}",
        flush=True,
    )
    train_opot_localizer(model, loader, args, num_classes)
    pseudo = generate_opot_pseudo(model, points, train_info, FEATURE_ROOTS[dataset], args)
    pseudo_root = args.project_root / f"{TAG}_pseudo_point_gaussian_seed{args.point_seed}"
    pseudo_csv, quality_json = export_pseudo_csv(dataset, split, pseudo, train_info, train_annotations, pseudo_root)
    quality = json.loads(quality_json.read_text())
    quality["ot_rho"] = args.ot_rho
    quality["ot_sigma"] = args.ot_sigma
    quality["snap_fraction"] = args.snap_fraction
    quality["transition_snap"] = bool(args.transition_snap)
    write_json(quality_json, quality)
    config_path = build_detector_config(dataset, split, pseudo_csv, args)
    return pseudo_csv, quality_json, config_path


def run_detector(config_path: Path, args: argparse.Namespace) -> dict:
    cmd = [str(args.python), "all_strict_map_psee.py", str(config_path), "--ngpu", "1"]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[stage2]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(args.project_root), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"SlimSTAD1D detector failed with code {proc.returncode}: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    result_path = Path(cfg["testing"]["output_path"]) / "strict_result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    return json.loads(result_path.read_text())


def task_summary_path(args: argparse.Namespace, dataset: str, split: int) -> Path:
    suffix = "_smoke" if args.smoke else ""
    return args.project_root / "logs" / TAG / f"opot_{dataset}_loso{split}{suffix}_task_summary.json"


def run_task(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pseudo_csv, quality_json, config_path = generate_pseudo_and_config(args.dataset, args.split, args)
    strict = run_detector(config_path, args)
    out = {
        "method": METHOD_NAME,
        "method_key": "opot",
        "dataset": args.dataset,
        "split": args.split,
        "smoke": bool(args.smoke),
        "pseudo_csv": str(pseudo_csv),
        "pseudo_quality": json.loads(quality_json.read_text()),
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
    pattern = re.compile(r"run_opot_twostage_slimstad1d_queue\.py\s+run-task.*?--dataset\s+(\S+).*?--split\s+(\d+)")
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
            data = json.loads(path.read_text())
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
                "split": item["split"],
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
    log_root = mkdir(args.project_root / "logs" / TAG)
    tasks = [(dataset, split) for dataset in args.datasets for split in (args.splits or ([0] if args.smoke else list(range(SPLIT_COUNTS[dataset]))))]
    done = {(r["dataset"], int(r["split"])) for r in load_existing_results(args, tasks)}
    active = active_run_tasks(args.project_root)
    queue = deque([task for task in tasks if task not in done and task not in active])
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    max_parallel = max(1, len(gpus) * max(1, args.workers_per_gpu))
    running, failures = [], []
    print(f"[queue] total={len(tasks)} done={len(done)} active={len(active)} queued={len(queue)} gpus={gpus}", flush=True)
    while queue or running:
        while queue and len(running) < max_parallel:
            dataset, split = queue.popleft()
            gpu_load = {gpu: 0 for gpu in gpus}
            for r in running:
                gpu_load[r["gpu"]] = gpu_load.get(r["gpu"], 0) + 1
            gpu = min(gpus, key=lambda g: (gpu_load[g], gpus.index(g)))
            task_name = f"opot_{dataset}_loso{split}{'_smoke' if args.smoke else ''}"
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
            ]
            if args.smoke:
                cmd.append("--smoke")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w") as log:
                proc = subprocess.Popen(cmd, cwd=str(args.project_root), env=env, stdout=log, stderr=subprocess.STDOUT)
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
                data = json.loads(summary.read_text())
                print(f"[queue] done opot {dataset} loso{split} avg={float(data['strict_result'].get('test_average_mAP', 0.0)):.6f}", flush=True)
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
    parser.add_argument("--lambda-mil", type=float, default=0.25)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--ot-rho", type=float, default=0.5)
    parser.add_argument("--ot-sigma", type=float, default=0.35)
    parser.add_argument("--snap-fraction", type=float, default=0.15)
    parser.add_argument("--no-transition-snap", dest="transition_snap", action="store_false")
    parser.set_defaults(transition_snap=True)
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
        args.classifier_epochs = min(args.classifier_epochs, 2)
        args.detector_epochs = min(args.detector_epochs, 2)
        args.max_train_windows = args.max_train_windows or 64
        args.num_workers = 0
    if args.command == "run-task":
        run_task(args)
    else:
        launch_queue(args)


if __name__ == "__main__":
    main()
