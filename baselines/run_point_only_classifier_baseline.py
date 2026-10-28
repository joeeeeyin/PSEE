#!/usr/bin/env python3
"""Point-neighborhood classifier baseline for point-supervised sensor TAL.

This baseline is intentionally simple:
  1. sample one point per training action with the same gaussian policy as PSEE;
  2. train only a temporal class classifier around those points;
  3. convert class-score peaks into segments using a training-set duration prior.

It does not use PSEE expansion, transition maps, pseudo-boundary reliability, or
full boundary labels for detector loss.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_PROJECT_ROOT = Path("/home/yinjiaxi/pointsupervision/STAD-main")
DEFAULT_DETR_ROOT = Path("/home/yinjiaxi/pointsupervision/detr_e2e_tad")
DEFAULT_FEATURE_ROOT = Path("/data1/yinjiaxi/wear/processed/inertial_features/50_samples_50_overlap")


def import_wear_second_encoder(detr_root: Path):
    sys.path.insert(0, str(detr_root))
    try:
        from tad.temporal_detr import WearSecondEncoder  # type: ignore

        return WearSecondEncoder
    except Exception:
        class WearSecondEncoder(nn.Module):
            def __init__(self, out_dim: int = 128) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(12, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                bsz, length, _ = x.shape
                x = x.view(bsz * length, 4, 3, 50).permute(0, 1, 3, 2)
                x = x.reshape(bsz * length, 12, 50)
                x = self.net(x).mean(dim=-1)
                return x.view(bsz, length, -1)

        return WearSecondEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_rng(seed: int, mode: str, video_id: str, annotation_index: int) -> np.random.RandomState:
    payload = f"{seed}:{mode}:{video_id}:{annotation_index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return np.random.RandomState(value)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def to_int(row: dict, key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def load_info(path: Path) -> Dict[str, dict]:
    info = {}
    for row in read_csv_rows(path):
        video = row["video"]
        fps = float(row.get("fps") or row.get("sample_fps") or 50.0)
        count = float(row.get("count") or row.get("sample_count") or 0)
        sample_count = float(row.get("sample_count") or 0)
        duration = count / fps if fps > 0 and count > 0 else 0.0
        info[video] = {
            "fps": fps,
            "count": count,
            "sample_count": sample_count,
            "duration": duration,
        }
    return info


def load_annotations(path: Path) -> List[dict]:
    rows = []
    for row in read_csv_rows(path):
        start = to_float(row, "start")
        end = to_float(row, "end")
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        rows.append(
            {
                "video": row["video"],
                "label": to_int(row, "type_idx"),
                "type": row.get("type", str(to_int(row, "type_idx"))),
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )
    return rows


def make_gaussian_points(annotations: Sequence[dict], info: Dict[str, dict], seed: int) -> Dict[str, List[dict]]:
    by_video: Dict[str, List[dict]] = defaultdict(list)
    for ann in annotations:
        by_video[ann["video"]].append(ann)

    points: Dict[str, List[dict]] = defaultdict(list)
    for video, anns in by_video.items():
        record_duration = float(info.get(video, {}).get("duration", 0.0))
        for idx, ann in enumerate(anns):
            rng = stable_rng(seed, "gaussian", video, idx)
            ratio = float(np.clip(rng.normal(loc=0.5, scale=1.0 / 6.0), 0.05, 0.95))
            point_time = ann["start"] + ratio * (ann["end"] - ann["start"])
            points[video].append(
                {
                    "time": point_time,
                    "position": point_time / record_duration if record_duration > 0 else 0.0,
                    "label": int(ann["label"]),
                    "annotation_index": idx,
                    "point_ratio": ratio,
                    "point_mode": "gaussian",
                }
            )
    return points


def duration_priors(
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    num_classes: int,
    mode: str,
) -> Tuple[np.ndarray, float]:
    """Estimate segment duration from point annotations only.

    This baseline must not use GT start/end lengths.  The proxy duration for a
    point is derived from the nearest neighboring point gap in the same record;
    for singleton records we fall back to the record duration, which is known
    from the feature sequence rather than from action boundaries.
    """
    if mode == "global":
        mode = "point_gap_global"
    elif mode == "class":
        mode = "point_gap_class"
    if mode not in {"point_gap_global", "point_gap_class"}:
        raise ValueError(f"Unsupported duration prior mode: {mode}")

    durations = []
    by_class: Dict[int, List[float]] = defaultdict(list)
    for video, video_points in points.items():
        ordered = sorted(video_points, key=lambda item: float(item["time"]))
        record_duration = float(info.get(video, {}).get("duration", 0.0))
        for idx, point in enumerate(ordered):
            center = float(point["time"])
            gaps = []
            if idx > 0:
                gaps.append(center - float(ordered[idx - 1]["time"]))
            if idx + 1 < len(ordered):
                gaps.append(float(ordered[idx + 1]["time"]) - center)
            proxy_duration = max(min(gaps), 1.0) if gaps else max(record_duration, 1.0)
            durations.append(proxy_duration)
            by_class[int(point["label"])].append(proxy_duration)
    global_mean = float(np.mean(durations)) if durations else 1.0
    priors = np.full(num_classes, global_mean, dtype=np.float32)
    if mode == "point_gap_class":
        for cls, vals in by_class.items():
            if 0 <= cls < num_classes and vals:
                priors[cls] = float(np.mean(vals))
    return priors, global_mean


def normalized_duration_prior_mode(mode: str) -> str:
    if mode == "global":
        return "point_gap_global"
    if mode == "class":
        return "point_gap_class"
    return mode


def load_feature(feature_root: Path, video: str) -> np.ndarray:
    path = feature_root / f"{video}.npy"
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature for {video}, got {arr.shape}")
    return arr


def normalize_feature(feat: np.ndarray) -> np.ndarray:
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    return (feat - mean) / (std + 1e-5)


def window_starts(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


class PointWindowDataset(Dataset):
    def __init__(
        self,
        videos: Sequence[str],
        points: Dict[str, List[dict]],
        info: Dict[str, dict],
        feature_root: Path,
        window: int,
        stride: int,
        max_windows: int = 0,
    ) -> None:
        self.points = points
        self.info = info
        self.feature_root = feature_root
        self.window = window
        self.samples: List[Tuple[str, int, int, List[dict]]] = []
        self.feature_cache: Dict[str, np.ndarray] = {}

        for video in videos:
            feat = load_feature(feature_root, video)
            self.feature_cache[video] = feat
            duration = max(float(info[video]["duration"]), 1e-6)
            video_points = points.get(video, [])
            for start in window_starts(len(feat), window, stride):
                end = min(start + window, len(feat))
                local_points = []
                start_time = start / max(len(feat), 1) * duration
                end_time = end / max(len(feat), 1) * duration
                for point in video_points:
                    if start_time <= point["time"] < end_time:
                        local = dict(point)
                        local["local_step"] = (point["time"] - start_time) / max(end_time - start_time, 1e-6) * (end - start)
                        local_points.append(local)
                if local_points:
                    self.samples.append((video, start, end, local_points))

        if max_windows > 0 and len(self.samples) > max_windows:
            self.samples = self.samples[:max_windows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video, start, end, points = self.samples[idx]
        feat = self.feature_cache[video][start:end]
        if len(feat) < self.window:
            pad = np.zeros((self.window - len(feat), feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        feat = normalize_feature(feat)
        return {"feat": torch.from_numpy(feat), "points": points, "video": video}


def collate(batch: Sequence[dict]) -> dict:
    feats = torch.stack([item["feat"] for item in batch], dim=0)
    return {
        "feat": feats,
        "points": [item["points"] for item in batch],
        "videos": [item["video"] for item in batch],
    }


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


class PointOnlyClassifier(nn.Module):
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

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.wear_encoder is not None:
            feat = self.wear_encoder(feat)
        x = feat.transpose(1, 2)
        x = self.encoder(x)
        return self.class_head(x)


def point_neighborhood_loss(
    logits: torch.Tensor,
    points: Sequence[List[dict]],
    input_len: int,
    radius: int,
) -> torch.Tensor:
    bsz, _, out_len = logits.shape
    losses = []
    for batch_idx in range(bsz):
        for point in points[batch_idx]:
            label = int(point["label"])
            out_idx = int(round(float(point["local_step"]) / max(float(input_len), 1.0) * out_len))
            out_idx = max(0, min(out_len - 1, out_idx))
            lo = max(0, out_idx - radius)
            hi = min(out_len, out_idx + radius + 1)
            target = torch.full((hi - lo,), label, dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits[batch_idx, :, lo:hi].transpose(0, 1), target, reduction="mean"))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def train_classifier(model: nn.Module, loader: DataLoader, args) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            feat = batch["feat"].to(args.device)
            logits = model(feat)
            loss = point_neighborhood_loss(logits, batch["points"], args.window, args.point_radius)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        print(f"[epoch {epoch:03d}] point_ce={epoch_loss / max(steps, 1):.4f}", flush=True)


def segment_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def nms(segments: List[dict], threshold: float) -> List[dict]:
    kept = []
    for pred in sorted(segments, key=lambda x: x["score"], reverse=True):
        if all(segment_iou((pred["start"], pred["end"]), (old["start"], old["end"])) < threshold for old in kept):
            kept.append(pred)
    return kept


@torch.no_grad()
def predict_records(
    model: nn.Module,
    videos: Sequence[str],
    info: Dict[str, dict],
    feature_root: Path,
    duration_prior: np.ndarray,
    args,
) -> List[dict]:
    model.eval()
    predictions = []
    for rec_idx, video in enumerate(videos):
        if args.max_eval_records > 0 and rec_idx >= args.max_eval_records:
            break
        feat = normalize_feature(load_feature(feature_root, video))
        duration = max(float(info[video]["duration"]), 1e-6)
        tensor = torch.from_numpy(feat).unsqueeze(0).to(args.device)
        logits = model(tensor)
        scores = torch.sigmoid(logits)[0].detach().cpu().numpy()
        out_len = scores.shape[1]
        per_video = []
        for cls in range(scores.shape[0]):
            cls_scores = scores[cls]
            topk = min(args.topk_per_class, len(cls_scores))
            if topk <= 0:
                continue
            idxs = np.argpartition(-cls_scores, topk - 1)[:topk]
            for idx in idxs:
                score = float(cls_scores[idx])
                if score < args.score_thresh:
                    continue
                center = (float(idx) + 0.5) / out_len * duration
                seg_len = max(float(duration_prior[cls]), 1e-3)
                start = max(0.0, center - 0.5 * seg_len)
                end = min(duration, center + 0.5 * seg_len)
                if end > start:
                    per_video.append({"video": video, "label": cls, "score": score, "start": start, "end": end})
        by_class: Dict[int, List[dict]] = defaultdict(list)
        for pred in per_video:
            by_class[int(pred["label"])].append(pred)
        for cls, preds in by_class.items():
            predictions.extend(nms(preds, args.nms_iou)[: args.keep_per_class])
    return predictions


def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(predictions: Sequence[dict], gt: Sequence[dict], thresholds: Sequence[float], num_classes: int) -> Dict[str, float]:
    gt_by_class_video: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for ann in gt:
        gt_by_class_video[int(ann["label"])][ann["video"]].append(dict(ann, used=False))
    preds_by_class: Dict[int, List[dict]] = defaultdict(list)
    for pred in predictions:
        preds_by_class[int(pred["label"])].append(pred)

    result = {}
    for threshold in thresholds:
        aps = []
        for cls in range(num_classes):
            class_gt = gt_by_class_video.get(cls, {})
            npos = sum(len(v) for v in class_gt.values())
            if npos == 0:
                continue
            class_preds = sorted(preds_by_class.get(cls, []), key=lambda x: x["score"], reverse=True)
            tp = np.zeros(len(class_preds), dtype=np.float32)
            fp = np.zeros(len(class_preds), dtype=np.float32)
            used: Dict[str, set] = defaultdict(set)
            for idx, pred in enumerate(class_preds):
                candidates = class_gt.get(pred["video"], [])
                best_iou, best_gt = 0.0, -1
                for gt_idx, ann in enumerate(candidates):
                    iou = segment_iou((pred["start"], pred["end"]), (ann["start"], ann["end"]))
                    if iou > best_iou:
                        best_iou, best_gt = iou, gt_idx
                if best_iou >= threshold and best_gt >= 0 and best_gt not in used[pred["video"]]:
                    tp[idx] = 1.0
                    used[pred["video"]].add(best_gt)
                else:
                    fp[idx] = 1.0
            if len(class_preds) == 0:
                aps.append(0.0)
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(npos, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
            aps.append(average_precision(rec, prec))
        result[f"mAP@{threshold:.1f}"] = 100.0 * float(np.mean(aps)) if aps else 0.0
    result["avg"] = float(np.mean([result[f"mAP@{t:.1f}"] for t in thresholds]))
    return result


def first_existing(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Missing expected file; tried: " + ", ".join(str(p) for p in candidates))


def split_paths(annotation_root: Path, dataset: str, split: int, split_dir: Path = None) -> Dict[str, Path]:
    if split_dir is None:
        if dataset == "wear":
            split_dir = annotation_root / f"outer_{split}_val_{(split + 1) % 18}_{(split + 2) % 18}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--detr-root", type=Path, default=DEFAULT_DETR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--dataset", default="wear")
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--point-seed", type=int, default=3407)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument(
        "--duration-prior",
        choices=["point_gap_global", "point_gap_class", "global", "class"],
        default="point_gap_global",
        help=(
            "Duration prior for turning classifier peaks into segments. "
            "point_gap_* modes use only point timestamps and record durations; "
            "global/class are accepted as backward-compatible aliases."
        ),
    )
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--topk-per-class", type=int, default=50)
    parser.add_argument("--keep-per-class", type=int, default=50)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-records", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_train_windows = args.max_train_windows or 64
        args.max_eval_records = args.max_eval_records or 2
        args.num_workers = 0

    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.annotation_root is None:
        args.annotation_root = args.project_root / ("wear_annotations_strict_map" if args.dataset == "wear" else "tal4har_annotations_psee_slimstad1d")
    args.output_dir = args.output_dir or (
        args.project_root
        / "output"
        / f"point_only_classifier_{args.dataset}_loso{args.split}{'_smoke' if args.smoke else ''}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = split_paths(args.annotation_root, args.dataset, args.split, args.split_dir)
    train_info = load_info(paths["train_info"])
    val_info = load_info(paths["val_info"])
    test_info = load_info(paths["test_info"])
    train_ann = load_annotations(paths["train_ann"])
    val_ann = load_annotations(paths["val_ann"])
    test_ann = load_annotations(paths["test_ann"])
    num_classes = max([a["label"] for a in train_ann + val_ann + test_ann]) + 1

    train_points = make_gaussian_points(train_ann, train_info, args.point_seed)
    priors, global_mean_duration = duration_priors(train_points, train_info, num_classes, args.duration_prior)
    train_videos = sorted(train_info.keys())
    val_videos = sorted(val_info.keys())
    test_videos = sorted(test_info.keys())

    dataset = PointWindowDataset(
        train_videos,
        train_points,
        train_info,
        args.feature_root,
        args.window,
        args.stride,
        max_windows=args.max_train_windows,
    )
    if len(dataset) == 0:
        raise RuntimeError("No point-supervised train windows were found.")
    sample_feature = load_feature(args.feature_root, train_videos[0])
    input_dim = int(sample_feature.shape[1])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    WearSecondEncoder = import_wear_second_encoder(args.detr_root)
    model = PointOnlyClassifier(WearSecondEncoder, input_dim, num_classes, args.hidden_dim, args.downsample).to(args.device)
    print(
        json.dumps(
            {
                "method": "point_only_classifier",
                "dataset": args.dataset,
                "split": args.split,
                "device": str(args.device),
                "num_classes": num_classes,
                "input_dim": input_dim,
                "num_train_windows": len(dataset),
                "protocol": {
                    "point_mode": "gaussian",
                    "point_seed": args.point_seed,
                    "train_seed": args.seed,
                    "window": args.window,
                    "stride": args.stride,
                    "downsample": args.downsample,
                    "point_radius_output_steps": args.point_radius,
                    "duration_prior": normalized_duration_prior_mode(args.duration_prior),
                    "duration_prior_source": "training_point_gap_proxy",
                    "global_mean_duration": global_mean_duration,
                    "no_gt_duration_prior": True,
                    "no_psee": True,
                    "no_reliability": True,
                    "no_transition": True,
                },
            },
            indent=2,
        ),
        flush=True,
    )

    train_classifier(model, loader, args)
    val_pred = predict_records(model, val_videos, val_info, args.feature_root, priors, args)
    test_pred = predict_records(model, test_videos, test_info, args.feature_root, priors, args)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    val_map = evaluate_map(val_pred, val_ann, thresholds, num_classes)
    test_map = evaluate_map(test_pred, test_ann, thresholds, num_classes)

    result = {
        "method": "point_only_classifier",
        "dataset": args.dataset,
        "split": args.split,
        "smoke": bool(args.smoke),
        "validation_mAP": val_map,
        "test_mAP": test_map,
        "test_average_mAP": test_map["avg"],
        "num_validation_predictions": len(val_pred),
        "num_test_predictions": len(test_pred),
        "duration_prior": priors.tolist(),
    }
    with (args.output_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (args.output_dir / "predictions_test.json").open("w") as f:
        json.dump(test_pred, f)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
