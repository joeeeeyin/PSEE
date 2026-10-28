#!/usr/bin/env python3
"""Adaptive gap-limited two-stage PointCls baseline with SlimSTAD-1D.

This is a low-hyperparameter counterpart to adaptive PSEE.  Stage 1 trains
only the point-neighborhood class classifier.  Pseudo boundaries are recovered
from the classifier CAM peak and Otsu support.  The CAM search is constrained
by neighboring point gaps rather than GT segment durations, which keeps this
baseline annotation-efficient without letting a weak classifier drift to a
far-away false peak.  It does not use GT duration statistics, dataset-specific
max durations, PSEE candidate ranking, or reliability weighting.  Stage 2
trains the same SlimSTAD-1D detector settings used by adaptive PSEE.
"""

import argparse
import copy
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from run_point_only_classifier_baseline import (
    PointOnlyClassifier,
    PointWindowDataset,
    collate,
    duration_priors,
    import_wear_second_encoder,
    load_annotations,
    load_feature,
    load_info,
    make_gaussian_points,
    normalize_feature,
    segment_iou,
    set_seed,
    split_paths,
    train_classifier,
)
from run_point_transition_classifier_baseline import transition_score


PROJECT_ROOT = Path(os.environ.get("STAD_PROJECT", "/home/yinjiaxi/pointsupervision/STAD-main"))
DETR_ROOT = Path(os.environ.get("DETR_PROJECT", "/home/yinjiaxi/pointsupervision/detr_e2e_tad"))
PYTHON = Path(os.environ.get("SLIMSTAD_PYTHON", "/home/yinjiaxi/miniconda3/envs/tal_for_har/bin/python"))
DATA_ROOT = Path(os.environ.get("POINTSUP_DATA_ROOT", "/data/yinjiaxi/pointsupervision_data"))
OLD_PROJECT_ROOT = "/home/yinjiaxi/SlimSTAD/STAD-main"
OLD_DETR_ROOT = "/home/yinjiaxi/detr_e2e_tad"
OLD_DATA_ROOT = "/data2/yinjiaxi"
TAG = "pointcls_twostage_adaptive_gap_gaussian_slimstad1d_20260902"


def add_runtime_library_paths(env: Dict[str, str], python_path: Path) -> None:
    env_root = Path(python_path).expanduser().resolve().parent.parent
    extra_paths = [env_root / "lib"]
    extra_paths.extend((env_root / "lib").glob("python*/site-packages/torch/lib"))
    existing = env.get("LD_LIBRARY_PATH", "")
    parts = [str(p) for p in extra_paths if p.exists()]
    if existing:
        parts.append(existing)
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)

FEATURE_ROOTS = {
    "wear": DATA_ROOT / "wear/processed/inertial_features/50_samples_50_overlap",
    "sbhar": DATA_ROOT / "imwut24_tal4har_data/sbhar/processed/inertial_features/50_samples_50_overlap",
    "wetlab": DATA_ROOT / "imwut24_tal4har_data/wetlab/processed/inertial_features/50_samples_50_overlap",
    "rwhar": DATA_ROOT / "imwut24_tal4har_data/rwhar/processed/inertial_features/50_samples_50_overlap",
    "xrfv2": DATA_ROOT / "xrfv2-multimodal-tal-caption-qa-no-rgb/processed/slimstad_imu30",
}

ANNOTATION_ROOTS = {
    "wear": PROJECT_ROOT / "wear_annotations_strict_map",
    "sbhar": PROJECT_ROOT / "tal4har_annotations_psee_slimstad1d",
    "wetlab": PROJECT_ROOT / "tal4har_annotations_psee_slimstad1d",
    "rwhar": PROJECT_ROOT / "tal4har_annotations_psee_slimstad1d",
    "xrfv2": PROJECT_ROOT / "xrfv2_annotations_weak_v2" / "imu30",
}

SPLIT_COUNTS = {
    "wear": 18,
    "sbhar": 30,
    "wetlab": 22,
    "rwhar": 15,
    "xrfv2": 16,
}

METHOD_NAMES = {
    "pointcls": "PointCls-2S-AdaptiveGap",
    "pointclst": "PointCls-T-2S-AdaptiveGap",
}


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    mkdir(path.parent)
    fields = ["video", "type", "type_idx", "start", "end", "startFrame", "endFrame"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: dict) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


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


def frame_from_coordinate(value: float, video_info: dict) -> float:
    """Convert the detector coordinate used in CSV start/end to raw-frame field.

    STAD's loader ignores the CSV start/end columns during detector training and
    reconstructs the target coordinate as startFrame * sample_count / count.
    Therefore this inverse conversion preserves exactly the pseudo coordinate.
    """
    count = float(video_info.get("count", 0.0))
    sample_count = float(video_info.get("sample_count", 0.0))
    if count <= 0 or sample_count <= 0:
        return float(value)
    return float(value) * count / sample_count


def bounded_argmax(scores: np.ndarray, lo_idx: int, hi_idx: int, fallback_idx: int) -> int:
    lo_idx = max(0, min(len(scores) - 1, lo_idx))
    hi_idx = max(lo_idx + 1, min(len(scores), hi_idx))
    if hi_idx <= lo_idx:
        return max(0, min(len(scores) - 1, fallback_idx))
    return lo_idx + int(np.argmax(scores[lo_idx:hi_idx]))


def time_to_index(value: float, duration: float, out_len: int) -> int:
    idx = int(round(value / max(duration, 1e-6) * max(out_len - 1, 1)))
    return max(0, min(out_len - 1, idx))


def index_to_time(index: int, duration: float, out_len: int) -> float:
    return float(index + 0.5) / max(float(out_len), 1.0) * duration


def otsu_threshold(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return float(np.mean(vals)) if vals.size else 0.0
    v_min, v_max = float(vals.min()), float(vals.max())
    if abs(v_max - v_min) < 1e-8:
        return float(np.median(vals))
    bins = int(np.clip(np.sqrt(vals.size), 8, 64))
    hist, edges = np.histogram(vals, bins=bins, range=(v_min, v_max))
    prob = hist.astype(np.float64) / float(max(int(hist.sum()), 1))
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return float(centers[int(np.nanargmax(between))])


def cam_support_interval(cam_cls: np.ndarray, center_idx: int, left_limit: int, right_limit: int) -> Tuple[int, int, float]:
    if right_limit <= left_limit + 1:
        return int(left_limit), int(max(left_limit + 1, right_limit)), 0.0
    local = cam_cls[left_limit:right_limit]
    threshold = otsu_threshold(local)
    left = int(center_idx)
    while left > left_limit and cam_cls[left - 1] >= threshold:
        left -= 1
    right = int(center_idx) + 1
    while right < right_limit and cam_cls[right] >= threshold:
        right += 1
    if right <= left:
        left = max(left_limit, int(center_idx) - 1)
        right = min(right_limit, int(center_idx) + 2)
    return int(left), int(right), float(threshold)


def point_ownership_limits(ordered: Sequence[dict], idx: int, duration: float) -> Tuple[float, float]:
    center = float(ordered[idx]["time"])
    left = 0.0 if idx == 0 else 0.5 * (float(ordered[idx - 1]["time"]) + center)
    right = duration if idx + 1 == len(ordered) else 0.5 * (center + float(ordered[idx + 1]["time"]))
    left = max(0.0, min(left, duration))
    right = max(left + 1e-3, min(right, duration))
    return left, right


def nearest_point_gap(ordered: Sequence[dict], idx: int, fallback: float, duration: float) -> float:
    center = float(ordered[idx]["time"])
    gaps = []
    if idx > 0:
        gaps.append(max(center - float(ordered[idx - 1]["time"]), 0.0))
    if idx + 1 < len(ordered):
        gaps.append(max(float(ordered[idx + 1]["time"]) - center, 0.0))
    if gaps:
        return max(min(gaps), 1e-3)
    if fallback and np.isfinite(fallback):
        return max(float(fallback), 1e-3)
    return max(float(duration), 1e-3)


def adaptive_point_gap_stats(points: Dict[str, List[dict]], info: Dict[str, dict]) -> dict:
    gaps = []
    ownerships = []
    for video, video_points in points.items():
        ordered = sorted(video_points, key=lambda item: float(item["time"]))
        duration = max(float(info.get(video, {}).get("duration", 0.0)), 1e-6)
        for idx, point in enumerate(ordered):
            center = float(point["time"])
            local_gaps = []
            if idx > 0:
                local_gaps.append(max(center - float(ordered[idx - 1]["time"]), 0.0))
            if idx + 1 < len(ordered):
                local_gaps.append(max(float(ordered[idx + 1]["time"]) - center, 0.0))
            if local_gaps:
                gaps.append(min(local_gaps))
            lo, hi = point_ownership_limits(ordered, idx, duration)
            ownerships.append(hi - lo)
    return {
        "point_gap_median": float(np.median(gaps)) if gaps else None,
        "point_gap_mean": float(np.mean(gaps)) if gaps else None,
        "ownership_median": float(np.median(ownerships)) if ownerships else None,
        "ownership_mean": float(np.mean(ownerships)) if ownerships else None,
        "num_points": int(sum(len(v) for v in points.values())),
    }


def snap_to_transition(
    target: float,
    duration: float,
    transition: np.ndarray,
    lo_time: float,
    hi_time: float,
) -> float:
    if len(transition) == 0 or hi_time <= lo_time:
        return float(np.clip(target, lo_time, hi_time))
    lo_idx = time_to_index(lo_time, duration, len(transition))
    hi_idx = time_to_index(hi_time, duration, len(transition)) + 1
    best = bounded_argmax(transition, lo_idx, hi_idx, time_to_index(target, duration, len(transition)))
    return float(best / max(len(transition) - 1, 1) * duration)


def localizer_predictions_for_points(
    method: str,
    model: torch.nn.Module,
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    feature_root: Path,
    point_gap_stats: dict,
    args: argparse.Namespace,
) -> Tuple[List[dict], Dict[str, dict]]:
    model.eval()
    pseudo_rows: List[dict] = []
    diagnostics = {
        "segments_per_record": {},
        "used_transition": method == "pointclst",
    }

    with torch.no_grad():
        for video in sorted(points):
            raw_feat = load_feature(feature_root, video)
            feat = normalize_feature(raw_feat)
            duration = max(float(info[video]["duration"]), 1e-6)
            tensor = torch.from_numpy(feat).unsqueeze(0).to(args.device)
            logits = model(tensor)
            scores = torch.sigmoid(logits)[0].detach().cpu().numpy()
            out_len = scores.shape[1]
            trans = None
            if method == "pointclst":
                trans = transition_score(raw_feat, args.transition_half_window, args.transition_smooth_radius)

            ordered = sorted(points[video], key=lambda item: float(item["time"]))
            diagnostics["segments_per_record"][video] = len(ordered)
            for idx, point in enumerate(ordered):
                label = int(point["label"])
                center_hint = float(point["time"])
                left_limit, right_limit = point_ownership_limits(ordered, idx, duration)
                center_idx = time_to_index(center_hint, duration, out_len)
                gap = nearest_point_gap(
                    ordered,
                    idx,
                    point_gap_stats.get("point_gap_median"),
                    duration,
                )
                search_radius = max(float(args.min_peak_search), float(args.peak_search_ratio) * gap)
                search_left = max(left_limit, center_hint - search_radius)
                search_right = min(right_limit, center_hint + search_radius)
                left_idx = time_to_index(search_left, duration, out_len)
                right_idx = time_to_index(search_right, duration, out_len) + 1
                left_idx = int(np.clip(left_idx, 0, max(center_idx, 0)))
                right_idx = int(np.clip(right_idx, min(center_idx + 1, out_len), out_len))
                if right_idx <= left_idx + 1:
                    left_idx = max(0, center_idx - 1)
                    right_idx = min(out_len, center_idx + 2)

                point_idx = center_idx
                center_idx = bounded_argmax(scores[label], left_idx, right_idx, point_idx)
                center_time = index_to_time(center_idx, duration, out_len)
                cam_l, cam_r, cam_thr = cam_support_interval(scores[label], center_idx, left_idx, right_idx)
                cam_l = min(cam_l, point_idx)
                cam_r = max(cam_r, point_idx + 1)
                start = max(left_limit, float(cam_l) / max(float(out_len), 1.0) * duration)
                end = min(right_limit, float(cam_r) / max(float(out_len), 1.0) * duration)
                if end <= start:
                    start = max(left_limit, float(point_idx) / max(float(out_len), 1.0) * duration)
                    end = min(right_limit, float(point_idx + 1) / max(float(out_len), 1.0) * duration)

                if method == "pointclst" and trans is not None:
                    seg_len = max(end - start, 1e-3)
                    radius = max(args.min_boundary_search_sec, args.boundary_search_ratio * seg_len)
                    snapped_start = snap_to_transition(
                        start,
                        duration,
                        trans,
                        max(left_limit, start - radius),
                        min(center_time, start + radius),
                    )
                    snapped_end = snap_to_transition(
                        end,
                        duration,
                        trans,
                        max(center_time, end - radius),
                        min(right_limit, end + radius),
                    )
                    if snapped_end > snapped_start:
                        start, end = snapped_start, snapped_end

                pseudo_rows.append(
                    {
                        "video": video,
                        "label": label,
                        "start": float(start),
                        "end": float(end),
                        "score": float(scores[label, center_idx]),
                        "point_score": float(scores[label, point_idx]),
                        "cam_peak_time": float(index_to_time(center_idx, duration, out_len)),
                        "point_time": center_hint,
                        "annotation_index": int(point["annotation_index"]),
                        "point_ratio": float(point.get("point_ratio", 0.5)),
                        "cam_threshold_otsu": float(cam_thr),
                        "ownership_left": float(left_limit),
                        "ownership_right": float(right_limit),
                        "nearest_point_gap": float(gap),
                        "search_left": float(search_left),
                        "search_right": float(search_right),
                        "search_radius": float(search_radius),
                    }
                )
    return pseudo_rows, diagnostics


def pseudo_quality(
    pseudo: Sequence[dict],
    train_annotations: Sequence[dict],
) -> Dict[str, float]:
    gt_by_video = group_annotations(train_annotations)
    ious = []
    start_errors = []
    end_errors = []
    ratio_errors = []
    for row in pseudo:
        gt_list = gt_by_video.get(row["video"], [])
        ann_idx = int(row.get("annotation_index", -1))
        if ann_idx < 0 or ann_idx >= len(gt_list):
            continue
        gt = gt_list[ann_idx]
        gt_segment = (float(gt["start"]), float(gt["end"]))
        pred_segment = (float(row["start"]), float(row["end"]))
        gt_len = max(gt_segment[1] - gt_segment[0], 1e-6)
        pred_len = max(pred_segment[1] - pred_segment[0], 1e-6)
        iou = segment_iou(pred_segment, gt_segment)
        ious.append(iou)
        start_errors.append(abs(pred_segment[0] - gt_segment[0]) / gt_len)
        end_errors.append(abs(pred_segment[1] - gt_segment[1]) / gt_len)
        ratio_errors.append(abs(pred_len / gt_len - 1.0))
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
        "start_error_over_duration": float(np.mean(start_errors)),
        "end_error_over_duration": float(np.mean(end_errors)),
        "duration_ratio_error": float(np.mean(ratio_errors)),
    }


def export_pseudo_csv(
    method: str,
    dataset: str,
    split: int,
    pseudo: Sequence[dict],
    points: Dict[str, List[dict]],
    info: Dict[str, dict],
    train_annotations: Sequence[dict],
    pseudo_root: Path,
) -> Tuple[Path, Path]:
    gt_by_video = group_annotations(train_annotations)
    out_dir = mkdir(pseudo_root / dataset / f"loso{split}")
    csv_name = "train_annotations_pointcls_000.csv" if method == "pointcls" else "train_annotations_pointclst_000.csv"
    csv_path = out_dir / csv_name
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
    meta_path = out_dir / ("pointcls_pseudo_quality.json" if method == "pointcls" else "pointclst_pseudo_quality.json")
    meta = {
        "dataset": dataset,
        "split": split,
        "method": METHOD_NAMES[method],
        "stage": "pseudo_boundary_generation",
        "point_sampling": "gaussian",
        "point_seed": 3407,
        "boundary_source": "point_classifier_gap_limited_cam_peak_otsu_support",
        "duration_prior_source": "disabled",
        "duration_limits": "not_used",
        "point_gap_constraint": "nearest annotated point gap only constrains CAM peak search",
        "transition_used": method == "pointclst",
        "no_gt_duration_prior": True,
        "reliability_weighted_detector": False,
        "pseudo_count": len(rows),
        "train_point_count": int(sum(len(v) for v in points.values())),
        **pseudo_quality(pseudo, train_annotations),
    }
    write_json(meta_path, meta)
    return csv_path, meta_path


def find_template_config(dataset: str, split: int, project_root: Path) -> Path:
    if dataset == "wear":
        candidates = [
            project_root
            / "configs"
            / "psee_clean_20260715_point_ablation"
            / f"psee_clean_20260715_point_gaussian_wear_loso{split}_b000.yaml",
        ]
    elif dataset == "xrfv2":
        candidates = [
            project_root
            / "configs"
            / "psee_clean_20260715_point_ablation"
            / f"psee_clean_20260715_point_gaussian_xrfv2_imu30_loso{split}_b000.yaml",
        ]
    else:
        dated = {
            "sbhar": "20260809",
            "rwhar": "20260811",
            "wetlab": "20260812",
        }[dataset]
        candidates = [
            project_root
            / "configs"
            / f"tal4har_sfnet_style_gaussian_slimstad1d_{dataset}_clean_yjx163_{dated}"
            / f"tal4har_sfnet_style_gaussian_slimstad1d_{dataset}_clean_yjx163_{dated}_{dataset}_loso{split}_b000.yaml",
        ]
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


def remap_config_paths(obj, args: argparse.Namespace):
    if isinstance(obj, dict):
        return {k: remap_config_paths(v, args) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_config_paths(v, args) for v in obj]
    if isinstance(obj, str):
        return (
            obj.replace(OLD_PROJECT_ROOT, str(args.project_root))
            .replace(OLD_DETR_ROOT, str(args.detr_root))
            .replace(OLD_DATA_ROOT, str(DATA_ROOT))
        )
    return obj


def build_detector_config(
    method: str,
    dataset: str,
    split: int,
    pseudo_csv: Path,
    args: argparse.Namespace,
) -> Path:
    template = find_template_config(dataset, split, args.project_root)
    cfg = remap_config_paths(yaml.safe_load(template.read_text()), args)
    run_name = f"{TAG}_{method}_{dataset}_loso{split}_b000"
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
    cfg["training"].pop("psee_reliability_path", None)

    cfg["testing"]["checkpoint_path"] = str(args.project_root / "models" / run_name)
    cfg["testing"]["output_path"] = str(args.project_root / "output" / run_name)
    cfg["testing"]["output_json"] = f"{run_name}.json"

    cfg["protocol"] = {
        **(cfg.get("protocol") or {}),
        "method": METHOD_NAMES[method],
        "detector": "SlimSTAD-1D",
        "two_stage": True,
        "point_sampling": "gaussian",
        "point_seed": 3407,
        "pseudo_boundary_source": "adaptive_gap_point_classifier_cam_peak_otsu_support"
        if method == "pointcls"
        else "adaptive_gap_point_classifier_cam_peak_otsu_support_plus_transition",
        "adaptive_pointcls_2s": {
            "localizer": "point-neighborhood class classifier",
            "boundary_rule": "Class CAM peak inside a nearest-point-gap-limited local search interval, followed by Otsu connected support; force the selected segment to contain the annotated point",
            "uses_transition": method == "pointclst",
            "uses_psee_candidate_ranking": False,
            "uses_reliability_weights": False,
        },
        "duration_prior_source": "disabled",
        "duration_limits": "not_used",
        "point_gap_constraint": "used only as a local search bound, not as a fixed pseudo duration",
        "no_gt_duration_prior": True,
        "reliability_weighted_detector": False,
        "fair_template_config": str(template),
        "detector_batch_size": 2,
        "detector_learning_rate": 5e-5,
        "detector_weight_decay": 1e-4,
        "detector_max_epoch": int(args.detector_epochs),
        "detector_seed": 2020,
    }

    config_path = mkdir(args.project_root / "configs" / TAG) / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config_path


def generate_pseudo_and_config(method: str, dataset: str, split: int, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    if dataset == "xrfv2":
        paths = split_paths(
            ANNOTATION_ROOTS[dataset],
            dataset,
            split,
            split_dir=ANNOTATION_ROOTS[dataset] / f"loso{split}",
        )
    else:
        paths = split_paths(ANNOTATION_ROOTS[dataset], dataset, split)
    train_info = load_info(paths["train_info"])
    train_annotations = load_annotations(paths["train_ann"])
    val_annotations = load_annotations(paths["val_ann"])
    test_annotations = load_annotations(paths["test_ann"])
    num_classes = class_count_from_annotations(train_annotations, val_annotations, test_annotations)
    points = make_gaussian_points(train_annotations, train_info, args.point_seed)
    point_gap_stats = adaptive_point_gap_stats(points, train_info)

    train_videos = sorted(train_info)
    sample_video = train_videos[0]
    input_dim = int(load_feature(FEATURE_ROOTS[dataset], sample_video).shape[1])
    WearSecondEncoder = import_wear_second_encoder(args.detr_root)
    model = PointOnlyClassifier(
        WearSecondEncoder,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        downsample=args.downsample,
    ).to(args.device)

    train_dataset = PointWindowDataset(
        train_videos,
        points,
        train_info,
        FEATURE_ROOTS[dataset],
        window=args.classifier_window,
        stride=args.classifier_stride,
        max_windows=args.max_train_windows,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    local_args = copy.copy(args)
    local_args.window = args.classifier_window
    local_args.point_radius = args.point_radius
    local_args.epochs = args.classifier_epochs
    local_args.batch_size = args.classifier_batch_size
    local_args.lr = args.classifier_lr
    local_args.weight_decay = args.classifier_weight_decay
    local_args.grad_clip_norm = args.grad_clip_norm
    print(
        f"[stage1] {METHOD_NAMES[method]} {dataset} loso{split}: "
        f"windows={len(train_dataset)} points={sum(len(v) for v in points.values())} "
        f"adaptive_gap_median={point_gap_stats.get('point_gap_median')}",
        flush=True,
    )
    train_classifier(model, loader, local_args)

    pseudo, diagnostics = localizer_predictions_for_points(
        method,
        model,
        points,
        train_info,
        FEATURE_ROOTS[dataset],
        point_gap_stats,
        args,
    )
    diagnostics["adaptive_point_gap_stats"] = point_gap_stats
    pseudo_root = args.project_root / f"{TAG}_pseudo_point_gaussian_seed{args.point_seed}"
    pseudo_csv, quality_json = export_pseudo_csv(
        method,
        dataset,
        split,
        pseudo,
        points,
        train_info,
        train_annotations,
        pseudo_root,
    )
    diag_path = quality_json.with_name(quality_json.stem + "_diagnostics.json")
    write_json(diag_path, diagnostics)
    config_path = build_detector_config(method, dataset, split, pseudo_csv, args)
    return pseudo_csv, quality_json, config_path


def run_detector(config_path: Path, args: argparse.Namespace) -> dict:
    cmd = [
        str(args.python),
        "all_strict_map_psee.py",
        str(config_path),
        "--ngpu",
        "1",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    add_runtime_library_paths(env, args.python)
    print("[stage2]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(args.project_root), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"SlimSTAD1D detector failed with code {proc.returncode}: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    result_path = Path(cfg["testing"]["output_path"]) / "strict_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing strict_result.json: {result_path}")
    return json.loads(result_path.read_text())


def task_summary_path(args: argparse.Namespace, method: str, dataset: str, split: int) -> Path:
    suffix = "_smoke" if args.smoke else ""
    return args.project_root / "logs" / TAG / f"{method}_{dataset}_loso{split}{suffix}_task_summary.json"


def task_key(method: str, dataset: str, split: int) -> Tuple[str, str, int]:
    return (method, dataset, int(split))


def load_existing_results(args: argparse.Namespace, tasks: Sequence[Tuple[str, str, int]]) -> List[dict]:
    results = []
    for method, dataset, split in tasks:
        path = task_summary_path(args, method, dataset, split)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("strict_result"):
            results.append(data)
    return results


def active_run_tasks(project_root: Path) -> set:
    """Return run-task jobs already alive for this script.

    This lets a restarted queue avoid duplicating worker processes that were
    orphaned intentionally when switching from a smaller GPU set to a larger one.
    """
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
    pattern = re.compile(
        r"run_pointcls_twostage_adaptive_slimstad1d_queue\.py\s+run-task.*?"
        r"--method\s+(\S+).*?--dataset\s+(\S+).*?--split\s+(\d+)"
    )
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
            active.add(task_key(match.group(1), match.group(2), int(match.group(3))))
    return active


def run_task(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pseudo_csv, quality_json, config_path = generate_pseudo_and_config(args.method, args.dataset, args.split, args)
    strict = run_detector(config_path, args)
    quality = json.loads(quality_json.read_text())
    out = {
        "method": METHOD_NAMES[args.method],
        "method_key": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "smoke": bool(args.smoke),
        "pseudo_csv": str(pseudo_csv),
        "pseudo_quality": quality,
        "detector_config": str(config_path),
        "strict_result": strict,
    }
    write_json(task_summary_path(args, args.method, args.dataset, args.split), out)
    print("TASK_RESULT", json.dumps(out, sort_keys=True), flush=True)


def split_list_for_dataset(dataset: str, smoke: bool, explicit: Sequence[int]) -> List[int]:
    if explicit:
        return list(explicit)
    if smoke:
        return [0]
    return list(range(SPLIT_COUNTS[dataset]))


def make_tasks(args: argparse.Namespace) -> List[Tuple[str, str, int]]:
    tasks = []
    for method in args.methods:
        for dataset in args.datasets:
            for split in split_list_for_dataset(dataset, args.smoke, args.splits):
                tasks.append((method, dataset, split))
    return tasks


def launch_queue(args: argparse.Namespace) -> None:
    log_root = mkdir(args.project_root / "logs" / TAG)
    all_tasks = make_tasks(args)
    total = len(all_tasks)
    existing_results = load_existing_results(args, all_tasks)
    done_keys = {task_key(item["method_key"], item["dataset"], int(item["split"])) for item in existing_results}
    active_keys = active_run_tasks(args.project_root)
    tasks = deque([task for task in all_tasks if task_key(*task) not in done_keys and task_key(*task) not in active_keys])
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        gpus = ["0"]
    max_parallel = max(1, len(gpus) * max(1, args.workers_per_gpu))
    running = []
    failures = []
    print(
        f"[queue] tasks={total} completed={len(done_keys)} active_external={len(active_keys - done_keys)} "
        f"queued={len(tasks)} gpus={gpus} max_parallel={max_parallel} smoke={args.smoke}",
        flush=True,
    )

    while tasks or running:
        while tasks and len(running) < max_parallel:
            method, dataset, split = tasks.popleft()
            if task_summary_path(args, method, dataset, split).exists():
                continue
            if task_key(method, dataset, split) in active_run_tasks(args.project_root):
                continue
            gpu_load = {gpu_id: 0 for gpu_id in gpus}
            for active in running:
                gpu_load[active["gpu"]] = gpu_load.get(active["gpu"], 0) + 1
            gpu = min(gpus, key=lambda gpu_id: (gpu_load.get(gpu_id, 0), gpus.index(gpu_id)))
            task_name = f"{method}_{dataset}_loso{split}{'_smoke' if args.smoke else ''}"
            log_path = log_root / f"{task_name}.log"
            cmd = [
                str(args.python),
                str(Path(__file__).resolve()),
                "run-task",
                "--method",
                method,
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
            add_runtime_library_paths(env, args.python)
            with log_path.open("w") as log:
                proc = subprocess.Popen(cmd, cwd=str(args.project_root), env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append({"proc": proc, "task": (method, dataset, split), "log": log_path, "gpu": gpu})
            print(f"[queue] started {task_name} gpu={gpu} log={log_path}", flush=True)

        time.sleep(args.poll_interval)
        still_running = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                still_running.append(item)
                continue
            method, dataset, split = item["task"]
            summary = task_summary_path(args, method, dataset, split)
            if code == 0 and summary.exists():
                data = json.loads(summary.read_text())
                avg = float(data["strict_result"].get("test_average_mAP", 0.0))
                print(f"[queue] done {method} {dataset} loso{split} avg={avg:.6f}", flush=True)
            else:
                failure = {
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "returncode": code,
                    "log": str(item["log"]),
                }
                failures.append(failure)
                print(f"[queue] failed {failure}", flush=True)
        running = still_running

        results = load_existing_results(args, all_tasks)
        completed = {task_key(item["method_key"], item["dataset"], int(item["split"])) for item in results}
        external_active = active_run_tasks(args.project_root) - {
            task_key(item["task"][0], item["task"][1], item["task"][2]) for item in running
        }
        external_active = external_active - completed
        finished = (not tasks and not running and not external_active)
        queue_summary = summarize_results(args, results, failures, total, finished=finished)
        write_json(args.summary_path, queue_summary)
        if failures and args.stop_on_failure:
            raise RuntimeError(f"Stopping after failure: {failures[-1]}")
        if not tasks and not running and external_active:
            print(f"[queue] waiting_for_external_active={len(external_active)}", flush=True)
            time.sleep(args.poll_interval)


def summarize_results(
    args: argparse.Namespace,
    results: Sequence[dict],
    failures: Sequence[dict],
    total_tasks: int,
    finished: bool,
) -> dict:
    by_group: Dict[str, List[float]] = defaultdict(list)
    rows = []
    for item in results:
        strict = item.get("strict_result", {})
        maps = [float(x) for x in strict.get("test_mAP", [])]
        avg = float(strict.get("test_average_mAP", 0.0))
        key = f"{item['method_key']}::{item['dataset']}"
        by_group[key].append(avg)
        rows.append(
            {
                "method": item["method"],
                "method_key": item["method_key"],
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
    means = {}
    for key, vals in by_group.items():
        means[key] = {
            "n": len(vals),
            "avg_mean": float(np.mean(vals)) if vals else None,
            "avg_std": float(np.std(vals)) if vals else None,
        }
    return {
        "tag": TAG,
        "smoke": bool(args.smoke),
        "finished": bool(finished),
        "total_tasks": total_tasks,
        "completed_tasks": len(results),
        "failed_tasks": len(failures),
        "rows": sorted(rows, key=lambda x: (x["dataset"], x["method_key"], x["split"])),
        "group_summary": means,
        "failures": list(failures),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


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
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--peak-search-ratio", type=float, default=0.5)
    parser.add_argument("--min-peak-search", type=float, default=2.0)
    parser.add_argument("--transition-half-window", type=int, default=8)
    parser.add_argument("--transition-smooth-radius", type=int, default=2)
    parser.add_argument("--boundary-search-ratio", type=float, default=0.25)
    parser.add_argument("--min-boundary-search-sec", type=float, default=2.0)
    parser.add_argument("--min-duration-ratio", type=float, default=0.25)
    parser.add_argument("--smoke", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    task_parser = sub.add_parser("run-task")
    add_common_args(task_parser)
    task_parser.add_argument("--method", choices=sorted(METHOD_NAMES), required=True)
    task_parser.add_argument("--dataset", choices=sorted(FEATURE_ROOTS), required=True)
    task_parser.add_argument("--split", type=int, required=True)

    queue_parser = sub.add_parser("queue")
    add_common_args(queue_parser)
    queue_parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_NAMES), default=["pointcls"])
    queue_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(FEATURE_ROOTS),
        default=["wear", "wetlab", "sbhar", "rwhar", "xrfv2"],
    )
    queue_parser.add_argument("--splits", nargs="*", type=int, default=[])
    queue_parser.add_argument("--gpus", default="0")
    queue_parser.add_argument("--workers-per-gpu", type=int, default=1)
    queue_parser.add_argument("--poll-interval", type=float, default=20.0)
    queue_parser.add_argument("--stop-on-failure", action="store_true")
    queue_parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "logs" / f"{TAG}_summary.json")

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
