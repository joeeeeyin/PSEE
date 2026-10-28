#!/usr/bin/env python3
"""Export a parameter-light PSEE variant for SlimSTAD detector training.

This script intentionally leaves the frozen PSEE/PSEE-v2 exporters untouched.
The low-parameter variant removes class-duration priors, dataset-specific
maximum duration constraints, and fixed multi-scale expansion factors. Pseudo
segments are selected from transition-aligned ownership regions, CAM support,
and candidate consensus.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

DETR_PROJECT = Path(os.environ.get("DETR_PROJECT", "/home/yinjiaxi/detr_e2e_tad"))
SLIMSTAD_PROJECT = Path(os.environ.get("STAD_PROJECT", "/home/yinjiaxi/SlimSTAD/STAD-main"))
BASE_EXPORTER = DETR_PROJECT / "tools" / "export_psee_for_slimstad.py"
LOWPARAM_TAG = "psee_boundary_v1_20260905"
DEFAULT_POINT_SEED = 3407
# The submitted mainline uses one segment-level reliability for the detector.
# Boundary reliability remains an explicit legacy/diagnostic option.
RELIABILITY_MODE = "segment_reliability"

sys.path.insert(0, str(DETR_PROJECT))
from tad import weak_sensor_curve as curve  # noqa: E402


def load_base_exporter():
    spec = importlib.util.spec_from_file_location("export_psee_base", BASE_EXPORTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base exporter from {BASE_EXPORTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    key = "::".join(str(p) for p in (seed, *parts))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def sampled_point_time(start: float, end: float, mode: str, rng: np.random.Generator) -> Tuple[float, float]:
    duration = max(float(end) - float(start), 1e-6)
    if mode == "center":
        ratio = 0.5
    elif mode == "random":
        ratio = float(rng.uniform(0.05, 0.95))
    elif mode == "gaussian":
        ratio = float(np.clip(rng.normal(0.5, 1.0 / 6.0), 0.05, 0.95))
    else:
        raise ValueError(f"Unsupported point sampling mode: {mode}")
    return float(start) + ratio * duration, ratio


def make_point_annotations(mode: str, seed: int):
    def _points(record: dict) -> List[dict]:
        duration = max(float(record["duration"]), 1e-6)
        points: List[dict] = []
        for idx, ann in enumerate(record["annotations"]):
            start = float(ann["segment"][0])
            end = float(ann["segment"][1])
            rng = stable_rng(seed, record["video_id"], idx, ann["label"], mode)
            t, ratio = sampled_point_time(start, end, mode, rng)
            points.append(
                {
                    "time": t,
                    "position": float(t / duration),
                    "label": int(ann["label"]),
                    "annotation_index": idx,
                    "point_ratio": float(ratio),
                    "sampling": mode,
                    "sampling_seed": int(seed),
                }
            )
        return points

    return _points


def to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def unique_sorted_ints(values: Iterable[int], lo: int, hi: int) -> List[int]:
    out = sorted({int(np.clip(v, lo, hi)) for v in values})
    return out


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
    total = max(int(hist.sum()), 1)
    prob = hist.astype(np.float64) / float(total)
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(between))
    return float(centers[idx])


def max_transition_index(transition: np.ndarray, lo: int, hi: int, fallback: int) -> int:
    n = int(len(transition))
    if n <= 0:
        return int(fallback)
    lo = int(np.clip(lo, 0, n - 1))
    hi = int(np.clip(hi, lo + 1, n))
    if hi <= lo:
        return int(np.clip(fallback, 0, n - 1))
    return int(lo + np.argmax(transition[lo:hi]))


def boundary_between_points(points: Sequence[dict], left_i: int, right_i: int, transition: np.ndarray, scale: float) -> int:
    n = len(transition)
    left_idx = int(np.clip(round(float(points[left_i]["time"]) * scale), 0, max(n - 1, 0)))
    right_idx = int(np.clip(round(float(points[right_i]["time"]) * scale), 0, max(n - 1, 0)))
    if right_idx <= left_idx + 1:
        return int(np.clip(round((left_idx + right_idx) * 0.5), 0, max(n - 1, 0)))
    return max_transition_index(transition, left_idx + 1, right_idx, round((left_idx + right_idx) * 0.5))


def cam_support_interval(cam_cls: np.ndarray, center_idx: int, left_limit: int, right_limit: int) -> Tuple[int, int, float]:
    if right_limit <= left_limit + 1:
        return left_limit, max(left_limit + 1, right_limit), 0.0
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


def percentile_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    if float(np.nanmax(arr) - np.nanmin(arr)) < 1e-12:
        return np.ones(n, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def boundary_confidence(scores: Sequence[float]) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 1.0
    arr = np.sort(arr)[::-1]
    spread = float(np.std(arr))
    if spread < 1e-6:
        return 0.5
    return float(np.clip(sigmoid((arr[0] - arr[1]) / spread), 0.05, 1.0))


def side_boundary_scores(
    cam_cls: np.ndarray,
    transition_np: np.ndarray,
    side: str,
    side_candidates: Sequence[int],
    center_idx: int,
    left_limit: int,
    right_limit: int,
) -> List[dict]:
    rows: List[dict] = []
    n = int(len(transition_np))
    for boundary in side_candidates:
        b = int(boundary)
        if side == "left":
            inner = cam_cls[b : center_idx + 1]
            outer = cam_cls[left_limit:b]
            trans_idx = int(np.clip(b, 0, max(n - 1, 0)))
            owner_penalty = 0.0 if b >= left_limit else 1.0
        else:
            inner = cam_cls[center_idx:b]
            outer = cam_cls[b:right_limit]
            trans_idx = int(np.clip(b - 1, 0, max(n - 1, 0)))
            owner_penalty = 0.0 if b <= right_limit else 1.0
        sem = float(np.mean(inner)) if len(inner) else 0.0
        out = float(np.mean(outer)) if len(outer) else 0.0
        cont = float(max(0.0, sem - out))
        trans = float(transition_np[trans_idx]) if n else 0.0
        rows.append(
            {
                "boundary": b,
                "side": side,
                "S_sem": sem,
                "S_trans": trans,
                "S_cont": cont,
                "S_owner": owner_penalty,
            }
        )
    if not rows:
        return rows

    metric_names = ["S_sem", "S_trans", "S_cont"]
    metric_ranks = [percentile_ranks([row[name] for row in rows]) for name in metric_names]
    owner_penalty = np.asarray([row["S_owner"] for row in rows], dtype=np.float64)
    owner_score = 1.0 - np.clip(owner_penalty, 0.0, 1.0)
    for i, row in enumerate(rows):
        row["score"] = float(np.mean([ranks[i] for ranks in metric_ranks] + [owner_score[i]]))
    return rows


def interval_overlap(a: int, b: int, c: int, d: int) -> int:
    return max(0, min(b, d) - max(a, c))


def lowparam_psee_segment(
    cam,
    transition,
    point: dict,
    points: Sequence[dict],
    point_index: int,
    class_medians,
    global_median,
    record_duration: float,
    min_d: float,
    max_d: float,
) -> Tuple[float, float, float, dict]:
    del class_medians, global_median, min_d, max_d
    cam_np = to_numpy(cam)
    transition_np = to_numpy(transition).reshape(-1)
    if cam_np.ndim == 1:
        cam_np = cam_np.reshape(1, -1)
    out_len = int(len(transition_np))
    if out_len <= 1 or record_duration <= 0:
        return [0.0, float(record_duration)], 1.0, {"fallback": "degenerate_sequence"}

    scale = float(out_len) / max(float(record_duration), 1e-6)
    label = int(point["label"])
    label_idx = int(np.clip(label, 0, cam_np.shape[0] - 1))
    center_idx = int(np.clip(round(float(point["time"]) * scale), 0, out_len - 1))
    sorted_points = sorted(points, key=lambda p: float(p["time"]))

    if point_index > 0:
        left_limit = boundary_between_points(sorted_points, point_index - 1, point_index, transition_np, scale)
    else:
        left_limit = 0
    if point_index + 1 < len(sorted_points):
        right_limit = boundary_between_points(sorted_points, point_index, point_index + 1, transition_np, scale)
    else:
        right_limit = out_len
    left_limit = int(np.clip(left_limit, 0, max(center_idx - 1, 0)))
    right_limit = int(np.clip(right_limit, min(center_idx + 2, out_len), out_len))
    if right_limit <= left_limit + 1:
        left_limit, right_limit = max(0, center_idx - 1), min(out_len, center_idx + 2)

    cam_cls = cam_np[label_idx]
    cam_l, cam_r, cam_thr = cam_support_interval(cam_cls, center_idx, left_limit, right_limit)
    left_peak = max_transition_index(transition_np, left_limit, max(center_idx, left_limit + 1), left_limit)
    right_peak = max_transition_index(transition_np, center_idx + 1, right_limit, right_limit - 1)

    left_candidates = unique_sorted_ints(
        [left_limit, cam_l, left_peak],
        left_limit,
        max(center_idx - 1, left_limit),
    )
    right_candidates = unique_sorted_ints(
        [right_limit, cam_r, right_peak + 1],
        min(center_idx + 1, right_limit),
        right_limit,
    )

    candidates: List[dict] = []
    for left in left_candidates:
        for right in right_candidates:
            if left <= center_idx < right and right > left:
                inside = cam_cls[left:right]
                left_out = cam_cls[left_limit:left]
                right_out = cam_cls[right:right_limit]
                outside_parts = [x for x in (left_out, right_out) if len(x)]
                outside = float(np.mean(np.concatenate(outside_parts))) if outside_parts else 0.0
                trans_left = float(transition_np[left]) if 0 <= left < out_len else 0.0
                trans_right_idx = int(np.clip(right - 1, 0, out_len - 1))
                trans_right = float(transition_np[trans_right_idx])
                support_len = max(cam_r - cam_l, 1)
                support_cover = float(interval_overlap(left, right, cam_l, cam_r) / support_len)
                candidates.append(
                    {
                        "left": int(left),
                        "right": int(right),
                        "cam_inside": float(np.mean(inside)) if len(inside) else 0.0,
                        "cam_contrast": float((np.mean(inside) if len(inside) else 0.0) - outside),
                        "transition_alignment": 0.5 * (trans_left + trans_right),
                        "support_coverage": support_cover,
                    }
                )
    if not candidates:
        fallback_l = max(left_limit, center_idx - 1)
        fallback_r = min(right_limit, center_idx + 2)
        candidates = [
            {
                "left": int(fallback_l),
                "right": int(fallback_r),
                "cam_inside": float(cam_cls[center_idx]),
                "cam_contrast": float(cam_cls[center_idx]),
                "transition_alignment": 0.0,
                "support_coverage": 1.0,
            }
        ]

    metric_names = ["cam_inside", "cam_contrast", "transition_alignment", "support_coverage"]
    metric_ranks = [percentile_ranks([cand[name] for cand in candidates]) for name in metric_names]
    for i, cand in enumerate(candidates):
        cand["score"] = float(np.mean([ranks[i] for ranks in metric_ranks]))
    best = max(candidates, key=lambda c: (c["score"], c["cam_contrast"], c["transition_alignment"]))

    lefts = np.asarray([cand["left"] for cand in candidates], dtype=np.float64)
    rights = np.asarray([cand["right"] for cand in candidates], dtype=np.float64)
    span = max(float(right_limit - left_limit), 1.0)
    left_iqr = float(np.percentile(lefts, 75) - np.percentile(lefts, 25)) if len(lefts) > 1 else 0.0
    right_iqr = float(np.percentile(rights, 75) - np.percentile(rights, 25)) if len(rights) > 1 else 0.0
    agreement = float(np.clip(1.0 - (left_iqr + right_iqr) / span, 0.0, 1.0))
    score = float(np.clip(best["score"], 0.0, 1.0))
    segment_reliability = float((2.0 * score * agreement) / (score + agreement + 1e-6))
    segment_reliability = float(np.clip(segment_reliability, 1e-6, 1.0))

    selected = best
    left_confidence = segment_reliability
    right_confidence = segment_reliability
    left_boundary_rows: List[dict] = []
    right_boundary_rows: List[dict] = []
    if RELIABILITY_MODE == "boundary_reliability":
        left_boundary_rows = side_boundary_scores(
            cam_cls,
            transition_np,
            "left",
            left_candidates,
            center_idx,
            left_limit,
            right_limit,
        )
        right_boundary_rows = side_boundary_scores(
            cam_cls,
            transition_np,
            "right",
            right_candidates,
            center_idx,
            left_limit,
            right_limit,
        )
        if left_boundary_rows:
            left_best = max(left_boundary_rows, key=lambda item: (item["score"], item["S_trans"], item["S_cont"]))
        else:
            left_best = {"boundary": int(best["left"]), "score": score}
        if right_boundary_rows:
            right_best = max(right_boundary_rows, key=lambda item: (item["score"], item["S_trans"], item["S_cont"]))
        else:
            right_best = {"boundary": int(best["right"]), "score": score}
        selected = dict(best)
        selected["left"] = int(left_best["boundary"])
        selected["right"] = int(right_best["boundary"])
        if selected["left"] > center_idx:
            selected["left"] = int(max(left_limit, center_idx - 1))
        if selected["right"] <= center_idx:
            selected["right"] = int(min(right_limit, center_idx + 1))
        if selected["right"] <= selected["left"]:
            selected["left"] = int(best["left"])
            selected["right"] = int(best["right"])
        left_confidence = boundary_confidence([row["score"] for row in left_boundary_rows])
        right_confidence = boundary_confidence([row["score"] for row in right_boundary_rows])

    reliability = float(np.clip(0.5 * (left_confidence + right_confidence), 1e-6, 1.0))
    if RELIABILITY_MODE == "no_reliability":
        left_confidence = right_confidence = reliability = 1.0

    start = max(0.0, float(selected["left"]) / scale)
    end = min(float(record_duration), float(selected["right"]) / scale)
    if end <= start:
        end = min(float(record_duration), start + 1.0 / scale)

    components = {
        "method": "psee_lowparam_v1",
        "reliability_mode": RELIABILITY_MODE,
        "point_time": float(point["time"]),
        "label": label,
        "center_idx": int(center_idx),
        "ownership_left_idx": int(left_limit),
        "ownership_right_idx": int(right_limit),
        "cam_support_left_idx": int(cam_l),
        "cam_support_right_idx": int(cam_r),
        "cam_threshold_otsu": float(cam_thr),
        "left_transition_peak_idx": int(left_peak),
        "right_transition_peak_idx": int(right_peak),
        "num_candidates": int(len(candidates)),
        "selected_score": score,
        "candidate_agreement": agreement,
        "segment_reliability": segment_reliability,
        "left_confidence": float(left_confidence),
        "right_confidence": float(right_confidence),
        "reliability": reliability,
        "selected": {
            key: value
            for key, value in selected.items()
            if key in {"left", "right", "cam_inside", "cam_contrast", "transition_alignment", "support_coverage", "score"}
        },
        "left_boundary_candidates": sorted(
            [
                {
                    key: row[key]
                    for key in ("boundary", "score", "S_sem", "S_trans", "S_cont", "S_owner")
                }
                for row in left_boundary_rows
            ],
            key=lambda item: item["score"],
            reverse=True,
        )[:5],
        "right_boundary_candidates": sorted(
            [
                {
                    key: row[key]
                    for key in ("boundary", "score", "S_sem", "S_trans", "S_cont", "S_owner")
                }
                for row in right_boundary_rows
            ],
            key=lambda item: item["score"],
            reverse=True,
        )[:5],
        "removed_hyperparams": [
            "class_duration_prior",
            "global_duration_prior",
            "dataset_specific_max_duration",
            "fixed_duration_multipliers",
            "manual_metric_weights",
        ],
    }
    return [start, end], reliability, components


def boundary_error_fields(segment: Sequence[float], gt_segment: Sequence[float], row: dict) -> None:
    gt_start = float(gt_segment[0])
    gt_end = float(gt_segment[1])
    gt_dur = max(gt_end - gt_start, 1e-6)
    row["left_boundary_error"] = abs(float(segment[0]) - gt_start)
    row["right_boundary_error"] = abs(float(segment[1]) - gt_end)
    row["left_boundary_error_norm"] = row["left_boundary_error"] / gt_dur
    row["right_boundary_error_norm"] = row["right_boundary_error"] / gt_dur


def correlation(xs: Sequence[float], ys: Sequence[float]):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summary_with_boundary_diagnostics(base_summary, quality, weak_quality, selected_count, rows):
    out = base_summary(quality, weak_quality, selected_count, rows)
    left_conf = np.asarray([row.get("left_confidence", row.get("reliability", 1.0)) for row in rows], dtype=np.float64)
    right_conf = np.asarray([row.get("right_confidence", row.get("reliability", 1.0)) for row in rows], dtype=np.float64)
    left_err = np.asarray([row.get("left_boundary_error_norm", np.nan) for row in rows], dtype=np.float64)
    right_err = np.asarray([row.get("right_boundary_error_norm", np.nan) for row in rows], dtype=np.float64)
    out.update(
        {
            "mean_left_confidence": float(np.nanmean(left_conf)) if left_conf.size else None,
            "mean_right_confidence": float(np.nanmean(right_conf)) if right_conf.size else None,
            "median_left_confidence": float(np.nanmedian(left_conf)) if left_conf.size else None,
            "median_right_confidence": float(np.nanmedian(right_conf)) if right_conf.size else None,
            "mean_left_boundary_error_norm": float(np.nanmean(left_err)) if np.isfinite(left_err).any() else None,
            "mean_right_boundary_error_norm": float(np.nanmean(right_err)) if np.isfinite(right_err).any() else None,
            "left_confidence_error_correlation": correlation(left_conf, left_err),
            "right_confidence_error_correlation": correlation(right_conf, right_err),
        }
    )
    return out


def build_psee_records_boundary_aware(base, model, records, selected, device, modality, min_d, max_d):
    class_medians, global_median = base.duration_priors(records, selected)
    output = []
    reliability_rows = []
    quality = []
    weak_quality = []
    for record_index, record in enumerate(records):
        cam, transition = base.infer_weak_maps(model, record, modality, device)
        points = sorted(base.point_annotations(record), key=lambda item: item["time"])
        annotations = []
        for point_index, point in enumerate(points):
            ann_idx = int(point["annotation_index"])
            key = (record_index, ann_idx)
            gt_ann = record["annotations"][ann_idx]
            if key in selected:
                segment = list(gt_ann["segment"])
                reliability = 1.0
                components = {
                    "source": "true_boundary",
                    "left_confidence": 1.0,
                    "right_confidence": 1.0,
                    "segment_reliability": 1.0,
                    "reliability_mode": RELIABILITY_MODE,
                }
            else:
                segment, reliability, components = lowparam_psee_segment(
                    cam,
                    transition,
                    point,
                    points,
                    point_index,
                    class_medians,
                    global_median,
                    float(record["duration"]),
                    min_d,
                    max_d,
                )
                components["source"] = "psee"
            annotations.append({"segment": segment, "label": int(point["label"])})
            iou = base.segment_iou_np(segment, gt_ann["segment"])
            quality.append(iou)
            if key not in selected:
                weak_quality.append(iou)
            row = {
                "video": record["video_id"],
                "annotation_index": ann_idx,
                "label": int(point["label"]),
                "start": float(segment[0]),
                "end": float(segment[1]),
                "reliability": float(reliability),
                **components,
            }
            boundary_error_fields(segment, gt_ann["segment"], row)
            reliability_rows.append(row)
        record_out = {key: value for key, value in record.items() if key != "annotations"}
        record_out["annotations"] = annotations
        output.append(record_out)
    return output, reliability_rows, class_medians, global_median, np.asarray(quality), np.asarray(weak_quality)


def patch_train_localizer(base, tag: str, point_mode: str, point_seed: int, checkpoint_reuse: bool):
    original_train_localizer = curve.train_localizer
    variant_root = DETR_PROJECT / "outputs" / f"{tag}_point_{point_mode}_seed{point_seed}"

    def _train_localizer(args, records, labels, modality, output_dir):
        if checkpoint_reuse and getattr(args, "localizer_checkpoint", None):
            return original_train_localizer(args, records, labels, modality, output_dir)
        try:
            relative = Path(output_dir).relative_to(DETR_PROJECT / "outputs")
        except ValueError:
            relative = Path(str(output_dir).lstrip("/"))
        clean_dir = variant_root / relative
        args2 = copy.copy(args)
        args2.localizer_checkpoint = None
        clean_dir.mkdir(parents=True, exist_ok=True)
        model = original_train_localizer(args2, records, labels, modality, clean_dir)
        (clean_dir / "lowparam_manifest.json").write_text(
            json.dumps(
                {
                    "tag": tag,
                    "point_mode": point_mode,
                    "point_seed": point_seed,
                    "checkpoint_reuse": False,
                    "localizer_epochs": int(getattr(args2, "localizer_epochs", -1)),
                },
                indent=2,
            )
        )
        return model

    curve.train_localizer = _train_localizer
    base.train_localizer = _train_localizer


def patch_base_exporter(base, tag: str, point_mode: str, point_seed: int, checkpoint_reuse: bool, reliability_mode: str = "segment_reliability"):
    global RELIABILITY_MODE
    RELIABILITY_MODE = reliability_mode
    points_fn = make_point_annotations(point_mode, point_seed)
    curve.point_annotations = points_fn
    base.point_annotations = points_fn
    base.duration_priors = lambda records, selected: ({}, None)
    base.psee_segment = lowparam_psee_segment
    base.build_psee_records = lambda model, records, selected, device, modality, min_d, max_d: build_psee_records_boundary_aware(
        base, model, records, selected, device, modality, min_d, max_d
    )
    original_summary = base.summary
    base.summary = lambda quality, weak_quality, selected_count, rows: summary_with_boundary_diagnostics(
        original_summary, quality, weak_quality, selected_count, rows
    )
    base.PSEE_WEAR_ROOT = SLIMSTAD_PROJECT / f"wear_pseudo_strict_map_{tag}_point_{point_mode}_seed{point_seed}"
    base.PSEE_WIFI_ROOT = SLIMSTAD_PROJECT / f"wifi_pseudo_{tag}_point_{point_mode}_seed{point_seed}"
    base.PSEE_XRF_ROOT = SLIMSTAD_PROJECT / f"xrfv2_pseudo_{tag}_point_{point_mode}_seed{point_seed}"
    patch_train_localizer(base, tag, point_mode, point_seed, checkpoint_reuse)


def annotate_manifest(base, args, tag: str, point_mode: str, point_seed: int):
    manifest_path = base.PSEE_WEAR_ROOT / f"wear_loso{args.split}" / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest["psee_variant"] = tag
    manifest["reliability_mode"] = RELIABILITY_MODE
    manifest["point_sampling"] = {"mode": point_mode, "seed": point_seed}
    manifest["low_parameter_design"] = {
        "duration_priors": "disabled",
        "duration_limits": "not used for pseudo-boundary selection",
        "ownership": "transition peak between neighboring point annotations",
        "candidate_generation": "CAM Otsu support and transition peaks, no fixed expansion multipliers",
        "candidate_selection": "boundary-aware rank aggregation of CAM support, transition alignment, continuity, and ownership safety",
        "reliability": "left/right boundary confidence from adaptive score-margin competition",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wear"], default="wear")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--budgets", nargs="+", type=int, default=[0])
    parser.add_argument("--localizer-epochs", type=int, default=25)
    parser.add_argument("--point-mode", choices=["center", "random", "gaussian"], default="gaussian")
    parser.add_argument("--point-seed", type=int, default=DEFAULT_POINT_SEED)
    parser.add_argument("--tag", default=LOWPARAM_TAG)
    parser.add_argument("--checkpoint-reuse", action="store_true")
    parser.add_argument(
        "--reliability-mode",
        choices=["no_reliability", "segment_reliability", "boundary_reliability"],
        default="segment_reliability",
    )
    args = parser.parse_args()

    base = load_base_exporter()
    patch_base_exporter(base, args.tag, args.point_mode, args.point_seed, args.checkpoint_reuse, args.reliability_mode)
    base_args = SimpleNamespace(
        dataset=args.dataset,
        split=args.split,
        budgets=args.budgets,
        localizer_epochs=args.localizer_epochs,
        localizer_lr=1e-3,
        detector_epochs=80,
        seed=2020,
    )
    base.export_dataset(base_args)
    annotate_manifest(base, args, args.tag, args.point_mode, args.point_seed)


if __name__ == "__main__":
    main()
