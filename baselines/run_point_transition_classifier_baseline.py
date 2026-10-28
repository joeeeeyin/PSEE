#!/usr/bin/env python3
"""Point classifier + unsupervised transition-boundary refinement baseline."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from run_point_only_classifier_baseline import (
    DEFAULT_DETR_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PROJECT_ROOT,
    PointOnlyClassifier,
    PointWindowDataset,
    collate,
    duration_priors,
    evaluate_map,
    import_wear_second_encoder,
    load_annotations,
    load_feature,
    load_info,
    make_gaussian_points,
    normalize_feature,
    nms,
    normalized_duration_prior_mode,
    set_seed,
    split_paths,
    train_classifier,
)


def moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values.astype(np.float32)
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / float(2 * radius + 1)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def transition_score(feat: np.ndarray, half_window: int, smooth_radius: int) -> np.ndarray:
    """Unsupervised sensor change score B[t] from local pre/post feature means."""
    x = normalize_feature(feat)
    steps, dims = x.shape
    if steps <= 2:
        return np.zeros(steps, dtype=np.float32)
    cumsum = np.vstack([np.zeros((1, dims), dtype=np.float32), np.cumsum(x, axis=0)])
    idx = np.arange(steps)
    left_start = np.maximum(0, idx - half_window)
    left_end = idx
    right_start = idx
    right_end = np.minimum(steps, idx + half_window)
    left_count = np.maximum(left_end - left_start, 1).astype(np.float32)[:, None]
    right_count = np.maximum(right_end - right_start, 1).astype(np.float32)[:, None]
    left_mean = (cumsum[left_end] - cumsum[left_start]) / left_count
    right_mean = (cumsum[right_end] - cumsum[right_start]) / right_count
    score = np.sqrt(np.mean((right_mean - left_mean) ** 2, axis=1))
    score[0] = 0.0
    score[-1] = 0.0
    score = moving_average(score, smooth_radius)
    if float(score.max()) > float(score.min()):
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score.astype(np.float32)


def snap_boundary(
    rough_time: float,
    duration: float,
    trans: np.ndarray,
    search_radius_sec: float,
) -> float:
    if len(trans) == 0:
        return float(np.clip(rough_time, 0.0, duration))
    rough_idx = int(round(rough_time / max(duration, 1e-6) * (len(trans) - 1)))
    radius_idx = max(1, int(round(search_radius_sec / max(duration, 1e-6) * (len(trans) - 1))))
    lo = max(0, rough_idx - radius_idx)
    hi = min(len(trans), rough_idx + radius_idx + 1)
    if hi <= lo:
        best = rough_idx
    else:
        best = lo + int(np.argmax(trans[lo:hi]))
    return float(best / max(len(trans) - 1, 1) * duration)


@torch.no_grad()
def predict_records_transition(
    model: torch.nn.Module,
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
        raw_feat = load_feature(feature_root, video)
        feat = normalize_feature(raw_feat)
        trans = transition_score(raw_feat, args.transition_half_window, args.transition_smooth_radius)
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
                rough_start = max(0.0, center - 0.5 * seg_len)
                rough_end = min(duration, center + 0.5 * seg_len)
                search_radius = max(args.min_boundary_search_sec, args.boundary_search_ratio * seg_len)
                start = snap_boundary(rough_start, duration, trans, search_radius)
                end = snap_boundary(rough_end, duration, trans, search_radius)
                if end <= start:
                    start, end = rough_start, rough_end
                min_len = max(1e-3, args.min_duration_ratio * seg_len)
                if end - start < min_len:
                    midpoint = 0.5 * (start + end)
                    start = max(0.0, midpoint - 0.5 * min_len)
                    end = min(duration, midpoint + 0.5 * min_len)
                if end > start:
                    per_video.append(
                        {
                            "video": video,
                            "label": cls,
                            "score": score,
                            "start": float(start),
                            "end": float(end),
                            "rough_start": float(rough_start),
                            "rough_end": float(rough_end),
                        }
                    )
        by_class = {}
        for pred in per_video:
            by_class.setdefault(int(pred["label"]), []).append(pred)
        for preds in by_class.values():
            predictions.extend(nms(preds, args.nms_iou)[: args.keep_per_class])
    return predictions


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
    parser.add_argument("--transition-half-window", type=int, default=8)
    parser.add_argument("--transition-smooth-radius", type=int, default=2)
    parser.add_argument("--boundary-search-ratio", type=float, default=0.25)
    parser.add_argument("--min-boundary-search-sec", type=float, default=2.0)
    parser.add_argument("--min-duration-ratio", type=float, default=0.25)
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
        / f"point_transition_classifier_{args.dataset}_loso{args.split}{'_smoke' if args.smoke else ''}"
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
    loader = torch.utils.data.DataLoader(
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
                "method": "point_transition_classifier",
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
                    "transition_half_window": args.transition_half_window,
                    "transition_smooth_radius": args.transition_smooth_radius,
                    "boundary_search_ratio": args.boundary_search_ratio,
                    "no_psee": True,
                    "no_reliability": True,
                    "transition_used_only_at_inference": True,
                },
            },
            indent=2,
        ),
        flush=True,
    )

    train_classifier(model, loader, args)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    val_pred = predict_records_transition(model, val_videos, val_info, args.feature_root, priors, args)
    test_pred = predict_records_transition(model, test_videos, test_info, args.feature_root, priors, args)
    val_map = evaluate_map(val_pred, val_ann, thresholds, num_classes)
    test_map = evaluate_map(test_pred, test_ann, thresholds, num_classes)
    result = {
        "method": "point_transition_classifier",
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
