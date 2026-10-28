import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict, deque
from pathlib import Path


DATASETS = ["wear", "wetlab", "rwhar", "sbhar", "xrfv2"]


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def first_col(row, names):
    for name in names:
        if name in row:
            return name
    return None


def interval_cols(row):
    start = first_col(row, ["start", "t-start", "startFrame", "start_frame"])
    end = first_col(row, ["end", "t-end", "endFrame", "end_frame"])
    return start, end


def key_cols(row):
    video = first_col(row, ["video", "video-id", "video_id", "filename", "file", "name"])
    label = first_col(row, ["type", "label", "class", "activity", "type_idx", "label_id"])
    return video, label


def tiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def fold_from_path(path):
    match = re.search(r"loso[_-]?(\d+)|wear_loso(\d+)", str(path))
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def dataset_from_path(path):
    text = str(path)
    low = text.lower()
    for dataset in DATASETS:
        if (
            f"/{dataset}/" in low
            or f"_{dataset}_" in low
            or f"{dataset}_loso" in low
            or low.startswith(f"{dataset}_")
            or f"/{dataset}_" in low
        ):
            return dataset
    return None


def gt_candidates(root, dataset, fold):
    if dataset == "wear":
        return [
            root / "wear_annotations_strict_map" / f"outer_{fold}_val_{(fold + 1) % 18}_{(fold + 2) % 18}" / "train_wear_annotation.csv",
            root / "wear_annotations_strict_map" / f"outer_{fold}_val_{(fold + 1) % 18}_{(fold + 2) % 18}" / "train_annotation.csv",
            root / "tal4har_annotations_psee_slimstad1d" / "wear" / f"loso{fold}" / "train_annotation.csv",
        ]
    if dataset == "xrfv2":
        return [
            root / "xrfv2_annotations_weak_v2" / "imu30" / f"loso{fold}" / "train_annotation.csv",
        ]
    return [
        root / "tal4har_annotations_psee_slimstad1d" / dataset / f"loso{fold}" / "train_annotation.csv",
    ]


def find_gt(root, dataset, fold):
    for candidate in gt_candidates(root, dataset, fold):
        if candidate.exists():
            return candidate
    return None


def score_pair(pseudo_csv, gt_csv):
    pred_rows = read_rows(pseudo_csv)
    gt_rows = read_rows(gt_csv)
    if not pred_rows or not gt_rows:
        return []

    pred_video, pred_label = key_cols(pred_rows[0])
    gt_video, gt_label = key_cols(gt_rows[0])
    pred_start, pred_end = interval_cols(pred_rows[0])
    gt_start, gt_end = interval_cols(gt_rows[0])
    if not all([pred_video, pred_label, gt_video, gt_label, pred_start, pred_end, gt_start, gt_end]):
        raise ValueError(f"missing columns: {pseudo_csv} {gt_csv}")

    buckets = defaultdict(deque)
    for row in gt_rows:
        buckets[(row.get(gt_video, ""), row.get(gt_label, ""))].append(row)

    scores = []
    for row in pred_rows:
        key = (row.get(pred_video, ""), row.get(pred_label, ""))
        if not buckets[key]:
            continue
        gt_row = buckets[key].popleft()
        scores.append(
            tiou(
                float(row[pred_start]),
                float(row[pred_end]),
                float(gt_row[gt_start]),
                float(gt_row[gt_end]),
            )
        )
    return scores


def score_pattern(root, method, pattern):
    root = Path(root)
    by_dataset = {dataset: [] for dataset in DATASETS}
    folds = {dataset: set() for dataset in DATASETS}
    missing_gt = []
    csvs = sorted(root.glob(pattern))
    for pseudo_csv in csvs:
        dataset = dataset_from_path(pseudo_csv)
        fold = fold_from_path(pseudo_csv)
        if dataset is None or fold is None:
            continue
        gt_csv = find_gt(root, dataset, fold)
        if gt_csv is None:
            missing_gt.append(str(pseudo_csv))
            continue
        scores = score_pair(pseudo_csv, gt_csv)
        by_dataset[dataset].extend(scores)
        folds[dataset].add(fold)
    return summarize(method, by_dataset, folds, missing_gt, len(csvs))


def summarize(method, by_dataset, folds, missing_gt=None, file_count=None):
    rows = []
    all_scores = []
    for dataset in DATASETS:
        scores = by_dataset.get(dataset, [])
        all_scores.extend(scores)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "folds": len(folds.get(dataset, [])),
                "instances": len(scores),
                "mean_tIoU": sum(scores) / len(scores) if scores else None,
                "median_tIoU": statistics.median(scores) if scores else None,
                "r@0.5": sum(score >= 0.5 for score in scores) / len(scores) if scores else None,
            }
        )
    rows.append(
        {
            "method": method,
            "dataset": "ALL",
            "folds": sum(len(v) for v in folds.values()),
            "instances": len(all_scores),
            "mean_tIoU": sum(all_scores) / len(all_scores) if all_scores else None,
            "median_tIoU": statistics.median(all_scores) if all_scores else None,
            "r@0.5": sum(score >= 0.5 for score in all_scores) / len(all_scores) if all_scores else None,
            "missing_gt": len(missing_gt or []),
            "file_count": file_count,
        }
    )
    return rows


def aggregate_snapshot_quality(method, paths):
    by_dataset = {dataset: [] for dataset in DATASETS}
    folds = {dataset: set() for dataset in DATASETS}
    for path in paths:
        data = json.load(open(path))

        def walk(node):
            if isinstance(node, dict):
                if "pseudo_quality" in node:
                    quality = node["pseudo_quality"]
                    dataset = quality.get("dataset") or node.get("dataset")
                    if dataset in by_dataset and quality.get("pseudo_mean_iou") is not None:
                        count = int(quality.get("pseudo_count") or quality.get("train_point_count") or 1)
                        by_dataset[dataset].extend([float(quality["pseudo_mean_iou"])] * count)
                        if quality.get("split") is not None:
                            folds[dataset].add(int(quality["split"]))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(data)
    return summarize(method, by_dataset, folds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--mode", choices=["patterns", "snapshot"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("items", nargs="+")
    args = parser.parse_args()

    if args.mode == "patterns":
        all_rows = []
        for pattern in args.items:
            all_rows.extend(score_pattern(args.root, args.method, pattern))
        # Merge repeated patterns for one method.
        merged = {dataset: [] for dataset in DATASETS}
        merged_folds = {dataset: set() for dataset in DATASETS}
        for pattern in args.items:
            root = Path(args.root)
            for pseudo_csv in sorted(root.glob(pattern)):
                dataset = dataset_from_path(pseudo_csv)
                fold = fold_from_path(pseudo_csv)
                if dataset is None or fold is None:
                    continue
                gt_csv = find_gt(root, dataset, fold)
                if gt_csv is not None:
                    merged[dataset].extend(score_pair(pseudo_csv, gt_csv))
                    merged_folds[dataset].add(fold)
        rows = summarize(args.method, merged, merged_folds)
    else:
        rows = aggregate_snapshot_quality(args.method, args.items)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
