#!/usr/bin/env python3
"""Export adaptive low-parameter PSEE pseudo boundaries for extra datasets.

This file is intentionally separate from the frozen PSEE/PSEE-v2 exporters.
It reuses the low-parameter PSEE segment selector introduced for Wear, and
only adapts the data loading/writing glue for TAL4HAR datasets and XRFv2-IMU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import torch


DETR_PROJECT = Path(os.environ.get("DETR_PROJECT", "/home/yinjiaxi/detr_e2e_tad"))
SLIMSTAD_PROJECT = Path(os.environ.get("STAD_PROJECT", "/home/yinjiaxi/SlimSTAD/STAD-main"))
TAL4HAR_DATA_ROOT = Path(os.environ.get("TAL4HAR_DATA_ROOT", "/data2/yinjiaxi/imwut24_tal4har_data"))

LOWPARAM_EXPORTER = DETR_PROJECT / "tools" / "export_psee_lowparam_for_slimstad.py"
BASE_EXPORTER = DETR_PROJECT / "tools" / "export_psee_for_slimstad.py"
TAL4HAR_EXPORTER = DETR_PROJECT / "tools" / "export_tal4har_psee_clean_for_slimstad.py"

TAG = "psee_lowparam_v1_20260830"
POINT_MODE = "gaussian"
POINT_SEED = 3407
XRF_MODALITY = "imu30"
TAL4HAR_DATASETS = ("sbhar", "rwhar", "wetlab")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lowparam_design() -> dict:
    return {
        "duration_priors": "disabled",
        "duration_limits": "not used for pseudo-boundary selection",
        "ownership": "transition peak between neighboring point annotations",
        "candidate_generation": "CAM Otsu support and transition peaks, no fixed expansion multipliers",
        "candidate_selection": "boundary-aware rank aggregation of CAM support, transition alignment, continuity, and ownership safety",
        "reliability": "left/right boundary confidence from adaptive score-margin competition",
    }


def annotate_manifest(path: Path, dataset: str, tag: str, point_mode: str, point_seed: int) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["psee_variant"] = tag
    manifest["dataset"] = dataset
    manifest["point_sampling"] = {"mode": point_mode, "seed": int(point_seed)}
    manifest["low_parameter_design"] = lowparam_design()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def export_xrfv2(
    lowparam,
    split: int,
    localizer_epochs: int,
    tag: str,
    point_mode: str,
    point_seed: int,
    force: bool,
    modality: str,
    reliability_mode: str,
) -> Path:
    base = lowparam.load_base_exporter()
    lowparam.patch_base_exporter(base, tag, point_mode, point_seed, checkpoint_reuse=False, reliability_mode=reliability_mode)
    manifest_path = base.PSEE_XRF_ROOT / modality / f"loso{split}" / "manifest_psee.json"
    if manifest_path.exists() and not force:
        print(f"SKIP_EXPORT xrfv2/{modality} loso{split}", flush=True)
        return manifest_path
    args = SimpleNamespace(
        dataset="xrfv2",
        split=int(split),
        modality=modality,
        budgets=[0],
        localizer_epochs=int(localizer_epochs),
    )
    base.export_dataset(args)
    annotate_manifest(manifest_path, f"xrfv2_{modality}", tag, point_mode, point_seed)
    return manifest_path


def configure_tal4har(tal4har, tag: str, point_mode: str, point_seed: int) -> None:
    tal4har.DETR_PROJECT = DETR_PROJECT
    tal4har.STAD_PROJECT = SLIMSTAD_PROJECT
    tal4har.DATA_ROOT = TAL4HAR_DATA_ROOT
    tal4har.RUN_TAG = tag
    tal4har.POINT_SEED = int(point_seed)


def tal4har_output_root(dataset: str, split: int, tag: str, point_mode: str, point_seed: int) -> Path:
    return (
        SLIMSTAD_PROJECT
        / f"tal4har_pseudo_{tag}_point_{point_mode}_seed{point_seed}"
        / dataset
        / f"loso{split}"
    )


def export_tal4har(
    lowparam,
    base,
    tal4har,
    dataset: str,
    split: int,
    localizer_epochs: int,
    tag: str,
    point_mode: str,
    point_seed: int,
    force: bool,
    reliability_mode: str,
) -> Path:
    configure_tal4har(tal4har, tag, point_mode, point_seed)
    points_fn = lowparam.make_point_annotations(point_mode, point_seed)
    lowparam.RELIABILITY_MODE = reliability_mode
    tal4har.point_annotations = points_fn

    output_root = tal4har_output_root(dataset, split, tag, point_mode, point_seed)
    manifest_path = output_root / "manifest_psee.json"
    if manifest_path.exists() and not force:
        print(f"SKIP_EXPORT {dataset} loso{split}", flush=True)
        return manifest_path

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, labels, protocol = tal4har.load_records(dataset, split, "train")
    model = tal4har.train_localizer(records, labels, dataset, split, device, localizer_epochs)

    records_out: List[dict] = []
    rows: List[dict] = []
    quality: List[float] = []
    weak_quality: List[float] = []
    for record in records:
        cam, transition = tal4har.infer_maps(model, record, dataset, device)
        points = sorted(points_fn(record), key=lambda item: float(item["time"]))
        annotations = []
        for point_index, point in enumerate(points):
            ann_idx = int(point["annotation_index"])
            gt_ann = record["annotations"][ann_idx]
            segment, reliability, components = lowparam.lowparam_psee_segment(
                cam,
                transition,
                point,
                points,
                point_index,
                {},
                None,
                float(record["duration"]),
                0.0,
                float(record["duration"]),
            )
            components["source"] = "psee_lowparam_v1"
            annotations.append({"segment": segment, "label": int(point["label"])})
            iou = float(base.segment_iou_np(segment, gt_ann["segment"]))
            quality.append(iou)
            weak_quality.append(iou)
            rows.append(
                {
                    "video": record["video_id"],
                    "annotation_index": ann_idx,
                    "label": int(point["label"]),
                    "start": float(segment[0]),
                    "end": float(segment[1]),
                    "reliability": float(reliability),
                    **components,
                }
            )
            lowparam.boundary_error_fields(segment, gt_ann["segment"], rows[-1])
        rec_out = {key: value for key, value in record.items() if key != "annotations"}
        rec_out["annotations"] = annotations
        records_out.append(rec_out)

    csv_path = output_root / "train_annotations_psee_000.csv"
    rel_path = output_root / "train_reliability_psee_000.json"
    tal4har.write_psee_csv(csv_path, records_out, labels)
    meta = {
        "budget": 0,
        "class_duration_priors": {},
        "global_duration_prior": None,
        "psee_variant": tag,
        "point_mode": point_mode,
        "point_seed": int(point_seed),
        "reliability_mode": reliability_mode,
    }
    base.write_reliability(rel_path, csv_path, rows, meta)
    budget_summary = lowparam.summary_with_boundary_diagnostics(
        base.summary,
        np.asarray(quality),
        np.asarray(weak_quality),
        0,
        rows,
    )
    budget_summary.update({"annotation_csv": str(csv_path), "reliability_json": str(rel_path), **meta})
    manifest = {
        "method": "PSEE-lowparam-v1-tal4har",
        "run_tag": tag,
        "dataset": dataset,
        "split": int(split),
        "boundary_seed": None,
        "point_seed": int(point_seed),
        "point_mode": point_mode,
        "protocol": protocol,
        "psee_variant": tag,
        "reliability_mode": reliability_mode,
        "low_parameter_design": lowparam_design(),
        "budgets": {"0": budget_summary},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"PSEE_LOWPARAM dataset={dataset} split=loso{split} budget=0 "
        f"mean_iou={budget_summary['mean_iou']*100:.2f} "
        f"R@0.5={budget_summary['recall_0.5']*100:.2f} "
        f"mean_R={budget_summary['mean_reliability']*100:.2f}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*TAL4HAR_DATASETS, "xrfv2"], required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--localizer-epochs", type=int, default=25)
    parser.add_argument("--point-mode", choices=["center", "random", "gaussian"], default=POINT_MODE)
    parser.add_argument("--point-seed", type=int, default=POINT_SEED)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--xrf-modality", default=XRF_MODALITY)
    parser.add_argument(
        "--reliability-mode",
        choices=["no_reliability", "segment_reliability", "boundary_reliability"],
        default="segment_reliability",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    lowparam = load_module("psee_lowparam_exporter", LOWPARAM_EXPORTER)
    base = load_module("psee_base_exporter_for_lowparam_extra", BASE_EXPORTER)

    if args.dataset == "xrfv2":
        export_xrfv2(
            lowparam,
            args.split,
            args.localizer_epochs,
            args.tag,
            args.point_mode,
            args.point_seed,
            args.force,
            args.xrf_modality,
            args.reliability_mode,
        )
    else:
        tal4har = load_module("tal4har_psee_exporter_for_lowparam", TAL4HAR_EXPORTER)
        export_tal4har(
            lowparam,
            base,
            tal4har,
            args.dataset,
            args.split,
            args.localizer_epochs,
            args.tag,
            args.point_mode,
            args.point_seed,
            args.force,
            args.reliability_mode,
        )


if __name__ == "__main__":
    main()
