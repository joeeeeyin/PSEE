#!/usr/bin/env python3
"""Run adaptive low-parameter PSEE + SlimSTAD-1D on extra datasets.

The script keeps detector settings aligned with the clean PSEE/SFPoint/
PointCls SlimSTAD-1D experiments and only swaps the training annotations plus
reliability weights to the new adaptive PSEE pseudo-boundary files.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


PROJECT = Path(os.environ.get("STAD_PROJECT", "/home/yinjiaxi/SlimSTAD/STAD-main"))
DETR_PROJECT = Path(os.environ.get("DETR_PROJECT", "/home/yinjiaxi/detr_e2e_tad"))
PYTHON = Path(os.environ.get("SLIMSTAD_PYTHON", "/home/yinjiaxi/miniconda3/envs/slimstad/bin/python"))
EXPORTER = DETR_PROJECT / "tools" / "export_psee_lowparam_tal4har_xrf_for_slimstad.py"

TAG = "psee_lowparam_v1_20260830"
POINT_MODE = "gaussian"
POINT_SEED = 3407
RELIABILITY_MODE = "segment_reliability"
ENCODER_NAME = "slimstad1d_psee_lowparam_v1_gaussian_2020"
CONFIG_DIR = PROJECT / "configs" / "psee_lowparam_v1_extra" / ENCODER_NAME
TASKS_PATH = PROJECT / "logs" / f"{ENCODER_NAME}_extra_tasks.json"
SUMMARY_PATH = PROJECT / "logs" / f"{ENCODER_NAME}_extra_summary.json"

SPLIT_COUNTS = {
    "sbhar": 30,
    "rwhar": 15,
    "wetlab": 22,
    "xrfv2": 16,
}

TAL4HAR_TEMPLATE_DATES = {
    "sbhar": "20260809",
    "rwhar": "20260811",
    "wetlab": "20260812",
}

FIXED_TRAINING = {
    "batch_size": 2,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "max_epoch": 80,
    "random_seed": 2020,
    "focal_loss": True,
    "grad_clip_norm": 0.5,
    "save_model_num": [],
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dataset_run_name(dataset: str, split: int, smoke: bool) -> str:
    label = "xrfv2_imu30" if dataset == "xrfv2" else dataset
    name = f"{ENCODER_NAME}__{label}_loso{split}_b000"
    return f"{name}_smoke" if smoke else name


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_splits(value: str | None, dataset: str) -> List[int]:
    if value:
        return [int(item) for item in parse_csv_list(value)]
    return list(range(SPLIT_COUNTS[dataset]))


def pseudo_root(dataset: str, split: int, tag: str) -> Path:
    if dataset == "xrfv2":
        return PROJECT / f"xrfv2_pseudo_{tag}_point_{POINT_MODE}_seed{POINT_SEED}" / "imu30" / f"loso{split}"
    return PROJECT / f"tal4har_pseudo_{tag}_point_{POINT_MODE}_seed{POINT_SEED}" / dataset / f"loso{split}"


def pseudo_paths(dataset: str, split: int, tag: str) -> tuple[Path, Path, dict]:
    manifest_path = pseudo_root(dataset, split, tag) / "manifest_psee.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = manifest["budgets"]["0"]
    anno = Path(quality["annotation_csv"])
    rel = Path(quality["reliability_json"])
    if not anno.exists():
        raise FileNotFoundError(anno)
    if not rel.exists():
        raise FileNotFoundError(rel)
    return anno, rel, quality


def source_config(dataset: str, split: int) -> Path:
    if dataset == "xrfv2":
        return PROJECT / "configs/psee_clean_20260715_point_ablation" / (
            f"psee_clean_20260715_point_gaussian_xrfv2_imu30_loso{split}_b000.yaml"
        )

    psee_candidate = PROJECT / "configs" / "tal4har_psee_gaussian_slimstad1d_clean_yjx163_20260808" / (
        f"tal4har_psee_gaussian_slimstad1d_clean_yjx163_20260808_{dataset}_loso{split}_b000.yaml"
    )
    if psee_candidate.exists():
        return psee_candidate

    dated = TAL4HAR_TEMPLATE_DATES[dataset]
    sfpoint_candidate = PROJECT / "configs" / f"tal4har_sfnet_style_gaussian_slimstad1d_{dataset}_clean_yjx163_{dated}" / (
        f"tal4har_sfnet_style_gaussian_slimstad1d_{dataset}_clean_yjx163_{dated}_{dataset}_loso{split}_b000.yaml"
    )
    if sfpoint_candidate.exists():
        return sfpoint_candidate

    matches = [
        path
        for path in (PROJECT / "configs").glob(f"**/*{dataset}_loso{split}_b000.yaml")
        if "slimstad1d" in str(path) and "gaussian" in path.name
    ]
    if not matches:
        raise FileNotFoundError(f"No SlimSTAD-1D template config for {dataset} loso{split}")
    return sorted(matches)[0]


def expected_backbone(dataset: str) -> str:
    return "wifi_1d" if dataset == "xrfv2" else "generic_imu_1d"


def write_detector_config(dataset: str, split: int, tag: str, detector_epochs: int, smoke: bool) -> Dict[str, object]:
    anno_csv, rel_json, quality = pseudo_paths(dataset, split, tag)
    template = source_config(dataset, split)
    cfg = copy.deepcopy(load_yaml(template))
    run_name = dataset_run_name(dataset, split, smoke)

    cfg["dataset"]["training"]["video_anno_path"] = str(anno_csv)
    cfg["dataset"]["training"]["psee_reliability_path"] = str(rel_json)

    cfg["model"] = dict(cfg.get("model", {}))
    backbone = cfg["model"].get("backbone")
    if backbone != expected_backbone(dataset):
        raise ValueError(f"{template} backbone={backbone}, expected {expected_backbone(dataset)}")

    cfg["training"] = dict(cfg["training"])
    for key, value in FIXED_TRAINING.items():
        cfg["training"][key] = value
    cfg["training"]["max_epoch"] = int(detector_epochs)
    cfg["training"]["checkpoint_path"] = str(PROJECT / "models" / run_name)
    cfg["training"]["resume"] = None
    cfg["training"].pop("init_checkpoint", None)

    cfg["testing"] = dict(cfg["testing"])
    cfg["testing"]["checkpoint_path"] = str(PROJECT / "models" / run_name)
    cfg["testing"]["output_path"] = str(PROJECT / "output" / run_name)
    cfg["testing"]["output_json"] = f"{run_name}.json"

    cfg["protocol"] = dict(cfg.get("protocol", {}))
    cfg["protocol"]["method"] = "PSEE-lowparam-v1 + SlimSTAD-1D encoder"
    cfg["protocol"]["pseudo_boundary_method"] = "PSEE_lowparam_v1_transition_owned_ranked_candidates"
    cfg["protocol"]["reliability_weighted_detector"] = True
    cfg["protocol"]["psee_annotation_csv"] = str(anno_csv)
    cfg["protocol"]["psee_reliability_json"] = str(rel_json)
    cfg["protocol"]["psee_quality"] = quality
    cfg["protocol"]["psee_lowparam_v1"] = {
        "tag": tag,
        "point_mode": POINT_MODE,
        "point_seed": POINT_SEED,
        "duration_priors": "disabled",
        "dataset_specific_max_duration": "not used",
        "candidate_scoring": "rank aggregation, no manual metric weights",
        "detector": "SlimSTAD-1D",
        "detector_epochs": int(detector_epochs),
        "template_config": str(template),
    }
    cfg["protocol"].setdefault("weak_supervision", {})
    cfg["protocol"]["weak_supervision"]["pseudo_boundary_method"] = "PSEE-lowparam-v1"
    cfg["protocol"]["weak_supervision"]["pseudo_quality"] = quality

    config_path = CONFIG_DIR / f"{run_name}.yaml"
    dump_yaml(config_path, cfg)
    return {
        "run_name": run_name,
        "dataset": dataset,
        "split": int(split),
        "budget": 0,
        "seed": 2020,
        "point_mode": POINT_MODE,
        "point_seed": POINT_SEED,
        "psee_tag": tag,
        "config": str(config_path),
        "output": str(PROJECT / "output" / run_name),
        "model": str(PROJECT / "models" / run_name),
        "quality": quality,
        "detector_epochs": int(detector_epochs),
        "smoke": bool(smoke),
    }


def export_pseudo(dataset: str, split: int, tag: str, localizer_epochs: int, force: bool, reliability_mode: str) -> None:
    cmd = [
        str(PYTHON),
        str(EXPORTER),
        "--dataset",
        dataset,
        "--split",
        str(split),
        "--localizer-epochs",
        str(localizer_epochs),
        "--point-mode",
        POINT_MODE,
        "--point-seed",
        str(POINT_SEED),
        "--tag",
        tag,
        "--reliability-mode",
        reliability_mode,
    ]
    if dataset == "xrfv2":
        cmd.extend(["--xrf-modality", "imu30"])
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["DETR_PROJECT"] = str(DETR_PROJECT)
    env["STAD_PROJECT"] = str(PROJECT)
    env["TAL4HAR_DATA_ROOT"] = "/data2/yinjiaxi/imwut24_tal4har_data"
    subprocess.run(cmd, cwd=str(DETR_PROJECT), env=env, check=True)


def task_summary(task: Dict[str, object]) -> Dict[str, object]:
    row = dict(task)
    result_path = Path(str(task["output"])) / "strict_result.json"
    row["strict_result_path"] = str(result_path)
    row["status"] = "done" if result_path.exists() else "missing"
    if result_path.exists():
        row["result"] = read_json(result_path, {})
    return row


def write_summary(tasks: List[Dict[str, object]], summary_path: Path) -> dict:
    existing = read_json(summary_path, {})
    summary = {}
    for task in tasks:
        key = str(task["run_name"])
        current = task_summary(task)
        if current["status"] == "missing" and key in existing and existing[key].get("status") == "failed":
            current = existing[key]
        summary[key] = current
    tmp = summary_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(summary_path)
    return summary


def mark_failed(task: Dict[str, object], summary_path: Path, error: Exception) -> None:
    summary = read_json(summary_path, {})
    row = dict(task)
    row["status"] = "failed"
    row["error"] = repr(error)
    summary[str(task["run_name"])] = row
    tmp = summary_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(summary_path)


def run_one(task: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    dataset = str(task["dataset"])
    split = int(task["split"])
    export_pseudo(dataset, split, str(task["psee_tag"]), int(task["localizer_epochs"]), bool(args.force_export), str(task.get("reliability_mode", RELIABILITY_MODE)))
    detector_task = write_detector_config(
        dataset,
        split,
        str(task["psee_tag"]),
        int(task["detector_epochs"]),
        bool(task["smoke"]),
    )
    result_path = Path(str(detector_task["output"])) / "strict_result.json"
    if result_path.exists() and not args.force_detector:
        print(f"SKIP_DETECTOR {detector_task['run_name']}", flush=True)
        return task_summary(detector_task)
    cmd = [str(PYTHON), "all_strict_map_psee.py", str(detector_task["config"]), "--ngpu", "1"]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    return task_summary(detector_task)


def selected_tasks(tasks: Sequence[Dict[str, object]], shard_index: int, num_shards: int) -> List[Dict[str, object]]:
    return [task for idx, task in enumerate(tasks) if idx % num_shards == shard_index]


def run_worker(args: argparse.Namespace) -> None:
    tasks = read_json(Path(args.tasks), [])
    summary_path = Path(args.summary)
    jobs = selected_tasks(tasks, args.shard_index, args.num_shards)
    print(
        f"LOWPARAM_EXTRA_WORKER shard={args.shard_index}/{args.num_shards} "
        f"jobs={len(jobs)} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    for idx, task in enumerate(jobs, 1):
        print(
            f"RUN {idx}/{len(jobs)} dataset={task['dataset']} split={task['split']} "
            f"tag={task['psee_tag']}",
            flush=True,
        )
        try:
            run_one(task, args)
        except Exception as exc:
            mark_failed(task, summary_path, exc)
            print(f"FAILED dataset={task['dataset']} split={task['split']} error={exc!r}", flush=True)
            raise
        write_summary(tasks, summary_path)
        print(f"DONE dataset={task['dataset']} split={task['split']}", flush=True)
    summary = write_summary(tasks, summary_path)
    done = sum(1 for row in summary.values() if row.get("status") == "done")
    failed = sum(1 for row in summary.values() if row.get("status") == "failed")
    print(f"LOWPARAM_EXTRA_WORKER_DONE done={done}/{len(tasks)} failed={failed}", flush=True)


def build_tasks(args: argparse.Namespace) -> List[Dict[str, object]]:
    datasets = parse_csv_list(args.datasets)
    tasks: List[Dict[str, object]] = []
    for dataset in datasets:
        splits = parse_splits(args.splits, dataset)
        for split in splits:
            tasks.append(
                {
                    "run_name": dataset_run_name(dataset, split, args.smoke),
                    "dataset": dataset,
                    "split": int(split),
                    "budget": 0,
                    "seed": 2020,
                    "point_mode": POINT_MODE,
                    "point_seed": POINT_SEED,
                    "reliability_mode": args.reliability_mode,
                    "psee_tag": args.tag,
                    "localizer_epochs": int(args.localizer_epochs),
                    "detector_epochs": int(args.detector_epochs),
                    "smoke": bool(args.smoke),
                    "output": str(PROJECT / "output" / dataset_run_name(dataset, split, args.smoke)),
                    "config": str(CONFIG_DIR / f"{dataset_run_name(dataset, split, args.smoke)}.yaml"),
                }
            )
    return tasks


def run_queue(args: argparse.Namespace) -> None:
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    tasks = build_tasks(args)
    tasks_path = Path(args.tasks)
    summary_path = Path(args.summary)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    write_summary(tasks, summary_path)

    print(f"LOWPARAM_EXTRA_QUEUE tasks={len(tasks)} datasets={args.datasets} gpus={gpus} tag={args.tag}", flush=True)
    log_handle = None
    if args.worker_log:
        log_path = Path(args.worker_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"\n===== queue start {time.strftime('%Y-%m-%d %H:%M:%S')} tasks={len(tasks)} =====\n")
        log_handle.flush()

    procs = []
    try:
        for shard_idx, gpu in enumerate(gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["DETR_PROJECT"] = str(DETR_PROJECT)
            env["STAD_PROJECT"] = str(PROJECT)
            env["TAL4HAR_DATA_ROOT"] = "/data2/yinjiaxi/imwut24_tal4har_data"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--tasks",
                str(tasks_path),
                "--summary",
                str(summary_path),
                "--shard-index",
                str(shard_idx),
                "--num-shards",
                str(len(gpus)),
            ]
            if args.force_export:
                cmd.append("--force-export")
            if args.force_detector:
                cmd.append("--force-detector")
            stdout = log_handle if log_handle is not None else None
            stderr = subprocess.STDOUT if log_handle is not None else None
            proc = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=stdout, stderr=stderr)
            procs.append(proc)
            print(f"START worker={shard_idx} gpu={gpu} pid={proc.pid}", flush=True)
        codes = [proc.wait() for proc in procs]
        if any(code != 0 for code in codes):
            raise SystemExit(f"worker failed codes={codes}")
        write_summary(tasks, summary_path)
        print(f"LOWPARAM_EXTRA_QUEUE_DONE summary={summary_path}", flush=True)
    finally:
        if log_handle is not None:
            log_handle.close()


def print_status(args: argparse.Namespace) -> None:
    summary = read_json(Path(args.summary), {})
    total = len(summary)
    done = sum(1 for row in summary.values() if row.get("status") == "done")
    failed = sum(1 for row in summary.values() if row.get("status") == "failed")
    by_dataset: Dict[str, Dict[str, int]] = {}
    for row in summary.values():
        dataset = str(row.get("dataset", "unknown"))
        by_dataset.setdefault(dataset, {"done": 0, "failed": 0, "total": 0})
        by_dataset[dataset]["total"] += 1
        if row.get("status") == "done":
            by_dataset[dataset]["done"] += 1
        elif row.get("status") == "failed":
            by_dataset[dataset]["failed"] += 1
    print(json.dumps({"total": total, "done": done, "failed": failed, "by_dataset": by_dataset}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tasks", default=str(TASKS_PATH))
    common.add_argument("--summary", default=str(SUMMARY_PATH))
    common.add_argument("--force-export", action="store_true")
    common.add_argument("--force-detector", action="store_true")

    run = sub.add_parser("run-task", parents=[common])
    run.add_argument("--dataset", choices=sorted(SPLIT_COUNTS), required=True)
    run.add_argument("--split", type=int, required=True)
    run.add_argument("--tag", default=TAG)
    run.add_argument("--reliability-mode", choices=["segment_reliability", "boundary_reliability", "no_reliability"], default=RELIABILITY_MODE)
    run.add_argument("--localizer-epochs", type=int, default=25)
    run.add_argument("--detector-epochs", type=int, default=80)
    run.add_argument("--smoke", action="store_true")

    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)

    queue = sub.add_parser("queue", parents=[common])
    queue.add_argument("--datasets", default="sbhar,rwhar,wetlab,xrfv2")
    queue.add_argument("--splits", default="")
    queue.add_argument("--gpus", default="0")
    queue.add_argument("--tag", default=TAG)
    queue.add_argument("--reliability-mode", choices=["segment_reliability", "boundary_reliability", "no_reliability"], default=RELIABILITY_MODE)
    queue.add_argument("--localizer-epochs", type=int, default=25)
    queue.add_argument("--detector-epochs", type=int, default=80)
    queue.add_argument("--worker-log", default="")
    queue.add_argument("--smoke", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--summary", default=str(SUMMARY_PATH))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run-task":
        task = {
            "run_name": dataset_run_name(args.dataset, args.split, args.smoke),
            "dataset": args.dataset,
            "split": args.split,
            "psee_tag": args.tag,
            "reliability_mode": args.reliability_mode,
            "localizer_epochs": args.localizer_epochs,
            "detector_epochs": args.detector_epochs,
            "smoke": args.smoke,
        }
        result = run_one(task, args)
        print(json.dumps(result, indent=2), flush=True)
    elif args.cmd == "worker":
        run_worker(args)
    elif args.cmd == "queue":
        run_queue(args)
    elif args.cmd == "status":
        print_status(args)


if __name__ == "__main__":
    main()
