#!/usr/bin/env python3
"""Run Wear PSEE-lowparam + SlimSTAD-1D smoke/full experiments.

The detector configuration inherits the clean Wear PSEE/SlimSTAD-1D protocol
and only replaces the pseudo-boundary annotation/reliability files.
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
from typing import Dict, List

import yaml

PROJECT = Path(os.environ.get("STAD_PROJECT", "/home/yinjiaxi/SlimSTAD/STAD-main"))
DETR_PROJECT = Path(os.environ.get("DETR_PROJECT", "/home/yinjiaxi/detr_e2e_tad"))
PYTHON = Path(os.environ.get("SLIMSTAD_PYTHON", "/home/yinjiaxi/miniconda3/envs/slimstad/bin/python"))
EXPORTER = DETR_PROJECT / "tools" / "export_psee_lowparam_for_slimstad.py"

TAG = "psee_lowparam_v1_20260830"
POINT_MODE = "gaussian"
POINT_SEED = 3407
RELIABILITY_MODE = "segment_reliability"
ENCODER_NAME = "slimstad1d_psee_lowparam_v1_gaussian_2020"
CONFIG_DIR = PROJECT / "configs" / "psee_lowparam_v1" / ENCODER_NAME
TASKS_PATH = PROJECT / "logs" / f"{ENCODER_NAME}_wear_tasks.json"
SUMMARY_PATH = PROJECT / "logs" / f"{ENCODER_NAME}_wear_summary.json"

FIXED_TRAINING = {
    "batch_size": 2,
    "learning_rate": 5e-5,
    "max_epoch": 80,
    "weight_decay": 1e-4,
    "grad_clip_norm": 0.5,
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


def source_config(split: int) -> Path:
    return PROJECT / "configs/psee_clean_20260715_point_ablation" / (
        f"psee_clean_20260715_point_gaussian_wear_loso{split}_b000.yaml"
    )


def pseudo_root(tag: str, split: int) -> Path:
    return PROJECT / f"wear_pseudo_strict_map_{tag}_point_{POINT_MODE}_seed{POINT_SEED}" / f"wear_loso{split}"


def pseudo_paths(tag: str, split: int) -> tuple[Path, Path, dict]:
    manifest_path = pseudo_root(tag, split) / "manifest_psee.json"
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


def write_detector_config(tag: str, split: int, detector_epochs: int, smoke: bool) -> Dict[str, object]:
    anno_csv, rel_json, quality = pseudo_paths(tag, split)
    cfg = copy.deepcopy(load_yaml(source_config(split)))
    run_name = f"{ENCODER_NAME}__wear_loso{split}_b000"
    if smoke:
        run_name += "_smoke"

    cfg["dataset"]["training"]["video_anno_path"] = str(anno_csv)
    cfg["dataset"]["training"]["psee_reliability_path"] = str(rel_json)

    cfg["model"] = dict(cfg.get("model", {}))
    if cfg["model"].get("backbone") != "wear_imu_1d":
        raise ValueError(f"Unexpected backbone in {source_config(split)}: {cfg['model'].get('backbone')}")

    cfg["training"] = dict(cfg["training"])
    for key, value in FIXED_TRAINING.items():
        cfg["training"][key] = value
    cfg["training"]["max_epoch"] = int(detector_epochs)
    cfg["training"].pop("init_checkpoint", None)
    cfg["training"]["checkpoint_path"] = str(PROJECT / "models" / run_name)

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
        "detector_epochs": int(detector_epochs),
    }
    cfg["protocol"].setdefault("weak_supervision", {})
    cfg["protocol"]["weak_supervision"]["pseudo_boundary_method"] = "PSEE-lowparam-v1"
    cfg["protocol"]["weak_supervision"]["pseudo_quality"] = quality

    cfg_path = CONFIG_DIR / f"{run_name}.yaml"
    dump_yaml(cfg_path, cfg)
    return {
        "run_name": run_name,
        "dataset": "wear",
        "split": split,
        "budget": 0,
        "seed": 2020,
        "point_mode": POINT_MODE,
        "point_seed": POINT_SEED,
        "psee_tag": tag,
        "config": str(cfg_path),
        "output": str(PROJECT / "output" / run_name),
        "model": str(PROJECT / "models" / run_name),
        "quality": quality,
        "detector_epochs": int(detector_epochs),
        "smoke": bool(smoke),
    }


def export_pseudo(tag: str, split: int, localizer_epochs: int, reliability_mode: str) -> None:
    cmd = [
        str(PYTHON),
        str(EXPORTER),
        "--dataset",
        "wear",
        "--split",
        str(split),
        "--budgets",
        "0",
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
    subprocess.run(cmd, cwd=str(DETR_PROJECT), check=True)


def task_summary(task: Dict[str, object]) -> Dict[str, object]:
    output = Path(str(task["output"]))
    result_path = output / "strict_result.json"
    row = dict(task)
    row["status"] = "missing"
    row["strict_result_path"] = str(result_path)
    if result_path.exists():
        row["status"] = "done"
        row["result"] = read_json(result_path, {})
    return row


def write_summary(tasks: List[Dict[str, object]], summary_path: Path) -> dict:
    summary = {str(task["run_name"]): task_summary(task) for task in tasks}
    tmp = summary_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(summary_path)
    return summary


def run_one(split: int, tag: str, localizer_epochs: int, detector_epochs: int, smoke: bool, force: bool, reliability_mode: str = RELIABILITY_MODE) -> Dict[str, object]:
    export_pseudo(tag, split, localizer_epochs, reliability_mode)
    task = write_detector_config(tag, split, detector_epochs, smoke)
    result_path = Path(str(task["output"])) / "strict_result.json"
    if result_path.exists() and not force:
        print(f"SKIP detector {task['run_name']}", flush=True)
        return task_summary(task)
    cmd = [str(PYTHON), "all_strict_map_psee.py", str(task["config"]), "--ngpu", "1"]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    return task_summary(task)


def run_worker(args) -> None:
    tasks = read_json(Path(args.tasks), [])
    selected = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]
    print(
        f"LOWPARAM_WORKER shard={args.shard_index}/{args.num_shards} jobs={len(selected)} "
        f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    for idx, task in enumerate(selected, 1):
        split = int(task["split"])
        print(f"RUN {idx}/{len(selected)} split={split} tag={task['psee_tag']}", flush=True)
        run_one(
            split=split,
            tag=str(task["psee_tag"]),
            localizer_epochs=int(task["localizer_epochs"]),
            detector_epochs=int(task["detector_epochs"]),
            smoke=bool(task["smoke"]),
            force=bool(args.force),
            reliability_mode=str(task.get("reliability_mode", RELIABILITY_MODE)),
        )
        write_summary(tasks, Path(args.summary))
        print(f"DONE split={split}", flush=True)
    summary = write_summary(tasks, Path(args.summary))
    done = sum(1 for row in summary.values() if row.get("status") == "done")
    print(f"LOWPARAM_WORKER_DONE completed={done}/{len(tasks)}", flush=True)


def build_tasks(splits: List[int], tag: str, localizer_epochs: int, detector_epochs: int, smoke: bool, reliability_mode: str) -> List[Dict[str, object]]:
    tasks = []
    for split in splits:
        run_name = f"{ENCODER_NAME}__wear_loso{split}_b000"
        if smoke:
            run_name += "_smoke"
        tasks.append(
            {
                "run_name": run_name,
                "dataset": "wear",
                "split": int(split),
                "budget": 0,
                "seed": 2020,
                "point_mode": POINT_MODE,
                "point_seed": POINT_SEED,
                "reliability_mode": reliability_mode,
                "psee_tag": tag,
                "localizer_epochs": int(localizer_epochs),
                "detector_epochs": int(detector_epochs),
                "smoke": bool(smoke),
                "output": str(PROJECT / "output" / run_name),
                "config": str(CONFIG_DIR / f"{run_name}.yaml"),
            }
        )
    return tasks


def run_queue(args) -> None:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    splits = [int(x) for x in args.splits.split(",") if x.strip()] if args.splits else list(range(18))
    tag = args.tag
    tasks = build_tasks(splits, tag, args.localizer_epochs, args.detector_epochs, args.smoke, args.reliability_mode)
    tasks_path = Path(args.tasks)
    summary_path = Path(args.summary)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    write_summary(tasks, summary_path)

    print(f"LOWPARAM_QUEUE tasks={len(tasks)} gpus={gpus} tag={tag}", flush=True)
    procs = []
    log_handle = None
    if args.worker_log:
        log_path = Path(args.worker_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"\n===== queue start {time.strftime('%Y-%m-%d %H:%M:%S')} tasks={len(tasks)} =====\n")
        log_handle.flush()
    try:
        for shard_idx, gpu in enumerate(gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
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
            if args.force:
                cmd.append("--force")
            stdout = log_handle if log_handle is not None else None
            stderr = subprocess.STDOUT if log_handle is not None else None
            proc = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=stdout, stderr=stderr)
            procs.append(proc)
            print(f"START worker={shard_idx} gpu={gpu} pid={proc.pid}", flush=True)
        codes = [proc.wait() for proc in procs]
        if any(code != 0 for code in codes):
            raise SystemExit(f"worker failed codes={codes}")
        write_summary(tasks, summary_path)
        print(f"LOWPARAM_QUEUE_DONE summary={summary_path}", flush=True)
    finally:
        if log_handle is not None:
            log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run-task")
    run.add_argument("--split", type=int, required=True)
    run.add_argument("--tag", default=TAG)
    run.add_argument("--reliability-mode", choices=["segment_reliability", "boundary_reliability", "no_reliability"], default=RELIABILITY_MODE)
    run.add_argument("--localizer-epochs", type=int, default=25)
    run.add_argument("--detector-epochs", type=int, default=80)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--force", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("--tasks", required=True)
    worker.add_argument("--summary", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--force", action="store_true")

    queue = sub.add_parser("queue")
    queue.add_argument("--gpus", default="1,2,3")
    queue.add_argument("--splits", default="")
    queue.add_argument("--tag", default=TAG)
    queue.add_argument("--reliability-mode", choices=["segment_reliability", "boundary_reliability", "no_reliability"], default=RELIABILITY_MODE)
    queue.add_argument("--localizer-epochs", type=int, default=25)
    queue.add_argument("--detector-epochs", type=int, default=80)
    queue.add_argument("--tasks", default=str(TASKS_PATH))
    queue.add_argument("--summary", default=str(SUMMARY_PATH))
    queue.add_argument("--worker-log", default="")
    queue.add_argument("--smoke", action="store_true")
    queue.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run-task":
        result = run_one(args.split, args.tag, args.localizer_epochs, args.detector_epochs, args.smoke, args.force, args.reliability_mode)
        print(json.dumps(result, indent=2), flush=True)
    elif args.cmd == "worker":
        run_worker(args)
    elif args.cmd == "queue":
        run_queue(args)


if __name__ == "__main__":
    main()
