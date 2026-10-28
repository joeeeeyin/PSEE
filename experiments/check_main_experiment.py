#!/usr/bin/env python3
"""Validate that every submitted main experiment has a local entry point.

This is a provenance check, not a training launcher.  The archived runners
still depend on the original SlimSTAD, detector, data, and environment roots.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("main_experiment_manifest.json")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = set()
    for experiment in manifest["main_experiments"]:
        for key in ("runner", "runners", "tool"):
            value = experiment.get(key)
            if isinstance(value, str):
                required.add(value)
            elif isinstance(value, list):
                required.update(value)
        for value in experiment.get("psee_runners", []):
            required.add(value)
        for value in experiment.get("localizer", []):
            required.add(value)

    missing = sorted(path for path in required if not (ROOT / path).exists())
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print(f"main experiment entries: {len(manifest['main_experiments'])}")
    print(f"required local files: {len(required)}")
    print(f"python AST files checked: {len(python_files)}")
    if missing:
        print("MISSING:")
        for path in missing:
            print(f"  {path}")
        return 1

    for path in (
        ROOT / "psee_localizer/run_psee_lowparam_slimstad1d_wear_queue.py",
        ROOT / "psee_localizer/run_psee_lowparam_slimstad1d_extra_queue.py",
        ROOT / "psee_localizer/export_psee_lowparam_for_slimstad.py",
        ROOT / "psee_localizer/export_psee_lowparam_tal4har_xrf_for_slimstad.py",
    ):
        text = path.read_text(encoding="utf-8")
        if "segment_reliability" not in text:
            print(f"MAINLINE RELIABILITY CHECK FAILED: {path}")
            return 1

    print("main experiment code map: OK")
    print("runtime note: upstream detector/STAD/data/checkpoint roots are external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
