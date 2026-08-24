#!/usr/bin/env python3
"""Fail-fast validation for immutable production refit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from skyfare.production.contract import PRODUCTION_ORIGIN, RUNTIME, manifest
from skyfare.production.runtime import INPUT_ROOT, ROOT, load_frame, sha256, validate_inputs
from skyfare.production.sequence import load_or_build_sequence_source


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hardware", action="store_true")
    return parser.parse_args()


def _effective_cpus() -> int:
    quota = os.cpu_count() or 1
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.is_file():
        value, period = path.read_text().split()
        if value != "max":
            quota = min(quota, max(1, int(int(value) / int(period))))
    return quota


def main() -> None:
    args = _args()
    contract = manifest()
    inputs = validate_inputs()
    input_manifest = json.loads((INPUT_ROOT / "input_manifest.json").read_text(encoding="utf-8"))
    if input_manifest.get("status") != "PASS":
        raise RuntimeError("input assembly manifest is not PASS")
    for task in ("CLASSIFICATION", "REGRESSION"):
        frame = load_frame(task)
        if frame["label_time"].max() >= pd.Timestamp(PRODUCTION_ORIGIN):
            raise RuntimeError(f"{task}: production cutoff violated")
    sequence_source = load_or_build_sequence_source("EXACT")
    if int(sequence_source.get("groups", 0)) <= 0 or int(sequence_source.get("batches", 0)) <= 0:
        raise RuntimeError("EXACT sequence cache prebuild failed")
    decision = json.loads(
        (ROOT / "artifacts/evidence/final_deployment_decision.json").read_text(encoding="utf-8")
    )
    if decision.get("status") != "FINAL_LOCK" or decision.get("recipes") != contract["deployable_recipes"]:
        raise RuntimeError("two-block final decision does not match production contract")
    if not args.skip_hardware:
        cpus = _effective_cpus()
        if cpus < RUNTIME.minimum_effective_cpus:
            raise RuntimeError(f"effective CPU contract failed: {cpus}")
        mem_gib = int(Path("/proc/meminfo").read_text().split("MemTotal:", 1)[1].split()[0]) // 1024 // 1024
        if mem_gib < RUNTIME.minimum_effective_ram_gib:
            raise RuntimeError(f"RAM contract failed: {mem_gib} GiB")
        free_gib = shutil.disk_usage(ROOT).free // 1024**3
        if free_gib < RUNTIME.minimum_free_disk_gib:
            raise RuntimeError(f"disk contract failed: {free_gib} GiB")
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        )
        gpu_memory = [int(line.strip()) for line in query.splitlines() if line.strip()]
        if len(gpu_memory) < RUNTIME.minimum_gpus or min(gpu_memory[:2]) < RUNTIME.minimum_gpu_memory_mib:
            raise RuntimeError(f"GPU contract failed: {gpu_memory}")
    print(
        f"[PRODUCTION PREFLIGHT PASS] jobs={contract['jobs']} inputs={len(inputs)} "
        f"cutoff={contract['training_cutoff_inclusive']} sequence_cache=READY",
        flush=True,
    )


if __name__ == "__main__":
    main()
