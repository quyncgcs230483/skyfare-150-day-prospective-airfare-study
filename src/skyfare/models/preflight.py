#!/usr/bin/env python3
"""Fail-fast V24 host, dependency, data and sealed-test preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from skyfare.models.selection_contract import OBSERVATION_CUTOFF, RUNTIME, TEST_BLOCKS, manifest
from skyfare.models.temporal_runtime import (
    CLASSIFICATION_FRAME,
    CONTROL_ROOT,
    OFFERS_FRAME,
    OUTPUT_ROOT,
    REGRESSION_FRAME,
    apply_window,
    inner_split,
    load_classification_frame,
    load_regression_frame,
    outer_split,
    sha256,
    write_json_atomic,
)

CONTROL_CONTRACTS = {
    "v23_classification_final_oof.parquet": {
        "rows": 232_696,
        "columns": {"row_key", "fold", "target", "V23_SELECTED"},
        "folds": {"F02": 12_726, "F03": 47_005, "F04": 58_661, "F05": 57_115, "F06": 57_189},
    },
    "v23_regression_final_oof.parquet": {
        "rows": 286_810,
        "columns": {"row_key", "fold", "target_session_price_vnd", "V23_SELECTED"},
        "folds": {"F02": 28_285, "F03": 54_398, "F04": 70_602, "F05": 65_227, "F06": 68_298},
    },
    "v22_distribution_oof.parquet": {
        "rows": 286_810,
        "columns": {"row_key", "fold", "v22_selected_q05", "v22_selected_q50", "v22_selected_q95"},
        "folds": {"F02": 28_285, "F03": 54_398, "F04": 70_602, "F05": 65_227, "F06": 68_298},
    },
    "v22_ranking_oof.parquet": {
        "rows": 286_810,
        "columns": {"row_key", "fold", "v22_rank_selected", "v22_rank_ensemble"},
        "folds": {"F02": 28_285, "F03": 54_398, "F04": 70_602, "F05": 65_227, "F06": 68_298},
    },
}


def _effective_cpus() -> int:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = affinity
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        raw_quota, raw_period = cpu_max.read_text().strip().split()
        if raw_quota != "max":
            quota = max(1, int(raw_quota) // int(raw_period))
    return min(affinity, quota)


def _memory_gib() -> float:
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.is_file() and limit.read_text().strip() != "max":
        return int(limit.read_text().strip()) / (1024 ** 3)
    pages = os.sysconf("SC_PHYS_PAGES")
    size = os.sysconf("SC_PAGE_SIZE")
    return pages * size / (1024 ** 3)


def _gpus() -> list[dict[str, object]]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        index, name, memory, capability = [item.strip() for item in line.split(",", 3)]
        rows.append({"index": int(index), "name": name, "memory_mib": int(memory), "compute_capability": capability})
    return rows


def _dependency_versions() -> dict[str, str]:
    required = ("numpy", "pandas", "scipy", "pyarrow", "scikit-learn", "joblib", "lightgbm", "catboost", "xgboost", "tensorflow")
    versions = {name: importlib.metadata.version(name) for name in required}
    tensorflow_major_minor = tuple(int(part) for part in versions["tensorflow"].split(".")[:2])
    if tensorflow_major_minor not in {(2, 18), (2, 19)}:
        raise RuntimeError(f"V24 requires TensorFlow 2.18 or 2.19; found {versions['tensorflow']}")
    return versions


def _control_gate() -> dict[str, object]:
    report = {}
    for name, expected in CONTROL_CONTRACTS.items():
        path = CONTROL_ROOT / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing V24 control {path}")
        frame = pd.read_parquet(path)
        missing = sorted(expected["columns"] - set(frame.columns))
        folds = frame["fold"].value_counts().sort_index().astype(int).to_dict() if "fold" in frame else {}
        if len(frame) != expected["rows"] or missing or folds != expected["folds"]:
            raise RuntimeError(
                f"V24 control contract mismatch name={name} rows={len(frame)}/{expected['rows']} "
                f"missing={missing} folds={folds}"
            )
        if frame["row_key"].duplicated().any():
            raise RuntimeError(f"V24 control row_key is not unique: {name}")
        report[name] = {"rows": len(frame), "folds": folds, "sha256": sha256(path)}
    return report


def _data_gate() -> dict[str, object]:
    for path in (CLASSIFICATION_FRAME, REGRESSION_FRAME, OFFERS_FRAME):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing V24 input {path}")
    cutoff = pd.Timestamp(OBSERVATION_CUTOFF) + pd.Timedelta(days=1)
    class_times = pd.read_parquet(CLASSIFICATION_FRAME, columns=["label_time"])["label_time"]
    reg_times = pd.read_parquet(REGRESSION_FRAME, columns=["label_time"])["label_time"]
    offer_times = pd.read_parquet(OFFERS_FRAME, columns=["feature_time"])["feature_time"]
    if pd.to_datetime(class_times).max() >= cutoff or pd.to_datetime(reg_times).max() >= cutoff:
        raise RuntimeError("V24 input crosses observation cutoff")
    if pd.to_datetime(offer_times).max() >= cutoff:
        raise RuntimeError("V24 offers cross observation cutoff")
    classification = load_classification_frame()
    regression = load_regression_frame()
    split_audit = {}
    for task, frame, folds in (
        ("CLASSIFICATION", classification, ("F02", "F03", "F04", "F05", "F06")),
        ("REGRESSION", regression, ("F01", "F02", "F03", "F04", "F05", "F06")),
    ):
        for fold in folds:
            train, valid = outer_split(frame, task, fold)
            windows = ("EXPANDING", "RECENT84") if task == "REGRESSION" else ("EXPANDING",)
            for window in windows:
                fit, head = inner_split(apply_window(train, task, fold, window), task)
                split_audit[f"{task}/{fold}/{window}"] = {"fit": len(fit), "head": len(head), "valid": len(valid)}
    return {
        "classification_sha256": sha256(CLASSIFICATION_FRAME),
        "regression_sha256": sha256(REGRESSION_FRAME),
        "offers_sha256": sha256(OFFERS_FRAME),
        "classification_rows": int(len(class_times)),
        "regression_rows": int(len(reg_times)),
        "offers_rows": int(len(offer_times)),
        "maximum_label_time": str(max(pd.to_datetime(class_times).max(), pd.to_datetime(reg_times).max())),
        "split_audit": split_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    cli = parser.parse_args()
    contract = manifest()
    gpu = _gpus()
    if len(gpu) < RUNTIME.minimum_gpus or min(item["memory_mib"] for item in gpu[:2]) < RUNTIME.minimum_gpu_memory_mib:
        raise RuntimeError(f"V24 requires two >=20GB GPUs; found {gpu}")
    cpu = _effective_cpus()
    memory = _memory_gib()
    disk = shutil.disk_usage(cli.root).free / (1024 ** 3)
    if cpu < RUNTIME.minimum_effective_cpus:
        raise RuntimeError(f"V24 requires {RUNTIME.minimum_effective_cpus} CPUs; found {cpu}")
    if memory < RUNTIME.minimum_effective_ram_gib:
        raise RuntimeError(f"V24 requires {RUNTIME.minimum_effective_ram_gib}GiB RAM; found {memory:.1f}")
    if disk < RUNTIME.minimum_free_disk_gib:
        raise RuntimeError(f"V24 requires {RUNTIME.minimum_free_disk_gib}GiB free disk; found {disk:.1f}")
    import tensorflow as tf
    tf_gpu = tf.config.list_physical_devices("GPU")
    if len(tf_gpu) != 2:
        raise RuntimeError(f"TensorFlow must see exactly two GPUs; found {tf_gpu}")
    report = {
        "status": "PASS",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "test_access": False,
        "sealed_test_blocks": TEST_BLOCKS,
        "host": {"gpus": gpu, "tensorflow_gpus": [str(item) for item in tf_gpu], "effective_cpus": cpu, "effective_ram_gib": memory, "free_disk_gib": disk},
        "dependencies": _dependency_versions(),
        "data": _data_gate(),
        "controls": _control_gate(),
    }
    write_json_atomic(OUTPUT_ROOT / "runtime" / "preflight_v24.json", report)
    print(f"[V24 PREFLIGHT PASS] gpus=2 cpus={cpu} ram={memory:.1f}GiB disk={disk:.1f}GiB test_access=false", flush=True)


if __name__ == "__main__":
    main()
