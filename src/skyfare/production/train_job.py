#!/usr/bin/env python3
"""Train and serialize one immutable final production job."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.production.contract import registry
from skyfare.production.models import train_and_serialize
from skyfare.production.runtime import (
    artifact_complete,
    current_code_sha256,
    job_root,
    sha256,
    training_frames,
    write_json_atomic,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_job(raw):
    expected = {str(job["job_id"]): job for job in registry()}
    identifier = str(raw.get("job_id", ""))
    if identifier not in expected or raw != expected[identifier]:
        raise RuntimeError(f"job is not in immutable production registry: {identifier}")
    return raw


def heartbeat_writer(root: Path, job, started: float):
    def write(status: str, step: int, metrics: dict[str, float]):
        write_json_atomic(
            root / "heartbeat.json",
            {
                "status": status,
                "step": int(step),
                "job_id": job["job_id"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "metrics": {
                    key: float(value)
                    for key, value in metrics.items()
                    if np.isscalar(value) and np.isfinite(value)
                },
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            },
        )

    return write


def main():
    cli = parse_args()
    job = validate_job(json.loads(cli.job_json))
    root = job_root(str(job["job_id"]))
    if artifact_complete(str(job["job_id"])) and not cli.force:
        print(f"[PRODUCTION JOB SKIP] {job['job_id']}", flush=True)
        return
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heartbeat = heartbeat_writer(root, job, started)
    try:
        full, fit, head = training_frames(str(job["task"]), str(job["window"]))
        print(
            f"[PRODUCTION JOB START] {job['job_id']} full={len(full):,} fit={len(fit):,} "
            f"head={len(head):,} resource={job['resource']}",
            flush=True,
        )
        parity, metadata = train_and_serialize(job, full, fit, head, root, heartbeat)
        artifacts = {}
        for path in sorted(root.iterdir()):
            if path.is_file() and path.name not in {"done.json", "failure.json", "heartbeat.json"}:
                artifacts[path.name] = sha256(path)
        done = {
            "status": "COMPLETE",
            "job": job,
            "full_rows": int(len(full)),
            "fit_rows": int(len(fit)),
            "head_rows": int(len(head)),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "code_sha256": current_code_sha256(),
            "artifacts": artifacts,
            "reload_parity": parity,
            "training": metadata,
            "post_cutoff_labels_read": False,
        }
        write_json_atomic(root / "done.json", done)
        (root / "failure.json").unlink(missing_ok=True)
        print(f"[PRODUCTION JOB PASS] {job['job_id']} elapsed={done['elapsed_seconds']:.1f}s", flush=True)
    except Exception as error:
        write_json_atomic(
            root / "failure.json",
            {
                "status": "FAILED",
                "job": job,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )
        print(f"[PRODUCTION JOB FAIL] {job['job_id']} {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
