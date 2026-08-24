#!/usr/bin/env python3
"""Compact production refit progress and stalled-heartbeat monitor."""

from __future__ import annotations

import json
import time

from skyfare.production.contract import RUNTIME, registry
from skyfare.production.runtime import artifact_complete, job_root


def main() -> None:
    complete = []
    running = []
    failed = []
    now = time.time()
    for job in registry():
        job_id = str(job["job_id"])
        root = job_root(job_id)
        if artifact_complete(job_id):
            complete.append(job_id)
        elif (root / "failure.json").is_file():
            failed.append(job_id)
        elif (root / "heartbeat.json").is_file():
            heartbeat = json.loads((root / "heartbeat.json").read_text(encoding="utf-8"))
            age = max(0, now - (root / "heartbeat.json").stat().st_mtime)
            running.append((job_id, heartbeat.get("status"), heartbeat.get("step"), age))
    print(f"PRODUCTION REFIT complete={len(complete)}/24 remaining={24-len(complete)} failed={len(failed)}")
    for job_id, status, step, age in running:
        marker = " STALE" if age > RUNTIME.heartbeat_stale_minutes * 60 else ""
        print(f"RUN {job_id} status={status} step={step} heartbeat_age={age:.0f}s{marker}")
    for job_id in failed:
        print(f"FAIL {job_id}")
    if len(complete) == 24:
        print("PRODUCTION REFIT COMPLETE: wait for result archive marker")


if __name__ == "__main__":
    main()
