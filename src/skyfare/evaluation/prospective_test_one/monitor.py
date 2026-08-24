#!/usr/bin/env python3
"""Human-readable Test 1 progress and ETA snapshot."""

from __future__ import annotations

import json
import time

from skyfare.evaluation.prospective_test_one.contract import registry
from skyfare.evaluation.prospective_test_one.runtime import OUTPUT_ROOT, artifact_complete, job_root


def main() -> None:
    jobs = registry()
    complete = [job for job in jobs if artifact_complete(str(job["job_id"]))]
    running = []
    now = time.time()
    for job in jobs:
        if job in complete:
            continue
        heartbeat = job_root(str(job["job_id"])) / "heartbeat.json"
        if heartbeat.is_file():
            payload = json.loads(heartbeat.read_text(encoding="utf-8"))
            age = max(0.0, now - heartbeat.stat().st_mtime)
            running.append((job, payload, age))
    elapsed = [
        float(json.loads((job_root(str(job["job_id"])) / "done.json").read_text())["elapsed_seconds"])
        for job in complete
    ]
    mean = sum(elapsed) / len(elapsed) if elapsed else 0.0
    slots = 10
    eta_minutes = (len(jobs) - len(complete)) * mean / slots / 60.0 if mean else None
    phase = "SCORE"
    status_path = OUTPUT_ROOT / "runtime" / "pipeline_status_test1.json"
    if status_path.is_file():
        phase = "COMPLETE"
    elif (OUTPUT_ROOT / "runtime" / "prediction_commit_test1.json").is_file():
        phase = "EVALUATE"
    print(f"TEST1 phase={phase} complete={len(complete)}/24 remaining={24-len(complete)}")
    print(f"ETA jobs={'unknown' if eta_minutes is None else f'{eta_minutes:.0f}m'}")
    for job, heartbeat, age in running:
        print(
            f"RUN {job['job_id']} status={heartbeat.get('status')} "
            f"elapsed={float(heartbeat.get('elapsed_seconds', 0))/60:.1f}m heartbeat_age={age:.0f}s"
        )


if __name__ == "__main__":
    main()
