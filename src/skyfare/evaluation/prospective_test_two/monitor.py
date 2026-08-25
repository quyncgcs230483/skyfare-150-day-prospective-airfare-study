#!/usr/bin/env python3
"""Lightweight Test 2 progress, current jobs, stale signals, and ETA."""

from __future__ import annotations

import json
import time

from skyfare.evaluation.prospective_test_two.contract import RUNTIME, registry
from skyfare.evaluation.prospective_test_two.runtime import (
    OUTPUT_ROOT,
    current_code_sha256,
    job_root,
    preflight_sha256,
)


def _done(
    job_id: str, expected_code_sha256: str, expected_preflight_sha256: str | None
) -> dict[str, object] | None:
    root = job_root(job_id)
    path = root / "done.json"
    if not path.is_file() or not (root / "predictions.parquet").is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    valid = (
        payload.get("status") == "COMPLETE"
        and payload.get("code_sha256") == expected_code_sha256
        and payload.get("preflight_sha256") == expected_preflight_sha256
    )
    return payload if valid else None


def main() -> None:
    jobs = registry()
    code_hash = current_code_sha256()
    preflight_hash = preflight_sha256()
    completed: list[tuple[dict[str, object], dict[str, object]]] = []
    running = []
    now = time.time()
    for job in jobs:
        identifier = str(job["job_id"])
        done = _done(identifier, code_hash, preflight_hash)
        if done is not None:
            completed.append((job, done))
            continue
        heartbeat = job_root(identifier) / "heartbeat.json"
        if heartbeat.is_file():
            try:
                payload = json.loads(heartbeat.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            running.append((job, payload, max(0.0, now - heartbeat.stat().st_mtime)))

    phase = "SCORE"
    if (OUTPUT_ROOT / "runtime" / "pipeline_status_test2.json").is_file():
        phase = "COMPLETE"
    elif (OUTPUT_ROOT / "runtime" / "prediction_commit_test2.json").is_file():
        phase = "EVALUATE_VERIFY_ARCHIVE"
    elif len(completed) == len(jobs):
        phase = "COMMIT"

    elapsed_by_resource: dict[str, list[float]] = {"CPU": [], "GPU": []}
    for job, done in completed:
        elapsed_by_resource[str(job["resource"])].append(float(done.get("elapsed_seconds", 0.0)))
    remaining_seconds = 0.0
    for resource, slots in (("CPU", RUNTIME.cpu_workers), ("GPU", RUNTIME.gpu_workers)):
        pending = sum(
            1
            for job in jobs
            if str(job["resource"]) == resource
            and _done(str(job["job_id"]), code_hash, preflight_hash) is None
        )
        history = elapsed_by_resource[resource]
        if pending and history:
            remaining_seconds = max(remaining_seconds, pending * (sum(history) / len(history)) / slots)

    print(
        f"TEST2 phase={phase} complete={len(completed)}/{len(jobs)} "
        f"remaining={len(jobs)-len(completed)} running={len(running)}"
    )
    print(f"ETA jobs={'unknown' if remaining_seconds == 0 else f'{remaining_seconds/60:.0f}m'}")
    for job, heartbeat, age in running:
        stale = " STALE" if age > RUNTIME.heartbeat_stale_minutes * 60 else ""
        print(
            f"RUN {job['job_id']} status={heartbeat.get('status')} "
            f"elapsed={float(heartbeat.get('elapsed_seconds', 0))/60:.1f}m "
            f"heartbeat_age={age:.0f}s{stale}"
        )
    recent = sorted(
        completed,
        key=lambda item: (job_root(str(item[0]["job_id"])) / "done.json").stat().st_mtime,
        reverse=True,
    )[:5]
    for job, done in recent:
        print(f"DONE {job['job_id']} elapsed={float(done.get('elapsed_seconds', 0))/60:.1f}m")


if __name__ == "__main__":
    main()
