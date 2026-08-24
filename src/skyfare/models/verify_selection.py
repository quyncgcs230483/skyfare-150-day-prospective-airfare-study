#!/usr/bin/env python3
"""Verify V24 coverage, artifacts, lock, no test access and failed-job absence."""

from __future__ import annotations

import json
from pathlib import Path

from skyfare.models.selection_contract import late_registry, manifest, screen_registry, smoke_registry
from skyfare.models.temporal_runtime import OUTPUT_ROOT, artifact_complete, sha256, write_json_atomic


def _pass_json(path: Path, expected: str = "PASS") -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected:
        raise RuntimeError(f"{path}: status={payload.get('status')!r}")
    return payload


def main() -> None:
    finalists = _pass_json(OUTPUT_ROOT / "analysis" / "finalists_v24.json")["finalists"]
    registries = {
        "smoke": smoke_registry(),
        "screen_history": screen_registry(),
        "late": late_registry(finalists),
    }
    coverage = {}
    for phase, jobs in registries.items():
        complete = [job for job in jobs if artifact_complete(str(job["job_id"]))]
        if len(complete) != len(jobs):
            missing = [job["job_id"] for job in jobs if not artifact_complete(str(job["job_id"]))]
            raise RuntimeError(f"{phase}: incomplete {len(complete)}/{len(jobs)} missing={missing[:10]}")
        coverage[phase] = len(complete)
    _pass_json(OUTPUT_ROOT / "analysis" / "research_report_v24.json")
    _pass_json(OUTPUT_ROOT / "analysis" / "buy_wait_policy_v24.json")
    lock = _pass_json(OUTPUT_ROOT / "freeze" / "V24_RECIPE_LOCK.json", "FROZEN")
    if lock.get("test_access") is not False or lock.get("next_action") != "RUN_TEST_1_WITHOUT_TUNING":
        raise RuntimeError("V24 freeze/test handoff changed")
    failures = [path for path in (OUTPUT_ROOT / "jobs").glob("*/failure.json") if not (path.parent / "done.json").is_file()]
    if failures:
        raise RuntimeError(f"unresolved failed jobs: {[str(path) for path in failures[:10]]}")
    oof = {}
    for task in ("classification", "point", "distribution", "ranking"):
        path = OUTPUT_ROOT / "analysis" / f"{task}_final_oof_v24.parquet"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing V24 OOF {task}")
        oof[task] = sha256(path)
    report = {
        "status": "PASS",
        "contract_id": manifest()["contract_id"],
        "contract_sha256": manifest()["contract_sha256"],
        "coverage": coverage,
        "production_jobs": coverage["screen_history"] + coverage["late"],
        "smoke_jobs": coverage["smoke"],
        "test_access": False,
        "automatic_promotion": False,
        "oof_sha256": oof,
        "lock_sha256": sha256(OUTPUT_ROOT / "freeze" / "V24_RECIPE_LOCK.json"),
    }
    write_json_atomic(OUTPUT_ROOT / "VERIFICATION_V24.json", report)
    print(f"[V24 VERIFY PASS] production={report['production_jobs']} smoke={report['smoke_jobs']} test_access=false", flush=True)


if __name__ == "__main__":
    main()
