#!/usr/bin/env python3
"""Verify Test 1 execution integrity without changing scientific decision."""

from __future__ import annotations

import json

import pandas as pd

from skyfare.evaluation.prospective_test_one.contract import registry
from skyfare.evaluation.prospective_test_one.runtime import (
    OUTPUT_ROOT,
    artifact_complete,
    sha256,
    write_json_atomic,
)


def main() -> None:
    required_json = {
        "preflight": OUTPUT_ROOT / "runtime" / "preflight_test1.json",
        "scheduler": OUTPUT_ROOT / "runtime" / "scheduler_test1.json",
        "commit": OUTPUT_ROOT / "runtime" / "prediction_commit_test1.json",
        "report": OUTPUT_ROOT / "analysis" / "TEST_1_REPORT.json",
    }
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required_json.items()}
    if payloads["preflight"].get("status") != "PASS":
        raise RuntimeError("Test 1 preflight did not pass")
    if payloads["scheduler"].get("status") != "PASS" or payloads["scheduler"].get("jobs_complete") != 24:
        raise RuntimeError("Test 1 scheduler coverage incomplete")
    if payloads["commit"].get("status") != "COMMITTED_BEFORE_LABEL_EVALUATION":
        raise RuntimeError("Test 1 prediction commit missing")
    incomplete = [str(job["job_id"]) for job in registry() if not artifact_complete(str(job["job_id"]))]
    if incomplete:
        raise RuntimeError(f"Test 1 job artifacts incomplete: {incomplete}")
    prediction_summary = {}
    for task, item in payloads["commit"]["files"].items():
        path = OUTPUT_ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"committed prediction hash mismatch: {task}")
        frame = pd.read_parquet(path)
        if len(frame) != int(item["rows"]) or frame["row_key"].duplicated().any():
            raise RuntimeError(f"committed prediction identity mismatch: {task}")
        prediction_summary[task] = {"rows": len(frame), "sha256": sha256(path)}
    report_status = payloads["report"].get("status")
    if report_status not in {"PASS", "FAIL"}:
        raise RuntimeError("Test 1 scientific decision missing")
    verification = {
        "status": "PASS",
        "integrity": "PASS",
        "scientific_decision": report_status,
        "automatic_promotion": False,
        "test_2_access": False,
        "jobs": 24,
        "predictions": prediction_summary,
        "json_hashes": {name: sha256(path) for name, path in required_json.items()},
    }
    write_json_atomic(OUTPUT_ROOT / "VERIFICATION_TEST_1.json", verification)
    write_json_atomic(
        OUTPUT_ROOT / "runtime" / "pipeline_status_test1.json",
        {
            "status": "COMPLETE",
            "integrity": "PASS",
            "scientific_decision": report_status,
            "next_action": payloads["report"]["next_action"],
            "test_2_access": False,
        },
    )
    print(f"[TEST1 VERIFY PASS] integrity=PASS scientific_decision={report_status}", flush=True)


if __name__ == "__main__":
    main()
