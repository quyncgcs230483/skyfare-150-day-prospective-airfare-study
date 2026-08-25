#!/usr/bin/env python3
"""Verify all production models, calibration, manifest, and frozen evidence."""

from __future__ import annotations

import json

from skyfare.production.contract import CONTRACT_ID, TASKS, registry
from skyfare.production.runtime import (
    OUTPUT_ROOT,
    ROOT,
    artifact_complete,
    sha256,
    validate_inputs,
    write_json_atomic,
)


def main() -> None:
    validate_inputs()
    incomplete = [str(job["job_id"]) for job in registry() if not artifact_complete(str(job["job_id"]))]
    if incomplete:
        raise RuntimeError(f"incomplete production jobs: {incomplete}")
    path = OUTPUT_ROOT / "deployment/DEPLOYMENT_MANIFEST_R1.json"
    deployment = json.loads(path.read_text(encoding="utf-8"))
    if deployment.get("status") != "PASS" or deployment.get("contract", {}).get("contract_id") != CONTRACT_ID:
        raise RuntimeError("deployment manifest contract failure")
    if len(deployment.get("jobs", [])) != 24 or set(deployment.get("ensembles", {})) != set(TASKS):
        raise RuntimeError("deployment job/task coverage failure")
    for entry in deployment["jobs"]:
        root = ROOT / entry["model_root"]
        for relative, expected in entry["artifacts"].items():
            target = root / relative
            if not target.is_file() or sha256(target) != expected:
                raise RuntimeError(f"deployment artifact changed: {entry['job_id']}/{relative}")
        if entry["reload_parity"].get("status") != "PASS":
            raise RuntimeError(f"reload parity failed: {entry['job_id']}")
    for task in TASKS:
        ensemble = deployment["ensembles"][task]
        if ensemble.get("recipe") != "V24_MEAN" or ensemble.get("member_count") != 6:
            raise RuntimeError(f"{task}: frozen recipe/member failure")
        calibration = ensemble.get("calibration")
        if task in {"CLASSIFICATION", "DISTRIBUTION"}:
            target = ROOT / calibration["path"]
            if sha256(target) != calibration["sha256"]:
                raise RuntimeError(f"{task}: calibration hash failure")
    decision = json.loads(
        (ROOT / "artifacts/evidence/final_deployment_decision.json").read_text(encoding="utf-8")
    )
    if decision.get("status") != "FINAL_LOCK":
        raise RuntimeError("two-block final lock evidence missing")
    report = {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "jobs": 24,
        "tasks": list(TASKS),
        "deployment_manifest_sha256": sha256(path),
        "reload_parity": "PASS",
        "input_integrity": "PASS",
        "two_block_final_lock": "PASS",
        "production_default_action": "BUY",
    }
    write_json_atomic(OUTPUT_ROOT / "deployment/PRODUCTION_VERIFICATION_R1.json", report)
    print("[PRODUCTION REFIT ALL PASS] jobs=24 tasks=4 serialized deployment ready", flush=True)


if __name__ == "__main__":
    main()
