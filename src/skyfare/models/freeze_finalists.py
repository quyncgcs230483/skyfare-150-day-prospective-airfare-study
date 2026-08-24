#!/usr/bin/env python3
"""Write immutable recipe and Test 1/Test 2 handoff lock."""

from __future__ import annotations

import json

from skyfare.models.selection_contract import CONTRACT_ID, OBSERVATION_CUTOFF, TEST_BLOCKS, manifest
from skyfare.models.temporal_runtime import OUTPUT_ROOT, sha256, write_json_atomic


def main() -> None:
    analysis_path = OUTPUT_ROOT / "analysis" / "research_report_v24.json"
    policy_path = OUTPUT_ROOT / "analysis" / "buy_wait_policy_v24.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "PASS" or policy.get("status") != "PASS":
        raise RuntimeError("analysis or policy not PASS")
    lock = {
        "status": "FROZEN",
        "contract_id": CONTRACT_ID,
        "contract_sha256": manifest()["contract_sha256"],
        "observation_cutoff": OBSERVATION_CUTOFF,
        "test_access": False,
        "test_blocks": TEST_BLOCKS,
        "tasks": analysis["tasks"],
        "buy_wait": {
            "threshold": policy["threshold"],
            "nominal_friction_vnd": policy["nominal_friction_vnd"],
            "fallback": policy["fallback_on_missing_prediction"],
        },
        "analysis_sha256": sha256(analysis_path),
        "policy_sha256": sha256(policy_path),
        "next_action": "RUN_TEST_1_WITHOUT_TUNING",
        "test_1_failure_action": "RETAIN_INCUMBENT; DO_NOT_TUNE_ON_TEST_1; PRESERVE_TEST_2",
        "automatic_promotion": False,
    }
    path = OUTPUT_ROOT / "freeze" / "V24_RECIPE_LOCK.json"
    write_json_atomic(path, lock)
    write_json_atomic(OUTPUT_ROOT / "runtime" / "pipeline_status_v24.json", {"status": "PASS", "phase": "FROZEN", "lock_sha256": sha256(path)})
    print("[V24 FREEZE PASS] next=TEST_1 test_access=false automatic_promotion=false", flush=True)


if __name__ == "__main__":
    main()
