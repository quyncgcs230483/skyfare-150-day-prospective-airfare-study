"""Immutable prospective Test 2 contract for frozen V24 deployable recipe."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from skyfare.models.selection_contract import LATE_SEEDS, QUANTILES, candidate_map


CONTRACT_ID = "SKYFARE_128_DAY_TEST_2_EVALUATION_R1"
V24_CONTRACT_ID = "SKYFARE_V24_FINAL_REFINEMENT_FREEZE_R1"
TEST1_CONTRACT_ID = "SKYFARE_128_DAY_TEST_1_EVALUATION_R2"
DEVELOPMENT_CUTOFF = "2026-07-28"
REFIT_CUTOFF = "2026-08-08"
TEST_1 = ("2026-07-29", "2026-08-08")
TEST_2 = ("2026-08-09", "2026-08-19")
TASKS = ("CLASSIFICATION", "POINT", "DISTRIBUTION", "RANKING")
FINALISTS = {
    "CLASSIFICATION": ("C24_FT_WIDE_D015", "C24_GRU_EXACT_MTL_U64"),
    "POINT": ("R24_GRU_EXACT_HUBER_U64", "R24_LSTM_EXACT_HUBER_U64"),
    "DISTRIBUTION": ("D24_CAT_Q7_D7", "D24_XGB_Q7_D7"),
    "RANKING": ("K24_LGBM_XENDCG_R84_L63", "K24_LGBM_LAMBDA_R84_L63"),
}
ORIGINAL_V24_RECIPES = {
    "CLASSIFICATION": "V24_MEAN",
    "POINT": "V24_BLEND_50",
    "DISTRIBUTION": "V24_BLEND_50",
    "RANKING": "V24_BLEND_75",
}
DEPLOYABLE_RECIPES = {task: "V24_MEAN" for task in TASKS}
BUY_WAIT_THRESHOLD = 0.30
NOMINAL_FRICTION_VND = 50_000


@dataclass(frozen=True)
class RuntimeContract:
    minimum_gpus: int = 2
    minimum_gpu_memory_mib: int = 20_000
    minimum_effective_cpus: int = 56
    minimum_effective_ram_gib: int = 80
    minimum_free_disk_gib: int = 25
    cpu_workers: int = 6
    cpu_threads_per_worker: int = 6
    gpu_workers: int = 4
    gpu_workers_per_device: int = 2
    gpu_host_threads: int = 4
    tensorflow_memory_limit_mib: int = 8_500
    heartbeat_seconds: int = 30
    heartbeat_stale_minutes: int = 35
    job_timeout_minutes: int = 120


RUNTIME = RuntimeContract()


def registry() -> list[dict[str, object]]:
    candidates = candidate_map()
    jobs: list[dict[str, object]] = []
    for task in TASKS:
        for candidate_id in FINALISTS[task]:
            candidate = candidates[candidate_id]
            for seed in LATE_SEEDS:
                jobs.append(
                    {
                        **asdict(candidate),
                        "fold": "TEST_2",
                        "seed": int(seed),
                        "phase": "TEST_2_SCORE",
                        "job_id": f"{candidate_id}__TEST_2__S{seed}",
                    }
                )
    return jobs


def validate_contract() -> dict[str, object]:
    candidates = candidate_map()
    if set(FINALISTS) != set(TASKS):
        raise RuntimeError("Test 2 finalist task coverage changed")
    for task, identifiers in FINALISTS.items():
        if len(identifiers) != 2 or len(set(identifiers)) != 2:
            raise RuntimeError(f"{task}: exactly two unique finalists required")
        for identifier in identifiers:
            if identifier not in candidates or candidates[identifier].task != task:
                raise RuntimeError(f"{task}: invalid V24 finalist {identifier}")
    jobs = registry()
    if len(jobs) != 24 or len({str(job["job_id"]) for job in jobs}) != 24:
        raise RuntimeError("Test 2 registry must contain exactly 24 jobs")
    if RUNTIME.cpu_workers * RUNTIME.cpu_threads_per_worker > RUNTIME.minimum_effective_cpus:
        raise RuntimeError("CPU allocation exceeds host contract")
    total_host_threads = (
        RUNTIME.cpu_workers * RUNTIME.cpu_threads_per_worker
        + RUNTIME.gpu_workers * RUNTIME.gpu_host_threads
    )
    if total_host_threads > RUNTIME.minimum_effective_cpus:
        raise RuntimeError("combined CPU and GPU host-thread allocation exceeds host contract")
    if RUNTIME.gpu_workers != RUNTIME.minimum_gpus * RUNTIME.gpu_workers_per_device:
        raise RuntimeError("GPU slot allocation changed")
    return {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "v24_contract_id": V24_CONTRACT_ID,
        "test1_contract_id": TEST1_CONTRACT_ID,
        "development_cutoff": DEVELOPMENT_CUTOFF,
        "refit_cutoff": REFIT_CUTOFF,
        "test_1_training_extension": list(TEST_1),
        "test_2": list(TEST_2),
        "training_protocol": {
            "mode": "EXPANDING_REFIT_AT_TEST_2_BOUNDARY",
            "allowed_labels": "label_time < 2026-08-09",
            "test_1_recipe_selection": False,
            "candidate_search": False,
            "hyperparameter_tuning": False,
        },
        "classification_observability": {
            "metric_denominator": "OBSERVED_AND_MATURE_ONLY",
            "right_censor_at_test_2_end": True,
            "missing_session_policy": "KEEP_OBSERVED_ROWS; NEVER_IMPUTE; REPORT_DENOMINATOR",
        },
        "jobs": len(jobs),
        "seeds": list(LATE_SEEDS),
        "finalists": {key: list(value) for key, value in FINALISTS.items()},
        "original_v24_recipes": ORIGINAL_V24_RECIPES,
        "deployable_recipes": DEPLOYABLE_RECIPES,
        "operational_fixes_integrated": {
            "target_free_sequence_point": "NATIVE_TEST2_SCORER",
            "ranking_query_join": "VALIDATED_NO_SUFFIX_JOIN",
            "restartable_scheduler": True,
            "runtime_hotfix_required": False,
        },
        "buy_wait": {
            "threshold": BUY_WAIT_THRESHOLD,
            "nominal_friction_vnd": NOMINAL_FRICTION_VND,
            "fallback": "BUY",
        },
        "automatic_promotion": False,
        "failure_action": "RETAIN_INCUMBENT; NO_POST_TEST_TUNING",
        "pass_action": "FINAL_FREEZE_AND_REPORT",
        "quantiles": list(QUANTILES),
        "runtime": asdict(RUNTIME),
    }


def manifest() -> dict[str, object]:
    payload = validate_contract()
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
