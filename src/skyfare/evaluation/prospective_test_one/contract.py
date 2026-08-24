"""Immutable prospective Test 1 execution contract for frozen V24 finalists."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from skyfare.models.selection_contract import LATE_SEEDS, QUANTILES, candidate_map


CONTRACT_ID = "SKYFARE_128_DAY_TEST_1_EVALUATION_R2"
V24_CONTRACT_ID = "SKYFARE_V24_FINAL_REFINEMENT_FREEZE_R1"
OBSERVATION_CUTOFF = "2026-07-28"
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
    minimum_effective_cpus: int = 40
    minimum_effective_ram_gib: int = 80
    minimum_free_disk_gib: int = 25
    cpu_workers: int = 6
    cpu_threads_per_worker: int = 6
    gpu_workers: int = 4
    gpu_workers_per_device: int = 2
    tensorflow_memory_limit_mib: int = 8_500
    heartbeat_seconds: int = 30
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
                        "fold": "TEST_1",
                        "seed": int(seed),
                        "phase": "TEST_1_SCORE",
                        "job_id": f"{candidate_id}__TEST_1__S{seed}",
                    }
                )
    return jobs


def validate_contract() -> dict[str, object]:
    candidates = candidate_map()
    if set(FINALISTS) != set(TASKS):
        raise RuntimeError("Test 1 finalist task coverage changed")
    for task, identifiers in FINALISTS.items():
        if len(identifiers) != 2 or len(set(identifiers)) != 2:
            raise RuntimeError(f"{task}: exactly two unique finalists required")
        for identifier in identifiers:
            if identifier not in candidates or candidates[identifier].task != task:
                raise RuntimeError(f"{task}: invalid V24 finalist {identifier}")
    jobs = registry()
    if len(jobs) != 24 or len({str(job["job_id"]) for job in jobs}) != 24:
        raise RuntimeError("Test 1 registry must contain exactly 24 jobs")
    if RUNTIME.cpu_workers * RUNTIME.cpu_threads_per_worker > RUNTIME.minimum_effective_cpus:
        raise RuntimeError("CPU allocation exceeds host contract")
    if RUNTIME.gpu_workers != RUNTIME.minimum_gpus * RUNTIME.gpu_workers_per_device:
        raise RuntimeError("GPU slot allocation changed")
    return {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "v24_contract_id": V24_CONTRACT_ID,
        "observation_cutoff": OBSERVATION_CUTOFF,
        "test_1": list(TEST_1),
        "test_2_preserved": list(TEST_2),
        "classification_observability": {
            "metric_denominator": "OBSERVED_AND_MATURE_ONLY",
            "right_censor_at_test_2_boundary": True,
            "missing_session_policy": "KEEP_OBSERVED_ROWS; NEVER_IMPUTE; REPORT_DENOMINATOR",
        },
        "jobs": len(jobs),
        "seeds": list(LATE_SEEDS),
        "finalists": {key: list(value) for key, value in FINALISTS.items()},
        "original_v24_recipes": ORIGINAL_V24_RECIPES,
        "deployable_recipes": DEPLOYABLE_RECIPES,
        "deployability_correction": {
            "timing": "BEFORE_TEST_1_LABEL_ACCESS",
            "reason": "V24 incumbent blend components have OOF predictions but no deployable model artifacts",
            "scope": "replace non-reproducible incumbent blend term with executable V24 finalist mean",
            "candidate_search": False,
            "hyperparameter_tuning": False,
        },
        "buy_wait": {
            "threshold": BUY_WAIT_THRESHOLD,
            "nominal_friction_vnd": NOMINAL_FRICTION_VND,
            "fallback": "BUY",
        },
        "automatic_promotion": False,
        "test_1_failure_action": "RETAIN_INCUMBENT; DO_NOT_TUNE_ON_TEST_1; PRESERVE_TEST_2",
        "test_1_pass_action": "RUN_TEST_2_WITH_SAME_DEPLOYABLE_LOCK",
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
