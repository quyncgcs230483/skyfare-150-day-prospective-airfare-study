"""Immutable contract for final production refit after two-block confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from skyfare.models.selection_contract import LATE_SEEDS, candidate_map

CONTRACT_ID = "SKYFARE_128_DAY_FINAL_PRODUCTION_REFIT_R1"
TRAINING_CUTOFF_INCLUSIVE = "2026-08-19"
PRODUCTION_ORIGIN = "2026-08-20"
TASKS = ("CLASSIFICATION", "POINT", "DISTRIBUTION", "RANKING")
FINALISTS = {
    "CLASSIFICATION": ("C24_FT_WIDE_D015", "C24_GRU_EXACT_MTL_U64"),
    "POINT": ("R24_GRU_EXACT_HUBER_U64", "R24_LSTM_EXACT_HUBER_U64"),
    "DISTRIBUTION": ("D24_CAT_Q7_D7", "D24_XGB_Q7_D7"),
    "RANKING": ("K24_LGBM_XENDCG_R84_L63", "K24_LGBM_LAMBDA_R84_L63"),
}
DEPLOYABLE_RECIPES = {task: "V24_MEAN" for task in TASKS}
BUY_WAIT_THRESHOLD = 0.30
NOMINAL_FRICTION_VND = 50_000
INPUT_SHA256 = {
    "classification_training_frame.parquet": "43abf21c149cac2ac7d13fe06f05972083c6bb4ff5579cd48f88bd168cae6c3d",
    "regression_training_frame.parquet": "c1d0d30b1ccfbe678dc24382485fe390b015051513cd046e32bd12af4ba7deff",
    "standard_offers.parquet": "48f9502f0cab025f84a9f83ad6d7d95193fa3a0d6675615c784ec0aa9e5e511a",
}
INPUT_ROWS = {
    "classification_training_frame.parquet": 720_155,
    "regression_training_frame.parquet": 1_106_785,
    "standard_offers.parquet": 1_114_445,
}


@dataclass(frozen=True)
class RuntimeContract:
    minimum_gpus: int = 2
    minimum_gpu_memory_mib: int = 20_000
    minimum_effective_cpus: int = 56
    minimum_effective_ram_gib: int = 80
    minimum_free_disk_gib: int = 12
    cpu_workers: int = 6
    cpu_threads_per_worker: int = 6
    gpu_workers: int = 4
    gpu_workers_per_device: int = 2
    gpu_host_threads: int = 4
    tensorflow_memory_limit_mib: int = 8_500
    heartbeat_seconds: int = 30
    heartbeat_stale_minutes: int = 35
    job_timeout_minutes: int = 180


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
                        "fold": "PRODUCTION",
                        "seed": int(seed),
                        "phase": "FINAL_PRODUCTION_REFIT",
                        "job_id": f"{candidate_id}__PRODUCTION__S{seed}",
                    }
                )
    return jobs


def validate_contract() -> dict[str, object]:
    candidates = candidate_map()
    if set(FINALISTS) != set(TASKS):
        raise RuntimeError("production finalist task coverage changed")
    for task, identifiers in FINALISTS.items():
        if len(identifiers) != 2 or len(set(identifiers)) != 2:
            raise RuntimeError(f"{task}: exactly two unique finalists required")
        for identifier in identifiers:
            if identifier not in candidates or candidates[identifier].task != task:
                raise RuntimeError(f"{task}: invalid frozen finalist {identifier}")
    jobs = registry()
    if len(jobs) != 24 or len({str(job["job_id"]) for job in jobs}) != 24:
        raise RuntimeError("production registry must contain exactly 24 jobs")
    if RUNTIME.cpu_workers * RUNTIME.cpu_threads_per_worker > RUNTIME.minimum_effective_cpus:
        raise RuntimeError("CPU allocation exceeds host contract")
    host_threads = (
        RUNTIME.cpu_workers * RUNTIME.cpu_threads_per_worker
        + RUNTIME.gpu_workers * RUNTIME.gpu_host_threads
    )
    if host_threads > RUNTIME.minimum_effective_cpus:
        raise RuntimeError("combined host-thread allocation exceeds host contract")
    if RUNTIME.gpu_workers != RUNTIME.minimum_gpus * RUNTIME.gpu_workers_per_device:
        raise RuntimeError("GPU slot allocation changed")
    return {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "training_cutoff_inclusive": TRAINING_CUTOFF_INCLUSIVE,
        "production_origin": PRODUCTION_ORIGIN,
        "jobs": len(jobs),
        "seeds": list(LATE_SEEDS),
        "finalists": {task: list(ids) for task, ids in FINALISTS.items()},
        "deployable_recipes": DEPLOYABLE_RECIPES,
        "input_sha256": INPUT_SHA256,
        "input_rows": INPUT_ROWS,
        "runtime": asdict(RUNTIME),
        "training_protocol": {
            "stage_1": "temporal head selects only rounds/epochs and calibration",
            "stage_2": "fresh refit on all labels through 2026-08-19",
            "candidate_search": False,
            "hyperparameter_tuning": False,
            "recipe_change": False,
            "post_cutoff_labels": False,
        },
        "serialization": {
            "native_tree_models": True,
            "keras_weights_plus_frozen_architecture": True,
            "encoder_state": True,
            "sequence_normalization_state": True,
            "reload_prediction_parity": True,
        },
        "buy_wait": {
            "status": "GUARDED_PILOT_NOT_PROVEN_SUPERIOR",
            "threshold": BUY_WAIT_THRESHOLD,
            "friction_vnd": NOMINAL_FRICTION_VND,
            "production_default": "BUY",
        },
    }


def manifest() -> dict[str, object]:
    payload = validate_contract()
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
