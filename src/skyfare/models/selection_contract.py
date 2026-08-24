"""Frozen V24 final development-refinement and pre-test freeze contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


CONTRACT_ID = "SKYFARE_V24_FINAL_REFINEMENT_FREEZE_R1"
OBSERVATION_CUTOFF = "2026-07-28"
TEST_BLOCKS = {
    "TEST_1": ("2026-07-29", "2026-08-08"),
    "TEST_2": ("2026-08-09", "2026-08-19"),
}
SCREEN_FOLDS = ("F02", "F03", "F04")
LATE_FOLDS = ("F05", "F06")
REPORT_FOLDS = SCREEN_FOLDS + LATE_FOLDS
HISTORY_FOLD = "F01"
SCREEN_SEEDS = (2026082501, 2026082502)
LATE_SEEDS = (2026082501, 2026082502, 2026082503)
SMOKE_SEED = 2026082590
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
EXCLUDED_FAMILIES = ("EXTRA_TREES", "EXTRA_TREES_QUANTILE")
TASKS = ("CLASSIFICATION", "POINT", "DISTRIBUTION", "RANKING")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    task: str
    family: str
    resource: str
    configuration: str
    window: str = "EXPANDING"


@dataclass(frozen=True)
class RuntimeContract:
    minimum_gpus: int = 2
    minimum_gpu_memory_mib: int = 20_000
    minimum_effective_cpus: int = 40
    minimum_effective_ram_gib: int = 80
    minimum_free_disk_gib: int = 30
    hard_wall_hours: float = 12.0
    finalization_reserve_minutes: int = 60
    initial_gpu_slots: int = 2
    maximum_gpu_slots: int = 4
    cpu_workers: int = 6
    cpu_threads_per_worker: int = 6
    heartbeat_seconds: int = 30
    job_timeout_minutes: int = 90
    adaptive_concurrency_probe: bool = False


RUNTIME = RuntimeContract()

CLASSIFICATION_CANDIDATES = (
    Candidate("C24_LGBM_L31_D003_BAL", "CLASSIFICATION", "LGBM_CLASS", "CPU", "L31_D003_BAL"),
    Candidate("C24_LGBM_L63_D004_BAL", "CLASSIFICATION", "LGBM_CLASS", "CPU", "L63_D004_BAL"),
    Candidate("C24_LGBM_L63_D006_MARKET", "CLASSIFICATION", "LGBM_CLASS", "CPU", "L63_D006_MARKET"),
    Candidate("C24_CAT_GPU_D7_BAL", "CLASSIFICATION", "CAT_CLASS", "GPU", "D7_BAL"),
    Candidate("C24_CAT_GPU_D8_MARKET", "CLASSIFICATION", "CAT_CLASS", "GPU", "D8_MARKET"),
    Candidate("C24_FT_SMALL_D010", "CLASSIFICATION", "FT_CLASS", "GPU", "SMALL_D010"),
    Candidate("C24_FT_WIDE_D015", "CLASSIFICATION", "FT_CLASS", "GPU", "WIDE_D015"),
    Candidate("C24_RNN_EXACT_MTL_U64", "CLASSIFICATION", "SEQ_CLASS", "GPU", "RNN_U64_D015"),
    Candidate("C24_RNN_EXACT_MTL_U96", "CLASSIFICATION", "SEQ_CLASS", "GPU", "RNN_U96_D020"),
    Candidate("C24_GRU_EXACT_MTL_U64", "CLASSIFICATION", "SEQ_CLASS", "GPU", "GRU_U64_D015"),
)

POINT_CANDIDATES = (
    Candidate("R24_LGBM_L31_HUBER", "POINT", "LGBM_POINT", "CPU", "L31_HUBER"),
    Candidate("R24_LGBM_L63_L1", "POINT", "LGBM_POINT", "CPU", "L63_L1"),
    Candidate("R24_LGBM_L63_L2", "POINT", "LGBM_POINT", "CPU", "L63_L2"),
    Candidate("R24_CAT_GPU_D7_RMSE", "POINT", "CAT_POINT", "GPU", "D7_RMSE"),
    Candidate("R24_CAT_GPU_D8_MAE", "POINT", "CAT_POINT", "GPU", "D8_MAE"),
    Candidate("R24_GRU_EXACT_HUBER_U64", "POINT", "SEQ_POINT", "GPU", "GRU_U64_HUBER"),
    Candidate("R24_LSTM_EXACT_HUBER_U64", "POINT", "SEQ_POINT", "GPU", "LSTM_U64_HUBER"),
    Candidate("R24_GRU_EXACT_LOGCOSH_U96", "POINT", "SEQ_POINT", "GPU", "GRU_U96_LOGCOSH"),
)

DISTRIBUTION_CANDIDATES = (
    Candidate("D24_XGB_Q7_D5", "DISTRIBUTION", "XGB_QUANTILE", "GPU", "D5"),
    Candidate("D24_XGB_Q7_D7", "DISTRIBUTION", "XGB_QUANTILE", "GPU", "D7"),
    Candidate("D24_XGB_Q7_D9", "DISTRIBUTION", "XGB_QUANTILE", "GPU", "D9"),
    Candidate("D24_CAT_Q7_D7", "DISTRIBUTION", "CAT_MULTIQUANTILE", "CPU", "D7"),
    Candidate("D24_CAT_NORMAL_D7", "DISTRIBUTION", "CAT_UNCERTAINTY", "GPU", "D7"),
    Candidate("D24_CAT_NORMAL_D8", "DISTRIBUTION", "CAT_UNCERTAINTY", "GPU", "D8"),
)

RANKING_CANDIDATES = (
    Candidate("K24_LGBM_LAMBDA_R84_L31", "RANKING", "LGBM_LAMBDARANK", "CPU", "L31", "RECENT84"),
    Candidate("K24_LGBM_LAMBDA_R84_L63", "RANKING", "LGBM_LAMBDARANK", "CPU", "L63", "RECENT84"),
    Candidate("K24_LGBM_LAMBDA_R84_L127", "RANKING", "LGBM_LAMBDARANK", "CPU", "L127", "RECENT84"),
    Candidate("K24_LGBM_XENDCG_R84_L63", "RANKING", "LGBM_XENDCG", "CPU", "L63", "RECENT84"),
)

CANDIDATES = (
    *CLASSIFICATION_CANDIDATES,
    *POINT_CANDIDATES,
    *DISTRIBUTION_CANDIDATES,
    *RANKING_CANDIDATES,
)


def candidate_map() -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in CANDIDATES}


def job_spec(candidate: Candidate, fold: str, seed: int, phase: str) -> dict[str, object]:
    return {
        **asdict(candidate),
        "fold": fold,
        "seed": int(seed),
        "phase": phase,
        "job_id": f"{candidate.candidate_id}__{fold}__S{seed}",
    }


def smoke_registry() -> list[dict[str, object]]:
    return [job_spec(candidate, "F04", SMOKE_SEED, "SMOKE") for candidate in CANDIDATES]


def screen_registry() -> list[dict[str, object]]:
    jobs = [
        job_spec(candidate, fold, seed, "SCREEN")
        for candidate in CANDIDATES
        for fold in SCREEN_FOLDS
        for seed in SCREEN_SEEDS
    ]
    jobs.extend(
        job_spec(candidate, HISTORY_FOLD, seed, "HISTORY")
        for candidate in (*DISTRIBUTION_CANDIDATES, *RANKING_CANDIDATES)
        for seed in SCREEN_SEEDS
    )
    return jobs


def late_registry(finalist_ids: dict[str, list[str]]) -> list[dict[str, object]]:
    expected = set(TASKS)
    if set(finalist_ids) != expected:
        raise RuntimeError(f"finalist task coverage mismatch: {sorted(finalist_ids)}")
    candidates = candidate_map()
    selected: list[Candidate] = []
    for task in TASKS:
        identifiers = finalist_ids[task]
        if len(identifiers) != 2 or len(set(identifiers)) != 2:
            raise RuntimeError(f"{task}: exactly two unique finalists required")
        for identifier in identifiers:
            candidate = candidates.get(identifier)
            if candidate is None or candidate.task != task:
                raise RuntimeError(f"{task}: invalid finalist {identifier}")
            selected.append(candidate)
    return [
        job_spec(candidate, fold, seed, "LATE")
        for candidate in selected
        for fold in LATE_FOLDS
        for seed in LATE_SEEDS
    ]


def validate_contract() -> dict[str, object]:
    ids = [candidate.candidate_id for candidate in CANDIDATES]
    if len(ids) != 28 or len(ids) != len(set(ids)):
        raise RuntimeError("V24 candidate count or identity changed")
    counts = {task: sum(candidate.task == task for candidate in CANDIDATES) for task in TASKS}
    if counts != {"CLASSIFICATION": 10, "POINT": 8, "DISTRIBUTION": 6, "RANKING": 4}:
        raise RuntimeError(f"V24 task coverage changed: {counts}")
    serialized = json.dumps([asdict(candidate) for candidate in CANDIDATES]).upper()
    if any(name in serialized for name in EXCLUDED_FAMILIES):
        raise RuntimeError("ExtraTrees entered V24")
    if len(smoke_registry()) != 28 or len(screen_registry()) != 188:
        raise RuntimeError("V24 smoke/screen job coverage changed")
    synthetic_finalists = {
        task: [candidate.candidate_id for candidate in CANDIDATES if candidate.task == task][:2]
        for task in TASKS
    }
    if len(late_registry(synthetic_finalists)) != 48:
        raise RuntimeError("V24 late job coverage changed")
    if RUNTIME.cpu_workers * RUNTIME.cpu_threads_per_worker > RUNTIME.minimum_effective_cpus:
        raise RuntimeError("V24 CPU pool exceeds minimum host contract")
    if RUNTIME.initial_gpu_slots != RUNTIME.minimum_gpus or RUNTIME.maximum_gpu_slots > 4:
        raise RuntimeError("V24 GPU slot contract invalid")
    return {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "observation_cutoff": OBSERVATION_CUTOFF,
        "prospective_access": False,
        "automatic_promotion": False,
        "extra_trees_present": False,
        "buy_wait_policy_included": True,
        "candidate_count": len(CANDIDATES),
        "candidate_counts": counts,
        "smoke_jobs": len(smoke_registry()),
        "screen_and_history_jobs": len(screen_registry()),
        "late_jobs": 48,
        "production_jobs": len(screen_registry()) + 48,
    }


def manifest() -> dict[str, object]:
    payload = {
        **validate_contract(),
        "sealed_test_blocks": TEST_BLOCKS,
        "screen_folds": list(SCREEN_FOLDS),
        "late_folds": list(LATE_FOLDS),
        "report_folds": list(REPORT_FOLDS),
        "history_fold": HISTORY_FOLD,
        "screen_seeds": list(SCREEN_SEEDS),
        "late_seeds": list(LATE_SEEDS),
        "quantiles": list(QUANTILES),
        "runtime": asdict(RUNTIME),
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "selection": {
            "classification": "equal-fold proper score; calibration, worst-fold, seed and subgroup gates",
            "point": "equal-fold MAPE; MAE/RMSE/regret/worst-fold guards",
            "distribution": "equal-fold WIS; coverage, width, tail balance and coherence guards",
            "ranking": "equal-query NDCG@5; pairwise concordance and worst-fold guards",
            "policy": "strict-prior BUY/WAIT tournament; mean regret, CVaR95, worst-fold, action-rate and friction gates",
            "screen": "F02-F04 only",
            "late": "F05-F06 development stability gate; not independent confirmation",
            "finalists_per_task": 2,
            "test_labels_used": False,
        },
        "stop_condition": "one frozen recipe per task plus one BUY/WAIT policy; then TEST_1",
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
