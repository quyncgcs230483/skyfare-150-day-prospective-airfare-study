"""Frozen V19 deep temporal research contract.

V19 is development-only. Prospective blocks stay sealed, ExtraTrees stay excluded,
and automatic deployment promotion is impossible by construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

CONTRACT_ID = "SKYFARE_V19_DEEP_TEMPORAL_RESEARCH_R1"
STATUS = "128_DAY_DEVELOPMENT_ONLY_NO_PROSPECTIVE_ACCESS"
OBSERVATION_CUTOFF = "2026-07-28"
REPORT_FOLDS = ("F02", "F03", "F04", "F05", "F06")
SCREEN_FOLDS = ("F02", "F03", "F04")
LATE_DEVELOPMENT_FOLDS = ("F05", "F06")
BASE_SEEDS = (2026082101, 2026082102)
LATE_SEEDS = (2026082101, 2026082102, 2026082103, 2026082104, 2026082105)
EXCLUDED_FAMILIES = ("EXTRA_TREES",)
PROSPECTIVE_BLOCKS = {
    "BLOCK_1": ("2026-07-29", "2026-08-08"),
    "BLOCK_2": ("2026-08-09", "2026-08-19"),
    "FINAL_EVALUATION": ("UNOPENED", "UNOPENED"),
}


@dataclass(frozen=True)
class Fold:
    fold: str
    validation_start: str
    validation_end: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    task: str
    sequence_mode: str
    objective: str
    sample_weighting: str
    point_loss: str = "NOT_APPLICABLE"


@dataclass(frozen=True)
class RuntimeContract:
    target_training_hours: float = 9.0
    hard_wall_hours: float = 10.0
    finalization_reserve_minutes: int = 45
    single_job_timeout_minutes: int = 45
    checkpoint_loss_minutes_max: int = 10
    smoke_epochs: int = 2
    max_epochs: int = 24
    core_job_count: int = 142
    score_blind_runtime_extension: bool = True


RUNTIME = RuntimeContract()


REGRESSION_FOLDS = (
    Fold("F01", "2026-04-08", "2026-04-14"),
    Fold("F02", "2026-05-01", "2026-05-07"),
    Fold("F03", "2026-05-15", "2026-05-21"),
    Fold("F04", "2026-06-06", "2026-06-12"),
    Fold("F05", "2026-07-01", "2026-07-07"),
    Fold("F06", "2026-07-22", "2026-07-28"),
)

CLASSIFICATION_FOLDS = (
    Fold("F01", "2026-04-08", "2026-04-14"),
    Fold("F02", "2026-05-01", "2026-05-07"),
    Fold("F03", "2026-05-15", "2026-05-21"),
    Fold("F04", "2026-06-06", "2026-06-12"),
    Fold("F05", "2026-06-20", "2026-06-26"),
    Fold("F06", "2026-07-07", "2026-07-13"),
)


CANDIDATES = (
    Candidate("C00_EXACT_GRU_BCE", "CLASSIFICATION", "EXACT", "SINGLE", "ROW"),
    Candidate(
        "C01_EXACT_GRU_WEIGHTED_CORRECTED",
        "CLASSIFICATION",
        "EXACT",
        "SINGLE",
        "CLASS_BALANCED_PRIOR_CORRECTED",
    ),
    Candidate("C02_DUD7_GRU_BCE", "CLASSIFICATION", "TEMPLATE", "SINGLE", "ROW"),
    Candidate("C03_DUAL_GATED_GRU_BCE", "CLASSIFICATION", "DUAL_GATED", "SINGLE", "ROW"),
    Candidate(
        "C04_DUAL_ATTENTION_GRU_BCE",
        "CLASSIFICATION",
        "DUAL_ATTENTION",
        "SINGLE",
        "ROW",
    ),
    Candidate(
        "C05_EXACT_MTL_UNCERTAINTY",
        "CLASSIFICATION",
        "EXACT",
        "MTL_UNCERTAINTY",
        "ROW",
    ),
    Candidate(
        "C06_DUAL_MTL_FIXED",
        "CLASSIFICATION",
        "DUAL_GATED",
        "MTL_FIXED",
        "ROW",
    ),
    Candidate(
        "C07_DUAL_MTL_UNCERTAINTY",
        "CLASSIFICATION",
        "DUAL_GATED",
        "MTL_UNCERTAINTY",
        "ROW",
    ),
    Candidate(
        "C08_DUAL_MTL_GRADNORM",
        "CLASSIFICATION",
        "DUAL_GATED",
        "MTL_GRADNORM",
        "ROW",
    ),
    Candidate(
        "C09_DUAL_GRU_SESSION_WEIGHTED",
        "CLASSIFICATION",
        "DUAL_GATED",
        "SINGLE",
        "EQUAL_TARGET_SESSION",
    ),
    Candidate(
        "C10_DUAL_ORDINAL_RETURN",
        "CLASSIFICATION",
        "DUAL_GATED",
        "ORDINAL_UNCERTAINTY",
        "ROW",
    ),
    Candidate("R00_EXACT_GRU_HUBER", "REGRESSION", "EXACT", "POINT", "ROW", "HUBER"),
    Candidate("R01_DUD7_GRU_HUBER", "REGRESSION", "TEMPLATE", "POINT", "ROW", "HUBER"),
    Candidate("R02_DUAL_GATED_GRU_HUBER", "REGRESSION", "DUAL_GATED", "POINT", "ROW", "HUBER"),
    Candidate(
        "R03_DUAL_ATTENTION_GRU_HUBER",
        "REGRESSION",
        "DUAL_ATTENTION",
        "POINT",
        "ROW",
        "HUBER",
    ),
    Candidate("R04_DUAL_GRU_LOGCOSH", "REGRESSION", "DUAL_GATED", "POINT", "ROW", "LOGCOSH"),
    Candidate("R05_DUAL_GRU_QUANTILE", "REGRESSION", "DUAL_GATED", "QUANTILE", "ROW", "PINBALL"),
)


def candidate_map() -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in CANDIDATES}


def fold_map(task: str) -> dict[str, Fold]:
    folds = CLASSIFICATION_FOLDS if task == "CLASSIFICATION" else REGRESSION_FOLDS
    return {fold.fold: fold for fold in folds}


def _validate_folds(name: str, folds: tuple[Fold, ...]) -> None:
    expected = tuple(f"F{index:02d}" for index in range(1, 7))
    if tuple(fold.fold for fold in folds) != expected:
        raise ValueError(f"{name}: fold identity differs from frozen rolling-origin contract")
    cutoff = date.fromisoformat(OBSERVATION_CUTOFF)
    for fold in folds:
        start = date.fromisoformat(fold.validation_start)
        end = date.fromisoformat(fold.validation_end)
        if start > end or end > cutoff:
            raise ValueError(f"{name}/{fold.fold}: illegal validation interval")


def validate_contract() -> dict[str, object]:
    _validate_folds("classification", CLASSIFICATION_FOLDS)
    _validate_folds("regression", REGRESSION_FOLDS)
    identifiers = [candidate.candidate_id for candidate in CANDIDATES]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs are not unique")
    serialized = json.dumps([asdict(candidate) for candidate in CANDIDATES]).upper()
    for excluded in EXCLUDED_FAMILIES:
        if excluded in serialized:
            raise ValueError(f"excluded family present: {excluded}")
    if len([candidate for candidate in CANDIDATES if candidate.task == "CLASSIFICATION"]) != 11:
        raise ValueError("classification candidate count changed")
    if len([candidate for candidate in CANDIDATES if candidate.task == "REGRESSION"]) != 6:
        raise ValueError("regression candidate count changed")
    expected_core = len(CANDIDATES) * len(SCREEN_FOLDS) * len(BASE_SEEDS) + 4 * len(
        LATE_DEVELOPMENT_FOLDS
    ) * len(LATE_SEEDS)
    if expected_core != RUNTIME.core_job_count:
        raise ValueError((expected_core, RUNTIME.core_job_count))
    if not 9.0 <= RUNTIME.target_training_hours < RUNTIME.hard_wall_hours <= 10.0:
        raise ValueError("runtime contract must target 9-10 hours")
    if RUNTIME.finalization_reserve_minutes < 45:
        raise ValueError("finalization reserve below 45 minutes")
    if REPORT_FOLDS != SCREEN_FOLDS + LATE_DEVELOPMENT_FOLDS:
        raise ValueError("screen/late folds do not cover report folds")
    return {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "protocol_status": STATUS,
        "candidate_count": len(CANDIDATES),
        "classification_candidates": 11,
        "regression_candidates": 6,
        "core_jobs": expected_core,
        "prospective_access": False,
        "extra_trees_present": False,
        "automatic_promotion": False,
    }


def manifest() -> dict[str, object]:
    return {
        **validate_contract(),
        "observation_cutoff": OBSERVATION_CUTOFF,
        "report_folds": list(REPORT_FOLDS),
        "screen_folds": list(SCREEN_FOLDS),
        "late_development_folds": list(LATE_DEVELOPMENT_FOLDS),
        "base_seeds": list(BASE_SEEDS),
        "late_seeds": list(LATE_SEEDS),
        "classification_folds": [asdict(fold) for fold in CLASSIFICATION_FOLDS],
        "regression_folds": [asdict(fold) for fold in REGRESSION_FOLDS],
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "runtime": asdict(RUNTIME),
        "sealed_prospective_blocks": PROSPECTIVE_BLOCKS,
        "selection": {
            "classification": "equal-fold 0.7*Brier + 0.3*log-loss; MAE-free proper scores",
            "regression": "equal-fold MAPE; MAE guard",
            "screen_only": list(SCREEN_FOLDS),
            "late_folds_are_not_independent_confirmation": True,
            "runtime_extension_uses_scores_only_to_identify_frozen_top2_per_task": True,
            "runtime_extension_count_depends_only_on_elapsed_fit_time": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
