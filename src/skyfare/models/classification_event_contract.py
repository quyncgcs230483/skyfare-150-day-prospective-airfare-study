"""Frozen standard-only Pilot115 closing contract."""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = "ANCHOR_EVENT_PILOT115_STANDARD_CLOSING_V2"
SEED = 20260725
CUTOFF = "2026-07-15"

SELECTION_FOLDS = [
    {
        "fold": "CL115_S1",
        "validation_start": "2026-06-06",
        "validation_end": "2026-06-12",
    },
    {
        "fold": "CL115_S2",
        "validation_start": "2026-06-13",
        "validation_end": "2026-06-19",
    },
    {
        "fold": "CL115_S3",
        "validation_start": "2026-06-20",
        "validation_end": "2026-06-26",
    },
]

LOCKED_EVALUATION_BLOCKS = [
    {
        "fold": "CL115_C1",
        "validation_start": "2026-06-27",
        "validation_end": "2026-07-03",
    },
    {
        "fold": "CL115_C2",
        "validation_start": "2026-07-04",
        "validation_end": "2026-07-10",
    },
]

TASKS: dict[str, dict[str, Any]] = {
    "EXACT_NEXT_PRICE": {
        "source": "exact_next",
        "population": "EXACT_OBSERVED_ONLY",
        "current_dud": None,
        "target_dud": None,
        "primary": True,
        "claim": "STANDARD_ONLY_CLOSING_EXPERIMENT",
        "selection_folds": ["CL115_S1", "CL115_S2", "CL115_S3"],
        "evaluation_blocks": ["CL115_C1", "CL115_C2"],
    },
    "DUD1_AM_PM_PRICE": {
        "source": "dud1",
        "population": "DUD1_AM_WITH_OBSERVED_PM_ONLY",
        "current_dud": 1,
        "target_dud": 1,
        "primary": False,
        "claim": "CLOSING_EXPERIMENT",
        "selection_folds": ["CL115_S1", "CL115_S2", "CL115_S3"],
        "evaluation_blocks": ["CL115_C1", "CL115_C2"],
    },
    "MULTI_45_TO_21": {
        "source": "multi_horizon",
        "population": "NAMED_HORIZON_ONLY",
        "current_dud": 45,
        "target_dud": 21,
        "primary": False,
        "claim": "DIAGNOSTIC_ONLY",
        "selection_folds": ["CL115_S2", "CL115_S3"],
        "evaluation_blocks": [],
    },
    "MULTI_30_TO_14": {
        "source": "multi_horizon",
        "population": "NAMED_HORIZON_ONLY",
        "current_dud": 30,
        "target_dud": 14,
        "primary": False,
        "claim": "DIAGNOSTIC_ONLY",
        "selection_folds": ["CL115_S1", "CL115_S2", "CL115_S3"],
        "evaluation_blocks": ["CL115_C1"],
    },
    "MULTI_21_TO_10": {
        "source": "multi_horizon",
        "population": "NAMED_HORIZON_ONLY",
        "current_dud": 21,
        "target_dud": 10,
        "primary": False,
        "claim": "DIAGNOSTIC_ONLY",
        "selection_folds": ["CL115_S1", "CL115_S2", "CL115_S3"],
        "evaluation_blocks": ["CL115_C1", "CL115_C2"],
    },
}

CATEGORICAL_FEATURES = [
    "route",
    "airline",
    "session_label",
    "departure_period",
    "anchor_source",
    "transition",
]

NUMERIC_FEATURES = [
    "days_until_departure",
    "target_dud",
    "horizon_gap_days",
    "flight_day_of_week",
    "flight_month",
    "is_peak_period",
    "departure_time_sin",
    "departure_time_cos",
    "log_price_vnd",
    "current_relative_log",
    "anchor_support_log1p",
    "competitor_airline_count",
    "competitor_offer_count",
    "log_current_over_competitor_min",
    "log_same_airline_alt_over_current",
    "prior_market_change_pct_per_day",
    "has_prior_market_change",
    "relative_history_eligible",
    "previous_relative_log",
    "market_shift_log",
    "relative_lag_age_hours",
    "prior_relative_count",
    "prior_relative_volatility",
    "prior_relative_trend_per_dud_day",
]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "RIDGE": {"alpha": 10.0},
    "ELASTIC_NET": {
        "alpha": 0.001,
        "l1_ratio": 0.20,
        "max_iter": 5000,
        "random_state": SEED,
    },
    "DECISION_TREE": {
        "max_depth": 12,
        "min_samples_leaf": 100,
        "min_samples_split": 200,
        "random_state": SEED,
    },
    "RANDOM_FOREST": {
        "n_estimators": 350,
        "max_depth": 16,
        "min_samples_leaf": 8,
        "max_features": 0.70,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "HIST_GBM": {
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "learning_rate": 0.05,
        "l2_regularization": 0.20,
        "early_stopping": False,
        "random_state": SEED,
    },
    "XGBOOST": {
        "n_estimators": 650,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 8,
        "reg_lambda": 7.0,
        "random_state": SEED,
    },
    "CATBOOST": {
        "iterations": 650,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7.0,
        "random_seed": SEED,
        "verbose": False,
    },
    "MLP": {
        "hidden_units": [128, 64],
        "dropout": 0.15,
        "learning_rate": 0.001,
        "batch_size": 2048,
        "max_epochs": 35,
        "patience": 4,
    },
}

TABULAR_MODELS = list(MODEL_CONFIGS)
DEEP_TASKS = [
    "EXACT_NEXT_PRICE",
    "DUD1_AM_PM_PRICE",
    "MULTI_21_TO_10",
]
SEQUENCE_MIN_LENGTH = 3
SEQUENCE_MAX_LENGTH = 16
SEQUENCE_FEATURES = [
    "current_relative_log",
    "market_shift_log",
    "days_until_departure",
    "log_price_vnd",
    "log_current_over_competitor_min",
]

BASELINES = [
    "CURRENT_PRICE_CARRY_FORWARD",
    "TRAIN_GLOBAL_MEDIAN_LOG_DELTA",
    "TRAIN_HIERARCHICAL_MEDIAN_LOG_DELTA",
]
HIERARCHY_MIN_SUPPORT = 100

LABELS: dict[str, dict[str, Any]] = {
    "DROP_50K": {"kind": "ABSOLUTE", "vnd": 50_000},
    "DROP_2PCT": {"kind": "PERCENT", "pct": 0.02},
    "DROP_3PCT": {"kind": "PERCENT", "pct": 0.03},
    "DROP_4PCT": {"kind": "PERCENT", "pct": 0.04},
    "DROP_5PCT": {"kind": "PERCENT", "pct": 0.05},
    "DROP_MAX_5PCT_50K": {"kind": "MAX", "pct": 0.05, "vnd": 50_000},
}

EVENT_POLICY = {
    "label": "DROP_3PCT",
    "drop_fraction": 0.03,
    "direct_model": "CATBOOST_BALANCED_VND",
    "delta_model": "CATBOOST_PRICE_DELTA",
    "baseline": "FIXED_HIERARCHY",
    "calibration": "ROLLING_CONDITIONAL_LOGISTIC",
    "calibration_features": [
        "raw_score_logit",
        "transition",
        "session_label",
        "is_warm",
        "days_until_departure",
        "relative_lag_age_hours",
        "prior_relative_volatility",
        "prior_relative_trend_per_dud_day",
    ],
    "selection_objective": (
        "MIN_VND_COST_SUBJECT_TO_BALANCE_FLOOR_THEN_MAX_MIN_CLASS_RECALL"
    ),
    "balance_floors": {
        "drop_recall": 0.60,
        "no_drop_recall": 0.60,
    },
    "safety_floors": {
        "drop_recall": 0.75,
        "no_drop_recall": 0.60,
    },
    "slice_router": {
        "dimensions": ["regime", "transition"],
        "minimum_rows": 500,
        "minimum_class_rows": 100,
        "fallback": "FIXED_HIERARCHY",
        "selection_source": "S1_S2_S3_ONLY",
    },
    "vnd_loss": {
        "unsafe_buy_multiplier": 2.0,
        "wait_friction_vnd": 25000.0,
    },
}

DROP_RECALL_FLOOR = EVENT_POLICY["balance_floors"]["drop_recall"]
NO_DROP_RECALL_FLOOR = EVENT_POLICY["balance_floors"]["no_drop_recall"]
UNSAFE_BUY_WEIGHT = EVENT_POLICY["vnd_loss"]["unsafe_buy_multiplier"]
WAIT_FRICTION_VND = EVENT_POLICY["vnd_loss"]["wait_friction_vnd"]
CALIBRATION_DATE_FRACTION = 0.20


def contract_payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "cutoff": CUTOFF,
        "milestone": 115,
        "scope": "STANDARD_ONLY_FULL_CLOSING_AND_DRIFT_AWARE_POLICY",
        "sources": {
            "allowed": [
                "fli_standard_offers.csv",
                "trip_com_standard_offers.csv",
            ],
            "nonstd_allowed": False,
            "tier_unresolved_allowed": False,
            "target_era": "TRIP_COM_BROWSER_ERA",
        },
        "production_claim_allowed": False,
        "fresh_confirmation_required": "MILESTONE120",
        "identity": {
            "semantic_key": [
                "route",
                "airline",
                "flight_date",
                "departure_HHMM",
            ],
            "display_proxy": "airline-HHMM",
            "physical_flight_claim_allowed": False,
            "collision_audit_required": True,
        },
        "tasks": TASKS,
        "target": "log(target_price_vnd / price_vnd)",
        "event_policy": EVENT_POLICY,
        "selection_folds": SELECTION_FOLDS,
        "locked_evaluation_blocks": LOCKED_EVALUATION_BLOCKS,
        "split": {
            "train": "feature_time < validation_start AND label_time < validation_start",
            "validation": "validation_start <= feature_time <= validation_end",
            "strict_feature_before_label": True,
            "random_split_allowed": False,
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "schedule_slot_id_predictor_allowed": False,
            "raw_flight_date_predictor_allowed": False,
            "future_feature_allowed": False,
        },
        "models": {
            "tabular": TABULAR_MODELS,
            "sequence": {
                "family": "GRU",
                "tasks": DEEP_TASKS,
                "excluded_tasks": {
                    "MULTI_45_TO_21": (
                        "ZERO_WARM_TRAIN_ROWS_IN_SUPPORTED_SELECTION_FOLDS"
                    ),
                    "MULTI_30_TO_14": (
                        "ZERO_WARM_TRAIN_ROWS_IN_S1_AND_S2"
                    ),
                },
                "population": "HISTORY_LENGTH_AT_LEAST_3",
                "minimum_length": SEQUENCE_MIN_LENGTH,
                "maximum_length": SEQUENCE_MAX_LENGTH,
                "winner_eligible_for_all_rows": False,
            },
            "chronos2": {
                "mode": "ZERO_SHOT_DETERMINISTIC_SAMPLE",
                "winner_eligible": False,
                "nonblocking": True,
            },
        },
        "baselines": {
            "candidates": BASELINES,
            "hierarchy_min_support": HIERARCHY_MIN_SUPPORT,
            "validation_oracle_baseline_selection_allowed": False,
            "confirmation_baseline": (
                "LOCK_BEST_MEAN_SELECTION_BASELINE_PER_TASK_BEFORE_C1_C2"
            ),
        },
        "selection": {
            "primary": (
                "EQUAL_TRANSITION_MAE_WITHIN_FOLD_THEN_EQUAL_FOLD_MEAN_SKILL_"
                "VS_LOCKED_BASELINE"
            ),
            "winner_requires_positive_mean_skill": True,
            "winner_requires_positive_fold_count": 2,
            "confirmation_cannot_reselect": True,
            "paired_cluster_bootstrap": {
                "cluster": "schedule_slot_id",
                "strata": ["fold", "transition"],
                "selection_replicates": 500,
                "confirmation_replicates": 1000,
            },
        },
        "reporting": {
            "metrics": ["MAE_VND", "MAPE", "R2", "MAE_SKILL"],
            "slices": ["COLD", "WARM", "TRANSITION", "FOLD"],
            "multi_horizon_pooling_allowed": False,
            "off_grid_claim": (
                "LOBWO_RECONSTRUCTION_EVIDENCE_NOT_TRUE_UNOBSERVED_DUD_GROUND_TRUTH"
            ),
            "named_horizon_missing_block_rule": (
                "REPORT_NOT_OBSERVED; NEVER POOL OR SYNTHESIZE A ROW"
            ),
        },
        "standard_only_components": {
            "anchor_nowcast": "RETRAINED_STANDARD_ONLY",
            "future_price": "REBUILT_FROM_STANDARD_ONLY_PRODUCTS",
            "event_policy": "RETRAINED_STANDARD_ONLY_NO_LEGACY_INPUT",
            "reobservation": "RETRAINED_STANDARD_ONLY_COVERAGE_FLAG_ONLY",
            "dud1": "RETRAINED_STANDARD_ONLY",
            "off_grid": "RECOMPUTED_STANDARD_ONLY_LOBWO",
        },
        "guards": [
            "NO_NONSTD_FILE_ACCESS",
            "NO_TIER_UNRESOLVED_ROWS",
            "NO_PSEUDO_LABEL",
            "NO_MISSING_LABEL_IMPUTATION",
            "NO_VALIDATION_LABEL_IN_TRAINING",
            "NO_SCHEDULE_SLOT_ID_AS_PREDICTOR",
            "NO_PHYSICAL_FLIGHT_NUMBER_CLAIM",
            "NO_MULTI_HORIZON_POOLING",
            "NO_C1_C2_RESELECTION",
            "CONFUSION_MATRIX_DIAGNOSTIC_NOT_SOLE_PROMOTION_GATE",
            "EVERY_MATRIX_REQUIRES_FOLD_TIME_SUPPORT_PREVALENCE_AND_VND_COST",
            "SLICE_ROUTER_SHRINKS_TO_HIERARCHY",
            "CHRONOS_FAILURE_CANNOT_BLOCK_MAIN_ARCHIVE",
        ],
    }
