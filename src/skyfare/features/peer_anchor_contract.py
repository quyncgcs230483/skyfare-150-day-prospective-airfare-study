"""Frozen contract for current-fare nowcast and exact-next drop pilots."""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = "STRICTLY_PRIOR_ANCHOR_EVENT_R2"
SEED = 20260723

BOOKING_WINDOWS = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1]
INTERIOR_BOOKING_WINDOWS = BOOKING_WINDOWS[1:-1]

MILESTONES: dict[str, dict[str, Any]] = {
    "110": {
        "cutoff": "2026-07-10",
        "offers": "pilot_offer_features_2026-07-10.parquet",
        "exact": "pilot_exact_next_targets_2026-07-10.parquet",
        "multi": "pilot_multi_horizon_targets_2026-07-10.parquet",
        "dud1": "pilot_dud1_intraday_targets_2026-07-10.parquet",
        "input_root": "data/processed/development/day_110",
        "bundle_name": "anchor_event_110_v2",
        "folds": [
            {"fold": "AE110_S1", "validation_start": "2026-06-05", "validation_end": "2026-06-11"},
            {"fold": "AE110_S2", "validation_start": "2026-06-12", "validation_end": "2026-06-18"},
            {"fold": "AE110_S3", "validation_start": "2026-06-19", "validation_end": "2026-06-25"},
        ],
    },
    "115": {
        "cutoff": "2026-07-15",
        "offers": "pilot115_offer_features_2026-07-15.parquet",
        "exact": "pilot115_exact_next_targets_2026-07-15.parquet",
        "multi": "pilot115_multi_horizon_targets_2026-07-15.parquet",
        "dud1": "pilot115_dud1_intraday_targets_2026-07-15.parquet",
        "input_root": "data/processed/development/day_115",
        "bundle_name": "anchor_event_115_v2",
        "folds": [
            {"fold": "AE115_S1", "validation_start": "2026-06-10", "validation_end": "2026-06-16"},
            {"fold": "AE115_S2", "validation_start": "2026-06-17", "validation_end": "2026-06-23"},
            {"fold": "AE115_S3", "validation_start": "2026-06-24", "validation_end": "2026-06-30"},
        ],
    },
}

PEER_ANCHOR_MIN_SUPPORT = 3
MATERIAL_DROP_FRACTION = 0.05
MATERIAL_DROP_VND = 50_000.0
UNSAFE_BUY_WEIGHT = 2.0
SEQUENCE_MIN_PRIOR = 3
SEQUENCE_MAX_PRIOR = 16

EVENT_TARGETS = {
    "ANY_DECREASE": {"kind": "PCT", "value": 0.0},
    "DROP_2PCT": {"kind": "PCT", "value": 0.02},
    "DROP_3PCT": {"kind": "PCT", "value": 0.03},
    "DROP_4PCT": {"kind": "PCT", "value": 0.04},
    "DROP_5PCT": {"kind": "PCT", "value": 0.05},
    "DROP_50K": {"kind": "VND", "value": MATERIAL_DROP_VND},
    "DROP_5PCT_OR_50K": {
        "kind": "MIN_PCT_VND",
        "fraction": MATERIAL_DROP_FRACTION,
        "vnd": MATERIAL_DROP_VND,
    },
}
PRIMARY_EVENT_TARGET = "DROP_5PCT_OR_50K"

CATEGORICAL_FEATURES = ["route", "airline", "session_label"]

REGRESSION_STATIC_FEATURES = [
    *CATEGORICAL_FEATURES,
    "days_until_departure",
    "flight_day_of_week",
    "flight_month",
    "is_peak_period",
    "departure_time_sin",
    "departure_time_cos",
    "anchor_log",
    "anchor_support_log1p",
    "competitor_airline_count",
]

EVENT_STATIC_FEATURES = [
    *REGRESSION_STATIC_FEATURES,
    "current_relative_log",
    "prior_market_change_pct_per_day",
    "has_prior_market_change",
    "is_auxiliary_target",
]

OFFGRID_EVENT_FEATURES = [
    *REGRESSION_STATIC_FEATURES,
]

WARM_FEATURES = [
    "previous_relative_log",
    "market_shift_log",
    "relative_lag_age_hours",
    "prior_relative_count",
    "prior_relative_volatility",
    "prior_relative_trend_per_dud_day",
]

GATE1_MODELS = {
    "regression": ["SPLINE_RIDGE", "DECISION_TREE", "CATBOOST"],
    "event": ["LOGISTIC", "DECISION_TREE", "CATBOOST"],
}

EXPANSION_MODELS = {
    "regression": ["RANDOM_FOREST", "HIST_GBM", "XGBOOST", "MLP"],
    "event": ["RANDOM_FOREST", "HIST_GBM", "XGBOOST", "MLP"],
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "SPLINE_RIDGE": {"alpha": 10.0, "n_knots": 5, "degree": 2},
    "LOGISTIC": {"C": 1.0, "max_iter": 2500, "class_weight": None},
    "DECISION_TREE_REGRESSION": {
        "max_depth": 12,
        "min_samples_leaf": 100,
        "min_samples_split": 200,
        "random_state": SEED,
    },
    "DECISION_TREE_EVENT": {
        "max_depth": 10,
        "min_samples_leaf": 100,
        "min_samples_split": 200,
        "class_weight": None,
        "random_state": SEED,
    },
    "CATBOOST_REGRESSION": {
        "iterations": 550,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    },
    "CATBOOST_EVENT": {
        "iterations": 550,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    },
    "RANDOM_FOREST_REGRESSION": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 10,
        "max_features": 0.7,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "RANDOM_FOREST_EVENT": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 10,
        "max_features": 0.7,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "HIST_GBM_REGRESSION": {
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "learning_rate": 0.05,
        "l2_regularization": 0.2,
        "early_stopping": False,
        "random_state": SEED,
    },
    "HIST_GBM_EVENT": {
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "learning_rate": 0.05,
        "l2_regularization": 0.2,
        "early_stopping": False,
        "random_state": SEED,
    },
    "XGBOOST_REGRESSION": {
        "n_estimators": 550,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8,
        "reg_lambda": 7.0,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "XGBOOST_EVENT": {
        "n_estimators": 550,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8,
        "reg_lambda": 7.0,
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
    },
    "MLP_REGRESSION": {
        "hidden_layer_sizes": (128, 64),
        "alpha": 0.001,
        "batch_size": 2048,
        "learning_rate_init": 0.001,
        "max_iter": 80,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 8,
        "random_state": SEED,
    },
    "MLP_EVENT": {
        "hidden_layer_sizes": (128, 64),
        "alpha": 0.001,
        "batch_size": 2048,
        "learning_rate_init": 0.001,
        "max_iter": 80,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 8,
        "random_state": SEED,
    },
    "GRU": {
        "units": 48,
        "dense_units": 32,
        "dropout": 0.15,
        "batch_size": 2048,
        "epochs": 35,
        "patience": 5,
        "learning_rate": 0.001,
    },
}


def milestone_contract(milestone: str) -> dict[str, Any]:
    if milestone not in MILESTONES:
        raise KeyError(f"Unknown milestone: {milestone}")
    return {
        "contract_version": CONTRACT_VERSION,
        "milestone": milestone,
        **MILESTONES[milestone],
        "prediction_unit": "ONE_OBSERVED_OFFER_IN_ONE_COMPLETED_BATCH",
        "regression": {
            "target": "current_relative_log=log(price_vnd/anchor_vnd)",
            "product": "CURRENT_LISTED_FARE_NOWCAST_IN_VND",
            "baseline": "anchor_vnd",
            "warm_baseline": "previous_relative_log_on_identical_rows",
        },
        "event": {
            "target": PRIMARY_EVENT_TARGET,
            "definition": "saving_vnd >= min(0.05 * current_price_vnd, 50000)",
            "screened_targets": EVENT_TARGETS,
            "training_sources": ["EXACT_ONLY", "EXACT_PLUS_NEAREST_LOWER_AUX"],
            "validation_source": "EXACT_ONLY",
            "action_truth_separate": True,
            "actions": ["BUY", "WAIT"],
            "uncertain_action_allowed": False,
            "unsafe_buy_weight": UNSAFE_BUY_WEIGHT,
        },
        "anchor": {
            "primary": "CURRENT_BATCH_ROUTE_FLIGHT_DATE_LEAVE_ONE_SCHEDULE_SLOT_OUT_MEDIAN",
            "minimum_peer_support": PEER_ANCHOR_MIN_SUPPORT,
            "fallback": "STRICT_PRIOR_ROUTE_AIRLINE_DUD_MEDIAN",
            "low_support_last_resort": "CURRENT_PEER_MEDIAN_WITH_EXPLICIT_LOW_SUPPORT_FLAG",
        },
        "sequence": {
            "minimum_strict_prior_batches": SEQUENCE_MIN_PRIOR,
            "maximum_context_batches": SEQUENCE_MAX_PRIOR,
            "representative_recurrent": "GRU",
            "variant_open_rule": "GRU_MUST_BEAT_LAST_VALUE_PERSISTENCE_ON_MATCHED_ROWS",
        },
        "gates": {
            "mean_lift_strictly_positive": True,
            "minimum_positive_folds": 2,
            "latest_fold_reported_not_silently_pooled": True,
            "chronos_non_blocking": True,
            "winner_selection_folds": [MILESTONES[milestone]["folds"][0]["fold"], MILESTONES[milestone]["folds"][1]["fold"]],
            "confirmation_fold": MILESTONES[milestone]["folds"][2]["fold"],
            "event_train_label_must_precede_validation": True,
        },
        "dud1": {
            "am_to_pm": "SEPARATE_SUPERVISED_INTRADAY_LANE",
            "pm_terminal_action": "BUY",
            "terminal_reason": "NO_LATER_MONITORED_BATCH_BEFORE_DEPARTURE",
        },
        "off_grid": {
            "served_dud_range": [1, 60],
            "observed_ground_truth_duds": BOOKING_WINDOWS,
            "method": "CAUSAL_SAME_BATCH_TWO_SIDED_LOG_INTERPOLATION_PLUS_STATIC_CORRECTION",
            "evidence": "ROLLING_ORIGIN_LEAVE_ONE_BOOKING_WINDOW_OUT",
            "display_status": "OFF_GRID_INTERPOLATED",
            "event_features": OFFGRID_EVENT_FEATURES,
            "event_current_price_feature_allowed": False,
            "dud0_supported": False,
        },
        "serving": {
            "fresh_completed_batch_required": True,
            "on_grid_price_status": "OBSERVED_LISTED_PRICE",
            "off_grid_price_status": "OFF_GRID_INTERPOLATED_FORECAST",
            "off_grid_schedule_requirement": "CACHED_SCHEDULE_SLOT_FROM_A_STRICTLY_PRIOR_OBSERVATION",
            "stale_batch_action": "REFUSE_CURRENT_ADVICE_AND_LABEL_HISTORICAL_REPLAY",
            "database_source": "POSTGRESQL_BATCH_DATABASE_NOT_LIVE_SCRAPING",
        },
        "features": {
            "regression_static": REGRESSION_STATIC_FEATURES,
            "event_static": EVENT_STATIC_FEATURES,
            "offgrid_event": OFFGRID_EVENT_FEATURES,
            "warm": WARM_FEATURES,
        },
    }
