"""Frozen contract for full-era Classification selection."""

from __future__ import annotations

import os

CONTRACT_VERSION = "TEMPORAL_FARE_DROP_CLASSIFICATION_R2"
BUNDLE_NAME = "temporal_fare_drop_classification"
OUTPUT_NAME = "classification"
SEED = 20260729
CUTOFF = "2026-07-15"
TARGET_NAME = "DROP_5PCT"
TARGET_FORMULA = "target_price_vnd <= 0.95 * source_price_vnd"

BOOKING_WINDOWS = (60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1)
NEXT_DUD = {60: 45, 45: 30, 30: 21, 21: 14, 14: 10, 10: 7, 7: 5, 5: 3, 3: 2, 2: 1}

FOLDS = (
    {"fold": "E1", "validation_start": "2026-04-08", "validation_end": "2026-04-14", "role": "HISTORICAL_SCREEN"},
    {"fold": "E2", "validation_start": "2026-05-01", "validation_end": "2026-05-07", "role": "BRIDGE_ONSET_SCREEN"},
    {"fold": "B1", "validation_start": "2026-05-08", "validation_end": "2026-05-14", "role": "SOURCE_BRIDGE_SCREEN"},
    {"fold": "S1", "validation_start": "2026-06-06", "validation_end": "2026-06-12", "role": "TRIP_META_FIT"},
    {"fold": "S2", "validation_start": "2026-06-13", "validation_end": "2026-06-19", "role": "TRIP_META_FIT"},
    {"fold": "S3", "validation_start": "2026-06-20", "validation_end": "2026-06-26", "role": "TRIP_DEVELOPMENT"},
    {"fold": "C1", "validation_start": "2026-06-27", "validation_end": "2026-07-03", "role": "LOCKED_OPENED_EVIDENCE"},
    {"fold": "C2", "validation_start": "2026-07-04", "validation_end": "2026-07-10", "role": "LOCKED_OPENED_EVIDENCE"},
)
META_FIT_FOLDS = ("E1", "E2", "B1", "S1", "S2", "S3")
LOCK_SCREEN_FOLDS: tuple[str, ...] = ()
CONFIRMATION_FOLDS = ("C1", "C2")
DEVELOPMENT_FOLDS = META_FIT_FOLDS

BRIDGE_POLICIES = ("WITHIN_ERA_ONLY", "INCLUDE_BRIDGE")
ERA_TRANSITIONS = (
    "FLI_LIBRARY_ERA->FLI_LIBRARY_ERA",
    "FLI_LIBRARY_ERA->TRIP_COM_BROWSER_ERA",
    "TRIP_COM_BROWSER_ERA->TRIP_COM_BROWSER_ERA",
)

TABULAR_MODELS = (
    "LOGISTIC", "LINEAR_SVM", "DECISION_TREE", "RANDOM_FOREST",
    "HIST_GBM", "XGBOOST", "CATBOOST", "DELTA_CATBOOST", "MLP",
)
RECURRENT_MODELS = ("RNN", "GRU", "LSTM", "BILSTM")
SEQUENCE_LENGTHS = (7, 14, 21)

CATEGORICAL_FEATURES = (
    "route", "airline", "session_label", "departure_period",
    "anchor_source", "transition",
)
NUMERIC_FEATURES = (
    "days_until_departure", "target_dud", "horizon_gap_days",
    "flight_day_of_week", "flight_month", "is_peak_period",
    "departure_time_sin", "departure_time_cos", "log_price_vnd",
    "current_relative_log", "anchor_support_log1p",
    "competitor_airline_count", "competitor_offer_count",
    "log_current_over_competitor_min", "log_same_airline_alt_over_current",
    "prior_market_change_pct_per_day", "has_prior_market_change",
    "relative_history_eligible", "previous_relative_log", "market_shift_log",
    "relative_lag_age_hours", "prior_relative_count",
    "prior_relative_volatility", "prior_relative_trend_per_dud_day",
)

FORBIDDEN_PREDICTORS = (
    "offer_id", "target_offer_id", "schedule_slot_id", "flight_date",
    "departure_time", "feature_time", "label_time", "target_session_key",
    "target_price_vnd", TARGET_NAME, "material_drop_next", "price_change_vnd",
    "price_change_pct", "source_target_era_transition", "collection_era",
    "target_collection_era", "bridge_label_stability", "identity_quality",
    "candidate_source", "regime", "fold", "fold_role",
    "target_observation_state", "prediction_path", "hierarchy_level",
)

ROUTING_AND_REPORTING_TAGS = (
    "schedule_slot_id", "identity_quality", "is_schedule_fallback",
    "candidate_source", "route_airline", "regime", "is_first_observation",
    "history_support_count", "history_support_band", "market_group",
    "coverage_band", "support_tier", "route_support_quartile",
    "train_route_airline_support", "route_airline_support_band",
    "anchor_source", "anchor_fallback_level", "anchor_support_band",
    "anchor_age_band", "anchor_collection_era", "anchor_is_fallback",
    "dud_support_mode", "target_batch_exists", "target_observation_state",
    "source_session_key", "target_session_key", "feature_time", "label_time",
    "collection_era", "target_collection_era", "source_target_era_transition",
    "bridge_label_stability", "fold", "fold_role", "data_cutoff",
    "prediction_path", "hierarchy_level", "model_version",
    "feature_contract_version", "baseline_version",
)

# Assigned only after a row enters a concrete temporal split. Keeping these
# absent from the unsplit cache avoids inventing one fold identity for rows
# reused as legal training history across several expanding folds.
SPLIT_ASSIGNED_TAGS = ("fold", "fold_role")

BASELINE = "FIXED_ROUTE_AIRLINE_TRANSITION_HIERARCHY"
BASELINE_VERSION = "FIXED_HIERARCHY_V2_FULL_ERA"
WEIGHTING = "BALANCED_VND_EQUAL_TRANSITION"
WINDOW = "EXPANDING_ALL_LEGAL_HISTORY"
MIN_RECURRENT_HISTORY = 3

ENSEMBLE_RECIPES = (
    "BEST_SINGLE", "EQUAL_WEIGHT", "TOP2_EQUAL", "TOP3_EQUAL", "TOP5_EQUAL",
    "INVERSE_BRIER", "SIMPLEX_BRIER", "SHRINKAGE_SIMPLEX_BRIER",
    "FAMILY_BALANCED", "HIERARCHY_XGBOOST_ALPHA_GRID",
    "HIERARCHY_ALL_SHRINKAGE_SIMPLEX",
)
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(5, 96, 5))
MIN_SLICE_ROWS = 100

PROMOTION_GATES = {
    "selection": "recipe and bridge policy selected on E1,E2,B1,S1,S2,S3 only",
    "feature_review": "grouped diagnostics reviewed before recipe freeze",
    "confirmation": "same frozen system beats hierarchy Brier on C1 and C2",
    "policy": "same frozen BUY/WAIT threshold has no greater VND regret than hierarchy policy",
    "reselection": False,
    "c3_included": False,
}


def fold_spec(name: str) -> dict[str, str]:
    return next(item for item in FOLDS if item["fold"] == name)


def require_vast() -> None:
    if os.environ.get("SKYFARE_EXECUTION_ENV") != "VAST":
        raise RuntimeError("Training refused: set SKYFARE_EXECUTION_ENV=VAST on Vast.")
