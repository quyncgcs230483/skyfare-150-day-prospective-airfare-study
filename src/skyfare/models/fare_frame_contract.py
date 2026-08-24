"""Frozen development contract for next-session fare forecasting."""

from __future__ import annotations

import os


CONTRACT_VERSION = "STRICTLY_PRIOR_FARE_FRAME_R2"
BUNDLE_NAME = "strictly_prior_fare_frame"
OUTPUT_NAME = "fare_frame"
SEED = 20260729
CUTOFF = "2026-07-15"

# Development spans historical Fli, source bridge, and later Trip.com regimes.
# C1/C2 remain locked opened evidence and never participate in selection.
FOLDS = (
    {"fold": "E1", "validation_start": "2026-04-08", "validation_end": "2026-04-14", "role": "HISTORICAL_DRIFT_SCREEN"},
    {"fold": "E2", "validation_start": "2026-05-01", "validation_end": "2026-05-07", "role": "HISTORICAL_DRIFT_SCREEN"},
    {"fold": "B1", "validation_start": "2026-05-15", "validation_end": "2026-05-21", "role": "SOURCE_BRIDGE_SCREEN"},
    {"fold": "S1", "validation_start": "2026-06-06", "validation_end": "2026-06-12", "role": "TRIP_SCREEN"},
    {"fold": "S2", "validation_start": "2026-06-13", "validation_end": "2026-06-19", "role": "TRIP_SCREEN"},
    {"fold": "S3", "validation_start": "2026-06-20", "validation_end": "2026-06-26", "role": "TRIP_SELECT"},
    {"fold": "C1", "validation_start": "2026-06-27", "validation_end": "2026-07-03", "role": "LOCKED_OPENED_EVALUATION"},
    {"fold": "C2", "validation_start": "2026-07-04", "validation_end": "2026-07-10", "role": "LOCKED_OPENED_EVALUATION"},
)
DEVELOPMENT_FOLDS = ("E1", "E2", "B1", "S1", "S2", "S3")
CONFIRMATION_FOLDS = ("C1", "C2")
HISTORICAL_FOLDS = ("E1", "E2")
BRIDGE_FOLDS = ("B1",)
TRIP_DEVELOPMENT_FOLDS = ("S1", "S2", "S3")

BOOKING_WINDOWS = (60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1)
TARGET_NAME = "target_anchor_relative_log"
TARGET_FORMULA = "log(target_session_price_vnd / strictly_prior_market_anchor_vnd)"
PRICE_FORMULA = "strictly_prior_market_anchor_vnd * exp(predicted_target_anchor_relative_log)"
SCHEDULE_SLOT_GRAIN = ("route", "airline", "flight_date", "departure_HHMM")
CANDIDATE_POLICY = "LATEST_STRICTLY_PRIOR_SAME_SCHEDULE_CACHE"
SERVING_ELIGIBLE_CANDIDATE_SOURCES = ("PRIOR_SAME_SLOT_CACHE",)
STANDARD_SOURCE_FILES = (
    "data/interim/standardised/fli_standard_offers.csv",
    "data/interim/standardised/trip_com_standard_offers.csv",
)

TABULAR_MODELS = (
    "RIDGE", "ELASTIC_NET", "DECISION_TREE", "RANDOM_FOREST",
    "HIST_GBM", "XGBOOST", "CATBOOST", "MLP",
)
RECURRENT_MODELS = ("RNN", "GRU", "LSTM", "BILSTM")
RECURRENT_SEQUENCE_LENGTHS = (7, 14, 21)
FOUNDATION_MODELS = ("CHRONOS2",)

CATEGORICAL_FEATURES = (
    "route",
    "airline",
    "source_session_label",
    "target_session_label",
    "session_transition",
    "departure_period",
    "prior_anchor_source",
)

# Every predictor exists at feature_time. Era, identity, target state, and
# outcome-derived drift bands remain attached only for audit and slicing.
NUMERIC_FEATURES = (
    "target_dud",
    "session_gap_hours",
    "flight_day_of_week",
    "flight_month",
    "is_peak_period",
    "departure_time_sin",
    "departure_time_cos",
    "prior_anchor_log",
    "prior_anchor_support_log1p",
    "prior_anchor_age_hours",
    "prior_market_change_pct_per_day",
    "has_prior_market_change",
    "prior_competitor_airline_count",
    "prior_competitor_offer_count",
    "prior_route_min_log_price",
    "prior_route_price_spread_log1p",
    "relative_history_eligible",
    "is_first_observation",
    "history_support_count",
    "previous_relative_log",
    "relative_lag_age_hours",
    "prior_relative_volatility",
    "prior_relative_trend_per_dud_day",
    "has_previous_same_schedule",
    "has_prior_relative_volatility",
    "has_prior_relative_trend",
)

GATE_CATEGORICAL_FEATURES = (
    "route", "airline", "source_session_label", "target_session_label",
    "session_transition", "departure_period", "history_support_band",
    "anchor_fallback_level", "anchor_age_band", "dud_support_mode",
)
GATE_NUMERIC_FEATURES = (
    "target_dud", "session_gap_hours", "departure_time_sin",
    "departure_time_cos", "prior_anchor_support_log1p",
    "prior_anchor_age_hours", "prior_market_change_pct_per_day",
    "has_prior_market_change", "history_support_count",
    "relative_lag_age_hours", "prior_relative_volatility",
    "prior_relative_trend_per_dud_day", "learned_vs_carry_log_ratio",
)
GATE_MARGIN_VND = 25_000.0
GATE_THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(5, 96, 5))
OPERATIONAL_GAP_MAX_HOURS = 30.0

ROUTING_AND_REPORTING_TAGS = (
    "schedule_slot_id", "identity_quality", "is_schedule_fallback",
    "candidate_source", "route_airline", "source_session_key",
    "target_session_key", "source_session_date", "target_session_date",
    "source_session_label", "target_session_label", "session_transition",
    "session_gap_hours", "session_gap_band", "regime",
    "is_first_observation", "history_support_count", "history_support_band",
    "market_group", "coverage_band", "support_tier",
    "route_support_quartile", "train_route_airline_support",
    "route_airline_support_band", "prior_anchor_source",
    "anchor_fallback_level", "anchor_support_band", "anchor_age_band",
    "anchor_collection_era", "anchor_is_fallback", "dud_support_mode",
    "target_batch_exists", "target_observation_state", "feature_time",
    "label_time", "source_collection_era", "collection_era",
    "source_target_era_transition", "fold", "fold_role", "data_cutoff",
    "prediction_path", "hierarchy_level", "model_version",
    "feature_contract_version", "baseline_version", "carry_available",
    "gate_probability", "gate_threshold", "gate_decision",
    "selected_expert", "actual_movement_band",
)

FORBIDDEN_PREDICTORS = (
    "offer_id", "target_offer_id", "schedule_slot_id", "flight_no",
    "flight_date", "departure_time", "target_session_key", "feature_time",
    "label_time", "target_session_price_vnd", "target_anchor_relative_log",
    "price_vnd", "current_relative_log", "candidate_source", "regime",
    "history_support_band", "market_group", "coverage_band", "support_tier",
    "route_support_quartile", "route_airline_support_band", "collection_era",
    "source_collection_era", "source_target_era_transition",
    "target_observation_state", "fold", "hierarchy_level", "prediction_path",
    "identity_quality", "actual_movement_band",
)

COMMON_BASELINE = "STRICT_PRIOR_HIERARCHICAL_ANCHOR_RESIDUAL"
BASELINE_VERSION = "STRICT_PRIOR_HIERARCHICAL_ANCHOR_RESIDUAL_V2"
HIERARCHY_MIN_SUPPORT = 100
BASELINE_HIERARCHY = (
    "ROUTE_AIRLINE_HHMM_DUD_SESSION", "ROUTE_AIRLINE_PERIOD_DUD_SESSION",
    "ROUTE_AIRLINE_DUD_SESSION", "ROUTE_DUD_SESSION", "DUD_SESSION",
    "ZERO_RESIDUAL_ANCHOR",
)

MODEL_ONLY_RECIPES = (
    "BEST_SINGLE", "EQUAL_WEIGHT", "TOP2_EQUAL", "TOP3_EQUAL",
    "TOP5_EQUAL", "INVERSE_MAPE", "SIMPLEX_MAPE",
    "SHRINKAGE_SIMPLEX_MAPE",
)
BASELINE_BLEND_RECIPES = (
    "BASELINE_XGBOOST_FIXED_40_60", "BASELINE_XGBOOST_ALPHA_GRID",
    "BASELINE_ALL_SHRINKAGE_SIMPLEX",
)
GATED_RECIPES = ("DRIFT_AWARE_HARD_GATE", "DRIFT_AWARE_SOFT_GATE")

MARKET_GROUPS = {
    "TRUNK": ("SGN-HAN", "HAN-SGN", "SGN-DAD", "DAD-SGN", "HAN-DAD", "DAD-HAN"),
    "TOURISM": (
        "SGN-PQC", "PQC-SGN", "HAN-PQC", "PQC-HAN", "DAD-PQC",
        "PQC-DAD", "SGN-CXR", "CXR-SGN", "HAN-CXR", "CXR-HAN",
    ),
    "REGIONAL_ALTERNATIVE": ("SGN-HPH", "HPH-SGN", "HAN-VCA", "VCA-HAN"),
}

SUPPORT_BINS = (-1, 0, 99, 499, 1_999, float("inf"))
SUPPORT_LABELS = ("UNSEEN_0", "LOW_1_99", "MEDIUM_100_499", "HIGH_500_1999", "VERY_HIGH_2000_PLUS")
HISTORY_SUPPORT_BINS = (-1, 0, 2, float("inf"))
HISTORY_SUPPORT_LABELS = ("FIRST_0", "COLD_1_2", "WARM_3_PLUS")
ANCHOR_AGE_BINS = (-1, 12, 24, 48, float("inf"))
ANCHOR_AGE_LABELS = ("FRESH_LE_12H", "AGE_12_24H", "AGE_24_48H", "STALE_GT_48H")
SESSION_GAP_BINS = (-1, 6, 18, 30, 48, float("inf"))
SESSION_GAP_LABELS = ("LE_6H", "H6_18", "H18_30", "H30_48", "LONG_GT_48H")

PROMOTION_GATES = {
    "selection_data": "E1_E2_B1_S1_S2_S3_ONLY",
    "confirmation_reselection_allowed": False,
    "minimum_slice_rows": 50,
    "overall": "locked gated system no worse than carry on carry-eligible OOF rows",
    "material_change": "locked gated system beats carry for absolute movement at least 5 percent",
    "era_coverage": "report Fli, bridge, and Trip.com slices separately",
    "operational_gap": "headline excludes session gaps over 30 hours; long gaps remain audit-only",
    "confirmation": "same frozen recipe and gate pass C1 and C2 without reselection",
}


def market_group_lookup() -> dict[str, str]:
    return {route: group for group, routes in MARKET_GROUPS.items() for route in routes}


def require_vast() -> None:
    if os.environ.get("SKYFARE_EXECUTION_ENV") != "VAST":
        raise RuntimeError("Training refused: set SKYFARE_EXECUTION_ENV=VAST on Vast instance.")
