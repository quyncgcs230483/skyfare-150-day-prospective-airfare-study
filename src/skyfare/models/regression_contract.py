"""Frozen contract for as-of-session future-departure slot fare forecasting."""

from __future__ import annotations

import os

CONTRACT_VERSION = "TEMPORAL_POINT_FARE_REGRESSION_R1"
BUNDLE_PREFIX = "temporal_point_fare_regression"
OUTPUT_PREFIX = "point_regression"
SEED = 20260729
CUTOFF = "2026-07-15"

FOLDS = (
    {"fold": "E1", "validation_start": "2026-04-08", "validation_end": "2026-04-14", "role": "HISTORICAL_DRIFT_SCREEN"},
    {"fold": "E2", "validation_start": "2026-05-01", "validation_end": "2026-05-07", "role": "HISTORICAL_DRIFT_SCREEN"},
    {"fold": "B1", "validation_start": "2026-05-15", "validation_end": "2026-05-21", "role": "SOURCE_BRIDGE_SCREEN"},
    {"fold": "S1", "validation_start": "2026-06-06", "validation_end": "2026-06-12", "role": "TRIP_SCREEN"},
    {"fold": "S2", "validation_start": "2026-06-13", "validation_end": "2026-06-19", "role": "TRIP_SCREEN"},
    {"fold": "S3", "validation_start": "2026-06-20", "validation_end": "2026-06-26", "role": "TRIP_DEVELOPMENT"},
    {"fold": "C1", "validation_start": "2026-06-27", "validation_end": "2026-07-03", "role": "LOCKED_CONFIRMATION"},
    {"fold": "C2", "validation_start": "2026-07-04", "validation_end": "2026-07-10", "role": "LOCKED_CONFIRMATION"},
)
SHARD_FOLDS = {
    "A": ("E1", "E2", "B1"),
    "B": ("S1", "S2", "S3"),
}
CONFIRMATION_FOLDS = ("C1", "C2")
BOOKING_WINDOWS = (60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1)
INTERIOR_BOOKING_WINDOWS = BOOKING_WINDOWS[1:-1]

TASK_NAME = "REGRESSION"
TARGET_NAME = "target_anchor_relative_log"
TARGET_FORMULA = "log(query_session_observed_fare_vnd / strictly_prior_hierarchical_anchor_vnd)"
PRICE_FORMULA = "strictly_prior_hierarchical_anchor_vnd * exp(predicted_anchor_relative_log)"
COMMON_BASELINE = "STRICT_PRIOR_HIERARCHICAL_FARE_SURFACE"
BASELINE_VERSION = "STRICT_PRIOR_HIERARCHICAL_FARE_SURFACE_V1"

QUERY_GRAIN = ("session_date", "model_session_label", "route", "flight_date")
SCHEDULE_SLOT_GRAIN = ("route", "airline", "flight_date", "departure_HHMM")
SCHEDULE_TEMPLATE_GRAIN = ("route", "airline", "departure_HHMM")
SERVING_ELIGIBLE_CANDIDATE_SOURCES = (
    "PRIOR_SAME_SLOT_CACHE",
    "PRIOR_SCHEDULE_TEMPLATE_CACHE",
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
    "model_session_label",
    "departure_period",
    "prior_anchor_source",
)
NUMERIC_FEATURES = (
    "query_dud",
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
    "history_support_count",
    "is_first_observation",
    "previous_relative_log",
    "relative_lag_age_hours",
    "prior_relative_volatility",
    "prior_relative_trend_per_dud_day",
    "has_previous_same_schedule",
    "has_prior_relative_volatility",
    "has_prior_relative_trend",
    "template_history_support_count",
    "template_previous_relative_log",
    "template_lag_age_hours",
    "template_prior_relative_volatility",
    "template_prior_relative_trend_per_dud_day",
    "has_previous_schedule_template",
    "has_template_relative_volatility",
    "has_template_relative_trend",
)

FORBIDDEN_PREDICTORS = (
    "offer_id", "target_offer_id", "query_id", "schedule_slot_id",
    "schedule_template_id", "flight_no", "flight_date", "departure_time",
    "session_key", "source_session_key", "feature_time", "label_time",
    "query_session_observed_fare_vnd", "target_anchor_relative_log",
    "price_vnd", "anchor_vnd", "current_relative_log",
    "competitor_min_price_other_airlines", "same_airline_alternative_min_price",
    "candidate_source", "identity_quality", "is_schedule_fallback",
    "regime", "history_support_band", "template_history_support_band",
    "market_group", "coverage_band", "support_tier",
    "route_support_quartile", "route_airline_support_band",
    "collection_era", "anchor_collection_era", "target_observation_state",
    "fold", "fold_role", "hierarchy_level", "prediction_path",
)

ROUTING_AND_REPORTING_TAGS = (
    "query_id", "schedule_slot_id", "schedule_template_id",
    "identity_quality", "is_schedule_fallback", "candidate_source",
    "route_airline", "regime", "is_first_observation",
    "history_support_count", "history_support_band",
    "template_history_support_count", "template_history_support_band",
    "market_group", "coverage_band", "support_tier",
    "route_support_quartile", "train_route_airline_support",
    "route_airline_support_band", "prior_anchor_source",
    "anchor_fallback_level", "anchor_support_band", "anchor_age_band",
    "anchor_collection_era", "anchor_is_fallback", "dud_support_mode",
    "target_batch_exists", "target_observation_state", "feature_time",
    "label_time", "collection_era", "fold", "fold_role", "data_cutoff",
    "prediction_path", "hierarchy_level", "model_version",
    "feature_contract_version", "baseline_version",
)

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


def market_group_lookup() -> dict[str, str]:
    return {route: group for group, routes in MARKET_GROUPS.items() for route in routes}


def fold_spec(name: str) -> dict[str, str]:
    selected = [item for item in FOLDS if item["fold"] == name]
    if len(selected) != 1:
        raise KeyError(name)
    return selected[0]


def require_vast() -> None:
    if os.environ.get("SKYFARE_EXECUTION_ENV") != "VAST":
        raise RuntimeError("Training refused: set SKYFARE_EXECUTION_ENV=VAST on Vast instance.")
