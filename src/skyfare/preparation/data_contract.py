"""Frozen contract for 128-day development-frame construction."""

from __future__ import annotations

import os


CONTRACT_VERSION = "DEVELOPMENT_FRAME_CONTRACT_R1"
OUTPUT_NAME = "development"
BUNDLE_NAME = "development_frames"
SEED = 20260730
DATA_CUTOFF = "2026-07-28"
OBSERVED_DAYS = 128

EXACT_TASK = "CLASSIFICATION"
FARE_TASK = "REGRESSION"

EXACT_BRIDGE_POLICY = "WITHIN_ERA_ONLY"
EXACT_RECIPE = "HIERARCHY_ALL_SHRINKAGE_SIMPLEX"
EXACT_FALLBACK = "CATBOOST"
EXACT_THRESHOLD = 0.30
EXACT_COMPONENTS = (
    "CATBOOST",
    "DELTA_CATBOOST",
    "GRU_L14",
    "GRU_L21",
    "GRU_L7",
    "LINEAR_SVM",
    "LOGISTIC",
    "LSTM_L14",
    "LSTM_L21",
    "LSTM_L7",
    "MLP",
    "RNN_L14",
    "RNN_L21",
    "RNN_L7",
    "XGBOOST",
)

FARE_RECIPE = "SIMPLEX_LOG_MSE"
FARE_FALLBACK = "CATBOOST"
FARE_COMPONENTS = (
    "BILSTM_L14",
    "BILSTM_L7",
    "GRU_L21",
    "GRU_L7",
    "LSTM_L21",
    "RNN_L14",
    "RNN_L7",
    "XGBOOST",
    "CATBOOST",
)

ALL_COMPONENTS = {
    EXACT_TASK: EXACT_COMPONENTS,
    FARE_TASK: FARE_COMPONENTS,
}

CANONICAL_DUDS = (60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1)
SUPPORTED_ACTIONS = ("BUY", "WAIT")


def parse_component(token: str) -> tuple[str, int | None]:
    if "_L" not in token:
        return token, None
    model, length = token.rsplit("_L", 1)
    return model, int(length)


def require_vast() -> None:
    if os.environ.get("SKYFARE_EXECUTION_ENV") != "VAST":
        raise RuntimeError(
            "Development-frame build refused: set SKYFARE_EXECUTION_ENV=VAST."
        )
