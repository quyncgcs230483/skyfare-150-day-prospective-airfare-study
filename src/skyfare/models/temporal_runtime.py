"""Deterministic IO, temporal splits, encoders and atomic resume for V24."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.models.selection_contract import OBSERVATION_CUTOFF

MODULE_DIR = Path(__file__).resolve().parent
LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
INPUT_ROOT = Path(os.environ.get("SKYFARE_MODEL_INPUT_ROOT", LAYOUT.processed)).resolve()
CONTROL_ROOT = Path(os.environ.get("SKYFARE_MODEL_CONTROL_ROOT", LAYOUT.controls)).resolve()
OUTPUT_ROOT = Path(
    os.environ.get("SKYFARE_MODEL_OUTPUT_ROOT", LAYOUT.artifacts / "model_selection")
).resolve()
CLASSIFICATION_FRAME = INPUT_ROOT / "classification_training_frame.parquet"
REGRESSION_FRAME = INPUT_ROOT / "regression_training_frame.parquet"
OFFERS_FRAME = INPUT_ROOT / "standard_offers.parquet"

CLASS_FOLDS = {
    "F02": ("2026-05-01", "2026-05-07"),
    "F03": ("2026-05-15", "2026-05-21"),
    "F04": ("2026-06-06", "2026-06-12"),
    "F05": ("2026-06-20", "2026-06-26"),
    "F06": ("2026-07-07", "2026-07-13"),
}
REG_FOLDS = {
    "F01": ("2026-04-08", "2026-04-14"),
    "F02": ("2026-05-01", "2026-05-07"),
    "F03": ("2026-05-15", "2026-05-21"),
    "F04": ("2026-06-06", "2026-06-12"),
    "F05": ("2026-07-01", "2026-07-07"),
    "F06": ("2026-07-22", "2026-07-28"),
}

CLASS_CATEGORICAL = (
    "route", "airline", "session_label", "departure_period", "anchor_source", "transition",
)
CLASS_NUMERIC = (
    "days_until_departure", "target_dud", "horizon_gap_days", "flight_day_of_week",
    "flight_month", "is_peak_period", "departure_time_sin", "departure_time_cos",
    "log_price_vnd", "current_relative_log", "anchor_support_log1p",
    "competitor_airline_count", "competitor_offer_count",
    "log_current_over_competitor_min", "log_same_airline_alt_over_current",
    "prior_market_change_pct_per_day", "has_prior_market_change",
    "relative_history_eligible", "previous_relative_log", "market_shift_log",
    "relative_lag_age_hours", "prior_relative_count", "prior_relative_volatility",
    "prior_relative_trend_per_dud_day",
)
REG_CATEGORICAL = (
    "route", "airline", "model_session_label", "departure_period", "prior_anchor_source",
)
REG_NUMERIC = (
    "query_dud", "flight_day_of_week", "flight_month", "is_peak_period",
    "departure_time_sin", "departure_time_cos", "prior_anchor_log",
    "prior_anchor_support_log1p", "prior_anchor_age_hours",
    "prior_market_change_pct_per_day", "has_prior_market_change",
    "prior_competitor_airline_count", "prior_competitor_offer_count",
    "prior_route_min_log_price", "prior_route_price_spread_log1p",
    "history_support_count", "is_first_observation", "previous_relative_log",
    "relative_lag_age_hours", "prior_relative_volatility",
    "prior_relative_trend_per_dud_day", "has_previous_same_schedule",
    "has_prior_relative_volatility", "has_prior_relative_trend",
    "template_history_support_count", "template_previous_relative_log",
    "template_lag_age_hours", "template_prior_relative_volatility",
    "template_prior_relative_trend_per_dud_day", "has_previous_schedule_template",
    "has_template_relative_volatility", "has_template_relative_trend",
)

CLASS_IO_COLUMNS = (
    *CLASS_CATEGORICAL, *CLASS_NUMERIC, "row_key", "offer_id", "target_offer_id",
    "feature_time", "label_time", "DROP_5PCT", "price_vnd", "target_price_vnd",
    "price_change_pct", "history_support_count", "regime", "route_airline",
    "candidate_source", "target_session_key", "transition",
)
REG_IO_COLUMNS = (
    *REG_CATEGORICAL, *REG_NUMERIC, "query_id", "target_offer_id", "source_session_key",
    "target_session_key",
    "feature_time", "label_time", "target_anchor_relative_log", "prior_anchor_vnd",
    "target_session_price_vnd", "query_session_observed_fare_vnd", "route_airline",
    "candidate_source", "regime",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".parquet") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def regression_row_keys(frame: pd.DataFrame) -> pd.Series:
    tokens = frame[["query_id", "target_offer_id", "source_session_key"]].astype("string").fillna("__NA__")
    return pd.util.hash_pandas_object(tokens, index=False).astype("uint64")


@lru_cache(maxsize=1)
def load_classification_frame() -> pd.DataFrame:
    frame = pd.read_parquet(CLASSIFICATION_FRAME, columns=list(dict.fromkeys(CLASS_IO_COLUMNS)))
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], errors="raise")
    frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
    if frame["row_key"].duplicated().any():
        raise RuntimeError("classification row_key is not unique")
    if set(frame["DROP_5PCT"].dropna().unique()) != {0, 1}:
        raise RuntimeError("classification target contract changed")
    return frame


@lru_cache(maxsize=1)
def load_regression_frame() -> pd.DataFrame:
    frame = pd.read_parquet(REGRESSION_FRAME, columns=list(dict.fromkeys(REG_IO_COLUMNS)))
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], errors="raise")
    frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
    frame["row_key"] = regression_row_keys(frame)
    if frame["row_key"].duplicated().any():
        raise RuntimeError("regression row_key is not unique")
    target = frame["target_anchor_relative_log"].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise RuntimeError("regression target contains non-finite values")
    return frame


def outer_split(frame: pd.DataFrame, task: str, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = CLASS_FOLDS if task == "CLASSIFICATION" else REG_FOLDS
    if fold not in folds:
        raise RuntimeError(f"{task}: invalid fold {fold}")
    start_text, end_text = folds[fold]
    start = pd.Timestamp(start_text)
    end = pd.Timestamp(end_text) + pd.Timedelta(days=1)
    if task == "CLASSIFICATION":
        train = frame.loc[frame["feature_time"].lt(start) & frame["label_time"].lt(start)].copy()
        valid = frame.loc[frame["feature_time"].ge(start) & frame["feature_time"].lt(end)].copy()
        if not train["label_time"].lt(start).all():
            raise RuntimeError(f"{fold}: classification outer purge failed")
    else:
        train = frame.loc[frame["label_time"].lt(start)].copy()
        valid = frame.loc[frame["label_time"].ge(start) & frame["label_time"].lt(end)].copy()
        if not train["label_time"].lt(valid["label_time"].min()).all():
            raise RuntimeError(f"{fold}: regression outer purge failed")
    if train.empty or valid.empty:
        raise RuntimeError(f"{task}/{fold}: empty outer split")
    cutoff = pd.Timestamp(OBSERVATION_CUTOFF) + pd.Timedelta(days=1)
    if not valid["label_time"].lt(cutoff).all():
        raise RuntimeError(f"{task}/{fold}: validation crosses observation cutoff")
    return train, valid


def apply_window(train: pd.DataFrame, task: str, fold: str, window: str) -> pd.DataFrame:
    if window == "EXPANDING":
        return train.copy()
    if window != "RECENT84":
        raise KeyError(window)
    folds = CLASS_FOLDS if task == "CLASSIFICATION" else REG_FOLDS
    time_column = "feature_time" if task == "CLASSIFICATION" else "label_time"
    start = pd.Timestamp(folds[fold][0]) - pd.Timedelta(days=84)
    result = train.loc[train[time_column].ge(start)].copy()
    if result.empty:
        raise RuntimeError(f"{task}/{fold}/{window}: empty training window")
    return result


def inner_split(train: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_column = "feature_time" if task == "CLASSIFICATION" else "label_time"
    dates = np.asarray(sorted(train[time_column].dt.normalize().dropna().unique()))
    minimum_fit_dates = 10
    minimum_head_dates = 6 if task == "CLASSIFICATION" else 7
    minimum = minimum_fit_dates + minimum_head_dates
    if len(dates) < minimum:
        raise RuntimeError(f"{task}: fewer than {minimum} dates for inner temporal split")
    requested = max(6, int(np.ceil(len(dates) * 0.15))) if task == "CLASSIFICATION" else max(10, min(21, int(np.ceil(len(dates) * 0.15))))
    reserve = min(requested, len(dates) - minimum_fit_dates)
    if reserve < minimum_head_dates:
        raise RuntimeError(f"{task}: inner temporal head too short: dates={reserve}")
    origin = pd.Timestamp(dates[-reserve])
    if task == "CLASSIFICATION":
        fit = train.loc[train["feature_time"].lt(origin) & train["label_time"].lt(origin)].copy()
        head = train.loc[train["feature_time"].ge(origin)].copy()
        valid_order = fit["label_time"].lt(head["feature_time"].min()).all()
    else:
        fit = train.loc[train["label_time"].lt(origin)].copy()
        head = train.loc[train["label_time"].ge(origin)].copy()
        valid_order = fit["label_time"].lt(head["label_time"].min()).all()
    if fit.empty or head.empty or not valid_order:
        raise RuntimeError(f"{task}: inner temporal purge failed")
    fit_dates = fit[time_column].dt.normalize().nunique()
    head_dates = head[time_column].dt.normalize().nunique()
    if fit_dates < minimum_fit_dates or head_dates < minimum_head_dates:
        raise RuntimeError(f"{task}: inner date coverage failed fit={fit_dates} head={head_dates}")
    return fit, head


def deterministic_subset(frame: pd.DataFrame, task: str, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    if task == "CLASSIFICATION":
        work = frame.assign(_date=frame["feature_time"].dt.normalize(), _stratum=frame["DROP_5PCT"])
    else:
        work = frame.assign(
            _date=frame["label_time"].dt.normalize(),
            _stratum=pd.cut(frame["query_dud"], [-np.inf, 3, 7, 14, 30, 60, np.inf], labels=False),
        )
    groups = max(work.groupby(["_date", "_stratum"], observed=True).ngroups, 1)
    quota = max(8, int(np.ceil(limit / groups)))
    sampled = (
        work.sort_values(["_date", "_stratum", "row_key"], kind="stable")
        .groupby(["_date", "_stratum"], observed=True, group_keys=False)
        .head(quota)
        .drop(columns=["_date", "_stratum"])
    )
    if len(sampled) > limit:
        sampled = sampled.iloc[np.linspace(0, len(sampled) - 1, limit, dtype=np.int64)]
    return sampled.sort_values(["feature_time", "row_key"], kind="stable")


def sample_weights(target: np.ndarray, balanced: bool) -> np.ndarray:
    target = np.asarray(target, dtype=np.int8)
    if not balanced:
        return np.ones(len(target), dtype=np.float32)
    classes, counts = np.unique(target, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise RuntimeError("binary split lacks both classes")
    mapping = {int(label): len(target) / (2.0 * count) for label, count in zip(classes, counts, strict=True)}
    result = np.asarray([mapping[int(value)] for value in target], dtype=np.float32)
    return result / result.mean()


@dataclass
class FeatureEncoder:
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categories: dict[str, list[str]]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, task: str) -> "FeatureEncoder":
        categorical = CLASS_CATEGORICAL if task == "CLASSIFICATION" else REG_CATEGORICAL
        numeric = CLASS_NUMERIC if task == "CLASSIFICATION" else REG_NUMERIC
        categories = {
            column: sorted(frame[column].astype("string").fillna("__MISSING__").unique().tolist())
            for column in categorical
        }
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).to_numpy(dtype=np.float64)
            medians[column] = median
            means[column] = float(np.mean(filled))
            scale = float(np.std(filled))
            scales[column] = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
        return cls(categorical, numeric, categories, medians, means, scales)

    def tree_matrix(self, frame: pd.DataFrame, *, standardized: bool = False) -> np.ndarray:
        categorical = np.empty((len(frame), len(self.categorical_columns)), dtype=np.float32)
        for index, column in enumerate(self.categorical_columns):
            mapping = {value: code + 1 for code, value in enumerate(self.categories[column])}
            values = frame[column].astype("string").fillna("__MISSING__")
            categorical[:, index] = values.map(mapping).fillna(0).to_numpy(dtype=np.float32)
        numeric = np.empty((len(frame), len(self.numeric_columns)), dtype=np.float32)
        for index, column in enumerate(self.numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column]).to_numpy(dtype=float)
            if standardized:
                values = (values - self.means[column]) / self.scales[column]
            numeric[:, index] = values.astype(np.float32)
        return np.column_stack([categorical, numeric]).astype(np.float32)

    def native_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self.categorical_columns:
            result[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
        for column in self.numeric_columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column]).astype(float)
        return result.reset_index(drop=True)

    def neural_inputs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categorical = np.empty((len(frame), len(self.categorical_columns)), dtype=np.int32)
        for index, column in enumerate(self.categorical_columns):
            mapping = {value: code + 1 for code, value in enumerate(self.categories[column])}
            values = frame[column].astype("string").fillna("__MISSING__")
            categorical[:, index] = values.map(mapping).fillna(0).to_numpy(dtype=np.int32)
        numeric = np.empty((len(frame), len(self.numeric_columns)), dtype=np.float32)
        for index, column in enumerate(self.numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column]).to_numpy(dtype=float)
            numeric[:, index] = ((values - self.means[column]) / self.scales[column]).astype(np.float32)
        return categorical, numeric

    def metadata(self) -> dict[str, object]:
        return {
            "categorical_cardinalities": {column: len(values) + 1 for column, values in self.categories.items()},
            "numeric_features": len(self.numeric_columns),
            "fit_only": True,
        }


def ranking_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    eligible = frame.groupby("query_id", sort=False)["query_id"].transform("size").ge(2)
    work = frame.loc[eligible].sort_values(["query_id", "row_key"], kind="stable").copy()
    rank = work.groupby("query_id", sort=False)["target_session_price_vnd"].rank(method="average", pct=True)
    relevance = np.clip(np.floor((1.0 - rank.to_numpy(dtype=float)) * 5.0), 0, 4).astype(np.int32)
    groups = work.groupby("query_id", sort=False).size().astype(int).tolist()
    if not groups or sum(groups) != len(work) or min(groups) < 2:
        raise RuntimeError("ranking group contract failed")
    return work, relevance, groups


def job_root(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


@lru_cache(maxsize=1)
def current_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(MODULE_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def artifact_complete(job_id: str) -> bool:
    root = job_root(job_id)
    done = root / "done.json"
    prediction = root / "predictions.parquet"
    if not done.is_file() or not prediction.is_file():
        return False
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "COMPLETE"
        and payload.get("code_sha256") == current_code_sha256()
        and payload.get("job", {}).get("job_id") == job_id
        and payload.get("prediction_sha256") == sha256(prediction)
        and int(payload.get("rows", -1)) > 0
    )
