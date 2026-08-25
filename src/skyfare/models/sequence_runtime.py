"""Deterministic IO, rolling-origin splits, and leakage-safe V19 sequences."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.models.deep_temporal_contract import OBSERVATION_CUTOFF, fold_map

CODE_ROOT = Path(__file__).resolve().parent
LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
INPUT_ROOT = Path(os.environ.get("SKYFARE_MODEL_INPUT_ROOT", LAYOUT.processed)).resolve()
CONTROL_ROOT = Path(os.environ.get("SKYFARE_MODEL_CONTROL_ROOT", LAYOUT.controls)).resolve()
OUTPUT_ROOT = Path(
    os.environ.get("SKYFARE_SEQUENCE_OUTPUT_ROOT", LAYOUT.artifacts / "sequence_models")
).resolve()
CLASSIFICATION_FRAME = INPUT_ROOT / "classification_training_frame.parquet"
REGRESSION_FRAME = INPUT_ROOT / "regression_training_frame.parquet"
OFFERS_FRAME = INPUT_ROOT / "standard_offers.parquet"
EXACT_LENGTH = 21
TEMPLATE_LENGTH = 42


CLASS_CATEGORICAL = (
    "route",
    "airline",
    "session_label",
    "departure_period",
    "anchor_source",
    "transition",
)
CLASS_NUMERIC = (
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
)
REG_CATEGORICAL = (
    "route",
    "airline",
    "model_session_label",
    "departure_period",
    "prior_anchor_source",
)
REG_NUMERIC = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".parquet"
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_times(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "feature_time",
        "label_time",
        "flight_date",
        "departure_time",
        "session_date",
        "target_session_date",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def temporal_split(
    frame: pd.DataFrame, task: str, fold: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = fold_map(task)[fold]
    start = pd.Timestamp(spec.validation_start)
    end = pd.Timestamp(spec.validation_end) + pd.Timedelta(days=1)
    if task == "CLASSIFICATION":
        train = frame.loc[
            frame["feature_time"].lt(start) & frame["label_time"].lt(start)
        ].copy()
        valid = frame.loc[
            frame["feature_time"].ge(start) & frame["feature_time"].lt(end)
        ].copy()
    elif task == "REGRESSION":
        train = frame.loc[frame["label_time"].lt(start)].copy()
        valid = frame.loc[
            frame["label_time"].ge(start) & frame["label_time"].lt(end)
        ].copy()
    else:
        raise KeyError(task)
    if train.empty or valid.empty:
        raise RuntimeError(f"{task}/{fold}: empty temporal train or validation")
    if not train["label_time"].lt(start).all():
        raise RuntimeError(f"{task}/{fold}: outer label-time purge failed")
    cutoff = pd.Timestamp(OBSERVATION_CUTOFF) + pd.Timedelta(days=1)
    if not valid["label_time"].lt(cutoff).all():
        raise RuntimeError(f"{task}/{fold}: validation label crosses 128-day cutoff")
    return train, valid


def inner_temporal_head(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.asarray(sorted(train["feature_time"].dt.normalize().dropna().unique()))
    if len(dates) < 16:
        raise RuntimeError("fewer than 16 dates for temporal early-stopping split")
    reserve = max(6, int(np.ceil(len(dates) * 0.15)))
    boundary = pd.Timestamp(dates[-reserve])
    fit = train.loc[
        train["feature_time"].lt(boundary) & train["label_time"].lt(boundary)
    ].copy()
    head = train.loc[train["feature_time"].ge(boundary)].copy()
    if fit.empty or head.empty:
        raise RuntimeError("empty fit/head temporal partition")
    if not fit["label_time"].lt(head["feature_time"].min()).all():
        raise RuntimeError("fit/head label-time purge failed")
    return fit, head


def balanced_smoke(
    frame: pd.DataFrame, target: str, max_rows: int = 4096, max_dates: int = 24
) -> pd.DataFrame:
    work = frame.copy()
    work["_date"] = work["feature_time"].dt.normalize()
    dates = sorted(work["_date"].dropna().unique())[-max_dates:]
    work = work.loc[work["_date"].isin(dates)].copy()
    if target in work and 1 < work[target].nunique() <= 20:
        groups = max(work.groupby(["_date", target], observed=True).ngroups, 1)
        quota = max(8, max_rows // groups)
        work = (
            work.sort_values(["_date", target], kind="stable")
            .groupby(["_date", target], observed=True, group_keys=False)
            .head(quota)
        )
    elif len(work) > max_rows:
        positions = np.linspace(0, len(work) - 1, max_rows, dtype=np.int64)
        work = work.iloc[positions]
    return work.drop(columns="_date").sort_values("feature_time", kind="stable")


def _template_tokens(frame: pd.DataFrame) -> pd.Series:
    minute = pd.to_numeric(frame["departure_minute"], errors="coerce").fillna(-1).astype(int)
    dud7 = (
        pd.to_numeric(frame["days_until_departure"], errors="coerce")
        .fillna(-7)
        .floordiv(7)
        .astype(int)
    )
    return (
        frame["route"].astype("string").fillna("__NA__")
        + "|"
        + frame["airline"].astype("string").fillna("__NA__")
        + "|"
        + minute.astype("string")
        + "|D7="
        + dud7.astype("string")
    )


def _raw_sequence_features(offers: pd.DataFrame) -> pd.DataFrame:
    price = pd.to_numeric(offers["price_vnd"], errors="coerce").clip(lower=1.0)
    market = pd.to_numeric(
        offers["temporal_market_median_price"], errors="coerce"
    ).clip(lower=1.0)
    anchor = pd.to_numeric(offers["anchor_vnd"], errors="coerce").clip(lower=1.0)
    denominator = market.fillna(anchor).fillna(price)
    relative = np.log(price / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    support = pd.to_numeric(
        offers["temporal_market_support"], errors="coerce"
    ).fillna(0.0)
    return pd.DataFrame(
        {
            "relative_log": relative.astype(float),
            "log_price": np.log(price).astype(float),
            "dud_scaled": pd.to_numeric(
                offers["days_until_departure"], errors="coerce"
            ).fillna(0.0).clip(0, 60).div(60),
            "support_log": np.log1p(support.clip(lower=0)),
            "batch_log": np.log1p(
                pd.to_numeric(offers["batch_offer_count"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0)
            ),
        },
        index=offers.index,
    )


def build_sequence_source(offers: pd.DataFrame, kind: str) -> dict[str, object]:
    """Aggregate one timestamp/session into one legal sequence observation."""
    work = offers[
        [
            "offer_id",
            "schedule_slot_id",
            "session_key",
            "feature_time",
            "route",
            "airline",
            "departure_minute",
            "days_until_departure",
            "price_vnd",
            "temporal_market_median_price",
            "temporal_market_support",
            "anchor_vnd",
            "batch_offer_count",
        ]
    ].copy()
    if kind == "EXACT":
        group_token = work["schedule_slot_id"].astype("string")
    elif kind == "TEMPLATE_DUD7":
        group_token = _template_tokens(work)
    else:
        raise KeyError(kind)
    work["_group_token"] = group_token
    codes, unique = pd.factorize(group_token, sort=True)
    if (codes < 0).any():
        raise RuntimeError(f"{kind}: missing sequence group")
    work["_group_code"] = codes.astype(np.int32)
    query_group = pd.Series(
        work["_group_code"].to_numpy(dtype=np.int32),
        index=work["offer_id"].astype("uint64"),
    )
    if not query_group.index.is_unique:
        raise RuntimeError(f"{kind}: offer identity is not unique")
    raw = _raw_sequence_features(work)
    for column in raw:
        work[column] = raw[column]
    batch = (
        work.groupby(
            ["_group_code", "session_key", "feature_time"],
            observed=True,
            sort=False,
            as_index=False,
        )
        .agg(
            relative_log=("relative_log", "median"),
            log_price=("log_price", "median"),
            dud_scaled=("dud_scaled", "median"),
            support_log=("support_log", "median"),
            batch_log=("batch_log", "median"),
            offers_in_batch=("offer_id", "size"),
        )
        .sort_values(["_group_code", "feature_time", "session_key"], kind="stable")
        .reset_index(drop=True)
    )
    batch["delta_hours"] = (
        batch.groupby("_group_code", observed=True)["feature_time"]
        .diff()
        .dt.total_seconds()
        .div(3600)
        .fillna(0)
        .clip(0, 24 * 128)
    )
    values = np.column_stack(
        [
            batch["relative_log"],
            batch["log_price"],
            batch["dud_scaled"],
            batch["support_log"],
            np.log1p(batch["offers_in_batch"]),
            np.log1p(batch["delta_hours"]),
        ]
    ).astype(np.float32)
    group_count = len(unique)
    starts = np.full(group_count, -1, dtype=np.int64)
    ends = np.full(group_count, -1, dtype=np.int64)
    grouped = batch.groupby("_group_code", observed=True, sort=False).indices
    for code, positions in grouped.items():
        starts[int(code)] = int(positions[0])
        ends[int(code)] = int(positions[-1]) + 1
    if (starts < 0).any() or (ends <= starts).any():
        raise RuntimeError(f"{kind}: incomplete group ranges")
    time_ns = batch["feature_time"].astype("int64").to_numpy()
    return {
        "kind": kind,
        "values": values,
        "query_group": query_group,
        "starts": starts,
        "ends": ends,
        "time_ns": time_ns,
        "session_key": batch["session_key"].astype("string").to_numpy(),
        "groups": group_count,
        "batches": len(batch),
    }


def load_or_build_sequence_source(kind: str) -> dict[str, object]:
    cache_root = OUTPUT_ROOT / "cache"
    token = kind.lower()
    cache = cache_root / f"sequence_{token}_v19.joblib"
    manifest = cache_root / f"sequence_{token}_v19.json"
    source_hash = sha256(OFFERS_FRAME)
    if cache.is_file() and manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                payload.get("status") == "COMPLETE"
                and payload.get("offers_sha256") == source_hash
                and payload.get("kind") == kind
                and payload.get("cache_sha256") == sha256(cache)
            ):
                return joblib.load(cache)
        except Exception:
            pass
    offers = normalize_times(pd.read_parquet(OFFERS_FRAME))
    source = build_sequence_source(offers, kind)
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.joblib")
    joblib.dump(source, temporary, compress=3)
    os.replace(temporary, cache)
    write_json_atomic(
        manifest,
        {
            "status": "COMPLETE",
            "kind": kind,
            "offers_sha256": source_hash,
            "cache_sha256": sha256(cache),
            "groups": source["groups"],
            "batches": source["batches"],
            "feature_count": int(source["values"].shape[1]),
        },
    )
    return source


def make_sequences(
    offer_ids: pd.Series,
    cutoff_times: pd.Series,
    source: dict[str, object],
    *,
    length: int,
    inclusive: bool,
    forbidden_sessions: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = source["query_group"].reindex(offer_ids.astype("uint64")).to_numpy()
    parsed = pd.to_datetime(cutoff_times, errors="coerce")
    if parsed.isna().any():
        raise RuntimeError("sequence cutoff contains invalid timestamps")
    cutoff_ns = parsed.astype("int64").to_numpy()
    forbidden = (
        forbidden_sessions.astype("string").to_numpy()
        if forbidden_sessions is not None
        else None
    )
    values = source["values"]
    starts = source["starts"]
    ends = source["ends"]
    time_ns = source["time_ns"]
    sessions = source["session_key"]
    sequences = np.zeros((len(offer_ids), length, values.shape[1]), dtype=np.float32)
    masks = np.zeros((len(offer_ids), length), dtype=bool)
    lengths = np.zeros(len(offer_ids), dtype=np.int16)
    side = "right" if inclusive else "left"
    for row, raw_group in enumerate(groups):
        if pd.isna(raw_group):
            continue
        code = int(raw_group)
        group_start, group_end = int(starts[code]), int(ends[code])
        stop = group_start + int(
            np.searchsorted(time_ns[group_start:group_end], cutoff_ns[row], side=side)
        )
        candidates = np.arange(max(group_start, stop - length), stop, dtype=np.int64)
        if forbidden is not None and len(candidates):
            candidates = candidates[sessions[candidates] != forbidden[row]]
            candidates = candidates[-length:]
        current = values[candidates]
        sequences[row, : len(current)] = current
        masks[row, : len(current)] = True
        lengths[row] = len(current)
        if len(candidates):
            if inclusive and time_ns[candidates].max() > cutoff_ns[row]:
                raise RuntimeError("inclusive sequence crosses feature-time cutoff")
            if not inclusive and time_ns[candidates].max() >= cutoff_ns[row]:
                raise RuntimeError("strict sequence reaches current feature-time batch")
            if forbidden is not None and np.any(sessions[candidates] == forbidden[row]):
                raise RuntimeError("target session leaked into sequence")
    if np.any(np.diff(masks.astype(np.int8), axis=1) > 0):
        raise RuntimeError("sequence masks are not right-padded")
    return sequences, masks, lengths


def normalize_sequence_sets(
    fit: np.ndarray,
    fit_mask: np.ndarray,
    *others: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, ...]:
    observed = fit[fit_mask]
    if not len(observed):
        raise RuntimeError("fit population contains no sequence history")
    mean = observed.mean(axis=0)
    std = np.maximum(observed.std(axis=0), 1e-6)

    def transform(sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = sequence.copy()
        result[mask] = (result[mask] - mean) / std
        return result

    return (transform(fit, fit_mask),) + tuple(
        transform(sequence, mask) for sequence, mask in others
    )


def job_root(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


def artifact_complete(job_id: str) -> bool:
    root = job_root(job_id)
    prediction = root / "predictions.parquet"
    done = root / "done.json"
    if not prediction.is_file() or not done.is_file():
        return False
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            return False
        if payload.get("prediction_sha256") != sha256(prediction):
            return False
        rows = len(pd.read_parquet(prediction, columns=["row_id"]))
        return rows == int(payload.get("rows", -1))
    except Exception:
        return False


def unique_row_id(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "CLASSIFICATION":
        if "row_key" not in frame:
            raise RuntimeError("classification row_key missing")
        result = frame["row_key"].to_numpy(dtype=np.uint64)
    else:
        tokens = (
            frame["target_offer_id"].astype("string")
            + "|"
            + frame["source_session_key"].astype("string")
        )
        result = pd.util.hash_pandas_object(tokens, index=False).to_numpy(dtype=np.uint64)
    if pd.Series(result).duplicated().any():
        raise RuntimeError(f"{task}: row identity duplicated")
    return result
