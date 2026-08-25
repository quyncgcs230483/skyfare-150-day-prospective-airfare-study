"""Point-in-time frame and artifact helpers for Regression."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.models.regression_contract import (
    ANCHOR_AGE_BINS,
    ANCHOR_AGE_LABELS,
    BASELINE_VERSION,
    BOOKING_WINDOWS,
    CATEGORICAL_FEATURES,
    CONTRACT_VERSION,
    CUTOFF,
    FORBIDDEN_PREDICTORS,
    HISTORY_SUPPORT_BINS,
    HISTORY_SUPPORT_LABELS,
    NUMERIC_FEATURES,
    SERVING_ELIGIBLE_CANDIDATE_SOURCES,
    SUPPORT_BINS,
    SUPPORT_LABELS,
    fold_spec,
    market_group_lookup,
)

LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
SHARD = os.environ.get("SKYFARE_FUTURE_DEPARTURE_SHARD", "A").upper()
OUTPUT_ROOT = Path(
    os.environ.get(
        "SKYFARE_FUTURE_DEPARTURE_OUTPUT_ROOT",
        str(LAYOUT.artifacts / "regression" / f"shard_{SHARD.lower()}"),
    )
).resolve()
FRAME_CACHE = OUTPUT_ROOT / "preflight/regression_training_frame.parquet"
LEDGER_CACHE = OUTPUT_ROOT / "preflight/candidate_coverage_ledger.parquet"
STANDARD_OFFERS_CACHE = OUTPUT_ROOT / "preflight/standard_offers.parquet"
RECURRENT_SEQUENCE_SOURCE = STANDARD_OFFERS_CACHE

# Reuse audited immutable loaders and feature construction. New task never reuses
# legacy target filtering, carry baseline, gate, or selection logic.
os.environ.setdefault("SKYFARE_NEXT_SESSION_OUTPUT_ROOT", str(OUTPUT_ROOT))
from skyfare.models import fare_frame_runtime as legacy  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_times(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "feature_time", "label_time", "session_date", "source_session_date",
        "flight_date", "departure_time", "previous_schedule_time",
        "previous_relative_time", "template_previous_time",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric.gt(0)))


def safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0))


def build_standard_offers() -> pd.DataFrame:
    offers = normalize_times(legacy.build_standard_offers())
    if not STANDARD_OFFERS_CACHE.is_file():
        STANDARD_OFFERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = STANDARD_OFFERS_CACHE.with_suffix(".tmp.parquet")
        offers.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(STANDARD_OFFERS_CACHE)
    return offers


def _schedule_template_history(offers: pd.DataFrame) -> pd.DataFrame:
    """Build strict-prior template history at completed-session granularity."""
    batches = (
        offers.groupby(
            ["route", "airline", "departure_minute", "session_key"],
            observed=True,
        )
        .agg(
            template_batch_relative=("current_relative_log", "median"),
            template_batch_rows=("price_vnd", "size"),
            template_batch_time=("feature_time", "max"),
            template_batch_dud=("days_until_departure", "median"),
        )
        .reset_index()
        .sort_values(
            ["route", "airline", "departure_minute", "template_batch_time", "session_key"],
            kind="stable",
        )
    )
    keys = ["route", "airline", "departure_minute"]
    grouped = batches.groupby(keys, observed=True, sort=False)
    batches["template_previous_relative_log"] = grouped["template_batch_relative"].shift(1)
    batches["template_previous_time"] = grouped["template_batch_time"].shift(1)
    batches["template_prior_batches"] = grouped.cumcount().astype("int32")
    cumulative_rows = grouped["template_batch_rows"].cumsum()
    batches["template_history_support_count"] = (
        cumulative_rows - batches["template_batch_rows"]
    ).astype("int32")

    relative = pd.to_numeric(batches["template_batch_relative"], errors="coerce").astype(float)
    dud = pd.to_numeric(batches["template_batch_dud"], errors="coerce").astype(float)
    count = batches["template_prior_batches"].astype(float)
    token = batches[keys].astype("string").agg("|".join, axis=1)
    sum_z = relative.groupby(token, observed=True).cumsum() - relative
    sum_z2 = (relative * relative).groupby(token, observed=True).cumsum() - relative * relative
    sum_x = dud.groupby(token, observed=True).cumsum() - dud
    sum_x2 = (dud * dud).groupby(token, observed=True).cumsum() - dud * dud
    sum_xz = (dud * relative).groupby(token, observed=True).cumsum() - dud * relative
    mean_z = sum_z / count.where(count.gt(0))
    variance = sum_z2 / count.where(count.gt(0)) - mean_z.pow(2)
    batches["template_prior_relative_volatility"] = np.sqrt(
        variance.clip(lower=0)
    ).where(count.ge(2))
    denominator = count * sum_x2 - sum_x.pow(2)
    numerator = count * sum_xz - sum_x * sum_z
    batches["template_prior_relative_trend_per_dud_day"] = (
        numerator / denominator
    ).where(count.ge(2) & denominator.abs().gt(1e-12))
    return batches[
        [
            *keys, "session_key", "template_previous_relative_log",
            "template_previous_time", "template_history_support_count",
            "template_prior_relative_volatility",
            "template_prior_relative_trend_per_dud_day",
        ]
    ]


def _anchor_stats(source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return source.groupby(keys, observed=True)["price_vnd"].agg(
        anchor_value="median", anchor_support="size"
    )


def _apply_anchor_level(
    target: pd.DataFrame,
    stats: pd.DataFrame,
    keys: list[str],
    level: str,
) -> None:
    missing = target["prior_anchor_vnd"].isna()
    if not missing.any() or stats.empty:
        return
    index = (
        pd.MultiIndex.from_frame(target[keys])
        if len(keys) > 1
        else pd.Index(target[keys[0]])
    )
    mapped = stats.reindex(index)
    use = missing & mapped["anchor_value"].notna().to_numpy()
    target.loc[use, "prior_anchor_vnd"] = mapped.loc[use.to_numpy(), "anchor_value"].to_numpy()
    target.loc[use, "prior_anchor_support"] = mapped.loc[use.to_numpy(), "anchor_support"].to_numpy()
    target.loc[use, "anchor_fallback_level"] = level


def _replace_anchor_from_completed_batch(
    ledger: pd.DataFrame,
    offers: pd.DataFrame,
) -> pd.DataFrame:
    """Use only immediately preceding completed batch; no query-batch prices."""
    result = ledger.copy()
    by_session = {
        str(key): part.copy()
        for key, part in offers.groupby("session_key", observed=True, sort=False)
    }
    pieces: list[pd.DataFrame] = []
    hierarchy = (
        (["route", "airline", "departure_period", "days_until_departure"], "ROUTE_AIRLINE_PERIOD_DUD"),
        (["route", "airline", "days_until_departure"], "ROUTE_AIRLINE_DUD"),
        (["route", "days_until_departure"], "ROUTE_DUD"),
        (["airline", "days_until_departure"], "AIRLINE_DUD"),
        (["days_until_departure"], "GLOBAL_DUD"),
    )
    for source_key, part in result.groupby("source_session_key", observed=True, sort=False):
        source = by_session.get(str(source_key))
        if source is None or source.empty:
            continue
        target = part.copy()
        target["prior_anchor_vnd"] = np.nan
        target["prior_anchor_support"] = 0
        target["anchor_fallback_level"] = "UNAVAILABLE"
        for keys, level in hierarchy:
            _apply_anchor_level(target, _anchor_stats(source, keys), keys, level)
        missing = target["prior_anchor_vnd"].isna()
        if missing.any():
            target.loc[missing, "prior_anchor_vnd"] = float(source["price_vnd"].median())
            target.loc[missing, "prior_anchor_support"] = len(source)
            target.loc[missing, "anchor_fallback_level"] = "GLOBAL_BATCH"
        source_time = pd.to_datetime(source["feature_time"], errors="coerce").max()
        target["feature_time"] = source_time
        target["prior_anchor_source"] = "LATEST_STRICTLY_PRIOR_COMPLETED_BATCH"
        target["prior_anchor_log"] = safe_log(target["prior_anchor_vnd"])
        target["prior_anchor_support_log1p"] = safe_log1p(target["prior_anchor_support"])
        target["prior_anchor_age_hours"] = (
            target["label_time"] - source_time
        ).dt.total_seconds() / 3600.0
        target["anchor_collection_era"] = legacy._mode(source["collection_era"])
        target["anchor_is_fallback"] = target["anchor_fallback_level"].ne(
            "ROUTE_AIRLINE_PERIOD_DUD"
        ).astype("int8")
        target["target_anchor_relative_log"] = np.log(
            target["query_session_observed_fare_vnd"] / target["prior_anchor_vnd"]
        )
        target["target_log_ratio"] = target["target_anchor_relative_log"]
        target = legacy._prior_market_context(source, target)
        pieces.append(target)
    if not pieces:
        raise RuntimeError("No completed-batch anchors could be constructed")
    return pd.concat(pieces, ignore_index=True)


def build_candidate_ledger() -> pd.DataFrame:
    offers = build_standard_offers()
    ledger = legacy.build_candidate_ledger().copy()
    ledger = normalize_times(ledger)
    ledger["query_session_observed_fare_vnd"] = ledger["target_session_price_vnd"]
    ledger["query_dud"] = ledger["target_dud"]
    ledger["session_date"] = pd.to_datetime(ledger["target_session_date"], errors="coerce")
    ledger["model_session_label"] = ledger["target_session_label"].astype("string")
    template = _schedule_template_history(offers)
    ledger["session_key"] = ledger["session_key"].astype("string")
    template["session_key"] = template["session_key"].astype("string")
    ledger = ledger.merge(
        template,
        on=["route", "airline", "departure_minute", "session_key"],
        how="left",
        validate="many_to_one",
    )
    hhmm = pd.to_numeric(ledger["departure_minute"], errors="coerce").fillna(-1).astype(int)
    ledger["departure_HHMM"] = hhmm // 60 * 100 + hhmm % 60
    ledger["schedule_template_id"] = (
        ledger["route"].astype("string") + "|"
        + ledger["airline"].astype("string") + "|"
        + ledger["departure_HHMM"].astype("string").str.zfill(4)
    )
    ledger["query_id"] = (
        pd.to_datetime(ledger["session_date"]).dt.strftime("%Y-%m-%d") + "|"
        + ledger["model_session_label"].astype("string") + "|"
        + ledger["route"].astype("string") + "|"
        + pd.to_datetime(ledger["flight_date"]).dt.strftime("%Y-%m-%d")
    )
    # Set feature_time to end of completed source batch before checking whether
    # exact-slot/template candidates were already available.
    ledger = _replace_anchor_from_completed_batch(ledger, offers)
    exact_legal = (
        pd.to_numeric(ledger["prior_observation_count"], errors="coerce").fillna(0).gt(0)
        & pd.to_datetime(ledger["previous_schedule_time"], errors="coerce").le(ledger["feature_time"])
    )
    template_legal = (
        pd.to_numeric(ledger["template_history_support_count"], errors="coerce").fillna(0).gt(0)
        & pd.to_datetime(ledger["template_previous_time"], errors="coerce").le(ledger["feature_time"])
    )
    ledger["candidate_source"] = np.select(
        [exact_legal, template_legal],
        ["PRIOR_SAME_SLOT_CACHE", "PRIOR_SCHEDULE_TEMPLATE_CACHE"],
        default="RETROSPECTIVE_TARGET_ONLY",
    )
    history = pd.to_numeric(ledger["prior_observation_count"], errors="coerce").fillna(0).astype(int)
    template_history = pd.to_numeric(
        ledger["template_history_support_count"], errors="coerce"
    ).fillna(0).astype(int)
    ledger["history_support_count"] = history
    ledger["template_history_support_count"] = template_history
    ledger["is_first_observation"] = history.eq(0).astype("int8")
    ledger["regime"] = np.where(history.ge(3), "WARM", "COLD")
    ledger["history_support_band"] = pd.cut(
        history, bins=HISTORY_SUPPORT_BINS, labels=HISTORY_SUPPORT_LABELS
    ).astype("string")
    ledger["template_history_support_band"] = pd.cut(
        template_history, bins=HISTORY_SUPPORT_BINS, labels=HISTORY_SUPPORT_LABELS
    ).astype("string")
    ledger["has_previous_schedule_template"] = template_legal.astype("int8")
    ledger["has_template_relative_volatility"] = ledger[
        "template_prior_relative_volatility"
    ].notna().astype("int8")
    ledger["has_template_relative_trend"] = ledger[
        "template_prior_relative_trend_per_dud_day"
    ].notna().astype("int8")
    ledger["template_lag_age_hours"] = (
        ledger["label_time"] - ledger["template_previous_time"]
    ).dt.total_seconds() / 3600.0
    ledger["anchor_support_band"] = pd.cut(
        ledger["prior_anchor_support"], bins=SUPPORT_BINS, labels=SUPPORT_LABELS
    ).astype("string")
    ledger["anchor_age_band"] = pd.cut(
        ledger["prior_anchor_age_hours"], bins=ANCHOR_AGE_BINS, labels=ANCHOR_AGE_LABELS
    ).astype("string")
    ledger["route_airline"] = ledger["route"].astype("string") + "|" + ledger["airline"].astype("string")
    ledger["market_group"] = ledger["route"].map(market_group_lookup()).fillna("UNMAPPED")
    ledger["dud_support_mode"] = np.where(
        ledger["query_dud"].isin(BOOKING_WINDOWS), "ON_GRID", "INTERIOR_OFF_GRID_INTERPOLATED"
    )
    ledger["target_batch_exists"] = True
    ledger["target_observation_state"] = "OBSERVED"
    ledger["data_cutoff"] = CUTOFF
    ledger["feature_contract_version"] = CONTRACT_VERSION
    ledger["baseline_version"] = BASELINE_VERSION
    if not ledger["feature_time"].lt(ledger["label_time"]).all():
        raise RuntimeError("Point-in-time failure: completed-batch feature_time is not before query label_time")
    return ledger.sort_values(["label_time", "query_id", "schedule_slot_id"], kind="stable").reset_index(drop=True)


def build_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = build_candidate_ledger()
    frame = ledger[
        ledger["candidate_source"].isin(SERVING_ELIGIBLE_CANDIDATE_SOURCES)
        & ledger["prior_anchor_vnd"].gt(0)
        & ledger["query_session_observed_fare_vnd"].gt(0)
    ].copy()
    required = [
        "offer_id", "target_offer_id", "query_id", "schedule_slot_id",
        "schedule_template_id", "feature_time", "label_time",
        "query_session_observed_fare_vnd", "target_anchor_relative_log",
        *CATEGORICAL_FEATURES, *NUMERIC_FEATURES,
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Future-departure frame missing columns: {missing}")
    forbidden = sorted(set(CATEGORICAL_FEATURES + NUMERIC_FEATURES).intersection(FORBIDDEN_PREDICTORS))
    if forbidden:
        raise RuntimeError(f"Forbidden predictors entered model: {forbidden}")
    if frame["target_offer_id"].duplicated().any():
        raise RuntimeError("Target offer appears more than once")
    return frame.reset_index(drop=True), ledger


def load_training_frame() -> pd.DataFrame:
    if FRAME_CACHE.is_file():
        return normalize_times(pd.read_parquet(FRAME_CACHE))
    frame, _ = build_training_frame()
    return frame


def _coverage_label(count: int) -> str:
    if count >= 5:
        return "UNIVERSAL_5"
    if count == 4:
        return "BROAD_4_5"
    if count >= 2:
        return "MID_2_3"
    if count == 1:
        return "SINGLE_1"
    return "UNOBSERVED_0"


def temporal_split(frame: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = fold_spec(fold)
    start = pd.Timestamp(spec["validation_start"])
    end = pd.Timestamp(spec["validation_end"]) + pd.Timedelta(days=1)
    train = frame[frame["label_time"].lt(start)].copy()
    valid = frame[frame["label_time"].ge(start) & frame["label_time"].lt(end)].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"{fold}: empty temporal train/valid split")
    if not train["label_time"].lt(valid["label_time"].min()).all():
        raise RuntimeError(f"{fold}: label-time purge failure")
    route_airline_counts = train.groupby("route_airline", observed=True).size()
    valid["train_route_airline_support"] = valid["route_airline"].map(route_airline_counts).fillna(0).astype(int)
    valid["route_airline_support_band"] = pd.cut(
        valid["train_route_airline_support"], bins=SUPPORT_BINS, labels=SUPPORT_LABELS
    ).astype("string")
    route_airlines = train.groupby("route", observed=True)["airline"].nunique()
    valid["coverage_band"] = valid["route"].map(route_airlines).fillna(0).astype(int).map(_coverage_label)
    valid["support_tier"] = np.where(
        valid["coverage_band"].isin(["UNIVERSAL_5", "BROAD_4_5"]),
        "HIGH_SUPPORT", "LOWER_SUPPORT",
    )
    route_support = train.groupby("route", observed=True).size()
    ranks = route_support.rank(method="first", pct=True)
    quartiles = pd.cut(
        ranks, [0, .25, .5, .75, 1.0],
        labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"], include_lowest=True,
    )
    valid["route_support_quartile"] = valid["route"].map(quartiles.astype("string")).fillna("UNSEEN")
    valid["fold"] = spec["fold"]
    valid["fold_role"] = spec["role"]
    return train, valid


def target_cell_weights(train: pd.DataFrame) -> np.ndarray:
    support = train.groupby(
        ["query_dud", "model_session_label"], observed=True
    )["query_dud"].transform("size")
    weights = 1.0 / support.to_numpy(dtype=float)
    return weights / max(float(np.mean(weights)), 1e-12)


def residual_to_price(
    anchor: np.ndarray,
    predicted_log_residual: np.ndarray,
    training_residual: np.ndarray,
) -> np.ndarray:
    lower, upper = np.quantile(training_residual, [0.005, 0.995])
    clipped = np.clip(np.asarray(predicted_log_residual, dtype=float), lower, upper)
    return np.asarray(anchor, dtype=float) * np.exp(clipped)


def stable_row_key(frame: pd.DataFrame) -> pd.Series:
    tokens = frame[["target_offer_id", "source_session_key"]].astype("string").fillna("__NA__").agg("|".join, axis=1)
    return pd.util.hash_pandas_object(tokens, index=False).astype("uint64")


def prediction_path(model: str, fold: str, sequence_length: int | None = None) -> Path:
    suffix = model if sequence_length is None else f"{model}_L{sequence_length}"
    return OUTPUT_ROOT / "base_predictions" / suffix / fold / "predictions.parquet"


def done_path(model: str, fold: str, sequence_length: int | None = None) -> Path:
    suffix = model if sequence_length is None else f"{model}_L{sequence_length}"
    return OUTPUT_ROOT / "base_predictions" / suffix / fold / "done.json"
