"""Point-in-time data, baseline, metric, and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from skyfare.core.paths import DataLayout
from skyfare.models.fare_frame_contract import (
    ANCHOR_AGE_BINS,
    ANCHOR_AGE_LABELS,
    BASELINE_VERSION,
    BOOKING_WINDOWS,
    CATEGORICAL_FEATURES,
    CONTRACT_VERSION,
    CUTOFF,
    FOLDS,
    FORBIDDEN_PREDICTORS,
    HIERARCHY_MIN_SUPPORT,
    HISTORY_SUPPORT_BINS,
    HISTORY_SUPPORT_LABELS,
    NUMERIC_FEATURES,
    SESSION_GAP_BINS,
    SESSION_GAP_LABELS,
    SUPPORT_BINS,
    SUPPORT_LABELS,
    market_group_lookup,
)

LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
OUTPUT_ROOT = Path(
    os.environ.get(
        "SKYFARE_NEXT_SESSION_OUTPUT_ROOT",
        str(LAYOUT.artifacts / "fare_frame"),
    )
).resolve()
FRAME_CACHE = OUTPUT_ROOT / "preflight/model_training_frame.parquet"
LEDGER_CACHE = OUTPUT_ROOT / "preflight/candidate_coverage_ledger.parquet"
STANDARD_OFFERS_CACHE = OUTPUT_ROOT / "preflight/standard_offers.parquet"
RECURRENT_SEQUENCE_SOURCE = Path(
    os.environ.get(
        "SKYFARE_NEXT_SESSION_SEQUENCE_PATH",
        str(STANDARD_OFFERS_CACHE),
    )
).resolve()

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
        "scraped_at",
        "feature_time",
        "label_time",
        "session_date",
        "target_session_date",
        "flight_date",
        "departure_time",
        "previous_schedule_time",
        "previous_relative_time",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric.gt(0)))


def safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0))


def _standard_raw_loader(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Load standard masters and rebuild session keys independent of datetime resolution."""
    from skyfare.features.audit_common import load_standard_only_raw
    from skyfare.preparation.temporal_sessions import (
        collection_session_labels,
        operational_session_dates,
    )

    data = load_standard_only_raw(cutoff)
    session_id = pd.to_datetime(data["session_id"], errors="coerce")
    data["session_date"] = pd.to_datetime(operational_session_dates(session_id))
    data["session_label"] = collection_session_labels(session_id).astype("string")
    day_number = data["session_date"].map(pd.Timestamp.toordinal).astype("int64")
    session_band = data["session_label"].eq("PM").astype("int8")
    data["session_key"] = day_number * 2 + session_band
    return data


def _attach_cross_era_market_history(offers: pd.DataFrame) -> pd.DataFrame:
    """Strict-prior route-airline-DUD market history across compatible standard eras."""
    result = offers.copy()
    keys = ["route", "airline", "days_until_departure"]
    batches = (
        result.groupby([*keys, "session_key"], observed=True)
        .agg(
            market_price=("price_vnd", "median"),
            market_support=("price_vnd", "size"),
            market_time=("feature_time", "max"),
            market_collection_era=("collection_era", _mode),
        )
        .reset_index()
        .sort_values([*keys, "market_time", "session_key"], kind="stable")
    )
    grouped = batches.groupby(keys, observed=True, sort=False)
    batches["temporal_market_median_price"] = grouped["market_price"].shift(1)
    batches["temporal_market_support"] = grouped["market_support"].shift(1)
    batches["temporal_market_time"] = grouped["market_time"].shift(1)
    batches["temporal_market_collection_era"] = grouped[
        "market_collection_era"
    ].shift(1)
    batches["prior2_market_price"] = grouped["market_price"].shift(2)
    batches["prior2_market_time"] = grouped["market_time"].shift(2)
    elapsed_days = (
        batches["temporal_market_time"] - batches["prior2_market_time"]
    ).dt.total_seconds() / 86_400.0
    batches["prior_market_change_pct_per_day"] = (
        100.0
        * (
            batches["temporal_market_median_price"]
            / batches["prior2_market_price"]
            - 1.0
        )
        / elapsed_days.where(elapsed_days.gt(0))
    )
    batches["has_prior_market_change"] = batches[
        "prior_market_change_pct_per_day"
    ].notna().astype("int8")
    columns = [
        *keys,
        "session_key",
        "temporal_market_median_price",
        "temporal_market_support",
        "temporal_market_time",
        "temporal_market_collection_era",
        "prior_market_change_pct_per_day",
        "has_prior_market_change",
    ]
    result = result.drop(
        columns=[column for column in columns[4:] if column in result],
        errors="ignore",
    ).merge(
        batches[columns],
        on=[*keys, "session_key"],
        how="left",
        validate="many_to_one",
    )
    result["temporal_market_age_hours"] = (
        result["feature_time"] - result["temporal_market_time"]
    ).dt.total_seconds() / 3600.0
    return result


def _attach_cross_era_schedule_history(offers: pd.DataFrame) -> pd.DataFrame:
    """Strict-prior same-slot price and relative history without era reset."""
    result = offers.sort_values(
        ["schedule_slot_id", "feature_time", "session_key"], kind="stable"
    ).copy()
    grouped = result.groupby("schedule_slot_id", observed=True, sort=False)
    result["prior_observation_count"] = grouped.cumcount().astype("int32")
    result["previous_price_same_schedule"] = grouped["price_vnd"].shift(1)
    result["previous_schedule_time"] = grouped["feature_time"].shift(1)
    result["previous_schedule_session_key"] = grouped["session_key"].shift(1)
    result["previous_schedule_session_label"] = grouped["session_label"].shift(1)
    result["previous_schedule_collection_era"] = grouped["collection_era"].shift(1)
    result["lag_age_hours"] = (
        result["feature_time"] - result["previous_schedule_time"]
    ).dt.total_seconds() / 3600.0
    result["has_same_schedule_history"] = result[
        "previous_price_same_schedule"
    ].notna().astype("int8")

    # Normalize nullable pandas numeric arrays before mixed arithmetic. Some
    # pandas versions fail when multiplying nullable Int64 and Float64 arrays.
    price = pd.to_numeric(result["price_vnd"], errors="coerce").astype("float64")
    dud = pd.to_numeric(
        result["days_until_departure"], errors="coerce"
    ).astype("float64")
    keys = result["schedule_slot_id"]
    count = result["prior_observation_count"].astype(float)
    prior_sum_price = price.groupby(keys, observed=True).cumsum() - price
    prior_sum_price2 = (price * price).groupby(keys, observed=True).cumsum() - price * price
    mean_price = prior_sum_price / count.where(count.gt(0))
    variance = prior_sum_price2 / count.where(count.gt(0)) - mean_price.pow(2)
    result["prior_price_volatility_vnd"] = np.sqrt(
        variance.clip(lower=0)
    ).where(count.ge(2))
    prior_sum_dud = dud.groupby(keys, observed=True).cumsum() - dud
    prior_sum_dud2 = (dud * dud).groupby(keys, observed=True).cumsum() - dud * dud
    prior_sum_dp = (dud * price).groupby(keys, observed=True).cumsum() - dud * price
    denominator = count * prior_sum_dud2 - prior_sum_dud.pow(2)
    numerator = count * prior_sum_dp - prior_sum_dud * prior_sum_price
    result["prior_price_trend_vnd_per_dud_day"] = (
        numerator / denominator
    ).where(count.ge(2) & denominator.abs().gt(1e-12))

    # Relative history uses peer-relative values computed within each completed batch.
    relative = pd.to_numeric(
        result["current_relative_log"], errors="coerce"
    ).astype("float64")
    valid = relative.notna().astype(float)
    value = relative.fillna(0.0)
    relative_count = valid.groupby(keys, observed=True).cumsum() - valid
    sum_z = value.groupby(keys, observed=True).cumsum() - value
    sum_z2 = (value * value).groupby(keys, observed=True).cumsum() - value * value
    weighted_dud = dud * valid
    sum_x = weighted_dud.groupby(keys, observed=True).cumsum() - weighted_dud
    sum_x2 = (dud * dud * valid).groupby(keys, observed=True).cumsum() - dud * dud * valid
    sum_xz = (dud * value).groupby(keys, observed=True).cumsum() - dud * value
    result["prior_relative_count"] = relative_count.astype("int32")
    result["previous_relative_log"] = relative.groupby(keys, observed=True).shift(1)
    result["previous_anchor_vnd"] = result.groupby(
        "schedule_slot_id", observed=True, sort=False
    )["anchor_vnd"].shift(1)
    result["previous_relative_time"] = result.groupby(
        "schedule_slot_id", observed=True, sort=False
    )["feature_time"].shift(1)
    result["previous_relative_session_key"] = result.groupby(
        "schedule_slot_id", observed=True, sort=False
    )["session_key"].shift(1)
    result["relative_lag_age_hours"] = (
        result["feature_time"] - result["previous_relative_time"]
    ).dt.total_seconds() / 3600.0
    result["market_shift_log"] = safe_log(result["anchor_vnd"]) - safe_log(
        result["previous_anchor_vnd"]
    )
    mean_z = sum_z / relative_count.where(relative_count.gt(0))
    relative_variance = sum_z2 / relative_count.where(relative_count.gt(0)) - mean_z.pow(2)
    result["prior_relative_volatility"] = np.sqrt(
        relative_variance.clip(lower=0)
    ).where(relative_count.ge(2))
    rel_denominator = relative_count * sum_x2 - sum_x.pow(2)
    rel_numerator = relative_count * sum_xz - sum_x * sum_z
    result["prior_relative_trend_per_dud_day"] = (
        rel_numerator / rel_denominator
    ).where(relative_count.ge(2) & rel_denominator.abs().gt(1e-12))
    result["relative_history_eligible"] = relative_count.ge(3).astype("int8")

    prior = result["previous_schedule_time"].notna()
    if not result.loc[prior, "previous_schedule_time"].lt(
        result.loc[prior, "feature_time"]
    ).all():
        raise RuntimeError("Cross-era same-slot history is not strictly prior")
    return result.sort_values(
        ["feature_time", "session_key", "schedule_slot_id"], kind="stable"
    ).reset_index(drop=True)


def build_standard_offers() -> pd.DataFrame:
    """Build all compatible standard observations through cutoff; no model fitting."""
    if STANDARD_OFFERS_CACHE.is_file():
        return normalize_times(pd.read_parquet(STANDARD_OFFERS_CACHE))

    from skyfare.features.audit_common import load_standard_only_raw
    from skyfare.features.candidate_feature_contract import build_candidate_frame
    from skyfare.features.peer_anchor import attach_peer_anchor

    # Import proves immutable standard loader exists; wrapper fixes session-key
    # construction for both pandas nanosecond and microsecond datetime storage.
    del load_standard_only_raw
    offers = build_candidate_frame(CUTOFF, raw_loader=_standard_raw_loader)
    offers["offer_id"] = pd.util.hash_pandas_object(
        offers[["collection_era", "session_key", "schedule_slot_id"]],
        index=False,
    ).astype("uint64")
    if offers["offer_id"].duplicated().any():
        raise RuntimeError("Standard offer_id is not unique")
    offers = _attach_cross_era_market_history(offers)
    offers = attach_peer_anchor(offers)
    offers = _attach_cross_era_schedule_history(offers)
    return normalize_times(offers)


def _mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "UNKNOWN"
    modes = values.mode()
    return str(modes.iloc[0]) if len(modes) == 1 else "MIXED"


def _fill_group_stat(
    target: pd.DataFrame,
    source: pd.DataFrame,
    keys: list[str],
    missing: pd.Series,
    level: str,
) -> None:
    stats = source.groupby(keys, observed=True)["price_vnd"].agg(
        anchor_value="median", anchor_support="size"
    )
    if stats.empty:
        return
    index = pd.MultiIndex.from_frame(target[keys]) if len(keys) > 1 else pd.Index(target[keys[0]])
    mapped = stats.reindex(index)
    use = missing & mapped["anchor_value"].notna().to_numpy()
    target.loc[use, "prior_anchor_vnd"] = mapped.loc[use.to_numpy(), "anchor_value"].to_numpy()
    target.loc[use, "prior_anchor_support"] = mapped.loc[use.to_numpy(), "anchor_support"].to_numpy()
    target.loc[use, "anchor_fallback_level"] = level


def _prior_market_context(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    keys = ["route", "days_until_departure"]
    stats = source.groupby(keys, observed=True).agg(
        prior_route_min_price=("price_vnd", "min"),
        prior_route_max_price=("price_vnd", "max"),
        prior_competitor_airline_count=("airline", "nunique"),
        prior_competitor_offer_count=("price_vnd", "size"),
    )
    index = pd.MultiIndex.from_frame(target[keys])
    mapped = stats.reindex(index).reset_index(drop=True)
    for column in mapped:
        target[column] = mapped[column].to_numpy()
    target["prior_route_min_log_price"] = safe_log(target["prior_route_min_price"])
    spread = target["prior_route_max_price"] - target["prior_route_min_price"]
    target["prior_route_price_spread_log1p"] = safe_log1p(spread)
    return target


def _session_pairs(offers: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        offers.groupby("session_key", observed=True)
        .agg(
            target_session_time=("feature_time", "min"),
            target_session_date=("session_date", "first"),
            target_session_label=("session_label", "first"),
            target_collection_era=("collection_era", _mode),
        )
        .sort_values("target_session_time", kind="stable")
        .reset_index()
    )
    sessions["source_session_key"] = sessions["session_key"].shift(1)
    sessions["feature_time"] = sessions["target_session_time"].shift(1)
    sessions["source_session_date"] = sessions["target_session_date"].shift(1)
    sessions["source_session_label"] = sessions["target_session_label"].shift(1)
    sessions["source_collection_era"] = sessions["target_collection_era"].shift(1)
    return sessions.dropna(subset=["source_session_key", "feature_time"]).copy()


def build_candidate_ledger() -> pd.DataFrame:
    """Build observed target rows using only latest strictly-prior schedule cache."""
    offers = build_standard_offers()
    offers["route"] = offers["route"].astype("string").str.upper().str.replace("→", "-", regex=False)
    offers["airline"] = offers["airline"].astype("string").str.upper()
    offers["session_key"] = offers["session_key"].astype("string")
    pairs = _session_pairs(offers)
    by_session = {str(key): part.copy() for key, part in offers.groupby("session_key", observed=True, sort=False)}
    parts: list[pd.DataFrame] = []

    for pair in pairs.itertuples(index=False):
        source = by_session[str(pair.source_session_key)]
        target = by_session[str(pair.session_key)].copy()
        target["target_session_key"] = str(pair.session_key)
        target["source_session_key"] = str(pair.source_session_key)
        target["target_session_date"] = pd.Timestamp(pair.target_session_date)
        target["target_session_label"] = str(pair.target_session_label)
        target["source_session_date"] = pd.Timestamp(pair.source_session_date)
        target["source_session_label"] = str(pair.source_session_label)
        target["source_collection_era"] = str(pair.source_collection_era)
        target["label_time"] = pd.Timestamp(pair.target_session_time)
        target["feature_time"] = pd.Timestamp(pair.feature_time)
        target["session_gap_hours"] = (
            target["label_time"] - target["feature_time"]
        ).dt.total_seconds() / 3600.0
        target["session_gap_band"] = pd.cut(
            target["session_gap_hours"],
            bins=SESSION_GAP_BINS,
            labels=SESSION_GAP_LABELS,
        ).astype("string")
        target["session_transition"] = (
            target["source_session_label"] + "->" + target["target_session_label"]
        )
        target["source_target_era_transition"] = (
            target["source_collection_era"]
            + "->"
            + target["collection_era"].astype("string")
        )
        target["target_offer_id"] = target["offer_id"]
        target["target_session_price_vnd"] = pd.to_numeric(target["price_vnd"], errors="coerce")
        target["target_dud"] = pd.to_numeric(target["days_until_departure"], errors="coerce")

        history_count = pd.to_numeric(target["prior_observation_count"], errors="coerce").fillna(0).astype(int)
        previous_time = pd.to_datetime(target["previous_schedule_time"], errors="coerce")
        legal_cache = history_count.gt(0) & previous_time.notna() & previous_time.le(target["feature_time"])
        target["candidate_source"] = np.where(
            legal_cache, "PRIOR_SAME_SLOT_CACHE", "RETROSPECTIVE_TARGET_ONLY"
        )
        target["carry_available"] = legal_cache.astype("int8")
        target["history_support_count"] = history_count
        target["is_first_observation"] = history_count.eq(0).astype("int8")
        target["regime"] = np.where(history_count.ge(3), "WARM", "COLD")
        target["history_support_band"] = pd.cut(
            history_count,
            bins=HISTORY_SUPPORT_BINS,
            labels=HISTORY_SUPPORT_LABELS,
        ).astype("string")

        target["prior_anchor_vnd"] = np.nan
        target["prior_anchor_support"] = 0
        target["anchor_fallback_level"] = "UNAVAILABLE"
        hierarchy = (
            (["route", "airline", "days_until_departure"], "ROUTE_AIRLINE_DUD"),
            (["route", "days_until_departure"], "ROUTE_DUD"),
            (["airline", "days_until_departure"], "AIRLINE_DUD"),
            (["days_until_departure"], "GLOBAL_DUD"),
        )
        for keys, level in hierarchy:
            _fill_group_stat(
                target,
                source,
                keys,
                target["prior_anchor_vnd"].isna(),
                level,
            )
        if target["prior_anchor_vnd"].isna().any():
            fallback = float(pd.to_numeric(source["price_vnd"], errors="coerce").median())
            missing = target["prior_anchor_vnd"].isna()
            target.loc[missing, "prior_anchor_vnd"] = fallback
            target.loc[missing, "prior_anchor_support"] = len(source)
            target.loc[missing, "anchor_fallback_level"] = "GLOBAL"

        target["prior_anchor_source"] = "STRICTLY_PRIOR_COMPLETED_BATCH"
        target["prior_anchor_log"] = safe_log(target["prior_anchor_vnd"])
        target["prior_anchor_support_log1p"] = safe_log1p(target["prior_anchor_support"])
        target["prior_anchor_age_hours"] = (
            target["label_time"] - target["feature_time"]
        ).dt.total_seconds() / 3600.0
        target["anchor_support_band"] = pd.cut(
            target["prior_anchor_support"], bins=SUPPORT_BINS, labels=SUPPORT_LABELS
        ).astype("string")
        target["anchor_age_band"] = pd.cut(
            target["prior_anchor_age_hours"], bins=ANCHOR_AGE_BINS, labels=ANCHOR_AGE_LABELS
        ).astype("string")
        target["anchor_collection_era"] = _mode(source["collection_era"])
        target["anchor_is_fallback"] = target["anchor_fallback_level"].ne("ROUTE_AIRLINE_DUD").astype("int8")
        target["target_anchor_relative_log"] = np.log(
            target["target_session_price_vnd"] / target["prior_anchor_vnd"]
        )
        target["target_log_ratio"] = target["target_anchor_relative_log"]
        target["target_batch_exists"] = True
        target["target_observation_state"] = "OBSERVED"
        target["data_cutoff"] = CUTOFF
        target["market_group"] = target["route"].map(market_group_lookup()).fillna("UNMAPPED")
        target["route_airline"] = target["route"] + "|" + target["airline"]
        target["dud_support_mode"] = np.where(
            target["target_dud"].isin(BOOKING_WINDOWS), "ON_GRID", "INTERIOR_OFF_GRID_INTERPOLATED"
        )
        target["departure_HHMM"] = (
            pd.to_numeric(target["departure_minute"], errors="coerce").fillna(-1).astype(int) // 60 * 100
            + pd.to_numeric(target["departure_minute"], errors="coerce").fillna(-1).astype(int) % 60
        )
        target = _prior_market_context(source, target)

        target["has_previous_same_schedule"] = legal_cache.astype("int8")
        target["has_prior_relative_volatility"] = target["prior_relative_volatility"].notna().astype("int8")
        target["has_prior_relative_trend"] = target["prior_relative_trend_per_dud_day"].notna().astype("int8")
        target["relative_history_eligible"] = history_count.ge(3).astype("int8")
        target["feature_contract_version"] = CONTRACT_VERSION
        target["baseline_version"] = BASELINE_VERSION
        movement = (
            target["target_session_price_vnd"]
            / pd.to_numeric(target["previous_price_same_schedule"], errors="coerce")
            - 1.0
        )
        movement_numeric = pd.to_numeric(movement, errors="coerce").astype(float)
        target["actual_movement_band"] = np.select(
            [
                movement_numeric.le(-0.10).fillna(False).to_numpy(dtype=bool),
                movement_numeric.le(-0.05).fillna(False).to_numpy(dtype=bool),
                movement_numeric.abs().le(0.01).fillna(False).to_numpy(dtype=bool),
                movement_numeric.lt(0.05).fillna(False).to_numpy(dtype=bool),
                movement_numeric.lt(0.10).fillna(False).to_numpy(dtype=bool),
            ],
            [
                "DROP_GE_10PCT",
                "DROP_5_10PCT",
                "STABLE_LE_1PCT",
                "MILD_WITHIN_5PCT",
                "RISE_5_10PCT",
            ],
            default="RISE_GE_10PCT",
        )
        target.loc[~legal_cache, "actual_movement_band"] = "NO_LEGAL_CARRY"
        parts.append(target)

    ledger = pd.concat(parts, ignore_index=True)
    ledger = normalize_times(ledger)
    if not ledger["feature_time"].lt(ledger["label_time"]).all():
        raise RuntimeError("Point-in-time failure: feature_time is not before label_time")
    return ledger.sort_values(["label_time", "target_offer_id"], kind="stable").reset_index(drop=True)


def build_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = build_candidate_ledger()
    frame = ledger[
        ledger["candidate_source"].eq("PRIOR_SAME_SLOT_CACHE")
        & ledger["prior_anchor_vnd"].gt(0)
        & ledger["target_session_price_vnd"].gt(0)
    ].copy()
    required = [
        "target_offer_id",
        "schedule_slot_id",
        "feature_time",
        "label_time",
        "target_session_price_vnd",
        "target_anchor_relative_log",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Next-session frame missing columns: {missing}")
    if set(CATEGORICAL_FEATURES + NUMERIC_FEATURES).intersection(FORBIDDEN_PREDICTORS):
        raise RuntimeError("Forbidden field declared as predictor")
    frame = frame.dropna(
        subset=["feature_time", "label_time", "target_session_price_vnd", "prior_anchor_vnd", "schedule_slot_id"]
    ).copy()
    if frame.duplicated("target_offer_id").any():
        raise RuntimeError("Target offer appears more than once")
    return frame.reset_index(drop=True), ledger


def load_training_frame() -> pd.DataFrame:
    if FRAME_CACHE.is_file():
        return normalize_times(pd.read_parquet(FRAME_CACHE))
    frame, _ = build_training_frame()
    return frame


def fold_spec(name: str) -> dict[str, str]:
    selected = [item for item in FOLDS if item["fold"] == name]
    if len(selected) != 1:
        raise KeyError(name)
    return selected[0]


def temporal_split(frame: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = fold_spec(fold)
    start = pd.Timestamp(spec["validation_start"])
    end = pd.Timestamp(spec["validation_end"]) + pd.Timedelta(days=1)
    train = frame[frame["label_time"].lt(start)].copy()
    valid = frame[frame["label_time"].ge(start) & frame["label_time"].lt(end)].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"{fold}: empty temporal train/valid split")
    if train["label_time"].max() >= start:
        raise RuntimeError(f"{fold}: label-time purge failure")
    return train, attach_fold_tags(train, valid, spec)


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


def attach_fold_tags(train: pd.DataFrame, valid: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    result = valid.copy()
    route_airline_counts = train.groupby("route_airline", observed=True).size()
    result["train_route_airline_support"] = result["route_airline"].map(route_airline_counts).fillna(0).astype(int)
    result["route_airline_support_band"] = pd.cut(
        result["train_route_airline_support"], bins=SUPPORT_BINS, labels=SUPPORT_LABELS
    ).astype("string")
    route_airlines = train.groupby("route", observed=True)["airline"].nunique()
    result["coverage_band"] = result["route"].map(route_airlines).fillna(0).astype(int).map(_coverage_label)
    result["support_tier"] = np.where(
        result["coverage_band"].isin(["UNIVERSAL_5", "BROAD_4_5"]), "HIGH_SUPPORT", "LOWER_SUPPORT"
    )
    route_support = train.groupby("route", observed=True).size()
    ranks = route_support.rank(method="first", pct=True)
    quartiles = pd.cut(ranks, [0, .25, .5, .75, 1.0], labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"], include_lowest=True)
    result["route_support_quartile"] = result["route"].map(quartiles.astype("string")).fillna("UNSEEN")
    result["fold"] = spec["fold"]
    result["fold_role"] = spec["role"]
    return result


def target_cell_weights(train: pd.DataFrame) -> np.ndarray:
    columns = ["target_dud", "target_session_label"]
    support = train.groupby(columns, observed=True)["target_dud"].transform("size")
    weights = 1.0 / support.to_numpy(dtype=float)
    return weights / max(float(np.mean(weights)), 1e-12)


def _hierarchy_lookup(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    keys: list[str],
    unresolved: np.ndarray,
    values: np.ndarray,
    sources: np.ndarray,
    supports: np.ndarray,
    source_name: str,
) -> None:
    stats = train.groupby(keys, observed=True)["target_anchor_relative_log"].agg(["median", "count"])
    index = pd.MultiIndex.from_frame(valid[keys]) if len(keys) > 1 else pd.Index(valid[keys[0]])
    mapped = stats.reindex(index)
    use = unresolved & mapped["count"].fillna(0).to_numpy().astype(int).__ge__(HIERARCHY_MIN_SUPPORT)
    values[use] = mapped.loc[use, "median"].to_numpy(dtype=float)
    supports[use] = mapped.loc[use, "count"].to_numpy(dtype=int)
    sources[use] = source_name
    unresolved[use] = False


def hierarchical_anchor_residual(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros(len(valid), dtype=float)
    sources = np.full(len(valid), "ZERO_RESIDUAL_ANCHOR", dtype=object)
    supports = np.zeros(len(valid), dtype=int)
    unresolved = np.ones(len(valid), dtype=bool)
    levels = (
        (["route", "airline", "departure_minute", "target_dud", "target_session_label"], "ROUTE_AIRLINE_HHMM_DUD_SESSION"),
        (["route", "airline", "departure_period", "target_dud", "target_session_label"], "ROUTE_AIRLINE_PERIOD_DUD_SESSION"),
        (["route", "airline", "target_dud", "target_session_label"], "ROUTE_AIRLINE_DUD_SESSION"),
        (["route", "target_dud", "target_session_label"], "ROUTE_DUD_SESSION"),
        (["target_dud", "target_session_label"], "DUD_SESSION"),
    )
    for keys, name in levels:
        _hierarchy_lookup(train, valid, keys, unresolved, values, sources, supports, name)
    return values, sources, supports


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


def slice_frames(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "OVERALL", "ALL", frame
    dimensions = (
        ("REGIME", "regime"),
        ("HISTORY_SUPPORT", "history_support_band"),
        ("COLLECTION_ERA", "collection_era"),
        ("ERA_TRANSITION", "source_target_era_transition"),
        ("SESSION_TRANSITION", "session_transition"),
        ("SESSION_GAP", "session_gap_band"),
        ("ACTUAL_MOVEMENT", "actual_movement_band"),
        ("MARKET_GROUP", "market_group"),
        ("COVERAGE_BAND", "coverage_band"),
        ("SUPPORT_TIER", "support_tier"),
        ("ROUTE_AIRLINE_SUPPORT", "route_airline_support_band"),
        ("ANCHOR_SOURCE", "anchor_fallback_level"),
        ("ANCHOR_SUPPORT", "anchor_support_band"),
        ("ANCHOR_AGE", "anchor_age_band"),
        ("DUD_SUPPORT", "dud_support_mode"),
        ("AIRLINE", "airline"),
        ("TARGET_DUD", "target_dud"),
        ("TARGET_SESSION", "target_session_label"),
        ("ROUTE_AIRLINE", "route_airline"),
    )
    for name, column in dimensions:
        if column in frame:
            for value, part in frame.groupby(column, observed=True, dropna=False):
                yield name, str(value), part


def metric_record(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    system: str,
    fold: str,
    slice_type: str = "OVERALL",
    slice_value: str = "ALL",
) -> dict[str, Any]:
    actual = frame["target_session_price_vnd"].to_numpy(dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    absolute = np.abs(actual - predicted)
    return {
        "system": system,
        "fold": fold,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "support": int(len(frame)),
        "schedule_slots": int(frame["schedule_slot_id"].nunique()),
        "mae_vnd": float(mean_absolute_error(actual, predicted)),
        "mape": float(np.mean(absolute / np.maximum(actual, 1.0))),
        "wmape": float(absolute.sum() / max(float(np.abs(actual).sum()), 1.0)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
    }
