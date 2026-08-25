"""Reconciled point-in-time candidate features for new-build pilots.

This module never fits a model. It defines candidate semantics once so EA04,
EA05 and the later feature builder inspect the same information set.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from skyfare.features.audit_common import (
    BOOKING_WINDOWS,
    ROOT,
    completed_batch_offers,
    load_new_build_raw,
)

PEAK_CONFIG = ROOT / "configs/peak_dates_vietnam_2026.json"


def booking_window_bucket(days: pd.Series) -> pd.Series:
    return pd.cut(
        days.astype(float),
        bins=[-np.inf, 3, 7, 14, 30, 60, np.inf],
        labels=["DUD01_03", "DUD04_07", "DUD08_14", "DUD15_30", "DUD31_60", "DUD61_PLUS"],
        ordered=True,
    )


def peak_dates() -> set[object]:
    config = json.loads(PEAK_CONFIG.read_text(encoding="utf-8"))
    values: set[object] = set()
    for period in config["periods"]:
        values.update(pd.date_range(period["start"], period["end"]).date)
    return values


def _other_airline_context(offers: pd.DataFrame) -> pd.DataFrame:
    """Market context from other airlines only, excluding target airline."""

    key = ["session_key", "route", "flight_date"]
    airline = (
        offers.groupby(key + ["airline"], observed=True)
        .agg(
            airline_min_price=("price_vnd", "min"),
            airline_median_price=("price_vnd", "median"),
            airline_offer_count=("price_vnd", "size"),
        )
        .reset_index()
    )
    left = airline[key + ["airline"]].rename(columns={"airline": "target_airline"})
    right = airline.rename(
        columns={
            "airline": "other_airline",
            "airline_min_price": "other_min_price",
            "airline_median_price": "other_median_price",
            "airline_offer_count": "other_offer_count",
        }
    )
    cross = left.merge(right, on=key, how="inner")
    cross = cross[cross["target_airline"].ne(cross["other_airline"])].copy()
    context = (
        cross.groupby(key + ["target_airline"], observed=True)
        .agg(
            competitor_min_price_other_airlines=("other_min_price", "min"),
            competitor_price_spread_other_airlines=("other_median_price", lambda x: x.max() - x.min()),
            competitor_airline_count=("other_airline", "nunique"),
            competitor_offer_count=("other_offer_count", "sum"),
        )
        .reset_index()
        .rename(columns={"target_airline": "airline"})
    )
    return context


def _causal_market_anchor(offers: pd.DataFrame) -> pd.DataFrame:
    """Latest market level and direction from strictly-prior batches."""

    key = ["collection_era", "route", "airline", "days_until_departure"]
    batch = (
        offers.groupby(key + ["session_key"], observed=True)
        .agg(
            batch_feature_time=("feature_time", "max"),
            batch_market_median_price=("price_vnd", "median"),
            batch_market_support=("price_vnd", "size"),
        )
        .reset_index()
        .sort_values(key + ["batch_feature_time"])
    )
    grouped = batch.groupby(key, observed=True, sort=False)
    batch["temporal_market_median_price"] = grouped["batch_market_median_price"].shift(1)
    batch["temporal_market_support"] = grouped["batch_market_support"].shift(1)
    batch["temporal_market_time"] = grouped["batch_feature_time"].shift(1)
    batch["temporal_market_collection_era"] = grouped["collection_era"].shift(1)
    previous_median = grouped["batch_market_median_price"].shift(2)
    previous_time = grouped["batch_feature_time"].shift(2)
    change_gap_hours = (
        batch["temporal_market_time"] - previous_time
    ).dt.total_seconds() / 3600
    batch["prior_market_change_pct_per_day"] = (
        100
        * (batch["temporal_market_median_price"] / previous_median - 1)
        / (change_gap_hours / 24)
    ).where(change_gap_hours.gt(0) & previous_median.gt(0))
    batch["has_prior_market_change"] = batch[
        "prior_market_change_pct_per_day"
    ].notna().astype("int8")
    return batch[
        key
        + [
            "session_key",
            "temporal_market_median_price",
            "temporal_market_support",
            "temporal_market_time",
            "temporal_market_collection_era",
            "prior_market_change_pct_per_day",
            "has_prior_market_change",
        ]
    ]


def _same_airline_alternatives(offers: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-offer-out same-airline schedule alternatives in current batch."""

    key = [offers["session_key"], offers["route"], offers["flight_date"], offers["airline"]]
    price = offers["price_vnd"].astype(float)
    group_size = price.groupby(key, observed=True).transform("size")
    minimum = price.groupby(key, observed=True).transform("min")
    minimum_count = price.eq(minimum).groupby(key, observed=True).transform("sum")
    next_minimum = price.where(price.gt(minimum)).groupby(key, observed=True).transform("min")
    leave_one_out = minimum.where(~(price.eq(minimum) & minimum_count.eq(1)), next_minimum)
    result = offers[["session_key", "route", "flight_date", "airline", "schedule_slot_id"]].copy()
    result["same_airline_alternative_min_price"] = leave_one_out.where(group_size.gt(1))
    result["same_airline_alternative_count"] = (group_size - 1).clip(lower=0)
    return result


def _same_schedule_history(offers: pd.DataFrame) -> pd.DataFrame:
    """Causal warm features from every earlier completed batch for one slot."""

    out = offers.sort_values(
        ["collection_era", "schedule_slot_id", "feature_time", "session_key"]
    ).copy()
    if out.duplicated(["collection_era", "schedule_slot_id", "session_key"]).any():
        raise RuntimeError("schedule-slot appears more than once in one completed batch")
    key = [out["collection_era"], out["schedule_slot_id"]]
    grouped = out.groupby(
        ["collection_era", "schedule_slot_id"],
        observed=True,
        sort=False,
    )
    count = grouped.cumcount().astype(float)
    price = out["price_vnd"].astype(float)
    dud = out["days_until_departure"].astype(float)

    sum_y = price.groupby(key, observed=True).cumsum() - price
    sum_x = dud.groupby(key, observed=True).cumsum() - dud
    sum_xy = (price * dud).groupby(key, observed=True).cumsum() - price * dud
    sum_x2 = (dud * dud).groupby(key, observed=True).cumsum() - dud * dud
    sum_y2 = (price * price).groupby(key, observed=True).cumsum() - price * price

    out["prior_observation_count"] = count
    out["previous_price_same_schedule"] = grouped["price_vnd"].shift(1)
    out["previous_schedule_time"] = grouped["feature_time"].shift(1)
    out["previous_schedule_session_key"] = grouped["session_key"].shift(1)
    out["previous_schedule_session_label"] = grouped["session_label"].shift(1)
    out["previous_schedule_collection_era"] = grouped["collection_era"].shift(1)
    out["lag_age_hours"] = (
        out["feature_time"] - out["previous_schedule_time"]
    ).dt.total_seconds() / 3600

    valid_mean = count.gt(0)
    mean_y = sum_y / count.where(valid_mean)
    variance = sum_y2 / count.where(valid_mean) - mean_y.pow(2)
    out["prior_price_volatility_vnd"] = np.sqrt(variance.clip(lower=0)).where(count.ge(2))

    denominator = count * sum_x2 - sum_x.pow(2)
    numerator = count * sum_xy - sum_x * sum_y
    out["prior_price_trend_vnd_per_dud_day"] = (numerator / denominator).where(
        count.ge(2) & denominator.abs().gt(1e-9)
    )
    return out


def build_candidate_frame(
    cutoff: pd.Timestamp | str | None = None,
    raw_loader: Callable[[pd.Timestamp], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build one point-in-time offer frame through an explicit snapshot cutoff.

    The default remains the Pilot110 cutoff for backwards reproducibility. Every
    later pilot must pass its cutoff explicitly so newer raw files cannot enter
    silently.
    """

    resolved_cutoff = None if cutoff is None else pd.Timestamp(cutoff).normalize()
    loader = raw_loader or load_new_build_raw
    raw = loader() if resolved_cutoff is None else loader(resolved_cutoff)
    offers = completed_batch_offers(raw)
    del raw
    gc.collect()
    offers = offers[offers["days_until_departure"].isin(BOOKING_WINDOWS)].copy()
    offers = offers[
        [
            "scraped_at",
            "session_id",
            "session_date",
            "session_label",
            "session_key",
            "route",
            "airline",
            "flight_date",
            "departure_time",
            "days_until_departure",
            "price_vnd",
            "schedule_slot_id",
            "is_schedule_fallback",
            "collection_era",
        ]
    ].copy()
    for column in ["route", "airline", "session_label", "collection_era"]:
        offers[column] = offers[column].astype("category")

    batch_times = offers.groupby("session_key", observed=True)["scraped_at"].transform("max")
    offers["feature_time"] = batch_times
    offers["flight_day_of_week"] = offers["flight_date"].dt.dayofweek.astype("int8")
    offers["flight_month"] = offers["flight_date"].dt.month.astype("int8")
    offers["is_peak_period"] = offers["flight_date"].dt.date.isin(peak_dates()).astype("int8")
    offers["departure_minute"] = (
        offers["departure_time"].dt.hour * 60 + offers["departure_time"].dt.minute
    ).astype(float)
    angle = 2 * np.pi * offers["departure_minute"] / (24 * 60)
    offers["departure_time_sin"] = np.sin(angle)
    offers["departure_time_cos"] = np.cos(angle)
    offers["departure_period"] = pd.cut(
        offers["departure_minute"],
        bins=[-1, 299, 719, 1019, 1439],
        labels=["NIGHT", "MORNING", "AFTERNOON", "EVENING"],
    ).astype("string")
    offers["booking_window_bucket"] = booking_window_bucket(offers["days_until_departure"])
    offers["identity_quality"] = np.where(
        offers["is_schedule_fallback"], "SCHEDULE_SLOT_FALLBACK", "RAW_FLIGHT_NO_AVAILABLE"
    )
    offers["batch_offer_count"] = offers.groupby("session_key", observed=True)["price_vnd"].transform("size")

    offers = offers.merge(
        _other_airline_context(offers),
        on=["session_key", "route", "flight_date", "airline"],
        how="left",
        validate="many_to_one",
    )
    offers = offers.merge(
        _same_airline_alternatives(offers),
        on=["session_key", "route", "flight_date", "airline", "schedule_slot_id"],
        how="left",
        validate="one_to_one",
    )
    offers = offers.merge(
        _causal_market_anchor(offers),
        on=["collection_era", "route", "airline", "days_until_departure", "session_key"],
        how="left",
        validate="many_to_one",
    )
    offers["temporal_market_age_hours"] = (
        offers["feature_time"] - offers["temporal_market_time"]
    ).dt.total_seconds() / 3600

    offers = _same_schedule_history(offers)
    offers["has_same_schedule_history"] = offers["previous_price_same_schedule"].notna().astype("int8")
    offers["current_vs_competitor_ratio"] = (
        offers["price_vnd"] / offers["competitor_min_price_other_airlines"]
    )
    offers["previous_vs_temporal_market_ratio"] = (
        offers["previous_price_same_schedule"] / offers["temporal_market_median_price"]
    )

    prior_mask = offers["previous_schedule_time"].notna()
    if not offers.loc[prior_mask, "previous_schedule_time"].lt(offers.loc[prior_mask, "feature_time"]).all():
        raise RuntimeError("same-schedule history contains non-prior timestamps")
    market_mask = offers["temporal_market_time"].notna()
    if not offers.loc[market_mask, "temporal_market_time"].lt(offers.loc[market_mask, "feature_time"]).all():
        raise RuntimeError("temporal market anchor contains non-prior timestamps")
    if not offers.loc[market_mask, "temporal_market_collection_era"].eq(
        offers.loc[market_mask, "collection_era"]
    ).all():
        raise RuntimeError("temporal market anchor crosses collection-era boundary")
    if not offers.loc[prior_mask, "previous_schedule_collection_era"].eq(
        offers.loc[prior_mask, "collection_era"]
    ).all():
        raise RuntimeError("same-schedule history crosses collection-era boundary")
    if offers.loc[prior_mask, "previous_schedule_session_key"].eq(
        offers.loc[prior_mask, "session_key"]
    ).any():
        raise RuntimeError("same-schedule history contains current-batch rows")
    if not offers["has_prior_market_change"].eq(
        offers["prior_market_change_pct_per_day"].notna().astype("int8")
    ).all():
        raise RuntimeError("prior market change mask does not match feature availability")
    return offers.sort_values(["feature_time", "session_key", "schedule_slot_id"]).reset_index(drop=True)


NUMERIC_CANDIDATES = [
    "days_until_departure",
    "flight_day_of_week",
    "flight_month",
    "is_peak_period",
    "departure_minute",
    "departure_time_sin",
    "departure_time_cos",
    "batch_offer_count",
    "competitor_min_price_other_airlines",
    "competitor_price_spread_other_airlines",
    "competitor_airline_count",
    "competitor_offer_count",
    "same_airline_alternative_min_price",
    "same_airline_alternative_count",
    "temporal_market_median_price",
    "temporal_market_support",
    "temporal_market_age_hours",
    "prior_market_change_pct_per_day",
    "has_prior_market_change",
    "price_vnd",
    "current_vs_competitor_ratio",
    "previous_price_same_schedule",
    "prior_price_trend_vnd_per_dud_day",
    "prior_price_volatility_vnd",
    "lag_age_hours",
    "prior_observation_count",
    "previous_vs_temporal_market_ratio",
]
