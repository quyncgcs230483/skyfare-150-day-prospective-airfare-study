"""Point-in-time label construction, temporal splits, weights, and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.models.classification_contract import (
    BASELINE_VERSION,
    BRIDGE_POLICIES,
    CUTOFF,
    NEXT_DUD,
    TARGET_NAME,
    fold_spec,
)


LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
OUTPUT_ROOT = Path(
    os.environ.get(
        "SKYFARE_EXACT_DROP5_OUTPUT_ROOT",
        LAYOUT.artifacts / "classification",
    )
).resolve()
PREFLIGHT_ROOT = OUTPUT_ROOT / "preflight"
OFFERS_CACHE = PREFLIGHT_ROOT / "standard_offers.parquet"
FRAME_CACHE = PREFLIGHT_ROOT / "classification_full_era.parquet"

from skyfare.models.classification_event_runtime import event_frame, hierarchy_probability
from skyfare.models.fare_frame_runtime import build_standard_offers


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


def stable_row_key(frame: pd.DataFrame) -> pd.Series:
    tokens = frame[["offer_id", "target_offer_id"]].astype("string").agg("|".join, axis=1)
    return pd.util.hash_pandas_object(tokens, index=False).astype("uint64")


def _band(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce").fillna(0),
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype("string")


def _market_group(route: pd.Series) -> pd.Series:
    trunk = {"SGN-HAN", "HAN-SGN", "SGN-DAD", "DAD-SGN", "HAN-DAD", "DAD-HAN"}
    tourism = {
        "SGN-PQC", "PQC-SGN", "HAN-PQC", "PQC-HAN", "DAD-PQC", "PQC-DAD",
        "SGN-CXR", "CXR-SGN", "HAN-CXR", "CXR-HAN",
    }
    return np.select(
        [route.astype(str).isin(trunk), route.astype(str).isin(tourism)],
        ["TRUNK", "TOURISM"],
        default="REGIONAL_ALTERNATIVE",
    )


def build_exact_frame(offers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pair same proxy slot/session at canonical next DUD across legal eras."""

    source = build_standard_offers() if offers is None else offers.copy()
    datetime_columns = [
        "feature_time", "session_date", "flight_date", "departure_time",
        "historical_anchor_time", "temporal_market_time",
    ]
    for column in datetime_columns:
        if column in source:
            source[column] = pd.to_datetime(source[column], errors="coerce")

    target_columns = [
        "offer_id", "schedule_slot_id", "session_label", "days_until_departure",
        "price_vnd", "feature_time", "session_date", "session_key", "collection_era",
        "route", "airline", "flight_date", "departure_minute",
    ]
    target = source[target_columns].rename(
        columns={
            "offer_id": "target_offer_id",
            "days_until_departure": "target_dud",
            "price_vnd": "target_price_vnd",
            "feature_time": "label_time",
            "session_date": "target_session_date",
            "session_key": "target_session_key",
            "collection_era": "target_collection_era",
            "route": "target_route",
            "airline": "target_airline",
            "flight_date": "target_flight_date",
            "departure_minute": "target_departure_minute",
        }
    )
    key = ["schedule_slot_id", "session_label", "target_dud"]
    if target.duplicated(key).any():
        raise RuntimeError("Target lookup is not unique at slot/session/DUD")

    current = source[source["days_until_departure"].isin(NEXT_DUD)].copy()
    current["current_dud"] = current["days_until_departure"].astype(int)
    current["target_dud"] = current["current_dud"].map(NEXT_DUD).astype(int)
    pairs = current.merge(target, on=key, how="inner", validate="one_to_one")
    expected_target_date = (
        pairs["flight_date"].dt.normalize()
        - pd.to_timedelta(pairs["target_dud"], unit="D")
    )
    legal = (
        pairs["label_time"].gt(pairs["feature_time"])
        & pairs["target_session_date"].eq(expected_target_date)
        & pairs["route"].astype(str).eq(pairs["target_route"].astype(str))
        & pairs["airline"].astype(str).eq(pairs["target_airline"].astype(str))
        & pairs["flight_date"].eq(pairs["target_flight_date"])
        & pd.to_numeric(pairs["departure_minute"], errors="coerce").eq(
            pd.to_numeric(pairs["target_departure_minute"], errors="coerce")
        )
    )
    pairs = pairs.loc[legal].copy()
    if pairs.empty:
        raise RuntimeError("No legal exact-next labels")

    pairs["transition"] = (
        pairs["current_dud"].astype(str) + "->" + pairs["target_dud"].astype(str)
    )
    pairs["horizon_gap_days"] = pairs["current_dud"] - pairs["target_dud"]
    pairs["source_target_era_transition"] = (
        pairs["collection_era"].astype(str)
        + "->"
        + pairs["target_collection_era"].astype(str)
    )
    pairs["price_change_vnd"] = pairs["target_price_vnd"] - pairs["price_vnd"]
    pairs["price_change_pct"] = 100.0 * pairs["price_change_vnd"] / pairs["price_vnd"]
    pairs[TARGET_NAME] = (
        pairs["target_price_vnd"] <= 0.95 * pairs["price_vnd"]
    ).astype("int8")
    pairs["material_drop_next"] = pairs[TARGET_NAME]
    bridge = pairs["source_target_era_transition"].eq(
        "FLI_LIBRARY_ERA->TRIP_COM_BROWSER_ERA"
    )
    robust_drop = pairs["target_price_vnd"] <= 0.95 * 0.95 * pairs["price_vnd"]
    robust_no_drop = pairs["target_price_vnd"] > 0.95 * 1.05 * pairs["price_vnd"]
    pairs["bridge_label_stability"] = np.select(
        [~bridge, robust_drop, robust_no_drop],
        ["NOT_BRIDGE", "ROBUST_DROP", "ROBUST_NO_DROP"],
        default="AMBIGUOUS_WITHIN_SOURCE_5PCT",
    )
    pairs["is_exact_next"] = True

    frame = event_frame(source, pairs)
    frame[TARGET_NAME] = frame[TARGET_NAME].astype("int8")
    frame["source_session_key"] = frame["session_key"].astype("string")
    frame["candidate_source"] = "OBSERVED_SOURCE_EXACT_PAIR"
    frame["route_airline"] = frame["route"].astype(str) + "|" + frame["airline"].astype(str)
    history = pd.to_numeric(frame.get("prior_relative_count", 0), errors="coerce").fillna(0)
    frame["history_support_count"] = history.astype(int)
    frame["is_first_observation"] = history.eq(0).astype("int8")
    frame["regime"] = np.where(history.ge(3), "WARM", "COLD")
    frame["history_support_band"] = _band(
        history, [-1, 0, 2, float("inf")], ["FIRST_0", "COLD_1_2", "WARM_3_PLUS"]
    )
    frame["market_group"] = _market_group(frame["route"])
    frame["coverage_band"] = "AUDIT_DERIVED_AFTER_SPLIT"
    frame["support_tier"] = "AUDIT_DERIVED_AFTER_SPLIT"
    frame["route_support_quartile"] = "AUDIT_DERIVED_AFTER_SPLIT"
    frame["train_route_airline_support"] = -1
    frame["route_airline_support_band"] = "AUDIT_DERIVED_AFTER_SPLIT"
    support = pd.to_numeric(frame.get("peer_anchor_support", 0), errors="coerce").fillna(0)
    frame["anchor_support_band"] = _band(
        support,
        [-1, 0, 9, 49, float("inf")],
        ["NONE_0", "LOW_1_9", "MEDIUM_10_49", "HIGH_50_PLUS"],
    )
    age_hours = (
        frame["feature_time"] - pd.to_datetime(frame.get("historical_anchor_time"), errors="coerce")
    ).dt.total_seconds().div(3600)
    frame["anchor_age_band"] = _band(
        age_hours.fillna(float("inf")),
        [-1, 12, 24, 48, float("inf")],
        ["FRESH_LE_12H", "AGE_12_24H", "AGE_24_48H", "STALE_GT_48H"],
    )
    frame["anchor_fallback_level"] = np.where(
        frame.get("anchor_is_fallback", False), "FALLBACK", "OBSERVED_PEER"
    )
    frame["anchor_collection_era"] = frame.get(
        "temporal_market_collection_era", frame["collection_era"]
    ).astype("string")
    frame["dud_support_mode"] = "ON_GRID"
    frame["target_batch_exists"] = True
    frame["target_observation_state"] = "OBSERVED"
    frame["data_cutoff"] = CUTOFF
    frame["baseline_version"] = BASELINE_VERSION
    frame["feature_contract_version"] = "EXACT_DROP5_FULL_ERA_LEGAL_FEATURES_V2"
    frame["model_version"] = "PENDING_SELECTION"
    frame["prediction_path"] = "PENDING"
    frame["hierarchy_level"] = "FIXED_ROUTE_AIRLINE_TRANSITION_SHRINKAGE"
    frame["row_key"] = stable_row_key(frame)
    if frame["row_key"].duplicated().any():
        raise RuntimeError("Exact-next row keys are not unique")
    return frame.sort_values(["feature_time", "row_key"], kind="stable").reset_index(drop=True)


def load_frame() -> pd.DataFrame:
    if not FRAME_CACHE.is_file():
        raise FileNotFoundError(FRAME_CACHE)
    frame = pd.read_parquet(FRAME_CACHE)
    for column in ["feature_time", "label_time", "session_date", "flight_date"]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def temporal_split(
    frame: pd.DataFrame, fold: str, bridge_policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bridge_policy not in BRIDGE_POLICIES:
        raise KeyError(bridge_policy)
    spec = fold_spec(fold)
    start = pd.Timestamp(spec["validation_start"])
    end = pd.Timestamp(spec["validation_end"]) + pd.Timedelta(days=1)
    train = frame[
        frame["feature_time"].lt(start) & frame["label_time"].lt(start)
    ].copy()
    if bridge_policy == "WITHIN_ERA_ONLY":
        train = train[
            ~train["source_target_era_transition"].eq(
                "FLI_LIBRARY_ERA->TRIP_COM_BROWSER_ERA"
            )
        ].copy()
    valid = frame[
        frame["feature_time"].ge(start) & frame["feature_time"].lt(end)
    ].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"{fold}/{bridge_policy}: empty train or valid")
    if not train["label_time"].lt(start).all():
        raise RuntimeError(f"{fold}/{bridge_policy}: label-time purge failure")
    if train[TARGET_NAME].nunique() < 2 or valid[TARGET_NAME].nunique() < 2:
        raise RuntimeError(f"{fold}/{bridge_policy}: one class missing")
    train["fold"] = fold
    valid["fold"] = fold
    train["fold_role"] = spec["role"]
    valid["fold_role"] = spec["role"]
    return train, valid


def classification_weights(frame: pd.DataFrame) -> np.ndarray:
    y = frame[TARGET_NAME].to_numpy(dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    if (counts == 0).any():
        raise RuntimeError("Training population lacks one class")
    class_weight = len(y) / (2.0 * counts[y])
    current = frame["price_vnd"].to_numpy(dtype=float)
    future = frame["target_price_vnd"].to_numpy(dtype=float)
    economic = np.where(
        y == 1,
        np.maximum(current - future, 25_000.0),
        25_000.0,
    )
    economic /= max(float(np.median(economic[economic > 0])), 1.0)
    transition_count = frame.groupby("transition", observed=True)["transition"].transform("size")
    transition_weight = len(frame) / (
        frame["transition"].nunique() * transition_count.to_numpy(dtype=float)
    )
    weight = class_weight * np.clip(economic, 0.25, 4.0) * transition_weight
    return weight / max(float(weight.mean()), 1e-12)


def baseline_probability(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return hierarchy_probability(train, valid, TARGET_NAME, "FIXED_HIERARCHY")[0]


def prediction_path(job: dict[str, Any]) -> Path:
    token = str(job["job_id"])
    return OUTPUT_ROOT / "base_oof" / token / "predictions.parquet"


def metadata_path(job: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / "base_oof" / str(job["job_id"]) / "fit_metadata.joblib"


def done_path(job: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / "base_oof" / str(job["job_id"]) / "done.json"
