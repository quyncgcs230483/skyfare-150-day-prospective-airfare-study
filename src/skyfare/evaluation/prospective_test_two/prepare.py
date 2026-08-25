#!/usr/bin/env python3
"""Build sealed prospective Test 2 frames from immutable daily raw files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.core.sources import CollectionSource, normalize_source_columns
from skyfare.features import audit_common, candidate_feature_contract
from skyfare.models import classification_runtime as class_runtime
from skyfare.models import fare_frame_runtime as next_runtime
from skyfare.models import regression_runtime as reg_runtime
from skyfare.preparation import merge_daily_offers as merge
from skyfare.preparation import temporal_sessions

TEST_START = pd.Timestamp("2026-08-09")
TEST_END = pd.Timestamp("2026-08-19")
TEST_END_EXCLUSIVE = TEST_END + pd.Timedelta(days=1)
HISTORY_CUTOFF = pd.Timestamp("2026-08-08")
DEVELOPMENT_CUTOFF = pd.Timestamp("2026-07-28")
EXPECTED_DUDS = {1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60}
RAW_COLUMNS = [
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session_semantics(frame: pd.DataFrame, temporal_sessions) -> pd.DataFrame:
    result = frame.copy()
    for column in ("scraped_at", "session_id", "flight_date", "departure_time"):
        result[column] = pd.to_datetime(result[column], errors="raise").astype("datetime64[ns]")
    result["days_until_departure"] = pd.to_numeric(
        result["days_until_departure"], errors="raise"
    ).astype("int64")
    result["price_vnd"] = pd.to_numeric(result["price_vnd"], errors="raise").astype(float)
    result["session_date"] = pd.to_datetime(
        temporal_sessions.operational_session_dates(result["session_id"])
    ).astype("datetime64[ns]")
    result["session_label"] = temporal_sessions.collection_session_labels(
        result["session_id"]
    ).astype("string")
    day_number = result["session_date"].map(pd.Timestamp.toordinal).astype("int64")
    result["session_key"] = day_number * 2 + result["session_label"].eq("PM").astype("int8")
    departure_hhmm = (
        result["departure_time"].dt.hour * 100 + result["departure_time"].dt.minute
    ).astype("int16")
    result["schedule_slot_id"] = pd.util.hash_pandas_object(
        pd.DataFrame(
            {
                "route": result["route"].astype(object),
                "airline": result["airline"].astype(object),
                "flight_date": result["flight_date"],
                "departure_hhmm": departure_hhmm,
            }
        ),
        index=False,
    ).astype("uint64")
    result["is_schedule_fallback"] = True
    result["collection_era"] = CollectionSource.TRIP_COM.value
    return result


def _validate_test_daily_files(daily_paths: list[Path], temporal_sessions) -> dict[str, object]:
    selected = [path for path in daily_paths if TEST_START <= pd.Timestamp(path.stem) <= TEST_END]
    expected = {day.strftime("%Y-%m-%d") for day in pd.date_range(TEST_START, TEST_END)}
    observed = {path.stem for path in selected}
    if observed != expected:
        raise RuntimeError(f"Test 2 raw date coverage mismatch missing={sorted(expected - observed)}")
    rows = 0
    hashes: dict[str, str] = {}
    session_coverage: dict[str, dict[str, object]] = {}
    missing_sessions: list[str] = []
    for path in selected:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        session_id = pd.to_datetime(frame["session_id"], errors="raise")
        frame["session_label"] = temporal_sessions.collection_session_labels(session_id)
        bands = set(frame["session_label"].dropna().astype(str))
        if not bands or not bands.issubset({"AM", "PM"}):
            raise RuntimeError(f"{path.name}: invalid collection-session coverage {bands}")
        for missing in sorted({"AM", "PM"} - bands):
            missing_sessions.append(f"{path.stem}/{missing}")
        day_report: dict[str, object] = {}
        for label, part in frame.groupby("session_label", observed=True):
            duds = set(pd.to_numeric(part["days_until_departure"], errors="raise").astype(int))
            routes = int(part["route"].nunique())
            airlines = int(part["airline"].nunique())
            if duds != EXPECTED_DUDS or routes != 20 or airlines < 4:
                raise RuntimeError(f"{path.name}/{label}: route-airline-DUD coverage incomplete")
            day_report[str(label)] = {
                "rows": len(part),
                "routes": routes,
                "airlines": airlines,
                "duds": sorted(duds),
            }
        session_coverage[path.stem] = day_report
        rows += len(frame)
        hashes[path.name] = sha256(path)
    return {
        "dates": sorted(observed),
        "rows": rows,
        "sha256": hashes,
        "session_coverage": session_coverage,
        "missing_sessions": missing_sessions,
        "missingness_policy": "KEEP_OBSERVED_ROWS; NEVER_IMPUTE; REPORT_DENOMINATOR",
    }


def _build_raw_loader(
    frozen_offers_path: Path,
    standard_trip: pd.DataFrame,
    temporal_sessions,
):
    frozen = pd.read_parquet(frozen_offers_path, columns=RAW_COLUMNS)
    frozen = _session_semantics(frozen, temporal_sessions)
    frozen["collection_era"] = pd.read_parquet(
        frozen_offers_path, columns=["collection_era"]
    )["collection_era"].astype("string").to_numpy()
    frozen = normalize_source_columns(frozen)
    if frozen["session_date"].max().normalize() != HISTORY_CUTOFF:
        raise RuntimeError("Test 1 history offers do not end at Test 2 refit cutoff")

    incoming = _session_semantics(standard_trip, temporal_sessions)
    incoming = incoming.loc[
        incoming["session_date"].ge(TEST_START)
        & incoming["session_date"].le(TEST_END),
        RAW_COLUMNS,
    ].copy()
    if set(incoming["session_date"].dt.strftime("%Y-%m-%d")) != {
        day.strftime("%Y-%m-%d") for day in pd.date_range(TEST_START, TEST_END)
    }:
        raise RuntimeError("Standard-tier Test 2 rows do not cover all 11 days")
    combined = pd.concat([frozen[RAW_COLUMNS], incoming], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["session_key", "schedule_slot_id", "scraped_at", "session_id"], kind="stable"
    ).reset_index(drop=True)
    if combined["session_date"].max().normalize() != TEST_END:
        raise RuntimeError("Combined source does not end at Test 2 cutoff")

    def loader(cutoff: pd.Timestamp) -> pd.DataFrame:
        cutoff = pd.Timestamp(cutoff).normalize()
        if cutoff != TEST_END:
            raise RuntimeError(f"Unexpected sealed cutoff {cutoff.date()}")
        return combined.copy()

    return loader, frozen, incoming


def _parity_report(rebuilt: pd.DataFrame, frozen: pd.DataFrame) -> dict[str, object]:
    old = rebuilt.loc[pd.to_datetime(rebuilt["session_date"]).le(HISTORY_CUTOFF)].copy()
    if len(old) != len(frozen):
        raise RuntimeError(f"Frozen-offer parity row count changed: {len(old)} != {len(frozen)}")
    columns = [
        "offer_id",
        "session_key",
        "schedule_slot_id",
        "price_vnd",
        "feature_time",
        "anchor_vnd",
        "temporal_market_median_price",
        "previous_price_same_schedule",
    ]
    left = old[columns].sort_values("offer_id").reset_index(drop=True)
    right = frozen[columns].sort_values("offer_id").reset_index(drop=True)
    if not left["offer_id"].astype("string").equals(right["offer_id"].astype("string")):
        raise RuntimeError("Frozen-offer parity identity changed")
    mismatches: dict[str, int] = {}
    for column in columns[1:]:
        if "time" in column:
            a = pd.to_datetime(left[column], errors="coerce")
            b = pd.to_datetime(right[column], errors="coerce")
            equal = a.eq(b) | (a.isna() & b.isna())
        elif pd.api.types.is_numeric_dtype(left[column]):
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            equal = pd.Series(np.isclose(a, b, rtol=0, atol=1e-8, equal_nan=True))
        else:
            equal = left[column].astype("string").eq(right[column].astype("string"))
        count = int((~equal).sum())
        if count:
            mismatches[column] = count
    if mismatches:
        raise RuntimeError(f"Frozen-offer semantic parity changed: {mismatches}")
    return {"status": "PASS", "rows": len(old), "columns": columns}


def _observability(offers: pd.DataFrame, next_dud: dict[int, int]) -> pd.DataFrame:
    source = offers.loc[
        offers["days_until_departure"].isin(next_dud)
        & offers["feature_time"].ge(TEST_START)
        & offers["feature_time"].lt(TEST_END_EXCLUSIVE)
    ].copy()
    source["current_dud"] = source["days_until_departure"].astype(int)
    source["target_dud"] = source["current_dud"].map(next_dud).astype(int)
    source["expected_target_session_date"] = (
        source["flight_date"].dt.normalize()
        - pd.to_timedelta(source["target_dud"], unit="D")
    )
    target = offers[
        ["schedule_slot_id", "session_label", "days_until_departure", "offer_id", "feature_time"]
    ].rename(
        columns={
            "days_until_departure": "target_dud",
            "offer_id": "observed_target_offer_id",
            "feature_time": "observed_label_time",
        }
    )
    audit = source.merge(
        target,
        on=["schedule_slot_id", "session_label", "target_dud"],
        how="left",
        validate="one_to_one",
    )
    mature = audit["expected_target_session_date"].le(TEST_END)
    observed = audit["observed_target_offer_id"].notna()
    audit["target_observation_state"] = np.select(
        [~mature, mature & observed, mature & ~observed],
        ["IMMATURE_RIGHT_CENSORED", "OBSERVED", "MATURE_NOT_OBSERVED"],
        default="INVALID",
    )
    audit["metric_eligible"] = audit["target_observation_state"].eq("OBSERVED")
    return audit


def build(
    development_input_root: Path,
    history_root: Path,
    output_root: Path,
    trip_raw_root: Path | None = None,
) -> dict[str, object]:
    development_input_root = development_input_root.resolve()
    history_root = history_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work = output_root / ".build"
    work.mkdir(parents=True, exist_ok=True)
    layout = DataLayout.resolve()
    daily_paths, cutoff = merge.discover_daily_files(
        (trip_raw_root or layout.raw_trip_com).resolve(), TEST_END
    )
    raw_audit = _validate_test_daily_files(daily_paths, temporal_sessions)
    # QH availability varies by day. Latest Test 2 day has four observed airlines.
    # Merge validation remains strict for every observed route, DUD, and AM/PM band.
    merge.EXPECTED_AIRLINES = 4
    _, standard_trip, _, aggregate_stats = merge.build_aggregates(daily_paths, cutoff)

    frozen_offers_path = history_root / "standard_offers.parquet"
    class_dev = development_input_root / "classification_training_frame.parquet"
    reg_dev = development_input_root / "regression_training_frame.parquet"
    for path in (frozen_offers_path, class_dev, reg_dev):
        if not path.is_file():
            raise FileNotFoundError(path)
    loader, frozen_raw, incoming = _build_raw_loader(
        frozen_offers_path, standard_trip, temporal_sessions
    )

    _ = audit_common
    candidate_feature_contract.PEAK_CONFIG = layout.controls / "peak_dates_vietnam_2026.json"

    next_runtime.CUTOFF = TEST_END.strftime("%Y-%m-%d")
    next_runtime._standard_raw_loader = loader
    next_runtime.STANDARD_OFFERS_CACHE = work / "standard_test2_offers.parquet"
    next_runtime.FRAME_CACHE = work / "unused_next_frame.parquet"
    next_runtime.LEDGER_CACHE = work / "unused_next_ledger.parquet"
    offers = next_runtime.normalize_times(next_runtime.build_standard_offers())
    if offers["feature_time"].max() >= TEST_END_EXCLUSIVE:
        raise RuntimeError("post-Test-2 row entered sealed offers")
    frozen_full = pd.read_parquet(frozen_offers_path)
    parity = _parity_report(offers, frozen_full)

    reg_runtime.CUTOFF = TEST_END.strftime("%Y-%m-%d")
    reg_runtime.STANDARD_OFFERS_CACHE = next_runtime.STANDARD_OFFERS_CACHE
    reg_runtime.RECURRENT_SEQUENCE_SOURCE = next_runtime.STANDARD_OFFERS_CACHE
    reg_runtime.legacy.CUTOFF = TEST_END.strftime("%Y-%m-%d")
    reg_runtime.legacy.STANDARD_OFFERS_CACHE = next_runtime.STANDARD_OFFERS_CACHE
    reg_runtime.FRAME_CACHE = work / "regression_full.parquet"
    reg_runtime.LEDGER_CACHE = work / "regression_ledger.parquet"
    regression, _ = reg_runtime.build_training_frame()
    regression = reg_runtime.normalize_times(regression)
    regression["row_key"] = reg_runtime.stable_row_key(regression)
    regression_test = regression.loc[
        regression["label_time"].ge(TEST_START)
        & regression["label_time"].lt(TEST_END_EXCLUSIVE)
    ].copy()

    class_runtime.CUTOFF = TEST_END.strftime("%Y-%m-%d")
    exact = class_runtime.build_exact_frame(offers)
    exact["feature_time"] = pd.to_datetime(exact["feature_time"], errors="raise")
    exact["label_time"] = pd.to_datetime(exact["label_time"], errors="raise")
    classification_test = exact.loc[
        exact["feature_time"].ge(TEST_START)
        & exact["feature_time"].lt(TEST_END_EXCLUSIVE)
        & exact["label_time"].lt(TEST_END_EXCLUSIVE)
        & ~exact["source_target_era_transition"].eq("FLI_LIBRARY_ERA->TRIP_COM_BROWSER_ERA")
    ].copy()
    observability = _observability(offers, class_runtime.NEXT_DUD)
    eligible = set(
        observability.loc[observability["metric_eligible"], "offer_id"].astype("string")
    )
    if not set(classification_test["offer_id"].astype("string")).issubset(eligible):
        raise RuntimeError("Classification metric frame contains censored row")
    if regression_test.empty or classification_test.empty:
        raise RuntimeError("Test 2 frame is empty")
    if regression_test["label_time"].max() >= TEST_END_EXCLUSIVE:
        raise RuntimeError("post-Test-2 regression label entered Test 2")
    if classification_test["label_time"].max() >= TEST_END_EXCLUSIVE:
        raise RuntimeError("post-Test-2 classification label entered Test 2")

    offers.to_parquet(output_root / "standard_offers.parquet", index=False, compression="zstd")
    regression_test.to_parquet(
        output_root / "regression_test_2_frame.parquet", index=False, compression="zstd"
    )
    classification_test.to_parquet(
        output_root / "classification_test_2_frame.parquet", index=False, compression="zstd"
    )
    observability.to_parquet(
        output_root / "classification_test_2_observability.parquet",
        index=False,
        compression="zstd",
    )

    artifact_names = [
        "standard_offers.parquet",
        "regression_test_2_frame.parquet",
        "classification_test_2_frame.parquet",
        "classification_test_2_observability.parquet",
    ]
    report = {
        "status": "PASS",
        "protocol": "PROSPECTIVE_TEST_2_R1_RIGHT_CENSOR_SAFE",
        "development_cutoff": str(DEVELOPMENT_CUTOFF.date()),
        "history_cutoff": str(HISTORY_CUTOFF.date()),
        "test_2": [str(TEST_START.date()), str(TEST_END.date())],
        "raw_test_2": raw_audit,
        "aggregate_stats": aggregate_stats,
        "standard_incoming_rows": len(incoming),
        "standard_offer_rows": len(offers),
        "classification_rows": len(classification_test),
        "classification_feature_dates": sorted(
            classification_test["feature_time"].dt.strftime("%Y-%m-%d").unique()
        ),
        "classification_label_max": classification_test["label_time"].max(),
        "classification_observability": {
            str(key): int(value)
            for key, value in observability["target_observation_state"].value_counts().items()
        },
        "classification_metric_denominator": "OBSERVED_AND_MATURE_ONLY",
        "regression_rows": len(regression_test),
        "regression_label_dates": sorted(
            regression_test["label_time"].dt.strftime("%Y-%m-%d").unique()
        ),
        "frozen_history_parity": parity,
        "frozen_input_sha256": {
            path.name: sha256(path) for path in (frozen_offers_path, class_dev, reg_dev)
        },
        "artifacts": {
            name: {
                "bytes": (output_root / name).stat().st_size,
                "sha256": sha256(output_root / name),
            }
            for name in artifact_names
        },
        "post_test_2_rows_accessed": 0,
    }
    (output_root / "SEALED_TEST_2_MANIFEST_R1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-input-root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trip-raw-root", type=Path)
    args = parser.parse_args()
    build(
        args.development_input_root,
        args.history_root,
        args.output_root,
        args.trip_raw_root,
    )


if __name__ == "__main__":
    main()
