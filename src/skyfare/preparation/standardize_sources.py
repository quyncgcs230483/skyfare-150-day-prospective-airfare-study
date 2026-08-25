"""Standardise Fli and Trip.com daily CSVs into one point-in-time offer table."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from skyfare.core.integrity import sha256, write_json_atomic
from skyfare.core.paths import DataLayout
from skyfare.core.sources import CollectionSource
from skyfare.preparation.merge_daily_offers import assign_price_tier
from skyfare.preparation.temporal_sessions import (
    collection_session_labels,
    operational_session_dates,
)

REQUIRED_COLUMNS = (
    "scraped_at",
    "session_id",
    "origin",
    "dest",
    "route",
    "days_until_departure",
    "flight_date",
    "airline",
    "airline_name",
    "flight_no",
    "departure_time",
    "price_vnd",
)
IDENTITY_COLUMNS = (
    "session_key",
    "route",
    "airline",
    "flight_date",
    "departure_time",
    "days_until_departure",
)


def _daily_files(root: Path, cutoff: pd.Timestamp) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.glob("????-??-??.csv")):
        try:
            day = pd.Timestamp(path.stem).normalize()
        except ValueError:
            continue
        if day <= cutoff:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No daily CSV files found in {root}")
    return paths


def _read_source(
    paths: list[Path],
    source: CollectionSource,
    project_root: Path,
    source_label: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frame = frame.copy()
        frame["collection_era"] = source.value
        frame["data_source"] = source_label or (
            "fli_library" if source is CollectionSource.FLI else "trip_com"
        )
        try:
            source_file = path.relative_to(project_root).as_posix()
        except ValueError:
            source_file = path.as_posix()
        frame["source_file"] = source_file
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("scraped_at", "session_id", "flight_date", "departure_time"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["price_vnd"] = pd.to_numeric(result["price_vnd"], errors="coerce")
    result["days_until_departure"] = pd.to_numeric(
        result["days_until_departure"], errors="coerce"
    ).astype("Int64")
    for column in (
        "origin",
        "dest",
        "route",
        "airline",
        "airline_name",
        "flight_no",
        "data_source",
        "collection_era",
        "source_file",
    ):
        if column in result:
            result[column] = result[column].astype("string")
    result = result.dropna(
        subset=[
            "scraped_at",
            "session_id",
            "route",
            "airline",
            "flight_date",
            "departure_time",
            "days_until_departure",
            "price_vnd",
        ]
    )
    result = result[result["price_vnd"].gt(0)].copy()
    result["session_date"] = pd.to_datetime(
        operational_session_dates(result["session_id"])
    )
    result["session_label"] = collection_session_labels(result["session_id"]).astype(
        "string"
    )
    day_number = (result["session_date"].astype("int64") // 86_400_000_000_000).astype(
        "int32"
    )
    result["session_key"] = (
        day_number * 2 + result["session_label"].eq("PM").astype("int8")
    )
    departure_hhmm = (
        result["departure_time"].dt.hour * 100 + result["departure_time"].dt.minute
    ).astype("int16")
    result["schedule_slot_id"] = pd.util.hash_pandas_object(
        pd.DataFrame(
            {
                "route": result["route"].astype("string"),
                "airline": result["airline"].astype("string"),
                "flight_date": result["flight_date"],
                "departure_hhmm": departure_hhmm,
            }
        ),
        index=False,
    ).astype("uint64")
    result["is_schedule_fallback"] = True
    result = result.sort_values([*IDENTITY_COLUMNS, "scraped_at"], kind="stable")
    return result.drop_duplicates(list(IDENTITY_COLUMNS), keep="last").reset_index(drop=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".csv", encoding="utf-8", newline=""
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".parquet") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(cutoff: pd.Timestamp) -> dict[str, object]:
    layout = DataLayout.resolve()
    layout.create_runtime_directories()
    fli_paths = _daily_files(layout.raw_fli, cutoff)
    manual_paths = _daily_files(layout.raw_google_flights_manual, cutoff)
    trip_paths = _daily_files(layout.raw_trip_com, cutoff)
    fli_library = _normalise(
        _read_source(fli_paths, CollectionSource.FLI, layout.root)
    )
    google_flights_manual = _normalise(
        _read_source(
            manual_paths,
            CollectionSource.FLI,
            layout.root,
            source_label="google_flights_manual_9g",
        )
    )
    fli = pd.concat(
        [fli_library, google_flights_manual], ignore_index=True, sort=False
    )
    trip = _normalise(_read_source(trip_paths, CollectionSource.TRIP_COM, layout.root))
    trip["price_tier"] = assign_price_tier(trip)
    trip_standard = trip[trip["price_tier"].eq("standard")].copy()

    fli_master = layout.standardised / "fli_standard_offers.csv"
    trip_master = layout.standardised / "trip_com_standard_offers.csv"
    standard_parquet = layout.standardised / "standard_offers.parquet"
    _atomic_csv(fli_master, fli)
    _atomic_csv(trip_master, trip_standard)
    combined = pd.concat([fli, trip_standard], ignore_index=True, sort=False)
    combined = combined[pd.to_datetime(combined["session_date"]).le(cutoff)].copy()
    combined = combined.sort_values([*IDENTITY_COLUMNS, "scraped_at"], kind="stable")
    combined = combined.drop_duplicates(list(IDENTITY_COLUMNS), keep="last")
    _atomic_parquet(standard_parquet, combined)

    manifest = {
        "status": "PASS",
        "cutoff_inclusive": cutoff.strftime("%Y-%m-%d"),
        "source_taxonomy": {
            CollectionSource.FLI.value: "Fli Python library for Google Flights",
            CollectionSource.TRIP_COM.value: "Trip.com browser collector",
        },
        "raw_files": {
            "fli_library": len(fli_paths),
            "google_flights_manual_9g": len(manual_paths),
            "trip_com": len(trip_paths),
        },
        "rows": {
            "fli_library": len(fli_library),
            "google_flights_manual_9g": len(google_flights_manual),
            "trip_com_standard": len(trip_standard),
            "combined": len(combined),
        },
        "outputs": {
            path.relative_to(layout.root).as_posix(): sha256(path)
            for path in (fli_master, trip_master, standard_parquet)
        },
    }
    write_json_atomic(
        layout.artifacts / "data_preparation" / "source_standardisation_manifest.json",
        manifest,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2026-07-28")
    args = parser.parse_args()
    report = build(pd.Timestamp(args.cutoff).normalize())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
