"""Shared new-build feature-selection audit loader.

Snapshot: new-build 110 pilot, cutoff 2026-07-10.
Branch: model-free Error Analysis / Feature Selection.
Inputs: Google Flights/fli standard CSV plus Trip daily CSVs through cutoff.
Outputs: none; callers write audit artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


from skyfare.core.paths import DataLayout
from skyfare.core.sources import CollectionSource
from skyfare.preparation.temporal_sessions import (
    collection_session_labels,
    operational_session_dates,
)

LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
FLI_STANDARD_PATH = LAYOUT.standardised / "fli_standard_offers.csv"
TRIP_DIR = LAYOUT.raw_trip_com
TRIP_STANDARD_PATH = LAYOUT.standardised / "trip_com_standard_offers.csv"
CUTOFF = pd.Timestamp("2026-07-10")
BOOKING_WINDOWS = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1]

CORE_COLUMNS = [
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
    "data_source",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(cutoff: pd.Timestamp = CUTOFF) -> list[Path]:
    trip = [
        path
        for path in sorted(TRIP_DIR.glob("2026-*.csv"))
        if pd.Timestamp(path.stem) <= cutoff
    ]
    if not FLI_STANDARD_PATH.exists() or not trip:
        raise FileNotFoundError("New-build raw inputs are incomplete")
    return [FLI_STANDARD_PATH, *trip]


def input_manifest(cutoff: pd.Timestamp = CUTOFF) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths(cutoff)
    ]


def standard_input_paths() -> list[Path]:
    """Only immutable standard-tier masters allowed by final methodology."""

    paths = [FLI_STANDARD_PATH, TRIP_STANDARD_PATH]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Standard-only raw inputs are incomplete: " + ", ".join(missing))
    if any("nonstd" in path.name.lower() for path in paths):
        raise RuntimeError("Non-standard source entered standard-only input list")
    return paths


def standard_input_manifest() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "allowed_tier": "STANDARD_ONLY",
        }
        for path in standard_input_paths()
    ]


def _read(path: Path, collection_era: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=lambda column: str(column).lstrip("\ufeff") in CORE_COLUMNS,
        dtype_backend="pyarrow",
    )
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    missing = sorted(set(CORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    frame = frame[CORE_COLUMNS].copy()
    frame["collection_era"] = collection_era
    try:
        source_file = path.relative_to(ROOT)
    except ValueError:
        source_file = path
    frame["source_file"] = str(source_file)
    return frame


def load_new_build_raw(cutoff: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """Load new-build source rows from immutable daily inputs.

    Trip master is intentionally excluded because old incremental merge missed
    partial-day supplements. Datetimes are interpreted as Vietnam local wall
    time; no UTC conversion is applied to naive source timestamps.
    """

    frames = [_read(FLI_STANDARD_PATH, CollectionSource.FLI.value)]
    frames.extend(
        _read(path, CollectionSource.TRIP_COM.value)
        for path in input_paths(cutoff)[1:]
    )
    data = pd.concat(frames, ignore_index=True, sort=False)

    for column in ["scraped_at", "session_id", "flight_date", "departure_time"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["price_vnd"] = pd.to_numeric(data["price_vnd"], errors="coerce")
    data["days_until_departure"] = pd.to_numeric(
        data["days_until_departure"], errors="coerce"
    ).astype("Int64")

    data["session_date"] = pd.to_datetime(
        operational_session_dates(data["session_id"])
    )
    data = data[data["session_date"].le(cutoff)].copy()
    data = data.dropna(
        subset=[
            "session_id",
            "route",
            "airline",
            "flight_date",
            "departure_time",
            "days_until_departure",
            "price_vnd",
        ]
    )
    data = data[data["price_vnd"].gt(0)].copy()

    data["session_label"] = collection_session_labels(data["session_id"]).astype(
        "string[pyarrow]"
    )
    band = data["session_label"].eq("PM").astype("int8")
    data["session_key"] = (
        (data["session_date"].astype("int64") // 86_400_000_000_000).astype("int32") * 2
        + band
    )
    data["departure_hhmm"] = (
        data["departure_time"].dt.hour * 100 + data["departure_time"].dt.minute
    ).astype("int16")
    data["schedule_slot_id"] = pd.util.hash_pandas_object(
        data[["route", "airline", "flight_date", "departure_hhmm"]], index=False
    ).astype("uint64")

    airline = data["airline"].astype("string[pyarrow]")

    raw_no = data["flight_no"].astype("string[pyarrow]").str.strip().str.replace(r"\.0$", "", regex=True)
    departure_hhmm_text = data["departure_hhmm"].astype("string[pyarrow]").str.zfill(4)
    fallback = airline.str.cat(departure_hhmm_text, sep="-")
    data["flight_no_clean"] = raw_no
    data["is_schedule_fallback"] = raw_no.str.upper().eq(fallback.str.upper()).fillna(False)
    numeric_no = raw_no.str.replace(r"\D", "", regex=True)
    data["verified_flight_no_proxy"] = np.where(
        data["is_schedule_fallback"],
        pd.NA,
        airline.str.cat(numeric_no.fillna(""), sep=""),
    )

    duplicate_columns = [
        "scraped_at",
        "session_id",
        "route",
        "days_until_departure",
        "flight_date",
        "airline",
        "flight_no_clean",
        "departure_time",
        "price_vnd",
    ]
    data["is_exact_duplicate"] = data.duplicated(duplicate_columns, keep=False)
    data["timezone_contract"] = "Asia/Ho_Chi_Minh_local_wall_time"
    for column in [
        "origin",
        "dest",
        "route",
        "airline",
        "airline_name",
        "data_source",
        "collection_era",
        "source_file",
        "session_label",
        "flight_no_clean",
        "verified_flight_no_proxy",
        "timezone_contract",
    ]:
        data[column] = data[column].astype("category")
    return data.sort_values(["session_id", "route", "airline", "flight_date", "departure_time"])


def load_standard_only_raw(cutoff: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """Load only standard Fli and Trip.com masters through cutoff.

    Fli ends on 2026-05-14. Trip.com starts on 2026-05-15. Source paths and
    row-level tier checks fail closed so nonstd rows cannot enter silently.
    """

    cutoff = pd.Timestamp(cutoff).normalize()
    fli = _read(FLI_STANDARD_PATH, CollectionSource.FLI.value)
    trip = _read(TRIP_STANDARD_PATH, CollectionSource.TRIP_COM.value)
    data = pd.concat([fli, trip], ignore_index=True, sort=False)

    for column in ["scraped_at", "session_id", "flight_date", "departure_time"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["price_vnd"] = pd.to_numeric(data["price_vnd"], errors="coerce")
    data["days_until_departure"] = pd.to_numeric(
        data["days_until_departure"], errors="coerce"
    ).astype("Int64")
    data["session_date"] = pd.to_datetime(
        operational_session_dates(data["session_id"])
    )

    fli_end = pd.Timestamp("2026-05-14")
    trip_start = pd.Timestamp("2026-05-15")
    valid_era = (
        data["collection_era"].eq(CollectionSource.FLI.value)
        & data["session_date"].le(fli_end)
    ) | (
        data["collection_era"].eq(CollectionSource.TRIP_COM.value)
        & data["session_date"].ge(trip_start)
    )
    data = data[valid_era & data["session_date"].le(cutoff)].copy()
    data = data.dropna(
        subset=[
            "session_id",
            "route",
            "airline",
            "flight_date",
            "departure_time",
            "days_until_departure",
            "price_vnd",
        ]
    )
    data = data[data["price_vnd"].gt(0)].copy()

    data["session_label"] = collection_session_labels(data["session_id"]).astype(
        "string[pyarrow]"
    )
    band = data["session_label"].eq("PM").astype("int8")
    data["session_key"] = (
        (data["session_date"].astype("int64") // 86_400_000_000_000).astype("int32") * 2
        + band
    )
    data["departure_hhmm"] = (
        data["departure_time"].dt.hour * 100 + data["departure_time"].dt.minute
    ).astype("int16")
    data["schedule_slot_id"] = pd.util.hash_pandas_object(
        data[["route", "airline", "flight_date", "departure_hhmm"]], index=False
    ).astype("uint64")

    airline = data["airline"].astype("string[pyarrow]")
    raw_no = (
        data["flight_no"]
        .astype("string[pyarrow]")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    departure_hhmm_text = data["departure_hhmm"].astype(
        "string[pyarrow]"
    ).str.zfill(4)
    fallback = airline.str.cat(departure_hhmm_text, sep="-")
    data["flight_no_clean"] = raw_no
    # Identity always uses airline-HHMM. Raw flight number remains audit-only.
    data["is_schedule_fallback"] = True
    data["verified_flight_no_proxy"] = pd.NA
    data["semantic_identity"] = fallback
    data["is_exact_duplicate"] = data.duplicated(
        [
            "scraped_at",
            "session_id",
            "route",
            "days_until_departure",
            "flight_date",
            "airline",
            "departure_time",
            "price_vnd",
        ],
        keep=False,
    )
    data["timezone_contract"] = "Asia/Ho_Chi_Minh_local_wall_time"
    for column in [
        "origin",
        "dest",
        "route",
        "airline",
        "airline_name",
        "data_source",
        "collection_era",
        "source_file",
        "session_label",
        "flight_no_clean",
        "semantic_identity",
        "timezone_contract",
    ]:
        data[column] = data[column].astype("category")
    if data["source_file"].astype(str).str.contains("nonstd", case=False).any():
        raise RuntimeError("Non-standard source row entered standard-only frame")
    return data.sort_values(
        ["session_id", "route", "airline", "flight_date", "departure_time"]
    )


def completed_batch_offers(data: pd.DataFrame) -> pd.DataFrame:
    """One deterministic offer state per completed AM/PM batch and slot.

    Latest scrape wins when a resumed collection observes same schedule slot
    more than once inside one AM/PM band. EA01 reports those collisions; this
    function does not hide their frequency.
    """

    ordered = data.sort_values(["session_key", "schedule_slot_id", "scraped_at", "session_id"])
    return ordered.drop_duplicates(["session_key", "schedule_slot_id"], keep="last").copy()


def next_window_map() -> dict[int, int]:
    return dict(zip(BOOKING_WINDOWS[:-1], BOOKING_WINDOWS[1:]))
