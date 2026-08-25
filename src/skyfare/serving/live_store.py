"""Batch-safe PostgreSQL ingestion and snapshot metadata helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from skyfare.preparation.temporal_sessions import (
    collection_session_labels,
    operational_session_dates,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_SQL = Path(__file__).with_name("schema.sql")
DEFAULT_DSN = "host=127.0.0.1 port=5432 dbname=fyp_flights user=postgres"
BOOKING_WINDOWS = {1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60}
EXPECTED_ROUTES = 20
EXPECTED_AIRLINES = 5
EXPECTED_AIRLINE_CODES = {"9G", "QH", "VJ", "VN", "VU"}
MINIMUM_AM_AIRLINES = 4

STANDARD_COLUMNS = [
    "row_key",
    "batch_id",
    "scraped_at",
    "session_id",
    "session_date",
    "session_label",
    "origin",
    "dest",
    "route",
    "days_until_departure",
    "flight_date",
    "airline",
    "airline_name",
    "flight_no",
    "departure_time",
    "departure_hhmm",
    "schedule_slot_key",
    "price_vnd",
    "data_source",
    "collection_era",
    "source_file",
]


def dsn() -> str:
    return os.environ.get("SKYFARE_DATABASE_URL", DEFAULT_DSN)


def _driver():
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError(
            "psycopg2 is required. Install requirements for "
            "the serving dependency group."
        ) from exc
    return psycopg2


@contextmanager
def connection() -> Iterator[object]:
    conn = _driver().connect(dsn())
    try:
        yield conn
    finally:
        conn.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def apply_schema() -> None:
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()


def _normalise_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _row_keys(frame: pd.DataFrame) -> pd.Series:
    key_columns = [
        "batch_id",
        "route",
        "airline",
        "flight_date",
        "departure_hhmm",
        "price_vnd",
        "scraped_at",
    ]
    token = frame[key_columns].astype("string").fillna("").agg("|".join, axis=1)
    duplicate_ordinal = token.groupby(token, sort=False).cumcount().astype(str)
    token = token + "|" + duplicate_ordinal
    hashed = pd.util.hash_pandas_object(token, index=False).astype("uint64")
    return hashed.map(lambda value: f"{int(value):016x}")


def prepare_rows(
    frame: pd.DataFrame,
    collection_era: str,
    source_file: Path,
    price_tier: pd.Series | str,
) -> pd.DataFrame:
    result = frame.copy()
    for column in ("scraped_at", "session_id", "flight_date", "departure_time"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["price_vnd"] = pd.to_numeric(result["price_vnd"], errors="coerce")
    result["days_until_departure"] = pd.to_numeric(
        result["days_until_departure"], errors="coerce"
    )
    result["session_date"] = pd.to_datetime(
        operational_session_dates(result["session_id"])
    )
    result["session_label"] = collection_session_labels(
        result["session_id"]
    ).astype("string")
    result["departure_hhmm"] = (
        result["departure_time"].dt.hour * 100
        + result["departure_time"].dt.minute
    )
    result["schedule_slot_key"] = (
        result["route"].astype("string").str.upper()
        + "|"
        + result["airline"].astype("string").str.upper()
        + "|"
        + result["flight_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + result["departure_hhmm"].astype("Int64").astype("string").str.zfill(4)
    )
    result["collection_era"] = collection_era
    result["source_file"] = str(source_file.resolve())
    if isinstance(price_tier, str):
        result["price_tier"] = price_tier
    else:
        result["price_tier"] = price_tier.astype("string")
    result = result.dropna(
        subset=[
            "scraped_at",
            "session_id",
            "session_date",
            "session_label",
            "route",
            "airline",
            "flight_date",
            "departure_time",
            "departure_hhmm",
            "days_until_departure",
            "price_vnd",
        ]
    ).copy()
    result = result[
        result["days_until_departure"].isin(BOOKING_WINDOWS)
        & result["price_vnd"].gt(0)
    ].copy()
    result["route"] = result["route"].astype(str).str.upper()
    result["airline"] = result["airline"].astype(str).str.upper()
    result["days_until_departure"] = result[
        "days_until_departure"
    ].astype("int16")
    result["departure_hhmm"] = result["departure_hhmm"].astype("int16")
    result["price_vnd"] = result["price_vnd"].round().astype("int64")
    for column in ("origin", "dest"):
        result[column] = result[column].astype(str).str.upper()
    for column in (
        "airline_name",
        "flight_no",
        "data_source",
    ):
        if column not in result:
            result[column] = pd.NA
    result["batch_id"] = (
        collection_era
        + "|"
        + result["session_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + result["session_label"].astype(str)
    )
    result["row_key"] = _row_keys(result)
    return result


def validate_batch(
    frame: pd.DataFrame,
    enforce_full_coverage: bool,
    allow_airline_subset: bool = False,
) -> dict[str, object]:
    if frame.empty:
        raise RuntimeError("Batch has no valid canonical observations")
    if frame["batch_id"].nunique() != 1:
        raise RuntimeError("Batch frame contains multiple batch identifiers")
    duplicate_slots = frame.duplicated("schedule_slot_key", keep="last")
    duds = set(frame["days_until_departure"].astype(int))
    routes = int(frame["route"].nunique())
    airline_codes = set(frame["airline"].astype(str).str.upper().unique())
    airlines = len(airline_codes)
    missing_airlines = sorted(EXPECTED_AIRLINE_CODES.difference(airline_codes))
    unexpected_airlines = sorted(airline_codes.difference(EXPECTED_AIRLINE_CODES))
    failures: list[str] = []
    if enforce_full_coverage and duds != BOOKING_WINDOWS:
        failures.append(f"DUD={sorted(duds)}")
    if enforce_full_coverage and routes != EXPECTED_ROUTES:
        failures.append(f"routes={routes}")
    if enforce_full_coverage and unexpected_airlines:
        failures.append(f"unexpected_airlines={unexpected_airlines}")
    if enforce_full_coverage and airlines != EXPECTED_AIRLINES:
        subset_is_valid = (
            allow_airline_subset
            and airlines >= MINIMUM_AM_AIRLINES
            and not unexpected_airlines
        )
        if not subset_is_valid:
            failures.append(f"airlines={airlines}")
    if failures:
        raise RuntimeError("Incomplete batch: " + "; ".join(failures))
    return {
        "routes": routes,
        "airlines": airlines,
        "airline_codes": sorted(airline_codes),
        "missing_expected_airlines": missing_airlines,
        "airline_subset_allowed": bool(
            allow_airline_subset and missing_airlines
        ),
        "duds": sorted(duds),
        "route_airline_pairs": int(
            frame[["route", "airline"]].drop_duplicates().shape[0]
        ),
        "duplicate_schedule_slots": int(duplicate_slots.sum()),
    }


def _copy_standard_frame(
    cursor: object,
    frame: pd.DataFrame,
    collection_era: str,
) -> None:
    payload = frame.copy()
    payload["collection_era"] = collection_era
    payload = payload[STANDARD_COLUMNS]
    for column in (
        "scraped_at",
        "session_id",
        "session_date",
        "flight_date",
        "departure_time",
    ):
        payload[column] = payload[column].map(_normalise_scalar)
    buffer = io.StringIO()
    payload.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    columns = ", ".join(STANDARD_COLUMNS)
    cursor.copy_expert(
        "COPY skyfare_live.standard_offers "
        f"({columns}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')",
        buffer,
    )


def ingest_batch(
    frame: pd.DataFrame,
    collection_era: str,
    source_file: Path,
    source_hash: str,
    enforce_full_coverage: bool,
    enforce_registry: bool = False,
    allow_airline_subset: bool = False,
    validation_context: dict[str, object] | None = None,
) -> dict[str, object]:
    batch_id = str(frame["batch_id"].iloc[0])
    session_date = pd.Timestamp(frame["session_date"].iloc[0]).date()
    session_label = str(frame["session_label"].iloc[0])
    started_at = pd.to_datetime(frame["scraped_at"], errors="coerce").min()
    completed_at = pd.to_datetime(frame["scraped_at"], errors="coerce").max()
    standard_candidates = frame[frame["price_tier"].eq("standard")].copy()
    duplicate_standard_slots = int(
        standard_candidates.duplicated(
            "schedule_slot_key", keep="last"
        ).sum()
    )
    standard = (
        standard_candidates
        .sort_values(
            ["schedule_slot_key", "scraped_at", "session_id"],
            kind="stable",
        )
        .drop_duplicates("schedule_slot_key", keep="last")
        .copy()
    )
    standard_report = validate_batch(
        standard,
        enforce_full_coverage,
        allow_airline_subset=allow_airline_subset,
    )
    report = {
        **standard_report,
        "standard_rows": int(len(standard)),
        "filtered_nonstandard_rows": int(
            frame["price_tier"].ne("standard").sum()
        ),
        "duplicate_standard_slots_removed": duplicate_standard_slots,
    }
    if validation_context:
        report.update(validation_context)
    with connection() as conn:
        with conn.cursor() as cursor:
            if enforce_registry:
                cursor.execute(
                    """
                    SELECT route, airline
                    FROM skyfare_live.route_airline_registry
                    WHERE active
                    """
                )
                allowed = {(str(route), str(airline)) for route, airline in cursor}
                observed = {
                    (str(route), str(airline))
                    for route, airline in standard[
                        ["route", "airline"]
                    ].drop_duplicates().itertuples(index=False, name=None)
                }
                unknown = sorted(observed.difference(allowed))
                if allowed and unknown:
                    raise RuntimeError(
                        "Batch contains unsupported route-airline pairs: "
                        + ", ".join(f"{route}/{airline}" for route, airline in unknown)
                    )
            cursor.execute(
                """
                SELECT status, source_sha256
                FROM skyfare_live.collection_batches
                WHERE batch_id = %s
                """,
                (batch_id,),
            )
            existing = cursor.fetchone()
            if existing == ("READY", source_hash):
                return {"batch_id": batch_id, "status": "READY", **report}
            cursor.execute(
                """
                INSERT INTO skyfare_live.collection_batches (
                    batch_id, collection_era, session_date, session_label,
                    source_file, source_sha256, status, started_at, completed_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'STAGING', %s, %s, NOW())
                ON CONFLICT (batch_id) DO UPDATE SET
                    source_file = EXCLUDED.source_file,
                    source_sha256 = EXCLUDED.source_sha256,
                    status = 'STAGING',
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    promoted_at = NULL,
                    failure_reason = NULL,
                    updated_at = NOW()
                """,
                (
                    batch_id,
                    collection_era,
                    session_date,
                    session_label,
                    str(source_file.resolve()),
                    source_hash,
                    started_at,
                    completed_at,
                ),
            )
            cursor.execute(
                "DELETE FROM skyfare_live.standard_offers WHERE batch_id = %s",
                (batch_id,),
            )
            if not standard.empty:
                _copy_standard_frame(cursor, standard, collection_era)
            cursor.execute(
                """
                INSERT INTO skyfare_live.route_airline_registry (
                    route, airline, first_seen, last_seen
                )
                SELECT route, airline, MIN(session_date), MAX(session_date)
                FROM skyfare_live.standard_offers
                WHERE batch_id = %s
                GROUP BY route, airline
                ON CONFLICT (route, airline) DO UPDATE SET
                    first_seen = LEAST(
                        skyfare_live.route_airline_registry.first_seen,
                        EXCLUDED.first_seen
                    ),
                    last_seen = GREATEST(
                        skyfare_live.route_airline_registry.last_seen,
                        EXCLUDED.last_seen
                    )
                """,
                (batch_id,),
            )
            cursor.execute(
                """
                UPDATE skyfare_live.collection_batches
                SET status = 'READY',
                    promoted_at = NOW(),
                    standard_rows = %s,
                    route_count = %s,
                    airline_count = %s,
                    dud_count = %s,
                    route_airline_count = %s,
                    validation_report = %s::jsonb,
                    updated_at = NOW()
                WHERE batch_id = %s
                """,
                (
                    len(standard),
                    report["routes"],
                    report["airlines"],
                    len(report["duds"]),
                    report["route_airline_pairs"],
                    json.dumps(report),
                    batch_id,
                ),
            )
        conn.commit()
    return {"batch_id": batch_id, "status": "READY", **report}


def latest_ready_snapshot() -> dict[str, object] | None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT row_to_json(x) FROM skyfare_live.latest_ready_snapshot x"
            )
            row = cursor.fetchone()
    return None if row is None else row[0]
