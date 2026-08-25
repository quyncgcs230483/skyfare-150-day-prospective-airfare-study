#!/usr/bin/env python3
"""Create, populate and audit SkyFare live PostgreSQL storage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.preparation.merge_daily_offers import assign_price_tier
from skyfare.serving.live_store import (
    apply_schema,
    connection,
    ingest_batch,
    prepare_rows,
    sha256,
    validate_batch,
)

LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
FLI_STANDARD = LAYOUT.standardised / "fli_standard_offers.csv"
TRIP_STANDARD = LAYOUT.standardised / "trip_com_standard_offers.csv"


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _ingest_prepared(
    prepared: pd.DataFrame,
    collection_era: str,
    source: Path,
    enforce_full_coverage: bool,
    enforce_registry: bool = False,
    allow_airline_subset: bool = False,
    validation_context: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    source_hash = sha256(source)
    reports = []
    for _, batch in prepared.groupby("batch_id", sort=True, observed=True):
        reports.append(
            ingest_batch(
                batch.copy(),
                collection_era=collection_era,
                source_file=source,
                source_hash=source_hash,
                enforce_full_coverage=enforce_full_coverage,
                enforce_registry=enforce_registry,
                allow_airline_subset=allow_airline_subset,
                validation_context=validation_context,
            )
        )
        print(
            "[READY]"
            f" {reports[-1]['batch_id']}"
            f" standard={reports[-1]['standard_rows']:,}",
            flush=True,
        )
    return reports


def _verify_am_collection_log(path: Path) -> dict[str, object]:
    log_path = (
        ROOT / "data" / "raw" / "collection_issues" / f"trip_all_{path.stem}.log"
    )
    if not log_path.is_file():
        raise RuntimeError(f"AM_ONLY requires collector log: {log_path}")
    content = log_path.read_text(encoding="utf-8", errors="replace")
    start = re.search(r"START\s+queries=(\d+)", content)
    end = re.search(r"END\s+rows=(\d+)\s+skipped=(\d+)/(\d+)", content)
    if not start or not end:
        raise RuntimeError("AM_ONLY collector log lacks START/END evidence")
    started_queries = int(start.group(1))
    rows, skipped, completed_queries = map(int, end.groups())
    if started_queries != 220 or completed_queries != 220 or skipped != 0:
        raise RuntimeError(
            "AM_ONLY collector incomplete: "
            f"started={started_queries}; completed={completed_queries}; "
            f"skipped={skipped}"
        )
    return {
        "collector_completion_verified": True,
        "collector_expected_queries": 220,
        "collector_completed_queries": completed_queries,
        "collector_skipped_queries": skipped,
        "collector_reported_rows": rows,
        "collector_log": str(log_path.resolve()),
        "collector_log_sha256": sha256(log_path),
    }


def bootstrap_standard(cutoff: str | None) -> dict[str, object]:
    resolved = None if cutoff is None else pd.Timestamp(cutoff).normalize()
    sources = [
        ("FLI_LIBRARY_ERA", FLI_STANDARD),
        ("TRIP_COM_BROWSER_ERA", TRIP_STANDARD),
    ]
    reports: list[dict[str, object]] = []
    for era, path in sources:
        raw = _read(path)
        prepared = prepare_rows(raw, era, path, "standard")
        if resolved is not None:
            prepared = prepared[
                prepared["session_date"].dt.normalize().le(resolved)
            ].copy()
        reports.extend(
            _ingest_prepared(
                prepared,
                collection_era=era,
                source=path,
                enforce_full_coverage=False,
            )
        )
    result = {
        "status": "BOOTSTRAP_PASS",
        "batches": len(reports),
        "standard_rows": int(
            sum(int(item["standard_rows"]) for item in reports)
        ),
    }
    print(json.dumps(result, indent=2))
    return result


def ingest_day(path: Path, allow_am_only: bool = False) -> dict[str, object]:
    raw = _read(path)
    prepared = prepare_rows(
        raw,
        collection_era="TRIP_COM_BROWSER_ERA",
        source_file=path,
        price_tier=assign_price_tier(raw),
    )
    if allow_am_only:
        prepared = prepared[prepared["session_label"].eq("AM")].copy()
    labels = {
        str(batch_id).rsplit("|", 1)[-1]
        for batch_id in prepared["batch_id"].drop_duplicates()
    }
    expected_labels = {"AM"} if allow_am_only else {"AM", "PM"}
    if labels != expected_labels:
        raise RuntimeError(f"Daily file lacks AM/PM batches: {sorted(labels)}")
    validation_context = (
        _verify_am_collection_log(path) if allow_am_only else None
    )
    for _, batch in prepared.groupby("batch_id", sort=True, observed=True):
        standard = (
            batch[batch["price_tier"].eq("standard")]
            .sort_values(
                ["schedule_slot_key", "scraped_at", "session_id"],
                kind="stable",
            )
            .drop_duplicates("schedule_slot_key", keep="last")
        )
        validate_batch(
            standard,
            enforce_full_coverage=True,
            allow_airline_subset=allow_am_only,
        )
    reports = _ingest_prepared(
        prepared,
        collection_era="TRIP_COM_BROWSER_ERA",
        source=path,
        enforce_full_coverage=True,
        enforce_registry=True,
        allow_airline_subset=allow_am_only,
        validation_context=validation_context,
    )
    result = {
        "status": "DAY_INGEST_PASS",
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "batches": reports,
    }
    print(json.dumps(result, indent=2))
    return result


def status() -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'READY') AS ready_batches,
                    COUNT(*) FILTER (WHERE status = 'FAILED') AS failed_batches,
                    COUNT(*) FILTER (WHERE status = 'STAGING') AS staging_batches,
                    MIN(session_date) FILTER (WHERE status = 'READY'),
                    MAX(session_date) FILTER (WHERE status = 'READY'),
                    SUM(standard_rows) FILTER (WHERE status = 'READY')
                FROM skyfare_live.collection_batches
                """
            )
            summary = cursor.fetchone()
            cursor.execute(
                """
                SELECT session_date, session_label, collection_era,
                       standard_rows, route_count, airline_count, dud_count,
                       route_airline_count, status
                FROM skyfare_live.collection_batches
                ORDER BY session_date DESC,
                         CASE session_label WHEN 'PM' THEN 1 ELSE 0 END DESC
                LIMIT 8
                """
            )
            latest = cursor.fetchall()
            cursor.execute(
                "SELECT COUNT(*) FROM skyfare_live.route_airline_registry "
                "WHERE active"
            )
            registry = cursor.fetchone()[0]
            cursor.execute(
                "SELECT row_to_json(x) "
                "FROM skyfare_live.latest_ready_snapshot x"
            )
            snapshot = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM skyfare_live.inference_audit")
            inference_audits = cursor.fetchone()[0]
    print(
        json.dumps(
            {
                "ready_batches": summary[0],
                "failed_batches": summary[1],
                "staging_batches": summary[2],
                "first_date": summary[3],
                "latest_date": summary[4],
                "standard_rows": summary[5],
                "active_route_airline_pairs": registry,
                "latest_batches": latest,
                "latest_snapshot": None if snapshot is None else snapshot[0],
                "inference_audits": inference_audits,
            },
            indent=2,
            default=str,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-schema")
    bootstrap = commands.add_parser("bootstrap-standard")
    bootstrap.add_argument("--through-date")
    ingest = commands.add_parser("ingest-day")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--allow-am-only", action="store_true")
    commands.add_parser("status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init-schema":
        apply_schema()
        print("[SCHEMA PASS] SKYFARE_LIVE_POSTGRES_V4_STANDARD_ROWS_ONLY")
    elif args.command == "bootstrap-standard":
        bootstrap_standard(args.through_date)
    elif args.command == "ingest-day":
        ingest_day(
            args.path.expanduser().resolve(),
            allow_am_only=args.allow_am_only,
        )
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
