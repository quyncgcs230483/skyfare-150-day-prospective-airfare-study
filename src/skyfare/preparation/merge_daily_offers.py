"""Deterministically rebuild Trip.com daily aggregates.

Old behavior appended only dates absent from ``trip_full.csv``. If a daily file
was merged while still partial, later rows for that same date were ignored
forever. This version always rebuilds from daily files through a declared
cutoff, validates latest-day completeness, then atomically replaces all three
aggregate outputs.

Examples:
    python -m skyfare.preparation.merge_daily_offers --check-only
    python -m skyfare.preparation.merge_daily_offers --through-date 2026-07-15
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.preparation.temporal_sessions import collection_session_labels  # noqa: E402

LAYOUT = DataLayout.resolve()
DEFAULT_TRIP_DIR = LAYOUT.raw_trip_com
DEFAULT_OUTPUT = LAYOUT.standardised / "trip_com_all_offers.csv"
DATE_START = pd.Timestamp("2026-05-15")
BOOKING_WINDOWS = {1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60}
EXPECTED_ROUTES = 20
EXPECTED_AIRLINES = 5

EFFECTIVE_CEILING = {
    "SGN-HAN": 5_000_000,
    "HAN-SGN": 5_000_000,
    "SGN-DAD": 3_800_000,
    "DAD-SGN": 3_800_000,
    "HAN-DAD": 4_500_000,
    "DAD-HAN": 4_500_000,
    "SGN-CXR": 2_800_000,
    "CXR-SGN": 2_800_000,
    "SGN-PQC": 2_800_000,
    "PQC-SGN": 2_800_000,
    "HAN-PQC": 5_800_000,
    "PQC-HAN": 5_800_000,
    "DAD-PQC": 3_800_000,
    "PQC-DAD": 3_800_000,
    "HAN-CXR": 5_000_000,
    "CXR-HAN": 5_000_000,
    "SGN-HPH": 3_500_000,
    "HPH-SGN": 3_500_000,
    "HAN-VCA": 5_500_000,
    "VCA-HAN": 5_500_000,
}


def assign_price_tier(frame: pd.DataFrame) -> pd.Series:
    """Vectorized equivalent of legacy row-wise fare-tier assignment."""
    price = pd.to_numeric(frame["price_vnd"], errors="coerce")
    ceiling = frame["route"].map(EFFECTIVE_CEILING)
    values = np.select(
        [
            price.isna(),
            ceiling.isna(),
            price.le(ceiling),
            price.le(10_000_000),
        ],
        ["soldout", "unknown", "standard", "high"],
        default="extreme",
    )
    return pd.Series(values, index=frame.index, dtype="object")


def discover_daily_files(
    trip_dir: Path, through_date: pd.Timestamp | None
) -> tuple[list[Path], pd.Timestamp]:
    date_map: dict[pd.Timestamp, Path] = {}
    for path in sorted(trip_dir.glob("????-??-??.csv")):
        try:
            day = pd.Timestamp(path.stem).normalize()
        except ValueError:
            continue
        if day >= DATE_START:
            date_map[day] = path
    if not date_map:
        raise FileNotFoundError(f"No daily Trip files found in {trip_dir}")

    cutoff = through_date.normalize() if through_date is not None else max(date_map)
    if cutoff not in date_map:
        raise FileNotFoundError(f"Missing cutoff file: {cutoff.date()}.csv")
    expected_dates = pd.date_range(DATE_START, cutoff, freq="D")
    missing = [str(day.date()) for day in expected_dates if day not in date_map]
    if missing:
        raise FileNotFoundError(f"Missing daily Trip dates: {missing}")
    return [date_map[day] for day in expected_dates], cutoff


def load_daily_file(path: Path, expected_columns: list[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if expected_columns is not None and frame.columns.tolist() != expected_columns:
        raise AssertionError(f"Schema drift in {path.name}")
    scraped_at = pd.to_datetime(frame["scraped_at"], format="mixed", errors="coerce")
    observed_dates = set(scraped_at.dt.strftime("%Y-%m-%d").dropna())
    if observed_dates != {path.stem}:
        raise AssertionError(
            f"{path.name}: scraped_at dates {sorted(observed_dates)}"
        )
    return frame


def latest_day_coverage(frame: pd.DataFrame, date_label: str) -> pd.DataFrame:
    """Require full AM and PM route/airline/DUD coverage before any write."""
    session_id = pd.to_datetime(frame["session_id"], format="mixed", errors="coerce")
    if session_id.isna().any():
        raise AssertionError(f"{date_label}: invalid session_id values")
    audit = frame.copy()
    audit["session_band"] = collection_session_labels(session_id)
    coverage = (
        audit.groupby("session_band", as_index=False)
        .agg(
            rows=("route", "size"),
            routes=("route", "nunique"),
            airlines=("airline", "nunique"),
            dud_count=("days_until_departure", "nunique"),
        )
        .sort_values("session_band")
    )
    observed_bands = set(coverage["session_band"])
    if observed_bands != {"AM", "PM"}:
        raise RuntimeError(
            f"{date_label}: incomplete latest day; bands={sorted(observed_bands)}"
        )
    for band in ["AM", "PM"]:
        block = audit.loc[audit["session_band"].eq(band)]
        duds = set(
            pd.to_numeric(block["days_until_departure"], errors="coerce")
            .dropna()
            .astype(int)
        )
        routes = block["route"].nunique()
        airlines = block["airline"].nunique()
        failures: list[str] = []
        if duds != BOOKING_WINDOWS:
            failures.append(
                f"DUD={sorted(duds)} (expected {sorted(BOOKING_WINDOWS)})"
            )
        if routes != EXPECTED_ROUTES:
            failures.append(f"routes={routes} (expected {EXPECTED_ROUTES})")
        if airlines != EXPECTED_AIRLINES:
            failures.append(
                f"airlines={airlines} (expected {EXPECTED_AIRLINES})"
            )
        if failures:
            raise RuntimeError(
                f"{date_label} {band}: incomplete latest day; " + "; ".join(failures)
            )
    return coverage


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def build_aggregates(
    paths: list[Path], cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for path in paths:
        frame = load_daily_file(path, expected_columns)
        if expected_columns is None:
            expected_columns = frame.columns.tolist()
        frames.append(frame)
        print(f"  [READ] {path.name}: {len(frame):,} rows")

    latest_coverage = latest_day_coverage(frames[-1], str(cutoff.date()))
    print("\n[LATEST-DAY PASS]")
    print(latest_coverage.to_string(index=False))

    combined = pd.concat(frames, ignore_index=True)
    raw_rows = len(combined)
    exact_duplicates = int(combined.duplicated().sum())
    combined = combined.drop_duplicates().copy()
    combined["price_tier"] = assign_price_tier(combined)
    sort_columns = [
        "scraped_at",
        "session_id",
        "route",
        "flight_date",
        "days_until_departure",
        "airline",
        "flight_no",
        "departure_time",
        "price_vnd",
    ]
    combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(
        drop=True
    )
    standard = combined.loc[combined["price_tier"].eq("standard")].copy()
    nonstandard = combined.loc[
        combined["price_tier"].isin(["high", "extreme", "unknown"])
    ].copy()
    stats = {
        "daily_files": len(paths),
        "raw_rows": raw_rows,
        "exact_duplicates_removed": exact_duplicates,
        "all_rows": len(combined),
        "standard_rows": len(standard),
        "nonstandard_rows": len(nonstandard),
        "soldout_rows": int(combined["price_tier"].eq("soldout").sum()),
    }
    return combined, standard, nonstandard, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-dir", type=Path, default=DEFAULT_TRIP_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the full Trip aggregate.",
    )
    parser.add_argument(
        "--through-date",
        help="Inclusive YYYY-MM-DD cutoff. Default: latest daily file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and rebuild in memory without writing aggregate files.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    started_at = datetime.now()
    trip_dir = args.trip_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    through_date = (
        pd.Timestamp(args.through_date).normalize() if args.through_date else None
    )
    paths, cutoff = discover_daily_files(trip_dir, through_date)

    print("=" * 72)
    print("TRIP.COM DAILY REBUILD - IDEMPOTENT + ATOMIC")
    print("=" * 72)
    print(f"Source       : {trip_dir}")
    print(f"Date range   : {DATE_START.date()} -> {cutoff.date()}")
    print(f"Daily files  : {len(paths)}")
    print(f"Output       : {output}")
    print(f"Mode         : {'CHECK ONLY' if args.check_only else 'WRITE'}\n")

    combined, standard, nonstandard, stats = build_aggregates(paths, cutoff)
    print("\n[REBUILD SUMMARY]")
    for key, value in stats.items():
        print(f"  {key:<26}: {value:,}")

    if args.check_only:
        print("\n[CHECK PASS] No files written.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    standard_path = output.with_name("trip_standard_offers.csv")
    nonstandard_path = output.with_name("trip_nonstandard_offers.csv")
    atomic_write_csv(combined, output)
    atomic_write_csv(standard, standard_path)
    atomic_write_csv(nonstandard, nonstandard_path)
    elapsed = (datetime.now() - started_at).total_seconds()
    print("\n[WRITE PASS] Atomic replacement complete")
    print(f"  {output}")
    print(f"  {standard_path}")
    print(f"  {nonstandard_path}")
    print(f"  duration_seconds={elapsed:.1f}")


if __name__ == "__main__":
    run()
