"""EA01 schedule-slot identity continuity audit for new-build 110.

Snapshot: new-build 110 pilot, cutoff 2026-07-10.
Branch: model-free pre-fit gate EA01.
Input: immutable raw source files rebuilt by audit_common.
Output: CSV/JSON/PNG/Markdown under feature_selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skyfare.features.audit_common import BOOKING_WINDOWS, CUTOFF, ROOT, completed_batch_offers, input_manifest, load_new_build_raw, next_window_map


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_DIR = ROOT / "artifacts/feature_research/figures"
REPORT = ROOT / "artifacts/feature_research/reports/EA01_IDENTITY_CONTINUITY.txt"


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(100 * numerator / denominator) if denominator else float("nan")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_new_build_raw()
    offers = completed_batch_offers(data)

    era = (
        data.groupby("collection_era", observed=True)
        .agg(
            rows=("schedule_slot_id", "size"),
            first_date=("session_date", "min"),
            last_date=("session_date", "max"),
            fallback_rows=("is_schedule_fallback", "sum"),
            exact_duplicate_rows=("is_exact_duplicate", "sum"),
            schedule_slots=("schedule_slot_id", "nunique"),
        )
        .reset_index()
    )
    era["fallback_pct"] = 100 * era["fallback_rows"] / era["rows"]
    era["exact_duplicate_pct"] = 100 * era["exact_duplicate_rows"] / era["rows"]
    era.to_csv(TABLE_DIR / "ea01_identity_by_collection_era.csv", index=False)

    collision = (
        data.groupby(["session_key", "schedule_slot_id"], observed=True)
        .agg(
            rows=("schedule_slot_id", "size"),
            unique_prices=("price_vnd", "nunique"),
            unique_raw_flight_numbers=("flight_no_clean", "nunique"),
            first_scrape=("scraped_at", "min"),
            last_scrape=("scraped_at", "max"),
        )
        .reset_index()
    )
    collision = collision[collision["rows"].gt(1)].copy()
    collision.to_csv(TABLE_DIR / "ea01_batch_slot_collisions.csv", index=False)

    verified = data[data["verified_flight_no_proxy"].notna()].copy()
    verified_group = (
        verified.groupby(
            ["route", "airline", "flight_date", "verified_flight_no_proxy"],
            observed=True,
        )
        .agg(
            rows=("schedule_slot_id", "size"),
            departure_slots=("departure_hhmm", "nunique"),
            first_slot=("departure_hhmm", "min"),
            last_slot=("departure_hhmm", "max"),
            observed_duds=("days_until_departure", "nunique"),
        )
        .reset_index()
    )
    verified_group["schedule_shift_observed"] = verified_group["departure_slots"].gt(1)
    verified_group.to_csv(TABLE_DIR / "ea01_verified_flight_schedule_shifts.csv", index=False)

    merge_map = (
        verified.groupby(["session_key", "schedule_slot_id"], observed=True)
        .agg(verified_numbers=("verified_flight_no_proxy", "nunique"), rows=("schedule_slot_id", "size"))
        .reset_index()
    )
    merge_map = merge_map[merge_map["verified_numbers"].gt(1)].copy()
    merge_map.to_csv(TABLE_DIR / "ea01_verified_slot_merge_risk.csv", index=False)

    grid = offers[offers["days_until_departure"].isin(BOOKING_WINDOWS)].copy()
    grid["entity_band"] = pd.util.hash_pandas_object(
        grid[["schedule_slot_id", "session_label"]], index=False
    ).astype("uint64")
    sets = {
        int(dud): set(part["entity_band"].astype(str))
        for dud, part in grid.groupby("days_until_departure", observed=True)
    }
    transition_rows = []
    for current_dud, target_dud in next_window_map().items():
        current = sets.get(current_dud, set())
        target = sets.get(target_dud, set())
        exact = current.intersection(target)
        transition_rows.append(
            {
                "current_dud": current_dud,
                "target_dud": target_dud,
                "current_entities": len(current),
                "exact_retained_entities": len(exact),
                "exact_retention_pct": pct(len(exact), len(current)),
                "missing_exact_entities": len(current.difference(target)),
            }
        )
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(TABLE_DIR / "ea01_exact_next_identity_retention.csv", index=False)

    collision_rows = int(collision["rows"].sum()) if len(collision) else 0
    shifted_groups = int(verified_group["schedule_shift_observed"].sum())
    shifted_rows = int(
        verified_group.loc[verified_group["schedule_shift_observed"], "rows"].sum()
    )
    summary = {
        "status": "EA01_COMPLETE_MODEL_FREE",
        "cutoff": str(CUTOFF.date()),
        "timezone_contract": "Asia/Ho_Chi_Minh local wall time",
        "raw_rows": int(len(data)),
        "completed_batch_offers": int(len(offers)),
        "exact_duplicate_rows": int(data["is_exact_duplicate"].sum()),
        "batch_slot_collision_groups": int(len(collision)),
        "batch_slot_collision_rows": collision_rows,
        "verified_raw_number_rows": int(len(verified)),
        "schedule_fallback_rows": int(data["is_schedule_fallback"].sum()),
        "verified_groups_with_departure_shift": shifted_groups,
        "verified_rows_in_shifted_groups": shifted_rows,
        "verified_slot_merge_risk_groups": int(len(merge_map)),
        "identity_claim": "SCHEDULE_SLOT_PROXY_NOT_PHYSICAL_FLIGHT",
        "trip_physical_flight_continuity_identifiable": False,
        "input_manifest": input_manifest(),
    }
    (TABLE_DIR / "ea01_identity_continuity_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.barplot(data=era, x="collection_era", y="fallback_pct", ax=axes[0], color="#E69F00")
    axes[0].set_title("Trip rows expose schedule slots, not verified flight numbers")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Rows using airline-HHMM fallback (%)")
    axes[0].tick_params(axis="x", rotation=12)
    sns.barplot(
        data=transitions,
        x=transitions["current_dud"].astype(str) + "->" + transitions["target_dud"].astype(str),
        y="exact_retention_pct",
        ax=axes[1],
        color="#0072B2",
    )
    axes[1].set_title("Exact schedule-slot retention varies by booking transition")
    axes[1].set_xlabel("DUD transition")
    axes[1].set_ylabel("Current entities observed again (%)")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("EA01. Identity supports schedule-slot claims only", fontweight="bold")
    fig.tight_layout()
    figure_path = FIG_DIR / "EA01_identity_continuity.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = f"""# EA01 - Identity Continuity

> New-build 110 cutoff: `{CUTOFF.date()}`. Model-free. No 110/115 model outcome used.

## Decision

Identity contract remains **schedule-slot proxy**, not physical flight.

- Raw rows: **{len(data):,}**
- Completed AM/PM offer states: **{len(offers):,}**
- Fli/Trip.com rows using `airline-HHMM` fallback: **{int(data['is_schedule_fallback'].sum()):,}**
- Verified-number groups with observed departure-time shift: **{shifted_groups:,}**
- Verified slot groups mapping to multiple raw numbers in one batch: **{len(merge_map):,}**
- Batch-slot collision groups requiring deterministic latest-state handling: **{len(collision):,}**

## Interpretation

`schedule_slot_id = route + airline + flight_date + departure_HHMM` is usable for
strictly-prior persistence, but cannot prove physical-flight continuity. Trip
fallback values cannot measure carrier-flight schedule changes. Every persistence
and sequence claim must use `same-schedule`, carry `identity_quality`, and report
exact transition retention.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"[EA01 PASS] rows={len(data):,} offers={len(offers):,} shifts={shifted_groups:,}")
    print(f"[EA01 PASS] {figure_path}")


if __name__ == "__main__":
    main()
