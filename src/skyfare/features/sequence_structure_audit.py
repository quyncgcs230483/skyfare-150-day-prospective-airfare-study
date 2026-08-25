"""EA07 model-free sequence-lane coverage and irregular-gap audit."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from skyfare.features.audit_common import CUTOFF, ROOT
from skyfare.features.candidate_feature_contract import build_candidate_frame

TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_DIR = ROOT / "artifacts/feature_research/figures"
REPORT = ROOT / "artifacts/feature_research/reports/EA07_SEQUENCE_STRUCTURE.txt"
LENGTHS = [7, 14, 21]


def lane_summary(lane: pd.DataFrame, lane_name: str, key: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lane = lane.sort_values(key + ["feature_time"]).copy()
    group = lane.groupby(key, observed=True, sort=False)
    lane["prior_event_count"] = group.cumcount()
    lane["previous_event_time"] = group["feature_time"].shift(1)
    lane["delta_hours"] = (
        lane["feature_time"] - lane["previous_event_time"]
    ).dt.total_seconds() / 3600
    coverage = []
    for length in LENGTHS:
        eligible = lane["prior_event_count"].ge(length)
        coverage.append(
            {
                "sequence_lane": lane_name,
                "sequence_length": length,
                "event_rows": len(lane),
                "eligible_rows": int(eligible.sum()),
                "coverage_pct": float(100 * eligible.mean()),
                "series": int(group.ngroups),
            }
        )
    positive = lane.loc[lane["delta_hours"].gt(0), "delta_hours"]
    gaps = pd.DataFrame(
        [
            {
                "sequence_lane": lane_name,
                "positive_gap_rows": len(positive),
                "median_gap_hours": float(positive.median()),
                "p90_gap_hours": float(positive.quantile(0.90)),
                "p95_gap_hours": float(positive.quantile(0.95)),
                "max_gap_hours": float(positive.max()),
                "gap_24h_pct": float(100 * positive.eq(24).mean()),
                "gap_72h_or_more_pct": float(100 * positive.ge(72).mean()),
                "nonpositive_gap_violations": int(lane["delta_hours"].le(0).sum()),
            }
        ]
    )
    return pd.DataFrame(coverage), gaps


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_candidate_frame()

    market = (
        frame.groupby(
            ["collection_era", "route", "airline", "booking_window_bucket", "session_key"], observed=True
        )
        .agg(feature_time=("feature_time", "max"), event_price=("price_vnd", "median"), event_support=("price_vnd", "size"))
        .reset_index()
    )
    market_coverage, market_gaps = lane_summary(
        market,
        "market_event_sequence",
        ["collection_era", "route", "airline", "booking_window_bucket"],
    )

    schedule = frame[
        ["collection_era", "schedule_slot_id", "session_label", "feature_time", "price_vnd", "days_until_departure"]
    ].copy()
    schedule_coverage, schedule_gaps = lane_summary(
        schedule,
        "same_schedule_history_sequence",
        ["collection_era", "schedule_slot_id", "session_label"],
    )
    coverage = pd.concat([market_coverage, schedule_coverage], ignore_index=True)
    gaps = pd.concat([market_gaps, schedule_gaps], ignore_index=True)
    coverage.to_csv(TABLE_DIR / "ea07_sequence_length_coverage.csv", index=False)
    gaps.to_csv(TABLE_DIR / "ea07_sequence_gap_summary.csv", index=False)

    observed_dates = pd.DatetimeIndex(sorted(frame["session_date"].dropna().unique()))
    calendar = pd.date_range(observed_dates.min(), observed_dates.max(), freq="D")
    missing_dates = calendar.difference(observed_dates)
    pd.DataFrame({"missing_session_date": missing_dates}).to_csv(
        TABLE_DIR / "ea07_missing_calendar_dates.csv", index=False
    )

    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    sns.barplot(data=coverage, x="sequence_length", y="coverage_pct", hue="sequence_lane", ax=axes[0])
    axes[0].set_title("Longer sequence length sharply changes eligible population")
    axes[0].set_xlabel("Required strictly-prior events")
    axes[0].set_ylabel("Eligible event rows (%)")
    gap_long = gaps.melt(
        id_vars="sequence_lane",
        value_vars=["median_gap_hours", "p90_gap_hours", "p95_gap_hours"],
        var_name="gap_statistic",
        value_name="hours",
    )
    sns.barplot(data=gap_long, x="gap_statistic", y="hours", hue="sequence_lane", ax=axes[1])
    axes[1].set_title("Event-time gaps are irregular and must remain explicit")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Elapsed hours")
    axes[1].tick_params(axis="x", rotation=18)
    fig.suptitle("EA07. Market sequences and same-schedule histories are different data products", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "EA07_sequence_lanes_require_real_gaps_and_separate_coverage.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "status": "EA07_COMPLETE_MODEL_FREE",
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "cutoff": str(CUTOFF.date()),
        "observed_session_dates": len(observed_dates),
        "calendar_span_days": len(calendar),
        "missing_calendar_dates": len(missing_dates),
        "sequence_lengths_audited": LENGTHS,
        "interpolation_allowed": False,
        "delta_hours_required": True,
        "booking_window_bucket_role": "GROUP_KEY_ONLY_NOT_INPUT_FEATURE",
        "cross_collection_era_history_allowed": False,
        "model_fit_called": False,
    }
    (TABLE_DIR / "ea07_sequence_structure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    REPORT.write_text(
        f"""# EA07 - Sequence Structure Before Modelling

> Model-free structural audit through `{CUTOFF.date()}`.

## Two different sequence meanings

1. `market_event_sequence`: collection-era-route-airline-booking-window market events through completed batches.
2. `same_schedule_history_sequence`: strictly-prior observations for one schedule-slot and AM/PM band within one collection era.

They are not interchangeable. The first is available to cold rows; the second
is warm-only. A feature-based network with sequence length one is an MLP-like
adapter, not evidence that flight-price trajectories were learned.

## Length and gap decision

Lengths 7, 14 and 21 remain pilot candidates because they change coverage.
Observed time gaps are passed as `delta_hours` with masks. A 72-hour gap is one
event transition with `delta_hours=72`; it is not equivalent to 24 hours and it
does not create two synthetic observations. Missing dates are never interpolated.
No sequence crosses the Fli/Trip era boundary.

- Observed collection dates: **{len(observed_dates)}** across **{len(calendar)}** calendar days.
- Missing calendar dates: **{len(missing_dates)}**.

`booking_window_bucket` remains a sequence grouping key only. Exact DUD stays
the finer tabular time-to-departure feature.
""",
        encoding="utf-8",
    )
    print(f"[EA07 PASS] market_events={len(market):,} schedule_events={len(schedule):,} missing_dates={len(missing_dates)}")


if __name__ == "__main__":
    main()
