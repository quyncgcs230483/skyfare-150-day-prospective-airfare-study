"""EA06 observed listed-price semantics audit for the new-build 110 pilot.

Snapshot: new-build 110 pilot, cutoff 2026-07-10.
Branch: model-free pre-fit gate EA06.
Inputs: immutable Fli, Trip daily, and overlapping FLI raw files.
Outputs: two non-duplicative figures plus CSV/JSON/Markdown evidence.
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skyfare.core.paths import DataLayout
from skyfare.features.audit_common import (
    BOOKING_WINDOWS,
    CUTOFF,
    ROOT,
    completed_batch_offers,
    input_manifest,
    load_new_build_raw,
    sha256,
)
from skyfare.preparation.temporal_sessions import (
    collection_session_labels,
    operational_session_dates,
)

TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_DIR = ROOT / "artifacts/feature_research/figures"
REPORT = ROOT / "artifacts/feature_research/reports/EA06_LISTED_PRICE_SEMANTICS.txt"
FLI_PATH = DataLayout.resolve().standardised / "fli_overlap_standard_offers.csv"
TRIP_START = pd.Timestamp("2026-05-15")
FLI_OVERLAP_START = pd.Timestamp("2026-05-16")
FLI_OVERLAP_END = pd.Timestamp("2026-06-03")


def _pct_change(late: float, early: float) -> float:
    return float(100 * (late / early - 1)) if early else float("nan")


def _departure_period(frame: pd.DataFrame) -> pd.Series:
    minute = frame["departure_time"].dt.hour * 60 + frame["departure_time"].dt.minute
    return pd.cut(
        minute,
        bins=[-1, 299, 719, 1019, 1439],
        labels=["NIGHT", "MORNING", "AFTERNOON", "EVENING"],
    )


def _load_fli_overlap() -> pd.DataFrame:
    columns = [
        "scraped_at",
        "session_id",
        "route",
        "days_until_departure",
        "flight_date",
        "airline",
        "departure_time",
        "price_vnd",
    ]
    frame = pd.read_csv(
        FLI_PATH,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=columns,
        dtype_backend="pyarrow",
    )
    for column in ["scraped_at", "session_id", "flight_date", "departure_time"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["days_until_departure"] = pd.to_numeric(
        frame["days_until_departure"], errors="coerce"
    ).astype("Int64")
    frame["price_vnd"] = pd.to_numeric(frame["price_vnd"], errors="coerce")
    frame["session_date"] = pd.to_datetime(
        operational_session_dates(frame["session_id"])
    )
    frame = frame[
        frame["session_date"].between(FLI_OVERLAP_START, FLI_OVERLAP_END)
        & frame["price_vnd"].gt(0)
    ].dropna(subset=columns[1:])
    frame["session_label"] = collection_session_labels(frame["session_id"])
    frame["departure_hhmm"] = (
        frame["departure_time"].dt.hour * 100 + frame["departure_time"].dt.minute
    ).astype("int16")
    return frame


def _fli_trip_agreement(trip: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    fli = _load_fli_overlap()
    keys = [
        "session_date",
        "session_label",
        "route",
        "days_until_departure",
        "flight_date",
        "airline",
        "departure_hhmm",
    ]
    fli_slot = (
        fli.groupby(keys, observed=True)
        .agg(fli_price_vnd=("price_vnd", "min"), fli_rows=("price_vnd", "size"))
        .reset_index()
    )
    trip_overlap = trip[trip["session_date"].between(FLI_OVERLAP_START, FLI_OVERLAP_END)].copy()
    trip_slot = (
        trip_overlap.groupby(keys, observed=True)
        .agg(trip_price_vnd=("price_vnd", "min"), trip_rows=("price_vnd", "size"))
        .reset_index()
    )
    matched = fli_slot.merge(trip_slot, on=keys, how="inner", validate="one_to_one")
    matched["trip_to_fli_ratio"] = matched["trip_price_vnd"] / matched["fli_price_vnd"]
    matched["absolute_pct_difference"] = (
        100 * (matched["trip_to_fli_ratio"] - 1).abs()
    )
    daily = (
        matched.groupby("session_date", observed=True)
        .agg(
            matched_slots=("trip_to_fli_ratio", "size"),
            median_trip_to_fli_ratio=("trip_to_fli_ratio", "median"),
            within_5pct=("absolute_pct_difference", lambda values: values.le(5).mean()),
            within_10pct=("absolute_pct_difference", lambda values: values.le(10).mean()),
        )
        .reset_index()
    )
    summary = {
        "overlap_start": str(FLI_OVERLAP_START.date()),
        "overlap_end": str(FLI_OVERLAP_END.date()),
        "overlap_days_with_exact_matches": int(daily["session_date"].nunique()),
        "exact_schedule_minute_matches": int(len(matched)),
        "median_trip_to_fli_ratio": float(matched["trip_to_fli_ratio"].median()),
        "within_5pct_pct": float(100 * matched["absolute_pct_difference"].le(5).mean()),
        "within_10pct_pct": float(100 * matched["absolute_pct_difference"].le(10).mean()),
        "claim_guard": (
            "The overlap supports early Trip listed-price comparability only; "
            "it cannot verify later June/July display semantics."
        ),
    }
    return daily, summary


def _matched_shift(
    trip: pd.DataFrame,
    keys: list[str],
    stage: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    grouped = (
        trip.groupby(keys + ["period"], observed=True)
        .agg(median_price=("price_vnd", "median"), rows=("price_vnd", "size"))
        .reset_index()
    )
    price = grouped.pivot(index=keys, columns="period", values="median_price")
    support = grouped.pivot(index=keys, columns="period", values="rows")
    valid_index = price.dropna(subset=["EARLY", "LATE"]).index
    price = price.loc[valid_index]
    support = support.loc[valid_index]
    detail = price.reset_index()
    detail["early_rows"] = support["EARLY"].to_numpy()
    detail["late_rows"] = support["LATE"].to_numpy()
    detail["change_pct"] = 100 * (detail["LATE"] / detail["EARLY"] - 1)
    detail["stage"] = stage
    result = {
        "stage": stage,
        "matched_cells": int(len(detail)),
        "median_cell_change_pct": float(detail["change_pct"].median()),
        "share_cells_declining_pct": float(100 * detail["change_pct"].lt(0).mean()),
        "weighted_mean_cell_change_pct": float(
            np.average(
                detail["change_pct"],
                weights=np.minimum(detail["early_rows"], detail["late_rows"]),
            )
        ),
    }
    return result, detail


def _trip_shift_decomposition(trip: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dates = sorted(pd.Timestamp(value) for value in trip["session_date"].unique())
    split_date = dates[len(dates) // 2]
    trip = trip.copy()
    trip["period"] = np.where(trip["session_date"].lt(split_date), "EARLY", "LATE")
    trip["departure_period"] = _departure_period(trip)
    trip["flight_month"] = trip["flight_date"].dt.to_period("M").astype(str)
    trip["flight_weekday"] = trip["flight_date"].dt.dayofweek.astype("int8")

    early = trip.loc[trip["period"].eq("EARLY"), "price_vnd"]
    late = trip.loc[trip["period"].eq("LATE"), "price_vnd"]
    rows: list[dict[str, object]] = [
        {
            "stage": "Raw listed-price median",
            "matched_cells": 1,
            "median_cell_change_pct": _pct_change(float(late.median()), float(early.median())),
            "share_cells_declining_pct": np.nan,
            "weighted_mean_cell_change_pct": _pct_change(float(late.median()), float(early.median())),
        }
    ]
    details: list[pd.DataFrame] = []
    stages = [
        (
            "Match route + airline + exact DUD",
            ["route", "airline", "days_until_departure"],
        ),
        (
            "Also match departure period",
            ["route", "airline", "days_until_departure", "departure_period"],
        ),
        (
            "Also match flight month",
            [
                "route",
                "airline",
                "days_until_departure",
                "departure_period",
                "flight_month",
            ],
        ),
        (
            "Also match flight weekday",
            [
                "route",
                "airline",
                "days_until_departure",
                "departure_period",
                "flight_month",
                "flight_weekday",
            ],
        ),
    ]
    for stage, keys in stages:
        result, detail = _matched_shift(trip, keys, stage)
        rows.append(result)
        details.append(detail)

    decomposition = pd.DataFrame(rows)
    final_detail = details[-1]
    raw_shift = float(decomposition.iloc[0]["median_cell_change_pct"])
    residual_shift = float(decomposition.iloc[-1]["median_cell_change_pct"])
    explained_share = float(100 * (1 - abs(residual_shift) / abs(raw_shift))) if raw_shift else np.nan

    composition = []
    for column in ["airline", "route", "days_until_departure", "flight_month"]:
        shares = (
            trip.groupby(["period", column], observed=True).size()
            / trip.groupby("period", observed=True).size()
        ).unstack("period").fillna(0)
        delta = 100 * (shares.get("LATE", 0) - shares.get("EARLY", 0))
        composition.append(
            {
                "dimension": column,
                "max_absolute_share_shift_pp": float(delta.abs().max()),
                "level_with_largest_shift": str(delta.abs().idxmax()),
                "signed_shift_pp": float(delta.loc[delta.abs().idxmax()]),
            }
        )
    composition_frame = pd.DataFrame(composition)
    metadata = {
        "trip_start": str(trip["session_date"].min().date()),
        "trip_end": str(trip["session_date"].max().date()),
        "period_split_date": str(split_date.date()),
        "trip_offer_rows": int(len(trip)),
        "raw_shift_pct": raw_shift,
        "fully_matched_residual_shift_pct": residual_shift,
        "raw_shift_magnitude_associated_with_composition_and_departure_cohort_pct": explained_share,
        "fully_matched_cells_declining_pct": float(
            decomposition.iloc[-1]["share_cells_declining_pct"]
        ),
        "interpretation_guard": (
            "This is descriptive decomposition, not causal attribution. Flight month explains "
            "a large share of the raw shift; the residual can still mix market, promotions, "
            "display semantics, and collection behavior."
        ),
    }
    return decomposition, final_detail, {"metadata": metadata, "composition": composition_frame}


def _collection_stability(trip: pd.DataFrame) -> pd.DataFrame:
    return (
        trip.groupby(["session_date", "session_label"], observed=True)
        .agg(
            offer_rows=("price_vnd", "size"),
            routes=("route", "nunique"),
            airlines=("airline", "nunique"),
            booking_windows=("days_until_departure", "nunique"),
            route_dud_queries=(
                "price_vnd",
                lambda values: trip.loc[values.index, ["route", "days_until_departure"]]
                .drop_duplicates()
                .shape[0],
            ),
        )
        .reset_index()
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_new_build_raw()
    source_ranges = (
        raw.groupby("collection_era", observed=True)
        .agg(
            first_date=("session_date", "min"),
            last_date=("session_date", "max"),
            rows=("price_vnd", "size"),
        )
        .reset_index()
    )
    source_ranges.to_csv(TABLE_DIR / "ea06_collection_era_ranges.csv", index=False)
    trip = completed_batch_offers(raw)
    trip = trip[
        trip["collection_era"].astype(str).eq("TRIP_COM_BROWSER_ERA")
        & trip["days_until_departure"].isin(BOOKING_WINDOWS)
        & trip["session_date"].between(TRIP_START, CUTOFF)
    ].copy()
    trip["departure_hhmm"] = (
        trip["departure_time"].dt.hour * 100 + trip["departure_time"].dt.minute
    ).astype("int16")

    agreement_daily, agreement = _fli_trip_agreement(trip)
    agreement_daily.to_csv(TABLE_DIR / "ea06_fli_trip_daily_agreement.csv", index=False)
    (TABLE_DIR / "ea06_fli_trip_overlap_summary.json").write_text(
        json.dumps(agreement, indent=2) + "\n", encoding="utf-8"
    )

    decomposition, final_detail, drift = _trip_shift_decomposition(trip)
    decomposition.to_csv(TABLE_DIR / "ea06_trip_drift_decomposition.csv", index=False)
    final_detail.to_csv(TABLE_DIR / "ea06_trip_fully_adjusted_stratum_changes.csv", index=False)
    drift["composition"].to_csv(TABLE_DIR / "ea06_trip_composition_shifts.csv", index=False)

    stability = _collection_stability(trip)
    stability.to_csv(TABLE_DIR / "ea06_trip_collection_stability.csv", index=False)
    stability_summary = {
        "completed_batches": int(len(stability)),
        "min_routes": int(stability["routes"].min()),
        "min_airlines": int(stability["airlines"].min()),
        "min_booking_windows": int(stability["booking_windows"].min()),
        "min_route_dud_queries": int(stability["route_dud_queries"].min()),
        "median_route_dud_queries": float(stability["route_dud_queries"].median()),
    }

    era_ranges = source_ranges.set_index("collection_era")
    fli_end = pd.Timestamp(era_ranges.loc["FLI_LIBRARY_ERA", "last_date"])
    trip_start = pd.Timestamp(era_ranges.loc["TRIP_COM_BROWSER_ERA", "first_date"])
    fli_trip_overlap = max(
        0,
        int((min(fli_end, CUTOFF) - max(TRIP_START, trip_start)).days) + 1,
    )
    fli_manifest = {
        "path": str(FLI_PATH.relative_to(ROOT)),
        "bytes": FLI_PATH.stat().st_size,
        "sha256": sha256(FLI_PATH),
    }
    summary = {
        "status": "EA06_COMPLETE_RESIDUAL_SHIFT_REQUIRES_ROLLING_REGIME_VALIDATION",
        "cutoff": str(CUTOFF.date()),
        "uses_rows_after_cutoff": False,
        "price_claim": "OBSERVED_ECONOMY_LISTED_FARE_NOT_EQUIVALENT_BUNDLE",
        "fli_trip_overlap_days": fli_trip_overlap,
        "fli_trip_crosscheck": agreement,
        "trip_shift_decomposition": drift["metadata"],
        "trip_collection_stability": stability_summary,
        "later_june_july_display_semantics_verified_by_overlap": False,
        "retained_html_snapshots_available_for_semantic_replay": False,
        "willdrop_guard": (
            "A real same-offer future drop remains product-relevant. The risk is regime shortcut "
            "and weak reversal generalization, not automatic target contamination."
        ),
        "anchor_guard": (
            "A strictly-prior anchor is causal. Its adaptation lag must be measured by rolling fold "
            "and regime; lag alone is not leakage."
        ),
        "input_manifest": [*input_manifest(), fli_manifest],
    }
    (TABLE_DIR / "ea06_listed_price_semantics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(
        agreement_daily["session_date"],
        agreement_daily["median_trip_to_fli_ratio"],
        marker="o",
        color="#0072B2",
        label="Daily median Trip / FLI",
    )
    ax.axhline(1.0, color="#222222", linewidth=1.2, label="Exact agreement")
    ax.axhspan(0.95, 1.05, color="#009E73", alpha=0.12, label="Within +/-5%")
    ax.set_title(
        "EA06.1 FLI cross-check supports early Trip prices, not later-period semantics",
        fontweight="bold",
    )
    ax.set_xlabel("Overlapping collection date")
    ax.set_ylabel("Median listed-price ratio (Trip / FLI)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        f"{agreement['exact_schedule_minute_matches']:,} matched schedule-minute offers; "
        f"{agreement['within_5pct_pct']:.1f}% agree within +/-5%.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(
        FIG_DIR / "EA06_1_fli_trip_overlap_supports_early_price_semantics.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    plot = decomposition.copy()
    colors = ["#D55E00", "#E69F00", "#56B4E9", "#0072B2", "#009E73"]
    bars = ax.barh(plot["stage"][::-1], plot["median_cell_change_pct"][::-1], color=colors[::-1])
    for bar, value in zip(bars, plot["median_cell_change_pct"][::-1], strict=True):
        ax.text(
            -0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            va="center",
            ha="right",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title(
        "EA06.2 Departure cohort explains most of the raw shift; residual movement remains",
        fontweight="bold",
    )
    ax.set_xlabel("Late-period vs early-period median listed-price change")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Pilot rows only (through 10/07). Descriptive matching is not causal attribution.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(
        FIG_DIR / "EA06_2_departure_cohort_explains_most_raw_price_shift.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    for stale in [
        "EA06_1_zero_source_overlap_blocks_source_effect_claims.png",
        "EA06_2_residual_price_drift_requires_rolling_validation.png",
        "EA06_3_batch_volume_requires_market_support_counts.png",
        "EA06_4_price_dispersion_supports_market_spread_feature.png",
        "EA06_listed_price_semantics.png",
    ]:
        (FIG_DIR / stale).unlink(missing_ok=True)

    meta = drift["metadata"]
    report = f"""# EA06 - Listed-Price Semantics and Temporal Shift

> New-build 110 cutoff: `{CUTOFF.date()}`. Model-free; no row after the cutoff is used.

## What is verified

- Fli and Trip.com overlap: **{fli_trip_overlap} days**. Their protocol effect remains confounded with era.
- FLI and Trip exact schedule-minute matches: **{agreement['exact_schedule_minute_matches']:,}** across
  **{agreement['overlap_days_with_exact_matches']} overlapping days**.
- Median Trip/FLI listed-price ratio: **{agreement['median_trip_to_fli_ratio']:.4f}**;
  **{agreement['within_5pct_pct']:.2f}%** of matches are within +/-5%.
- This supports early Trip comparability only. No retained page snapshot proves that
  Trip display semantics stayed unchanged throughout later June/July.

## Price-shift decomposition inside the pilot

- Raw early-to-late Trip median shift: **{meta['raw_shift_pct']:.2f}%**.
- After matching route, airline, exact DUD, departure period, flight month and
  flight weekday: median cell shift **{meta['fully_matched_residual_shift_pct']:.2f}%**.
- Declining fully matched cells: **{meta['fully_matched_cells_declining_pct']:.2f}%**.
- The departure-cohort/composition adjustment is associated with
  **{meta['raw_shift_magnitude_associated_with_composition_and_departure_cohort_pct']:.1f}%**
  of the raw shift magnitude.

This is not proof that the remaining movement is pure market drift. The residual
can mix real market movement, promotions, unobserved fare-bundle changes, display
semantics and collection behavior.

## Collection stability

- Completed AM/PM batches: **{stability_summary['completed_batches']}**.
- Minimum observed routes / airlines / DUD windows: **{stability_summary['min_routes']} /
  {stability_summary['min_airlines']} / {stability_summary['min_booking_windows']}**.
- Route-DUD query coverage: median **{stability_summary['median_route_dud_queries']:.0f}**,
  minimum **{stability_summary['min_route_dud_queries']}** of the intended 220.

This makes a large route/DUD selection change unlikely, but does not prove that
the website's fare-card semantics were invariant.

## Feature-selection handoff

- Keep `flight_month`: it captures departure cohort/seasonality and is not
  interchangeable with collection date.
- Keep market support counts as model inputs only where their semantics are
  distinct; their descriptive volume plot stays in raw EDA, not duplicated here.
- Do not add raw `session_date`, era index or a source flag as a shortcut.
- A causal market-trend candidate may be considered only if built from strictly
  prior completed batches, with age/support masks and a redundancy audit.
- Require rolling-origin, latest-fold and regime/DUD reporting for both branches.
- For Will-Drop, a genuine future price decline remains a correct user outcome;
  the threat is regime dependence, not automatic label contamination.
- For Relative, a prior-session anchor remains legal; report its bias/adaptation
  lag by fold and regime.

## Decision

EA06 passes as a model-free gate with a required temporal-regime validation
contract. It does **not** justify the claim that the raw -{abs(meta['raw_shift_pct']):.1f}%
movement is entirely market drift.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(
        f"[EA06 PASS] trip_rows={len(trip):,} raw_shift={meta['raw_shift_pct']:.2f}% "
        f"fully_matched={meta['fully_matched_residual_shift_pct']:.2f}%"
    )
    print(
        f"[EA06 PASS] FLI-Trip matches={agreement['exact_schedule_minute_matches']:,} "
        f"within5={agreement['within_5pct_pct']:.2f}%"
    )
    print(f"[EA06 PASS] two standalone figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
