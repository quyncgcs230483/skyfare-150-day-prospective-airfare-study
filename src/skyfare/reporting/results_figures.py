#!/usr/bin/env python3
"""Build publication-ready EDA and prospective evaluation figures."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from skyfare.core.sources import CollectionSource, normalize_source_columns

COLORS = {
    "ink": "#1F2933",
    "muted": "#667681",
    "line": "#70828C",
    "grid": "#D9E2E7",
    "teal": "#3B8D8A",
    "teal_light": "#DCEFED",
    "blue": "#5F91BC",
    "blue_light": "#E6EFF8",
    "orange": "#D88B37",
    "orange_light": "#FCEBD9",
    "gray": "#9AA6AD",
    "gray_light": "#EEF2F4",
    "navy": "#334E68",
    "red": "#B95C58",
}

AIRLINE_COLORS = {
    "VN": COLORS["teal"],
    "VJ": COLORS["blue"],
    "9G": COLORS["orange"],
    "VU": COLORS["navy"],
    "QH": "#A66F86",
}

MIN_AIRLINE_DAY_SUPPORT = 100
BOOKING_WINDOWS = [60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1]
FLI_LIBRARY_ERA = CollectionSource.FLI.value
TRIP_COM_BROWSER_ERA = CollectionSource.TRIP_COM.value
SOURCE_LABELS = {
    FLI_LIBRARY_ERA: "Fli library",
    TRIP_COM_BROWSER_ERA: "Trip.com",
}
SOURCE_COLORS = {
    "Fli library": COLORS["blue"],
    "Trip.com": COLORS["teal"],
}


def normalize_collection_era(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose canonical collector names through the shared source taxonomy."""

    return normalize_source_columns(frame)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Helvetica",
            "font.size": 8.4,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.7,
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["line"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "text.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.48)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=COLORS["teal"],
        va="top",
        ha="left",
    )


def add_period_shading(ax: plt.Axes) -> None:
    test1_start = pd.Timestamp("2026-07-29")
    test2_start = pd.Timestamp("2026-08-09")
    end = pd.Timestamp("2026-08-20")
    ax.axvspan(test1_start, test2_start, color=COLORS["blue_light"], alpha=0.72, zorder=0)
    ax.axvspan(test2_start, end, color=COLORS["orange_light"], alpha=0.72, zorder=0)
    ax.axvline(test1_start, color=COLORS["blue"], linewidth=1.0, linestyle=(0, (3, 3)))
    ax.axvline(test2_start, color=COLORS["orange"], linewidth=1.0, linestyle=(0, (3, 3)))


def add_collection_transition(ax: plt.Axes, transition: pd.Timestamp, *, label: bool = False) -> None:
    ax.axvline(transition, color=COLORS["line"], linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)
    if label:
        ax.annotate(
            "Collection protocol\ntransition",
            xy=(transition, 0.96),
            xycoords=("data", "axes fraction"),
            xytext=(8, -4),
            textcoords="offset points",
            color=COLORS["muted"],
            fontsize=8.3,
            fontweight="bold",
            ha="left",
            va="top",
        )


def read_member_bytes(archive: Path, member_name: str) -> bytes:
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Unable to read {member_name} from {archive}")
        return handle.read()


def extract_member(archive: Path, member_name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Unable to read {member_name} from {archive}")
        with destination.open("wb") as output:
            while chunk := handle.read(1024 * 1024):
                output.write(chunk)
    return destination


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), pad_inches=0.08)
    plt.close(fig)


def validate_summary(summary: dict[str, object]) -> None:
    eda = summary["eda"]
    if eda["rows"] != 1_114_445 or eda["dates"] != 150 or eda["routes"] != 20:
        raise RuntimeError(f"Frozen EDA contract mismatch: {eda}")
    if sum(eda["airlines"].values()) != eda["rows"] or len(eda["airlines"]) != 5:
        raise RuntimeError(f"Airline inventory mismatch: {eda['airlines']}")
    if eda["missing_collection_dates"] != ["2026-05-08", "2026-05-09"]:
        raise RuntimeError(f"Unexpected collection gaps: {eda['missing_collection_dates']}")
    if eda["minimum_airline_day_support"] != MIN_AIRLINE_DAY_SUPPORT:
        raise RuntimeError("Airline-day support rule mismatch")
    if sum(eda["sources"].values()) != eda["rows"]:
        raise RuntimeError(f"Source inventory mismatch: {eda['sources']}")
    if eda["temporal_fare_index"]["observed_days"] != 150:
        raise RuntimeError(f"Temporal fare-index coverage mismatch: {eda['temporal_fare_index']}")
    if eda["temporal_fare_index"]["normalisation"] != "route_airline_dud_session_label":
        raise RuntimeError(f"Temporal fare-index definition mismatch: {eda['temporal_fare_index']}")

    evaluation = summary["evaluation"]
    expected_skill = {
        "Classification\nBrier score": 17.992536557551396,
        "Regression\nMAPE": 48.50728556863948,
        "Distribution\nWIS": 34.77687751476064,
        "Ranking\nNDCG@5": 24.46739951593191,
    }
    for metric, expected in expected_skill.items():
        actual = evaluation["relative_skill_percent"]["Pooled"][metric]
        if not np.isclose(actual, expected, rtol=0, atol=1e-10):
            raise RuntimeError(f"Pooled skill mismatch for {metric}: {actual} != {expected}")

    expected_coverage = {"50": 0.4865457759260136, "80": 0.7812130107475972, "90": 0.8853273992708679}
    for level, expected in expected_coverage.items():
        actual = evaluation["pooled_coverage"][level]
        if not np.isclose(actual, expected, rtol=0, atol=1e-10):
            raise RuntimeError(f"Coverage mismatch for {level}: {actual} != {expected}")

    for item in evaluation["policy_regret_reduction_vnd"]:
        if not item["low"] <= item["effect"] <= item["high"]:
            raise RuntimeError(f"Invalid policy interval: {item}")
    expected_calibration_rows = {"Test 1": 41_573, "Test 2": 42_146}
    for block, expected_rows in expected_calibration_rows.items():
        calibration = evaluation["classification_calibration"][block]
        if calibration["rows"] != expected_rows or sum(calibration["rows_per_bin"]) != expected_rows:
            raise RuntimeError(f"Calibration inventory mismatch for {block}: {calibration}")

    expected_diagnostics = {
        "Test 1": {"rows": 41_573, "auc": 0.662589823437386, "auprc": 0.18870621481395916},
        "Test 2": {"rows": 42_146, "auc": 0.6551827317359795, "auprc": 0.24772246483213742},
    }
    for block, expected in expected_diagnostics.items():
        actual = evaluation["classification_diagnostics"][block]
        if actual["rows"] != expected["rows"]:
            raise RuntimeError(f"Classification diagnostic rows mismatch for {block}: {actual}")
        for metric in ["auc", "auprc"]:
            if not np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-12):
                raise RuntimeError(f"Classification diagnostic mismatch for {block} {metric}: {actual}")

    expected_regression = {
        "Test 1": {"rows": 95_962, "r_squared": 0.8519733247116501},
        "Test 2": {"rows": 94_127, "r_squared": 0.8124754447607193},
    }
    for block, expected in expected_regression.items():
        actual = evaluation["regression_diagnostics"][block]
        if actual["rows"] != expected["rows"] or not np.isclose(
            actual["r_squared"], expected["r_squared"], rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"Regression diagnostic mismatch for {block}: {actual}")


def _build_eda_figure_legacy(source: pd.DataFrame, output: Path) -> dict[str, object]:
    source = normalize_collection_era(source)
    source["session_date"] = pd.to_datetime(source["session_date"])
    transition = source.loc[
        source["collection_era"].eq(TRIP_COM_BROWSER_ERA), "session_date"
    ].min()
    if pd.isna(transition):
        raise RuntimeError("Trip.com collection transition not found")
    airline_order = [name for name in ["VN", "VJ", "9G", "VU", "QH"] if name in set(source["airline"])]

    daily_counts = (
        source.groupby(["session_date", "airline"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=airline_order)
        .sort_index()
    )
    full_dates = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D")
    daily_counts = daily_counts.reindex(full_dates)
    daily_total = daily_counts.sum(axis=1, min_count=1)
    total_median = daily_total.rolling(7, min_periods=3, center=True).median()
    total_median = total_median.where(daily_total.notna())
    daily_share = daily_counts.div(daily_total, axis=0).mul(100)
    daily_drift = (
        source.groupby(["session_date", "airline"], observed=True)["current_relative_log"]
        .median()
        .unstack()
        .reindex(columns=airline_order)
        .sort_index()
        .reindex(full_dates)
    )
    supported_drift = daily_drift.where(daily_counts >= MIN_AIRLINE_DAY_SUPPORT)
    smoothed_drift = supported_drift.rolling(7, min_periods=3, center=True).median()
    smoothed_drift = smoothed_drift.where(supported_drift.notna())
    missing_dates = daily_total.index[daily_total.isna()]

    date_min = daily_counts.index.min()
    date_max = daily_counts.index.max() + pd.Timedelta(days=1)
    test1_start = pd.Timestamp("2026-07-29")
    test2_start = pd.Timestamp("2026-08-09")

    fig = plt.figure(figsize=(7.25, 6.15), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.48],
        hspace=0.27,
        left=0.12,
        right=0.985,
        top=0.96,
        bottom=0.09,
    )
    top = outer[0].subgridspec(3, 1, height_ratios=[0.17, 1.0, 0.19], hspace=0.05)
    band = fig.add_subplot(top[0])
    volume_ax = fig.add_subplot(top[1], sharex=band)
    share_ax = fig.add_subplot(top[2], sharex=band)
    drift_grid = outer[1].subgridspec(len(airline_order), 1, hspace=0.07)
    drift_axes = [fig.add_subplot(drift_grid[index]) for index in range(len(airline_order))]

    segments = [
        (date_min, transition, COLORS["gray_light"], "Development · initial collection", COLORS["muted"]),
        (transition, test1_start, COLORS["teal_light"], "Development · expanded collection", COLORS["teal"]),
        (test1_start, test2_start, COLORS["blue_light"], "Test 1", COLORS["blue"]),
        (test2_start, date_max, COLORS["orange_light"], "Test 2", COLORS["orange"]),
    ]
    band.set_xlim(date_min, date_max)
    band.set_ylim(0, 1)
    for start, end, fill, label, color in segments:
        band.axvspan(start, end, ymin=0.15, ymax=0.86, color=fill, ec="none")
        band.text(
            start + (end - start) / 2,
            0.5,
            label,
            color=color,
            fontsize=6.9,
            fontweight="bold",
            ha="center",
            va="center",
        )
    band.axis("off")

    volume_ax.plot(daily_total.index, daily_total, color=COLORS["gray"], linewidth=0.75, alpha=0.55)
    volume_ax.plot(total_median.index, total_median, color=COLORS["ink"], linewidth=1.55)
    volume_ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["gray"], linewidth=0.9, alpha=0.7, label="Raw daily volume"),
            Line2D([0], [0], color=COLORS["ink"], linewidth=1.6, label="7-day rolling median"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=6.7,
        handlelength=2.2,
        borderaxespad=0.3,
    )
    if len(missing_dates):
        missing_midpoint = missing_dates[0] + (missing_dates[-1] - missing_dates[0]) / 2
        volume_ax.text(
            missing_midpoint,
            0.05,
            "No collection\n8–9 May",
            transform=volume_ax.get_xaxis_transform(),
            color=COLORS["muted"],
            fontsize=6.2,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    volume_ax.set_ylabel("Offers/day")
    volume_ax.set_ylim(0, daily_total.max() * 1.08)
    volume_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    volume_ax.tick_params(axis="x", labelbottom=False, length=0)
    clean_axis(volume_ax)

    share_ax.stackplot(
        daily_share.index,
        [daily_share[name].to_numpy() for name in airline_order],
        colors=[AIRLINE_COLORS[name] for name in airline_order],
        linewidth=0,
        alpha=0.9,
    )
    share_ax.set_ylim(0, 100)
    share_ax.set_yticks([])
    share_ax.tick_params(axis="x", labelbottom=False, length=0)
    share_ax.spines[:].set_visible(False)
    share_ax.text(
        -0.014,
        0.5,
        "Airline share",
        transform=share_ax.transAxes,
        color=COLORS["muted"],
        fontsize=7.2,
        ha="right",
        va="center",
    )
    share_ax.legend(
        handles=[Patch(facecolor=AIRLINE_COLORS[name], edgecolor="none", label=name) for name in airline_order],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
        ncol=len(airline_order),
        fontsize=6.7,
        handlelength=1.0,
        handleheight=0.7,
        columnspacing=1.4,
        borderaxespad=0,
    )

    for axis in [volume_ax, share_ax]:
        for boundary, color in [(transition, COLORS["gray"]), (test1_start, COLORS["blue"]), (test2_start, COLORS["orange"])]:
            axis.axvline(boundary, color=color, linewidth=0.65, alpha=0.42)

    shared_ymin, shared_ymax = -0.22, 0.27
    for index, (axis, airline) in enumerate(zip(drift_axes, airline_order, strict=True)):
        axis.plot(
            smoothed_drift.index,
            smoothed_drift[airline],
            color=AIRLINE_COLORS[airline],
            linewidth=1.65,
        )
        axis.axhline(0, color=COLORS["line"], linewidth=0.72, alpha=0.55)
        axis.set_ylim(shared_ymin, shared_ymax)
        axis.set_xlim(date_min, date_max)
        axis.set_yticks([])
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_visible(index == len(airline_order) - 1)
        axis.tick_params(axis="x", labelbottom=index == len(airline_order) - 1, length=3)
        axis.text(
            -0.018,
            0.5,
            airline,
            transform=axis.transAxes,
            color=AIRLINE_COLORS[airline],
            fontsize=8.3,
            fontweight="bold",
            ha="right",
            va="center",
        )
        for boundary, color in [(transition, COLORS["gray"]), (test1_start, COLORS["blue"]), (test2_start, COLORS["orange"])]:
            axis.axvline(boundary, color=color, linewidth=0.55, alpha=0.3)

    drift_axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    drift_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    drift_axes[-1].set_xlabel("Collection date")

    top_position = band.get_position()
    drift_position = drift_axes[0].get_position()
    fig.text(top_position.x0 - 0.045, top_position.y1 + 0.012, "A", color=COLORS["teal"], fontsize=10, fontweight="bold")
    fig.text(top_position.x0, top_position.y1 + 0.012, "Offer volume and airline mix", fontsize=10.2, fontweight="bold")
    fig.text(drift_position.x0 - 0.045, drift_position.y1 + 0.014, "B", color=COLORS["teal"], fontsize=10, fontweight="bold")
    fig.text(drift_position.x0, drift_position.y1 + 0.014, "Airline fare drift from strictly prior anchors", fontsize=10.2, fontweight="bold")
    fig.text(
        0.035,
        (drift_axes[0].get_position().y1 + drift_axes[-1].get_position().y0) / 2,
        "Relative log fare\n(shared range −0.22 to +0.27)",
        rotation=90,
        color=COLORS["muted"],
        fontsize=7.6,
        ha="center",
        va="center",
    )

    save_figure(fig, output)
    return {
        "rows": int(len(source)),
        "dates": int(source["session_date"].nunique()),
        "date_min": str(source["session_date"].min().date()),
        "date_max": str(source["session_date"].max().date()),
        "collection_transition": str(transition.date()),
        "missing_collection_dates": [str(value.date()) for value in missing_dates],
        "volume_smoothing": "centered 7-day rolling median; no interpolation across unobserved dates",
        "drift_definition": "log(observed fare / strictly prior anchor)",
        "drift_smoothing": "centered 7-day rolling median; gaps retained",
        "minimum_airline_day_support": MIN_AIRLINE_DAY_SUPPORT,
        "airlines": {name: int((source["airline"] == name).sum()) for name in airline_order},
        "routes": int(source["route"].nunique()),
    }


def build_eda_figure_v2(source: pd.DataFrame, output: Path) -> dict[str, object]:
    source = normalize_collection_era(source)
    source["session_date"] = pd.to_datetime(source["session_date"])
    eras = set(source["collection_era"].astype(str).unique())
    if eras != set(SOURCE_LABELS):
        raise RuntimeError(f"Unexpected collection-source inventory: {sorted(eras)}")
    source["data_source"] = source["collection_era"].map(SOURCE_LABELS)
    transition = source.loc[source["data_source"].eq("Trip.com"), "session_date"].min()
    if pd.isna(transition):
        raise RuntimeError("Trip.com source transition not found")
    if source["offer_id"].duplicated().any():
        raise RuntimeError("Standard offer_id is not unique; source counts require a deduplication audit")

    unique_offers = source.drop_duplicates("offer_id").copy()
    airline_order = [name for name in ["VN", "VJ", "9G", "VU", "QH"] if name in set(unique_offers["airline"])]
    full_dates = pd.date_range(
        unique_offers["session_date"].min(), unique_offers["session_date"].max(), freq="D"
    )
    date_min = full_dates.min()
    date_max = full_dates.max() + pd.Timedelta(days=1)
    test1_start = pd.Timestamp("2026-07-29")
    test2_start = pd.Timestamp("2026-08-09")

    daily_source = (
        unique_offers.groupby(["session_date", "data_source"], observed=True)["offer_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=["Fli library", "Trip.com"], fill_value=0)
        .sort_index()
        .reindex(full_dates)
    )
    daily_total = daily_source.sum(axis=1, min_count=1)
    missing_dates = daily_total.index[daily_total.isna()]

    airline_source = (
        unique_offers.groupby(["airline", "data_source"], observed=True)["offer_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=["Fli library", "Trip.com"], fill_value=0)
    )
    airline_source["Total"] = airline_source.sum(axis=1)
    airline_source = airline_source.sort_values("Total", ascending=False)

    collected_dates = int(unique_offers["session_date"].nunique())
    fare_rows = unique_offers[
        unique_offers["price_vnd"].gt(0)
        & unique_offers["days_until_departure"].isin(BOOKING_WINDOWS)
    ].copy()
    fare_strata = ["route", "airline", "days_until_departure", "session_label"]
    fare_reference = (
        fare_rows.groupby(fare_strata, observed=True)["price_vnd"]
        .median()
        .rename("stratum_reference_vnd")
        .reset_index()
    )
    fare_rows = fare_rows.merge(
        fare_reference,
        on=fare_strata,
        how="left",
        validate="many_to_one",
    )
    fare_rows["log_relative_fare"] = np.log(
        fare_rows["price_vnd"] / fare_rows["stratum_reference_vnd"]
    )
    daily_log_fare = fare_rows.groupby("session_date", observed=True)[
        "log_relative_fare"
    ].median()
    daily_fare_index = (100 * np.exp(daily_log_fare)).reindex(full_dates)
    rolling_fare_index = daily_fare_index.rolling(7, min_periods=3).median()

    fig = plt.figure(figsize=(7.25, 5.45), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.90, 1.10],
        width_ratios=[1.47, 1.0],
        hspace=0.30,
        wspace=0.31,
        left=0.095,
        right=0.98,
        top=0.94,
        bottom=0.155,
    )
    volume_ax = fig.add_subplot(outer[0, 0])
    coverage_ax = fig.add_subplot(outer[0, 1])
    bottom = outer[1, :].subgridspec(2, 1, height_ratios=[0.12, 1.0], hspace=0.025)
    band = fig.add_subplot(bottom[0])
    drift_ax = fig.add_subplot(bottom[1], sharex=band)

    volume_ax.stackplot(
        daily_source.index,
        [daily_source[name].fillna(0).to_numpy() for name in ["Fli library", "Trip.com"]],
        colors=[SOURCE_COLORS["Fli library"], SOURCE_COLORS["Trip.com"]],
        alpha=0.68,
        linewidth=0,
    )
    volume_ax.plot(daily_total.index, daily_total, color=COLORS["ink"], linewidth=1.35, zorder=3)
    volume_ax.legend(
        handles=[
            Patch(facecolor=SOURCE_COLORS["Fli library"], edgecolor="none", alpha=0.68, label="Fli library"),
            Patch(facecolor=SOURCE_COLORS["Trip.com"], edgecolor="none", alpha=0.68, label="Trip.com"),
            Line2D([0], [0], color=COLORS["ink"], linewidth=1.35, label="Total"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=6.5,
        ncol=3,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.25,
    )
    if len(missing_dates):
        missing_midpoint = missing_dates[0] + (missing_dates[-1] - missing_dates[0]) / 2
        volume_ax.text(
            missing_midpoint,
            0.05,
            "No collection\n8–9 May",
            transform=volume_ax.get_xaxis_transform(),
            color=COLORS["muted"],
            fontsize=6.0,
            fontweight="semibold",
            ha="center",
            va="bottom",
        )
    for boundary, color in [
        (transition, COLORS["gray"]),
        (test1_start, COLORS["blue"]),
        (test2_start, COLORS["orange"]),
    ]:
        volume_ax.axvline(boundary, color=color, linewidth=0.60, alpha=0.36)
    volume_ax.set_xlim(date_min, date_max)
    volume_ax.set_ylim(0, daily_total.max() * 1.08)
    volume_ax.set_ylabel("Unique offers per day")
    volume_ax.xaxis.set_major_locator(mdates.MonthLocator())
    volume_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    volume_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    clean_axis(volume_ax)

    y = np.arange(len(airline_source))
    left = np.zeros(len(airline_source))
    for source_name in ["Fli library", "Trip.com"]:
        values = airline_source[source_name].to_numpy()
        coverage_ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=SOURCE_COLORS[source_name],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.25,
        )
        for index, (start, value, total) in enumerate(zip(left, values, airline_source["Total"], strict=True)):
            if value >= 35_000 and value / total >= 0.14:
                coverage_ax.text(
                    start + value / 2,
                    index,
                    f"{value / total:.0%}",
                    color="white",
                    fontsize=6.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
        left += values
    grand_total = float(airline_source["Total"].sum())
    label_offset = float(airline_source["Total"].max()) * 0.012
    for index, total in enumerate(airline_source["Total"]):
        coverage_ax.text(
            total + label_offset,
            index,
            f"{total:,.0f} · {total / grand_total:.1%}",
            fontsize=6.3,
            ha="left",
            va="center",
        )
    coverage_ax.set_yticks(y, airline_source.index)
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, airline_source["Total"].max() * 1.28)
    coverage_ax.set_xlabel("Unique offers")
    coverage_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    coverage_ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k")
    )
    clean_axis(coverage_ax, grid_axis="x")

    segments = [
        (date_min, transition, COLORS["gray_light"], "Fli development", COLORS["muted"]),
        (transition, test1_start, COLORS["teal_light"], "Trip.com development", COLORS["teal"]),
        (test1_start, test2_start, COLORS["blue_light"], "Test 1", COLORS["blue"]),
        (test2_start, date_max, COLORS["orange_light"], "Test 2", COLORS["orange"]),
    ]
    band.set_xlim(date_min, date_max)
    band.set_ylim(0, 1)
    for start, end, fill, label, color in segments:
        band.axvspan(start, end, ymin=0.12, ymax=0.88, color=fill, ec="none")
        band.text(
            start + (end - start) / 2,
            0.5,
            label,
            color=color,
            fontsize=6.6,
            fontweight="bold",
            ha="center",
            va="center",
        )
    band.axis("off")

    drift_ax.plot(
        full_dates,
        daily_fare_index,
        color=COLORS["blue"],
        linewidth=0.9,
        alpha=0.42,
        label="Daily adjusted fare index",
    )
    drift_ax.plot(
        full_dates,
        rolling_fare_index,
        color=COLORS["teal"],
        linewidth=1.8,
        label="7-day rolling median",
    )
    drift_ax.axhline(
        100,
        color=COLORS["line"],
        linewidth=0.9,
        linestyle=(0, (4, 4)),
        label="Reference level (100)",
    )
    add_period_shading(drift_ax)
    add_collection_transition(drift_ax, transition)
    drift_ax.set_xlim(date_min, date_max)
    drift_ax.set_ylabel("Comparable-fare index")
    drift_ax.set_xlabel("Observation date (2026)", labelpad=31)
    drift_ax.xaxis.set_major_locator(mdates.MonthLocator())
    drift_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    drift_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    clean_axis(drift_ax)
    drift_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=6.8,
        ncol=3,
        handlelength=2.0,
        borderaxespad=0.3,
        columnspacing=1.0,
    )

    for axis, label, title in [
        (volume_ax, "A", "Daily unique offers by source"),
        (coverage_ax, "B", "Airline coverage by source"),
        (band, "C", "Observed comparable-fare drift"),
    ]:
        position = axis.get_position()
        fig.text(
            position.x0 - 0.035,
            position.y1 + 0.012,
            label,
            color=COLORS["teal"],
            fontsize=9.5,
            fontweight="bold",
        )
        fig.text(position.x0, position.y1 + 0.012, title, fontsize=9.5, fontweight="bold")

    save_figure(fig, output)
    observed_fare_index = daily_fare_index.dropna()
    return {
        "rows": int(len(unique_offers)),
        "dates": collected_dates,
        "date_min": str(unique_offers["session_date"].min().date()),
        "date_max": str(unique_offers["session_date"].max().date()),
        "collection_transition": str(transition.date()),
        "missing_collection_dates": [str(value.date()) for value in missing_dates],
        "offer_count_definition": "unique offer_id after final standard-population deduplication",
        "sources": {
            name: int((unique_offers["data_source"] == name).sum())
            for name in ["Fli library", "Trip.com"]
        },
        "airline_source_rows": {
            airline: {
                name: int(airline_source.loc[airline, name])
                for name in ["Fli library", "Trip.com"]
            }
            for airline in airline_source.index
        },
        "temporal_fare_index": {
            "definition": "100 * exp(daily median log(fare / stratum median fare))",
            "normalisation": "route_airline_dud_session_label",
            "smoothing": "trailing 7-day rolling median; collection gaps retained",
            "reference": 100,
            "observed_days": int(observed_fare_index.size),
            "minimum": round(float(observed_fare_index.min()), 2),
            "maximum": round(float(observed_fare_index.max()), 2),
            "peak_to_trough_points": round(
                float(observed_fare_index.max() - observed_fare_index.min()),
                2,
            ),
        },
        "minimum_airline_day_support": MIN_AIRLINE_DAY_SUPPORT,
        "airlines": {
            name: int((unique_offers["airline"] == name).sum()) for name in airline_order
        },
        "routes": int(unique_offers["route"].nunique()),
    }


def load_test_block(
    block: str,
    report: dict[str, object],
    classification_predictions: pd.DataFrame,
    classification_labels: pd.DataFrame,
    point_predictions: pd.DataFrame,
    regression_labels: pd.DataFrame,
) -> dict[str, object]:
    if report.get("status") != "PASS" or report.get("evaluation_complete") is not True:
        raise RuntimeError(f"{block} report is not a completed PASS")
    merged = classification_predictions[["row_key", "probability"]].merge(
        classification_labels[["row_key", "DROP_5PCT"]],
        on="row_key",
        validate="one_to_one",
    )
    merged["bin"] = pd.qcut(merged["probability"], q=6, duplicates="drop")
    reliability = (
        merged.groupby("bin", observed=True)
        .agg(predicted=("probability", "mean"), observed=("DROP_5PCT", "mean"), rows=("row_key", "size"))
        .reset_index(drop=True)
    )
    regression = point_predictions[
        ["row_key", "query_dud", "predicted_price_vnd"]
    ].merge(
        regression_labels[["row_key", "target_session_price_vnd"]],
        on="row_key",
        validate="one_to_one",
    )
    if regression[["predicted_price_vnd", "target_session_price_vnd"]].isna().any().any():
        raise RuntimeError(f"{block} regression diagnostics contain missing values")
    return {
        "block": block,
        "report": report,
        "classification": merged,
        "regression": regression,
        "reliability": reliability,
        "rows": len(merged),
    }


def relative_skill(report: dict[str, object]) -> dict[str, float]:
    classification = report["classification"]
    point = report["point"]
    distribution = report["distribution"]
    ranking = report["ranking"]
    return {
        "Classification\nBrier score": 100
        * (1 - classification["model"]["brier"] / classification["baseline_constant_dev_rate"]["brier"]),
        "Regression\nMAPE": 100 * (1 - point["model"]["mape"] / point["baseline_prior_anchor"]["mape"]),
        "Distribution\nWIS": 100
        * (1 - distribution["model"]["wis"] / distribution["baseline_empirical_dev_residual"]["wis"]),
        "Ranking\nNDCG@5": 100
        * (ranking["model"]["ndcg5"] / ranking["baseline_prior_anchor"]["ndcg5"] - 1),
    }


def pooled_metric(blocks: list[dict[str, object]], path: tuple[str, ...], weight_path: tuple[str, ...]) -> float:
    values = []
    weights = []
    for block in blocks:
        cursor = block["report"]
        for key in path:
            cursor = cursor[key]
        weight = block["report"]
        for key in weight_path:
            weight = weight[key]
        values.append(float(cursor))
        weights.append(float(weight))
    return float(np.average(values, weights=weights))


def pooled_skill(blocks: list[dict[str, object]]) -> dict[str, float]:
    specs = {
        "Classification\nBrier score": (
            ("classification", "model", "brier"),
            ("classification", "baseline_constant_dev_rate", "brier"),
            ("buy_wait", "frozen", "rows"),
            "lower",
        ),
        "Regression\nMAPE": (
            ("point", "model", "mape"),
            ("point", "baseline_prior_anchor", "mape"),
            ("point", "paired_day_bootstrap", "day_blocks"),
            "lower",
        ),
        "Distribution\nWIS": (
            ("distribution", "model", "wis"),
            ("distribution", "baseline_empirical_dev_residual", "wis"),
            ("distribution", "model", "coverage50"),
            "lower",
        ),
        "Ranking\nNDCG@5": (
            ("ranking", "model", "ndcg5"),
            ("ranking", "baseline_prior_anchor", "ndcg5"),
            ("ranking", "model", "queries"),
            "higher",
        ),
    }
    row_weights = {
        "Classification\nBrier score": [block["rows"] for block in blocks],
        "Regression\nMAPE": [95962, 94127],
        "Distribution\nWIS": [95962, 94127],
        "Ranking\nNDCG@5": [
            block["report"]["ranking"]["model"]["queries"] for block in blocks
        ],
    }
    result = {}
    for name, (model_path, baseline_path, _, direction) in specs.items():
        model_values = []
        baseline_values = []
        for block in blocks:
            model = block["report"]
            baseline = block["report"]
            for key in model_path:
                model = model[key]
            for key in baseline_path:
                baseline = baseline[key]
            model_values.append(float(model))
            baseline_values.append(float(baseline))
        weights = row_weights[name]
        model_pooled = float(np.average(model_values, weights=weights))
        baseline_pooled = float(np.average(baseline_values, weights=weights))
        if direction == "lower":
            result[name] = 100 * (1 - model_pooled / baseline_pooled)
        else:
            result[name] = 100 * (model_pooled / baseline_pooled - 1)
    return result


def _build_evaluation_figure_legacy(blocks: list[dict[str, object]], output: Path) -> dict[str, object]:
    skills = {
        "Test 1": relative_skill(blocks[0]["report"]),
        "Test 2": relative_skill(blocks[1]["report"]),
        "Pooled": pooled_skill(blocks),
    }
    task_order = list(skills["Test 1"])
    series_colors = {"Test 1": COLORS["blue"], "Test 2": COLORS["orange"], "Pooled": COLORS["ink"]}

    fig = plt.figure(figsize=(7.25, 5.85), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.985,
        top=0.95,
        bottom=0.10,
        wspace=0.40,
        hspace=0.38,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    y = np.arange(len(task_order))
    offsets = {"Test 1": -0.20, "Test 2": 0.0, "Pooled": 0.20}
    for label in ["Test 1", "Test 2", "Pooled"]:
        values = [skills[label][task] for task in task_order]
        marker = "D" if label == "Pooled" else "o"
        ax1.scatter(values, y + offsets[label], s=29, marker=marker, color=series_colors[label], zorder=3)
    display_tasks = [
        "Classification · Brier",
        "Regression · MAPE",
        "Distribution · WIS",
        "Ranking · NDCG@5",
    ]
    pooled_values = [skills["Pooled"][task] for task in task_order]
    for index, value in enumerate(pooled_values):
        ax1.text(value + 1.5, index + offsets["Pooled"], f"{value:.1f}%", fontsize=7.4, va="center")
    label_y = y[0] + np.array([offsets["Test 1"], offsets["Test 2"], offsets["Pooled"]])
    for name, y_position in zip(["T1", "T2", "Pooled"], label_y, strict=True):
        ax1.text(
            54.6,
            y_position,
            name,
            color=series_colors[{"T1": "Test 1", "T2": "Test 2"}.get(name, name)],
            fontsize=7.1,
            fontweight="bold",
            va="center",
            ha="right",
        )
    ax1.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax1.set_yticks(y, display_tasks)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 57)
    ax1.set_xlabel("Improvement over task baseline (%)")
    clean_axis(ax1, grid_axis="x")

    for block, color in zip(blocks, [COLORS["blue"], COLORS["orange"]], strict=True):
        rel = block["reliability"]
        ax2.plot(rel["predicted"], rel["observed"], marker="o", markersize=3.3, linewidth=1.45, color=color)
        endpoint = rel.iloc[-1]
        ax2.annotate(
            block["block"],
            xy=(endpoint["predicted"], endpoint["observed"]),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="center",
        )
    ax2.plot([0, 0.4], [0, 0.4], color=COLORS["gray"], linewidth=0.9, linestyle=(0, (3, 3)))
    ax2.text(0.385, 0.372, "Perfect", color=COLORS["muted"], fontsize=6.9, ha="right", rotation=37)
    ax2.set_xlim(0, 0.4)
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Observed drop frequency")
    clean_axis(ax2, grid_axis="both")

    nominal = np.array([0.5, 0.8, 0.9])
    pooled_coverages = []
    coverage_by_series: dict[str, np.ndarray] = {}
    for block in blocks:
        report = block["report"]
        observed = np.array([report["distribution"]["model"][f"coverage{level}"] for level in [50, 80, 90]])
        coverage_by_series[block["block"]] = observed
        pooled_coverages.append(observed)
    weights = np.array([95962, 94127])
    pooled_coverage = np.average(np.vstack(pooled_coverages), axis=0, weights=weights)
    coverage_by_series["Pooled"] = pooled_coverage
    coverage_y = np.arange(len(nominal))
    coverage_offsets = {"Test 1": -0.18, "Test 2": 0.0, "Pooled": 0.18}
    for label in ["Test 1", "Test 2", "Pooled"]:
        errors = (coverage_by_series[label] - nominal) * 100
        marker = "D" if label == "Pooled" else "o"
        ax3.scatter(
            errors,
            coverage_y + coverage_offsets[label],
            s=29,
            marker=marker,
            color=series_colors[label],
            zorder=3,
        )
    pooled_errors = (pooled_coverage - nominal) * 100
    for index, error in enumerate(pooled_errors):
        formatted = f"{error:+.2f}".replace("-", "−")
        ax3.text(
            error + 0.12,
            coverage_y[index] + coverage_offsets["Pooled"],
            f"{formatted} pp",
            color=COLORS["ink"],
            fontsize=6.8,
            va="center",
        )
    for name, offset in [("T1", -0.18), ("T2", 0.0), ("Pooled", 0.18)]:
        ax3.text(
            -3.8,
            coverage_y[0] + offset,
            name,
            color=series_colors[{"T1": "Test 1", "T2": "Test 2"}.get(name, name)],
            fontsize=7.1,
            fontweight="bold",
            va="center",
        )
    ax3.axvline(0, color=COLORS["line"], linewidth=1.0, alpha=0.75)
    ax3.set_xlim(-4.1, 1.0)
    ax3.set_yticks(coverage_y, ["50% interval", "80% interval", "90% interval"])
    ax3.invert_yaxis()
    ax3.set_xlabel("Observed − nominal coverage (percentage points)")
    clean_axis(ax3, grid_axis="x")

    policy = []
    for block in blocks:
        item = block["report"]["buy_wait"]["paired_day_bootstrap_vs_buy_all"]
        policy.append(
            {
                "name": block["block"],
                "effect": -float(item["mean_model_minus_baseline"]),
                "low": -float(item["ci95_high"]),
                "high": -float(item["ci95_low"]),
            }
        )
    pooled_effect = 724.50
    policy.append({"name": "Pooled", "effect": pooled_effect, "low": -187.37, "high": 1671.25})
    y_policy = np.arange(len(policy))
    for index, item in enumerate(policy):
        color = [COLORS["blue"], COLORS["orange"], COLORS["ink"]][index]
        ax4.errorbar(
            item["effect"],
            index,
            xerr=[[item["effect"] - item["low"]], [item["high"] - item["effect"]]],
            fmt="o",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=3,
            zorder=3,
        )
        ax4.text(
            5000,
            index,
            f"{item['effect']:,.0f} [{item['low']:,.0f}, {item['high']:,.0f}]",
            color=COLORS["ink"],
            fontsize=7.1,
            ha="right",
            va="center",
        )
    ax4.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax4.set_yticks(y_policy, [item["name"] for item in policy])
    ax4.invert_yaxis()
    ax4.set_xlim(-1050, 5200)
    ax4.set_xlabel("Regret reduction vs always buy (VND)")
    clean_axis(ax4, grid_axis="x")

    panel_titles = [
        (ax1, "A", "Relative skill"),
        (ax2, "B", "Probability calibration"),
        (ax3, "C", "Coverage error"),
        (ax4, "D", "Policy regret reduction"),
    ]
    for axis, label, title in panel_titles:
        position = axis.get_position()
        fig.text(position.x0 - 0.035, position.y1 + 0.012, label, color=COLORS["teal"], fontsize=9.5, fontweight="bold")
        fig.text(position.x0, position.y1 + 0.012, title, fontsize=9.5, fontweight="bold")

    save_figure(fig, output)
    calibration_summary = {
        block["block"]: {
            "rows": int(block["rows"]),
            "bins": 6,
            "binning": "equal-frequency",
            "rows_per_bin": [int(value) for value in block["reliability"]["rows"]],
        }
        for block in blocks
    }
    policy_rows = [int(block["report"]["buy_wait"]["frozen"]["rows"]) for block in blocks]
    pooled_wait_rate = float(
        np.average(
            [float(block["report"]["buy_wait"]["frozen"]["wait_rate"]) for block in blocks],
            weights=policy_rows,
        )
    )
    return {
        "relative_skill_percent": skills,
        "relative_skill_definition": {
            "lower_is_better": "100 * (1 - model / baseline)",
            "higher_is_better": "100 * (model / baseline - 1)",
            "uncertainty": "descriptive point estimates; raw-metric bootstrap evidence is reported separately",
        },
        "classification_calibration": calibration_summary,
        "pooled_coverage": {str(level): float(value) for level, value in zip([50, 80, 90], pooled_coverage, strict=True)},
        "pooled_coverage_error_percentage_points": {
            str(level): float(error) for level, error in zip([50, 80, 90], pooled_errors, strict=True)
        },
        "policy_regret_reduction_vnd": policy,
        "policy_assumptions": {
            "baseline": "always buy",
            "bootstrap_draws": 4000,
            "friction_vnd": 50000,
            "wait_threshold": 0.30,
            "pooled_wait_rate": pooled_wait_rate,
        },
    }


def metric_at(report: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = report
    for key in path:
        value = value[key]
    return float(value)


def build_classification_report_figure(
    blocks: list[dict[str, object]], output: Path
) -> dict[str, object]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.72),
        gridspec_kw={"left": 0.08, "right": 0.985, "top": 0.78, "bottom": 0.18, "wspace": 0.34},
    )
    colors = {"Test 1": COLORS["blue"], "Test 2": COLORS["orange"]}
    summary: dict[str, dict[str, float | int]] = {}

    for block in blocks:
        label = block["block"]
        color = colors[label]
        frame = block["classification"]
        target = frame["DROP_5PCT"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)
        prevalence = float(target.mean())
        precision, recall, _ = precision_recall_curve(target, probability)
        false_positive, true_positive, _ = roc_curve(target, probability)
        auprc = float(average_precision_score(target, probability))
        auc = float(roc_auc_score(target, probability))
        model_metrics = block["report"]["classification"]["model"]

        axes[0].plot(recall, precision, color=color, linewidth=1.55)
        axes[0].axhline(
            prevalence,
            color=color,
            linewidth=0.75,
            linestyle=(0, (2, 3)),
            alpha=0.58,
        )
        axes[1].plot(false_positive, true_positive, color=color, linewidth=1.55)
        reliability = block["reliability"]
        axes[2].plot(
            reliability["predicted"],
            reliability["observed"],
            color=color,
            linewidth=1.55,
            marker="o",
            markersize=2.8,
        )
        summary[label] = {
            "rows": int(len(frame)),
            "prevalence": prevalence,
            "auc": auc,
            "auprc": auprc,
            "auprc_over_prevalence": auprc / prevalence,
            "brier": float(model_metrics["brier"]),
            "ece15": float(model_metrics["ece15"]),
        }

    axes[1].plot(
        [0, 1], [0, 1], color=COLORS["gray"], linewidth=0.8, linestyle=(0, (3, 3))
    )
    axes[2].plot(
        [0, 0.4], [0, 0.4], color=COLORS["gray"], linewidth=0.8, linestyle=(0, (3, 3))
    )

    metric_lines = [
        (
            f"T1 AUPRC {summary['Test 1']['auprc']:.3f} / base {summary['Test 1']['prevalence']:.3f}",
            f"T2 {summary['Test 2']['auprc']:.3f} / {summary['Test 2']['prevalence']:.3f}",
        ),
        (
            f"T1 AUC {summary['Test 1']['auc']:.3f}",
            f"T2 {summary['Test 2']['auc']:.3f}",
        ),
        (
            f"T1 Brier {summary['Test 1']['brier']:.3f} · ECE {summary['Test 1']['ece15']:.3f}",
            f"T2 Brier {summary['Test 2']['brier']:.3f} · ECE {summary['Test 2']['ece15']:.3f}",
        ),
    ]
    titles = ["Precision–recall", "ROC curve", "Probability calibration"]
    labels = ["A", "B", "C"]
    for axis, title, label, metrics in zip(axes, titles, labels, metric_lines, strict=True):
        axis.set_title(title, loc="left", fontsize=8.6, pad=25)
        stacked_metrics = title == "Probability calibration"
        axis.text(
            0.0,
            1.025,
            metrics[0],
            transform=axis.transAxes,
            color=COLORS["blue"],
            fontsize=6.0,
            fontweight="bold",
            ha="left",
        )
        axis.text(
            0.0 if stacked_metrics else 1.0,
            0.965 if stacked_metrics else 1.025,
            metrics[1],
            transform=axis.transAxes,
            color=COLORS["orange"],
            fontsize=6.0,
            fontweight="bold",
            ha="left" if stacked_metrics else "right",
        )
        axis.text(
            -0.075,
            1.205,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            color=COLORS["teal"],
            ha="left",
            va="top",
            clip_on=False,
        )
        clean_axis(axis, grid_axis="both")

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[2].set_xlim(0, 0.4)
    axes[2].set_ylim(0, 0.4)
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed drop frequency")

    save_figure(fig, output)
    return summary


def build_regression_report_figure(
    blocks: list[dict[str, object]], output: Path
) -> dict[str, object]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.82),
        gridspec_kw={
            "left": 0.08,
            "right": 0.985,
            "top": 0.80,
            "bottom": 0.20,
            "wspace": 0.34,
            "width_ratios": [1.0, 1.0, 1.15],
        },
    )
    colors = {"Test 1": COLORS["blue"], "Test 2": COLORS["orange"]}
    density_cmap = LinearSegmentedColormap.from_list(
        "skyfare_regression_density",
        ["#F3F7F8", COLORS["blue_light"], COLORS["blue"], COLORS["navy"]],
    )
    all_prices = np.concatenate(
        [
            block["regression"][["predicted_price_vnd", "target_session_price_vnd"]]
            .to_numpy(dtype=float)
            .ravel()
            for block in blocks
        ]
    )
    display_limit = float(np.quantile(all_prices, 0.999))
    summary: dict[str, dict[str, object]] = {}

    for axis, block, label in zip(axes[:2], blocks, ["A", "B"], strict=True):
        frame = block["regression"]
        observed = frame["target_session_price_vnd"].to_numpy(dtype=float)
        predicted = frame["predicted_price_vnd"].to_numpy(dtype=float)
        axis.hexbin(
            observed / 1_000_000,
            predicted / 1_000_000,
            gridsize=43,
            mincnt=1,
            bins="log",
            cmap=density_cmap,
            extent=(0, display_limit / 1_000_000, 0, display_limit / 1_000_000),
            linewidths=0,
            rasterized=True,
        )
        axis.plot(
            [0, display_limit / 1_000_000],
            [0, display_limit / 1_000_000],
            color=COLORS["red"],
            linewidth=0.85,
            linestyle=(0, (3, 3)),
        )
        report_metrics = block["report"]["point"]["model"]
        r_squared = float(r2_score(observed, predicted))
        summary[block["block"]] = {
            "rows": int(len(frame)),
            "r_squared": r_squared,
            "mape": float(report_metrics["mape"]),
            "mae_vnd": float(report_metrics["mae"]),
            "rmse_vnd": float(report_metrics["rmse"]),
        }
        axis.set_title(block["block"], loc="left", fontsize=8.6, pad=25)
        axis.text(
            0.0,
            1.025,
            f"R² {r_squared:.3f}  ·  MAPE {report_metrics['mape']:.2%}",
            transform=axis.transAxes,
            color=colors[block["block"]],
            fontsize=6.3,
            fontweight="bold",
            ha="left",
        )
        axis.text(
            0.98,
            0.05,
            "Identity",
            transform=axis.transAxes,
            color=COLORS["red"],
            fontsize=5.8,
            ha="right",
        )
        axis.set_xlim(0, display_limit / 1_000_000)
        axis.set_ylim(0, display_limit / 1_000_000)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Observed fare (million VND)")
        axis.set_ylabel("Predicted fare (million VND)")
        axis.text(
            -0.075,
            1.235,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            color=COLORS["teal"],
            ha="left",
            va="top",
            clip_on=False,
        )
        clean_axis(axis, grid_axis="both")

    positions = np.arange(len(BOOKING_WINDOWS))
    dud_summary: dict[str, dict[str, list[float]]] = {}
    for block in blocks:
        frame = block["regression"].copy()
        frame["absolute_percentage_error"] = (
            (frame["predicted_price_vnd"] - frame["target_session_price_vnd"]).abs()
            / frame["target_session_price_vnd"]
        )
        grouped = (
            frame.groupby("query_dud", observed=True)["absolute_percentage_error"]
            .agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
            )
            .reindex(BOOKING_WINDOWS)
        )
        color = colors[block["block"]]
        median = 100 * grouped["median"].to_numpy(dtype=float)
        q25 = 100 * grouped["q25"].to_numpy(dtype=float)
        q75 = 100 * grouped["q75"].to_numpy(dtype=float)
        axes[2].fill_between(positions, q25, q75, color=color, alpha=0.065, linewidth=0)
        axes[2].plot(
            positions,
            median,
            color=color,
            linewidth=1.55,
            marker="o",
            markersize=2.7,
            label=block["block"],
        )
        dud_summary[block["block"]] = {
            "median_ape_percent": [float(value) for value in median],
            "q25_ape_percent": [float(value) for value in q25],
            "q75_ape_percent": [float(value) for value in q75],
        }

    axes[2].set_title("Error by booking window", loc="left", fontsize=8.6, pad=25)
    axes[2].text(
        0.0,
        1.025,
        "Line: median APE",
        transform=axes[2].transAxes,
        color=COLORS["muted"],
        fontsize=6.1,
        ha="left",
    )
    axes[2].text(
        1.0,
        1.025,
        "Band: interquartile range",
        transform=axes[2].transAxes,
        color=COLORS["muted"],
        fontsize=6.1,
        ha="right",
    )
    axes[2].set_xticks(positions, [str(value) for value in BOOKING_WINDOWS])
    axes[2].set_xlabel("Days until departure")
    axes[2].set_ylabel("Absolute percentage error (%)")
    axes[2].legend(frameon=False, fontsize=6.6, ncol=2, loc="upper left")
    axes[2].text(
        -0.075,
        1.235,
        "C",
        transform=axes[2].transAxes,
        fontsize=10,
        fontweight="bold",
        color=COLORS["teal"],
        ha="left",
        va="top",
        clip_on=False,
    )
    clean_axis(axes[2])

    summary["booking_window_error"] = {
        "booking_windows": BOOKING_WINDOWS,
        "series": dud_summary,
    }
    summary["display_axis_cap_vnd"] = display_limit
    summary["display_axis_quantile"] = 0.999
    save_figure(fig, output)
    return summary


def build_distribution_ranking_figure(
    blocks: list[dict[str, object]], output: Path
) -> dict[str, object]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.78),
        gridspec_kw={
            "left": 0.075,
            "right": 0.985,
            "top": 0.76,
            "bottom": 0.21,
            "wspace": 0.38,
            "width_ratios": [1.0, 1.08, 1.0],
        },
    )
    labels = [block["block"] for block in blocks]
    x = np.arange(len(labels), dtype=float)
    width = 0.27

    wis_baseline = np.array(
        [
            metric_at(
                block["report"],
                ("distribution", "baseline_empirical_dev_residual", "wis"),
            )
            for block in blocks
        ]
    )
    wis_finalist = np.array(
        [metric_at(block["report"], ("distribution", "model", "wis")) for block in blocks]
    )
    ndcg_baseline = np.array(
        [metric_at(block["report"], ("ranking", "baseline_prior_anchor", "ndcg5")) for block in blocks]
    )
    ndcg_finalist = np.array(
        [metric_at(block["report"], ("ranking", "model", "ndcg5")) for block in blocks]
    )
    nominal = np.array([50.0, 80.0, 90.0])
    coverage = {
        block["block"]: np.array(
            [
                100.0
                * metric_at(
                    block["report"],
                    ("distribution", "model", f"coverage{level}"),
                )
                for level in [50, 80, 90]
            ]
        )
        for block in blocks
    }

    if not np.all(wis_finalist < wis_baseline):
        raise RuntimeError("Distribution figure requires finalist WIS below baseline in both tests")
    if not np.all(ndcg_finalist > ndcg_baseline):
        raise RuntimeError("Ranking figure requires finalist NDCG@5 above baseline in both tests")

    distribution_weights = np.array([95_962.0, 94_127.0])
    ranking_weights = np.array(
        [metric_at(block["report"], ("ranking", "model", "queries")) for block in blocks]
    )
    pooled_wis_gain = 100.0 * (
        1.0
        - np.average(wis_finalist, weights=distribution_weights)
        / np.average(wis_baseline, weights=distribution_weights)
    )
    pooled_ndcg_gain = 100.0 * (
        np.average(ndcg_finalist, weights=ranking_weights)
        / np.average(ndcg_baseline, weights=ranking_weights)
        - 1.0
    )

    def paired_bars(
        axis: plt.Axes,
        baseline: np.ndarray,
        finalist: np.ndarray,
        *,
        ylabel: str,
        upper: float,
        decimals: int,
    ) -> None:
        baseline_bars = axis.bar(
            x - width / 1.8,
            baseline,
            width,
            color=COLORS["gray_light"],
            edgecolor=COLORS["gray"],
            linewidth=0.8,
            zorder=3,
        )
        finalist_bars = axis.bar(
            x + width / 1.8,
            finalist,
            width,
            color=COLORS["teal"],
            edgecolor=COLORS["teal"],
            linewidth=0.8,
            zorder=3,
        )
        for bars, color in [
            (baseline_bars, COLORS["muted"]),
            (finalist_bars, COLORS["teal"]),
        ]:
            for bar in bars:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + upper * 0.025,
                    f"{bar.get_height():.{decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.1,
                    fontweight="bold",
                    color=color,
                )
        axis.set_xticks(x, labels)
        axis.set_ylim(0, upper)
        axis.set_ylabel(ylabel)
        clean_axis(axis, grid_axis="y")

    paired_bars(
        axes[0],
        wis_baseline,
        wis_finalist,
        ylabel="Weighted interval score",
        upper=float(wis_baseline.max() * 1.34),
        decimals=3,
    )
    axes[0].set_title("Distributional accuracy", loc="left", fontsize=8.6, pad=27)
    axes[0].text(
        0.0,
        1.025,
        f"Pooled WIS reduction {pooled_wis_gain:.1f}%",
        transform=axes[0].transAxes,
        color=COLORS["teal"],
        fontsize=6.1,
        fontweight="bold",
    )

    axes[1].plot(
        [45, 93],
        [45, 93],
        color=COLORS["gray"],
        linewidth=0.85,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    test_colors = {"Test 1": COLORS["blue"], "Test 2": COLORS["orange"]}
    for label in labels:
        axes[1].plot(
            nominal,
            coverage[label],
            color=test_colors[label],
            linewidth=1.45,
            marker="o",
            markersize=3.6,
            zorder=3,
        )
        axes[1].text(
            nominal[-1] + 0.8,
            coverage[label][-1] + (1.15 if label == "Test 1" else -1.35),
            label,
            color=test_colors[label],
            fontsize=6.2,
            fontweight="bold",
            ha="left",
            va="center",
        )
    max_coverage_error = max(
        float(np.max(np.abs(observed - nominal))) for observed in coverage.values()
    )
    axes[1].set_xlim(45, 96)
    axes[1].set_ylim(45, 94)
    axes[1].set_xticks([50, 60, 70, 80, 90])
    axes[1].set_yticks([50, 60, 70, 80, 90])
    axes[1].set_xlabel("Nominal coverage (%)")
    axes[1].set_ylabel("Observed coverage (%)")
    axes[1].set_title("Prediction-interval calibration", loc="left", fontsize=8.6, pad=27)
    axes[1].text(
        0.0,
        1.025,
        f"Maximum absolute deviation {max_coverage_error:.1f} pp",
        transform=axes[1].transAxes,
        color=COLORS["muted"],
        fontsize=6.1,
        fontweight="bold",
    )
    clean_axis(axes[1], grid_axis="both")

    paired_bars(
        axes[2],
        ndcg_baseline,
        ndcg_finalist,
        ylabel="NDCG@5",
        upper=1.0,
        decimals=3,
    )
    axes[2].set_title("Offer-ranking quality", loc="left", fontsize=8.6, pad=27)
    axes[2].text(
        0.0,
        1.025,
        f"Pooled NDCG@5 gain {pooled_ndcg_gain:.1f}%",
        transform=axes[2].transAxes,
        color=COLORS["teal"],
        fontsize=6.1,
        fontweight="bold",
    )

    for axis, label in zip(axes, ["A", "B", "C"], strict=True):
        axis.text(
            -0.075,
            1.205,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            color=COLORS["teal"],
            ha="left",
            va="top",
            clip_on=False,
        )

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS["gray_light"],
                edgecolor=COLORS["gray"],
                label="Baseline",
            ),
            Patch(facecolor=COLORS["teal"], edgecolor=COLORS["teal"], label="Finalist"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=6.6,
        handlelength=1.1,
        columnspacing=1.3,
    )

    summary = {
        "distribution": {
            label: {
                "baseline_wis": float(baseline),
                "finalist_wis": float(finalist),
                "coverage_percent": {
                    str(level): float(value)
                    for level, value in zip([50, 80, 90], coverage[label], strict=True)
                },
            }
            for label, baseline, finalist in zip(labels, wis_baseline, wis_finalist, strict=True)
        },
        "ranking": {
            label: {
                "baseline_ndcg5": float(baseline),
                "finalist_ndcg5": float(finalist),
            }
            for label, baseline, finalist in zip(labels, ndcg_baseline, ndcg_finalist, strict=True)
        },
        "pooled_wis_reduction_percent": float(pooled_wis_gain),
        "pooled_ndcg5_gain_percent": float(pooled_ndcg_gain),
        "maximum_absolute_coverage_error_pp": float(max_coverage_error),
    }
    save_figure(fig, output)
    return summary


def build_evaluation_figure_v2(
    blocks: list[dict[str, object]], output: Path
) -> dict[str, object]:
    skills = {
        "Test 1": relative_skill(blocks[0]["report"]),
        "Test 2": relative_skill(blocks[1]["report"]),
        "Pooled": pooled_skill(blocks),
    }
    task_specs = [
        {
            "key": "Classification\nBrier score",
            "title": "Classification\nBrier score",
            "model": ("classification", "model", "brier"),
            "baseline": ("classification", "baseline_constant_dev_rate", "brier"),
            "weights": [block["rows"] for block in blocks],
            "direction": "lower",
        },
        {
            "key": "Regression\nMAPE",
            "title": "Regression\nMAPE",
            "model": ("point", "model", "mape"),
            "baseline": ("point", "baseline_prior_anchor", "mape"),
            "weights": [95_962, 94_127],
            "direction": "lower",
        },
        {
            "key": "Distribution\nWIS",
            "title": "Distribution\nWIS",
            "model": ("distribution", "model", "wis"),
            "baseline": ("distribution", "baseline_empirical_dev_residual", "wis"),
            "weights": [95_962, 94_127],
            "direction": "lower",
        },
        {
            "key": "Ranking\nNDCG@5",
            "title": "Ranking\nNDCG@5",
            "model": ("ranking", "model", "ndcg5"),
            "baseline": ("ranking", "baseline_prior_anchor", "ndcg5"),
            "weights": [block["report"]["ranking"]["model"]["queries"] for block in blocks],
            "direction": "higher",
        },
    ]
    raw_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for spec in task_specs:
        raw_metrics[spec["key"]] = {}
        baselines = []
        finalists = []
        for block in blocks:
            baseline = metric_at(block["report"], spec["baseline"])
            finalist = metric_at(block["report"], spec["model"])
            baselines.append(baseline)
            finalists.append(finalist)
            raw_metrics[spec["key"]][block["block"]] = {
                "baseline": baseline,
                "finalist": finalist,
            }
        raw_metrics[spec["key"]]["Pooled"] = {
            "baseline": float(np.average(baselines, weights=spec["weights"])),
            "finalist": float(np.average(finalists, weights=spec["weights"])),
            "improvement_percent": float(skills["Pooled"][spec["key"]]),
        }

    fig = plt.figure(figsize=(7.25, 5.35), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.86, 1.16],
        hspace=0.42,
        left=0.09,
        right=0.985,
        top=0.88,
        bottom=0.10,
    )
    top = outer[0].subgridspec(1, 4, wspace=0.42)
    mini_axes = [fig.add_subplot(top[0, index]) for index in range(4)]
    bottom = outer[1].subgridspec(1, 3, wspace=0.42)
    calibration_ax = fig.add_subplot(bottom[0, 0])
    coverage_ax = fig.add_subplot(bottom[0, 1])
    policy_ax = fig.add_subplot(bottom[0, 2])
    series_colors = {"Test 1": COLORS["blue"], "Test 2": COLORS["orange"], "Pooled": COLORS["ink"]}

    for axis, spec in zip(mini_axes, task_specs, strict=True):
        task = raw_metrics[spec["key"]]
        all_values = [
            task[block][field]
            for block in ["Test 1", "Test 2"]
            for field in ["baseline", "finalist"]
        ]
        span = max(all_values) - min(all_values)
        pad = max(span * 0.24, max(all_values) * 0.018)
        axis.set_xlim(min(all_values) - pad, max(all_values) + pad)
        for y, block_name in zip([1, 0], ["Test 1", "Test 2"], strict=True):
            baseline = task[block_name]["baseline"]
            finalist = task[block_name]["finalist"]
            color = series_colors[block_name]
            axis.plot([baseline, finalist], [y, y], color=color, linewidth=2.0, alpha=0.82, zorder=2)
            axis.scatter(
                baseline,
                y,
                s=31,
                facecolor="white",
                edgecolor=color,
                linewidth=1.35,
                zorder=3,
            )
            axis.scatter(finalist, y, s=31, color=color, edgecolor="white", linewidth=0.45, zorder=4)
            axis.annotate(
                f"{baseline:.3f}",
                (baseline, y),
                xytext=(0, -11),
                textcoords="offset points",
                color=COLORS["muted"],
                fontsize=6.4,
                ha="center",
                va="top",
            )
            axis.annotate(
                f"{finalist:.3f}",
                (finalist, y),
                xytext=(0, 8),
                textcoords="offset points",
                color=color,
                fontsize=6.5,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
        arrow_start, arrow_end = (
            ((0.82, 0.89), (0.18, 0.89))
            if spec["direction"] == "lower"
            else ((0.18, 0.89), (0.82, 0.89))
        )
        axis.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 0.75},
        )
        axis.text(
            0.5,
            0.92,
            f"{spec['direction']} is better",
            transform=axis.transAxes,
            color=COLORS["muted"],
            fontsize=6.3,
            ha="center",
            va="bottom",
        )
        axis.text(
            0.5,
            -0.24,
            f"Pooled improvement  {skills['Pooled'][spec['key']]:.1f}%",
            transform=axis.transAxes,
            color=COLORS["ink"],
            fontsize=6.5,
            fontweight="bold",
            ha="center",
            va="top",
        )
        axis.set_title(spec["title"], fontsize=8.2, pad=8)
        axis.set_ylim(-0.42, 1.42)
        axis.set_yticks([1, 0], ["Test 1", "Test 2"])
        axis.tick_params(axis="x", labelsize=6.6)
        axis.tick_params(axis="y", labelsize=6.8, length=0)
        axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        clean_axis(axis, grid_axis="x")
        axis.spines["left"].set_visible(False)

    for block, color in zip(blocks, [COLORS["blue"], COLORS["orange"]], strict=True):
        reliability = block["reliability"]
        calibration_ax.plot(
            reliability["predicted"],
            reliability["observed"],
            marker="o",
            markersize=3.0,
            linewidth=1.35,
            color=color,
        )
        endpoint = reliability.iloc[-1]
        calibration_ax.annotate(
            block["block"],
            xy=(endpoint["predicted"], endpoint["observed"]),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=6.8,
            fontweight="bold",
            ha="left",
            va="center",
        )
    calibration_ax.plot(
        [0, 0.4], [0, 0.4], color=COLORS["gray"], linewidth=0.85, linestyle=(0, (3, 3))
    )
    calibration_ax.text(
        0.385, 0.37, "Perfect", color=COLORS["muted"], fontsize=6.5, ha="right", rotation=38
    )
    calibration_ax.set_xlim(0, 0.4)
    calibration_ax.set_ylim(0, 0.4)
    calibration_ax.set_xlabel("Mean predicted probability")
    calibration_ax.set_ylabel("Observed drop frequency")
    clean_axis(calibration_ax, grid_axis="both")

    nominal = np.array([0.5, 0.8, 0.9])
    pooled_coverages = []
    coverage_by_series: dict[str, np.ndarray] = {}
    for block in blocks:
        observed = np.array(
            [block["report"]["distribution"]["model"][f"coverage{level}"] for level in [50, 80, 90]]
        )
        coverage_by_series[block["block"]] = observed
        pooled_coverages.append(observed)
    pooled_coverage = np.average(np.vstack(pooled_coverages), axis=0, weights=np.array([95_962, 94_127]))
    coverage_by_series["Pooled"] = pooled_coverage
    coverage_y = np.arange(len(nominal))
    coverage_offsets = {"Test 1": -0.18, "Test 2": 0.0, "Pooled": 0.18}
    for label in ["Test 1", "Test 2", "Pooled"]:
        errors = (coverage_by_series[label] - nominal) * 100
        marker = "D" if label == "Pooled" else "o"
        coverage_ax.scatter(
            errors,
            coverage_y + coverage_offsets[label],
            s=27,
            marker=marker,
            color=series_colors[label],
            zorder=3,
        )
    pooled_errors = (pooled_coverage - nominal) * 100
    for index, error in enumerate(pooled_errors):
        formatted = f"{error:+.2f}".replace("-", "−")
        coverage_ax.text(
            error + 0.12,
            coverage_y[index] + coverage_offsets["Pooled"],
            f"{formatted} pp",
            color=COLORS["ink"],
            fontsize=6.5,
            va="center",
        )
    for name, offset in [("T1", -0.18), ("T2", 0.0), ("Pooled", 0.18)]:
        coverage_ax.text(
            -3.8,
            coverage_y[0] + offset,
            name,
            color=series_colors[{"T1": "Test 1", "T2": "Test 2"}.get(name, name)],
            fontsize=6.6,
            fontweight="bold",
            va="center",
        )
    coverage_ax.axvline(0, color=COLORS["line"], linewidth=0.9, alpha=0.75)
    coverage_ax.set_xlim(-4.1, 1.0)
    coverage_ax.set_yticks(coverage_y, ["50% interval", "80% interval", "90% interval"])
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlabel("Observed − nominal coverage (pp)")
    clean_axis(coverage_ax, grid_axis="x")

    policy = []
    for block in blocks:
        item = block["report"]["buy_wait"]["paired_day_bootstrap_vs_buy_all"]
        policy.append(
            {
                "name": block["block"],
                "effect": -float(item["mean_model_minus_baseline"]),
                "low": -float(item["ci95_high"]),
                "high": -float(item["ci95_low"]),
            }
        )
    policy.append({"name": "Pooled", "effect": 724.50, "low": -187.37, "high": 1671.25})
    for index, item in enumerate(policy):
        color = [COLORS["blue"], COLORS["orange"], COLORS["ink"]][index]
        policy_ax.errorbar(
            item["effect"],
            index,
            xerr=[[item["effect"] - item["low"]], [item["high"] - item["effect"]]],
            fmt="o",
            markersize=5.0,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3,
            zorder=3,
        )
        policy_ax.text(
            5000,
            index,
            f"{item['effect']:,.0f} [{item['low']:,.0f}, {item['high']:,.0f}]",
            color=COLORS["ink"],
            fontsize=6.3,
            ha="right",
            va="center",
        )
    policy_ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    policy_ax.set_yticks(np.arange(len(policy)), [item["name"] for item in policy])
    policy_ax.invert_yaxis()
    policy_ax.set_xlim(-1050, 5200)
    policy_ax.set_xlabel("Regret reduction vs always buy (VND)")
    clean_axis(policy_ax, grid_axis="x")

    first_position = mini_axes[0].get_position()
    fig.text(
        first_position.x0 - 0.032,
        0.955,
        "A",
        color=COLORS["teal"],
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(first_position.x0, 0.955, "Baseline-to-finalist performance", fontsize=9.5, fontweight="bold")
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="white",
                markeredgecolor=COLORS["muted"], markeredgewidth=1.2, label="Baseline",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["muted"],
                markeredgecolor="white", markeredgewidth=0.4, label="Finalist",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.94, 0.977),
        frameon=False,
        ncol=2,
        fontsize=6.8,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    for axis, label, title in [
        (calibration_ax, "B", "Probability calibration"),
        (coverage_ax, "C", "Coverage error"),
        (policy_ax, "D", "Policy regret reduction"),
    ]:
        position = axis.get_position()
        fig.text(
            position.x0 - 0.032,
            position.y1 + 0.014,
            label,
            color=COLORS["teal"],
            fontsize=9.5,
            fontweight="bold",
        )
        fig.text(position.x0, position.y1 + 0.014, title, fontsize=9.5, fontweight="bold")

    save_figure(fig, output)
    calibration_summary = {
        block["block"]: {
            "rows": int(block["rows"]),
            "bins": 6,
            "binning": "equal-frequency",
            "rows_per_bin": [int(value) for value in block["reliability"]["rows"]],
        }
        for block in blocks
    }
    policy_rows = [int(block["report"]["buy_wait"]["frozen"]["rows"]) for block in blocks]
    pooled_wait_rate = float(
        np.average(
            [float(block["report"]["buy_wait"]["frozen"]["wait_rate"]) for block in blocks],
            weights=policy_rows,
        )
    )
    return {
        "baseline_to_finalist_raw_metrics": raw_metrics,
        "relative_skill_percent": skills,
        "relative_skill_definition": {
            "lower_is_better": "100 * (1 - finalist / baseline)",
            "higher_is_better": "100 * (finalist / baseline - 1)",
            "uncertainty": "Panel A reports descriptive raw point estimates; bootstrap inference remains in the result tables",
        },
        "classification_calibration": calibration_summary,
        "pooled_coverage": {
            str(level): float(value) for level, value in zip([50, 80, 90], pooled_coverage, strict=True)
        },
        "pooled_coverage_error_percentage_points": {
            str(level): float(error) for level, error in zip([50, 80, 90], pooled_errors, strict=True)
        },
        "policy_regret_reduction_vnd": policy,
        "policy_assumptions": {
            "baseline": "always buy",
            "bootstrap_draws": 4000,
            "friction_vnd": 50000,
            "wait_threshold": 0.30,
            "pooled_wait_rate": pooled_wait_rate,
        },
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    release_assets = root / "artifacts" / "release_assets"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production-archive",
        type=Path,
        default=release_assets / "SKYFARE_128_DAY_FINAL_PRODUCTION_MODELS_R1.tar.gz",
    )
    parser.add_argument(
        "--test1-archive",
        type=Path,
        default=release_assets / "SKYFARE_TEST_1_EVALUATION_RESULTS_R2.tar.gz",
    )
    parser.add_argument(
        "--test2-archive",
        type=Path,
        default=release_assets / "SKYFARE_TEST_2_EVALUATION_RESULTS_R1.tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reports" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    for archive in [args.production_archive, args.test1_archive, args.test2_archive]:
        if not archive.is_file():
            raise FileNotFoundError(archive)

    with tempfile.TemporaryDirectory(prefix="skyfare-results-figures-") as temporary:
        temp = Path(temporary)
        source_path = extract_member(
            args.production_archive,
            "skyfare_128_day_final_production_refit_r1/inputs/standard_128_day_offers.parquet",
            temp / "source.parquet",
        )
        source = pd.read_parquet(
            source_path,
            columns=[
                "offer_id",
                "session_date",
                "route",
                "airline",
                "collection_era",
                "days_until_departure",
                "session_label",
                "price_vnd",
            ],
        )

        blocks = []
        for number, archive, prefix in [
            (1, args.test1_archive, "test1_outputs"),
            (2, args.test2_archive, "test2_outputs"),
        ]:
            report = json.loads(
                read_member_bytes(archive, f"{prefix}/analysis/TEST_{number}_REPORT.json")
            )
            prediction_path = extract_member(
                archive,
                f"{prefix}/frozen_predictions/classification_test{number}_predictions.parquet",
                temp / f"classification_predictions_test{number}.parquet",
            )
            label_path = extract_member(
                archive,
                f"{prefix}/staging/classification_labels.parquet",
                temp / f"classification_labels_test{number}.parquet",
            )
            point_prediction_path = extract_member(
                archive,
                f"{prefix}/frozen_predictions/point_test{number}_predictions.parquet",
                temp / f"point_predictions_test{number}.parquet",
            )
            regression_label_path = extract_member(
                archive,
                f"{prefix}/staging/regression_labels.parquet",
                temp / f"regression_labels_test{number}.parquet",
            )
            blocks.append(
                load_test_block(
                    f"Test {number}",
                    report,
                    pd.read_parquet(prediction_path),
                    pd.read_parquet(label_path),
                    pd.read_parquet(point_prediction_path),
                    pd.read_parquet(regression_label_path),
                )
            )

        classification_diagnostics = build_classification_report_figure(
            blocks,
            args.output_dir / "classification_prospective_diagnostics",
        )
        regression_diagnostics = build_regression_report_figure(
            blocks,
            args.output_dir / "regression_prospective_diagnostics",
        )
        distribution_ranking_diagnostics = build_distribution_ranking_figure(
            blocks,
            args.output_dir / "distribution_ranking_prospective_diagnostics",
        )
        summary = {
            "eda": build_eda_figure_v2(
                source,
                args.output_dir / "airline_composition_temporal_drift",
            ),
            "evaluation": build_evaluation_figure_v2(
                blocks,
                args.output_dir / "prospective_evaluation_decision_utility",
            ),
        }
        summary["evaluation"]["classification_diagnostics"] = classification_diagnostics
        summary["evaluation"]["regression_diagnostics"] = regression_diagnostics
        summary["evaluation"][
            "distribution_ranking_diagnostics"
        ] = distribution_ranking_diagnostics
        validate_summary(summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "figure_data_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(f"[RESULT FIGURES PASS] output={args.output_dir}")


if __name__ == "__main__":
    main()
