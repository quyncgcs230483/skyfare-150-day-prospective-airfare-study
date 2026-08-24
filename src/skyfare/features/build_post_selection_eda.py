"""Generate branch-specific model-free EDA after Feature Selection.

Snapshot: new-build 110 pilot, cutoff 2026-07-10.
Branches: Relative, Will-Drop and Sequence.
Inputs: reproduced EA02/EA05/EA07 tables plus the locked pre-fit contract.
Outputs: five decision-oriented figures, one Markdown contract and one manifest.
No estimator is fitted by this script.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skyfare.features.audit_common import BOOKING_WINDOWS, CUTOFF, ROOT


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_ROOT = ROOT / "artifacts/feature_research/figures/branch_eda"
REL_FIG_DIR = FIG_ROOT / "relative"
WD_FIG_DIR = FIG_ROOT / "willdrop"
REPORT = ROOT / "artifacts/feature_research/reports/BRANCH_EDA_CONTRACT.txt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    REL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    WD_FIG_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        TABLE_DIR / "ea02_multi_horizon_coverage.csv",
        TABLE_DIR / "ea02_exact_next_transition_targets.csv",
        TABLE_DIR / "ea05_regression_causal_baselines.csv",
        TABLE_DIR / "ea05_willdrop_causal_baselines.csv",
        TABLE_DIR / "ea07_sequence_length_coverage.csv",
        TABLE_DIR / "fs07_prefit_contract_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Branch EDA requires reproduced pre-fit gates: {missing}")

    horizons = pd.read_csv(required[0])
    exact = pd.read_csv(required[1])
    regression = pd.read_csv(required[2])
    willdrop = pd.read_csv(required[3])
    sequence = pd.read_csv(required[4])
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Relative 01: target mask/coverage matrix.
    coverage = horizons.pivot(index="current_dud", columns="target_dud", values="coverage_of_mature_pct")
    coverage = coverage.reindex(index=BOOKING_WINDOWS[:-1], columns=BOOKING_WINDOWS[1:])
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    sns.heatmap(
        coverage,
        cmap="Blues",
        vmin=70,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Observed / mature targets (%)"},
        ax=ax,
    )
    ax.set_title("REL01. Horizon-specific masks are mandatory because target coverage is unequal", fontweight="bold")
    ax.set_xlabel("Target DUD")
    ax.set_ylabel("Current DUD")
    save(fig, REL_FIG_DIR / "REL01_horizon_specific_masks_are_mandatory.png")

    # Relative 02: strongest exact-next baselines by DUD transition.
    exact_reg = regression[regression["task"].eq("future_price_exact_next")].copy()
    exact_reg["transition"] = exact_reg["current_dud"].astype(int).astype(str) + "->" + exact_reg["target_dud"].astype(int).astype(str)
    baseline_labels = {
        "current_price_carry_forward": "Current price carry-forward",
        "causal_route_airline_target_dud_market": "Causal route-airline-target-DUD",
        "current_other_airline_minimum": "Other-airline minimum",
    }
    exact_reg["baseline_label"] = exact_reg["baseline"].map(baseline_labels)
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    sns.lineplot(data=exact_reg, x="transition", y="mape_pct", hue="baseline_label", marker="o", ax=ax)
    ax.set_title("REL02. Exact-next ML must beat current-price carry-forward on every DUD transition", fontweight="bold")
    ax.set_xlabel("Exact-next DUD transition")
    ax.set_ylabel("MAPE (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Causal baseline", frameon=True)
    style_axis(ax)
    save(fig, REL_FIG_DIR / "REL02_exact_next_must_beat_current_price_carry_forward.png")

    # Relative 03: h=0 cold versus warm baseline contract.
    h0 = regression[regression["task"].eq("relative_h0")].copy()
    h0_names = {
        "prior_route_airline_dud_market_median": "Prior market median (all)",
        "current_other_airline_minimum": "Other-airline minimum (all)",
        "previous_same_schedule_carry_forward": "Previous schedule price (warm)",
    }
    h0["baseline_label"] = h0["baseline"].map(h0_names)
    h0 = h0.sort_values("mape_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    colors = ["#0072B2" if population == "all" else "#D55E00" for population in h0["population"]]
    bars = ax.barh(h0["baseline_label"], h0["mape_pct"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in h0["mape_pct"]], padding=4)
    ax.set_xlim(0, max(h0["mape_pct"]) * 1.18)
    ax.set_title("REL03. Cold and warm offers require different h=0 baselines", fontweight="bold")
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("")
    style_axis(ax)
    save(fig, REL_FIG_DIR / "REL03_cold_and_warm_require_different_baselines.png")

    # Will-Drop 01: prevalence and usable label count by transition.
    exact = exact.sort_values("current_dud", ascending=False).copy()
    exact["transition"] = exact["current_dud"].astype(int).astype(str) + "->" + exact["target_dud"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    bars = ax.bar(exact["transition"], exact["wait_prevalence_pct"], color="#009E73")
    labels = [f"{rate:.0f}%\nn={int(rows):,}" for rate, rows in zip(exact["wait_prevalence_pct"], exact["observed_target_rows"])]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_ylim(0, max(exact["wait_prevalence_pct"]) * 1.28)
    ax.set_title("WD01. DUD-specific prevalence makes one global class threshold fragile", fontweight="bold")
    ax.set_xlabel("Exact-next DUD transition")
    ax.set_ylabel("WAIT prevalence (%)")
    ax.tick_params(axis="x", rotation=45)
    style_axis(ax)
    save(fig, WD_FIG_DIR / "WD01_dud_specific_prevalence_requires_stratified_policy_reporting.png")

    # Will-Drop 02: actual money consequences conditional on material moves.
    money = exact[["transition", "median_wait_savings_vnd", "median_rise_cost_vnd"]].melt(
        id_vars="transition", var_name="outcome", value_name="median_vnd"
    )
    money["outcome"] = money["outcome"].map(
        {
            "median_wait_savings_vnd": "Median saving when WAIT label is true",
            "median_rise_cost_vnd": "Median extra cost when price rises >=5%",
        }
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    sns.barplot(data=money, x="transition", y="median_vnd", hue="outcome", ax=ax)
    ax.set_title("WD02. Equal 5% labels hide unequal VND consequences across booking windows", fontweight="bold")
    ax.set_xlabel("Exact-next DUD transition")
    ax.set_ylabel("Median consequence (VND)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Observed outcome", frameon=True)
    style_axis(ax)
    save(fig, WD_FIG_DIR / "WD02_fixed_percentage_labels_hide_unequal_vnd_consequences.png")

    # A compact contract table: what each EDA block decides before model fitting.
    branch_contract = pd.DataFrame(
        [
            ["Relative", "REL01", "target mask", "report every horizon separately; no aggregate without coverage"],
            ["Relative", "REL02", "exact-next baseline", "compare ML against current-price carry-forward on matched rows"],
            ["Relative", "REL03", "population baseline", "cold and warm headline metrics must remain separate"],
            ["Will-Drop", "WD01", "class support", "report prevalence and n by DUD; baseline delta is within fold"],
            ["Will-Drop", "WD02", "money consequence", "5% remains label; policy also uses VND regret"],
            ["Sequence", "EA07", "lane and length", "market 7/14/21; schedule 7 sparse; schedule 14/21 rejected"],
        ],
        columns=["branch", "figure", "decision_axis", "locked_handoff"],
    )
    branch_contract.to_csv(TABLE_DIR / "branch_eda_decision_contract.csv", index=False)

    report_table = branch_contract.to_markdown(index=False)
    sequence_table = sequence.to_markdown(index=False)
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    REPORT.write_text(
        f"""# Branch EDA Contract After Feature Selection

> Status: `MODEL_FREE_BRANCH_EDA_COMPLETE`
>
> Created `{now}` at cutoff `{CUTOFF.date()}`. No estimator, calibrator or
> threshold was fitted. Archived 110/115 outcomes were not used.

## Why this EDA exists

This is not another raw-market overview. It validates the target masks,
baselines, population split and sequence products implied by the prediction
and feature contracts.

{report_table}

## Sequence structural evidence

{sequence_table}

## Training handoff

- Relative reports h=0, exact-next and longer horizons separately.
- Exact-next regression must beat current-price carry-forward on matched rows.
- Cold and warm populations keep separate baselines and metrics.
- Will-Drop reports within-DUD prevalence, baseline delta and VND consequence.
- Sequence adapters may not silently swap market sequences for same-schedule
  histories; they are different data products.
""",
        encoding="utf-8",
    )

    outputs = [
        REL_FIG_DIR / "REL01_horizon_specific_masks_are_mandatory.png",
        REL_FIG_DIR / "REL02_exact_next_must_beat_current_price_carry_forward.png",
        REL_FIG_DIR / "REL03_cold_and_warm_require_different_baselines.png",
        WD_FIG_DIR / "WD01_dud_specific_prevalence_requires_stratified_policy_reporting.png",
        WD_FIG_DIR / "WD02_fixed_percentage_labels_hide_unequal_vnd_consequences.png",
        TABLE_DIR / "branch_eda_decision_contract.csv",
        REPORT,
    ]
    manifest = {
        "status": "MODEL_FREE_BRANCH_EDA_COMPLETE",
        "created_at": now,
        "cutoff": str(CUTOFF.date()),
        "model_fit_called": False,
        "uses_archived_110_115_outcomes": False,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in required
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in outputs
        ],
    }
    (TABLE_DIR / "branch_eda_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[BRANCH EDA PASS] Relative=3 Will-Drop=2 Sequence=reused EA07; model_fit_called=false")


if __name__ == "__main__":
    main()
