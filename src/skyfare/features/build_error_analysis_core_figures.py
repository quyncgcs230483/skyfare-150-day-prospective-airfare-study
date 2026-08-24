"""Build six curated old-system Error Analysis figures from locked evidence.

No model fit. Five figures come from saved CSV evidence. F1 keeps only two
already-rendered panels from the locked 90days figure; canonical notebook code
now regenerates those panels directly when full audit is rerun.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "artifacts/feature_research/error_analysis/tables"
FIGURES = ROOT / "artifacts/feature_research/error_analysis/figures"
MANIFEST = ROOT / "artifacts/feature_research/tables/error_analysis_core_figure_manifest.json"

SAME_HISTORY = {
    "price_lag_1step",
    "price_rolling_mean_7d",
    "price_rolling_mean_14d",
    "elasticity_signal",
    "lag_hours",
}
CROSS_SECTION = {"competition_price_diff"}
COLORS = {
    "static": "#4C78A8",
    "same-schedule history": "#E45756",
    "current-batch market": "#72B7B2",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curate_f1() -> str:
    path = FIGURES / "main_F1_offline_vs_at_serve_pred_actual.png"
    image = mpimg.imread(path)
    height, width = image.shape[:2]
    if width < 2200:
        # Idempotently correct the legacy panel title. The underlying points and
        # locked predictions are unchanged; only the semantic label is repaired.
        pil = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(pil)
        draw.rectangle((1110, 52, 2135, 155), fill="white")
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 25)
        lines = [
            "AT SERVE - no same-schedule history",
            "competitor context kept (R2=0.45)",
        ]
        y = 70
        for line in lines:
            box = draw.textbbox((0, 0), line, font=font)
            x = 1622 - (box[2] - box[0]) / 2
            draw.text((x, y), line, fill="#262626", font=font)
            y += 31
        pil.save(path)
        return "cleaned_presentation_crop_of_locked_two_valid_panels"
    if width < 3000:
        # Clean the first presentation crop without touching locked predictions.
        # The prior crop retained part of the old three-panel title and annotation.
        cropped = image[82:, :]
        fig, ax = plt.subplots(figsize=(14, 5.2))
        ax.imshow(cropped)
        ax.add_patch(
            Rectangle(
                (1240, 145),
                570,
                100,
                facecolor="white",
                edgecolor="#C44E1A",
                linewidth=1.5,
                zorder=4,
            )
        )
        ax.text(
            1255,
            182,
            "Cold-start predictions flatten\nwithout same-schedule history",
            color="#A63D12",
            fontsize=11,
            va="center",
            zorder=5,
        )
        ax.axis("off")
        fig.suptitle(
            "F1. Removing unavailable same-schedule history exposes cold-start loss",
            fontsize=15,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.91)
        save(fig, path.name)
        return "cleaned_presentation_crop_of_locked_two_valid_panels"
    cropped = image[max(35, int(height * 0.035)) :, : int(width * 2 / 3)]
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.imshow(cropped)
    ax.axis("off")
    fig.suptitle(
        "F1. Removing unavailable same-schedule history exposes cold-start loss",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    save(fig, path.name)
    return "presentation_crop_of_locked_two_valid_panels"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")

    headline = pd.read_csv(TABLES / "e_headline_failure_ranking.csv")
    headline = headline[~headline["claim_scope"].eq("stress test only")].copy()
    headline["failure_mechanism"] = headline["failure_mechanism"].replace(
        {"Honest cold-start: same-flight history unavailable": "Honest cold-start: same-schedule history unavailable"}
    )
    headline = headline.sort_values("r2_drop")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    scope_colors = {
        "serve-time claim": "#D55E00",
        "observed first-contact slice": "#CC79A7",
        "temporal robustness": "#0072B2",
    }
    bars = ax.barh(
        headline["failure_mechanism"],
        headline["r2_drop"],
        color=headline["claim_scope"].map(scope_colors),
    )
    for bar, value in zip(bars, headline["r2_drop"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.set_title("F0. Cold-start damage is almost 9x measured temporal-drift damage", fontweight="bold")
    ax.set_xlabel("R2 loss relative to offline evaluation")
    ax.set_ylabel("")
    save(fig, "main_F0_headline_failure_ranking.png")

    f1_method = curate_f1()

    availability = pd.read_csv(TABLES / "e_main_history_feature_availability.csv")
    order = (
        availability[availability["contact"].eq("First contact")]
        .sort_values("available_pct")["feature"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(data=availability, y="feature", x="available_pct", hue="contact", order=order, ax=ax)
    ax.set_title("F3. Five persistence features disappear at first contact", fontweight="bold")
    ax.set_xlabel("Rows where feature is available (%)")
    ax.set_ylabel("")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False)
    save(fig, "main_F3_first_contact_feature_availability.png")

    fixability = pd.read_csv(TABLES / "e_fixability_database.csv")
    selected = fixability[
        fixability["milestone"].eq("90days")
        & ~fixability["strategy"].eq("all_economic_removed_stress")
    ].copy()
    order = [
        "offline_history_available",
        "at_serve_no_same_flight_history",
        "db_route_airline_history_proxy",
        "db_recent_comparable_anchor",
        "anchor_alone_reference",
    ]
    labels = {
        "offline_history_available": "Offline\nhistory",
        "at_serve_no_same_flight_history": "Cold\nno history",
        "db_route_airline_history_proxy": "DB route-airline\nproxy",
        "db_recent_comparable_anchor": "DB recent\nanchor fill",
        "anchor_alone_reference": "DB anchor\nalone",
    }
    selected = selected.set_index("strategy").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(np.arange(len(selected)), selected["r2"], color=["#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#009E73"])
    ax.set_xticks(np.arange(len(selected)), [labels[x] for x in order])
    for bar, value in zip(bars, selected["r2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center")
    ax.set_title("F4. Database proxies recover only part of lost cold-start signal", fontweight="bold")
    ax.set_ylabel("R2 on 90days test rows")
    ax.set_xlabel("")
    save(fig, "main_F4_serve_and_database_strategies.png")

    oot = pd.read_csv(TABLES / "e3_temporal_oot_retrain.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sns.barplot(data=oot, x="eval_short", y="reg_r2", color="#0072B2", ax=axes[0])
    sns.barplot(data=oot, x="eval_short", y="reg_mape_pct", color="#E69F00", ax=axes[1])
    axes[0].set_title("R2 remains above 0.76")
    axes[1].set_title("MAPE rises modestly")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("R2")
    axes[1].set_ylabel("MAPE (%)")
    fig.suptitle("F5. Temporal drift is real, but much smaller than cold-start loss", fontweight="bold")
    save(fig, "main_F5_temporal_oot_drift.png")

    importance = pd.read_csv(TABLES / "e4_feature_importance_90days.csv")
    milestone_importance = pd.concat(
        [pd.read_csv(path) for path in sorted(TABLES.glob("e4_feature_importance_*days.csv"))],
        ignore_index=True,
    )
    lag_importance = milestone_importance.loc[
        milestone_importance["feature"].eq("price_lag_1step"), "importance"
    ]
    importance["corrected_family"] = np.select(
        [
            importance["feature"].isin(SAME_HISTORY),
            importance["feature"].isin(CROSS_SECTION),
        ],
        ["same-schedule history", "current-batch market"],
        default="static",
    )
    importance = importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(
        importance["feature"],
        importance["importance"],
        color=importance["corrected_family"].map(COLORS),
    )
    same_sum = importance.loc[importance["corrected_family"].eq("same-schedule history"), "importance"].sum()
    cross_sum = importance.loc[importance["corrected_family"].eq("current-batch market"), "importance"].sum()
    ax.set_title("F6. Old RF relied mainly on warm-only persistence", fontweight="bold")
    ax.set_xlabel("RF impurity importance (diagnostic only)")
    ax.set_ylabel("")
    ax.text(
        0.99,
        0.03,
        f"Same-schedule history: {same_sum:.1%}\nCurrent-batch market: {cross_sum:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.95},
    )
    fig.text(
        0.5,
        0.01,
        (
            "90days illustration only. price_lag_1step impurity importance varies "
            f"{lag_importance.min():.2f}-{lag_importance.max():.2f} across 80/85/90days; "
            "F0 and F3 provide primary failure evidence."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.12)
    save(fig, "main_F6_corrected_feature_importance_90days.png")

    outputs = sorted(FIGURES.glob("main_F*.png"))
    manifest = {
        "status": "CURATED_SIX_FIGURES_NO_MODEL_FIT",
        "f1_materialization": f1_method,
        "figures": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in outputs
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] wrote {len(outputs)} curated figures")
    print(f"[PASS] manifest {MANIFEST}")


if __name__ == "__main__":
    main()
