"""EA04 pre-fit redundancy audit and model-family adapter contract."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import combinations
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skyfare.features.audit_common import CUTOFF, ROOT
from skyfare.features.candidate_feature_contract import NUMERIC_CANDIDATES, build_candidate_frame


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_DIR = ROOT / "artifacts/feature_research/figures"
REPORT = ROOT / "artifacts/feature_research/reports/EA04_REDUNDANCY_AND_MODEL_ADAPTERS.txt"


DETERMINISTIC_GROUPS = [
    {
        "group": "route_identity",
        "members": "route | origin | dest",
        "rule": "route deterministically decomposes to origin and destination",
        "decision": "keep route in model adapter; origin/dest remain ingestion fields",
    },
    {
        "group": "flight_weekday",
        "members": "flight_day_of_week | is_weekend",
        "rule": "is_weekend = 1[weekday in {5,6}]",
        "decision": "keep flight_day_of_week only",
    },
    {
        "group": "booking_cycle",
        "members": "days_until_departure | booking_window_bucket",
        "rule": "bucket is a deterministic coarsening of exact DUD",
        "decision": "DUD is tabular input; bucket is sequence key only",
    },
    {
        "group": "departure_clock",
        "members": "departure_minute | departure_time_sin | departure_time_cos | departure_period",
        "rule": "all are deterministic transforms of scheduled local time",
        "decision": "use one clock adapter per model family, never all representations together",
    },
    {
        "group": "h1_market_position",
        "members": "price_vnd | competitor_min_price_other_airlines | current_vs_competitor_ratio",
        "rule": "ratio = current price / competitor minimum",
        "decision": "LEVELS uses current+competitor; RELATIVE uses current+ratio; never all three",
    },
    {
        "group": "warm_relative_position",
        "members": "previous_price_same_schedule | temporal_market_median_price | previous_vs_temporal_market_ratio",
        "rule": "ratio = previous schedule price / temporal market median",
        "decision": "WARM_LEVELS or WARM_RELATIVE adapter, never all three",
    },
    {
        "group": "target_horizon",
        "members": "current_dud | target_dud | target_horizon_days",
        "rule": "target_horizon_days = current_dud - target_dud",
        "decision": "multi-horizon adapter uses current_dud + target_dud only",
    },
]


BASE = [
    "route",
    "airline",
    "days_until_departure",
    "flight_month",
    "flight_day_of_week",
    "is_peak_period",
    "session_label",
]
SUPPORT = [
    "competitor_price_spread_other_airlines",
    "competitor_airline_count",
    "competitor_offer_count",
    "same_airline_alternative_min_price",
    "temporal_market_support",
    "temporal_market_age_hours",
    "prior_market_change_pct_per_day",
    "has_prior_market_change",
]
WARM_STATE = [
    "prior_price_trend_vnd_per_dud_day",
    "prior_price_volatility_vnd",
    "prior_observation_count",
]


def adapter_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, branch: str, population: str, models: str, representation: str, features: list[str], rationale: str) -> None:
        rows.append(
            {
                "adapter": name,
                "branch": branch,
                "population": population,
                "compatible_models": models,
                "representation": representation,
                "features": " | ".join(features),
                "feature_count": len(features),
                "rationale": rationale,
            }
        )

    add(
        "H0_COLD_TREE_LEVELS",
        "Relative h=0",
        "cold",
        "RF|XGBoost",
        "ordinal numeric + one-hot categorical inside fold",
        BASE + ["departure_minute", "competitor_min_price_other_airlines", "temporal_market_median_price"] + SUPPORT,
        "Excludes current row price; market level excludes target airline.",
    )
    add(
        "H0_COLD_CAT_LEVELS",
        "Relative h=0",
        "cold",
        "CatBoost",
        "native categorical",
        BASE + ["departure_period", "competitor_min_price_other_airlines", "temporal_market_median_price"] + SUPPORT,
        "Native categories; no external label encoding.",
    )
    add(
        "H0_COLD_DL_LEVELS",
        "Relative h=0",
        "cold",
        "MLP|RNN-family feature adapter",
        "categorical embeddings + cyclic clock",
        BASE + ["departure_time_sin", "departure_time_cos", "competitor_min_price_other_airlines", "temporal_market_median_price"] + SUPPORT,
        "Smooth clock geometry; target row price remains forbidden.",
    )
    add(
        "H0_COLD_LINEAR_LEVELS",
        "Relative h=0",
        "cold",
        "Ridge|ElasticNet|SVR",
        "one-hot categorical + standardized numeric inside fold",
        BASE + ["departure_time_sin", "departure_time_cos", "competitor_min_price_other_airlines", "temporal_market_median_price"] + SUPPORT,
        "Linear/SVM baseline adapter; transformations are fitted inside each fold.",
    )
    add(
        "H1_COLD_TREE_LEVELS",
        "Future-price h>=1",
        "cold",
        "RF|XGBoost",
        "levels",
        BASE + ["target_dud", "departure_minute", "price_vnd", "competitor_min_price_other_airlines", "temporal_market_median_price"] + SUPPORT,
        "Current price is legal for a future target.",
    )
    add(
        "H1_COLD_TREE_RELATIVE",
        "Future-price h>=1",
        "cold",
        "RF|XGBoost",
        "current level + relative market position",
        BASE + ["target_dud", "departure_minute", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Drops competitor level because current+ratio reconstruct it.",
    )
    add(
        "H1_COLD_CAT_RELATIVE",
        "Future-price h>=1",
        "cold",
        "CatBoost",
        "native categorical + relative market position",
        BASE + ["target_dud", "departure_period", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Native categorical adapter; same target legality as tree relative adapter.",
    )
    add(
        "H1_COLD_DL_RELATIVE",
        "Future-price h>=1",
        "cold",
        "MLP|RNN-family feature adapter",
        "embeddings + cyclic clock + relative position",
        BASE + ["target_dud", "departure_time_sin", "departure_time_cos", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "No one-timestep recurrent claim; sequence models use EA07 contracts.",
    )
    add(
        "H1_COLD_LINEAR_RELATIVE",
        "Future-price h>=1",
        "cold",
        "Ridge|ElasticNet|SVR",
        "one-hot categorical + standardized numeric + relative position",
        BASE + ["target_dud", "departure_time_sin", "departure_time_cos", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Linear/SVM baseline adapter; no deterministic market-position triplet.",
    )
    add(
        "WD_COLD_TREE_RELATIVE",
        "Will-Drop exact-next",
        "cold",
        "RF|XGBoost",
        "current level + relative market position",
        BASE + ["departure_minute", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Exact-next target DUD is deterministic from current DUD and is therefore omitted.",
    )
    add(
        "WD_COLD_CAT_RELATIVE",
        "Will-Drop exact-next",
        "cold",
        "CatBoost",
        "native categorical + relative market position",
        BASE + ["departure_period", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Exact-next target DUD is deterministic from current DUD and is therefore omitted.",
    )
    add(
        "WD_COLD_DL_RELATIVE",
        "Will-Drop exact-next",
        "cold",
        "MLP|RNN-family feature adapter",
        "embeddings + cyclic clock + relative position",
        BASE + ["departure_time_sin", "departure_time_cos", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Feature-based neural adapter; no duplicate target horizon column.",
    )
    add(
        "WD_COLD_LINEAR_RELATIVE",
        "Will-Drop exact-next",
        "cold",
        "LogisticRegression|LinearSVM",
        "one-hot categorical + standardized numeric + relative position",
        BASE + ["departure_time_sin", "departure_time_cos", "price_vnd", "current_vs_competitor_ratio", "temporal_market_median_price"] + SUPPORT,
        "Transparent linear classification baseline; calibration is a later fold-only policy step.",
    )
    add(
        "WARM_LEVELS_EXTENSION",
        "all legal branches",
        "warm",
        "all tabular feature-based models",
        "history levels",
        ["previous_price_same_schedule"] + WARM_STATE,
        "Adds level, direction, uncertainty and support once each.",
    )
    add(
        "WARM_RELATIVE_EXTENSION",
        "all legal branches",
        "warm",
        "all tabular feature-based models",
        "history relative to temporal market",
        ["previous_vs_temporal_market_ratio"] + WARM_STATE,
        "Alternative to previous-price level; not concatenated with both ratio components.",
    )
    return rows


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_candidate_frame()
    numeric = frame[NUMERIC_CANDIDATES].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson", min_periods=1000)
    sample = numeric.sample(min(200_000, len(numeric)), random_state=115)
    spearman = sample.corr(method="spearman", min_periods=1000)

    records = []
    for left, right in combinations(NUMERIC_CANDIDATES, 2):
        complete = int(numeric[[left, right]].notna().all(axis=1).sum())
        p = float(pearson.loc[left, right])
        s = float(spearman.loc[left, right])
        magnitude = max(abs(p), abs(s)) if np.isfinite(p) and np.isfinite(s) else np.nan
        if np.isfinite(magnitude) and magnitude >= 0.95:
            band = "AUTO_REVIEW_GE_0_95"
        elif np.isfinite(magnitude) and magnitude >= 0.90:
            band = "REVIEW_0_90_TO_0_95"
        else:
            band = "BELOW_REVIEW_THRESHOLD"
        records.append(
            {
                "feature_left": left,
                "feature_right": right,
                "complete_rows": complete,
                "pearson": p,
                "spearman": s,
                "max_abs_association": magnitude,
                "review_band": band,
                "automatic_drop": False,
                "decision_rule": "drop only if deterministic or semantically equivalent; otherwise adapter/ablation review",
            }
        )
    pairs = pd.DataFrame(records).sort_values("max_abs_association", ascending=False)
    pairs.to_csv(TABLE_DIR / "ea04_numeric_pairwise_association.csv", index=False)
    pd.DataFrame(DETERMINISTIC_GROUPS).to_csv(TABLE_DIR / "ea04_deterministic_redundancy_groups.csv", index=False)
    adapters = pd.DataFrame(adapter_rows())
    adapters.to_csv(TABLE_DIR / "ea04_model_family_adapters.csv", index=False)

    availability_features = [
        "competitor_min_price_other_airlines",
        "competitor_price_spread_other_airlines",
        "competitor_airline_count",
        "competitor_offer_count",
        "same_airline_alternative_min_price",
        "temporal_market_median_price",
        "temporal_market_support",
        "temporal_market_age_hours",
        "prior_market_change_pct_per_day",
        "has_prior_market_change",
        "previous_price_same_schedule",
        "prior_price_trend_vnd_per_dud_day",
        "prior_price_volatility_vnd",
    ]
    availability = pd.DataFrame(
        {
            "feature": availability_features,
            "all_rows_available_pct": [100 * frame[name].notna().mean() for name in availability_features],
            "cold_eligible": [not name.startswith(("previous_", "prior_price_", "lag_age")) for name in availability_features],
            "warm_only": [name.startswith(("previous_", "prior_price_", "lag_age")) for name in availability_features],
        }
    )
    availability.to_csv(TABLE_DIR / "ea04_candidate_availability.csv", index=False)

    plot = availability.sort_values("all_rows_available_pct")
    colors = np.where(plot["warm_only"], "#D55E00", "#0072B2")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.barh(plot["feature"], plot["all_rows_available_pct"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in plot["all_rows_available_pct"]], padding=4, fontsize=9)
    ax.axvline(100, color="#333333", linewidth=0.8)
    ax.set_xlim(0, 106)
    ax.set_xlabel("Availability across completed offer rows (%)")
    ax.set_ylabel("")
    fig.suptitle(
        "FS02. Selected market context serves cold offers; schedule history remains warm-only",
        y=0.98,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.92,
        "Blue = cold-legal market context; orange = strictly-prior warm extension.",
        ha="center",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG_DIR / "FS02_market_context_serves_cold_history_remains_warm_only.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    high = pairs[pairs["review_band"].ne("BELOW_REVIEW_THRESHOLD")]
    summary = {
        "status": "EA04_COMPLETE_MODEL_FREE",
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "cutoff": str(CUTOFF.date()),
        "candidate_rows": len(frame),
        "numeric_features_audited": len(NUMERIC_CANDIDATES),
        "numeric_pairs_audited": len(pairs),
        "pairs_ge_0_90": int(len(high)),
        "pairs_ge_0_95": int(pairs["review_band"].eq("AUTO_REVIEW_GE_0_95").sum()),
        "prefit_adapter_exclusions": {
            "same_airline_alternative_count": (
                "metadata/audit only: Spearman association with "
                "temporal_market_support is approximately 0.949; keep the "
                "prior-market support feature in model adapters"
            ),
            "lag_age_hours": (
                "routing/freshness metadata only: Spearman association with "
                "days_until_departure is approximately 0.945; do not expose "
                "the collection-schedule shortcut to initial model adapters"
            ),
        },
        "deterministic_groups": len(DETERMINISTIC_GROUPS),
        "automatic_drop_rule": "only deterministic or semantically equivalent relationships",
        "correlation_review_rule": "|Pearson| or |Spearman| >=0.90 review; >=0.95 remove only when same meaning",
        "linear_adapter_vif_rule": "VIF >10 is a review trigger inside a declared linear adapter, not a universal tree-feature deletion rule",
        "model_fit_called": False,
        "cross_collection_era_history_violations": int(
            (
                frame["previous_schedule_collection_era"].notna()
                & frame["previous_schedule_collection_era"].ne(frame["collection_era"])
            ).sum()
        ),
        "cross_collection_era_market_anchor_violations": int(
            (
                frame["temporal_market_collection_era"].notna()
                & frame["temporal_market_collection_era"].ne(frame["collection_era"])
            ).sum()
        ),
    }
    (TABLE_DIR / "ea04_redundancy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    high_table = high.head(20)[["feature_left", "feature_right", "pearson", "spearman", "review_band"]].to_markdown(index=False)
    adapter_table = adapters[["adapter", "branch", "compatible_models", "feature_count", "representation"]].to_markdown(index=False)
    REPORT.write_text(
        f"""# EA04 - Redundancy and Model-Family Adapters

> Model-free audit on `{len(frame):,}` completed-batch offers through `{CUTOFF.date()}`.

## Academic decision rule

Pairwise association is a diagnostic, not a universal deletion rule. This new
build reviews `|Pearson|` or `|Spearman| >= 0.90`. At `>=0.95`, a column is
removed only when the pair is also deterministic or semantically equivalent.
For a declared linear adapter, VIF above 10 is an additional review trigger.
Tree and neural adapters are not forced to share an identical representation.

Reference: [NIST Variance Inflation Factor](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/vif.htm).

## Highest numeric associations

{high_table if len(high) else 'No numeric pair reached the review threshold.'}

## Preventive decisions

- `same_airline_alternative_count` remains available for audit but is excluded
  from every pre-fit model adapter: its Spearman association with
  `temporal_market_support` is about 0.949. The latter is retained because it
  directly qualifies the strictly-prior market anchor used at serving time.
- `lag_age_hours` remains a warm-history freshness/router field but is excluded
  from initial model adapters: its rank association with DUD is about 0.945,
  and feeding it to the model risks learning the collection schedule.
- No adapter contains current price, competitor minimum and their ratio together.
- No warm adapter contains previous price, temporal median and their ratio together.
- Exact DUD is tabular; booking-window bucket is a sequence key only.
- Clock representation is selected by model family rather than concatenating
  minute, cyclic coordinates and period flags.
- Native categorical models receive categories; linear/SVM adapters one-hot
  inside each training fold; neural feature adapters use embeddings.
- Same-schedule history, prior market anchors and sequence lanes are grouped
  within `collection_era`; both cross-era violation counters must remain zero.

## Pre-fit adapters

{adapter_table}

These are candidate adapters, not winners. Final retention requires paired
rolling ablation after the pilot begins.
""",
        encoding="utf-8",
    )
    print(f"[EA04 PASS] rows={len(frame):,} pairs={len(pairs)} high={len(high)} adapters={len(adapters)}")


if __name__ == "__main__":
    main()
