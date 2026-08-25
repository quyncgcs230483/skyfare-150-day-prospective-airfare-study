"""FS01 feature research, lineage and prevention-oriented risk register."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd

from skyfare.features.audit_common import CUTOFF, ROOT

TABLE_DIR = ROOT / "artifacts/feature_research/tables"
FIG_DIR = ROOT / "artifacts/feature_research/figures"
REPORT = ROOT / "artifacts/feature_research/reports/FS01_FEATURE_RESEARCH_AND_LINEAGE.txt"


SOURCES = {
    "survey": "https://www.sciencedirect.com/science/article/pii/S131915781830884X",
    "departure": "https://www.sciencedirect.com/science/article/pii/S0965856416300258",
    "catboost": "https://catboost.ai/docs/en/features/categorical-features",
    "onehot": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html",
    "vif": "https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/vif.htm",
    "elasticity": "https://openstax.org/books/principles-economics-3e/pages/5-1-price-elasticity-of-demand-and-price-elasticity-of-supply",
    "tet": "https://xaydungchinhsach.chinhphu.vn/de-xuat-phuong-an-nghi-tet-am-lich-nghi-le-quoc-khanh-nam-2026-119251002130522291.htm",
    "holidays": "https://baochinhphu.vn/chinh-thuc-bo-noi-vu-ra-thong-bao-lich-nghi-gio-to-hung-vuong-le-30-4-1-5-102260410213402453.htm",
}


def r(old: str, evidence: str, risk: str, design: str, new: str, disposition: str, family: str, source: str) -> dict[str, str]:
    return locals()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        r("route + origin + dest", "Route is directly reconstructible from origin and destination.", "Three deterministic representations inflate the adapter without adding information.", "Use route as the canonical market category; retain origin/dest as ingestion fields only.", "route", "KEEP", "F1 product", SOURCES["survey"]),
        r("airline + airline_name", "Carrier code and carrier name identify the same airline.", "Equivalent categories split importance and complicate train/serve mapping.", "Use stable airline code only.", "airline", "KEEP", "F1 product", SOURCES["survey"]),
        r("route_encoded / airline_encoded", "Legacy LabelEncoder assigned arbitrary integers to nominal categories.", "Distance-based and linear models interpret artificial order and spacing.", "Use model adapters: one-hot for linear/SVM, native categorical for CatBoost/XGBoost, embeddings for neural models.", "route; airline", "REDEFINE", "F1 product", SOURCES["catboost"]),
        r("dep_hour + morning + evening", "All are deterministic functions of local departure time.", "Redundant clock encodings add correlated columns and discontinuity at midnight.", "Use one representation per model adapter: cyclic sin/cos or categorical departure period.", "departure_time_sin/cos OR departure_period", "REDEFINE", "F1 product", SOURCES["departure"]),
        r("flight_no / flight_no_norm", "Raw audit shows airline-HHMM is a schedule-slot proxy, not verified physical flight identity.", "History can be joined to the wrong physical flight when schedules move.", "Use schedule_slot_id only as routing/group key, attach identity quality, and report continuity.", "schedule_slot_id; identity_quality", "REDEFINE", "F1 identity support", "EA01 raw identity audit"),
        r("route_distance = {0,1,2}", "Legacy value is a hand-written ordinal route class and is deterministic from route; it is not distance in kilometres.", "Name overclaims semantics and integer codes create false geometry.", "Exclude now; introduce route_distance_km only after a versioned airport-coordinate lookup is available.", "route_distance_km", "DEFER", "F1 product", "Legacy source audit"),
        r("days_until_departure + booking_window_bucket", "Bucket is a deterministic coarse version of exact DUD.", "Using both as tabular inputs duplicates booking-cycle position.", "Keep exact DUD as tabular input; keep bucket only as a sequence grouping key.", "days_until_departure", "KEEP", "F2 time", SOURCES["survey"]),
        r("day_of_week + is_weekend", "Weekend is exactly derived from weekday.", "Deterministic duplicate adds no dimension.", "Keep flight weekday; do not add is_weekend to the same adapter.", "flight_day_of_week", "KEEP", "F2 time", SOURCES["survey"]),
        r("is_peak hard-coded dates", "The concept is domain-relevant, but legacy Tet 27/01-02/02 conflicts with the official 2026 period in February.", "A wrong calendar silently assigns demand regime to the wrong flights.", "Keep binary feature using a versioned official Vietnamese peak-period config.", "is_peak_period", "REDEFINE", "F2 time", SOURCES["tet"]),
        r("competition_price_diff", "The corrected legacy column actually stored the minimum other-airline price, not a difference.", "Misleading name invites accidental subtraction/target leakage and wrong report claims.", "Compute other-airline context after full batch completion; exclude target airline and use semantic names plus support counts.", "competitor_min_price_other_airlines; competitor_price_spread_other_airlines; competitor_airline_count", "REDEFINE", "F3 market", SOURCES["departure"]),
        r("no same-airline schedule context", "Raw batches contain multiple departures by the same airline on the same route and flight date.", "Route-airline aggregates cannot express whether one schedule slot is expensive relative to the carrier's own alternatives.", "Add a leave-one-offer-out same-airline alternative level and count; never use the target row itself.", "same_airline_alternative_min_price; same_airline_alternative_count", "ADD", "F3 market", SOURCES["departure"]),
        r("xsec_ratio_now", "The numerator is current row price.", "It leaks h=0 current-price target but is legal for forecasting future price and Will-Drop.", "Declare it only in h>=1/Will-Drop relative-position adapter; never in h=0.", "current_vs_competitor_ratio", "CONDITIONAL", "F3 market", "Prediction-time legality"),
        r("price_lag_1step", "Legacy positional shift could use another row from the same collection session.", "History may not exist at first contact and same-session use is not serve-causal.", "Warm-only latest price from strictly earlier completed batch; preserve missing cold rows.", "previous_price_same_schedule", "REDEFINE", "F4 history", SOURCES["survey"]),
        r("rolling_mean_7d / rolling_mean_14d", "Legacy rolling windows counted 7/14 rows, not days, and were almost duplicate summaries.", "Names overclaim time semantics; near-duplicates split importance.", "Drop raw rolling means; represent one level, one direction, one volatility and their support/age.", "previous_price_same_schedule; prior_price_trend_vnd_per_dud_day; prior_price_volatility_vnd", "REPLACE", "F4 history", SOURCES["survey"]),
        r("elasticity_signal", "Economic price elasticity requires quantity-demand response; raw data contain listed prices but no demand quantity.", "Calling a price-vs-DUD slope elasticity is economically false.", "Retain the causal trajectory idea but rename it to the quantity it actually measures.", "prior_price_trend_vnd_per_dud_day", "RENAME", "F4 history", SOURCES["elasticity"]),
        r("lag_hours", "History relevance depends on elapsed time, but legacy time was approximated from AM=0/PM=12.", "Approximate timestamps can distort freshness and allow same-session history.", "Use actual completed-batch feature_time, require previous_feature_time < current_feature_time, and reserve lag age for freshness/routing rather than the initial model adapter.", "lag_age_hours", "REDEFINE", "F4 history", "Point-in-time contract"),
        r("price_tier / seats_left / is_soldout", "price_tier is source-specific; seats_left is mostly absent; soldout has little usable variation.", "Source artifacts and near-constant fields do not represent a stable serving contract.", "Exclude from primary candidate set; revisit only after consistent collection coverage.", "none", "EXCLUDE", "raw support", "Raw schema audit"),
        r("source/src", "Fli and Trip have zero overlapping dates, while FLI and Trip provide a 19-day early cross-check.", "A Fli/Trip flag is also an era flag, so that protocol effect cannot be identified separately from time; the FLI overlap does not verify later June/July semantics.", "Use collection era for sensitivity/disclosure, not as a causal primary feature; retain the FLI/Trip agreement as bounded early-period evidence.", "collection_era", "METADATA", "F5 support", "EA06 raw semantics audit"),
        r("no causal market-direction feature", "After controlling route, airline, exact DUD, departure period, flight month and weekday, EA06 still observes a -6.83% median early-to-late listed-price shift.", "A model with only a prior market level cannot distinguish a rising market from a falling market and may instead learn calendar/era shortcuts.", "Add one cold+warm market-direction candidate from two strictly-prior comparable route-airline-DUD batch medians; preserve missingness and test it by paired rolling ablation.", "prior_market_change_pct_per_day; has_prior_market_change", "ADD", "F3 market", "EA06 drift decomposition"),
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "fs01_feature_lineage.csv", index=False)
    risk = frame[["evidence", "risk", "design", "family", "source"]].copy()
    risk.columns = ["observed_evidence", "inferred_risk", "preventive_design", "feature_family", "evidence_source"]
    risk.to_csv(TABLE_DIR / "fs01_design_risk_register.csv", index=False)

    counts = frame["disposition"].value_counts().reindex(
        ["KEEP", "ADD", "REDEFINE", "REPLACE", "RENAME", "CONDITIONAL", "DEFER", "EXCLUDE", "METADATA"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9", "#999999", "#C44E52", "#666666"]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Legacy feature decisions")
    ax.set_ylabel("")
    fig.suptitle(
        "FS01. Most legacy features need semantic correction, not blind reuse",
        y=0.98,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.92,
        "Each decision is justified by old-task evidence, raw-data semantics or literature before model fitting.",
        ha="center",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG_DIR / "FS01_legacy_features_require_semantic_selection_before_training.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    source_lines = "\n".join(f"- [{name}]({url})" for name, url in SOURCES.items())
    table = frame[["old", "new", "disposition", "design"]].to_markdown(index=False)
    risk_table = risk.to_markdown(index=False)
    REPORT.write_text(
        f"""# FS01 - Feature Research, Lineage and Preventive Design

> Created `{now}` before the first new-build model fit. Cutoff `{CUTOFF.date()}`.
> This is a new-build design document. It does not cite Relative/Will-Drop
> outcome from archived 110/115 milestones.

## Decision rule

For every feature family: **observed evidence -> inferred deployment risk ->
preventive design**. A feature enters a candidate adapter only when it is
serve-available for its declared population, target/horizon-legal,
point-in-time causal, semantically distinct, and representable by that model
family. Predictive retention is decided later by paired rolling ablation.

## Design Risk Register

{risk_table}

Every preventive design above is supported by old absolute-task evidence,
raw-data semantics or an a-priori literature/domain source. It does not use an
archived Relative/Will-Drop 110/115 outcome as retrospective justification.

## Selected lineage

{table}

## Important corrections

- `is_peak_period` is retained, but the official 2026 periods are versioned in
  `configs/peak_dates_vietnam_2026.json`; the legacy Tet dates were wrong.
- `elasticity_signal` is not elasticity because quantity demanded is absent.
  The causal idea survives only as `prior_price_trend_vnd_per_dud_day`.
- `route_distance` was an ordinal class, not kilometres. It is deferred rather
  than silently presented as physical distance.
- Market context and target-containing ratios are different adapters. h=0 may
  use other-airline levels that exclude the target row; it may not use the
  current row price or a ratio containing it.

## Research and authoritative references

{source_lines}

The airline-price survey supports historical price, purchase timing and
departure timing as common information families. It does not license any
particular engineered column by itself; legality and raw-data availability
remain project-specific gates.
""",
        encoding="utf-8",
    )

    metadata = {
        "status": "FS01_LOCKED_BEFORE_MODEL_FIT",
        "created_at": now,
        "rows": len(frame),
        "uses_archived_110_115_model_outcomes": False,
        "cutoff": str(CUTOFF.date()),
    }
    (TABLE_DIR / "fs01_feature_lineage_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[FS01 PASS] decisions={len(frame)} report={REPORT}")


if __name__ == "__main__":
    main()
