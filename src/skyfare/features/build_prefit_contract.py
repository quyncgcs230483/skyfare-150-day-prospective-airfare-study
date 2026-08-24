"""Consolidate model-free gates into the pre-fit candidate feature contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from skyfare.features.audit_common import CUTOFF, ROOT


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
REPORT = ROOT / "artifacts/feature_research/reports/FEATURE_SELECTION_PREFIT_CONTRACT.txt"


def feature(name: str, family: str, population: str, branch: str, status: str, representation: str, unique_information: str) -> dict[str, str]:
    return locals()


def digest(path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    required = [
        TABLE_DIR / "fs01_feature_lineage.csv",
        TABLE_DIR / "ea03_feature_legality_matrix.csv",
        TABLE_DIR / "ea04_numeric_pairwise_association.csv",
        TABLE_DIR / "ea04_deterministic_redundancy_groups.csv",
        TABLE_DIR / "ea04_model_family_adapters.csv",
        TABLE_DIR / "ea05_regression_causal_baselines.csv",
        TABLE_DIR / "ea05_willdrop_causal_baselines.csv",
        TABLE_DIR / "ea07_sequence_length_coverage.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"FS07 cannot lock before prior gates exist: {missing}")

    rows = [
        feature("route", "F1 product", "cold+warm", "all", "CANDIDATE", "categorical adapter", "market identity"),
        feature("airline", "F1 product", "cold+warm", "all", "CANDIDATE", "categorical adapter", "carrier business model"),
        feature("departure_clock_adapter", "F1 product", "cold+warm", "all", "CANDIDATE_VARIANT", "minute OR cyclic sin/cos OR period", "schedule convenience without duplicate encodings"),
        feature("days_until_departure", "F2 time", "cold+warm", "all", "CANDIDATE", "numeric", "current booking-cycle position"),
        feature("target_dud", "F2 time", "cold+warm", "future-price multi-horizon", "CANDIDATE", "numeric/embedding", "requested future booking position"),
        feature("flight_month", "F2 time", "cold+warm", "all", "CANDIDATE", "categorical", "flight-date seasonality"),
        feature("flight_day_of_week", "F2 time", "cold+warm", "all", "CANDIDATE", "categorical", "flight weekday demand"),
        feature("is_peak_period", "F2 time", "cold+warm", "all", "CANDIDATE", "binary from versioned official config", "Vietnamese holiday travel regime"),
        feature("session_label", "F5 observation", "cold+warm", "all", "CANDIDATE", "categorical", "AM/PM collection regime"),
        feature("competitor_min_price_other_airlines", "F3 market", "cold+warm", "all", "CANDIDATE", "level/log", "other-airline market level excluding target airline"),
        feature("competitor_price_spread_other_airlines", "F3 market", "cold+warm", "all", "CANDIDATE", "numeric", "cross-airline market dispersion"),
        feature("competitor_airline_count", "F3 market", "cold+warm", "all", "CANDIDATE", "count", "cross-airline support"),
        feature("competitor_offer_count", "F3 market", "cold+warm", "all", "CANDIDATE", "count", "offer support behind competitor context"),
        feature("same_airline_alternative_min_price", "F3 market", "cold+warm", "all", "CANDIDATE", "level/log", "own-carrier alternative departure level"),
        feature("same_airline_alternative_count", "F3 market", "cold+warm", "audit only", "EXCLUDE_REDUNDANT", "count", "near-duplicate support signal; retained outside model adapters for audit"),
        feature("temporal_market_median_price", "F3 market", "cold+warm", "all", "CANDIDATE", "level/log", "strictly-prior market level"),
        feature("temporal_market_support", "F3 market", "cold+warm", "all", "CANDIDATE", "count", "support behind prior market level"),
        feature("temporal_market_age_hours", "F3 market", "cold+warm", "all", "CANDIDATE", "numeric", "freshness of prior market context"),
        feature("prior_market_change_pct_per_day", "F3 market", "cold+warm", "all", "CANDIDATE", "numeric + mask", "strictly-prior market direction distinct from prior market level"),
        feature("has_prior_market_change", "F3 market", "cold+warm", "all", "CANDIDATE_MASK", "binary", "explicit availability; no imputed market direction"),
        feature("price_vnd", "current state", "cold+warm", "future-price h>=1 + Will-Drop", "CONDITIONAL", "level/log", "observed current price; forbidden as h=0 input"),
        feature("current_vs_competitor_ratio", "current state", "cold+warm", "future-price h>=1 + Will-Drop", "ADAPTER_VARIANT", "ratio", "relative current market position; not with both components"),
        feature("previous_price_same_schedule", "F4 history", "warm", "all", "WARM_CANDIDATE", "level/log", "latest strictly-prior schedule level"),
        feature("previous_vs_temporal_market_ratio", "F4 history", "warm", "all", "WARM_ADAPTER_VARIANT", "ratio", "prior schedule position versus market; alternative to level components"),
        feature("prior_price_trend_vnd_per_dud_day", "F4 history", "warm", "all", "WARM_CANDIDATE", "numeric + mask", "trajectory direction, not elasticity"),
        feature("prior_price_volatility_vnd", "F4 history", "warm", "all", "WARM_CANDIDATE", "numeric + mask", "trajectory uncertainty"),
        feature("lag_age_hours", "F4 history", "warm", "routing only", "WARM_ROUTING_METADATA", "numeric + mask", "history freshness; excluded from initial model adapters after correlation audit"),
        feature("prior_observation_count", "F4 history", "warm", "all", "WARM_CANDIDATE", "count", "history support"),
        feature("schedule_slot_id", "identity", "cold+warm", "routing/sequence", "GROUP_KEY_ONLY", "uint64 hash of semantic tuple", "same-schedule routing; not physical flight claim"),
        feature("identity_quality", "identity", "cold+warm", "routing/reporting", "METADATA_ONLY", "categorical", "continuity confidence"),
        feature("collection_era", "F5 support", "cold+warm", "sensitivity", "METADATA_ONLY", "categorical", "source-era confound disclosure"),
        feature("batch_offer_count", "F5 support", "cold+warm", "sensitivity", "METADATA_ONLY", "count", "collection volume; comparable-market counts preferred"),
        feature("booking_window_bucket", "sequence", "cold+warm", "market-event sequence", "GROUP_KEY_ONLY", "categorical group key", "coarse market series identity; not tabular input"),
        feature("market_batch_age_hours", "serving", "cold+warm", "operational gate", "DEFER_TRAINING", "numeric metadata", "requires historical query-time logging"),
        feature("route_distance_km", "F1 product", "cold+warm", "future work", "DEFER", "numeric", "requires versioned airport-coordinate source"),
    ]
    contract = pd.DataFrame(rows)
    contract.to_csv(TABLE_DIR / "fs07_prefit_feature_contract.csv", index=False)

    pairs = pd.read_csv(TABLE_DIR / "ea04_numeric_pairwise_association.csv")
    high = pairs[pairs["review_band"].ne("BELOW_REVIEW_THRESHOLD")]
    baselines = pd.read_csv(TABLE_DIR / "ea05_regression_causal_baselines.csv")
    h0 = baselines[baselines["task"].eq("relative_h0")]
    exact = baselines[baselines["task"].eq("future_price_exact_next")]
    sequence = pd.read_csv(TABLE_DIR / "ea07_sequence_length_coverage.csv")
    adapters = pd.read_csv(TABLE_DIR / "ea04_model_family_adapters.csv")

    h0_table = h0[["baseline", "population", "coverage_pct", "mape_pct", "r2"]].to_markdown(index=False)
    exact_table = exact.groupby("baseline", observed=True).agg(
        eligible_rows=("eligible_rows", "sum"),
        mean_mape_pct=("mape_pct", "mean"),
        mean_r2=("r2", "mean"),
    ).reset_index().to_markdown(index=False)
    sequence_table = sequence.to_markdown(index=False)
    selected_table = contract[contract["status"].isin(["CANDIDATE", "CONDITIONAL", "WARM_CANDIDATE"])][
        ["name", "family", "population", "branch", "unique_information"]
    ].to_markdown(index=False)
    high_table = high[["feature_left", "feature_right", "pearson", "spearman", "review_band"]].to_markdown(index=False)

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    REPORT.write_text(
        f"""# Feature Selection - Pre-Fit Candidate Contract

> Status: `LOCKED_BEFORE_FIRST_NEW_BUILD_MODEL_FIT`
>
> Created `{now}` using raw-data/error-analysis evidence only. Archived
> Relative/Will-Drop 110/115 outcomes are not selection evidence.

## What has been selected before training

{selected_table}

The table deliberately has no fixed "CORE13" count. It contains semantic
dimensions, while model-family adapters choose one legal representation for
categorical and clock variables. The full machine-readable contract includes
conditional, metadata, grouping and deferred rows.

## Correlation and redundancy result

- Numeric candidates audited: **{len({column for column in pairs['feature_left']} | {column for column in pairs['feature_right']})}**.
- Numeric pairs audited: **{len(pairs)}**.
- Pairs reaching `|r| or |rho| >= 0.90`: **{len(high)}**.
- Pairs reaching `>=0.95`: **{int(high['max_abs_association'].ge(0.95).sum())}**.
- `same_airline_alternative_count` is excluded from every model adapter because
  it is near-duplicate support information (`rho` about 0.949) beside
  `temporal_market_support`; it remains audit metadata only.
- `lag_age_hours` is excluded from initial model adapters and retained only for
  freshness/routing. Its high association with DUD could otherwise expose a
  collection-schedule shortcut even though the two concepts differ semantically.
- Deterministic groups are separated by adapter before `.fit()`.

{high_table}

## Baselines the pilot must beat

### h=0

{h0_table}

### Future price, exact next window

{exact_table}

## Sequence decision before training

{sequence_table}

Same-schedule sequence lengths 14 and 21 have zero eligible rows. They are
therefore rejected before modelling. Length 7 is only a sparse warm extension;
market-event sequences may ablate 7/14/21 because their coverage remains high.

## Adapter rule

The **same semantic candidate pool** is evaluated across all relevant model
families, but representation is not forced to be identical:

- RF/XGBoost: numeric clock or one-hot categories fitted inside each fold.
- CatBoost: native categorical columns, following official CatBoost guidance.
- Linear/SVM: one-hot encoding fitted inside each fold.
- Feature-based neural models: categorical embeddings plus cyclic clock.
- True sequence models: EA07 event tensors with real `delta_hours` and masks.

`prior_market_change_pct_per_day` remains raw for tree candidates. Linear and
neural adapters apply deterministic signed-`log1p(abs(x))` followed by a scaler
fitted on that fold's training rows only. No clipping/scaling statistic is
computed on validation data. Lag, anchor and sequence construction is isolated
by `collection_era`; Trip rows never inherit Fli state.

Candidate value is decided by paired rolling family ablation. Feature
importance and residual correlation are post-fit diagnostics, never pre-fit
selection goals.
""",
        encoding="utf-8",
    )

    manifest = {
        "status": "LOCKED_BEFORE_FIRST_NEW_BUILD_MODEL_FIT",
        "created_at": now,
        "cutoff": str(CUTOFF.date()),
        "candidate_contract_rows": len(contract),
        "adapter_count": len(adapters),
        "uses_archived_110_115_outcomes": False,
        "model_fit_called": False,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}
            for path in required
        ],
    }
    (TABLE_DIR / "fs07_prefit_contract_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[FS07 PASS] contract_rows={len(contract)} adapters={len(adapters)} report={REPORT}")


if __name__ == "__main__":
    main()
