"""EA03 pre-fit Feature Legality Matrix and selection criteria.

Snapshot: new-build 110 pilot, cutoff 2026-07-10.
Branch: semantic Feature Selection before first feature construction.
Inputs: EA01, EA02 and EA06 model-free audit summaries.
Outputs: legality CSV, criteria JSON and Markdown report. No model fit.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from skyfare.features.audit_common import CUTOFF, ROOT

TABLE_DIR = ROOT / "artifacts/feature_research/tables"
REPORT = ROOT / "artifacts/feature_research/reports/EA03_FEATURE_LEGALITY_MATRIX.txt"


def row(
    feature: str,
    family: str,
    dimension: str,
    formula: str,
    cold: str,
    warm: str,
    h0: str,
    h1: str,
    willdrop: str,
    point_in_time_rule: str,
    disposition: str,
    representation: str,
    reason: str,
) -> dict[str, str]:
    return locals()


def main() -> None:
    required = {
        "EA01": TABLE_DIR / "ea01_identity_continuity_summary.json",
        "EA02": TABLE_DIR / "ea02_target_horizon_summary.json",
        "EA06": TABLE_DIR / "ea06_listed_price_semantics_summary.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"EA03 cannot run before model-free audits: {missing}")
    audits = {name: json.loads(path.read_text()) for name, path in required.items()}

    rows = [
        row("route", "F1_product", "market identity", "origin + destination", "YES", "YES", "YES", "YES", "YES", "known in batch", "CANDIDATE", "native categorical / one-hot", "Core market identity"),
        row("airline", "F1_product", "business model", "carrier code", "YES", "YES", "YES", "YES", "YES", "known in batch", "CANDIDATE", "native categorical / one-hot", "Systematic airline pricing differences"),
        row("departure_clock_adapter", "F1_product", "schedule convenience", "one of minute / cyclic sin-cos / categorical period", "YES", "YES", "YES", "YES", "YES", "known before departure", "MODEL_ADAPTER_VARIANT", "model-specific", "Never concatenate deterministic clock representations in one adapter"),
        row("schedule_slot_id", "F1_product", "schedule-slot proxy", "route|airline|flight_date|HHMM", "YES", "YES", "CONDITIONAL", "CONDITIONAL", "CONDITIONAL", "known in batch; EA01 quality required", "ROUTING_OR_GROUP_KEY", "categorical/group key", "Not verified physical flight; high-cardinality identifier needs ablation"),
        row("identity_quality", "F5_support", "identity confidence", "SCHEDULE_SLOT_PROXY or VERIFIED_FLIGHT_NO", "YES", "YES", "YES", "YES", "YES", "known from ingestion", "REQUIRED_METADATA", "categorical/mask", "Prevents physical-flight overclaim"),
        row("route_distance_km", "F1_product", "physical distance prior", "versioned airport-coordinate distance", "YES", "YES", "YES", "YES", "YES", "static lookup versioned before fit", "DEFER", "numeric", "Legacy 0/1/2 route class was not kilometres and duplicated route"),
        row("days_until_departure", "F2_time", "booking-cycle position", "flight_date - session_date", "YES", "YES", "YES", "YES", "YES", "known at prediction", "CANDIDATE", "numeric", "Primary revenue-management clock"),
        row("target_dud", "F2_time", "requested future booking position", "future DUD on observed grid", "YES", "YES", "NO", "YES", "NO", "defined by requested future window", "CANDIDATE_H1_PLUS", "numeric/embedding", "Together with current DUD identifies multi-horizon target"),
        row("target_horizon_days", "F2_time", "forecast horizon", "current_DUD - target_DUD", "YES", "YES", "NO", "YES", "NO", "defined by requested future window", "EXCLUDE_DETERMINISTIC", "metadata only", "Deterministic from current DUD and target DUD"),
        row("month", "F2_time", "flight season", "month(flight_date)", "YES", "YES", "YES", "YES", "YES", "known before departure", "CANDIDATE", "categorical/cyclic ablation", "Flight-date seasonality"),
        row("day_of_week", "F2_time", "flight weekday", "weekday(flight_date)", "YES", "YES", "YES", "YES", "YES", "known before departure", "CANDIDATE", "categorical/cyclic ablation", "Demand calendar"),
        row("is_peak", "F2_time", "Vietnam peak event", "flight_date in versioned peak config", "YES", "YES", "YES", "YES", "YES", "config must predate prediction", "CANDIDATE", "binary", "Keep domain concept; move dates to config"),
        row("current_price_vnd", "F3_market", "current offer level", "observed target-row price in completed batch", "YES", "YES", "NO", "YES", "YES", "known only for future-target tasks", "CANDIDATE_H1_PLUS", "numeric/log", "Target leakage for h=0; valid input for h>=1 and Will-Drop"),
        row("competitor_min_price_other_airlines", "F3_market", "competitor level", "min price of other-airline rows in same route/batch", "YES", "YES", "YES", "YES", "YES", "whole batch complete; target row excluded", "CANDIDATE", "numeric/log", "Legal cross-sectional market context"),
        row("competitor_price_spread_other_airlines", "F3_market", "market dispersion", "max-min of other-airline median prices in same route/flight-date/batch", "YES", "YES", "YES", "YES", "YES", "whole batch complete; target airline excluded", "CANDIDATE", "numeric", "Uncertainty in current market"),
        row("competitor_airline_count", "F3_market", "market support", "number of other airlines in comparable market", "YES", "YES", "YES", "YES", "YES", "whole batch complete", "CANDIDATE", "numeric/count", "Distinguishes broad from thin competition"),
        row("competitor_offer_count", "F3_market", "market support", "number of other-airline offers in comparable market", "YES", "YES", "YES", "YES", "YES", "whole batch complete", "CANDIDATE", "numeric/count", "Support for market level and spread"),
        row("same_airline_alternative_min_price", "F3_market", "own-carrier alternative level", "leave-one-offer-out minimum among same-airline other departures", "YES", "YES", "YES", "YES", "YES", "whole batch complete; current schedule row excluded", "CANDIDATE", "numeric/log", "Differentiates one departure from same-airline alternatives"),
        row("same_airline_alternative_count", "F3_market", "own-carrier support", "count same-airline other departures", "YES", "YES", "YES", "YES", "YES", "whole batch complete", "CANDIDATE", "numeric/count", "Support for own-carrier alternative level"),
        row("batch_offer_count", "F5_support", "collection volume", "all offer rows in completed batch", "YES", "YES", "YES", "YES", "YES", "whole batch complete", "SENSITIVITY_METADATA", "numeric/count", "Mostly collection-era volume; comparable-market counts are preferred model inputs"),
        row("xsec_ratio_now", "F3_market", "relative position", "current_price / causal batch anchor", "YES", "YES", "NO", "YES", "YES", "whole batch complete", "CANDIDATE_H1_PLUS", "numeric", "Contains current target at h=0"),
        row("previous_price_same_schedule", "F4_history", "persistence level", "latest price from prior completed batch for schedule slot", "NO", "YES", "YES", "YES", "YES", "previous_batch_end < current_batch_start", "WARM_CANDIDATE", "numeric/log + missing mask", "Strong baseline; must beat carry-forward"),
        row("previous_ratio_same_schedule", "F4_history", "relative persistence", "latest causal ratio for schedule slot", "NO", "YES", "YES", "YES", "YES", "previous_batch_end < current_batch_start", "WARM_CANDIDATE", "numeric + missing mask", "Warm-only relative state"),
        row("prior_price_trend_vnd_per_dud_day", "F4_history", "trajectory direction", "causal OLS slope of price over DUD using strictly-prior observations", "NO", "YES", "YES", "YES", "YES", "all source times strictly earlier", "WARM_CANDIDATE", "numeric + support mask", "Renamed from false elasticity claim; adds direction beyond level"),
        row("prior_price_volatility_vnd", "F4_history", "trajectory uncertainty", "std over strictly-prior observations", "NO", "YES", "YES", "YES", "YES", "all source times strictly earlier", "WARM_CANDIDATE", "numeric + support mask", "Adds volatility beyond level"),
        row("lag_age_hours", "F4_history", "history freshness", "feature_time - prior_feature_time", "NO", "YES", "YES", "YES", "YES", "positive strictly-prior delta", "WARM_ROUTING_METADATA", "numeric + missing mask; excluded from initial model adapters", "Old prices lose relevance with age; routing use avoids a DUD-correlated collection shortcut"),
        row("prior_observation_count", "F4_history", "history support", "count strictly-prior observations", "NO", "YES", "YES", "YES", "YES", "count only earlier completed batches", "WARM_CANDIDATE", "numeric/count", "Confidence/support metadata"),
        row("market_batch_age_hours", "F5_support", "batch freshness", "query_time - batch_completion_time", "YES", "YES", "YES", "YES", "YES", "known at serving but unavailable in historical training rows", "OPERATIONAL_GATE_ONLY", "numeric metadata", "Do not train until query-time is recorded consistently"),
        row("session_label", "F5_support", "AM/PM collection session", "AM 06:00-17:59; PM 18:00-05:59 on prior operational date", "YES", "YES", "YES", "YES", "YES", "known from intended collection session", "CANDIDATE", "categorical", "Training collection regime; keep separate from 00:00/12:00 GUI daypart"),
        row("collection_era", "F5_support", "source-era support", "FLI_LIBRARY_ERA or TRIP_COM_BROWSER_ERA", "YES", "YES", "CONDITIONAL", "CONDITIONAL", "CONDITIONAL", "known from ingestion", "SENSITIVITY_METADATA", "categorical/metadata", "EA06 shows source and era are non-identifiable"),
        row("temporal_market_median_price", "F3_market", "causal market level", "prior-session route/airline/DUD market median", "YES", "YES", "YES", "YES", "YES", "anchor_session < current_session", "CANDIDATE", "numeric + age/support", "Baseline and fair-value context"),
        row("temporal_market_support", "F3_market", "causal anchor support", "offer count behind prior-session market median", "YES", "YES", "YES", "YES", "YES", "anchor_session < current_session", "CANDIDATE", "numeric/count", "Separates supported from thin anchors"),
        row("temporal_market_age_hours", "F3_market", "causal anchor freshness", "feature_time - prior market time", "YES", "YES", "YES", "YES", "YES", "strictly positive", "CANDIDATE", "numeric", "Older market anchors are less comparable"),
        row("prior_market_change_pct_per_day", "F3_market", "causal market direction", "100 * (prior_market_median_t-1 / prior_market_median_t-2 - 1) / elapsed_days", "YES", "YES", "YES", "YES", "YES", "both source batches strictly precede feature_time", "CANDIDATE", "numeric + availability mask", "EA06 found residual temporal movement after flight-cohort controls; this supplies direction without using session date or collection era"),
        row("has_prior_market_change", "F3_market", "market-direction availability", "1[at least two strictly-prior comparable market batches exist]", "YES", "YES", "YES", "YES", "YES", "derived only from prior-batch availability", "CANDIDATE_MASK", "binary", "Preserves missingness instead of inventing a market trend"),
        row("booking_window_bucket", "sequence_key", "series grouping", "discrete DUD group", "YES", "YES", "NO_AS_TABULAR", "NO_AS_TABULAR", "NO_AS_TABULAR", "known at prediction", "GROUP_KEY_ONLY", "sequence grouping key", "Keep as series key; raw DUD is finer tabular predictor"),
        row("future_price_vnd", "target", "label", "observed same-entity future-window price", "NO", "NO", "NO", "NO", "NO", "exists only at label time", "TARGET_ONLY", "target", "Never a feature"),
        row("will_drop_exact_next", "target", "label", "1[next_price <= 0.95*current_price]", "NO", "NO", "NO", "NO", "NO", "exists only after exact-next label time", "TARGET_ONLY", "target", "Never a feature"),
    ]
    matrix = pd.DataFrame(rows)
    matrix.to_csv(TABLE_DIR / "ea03_feature_legality_matrix.csv", index=False)

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    criteria = {
        "status": "LOCKED_BEFORE_FIRST_NEW_BUILD_FEATURE_DATASET_OR_MODEL_FIT",
        "created_at": now,
        "cutoff": str(CUTOFF.date()),
        "before_first_new_build_110_fit": True,
        "source_audits": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "status": audits[name]["status"],
            }
            for name, path in required.items()
        },
        "acceptance_rules": [
            "serve_available_for_declared_population",
            "legal_for_declared_target_and_horizon",
            "point_in_time_causal",
            "semantically_nonredundant_or_distinct_dimension",
            "retained_finally_only_after_rolling_paired_ablation",
        ],
        "feature_sets": {
            "cold": ["F1_product", "F2_time", "F3_market", "F5_support"],
            "warm_extension": ["F4_history"],
            "sequence_keys": ["market_event_sequence", "same_schedule_history_sequence"],
        },
        "presentation_contract": {
            "figures": ["FS1_feature_lineage", "FS2_cold_warm_availability", "FS6_family_ablation_after_fit"],
            "tables": ["FS3_legality_matrix", "FS4_redundancy_groups", "FS5_baseline_matrix", "FS7_final_feature_contract"],
        },
        "importance_rule": "Importance is post-fit diagnostic, not pre-fit selection evidence.",
        "correlation_rule": "Correlation alone does not remove a feature; deterministic or semantic equivalence can.",
    }
    (TABLE_DIR / "feature_selection_criteria.json").write_text(
        json.dumps(criteria, indent=2) + "\n", encoding="utf-8"
    )

    legal_counts = matrix["disposition"].value_counts().to_dict()
    report = f"""# EA03 - Feature Legality Matrix

> Created `{now}` before new-build feature construction and model fitting.

## Method

Feature legality is target- and horizon-specific. Semantic selection precedes
`.fit()`; predictive retention later requires rolling paired ablation across
compatible model families.

## Key decisions

- `current_price_vnd` and `xsec_ratio_now`: forbidden for h=0, legal for h>=1 and Will-Drop.
- same-schedule history: warm-only, strictly prior, never imputed into cold rows.
- `schedule_slot_id`: proxy/group key pending EA01 quality; not physical flight.
- `booking_window_bucket`: sequence grouping key only; raw DUD remains tabular input.
- `collection_era`: sensitivity/support metadata because EA06 cannot separate source from era.

Disposition counts: `{legal_counts}`.

Outputs are **3 figures + 4 tables**, not seven figures. FS6 and final FS7 remain
post-fit because empirical value requires frozen rolling predictions.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"[EA03 PASS] features={len(matrix)} criteria={TABLE_DIR / 'feature_selection_criteria.json'}")


if __name__ == "__main__":
    main()
