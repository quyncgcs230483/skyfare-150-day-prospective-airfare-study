"""Build reproducible evidence linking old-model errors to feature selection.

Scope: old absolute regression and is_good_price classifier only.
This script reads locked diagnostic tables. It does not fit models or select
features. Outputs support new-build 110 model-free design gates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "artifacts/feature_research/error_analysis/tables"
OUTPUT = ROOT / "artifacts/feature_research/tables"

SAME_SCHEDULE_HISTORY = {
    "price_lag_1step",
    "price_rolling_mean_7d",
    "price_rolling_mean_14d",
    "elasticity_signal",
    "lag_hours",
}
CROSS_SECTION = {"competition_price_diff"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (INPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in filters.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {filters}; found {len(matches)}")
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    headline = read_csv("e_headline_failure_ranking.csv")
    availability = read_csv("e_main_history_feature_availability.csv")
    importance = read_csv("e4_feature_importance_90days.csv")
    serve = read_csv("e4_first_obs_and_serve_skew_regression.csv")
    temporal = read_csv("e3_temporal_oot_retrain.csv")
    baselines = read_csv("e2_surface_metrics_and_baselines.csv")

    cold = one(
        headline,
        failure_mechanism="Honest cold-start: same-flight history unavailable",
    )
    drift = one(headline, failure_mechanism="June temporal drift")
    offline = one(serve, milestone="90days", mode="normal", slice="all_test")
    first = one(
        serve,
        milestone="90days",
        mode="normal",
        slice="first_observation_only",
    )
    warm = one(
        serve,
        milestone="90days",
        mode="normal",
        slice="history_available_obs_gt1",
    )
    oot = one(
        temporal,
        evaluation="temporal_OOT_train_pre_June_test_June",
    )
    lag_baseline = one(
        baselines,
        milestone="90days",
        model="lag_1step",
        slice="history_available_obs_gt1",
    )

    same_history_importance = sum(
        float(row["importance"])
        for row in importance
        if row["feature"] in SAME_SCHEDULE_HISTORY
    )
    cross_section_importance = sum(
        float(row["importance"])
        for row in importance
        if row["feature"] in CROSS_SECTION
    )
    legacy_mislabeled_sum = sum(
        float(row["importance"])
        for row in importance
        if row["feature_family"] == "history_dependent"
    )

    cold_drop = float(cold["r2_drop"])
    drift_drop = float(drift["r2_drop"])
    summary = {
        "scope": {
            "regression_target": "price_vnd",
            "classification_target": "is_good_price",
            "not_110_or_115_target_evaluation": True,
        },
        "headline_90days": {
            "offline_r2": float(offline["r2"]),
            "first_observation_r2": float(first["r2"]),
            "warm_r2": float(warm["r2"]),
            "cold_start_r2_drop": cold_drop,
            "temporal_oot_r2": float(oot["reg_r2"]),
            "temporal_drift_r2_drop": drift_drop,
            "cold_drop_over_drift_drop": cold_drop / drift_drop,
        },
        "availability_first_contact_pct": {
            row["feature"]: float(row["available_pct"])
            for row in availability
            if row["contact"] == "First contact"
        },
        "importance_90days": {
            "same_schedule_history_sum_corrected": same_history_importance,
            "cross_section_competitor_sum": cross_section_importance,
            "legacy_history_dependent_sum_mislabeled": legacy_mislabeled_sum,
            "legacy_error": (
                "competition_price_diff was counted as history_dependent even "
                "though it is cross-sectional"
            ),
        },
        "old_absolute_warm_baseline_90days": {
            "lag_1step_r2": float(lag_baseline["r2"]),
            "rf_reference_r2": float(warm["r2"]),
            "rf_minus_lag_r2": float(warm["r2"]) - float(lag_baseline["r2"]),
            "population": "history_available_obs_gt1",
            "warning": (
                "This does not reproduce the chat-only 115 Relative claim "
                "56.24% versus 31.97%. That claim remains unverified."
            ),
        },
    }

    bridge_rows: list[dict[str, object]] = [
        {
            "id": "EAF01",
            "old_failure": "Old targets differ from rebuilt targets",
            "measured_evidence": "TARGET_REG=price_vnd; TARGET_CLS=is_good_price",
            "feature_requirement": "Judge legality separately for h=0, h>=1, and exact-next classification",
            "candidate_family": "all",
            "population": "all",
            "required_gate": "target_horizon_contract",
            "status": "MUST_ADD_BEFORE_SELECTION",
        },
        {
            "id": "EAF02",
            "old_failure": "Same-schedule history absent at first contact",
            "measured_evidence": "five history features have 0% first-contact availability",
            "feature_requirement": "cold path cannot require persistence; warm path may use strictly-prior persistence",
            "candidate_family": "persistence_and_trajectory",
            "population": "cold_vs_warm",
            "required_gate": "point_in_time_availability",
            "status": "VALID_OLD_EVIDENCE",
        },
        {
            "id": "EAF03",
            "old_failure": "Cold-start loss dominates measured temporal drift",
            "measured_evidence": f"R2 drop {cold_drop:.6f} vs {drift_drop:.6f} ({cold_drop / drift_drop:.2f}x)",
            "feature_requirement": "report cold/warm/routed metrics; keep rolling OOT as secondary robustness gate",
            "candidate_family": "serve_available_context",
            "population": "cold_vs_warm",
            "required_gate": "population_routing",
            "status": "VALID_OLD_EVIDENCE",
        },
        {
            "id": "EAF04",
            "old_failure": "RF importance family was mislabeled",
            "measured_evidence": f"legacy={legacy_mislabeled_sum:.6f}; corrected same-history={same_history_importance:.6f}; cross-section={cross_section_importance:.6f}",
            "feature_requirement": "separate persistence from current-batch market context; do not select from impurity rank",
            "candidate_family": "persistence_vs_market_context",
            "population": "cold_vs_warm",
            "required_gate": "semantic_family_audit",
            "status": "OLD_ANALYSIS_CORRECTION",
        },
        {
            "id": "EAF05",
            "old_failure": "Naive baseline comparison not matched to new horizons",
            "measured_evidence": f"old absolute warm RF R2={float(warm['r2']):.6f}; lag-one R2={float(lag_baseline['r2']):.6f}",
            "feature_requirement": "compare strongest legal baseline on identical rows and horizon before ML claim",
            "candidate_family": "baseline_router",
            "population": "cold_vs_warm_by_horizon",
            "required_gate": "causal_baseline_matrix",
            "status": "MUST_RECOMPUTE_FOR_NEW_TARGET",
        },
        {
            "id": "EAF06",
            "old_failure": "Schedule-slot proxy treated as physical flight",
            "measured_evidence": "flight_no_norm encodes airline-HHMM",
            "feature_requirement": "audit collisions, time shifts, continuity, and unmatched transitions",
            "candidate_family": "identity_and_schedule_slot",
            "population": "all",
            "required_gate": "identity_continuity",
            "status": "MUST_ADD_BEFORE_SELECTION",
        },
        {
            "id": "EAF07",
            "old_failure": "Source and era are confounded",
            "measured_evidence": "Fli and Trip occupy non-overlapping calendar eras",
            "feature_requirement": "describe support and drift without attributing cause to scraper alone",
            "candidate_family": "source_era_metadata",
            "population": "all",
            "required_gate": "source_era_and_price_semantics",
            "status": "MUST_ADD_BEFORE_SELECTION",
        },
        {
            "id": "EAF08",
            "old_failure": "Old tabular audit did not study same-schedule sequences",
            "measured_evidence": "history was compressed into lag/rolling columns",
            "feature_requirement": "audit market sequence and same-schedule sequence separately; test K structurally",
            "candidate_family": "sequence",
            "population": "market_vs_warm",
            "required_gate": "sequence_support",
            "status": "MUST_ADD_BEFORE_SELECTION",
        },
        {
            "id": "EAF09",
            "old_failure": "Old classifier target and safety denominator do not match product",
            "measured_evidence": "is_good_price is static; unsafe BUY requires false_BUY/predicted_BUY",
            "feature_requirement": "build exact-next labels, maturity, prevalence, and VND payoff inputs before fit; audit BUY safety after predictions exist",
            "candidate_family": "classification_and_policy",
            "population": "by_DUD_and_price_band",
            "required_gate": "target_policy_inputs_then_postfit_safety",
            "status": "SPLIT_PREFIT_AND_POSTFIT",
        },
        {
            "id": "EAF10",
            "old_failure": "Oracle best-window value can overstate deployable policy value",
            "measured_evidence": "not measured in old error analysis",
            "feature_requirement": "report oracle opportunity ceiling before fit and realized frozen-policy regret after predictions exist",
            "candidate_family": "multi_horizon_decision",
            "population": "eligible_price_paths",
            "required_gate": "oracle_prefit_then_realized_postfit",
            "status": "SPLIT_PREFIT_AND_POSTFIT",
        },
    ]

    figure_disposition: list[dict[str, str]] = [
        {
            "artifact": "main_F0_headline_failure_ranking.png",
            "role": "MAIN_REBUILT",
            "reason": "Quantifies cold-start loss versus drift; impossible stress lane removed",
        },
        {
            "artifact": "main_F1_offline_vs_at_serve_pred_actual.png",
            "role": "MAIN_REBUILT",
            "reason": "Shows offline versus honest cold-start only; stress panel removed",
        },
        {
            "artifact": "main_F2_prediction_distribution_collapse.png",
            "role": "DELETE",
            "reason": "Duplicates F1 and depends on impossible all-economic stress lane",
        },
        {
            "artifact": "main_F3_first_contact_feature_availability.png",
            "role": "MAIN_REBUILT",
            "reason": "Links unavailable history to observed failure",
        },
        {
            "artifact": "main_F4_serve_and_database_strategies.png",
            "role": "MAIN_REBUILT",
            "reason": "Answers whether coarse database proxies rescue old model; stress strategy removed",
        },
        {
            "artifact": "appendix_classifier_postmortem.png",
            "role": "DELETE",
            "reason": "Old is_good_price plot answers deprecated target; source table retains structural lesson",
        },
        {
            "artifact": "appendix_drift_modest_residual_and_prediction_time_support.png",
            "role": "DELETE",
            "reason": "Duplicates F0/F1 and adds no feature-selection decision",
        },
        {
            "artifact": "appendix_drift_modest_temporal_oot.png",
            "role": "DELETE_REPLACED",
            "reason": "Replaced by main_F5_temporal_oot_drift.png",
        },
        {
            "artifact": "appendix_drift_residual_by_dte_previous.png",
            "role": "DELETE",
            "reason": "Older residual slice; not needed for feature handoff",
        },
        {
            "artifact": "appendix_drift_residual_over_time_previous.png",
            "role": "DELETE",
            "reason": "Older residual slice; not needed for feature handoff",
        },
        {
            "artifact": "appendix_mechanism_r2_tests.png",
            "role": "DELETE",
            "reason": "Mechanism detail; main story already has measured ranking",
        },
        {
            "artifact": "appendix_offline_best_case_classifier_confusion_matrix.png",
            "role": "DELETE",
            "reason": "Deprecated classifier target and best-case surface",
        },
        {
            "artifact": "appendix_offline_best_case_feature_importance_80days.png",
            "role": "DELETE_REPLACED",
            "reason": "Wrong family label; replaced by one corrected 90days figure",
        },
        {
            "artifact": "appendix_offline_best_case_feature_importance_85days.png",
            "role": "DELETE_REPLACED",
            "reason": "Wrong family label; replaced by one corrected 90days figure",
        },
        {
            "artifact": "appendix_offline_best_case_feature_importance_90days.png",
            "role": "DELETE_REPLACED",
            "reason": "Wrong family label; replaced by one corrected 90days figure",
        },
        {
            "artifact": "appendix_offline_best_case_pred_vs_actual.png",
            "role": "DELETE",
            "reason": "Offline best case is context, not deployable evidence",
        },
        {
            "artifact": "appendix_offline_best_case_surface_metrics_and_baselines.png",
            "role": "DELETE",
            "reason": "Old target baseline context; recompute for new horizons",
        },
        {
            "artifact": "appendix_previous_serve_skew_flattening.png",
            "role": "DELETE",
            "reason": "Duplicates F1/F2 mechanism",
        },
        {
            "artifact": "appendix_previous_serve_skew_scatter.png",
            "role": "DELETE",
            "reason": "Duplicates F1/F2 mechanism",
        },
        {
            "artifact": "appendix_serve_mode_metric_bars.png",
            "role": "DELETE",
            "reason": "Detailed metrics retained; headline uses F0",
        },
        {
            "artifact": "appendix_superseded_F4_all_economic_removed.png",
            "role": "DELETE",
            "reason": "Stress removes legal competitor context; never normal serve claim",
        },
        {
            "artifact": "main_F5_temporal_oot_drift.png",
            "role": "MAIN_NEW",
            "reason": "Direct OOT evidence retained without duplicate residual panels",
        },
        {
            "artifact": "main_F6_corrected_feature_importance_90days.png",
            "role": "MAIN_NEW",
            "reason": "One corrected family view replaces three mislabeled figures",
        },
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "error_analysis_quantitative_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT / "error_analysis_feature_selection_bridge.csv", bridge_rows)
    write_csv(OUTPUT / "error_analysis_artifact_disposition.csv", figure_disposition)
    print(f"[PASS] wrote {len(bridge_rows)} bridge requirements")
    print(f"[PASS] classified {len(figure_disposition)} error-analysis figures")
    print(f"[PASS] cold-start R2 drop is {cold_drop / drift_drop:.2f}x drift drop")
    print(
        "[PASS] corrected 90days importance: "
        f"same-history={same_history_importance:.6f}, "
        f"cross-section={cross_section_importance:.6f}"
    )


if __name__ == "__main__":
    main()
