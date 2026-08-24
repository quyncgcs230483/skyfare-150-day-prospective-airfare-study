"""Freeze the leakage-controlled prefit protocol before any estimator fit.

Inputs are model-free feature-selection artifacts only. This script writes
split, candidate and acceptance contracts; it never imports an estimator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from skyfare.features.audit_common import CUTOFF, ROOT


TABLE_DIR = ROOT / "artifacts/feature_research/tables"
REPORT = ROOT / "artifacts/feature_research/reports/PREFIT_TRAINING_CONTRACT.txt"


RUNTIME_CONTRACT = {
    "base_image": "tensorflow/tensorflow:2.18.0-gpu-jupyter",
    "python": "3.10.12",
    "cuda": "12.5.1",
    "cudnn_major": "9",
    "direct_dependencies": {
        "numpy": "2.0.2",
        "pandas": "2.3.3",
        "pyarrow": "19.0.1",
        "scipy": "1.15.3",
        "scikit-learn": "1.6.1",
        "joblib": "1.4.2",
        "tensorflow": "2.18.0",
        "xgboost": "3.2.0",
        "catboost": "1.2.10",
        "chronos-forecasting": "2.2.0",
    },
    "chronos_model": {
        "repository": "amazon/chronos-2",
        "revision": "95a9710e2596287d08352589f42634fa5abdf0a7",
        "offline_only": True,
        "weights_required_before_stage1": False,
        "weights_required_before_foundation_diagnostic": True,
    },
    "provenance": {
        "role": "dependency compatibility evidence only; no archived predictions or outcomes reused",
        "tested_environment_evidence": "_ARCHIVE_v1_pre_rebuild/payload/03_models/snapshot_2026-07-15/persistence_outputs/outputs/vast_environment_preflight.json",
        "tree_version_evidence": "_ARCHIVE_v1_pre_rebuild/payload/05_docs/model_evaluation/110days/pretest_model_bundle_manifest.json",
    },
}


MARKET_DIRECTION_FEATURES = {
    "prior_market_change_pct_per_day",
    "has_prior_market_change",
}
MARKET_LEVEL_FEATURES = {
    "competitor_min_price_other_airlines",
    "current_vs_competitor_ratio",
    "temporal_market_median_price",
    "competitor_price_spread_other_airlines",
    "competitor_airline_count",
    "competitor_offer_count",
    "same_airline_alternative_min_price",
    "temporal_market_support",
    "temporal_market_age_hours",
}


SELECTION_COMMON_MULTI_HORIZON_TRANSITIONS = [
    "2->1",
    "3->1", "3->2",
    "5->1", "5->2", "5->3",
    "7->1", "7->2", "7->3", "7->5",
    "10->1", "10->2", "10->3", "10->5", "10->7",
    "14->1", "14->2", "14->3", "14->5", "14->7", "14->10",
    "21->1", "21->2", "21->3", "21->5", "21->7", "21->10", "21->14",
    "30->10", "30->14", "30->21",
    "45->30",
    "60->45",
]

CONFIRMATION_COMMON_MULTI_HORIZON_TRANSITIONS = [
    "2->1",
    "3->1", "3->2",
    "5->1", "5->2", "5->3",
    "7->1", "7->2", "7->3", "7->5",
    "10->5", "10->7",
    "14->10",
]


STAGE1_FIXED_CONFIGS = {
    "DECISION_TREE_REG_SCREEN_V2": {
        "max_depth": 14,
        "min_samples_leaf": 50,
        "min_samples_split": 100,
        "random_state": 115,
        "stage2_pruning_rule": "ccp_alpha selected on training-fold inner temporal tail only",
    },
    "DECISION_TREE_CLS_SCREEN_V2": {
        "max_depth": 12,
        "min_samples_leaf": 50,
        "min_samples_split": 100,
        "class_weight": "TRAIN_FOLD_BALANCED",
        "random_state": 115,
        "stage2_pruning_rule": "ccp_alpha selected on training-fold inner temporal tail only",
    },
    "RF_SCREEN_V1": {
        "n_estimators": 400,
        "max_depth": 18,
        "min_samples_leaf": 4,
        "max_features": 0.7,
        "random_state": 115,
        "n_jobs": -1,
    },
    "XGB_SCREEN_V1": {
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
        "random_state": 115,
    },
    "CATBOOST_SCREEN_V1": {
        "iterations": 800,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_seed": 115,
        "verbose": False,
    },
    "HISTGBM_SCREEN_V1": {
        "max_iter": 300,
        "max_leaf_nodes": 63,
        "learning_rate": 0.05,
        "l2_regularization": 0.1,
        "early_stopping": False,
        "random_state": 115,
    },
    "RIDGE_SCREEN_V1": {"alpha": 10.0},
    "ELASTICNET_SCREEN_V1": {
        "alpha": 0.001,
        "l1_ratio": 0.2,
        "max_iter": 5000,
        "random_state": 115,
    },
    "LOGISTIC_SCREEN_V1": {"C": 1.0, "max_iter": 3000, "class_weight": "TRAIN_FOLD_BALANCED"},
    "LINEARSVM_SCREEN_V1": {"C": 1.0, "class_weight": "TRAIN_FOLD_BALANCED"},
    "MLP_SCREEN_V1": {
        "hidden_units": [128, 64],
        "dropout": 0.15,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 2048,
        "max_epochs": 40,
        "inner_tail_patience": 4,
        "random_seed": 115,
    },
    "RECURRENT_SCREEN_V1": {
        "recurrent_units": 64,
        "dense_units": 32,
        "dropout": 0.15,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 1024,
        "max_epochs": 30,
        "inner_tail_patience": 4,
        "random_seed": 115,
    },
    "CHRONOS2_ZERO_SHOT_V1": {
        "fit_calls": 0,
        "repository": RUNTIME_CONTRACT["chronos_model"]["repository"],
        "revision": RUNTIME_CONTRACT["chronos_model"]["revision"],
        "chronos_forecasting_version": RUNTIME_CONTRACT["direct_dependencies"]["chronos-forecasting"],
        "offline_only": True,
    },
}


SELECTION_FOLDS = [
    {"fold": "PILOT_S1", "validation_start": "2026-06-06", "validation_end": "2026-06-12"},
    {"fold": "PILOT_S2", "validation_start": "2026-06-13", "validation_end": "2026-06-19"},
    {"fold": "PILOT_S3", "validation_start": "2026-06-20", "validation_end": "2026-06-26"},
]

CONFIRMATION_BLOCKS = [
    {
        "block": "PILOT_C1",
        "validation_start": "2026-06-27",
        "validation_end": "2026-07-03",
        "role": "FROZEN_MODEL_CONFIRMATION_AND_POLICY_SELECTION",
    },
    {
        "block": "PILOT_C2",
        "validation_start": "2026-07-04",
        "validation_end": "2026-07-10",
        "role": "FROZEN_POLICY_CONFIRMATION_NO_RESELECTION",
    },
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small contract table without the optional tabulate package."""

    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def candidate(
    candidate_id: str,
    stage: str,
    branch: str,
    target: str,
    population: str,
    model_family: str,
    architecture: str,
    adapter: str,
    feature_variant: str,
    sequence_lane: str = "NONE",
    sequence_length: str = "NONE",
    promotion_gate: str = "PREFIT_REGISTERED",
    target_representation: str = "NATIVE",
) -> dict[str, str]:
    return locals()


def registry_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    reg_models = [
        ("decision_tree", "DecisionTreeRegressor", "H1_COLD_TREE_LEVELS"),
        ("random_forest", "RF", "H1_COLD_TREE_LEVELS"),
        ("xgboost", "XGBoost", "H1_COLD_TREE_RELATIVE"),
        ("catboost", "CatBoost", "H1_COLD_CAT_RELATIVE"),
        ("hist_gradient_boosting", "HistGBM", "H1_COLD_TREE_RELATIVE"),
        ("ridge", "Ridge", "H1_COLD_LINEAR_RELATIVE"),
        ("elastic_net", "ElasticNet", "H1_COLD_LINEAR_RELATIVE"),
        ("mlp", "MLP", "H1_COLD_DL_RELATIVE"),
    ]
    cls_models = [
        ("decision_tree", "DecisionTreeClassifier", "WD_COLD_TREE_RELATIVE"),
        ("random_forest", "RF", "WD_COLD_TREE_RELATIVE"),
        ("xgboost", "XGBoost", "WD_COLD_TREE_RELATIVE"),
        ("catboost", "CatBoost", "WD_COLD_CAT_RELATIVE"),
        ("hist_gradient_boosting", "HistGBM", "WD_COLD_TREE_RELATIVE"),
        ("logistic_regression", "LogisticRegression", "WD_COLD_LINEAR_RELATIVE"),
        ("linear_svm_calibrated", "LinearSVM", "WD_COLD_LINEAR_RELATIVE"),
        ("mlp", "MLP", "WD_COLD_DL_RELATIVE"),
    ]
    feature_variants = [
        "STATIC_CURRENT_STATE",
        "PLUS_MARKET_LEVEL_CONTEXT",
        "PLUS_PRIOR_MARKET_DIRECTION",
    ]
    target_representations = [
        ("DIRECT_VND", "multi_horizon_price_vnd"),
        ("LOG_FUTURE_OVER_CURRENT", "multi_horizon_log_price_ratio"),
    ]
    for slug, family, adapter in reg_models:
        for target_representation, target in target_representations:
            for variant in feature_variants:
                rows.append(
                    candidate(
                        f"REG_PATH_{slug}__{target_representation.lower()}__{variant.lower()}",
                        "STAGE1_FAMILY_SCREEN",
                        "future_price",
                        target,
                        "cold_feature_path_all_rows",
                        family,
                        slug,
                        adapter,
                        variant,
                        target_representation=target_representation,
                    )
                )
    for slug, family, adapter in cls_models:
        for variant in feature_variants:
            rows.append(
                candidate(
                    f"WD_EXACT_{slug}__{variant.lower()}",
                    "STAGE1_FAMILY_SCREEN",
                    "willdrop",
                    "exact_next_drop_ge_5pct",
                    "cold_feature_path_all_rows",
                    family,
                    slug,
                    adapter,
                    variant,
                )
            )

    for branch, target, target_representation in [
        ("future_price", "multi_horizon_price_vnd", "DIRECT_VND"),
        ("future_price", "multi_horizon_log_price_ratio", "LOG_FUTURE_OVER_CURRENT"),
        ("willdrop", "exact_next_drop_ge_5pct", "BINARY"),
    ]:
        # All architectures first use K=14. GRU additionally screens K=7/21;
        # the promoted architecture repeats K sensitivity in Stage 2. This
        # tests sequence length without multiplying every architecture by K.
        for architecture in ["RNN", "GRU", "LSTM", "BiLSTM"]:
            rows.append(
                candidate(
                    f"{branch.upper()}_{target_representation}_MARKETSEQ_{architecture}_L14",
                    "STAGE1_SEQUENCE_SCREEN",
                    branch,
                    target,
                    "cold_feature_path_all_rows",
                    "RecurrentSequence",
                    architecture,
                    "EA07_MARKET_EVENT_SEQUENCE",
                    "STATIC_PLUS_MARKET_SEQUENCE",
                    "market_event_sequence",
                    "14",
                    target_representation=target_representation,
                )
            )
        for length in [7, 21]:
            rows.append(
                candidate(
                    f"{branch.upper()}_{target_representation}_MARKETSEQ_GRU_L{length}",
                    "STAGE1_SEQUENCE_SCREEN",
                    branch,
                    target,
                    "cold_feature_path_all_rows",
                    "RecurrentSequence",
                    "GRU",
                    "EA07_MARKET_EVENT_SEQUENCE",
                    "STATIC_PLUS_MARKET_SEQUENCE",
                    "market_event_sequence",
                    str(length),
                    target_representation=target_representation,
                )
            )
    for branch, target in [
        ("future_price", "PROMOTED_STAGE1_TARGET_REPRESENTATION"),
        ("willdrop", "exact_next_drop_ge_5pct"),
    ]:
        for architecture in ["RNN", "GRU", "LSTM", "BiLSTM"]:
            rows.append(
                candidate(
                    f"{branch.upper()}_SCHEDULESEQ_{architecture}_L7",
                    "STAGE2_WARM_SEQUENCE",
                    branch,
                    target,
                    "warm_only_matched_rows",
                    "RecurrentSequence",
                    architecture,
                    "EA07_SAME_SCHEDULE_SEQUENCE",
                    "COLD_ADAPTER_PLUS_WARM_SEQUENCE",
                    "same_schedule_history_sequence",
                    "7",
                    "RUN_ONLY_AFTER_STABLE_COLD_FAMILY_EXISTS",
                    target_representation="PROMOTED_FROM_STAGE1",
                )
            )

    rows.extend(
        [
            candidate(
                "REG_PATH_CHRONOS2_ZERO_SHOT",
                    "STAGE3_FOUNDATION_DIAGNOSTIC",
                "future_price",
                "multi_horizon_price_vnd",
                    "market_context_rows_not_offer_target_rows",
                "FoundationTimeSeries",
                "Chronos2",
                "PINNED_ZERO_SHOT",
                    "MARKET_MEDIAN_CONTEXT_ONLY",
                "market_event_sequence",
                    "MODEL_CONTEXT_LIMIT",
                    "BLOCKED_UNTIL_OFFER_LEVEL_TARGET_ADAPTER_IS_PREFIT_DECLARED",
            ),
            candidate(
                "WD_EXACT_CHRONOS2_DERIVED",
                    "STAGE3_FOUNDATION_DIAGNOSTIC",
                "willdrop",
                "exact_next_drop_ge_5pct",
                    "market_context_rows_not_offer_target_rows",
                "FoundationTimeSeries",
                "Chronos2",
                "PINNED_ZERO_SHOT_DERIVED_PROBABILITY",
                    "MARKET_MEDIAN_SAMPLE_DROP_PROBABILITY",
                "market_event_sequence",
                    "MODEL_CONTEXT_LIMIT",
                    "BLOCKED_UNTIL_OFFER_LEVEL_TARGET_ADAPTER_IS_PREFIT_DECLARED",
            ),
        ]
    )

    for branch, target in [
        ("future_price", "multi_horizon_price_vnd"),
        ("relative_h0", "current_fair_value_vnd"),
        ("willdrop", "exact_next_drop_ge_5pct"),
    ]:
        rows.append(
            candidate(
                f"{branch.upper()}_PROMOTED_TABULAR_TEMPLATE",
                "STAGE2_PROMOTED_TUNING",
                branch,
                target,
                "cold_then_warm_matched",
                "PROMOTED_FROM_STAGE1",
                "bounded_tuning",
                "LEGAL_ADAPTER_FOR_BRANCH",
                "FULL_NONREDUNDANT_PLUS_WARM_ABLATION",
                promotion_gate="ONLY_FAMILIES_PASSING_BRANCH_BASELINE_AND_STABILITY",
            )
        )

    rows.append(
        candidate(
            "FUTURE_PRICE_EXACT_SPECIALIST_TEMPLATE",
            "STAGE2_EXACT_SPECIALIST",
            "future_price",
            "exact_next_price_vnd",
            "matched_exact_next_rows",
            "PROMOTED_FROM_STAGE1",
            "bounded_tuning",
            "LEGAL_ADAPTER_FOR_BRANCH",
            "SAME_FEATURE_VARIANT_AS_PROMOTED_PATH_MODEL",
            promotion_gate="ONLY_REPLACE_EXACT_SLICE_IF_EVERY_FOLD_BEATS_PATH_MODEL_AND_BASELINE",
        )
    )

    rows.append(
        candidate(
            "DUD1_INTRADAY_PROMOTED_TABULAR_TEMPLATE",
            "STAGE2_DUD1_INTRADAY_AUXILIARY",
            "willdrop_intraday",
            "dud1_am_to_pm_drop_ge_5pct",
            "matched_dud1_am_rows",
            "PROMOTED_FROM_STAGE1_WILLDROP",
            "bounded_tuning",
            "LEGAL_WILLDROP_ADAPTER",
            "SAME_FEATURE_VARIANT_AS_PROMOTED_WILLDROP",
            promotion_gate="RUN_ONLY_AFTER_STABLE_EXACT_NEXT_FAMILY_EXISTS",
        )
    )

    rows.extend(
        [
            candidate(
                "SOURCE_MIXED_SENSITIVITY_TEMPLATE",
                "STAGE3_SENSITIVITY",
                "all",
                "same_as_promoted_candidate",
                "matched_rows",
                "BEST_STABLE_REFERENCE",
                "fixed_from_stage2",
                "same_adapter",
                "TRIP_ONLY_VS_MIXED_WITH_ERA_ISOLATED_STATE",
                promotion_gate="SENSITIVITY_ONLY_NOT_WINNER_SELECTION",
            ),
            candidate(
                "VALIDATION_PROTOCOL_SENSITIVITY_TEMPLATE",
                "STAGE3_SENSITIVITY",
                "all",
                "same_as_promoted_candidate",
                "matched_rows",
                "BEST_STABLE_REFERENCE",
                "fixed_from_stage2",
                "same_adapter",
                "EXPANDING_VS_SLIDING35_VS_RECENCY21",
                promotion_gate="SENSITIVITY_ONLY_HEADLINE_REMAINS_EXPANDING",
            ),
            candidate(
                "OOF_ENSEMBLE_TEMPLATE",
                "STAGE3_OPTIONAL_ENSEMBLE",
                "all",
                "branch_specific",
                "matched_rows",
                "TOP_STABLE_COMPONENTS_ONLY",
                "nonnegative_oof_blend",
                "component_adapters_unchanged",
                "NO_NEW_FEATURES",
                promotion_gate="ONLY_IF_ALL_COMPONENTS_PASS_AND_RESIDUAL_CORRELATION_LT_0_95",
            ),
            candidate(
                "PROMOTED_SEQUENCE_LENGTH_SENSITIVITY_TEMPLATE",
                "STAGE2_SEQUENCE_LENGTH_SENSITIVITY",
                "all",
                "branch_specific",
                "matched_rows",
                "RecurrentSequence",
                "promoted_architecture",
                "EA07_MARKET_EVENT_SEQUENCE",
                "K7_VS_K14_VS_K21",
                "market_event_sequence",
                "7|14|21",
                promotion_gate="RUN_ONLY_FOR_PROMOTED_STABLE_SEQUENCE_ARCHITECTURE",
            ),
        ]
    )
    return rows


def resolve_stage1_feature_lists(registry: pd.DataFrame, adapters: pd.DataFrame) -> pd.DataFrame:
    """Materialize the exact feature list behind every Stage-1 tabular row."""

    adapter_features = {
        row.adapter: [value.strip() for value in row.features.split("|")]
        for row in adapters.itertuples(index=False)
    }
    resolved: list[str] = []
    counts: list[int] = []
    for row in registry.itertuples(index=False):
        if row.stage != "STAGE1_FAMILY_SCREEN":
            resolved.append(f"CONTRACT:{row.adapter}")
            counts.append(0)
            continue
        if row.adapter not in adapter_features:
            raise RuntimeError(f"Missing model-family adapter: {row.adapter}")
        full = adapter_features[row.adapter]
        static = [
            feature
            for feature in full
            if feature not in MARKET_LEVEL_FEATURES and feature not in MARKET_DIRECTION_FEATURES
        ]
        market = [feature for feature in full if feature in MARKET_LEVEL_FEATURES]
        direction = [feature for feature in full if feature in MARKET_DIRECTION_FEATURES]
        if row.feature_variant == "STATIC_CURRENT_STATE":
            features = static
        elif row.feature_variant == "PLUS_MARKET_LEVEL_CONTEXT":
            features = static + market
        elif row.feature_variant == "PLUS_PRIOR_MARKET_DIRECTION":
            features = static + market + direction
        else:
            raise RuntimeError(f"Unknown Stage-1 feature variant: {row.feature_variant}")
        if not features or len(features) != len(set(features)):
            raise RuntimeError(f"Invalid resolved feature list for {row.candidate_id}")
        resolved.append(" | ".join(features))
        counts.append(len(features))
    output = registry.copy()
    output["resolved_features"] = resolved
    output["resolved_feature_count"] = counts
    config_map = {
        "DecisionTreeRegressor": "DECISION_TREE_REG_SCREEN_V2",
        "DecisionTreeClassifier": "DECISION_TREE_CLS_SCREEN_V2",
        "RF": "RF_SCREEN_V1",
        "XGBoost": "XGB_SCREEN_V1",
        "CatBoost": "CATBOOST_SCREEN_V1",
        "HistGBM": "HISTGBM_SCREEN_V1",
        "Ridge": "RIDGE_SCREEN_V1",
        "ElasticNet": "ELASTICNET_SCREEN_V1",
        "LogisticRegression": "LOGISTIC_SCREEN_V1",
        "LinearSVM": "LINEARSVM_SCREEN_V1",
        "MLP": "MLP_SCREEN_V1",
        "RecurrentSequence": "RECURRENT_SCREEN_V1",
        "FoundationTimeSeries": "CHRONOS2_ZERO_SHOT_V1",
    }
    output["config_id"] = output["model_family"].map(config_map).fillna("CONDITIONAL_STAGE_CONFIG")
    return output


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        ROOT / "artifacts/feature_research/reports/PREDICTION_CONTRACT_V2.txt",
        TABLE_DIR / "model_free_gates_execution_manifest.json",
        TABLE_DIR / "fs07_prefit_contract_manifest.json",
        TABLE_DIR / "branch_eda_manifest.json",
        TABLE_DIR / "ea04_redundancy_summary.json",
        TABLE_DIR / "ea05_causal_baseline_summary.json",
        TABLE_DIR / "ea06_listed_price_semantics_summary.json",
        TABLE_DIR / "ea07_sequence_structure_summary.json",
        ROOT / "artifacts/feature_research/reports/OOF_PROTOCOL_RESEARCH_AND_DATA_AUDIT.txt",
        TABLE_DIR / "ea04_model_family_adapters.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Pilot contract cannot lock before pre-fit gates: {missing}")

    gate = json.loads(required[1].read_text())
    redundancy = json.loads(required[4].read_text())
    if gate["status"] != "ALL_MODEL_FREE_FEATURE_SELECTION_GATES_REPRODUCED_NO_MODEL_FIT":
        raise RuntimeError("Model-free gate manifest is not ready")
    if gate["model_fit_called"] or gate["uses_archived_110_115_outcomes"]:
        raise RuntimeError("Pre-fit evidence is contaminated by fit or archived outcomes")
    if redundancy.get("cross_collection_era_history_violations", -1) != 0:
        raise RuntimeError("Same-schedule history crosses collection era")
    if redundancy.get("cross_collection_era_market_anchor_violations", -1) != 0:
        raise RuntimeError("Market anchor crosses collection era")

    rows = registry_rows()
    registry = resolve_stage1_feature_lists(pd.DataFrame(rows), pd.read_csv(required[-1]))
    if registry["candidate_id"].duplicated().any():
        raise RuntimeError("Candidate ids are not unique")
    registry.to_csv(TABLE_DIR / "prefit_candidate_registry.csv", index=False)

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    contract = {
        "status": "AUTHORITATIVE_V3_BEFORE_FIRST_PILOT_FIT",
        "supersedes": "historical_training_acceptance_contract.json",
        "supersession_reason": "separate selection from confirmation and screen drift-resistant regression target before any model fit",
        "created_at": now,
        "pilot_cutoff": str(CUTOFF.date()),
        "pilot_role": "OPEN_EXPLORATORY_DEVELOPMENT_NOT_CONFIRMATORY_TEST",
        "prediction_unit": "EVERY_OFFER_IN_COMPLETED_AM_PM_DATABASE_BATCH",
        "identity_contract": {
            "prediction_granularity": "one route-flight_date-airline-departure_HHMM schedule-slot offer",
            "scrape_session_role": "observation timing only; never collapses departure offers",
            "flight_no_norm_semantics": "airline-HHMM schedule-slot proxy",
        },
        "runtime_contract": RUNTIME_CONTRACT,
        "source_contract": {
            "primary_lane": "TRIP_COM_BROWSER_ERA_ONLY",
            "primary_start": "2026-05-15",
            "mixed_lane_role": "SENSITIVITY_ONLY",
            "source_feature_in_primary_model": False,
            "cross_collection_era_lag_anchor_sequence_allowed": False,
        },
        "targets": {
            "regression_primary_product": "POOLED_DIRECT_MULTI_HORIZON_PRICE_PATH_WITH_TARGET_DUD",
            "regression_primary_operational": "EXACT_NEXT_SPECIALIST",
            "stage1_regression_screen": "POOLED_DIRECT_MULTI_HORIZON_WITH_EXACT_NEXT_MANDATORY_SLICE",
            "regression_secondary": "H0_CURRENT_FAIR_VALUE",
            "classification": "EXACT_NEXT_DROP_AT_LEAST_5_PERCENT",
            "classification_scope": "DUD60_TO_DUD2_WITH_SAME_AM_PM_BAND",
            "dud1_auxiliary": "DUD1_AM_TO_SAME_DAY_PM_DROP_AT_LEAST_5_PERCENT",
            "dud1_pm": "TERMINAL_NO_LATER_MONITORED_OBSERVATION",
            "missing_target_policy": "MASK_NOT_INTERPOLATE",
            "stage1_regression_target_representations": {
                "DIRECT_VND": "target_price_vnd",
                "LOG_FUTURE_OVER_CURRENT": "log(target_price_vnd / price_vnd)",
                "comparison_metric": "inverse-transform every prediction to VND and score on identical rows",
                "baseline": "current-price carry-forward on identical rows",
            },
        },
        "choice_set_policy": {
            "role": "deterministic current-batch recommendation layer; not a model target",
            "same_route": True,
            "same_flight_date": True,
            "all_airlines": True,
            "departure_time_distance_minutes_max": 60,
            "window_direction": "SYMMETRIC_PLUS_MINUS_60_MINUTES",
            "each_alternative_has_independent_forecasts": True,
        },
        "foundation_model_scope": {
            "chronos2_stage1_winner_eligible": False,
            "reason": "registered context is an irregular aggregate market-median series and no legal adapter maps it to each offer and arbitrary target_dud",
            "allowed_role_before_adapter_exists": "documented optional diagnostic only; zero Stage-1 inference runs",
            "reentry_gate": "declare an offer-level target adapter, regular-time semantics, matched baseline and coverage before execution",
        },
        "validation": {
            "primary_protocol": "EXPANDING_WINDOW_ROLLING_ORIGIN",
            "selection_folds": SELECTION_FOLDS,
            "confirmation_blocks": CONFIRMATION_BLOCKS,
            "selection_common_exact_transitions": [
                "2->1", "3->2", "5->3", "7->5", "10->7",
                "14->10", "21->14", "30->21", "45->30", "60->45",
            ],
            "confirmation_common_exact_transitions": [
                "2->1", "3->2", "5->3", "7->5", "10->7", "14->10",
            ],
            "selection_common_multi_horizon_transitions": SELECTION_COMMON_MULTI_HORIZON_TRANSITIONS,
            "confirmation_common_multi_horizon_transitions": CONFIRMATION_COMMON_MULTI_HORIZON_TRANSITIONS,
            "equal_fold_mean": True,
            "row_weighted_mean_for_selection": False,
            "latest_fold_reported_separately": True,
            "winner_aggregation": "equal weight over structurally comparable DUD transitions present in every fold",
            "non_common_transition_role": "trained and reported diagnostically, but excluded from winner aggregation",
            "exact_next_promotion_gate": "equal mean over locked common exact-next transitions; row-weighted exact metric is diagnostic only",
            "warm_router_comparison": "matched warm rows, equal locked-transition mean in each selection fold",
            "natural_fold_mix_role": "diagnostic only; never winner selection",
            "latest_fold_gate": "positive only on transition cells comparable with earlier folds",
            "cold_reporting": "fold x transition x sample-size; no unconditional cold claim",
            "oof_definition": "predictions on each temporal validation fold from models fit only on earlier permitted rows",
            "random_kfold_allowed": False,
            "c1_c2_model_reselection_allowed": False,
            "c2_threshold_reselection_allowed": False,
            "regression_h0_train_rule": "feature_time < validation_start",
            "labeled_task_train_rule": "feature_time < validation_start AND label_time < validation_start",
            "validation_rule": "validation_start <= feature_time <= validation_end AND target observed AND mature",
            "cluster_ci_unit": "schedule_slot_id across all DUD and AM_PM observations",
            "sensitivity_protocols": {
                "sliding_window_days": 35,
                "recency_weight_half_life_days": 21,
                "role": "reference-candidate sensitivity only; not a new validation truth",
            },
        },
        "feature_screen": {
            "incremental_variants": [
                "STATIC_CURRENT_STATE",
                "PLUS_MARKET_LEVEL_CONTEXT",
                "PLUS_PRIOR_MARKET_DIRECTION",
            ],
            "same_variants_across_all_compatible_tabular_model_families": True,
            "paired_on_same_rows": True,
            "warm_ablation_on_matched_warm_rows": True,
            "market_direction_tree_representation": "RAW",
            "market_direction_linear_neural_representation": "SIGNED_LOG1P_THEN_TRAIN_FOLD_SCALER",
        },
        "tuning": {
            "stage1": "fixed predeclared screening complexity",
            "stage2": "bounded tuning only for promoted stable families",
            "tree_trials_per_promoted_family": 30,
            "linear_trials_per_promoted_family": 15,
            "neural_trials_per_promoted_family": 20,
            "random_seeds": [115, 215, 315],
            "early_stopping_uses_training_inner_tail_only": True,
            "validation_fold_as_early_stopping_set_allowed": False,
            "stage1_fixed_configs": STAGE1_FIXED_CONFIGS,
            "classification_balance": "weights computed inside each training fold only",
        },
        "multi_horizon_training": {
            "pooled_direct_key": "current_dud plus target_dud",
            "minimum_training_rows_per_transition": 2000,
            "transition_weighting": "equal total weight per eligible current_dud-to-target_dud cell within each training fold",
            "unsupported_transition_fallback": "strongest causal baseline",
        },
        "regression_acceptance": {
            "baseline_multi_horizon": "CURRENT_PRICE_CARRY_FORWARD_ON_MATCHED_ROWS",
            "baseline_exact_next": "CURRENT_PRICE_CARRY_FORWARD_ON_MATCHED_ROWS",
            "baseline_h0_cold": "PRIOR_ROUTE_AIRLINE_DUD_MARKET_MEDIAN",
            "baseline_h0_warm": "PREVIOUS_SAME_SCHEDULE_CARRY_FORWARD",
            "primary_skill": "1 - MAE_MODEL / MAE_BASELINE",
            "target_representation_selection": "same fold-transition-population cells; VND-space metrics after inverse transform",
            "mean_equal_fold_skill_gt": 0.0,
            "every_fold_skill_gt": 0.0,
            "latest_fold_skill_gt": 0.0,
            "r2_gt": 0.0,
            "multi_horizon_aggregation": "equal horizon mean plus each horizon separately",
            "stage1_promotion_requires": [
                "multi_horizon equal-fold skill > 0 on every fold",
                "exact-next slice skill > 0 on every fold",
                "latest-fold multi-horizon and exact-next skill > 0",
            ],
            "exact_specialist_rule": "Stage-2 exact specialist may replace only the exact-next slice when it beats the promoted path model and carry-forward baseline on every fold",
            "fallback_if_no_model_passes": "serve strongest causal baseline for that population",
        },
        "willdrop_acceptance": {
            "ranking_metric": "PR_AUC_MODEL_MINUS_WITHIN_FOLD_DUD_ONLY_PR_AUC",
            "primary_aggregation": "equal weight over preflight-resolved common exact-next transition cells",
            "required_reporting_granularity": "fold x current_dud x target_dud x cold_warm",
            "natural_prevalence_mix_for_selection": False,
            "mean_equal_fold_delta_gt": 0.0,
            "every_fold_delta_gt": 0.0,
            "latest_fold_delta_gt": 0.0,
            "calibration_metrics": ["Brier", "ECE"],
            "fallback_if_no_model_passes": "DUD_ONLY_PROBABILITY_PLUS_UNCERTAIN_POLICY",
        },
        "safety_policy": {
            "probability_direction": "higher means WAIT",
            "internal_actions": ["BUY", "UNCERTAIN", "WAIT"],
            "ui_actions": ["BUY", "WAIT"],
            "ui_uncertain_mapping": "WAIT_WITH_LOW_CONFIDENCE_REASON",
            "unsafe_buy_denominator": "false_BUY / (true_BUY + false_BUY)",
            "wait_recall_min_each_policy_validation_slice": 0.75,
            "unsafe_buy_wilson_upper_max": 0.15,
            "action_coverage_min": 0.60,
            "buy_coverage_min": 0.20,
            "false_buy_vnd_cost_multiplier": 2.0,
            "threshold_candidates": ["GLOBAL_TWO_THRESHOLD", "DUD_SPECIFIC_WITH_GLOBAL_FALLBACK"],
            "dud_specific_min_actual_buy": 500,
            "dud_specific_min_actual_wait": 500,
            "selection_order": [
                "satisfy safety and coverage gates",
                "minimize asymmetric expected VND regret",
                "maximize macro F1",
                "prefer simpler global policy on tie",
            ],
            "policy_development": "temporal OOF development only; later prospective holdout cannot retune",
            "calibrator_fit_folds": ["PILOT_S1", "PILOT_S2", "PILOT_S3"],
            "policy_threshold_selection_fold": "PILOT_C1",
            "frozen_policy_confirmation_fold": "PILOT_C2",
            "pilot_disclosure": "C1 and C2 are internal chronological confirmation blocks inside an open pilot, not an external prospective holdout",
            "no_feasible_policy_action": "mark policy infeasible; internal UNCERTAIN maps to user-facing WAIT; never force unsafe BUY",
            "dud1_am_action": "use separately calibrated DUD1 AM-to-PM auxiliary when label support is sufficient",
            "dud1_pm_action": "BUY_NOW_OR_COMPARE_CURRENT_ALTERNATIVES with probability NA and explicit reason code",
            "dud_threshold_rule": "DUD-specific only when minimum actual BUY and WAIT support pass; otherwise predeclared hierarchical/global fallback",
        },
        "uncertainty": {
            "quantiles": [0.1, 0.5, 0.9],
            "fit_after_point_winner": True,
            "quantile_crossing_fix": "monotone sort or constrained objective declared in artifact",
            "metrics": ["pinball", "P10_P90_coverage", "interval_width"],
        },
        "ensemble": {
            "optional": True,
            "weights": "nonnegative, sum to one, learned from temporal OOF only",
            "component_gate": "every component independently passes branch acceptance",
            "diversity_gate": "matched-row residual or score correlation < 0.95",
            "must_beat_best_component_each_fold": True,
        },
        "non_negotiable": [
            "no row after 2026-07-10",
            "no archived 110/115 outcome used for candidate selection",
            "no random split",
            "no preprocessing fit outside training fold",
            "no current-row price in h0 features",
            "no cross-era lag anchor label or sequence state",
            "no current-batch history",
            "no session-label partition for warm history; any earlier completed batch is legal",
            "no future backfill or interpolation",
            "no policy metric with false_BUY divided by actual_WAIT",
            "no mixing DUD1 AM-to-PM with booking-window exact-next metrics",
            "no recommendation outside same route/date and plus-minus 60 departure minutes",
            "no C1 or C2 model-family, feature, target-representation, hyperparameter or ensemble reselection",
        ],
        "candidate_rows": len(registry),
        "stage1_actual_model_fits": int(
            registry["stage"].isin(["STAGE1_FAMILY_SCREEN", "STAGE1_SEQUENCE_SCREEN"])
            .sum()
            * len(SELECTION_FOLDS)
        ),
        "stage1_zero_shot_inference_runs": 0,
    }
    contract_path = TABLE_DIR / "prefit_training_acceptance_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    stage_counts = registry.groupby("stage", observed=True).size().rename("registered_rows").reset_index()
    model_counts = registry.groupby(["stage", "model_family"], observed=True).size().rename("registered_rows").reset_index()
    stage_table = markdown_table(stage_counts)
    model_table = markdown_table(model_counts)
    REPORT.write_text(
        f"""# Leakage-controlled prefit training contract

> Status: `AUTHORITATIVE_V3_BEFORE_FIRST_PILOT_FIT`
>
> Created `{now}`. The pilot is open exploratory development, not a sealed
> confirmatory test. No estimator, calibrator, threshold or ensemble was fitted
> by this contract builder.

## Why training is now staged

Every compatible tabular family first sees the same incremental comparison:
static/current state -> market level -> strictly-prior market direction. This
answers whether a feature family helps RF, boosting, linear and neural models
without forcing one representation onto every algorithm. Future-price models
also compare direct VND with `log(future/current)` on identical rows, then score
both in VND after inverse transform. Only families that
beat the correct baseline on every temporal fold are tuned or extended.

For regression, Stage 1 fits the declared **pooled direct multi-horizon price
path**, not a different surrogate task. Exact-next is a mandatory reported slice
and promotion gate of that same path model. In Stage 2, an exact-next specialist
may replace only that operational slice, and only if it beats both the promoted
path model and carry-forward baseline on every matched temporal fold. This keeps
the full path as the primary product without sacrificing next-window advice.

Chronos-2 is recorded only as a blocked foundation diagnostic. The available
sequence is an irregular aggregate market-median context, while the product
target is an individual offer at an arbitrary `target_dud`. Without a declared
offer-level adapter, treating its zero-shot median forecast as an offer forecast
would compare different tasks. It therefore has zero Stage-1 inference runs and
cannot enter winner selection.

{stage_table}

{model_table}

## Rolling-origin is not a feature

Rolling-origin is the evaluation protocol. Changing feature candidates does not
make random K-fold preferable: each validation prediction must still come from
strictly earlier data. The concatenated validation predictions are temporal
OOF predictions. They may support calibration or a constrained ensemble, but
they never make future rows part of training.

## Three guards to remember

1. **Source state isolation:** Trip primary rows cannot inherit Fli lag,
   anchor, label transition or sequence state.
2. **Label-time purge:** future-price and Will-Drop training rows require
   `label_time < validation_start`; splitting only by feature date is not enough.
3. **Matched baseline:** every model/feature claim is measured on the same rows,
   horizon and population as its strongest legal baseline.

These three guards are important, important, important. A run that violates any
one of them is invalid even when its metric is attractive.
""",
        encoding="utf-8",
    )

    inputs = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in required
    ]
    outputs = [registry_path := TABLE_DIR / "prefit_candidate_registry.csv", contract_path, REPORT]
    manifest = {
        "status": "PREFIT_CONTRACT_COMPLETE_NO_MODEL_FIT",
        "supersedes": "historical_prefit_training_manifest.json",
        "created_at": now,
        "cutoff": str(CUTOFF.date()),
        "inputs": inputs,
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in outputs
        ],
        "model_fit_called": False,
        "calibrator_fit_called": False,
        "threshold_selected": False,
        "uses_archived_110_115_outcomes": False,
    }
    (TABLE_DIR / "prefit_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[PILOT CONTRACT PASS] candidates={len(registry)} no_model_fit=true")


if __name__ == "__main__":
    main()
