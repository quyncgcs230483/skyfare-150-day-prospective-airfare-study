#!/usr/bin/env python3
"""One-time evaluation of committed Test 2 predictions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from skyfare.evaluation.prospective_test_two.contract import BUY_WAIT_THRESHOLD, NOMINAL_FRICTION_VND, QUANTILES
from skyfare.evaluation.metrics import classification_metrics, distribution_metrics, point_metrics, ranking_metrics
from skyfare.evaluation.prospective_test_two.runtime import (
    CLASS_LABELS,
    OUTPUT_ROOT,
    REG_LABELS,
    ROOT,
    load_dev_frame,
    preflight_sha256,
    sha256,
    write_json_atomic,
)


def _join(path: str, labels: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.read_parquet(OUTPUT_ROOT / "frozen_predictions" / path)
    label_payload = labels.copy()
    if "query_id" in prediction and "query_id" in label_payload:
        identity = prediction[["row_key", "query_id"]].merge(
            label_payload[["row_key", "query_id"]],
            on="row_key",
            how="inner",
            validate="one_to_one",
            suffixes=("_prediction", "_label"),
        )
        if len(identity) != len(prediction) or not identity[
            "query_id_prediction"
        ].astype(str).equals(identity["query_id_label"].astype(str)):
            raise RuntimeError("prediction/label query identities disagree")
        label_payload = label_payload.drop(columns=["query_id"])
    merged = prediction.merge(label_payload, on="row_key", how="inner", validate="one_to_one")
    if len(merged) != len(prediction) or len(merged) != len(labels):
        raise RuntimeError(
            f"prediction/label row coverage mismatch prediction={len(prediction)} "
            f"labels={len(labels)} joined={len(merged)}"
        )
    return merged


def _bootstrap(delta: np.ndarray, dates: pd.Series, seed: int) -> dict[str, float]:
    work = pd.DataFrame({"delta": np.asarray(delta, dtype=float), "date": pd.to_datetime(dates).dt.normalize()})
    daily = work.groupby("date", sort=True)["delta"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(daily, size=(4000, len(daily)), replace=True), axis=1)
    return {
        "mean_model_minus_baseline": float(np.mean(daily)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "day_blocks": int(len(daily)),
        "draws": 4000,
    }


def _policy(frame: pd.DataFrame, action: str) -> tuple[dict[str, float], np.ndarray]:
    current = frame["price_vnd"].to_numpy(dtype=float)
    future = frame["target_price_vnd"].to_numpy(dtype=float)
    if action == "FROZEN":
        wait = frame["probability"].to_numpy(dtype=float) >= BUY_WAIT_THRESHOLD
    elif action == "BUY_ALL":
        wait = np.zeros(len(frame), dtype=bool)
    elif action == "WAIT_ALL":
        wait = np.ones(len(frame), dtype=bool)
    else:
        raise KeyError(action)
    oracle = np.minimum(current, future + NOMINAL_FRICTION_VND)
    regret = np.maximum(np.where(wait, future + NOMINAL_FRICTION_VND, current) - oracle, 0.0)
    cutoff = float(np.quantile(regret, 0.95))
    tail = regret[regret >= cutoff]
    return {
        "mean_regret_vnd": float(regret.mean()),
        "cvar95_regret_vnd": float(tail.mean()) if len(tail) else 0.0,
        "p95_regret_vnd": cutoff,
        "wait_rate": float(wait.mean()),
        "rows": int(len(frame)),
    }, regret


def main() -> None:
    commit_path = OUTPUT_ROOT / "runtime" / "prediction_commit_test2.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("status") != "COMMITTED_BEFORE_LABEL_EVALUATION":
        raise RuntimeError("Test 2 prediction commit missing")
    if commit.get("preflight_sha256") != preflight_sha256():
        raise RuntimeError("Test 2 staged inputs changed after prediction commit")
    for task, item in commit["files"].items():
        path = OUTPUT_ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"committed prediction changed before evaluation: {task}")

    class_labels = pd.read_parquet(CLASS_LABELS)
    reg_labels = pd.read_parquet(REG_LABELS)
    classification = _join("classification_test2_predictions.parquet", class_labels)
    point = _join("point_test2_predictions.parquet", reg_labels)
    distribution = _join("distribution_test2_predictions.parquet", reg_labels)
    ranking = _join("ranking_test2_predictions.parquet", reg_labels)

    class_target = classification["DROP_5PCT"].to_numpy(dtype=np.int8)
    class_probability = classification["probability"].to_numpy(dtype=float)
    class_rate = float(load_dev_frame("CLASSIFICATION")["DROP_5PCT"].mean())
    class_baseline = np.full(len(classification), np.clip(class_rate, 1e-7, 1.0 - 1e-7))
    class_metrics = classification_metrics(class_target, class_probability)
    class_base_metrics = classification_metrics(class_target, class_baseline)
    model_class_loss = -(class_target * np.log(class_probability) + (1 - class_target) * np.log1p(-class_probability)) + np.square(class_probability - class_target)
    base_class_loss = -(class_target * np.log(class_baseline) + (1 - class_target) * np.log1p(-class_baseline)) + np.square(class_baseline - class_target)

    point_target = point["target_session_price_vnd"].to_numpy(dtype=float)
    point_prediction = point["predicted_price_vnd"].to_numpy(dtype=float)
    point_baseline = point["prior_anchor_vnd"].to_numpy(dtype=float)
    point_result = point_metrics(point_target, point_prediction)
    point_base_result = point_metrics(point_target, point_baseline)

    q_columns = [f"q{int(round(level * 100)):02d}_relative_log" for level in QUANTILES]
    q_model = distribution[q_columns].to_numpy(dtype=float)
    reg_dev = load_dev_frame("REGRESSION")
    dev_target = reg_dev["target_anchor_relative_log"].to_numpy(dtype=float)
    q_baseline_values = np.quantile(dev_target, np.asarray(QUANTILES, dtype=float))
    q_baseline = np.tile(q_baseline_values, (len(distribution), 1))
    dist_target = distribution["target_anchor_relative_log"].to_numpy(dtype=float)
    dist_result = distribution_metrics(dist_target, q_model)
    dist_base_result = distribution_metrics(dist_target, q_baseline)
    levels = np.asarray(QUANTILES, dtype=float)
    model_dist_loss = 2.0 * np.mean(
        np.maximum(
            levels[None, :] * (dist_target[:, None] - q_model),
            (levels[None, :] - 1.0) * (dist_target[:, None] - q_model),
        ),
        axis=1,
    )
    base_dist_loss = 2.0 * np.mean(
        np.maximum(
            levels[None, :] * (dist_target[:, None] - q_baseline),
            (levels[None, :] - 1.0) * (dist_target[:, None] - q_baseline),
        ),
        axis=1,
    )

    ranking_frame = ranking.rename(columns={"target_session_price_vnd": "target_price_vnd"})
    ranking_result = ranking_metrics(ranking_frame, ranking["score"].to_numpy(dtype=float))
    ranking_baseline_score = -ranking["prior_anchor_vnd"].to_numpy(dtype=float)
    ranking_base_result = ranking_metrics(ranking_frame, ranking_baseline_score)

    policy_result, policy_regret = _policy(classification, "FROZEN")
    buy_result, buy_regret = _policy(classification, "BUY_ALL")
    wait_result, _ = _policy(classification, "WAIT_ALL")

    gates = {
        "classification_proper_score_beats_constant_prior": class_metrics["proper_score"] <= class_base_metrics["proper_score"],
        "point_mape_beats_prior_anchor": point_result["mape"] <= point_base_result["mape"],
        "distribution_wis_beats_empirical_residual": dist_result["wis"] <= dist_base_result["wis"],
        "distribution_quantiles_coherent": dist_result["crossing_rate"] == 0.0,
        "ranking_ndcg5_beats_prior_anchor": ranking_result["ndcg5"] >= ranking_base_result["ndcg5"],
        "policy_mean_regret_beats_buy_all": policy_result["mean_regret_vnd"] <= buy_result["mean_regret_vnd"],
        "policy_cvar_not_materially_worse_than_buy_all": policy_result["cvar95_regret_vnd"] <= buy_result["cvar95_regret_vnd"] * 1.05 + 1.0,
    }
    test1_report_path = ROOT / "artifacts/evidence/prospective_test_one/report.json"
    test1_report = json.loads(test1_report_path.read_text(encoding="utf-8"))
    test1_confirmed = (
        test1_report.get("status") == "PASS"
        and test1_report.get("test_2_access") is False
        and len(test1_report.get("gates", {})) == 7
        and all(value is True for value in test1_report.get("gates", {}).values())
    )
    test2_passed = all(gates.values())
    decision = "PASS" if test1_confirmed and test2_passed else "FAIL"
    report = {
        "status": decision,
        "evaluation_complete": True,
        "automatic_promotion": False,
        "test_2_access": True,
        "next_action": "FINAL_FREEZE_AND_REPORT" if decision == "PASS" else "RETAIN_INCUMBENT; NO_POST_TEST_TUNING",
        "two_block_confirmation": {
            "test_1": "PASS" if test1_confirmed else "FAIL",
            "test_2": "PASS" if test2_passed else "FAIL",
            "decision": "CONFIRMED" if decision == "PASS" else "NOT_CONFIRMED",
            "recipe_changed_between_blocks": False,
        },
        "classification": {
            "model": class_metrics,
            "baseline_constant_dev_rate": class_base_metrics,
            "paired_day_bootstrap": _bootstrap(model_class_loss - base_class_loss, classification["feature_time"], 24201),
        },
        "point": {
            "model": point_result,
            "baseline_prior_anchor": point_base_result,
            "paired_day_bootstrap": _bootstrap(
                np.abs(point_prediction - point_target) / np.maximum(point_target, 1.0)
                - np.abs(point_baseline - point_target) / np.maximum(point_target, 1.0),
                point["label_time"],
                24202,
            ),
        },
        "distribution": {
            "model": dist_result,
            "baseline_empirical_dev_residual": dist_base_result,
            "paired_day_bootstrap": _bootstrap(model_dist_loss - base_dist_loss, distribution["label_time"], 24203),
        },
        "ranking": {"model": ranking_result, "baseline_prior_anchor": ranking_base_result},
        "buy_wait": {
            "frozen": policy_result,
            "buy_all": buy_result,
            "wait_all": wait_result,
            "paired_day_bootstrap_vs_buy_all": _bootstrap(policy_regret - buy_regret, classification["feature_time"], 24204),
            "threshold": BUY_WAIT_THRESHOLD,
            "friction_vnd": NOMINAL_FRICTION_VND,
        },
        "gates": gates,
    }
    write_json_atomic(OUTPUT_ROOT / "analysis" / "TEST_2_REPORT.json", report)
    print(f"[TEST2 EVALUATION {decision}] gates={sum(gates.values())}/{len(gates)}", flush=True)


if __name__ == "__main__":
    main()
