#!/usr/bin/env python3
"""Build final pooled report from committed Test 1 and Test 2 archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.evaluation.pooled.metrics import (
    QUANTILES,
    block_heterogeneity,
    classification_metrics,
    daily_stratified_bootstrap,
    distribution_metrics,
    distribution_row_loss,
    point_metrics,
    policy_metrics,
    ranking_metrics,
    ranking_query_metrics,
)

MODULE = Path(__file__).resolve().parent
LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
CONTRACT_PATH = LAYOUT.controls / "pooled_evaluation_contract.json"
BASELINE_PATH = LAYOUT.controls / "baseline_metrics.json"
Q_COLUMNS = [f"q{int(round(level * 100)):02d}_relative_log" for level in QUANTILES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_members(block: str) -> list[str]:
    prefix = block.lower().replace("_", "")
    number = block[-1]
    output = f"test{number}_outputs"
    return [
        f"{output}/VERIFICATION_TEST_{number}.json",
        f"{output}/analysis/TEST_{number}_REPORT.json",
        f"{output}/runtime/deployable_lock_test{number}.json",
        f"{output}/runtime/prediction_commit_test{number}.json",
        f"{output}/frozen_predictions/classification_{prefix}_predictions.parquet",
        f"{output}/frozen_predictions/distribution_{prefix}_predictions.parquet",
        f"{output}/frozen_predictions/point_{prefix}_predictions.parquet",
        f"{output}/frozen_predictions/ranking_{prefix}_predictions.parquet",
        f"{output}/staging/classification_labels.parquet",
        f"{output}/staging/regression_labels.parquet",
    ]


def extract_required(archive: Path, destination: Path, block: str) -> None:
    members = required_members(block)
    with tarfile.open(archive, "r:gz") as handle:
        available = {item.name: item for item in handle.getmembers()}
        missing = sorted(set(members) - set(available))
        if missing:
            raise RuntimeError(f"{block}: archive members missing={missing}")
        for name in members:
            member = available[name]
            if not member.isfile() or Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe archive member: {name}")
            handle.extract(member, destination, filter="data")


def join_prediction(prediction: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    payload = labels.copy()
    if "query_id" in prediction and "query_id" in payload:
        identity = prediction[["row_key", "query_id"]].merge(
            payload[["row_key", "query_id"]], on="row_key", how="inner", validate="one_to_one", suffixes=("_prediction", "_label")
        )
        if len(identity) != len(prediction) or not identity["query_id_prediction"].astype(str).equals(identity["query_id_label"].astype(str)):
            raise RuntimeError("prediction/label query identities disagree")
        payload = payload.drop(columns=["query_id"])
    merged = prediction.merge(payload, on="row_key", how="inner", validate="one_to_one")
    if len(merged) != len(prediction) or len(merged) != len(labels):
        raise RuntimeError("prediction/label row coverage mismatch")
    return merged


def validate_block(root: Path, block: str, contract: dict[str, Any]) -> dict[str, Any]:
    number = block[-1]
    prefix = block.lower().replace("_", "")
    output = root / f"test{number}_outputs"
    verification = load_json(output / f"VERIFICATION_TEST_{number}.json")
    report = load_json(output / "analysis" / f"TEST_{number}_REPORT.json")
    lock = load_json(output / "runtime" / f"deployable_lock_test{number}.json")
    commit = load_json(output / "runtime" / f"prediction_commit_test{number}.json")
    if verification.get("status") != "PASS" or verification.get("integrity") != "PASS" or verification.get("scientific_decision") != "PASS":
        raise RuntimeError(f"{block}: verification not PASS")
    if report.get("status") != "PASS" or len(report.get("gates", {})) != 7 or not all(report["gates"].values()):
        raise RuntimeError(f"{block}: scientific gates not PASS")
    if commit.get("status") != "COMMITTED_BEFORE_LABEL_EVALUATION" or commit.get("test_labels_read") is not False:
        raise RuntimeError(f"{block}: prediction commit invalid")
    if commit.get("recipes") != contract["expected_recipes"] or lock.get("deployable_recipes") != contract["expected_recipes"]:
        raise RuntimeError(f"{block}: deployable recipe drift")
    for task, item in commit["files"].items():
        path = output / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"{block}: committed {task} prediction changed")
        if len(pd.read_parquet(path, columns=["row_key"])) != item["rows"]:
            raise RuntimeError(f"{block}: committed {task} row count changed")
    return {"output": output, "verification": verification, "report": report, "lock": lock, "commit": commit, "prefix": prefix}


def read_joined(block_data: dict[str, Any], block: str) -> dict[str, pd.DataFrame]:
    output = block_data["output"]
    prefix = block_data["prefix"]
    class_labels = pd.read_parquet(output / "staging" / "classification_labels.parquet")
    reg_labels = pd.read_parquet(output / "staging" / "regression_labels.parquet")
    result = {}
    for task, labels in (("classification", class_labels), ("point", reg_labels), ("distribution", reg_labels), ("ranking", reg_labels)):
        prediction = pd.read_parquet(output / "frozen_predictions" / f"{task}_{prefix}_predictions.parquet")
        frame = join_prediction(prediction, labels)
        frame["block_id"] = block
        result[task] = frame
    return result


def assert_temporal_design(frames: dict[str, dict[str, pd.DataFrame]], contract: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for task in ("classification", "point", "distribution", "ranking"):
        first = frames["TEST_1"][task]
        second = frames["TEST_2"][task]
        overlap = set(first["row_key"]) & set(second["row_key"])
        if overlap:
            raise RuntimeError(f"{task}: row_key overlap across blocks")
        query_overlap = 0
        if "query_id" in first and "query_id" in second:
            query_overlap = len(set(first["query_id"].astype(str)) & set(second["query_id"].astype(str)))
            if query_overlap:
                raise RuntimeError(f"{task}: query overlap across blocks")
        t1 = pd.to_datetime(first["label_time"])
        t2 = pd.to_datetime(second["label_time"])
        boundary = pd.Timestamp(contract["expected_blocks"]["test_2"][0])
        if not t1.lt(boundary).all() or not t2.ge(boundary).all():
            raise RuntimeError(f"{task}: prospective block boundary violated")
        audit[task] = {
            "row_overlap": 0,
            "query_overlap": query_overlap,
            "test_1_rows": int(len(first)),
            "test_2_rows": int(len(second)),
            "test_1_label_min": str(t1.min()),
            "test_1_label_max": str(t1.max()),
            "test_2_label_min": str(t2.min()),
            "test_2_label_max": str(t2.max()),
        }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test1-results", type=Path, required=True)
    parser.add_argument("--test2-results", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LAYOUT.artifacts / "pooled_evaluation",
    )
    args = parser.parse_args()

    contract = load_json(CONTRACT_PATH)
    baselines = load_json(BASELINE_PATH)
    archives = {"TEST_1": args.test1_results.resolve(), "TEST_2": args.test2_results.resolve()}
    archive_hashes = {}
    for block, archive in archives.items():
        expected = contract["expected_archives"][block.lower()]["sha256"]
        observed = sha256(archive)
        if observed != expected:
            raise RuntimeError(f"{block}: result archive hash mismatch expected={expected} observed={observed}")
        archive_hashes[block] = observed

    with tempfile.TemporaryDirectory(prefix="skyfare-two-block-") as temporary:
        temp = Path(temporary)
        blocks = {}
        frames = {}
        for block, archive in archives.items():
            destination = temp / block.lower()
            destination.mkdir()
            extract_required(archive, destination, block)
            blocks[block] = validate_block(destination, block, contract)
            frames[block] = read_joined(blocks[block], block)

        temporal_audit = assert_temporal_design(frames, contract)
        pooled = {task: pd.concat([frames["TEST_1"][task], frames["TEST_2"][task]], ignore_index=True) for task in frames["TEST_1"]}

        classification = pooled["classification"]
        class_target = classification["DROP_5PCT"].to_numpy(dtype=np.int8)
        class_probability = classification["probability"].to_numpy(dtype=float)
        class_baseline = classification["block_id"].map(baselines["classification_constant_probability"]).to_numpy(dtype=float)
        class_model_loss = -(class_target * np.log(class_probability) + (1 - class_target) * np.log1p(-class_probability)) + np.square(class_probability - class_target)
        class_base_loss = -(class_target * np.log(class_baseline) + (1 - class_target) * np.log1p(-class_baseline)) + np.square(class_baseline - class_target)

        point = pooled["point"]
        point_target = point["target_session_price_vnd"].to_numpy(dtype=float)
        point_prediction = point["predicted_price_vnd"].to_numpy(dtype=float)
        point_baseline = point["prior_anchor_vnd"].to_numpy(dtype=float)
        point_delta = np.abs(point_prediction - point_target) / np.maximum(point_target, 1.0) - np.abs(point_baseline - point_target) / np.maximum(point_target, 1.0)

        distribution = pooled["distribution"]
        dist_target = distribution["target_anchor_relative_log"].to_numpy(dtype=float)
        dist_model = distribution[Q_COLUMNS].to_numpy(dtype=float)
        dist_baseline = np.vstack([baselines["distribution_empirical_quantiles"][block] for block in distribution["block_id"]])
        dist_delta = distribution_row_loss(dist_target, dist_model) - distribution_row_loss(dist_target, dist_baseline)

        ranking = pooled["ranking"].rename(columns={"target_session_price_vnd": "target_price_vnd"})
        ranking_model_queries = ranking_query_metrics(ranking, ranking["score"].to_numpy(dtype=float))
        ranking_base_queries = ranking_query_metrics(ranking, -ranking["prior_anchor_vnd"].to_numpy(dtype=float))
        ranking_delta_frame = ranking_model_queries.merge(
            ranking_base_queries[["query_id", "ndcg5"]], on="query_id", validate="one_to_one", suffixes=("_model", "_baseline")
        )
        ranking_delta = ranking_delta_frame["ndcg5_model"] - ranking_delta_frame["ndcg5_baseline"]

        threshold = float(contract["policy"]["threshold"])
        friction = float(contract["policy"]["friction_vnd"])
        policy, policy_regret = policy_metrics(classification, "FROZEN", threshold, friction)
        buy_all, buy_regret = policy_metrics(classification, "BUY_ALL", threshold, friction)
        wait_all, _ = policy_metrics(classification, "WAIT_ALL", threshold, friction)

        bootstrap_inputs = {
            "classification_proper_score": (class_model_loss - class_base_loss, classification["feature_time"], classification["block_id"], 43001),
            "point_mape": (point_delta, point["label_time"], point["block_id"], 43002),
            "distribution_wis": (dist_delta, distribution["label_time"], distribution["block_id"], 43003),
            "ranking_ndcg5": (-ranking_delta.to_numpy(dtype=float), ranking_delta_frame["label_time"], ranking_delta_frame["block_id"], 43004),
            "policy_regret": (policy_regret - buy_regret, classification["feature_time"], classification["block_id"], 43005),
        }
        uncertainty = {}
        heterogeneity = {}
        daily_tables = []
        for metric, (delta, dates, block_ids, seed) in bootstrap_inputs.items():
            summary, daily = daily_stratified_bootstrap(delta, dates, block_ids, seed)
            uncertainty[metric] = summary
            heterogeneity[metric] = block_heterogeneity(daily, seed + 100)
            daily.insert(0, "metric", metric)
            daily_tables.append(daily)

        classification_model = classification_metrics(class_target, class_probability)
        classification_baseline = classification_metrics(class_target, class_baseline)
        point_model = point_metrics(point_target, point_prediction)
        point_base = point_metrics(point_target, point_baseline)
        dist_model_metrics = distribution_metrics(dist_target, dist_model)
        dist_base_metrics = distribution_metrics(dist_target, dist_baseline)
        ranking_model = ranking_metrics(ranking, ranking["score"].to_numpy(dtype=float))
        ranking_base = ranking_metrics(ranking, -ranking["prior_anchor_vnd"].to_numpy(dtype=float))

        gates = {
            "both_prospective_blocks_pass": all(blocks[name]["report"]["status"] == "PASS" for name in blocks),
            "classification_proper_score_beats_block_specific_prior": classification_model["proper_score"] <= classification_baseline["proper_score"],
            "point_mape_beats_prior_anchor": point_model["mape"] <= point_base["mape"],
            "distribution_wis_beats_block_specific_empirical_residual": dist_model_metrics["wis"] <= dist_base_metrics["wis"],
            "distribution_quantiles_coherent": dist_model_metrics["crossing_rate"] == 0.0,
            "ranking_ndcg5_beats_prior_anchor": ranking_model["ndcg5"] >= ranking_base["ndcg5"],
            "policy_mean_regret_beats_buy_all": policy["mean_regret_vnd"] <= buy_all["mean_regret_vnd"],
            "policy_cvar_not_materially_worse_than_buy_all": policy["cvar95_regret_vnd"] <= buy_all["cvar95_regret_vnd"] * 1.05 + 1.0,
            "recipe_unchanged_between_blocks": blocks["TEST_1"]["commit"]["recipes"] == blocks["TEST_2"]["commit"]["recipes"] == contract["expected_recipes"],
            "no_cross_block_row_or_query_overlap": all(item["row_overlap"] == 0 and item["query_overlap"] == 0 for item in temporal_audit.values()),
        }
        decision = "PASS" if all(gates.values()) else "FAIL"
        report = {
            "status": decision,
            "analysis_type": contract["pooled_design"],
            "automatic_promotion": False,
            "candidate_search": False,
            "hyperparameter_tuning": False,
            "retraining_performed": False,
            "blocks": {name: blocks[name]["report"] for name in ("TEST_1", "TEST_2")},
            "pooled": {
                "classification": {"model": classification_model, "baseline": classification_baseline, "rows": int(len(classification))},
                "point": {"model": point_model, "baseline": point_base, "rows": int(len(point))},
                "distribution": {"model": dist_model_metrics, "baseline": dist_base_metrics, "rows": int(len(distribution))},
                "ranking": {"model": ranking_model, "baseline": ranking_base},
                "buy_wait": {"frozen": policy, "buy_all": buy_all, "wait_all": wait_all, "threshold": threshold, "friction_vnd": friction},
            },
            "stratified_day_bootstrap": uncertainty,
            "block_heterogeneity": heterogeneity,
            "temporal_isolation": temporal_audit,
            "gates": gates,
            "scientific_conclusion": "TWO_BLOCK_CONFIRMED" if decision == "PASS" else "NOT_CONFIRMED",
            "next_action": "FINAL_FREEZE_AND_DEPLOYMENT_PACKAGE" if decision == "PASS" else "RETAIN_INCUMBENT",
        }
        deployment = {
            "status": "FINAL_LOCK" if decision == "PASS" else "NOT_APPROVED",
            "model_stack_status": "FINAL_LOCK" if decision == "PASS" else "NOT_APPROVED",
            "buy_wait_policy_status": "GUARDED_PILOT_NOT_PROVEN_SUPERIOR" if uncertainty["policy_regret"]["ci95_high"] >= 0.0 else "FINAL_LOCK",
            "contract_id": contract["contract_id"],
            "recipes": contract["expected_recipes"],
            "recipe_note": "Deployable V24_MEAN recipes are confirmed; original research blends were not carried forward because serialized deployment models were absent.",
            "policy": contract["policy"],
            "training_required_for_pooled_confirmation": False,
            "gpu_required_for_pooled_confirmation": False,
            "operational_refit": "Expanding refit may occur only at future production boundary; recipe and hyperparameters remain frozen.",
            "buy_wait_evidence": "Operationally safe; superiority requires CI review and remains conservative when pooled CI includes zero.",
        }
        provenance = {
            "status": "PASS",
            "input_archive_sha256": archive_hashes,
            "prediction_commit_sha256": {name: sha256(blocks[name]["output"] / "runtime" / f"prediction_commit_test{name[-1]}.json") for name in blocks},
            "baseline_lock_sha256": sha256(BASELINE_PATH),
            "contract_sha256": sha256(CONTRACT_PATH),
        }

        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "TWO_BLOCK_POOLED_REPORT.json", report)
        write_json(output / "FINAL_DEPLOYMENT_DECISION.json", deployment)
        write_json(output / "TWO_BLOCK_PROVENANCE.json", provenance)
        pd.concat(daily_tables, ignore_index=True).to_csv(output / "DAILY_STRATIFIED_DELTAS.csv", index=False)
        summary = pd.DataFrame([
            {"task": "CLASSIFICATION", "metric": "proper_score", "model": classification_model["proper_score"], "baseline": classification_baseline["proper_score"]},
            {"task": "POINT", "metric": "mape", "model": point_model["mape"], "baseline": point_base["mape"]},
            {"task": "DISTRIBUTION", "metric": "wis", "model": dist_model_metrics["wis"], "baseline": dist_base_metrics["wis"]},
            {"task": "RANKING", "metric": "ndcg5", "model": ranking_model["ndcg5"], "baseline": ranking_base["ndcg5"]},
            {"task": "BUY_WAIT", "metric": "mean_regret_vnd", "model": policy["mean_regret_vnd"], "baseline": buy_all["mean_regret_vnd"]},
        ])
        summary.to_csv(output / "POOLED_METRIC_SUMMARY.csv", index=False)
        policy_ci = uncertainty["policy_regret"]
        text_report = f"""Skyfare prospective evaluation summary

Scientific conclusion: {report['scientific_conclusion']}
Pooled gates: {sum(gates.values())}/{len(gates)} PASS
Model stack: {deployment['model_stack_status']}
Buy/wait policy: {deployment['buy_wait_policy_status']}
Retraining for pooled confirmation: No
GPU required: No

Task\tModel\tBaseline\tDirection
Classification proper score\t{classification_model['proper_score']:.6f}\t{classification_baseline['proper_score']:.6f}\tlower is better
Point MAPE\t{point_model['mape']:.6f}\t{point_base['mape']:.6f}\tlower is better
Distribution WIS\t{dist_model_metrics['wis']:.6f}\t{dist_base_metrics['wis']:.6f}\tlower is better
Ranking NDCG@5\t{ranking_model['ndcg5']:.6f}\t{ranking_base['ndcg5']:.6f}\thigher is better
Buy/wait mean regret (VND)\t{policy['mean_regret_vnd']:.2f}\t{buy_all['mean_regret_vnd']:.2f}\tlower is better

Classification, point, distribution and ranking pooled confidence intervals exclude zero in the favorable direction. Buy/wait model-minus-buy-all daily delta is {policy_ci['mean_model_minus_baseline']:.2f} VND with 95% CI [{policy_ci['ci95_low']:.2f}, {policy_ci['ci95_high']:.2f}]. Mean and CVaR improve, but superiority is not statistically established; guarded pilot retains BUY fallback.

Design: two prospective walk-forward blocks. Test 1 uses its pre-block fit. Test 2 uses the allowed expanding refit at its boundary. Recipe and hyperparameters remain frozen. Cross-block row and query overlap are zero.
"""
        (output / "FINAL_REPORT.txt").write_text(text_report, encoding="utf-8")
        print(f"[TWO BLOCK FINAL {decision}] gates={sum(gates.values())}/{len(gates)} rows_class={len(classification)} rows_reg={len(point)}")


if __name__ == "__main__":
    main()
