#!/usr/bin/env python3
"""Run one V24 fitting job with atomic prediction and completion artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.calibration.methods import calibrate_probability, calibrate_quantiles
from skyfare.evaluation.metrics import (
    classification_metrics,
    distribution_metrics,
    point_metrics,
    ranking_metrics,
)
from skyfare.models.candidate_models import fit_predict
from skyfare.models.selection_contract import QUANTILES, candidate_map
from skyfare.models.temporal_runtime import (
    apply_window,
    artifact_complete,
    current_code_sha256,
    deterministic_subset,
    inner_split,
    job_root,
    load_classification_frame,
    load_regression_frame,
    outer_split,
    sha256,
    write_json_atomic,
    write_parquet_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_job(raw: dict[str, object]) -> dict[str, object]:
    required = {
        "candidate_id", "task", "family", "resource", "configuration", "window",
        "fold", "seed", "phase", "job_id",
    }
    if set(raw) != required:
        raise RuntimeError(f"job fields changed missing={sorted(required - set(raw))} extra={sorted(set(raw) - required)}")
    candidate = candidate_map().get(str(raw["candidate_id"]))
    if candidate is None:
        raise RuntimeError(f"unknown candidate {raw['candidate_id']}")
    for key, value in asdict(candidate).items():
        if raw[key] != value:
            raise RuntimeError(f"job candidate mismatch {key}: {raw[key]!r} != {value!r}")
    expected_id = f"{candidate.candidate_id}__{raw['fold']}__S{int(raw['seed'])}"
    if raw["job_id"] != expected_id:
        raise RuntimeError("job identity mismatch")
    if raw["phase"] not in {"SMOKE", "SCREEN", "HISTORY", "LATE"}:
        raise RuntimeError(f"invalid phase {raw['phase']}")
    return raw


def heartbeat_writer(root: Path, job: dict[str, object], started: float):
    def write(status: str, epoch: int, metrics: dict[str, float]) -> None:
        write_json_atomic(
            root / "heartbeat.json",
            {
                "status": status,
                "job_id": job["job_id"],
                "epoch": int(epoch),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "metrics": {
                    key: float(value)
                    for key, value in metrics.items()
                    if np.isscalar(value) and np.isfinite(value)
                },
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            },
        )

    return write


def prepare_frames(job: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = str(job["task"])
    source_task = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
    frame = load_classification_frame() if source_task == "CLASSIFICATION" else load_regression_frame()
    train, valid = outer_split(frame, source_task, str(job["fold"]))
    train = apply_window(train, source_task, str(job["fold"]), str(job["window"]))
    fit, head = inner_split(train, source_task)
    if job["phase"] == "SMOKE":
        fit = deterministic_subset(fit, source_task, 4096)
        head = deterministic_subset(head, source_task, 2048)
        valid = deterministic_subset(valid, source_task, 2048)
    return fit, head, valid


def prediction_artifact(
    job: dict[str, object],
    head: pd.DataFrame,
    valid: pd.DataFrame,
    head_prediction: np.ndarray,
    valid_prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    task = str(job["task"])
    base = {
        "candidate_id": str(job["candidate_id"]),
        "fold": str(job["fold"]),
        "seed": int(job["seed"]),
        "phase": str(job["phase"]),
    }
    if task == "CLASSIFICATION":
        probability, calibration = calibrate_probability(
            head["DROP_5PCT"].to_numpy(dtype=np.int8), head_prediction, valid_prediction
        )
        result = pd.DataFrame(
            {
                "row_key": valid["row_key"].to_numpy(dtype=np.uint64),
                "offer_id": valid["offer_id"].to_numpy(dtype=np.uint64),
                "target_offer_id": valid["target_offer_id"].to_numpy(dtype=np.uint64),
                "feature_time": valid["feature_time"].to_numpy(),
                "label_time": valid["label_time"].to_numpy(),
                "target": valid["DROP_5PCT"].to_numpy(dtype=np.int8),
                "raw_score": np.asarray(valid_prediction, dtype=float).reshape(-1),
                "probability": probability,
                "price_vnd": valid["price_vnd"].to_numpy(dtype=float),
                "target_price_vnd": valid["target_price_vnd"].to_numpy(dtype=float),
                "route": valid["route"].astype(str).to_numpy(),
                "airline": valid["airline"].astype(str).to_numpy(),
                "regime": valid["regime"].astype(str).to_numpy(),
                "target_dud": valid["target_dud"].to_numpy(dtype=float),
                **base,
            }
        )
        return result, {"metrics": classification_metrics(result["target"], result["probability"]), "calibration": calibration}

    if task == "POINT":
        predicted_relative = np.asarray(valid_prediction, dtype=float).reshape(-1)
        predicted_price = np.maximum(valid["prior_anchor_vnd"].to_numpy(dtype=float), 1.0) * np.exp(predicted_relative)
        result = pd.DataFrame(
            {
                "row_key": valid["row_key"].to_numpy(dtype=np.uint64),
                "query_id": valid["query_id"].astype(str).to_numpy(),
                "target_offer_id": valid["target_offer_id"].to_numpy(dtype=np.uint64),
                "feature_time": valid["feature_time"].to_numpy(),
                "label_time": valid["label_time"].to_numpy(),
                "query_dud": valid["query_dud"].to_numpy(dtype=float),
                "target_relative_log": valid["target_anchor_relative_log"].to_numpy(dtype=float),
                "predicted_relative_log": predicted_relative,
                "target_price_vnd": valid["target_session_price_vnd"].to_numpy(dtype=float),
                "predicted_price_vnd": predicted_price,
                "route_airline": valid["route_airline"].astype(str).to_numpy(),
                "regime": valid["regime"].astype(str).to_numpy(),
                **base,
            }
        )
        return result, {"metrics": point_metrics(result["target_price_vnd"], result["predicted_price_vnd"])}

    if task == "DISTRIBUTION":
        calibrated, calibration = calibrate_quantiles(head, head_prediction, valid, valid_prediction)
        payload: dict[str, object] = {
            "row_key": valid["row_key"].to_numpy(dtype=np.uint64),
            "query_id": valid["query_id"].astype(str).to_numpy(),
            "target_offer_id": valid["target_offer_id"].to_numpy(dtype=np.uint64),
            "feature_time": valid["feature_time"].to_numpy(),
            "label_time": valid["label_time"].to_numpy(),
            "query_dud": valid["query_dud"].to_numpy(dtype=float),
            "target_relative_log": valid["target_anchor_relative_log"].to_numpy(dtype=float),
            "target_price_vnd": valid["target_session_price_vnd"].to_numpy(dtype=float),
            "prior_anchor_vnd": valid["prior_anchor_vnd"].to_numpy(dtype=float),
            "route_airline": valid["route_airline"].astype(str).to_numpy(),
            "regime": valid["regime"].astype(str).to_numpy(),
            **base,
        }
        anchor = np.maximum(valid["prior_anchor_vnd"].to_numpy(dtype=float), 1.0)
        for index, level in enumerate(QUANTILES):
            token = f"q{int(round(level * 100)):02d}"
            payload[f"{token}_relative_log"] = calibrated[:, index]
            payload[f"{token}_price_vnd"] = anchor * np.exp(calibrated[:, index])
        result = pd.DataFrame(payload)
        return result, {
            "metrics": distribution_metrics(result["target_relative_log"], calibrated),
            "calibration": calibration,
        }

    if task == "RANKING":
        score = np.asarray(valid_prediction, dtype=float).reshape(-1)
        result = pd.DataFrame(
            {
                "row_key": valid["row_key"].to_numpy(dtype=np.uint64),
                "query_id": valid["query_id"].astype(str).to_numpy(),
                "target_offer_id": valid["target_offer_id"].to_numpy(dtype=np.uint64),
                "feature_time": valid["feature_time"].to_numpy(),
                "label_time": valid["label_time"].to_numpy(),
                "query_dud": valid["query_dud"].to_numpy(dtype=float),
                "target_price_vnd": valid["target_session_price_vnd"].to_numpy(dtype=float),
                "score": score,
                "route_airline": valid["route_airline"].astype(str).to_numpy(),
                "regime": valid["regime"].astype(str).to_numpy(),
                **base,
            }
        )
        return result, {"metrics": ranking_metrics(valid, score)}
    raise KeyError(task)


def main() -> None:
    cli = parse_args()
    job = validate_job(json.loads(cli.job_json))
    root = job_root(str(job["job_id"]))
    if artifact_complete(str(job["job_id"])) and not cli.force:
        print(f"[V24 JOB SKIP] {job['job_id']}", flush=True)
        return
    if cli.force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heartbeat = heartbeat_writer(root, job, started)
    try:
        fit, head, valid = prepare_frames(job)
        print(
            f"[V24 JOB START] {job['job_id']} fit={len(fit):,} head={len(head):,} valid={len(valid):,} "
            f"resource={job['resource']} task={job['task']}",
            flush=True,
        )
        head_prediction, valid_prediction, model_metadata = fit_predict(
            job, fit, head, valid, smoke=job["phase"] == "SMOKE", artifact_dir=root, heartbeat=heartbeat
        )
        artifact, report = prediction_artifact(job, head, valid, head_prediction, valid_prediction)
        prediction = root / "predictions.parquet"
        write_parquet_atomic(prediction, artifact)
        done = {
            "status": "COMPLETE",
            "job": job,
            "rows": int(len(artifact)),
            "fit_rows": int(len(fit)),
            "head_rows": int(len(head)),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "prediction_sha256": sha256(prediction),
            "code_sha256": current_code_sha256(),
            "model": model_metadata,
            **report,
        }
        write_json_atomic(root / "done.json", done)
        (root / "failure.json").unlink(missing_ok=True)
        print(
            f"[V24 JOB PASS] {job['job_id']} rows={len(artifact):,} elapsed={done['elapsed_seconds']:.1f}s",
            flush=True,
        )
    except Exception as error:
        write_json_atomic(
            root / "failure.json",
            {
                "status": "FAILED",
                "job": job,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )
        print(f"[V24 JOB FAIL] {job['job_id']} {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
