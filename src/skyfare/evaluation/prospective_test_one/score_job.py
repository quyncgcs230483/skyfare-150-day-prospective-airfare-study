#!/usr/bin/env python3
"""Refit one frozen V24 finalist and score target-free Test 1 features."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.calibration.methods import calibrate_probability, calibrate_quantiles
from skyfare.evaluation.prospective_test_one.contract import QUANTILES, registry
from skyfare.evaluation.prospective_test_one.runtime import (
    artifact_complete,
    current_code_sha256,
    job_root,
    preflight_sha256,
    sha256,
    training_frames,
    write_json_atomic,
    write_parquet_atomic,
)
from skyfare.models.candidate_models import fit_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_job(raw: dict[str, object]) -> dict[str, object]:
    expected = {str(job["job_id"]): job for job in registry()}
    identifier = str(raw.get("job_id", ""))
    if identifier not in expected or raw != expected[identifier]:
        raise RuntimeError(f"job is not in immutable Test 1 registry: {identifier}")
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


def prediction_artifact(
    job: dict[str, object],
    head: pd.DataFrame,
    valid: pd.DataFrame,
    head_prediction: np.ndarray,
    valid_prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    task = str(job["task"])
    common = {
        "candidate_id": str(job["candidate_id"]),
        "seed": int(job["seed"]),
        "phase": "TEST_1_SCORE",
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
                "raw_score": np.asarray(valid_prediction, dtype=float).reshape(-1),
                "probability": probability,
                "price_vnd": valid["price_vnd"].to_numpy(dtype=float),
                "route": valid["route"].astype(str).to_numpy(),
                "airline": valid["airline"].astype(str).to_numpy(),
                "regime": valid["regime"].astype(str).to_numpy(),
                "target_dud": valid["target_dud"].to_numpy(dtype=float),
                **common,
            }
        )
        return result, {"calibration": calibration}

    base: dict[str, object] = {
        "row_key": valid["row_key"].to_numpy(dtype=np.uint64),
        "query_id": valid["query_id"].astype(str).to_numpy(),
        "target_offer_id": valid["target_offer_id"].to_numpy(dtype=np.uint64),
        "feature_time": valid["feature_time"].to_numpy(),
        "label_time": valid["label_time"].to_numpy(),
        "query_dud": valid["query_dud"].to_numpy(dtype=float),
        "prior_anchor_vnd": valid["prior_anchor_vnd"].to_numpy(dtype=float),
        "route_airline": valid["route_airline"].astype(str).to_numpy(),
        "regime": valid["regime"].astype(str).to_numpy(),
        **common,
    }
    if task == "POINT":
        relative = np.asarray(valid_prediction, dtype=float).reshape(-1)
        result = pd.DataFrame(
            {
                **base,
                "predicted_relative_log": relative,
                "predicted_price_vnd": np.maximum(valid["prior_anchor_vnd"].to_numpy(dtype=float), 1.0)
                * np.exp(relative),
            }
        )
        return result, {}
    if task == "DISTRIBUTION":
        calibrated, calibration = calibrate_quantiles(head, head_prediction, valid, valid_prediction)
        anchor = np.maximum(valid["prior_anchor_vnd"].to_numpy(dtype=float), 1.0)
        payload = dict(base)
        for index, level in enumerate(QUANTILES):
            token = f"q{int(round(level * 100)):02d}"
            payload[f"{token}_relative_log"] = calibrated[:, index]
            payload[f"{token}_price_vnd"] = anchor * np.exp(calibrated[:, index])
        return pd.DataFrame(payload), {"calibration": calibration}
    if task == "RANKING":
        return pd.DataFrame({**base, "score": np.asarray(valid_prediction, dtype=float).reshape(-1)}), {}
    raise KeyError(task)


def main() -> None:
    cli = parse_args()
    job = validate_job(json.loads(cli.job_json))
    root = job_root(str(job["job_id"]))
    if artifact_complete(str(job["job_id"])) and not cli.force:
        print(f"[TEST1 JOB SKIP] {job['job_id']}", flush=True)
        return
    if cli.force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heartbeat = heartbeat_writer(root, job, started)
    try:
        fit, head, valid = training_frames(str(job["task"]), str(job["window"]))
        print(
            f"[TEST1 JOB START] {job['job_id']} fit={len(fit):,} head={len(head):,} "
            f"valid={len(valid):,} resource={job['resource']}",
            flush=True,
        )
        head_prediction, valid_prediction, model_metadata = fit_predict(
            job, fit, head, valid, smoke=False, artifact_dir=root, heartbeat=heartbeat
        )
        artifact, report = prediction_artifact(job, head, valid, head_prediction, valid_prediction)
        forbidden = {
            "target",
            "DROP_5PCT",
            "target_price_vnd",
            "target_relative_log",
            "target_anchor_relative_log",
            "target_session_price_vnd",
        }
        leaked = sorted(forbidden.intersection(artifact.columns))
        if leaked:
            raise RuntimeError(f"target columns entered scorer output: {leaked}")
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
            "preflight_sha256": preflight_sha256(),
            "test_labels_read": False,
            "model": model_metadata,
            **report,
        }
        write_json_atomic(root / "done.json", done)
        (root / "failure.json").unlink(missing_ok=True)
        print(
            f"[TEST1 JOB PASS] {job['job_id']} rows={len(artifact):,} "
            f"elapsed={done['elapsed_seconds']:.1f}s",
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
        print(f"[TEST1 JOB FAIL] {job['job_id']} {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
