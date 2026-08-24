#!/usr/bin/env python3
"""Fit frozen ensemble calibration and emit deployment manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.production.calibration import (
    apply_probability,
    apply_quantiles,
    fit_probability,
    fit_quantiles,
)
from skyfare.production.contract import FINALISTS, TASKS, manifest as contract_manifest, registry
from skyfare.production.runtime import (
    OUTPUT_ROOT,
    ROOT,
    artifact_complete,
    current_code_sha256,
    job_root,
    sha256,
    write_json_atomic,
)


def _load_task(task: str) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, object]]]:
    jobs = [job for job in registry() if job["task"] == task]
    frames: list[pd.DataFrame] = []
    entries: list[dict[str, object]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        if not artifact_complete(job_id):
            raise RuntimeError(f"incomplete production job: {job_id}")
        root = job_root(job_id)
        frame = pd.read_parquet(root / "head_predictions.parquet")
        if frames and not np.array_equal(frame["row_key"].to_numpy(), frames[0]["row_key"].to_numpy()):
            raise RuntimeError(f"{task}: temporal head row identity differs across models")
        frames.append(frame)
        done = json.loads((root / "done.json").read_text(encoding="utf-8"))
        entries.append(
            {
                "job_id": job_id,
                "candidate_id": job["candidate_id"],
                "family": job["family"],
                "configuration": job["configuration"],
                "seed": job["seed"],
                "resource": job["resource"],
                "model_root": str(root.relative_to(ROOT)),
                "artifacts": done["artifacts"],
                "reload_parity": done["reload_parity"],
                "training": done["training"],
            }
        )
    column = "raw_score" if task == "CLASSIFICATION" else "prediction"
    if task == "DISTRIBUTION":
        quantile_columns = [column for column in frames[0] if column.startswith("q")]
        prediction = np.mean([frame[quantile_columns].to_numpy(dtype=float) for frame in frames], axis=0)
    else:
        prediction = np.mean([frame[column].to_numpy(dtype=float) for frame in frames], axis=0)
    return frames[0], prediction, entries


def main() -> None:
    output = OUTPUT_ROOT / "deployment"
    output.mkdir(parents=True, exist_ok=True)
    ensembles: dict[str, object] = {}
    job_entries: list[dict[str, object]] = []
    for task in TASKS:
        head, raw, entries = _load_task(task)
        state: dict[str, object] | None = None
        if task == "CLASSIFICATION":
            state = fit_probability(head["target"].to_numpy(dtype=np.int8), raw)
            before = apply_probability(raw, state)
            calibration_path = output / "classification_calibration.json"
            write_json_atomic(calibration_path, state)
            loaded = json.loads(calibration_path.read_text(encoding="utf-8"))
            after = apply_probability(raw, loaded)
        elif task == "DISTRIBUTION":
            state = fit_quantiles(head, raw)
            before = apply_quantiles(head, raw, state)
            calibration_path = output / "distribution_calibration.json"
            write_json_atomic(calibration_path, state)
            loaded = json.loads(calibration_path.read_text(encoding="utf-8"))
            after = apply_quantiles(head, raw, loaded)
        else:
            calibration_path = None
            before = raw
            after = raw.copy()
        maximum = float(np.max(np.abs(np.asarray(before) - np.asarray(after))))
        if maximum > 1e-12:
            raise RuntimeError(f"{task}: calibration reload drift {maximum}")
        ensembles[task] = {
            "recipe": "V24_MEAN",
            "candidate_ids": list(FINALISTS[task]),
            "members": [entry["job_id"] for entry in entries],
            "member_count": len(entries),
            "temporal_head_rows": len(head),
            "calibration": (
                {
                    "path": str(calibration_path.relative_to(ROOT)),
                    "sha256": sha256(calibration_path),
                    "recipe": state["recipe"],
                    "reload_max_abs_difference": maximum,
                }
                if calibration_path is not None
                else None
            ),
        }
        job_entries.extend(entries)
    deployment = {
        "status": "PASS",
        "contract": contract_manifest(),
        "code_sha256": current_code_sha256(),
        "jobs": job_entries,
        "ensembles": ensembles,
        "policy": {
            "production_action": "BUY",
            "pilot_status": "GUARDED_PILOT_NOT_PROVEN_SUPERIOR",
            "pilot_automatic": False,
        },
        "inference_entrypoint": "python -m skyfare.production.inference",
    }
    manifest_path = output / "DEPLOYMENT_MANIFEST_R1.json"
    write_json_atomic(manifest_path, deployment)
    summary = {
        "status": "PASS",
        "production_jobs": len(job_entries),
        "tasks": list(TASKS),
        "recipes": {task: ensembles[task]["recipe"] for task in TASKS},
        "training_cutoff_inclusive": deployment["contract"]["training_cutoff_inclusive"],
        "deployment_manifest_sha256": sha256(manifest_path),
        "buy_wait": deployment["policy"],
    }
    write_json_atomic(output / "PRODUCTION_REFIT_SUMMARY_R1.json", summary)
    print("[PRODUCTION FINALIZE PASS] jobs=24 tasks=4 recipes=V24_MEAN", flush=True)


if __name__ == "__main__":
    main()
