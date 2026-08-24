#!/usr/bin/env python3
"""Commit target-free Test 2 predictions before label evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyfare.evaluation.prospective_test_two.contract import FINALISTS, QUANTILES, registry
from skyfare.evaluation.metrics import rearrange
from skyfare.evaluation.prospective_test_two.runtime import (
    OUTPUT_ROOT,
    artifact_complete,
    job_root,
    preflight_sha256,
    sha256,
    write_json_atomic,
    write_parquet_atomic,
)


def _load_task(task: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for job in registry():
        if job["task"] != task:
            continue
        identifier = str(job["job_id"])
        if not artifact_complete(identifier):
            raise RuntimeError(f"cannot freeze incomplete job {identifier}")
        frames.append(pd.read_parquet(job_root(identifier) / "predictions.parquet"))
    if len(frames) != 6:
        raise RuntimeError(f"{task}: expected six finalist-seed predictions; found={len(frames)}")
    return frames


def _mean_aligned(frames: list[pd.DataFrame], value_columns: list[str]) -> pd.DataFrame:
    base = frames[0].sort_values("row_key", kind="stable").reset_index(drop=True)
    keys = base["row_key"].to_numpy(dtype=np.uint64)
    values: list[np.ndarray] = []
    for frame in frames:
        current = frame.sort_values("row_key", kind="stable").reset_index(drop=True)
        if not np.array_equal(keys, current["row_key"].to_numpy(dtype=np.uint64)):
            raise RuntimeError("prediction row identity differs across frozen jobs")
        values.append(current[value_columns].to_numpy(dtype=float))
    result = base.drop(columns=["candidate_id", "seed", "phase", *value_columns], errors="ignore")
    mean = np.mean(np.stack(values, axis=0), axis=0)
    for index, column in enumerate(value_columns):
        result[column] = mean[:, index]
    return result


def main() -> None:
    commit_root = OUTPUT_ROOT / "frozen_predictions"
    outputs = {}

    classification = _mean_aligned(_load_task("CLASSIFICATION"), ["probability"])
    classification = classification.drop(columns=["raw_score"], errors="ignore")
    classification["probability"] = np.clip(classification["probability"], 1e-7, 1.0 - 1e-7)
    outputs["CLASSIFICATION"] = commit_root / "classification_test2_predictions.parquet"
    write_parquet_atomic(outputs["CLASSIFICATION"], classification)

    point = _mean_aligned(_load_task("POINT"), ["predicted_price_vnd"])
    point["predicted_price_vnd"] = np.maximum(point["predicted_price_vnd"], 1.0)
    point["predicted_relative_log"] = np.log(
        point["predicted_price_vnd"].to_numpy(dtype=float)
        / np.maximum(point["prior_anchor_vnd"].to_numpy(dtype=float), 1.0)
    )
    outputs["POINT"] = commit_root / "point_test2_predictions.parquet"
    write_parquet_atomic(outputs["POINT"], point)

    quantile_columns = [f"q{int(round(level * 100)):02d}_relative_log" for level in QUANTILES]
    distribution = _mean_aligned(_load_task("DISTRIBUTION"), quantile_columns)
    ordered = rearrange(distribution[quantile_columns].to_numpy(dtype=float))
    anchor = np.maximum(distribution["prior_anchor_vnd"].to_numpy(dtype=float), 1.0)
    for index, level in enumerate(QUANTILES):
        token = f"q{int(round(level * 100)):02d}"
        distribution[f"{token}_relative_log"] = ordered[:, index]
        distribution[f"{token}_price_vnd"] = anchor * np.exp(ordered[:, index])
    outputs["DISTRIBUTION"] = commit_root / "distribution_test2_predictions.parquet"
    write_parquet_atomic(outputs["DISTRIBUTION"], distribution)

    ranking = _mean_aligned(_load_task("RANKING"), ["score"])
    outputs["RANKING"] = commit_root / "ranking_test2_predictions.parquet"
    write_parquet_atomic(outputs["RANKING"], ranking)

    payload = {
        "status": "COMMITTED_BEFORE_LABEL_EVALUATION",
        "recipes": {task: "V24_MEAN" for task in FINALISTS},
        "finalists": {task: list(values) for task, values in FINALISTS.items()},
        "jobs": 24,
        "preflight_sha256": preflight_sha256(),
        "test_labels_read": False,
        "files": {
            task: {
                "path": str(path.relative_to(OUTPUT_ROOT)),
                "sha256": sha256(path),
                "rows": int(len(pd.read_parquet(path, columns=["row_key"]))),
            }
            for task, path in outputs.items()
        },
    }
    write_json_atomic(OUTPUT_ROOT / "runtime" / "prediction_commit_test2.json", payload)
    print("[TEST2 PREDICTION COMMIT PASS] 4 tasks; labels unread", flush=True)


if __name__ == "__main__":
    main()
