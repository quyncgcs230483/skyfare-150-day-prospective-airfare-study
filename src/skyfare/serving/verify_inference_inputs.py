#!/usr/bin/env python3
"""Verify prepared frames against every frozen encoder without fitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.models import sequence_runtime
from skyfare.production.runtime import load_encoder
from skyfare.production.sequence import load_normalizer, sequence_inputs

MODEL_FILE = {
    "FT_CLASS": "model.weights.h5",
    "SEQ_CLASS": "model.weights.h5",
    "SEQ_POINT": "model.weights.h5",
    "CAT_MULTIQUANTILE": "model.cbm",
    "XGB_QUANTILE": "model.ubj",
    "LGBM_LAMBDARANK": "model.txt",
    "LGBM_XENDCG": "model.txt",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _finite(values: object, label: str) -> None:
    for index, value in enumerate(values if isinstance(values, tuple) else (values,)):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError(f"{label}[{index}] contains non-finite values")


def main() -> None:
    args = _args()
    feature_root = args.feature_root.resolve()
    artifact_root = args.artifact_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or len(manifest.get("jobs", [])) != 24:
        raise RuntimeError("Frozen deployment manifest is incomplete")
    classification = pd.read_parquet(
        feature_root / "classification_features.parquet"
    ).head(512)
    regression = pd.read_parquet(feature_root / "regression_features.parquet").head(
        512
    )
    sequence_runtime.OFFERS_FRAME = feature_root / "standard_offers.parquet"
    sequence_runtime.OUTPUT_ROOT = feature_root / "compatibility_sequence_cache"
    task_by_job = {
        str(job_id): task
        for task, ensemble in manifest["ensembles"].items()
        for job_id in ensemble["members"]
    }

    verified = 0
    for entry in manifest["jobs"]:
        family = str(entry["family"])
        task = task_by_job[str(entry["job_id"])]
        frame = classification if task == "CLASSIFICATION" else regression
        root = artifact_root / str(entry["model_root"])
        model = root / MODEL_FILE[family]
        encoder_path = root / "encoder.json"
        if not model.is_file() or not encoder_path.is_file():
            raise FileNotFoundError(f"Frozen model artifact missing: {root}")
        encoder = load_encoder(encoder_path)
        if family == "FT_CLASS":
            _finite(encoder.neural_inputs(frame), str(entry["job_id"]))
        elif family in {"SEQ_CLASS", "SEQ_POINT"}:
            source_task = "CLASSIFICATION" if family == "SEQ_CLASS" else "POINT"
            inputs, _ = sequence_inputs(
                frame,
                source_task,
                encoder,
                load_normalizer(root / "sequence_normalizer.json"),
            )
            _finite(tuple(inputs), str(entry["job_id"]))
        elif family == "CAT_MULTIQUANTILE":
            native = encoder.native_frame(frame)
            if native.empty:
                raise RuntimeError(f"{entry['job_id']}: empty CatBoost input")
        else:
            _finite(encoder.tree_matrix(frame), str(entry["job_id"]))
        verified += 1
    for task in ("CLASSIFICATION", "DISTRIBUTION"):
        calibration = artifact_root / manifest["ensembles"][task]["calibration"]["path"]
        if not calibration.is_file():
            raise FileNotFoundError(calibration)
    print(
        json.dumps(
            {
                "status": "PASS",
                "jobs": verified,
                "classification_rows_checked": len(classification),
                "regression_rows_checked": len(regression),
                "post_cutoff_labels_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
