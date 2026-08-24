#!/usr/bin/env python3
"""Load frozen production artifacts and score prepared feature frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.production.calibration import apply_probability, apply_quantiles
from skyfare.production.contract import BUY_WAIT_THRESHOLD, QUANTILES
from skyfare.production.runtime import OUTPUT_ROOT, ROOT, load_encoder, write_parquet_atomic
from skyfare.models import candidate_models as original_models
from skyfare.evaluation.metrics import rearrange
from skyfare.production.models import _build_sequence_class, _build_sequence_point
from skyfare.production.sequence import load_normalizer, sequence_inputs


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-frame", type=Path, required=True)
    parser.add_argument("--regression-frame", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_ROOT / "deployment/DEPLOYMENT_MANIFEST_R1.json",
    )
    return parser.parse_args()


def _model_path(entry: dict[str, object]) -> Path:
    return ROOT / str(entry["model_root"])


def _predict_member(entry: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    root = _model_path(entry)
    family = str(entry["family"])
    encoder = load_encoder(root / "encoder.json")
    if family == "FT_CLASS":
        tf = original_models._configure_tensorflow(int(entry["seed"]))
        model = original_models._ft_model(tf, encoder, str(entry["configuration"]))
        model.load_weights(root / "model.weights.h5")
        return model.predict(list(encoder.neural_inputs(frame)), batch_size=2048, verbose=0).reshape(-1)
    if family in {"SEQ_CLASS", "SEQ_POINT"}:
        tf = original_models._configure_tensorflow(int(entry["seed"]))
        task = "CLASSIFICATION" if family == "SEQ_CLASS" else "POINT"
        inputs, _ = sequence_inputs(frame, task, encoder, load_normalizer(root / "sequence_normalizer.json"))
        configuration = str(entry["configuration"])
        if family == "SEQ_CLASS":
            model = _build_sequence_class(tf, encoder, configuration, inputs[0].shape[1:], inputs[2].shape[1])
            model.load_weights(root / "model.weights.h5")
            return model.predict(inputs, batch_size=2048, verbose=0)["drop5"].reshape(-1)
        model = _build_sequence_point(tf, encoder, configuration, inputs[0].shape[1:], inputs[2].shape[1])
        model.load_weights(root / "model.weights.h5")
        return model.predict(inputs, batch_size=2048, verbose=0).reshape(-1)
    if family == "CAT_MULTIQUANTILE":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(root / "model.cbm")
        return rearrange(model.predict(encoder.native_frame(frame)))
    if family == "XGB_QUANTILE":
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(root / "model.ubj")
        return rearrange(model.predict(encoder.tree_matrix(frame)))
    if family in {"LGBM_LAMBDARANK", "LGBM_XENDCG"}:
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(root / "model.txt"))
        return np.asarray(model.predict(encoder.tree_matrix(frame)), dtype=float)
    raise RuntimeError(f"unsupported deployment family: {family}")


def _members(manifest: dict[str, object], task: str) -> list[dict[str, object]]:
    identifiers = set(manifest["ensembles"][task]["members"])
    result = [entry for entry in manifest["jobs"] if entry["job_id"] in identifiers]
    if len(result) != 6:
        raise RuntimeError(f"{task}: expected six frozen ensemble members")
    return result


def main() -> None:
    args = _args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("deployment manifest is not PASS")
    classification = pd.read_parquet(args.classification_frame)
    regression = pd.read_parquet(args.regression_frame)
    for frame in (classification, regression):
        if "row_key" not in frame:
            raise RuntimeError("prepared inference frame requires row_key")
    class_raw = np.mean([_predict_member(entry, classification) for entry in _members(manifest, "CLASSIFICATION")], axis=0)
    class_state = json.loads(
        (ROOT / manifest["ensembles"]["CLASSIFICATION"]["calibration"]["path"]).read_text(encoding="utf-8")
    )
    probability = apply_probability(class_raw, class_state)
    class_output = pd.DataFrame(
        {
            "row_key": classification["row_key"].to_numpy(),
            "drop_5pct_probability": probability,
            "operational_action": "BUY",
            "guarded_pilot_action": np.where(probability >= BUY_WAIT_THRESHOLD, "WAIT", "BUY"),
        }
    )
    point_raw = np.mean([_predict_member(entry, regression) for entry in _members(manifest, "POINT")], axis=0)
    prior_anchor = regression["prior_anchor_vnd"].to_numpy(dtype=float)
    point_output = pd.DataFrame(
        {
            "row_key": regression["row_key"].to_numpy(),
            "predicted_price_vnd": prior_anchor * np.exp(point_raw),
        }
    )
    distribution_raw = np.mean(
        [_predict_member(entry, regression) for entry in _members(manifest, "DISTRIBUTION")], axis=0
    )
    distribution_state = json.loads(
        (ROOT / manifest["ensembles"]["DISTRIBUTION"]["calibration"]["path"]).read_text(encoding="utf-8")
    )
    calibrated = apply_quantiles(regression, distribution_raw, distribution_state)
    distribution_payload: dict[str, object] = {"row_key": regression["row_key"].to_numpy()}
    for index, level in enumerate(QUANTILES):
        distribution_payload[f"predicted_price_q{int(round(level * 100)):02d}_vnd"] = prior_anchor * np.exp(
            calibrated[:, index]
        )
    distribution_output = pd.DataFrame(distribution_payload)
    ranking = np.mean([_predict_member(entry, regression) for entry in _members(manifest, "RANKING")], axis=0)
    ranking_output = pd.DataFrame({"row_key": regression["row_key"].to_numpy(), "ranking_score": ranking})
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(args.output_root / "classification_predictions.parquet", class_output)
    write_parquet_atomic(args.output_root / "point_predictions.parquet", point_output)
    write_parquet_atomic(args.output_root / "distribution_predictions.parquet", distribution_output)
    write_parquet_atomic(args.output_root / "ranking_predictions.parquet", ranking_output)
    print(
        f"[PRODUCTION INFERENCE PASS] classification={len(class_output):,} regression={len(regression):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
