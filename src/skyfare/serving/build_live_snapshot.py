#!/usr/bin/env python3
"""Publish an atomic, verified prediction snapshot for the Reflex application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from skyfare.core.paths import DataLayout

TRAINING_CUTOFF = "2026-08-19"
CANONICAL_DUDS = frozenset({1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60})
REQUIRED_COLUMNS = {
    "session_date",
    "session_label",
    "route",
    "flight_date",
    "airline",
    "departure_HHMM",
    "schedule_slot_id",
    "predicted_price_vnd",
    "baseline_price_vnd",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".json"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _args() -> argparse.Namespace:
    layout = DataLayout.resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-provenance", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=layout.artifacts / "serving"
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    cutoff = pd.Timestamp(args.through_date).normalize()
    model = _read_json(args.model_manifest)
    provenance = _read_json(args.prediction_provenance)
    if model.get("status") != "PASS":
        raise RuntimeError("Production model manifest is not PASS")
    model_cutoff = str(
        model.get("training_cutoff_inclusive", model.get("training_cutoff", ""))
    )
    if model_cutoff != TRAINING_CUTOFF:
        raise RuntimeError(f"Frozen training cutoff mismatch: {model_cutoff}")
    if provenance.get("status") != "PASS":
        raise RuntimeError("Prediction provenance is not PASS")
    if provenance.get("post_cutoff_labels_used") is not False:
        raise RuntimeError("Post-cutoff labels must not enter serving features")
    if str(provenance.get("observation_cutoff")) != cutoff.strftime("%Y-%m-%d"):
        raise RuntimeError("Prediction provenance cutoff mismatch")

    frame = pd.read_parquet(args.predictions)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction columns missing: {missing}")
    for column in ("session_date", "flight_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if frame.empty or frame[["session_date", "flight_date"]].isna().any().any():
        raise RuntimeError("Prediction frame is empty or has invalid dates")
    if frame["session_date"].dt.normalize().max() != cutoff:
        raise RuntimeError("Prediction frame does not reach declared booking date")
    duds = (frame["flight_date"] - frame["session_date"]).dt.days
    if not duds.between(1, 60).all():
        raise RuntimeError("Serving DUD must remain within 1-60")
    if "dud_support_mode" in frame:
        on_grid = frame["dud_support_mode"].astype(str).eq("ON_GRID")
        if set(duds[on_grid].unique()).difference(CANONICAL_DUDS):
            raise RuntimeError("ON_GRID row uses non-canonical DUD")
    if not pd.to_numeric(frame["predicted_price_vnd"], errors="coerce").gt(0).all():
        raise RuntimeError("Predicted prices must be positive")
    if not pd.to_numeric(frame["baseline_price_vnd"], errors="coerce").gt(0).all():
        raise RuntimeError("Baseline prices must be positive")

    prediction_hash = _sha256(args.predictions)
    snapshot_id = f"{cutoff.strftime('%Y%m%d')}-{prediction_hash[:12]}"
    relative_root = Path("snapshots") / snapshot_id
    snapshot_root = args.output_root / relative_root
    snapshot_root.mkdir(parents=True, exist_ok=True)
    output = snapshot_root / "predictions.parquet"
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)
    manifest = {
        "status": "PASS",
        "snapshot_id": snapshot_id,
        "model_version": str(model.get("contract_id", "FINAL_PRODUCTION_REFIT_R1")),
        "training_cutoff_inclusive": TRAINING_CUTOFF,
        "observation_cutoff": cutoff.strftime("%Y-%m-%d"),
        "post_cutoff_labels_used": False,
        "rows": len(frame),
        "routes": int(frame["route"].nunique()),
        "airlines": int(frame["airline"].nunique()),
        "route_airline_pairs": int(frame[["route", "airline"]].drop_duplicates().shape[0]),
        "predictions_sha256": _sha256(output),
        "source_predictions_sha256": prediction_hash,
        "prediction_provenance_sha256": _sha256(args.prediction_provenance),
        "model_manifest_sha256": _sha256(args.model_manifest),
    }
    manifest_path = snapshot_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    current = {
        "snapshot_id": snapshot_id,
        "manifest": str(relative_root / "manifest.json"),
        "predictions": str(relative_root / "predictions.parquet"),
        "manifest_sha256": _sha256(manifest_path),
    }
    _atomic_json(args.output_root / "current.json", current)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
