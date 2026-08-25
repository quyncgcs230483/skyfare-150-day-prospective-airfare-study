#!/usr/bin/env python3
"""Prepare target-free post-freeze inference frames from observed offer history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skyfare.core.integrity import sha256, write_json_atomic
from skyfare.core.paths import DataLayout
from skyfare.models import classification_runtime, regression_runtime
from skyfare.models.classification_contract import (
    CATEGORICAL_FEATURES as CLASS_CATEGORICAL,
)
from skyfare.models.classification_contract import NUMERIC_FEATURES as CLASS_NUMERIC
from skyfare.models.regression_contract import (
    CATEGORICAL_FEATURES as REG_CATEGORICAL,
)
from skyfare.models.regression_contract import NUMERIC_FEATURES as REG_NUMERIC

CLASS_TARGET_COLUMNS = {
    "DROP_5PCT",
    "material_drop_next",
    "target_price_vnd",
    "price_change_vnd",
    "price_change_pct",
    "label_time",
}
REG_TARGET_COLUMNS = {
    "target_anchor_relative_log",
    "target_log_ratio",
    "target_session_price_vnd",
}


def _args() -> argparse.Namespace:
    layout = DataLayout.resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--through-date", default="2026-08-24")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=layout.artifacts / "serving" / "inference_build",
    )
    return parser.parse_args()


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _require_predictors(
    frame: pd.DataFrame,
    categorical: tuple[str, ...],
    numeric: tuple[str, ...],
    task: str,
) -> None:
    missing = sorted(set(categorical + numeric).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{task} inference predictors missing: {missing}")


def build(through_date: str, output_root: Path) -> dict[str, object]:
    cutoff = pd.Timestamp(through_date).normalize()
    cutoff_text = cutoff.strftime("%Y-%m-%d")
    output_root = output_root.resolve()
    offers_path = output_root / "standard_offers.parquet"

    regression_runtime.CUTOFF = cutoff_text
    regression_runtime.STANDARD_OFFERS_CACHE = offers_path
    regression_runtime.RECURRENT_SEQUENCE_SOURCE = offers_path
    regression_runtime.legacy.CUTOFF = cutoff_text
    regression_runtime.legacy.STANDARD_OFFERS_CACHE = offers_path
    regression_runtime.legacy.RECURRENT_SEQUENCE_SOURCE = offers_path
    regression_runtime.FRAME_CACHE = output_root / "regression_features.parquet"
    regression_runtime.LEDGER_CACHE = output_root / "regression_ledger.parquet"

    offers = regression_runtime.build_standard_offers()
    if offers.empty:
        raise RuntimeError("Post-freeze standard offer frame is empty")
    observed = pd.to_datetime(offers["session_date"], errors="coerce").dt.normalize()
    if observed.max() != cutoff:
        raise RuntimeError(f"Offer history ends {observed.max()}, expected {cutoff}")
    _write_parquet(offers_path, offers)

    classification = classification_runtime.build_inference_frame(offers, cutoff)
    _require_predictors(
        classification,
        CLASS_CATEGORICAL,
        CLASS_NUMERIC,
        "Classification",
    )
    leaked_class = sorted(CLASS_TARGET_COLUMNS.intersection(classification.columns))
    if leaked_class:
        raise RuntimeError(f"Classification inference target columns present: {leaked_class}")

    regression, ledger = regression_runtime.build_training_frame()
    current = pd.to_datetime(regression["session_date"], errors="coerce").dt.normalize()
    regression = regression.loc[current.eq(cutoff)].copy()
    if regression.empty:
        raise RuntimeError(f"No regression inference rows for {cutoff_text}")
    _require_predictors(regression, REG_CATEGORICAL, REG_NUMERIC, "Regression")
    regression["row_key"] = regression_runtime.stable_row_key(regression)
    regression["observed_price_vnd"] = regression[
        "query_session_observed_fare_vnd"
    ]
    regression = regression.drop(
        columns=sorted(REG_TARGET_COLUMNS.intersection(regression.columns))
    )
    if regression["row_key"].duplicated().any():
        raise RuntimeError("Regression inference row keys are not unique")
    if not pd.to_datetime(regression["feature_time"]).lt(
        pd.to_datetime(regression["session_date"]) + pd.Timedelta(days=1)
    ).all():
        raise RuntimeError("Regression inference feature time exceeds observation day")

    class_path = output_root / "classification_features.parquet"
    regression_path = output_root / "regression_features.parquet"
    ledger_path = output_root / "regression_ledger.parquet"
    _write_parquet(class_path, classification)
    _write_parquet(regression_path, regression)
    _write_parquet(ledger_path, ledger)
    manifest = {
        "status": "PASS",
        "observation_cutoff": cutoff_text,
        "post_cutoff_labels_used": False,
        "standard_offers": {"rows": len(offers), "sha256": sha256(offers_path)},
        "classification": {
            "rows": len(classification),
            "sha256": sha256(class_path),
            "target_columns_present": [],
        },
        "regression": {
            "rows": len(regression),
            "sha256": sha256(regression_path),
            "target_columns_present": [],
        },
    }
    write_json_atomic(output_root / "prediction_provenance.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    args = _args()
    build(args.through_date, args.output_root)


if __name__ == "__main__":
    main()
