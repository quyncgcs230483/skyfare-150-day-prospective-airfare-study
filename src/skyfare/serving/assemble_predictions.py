#!/usr/bin/env python3
"""Join task-specific model outputs into one verified application frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skyfare.core.integrity import sha256, write_json_atomic


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build(feature_root: Path, prediction_root: Path, output: Path) -> dict[str, object]:
    classification = pd.read_parquet(
        feature_root / "classification_features.parquet"
    )
    regression = pd.read_parquet(feature_root / "regression_features.parquet")
    class_prediction = pd.read_parquet(
        prediction_root / "classification_predictions.parquet"
    )
    point = pd.read_parquet(prediction_root / "point_predictions.parquet")
    distribution = pd.read_parquet(
        prediction_root / "distribution_predictions.parquet"
    )
    ranking = pd.read_parquet(prediction_root / "ranking_predictions.parquet")

    class_map = classification[["row_key", "offer_id"]].merge(
        class_prediction,
        on="row_key",
        how="inner",
        validate="one_to_one",
    )
    class_map = class_map.rename(
        columns={
            "drop_5pct_probability": "drop_probability",
            "guarded_pilot_action": "action",
        }
    )
    class_map = class_map[
        ["offer_id", "drop_probability", "action", "operational_action"]
    ]

    frame = regression.merge(point, on="row_key", how="inner", validate="one_to_one")
    frame = frame.merge(
        distribution, on="row_key", how="inner", validate="one_to_one"
    ).merge(ranking, on="row_key", how="inner", validate="one_to_one")
    frame["offer_id"] = frame["target_offer_id"].astype("uint64")
    frame = frame.merge(class_map, on="offer_id", how="left", validate="one_to_one")
    frame["action"] = frame["action"].fillna("BUY")
    frame["operational_action"] = frame["operational_action"].fillna("BUY")
    frame["session_label"] = frame["model_session_label"].astype("string")
    frame["baseline_price_vnd"] = frame["prior_anchor_vnd"]
    frame["departure_HHMM"] = pd.to_numeric(
        frame["departure_HHMM"], errors="raise"
    ).astype("int16")
    frame["schedule_slot_id"] = frame["schedule_slot_id"].astype("string")
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="raise")
    frame["flight_date"] = pd.to_datetime(frame["flight_date"], errors="raise")
    frame["dud_support_mode"] = "ON_GRID"
    frame["interpolation_interval"] = ""
    frame["interpolation_note"] = "Canonical booking-window prediction"

    columns = [
        "session_date",
        "session_label",
        "route",
        "flight_date",
        "airline",
        "departure_HHMM",
        "schedule_slot_id",
        "query_dud",
        "dud_support_mode",
        "predicted_price_vnd",
        "baseline_price_vnd",
        "observed_price_vnd",
        "drop_probability",
        "action",
        "operational_action",
        "ranking_score",
        "interpolation_interval",
        "interpolation_note",
    ]
    quantiles = sorted(
        column for column in frame if column.startswith("predicted_price_q")
    )
    frame = frame[[*columns, *quantiles]].sort_values(
        ["session_date", "session_label", "route", "flight_date", "ranking_score"],
        ascending=[True, True, True, True, False],
        kind="stable",
    )
    if frame.empty or not frame["predicted_price_vnd"].gt(0).all():
        raise RuntimeError("Assembled prediction frame is empty or invalid")
    if frame.duplicated(
        [
            "session_date",
            "session_label",
            "route",
            "flight_date",
            "airline",
            "departure_HHMM",
        ]
    ).any():
        raise RuntimeError("Assembled serving identity is not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    manifest = {
        "status": "PASS",
        "rows": len(frame),
        "routes": int(frame["route"].nunique()),
        "airlines": int(frame["airline"].nunique()),
        "output_sha256": sha256(output),
        "classification_rows": len(class_map),
        "regression_rows": len(regression),
    }
    write_json_atomic(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    args = _args()
    build(args.feature_root, args.prediction_root, args.output)


if __name__ == "__main__":
    main()
