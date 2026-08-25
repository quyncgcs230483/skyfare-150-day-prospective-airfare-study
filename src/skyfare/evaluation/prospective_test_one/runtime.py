"""IO, temporal isolation, and atomic artifacts for prospective Test 1."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.core.paths import DataLayout
from skyfare.evaluation.prospective_test_one.contract import OBSERVATION_CUTOFF, TEST_1, TEST_2
from skyfare.models.temporal_runtime import (
    CLASS_CATEGORICAL,
    CLASS_NUMERIC,
    REG_CATEGORICAL,
    REG_NUMERIC,
    inner_split,
    regression_row_keys,
)

MODULE_DIR = Path(__file__).resolve().parent
LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
DEV_INPUT_ROOT = Path(
    os.environ.get("SKYFARE_TEST_ONE_DEVELOPMENT_INPUT_ROOT", LAYOUT.processed / "development")
).resolve()
SEALED_ROOT = Path(
    os.environ.get("SKYFARE_TEST_ONE_SEALED_ROOT", LAYOUT.processed / "prospective_test_one")
).resolve()
OUTPUT_ROOT = Path(
    os.environ.get("SKYFARE_TEST_ONE_OUTPUT_ROOT", LAYOUT.artifacts / "prospective_test_one")
).resolve()
STAGING_ROOT = OUTPUT_ROOT / "staging"
CLASS_DEV = DEV_INPUT_ROOT / "classification_training_frame.parquet"
REG_DEV = DEV_INPUT_ROOT / "regression_training_frame.parquet"
CLASS_SOURCE = SEALED_ROOT / "classification_test_1_frame.parquet"
REG_SOURCE = SEALED_ROOT / "regression_test_1_frame.parquet"
OFFERS_SOURCE = SEALED_ROOT / "standard_offers.parquet"
OBSERVABILITY_SOURCE = SEALED_ROOT / "classification_test_1_observability.parquet"
CLASS_FEATURES = STAGING_ROOT / "classification_features.parquet"
REG_FEATURES = STAGING_ROOT / "regression_features.parquet"
CLASS_LABELS = STAGING_ROOT / "classification_labels.parquet"
REG_LABELS = STAGING_ROOT / "regression_labels.parquet"

CLASS_TARGET_COLUMNS = ("DROP_5PCT", "target_price_vnd", "price_change_pct")
REG_TARGET_COLUMNS = ("target_anchor_relative_log", "target_session_price_vnd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".parquet") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalized_dates(frame: pd.DataFrame, column: str) -> set[str]:
    values = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    return {value.strftime("%Y-%m-%d") for value in values.unique()}


def expected_test_dates() -> set[str]:
    return {
        value.strftime("%Y-%m-%d")
        for value in pd.date_range(TEST_1[0], TEST_1[1], freq="D")
    }


@lru_cache(maxsize=2)
def load_dev_frame(task: str) -> pd.DataFrame:
    path = CLASS_DEV if task == "CLASSIFICATION" else REG_DEV
    frame = pd.read_parquet(path)
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], errors="raise")
    frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
    if task != "CLASSIFICATION":
        frame["row_key"] = regression_row_keys(frame)
    return frame


@lru_cache(maxsize=2)
def load_test_features(task: str) -> pd.DataFrame:
    path = CLASS_FEATURES if task == "CLASSIFICATION" else REG_FEATURES
    frame = pd.read_parquet(path)
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], errors="raise")
    frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
    return frame


def training_frames(task: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_task = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
    frame = load_dev_frame(source_task)
    start = pd.Timestamp(TEST_1[0])
    if source_task == "CLASSIFICATION":
        train = frame.loc[frame["feature_time"].lt(start) & frame["label_time"].lt(start)].copy()
        time_column = "feature_time"
    else:
        train = frame.loc[frame["label_time"].lt(start)].copy()
        time_column = "label_time"
    if train.empty or not train["label_time"].lt(start).all():
        raise RuntimeError(f"{task}: prospective training purge failed")
    if window == "RECENT84":
        train = train.loc[train[time_column].ge(start - pd.Timedelta(days=84))].copy()
    elif window != "EXPANDING":
        raise RuntimeError(f"unknown training window {window}")
    fit, head = inner_split(train, source_task)
    valid = load_test_features(source_task).copy()
    return fit, head, valid


def required_feature_columns(task: str) -> set[str]:
    if task == "CLASSIFICATION":
        return {
            *CLASS_CATEGORICAL,
            *CLASS_NUMERIC,
            "row_key",
            "offer_id",
            "target_offer_id",
            "target_session_key",
            "feature_time",
            "label_time",
            "price_vnd",
            "route",
            "airline",
            "regime",
            "target_dud",
        }
    return {
        *REG_CATEGORICAL,
        *REG_NUMERIC,
        "row_key",
        "query_id",
        "target_offer_id",
        "source_session_key",
        "target_session_key",
        "feature_time",
        "label_time",
        "query_dud",
        "prior_anchor_vnd",
        "route_airline",
        "regime",
    }


def job_root(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


@lru_cache(maxsize=1)
def current_code_sha256() -> str:
    digest = hashlib.sha256()
    for directory in (MODULE_DIR, Path(__file__).resolve().parents[2] / "models"):
        for path in sorted(directory.glob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def preflight_sha256() -> str | None:
    path = OUTPUT_ROOT / "runtime" / "preflight_test1.json"
    return sha256(path) if path.is_file() else None


def artifact_complete(job_id: str) -> bool:
    root = job_root(job_id)
    done = root / "done.json"
    prediction = root / "predictions.parquet"
    if not done.is_file() or not prediction.is_file():
        return False
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "COMPLETE"
        and payload.get("code_sha256") == current_code_sha256()
        and payload.get("preflight_sha256") == preflight_sha256()
        and payload.get("prediction_sha256") == sha256(prediction)
        and payload.get("job", {}).get("job_id") == job_id
    )


def assert_no_test2(frame: pd.DataFrame) -> None:
    cutoff = pd.Timestamp(OBSERVATION_CUTOFF) + pd.Timedelta(days=1)
    if frame["label_time"].min() < cutoff:
        raise RuntimeError("Test 1 source contains pre-Test-1 labels")
    if frame["label_time"].max() >= pd.Timestamp(TEST_2[0]):
        raise RuntimeError("Test 2 label entered Test 1 source")


def finite(values: pd.Series, name: str) -> None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite values in {name}")
