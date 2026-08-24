"""IO, temporal training split, resume, and artifact integrity for production refit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyfare.production.contract import INPUT_ROWS, INPUT_SHA256, PRODUCTION_ORIGIN
from skyfare.core.paths import DataLayout


MODULE_DIR = Path(__file__).resolve().parent
LAYOUT = DataLayout.resolve()
ROOT = LAYOUT.root
INPUT_ROOT = Path(
    os.environ.get("SKYFARE_PRODUCTION_INPUT_ROOT", LAYOUT.processed / "production")
).resolve()
OUTPUT_ROOT = Path(
    os.environ.get("SKYFARE_PRODUCTION_OUTPUT_ROOT", LAYOUT.artifacts / "production_refit")
).resolve()

from skyfare.models.temporal_runtime import FeatureEncoder, inner_split, ranking_frame, regression_row_keys  # noqa: E402


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


def validate_inputs(*, verify_hashes: bool = True) -> dict[str, object]:
    report: dict[str, object] = {}
    for name, expected_rows in INPUT_ROWS.items():
        path = INPUT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = len(pd.read_parquet(path, columns=["feature_time"]))
        if rows != expected_rows:
            raise RuntimeError(f"{name}: row count changed {rows} != {expected_rows}")
        digest = sha256(path) if verify_hashes else None
        if verify_hashes and digest != INPUT_SHA256[name]:
            raise RuntimeError(f"{name}: sha256 changed")
        report[name] = {"rows": rows, "sha256": digest or "SKIPPED"}
    return report


@lru_cache(maxsize=2)
def load_frame(task: str) -> pd.DataFrame:
    source_task = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
    name = (
        "classification_training_frame.parquet"
        if source_task == "CLASSIFICATION"
        else "regression_training_frame.parquet"
    )
    frame = pd.read_parquet(INPUT_ROOT / name)
    frame["feature_time"] = pd.to_datetime(frame["feature_time"], errors="raise")
    frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
    if source_task == "REGRESSION":
        frame["row_key"] = regression_row_keys(frame)
    if frame["row_key"].duplicated().any():
        raise RuntimeError(f"{task}: duplicate row identity")
    origin = pd.Timestamp(PRODUCTION_ORIGIN)
    if not frame["label_time"].lt(origin).all():
        raise RuntimeError(f"{task}: post-cutoff label entered production refit")
    return frame


def training_frames(task: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_task = "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION"
    full = load_frame(source_task).copy()
    if window == "RECENT84":
        cutoff = pd.Timestamp(PRODUCTION_ORIGIN) - pd.Timedelta(days=84)
        full = full.loc[full["label_time"].ge(cutoff)].copy()
    elif window != "EXPANDING":
        raise RuntimeError(f"unknown training window {window}")
    if full.empty:
        raise RuntimeError(f"{task}: empty production training frame")
    fit, head = inner_split(full, source_task)
    return full, fit, head


def save_encoder(path: Path, encoder: FeatureEncoder) -> None:
    write_json_atomic(path, asdict(encoder))


def load_encoder(path: Path) -> FeatureEncoder:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FeatureEncoder(
        categorical_columns=tuple(payload["categorical_columns"]),
        numeric_columns=tuple(payload["numeric_columns"]),
        categories={key: list(value) for key, value in payload["categories"].items()},
        medians={key: float(value) for key, value in payload["medians"].items()},
        means={key: float(value) for key, value in payload["means"].items()},
        scales={key: float(value) for key, value in payload["scales"].items()},
    )


def job_root(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


@lru_cache(maxsize=1)
def current_code_sha256() -> str:
    digest = hashlib.sha256()
    for directory in (MODULE_DIR, Path(__file__).resolve().parents[1] / "models"):
        for path in sorted(directory.glob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def artifact_complete(job_id: str) -> bool:
    root = job_root(job_id)
    done_path = root / "done.json"
    if not done_path.is_file():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "COMPLETE" or done.get("code_sha256") != current_code_sha256():
            return False
        if done.get("job", {}).get("job_id") != job_id:
            return False
        for relative, expected in done.get("artifacts", {}).items():
            path = root / relative
            if not path.is_file() or sha256(path) != expected:
                return False
        return done.get("reload_parity", {}).get("status") == "PASS"
    except (OSError, ValueError, TypeError):
        return False


def deterministic_sample(frame: pd.DataFrame, limit: int = 256) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    return frame.sort_values("row_key", kind="stable").iloc[
        np.linspace(0, len(frame) - 1, limit, dtype=np.int64)
    ].copy()


def ranking_training_frame(frame: pd.DataFrame):
    return ranking_frame(frame)
