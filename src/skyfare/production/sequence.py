"""Sequence construction with persisted normalization for production inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from skyfare.production.runtime import INPUT_ROOT, OUTPUT_ROOT, write_json_atomic

os.environ.setdefault("SKYFARE_SEQUENCE_INPUT_ROOT", str(INPUT_ROOT))
os.environ.setdefault("SKYFARE_SEQUENCE_OUTPUT_ROOT", str(OUTPUT_ROOT))

from skyfare.models.sequence_runtime import (
    EXACT_LENGTH,
    load_or_build_sequence_source,
    make_sequences,
)


def raw_sequence_arrays(frame: pd.DataFrame, task: str, encoder):
    identifier = "offer_id" if task == "CLASSIFICATION" else "target_offer_id"
    sequence, mask, length = make_sequences(
        frame[identifier],
        frame["feature_time"],
        load_or_build_sequence_source("EXACT"),
        length=EXACT_LENGTH,
        inclusive=task != "CLASSIFICATION",
        forbidden_sessions=frame["target_session_key"],
    )
    context = encoder.tree_matrix(frame, standardized=True)
    support = np.column_stack([np.log1p(length), length > 0]).astype(np.float32)
    context = np.column_stack([context, support]).astype(np.float32)
    return sequence, mask, context


def fit_normalizer(sequence: np.ndarray, mask: np.ndarray) -> dict[str, list[float]]:
    observed = sequence[mask]
    if not len(observed):
        raise RuntimeError("sequence fit population contains no history")
    mean = observed.mean(axis=0)
    scale = np.maximum(observed.std(axis=0), 1e-6)
    return {"mean": mean.astype(float).tolist(), "scale": scale.astype(float).tolist()}


def normalize(sequence: np.ndarray, mask: np.ndarray, state: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(state["mean"], dtype=np.float32)
    scale = np.asarray(state["scale"], dtype=np.float32)
    result = sequence.copy()
    result[mask] = (result[mask] - mean) / scale
    return result


def sequence_inputs(frame: pd.DataFrame, task: str, encoder, state=None):
    sequence, mask, context = raw_sequence_arrays(frame, task, encoder)
    fitted = fit_normalizer(sequence, mask) if state is None else state
    return [normalize(sequence, mask, fitted), mask, context], fitted


def save_normalizer(path: Path, state: dict[str, list[float]]) -> None:
    write_json_atomic(path, state)


def load_normalizer(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if sorted(payload) != ["mean", "scale"]:
        raise RuntimeError("sequence normalization state changed")
    return payload
