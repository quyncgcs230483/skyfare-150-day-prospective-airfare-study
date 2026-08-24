"""Test 2 model dispatch with target-free sequence-point scoring."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from skyfare.models import candidate_models as models


def _sequence_point_target_free(
    job: dict[str, object],
    frames: tuple[pd.DataFrame, ...],
    encoder: models.FeatureEncoder,
    smoke: bool,
    artifact_dir: Path,
    heartbeat: Callable[[str, int, dict[str, float]], None] | None,
):
    tf = models._configure_tensorflow(int(job["seed"]))
    configuration = str(job["configuration"])
    architecture = configuration.split("_")[0]
    units = 96 if "U96" in configuration else 64
    inputs = models._sequence_arrays(frames, "POINT", encoder)
    model_inputs, hidden = models._sequence_backbone(
        tf,
        inputs[0][0].shape[1:],
        inputs[0][2].shape[1],
        architecture,
        units,
        0.15,
    )
    output = tf.keras.layers.Dense(1, name="relative_log")(hidden)
    model = tf.keras.Model(model_inputs, output)
    loss = (
        tf.keras.losses.LogCosh()
        if "LOGCOSH" in configuration
        else tf.keras.losses.Huber(delta=0.05)
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-5),
        loss=loss,
    )

    # Valid/Test 2 labels are intentionally absent during fitting and prediction.
    fit_target = frames[0]["target_anchor_relative_log"].to_numpy(np.float32)
    head_target = frames[1]["target_anchor_relative_log"].to_numpy(np.float32)
    raw_head, raw_valid, metadata = models._fit_neural(
        tf,
        model,
        list(inputs[0]),
        fit_target,
        list(inputs[1]),
        head_target,
        list(inputs[2]),
        smoke=smoke,
        artifact_dir=artifact_dir,
        heartbeat=heartbeat,
    )
    return (
        raw_head.reshape(-1),
        raw_valid.reshape(-1),
        {
            "device": "GPU",
            "architecture": architecture,
            "loss": loss.name,
            "target_free_sequence_point": True,
            **metadata,
        },
    )


def fit_predict(
    job: dict[str, object],
    fit: pd.DataFrame,
    head: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    smoke: bool,
    artifact_dir: Path,
    heartbeat: Callable[[str, int, dict[str, float]], None] | None = None,
):
    if str(job["family"]) != "SEQ_POINT":
        return models.fit_predict(
            job,
            fit,
            head,
            valid,
            smoke=smoke,
            artifact_dir=artifact_dir,
            heartbeat=heartbeat,
        )

    started = time.monotonic()
    if heartbeat:
        heartbeat("FIT_STARTED", 0, {})
    encoder = models.FeatureEncoder.fit(fit, "REGRESSION")
    head_prediction, valid_prediction, metadata = _sequence_point_target_free(
        job,
        (fit, head, valid),
        encoder,
        smoke,
        artifact_dir,
        heartbeat,
    )
    elapsed = time.monotonic() - started
    if heartbeat:
        heartbeat("FIT_COMPLETE", 1, {"fit_seconds": elapsed})
    return head_prediction, valid_prediction, {
        **metadata,
        "encoder": encoder.metadata(),
        "fit_seconds": elapsed,
    }
