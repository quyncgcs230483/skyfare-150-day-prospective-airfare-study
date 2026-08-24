"""Two-stage final refit with native serialization and reload parity."""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from skyfare.production.contract import QUANTILES
from skyfare.production.runtime import (
    FeatureEncoder,
    deterministic_sample,
    ranking_training_frame,
    save_encoder,
    write_parquet_atomic,
)
from skyfare.production.sequence import save_normalizer, sequence_inputs


from skyfare.models import candidate_models as models
from skyfare.evaluation.metrics import rearrange


Heartbeat = Callable[[str, int, dict[str, float]], None]


def _parity(before: np.ndarray, after: np.ndarray, tolerance: float) -> dict[str, object]:
    left = np.asarray(before, dtype=float)
    right = np.asarray(after, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError(f"reload parity shape/finite failure {left.shape} != {right.shape}")
    maximum = float(np.max(np.abs(left - right))) if left.size else 0.0
    if maximum > tolerance:
        raise RuntimeError(f"reload prediction drift {maximum} > {tolerance}")
    return {"status": "PASS", "rows": int(left.shape[0]), "max_abs_difference": maximum, "tolerance": tolerance}


def _epoch_callback(tf: Any, heartbeat: Heartbeat):
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            heartbeat("FULL_REFIT_EPOCH", int(epoch) + 1, logs or {})

    return Callback()


def _fit_stage1_neural(
    tf: Any,
    model: Any,
    fit_inputs: list[np.ndarray],
    fit_target: Any,
    head_inputs: list[np.ndarray],
    head_target: Any,
    artifact_dir: Path,
    heartbeat: Heartbeat,
    sample_weight: Any = None,
):
    stage = artifact_dir / "stage1"
    stage.mkdir(parents=True, exist_ok=True)
    head_prediction, _, metadata = models._fit_neural(
        tf,
        model,
        fit_inputs,
        fit_target,
        head_inputs,
        head_target,
        head_inputs,
        smoke=False,
        artifact_dir=stage,
        sample_weight=sample_weight,
        heartbeat=heartbeat,
    )
    validation = metadata.get("history", {}).get("val_loss", [])
    epochs = int(np.argmin(np.asarray(validation, dtype=float))) + 1 if validation else int(metadata["epochs"])
    metadata["selected_epoch"] = max(1, epochs)
    return head_prediction, max(1, epochs), metadata


def _build_sequence_class(tf: Any, encoder: FeatureEncoder, configuration: str, input_shape, context_dim):
    architecture = configuration.split("_")[0]
    units = 96 if "U96" in configuration else 64
    dropout = 0.20 if "D020" in configuration else 0.15
    model_inputs, hidden = models._sequence_backbone(tf, input_shape, context_dim, architecture, units, dropout)
    outputs = {
        "drop5": tf.keras.layers.Dense(1, name="drop5")(hidden),
        "direction3": tf.keras.layers.Dense(3, name="direction3")(hidden),
        "log_return": tf.keras.layers.Dense(1, name="log_return")(hidden),
    }
    model = tf.keras.Model(model_inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-5),
        loss={
            "drop5": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            "direction3": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "log_return": tf.keras.losses.Huber(delta=0.05),
        },
        loss_weights={"drop5": 1.0, "direction3": 0.25, "log_return": 0.25},
    )
    return model


def _sequence_class_targets(frame: pd.DataFrame):
    change = frame["price_change_pct"].to_numpy(dtype=float)
    direction = np.where(change < -0.01, 0, np.where(change > 0.01, 2, 1)).astype(np.int32)
    log_return = np.log(
        np.maximum(frame["target_price_vnd"].to_numpy(dtype=float), 1.0)
        / np.maximum(frame["price_vnd"].to_numpy(dtype=float), 1.0)
    ).astype(np.float32)
    return {
        "drop5": frame["DROP_5PCT"].to_numpy(dtype=np.float32),
        "direction3": direction,
        "log_return": log_return,
    }


def _build_sequence_point(tf: Any, encoder: FeatureEncoder, configuration: str, input_shape, context_dim):
    architecture = configuration.split("_")[0]
    units = 96 if "U96" in configuration else 64
    model_inputs, hidden = models._sequence_backbone(tf, input_shape, context_dim, architecture, units, 0.15)
    model = tf.keras.Model(model_inputs, tf.keras.layers.Dense(1, name="relative_log")(hidden))
    loss = tf.keras.losses.LogCosh() if "LOGCOSH" in configuration else tf.keras.losses.Huber(delta=0.05)
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-5), loss=loss)
    return model


def _train_ft(job, full, fit, head, artifact_dir: Path, heartbeat: Heartbeat):
    seed = int(job["seed"])
    configuration = str(job["configuration"])
    tf = models._configure_tensorflow(seed)
    stage_encoder = FeatureEncoder.fit(fit, "CLASSIFICATION")
    stage_model = models._ft_model(tf, stage_encoder, configuration)
    stage_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    fit_inputs = list(stage_encoder.neural_inputs(fit))
    head_inputs = list(stage_encoder.neural_inputs(head))
    head_raw, epochs, stage_metadata = _fit_stage1_neural(
        tf,
        stage_model,
        fit_inputs,
        fit["DROP_5PCT"].to_numpy(dtype=np.float32),
        head_inputs,
        head["DROP_5PCT"].to_numpy(dtype=np.float32),
        artifact_dir,
        heartbeat,
        sample_weight=models._class_weights(fit, configuration),
    )
    del stage_model, fit_inputs, head_inputs
    tf.keras.backend.clear_session()
    gc.collect()

    encoder = FeatureEncoder.fit(full, "CLASSIFICATION")
    full_inputs = list(encoder.neural_inputs(full))
    model = models._ft_model(tf, encoder, configuration)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model.fit(
        full_inputs,
        full["DROP_5PCT"].to_numpy(dtype=np.float32),
        sample_weight=models._class_weights(full, configuration),
        epochs=epochs,
        batch_size=1024,
        shuffle=False,
        callbacks=[_epoch_callback(tf, heartbeat), tf.keras.callbacks.TerminateOnNaN()],
        verbose=2,
    )
    weights = artifact_dir / "model.weights.h5"
    model.save_weights(weights)
    save_encoder(artifact_dir / "encoder.json", encoder)
    sample_inputs = [value[:256] for value in full_inputs]
    before = model.predict(sample_inputs, batch_size=256, verbose=0).reshape(-1)
    del model, full_inputs
    tf.keras.backend.clear_session()
    reloaded = models._ft_model(tf, encoder, configuration)
    reloaded.load_weights(weights)
    after = reloaded.predict(sample_inputs, batch_size=256, verbose=0).reshape(-1)
    shutil.rmtree(artifact_dir / "stage1", ignore_errors=True)
    return np.asarray(head_raw).reshape(-1), _parity(before, after, 1e-6), {"fixed_epochs": epochs, "stage1": stage_metadata}


def _train_sequence(job, full, fit, head, artifact_dir: Path, heartbeat: Heartbeat):
    task = str(job["task"])
    source_task = "CLASSIFICATION" if task == "CLASSIFICATION" else "POINT"
    seed = int(job["seed"])
    configuration = str(job["configuration"])
    tf = models._configure_tensorflow(seed)
    stage_encoder = FeatureEncoder.fit(fit, "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION")
    fit_inputs, stage_state = sequence_inputs(fit, source_task, stage_encoder)
    head_inputs, _ = sequence_inputs(head, source_task, stage_encoder, stage_state)
    if task == "CLASSIFICATION":
        stage_model = _build_sequence_class(tf, stage_encoder, configuration, fit_inputs[0].shape[1:], fit_inputs[2].shape[1])
        fit_target = _sequence_class_targets(fit)
        head_target = _sequence_class_targets(head)
        sample_weight = {
            "drop5": models._class_weights(fit, configuration),
            "direction3": np.ones(len(fit), dtype=np.float32),
            "log_return": np.ones(len(fit), dtype=np.float32),
        }
    else:
        stage_model = _build_sequence_point(tf, stage_encoder, configuration, fit_inputs[0].shape[1:], fit_inputs[2].shape[1])
        fit_target = fit["target_anchor_relative_log"].to_numpy(dtype=np.float32)
        head_target = head["target_anchor_relative_log"].to_numpy(dtype=np.float32)
        sample_weight = None
    head_raw, epochs, stage_metadata = _fit_stage1_neural(
        tf,
        stage_model,
        fit_inputs,
        fit_target,
        head_inputs,
        head_target,
        artifact_dir,
        heartbeat,
        sample_weight=sample_weight,
    )
    if task == "CLASSIFICATION":
        head_raw = head_raw["drop5"]
    del stage_model, fit_inputs, head_inputs
    tf.keras.backend.clear_session()
    gc.collect()

    encoder = FeatureEncoder.fit(full, "CLASSIFICATION" if task == "CLASSIFICATION" else "REGRESSION")
    full_inputs, normalizer = sequence_inputs(full, source_task, encoder)
    if task == "CLASSIFICATION":
        model = _build_sequence_class(tf, encoder, configuration, full_inputs[0].shape[1:], full_inputs[2].shape[1])
        target = _sequence_class_targets(full)
        weight = {
            "drop5": models._class_weights(full, configuration),
            "direction3": np.ones(len(full), dtype=np.float32),
            "log_return": np.ones(len(full), dtype=np.float32),
        }
    else:
        model = _build_sequence_point(tf, encoder, configuration, full_inputs[0].shape[1:], full_inputs[2].shape[1])
        target = full["target_anchor_relative_log"].to_numpy(dtype=np.float32)
        weight = None
    model.fit(
        full_inputs,
        target,
        sample_weight=weight,
        epochs=epochs,
        batch_size=1024,
        shuffle=False,
        callbacks=[_epoch_callback(tf, heartbeat), tf.keras.callbacks.TerminateOnNaN()],
        verbose=2,
    )
    weights_path = artifact_dir / "model.weights.h5"
    model.save_weights(weights_path)
    save_encoder(artifact_dir / "encoder.json", encoder)
    save_normalizer(artifact_dir / "sequence_normalizer.json", normalizer)
    sample_inputs = [value[:256] for value in full_inputs]
    raw_before = model.predict(sample_inputs, batch_size=256, verbose=0)
    before = raw_before["drop5"].reshape(-1) if task == "CLASSIFICATION" else raw_before.reshape(-1)
    del model, full_inputs
    tf.keras.backend.clear_session()
    if task == "CLASSIFICATION":
        reloaded = _build_sequence_class(tf, encoder, configuration, sample_inputs[0].shape[1:], sample_inputs[2].shape[1])
    else:
        reloaded = _build_sequence_point(tf, encoder, configuration, sample_inputs[0].shape[1:], sample_inputs[2].shape[1])
    reloaded.load_weights(weights_path)
    raw_after = reloaded.predict(sample_inputs, batch_size=256, verbose=0)
    after = raw_after["drop5"].reshape(-1) if task == "CLASSIFICATION" else raw_after.reshape(-1)
    shutil.rmtree(artifact_dir / "stage1", ignore_errors=True)
    return np.asarray(head_raw).reshape(-1), _parity(before, after, 1e-6), {"fixed_epochs": epochs, "stage1": stage_metadata}


def _cat_quantile_model(job, iterations: int, *, early_stopping: bool):
    from catboost import CatBoostRegressor

    alpha = ",".join(f"{level:.2f}" for level in QUANTILES)
    kwargs = dict(
        loss_function=f"MultiQuantile:alpha={alpha}",
        eval_metric=f"MultiQuantile:alpha={alpha}",
        iterations=iterations,
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        random_seed=int(job["seed"]),
        task_type="CPU",
        thread_count=int(os.environ.get("SKYFARE_V24_CPU_THREADS", "6")),
        allow_writing_files=False,
        verbose=False,
    )
    if early_stopping:
        kwargs.update(od_type="Iter", od_wait=75)
    return CatBoostRegressor(**kwargs)


def _train_cat_quantile(job, full, fit, head, artifact_dir: Path, heartbeat: Heartbeat):
    from catboost import CatBoostRegressor
    from skyfare.models.temporal_runtime import REG_CATEGORICAL

    stage_encoder = FeatureEncoder.fit(fit, "REGRESSION")
    stage_model = _cat_quantile_model(job, 1400, early_stopping=True)
    stage_model.fit(
        stage_encoder.native_frame(fit),
        fit["target_anchor_relative_log"].to_numpy(dtype=float),
        cat_features=list(REG_CATEGORICAL),
        eval_set=(stage_encoder.native_frame(head), head["target_anchor_relative_log"].to_numpy(dtype=float)),
        use_best_model=True,
    )
    rounds = max(1, int(stage_model.get_best_iteration()) + 1)
    head_raw = rearrange(stage_model.predict(stage_encoder.native_frame(head)))
    heartbeat("STAGE1_COMPLETE", rounds, {"fixed_rounds": rounds})
    encoder = FeatureEncoder.fit(full, "REGRESSION")
    model = _cat_quantile_model(job, rounds, early_stopping=False)
    matrix = encoder.native_frame(full)
    model.fit(matrix, full["target_anchor_relative_log"].to_numpy(dtype=float), cat_features=list(REG_CATEGORICAL))
    model_path = artifact_dir / "model.cbm"
    model.save_model(model_path)
    save_encoder(artifact_dir / "encoder.json", encoder)
    sample = encoder.native_frame(deterministic_sample(full))
    before = rearrange(model.predict(sample))
    reloaded = CatBoostRegressor()
    reloaded.load_model(model_path)
    after = rearrange(reloaded.predict(sample))
    return head_raw, _parity(before, after, 1e-10), {"fixed_rounds": rounds}


def _xgb_model(job, iterations: int, *, early_stopping: bool):
    from xgboost import XGBRegressor

    kwargs = dict(
        objective="reg:quantileerror",
        quantile_alpha=np.asarray(QUANTILES),
        n_estimators=iterations,
        learning_rate=0.025,
        max_depth=int(str(job["configuration"])[1:]),
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.03,
        reg_lambda=2.0,
        tree_method="hist",
        device=os.environ.get("SKYFARE_V24_XGB_DEVICE", "cuda:0"),
        max_bin=256,
        random_state=int(job["seed"]),
        n_jobs=int(os.environ.get("SKYFARE_V24_GPU_HOST_THREADS", "4")),
        verbosity=0,
    )
    if early_stopping:
        kwargs["early_stopping_rounds"] = 75
    return XGBRegressor(**kwargs)


def _train_xgb(job, full, fit, head, artifact_dir: Path, heartbeat: Heartbeat):
    from xgboost import XGBRegressor

    stage_encoder = FeatureEncoder.fit(fit, "REGRESSION")
    stage_model = _xgb_model(job, 1800, early_stopping=True)
    stage_model.fit(
        stage_encoder.tree_matrix(fit),
        fit["target_anchor_relative_log"].to_numpy(dtype=float),
        eval_set=[(stage_encoder.tree_matrix(head), head["target_anchor_relative_log"].to_numpy(dtype=float))],
        verbose=False,
    )
    rounds = max(1, int(stage_model.best_iteration) + 1)
    head_raw = rearrange(stage_model.predict(stage_encoder.tree_matrix(head)))
    heartbeat("STAGE1_COMPLETE", rounds, {"fixed_rounds": rounds})
    encoder = FeatureEncoder.fit(full, "REGRESSION")
    model = _xgb_model(job, rounds, early_stopping=False)
    model.fit(encoder.tree_matrix(full), full["target_anchor_relative_log"].to_numpy(dtype=float), verbose=False)
    model_path = artifact_dir / "model.ubj"
    model.save_model(model_path)
    save_encoder(artifact_dir / "encoder.json", encoder)
    sample = encoder.tree_matrix(deterministic_sample(full))
    before = rearrange(model.predict(sample))
    reloaded = XGBRegressor()
    reloaded.load_model(model_path)
    after = rearrange(reloaded.predict(sample))
    return head_raw, _parity(before, after, 1e-6), {"fixed_rounds": rounds}


def _ranker_model(job, iterations: int):
    import lightgbm as lgb

    objective = "lambdarank" if job["family"] == "LGBM_LAMBDARANK" else "rank_xendcg"
    return lgb.LGBMRanker(
        objective=objective,
        n_estimators=iterations,
        learning_rate=0.025,
        num_leaves=int(str(job["configuration"])[1:]),
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=int(job["seed"]),
        n_jobs=int(os.environ.get("SKYFARE_V24_CPU_THREADS", "6")),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        label_gain=[0, 1, 3, 7, 15],
        lambdarank_truncation_level=8,
    )


def _train_ranker(job, full, fit, head, artifact_dir: Path, heartbeat: Heartbeat):
    import lightgbm as lgb

    stage_encoder = FeatureEncoder.fit(fit, "REGRESSION")
    fit_sorted, fit_target, fit_groups = ranking_training_frame(fit)
    head_sorted, head_target, head_groups = ranking_training_frame(head)
    stage_model = _ranker_model(job, 1600)
    stage_model.fit(
        stage_encoder.tree_matrix(fit_sorted),
        fit_target,
        group=fit_groups,
        eval_set=[(stage_encoder.tree_matrix(head_sorted), head_target)],
        eval_group=[head_groups],
        eval_at=[3, 5],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    rounds = max(1, int(stage_model.best_iteration_ or stage_model.n_estimators))
    all_head_sorted = head.sort_values(["query_id", "row_key"], kind="stable")
    sorted_prediction = stage_model.predict(stage_encoder.tree_matrix(all_head_sorted))
    head_raw = models._restore_score(all_head_sorted, sorted_prediction, head)
    heartbeat("STAGE1_COMPLETE", rounds, {"fixed_rounds": rounds})
    encoder = FeatureEncoder.fit(full, "REGRESSION")
    full_sorted, full_target, full_groups = ranking_training_frame(full)
    model = _ranker_model(job, rounds)
    model.fit(encoder.tree_matrix(full_sorted), full_target, group=full_groups)
    model_path = artifact_dir / "model.txt"
    model.booster_.save_model(model_path)
    save_encoder(artifact_dir / "encoder.json", encoder)
    sample = deterministic_sample(full_sorted)
    matrix = encoder.tree_matrix(sample)
    before = model.predict(matrix)
    reloaded = lgb.Booster(model_file=str(model_path))
    after = reloaded.predict(matrix)
    return head_raw, _parity(before, after, 1e-10), {"fixed_rounds": rounds}


def _save_head_prediction(job, head: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    task = str(job["task"])
    if task == "CLASSIFICATION":
        frame = pd.DataFrame(
            {
                "row_key": head["row_key"].to_numpy(dtype=np.uint64),
                "target": head["DROP_5PCT"].to_numpy(dtype=np.int8),
                "raw_score": np.asarray(prediction, dtype=float).reshape(-1),
            }
        )
    elif task == "DISTRIBUTION":
        values = np.asarray(prediction, dtype=float)
        payload = {
            "row_key": head["row_key"].to_numpy(dtype=np.uint64),
            "target": head["target_anchor_relative_log"].to_numpy(dtype=float),
            "query_dud": head["query_dud"].to_numpy(dtype=float),
            "regime": head["regime"].astype(str).to_numpy(),
        }
        for index, level in enumerate(QUANTILES):
            payload[f"q{int(round(level * 100)):02d}"] = values[:, index]
        frame = pd.DataFrame(payload)
    else:
        values = np.asarray(prediction, dtype=float)
        frame = pd.DataFrame(
            {
                "row_key": head["row_key"].to_numpy(dtype=np.uint64),
                "prediction": values.reshape(len(values), -1)[:, 0],
            }
        )
    write_parquet_atomic(path, frame)


def train_and_serialize(job, full, fit, head, artifact_dir: Path, heartbeat: Heartbeat):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    family = str(job["family"])
    if family == "FT_CLASS":
        head_prediction, parity, metadata = _train_ft(job, full, fit, head, artifact_dir, heartbeat)
    elif family in {"SEQ_CLASS", "SEQ_POINT"}:
        head_prediction, parity, metadata = _train_sequence(job, full, fit, head, artifact_dir, heartbeat)
    elif family == "CAT_MULTIQUANTILE":
        head_prediction, parity, metadata = _train_cat_quantile(job, full, fit, head, artifact_dir, heartbeat)
    elif family == "XGB_QUANTILE":
        head_prediction, parity, metadata = _train_xgb(job, full, fit, head, artifact_dir, heartbeat)
    elif family in {"LGBM_LAMBDARANK", "LGBM_XENDCG"}:
        head_prediction, parity, metadata = _train_ranker(job, full, fit, head, artifact_dir, heartbeat)
    else:
        raise RuntimeError(f"unsupported frozen production family: {family}")
    _save_head_prediction(job, head, head_prediction, artifact_dir / "head_predictions.parquet")
    return parity, metadata
