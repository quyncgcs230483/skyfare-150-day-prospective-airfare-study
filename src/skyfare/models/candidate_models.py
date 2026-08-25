"""Bounded V24 model refinements behind one fit/predict interface."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from skyfare.evaluation.metrics import rearrange
from skyfare.models.selection_contract import QUANTILES
from skyfare.models.temporal_runtime import (
    CLASS_CATEGORICAL,
    REG_CATEGORICAL,
    FeatureEncoder,
    ranking_frame,
    sample_weights,
)

SUPPORTED_FAMILIES = {
    "LGBM_CLASS", "CAT_CLASS", "FT_CLASS", "SEQ_CLASS",
    "LGBM_POINT", "CAT_POINT", "SEQ_POINT",
    "XGB_QUANTILE", "CAT_MULTIQUANTILE", "CAT_UNCERTAINTY",
    "LGBM_LAMBDARANK", "LGBM_XENDCG",
}


def _finite(name: str, values: np.ndarray, expected: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape[0] != expected or not np.isfinite(result).all():
        raise RuntimeError(f"{name}: invalid prediction shape or non-finite value")
    return result


def _class_weights(frame: pd.DataFrame, configuration: str) -> np.ndarray:
    target = frame["DROP_5PCT"].to_numpy(dtype=np.int8)
    balanced = "BAL" in configuration or "MARKET" in configuration
    weight = sample_weights(target, balanced)
    if "MARKET" in configuration:
        route_count = frame.groupby("route", observed=True)["route"].transform("size").to_numpy(dtype=float)
        market = 1.0 / np.sqrt(np.maximum(route_count, 1.0))
        market /= market.mean()
        weight = weight * np.clip(market, 0.35, 3.0)
        weight /= weight.mean()
    return weight.astype(np.float32)


def _lgbm_class(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    import lightgbm as lgb

    fit, head, valid = frames
    configuration = str(job["configuration"])
    leaves = 31 if "L31" in configuration else 63
    depth = 3 if "D003" in configuration else (4 if "D004" in configuration else 6)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=16 if smoke else 1400,
        learning_rate=0.05 if smoke else 0.025,
        num_leaves=leaves,
        max_depth=depth,
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
    )
    x_fit, x_head, x_valid = (encoder.tree_matrix(frame) for frame in frames)
    model.fit(
        x_fit,
        fit["DROP_5PCT"].to_numpy(dtype=np.int8),
        sample_weight=_class_weights(fit, configuration),
        eval_set=[(x_head, head["DROP_5PCT"].to_numpy(dtype=np.int8))],
        eval_metric="binary_logloss",
        categorical_feature=list(range(len(CLASS_CATEGORICAL))),
        callbacks=[lgb.early_stopping(4 if smoke else 75, verbose=False), lgb.log_evaluation(0)],
    )
    return (
        _finite("lgbm class head", model.predict(x_head, raw_score=True), len(head)).reshape(-1),
        _finite("lgbm class valid", model.predict(x_valid, raw_score=True), len(valid)).reshape(-1),
        {"device": "CPU", "best_iteration": int(model.best_iteration_ or model.n_estimators)},
    )


def _cat_class(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    from catboost import CatBoostClassifier

    fit, head, valid = frames
    configuration = str(job["configuration"])
    depth = 7 if "D7" in configuration else 8
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=16 if smoke else 1300,
        depth=depth,
        learning_rate=0.05 if smoke else 0.03,
        l2_leaf_reg=6.0,
        random_seed=int(job["seed"]),
        task_type="GPU",
        devices=os.environ.get("SKYFARE_V24_GPU_DEVICE", "0"),
        thread_count=int(os.environ.get("SKYFARE_V24_GPU_HOST_THREADS", "4")),
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=4 if smoke else 75,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        gpu_ram_part=float(os.environ.get("SKYFARE_V24_CAT_GPU_RAM_PART", "0.42")),
    )
    x_fit, x_head, x_valid = (encoder.native_frame(frame) for frame in frames)
    model.fit(
        x_fit,
        fit["DROP_5PCT"].to_numpy(dtype=np.int8),
        cat_features=list(CLASS_CATEGORICAL),
        sample_weight=_class_weights(fit, configuration),
        eval_set=(x_head, head["DROP_5PCT"].to_numpy(dtype=np.int8)),
        use_best_model=True,
    )
    return (
        _finite("cat class head", model.predict(x_head, prediction_type="RawFormulaVal"), len(head)).reshape(-1),
        _finite("cat class valid", model.predict(x_valid, prediction_type="RawFormulaVal"), len(valid)).reshape(-1),
        {"device": "GPU", "best_iteration": int(model.get_best_iteration()), "gpu_nondeterministic": True},
    )


def _configure_tensorflow(seed: int):
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    memory_limit = int(os.environ.get("SKYFARE_V24_TF_MEMORY_LIMIT_MB", "0"))
    for device in devices:
        try:
            if memory_limit > 0:
                tf.config.set_logical_device_configuration(
                    device, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            else:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            int(os.environ.get("SKYFARE_V24_GPU_HOST_THREADS", "4"))
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    return tf


def _ft_model(tf: Any, encoder: FeatureEncoder, configuration: str):
    wide = "WIDE" in configuration
    token_dim, blocks, dropout = (48, 3, 0.15) if wide else (32, 2, 0.10)
    categorical = tf.keras.Input((len(encoder.categorical_columns),), dtype="int32", name="categorical")
    numeric = tf.keras.Input((len(encoder.numeric_columns),), dtype="float32", name="numeric")
    tokens = []
    for index, column in enumerate(encoder.categorical_columns):
        selected = tf.keras.layers.Lambda(lambda value, i=index: value[:, i], name=f"cat_{index}")(categorical)
        token = tf.keras.layers.Embedding(len(encoder.categories[column]) + 1, token_dim)(selected)
        tokens.append(tf.keras.layers.Reshape((1, token_dim))(token))
    for index in range(len(encoder.numeric_columns)):
        selected = tf.keras.layers.Lambda(lambda value, i=index: value[:, i:i + 1], name=f"num_{index}")(numeric)
        tokens.append(tf.keras.layers.Reshape((1, token_dim))(tf.keras.layers.Dense(token_dim)(selected)))
    cls_index = tf.keras.layers.Lambda(lambda value: tf.zeros((tf.shape(value)[0], 1), tf.int32))(numeric)
    cls = tf.keras.layers.Embedding(1, token_dim)(cls_index)
    hidden = tf.keras.layers.Concatenate(axis=1)([cls, *tokens])
    for _ in range(blocks):
        normalized = tf.keras.layers.LayerNormalization()(hidden)
        attended = tf.keras.layers.MultiHeadAttention(4, token_dim // 4, dropout=dropout)(normalized, normalized)
        hidden = tf.keras.layers.Add()([hidden, attended])
        normalized = tf.keras.layers.LayerNormalization()(hidden)
        feed = tf.keras.layers.Dense(token_dim * 2, activation="gelu")(normalized)
        feed = tf.keras.layers.Dropout(dropout)(feed)
        hidden = tf.keras.layers.Add()([hidden, tf.keras.layers.Dense(token_dim)(feed)])
    output = tf.keras.layers.Lambda(lambda value: value[:, 0, :])(hidden)
    output = tf.keras.layers.Dense(1, name="drop5_logit")(tf.keras.layers.LayerNormalization()(output))
    return tf.keras.Model([categorical, numeric], output)


def _fit_neural(
    tf: Any,
    model: Any,
    fit_inputs: Any,
    fit_target: Any,
    head_inputs: Any,
    head_target: Any,
    valid_inputs: Any,
    *,
    smoke: bool,
    artifact_dir: Path,
    sample_weight: Any = None,
    heartbeat: Callable[[str, int, dict[str, float]], None] | None = None,
):
    class Heartbeat(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if heartbeat:
                heartbeat("FIT_EPOCH", int(epoch) + 1, logs or {})

    checkpoint = artifact_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2 if smoke else 5, min_delta=1e-5, restore_best_weights=True),
        Heartbeat(),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        fit_inputs,
        fit_target,
        sample_weight=sample_weight,
        validation_data=(head_inputs, head_target),
        epochs=2 if smoke else 30,
        batch_size=512 if smoke else 1024,
        shuffle=False,
        callbacks=callbacks,
        verbose=2,
    )
    head_prediction = model.predict(head_inputs, batch_size=2048, verbose=0)
    valid_prediction = model.predict(valid_inputs, batch_size=2048, verbose=0)
    return head_prediction, valid_prediction, {
        "epochs": len(history.history["loss"]),
        "history": {name: [float(value) for value in values] for name, values in history.history.items()},
    }


def _ft_class(
    job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool,
    artifact_dir: Path, heartbeat: Callable[[str, int, dict[str, float]], None] | None,
):
    tf = _configure_tensorflow(int(job["seed"]))
    inputs = [encoder.neural_inputs(frame) for frame in frames]
    model = _ft_model(tf, encoder, str(job["configuration"]))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    raw_head, raw_valid, metadata = _fit_neural(
        tf, model, list(inputs[0]), frames[0]["DROP_5PCT"].to_numpy(dtype=np.float32),
        list(inputs[1]), frames[1]["DROP_5PCT"].to_numpy(dtype=np.float32), list(inputs[2]),
        smoke=smoke, artifact_dir=artifact_dir,
        sample_weight=_class_weights(frames[0], str(job["configuration"])), heartbeat=heartbeat,
    )
    return raw_head.reshape(-1), raw_valid.reshape(-1), {"device": "GPU", **metadata}


def _sequence_arrays(frames: tuple[pd.DataFrame, ...], task: str, encoder: FeatureEncoder):
    from skyfare.models.sequence_runtime import (
        EXACT_LENGTH,
        load_or_build_sequence_source,
        make_sequences,
        normalize_sequence_sets,
    )

    full = pd.concat(frames, ignore_index=True)
    identifier = "offer_id" if task == "CLASSIFICATION" else "target_offer_id"
    sequence, mask, length = make_sequences(
        full[identifier], full["feature_time"], load_or_build_sequence_source("EXACT"),
        length=EXACT_LENGTH, inclusive=task != "CLASSIFICATION", forbidden_sessions=full["target_session_key"],
    )
    sizes = [len(frame) for frame in frames]
    boundaries = np.cumsum([0, *sizes])
    sequences = tuple(sequence[boundaries[i]:boundaries[i + 1]] for i in range(3))
    masks = tuple(mask[boundaries[i]:boundaries[i + 1]] for i in range(3))
    lengths = tuple(length[boundaries[i]:boundaries[i + 1]] for i in range(3))
    normalized = normalize_sequence_sets(
        sequences[0], masks[0], (sequences[1], masks[1]), (sequences[2], masks[2])
    )
    contexts = []
    for frame, history_length in zip(frames, lengths, strict=True):
        context = encoder.tree_matrix(frame, standardized=True)
        support = np.column_stack([np.log1p(history_length), history_length > 0]).astype(np.float32)
        contexts.append(np.column_stack([context, support]).astype(np.float32))
    return tuple([normalized[i], masks[i], contexts[i]] for i in range(3))


def _safe_recurrent_mask(tf: Any, mask: Any):
    return tf.keras.layers.Lambda(
        lambda value: tf.where(
            tf.reduce_any(value, axis=1, keepdims=True), value,
            tf.concat([tf.ones_like(value[:, :1]), tf.zeros_like(value[:, 1:])], axis=1),
        ),
        output_shape=(mask.shape[-1],),
    )(mask)


def _sequence_backbone(tf: Any, sequence_shape: tuple[int, int], context_dim: int, architecture: str, units: int, dropout: float):
    sequence = tf.keras.Input(sequence_shape, name="sequence")
    mask = tf.keras.Input((sequence_shape[0],), dtype="bool", name="sequence_mask")
    context = tf.keras.Input((context_dim,), name="context")
    safe_mask = _safe_recurrent_mask(tf, mask)
    layer = {"RNN": tf.keras.layers.SimpleRNN, "GRU": tf.keras.layers.GRU, "LSTM": tf.keras.layers.LSTM}[architecture]
    encoded = layer(units, name=f"{architecture.lower()}_encoder")(sequence, mask=safe_mask)
    has_history = tf.keras.layers.Lambda(
        lambda value: tf.cast(tf.reduce_any(value, axis=1, keepdims=True), tf.float32),
        output_shape=(1,),
    )(mask)
    encoded = tf.keras.layers.Multiply()([encoded, has_history])
    context_hidden = tf.keras.layers.Dense(64, activation="relu")(context)
    hidden = tf.keras.layers.Concatenate()([encoded, context_hidden])
    hidden = tf.keras.layers.Dense(96, activation="relu")(hidden)
    return (sequence, mask, context), tf.keras.layers.Dropout(dropout)(hidden)


def _sequence_class(
    job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool,
    artifact_dir: Path, heartbeat: Callable[[str, int, dict[str, float]], None] | None,
):
    tf = _configure_tensorflow(int(job["seed"]))
    configuration = str(job["configuration"])
    architecture = configuration.split("_")[0]
    units = 96 if "U96" in configuration else 64
    dropout = 0.20 if "D020" in configuration else 0.15
    inputs = _sequence_arrays(frames, "CLASSIFICATION", encoder)
    model_inputs, hidden = _sequence_backbone(tf, inputs[0][0].shape[1:], inputs[0][2].shape[1], architecture, units, dropout)
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

    def targets(frame: pd.DataFrame):
        change = frame["price_change_pct"].to_numpy(dtype=float)
        direction = np.where(change < -0.01, 0, np.where(change > 0.01, 2, 1)).astype(np.int32)
        log_return = np.log(
            np.maximum(frame["target_price_vnd"].to_numpy(dtype=float), 1.0)
            / np.maximum(frame["price_vnd"].to_numpy(dtype=float), 1.0)
        ).astype(np.float32)
        return {"drop5": frame["DROP_5PCT"].to_numpy(dtype=np.float32), "direction3": direction, "log_return": log_return}

    raw_head, raw_valid, metadata = _fit_neural(
        tf, model, list(inputs[0]), targets(frames[0]), list(inputs[1]), targets(frames[1]), list(inputs[2]),
        smoke=smoke, artifact_dir=artifact_dir,
        sample_weight={
            "drop5": _class_weights(frames[0], configuration),
            "direction3": np.ones(len(frames[0]), dtype=np.float32),
            "log_return": np.ones(len(frames[0]), dtype=np.float32),
        },
        heartbeat=heartbeat,
    )
    return raw_head["drop5"].reshape(-1), raw_valid["drop5"].reshape(-1), {"device": "GPU", "architecture": architecture, **metadata}


def _lgbm_point(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    import lightgbm as lgb

    fit, head, valid = frames
    configuration = str(job["configuration"])
    leaves = 31 if "L31" in configuration else 63
    objective = "huber" if "HUBER" in configuration else ("regression_l1" if "L1" in configuration else "regression")
    model = lgb.LGBMRegressor(
        objective=objective, n_estimators=16 if smoke else 1600,
        learning_rate=0.05 if smoke else 0.025, num_leaves=leaves, min_child_samples=100,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.5,
        random_state=int(job["seed"]), n_jobs=int(os.environ.get("SKYFARE_V24_CPU_THREADS", "6")),
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    x_fit, x_head, x_valid = (encoder.tree_matrix(frame) for frame in frames)
    model.fit(
        x_fit, fit["target_anchor_relative_log"].to_numpy(dtype=float),
        eval_set=[(x_head, head["target_anchor_relative_log"].to_numpy(dtype=float))],
        callbacks=[lgb.early_stopping(4 if smoke else 75, verbose=False), lgb.log_evaluation(0)],
    )
    return model.predict(x_head), model.predict(x_valid), {
        "device": "CPU", "objective": objective, "best_iteration": int(model.best_iteration_ or model.n_estimators)
    }


def _cat_point(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    from catboost import CatBoostRegressor

    fit, head, valid = frames
    configuration = str(job["configuration"])
    loss = "MAE" if "MAE" in configuration else "RMSE"
    model = CatBoostRegressor(
        loss_function=loss, eval_metric=loss, iterations=16 if smoke else 1400,
        depth=7 if "D7" in configuration else 8, learning_rate=0.05 if smoke else 0.03,
        l2_leaf_reg=6.0, random_seed=int(job["seed"]), task_type="GPU",
        devices=os.environ.get("SKYFARE_V24_GPU_DEVICE", "0"),
        thread_count=int(os.environ.get("SKYFARE_V24_GPU_HOST_THREADS", "4")),
        allow_writing_files=False, verbose=False, od_type="Iter", od_wait=4 if smoke else 75,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        gpu_ram_part=float(os.environ.get("SKYFARE_V24_CAT_GPU_RAM_PART", "0.42")),
    )
    x_fit, x_head, x_valid = (encoder.native_frame(frame) for frame in frames)
    model.fit(
        x_fit, fit["target_anchor_relative_log"].to_numpy(dtype=float),
        cat_features=list(REG_CATEGORICAL),
        eval_set=(x_head, head["target_anchor_relative_log"].to_numpy(dtype=float)), use_best_model=True,
    )
    return model.predict(x_head), model.predict(x_valid), {
        "device": "GPU", "loss": loss, "best_iteration": int(model.get_best_iteration()), "gpu_nondeterministic": True,
    }


def _sequence_point(
    job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool,
    artifact_dir: Path, heartbeat: Callable[[str, int, dict[str, float]], None] | None,
):
    tf = _configure_tensorflow(int(job["seed"]))
    configuration = str(job["configuration"])
    architecture = configuration.split("_")[0]
    units = 96 if "U96" in configuration else 64
    inputs = _sequence_arrays(frames, "POINT", encoder)
    model_inputs, hidden = _sequence_backbone(tf, inputs[0][0].shape[1:], inputs[0][2].shape[1], architecture, units, 0.15)
    output = tf.keras.layers.Dense(1, name="relative_log")(hidden)
    model = tf.keras.Model(model_inputs, output)
    loss = tf.keras.losses.LogCosh() if "LOGCOSH" in configuration else tf.keras.losses.Huber(delta=0.05)
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-5), loss=loss)
    target = [frame["target_anchor_relative_log"].to_numpy(dtype=np.float32) for frame in frames]
    raw_head, raw_valid, metadata = _fit_neural(
        tf, model, list(inputs[0]), target[0], list(inputs[1]), target[1], list(inputs[2]),
        smoke=smoke, artifact_dir=artifact_dir, heartbeat=heartbeat,
    )
    return raw_head.reshape(-1), raw_valid.reshape(-1), {"device": "GPU", "architecture": architecture, "loss": loss.name, **metadata}


def _xgb_quantile(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    from xgboost import XGBRegressor

    fit, head, valid = frames
    depth = int(str(job["configuration"])[1:])
    model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=np.asarray(QUANTILES),
        n_estimators=16 if smoke else 1800, learning_rate=0.05 if smoke else 0.025,
        max_depth=depth, min_child_weight=30, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.03, reg_lambda=2.0, tree_method="hist",
        device=os.environ.get("SKYFARE_V24_XGB_DEVICE", "cuda:0"), max_bin=256,
        random_state=int(job["seed"]), n_jobs=int(os.environ.get("SKYFARE_V24_GPU_HOST_THREADS", "4")),
        early_stopping_rounds=4 if smoke else 75, verbosity=0,
    )
    x_fit, x_head, x_valid = (encoder.tree_matrix(frame) for frame in frames)
    model.fit(
        x_fit, fit["target_anchor_relative_log"].to_numpy(dtype=float),
        eval_set=[(x_head, head["target_anchor_relative_log"].to_numpy(dtype=float))], verbose=False,
    )
    return rearrange(model.predict(x_head)), rearrange(model.predict(x_valid)), {
        "device": "GPU", "best_iteration": int(model.best_iteration), "depth": depth,
    }


def _cat_multi_quantile(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    from catboost import CatBoostRegressor

    fit, head, valid = frames
    alpha = ",".join(f"{level:.2f}" for level in QUANTILES)
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alpha}", eval_metric=f"MultiQuantile:alpha={alpha}",
        iterations=16 if smoke else 1400, depth=7, learning_rate=0.05 if smoke else 0.03,
        l2_leaf_reg=6.0, random_seed=int(job["seed"]), task_type="CPU",
        thread_count=int(os.environ.get("SKYFARE_V24_CPU_THREADS", "6")), allow_writing_files=False,
        verbose=False, od_type="Iter", od_wait=4 if smoke else 75,
    )
    x_fit, x_head, x_valid = (encoder.native_frame(frame) for frame in frames)
    model.fit(
        x_fit, fit["target_anchor_relative_log"].to_numpy(dtype=float), cat_features=list(REG_CATEGORICAL),
        eval_set=(x_head, head["target_anchor_relative_log"].to_numpy(dtype=float)), use_best_model=True,
    )
    return rearrange(model.predict(x_head)), rearrange(model.predict(x_valid)), {
        "device": "CPU", "best_iteration": int(model.get_best_iteration())
    }


def _cat_uncertainty(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    from catboost import CatBoostRegressor

    fit, head, valid = frames
    depth = int(str(job["configuration"])[1:])
    model = CatBoostRegressor(
        loss_function="RMSEWithUncertainty", eval_metric="RMSEWithUncertainty",
        iterations=16 if smoke else 1500, depth=depth, learning_rate=0.05 if smoke else 0.025,
        l2_leaf_reg=6.0, random_seed=int(job["seed"]), task_type="GPU",
        devices=os.environ.get("SKYFARE_V24_GPU_DEVICE", "0"),
        thread_count=int(os.environ.get("SKYFARE_V24_GPU_HOST_THREADS", "4")),
        allow_writing_files=False, verbose=False, od_type="Iter", od_wait=4 if smoke else 75,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        gpu_ram_part=float(os.environ.get("SKYFARE_V24_CAT_GPU_RAM_PART", "0.42")),
    )
    x_fit, x_head, x_valid = (encoder.native_frame(frame) for frame in frames)
    model.fit(
        x_fit, fit["target_anchor_relative_log"].to_numpy(dtype=float), cat_features=list(REG_CATEGORICAL),
        eval_set=(x_head, head["target_anchor_relative_log"].to_numpy(dtype=float)), use_best_model=True,
    )

    def predict(matrix: pd.DataFrame) -> np.ndarray:
        raw = _finite("cat uncertainty", model.predict(matrix, prediction_type="RawFormulaVal"), len(matrix))
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise RuntimeError(f"CatBoost uncertainty output changed: {raw.shape}")
        scale = np.clip(np.exp(raw[:, 1]), 1e-4, 3.0)
        return rearrange(raw[:, :1] + scale[:, None] * norm.ppf(np.asarray(QUANTILES))[None, :])

    return predict(x_head), predict(x_valid), {
        "device": "GPU", "best_iteration": int(model.get_best_iteration()), "distribution": "Normal"
    }


def _restore_score(sorted_frame: pd.DataFrame, score: np.ndarray, original: pd.DataFrame) -> np.ndarray:
    return pd.Series(np.asarray(score, dtype=float), index=sorted_frame.index).reindex(original.index).to_numpy(dtype=float)


def _lgbm_ranker(job: dict[str, object], frames: tuple[pd.DataFrame, ...], encoder: FeatureEncoder, smoke: bool):
    import lightgbm as lgb

    fit, head, valid = frames
    fit_sorted, fit_target, fit_groups = ranking_frame(fit)
    head_ranked, head_target, head_groups = ranking_frame(head)
    head_sorted = head.sort_values(["query_id", "row_key"], kind="stable")
    valid_sorted = valid.sort_values(["query_id", "row_key"], kind="stable")
    objective = "lambdarank" if job["family"] == "LGBM_LAMBDARANK" else "rank_xendcg"
    leaves = int(str(job["configuration"])[1:])
    model = lgb.LGBMRanker(
        objective=objective, n_estimators=16 if smoke else 1600,
        learning_rate=0.05 if smoke else 0.025, num_leaves=leaves, min_child_samples=80,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.5,
        random_state=int(job["seed"]), n_jobs=int(os.environ.get("SKYFARE_V24_CPU_THREADS", "6")),
        deterministic=True, force_col_wise=True, verbosity=-1, label_gain=[0, 1, 3, 7, 15],
        lambdarank_truncation_level=8,
    )
    model.fit(
        encoder.tree_matrix(fit_sorted), fit_target, group=fit_groups,
        eval_set=[(encoder.tree_matrix(head_ranked), head_target)], eval_group=[head_groups], eval_at=[3, 5],
        callbacks=[lgb.early_stopping(4 if smoke else 75, verbose=False), lgb.log_evaluation(0)],
    )
    return (
        _restore_score(head_sorted, model.predict(encoder.tree_matrix(head_sorted)), head),
        _restore_score(valid_sorted, model.predict(encoder.tree_matrix(valid_sorted)), valid),
        {"device": "CPU", "objective": objective, "leaves": leaves, "best_iteration": int(model.best_iteration_ or model.n_estimators)},
    )


def fit_predict(
    job: dict[str, object], fit: pd.DataFrame, head: pd.DataFrame, valid: pd.DataFrame, *,
    smoke: bool, artifact_dir: Path,
    heartbeat: Callable[[str, int, dict[str, float]], None] | None = None,
):
    started = time.monotonic()
    if heartbeat:
        heartbeat("FIT_STARTED", 0, {})
    encoder = FeatureEncoder.fit(fit, "CLASSIFICATION" if job["task"] == "CLASSIFICATION" else "REGRESSION")
    frames = (fit, head, valid)
    family = str(job["family"])
    if family == "LGBM_CLASS":
        result = _lgbm_class(job, frames, encoder, smoke)
    elif family == "CAT_CLASS":
        result = _cat_class(job, frames, encoder, smoke)
    elif family == "FT_CLASS":
        result = _ft_class(job, frames, encoder, smoke, artifact_dir, heartbeat)
    elif family == "SEQ_CLASS":
        result = _sequence_class(job, frames, encoder, smoke, artifact_dir, heartbeat)
    elif family == "LGBM_POINT":
        result = _lgbm_point(job, frames, encoder, smoke)
    elif family == "CAT_POINT":
        result = _cat_point(job, frames, encoder, smoke)
    elif family == "SEQ_POINT":
        result = _sequence_point(job, frames, encoder, smoke, artifact_dir, heartbeat)
    elif family == "XGB_QUANTILE":
        result = _xgb_quantile(job, frames, encoder, smoke)
    elif family == "CAT_MULTIQUANTILE":
        result = _cat_multi_quantile(job, frames, encoder, smoke)
    elif family == "CAT_UNCERTAINTY":
        result = _cat_uncertainty(job, frames, encoder, smoke)
    elif family in {"LGBM_LAMBDARANK", "LGBM_XENDCG"}:
        result = _lgbm_ranker(job, frames, encoder, smoke)
    else:
        raise KeyError(family)
    head_prediction, valid_prediction, metadata = result
    elapsed = time.monotonic() - started
    if heartbeat:
        heartbeat("FIT_COMPLETE", 1, {"fit_seconds": elapsed})
    return head_prediction, valid_prediction, {
        **metadata, "encoder": encoder.metadata(), "fit_seconds": elapsed,
    }
