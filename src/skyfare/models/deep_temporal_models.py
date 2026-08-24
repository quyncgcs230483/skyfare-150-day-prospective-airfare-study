"""V19 dual-scale GRU models and multitask objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def primary_sample_weight(sample_weight: Any) -> Any:
    if sample_weight is None:
        return None
    if isinstance(sample_weight, Mapping):
        if "drop5" in sample_weight:
            return sample_weight["drop5"]
        return sample_weight.get("ordinal")
    return sample_weight


def _masked_gru(tf: Any, sequence: Any, mask: Any, name: str) -> Any:
    # cuDNN rejects batches containing an all-False mask. Give empty histories
    # one temporary zero timestep, then gate their projected embedding to zero.
    safe_mask = tf.keras.layers.Lambda(
        lambda value: tf.where(
            tf.reduce_any(value, axis=1, keepdims=True),
            value,
            tf.concat(
                [
                    tf.ones_like(value[:, :1], dtype=tf.bool),
                    tf.zeros_like(value[:, 1:], dtype=tf.bool),
                ],
                axis=1,
            ),
        ),
        output_shape=(mask.shape[-1],),
        name=f"{name}_cudnn_safe_mask",
    )(mask)
    encoded = tf.keras.layers.GRU(48, name=f"{name}_gru")(sequence, mask=safe_mask)
    projected = tf.keras.layers.Dense(
        48, activation="relu", name=f"{name}_projection"
    )(encoded)
    has_history = tf.keras.layers.Lambda(
        lambda value: tf.cast(
            tf.reduce_any(value, axis=1, keepdims=True), tf.float32
        ),
        output_shape=(1,),
        name=f"{name}_has_history",
    )(mask)
    return tf.keras.layers.Multiply(name=f"{name}_empty_history_zero")(
        [projected, has_history]
    )


def build_backbone(
    tf: Any,
    *,
    exact_shape: tuple[int, int],
    template_shape: tuple[int, int],
    context_dim: int,
    sequence_mode: str,
):
    exact = tf.keras.Input(shape=exact_shape, name="exact_sequence")
    exact_mask = tf.keras.Input(shape=(exact_shape[0],), dtype="bool", name="exact_mask")
    template = tf.keras.Input(shape=template_shape, name="template_sequence")
    template_mask = tf.keras.Input(
        shape=(template_shape[0],), dtype="bool", name="template_mask"
    )
    context = tf.keras.Input(shape=(context_dim,), name="context")
    exact_encoded = _masked_gru(tf, exact, exact_mask, "exact")
    template_encoded = _masked_gru(tf, template, template_mask, "template")
    context_encoded = tf.keras.layers.Dense(
        48, activation="relu", name="context_projection"
    )(context)
    if sequence_mode == "EXACT":
        zero_template = tf.keras.layers.Lambda(
            lambda value: value * 0.0, name="template_zero_connection"
        )(template_encoded)
        temporal = tf.keras.layers.Add(name="exact_only")(
            [exact_encoded, zero_template]
        )
        fused = tf.keras.layers.Concatenate(name="exact_context")(
            [temporal, context_encoded]
        )
    elif sequence_mode == "TEMPLATE":
        zero_exact = tf.keras.layers.Lambda(
            lambda value: value * 0.0, name="exact_zero_connection"
        )(exact_encoded)
        temporal = tf.keras.layers.Add(name="template_only")(
            [template_encoded, zero_exact]
        )
        fused = tf.keras.layers.Concatenate(name="template_context")(
            [temporal, context_encoded]
        )
    elif sequence_mode == "DUAL_GATED":
        gate_source = tf.keras.layers.Concatenate(name="gate_source")(
            [exact_encoded, template_encoded, context_encoded]
        )
        gate = tf.keras.layers.Dense(48, activation="sigmoid", name="dual_gate")(
            gate_source
        )
        inverse_gate = tf.keras.layers.Lambda(
            lambda value: 1.0 - value, name="dual_inverse_gate"
        )(gate)
        exact_part = tf.keras.layers.Multiply(name="dual_exact_part")(
            [gate, exact_encoded]
        )
        template_part = tf.keras.layers.Multiply(name="dual_template_part")(
            [inverse_gate, template_encoded]
        )
        temporal = tf.keras.layers.Add(name="dual_temporal")(
            [exact_part, template_part]
        )
        fused = tf.keras.layers.Concatenate(name="dual_context")(
            [temporal, context_encoded]
        )
    elif sequence_mode == "DUAL_ATTENTION":
        tokens = tf.keras.layers.Lambda(
            lambda values: tf.stack(values, axis=1), name="fusion_tokens"
        )([exact_encoded, template_encoded, context_encoded])
        attended = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=24, dropout=0.10, name="fusion_attention"
        )(tokens, tokens)
        attended = tf.keras.layers.Add(name="fusion_residual")([tokens, attended])
        attended = tf.keras.layers.LayerNormalization(name="fusion_norm")(attended)
        fused = tf.keras.layers.GlobalAveragePooling1D(name="fusion_pool")(attended)
    else:
        raise KeyError(sequence_mode)
    fused = tf.keras.layers.Dense(64, activation="relu", name="shared_dense")(fused)
    fused = tf.keras.layers.Dropout(0.15, name="shared_dropout")(fused)
    return tf.keras.Model(
        [exact, exact_mask, template, template_mask, context],
        fused,
        name=f"v19_{sequence_mode.lower()}_backbone",
    )


def build_classification_model(
    tf: Any,
    *,
    exact_shape: tuple[int, int],
    template_shape: tuple[int, int],
    context_dim: int,
    sequence_mode: str,
    objective: str,
):
    backbone = build_backbone(
        tf,
        exact_shape=exact_shape,
        template_shape=template_shape,
        context_dim=context_dim,
        sequence_mode=sequence_mode,
    )

    class ResearchClassifier(tf.keras.Model):
        def __init__(self):
            super().__init__(name=f"v19_{sequence_mode.lower()}_{objective.lower()}")
            self.backbone = backbone
            self.objective = objective
            self.drop_head = tf.keras.layers.Dense(1, name="head_drop5")
            self.direction_head = None
            self.return_head = None
            self.ordinal_base = None
            self.ordinal_decrements = None
            if objective.startswith("MTL_"):
                self.direction_head = tf.keras.layers.Dense(3, name="head_direction3")
                self.return_head = tf.keras.layers.Dense(1, name="head_log_return")
                self.loss_tasks = ("drop5", "direction3", "log_return")
            elif objective == "ORDINAL_UNCERTAINTY":
                self.ordinal_base = tf.keras.layers.Dense(1, name="head_ordinal_base")
                self.ordinal_decrements = tf.keras.layers.Dense(
                    3, name="head_ordinal_decrements"
                )
                self.return_head = tf.keras.layers.Dense(1, name="head_log_return")
                self.loss_tasks = ("ordinal", "log_return")
            else:
                self.loss_tasks = ("drop5",)
            self.fixed_weights = {
                "drop5": 1.0,
                "direction3": 0.30,
                "log_return": 0.20,
                "ordinal": 1.0,
            }
            self.log_vars = {}
            if objective in {"MTL_UNCERTAINTY", "ORDINAL_UNCERTAINTY"}:
                self.log_vars = {
                    task: self.add_weight(
                        name=f"log_var_{task}",
                        shape=(),
                        initializer="zeros",
                        trainable=True,
                    )
                    for task in self.loss_tasks
                }
            self.gradnorm_weights = {}
            self.initial_losses = {}
            self.gradnorm_initialized = None
            if objective == "MTL_GRADNORM":
                self.gradnorm_weights = {
                    task: self.add_weight(
                        name=f"gradnorm_weight_{task}",
                        shape=(),
                        initializer="ones",
                        trainable=False,
                    )
                    for task in self.loss_tasks
                }
                self.initial_losses = {
                    task: self.add_weight(
                        name=f"initial_loss_{task}",
                        shape=(),
                        initializer="ones",
                        trainable=False,
                    )
                    for task in self.loss_tasks
                }
                self.gradnorm_initialized = self.add_weight(
                    name="gradnorm_initialized",
                    shape=(),
                    initializer="zeros",
                    trainable=False,
                )
            metric_names = set(self.loss_tasks) | {"drop5"}
            self.total_tracker = tf.keras.metrics.Mean(name="loss")
            self.task_trackers = {
                task: tf.keras.metrics.Mean(name=f"{task}_loss")
                for task in sorted(metric_names)
            }
            self.gradnorm_tracker = (
                tf.keras.metrics.Mean(name="gradnorm_imbalance")
                if objective == "MTL_GRADNORM"
                else None
            )

        @property
        def metrics(self):
            result = [self.total_tracker, *self.task_trackers.values()]
            if self.gradnorm_tracker is not None:
                result.append(self.gradnorm_tracker)
            return result

        def call(self, inputs, training=False):
            shared = self.backbone(inputs, training=training)
            if self.objective == "ORDINAL_UNCERTAINTY":
                base = self.ordinal_base(shared)
                decrements = tf.nn.softplus(self.ordinal_decrements(shared))
                cumulative = tf.cumsum(decrements, axis=1)
                ordinal = tf.concat([base, base - cumulative], axis=1)
                return {
                    "drop5": ordinal[:, 3:4],
                    "ordinal": ordinal,
                    "log_return": self.return_head(shared),
                }
            result = {"drop5": self.drop_head(shared)}
            if self.direction_head is not None:
                result["direction3"] = self.direction_head(shared)
                result["log_return"] = self.return_head(shared)
            return result

        @staticmethod
        def _dense_gradient(gradient, variable):
            if gradient is None:
                return tf.zeros_like(variable)
            if isinstance(gradient, tf.IndexedSlices):
                return tf.convert_to_tensor(gradient)
            return gradient

        def _apply_row_weight(self, loss, sample_weight):
            weight = primary_sample_weight(sample_weight)
            if weight is None:
                return tf.reduce_mean(loss)
            weight = tf.cast(tf.reshape(weight, (-1,)), tf.float32)
            row_loss = tf.reshape(loss, (tf.shape(loss)[0], -1))
            row_loss = tf.reduce_mean(row_loss, axis=1)
            return tf.reduce_sum(row_loss * weight) / tf.maximum(
                tf.reduce_sum(weight), 1.0
            )

        def _compute_losses(self, targets, outputs, sample_weight):
            drop = tf.keras.losses.binary_crossentropy(
                tf.cast(targets["drop5"], tf.float32),
                outputs["drop5"],
                from_logits=True,
            )
            losses = {"drop5": self._apply_row_weight(drop, sample_weight)}
            if "direction3" in outputs:
                losses["direction3"] = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        tf.cast(targets["direction3"], tf.int32),
                        outputs["direction3"],
                        from_logits=True,
                    )
                )
                losses["log_return"] = tf.reduce_mean(
                    tf.keras.losses.huber(
                        tf.cast(targets["log_return"], tf.float32),
                        outputs["log_return"],
                        delta=1.0,
                    )
                )
            if "ordinal" in outputs:
                ordinal = tf.keras.losses.binary_crossentropy(
                    tf.cast(targets["ordinal"], tf.float32),
                    outputs["ordinal"],
                    from_logits=True,
                )
                losses["ordinal"] = self._apply_row_weight(ordinal, sample_weight)
                losses["log_return"] = tf.reduce_mean(
                    tf.keras.losses.huber(
                        tf.cast(targets["log_return"], tf.float32),
                        outputs["log_return"],
                        delta=1.0,
                    )
                )
            return losses

        def _total(self, losses):
            if self.log_vars:
                return tf.add_n(
                    [
                        tf.exp(-self.log_vars[task]) * losses[task]
                        + self.log_vars[task]
                        for task in self.loss_tasks
                    ]
                )
            if self.gradnorm_weights:
                return tf.add_n(
                    [self.gradnorm_weights[task] * losses[task] for task in self.loss_tasks]
                )
            return tf.add_n(
                [self.fixed_weights[task] * losses[task] for task in self.loss_tasks]
            )

        def _update_gradnorm(self, tape, losses):
            shared_variables = self.backbone.trainable_variables
            base_norms = {}
            for task in self.loss_tasks:
                gradients = tape.gradient(losses[task], shared_variables)
                dense = [
                    self._dense_gradient(gradient, variable)
                    for gradient, variable in zip(
                        gradients, shared_variables, strict=True
                    )
                ]
                base_norms[task] = tf.stop_gradient(tf.linalg.global_norm(dense))
            def initialize_losses():
                for task in self.loss_tasks:
                    self.initial_losses[task].assign(
                        tf.maximum(tf.stop_gradient(losses[task]), 1e-8)
                    )
                self.gradnorm_initialized.assign(1.0)
                return tf.constant(0.0)

            tf.cond(
                tf.equal(self.gradnorm_initialized, 0.0),
                initialize_losses,
                lambda: tf.constant(0.0),
            )
            rates = tf.stack(
                [
                    tf.stop_gradient(losses[task]) / self.initial_losses[task]
                    for task in self.loss_tasks
                ]
            )
            inverse_rates = rates / tf.maximum(tf.reduce_mean(rates), 1e-8)
            norms = tf.stack(
                [
                    self.gradnorm_weights[task] * base_norms[task]
                    for task in self.loss_tasks
                ]
            )
            target = tf.stop_gradient(
                tf.reduce_mean(norms) * tf.pow(inverse_rates, 1.5)
            )
            learning_rate = tf.constant(0.01, dtype=tf.float32)
            for index, task in enumerate(self.loss_tasks):
                gradient = tf.sign(norms[index] - target[index]) * base_norms[task]
                self.gradnorm_weights[task].assign(
                    tf.maximum(self.gradnorm_weights[task] - learning_rate * gradient, 0.05)
                )
            total_weight = tf.add_n(list(self.gradnorm_weights.values()))
            scale = tf.cast(len(self.loss_tasks), tf.float32) / tf.maximum(total_weight, 1e-8)
            for task in self.loss_tasks:
                self.gradnorm_weights[task].assign(self.gradnorm_weights[task] * scale)
            imbalance = tf.reduce_mean(tf.abs(norms - target))
            self.gradnorm_tracker.update_state(imbalance)

        def train_step(self, data):
            inputs, targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            persistent = self.objective == "MTL_GRADNORM"
            with tf.GradientTape(persistent=persistent) as tape:
                outputs = self(inputs, training=True)
                losses = self._compute_losses(targets, outputs, sample_weight)
                total = self._total(losses)
            gradients = tape.gradient(total, self.trainable_variables)
            pairs = [
                (self._dense_gradient(gradient, variable), variable)
                for gradient, variable in zip(
                    gradients, self.trainable_variables, strict=True
                )
            ]
            self.optimizer.apply_gradients(pairs)
            if persistent:
                self._update_gradnorm(tape, losses)
                del tape
            self.total_tracker.update_state(total)
            for task, tracker in self.task_trackers.items():
                tracker.update_state(losses[task])
            return {metric.name: metric.result() for metric in self.metrics}

        def test_step(self, data):
            inputs, targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            outputs = self(inputs, training=False)
            losses = self._compute_losses(targets, outputs, sample_weight)
            total = self._total(losses)
            self.total_tracker.update_state(total)
            for task, tracker in self.task_trackers.items():
                tracker.update_state(losses[task])
            return {metric.name: metric.result() for metric in self.metrics}

    return ResearchClassifier()


def quantile_loss(tf: Any, quantiles=(0.10, 0.50, 0.90)):
    quantile_tensor = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        target = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        error = target - tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(
            tf.maximum(quantile_tensor * error, (quantile_tensor - 1.0) * error),
            axis=1,
        )

    loss.__name__ = "pinball_q10_q50_q90"
    return loss


def build_regression_model(
    tf: Any,
    *,
    exact_shape: tuple[int, int],
    template_shape: tuple[int, int],
    context_dim: int,
    sequence_mode: str,
    objective: str,
):
    backbone = build_backbone(
        tf,
        exact_shape=exact_shape,
        template_shape=template_shape,
        context_dim=context_dim,
        sequence_mode=sequence_mode,
    )
    units = 3 if objective == "QUANTILE" else 1
    output = tf.keras.layers.Dense(units, name="head_regression")(backbone.output)
    return tf.keras.Model(
        backbone.inputs,
        output,
        name=f"v19_{sequence_mode.lower()}_{objective.lower()}",
    )
