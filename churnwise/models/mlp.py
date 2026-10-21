"""TensorFlow/Keras MLP construction for binary churn prediction.

The network is deliberately small for a tabular churn dataset:

    input (d) -> Dense(64, ReLU) -> Dropout(0.30)
              -> Dense(32, ReLU) -> Dropout(0.20)
              -> Dense(1, sigmoid) = P(churn)

Binary cross-entropy is the loss, Adam is the optimizer, and Keras
autodifferentiation supplies the gradients for backpropagation.
"""

from __future__ import annotations

import os
import random
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from churnwise.models.contracts import MLPConfig
from churnwise.models.errors import CODE_INVALID_MODEL, ModelingError


def validate_config(config: MLPConfig) -> None:
    problems = config.validate()
    if problems:
        raise ModelingError(
            CODE_INVALID_MODEL,
            "invalid MLP configuration: " + "; ".join(problems),
            "Correct the architecture or training options and try again.",
        )


def set_global_seed(seed: int, *, deterministic: bool = False) -> list[str]:
    """Seed Python, NumPy and TensorFlow; optionally enable op determinism."""
    warnings: list[str] = []
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - platform dependent
            warnings.append(f"TensorFlow op determinism could not be enabled: {exc}")
    return warnings


def build_mlp(input_dim: int, config: MLPConfig) -> tf.keras.Model:
    """Construct and compile the churn MLP for ``input_dim`` transformed features."""
    validate_config(config)
    if not isinstance(input_dim, int) or input_dim < 1:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"input_dim must be a positive integer, got {input_dim!r}",
            "Use the transformed feature count from the Phase 5 dataset.",
        )

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    hidden = inputs
    for index, (units, dropout_rate) in enumerate(
        zip(config.hidden_units, config.dropout), start=1
    ):
        hidden = tf.keras.layers.Dense(
            int(units),
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=(
                tf.keras.regularizers.l2(float(config.l2)) if config.l2 > 0 else None
            ),
            name=f"hidden_{index}",
        )(hidden)
        if dropout_rate > 0:
            hidden = tf.keras.layers.Dropout(
                float(dropout_rate), name=f"dropout_{index}"
            )(hidden)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="churn_probability"
    )(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="churn_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config.learning_rate)),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def architecture_summary(model: tf.keras.Model) -> list[str]:
    """Return a compact layer summary without printing to stdout."""
    lines: list[str] = []
    model.summary(print_fn=lines.append, line_length=88)
    return lines


def parameter_count(model: tf.keras.Model) -> int:
    return int(model.count_params())


def learning_rate_of(model: tf.keras.Model) -> float:
    value = model.optimizer.learning_rate
    if hasattr(value, "numpy"):
        return float(value.numpy())
    return float(tf.keras.backend.get_value(value))


__all__ = [
    "architecture_summary",
    "build_mlp",
    "learning_rate_of",
    "parameter_count",
    "set_global_seed",
    "validate_config",
]
