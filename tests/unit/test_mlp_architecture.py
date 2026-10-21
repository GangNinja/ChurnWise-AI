"""Model construction, compile configuration and validation (Phase 6, §2-3)."""

from __future__ import annotations

import numpy as np
import pytest

from churnwise.models.contracts import MLPConfig, coerce_dropout
from churnwise.models.errors import CODE_INVALID_MODEL, ModelingError
from churnwise.models.mlp import (
    build_mlp,
    learning_rate_of,
    parameter_count,
    set_global_seed,
    validate_config,
)


def test_default_architecture_is_small_relu_mlp_with_sigmoid_output():
    config = MLPConfig(hidden_units=(16, 8), dropout=(0.1, 0.2), l2=1e-3)
    model = build_mlp(6, config)

    assert model.input_shape == (None, 6)
    assert model.output_shape == (None, 1)
    hidden = [layer for layer in model.layers if layer.name.startswith("hidden_")]
    assert [layer.units for layer in hidden] == [16, 8]
    assert all(layer.activation.__name__ == "relu" for layer in hidden)
    assert all(layer.kernel_initializer.__class__.__name__.startswith("He") for layer in hidden)
    output = model.get_layer("churn_probability")
    assert output.units == 1
    assert output.activation.__name__ == "sigmoid"
    assert parameter_count(model) > 0


def test_compile_uses_binary_crossentropy_and_adam_with_configured_rate():
    model = build_mlp(3, MLPConfig(learning_rate=0.002, hidden_units=(4,), dropout=(0.0,)))
    assert model.loss.__class__.__name__ == "BinaryCrossentropy"
    assert model.optimizer.__class__.__name__ == "Adam"
    assert learning_rate_of(model) == pytest.approx(0.002)
    model.train_on_batch(
        np.zeros((4, 3), dtype=np.float32),
        np.array([0, 1, 0, 1], dtype=np.float32),
    )
    assert "accuracy" in model.get_metrics_result()


def test_dropout_and_l2_are_configurable_and_optional():
    with_regularisation = build_mlp(
        5, MLPConfig(hidden_units=(8,), dropout=(0.25,), l2=1e-2)
    )
    assert any(layer.__class__.__name__ == "Dropout" for layer in with_regularisation.layers)
    dense = with_regularisation.get_layer("hidden_1")
    assert float(dense.kernel_regularizer.l2) == pytest.approx(1e-2)

    without = build_mlp(5, MLPConfig(hidden_units=(8,), dropout=(0.0,), l2=0.0))
    assert not any(layer.__class__.__name__ == "Dropout" for layer in without.layers)
    assert without.get_layer("hidden_1").kernel_regularizer is None


@pytest.mark.parametrize(
    "bad_config",
    [
        MLPConfig(hidden_units=(), dropout=()),
        MLPConfig(hidden_units=(8, 4), dropout=(0.2,)),
        MLPConfig(dropout=(1.5,), hidden_units=(8,)),
        MLPConfig(learning_rate=0.0),
        MLPConfig(batch_size=0),
        MLPConfig(epochs=0),
        MLPConfig(threshold=1.5),
        MLPConfig(seed=-1),
        MLPConfig(min_learning_rate=2.0, learning_rate=1.0),
    ],
)
def test_invalid_configurations_are_rejected_with_actionable_code(bad_config):
    with pytest.raises(ModelingError) as caught:
        validate_config(bad_config)
    assert caught.value.code == CODE_INVALID_MODEL
    assert caught.value.remediation


@pytest.mark.parametrize(
    "malformed",
    [
        MLPConfig(hidden_units=None, dropout=None),  # type: ignore[arg-type]
        MLPConfig(hidden_units=(8,), dropout=None),  # type: ignore[arg-type]
        MLPConfig(hidden_units=(8,), dropout=("bad",)),  # type: ignore[arg-type]
        MLPConfig(hidden_units=(True,), dropout=(0.2,)),  # type: ignore[arg-type]
    ],
)
def test_malformed_container_configurations_raise_modeling_errors(malformed):
    with pytest.raises(ModelingError) as caught:
        validate_config(malformed)
    assert caught.value.code == CODE_INVALID_MODEL


@pytest.mark.parametrize("input_dim", [0, -1, 2.5, "5"])
def test_invalid_input_dimension_is_rejected(input_dim):
    with pytest.raises(ModelingError) as caught:
        build_mlp(input_dim, MLPConfig())
    assert caught.value.code == CODE_INVALID_MODEL


def test_wrong_input_width_fails_loudly():
    model = build_mlp(4, MLPConfig(hidden_units=(4,), dropout=(0.0,)))
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3), dtype=np.float32), verbose=0)


def test_coerce_dropout_expands_single_rate():
    assert coerce_dropout([0.2], [64, 32]) == (0.2, 0.2)
    assert coerce_dropout([0.1, 0.3], [64, 32]) == (0.1, 0.3)


def test_seed_utility_is_repeatable_for_numpy():
    set_global_seed(123)
    first = np.random.random(4)
    set_global_seed(123)
    second = np.random.random(4)
    assert np.array_equal(first, second)
