"""MLP explainer integration (Phase 8) on a small deterministic Keras model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("shap")

from churnwise.explain import ExplainConfig, build_explainer, chunked_explain
from churnwise.models.contracts import MLPConfig
from churnwise.models.mlp import build_mlp, set_global_seed

FEATURE_NAMES = [f"numeric__f{index}" for index in range(4)]


@pytest.fixture(scope="module")
def tiny_mlp():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(120, 4))
    labels = (features[:, 0] + 0.5 * features[:, 1] - 0.4 * features[:, 2] > 0).astype(int)
    set_global_seed(3)
    model = build_mlp(
        4,
        MLPConfig(
            hidden_units=(8,),
            dropout=(0.0,),
            epochs=1,
            batch_size=32,
            verbose=0,
        ),
    )
    model.fit(
        features[:80],
        labels[:80].astype("float32"),
        epochs=1,
        batch_size=32,
        verbose=0,
    )
    return model, features, labels


def _predict(model):
    return lambda matrix: np.asarray(model.predict(matrix, verbose=0), dtype=float).reshape(-1)


def test_mlp_explanations_are_additive_and_probability_space(tiny_mlp) -> None:
    model, features, _ = tiny_mlp
    config = ExplainConfig(
        mlp_sample_size=8,
        mlp_background_size=12,
        chunk_size=4,
        tolerance=1e-4,
    )
    explainer = build_explainer(
        "mlp",
        model,
        _predict(model),
        background=features[:12],
        feature_names=FEATURE_NAMES,
        config=config,
    )
    assert explainer.output_space == "probability"
    assert explainer.explainer_name in {"shap.DeepExplainer", "shap.Explainer(permutation)"}
    explanation = chunked_explain(explainer, features[40:48], config=config)
    assert explanation.values.shape == (8, 4)
    assert explanation.additivity_error <= config.tolerance
    assert np.allclose(
        explanation.predictions,
        np.asarray(model.predict(features[40:48], verbose=0)).reshape(-1),
        atol=1e-6,
    )
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    assert np.allclose(reconstructed, explanation.predictions, atol=config.tolerance)


def test_mlp_explainer_falls_back_honestly_when_deep_fails(monkeypatch, tiny_mlp) -> None:
    import shap

    model, features, _ = tiny_mlp

    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("deep explainer unsupported in this build")

    monkeypatch.setattr(shap, "DeepExplainer", Broken)
    config = ExplainConfig(mlp_sample_size=2, mlp_background_size=8, permutation_max_evals=2 * 4 + 1)
    explainer = build_explainer(
        "mlp",
        model,
        _predict(model),
        background=features[:8],
        feature_names=FEATURE_NAMES,
        config=config,
    )
    assert explainer.selected_by_fallback is True
    assert "DeepExplainer" in (explainer.fallback_reason or "")
    explanation = chunked_explain(explainer, features[40:42], config=config)
    assert explanation.additivity_error <= config.tolerance
    assert explainer.warnings
