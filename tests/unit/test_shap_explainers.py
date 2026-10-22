"""SHAP explainer adapters for synthetic LR and RF models (Phase 8)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from churnwise.explain import (
    CODE_UNSUPPORTED_MODEL,
    CODE_VALIDATION_FAILED,
    ExplainConfig,
    ExplainError,
    build_explainer,
    chunked_explain,
)

FEATURE_NAMES = ["numeric__a", "numeric__b", "numeric__c"]


@pytest.fixture(scope="module")
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(160, 3))
    logit = 1.2 * features[:, 0] - 0.7 * features[:, 1] + 0.4 * features[:, 2]
    labels = (logit + rng.normal(scale=0.4, size=len(features)) > 0).astype(int)
    return features, labels


@pytest.fixture(scope="module")
def config() -> ExplainConfig:
    return ExplainConfig(sample_size=20, background_size=25, chunk_size=7)


@pytest.fixture(scope="module")
def logistic_model(classification_data):
    from sklearn.linear_model import LogisticRegression

    features, labels = classification_data
    return LogisticRegression(max_iter=500).fit(features, labels)


@pytest.fixture(scope="module")
def forest_model(classification_data):
    from sklearn.ensemble import RandomForestClassifier

    features, labels = classification_data
    return RandomForestClassifier(n_estimators=25, random_state=4).fit(features, labels)


def test_logistic_explainer_reports_log_odds_and_additivity(
    logistic_model, classification_data, config
) -> None:
    features, _ = classification_data
    background = features[:25]
    explainer = build_explainer(
        "logistic_regression",
        logistic_model,
        lambda matrix: logistic_model.predict_proba(matrix)[:, 1],
        background=background,
        feature_names=FEATURE_NAMES,
        config=config,
    )
    assert explainer.explainer_name == "shap.LinearExplainer"
    assert explainer.output_space == "log_odds"
    explanation = chunked_explain(explainer, features[60:80], config=config)
    assert explanation.values.shape == (20, 3)
    assert explanation.output_space == "log_odds"
    assert explanation.additivity_error <= config.tolerance
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    expected = logistic_model.decision_function(features[60:80])
    assert np.allclose(reconstructed, expected, atol=1e-8)
    assert np.allclose(explanation.predictions, logistic_model.predict_proba(features[60:80])[:, 1])


def test_forest_explainer_extracts_positive_class_and_probability_space(
    forest_model, classification_data, config
) -> None:
    features, _ = classification_data
    explainer = build_explainer(
        "random_forest",
        forest_model,
        lambda matrix: forest_model.predict_proba(matrix)[:, 1],
        background=features[:25],
        feature_names=FEATURE_NAMES,
        config=config,
    )
    assert explainer.explainer_name == "shap.TreeExplainer"
    assert explainer.output_space == "probability"
    explanation = chunked_explain(explainer, features[60:80], config=config)
    assert explanation.values.shape == (20, 3)
    assert explanation.additivity_error <= config.tolerance
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    assert np.allclose(reconstructed, explanation.predictions, atol=1e-8)


def test_chunking_preserves_values_and_validation(forest_model, classification_data, config) -> None:
    features, _ = classification_data
    explainer = build_explainer(
        "random_forest",
        forest_model,
        lambda matrix: forest_model.predict_proba(matrix)[:, 1],
        background=features[:25],
        feature_names=FEATURE_NAMES,
        config=config,
    )
    chunked = chunked_explain(explainer, features[60:80], config=config)
    single = explainer.explain(features[60:80], tolerance=config.tolerance)
    assert np.allclose(chunked.values, single.values)


def test_legacy_list_shap_outputs_are_normalised() -> None:
    from churnwise.explain.explainers import _positive_class

    values = [np.zeros((2, 3)), np.ones((2, 3))]
    array, bases = _positive_class(values, [0.4, 0.6])
    assert array.shape == (2, 3)
    assert np.all(array == 1.0)
    assert np.all(bases == 0.6)

    single, single_base = _positive_class([np.full((2, 3), 2.0)], [0.5])
    assert single.shape == (2, 3)
    assert np.all(single == 2.0)
    assert np.all(single_base == 0.5)


def test_tree_explainer_initialization_failure_is_actionable(
    monkeypatch, forest_model, classification_data, config
) -> None:
    import shap

    features, _ = classification_data

    def boom(*args, **kwargs):
        raise RuntimeError("tree explainer unsupported")

    monkeypatch.setattr(shap, "TreeExplainer", boom)
    with pytest.raises(ExplainError) as raised:
        build_explainer(
            "random_forest",
            forest_model,
            lambda matrix: forest_model.predict_proba(matrix)[:, 1],
            background=features[:10],
            feature_names=FEATURE_NAMES,
            config=config,
        )
    assert raised.value.code == CODE_UNSUPPORTED_MODEL
    assert "could not initialise" in raised.value.message


def test_linear_explainer_initialization_failure_is_actionable(
    monkeypatch, logistic_model, classification_data, config
) -> None:
    import shap

    features, _ = classification_data

    def boom(*args, **kwargs):
        raise RuntimeError("linear explainer unsupported")

    monkeypatch.setattr(shap, "LinearExplainer", boom)
    with pytest.raises(ExplainError) as raised:
        build_explainer(
            "logistic_regression",
            logistic_model,
            lambda matrix: logistic_model.predict_proba(matrix)[:, 1],
            background=features[:10],
            feature_names=FEATURE_NAMES,
            config=config,
        )
    assert raised.value.code == CODE_UNSUPPORTED_MODEL
    assert "could not initialise" in raised.value.message


def test_unsupported_family_is_rejected(logistic_model, classification_data, config) -> None:
    features, _ = classification_data
    with pytest.raises(ExplainError) as raised:
        build_explainer(
            "gradient_boosting",
            logistic_model,
            lambda matrix: np.zeros(len(matrix)),
            background=features[:10],
            feature_names=FEATURE_NAMES,
            config=config,
        )
    assert raised.value.code == CODE_UNSUPPORTED_MODEL
    assert "logistic_regression" in raised.value.remediation


def test_wrong_feature_width_is_rejected(logistic_model, classification_data, config) -> None:
    features, _ = classification_data
    with pytest.raises(ExplainError) as raised:
        build_explainer(
            "logistic_regression",
            logistic_model,
            lambda matrix: logistic_model.predict_proba(matrix[:, :3])[:, 1],
            background=features[:10, :2],
            feature_names=FEATURE_NAMES,
            config=config,
        )
    assert raised.value.code == CODE_VALIDATION_FAILED


def test_prediction_failure_is_actionable(logistic_model, classification_data, config) -> None:
    features, _ = classification_data

    def broken_predict(matrix: np.ndarray) -> np.ndarray:
        raise RuntimeError("prediction backend unavailable")

    explainer = build_explainer(
        "logistic_regression",
        logistic_model,
        broken_predict,
        background=features[:10],
        feature_names=FEATURE_NAMES,
        config=config,
    )
    with pytest.raises(ExplainError) as raised:
        explainer.explain(features[:2], tolerance=config.tolerance)
    assert raised.value.code == "CW-XAI-003"
    assert "prediction failed" in raised.value.message


def test_non_finite_shap_values_are_rejected(logistic_model, classification_data, config) -> None:
    features, _ = classification_data
    explainer = build_explainer(
        "logistic_regression",
        logistic_model,
        lambda matrix: logistic_model.predict_proba(matrix)[:, 1],
        background=features[:10],
        feature_names=FEATURE_NAMES,
        config=config,
    )

    def nan_draw(matrix: np.ndarray):
        return np.full((len(matrix), 3), np.nan), np.zeros(len(matrix))

    explainer._draw = nan_draw  # type: ignore[assignment]
    with pytest.raises(ExplainError) as raised:
        explainer.explain(features[:2], tolerance=config.tolerance)
    assert raised.value.code == "CW-XAI-004"
    assert "non-finite" in raised.value.message


def test_out_of_range_probabilities_are_rejected(logistic_model, classification_data, config) -> None:
    features, _ = classification_data
    explainer = build_explainer(
        "logistic_regression",
        logistic_model,
        lambda matrix: np.full(len(matrix), 1.5),
        background=features[:10],
        feature_names=FEATURE_NAMES,
        config=config,
    )
    with pytest.raises(ExplainError) as raised:
        explainer.explain(features[:2], tolerance=config.tolerance)
    assert raised.value.code == "CW-XAI-004"
    assert "outside [0, 1]" in raised.value.message


def test_additivity_failure_raises_instead_of_returning_values(logistic_model) -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(10, 3))
    explainer = build_explainer(
        "logistic_regression",
        logistic_model,
        lambda matrix: logistic_model.predict_proba(matrix)[:, 1],
        background=features,
        feature_names=FEATURE_NAMES,
        config=ExplainConfig(background_size=10),
    )

    def broken_draw(matrix: np.ndarray):
        return np.full((len(matrix), 3), 5.0), np.zeros(len(matrix))

    explainer._draw = broken_draw  # type: ignore[assignment]
    with pytest.raises(ExplainError) as raised:
        explainer.explain(features[:2], tolerance=1e-6)
    assert raised.value.code == "CW-XAI-004"
    assert "additivity" in raised.value.message


def test_mlp_without_working_explainers_fails_honestly(monkeypatch, classification_data) -> None:
    import shap

    features, _ = classification_data

    def boom(*args, **kwargs):
        raise RuntimeError("explainer unavailable")

    monkeypatch.setattr(shap, "DeepExplainer", boom)
    monkeypatch.setattr(shap, "Explainer", boom)
    with pytest.raises(ExplainError) as raised:
        build_explainer(
            "mlp",
            object(),
            lambda matrix: np.full(len(matrix), 0.5),
            background=features[:10],
            feature_names=FEATURE_NAMES,
            config=ExplainConfig(mlp_background_size=10),
        )
    assert raised.value.code == CODE_UNSUPPORTED_MODEL
    assert "DeepExplainer" in raised.value.message


def test_missing_shap_dependency_is_actionable(monkeypatch, classification_data) -> None:
    features, _ = classification_data
    monkeypatch.setitem(sys.modules, "shap", None)
    with pytest.raises(ExplainError) as raised:
        build_explainer(
            "random_forest",
            object(),
            lambda matrix: np.full(len(matrix), 0.5),
            background=features[:5],
            feature_names=FEATURE_NAMES,
            config=ExplainConfig(),
        )
    assert raised.value.code == CODE_UNSUPPORTED_MODEL
    assert "not installed" in raised.value.message
    assert "requirements.txt" in raised.value.remediation


def test_config_rejects_unbounded_sampling() -> None:
    problems = ExplainConfig(sample_size=0, background_size=0, local_examples=-1).validate()
    assert any("sample_size" in problem for problem in problems)
    assert any("background_size" in problem for problem in problems)
    assert any("local_examples" in problem for problem in problems)
    assert ExplainConfig().validate() == []
    invalid = ExplainConfig(tolerance=float("nan"))  # type: ignore[arg-type]
    assert any("tolerance" in problem for problem in invalid.validate())
    non_numeric = ExplainConfig(tolerance="strict")  # type: ignore[arg-type]
    assert any("tolerance" in problem for problem in non_numeric.validate())
