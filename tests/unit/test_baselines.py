"""Logistic Regression and Random Forest baselines (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from churnwise.models import (
    LogisticRegressionConfig,
    ModelingError,
    RandomForestConfig,
)
from churnwise.models.baselines import (
    baseline_metadata,
    build_estimator,
    fit_baseline,
    load_baseline,
    save_baseline,
)
from churnwise.models.errors import (
    CODE_INVALID_MODEL,
    CODE_MODEL_ARTIFACT,
    CODE_TRAINING_FAILED,
)
from tests.models_helpers import make_prepared_dataset

WEIGHTS = {0: 1.0, 1: 2.0}


def fixture_matrix():
    dataset = make_prepared_dataset(rows=240, seed=5)
    return dataset.train.X, dataset.train.y, dataset.feature_names


def test_logistic_regression_fits_and_returns_bounded_probabilities():
    X, y, names = fixture_matrix()
    fitted = fit_baseline(
        "logistic_regression",
        X,
        y,
        config=LogisticRegressionConfig(max_iter=500),
        class_weight=WEIGHTS,
        feature_names=names,
    )
    probabilities = fitted.predict_proba(X[:20])
    assert probabilities.shape == (20,)
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    assert fitted.family == "logistic_regression"
    assert fitted.train_rows == len(X)
    assert fitted.feature_names == names
    assert fitted.class_weight == WEIGHTS
    assert fitted.config["solver"] == "lbfgs"


def test_random_forest_is_reproducible_with_a_fixed_seed():
    X, y, names = fixture_matrix()
    config = RandomForestConfig(n_estimators=40, n_jobs=1, random_state=17)
    first = fit_baseline(
        "random_forest", X, y, config=config, class_weight=WEIGHTS, feature_names=names
    )
    second = fit_baseline(
        "random_forest", X, y, config=config, class_weight=WEIGHTS, feature_names=names
    )
    assert np.array_equal(first.predict_proba(X[:40]), second.predict_proba(X[:40]))
    assert first.config["n_estimators"] == 40


def test_random_forest_default_is_serial_for_bitwise_stable_predictions():
    """Parallel forest averaging can differ by one ULP; the default is serial."""
    X, y, names = fixture_matrix()
    config = RandomForestConfig(n_estimators=30, random_state=17)
    assert config.n_jobs == 1
    fitted = fit_baseline(
        "random_forest", X, y, config=config, class_weight=WEIGHTS, feature_names=names
    )
    first = fitted.predict_proba(X[:40])
    second = fitted.predict_proba(X[:40])
    assert np.array_equal(first, second)
    assert fitted.estimator.n_jobs == 1


@pytest.mark.parametrize(
    "bad_config",
    [
        LogisticRegressionConfig(max_iter=0),
        LogisticRegressionConfig(C=0.0),
        LogisticRegressionConfig(penalty="l1", solver="lbfgs"),
        RandomForestConfig(n_estimators=0),
        RandomForestConfig(min_samples_leaf=0),
        RandomForestConfig(n_jobs=0),
    ],
)
def test_invalid_baseline_configuration_raises_modeling_error(bad_config):
    family = (
        "logistic_regression"
        if isinstance(bad_config, LogisticRegressionConfig)
        else "random_forest"
    )
    with pytest.raises(ModelingError) as caught:
        build_estimator(family, config=bad_config)
    assert caught.value.code == CODE_INVALID_MODEL
    assert caught.value.remediation


def test_single_class_training_partition_is_rejected_before_fitting():
    X, _, names = fixture_matrix()
    labels = np.zeros(len(X), dtype=np.int8)
    with pytest.raises(ModelingError) as caught:
        fit_baseline(
            "logistic_regression", X, labels, class_weight=WEIGHTS, feature_names=names
        )
    assert caught.value.code == CODE_TRAINING_FAILED


def test_feature_width_mismatch_is_rejected():
    X, y, names = fixture_matrix()
    with pytest.raises(ModelingError) as caught:
        fit_baseline(
            "logistic_regression",
            X[:, :2],
            y,
            class_weight=WEIGHTS,
            feature_names=names,
        )
    assert caught.value.code == CODE_INVALID_MODEL


def test_baseline_artifact_round_trip_and_digest_verification(tmp_path):
    X, y, names = fixture_matrix()
    fitted = fit_baseline(
        "random_forest",
        X,
        y,
        config=RandomForestConfig(n_estimators=30, n_jobs=1),
        class_weight=WEIGHTS,
        feature_names=names,
    )
    directory = tmp_path / "random_forest"
    payload = save_baseline(
        directory, fitted, baseline_metadata(fitted, seed=11)
    )
    assert payload["model_sha256"]
    restored = load_baseline(
        directory / "model.joblib", expected_sha256=payload["model_sha256"]
    )
    assert isinstance(restored, type(fitted))
    assert np.array_equal(restored.predict_proba(X[:10]), fitted.predict_proba(X[:10]))

    with pytest.raises(ModelingError) as caught:
        load_baseline(directory / "model.joblib", expected_sha256="0" * 64)
    assert caught.value.code == CODE_MODEL_ARTIFACT


def test_missing_baseline_artifact_is_actionable(tmp_path):
    with pytest.raises(ModelingError) as caught:
        load_baseline(tmp_path / "absent.joblib")
    assert caught.value.code == CODE_MODEL_ARTIFACT


def test_unknown_baseline_family_is_rejected():
    with pytest.raises(ModelingError) as caught:
        build_estimator("gradient_boosting")
    assert caught.value.code == CODE_INVALID_MODEL
