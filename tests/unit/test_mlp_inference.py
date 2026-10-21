"""Probability prediction, thresholds, raw records and artifact loading (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnwise.config import Settings
from churnwise.models import (
    ChurnPredictor,
    MLPConfig,
    MLPTrainer,
    ModelRegistry,
    ModelingError,
    build_mlp,
)
from churnwise.models.errors import CODE_MODEL_ARTIFACT, CODE_PREDICTION_FAILED
from tests.models_helpers import make_prepared_dataset, persist_phase5, synthetic_frame

FAST = MLPConfig(
    seed=3,
    hidden_units=(8,),
    dropout=(0.0,),
    l2=0.0,
    learning_rate=0.01,
    batch_size=16,
    epochs=3,
    early_stopping_patience=10,
    reduce_lr_patience=10,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("models-inference")
    settings = Settings(project_root=root)
    dataset = persist_phase5(make_prepared_dataset(), settings)
    result = MLPTrainer(settings, FAST).train(dataset, persist=True)
    return settings, dataset, result


def test_probabilities_are_finite_bounded_and_row_aligned(trained):
    settings, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    batch = predictor.predict(dataset.test.X)

    assert batch.probabilities.shape == (len(dataset.test.X),)
    assert batch.classes.shape == batch.probabilities.shape
    assert np.isfinite(batch.probabilities).all()
    assert ((batch.probabilities >= 0.0) & (batch.probabilities <= 1.0)).all()
    assert batch.row_positions == list(range(len(dataset.test.X)))
    assert batch.positive_label == "Yes"
    assert set(np.unique(batch.classes)) <= {0, 1}


def test_threshold_controls_class_predictions(trained):
    _, dataset, result = trained
    probabilities = ChurnPredictor(result.model, result.metadata, dataset=dataset).predict_proba(
        dataset.test.X
    )
    low = ChurnPredictor(result.model, result.metadata, dataset=dataset, threshold=0.05).predict(
        dataset.test.X
    )
    high = ChurnPredictor(result.model, result.metadata, dataset=dataset, threshold=0.95).predict(
        dataset.test.X
    )
    assert low.classes.sum() >= high.classes.sum()
    assert np.array_equal(low.classes, (probabilities >= 0.05).astype(int))
    assert np.array_equal(high.classes, (probabilities >= 0.95).astype(int))
    assert low.threshold == pytest.approx(0.05)


def test_model_input_width_must_match_metadata_features(trained):
    _, dataset, result = trained
    wrong_model = build_mlp(3, MLPConfig(hidden_units=(4,), dropout=(0.0,)))
    with pytest.raises(ModelingError) as caught:
        ChurnPredictor(wrong_model, result.metadata, dataset=dataset)
    assert caught.value.code == CODE_PREDICTION_FAILED
    assert "input width" in caught.value.message


def test_invalid_threshold_and_feature_width_are_rejected(trained):
    _, dataset, result = trained
    with pytest.raises(ModelingError) as caught:
        ChurnPredictor(result.model, result.metadata, dataset=dataset, threshold=1.5)
    assert caught.value.code == CODE_PREDICTION_FAILED

    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    with pytest.raises(ModelingError) as caught:
        predictor.predict(np.zeros((2, 2), dtype=np.float32))
    assert caught.value.code == CODE_PREDICTION_FAILED
    assert "columns" in caught.value.message


def test_non_numeric_matrix_values_are_rejected(trained):
    _, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    with pytest.raises(ModelingError) as caught:
        predictor.predict([["not", "numeric"]])
    assert caught.value.code == CODE_PREDICTION_FAILED
    assert "numeric matrix" in caught.value.message


def test_non_finite_matrix_values_are_rejected(trained):
    _, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    matrix = np.array(dataset.test.X[:1], copy=True)
    matrix[0, 0] = np.nan
    with pytest.raises(ModelingError) as caught:
        predictor.predict(matrix)
    assert caught.value.code == CODE_PREDICTION_FAILED
    assert "non-finite" in caught.value.message


def test_raw_records_use_the_fitted_phase5_pipeline(trained):
    _, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    raw = synthetic_frame(rows=6, seed=99).drop(columns=["churn"])

    batch = predictor.predict_frame(raw)
    expected = predictor.predict_proba(dataset.transform_new(raw))
    assert batch.n_rows == 6
    assert np.allclose(batch.probabilities, expected)
    assert batch.row_positions == list(range(6))

    single = predictor.predict_one(raw.iloc[0].to_dict())
    assert 0.0 <= single.probability <= 1.0
    assert single.predicted_class == int(single.probability >= predictor.threshold)
    assert single.threshold == pytest.approx(predictor.threshold)


def test_raw_records_missing_required_columns_fail_actionably(trained):
    _, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    with pytest.raises(ModelingError) as caught:
        predictor.predict_frame(pd.DataFrame({"tenure_months": [1.0]}))
    assert caught.value.code == CODE_PREDICTION_FAILED
    assert caught.value.remediation


def test_infinite_raw_values_are_imputed_not_propagated(trained):
    _, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    raw = synthetic_frame(rows=2, seed=5).drop(columns=["churn"])
    raw.loc[0, "monthly_charges"] = np.inf
    batch = predictor.predict_frame(raw)
    assert np.isfinite(batch.probabilities).all()


def test_saved_model_reloads_with_consistent_predictions(trained):
    settings, dataset, result = trained
    predictor = ChurnPredictor(result.model, result.metadata, dataset=dataset)
    original = predictor.predict_proba(dataset.test.X)

    reloaded = ChurnPredictor.load(
        settings, result.run_id, with_preprocessing=True
    )
    restored = reloaded.predict_proba(dataset.test.X)
    assert np.allclose(original, restored, rtol=0, atol=1e-6)

    raw = synthetic_frame(rows=4, seed=21).drop(columns=["churn"])
    assert np.allclose(
        predictor.predict_frame(raw).probabilities,
        reloaded.predict_frame(raw).probabilities,
        rtol=0,
        atol=1e-6,
    )


def test_matrix_only_loading_rejects_raw_records(trained):
    settings, _, result = trained
    predictor = ChurnPredictor.load(
        settings, result.run_id, with_preprocessing=False
    )
    with pytest.raises(ModelingError) as caught:
        predictor.predict_frame(synthetic_frame(rows=2).drop(columns=["churn"]))
    assert caught.value.code == CODE_PREDICTION_FAILED


def test_metadata_run_id_mismatch_is_refused(trained):
    settings, _, result = trained
    registry = ModelRegistry(settings)
    metadata_path = Path(registry.paths_for(result.run_id)["metadata"])
    original = metadata_path.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["model_run_id"] = "renamed-run-id"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with pytest.raises(ModelingError) as caught:
            registry.load(result.run_id)
        assert caught.value.code == CODE_MODEL_ARTIFACT
        assert "different run id" in caught.value.message
    finally:
        metadata_path.write_text(original, encoding="utf-8")


def test_tampered_model_artifact_is_refused(trained):
    settings, _, result = trained
    registry = ModelRegistry(settings)
    path = Path(registry.paths_for(result.run_id)["model"])
    original = path.read_bytes()
    tampered = bytearray(original)
    tampered[-8] ^= 0xFF
    path.write_bytes(bytes(tampered))
    try:
        with pytest.raises(ModelingError) as caught:
            registry.load(result.run_id)
        assert caught.value.code == CODE_MODEL_ARTIFACT
        assert "digest" in caught.value.message
    finally:
        path.write_bytes(original)


def test_metadata_records_threshold_and_positive_class_mapping(trained):
    _, dataset, result = trained
    payload = json.loads(json.dumps(result.metadata))
    assert payload["training"]["threshold"] == 0.5
    assert payload["preprocessing"]["target"]["positive_label"] == "Yes"
    assert payload["preprocessing"]["feature_count"] == len(dataset.feature_names)
    assert payload["architecture"]["activation"] == "relu"
    assert payload["architecture"]["output_activation"] == "sigmoid"
