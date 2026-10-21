"""Training workflow, history recording, early stopping and diagnostics (Phase 6)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from churnwise.config import Settings
from churnwise.models import MLPConfig, MLPTrainer, ModelingError
from churnwise.models.contracts import TrainingResult
from churnwise.models.errors import CODE_INVALID_MODEL
from tests.models_helpers import make_prepared_dataset

FAST = MLPConfig(
    seed=11,
    hidden_units=(8,),
    dropout=(0.0,),
    l2=0.0,
    learning_rate=0.01,
    batch_size=16,
    epochs=5,
    early_stopping_patience=10,
    reduce_lr_patience=10,
    threshold=0.5,
)


def settings_for(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path)


def test_training_records_history_diagnostics_and_leaves_test_split_alone(tmp_path):
    dataset = make_prepared_dataset()
    result = MLPTrainer(settings_for(tmp_path), FAST).train(dataset, persist=False)

    assert isinstance(result, TrainingResult)
    assert {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(result.history)
    lengths = {key: len(values) for key, values in result.history.items()}
    assert len(set(lengths.values())) == 1
    assert 1 <= lengths["loss"] <= FAST.epochs
    assert all(np.isfinite(values).all() for values in result.history.values())

    diagnostics = result.diagnostics
    assert diagnostics["train_rows"] == len(dataset.train.y)
    assert diagnostics["val_rows"] == len(dataset.val.y)
    assert diagnostics["test_rows_used"] == 0
    assert diagnostics["feature_count"] == len(dataset.feature_names)
    assert diagnostics["best_epoch"] <= diagnostics["epochs_completed"]
    assert diagnostics["best_val_loss"] == pytest.approx(
        min(result.history["val_loss"]), rel=0, abs=0
    )
    assert isinstance(diagnostics["duration_seconds"], float)
    assert diagnostics["duration_seconds"] >= 0.0

    train_rows = set(dataset.train.row_indices)
    val_rows = set(dataset.val.row_indices)
    test_rows = set(dataset.test.row_indices)
    assert not train_rows & test_rows
    assert not val_rows & test_rows


def test_early_stopping_stops_before_epoch_limit_and_restores_best(tmp_path):
    # Labels carry no signal, so validation loss cannot keep improving and the
    # patience-0 callback must stop the run well before the epoch limit.
    dataset = make_prepared_dataset()
    rng = np.random.default_rng(404)
    train_y = rng.integers(0, 2, size=len(dataset.train.y)).astype(np.int8)
    val_y = rng.integers(0, 2, size=len(dataset.val.y)).astype(np.int8)
    noisy = dataclasses.replace(
        dataset,
        train=dataclasses.replace(dataset.train, y=train_y),
        val=dataclasses.replace(dataset.val, y=val_y),
    )
    config = dataclasses.replace(
        FAST, epochs=40, early_stopping_patience=0, reduce_lr_patience=0
    )
    result = MLPTrainer(settings_for(tmp_path), config).train(noisy, persist=False)
    assert result.diagnostics["stopped_early"] is True
    assert result.diagnostics["epochs_completed"] < config.epochs
    assert result.diagnostics["max_epochs_reached"] is False
    # restore_best_weights keeps the best monitored checkpoint, not the last epoch.
    assert result.diagnostics["best_epoch"] <= result.diagnostics["epochs_completed"]


def test_persisted_run_writes_model_metadata_history_curves_and_reports(tmp_path):
    dataset = make_prepared_dataset()
    settings = settings_for(tmp_path)
    result = MLPTrainer(settings, FAST).train(dataset, persist=True)

    model_path = Path(result.paths["model"])
    assert model_path.is_file()
    assert Path(result.paths["history"]).is_file()
    assert Path(result.paths["metadata"]).is_file()
    assert Path(result.paths["curves"]).is_file()
    assert Path(result.paths["report_md"]).is_file()
    assert Path(result.paths["checkpoint"]).is_file()
    assert result.metadata["artifacts"]["checkpoint"] == result.paths["checkpoint"]
    assert result.metadata["model_sha256"]
    assert result.metadata["architecture"]["input_dim"] == len(dataset.feature_names)
    assert result.metadata["training"]["loss"] == "binary_crossentropy"
    assert result.metadata["training"]["optimizer"] == "adam"
    assert result.metadata["preprocessing"]["run_id"] == dataset.run_id
    assert result.metadata["preprocessing"]["dataset_sha256"] == "d" * 64
    assert result.metadata["preprocessing"]["target"]["positive_label"] == "Yes"
    assert result.metadata["reproducibility"]["seeds"]["tensorflow"] == FAST.seed
    assert result.metadata["environment"]["tensorflow"]

    saved = json.loads(Path(result.paths["history"]).read_text(encoding="utf-8"))
    assert saved["loss"] == result.history["loss"]


def test_class_weights_and_validation_are_passed_to_training(tmp_path):
    dataset = make_prepared_dataset()
    result = MLPTrainer(settings_for(tmp_path), FAST).train(dataset, persist=False)
    assert result.metadata["training"]["class_weight"] == {"0": 1.0, "1": 2.0}
    assert result.diagnostics["class_weight"] == {"0": 1.0, "1": 2.0}


def test_overlapping_split_row_indices_are_rejected(tmp_path):
    dataset = make_prepared_dataset()
    overlapping = dataclasses.replace(
        dataset,
        val=dataclasses.replace(
            dataset.val, row_indices=dataset.train.row_indices[:5]
        ),
    )
    with pytest.raises(ModelingError) as caught:
        MLPTrainer(settings_for(tmp_path), FAST).train(overlapping, persist=False)
    assert caught.value.code == CODE_INVALID_MODEL
    assert "share" in caught.value.message


def test_invalid_class_weights_are_rejected_before_fitting(tmp_path):
    dataset = make_prepared_dataset()
    broken = dataclasses.replace(
        dataset,
        schema={**dataset.schema, "class_weights": {"0": "not-a-number", "1": 2.0}},
    )
    with pytest.raises(ModelingError) as caught:
        MLPTrainer(settings_for(tmp_path), FAST).train(broken, persist=False)
    assert caught.value.code == CODE_INVALID_MODEL
    assert "class weights" in caught.value.message


def test_non_finite_training_features_are_rejected_before_fitting(tmp_path):
    dataset = make_prepared_dataset()
    broken_X = np.array(dataset.train.X, copy=True)
    broken_X[0, 0] = np.nan
    broken = dataclasses.replace(
        dataset,
        train=dataclasses.replace(dataset.train, X=broken_X),
    )
    with pytest.raises(ModelingError) as caught:
        MLPTrainer(settings_for(tmp_path), FAST).train(broken, persist=False)
    assert caught.value.code == CODE_INVALID_MODEL
    assert "non-finite" in caught.value.message


def test_ragged_feature_matrix_is_rejected_without_a_traceback(tmp_path):
    dataset = make_prepared_dataset()
    ragged = dataclasses.replace(
        dataset,
        train=dataclasses.replace(
            dataset.train, X=[[1.0, 2.0, 3.0], [4.0]]  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(ModelingError) as caught:
        MLPTrainer(settings_for(tmp_path), FAST).train(ragged, persist=False)
    assert caught.value.code == CODE_INVALID_MODEL
    assert "numeric matrix" in caught.value.message


def test_labels_outside_zero_one_and_shape_mismatch_are_rejected(tmp_path):
    dataset = make_prepared_dataset()
    trainer = MLPTrainer(settings_for(tmp_path), FAST)

    bad_labels = np.array(dataset.train.y, copy=True)
    bad_labels[0] = 2
    with pytest.raises(ModelingError) as caught:
        trainer.train(
            dataclasses.replace(
                dataset, train=dataclasses.replace(dataset.train, y=bad_labels)
            ),
            persist=False,
        )
    assert caught.value.code == CODE_INVALID_MODEL

    with pytest.raises(ModelingError) as caught:
        trainer.train(
            dataclasses.replace(
                dataset,
                train=dataclasses.replace(dataset.train, X=dataset.train.X[:-3]),
            ),
            persist=False,
        )
    assert "feature rows" in caught.value.message


def test_oversampled_training_rows_are_reflected_in_diagnostics(tmp_path):
    dataset = make_prepared_dataset()
    duplicated = [0, 1, 2, 3]
    oversampled = dataclasses.replace(dataset, oversample_indices=duplicated)
    result = MLPTrainer(settings_for(tmp_path), FAST).train(oversampled, persist=False)
    assert result.diagnostics["train_rows"] == len(dataset.train.y) + len(duplicated)
    assert result.diagnostics["train_positive_count"] == int(
        oversampled.resampled_train().y.sum()
    )


def test_seeded_training_is_repeatable_at_fixed_seed(tmp_path):
    dataset = make_prepared_dataset()
    first = MLPTrainer(settings_for(tmp_path), FAST).train(dataset, persist=False)
    second = MLPTrainer(settings_for(tmp_path), FAST).train(dataset, persist=False)
    assert first.history["loss"] == pytest.approx(second.history["loss"], rel=1e-6, abs=1e-6)
