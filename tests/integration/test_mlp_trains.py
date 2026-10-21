"""Phase 6 end-to-end: CLI train/predict and real Phase 5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnwise.cli import main
from churnwise.config import Settings
from churnwise.models import MLPConfig, MLPTrainer
from churnwise.preprocessing import PreprocessingCoordinator
from tests.models_helpers import make_prepared_dataset, persist_phase5, synthetic_frame


def settings_for(root: Path) -> Settings:
    return Settings(project_root=root)


def test_cli_trains_then_predicts_raw_csv(tmp_path, capsys):
    settings = settings_for(tmp_path)
    dataset = persist_phase5(make_prepared_dataset(), settings)

    train_code = main(
        [
            "train",
            "--preprocessing-run-id",
            dataset.run_id,
            "--epochs",
            "2",
            "--hidden",
            "8",
            "--dropout",
            "0",
            "--batch-size",
            "16",
            "--json",
        ],
        settings=settings,
    )
    train_output = capsys.readouterr().out
    assert train_code == 0, train_output
    payload = json.loads(train_output)
    assert payload["metadata"]["phase"] == 6
    assert payload["metadata"]["architecture"]["input_dim"] == len(dataset.feature_names)
    assert payload["diagnostics"]["test_rows_used"] == 0
    assert Path(payload["paths"]["model"]).is_file()

    raw = synthetic_frame(rows=5, seed=31).drop(columns=["churn"])
    input_path = tmp_path / "customers.csv"
    raw.to_csv(input_path, index=False)
    output_path = tmp_path / "predictions.csv"

    predict_code = main(
        [
            "predict",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            "0.4",
        ],
        settings=settings,
    )
    captured = capsys.readouterr().out
    assert predict_code == 0, captured
    predictions = pd.read_csv(output_path)
    assert list(predictions.columns) == [
        "row_index",
        "churn_probability",
        "churn_prediction",
    ]
    assert len(predictions) == len(raw)
    assert predictions["row_index"].tolist() == list(range(len(raw)))
    assert predictions["churn_probability"].between(0.0, 1.0).all()
    assert set(predictions["churn_prediction"]) <= {0, 1}


def test_cli_reports_missing_model_and_bad_threshold(tmp_path, capsys):
    settings = settings_for(tmp_path)
    input_path = tmp_path / "customers.csv"
    synthetic_frame(rows=2).drop(columns=["churn"]).to_csv(input_path, index=False)

    assert main(["predict", "--input", str(input_path)], settings=settings) == 2
    assert "no trained model" in capsys.readouterr().err

    assert (
        main(
            ["predict", "--input", str(input_path), "--threshold", "2"],
            settings=settings,
        )
        == 3
    )


def test_training_integrates_with_the_actual_phase5_artifacts(tmp_path):
    repo_settings = Settings(project_root=Path.cwd())
    prepared_root = Path(repo_settings.artifact_dir) / "prepared"
    runs = sorted(
        path.name for path in prepared_root.glob("*") if (path / "summary.json").is_file()
    ) if prepared_root.is_dir() else []
    if not runs:
        pytest.skip("no persisted Phase 5 run is available in this checkout")

    dataset = PreprocessingCoordinator(repo_settings).load(runs[-1])
    assert dataset.feature_names
    assert len(dataset.train.X) > 0 and len(dataset.val.X) > 0

    config = MLPConfig(
        seed=5,
        hidden_units=(8,),
        dropout=(0.0,),
        l2=0.0,
        learning_rate=0.01,
        batch_size=128,
        epochs=2,
        early_stopping_patience=10,
        reduce_lr_patience=10,
    )
    result = MLPTrainer(settings_for(tmp_path), config).train(dataset, persist=False)

    assert result.metadata["preprocessing"]["run_id"] == dataset.run_id
    assert result.metadata["architecture"]["input_dim"] == len(dataset.feature_names)
    assert result.diagnostics["train_rows"] == len(dataset.train.y)
    assert result.diagnostics["val_rows"] == len(dataset.val.y)
    assert result.diagnostics["test_rows_used"] == 0
    assert np.isfinite(result.history["val_loss"]).all()
    probabilities = result.model.predict(dataset.val.X, verbose=0).reshape(-1)
    assert probabilities.shape == (len(dataset.val.X),)
    assert np.isfinite(probabilities).all()
