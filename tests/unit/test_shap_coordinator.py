"""ExplainCoordinator against a persisted synthetic Phase 5/7 hand-off."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from churnwise.config import Settings
from churnwise.explain import (
    CODE_NO_MODEL,
    ExplainConfig,
    ExplainCoordinator,
    ExplainError,
)
from churnwise.models.baselines import baseline_metadata, fit_baseline, save_baseline
from tests.models_helpers import make_prepared_dataset, persist_phase5, synthetic_frame

BENCHMARK_ID = "20261021T000000Z-fixture-bench"


def _settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path)


def _fixture(tmp_path: Path) -> tuple[Settings, Any]:
    settings = _settings(tmp_path)
    dataset = persist_phase5(make_prepared_dataset(rows=240, seed=9), settings)
    names = list(dataset.feature_names)
    fitted = fit_baseline(
        "logistic_regression",
        dataset.train.X,
        dataset.train.y,
        feature_names=names,
    )
    metadata = baseline_metadata(fitted, seed=42)
    metadata["preprocessing"] = {
        "run_id": dataset.run_id,
        "feature_names": names,
        "split_fingerprint": dataset.split_spec.fingerprint,
    }
    evaluation = settings.artifact_dir / "evaluation" / BENCHMARK_ID
    payload = save_baseline(evaluation / "models" / "logistic_regression", fitted, metadata)
    (evaluation / "models" / "logistic_regression" / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    metrics = {
        "schema_version": "1.0",
        "families": {
            "logistic_regression": {
                "status": "completed",
                "threshold": {"selected": 0.5, "default": 0.5},
                "model": {"feature_count": len(names)},
            }
        },
        "preprocessing": {
            "run_id": dataset.run_id,
            "feature_names": names,
            "split_fingerprint": dataset.split_spec.fingerprint,
        },
        "selection": {"selected_family": "logistic_regression"},
    }
    (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return settings, dataset


def _config() -> ExplainConfig:
    return ExplainConfig(
        sample_size=25,
        background_size=25,
        mlp_sample_size=6,
        mlp_background_size=10,
        local_examples=2,
        seed=42,
    )


def test_coordinator_persists_global_and_local_artifacts(tmp_path: Path) -> None:
    settings, dataset = _fixture(tmp_path)
    result = ExplainCoordinator(settings, _config()).run(
        "logistic_regression", benchmark_run_id=BENCHMARK_ID
    )

    assert result.complete
    assert result.global_explanation is not None
    block = result.global_explanation
    assert block.output_space == "log_odds"
    assert block.sample["split"] == "val"
    assert block.sample["sampled_rows"] == 25
    assert block.sample["test_rows_used"] == 0
    assert block.additivity_error <= _config().tolerance
    assert block.feature_importance[0]["rank"] == 1
    assert "mapping" in block.explainer
    assert len(result.local_explanations) == 2

    root = settings.artifact_dir / "explanations" / result.run_id
    assert (root / "global.json").is_file()
    assert (root / "importance.png").is_file()
    assert (root / "summary.png").is_file()
    assert list(root.glob("dependence_*.png"))
    assert (root / "run_manifest.json").is_file()
    local_json = sorted((root / "local").glob("*.json"))
    local_png = sorted((root / "local").glob("*.png"))
    assert len(local_json) == 2
    assert len(local_png) == 2

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == 8
    assert manifest["preprocessing_run_id"] == dataset.run_id
    assert manifest["model"]["model_sha256"]
    assert manifest["sampling"]["test_rows_used"] == 0
    assert "no identifiers" in manifest["privacy"]
    assert "causal" in manifest["limitations"][0]

    payload = json.loads((root / "global.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["feature_importance"][0]["feature"]
    assert payload["provenance"]["dataset_sha256"]


def test_coordinator_reports_missing_baseline_artifact(tmp_path: Path) -> None:
    settings, _ = _fixture(tmp_path)
    evaluation = settings.artifact_dir / "evaluation" / BENCHMARK_ID
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    metrics["families"]["random_forest"] = {
        "status": "completed",
        "threshold": {"selected": 0.5},
        "model": {"feature_count": 8},
    }
    (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ExplainError) as raised:
        ExplainCoordinator(settings, _config()).run(
            "random_forest", benchmark_run_id=BENCHMARK_ID
        )
    assert raised.value.code == CODE_NO_MODEL
    assert "artifact" in raised.value.message


def test_coordinator_rejects_mismatched_preprocessing_provenance(tmp_path: Path) -> None:
    settings, _ = _fixture(tmp_path)
    metadata_path = (
        settings.artifact_dir
        / "evaluation"
        / BENCHMARK_ID
        / "models"
        / "logistic_regression"
        / "metadata.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["preprocessing"]["split_fingerprint"] = "0" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExplainError) as raised:
        ExplainCoordinator(settings, _config()).run(
            "logistic_regression", benchmark_run_id=BENCHMARK_ID
        )
    assert raised.value.code == CODE_NO_MODEL
    assert "split_fingerprint" in raised.value.message


def test_coordinator_reports_missing_benchmark(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ExplainError) as raised:
        ExplainCoordinator(settings, _config()).run("logistic_regression")
    assert raised.value.code == CODE_NO_MODEL


def test_custom_records_are_local_only_and_privacy_safe(tmp_path: Path) -> None:
    settings, dataset = _fixture(tmp_path)
    raw = synthetic_frame(rows=6, seed=3).drop(columns=["churn"])
    result = ExplainCoordinator(settings, _config()).explain_records(
        raw, family="logistic_regression", benchmark_run_id=BENCHMARK_ID
    )

    assert result.global_explanation is None
    assert len(result.local_explanations) == 6
    assert all(item.row_position is None or isinstance(item.row_position, int) for item in result.local_explanations)
    assert result.manifest["mode"] == "custom_records"
    assert "identifiers are not" in result.manifest["privacy"]
    assert result.manifest["rows_explained"] == 6
    assert result.manifest["dataset_sha256"]
    assert result.manifest["split_fingerprint"]
    assert result.manifest["output_space"] == "log_odds"
    assert result.manifest["explainer"]["name"] == "shap.LinearExplainer"
    for item in result.local_explanations:
        assert 0.0 <= item.probability <= 1.0
        assert item.summary
        assert item.additivity_error <= _config().tolerance
    assert result.paths["run_manifest"]


def test_single_customer_record_can_be_explained(tmp_path: Path) -> None:
    settings, _ = _fixture(tmp_path)
    raw = synthetic_frame(rows=1, seed=13).drop(columns=["churn"])
    result = ExplainCoordinator(settings, _config()).explain_records(
        raw, family="logistic_regression", benchmark_run_id=BENCHMARK_ID
    )
    assert result.complete
    assert len(result.local_explanations) == 1
    item = result.local_explanations[0]
    assert 0.0 <= item.probability <= 1.0
    assert item.summary
    assert item.top_positive or item.top_negative
    assert result.manifest["rows_explained"] == 1


def test_custom_records_reject_oversized_batches(tmp_path: Path) -> None:
    settings, _ = _fixture(tmp_path)
    config = ExplainConfig(max_local_batch=2)
    frame = synthetic_frame(rows=4, seed=3).drop(columns=["churn"])
    with pytest.raises(ExplainError) as raised:
        ExplainCoordinator(settings, config).explain_records(
            frame, family="logistic_regression", benchmark_run_id=BENCHMARK_ID
        )
    assert raised.value.code == "CW-XAI-001"
    assert "batch limit" in raised.value.message


def test_explanations_never_touch_test_rows(tmp_path: Path) -> None:
    settings, dataset = _fixture(tmp_path)
    result = ExplainCoordinator(settings, _config()).run(
        "logistic_regression", benchmark_run_id=BENCHMARK_ID
    )
    assert result.global_explanation is not None
    positions = set(result.global_explanation.sample["positions"])
    assert positions
    assert max(positions) < len(dataset.val.X)
    assert result.manifest["sampling"]["available_rows"] == len(dataset.val.X)
