"""Benchmark isolation, selection provenance and failure handling (Phase 7)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from churnwise.config import Settings
from churnwise.evaluation import (
    BenchmarkCoordinator,
    EvaluationConfig,
    EvaluationError,
)
from churnwise.models import MLPConfig, MLPTrainer
from tests.models_helpers import make_prepared_dataset, persist_phase5

FAST = MLPConfig(
    seed=21,
    hidden_units=(8,),
    dropout=(0.0,),
    l2=0.0,
    learning_rate=0.01,
    batch_size=16,
    epochs=4,
    early_stopping_patience=10,
    reduce_lr_patience=10,
)


@pytest.fixture(scope="module")
def benchmark_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase7-unit")
    settings = Settings(project_root=root)
    dataset = persist_phase5(make_prepared_dataset(rows=360, seed=21), settings)
    result = MLPTrainer(settings, FAST).train(dataset, persist=True)
    return settings, dataset, result.run_id


def _run(settings, dataset, mlp_run_id, *, run_id, persist=False, config=None):
    return BenchmarkCoordinator(
        settings, config or EvaluationConfig(seed=21, cv_folds=0)
    ).run(
        dataset,
        mlp_run_id=mlp_run_id,
        benchmark_run_id=run_id,
        persist=persist,
    )


def test_benchmark_evaluates_every_family_on_shared_held_out_rows(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    result = _run(
        settings,
        dataset,
        mlp_run_id,
        run_id="unit-complete-benchmark",
        persist=True,
    )

    assert result.complete
    assert set(result.families) == {"logistic_regression", "random_forest", "mlp"}
    assert result.n_test == len(dataset.test.y)
    assert result.metrics["validity"]["test_evaluation_calls"] == 1
    assert result.metrics["validity"]["test_evaluated_once"] is True
    assert result.metrics["validity"]["thresholds_frozen_before_test"] is True
    assert result.metrics["selection"]["test_used_for_selection"] is False

    for family in result.families:
        block = result.family(family)
        assert block["status"] == "completed"
        metrics = block["test_metrics"]
        counts = metrics["confusion_matrix"]
        assert counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"] == result.n_test
        assert 0.0 <= block["threshold"]["selected"] <= 1.0
        assert block["validation_metrics"]["threshold"] == block["threshold"]["selected"]
        assert block["test_metrics_at_0_5"]["threshold"] == 0.5
        assert block["test_metrics"]["n"] == result.n_test
        assert block["model"]["train_rows"] == len(dataset.train.y)
        assert block["curves"]["roc"]["auc"] is not None

    selected = result.selected_family
    assert selected in result.families
    ranking = result.metrics["selection"]["ranking"]
    assert len(ranking) == 3
    assert result.metrics["selection"]["selected_validation_value"] == max(
        row["validation_value"] for row in ranking
    )

    root = Path(result.paths["root"])
    for name in (
        "metrics.json",
        "benchmark.md",
        "run_manifest.json",
        "predictions.npz",
        "curves.json",
        "threshold_analysis.json",
        "roc_curves.png",
        "precision_recall_curves.png",
        "metric_comparison.png",
        "threshold_analysis.png",
        "probability_distributions.png",
    ):
        assert (root / name).is_file(), name
    for family in result.families:
        assert (root / f"confusion_matrix_{family}.png").is_file()
        assert (root / "models" / family / "model.joblib").is_file() or family == "mlp"

    markdown = (root / "benchmark.md").read_text(encoding="utf-8")
    assert "held-out test" in markdown
    assert "Threshold" in markdown
    assert "Selection uses validation predictions only" in markdown


def test_poisoning_test_labels_cannot_change_thresholds_or_selection(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    baseline = _run(
        settings, dataset, mlp_run_id, run_id="unit-label-baseline"
    )
    rng = np.random.default_rng(999)
    poisoned = dataclasses.replace(
        dataset,
        test=dataclasses.replace(
            dataset.test,
            y=rng.integers(0, 2, size=len(dataset.test.y)).astype(np.int8),
        ),
    )
    mutated = _run(settings, poisoned, mlp_run_id, run_id="unit-label-poisoned")

    assert mutated.selected_family == baseline.selected_family
    for family in baseline.families:
        assert mutated.threshold(family) == pytest.approx(baseline.threshold(family))
        assert (
            mutated.family(family)["validation_metrics"]["f1"]
            == pytest.approx(baseline.family(family)["validation_metrics"]["f1"])
        )
    assert any(
        mutated.test_metrics(family)["f1"] != baseline.test_metrics(family)["f1"]
        for family in baseline.families
    )


def test_repeated_runs_with_fixed_settings_reproduce_metrics(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    first = _run(settings, dataset, mlp_run_id, run_id="unit-repro-a")
    second = _run(settings, dataset, mlp_run_id, run_id="unit-repro-b")
    assert first.selected_family == second.selected_family
    for family in first.families:
        assert first.threshold(family) == pytest.approx(second.threshold(family))
        left = first.test_metrics(family)
        right = second.test_metrics(family)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
            if left[key] is None:
                assert right[key] is None
            else:
                assert right[key] == pytest.approx(left[key], abs=1e-12)
        assert left["confusion_matrix"] == right["confusion_matrix"]


def test_failed_mlp_does_not_corrupt_baseline_results(benchmark_env):
    settings, dataset, _ = benchmark_env
    result = _run(
        settings,
        dataset,
        "missing-run-id-000",
        run_id="unit-missing-mlp",
    )
    assert result.complete is False
    assert result.metrics["missing_families"] == ["mlp"]
    assert result.family("mlp")["status"] == "failed"
    assert result.family("mlp")["error"]
    for family in ("logistic_regression", "random_forest"):
        assert result.family(family)["status"] == "completed"
        assert result.test_metrics(family)["n"] == result.n_test
    assert result.selected_family in {"logistic_regression", "random_forest"}
    assert any("MLP Neural Network could not be evaluated" in item for item in result.warnings)


def test_no_mlp_artifact_is_recorded_as_a_missing_family(tmp_path):
    settings = Settings(project_root=tmp_path / "no-model")
    dataset = make_prepared_dataset(rows=240, seed=31)
    result = BenchmarkCoordinator(
        settings, EvaluationConfig(seed=31, cv_folds=0)
    ).run(dataset, benchmark_run_id="unit-no-mlp", persist=False)
    assert result.complete is False
    assert result.family("mlp")["status"] == "failed"
    assert result.family("mlp")["error"]["code"] == "CW-EVL-006"
    assert result.family("logistic_regression")["status"] == "completed"
    assert result.family("random_forest")["status"] == "completed"


def test_mlp_from_another_preprocessing_run_is_rejected(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    foreign = dataclasses.replace(dataset, run_id="unit-foreign-preprocessing")
    result = _run(settings, foreign, mlp_run_id, run_id="unit-foreign-mlp")
    block = result.family("mlp")
    assert block["status"] == "failed"
    assert block["error"]["code"] == "CW-EVL-002"
    assert result.complete is False
    assert result.family("random_forest")["status"] == "completed"


def test_invalid_evaluation_configuration_fails_before_any_fit(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    with pytest.raises(EvaluationError) as caught:
        BenchmarkCoordinator(
            settings,
            EvaluationConfig(threshold_strategy="magic", cv_folds=0),
        )
    assert caught.value.code == "CW-EVL-001"
    with pytest.raises(EvaluationError) as caught:
        _run(
            settings,
            dataset,
            mlp_run_id,
            run_id="unit-bad-cv",
            config=EvaluationConfig(cv_folds=1),
        )
    assert caught.value.code == "CW-EVL-001"


def test_all_families_failing_raises_instead_of_fabricating_results(benchmark_env):
    settings, dataset, _ = benchmark_env
    broken = dataclasses.replace(
        dataset,
        train=dataclasses.replace(
            dataset.train, y=np.zeros(len(dataset.train.y), dtype=np.int8)
        ),
    )
    with pytest.raises(EvaluationError) as caught:
        _run(settings, broken, "missing-run-id-000", run_id="unit-all-failed")
    assert caught.value.code == "CW-EVL-005"
