"""Phase 7 integration: identical test rows, saved artifacts and CLI flows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from churnwise.cli import main
from churnwise.config import Settings
from churnwise.evaluation import BenchmarkCoordinator, EvaluationConfig
from churnwise.models import MLPConfig, MLPTrainer
from churnwise.preprocessing import PreprocessingCoordinator
from tests.models_helpers import make_prepared_dataset, persist_phase5

FAST = MLPConfig(
    seed=23,
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
    root = tmp_path_factory.mktemp("phase7-integration")
    settings = Settings(project_root=root)
    dataset = persist_phase5(make_prepared_dataset(rows=420, seed=23), settings)
    trained = MLPTrainer(settings, FAST).train(dataset, persist=True)
    return settings, dataset, trained.run_id


def test_all_families_share_test_rows_matrix_and_split_identity(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    result = BenchmarkCoordinator(
        settings, EvaluationConfig(seed=23, cv_folds=0)
    ).run(
        dataset,
        mlp_run_id=mlp_run_id,
        benchmark_run_id="integration-parity",
        persist=True,
    )
    assert result.complete
    assert result.metrics["preprocessing"]["split_fingerprint"] == dataset.split_spec.fingerprint
    assert result.metrics["preprocessing"]["dataset_sha256"] == dataset.split_spec.dataset_sha256
    assert result.metrics["dataset"]["feature_count"] == len(dataset.feature_names)
    assert result.metrics["dataset"]["n_test"] == len(dataset.test.y)
    digests = result.metrics["dataset"]["matrix_sha256"]
    for split in ("train", "val", "test"):
        expected = hashlib.sha256(
            np.ascontiguousarray(getattr(dataset, split).X, dtype=np.float64).tobytes()
        ).hexdigest()
        assert digests[split] == expected

    root = Path(result.paths["root"])
    with np.load(root / "predictions.npz", allow_pickle=False) as stored:
        assert len(stored["y_test"]) == len(dataset.test.y)
        assert np.array_equal(stored["y_test"], dataset.test.y.astype(np.int8))
        for family in result.families:
            assert stored[f"proba_test_{family}"].shape == (len(dataset.test.y),)
            assert stored[f"proba_val_{family}"].shape == (len(dataset.val.y),)
            assert np.isfinite(stored[f"proba_test_{family}"]).all()
            assert stored[f"proba_test_{family}"].min() >= 0.0
            assert stored[f"proba_test_{family}"].max() <= 1.0

    train_rows = {
        result.family(family)["model"]["train_rows"] for family in result.families
    }
    assert train_rows == {len(dataset.train.y)}

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_model_family"] == result.selected_family
    assert manifest["test_evaluation_calls"] == 1
    assert manifest["preprocessing_run_id"] == dataset.run_id
    assert manifest["test_rows_evaluated"] == len(dataset.test.y)
    for family in result.families:
        assert manifest["thresholds"][family] == pytest.approx(result.threshold(family))

    thresholds = json.loads(
        (root / "threshold_analysis.json").read_text(encoding="utf-8")
    )
    assert thresholds["selected_on"] == "validation"
    for family in result.families:
        payload = thresholds["families"][family]
        assert payload["selected_threshold"] == pytest.approx(result.threshold(family))
        assert payload["sweep"], "the validation threshold sweep must be saved"


def test_regenerate_reproduces_metrics_from_stored_predictions(benchmark_env):
    settings, dataset, mlp_run_id = benchmark_env
    result = BenchmarkCoordinator(
        settings, EvaluationConfig(seed=23, cv_folds=0)
    ).run(
        dataset,
        mlp_run_id=mlp_run_id,
        benchmark_run_id="integration-regenerate",
        persist=True,
    )
    root = Path(result.paths["root"])
    before = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    regenerated = BenchmarkCoordinator(
        settings, EvaluationConfig(seed=23, cv_folds=0)
    ).regenerate(result.benchmark_run_id)
    after = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    assert after["regeneration"]["models_refitted"] is False
    assert after["regeneration"]["test_rows_rescored"] is False
    assert after["regeneration"]["verified_against_stored_metrics"] is True
    for family in result.families:
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
            expected = before["families"][family]["test_metrics"][key]
            observed = after["families"][family]["test_metrics"][key]
            if expected is None:
                assert observed is None
            else:
                assert observed == pytest.approx(expected, abs=1e-12)
        assert (
            after["families"][family]["test_metrics"]["confusion_matrix"]
            == before["families"][family]["test_metrics"]["confusion_matrix"]
        )
    assert regenerated.metrics["regeneration"]["verified_against_stored_metrics"] is True


def test_cli_evaluate_then_regenerate(benchmark_env, capsys):
    settings, dataset, mlp_run_id = benchmark_env
    code = main(
        [
            "evaluate",
            "--preprocessing-run-id",
            dataset.run_id,
            "--mlp-run-id",
            mlp_run_id,
            "--benchmark-run-id",
            "integration-cli",
            "--cv-folds",
            "0",
            "--json",
        ],
        settings=settings,
    )
    output = capsys.readouterr().out
    assert code == 0, output
    payload = json.loads(output)
    assert payload["metrics"]["complete"] is True
    assert payload["metrics"]["validity"]["test_evaluation_calls"] == 1
    root = Path(settings.artifact_dir) / "evaluation" / "integration-cli"
    assert (root / "metrics.json").is_file()
    assert (root / "benchmark.md").is_file()

    regenerate_code = main(
        ["evaluate", "--regenerate", "integration-cli"], settings=settings
    )
    captured = capsys.readouterr().out
    assert regenerate_code == 0, captured
    assert "regenerated from stored predictions" in captured


def test_cli_evaluate_rejects_an_invalid_configuration(benchmark_env, capsys):
    settings, dataset, mlp_run_id = benchmark_env
    code = main(
        [
            "evaluate",
            "--preprocessing-run-id",
            dataset.run_id,
            "--mlp-run-id",
            mlp_run_id,
            "--threshold-strategy",
            "recall_floor",
        ],
        settings=settings,
    )
    assert code == 3
    assert "recall_floor" in capsys.readouterr().err


def test_benchmark_integrates_with_the_actual_phase5_and_phase6_artifacts():
    repo_settings = Settings(project_root=Path.cwd())
    prepared_root = Path(repo_settings.artifact_dir) / "prepared"
    runs = (
        sorted(
            path.name
            for path in prepared_root.glob("*")
            if (path / "summary.json").is_file()
        )
        if prepared_root.is_dir()
        else []
    )
    if not runs:
        pytest.skip("no persisted Phase 5 run is available in this checkout")

    dataset = PreprocessingCoordinator(repo_settings).load(runs[-1])
    model_root = Path(repo_settings.artifact_dir) / "models"
    matching: list[str] = []
    if model_root.is_dir():
        for path in model_root.iterdir():
            metadata_path = path / "mlp" / "metadata.json"
            if not metadata_path.is_file():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (payload.get("preprocessing") or {}).get("run_id") == dataset.run_id:
                matching.append(path.name)
    if not matching:
        pytest.skip("no Phase 6 MLP artifact matches the latest Phase 5 run")

    result = BenchmarkCoordinator(
        repo_settings, EvaluationConfig(seed=repo_settings.seed, cv_folds=0)
    ).run(
        dataset,
        mlp_run_id=sorted(matching)[-1],
        benchmark_run_id="integration-real-artifacts",
        persist=False,
    )
    assert result.complete
    assert result.n_test == dataset.split_spec.sizes["test"]
    assert result.metrics["dataset"]["test_rows_used"] == result.n_test
    for family in result.families:
        metrics = result.test_metrics(family)
        assert metrics["n"] == result.n_test
        counts = metrics["confusion_matrix"]
        assert counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"] == result.n_test
        assert metrics["roc_auc"] is not None
