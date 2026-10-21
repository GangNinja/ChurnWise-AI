"""Fold-safe cross-validation: refitting behaviour and honest unavailability."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from churnwise.config import Settings
from churnwise.evaluation.cv import cross_validate
from churnwise.evaluation.folds import FoldContext
from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.models.baselines import LogisticRegressionConfig


def _context(rows: int = 240) -> tuple[FoldContext, SimpleNamespace]:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "color": rng.choice(["a", "b"], size=rows, p=[0.6, 0.4]),
            "y": rng.integers(0, 2, size=rows),
        }
    )
    # A level present in exactly one row: it is validation in exactly one fold,
    # so that fold's training rows lack it and its matrix must be narrower. A
    # frozen full-train transform would keep all four columns in every fold.
    frame.loc[0, "color"] = "rare"
    context = FoldContext(
        frame=frame,
        target_column="y",
        config=PreprocessConfig(seed=1),
        roles=SimpleNamespace(numeric=["x"], categorical=["color"], id=None, dropped=[]),
        derived_specs=[],
    )
    train_rows = list(range(0, 180))
    val_rows = list(range(180, rows))
    dataset = SimpleNamespace(
        train=SimpleNamespace(row_indices=train_rows, y=frame.loc[train_rows, "y"].tolist()),
        val=SimpleNamespace(row_indices=val_rows, y=frame.loc[val_rows, "y"].tolist()),
    )
    return context, dataset


def test_cv_refits_preprocessing_per_fold() -> None:
    context, dataset = _context()
    summary = cross_validate(
        dataset,
        Settings(),
        "logistic_regression",
        folds=4,
        seed=7,
        logistic_config=LogisticRegressionConfig(random_state=7),
        context=context,
    )

    assert summary.status == "completed"
    assert summary.folds_completed == 4
    assert summary.preprocessing["refitted_per_fold"] is True
    widths = [fold["feature_count"] for fold in summary.fold_details]
    assert min(widths) == 3, "the fold holding the rare row must encode only two levels"
    assert max(widths) == 4
    for fold in summary.fold_details:
        assert fold["n_train"] + fold["n_validation"] == 180 + 60
        assert fold["n_validation"] == 60
        assert fold["f1"] is not None
        assert fold["roc_auc"] is not None


def test_cv_summary_matches_fold_metrics() -> None:
    context, dataset = _context()
    summary = cross_validate(
        dataset,
        Settings(),
        "logistic_regression",
        folds=3,
        seed=11,
        logistic_config=LogisticRegressionConfig(random_state=11),
        context=context,
    )

    assert summary.status == "completed"
    values = [fold["f1"] for fold in summary.fold_details]
    assert summary.metrics["f1"]["mean"] == pytest.approx(float(np.mean(values)))
    assert summary.metrics["f1"]["n"] == 3
    assert "test" not in dataset.__dict__, "CV must not require the test partition"
    assert any("test partition is not used" in note for note in summary.notes)


def test_cv_reports_unavailable_without_reconstruction(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    dataset = SimpleNamespace(
        schema={},
        provenance={"source": {}},
        dataset_id="synthetic",
        train=SimpleNamespace(row_indices=[0], y=[0]),
        val=SimpleNamespace(row_indices=[1], y=[1]),
    )

    summary = cross_validate(dataset, settings, "logistic_regression", folds=3)

    assert summary.status == "unavailable"
    assert summary.folds_completed == 0
    assert summary.preprocessing["refitted_per_fold"] is False
    assert summary.preprocessing["reason"]
    assert any("fold-specific preprocessing" in note for note in summary.notes)


def test_cv_rejects_single_fold() -> None:
    context, dataset = _context()
    from churnwise.evaluation.errors import EvaluationError

    with pytest.raises(EvaluationError):
        cross_validate(dataset, Settings(), "logistic_regression", folds=1, context=context)
