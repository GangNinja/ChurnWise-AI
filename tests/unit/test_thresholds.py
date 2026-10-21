"""Threshold selection strategies and their validation-only contract (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from churnwise.evaluation import (
    EvaluationConfig,
    select_threshold,
    threshold_analysis,
    threshold_sweep,
)
from churnwise.evaluation.errors import (
    CODE_INVALID_CONFIG,
    CODE_METRIC_FAILED,
    EvaluationError,
)

LABELS = np.array([1, 1, 0, 0, 0])
SCORES = np.array([0.9, 0.4, 0.6, 0.2, 0.1])


def test_fixed_strategy_ignores_the_data():
    choice = select_threshold(LABELS, SCORES, strategy="fixed", fixed_threshold=0.35)
    assert choice.threshold == 0.35
    assert choice.strategy == "fixed"
    assert choice.selected_on == "validation"
    assert choice.validation_metrics["threshold"] == 0.35


def test_max_f1_chooses_the_best_validation_threshold_with_a_centred_tie_break():
    rows = threshold_sweep(LABELS, SCORES)
    best = max(row["f1"] for row in rows if row["f1"] is not None)
    best_thresholds = [
        row["threshold"]
        for row in rows
        if row["f1"] is not None and row["f1"] == pytest.approx(best)
    ]
    expected = max(best_thresholds, key=lambda value: -abs(value - 0.5))
    choice = select_threshold(LABELS, SCORES, strategy="max_f1")
    assert choice.validation_metrics["f1"] == pytest.approx(best)
    assert choice.threshold == pytest.approx(expected)
    assert choice.threshold == pytest.approx(0.4)
    assert "frozen" in " ".join(choice.notes)


def test_max_youden_chooses_a_defined_threshold():
    choice = select_threshold(LABELS, SCORES, strategy="max_youden")
    rows = threshold_sweep(LABELS, SCORES)
    best = max(row["youden"] for row in rows if row["youden"] is not None)
    assert choice.validation_metrics["recall"] is not None
    assert (
        choice.validation_metrics["recall"]
        + choice.validation_metrics["specificity"]
        - 1.0
    ) == pytest.approx(best)


def test_recall_floor_chooses_the_highest_qualifying_threshold():
    rows = threshold_sweep(LABELS, SCORES)
    qualifying = [row["threshold"] for row in rows if row["recall"] == pytest.approx(1.0)]
    choice = select_threshold(
        LABELS, SCORES, strategy="recall_floor", recall_floor=1.0
    )
    assert choice.threshold == pytest.approx(max(qualifying))
    assert choice.threshold == pytest.approx(0.4)
    assert choice.validation_metrics["recall"] == pytest.approx(1.0)


def test_unreachable_recall_floor_falls_back_and_says_so():
    # No grid threshold is low enough to flag either positive row, so the recall
    # floor cannot be met; the documented fallback is the F1 optimum.
    labels = np.array([1, 1, 0])
    scores = np.array([0.001, 0.002, 0.5])
    choice = select_threshold(
        labels, scores, strategy="recall_floor", recall_floor=0.9
    )
    assert choice.strategy == "recall_floor"
    assert "fallback" in choice.objective
    assert any("recall floor" in note for note in choice.notes)
    assert choice.threshold == pytest.approx(0.5)


def test_single_class_validation_is_reported_not_fabricated():
    choice = select_threshold(
        np.zeros(6, dtype=int),
        np.linspace(0.1, 0.9, 6),
        strategy="fixed",
        fixed_threshold=0.5,
    )
    assert choice.validation_metrics["roc_auc"] is None
    assert "roc_auc" in choice.validation_metrics["undefined"]
    with pytest.raises(EvaluationError) as caught:
        select_threshold(
            np.zeros(6, dtype=int), np.linspace(0.1, 0.9, 6), strategy="max_f1"
        )
    assert caught.value.code == CODE_METRIC_FAILED


@pytest.mark.parametrize(
    "strategy,floor",
    [("unknown", None), ("recall_floor", None), ("max_f1", 1.4)],
)
def test_invalid_threshold_configuration_is_rejected(strategy, floor):
    with pytest.raises(EvaluationError) as caught:
        select_threshold(LABELS, SCORES, strategy=strategy, recall_floor=floor)
    assert caught.value.code == CODE_INVALID_CONFIG


def test_threshold_analysis_records_the_default_and_the_frozen_choice():
    choice = select_threshold(LABELS, SCORES, strategy="fixed", fixed_threshold=0.5)
    payload = threshold_analysis(LABELS, SCORES, choice)
    assert payload["selected_on"] == "validation"
    assert payload["selected_threshold"] == 0.5
    assert payload["default_threshold"] == 0.5
    assert payload["default_threshold_row"]["threshold"] == pytest.approx(0.5)
    assert payload["grid_points"] == len(payload["sweep"])
    thresholds = [row["threshold"] for row in payload["sweep"]]
    assert thresholds == sorted(thresholds)
    assert 0.5 in thresholds


def test_evaluation_config_validates_cv_and_selection_options():
    assert EvaluationConfig().validate() == []
    assert EvaluationConfig(cv_folds=0).validate() == []
    assert EvaluationConfig(cv_folds=1).validate()
    assert EvaluationConfig(threshold_strategy="recall_floor").validate()
    assert EvaluationConfig(threshold_strategy="best").validate()
    assert EvaluationConfig(selection_metric="auc").validate()
    assert EvaluationConfig(selection_min_recall=1.5).validate()
