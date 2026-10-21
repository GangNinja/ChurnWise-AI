"""Metric correctness, positive-class conventions and edge cases (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnwise.evaluation import (
    binary_metrics,
    confusion_counts,
    precision_recall_curve_data,
    roc_curve_data,
    threshold_grid,
    threshold_sweep,
)
from churnwise.evaluation.errors import (
    CODE_INVALID_CONFIG,
    CODE_METRIC_FAILED,
    EvaluationError,
)


def reference_data(seed: int = 3, rows: int = 400):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=rows)
    scores = np.clip(0.15 + 0.65 * rng.random(rows) + 0.15 * labels, 0.0, 1.0)
    return labels, scores


def test_metrics_match_sklearn_reference_within_tolerance():
    labels, scores = reference_data()
    predictions = (scores >= 0.5).astype(int)
    block = binary_metrics(labels, scores, 0.5)

    assert block["accuracy"] == pytest.approx(
        accuracy_score(labels, predictions), abs=1e-9
    )
    assert block["precision"] == pytest.approx(
        precision_score(labels, predictions), abs=1e-9
    )
    assert block["recall"] == pytest.approx(
        recall_score(labels, predictions), abs=1e-9
    )
    assert block["f1"] == pytest.approx(f1_score(labels, predictions), abs=1e-9)
    assert block["roc_auc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)
    assert block["pr_auc"] == pytest.approx(
        average_precision_score(labels, scores), abs=1e-9
    )
    assert block["brier"] == pytest.approx(brier_score_loss(labels, scores), abs=1e-9)
    assert block["log_loss"] == pytest.approx(
        log_loss(labels, scores, labels=[0, 1]), abs=1e-9
    )


def test_confusion_matrix_label_ordering_and_reconciliation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    block = binary_metrics(labels, scores, 0.5)

    counts = block["confusion_matrix"]
    assert counts["labels"] == [0, 1]
    assert counts["matrix"] == [[1, 1], [1, 1]]
    assert (counts["tn"], counts["fp"], counts["fn"], counts["tp"]) == (1, 1, 1, 1)
    assert counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"] == block["n"]
    assert counts["matrix"] == confusion_matrix(labels, (scores >= 0.5)).tolist()
    assert block["precision"] == pytest.approx(0.5)
    assert block["recall"] == pytest.approx(0.5)


def test_precision_is_undefined_rather_than_zero_without_predicted_positives():
    labels = np.array([0, 0, 1])
    scores = np.array([0.05, 0.10, 0.20])
    block = binary_metrics(labels, scores, 0.5)
    assert block["predicted_positive"] == 0
    assert block["precision"] is None
    assert block["f1"] is None
    assert block["recall"] == pytest.approx(0.0)
    assert "precision" in block["undefined"]
    assert any("no row was predicted as churn" in note for note in block["notes"])


def test_single_class_partition_records_unavailable_ranking_metrics():
    labels = np.zeros(12, dtype=int)
    scores = np.linspace(0.05, 0.95, 12)
    block = binary_metrics(labels, scores, 0.5)
    assert block["accuracy"] is not None
    assert block["roc_auc"] is None
    assert block["pr_auc"] is None
    assert block["recall"] is None
    assert block["brier"] is not None
    assert {"roc_auc", "pr_auc", "recall", "log_loss"} <= set(block["undefined"])
    assert any("single class" in note or "only one class" in note for note in block["notes"])
    assert roc_curve_data(labels, scores)["auc"] is None
    assert precision_recall_curve_data(labels, scores)["pr_auc"] is None


def test_curves_come_from_probabilities_and_expose_auc():
    labels = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.05, 0.20, 0.65, 0.80, 0.95, 0.30])
    roc = roc_curve_data(labels, scores)
    pr = precision_recall_curve_data(labels, scores)
    assert roc["auc"] == pytest.approx(1.0)
    assert roc["fpr"][0] == pytest.approx(0.0)
    assert roc["tpr"][-1] == pytest.approx(1.0)
    assert pr["pr_auc"] == pytest.approx(1.0)
    assert len(pr["precision"]) == len(pr["recall"])
    assert len(pr["thresholds"]) == len(pr["precision"]) - 1


def test_threshold_sweep_is_consistent_and_monotone_in_predicted_positives():
    labels, scores = reference_data(seed=8, rows=300)
    rows = threshold_sweep(labels, scores, threshold_grid(41))
    assert [row["threshold"] for row in rows] == sorted(
        row["threshold"] for row in rows
    )
    predicted = [row["predicted_positive"] for row in rows]
    assert predicted == sorted(predicted, reverse=True)
    for row in rows:
        assert row["tn"] + row["fp"] + row["fn"] + row["tp"] == len(labels)
        if row["precision"] is not None:
            assert 0.0 <= row["precision"] <= 1.0
        if row["recall"] is not None:
            assert 0.0 <= row["recall"] <= 1.0
        if row["f1"] is not None:
            assert 0.0 <= row["f1"] <= 1.0


def test_threshold_grid_always_contains_default_and_is_deterministic():
    first = threshold_grid(25)
    second = threshold_grid(25)
    assert first == second
    assert 0.5 in first
    assert first == sorted(first)


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0, 1], [0.2]),  # length mismatch
        ([0, 1], [0.2, float("nan")]),  # non-finite
        ([0, 1], [0.2, 1.4]),  # outside [0, 1]
        ([0, 2], [0.2, 0.8]),  # non-binary label
        ([], []),  # empty
    ],
)
def test_invalid_metric_inputs_fail_with_actionable_code(labels, scores):
    with pytest.raises(EvaluationError) as caught:
        binary_metrics(labels, scores, 0.5)
    assert caught.value.code == CODE_METRIC_FAILED
    assert caught.value.remediation


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(EvaluationError) as caught:
        binary_metrics([0, 1], [0.2, 0.8], threshold)
    assert caught.value.code == CODE_INVALID_CONFIG


def test_confusion_counts_rejects_mismatched_predictions():
    with pytest.raises(EvaluationError) as caught:
        confusion_counts([0, 1], [0])
    assert caught.value.code == CODE_METRIC_FAILED
