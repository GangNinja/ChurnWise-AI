"""Binary churn metrics computed from genuine probability predictions.

Conventions used everywhere in Phase 7:

* label ``1`` is the positive class - the source churn label mapped by Phase 4/5
  to ``1``;
* ``precision``, ``recall``, ``F1``, ``PR-AUC`` and the reported confusion matrix
  describe that positive class, not an average over classes;
* precision, recall, F1 and AUC are ``None`` - never a fabricated ``0.0`` - when
  their denominator is undefined, and the reason is recorded in ``undefined`` and
  ``notes``.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from churnwise.evaluation.errors import (
    CODE_INVALID_CONFIG,
    CODE_METRIC_FAILED,
    EvaluationError,
)

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

#: Metrics reported for every family. ``auc`` entries are threshold-free.
REPORTED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def validate_threshold(threshold: float) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"threshold must be a number in [0, 1], got {threshold!r}",
            "Pass a probability threshold between 0 and 1.",
        ) from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"threshold must be in [0, 1], got {value!r}",
            "Pass a probability threshold between 0 and 1.",
        )
    return value


def _validated(
    y_true: Any, probabilities: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and validate labels/probabilities before any metric is computed."""
    try:
        labels = np.asarray(y_true).reshape(-1)
        scores = np.asarray(probabilities, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            f"labels or probabilities are not numeric: {type(exc).__name__}: {exc}",
            "Pass the Phase 5 labels and model probabilities unchanged.",
        ) from exc
    if labels.size == 0:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            "cannot compute metrics for an empty partition",
            "Evaluate a non-empty held-out split.",
        )
    if scores.size != labels.size:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            f"{labels.size} labels but {scores.size} probabilities",
            "Predict one probability per evaluated row, in the same order.",
        )
    if not np.isfinite(scores).all():
        raise EvaluationError(
            CODE_METRIC_FAILED,
            "probabilities contain non-finite values",
            "Reload a verified model artifact or retrain the model.",
        )
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            "probabilities fall outside [0, 1]",
            "Use model probabilities, not logits or class labels.",
        )
    if not np.isin(labels, (0, 1)).all():
        raise EvaluationError(
            CODE_METRIC_FAILED,
            "labels must be binary 0/1 churn indicators",
            "Use the Phase 5 target mapping; positive means churned.",
        )
    return labels.astype(int), scores


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return the 2x2 confusion matrix with an explicit label ordering."""
    try:
        labels = np.asarray(y_true).reshape(-1).astype(int)
        predictions = np.asarray(y_pred).reshape(-1).astype(int)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            f"labels or predictions are not numeric: {type(exc).__name__}: {exc}",
            "Pass binary 0/1 labels and predictions.",
        ) from exc
    if labels.size != predictions.size:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            f"{labels.size} labels but {predictions.size} predictions",
            "Predict one class per evaluated row, in the same order.",
        )
    if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise EvaluationError(
            CODE_METRIC_FAILED,
            "confusion matrix labels and predictions must be 0/1",
            "Threshold probabilities with >= to obtain 0/1 classes.",
        )
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    return {
        "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
        "matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": tn + fp + fn + tp,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    total = precision + recall
    if total <= 0:
        return 0.0
    return 2.0 * precision * recall / total


def binary_metrics(
    y_true: Any,
    probabilities: Any,
    threshold: float = 0.5,
    *,
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Compute the Phase 7 metric block for one model at one frozen threshold."""
    labels, scores = _validated(y_true, probabilities)
    value = validate_threshold(threshold)
    predictions = (scores >= value).astype(int)
    counts = confusion_counts(labels, predictions)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    n = counts["n"]
    positives = int(labels.sum())
    negatives = int(n - positives)

    undefined: list[str] = []
    notes: list[str] = []

    precision = _ratio(tp, tp + fp)
    if precision is None:
        undefined.append("precision")
        notes.append("precision is undefined: no row was predicted as churn")
    recall = _ratio(tp, tp + fn)
    if recall is None:
        undefined.append("recall")
        notes.append("recall is undefined: the partition has no positive (churn) rows")
    f1 = _f1(precision, recall)
    if f1 is None:
        undefined.append("f1")
    specificity = _ratio(tn, tn + fp)
    if specificity is None:
        undefined.append("specificity")
        notes.append("specificity is undefined: the partition has no negative rows")
    balanced = (
        None
        if recall is None or specificity is None
        else (recall + specificity) / 2.0
    )
    if balanced is None:
        undefined.append("balanced_accuracy")

    precision_negative = _ratio(tn, tn + fn)
    recall_negative = _ratio(tn, tn + fp)
    f1_negative = _f1(precision_negative, recall_negative)
    macro_precision = (
        None
        if precision is None or precision_negative is None
        else (precision + precision_negative) / 2.0
    )
    macro_recall = (
        None
        if recall is None or recall_negative is None
        else (recall + recall_negative) / 2.0
    )
    macro_f1 = (
        None
        if f1 is None or f1_negative is None
        else (f1 + f1_negative) / 2.0
    )
    for name, macro in (
        ("macro_precision", macro_precision),
        ("macro_recall", macro_recall),
        ("macro_f1", macro_f1),
    ):
        if macro is None:
            undefined.append(name)

    if positives > 0 and negatives > 0:
        roc_auc: float | None = float(roc_auc_score(labels, scores))
        pr_auc: float | None = float(average_precision_score(labels, scores))
        log_loss_value: float | None = float(log_loss(labels, scores, labels=[0, 1]))
    else:
        roc_auc = None
        pr_auc = None
        log_loss_value = None
        undefined.extend(["roc_auc", "pr_auc", "log_loss"])
        notes.append(
            "ROC-AUC, PR-AUC and log loss are undefined: the partition contains "
            "only one class"
        )
    brier = float(brier_score_loss(labels, scores))

    return {
        "positive_class": POSITIVE_LABEL,
        "positive_label": positive_label,
        "threshold": value,
        "n": n,
        "n_positive": positives,
        "n_negative": negatives,
        "positive_rate": float(positives / n),
        "predicted_positive": int(predictions.sum()),
        "predicted_positive_rate": float(predictions.sum() / n),
        "confusion_matrix": counts,
        "accuracy": float((tp + tn) / n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": log_loss_value,
        "class_metrics": {
            str(NEGATIVE_LABEL): {
                "precision": precision_negative,
                "recall": recall_negative,
                "f1": f1_negative,
                "support": negatives,
            },
            str(POSITIVE_LABEL): {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": positives,
            },
        },
        "undefined": undefined,
        "notes": notes,
    }


def threshold_grid(points: int = 99) -> list[float]:
    """Deterministic probability grid that always contains 0.5."""
    if not isinstance(points, int) or points < 3:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"threshold grid needs at least 3 points, got {points!r}",
            "Use the default grid or pass points >= 3.",
        )
    values = {round(float(value), 6) for value in np.linspace(0.01, 0.99, points)}
    values.add(0.5)
    return sorted(values)


def threshold_sweep(
    y_true: Any,
    probabilities: Any,
    grid: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute threshold-dependent metrics for every threshold in ``grid``.

    The sweep is vectorised so a 99-point grid over a few thousand rows stays
    cheap. Values that are undefined at a threshold are ``None``.
    """
    labels, scores = _validated(y_true, probabilities)
    thresholds = list(grid) if grid is not None else threshold_grid()
    values = np.asarray([validate_threshold(item) for item in thresholds], dtype=float)
    if values.size == 0:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            "threshold grid is empty",
            "Pass at least one threshold between 0 and 1.",
        )
    order = np.argsort(values, kind="stable")
    values = values[order]
    predictions = scores[:, None] >= values[None, :]
    positive = labels == 1
    negative = ~positive
    tp = (predictions & positive[:, None]).sum(axis=0)
    fp = (predictions & negative[:, None]).sum(axis=0)
    fn = (~predictions & positive[:, None]).sum(axis=0)
    tn = (~predictions & negative[:, None]).sum(axis=0)
    total_positive = int(positive.sum())
    total_negative = int(negative.sum())
    rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(values):
        tp_i = int(tp[index])
        fp_i = int(fp[index])
        fn_i = int(fn[index])
        tn_i = int(tn[index])
        precision = _ratio(tp_i, tp_i + fp_i)
        recall = _ratio(tp_i, tp_i + fn_i)
        specificity = _ratio(tn_i, tn_i + fp_i)
        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp_i,
                "fp": fp_i,
                "fn": fn_i,
                "tn": tn_i,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "specificity": specificity,
                "accuracy": float((tp_i + tn_i) / max(tp_i + tn_i + fp_i + fn_i, 1)),
                "youden": (
                    None
                    if recall is None or specificity is None
                    else float(recall + specificity - 1.0)
                ),
                "predicted_positive": tp_i + fp_i,
                "predicted_positive_rate": float(
                    (tp_i + fp_i) / max(total_positive + total_negative, 1)
                ),
            }
        )
    return rows


def _curve_arrays(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray, float | None]:
    labels, scores = _validated(y_true, probabilities)
    if labels.sum() in (0, labels.size):
        return np.asarray([]), np.asarray([]), None
    auc_value = float(roc_auc_score(labels, scores))
    return labels, scores, auc_value


def roc_curve_data(y_true: Any, probabilities: Any) -> dict[str, Any]:
    """ROC points from real probabilities plus the threshold-free AUC."""
    labels, scores, auc_value = _curve_arrays(y_true, probabilities)
    if auc_value is None:
        return {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "auc": None,
            "notes": ["ROC curve is undefined: the partition contains only one class"],
        }
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [
            float(value) if np.isfinite(value) else None for value in thresholds
        ],
        "auc": auc_value,
        "notes": [],
    }


def precision_recall_curve_data(y_true: Any, probabilities: Any) -> dict[str, Any]:
    """Precision-recall points from real probabilities plus average precision."""
    labels, scores = _validated(y_true, probabilities)
    if labels.sum() == 0:
        return {
            "precision": [],
            "recall": [],
            "thresholds": [],
            "pr_auc": None,
            "notes": [
                "precision-recall curve is undefined: the partition has no churn rows"
            ],
        }
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return {
        "precision": [float(value) for value in precision],
        "recall": [float(value) for value in recall],
        "thresholds": [float(value) for value in thresholds],
        "pr_auc": float(average_precision_score(labels, scores)),
        "notes": [],
    }


def metrics_from_predictions(
    y_true: Any,
    probabilities: Any,
    threshold: float,
    *,
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Alias used by regenerate flows so the metric path stays single."""
    return binary_metrics(
        y_true, probabilities, threshold, positive_label=positive_label
    )


def summarise_undefined(blocks: Iterable[dict[str, Any]]) -> list[str]:
    """Collect unique undefined-metric notes across metric blocks."""
    notes: list[str] = []
    for block in blocks:
        for note in block.get("notes") or []:
            if note not in notes:
                notes.append(note)
    return notes


__all__ = [
    "NEGATIVE_LABEL",
    "POSITIVE_LABEL",
    "REPORTED_METRICS",
    "binary_metrics",
    "confusion_counts",
    "metrics_from_predictions",
    "precision_recall_curve_data",
    "roc_curve_data",
    "summarise_undefined",
    "threshold_grid",
    "threshold_sweep",
    "validate_threshold",
]
