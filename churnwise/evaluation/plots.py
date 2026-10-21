"""Publication-quality figures generated only from measured Phase 7 results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from churnwise.evaluation.contracts import FAMILY_LABELS
from churnwise.evaluation.errors import CODE_ARTIFACT_FAILED, EvaluationError

COLOURS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "mlp": "#d62728",
}
_DEFAULT_COLOUR = "#7f7f7f"


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def _save(figure: Any, path: Path) -> Path:
    plotted = Path(path)
    plotted.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.tight_layout()
        figure.savefig(plotted, bbox_inches="tight")
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            CODE_ARTIFACT_FAILED,
            f"plot could not be written to {plotted}: {exc}",
            "Check that the artifact directory is writable.",
        ) from exc
    finally:
        plt = _matplotlib()
        plt.close(figure)
    return plotted


def plot_confusion_matrix(
    metrics: Mapping[str, Any], family: str, path: Path
) -> Path:
    """One 2x2 confusion matrix with counts and row-normalised percentages."""
    plt = _matplotlib()
    matrix = (metrics.get("confusion_matrix") or {}).get("matrix")
    if not matrix:
        raise EvaluationError(
            CODE_ARTIFACT_FAILED,
            "confusion matrix data is missing",
            "Recompute metrics before plotting.",
        )
    values = np.asarray(matrix, dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals > 0,
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(values, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Customers")
    labels = ["not churned (0)", "churned (1)"]
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{int(values[row, column])}\n({shares[row, column]:.1%} of row)",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if shares[row, column] > 0.55 else "black",
            )
    axis.set_xticks([0, 1], labels)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title(
        f"{FAMILY_LABELS.get(family, family)} — confusion matrix\n"
        f"threshold {float(metrics.get('threshold', 0.0)):.3f}, n={int(metrics.get('n', 0))}"
    )
    axis.grid(False)
    return _save(figure, path)


def plot_roc_curves(curves: Mapping[str, Mapping[str, Any]], path: Path) -> Path:
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    drawn = False
    for family, payload in curves.items():
        roc = payload.get("roc") or {}
        fpr = roc.get("fpr") or []
        tpr = roc.get("tpr") or []
        if not fpr or not tpr:
            continue
        axis.plot(
            fpr,
            tpr,
            color=COLOURS.get(family, _DEFAULT_COLOUR),
            label=f"{FAMILY_LABELS.get(family, family)} (AUC {roc.get('auc', 0):.3f})",
            linewidth=1.8,
        )
        drawn = True
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="chance")
    axis.set_xlabel("False positive rate (1 − specificity)")
    axis.set_ylabel("True positive rate (recall)")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_title("ROC curves — held-out test partition")
    axis.legend(loc="lower right")
    if not drawn:
        axis.text(
            0.5,
            0.5,
            "ROC curves unavailable\n(single-class or failed evaluation)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    return _save(figure, path)


def plot_precision_recall_curves(
    curves: Mapping[str, Mapping[str, Any]], path: Path
) -> Path:
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    drawn = False
    for family, payload in curves.items():
        pr = payload.get("pr") or {}
        precision = pr.get("precision") or []
        recall = pr.get("recall") or []
        if not precision or not recall:
            continue
        axis.plot(
            recall,
            precision,
            color=COLOURS.get(family, _DEFAULT_COLOUR),
            label=f"{FAMILY_LABELS.get(family, family)} (AP {pr.get('pr_auc', 0):.3f})",
            linewidth=1.8,
        )
        drawn = True
    axis.set_xlabel("Recall (positive class)")
    axis.set_ylabel("Precision (positive class)")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Precision-recall curves — held-out test partition")
    axis.legend(loc="lower left")
    if not drawn:
        axis.text(
            0.5,
            0.5,
            "Precision-recall curves unavailable\n(no positive rows or failed evaluation)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    return _save(figure, path)


def plot_metric_comparison(
    families: Mapping[str, Mapping[str, Any]], path: Path
) -> Path:
    """Grouped bars for threshold-dependent and threshold-free metrics."""
    plt = _matplotlib()
    family_names = [name for name in families if families[name].get("test_metrics")]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    threshold_metrics = ["accuracy", "precision", "recall", "f1"]
    auc_metrics = ["roc_auc", "pr_auc"]
    width = 0.8 / max(len(family_names), 1)
    for index, family in enumerate(family_names):
        block = families[family]
        metrics = block.get("test_metrics") or {}
        threshold = (block.get("threshold") or {}).get("selected")
        offset = (index - (len(family_names) - 1) / 2) * width
        values = [metrics.get(name) for name in threshold_metrics]
        positions = np.arange(len(threshold_metrics)) + offset
        axis = axes[0]
        for position, value in zip(positions, values):
            if value is None:
                continue
            axis.bar(
                position,
                float(value),
                width=width * 0.92,
                color=COLOURS.get(family, _DEFAULT_COLOUR),
                label=FAMILY_LABELS.get(family, family) if position == positions[0] else None,
            )
        axis = axes[1]
        values = [metrics.get(name) for name in auc_metrics]
        positions = np.arange(len(auc_metrics)) + offset
        for position, value in zip(positions, values):
            if value is None:
                continue
            axis.bar(
                position,
                float(value),
                width=width * 0.92,
                color=COLOURS.get(family, _DEFAULT_COLOUR),
                label=(
                    f"{FAMILY_LABELS.get(family, family)} "
                    f"(threshold {threshold:.3f})"
                    if threshold is not None
                    else FAMILY_LABELS.get(family, family)
                )
                if position == positions[0]
                else None,
            )

    axes[0].set_xticks(np.arange(len(threshold_metrics)), threshold_metrics)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification metrics at the frozen threshold")
    axes[1].set_xticks(np.arange(len(auc_metrics)), auc_metrics)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Threshold-free ranking metrics")
    for axis in axes:
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle("ChurnWise AI — held-out test comparison", fontsize=12)
    return _save(figure, path)


def plot_threshold_analysis(
    sweeps: Mapping[str, Mapping[str, Any]], path: Path
) -> Path:
    """Precision/recall/F1 against thresholds using validation predictions."""
    plt = _matplotlib()
    family_names = [name for name in sweeps if (sweeps[name].get("sweep") or [])]
    if not family_names:
        family_names = ["_empty"]
    figure, axes = plt.subplots(
        1, len(family_names), figsize=(4.6 * len(family_names), 4.2), squeeze=False
    )
    for axis, family in zip(axes[0], family_names):
        payload = sweeps.get(family) or {}
        rows = payload.get("sweep") or []
        if not rows:
            axis.text(
                0.5,
                0.5,
                "threshold sweep unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(family)
            continue
        thresholds = [row["threshold"] for row in rows]
        for name, colour in (
            ("precision", "#1f77b4"),
            ("recall", "#d62728"),
            ("f1", "#2ca02c"),
        ):
            values = [row.get(name) for row in rows]
            axis.plot(
                thresholds,
                [np.nan if value is None else value for value in values],
                label=name,
                color=colour,
                linewidth=1.6,
            )
        selected = payload.get("selected_threshold")
        if selected is not None:
            axis.axvline(
                float(selected),
                color="#444444",
                linestyle="--",
                linewidth=1.2,
                label=f"selected {float(selected):.3f}",
            )
        axis.axvline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Validation probability threshold")
        axis.set_ylabel("Score (positive class)")
        axis.set_title(FAMILY_LABELS.get(family, family))
        axis.legend(loc="lower left", fontsize=8)
    figure.suptitle("Threshold analysis on validation predictions", fontsize=12)
    return _save(figure, path)


def plot_probability_distributions(
    predictions: Mapping[str, Mapping[str, Any]],
    labels: Any,
    path: Path,
) -> Path:
    """Probability histograms by true class (aggregate diagnostics only)."""
    plt = _matplotlib()
    family_names = list(predictions)
    values = np.asarray(labels).reshape(-1).astype(int)
    figure, axes = plt.subplots(
        1, max(len(family_names), 1), figsize=(4.4 * max(len(family_names), 1), 3.8),
        squeeze=False,
    )
    bins = np.linspace(0.0, 1.0, 21)
    for axis, family in zip(axes[0], family_names or ["_empty"]):
        probabilities = predictions.get(family)
        if probabilities is None:
            axis.text(
                0.5,
                0.5,
                "probabilities unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(FAMILY_LABELS.get(family, family))
            continue
        scores = np.asarray(probabilities, dtype=float).reshape(-1)
        axis.hist(
            scores[values == 0],
            bins=bins,
            alpha=0.65,
            label="not churned (0)",
            color="#1f77b4",
        )
        axis.hist(
            scores[values == 1],
            bins=bins,
            alpha=0.65,
            label="churned (1)",
            color="#d62728",
        )
        axis.set_xlabel("Predicted P(churn)")
        axis.set_ylabel("Customers")
        axis.set_title(FAMILY_LABELS.get(family, family))
        axis.legend(fontsize=8)
    figure.suptitle("Test probability distributions by true class", fontsize=12)
    return _save(figure, path)


__all__ = [
    "plot_confusion_matrix",
    "plot_metric_comparison",
    "plot_precision_recall_curves",
    "plot_probability_distributions",
    "plot_roc_curves",
    "plot_threshold_analysis",
]
