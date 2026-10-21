"""Validation-only threshold selection for the Phase 7 benchmark.

A threshold is selected from validation predictions, frozen, and only then
applied to the test partition. The selection table is persisted so a reader can
audit the choice; the test labels are never an input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from churnwise.evaluation.contracts import THRESHOLD_STRATEGIES
from churnwise.evaluation.errors import (
    CODE_INVALID_CONFIG,
    CODE_METRIC_FAILED,
    EvaluationError,
)
from churnwise.evaluation.metrics import (
    binary_metrics,
    threshold_grid,
    threshold_sweep,
    validate_threshold,
)


@dataclass(frozen=True)
class ThresholdChoice:
    """One frozen operating threshold and the evidence behind it."""

    threshold: float
    strategy: str
    selected_on: str = "validation"
    recall_floor: float | None = None
    objective: str = ""
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": float(self.threshold),
            "strategy": self.strategy,
            "selected_on": self.selected_on,
            "recall_floor": self.recall_floor,
            "objective": self.objective,
            "validation_metrics": dict(self.validation_metrics),
            "notes": list(self.notes),
        }


def _best_row(
    rows: Sequence[dict[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    """Pick the best row deterministically; ties go to the threshold nearest 0.5."""
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None

    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (float(row[metric]), -abs(float(row["threshold"]) - 0.5))

    return max(candidates, key=key)


def select_threshold(
    y_true: Any,
    probabilities: Any,
    *,
    strategy: str = "max_f1",
    fixed_threshold: float = 0.5,
    recall_floor: float | None = None,
    grid: Sequence[float] | None = None,
) -> ThresholdChoice:
    """Choose an operating threshold using validation predictions only.

    Strategies:

    ``fixed``
        Use ``fixed_threshold`` verbatim (0.5 by default).
    ``max_f1``
        Maximise positive-class F1 on the validation sweep.
    ``max_youden``
        Maximise ``recall + specificity - 1`` (Youden's J).
    ``recall_floor``
        Among thresholds with ``recall >= recall_floor``, choose the highest
        threshold, which maximises specificity while meeting the recall floor.
        If no threshold qualifies, fall back to ``max_f1`` and record why.
    """
    if strategy not in THRESHOLD_STRATEGIES:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"unknown threshold strategy {strategy!r}",
            f"Use one of {list(THRESHOLD_STRATEGIES)}.",
        )
    value = validate_threshold(fixed_threshold)
    if strategy == "recall_floor" and recall_floor is None:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            "recall_floor strategy requires a recall floor",
            "Pass --recall-floor between 0 and 1.",
        )
    if recall_floor is not None and not 0.0 < float(recall_floor) <= 1.0:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"recall_floor must be in (0, 1], got {recall_floor!r}",
            "Pass --recall-floor between 0 and 1.",
        )
    if strategy == "fixed":
        chosen = value
        objective = "fixed operating threshold"
        notes = [
            "The 0.5 default is an arbitrary probability cut-off, not a validated "
            "business decision."
            if abs(value - 0.5) < 1e-12
            else f"A fixed threshold of {value:.4f} was configured in advance."
        ]
    else:
        rows = threshold_sweep(y_true, probabilities, grid or threshold_grid())
        notes = []
        if strategy == "recall_floor":
            floor = float(recall_floor)  # validated above
            qualifying = [
                row
                for row in rows
                if row.get("recall") is not None and float(row["recall"]) >= floor
            ]
            if qualifying:
                best = max(qualifying, key=lambda row: float(row["threshold"]))
                chosen = float(best["threshold"])
                objective = f"highest threshold with validation recall >= {floor:.2f}"
            else:
                best = _best_row(rows, "f1")
                if best is None:
                    raise EvaluationError(
                        CODE_METRIC_FAILED,
                        "no threshold produced a defined F1 on validation",
                        "Check the validation labels and model probabilities.",
                    )
                chosen = float(best["threshold"])
                objective = "fallback to maximum validation F1"
                notes.append(
                    f"no threshold reached the recall floor {floor:.2f}; the "
                    "validation F1 optimum was used instead"
                )
        else:
            metric = "f1" if strategy == "max_f1" else "youden"
            best = _best_row(rows, metric)
            if best is None:
                raise EvaluationError(
                    CODE_METRIC_FAILED,
                    f"no threshold produced a defined {metric} on validation",
                    "Check the validation labels and model probabilities.",
                )
            chosen = float(best["threshold"])
            objective = (
                "maximum validation positive-class F1"
                if strategy == "max_f1"
                else "maximum validation Youden J (recall + specificity - 1)"
            )
        notes.append(
            "The threshold was chosen on validation predictions and frozen before "
            "the test partition was evaluated."
        )
        notes.append(
            "ROC-AUC ranks across all thresholds; the threshold above is a separate "
            "operating point, not the AUC."
        )
    metrics = binary_metrics(y_true, probabilities, chosen)
    return ThresholdChoice(
        threshold=float(chosen),
        strategy=strategy,
        recall_floor=None if recall_floor is None else float(recall_floor),
        objective=objective,
        validation_metrics=metrics,
        notes=notes,
    )


def threshold_analysis(
    y_true: Any,
    probabilities: Any,
    choice: ThresholdChoice,
    *,
    grid: Sequence[float] | None = None,
    points: int = 99,
) -> dict[str, Any]:
    """Sweep rows plus the frozen choice, ready to persist as JSON."""
    rows = threshold_sweep(
        y_true, probabilities, grid or threshold_grid(points)
    )
    return {
        "selected_on": "validation",
        "strategy": choice.strategy,
        "selected_threshold": float(choice.threshold),
        "objective": choice.objective,
        "recall_floor": choice.recall_floor,
        "default_threshold": 0.5,
        "grid_points": len(rows),
        "sweep": rows,
        "notes": list(choice.notes),
        "default_threshold_row": _row_at(rows, 0.5),
    }


def _row_at(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    for row in rows:
        if abs(float(row["threshold"]) - float(threshold)) < 1e-9:
            return dict(row)
    nearest = min(rows, key=lambda row: abs(float(row["threshold"]) - threshold))
    payload = dict(nearest)
    payload["note"] = "closest grid point to the requested threshold"
    return payload


def threshold_table_rows(
    choices: dict[str, ThresholdChoice],
) -> list[dict[str, Any]]:
    """Compact comparison of frozen thresholds across families."""
    rows: list[dict[str, Any]] = []
    for family, choice in choices.items():
        metrics = choice.validation_metrics
        rows.append(
            {
                "family": family,
                "threshold": float(choice.threshold),
                "strategy": choice.strategy,
                "validation_f1": metrics.get("f1"),
                "validation_recall": metrics.get("recall"),
                "validation_precision": metrics.get("precision"),
            }
        )
    return rows


__all__ = [
    "ThresholdChoice",
    "select_threshold",
    "threshold_analysis",
    "threshold_table_rows",
]
