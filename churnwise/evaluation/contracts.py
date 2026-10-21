"""Stable Phase 7 contracts: benchmark configuration and result handles.

The benchmark writes JSON/NPZ artifacts; :class:`BenchmarkResult` is the
in-memory handle used by the CLI, tests and later phases. Every number in
``metrics`` is measured on the partitions named in its own payload.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from churnwise.models.contracts import (
    LOGISTIC_FAMILY,
    MLP_FAMILY,
    RANDOM_FOREST_FAMILY,
    SUPPORTED_MODEL_FAMILIES,
)

EVALUATION_SCHEMA_VERSION = "1.0"

SUPPORTED_FAMILIES = SUPPORTED_MODEL_FAMILIES

FAMILY_LABELS = {
    LOGISTIC_FAMILY: "Logistic Regression",
    RANDOM_FOREST_FAMILY: "Random Forest",
    MLP_FAMILY: "MLP Neural Network",
}

THRESHOLD_STRATEGIES = ("fixed", "max_f1", "max_youden", "recall_floor")
SELECTION_METRICS = ("f1", "roc_auc", "pr_auc", "recall", "balanced_accuracy", "accuracy")

#: Bounds that keep a student laptop responsive while still comparable.
MAX_CV_FOLDS = 10


@dataclass(frozen=True)
class EvaluationConfig:
    """Every choice that affects thresholds, selection or reported metrics.

    ``threshold_strategy`` is applied to **validation** predictions only. The
    default 0.5 threshold is always reported as a secondary view so a reader can
    see what a naive cut-off would have produced.
    """

    seed: int = 42
    threshold_strategy: str = "max_f1"
    fixed_threshold: float = 0.5
    recall_floor: float | None = None
    selection_metric: str = "f1"
    selection_min_recall: float | None = None
    cv_folds: int = 5
    version: str = EVALUATION_SCHEMA_VERSION

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            problems.append("seed must be a non-negative integer")
        if self.threshold_strategy not in THRESHOLD_STRATEGIES:
            problems.append(
                f"threshold_strategy must be one of {list(THRESHOLD_STRATEGIES)}"
            )
        if not 0.0 <= float(self.fixed_threshold) <= 1.0:
            problems.append("fixed_threshold must be in [0, 1]")
        if self.recall_floor is not None and not 0.0 < float(self.recall_floor) <= 1.0:
            problems.append("recall_floor must be in (0, 1]")
        if self.threshold_strategy == "recall_floor" and self.recall_floor is None:
            problems.append("recall_floor strategy requires --recall-floor")
        if self.selection_metric not in SELECTION_METRICS:
            problems.append(f"selection_metric must be one of {list(SELECTION_METRICS)}")
        if self.selection_min_recall is not None and not (
            0.0 < float(self.selection_min_recall) <= 1.0
        ):
            problems.append("selection_min_recall must be in (0, 1]")
        if (
            not isinstance(self.cv_folds, int)
            or isinstance(self.cv_folds, bool)
            or self.cv_folds < 0
            or self.cv_folds > MAX_CV_FOLDS
        ):
            problems.append(f"cv_folds must be an integer between 0 and {MAX_CV_FOLDS}")
        elif self.cv_folds == 1:
            problems.append("cv_folds must be 0 (disabled) or at least 2")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "threshold_strategy": self.threshold_strategy,
            "fixed_threshold": float(self.fixed_threshold),
            "recall_floor": (
                None if self.recall_floor is None else float(self.recall_floor)
            ),
            "selection_metric": self.selection_metric,
            "selection_min_recall": (
                None
                if self.selection_min_recall is None
                else float(self.selection_min_recall)
            ),
            "cv_folds": self.cv_folds,
            "version": self.version,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint()}


@dataclass
class BenchmarkResult:
    """A persisted (or in-memory) benchmark and its structured metrics."""

    benchmark_run_id: str
    metrics: dict[str, Any]
    paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def families(self) -> dict[str, Any]:
        return dict(self.metrics.get("families") or {})

    @property
    def selected_family(self) -> str | None:
        return (self.metrics.get("selection") or {}).get("selected_family")

    @property
    def complete(self) -> bool:
        return bool(self.metrics.get("complete"))

    @property
    def n_test(self) -> int:
        return int((self.metrics.get("dataset") or {}).get("n_test") or 0)

    def family(self, family: str) -> dict[str, Any]:
        """Return one family's measured block; fail loudly when it is absent."""
        payload = self.families.get(family)
        if payload is None:
            raise KeyError(f"family {family!r} is not part of this benchmark")
        return payload

    def test_metrics(self, family: str) -> dict[str, Any]:
        return dict(self.family(family).get("test_metrics") or {})

    def threshold(self, family: str) -> float | None:
        value = (self.family(family).get("threshold") or {}).get("selected")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_run_id": self.benchmark_run_id,
            "metrics": self.metrics,
            "paths": dict(self.paths),
            "warnings": list(self.warnings),
        }


__all__ = [
    "BenchmarkResult",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationConfig",
    "FAMILY_LABELS",
    "LOGISTIC_FAMILY",
    "MAX_CV_FOLDS",
    "MLP_FAMILY",
    "RANDOM_FOREST_FAMILY",
    "SELECTION_METRICS",
    "SUPPORTED_FAMILIES",
    "THRESHOLD_STRATEGIES",
]
