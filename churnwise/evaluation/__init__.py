"""Phase 7 comparative benchmarking and held-out evaluation.

Public entry points:

* :class:`BenchmarkCoordinator` - fit baselines, load the Phase 6 MLP, select
  thresholds on validation and evaluate the test partition once.
* :func:`evaluate` - one-call convenience wrapper.
* :class:`EvaluationConfig` - threshold, selection and CV settings.
* :class:`BenchmarkResult` - structured metrics plus artifact paths.
* :func:`binary_metrics` / :func:`threshold_sweep` / :func:`roc_curve_data` -
  reusable, audited metric primitives.
* :func:`select_threshold` - validation-only operating-threshold selection.
* :class:`EvaluationError` - stable ``CW-EVL-*`` failures with remediation.
"""

from __future__ import annotations

from churnwise.evaluation.benchmark import (
    BenchmarkCoordinator,
    evaluate,
    validate_benchmark_id,
)
from churnwise.evaluation.contracts import (
    EVALUATION_SCHEMA_VERSION,
    FAMILY_LABELS,
    LOGISTIC_FAMILY,
    MLP_FAMILY,
    RANDOM_FOREST_FAMILY,
    SUPPORTED_FAMILIES,
    BenchmarkResult,
    EvaluationConfig,
)
from churnwise.evaluation.cv import CrossValidationSummary, cross_validate
from churnwise.evaluation.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_FAMILY_FAILED,
    CODE_INCOMPATIBLE_INPUTS,
    CODE_INVALID_CONFIG,
    CODE_METRIC_FAILED,
    CODE_NO_MODEL,
    EvaluationError,
)
from churnwise.evaluation.metrics import (
    binary_metrics,
    confusion_counts,
    precision_recall_curve_data,
    roc_curve_data,
    threshold_grid,
    threshold_sweep,
)
from churnwise.evaluation.thresholds import (
    ThresholdChoice,
    select_threshold,
    threshold_analysis,
)

__all__ = [
    "CODE_ARTIFACT_FAILED",
    "CODE_FAMILY_FAILED",
    "CODE_INCOMPATIBLE_INPUTS",
    "CODE_INVALID_CONFIG",
    "CODE_METRIC_FAILED",
    "CODE_NO_MODEL",
    "EVALUATION_SCHEMA_VERSION",
    "FAMILY_LABELS",
    "LOGISTIC_FAMILY",
    "MLP_FAMILY",
    "RANDOM_FOREST_FAMILY",
    "SUPPORTED_FAMILIES",
    "BenchmarkCoordinator",
    "BenchmarkResult",
    "CrossValidationSummary",
    "EvaluationConfig",
    "EvaluationError",
    "ThresholdChoice",
    "binary_metrics",
    "confusion_counts",
    "cross_validate",
    "evaluate",
    "precision_recall_curve_data",
    "roc_curve_data",
    "select_threshold",
    "threshold_analysis",
    "threshold_grid",
    "threshold_sweep",
    "validate_benchmark_id",
]
