"""Stable Phase 8 contracts for explanations, sampling and provenance.

The module deliberately keeps two output spaces explicit:

``probability``
    SHAP values add up to the model's P(churn = 1). Tree and neural-network
    explainers used here operate in this space.
``log_odds``
    SHAP values add up to the Logistic Regression decision function
    (``log(p / (1 - p))``). A value in this space is *not* a probability
    change; :class:`LocalExplanation` reports the probability separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from churnwise.explain.errors import CODE_INVALID_CONFIG, ExplainError

EXPLAIN_SCHEMA_VERSION = "1.0"
OUTPUT_SPACE_PROBABILITY = "probability"
OUTPUT_SPACE_LOG_ODDS = "log_odds"
OUTPUT_SPACES = (OUTPUT_SPACE_PROBABILITY, OUTPUT_SPACE_LOG_ODDS)
SUPPORTED_FAMILIES = ("logistic_regression", "random_forest", "mlp")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ExplainConfig:
    """Bounds that keep explanations affordable on a CPU-only laptop.

    The defaults explain at most 200 validation rows for tree/linear models and
    60 for the MLP (whose permutation fallback is much slower). Backgrounds are
    drawn from training rows only and never from the test partition.
    """

    sample_size: int = 200
    background_size: int = 100
    mlp_sample_size: int = 60
    mlp_background_size: int = 50
    local_examples: int = 5
    top_k: int = 5
    seed: int = 42
    tolerance: float = 1e-4
    chunk_size: int = 100
    max_local_batch: int = 500
    permutation_max_evals: int = 500

    def validate(self) -> list[str]:
        problems: list[str] = []
        limits = {
            "sample_size": (self.sample_size, 1, 5000),
            "background_size": (self.background_size, 1, 1000),
            "mlp_sample_size": (self.mlp_sample_size, 1, 1000),
            "mlp_background_size": (self.mlp_background_size, 1, 500),
            "local_examples": (self.local_examples, 0, 100),
            "top_k": (self.top_k, 1, 50),
            "chunk_size": (self.chunk_size, 1, 1000),
            "max_local_batch": (self.max_local_batch, 1, 5000),
            "permutation_max_evals": (self.permutation_max_evals, 2, 5000),
        }
        for name, (value, low, high) in limits.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not low <= value <= high
            ):
                problems.append(f"{name} must be an integer between {low} and {high}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            problems.append("seed must be a non-negative integer")
        try:
            tolerance = float(self.tolerance)
        except (TypeError, ValueError):
            problems.append("tolerance must be a finite number in (0, 1]")
        else:
            if not math.isfinite(tolerance) or not 0.0 < tolerance <= 1.0:
                problems.append("tolerance must be a finite number in (0, 1]")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "sample_size": int(self.sample_size),
            "background_size": int(self.background_size),
            "mlp_sample_size": int(self.mlp_sample_size),
            "mlp_background_size": int(self.mlp_background_size),
            "local_examples": int(self.local_examples),
            "top_k": int(self.top_k),
            "seed": int(self.seed),
            "tolerance": float(self.tolerance),
            "chunk_size": int(self.chunk_size),
            "max_local_batch": int(self.max_local_batch),
            "permutation_max_evals": int(self.permutation_max_evals),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint()}

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def sample_limit(self, family: str) -> int:
        return int(self.mlp_sample_size if family == "mlp" else self.sample_size)

    def background_limit(self, family: str) -> int:
        return int(
            self.mlp_background_size if family == "mlp" else self.background_size
        )

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise ExplainError(
                CODE_INVALID_CONFIG,
                "invalid explanation configuration: " + "; ".join(problems),
                "Correct the explain options and retry.",
            )


@dataclass
class FeatureGroup:
    """One original feature and the transformed columns it produced."""

    source: str
    kind: str
    columns: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "columns": list(self.columns),
            "labels": list(self.labels),
            "positions": list(self.positions),
            "n_columns": len(self.columns),
        }


@dataclass
class RawExplanation:
    """Model-width SHAP values plus the evidence needed to audit them."""

    values: Any
    base_values: Any
    predictions: Any
    additivity_error: float
    output_space: str
    predictions_raw: Any | None = None
    explainer_name: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(len(self.predictions))


@dataclass
class Contribution:
    """One original feature's aggregated contribution for one customer."""

    feature: str
    kind: str
    contribution: float
    direction: str
    value: float | None = None
    active_level: str | None = None
    columns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "contribution": float(self.contribution),
            "direction": self.direction,
            "value": None if self.value is None else float(self.value),
            "active_level": self.active_level,
            "columns": [dict(item) for item in self.columns],
        }


@dataclass
class LocalExplanation:
    """One customer's prediction, drivers and data-grounded summary."""

    family: str
    output_space: str
    probability: float
    predicted_class: int
    threshold: float
    positive_label: str | None
    base_value: float
    model_output: float
    contributions: list[Contribution] = field(default_factory=list)
    top_positive: list[Contribution] = field(default_factory=list)
    top_negative: list[Contribution] = field(default_factory=list)
    summary: str = ""
    additivity_error: float = 0.0
    row_position: int | None = None
    plot_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "output_space": self.output_space,
            "probability": float(self.probability),
            "predicted_class": int(self.predicted_class),
            "threshold": float(self.threshold),
            "positive_label": self.positive_label,
            "base_value": float(self.base_value),
            "model_output": float(self.model_output),
            "contributions": [item.to_dict() for item in self.contributions],
            "top_positive": [item.to_dict() for item in self.top_positive],
            "top_negative": [item.to_dict() for item in self.top_negative],
            "summary": self.summary,
            "additivity_error": float(self.additivity_error),
            "row_position": self.row_position,
            "plot_path": self.plot_path,
            "warnings": list(self.warnings),
        }


def _json_float(value: Any) -> float | None:
    number = _finite(value)
    return number


@dataclass
class GlobalExplanation:
    """Dataset-level SHAP ranking plus sampling and validation evidence."""

    run_id: str
    family: str
    output_space: str
    generated_at: str = ""
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    base_value: float = 0.0
    sample: dict[str, Any] = field(default_factory=dict)
    additivity_error: float = 0.0
    explainer: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "phase": 8,
            "run_id": self.run_id,
            "family": self.family,
            "output_space": self.output_space,
            "generated_at": self.generated_at,
            "base_value": _json_float(self.base_value),
            "feature_importance": [
                {
                    "rank": int(item.get("rank", 0)),
                    "feature": item.get("feature"),
                    "kind": item.get("kind"),
                    "mean_abs_shap": _json_float(item.get("mean_abs_shap")),
                    "mean_shap": _json_float(item.get("mean_shap")),
                    "share": _json_float(item.get("share")),
                    "positive_share": _json_float(item.get("positive_share")),
                    "n_columns": int(item.get("n_columns", 0)),
                    "columns": list(item.get("columns") or []),
                    "direction": item.get("direction"),
                }
                for item in self.feature_importance
            ],
            "sample": dict(self.sample),
            "additivity_error": float(self.additivity_error),
            "explainer": dict(self.explainer),
            "provenance": dict(self.provenance),
            "plots": dict(self.plots),
            "warnings": list(self.warnings),
            "duration_seconds": float(self.duration_seconds),
            "config": dict(self.config),
        }

    def top_features(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.feature_importance[: max(1, int(limit))]


@dataclass
class ExplainResult:
    """A persisted explanation run: global ranking, local examples and files."""

    run_id: str
    family: str
    global_explanation: GlobalExplanation | None = None
    local_explanations: list[LocalExplanation] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.global_explanation is not None or bool(self.local_explanations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "phase": 8,
            "run_id": self.run_id,
            "family": self.family,
            "complete": self.complete,
            "global": (
                self.global_explanation.to_dict() if self.global_explanation else None
            ),
            "local": [item.to_dict() for item in self.local_explanations],
            "paths": dict(self.paths),
            "manifest": dict(self.manifest),
            "warnings": list(self.warnings),
        }


__all__ = [
    "EXPLAIN_SCHEMA_VERSION",
    "SUPPORTED_FAMILIES",
    "OUTPUT_SPACES",
    "OUTPUT_SPACE_LOG_ODDS",
    "OUTPUT_SPACE_PROBABILITY",
    "Contribution",
    "ExplainConfig",
    "ExplainResult",
    "FeatureGroup",
    "GlobalExplanation",
    "LocalExplanation",
    "RawExplanation",
]
