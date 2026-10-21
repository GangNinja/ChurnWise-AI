"""Stable Phase 6 contracts: MLP configuration and prediction records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

MODEL_SCHEMA_VERSION = "1.0"
MODEL_FAMILY = "mlp"

# Shared model-family identifiers used by Phase 7 benchmarking.
MLP_FAMILY = MODEL_FAMILY
LOGISTIC_FAMILY = "logistic_regression"
RANDOM_FOREST_FAMILY = "random_forest"
SUPPORTED_MODEL_FAMILIES = (LOGISTIC_FAMILY, RANDOM_FOREST_FAMILY, MLP_FAMILY)


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters that define the trained MLP and its inference defaults.

    The defaults follow the audited architecture: two ReLU hidden layers
    (64/32), dropout (0.30/0.20), L2 1e-4, Adam 1e-3, batch 64 and at most 150
    epochs with validation-loss early stopping.
    """

    seed: int = 42
    hidden_units: tuple[int, ...] = (64, 32)
    dropout: tuple[float, ...] = (0.30, 0.20)
    l2: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 150
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-5
    threshold: float = 0.5
    deterministic: bool = False
    shuffle: bool = True
    verbose: int = 0

    def validate(self) -> list[str]:
        problems: list[str] = []
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            problems.append("seed must be a non-negative integer")

        # Validate container types before using len()/iteration, so a malformed
        # Python configuration produces a ModelingError instead of a TypeError.
        hidden_units = (
            self.hidden_units
            if isinstance(self.hidden_units, (tuple, list))
            else None
        )
        if hidden_units is None or not hidden_units:
            problems.append("hidden_units must contain between 1 and 8 layers")
        elif len(hidden_units) > 8:
            problems.append("hidden_units must contain between 1 and 8 layers")
        elif any(
            not isinstance(units, int)
            or isinstance(units, bool)
            or units < 1
            or units > 4096
            for units in hidden_units
        ):
            problems.append("hidden_units must be integers between 1 and 4096")

        dropout = (
            self.dropout if isinstance(self.dropout, (tuple, list)) else None
        )
        if dropout is None:
            problems.append("dropout must be a sequence with one rate per hidden layer")
        elif hidden_units is not None and len(dropout) != len(hidden_units):
            problems.append("dropout must have one rate per hidden layer")
        elif dropout:
            try:
                bad_rate = any(not 0.0 <= float(rate) < 1.0 for rate in dropout)
            except (TypeError, ValueError):
                problems.append("dropout rates must be numeric values in [0, 1)")
            else:
                if bad_rate:
                    problems.append("dropout rates must be in [0, 1)")

        if not 0.0 <= float(self.l2):
            problems.append("l2 must be non-negative")
        if not 0.0 < float(self.learning_rate):
            problems.append("learning_rate must be positive")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            problems.append("batch_size must be a positive integer")
        if not isinstance(self.epochs, int) or self.epochs < 1:
            problems.append("epochs must be a positive integer")
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 0:
            problems.append("early_stopping_patience must be a non-negative integer")
        if not isinstance(self.reduce_lr_patience, int) or self.reduce_lr_patience < 0:
            problems.append("reduce_lr_patience must be a non-negative integer")
        if not 0.0 < float(self.reduce_lr_factor) <= 1.0:
            problems.append("reduce_lr_factor must be in (0, 1]")
        if not 0.0 <= float(self.min_learning_rate) <= float(self.learning_rate):
            problems.append("min_learning_rate must be in [0, learning_rate]")
        if not 0.0 <= float(self.threshold) <= 1.0:
            problems.append("threshold must be in [0, 1]")
        if isinstance(self.verbose, bool) or self.verbose not in (0, 1, 2):
            problems.append("verbose must be 0, 1 or 2")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "hidden_units": list(self.hidden_units),
            "dropout": list(self.dropout),
            "l2": float(self.l2),
            "learning_rate": float(self.learning_rate),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "reduce_lr_factor": float(self.reduce_lr_factor),
            "min_learning_rate": float(self.min_learning_rate),
            "threshold": float(self.threshold),
            "deterministic": bool(self.deterministic),
            "shuffle": bool(self.shuffle),
            "verbose": self.verbose,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint()}

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class TrainingResult:
    """A trained model plus every record needed to audit or load it."""

    model: Any
    history: dict[str, list[float]]
    metadata: dict[str, Any]
    paths: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("model_run_id", ""))


@dataclass
class Prediction:
    """One customer's churn probability and thresholded class."""

    probability: float
    predicted_class: int
    threshold: float
    positive_label: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": float(self.probability),
            "predicted_class": int(self.predicted_class),
            "threshold": float(self.threshold),
            "positive_label": self.positive_label,
        }


@dataclass
class PredictionBatch:
    """Ordered predictions; positions preserve the caller's row order."""

    probabilities: Any
    classes: Any
    threshold: float
    positive_label: str | None
    row_positions: list[int]

    @property
    def n_rows(self) -> int:
        return len(self.row_positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "positive_label": self.positive_label,
            "row_positions": list(self.row_positions),
            "probabilities": [float(value) for value in self.probabilities],
            "classes": [int(value) for value in self.classes],
        }


def coerce_dropout(dropout: Sequence[float], hidden_units: Sequence[int]) -> tuple[float, ...]:
    """Allow one dropout rate to apply to every hidden layer."""
    values = tuple(float(value) for value in dropout)
    if len(values) == 1 and len(hidden_units) > 1:
        return values * len(hidden_units)
    return values
