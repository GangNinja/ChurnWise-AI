"""Model construction, training, persistence and inference.

Phase 6 provides the Keras MLP; Phase 7 adds the scikit-learn baselines used by
the comparative benchmark.

Public entry points:

* :func:`build_mlp` - construct the Keras architecture.
* :class:`MLPConfig` - architecture and training hyperparameters.
* :class:`MLPTrainer` - fit on Phase 5 train/validation partitions.
* :class:`ModelRegistry` - trusted artifact persistence and reload.
* :class:`ChurnPredictor` - probability/class inference for matrices or raw rows.
* :class:`ModelingError` - stable ``CW-MOD-*`` failures with remediation.
* :func:`fit_baseline` - fit a documented Logistic Regression/Random Forest.
"""

from __future__ import annotations

from typing import Any

from churnwise.config import Settings
from churnwise.models.baselines import (
    FittedBaseline,
    LogisticRegressionConfig,
    RandomForestConfig,
    fit_baseline,
    load_baseline,
)
from churnwise.models.contracts import (
    MODEL_FAMILY,
    MODEL_SCHEMA_VERSION,
    MLPConfig,
    Prediction,
    PredictionBatch,
    TrainingResult,
)
from churnwise.models.errors import (
    CODE_INVALID_MODEL,
    CODE_MODEL_ARTIFACT,
    CODE_PREDICTION_FAILED,
    CODE_TRAINING_FAILED,
    ModelingError,
)
from churnwise.models.inference import ChurnPredictor
from churnwise.models.mlp import build_mlp
from churnwise.models.registry import LoadedModel, ModelRegistry
from churnwise.models.trainer import MLPTrainer

__all__ = [
    "CODE_INVALID_MODEL",
    "CODE_MODEL_ARTIFACT",
    "CODE_PREDICTION_FAILED",
    "CODE_TRAINING_FAILED",
    "MODEL_FAMILY",
    "MODEL_SCHEMA_VERSION",
    "ChurnPredictor",
    "FittedBaseline",
    "LoadedModel",
    "LogisticRegressionConfig",
    "RandomForestConfig",
    "MLPConfig",
    "MLPTrainer",
    "ModelRegistry",
    "ModelingError",
    "Prediction",
    "PredictionBatch",
    "TrainingResult",
    "build_mlp",
    "fit_baseline",
    "load_baseline",
    "train_model",
]


def train_model(
    dataset: Any,
    *,
    settings: Settings,
    config: MLPConfig | None = None,
    model_run_id: str | None = None,
    persist: bool = True,
) -> TrainingResult:
    """One-call convenience wrapper around :class:`MLPTrainer`."""
    return MLPTrainer(settings, config).train(
        dataset, model_run_id=model_run_id, persist=persist
    )
