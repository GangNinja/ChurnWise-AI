"""Reusable MLP inference for transformed matrices and raw customer records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from churnwise.config import Settings
from churnwise.models.contracts import Prediction, PredictionBatch
from churnwise.models.errors import CODE_PREDICTION_FAILED, ModelingError
from churnwise.models.registry import LoadedModel, ModelRegistry, validate_model_run_id


class ChurnPredictor:
    """Predict churn probabilities with a verified Phase 6 model.

    ``dataset`` is optional for matrix predictions and required for raw
    DataFrames; it supplies the exact fitted Phase 5 preprocessing pipeline.
    """

    def __init__(
        self,
        model: Any,
        metadata: dict[str, Any],
        *,
        dataset: Any | None = None,
        threshold: float | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.dataset = dataset
        default_threshold = float(
            (metadata.get("training") or {}).get("threshold", 0.5)
        )
        self.threshold = float(default_threshold if threshold is None else threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"threshold must be in [0, 1], got {self.threshold!r}",
                "Pass a probability threshold between 0 and 1.",
            )
        self.positive_label = (
            (metadata.get("preprocessing") or {}).get("target") or {}
        ).get("positive_label")
        self.feature_names = list(
            (metadata.get("preprocessing") or {}).get("feature_names") or []
        )
        if not self.feature_names:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "model metadata has no transformed feature names",
                "Regenerate the model metadata from the Phase 5 run.",
            )
        input_shape = getattr(self.model, "input_shape", None)
        if input_shape:
            try:
                model_width = int(input_shape[-1])
            except (TypeError, ValueError, IndexError):
                model_width = None
            if model_width is not None and model_width != len(self.feature_names):
                raise ModelingError(
                    CODE_PREDICTION_FAILED,
                    f"model input width {model_width} does not match "
                    f"{len(self.feature_names)} metadata feature names",
                    "Reload the model and metadata from the same training run.",
                )

    @classmethod
    def load(
        cls,
        settings: Settings,
        model_run_id: str,
        *,
        threshold: float | None = None,
        with_preprocessing: bool = True,
    ) -> "ChurnPredictor":
        validate_model_run_id(model_run_id)
        loaded: LoadedModel = ModelRegistry(settings).load(model_run_id)
        dataset = None
        if with_preprocessing:
            preprocessing_run_id = str(
                (loaded.metadata.get("preprocessing") or {}).get("run_id", "")
            )
            if not preprocessing_run_id:
                raise ModelingError(
                    CODE_PREDICTION_FAILED,
                    "model metadata has no preprocessing run id",
                    "Regenerate the model with the Phase 5 hand-off.",
                )
            from churnwise.preprocessing import PreprocessingCoordinator, PreprocessingError

            try:
                dataset = PreprocessingCoordinator(settings).load(preprocessing_run_id)
            except PreprocessingError as exc:
                raise ModelingError(
                    CODE_PREDICTION_FAILED,
                    f"Phase 5 preprocessing could not be loaded: {exc.message}",
                    exc.remediation,
                ) from exc
        return cls(
            loaded.model,
            loaded.metadata,
            dataset=dataset,
            threshold=threshold,
        )

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return P(churn) for a transformed matrix, preserving row order."""
        matrix = self._matrix(features)
        probabilities = np.asarray(self.model.predict(matrix, verbose=0), dtype=float).reshape(-1)
        if len(probabilities) != len(matrix):
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"model returned {len(probabilities)} probabilities for {len(matrix)} rows",
                "Reload the model and verify its input feature count.",
            )
        if not np.isfinite(probabilities).all():
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "model produced non-finite probabilities",
                "Reload a verified model artifact or retrain the model.",
            )
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "model probabilities fall outside [0, 1]",
                "Check the model output activation; it must be sigmoid.",
            )
        return np.clip(probabilities, 0.0, 1.0)

    def predict_frame(self, frame: pd.DataFrame) -> PredictionBatch:
        """Transform raw customer records with Phase 5 and predict churn."""
        if not isinstance(frame, pd.DataFrame):
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "predict_frame expects a pandas DataFrame",
                "Build a DataFrame with the schema.required_columns fields.",
            )
        if self.dataset is None:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "raw-record prediction requires the fitted Phase 5 preprocessing pipeline",
                "Load the predictor with with_preprocessing=True.",
            )
        try:
            matrix = self.dataset.transform_new(frame)
        except Exception as exc:  # PreprocessingError carries the actionable message
            message = getattr(exc, "message", str(exc))
            remediation = getattr(exc, "remediation", "")
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"raw records could not be transformed: {message}",
                remediation or "Provide every column required by the Phase 5 schema.",
            ) from exc
        probabilities = self.predict_proba(matrix)
        return self._batch(probabilities)

    def predict(self, values: Any) -> PredictionBatch:
        """Dispatch to raw-record or transformed-matrix prediction."""
        if isinstance(values, pd.DataFrame):
            return self.predict_frame(values)
        return self._batch(self.predict_proba(values))

    def predict_one(self, record: Mapping[str, Any] | pd.Series) -> Prediction:
        """Predict one customer from a raw record mapping/Series."""
        if isinstance(record, pd.Series):
            frame = record.to_frame().T
        elif isinstance(record, Mapping):
            frame = pd.DataFrame([dict(record)])
        else:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "predict_one expects a mapping or pandas Series",
                "Pass one raw customer record with the required schema columns.",
            )
        batch = self.predict_frame(frame)
        return Prediction(
            probability=float(batch.probabilities[0]),
            predicted_class=int(batch.classes[0]),
            threshold=float(self.threshold),
            positive_label=self.positive_label,
        )

    # -- helpers -----------------------------------------------------------
    def _matrix(self, values: Any) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            if self.dataset is None:
                raise ModelingError(
                    CODE_PREDICTION_FAILED,
                    "raw-record prediction requires the fitted Phase 5 preprocessing pipeline",
                    "Load the predictor with with_preprocessing=True.",
                )
            try:
                values = self.dataset.transform_new(values)
            except Exception as exc:
                message = getattr(exc, "message", str(exc))
                remediation = getattr(exc, "remediation", "")
                raise ModelingError(
                    CODE_PREDICTION_FAILED,
                    f"raw records could not be transformed: {message}",
                    remediation or "Provide every column required by the Phase 5 schema.",
                ) from exc
        try:
            matrix = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"features are not a numeric matrix: {type(exc).__name__}: {exc}",
                "Pass numeric transformed features or raw records through predict_frame.",
            ) from exc
        if matrix.ndim != 2:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"features must be a 2-D matrix, got shape {matrix.shape}",
                "Pass one row per customer with all transformed features.",
            )
        if matrix.shape[1] != len(self.feature_names):
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"features have {matrix.shape[1]} columns; model expects {len(self.feature_names)}",
                "Use the fitted Phase 5 transform and the model's feature_names.",
            )
        if not np.isfinite(matrix).all():
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                "features contain non-finite values",
                "Clean or impute the input features before prediction.",
            )
        return matrix

    def _batch(self, probabilities: np.ndarray) -> PredictionBatch:
        classes = (probabilities >= self.threshold).astype(int)
        return PredictionBatch(
            probabilities=probabilities,
            classes=classes,
            threshold=float(self.threshold),
            positive_label=self.positive_label,
            row_positions=list(range(len(probabilities))),
        )


__all__ = ["ChurnPredictor"]
