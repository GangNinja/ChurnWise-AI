"""The Phase 5 hand-off object consumed by training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from churnwise.preprocessing.contracts import ClassBalance, PreprocessConfig, SplitData, SplitSpec
from churnwise.preprocessing.errors import CODE_TRANSFORM_FAILED, PreprocessingError
from churnwise.preprocessing.features import apply_derived
from churnwise.preprocessing.pipeline import clean_frame, plan_from_schema


@dataclass
class PreparedDataset:
    run_id: str
    dataset_id: str
    config: PreprocessConfig
    split_spec: SplitSpec
    schema: dict[str, Any]
    train: SplitData
    val: SplitData
    test: SplitData
    class_balance: ClassBalance
    preprocessor: Any
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    oversample_indices: list[int] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.schema.get("feature_names") or [])

    @property
    def feature_sources(self) -> dict[str, str]:
        return {
            item["name"]: item["source"]
            for item in self.schema.get("output_features", [])
        }

    @property
    def class_weights(self) -> dict[int, float]:
        return {int(key): float(value) for key, value in self.schema.get("class_weights", {}).items()}

    def resampled_train(self) -> SplitData:
        """Return the training partition with the configured oversampling.

        Validation and test partitions are never resampled. With the default
        ``class_weight`` strategy this returns the untouched training split.
        """
        if not self.oversample_indices:
            return self.train
        extra = np.asarray(self.oversample_indices, dtype=np.int64)
        X = np.concatenate([self.train.X, self.train.X[extra]], axis=0)
        y = np.concatenate([self.train.y, self.train.y[extra]], axis=0)
        rows = [*self.train.row_indices, *(self.train.row_indices[int(i)] for i in extra)]
        return SplitData("train_resampled", X, y, rows)

    def split(self, name: str) -> SplitData:
        if name == "train":
            return self.train
        if name in {"val", "validation"}:
            return self.val
        if name == "test":
            return self.test
        raise KeyError(f"unknown split {name!r}")

    def as_frame(self, name: str) -> pd.DataFrame:
        split = self.split(name)
        return pd.DataFrame(split.X, columns=self.feature_names)

    def transform_new(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform new customer records with the fitted, frozen pipeline."""
        if not isinstance(frame, pd.DataFrame):
            raise PreprocessingError(
                CODE_TRANSFORM_FAILED,
                "transform_new expects a pandas DataFrame",
                "Build a DataFrame with the columns listed in schema.required_columns.",
            )
        required = list(self.schema.get("required_columns") or [])
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise PreprocessingError(
                CODE_TRANSFORM_FAILED,
                f"inference frame is missing required columns: {', '.join(missing)}",
                "Provide every column listed in schema.required_columns.",
            )
        prepared = apply_derived(frame, self.schema.get("derived", []))
        plan = plan_from_schema(self.schema)
        matrix = np.asarray(
            self.preprocessor.transform(clean_frame(prepared, plan)), dtype=np.float64
        )
        expected = len(self.feature_names)
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise PreprocessingError(
                CODE_TRANSFORM_FAILED,
                f"transform_new produced shape {matrix.shape}; schema expects {expected} columns",
                "Reload the preprocessor and schema from the same run_id.",
            )
        if not np.isfinite(matrix).all():
            raise PreprocessingError(
                CODE_TRANSFORM_FAILED,
                "transform_new produced non-finite values",
                "Inspect the input values; infinite numeric values are treated as missing.",
            )
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "split_fingerprint": self.split_spec.fingerprint,
            "dataset_sha256": self.split_spec.dataset_sha256,
            "seed": self.split_spec.seed,
            "sizes": dict(self.split_spec.sizes),
            "class_balance": self.class_balance.to_dict(),
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_sources": self.feature_sources,
            "warnings": list(self.warnings),
            "oversampled_train_rows": len(self.oversample_indices),
            "paths": dict(self.paths),
            "provenance": dict(self.provenance),
            "constraints": dict(self.split_spec.constraints),
        }
