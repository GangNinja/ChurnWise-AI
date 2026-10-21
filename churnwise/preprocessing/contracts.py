"""Stable Phase 5 contracts: preprocessing configuration and split identity.

Every value that changes the transformed output is part of
:class:`PreprocessConfig` and therefore part of its fingerprint. The run id
embeds the first eight characters of that fingerprint, exactly as the
architecture specifies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "1.0"

IMBALANCE_STRATEGIES = ("class_weight", "oversample", "none")
OUTLIER_POLICIES = ("report_only",)


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic, JSON-serialisable preprocessing settings."""

    seed: int = 42
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    test_fraction: float = 0.20
    imbalance: str = "class_weight"
    oversample_ratio: float = 1.0
    numeric_imputation: str = "median"
    categorical_imputation: str = "missing_category"
    scaling: str = "standard"
    missing_indicator_threshold: float = 0.01
    max_categorical_levels: int = 30
    min_frequency: float = 0.0
    derived_features: bool = True
    drop_duplicate_rows: bool = True
    min_rows: int = 100
    min_class_count: int = 2
    numeric_coerce_threshold: float = 0.98
    outlier_policy: str = "report_only"
    version: str = PIPELINE_VERSION

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.seed < 0:
            problems.append("seed must be non-negative")
        fractions = (self.train_fraction, self.val_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions):
            problems.append("split fractions must all be positive")
        if abs(sum(fractions) - 1.0) > 1e-6:
            problems.append("split fractions must sum to 1.0")
        if self.imbalance not in IMBALANCE_STRATEGIES:
            problems.append(f"imbalance must be one of {list(IMBALANCE_STRATEGIES)}")
        if not (0.0 < self.oversample_ratio <= 1.0):
            problems.append("oversample_ratio must be in (0, 1]")
        if self.numeric_imputation != "median":
            problems.append("only median numeric imputation is implemented")
        if self.categorical_imputation not in {"missing_category", "most_frequent"}:
            problems.append("categorical_imputation must be missing_category or most_frequent")
        if self.scaling != "standard":
            problems.append("only standard scaling is implemented")
        if not (0.0 <= self.missing_indicator_threshold < 1.0):
            problems.append("missing_indicator_threshold must be in [0, 1)")
        if not (2 <= self.max_categorical_levels <= 500):
            problems.append("max_categorical_levels must be between 2 and 500")
        if not (0.0 <= self.min_frequency < 1.0):
            problems.append("min_frequency must be in [0, 1)")
        if self.min_rows < 20:
            problems.append("min_rows must be at least 20")
        if not (2 <= self.min_class_count <= 10_000):
            problems.append("min_class_count must be at least 2")
        if not (0.5 <= self.numeric_coerce_threshold <= 1.0):
            problems.append("numeric_coerce_threshold must be in [0.5, 1.0]")
        if self.outlier_policy not in OUTLIER_POLICIES:
            problems.append(f"outlier_policy must be one of {list(OUTLIER_POLICIES)}")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "imbalance": self.imbalance,
            "oversample_ratio": self.oversample_ratio,
            "numeric_imputation": self.numeric_imputation,
            "categorical_imputation": self.categorical_imputation,
            "scaling": self.scaling,
            "missing_indicator_threshold": self.missing_indicator_threshold,
            "max_categorical_levels": self.max_categorical_levels,
            "min_frequency": self.min_frequency,
            "derived_features": self.derived_features,
            "drop_duplicate_rows": self.drop_duplicate_rows,
            "min_rows": self.min_rows,
            "min_class_count": self.min_class_count,
            "numeric_coerce_threshold": self.numeric_coerce_threshold,
            "outlier_policy": self.outlier_policy,
            "version": self.version,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint()}


@dataclass
class SplitSpec:
    """Reproducible split identity shared by every downstream consumer."""

    run_id: str
    dataset_id: str
    dataset_sha256: str
    strategy: str
    seed: int
    ratios: dict[str, float]
    sizes: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    class_rates: dict[str, float | None]
    fingerprint: str
    row_indices: dict[str, list[int]]
    constraints: dict[str, Any] = field(
        default_factory=lambda: {"test_touched": False, "fitted_on": "train_only"}
    )
    duplicate_rows_dropped: int = 0
    target_missing_dropped: int = 0
    id_column: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "strategy": self.strategy,
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "sizes": dict(self.sizes),
            "class_counts": {key: dict(value) for key, value in self.class_counts.items()},
            "class_rates": dict(self.class_rates),
            "fingerprint": self.fingerprint,
            "constraints": dict(self.constraints),
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "target_missing_dropped": self.target_missing_dropped,
            "id_column": self.id_column,
            "row_indices": {key: list(value) for key, value in self.row_indices.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitSpec":
        return cls(
            run_id=str(payload["run_id"]),
            dataset_id=str(payload["dataset_id"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            strategy=str(payload["strategy"]),
            seed=int(payload["seed"]),
            ratios={k: float(v) for k, v in payload["ratios"].items()},
            sizes={k: int(v) for k, v in payload["sizes"].items()},
            class_counts={
                name: {str(label): int(count) for label, count in counts.items()}
                for name, counts in payload["class_counts"].items()
            },
            class_rates={k: (None if v is None else float(v)) for k, v in payload["class_rates"].items()},
            fingerprint=str(payload["fingerprint"]),
            row_indices={k: [int(i) for i in v] for k, v in payload["row_indices"].items()},
            constraints=dict(payload.get("constraints") or {}),
            duplicate_rows_dropped=int(payload.get("duplicate_rows_dropped", 0)),
            target_missing_dropped=int(payload.get("target_missing_dropped", 0)),
            id_column=payload.get("id_column"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class SplitData:
    """One prepared partition; arrays are float64 features and int8 labels."""

    name: str
    X: Any
    y: Any
    row_indices: list[int]

    @property
    def n_rows(self) -> int:
        return int(len(self.y))

    @property
    def positive_rate(self) -> float | None:
        if self.n_rows == 0:
            return None
        return float(self.y.mean())


@dataclass
class ClassBalance:
    """Class counts and imbalance ratios for the full data and each split."""

    counts: dict[str, dict[str, int]]
    rates: dict[str, float | None]
    strategy: str
    class_weights: dict[str, float]
    ratio: float | None
    resampled: bool = False
    resampled_train_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {split: dict(counts) for split, counts in self.counts.items()},
            "rates": dict(self.rates),
            "strategy": self.strategy,
            "class_weights": dict(self.class_weights),
            "ratio": self.ratio,
            "resampled": self.resampled,
            "resampled_train_rows": self.resampled_train_rows,
        }
