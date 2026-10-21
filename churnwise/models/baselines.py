"""Phase 7 scikit-learn baselines: Logistic Regression and Random Forest.

Both estimators consume the *same* Phase 5 transformed matrix as the MLP, use the
same Phase 5 class weights and fixed, documented hyperparameters. They produce
``P(churn = 1)`` through ``predict_proba``. Nothing here tunes on the test split;
the benchmark only fits on the training partition.

Defaults follow ``docs/ARCHITECTURE.md`` section 8.3 with one deliberate change:
class weights are supplied by the caller from the Phase 5 training split, so all
three model families receive identical imbalance treatment.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from churnwise.models.contracts import LOGISTIC_FAMILY, RANDOM_FOREST_FAMILY
from churnwise.models.errors import (
    CODE_INVALID_MODEL,
    CODE_MODEL_ARTIFACT,
    CODE_PREDICTION_FAILED,
    CODE_TRAINING_FAILED,
    ModelingError,
)

BASELINE_FAMILIES = (LOGISTIC_FAMILY, RANDOM_FOREST_FAMILY)
BASELINE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Documented, fixed Logistic Regression configuration."""

    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    tol: float = 1e-4
    class_weight: Any = "balanced"
    random_state: int = 42

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.penalty not in {"l1", "l2", "elasticnet"}:
            problems.append("penalty must be one of l1, l2, elasticnet")
        if self.solver not in {"lbfgs", "liblinear", "saga", "newton-cg", "newton-cholesky"}:
            problems.append("solver is not supported by scikit-learn LogisticRegression")
        if not 0.0 < float(self.C) <= 1e6:
            problems.append("C must be positive")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            problems.append("max_iter must be a positive integer")
        if not 0.0 < float(self.tol) < 1.0:
            problems.append("tol must be in (0, 1)")
        if self.penalty == "l1" and self.solver not in {"liblinear", "saga"}:
            problems.append("l1 penalty requires the liblinear or saga solver")
        if self.penalty == "elasticnet" and self.solver != "saga":
            problems.append("elasticnet penalty requires the saga solver")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "penalty": self.penalty,
            "C": float(self.C),
            "solver": self.solver,
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "class_weight": self.class_weight,
            "random_state": int(self.random_state),
        }


@dataclass(frozen=True)
class RandomForestConfig:
    """Documented, fixed Random Forest configuration.

    ``n_estimators=300`` and ``min_samples_leaf=2`` keep the forest small enough
    for a CPU-only student laptop while remaining a genuine ensemble.
    """

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    bootstrap: bool = True
    class_weight: Any = "balanced_subsample"
    n_jobs: int = -1
    random_state: int = 42

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not isinstance(self.n_estimators, int) or not 10 <= self.n_estimators <= 2000:
            problems.append("n_estimators must be an integer between 10 and 2000")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth < 1
        ):
            problems.append("max_depth must be None or a positive integer")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            problems.append("min_samples_split must be an integer >= 2")
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            problems.append("min_samples_leaf must be a positive integer")
        if self.max_features not in {"sqrt", "log2", "all", None}:
            problems.append("max_features must be sqrt, log2, all or None")
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            problems.append("n_jobs must be a non-zero integer")
        return problems

    def payload(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "max_depth": self.max_depth,
            "min_samples_split": int(self.min_samples_split),
            "min_samples_leaf": int(self.min_samples_leaf),
            "max_features": self.max_features,
            "bootstrap": bool(self.bootstrap),
            "class_weight": self.class_weight,
            "n_jobs": int(self.n_jobs),
            "random_state": int(self.random_state),
        }


BASELINE_DEFAULTS = {
    LOGISTIC_FAMILY: LogisticRegressionConfig(),
    RANDOM_FOREST_FAMILY: RandomForestConfig(),
}


@dataclass
class FittedBaseline:
    """A fitted scikit-learn baseline plus the evidence needed to audit it."""

    family: str
    estimator: Any
    config: dict[str, Any]
    feature_names: list[str]
    train_rows: int
    fit_seconds: float
    class_weight: dict[int, float] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return ``P(churn = 1)`` for a Phase 5 transformed matrix."""
        matrix = _validate_matrix(features, len(self.feature_names))
        try:
            probabilities = np.asarray(
                self.estimator.predict_proba(matrix), dtype=float
            )
        except (AttributeError, ValueError, RuntimeError) as exc:
            raise ModelingError(
                CODE_TRAINING_FAILED,
                f"{self.family} could not produce probabilities: "
                f"{type(exc).__name__}: {exc}",
                "Refit the baseline on the Phase 5 training matrix.",
            ) from exc
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"{self.family} returned probabilities with shape "
                f"{probabilities.shape}; expected (n, 2)",
                "Use a binary classifier with a positive class at column 1.",
            )
        scores = probabilities[:, 1]
        if len(scores) != len(matrix):
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"{self.family} returned {len(scores)} probabilities for "
                f"{len(matrix)} rows",
                "Refit the baseline and retry.",
            )
        if not np.isfinite(scores).all() or scores.min() < 0.0 or scores.max() > 1.0:
            raise ModelingError(
                CODE_PREDICTION_FAILED,
                f"{self.family} produced probabilities outside [0, 1]",
                "Refit the baseline; the estimator output is not a probability.",
            )
        return np.clip(scores, 0.0, 1.0)


def _validate_matrix(features: Any, expected_width: int) -> np.ndarray:
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"baseline features are not numeric: {type(exc).__name__}: {exc}",
            "Pass the Phase 5 transformed matrix.",
        ) from exc
    if matrix.ndim != 2:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"baseline features must be 2-D, got shape {matrix.shape}",
            "Pass one row per customer with all transformed features.",
        )
    if matrix.shape[1] != expected_width:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"features have {matrix.shape[1]} columns; expected {expected_width}",
            "Use the same Phase 5 transform as the MLP.",
        )
    if not np.isfinite(matrix).all():
        raise ModelingError(
            CODE_INVALID_MODEL,
            "baseline features contain non-finite values",
            "Regenerate the Phase 5 run before fitting.",
        )
    return matrix


def _validate_labels(labels: Any, rows: int) -> np.ndarray:
    try:
        values = np.asarray(labels).reshape(-1).astype(int)
    except (TypeError, ValueError) as exc:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"training labels are not binary: {type(exc).__name__}: {exc}",
            "Use the Phase 5 0/1 churn labels.",
        ) from exc
    if len(values) != rows:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{len(values)} labels for {rows} training rows",
            "Regenerate the Phase 5 split.",
        )
    if not np.isin(values, (0, 1)).all():
        raise ModelingError(
            CODE_INVALID_MODEL,
            "training labels must be 0/1 churn indicators",
            "Use the Phase 5 target mapping; positive means churned.",
        )
    if np.unique(values).size < 2:
        raise ModelingError(
            CODE_TRAINING_FAILED,
            "the training partition contains a single class",
            "Acquire a dataset with both churn classes or adjust the split.",
        )
    return values


def _weights(class_weight: Any) -> dict[int, float] | str | None:
    if isinstance(class_weight, dict):
        try:
            return {int(key): float(value) for key, value in class_weight.items()}
        except (TypeError, ValueError) as exc:
            raise ModelingError(
                CODE_INVALID_MODEL,
                f"class weights are not numeric: {type(exc).__name__}: {exc}",
                "Regenerate the Phase 5 class weights from the training split.",
            ) from exc
    return class_weight


def build_estimator(
    family: str,
    *,
    config: LogisticRegressionConfig | RandomForestConfig | None = None,
    class_weight: Any = None,
) -> Any:
    """Construct one unfitted baseline estimator."""
    if family == LOGISTIC_FAMILY:
        resolved = config or BASELINE_DEFAULTS[LOGISTIC_FAMILY]
        if not isinstance(resolved, LogisticRegressionConfig):
            raise ModelingError(
                CODE_INVALID_MODEL,
                "logistic regression needs a LogisticRegressionConfig",
                "Use the documented default configuration.",
            )
        problems = resolved.validate()
        if problems:
            raise ModelingError(
                CODE_INVALID_MODEL,
                "invalid logistic regression configuration: " + "; ".join(problems),
                "Correct the baseline options and retry.",
            )
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty=resolved.penalty,
            C=float(resolved.C),
            solver=resolved.solver,
            max_iter=int(resolved.max_iter),
            tol=float(resolved.tol),
            class_weight=_weights(
                resolved.class_weight if class_weight is None else class_weight
            ),
            random_state=int(resolved.random_state),
        )
    if family == RANDOM_FOREST_FAMILY:
        resolved = config or BASELINE_DEFAULTS[RANDOM_FOREST_FAMILY]
        if not isinstance(resolved, RandomForestConfig):
            raise ModelingError(
                CODE_INVALID_MODEL,
                "random forest needs a RandomForestConfig",
                "Use the documented default configuration.",
            )
        problems = resolved.validate()
        if problems:
            raise ModelingError(
                CODE_INVALID_MODEL,
                "invalid random forest configuration: " + "; ".join(problems),
                "Correct the baseline options and retry.",
            )
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(resolved.n_estimators),
            max_depth=resolved.max_depth,
            min_samples_split=int(resolved.min_samples_split),
            min_samples_leaf=int(resolved.min_samples_leaf),
            max_features=resolved.max_features,
            bootstrap=bool(resolved.bootstrap),
            class_weight=_weights(
                resolved.class_weight if class_weight is None else class_weight
            ),
            n_jobs=int(resolved.n_jobs),
            random_state=int(resolved.random_state),
        )
    raise ModelingError(
        CODE_INVALID_MODEL,
        f"unknown baseline family {family!r}",
        f"Use one of {list(BASELINE_FAMILIES)}.",
    )


def fit_baseline(
    family: str,
    features: Any,
    labels: Any,
    *,
    config: LogisticRegressionConfig | RandomForestConfig | None = None,
    class_weight: Any = None,
    feature_names: list[str] | None = None,
) -> FittedBaseline:
    """Fit one baseline on training rows only and record its configuration."""
    estimator = build_estimator(family, config=config, class_weight=class_weight)
    raw = np.asarray(features)
    if raw.ndim != 2:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"baseline features must be 2-D, got shape {raw.shape}",
            "Pass the Phase 5 transformed training matrix.",
        )
    matrix = _validate_matrix(features, len(feature_names) if feature_names else raw.shape[1])
    values = _validate_labels(labels, len(matrix))
    resolved_config = config or BASELINE_DEFAULTS[family]
    resolved_weight = (
        _weights(class_weight) if class_weight is not None else resolved_config.class_weight
    )
    started = time.perf_counter()
    try:
        estimator.fit(matrix, values)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ModelingError(
            CODE_TRAINING_FAILED,
            f"{family} training failed: {type(exc).__name__}: {exc}",
            "Check the Phase 5 matrix, class weights and scikit-learn version.",
        ) from exc
    duration = time.perf_counter() - started
    if isinstance(resolved_weight, dict):
        weights = {int(key): float(value) for key, value in resolved_weight.items()}
    else:
        weights = {}
    return FittedBaseline(
        family=family,
        estimator=estimator,
        config=resolved_config.payload(),
        feature_names=list(feature_names or []),
        train_rows=int(len(matrix)),
        fit_seconds=float(duration),
        class_weight=weights,
    )


def baseline_metadata(fitted: FittedBaseline, *, seed: int) -> dict[str, Any]:
    """Structured metadata for one fitted baseline artifact."""
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "phase": 7,
        "model_family": fitted.family,
        "framework": "scikit-learn",
        "config": dict(fitted.config),
        "seed": int(seed),
        "class_weight": {str(key): float(value) for key, value in fitted.class_weight.items()},
        "training": {
            "train_rows": int(fitted.train_rows),
            "fit_seconds": float(fitted.fit_seconds),
        },
        "feature_names": list(fitted.feature_names),
        "feature_count": len(fitted.feature_names),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_baseline(directory: Path, fitted: FittedBaseline, metadata: dict[str, Any]) -> dict[str, Any]:
    """Persist a baseline with joblib and return paths plus a digest."""
    import joblib

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.joblib"
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    try:
        joblib.dump(fitted, temporary)
        temporary.replace(model_path)
    except OSError as exc:
        raise ModelingError(
            CODE_MODEL_ARTIFACT,
            f"baseline artifact could not be written: {exc}",
            "Check that the artifact directory is writable.",
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256(model_path)
    payload = {
        **metadata,
        "model_sha256": digest,
        "model_size_bytes": int(model_path.stat().st_size),
        "artifacts": {"model": str(model_path)},
    }
    return payload


def load_baseline(model_path: Path | str, *, expected_sha256: str | None = None) -> FittedBaseline:
    """Load a persisted baseline after verifying its digest."""
    import joblib

    path = Path(model_path)
    if not path.is_file():
        raise ModelingError(
            CODE_MODEL_ARTIFACT,
            f"baseline artifact is missing: {path}",
            "Rerun the evaluate command to regenerate it.",
        )
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise ModelingError(
            CODE_MODEL_ARTIFACT,
            "baseline digest does not match its metadata",
            "Regenerate the evaluation; the artifact may be corrupt or edited.",
        )
    try:
        loaded = joblib.load(path)
    except (OSError, ValueError, EOFError) as exc:
        raise ModelingError(
            CODE_MODEL_ARTIFACT,
            f"baseline artifact could not be loaded: {type(exc).__name__}: {exc}",
            "Regenerate the evaluation with this project version.",
        ) from exc
    if not isinstance(loaded, FittedBaseline):
        raise ModelingError(
            CODE_MODEL_ARTIFACT,
            "baseline artifact does not contain a FittedBaseline",
            "Regenerate the evaluation; do not load third-party joblib files.",
        )
    return loaded


__all__ = [
    "BASELINE_DEFAULTS",
    "BASELINE_FAMILIES",
    "BASELINE_SCHEMA_VERSION",
    "FittedBaseline",
    "LogisticRegressionConfig",
    "RandomForestConfig",
    "baseline_metadata",
    "build_estimator",
    "fit_baseline",
    "load_baseline",
    "save_baseline",
]
