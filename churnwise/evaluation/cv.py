"""Optional fold-safe stratified cross-validation on train+validation.

Folds are drawn from ``train + validation`` rows only. Inside each fold, the
Phase 5 column plan, imputers, scaler and one-hot encoder are refitted on that
fold's training rows and the validation rows are transform-only. This satisfies
the fold-isolation rule; no test row participates in CV.

The Phase 5 hand-off contains transformed matrices, so the base frame is
reconstructed and verified by :mod:`churnwise.evaluation.folds` first. If that
verification is unavailable, CV is reported as ``unavailable`` rather than being
quietly run with a shared transform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from churnwise.config import Settings
from churnwise.evaluation.contracts import MLP_FAMILY
from churnwise.evaluation.errors import CODE_METRIC_FAILED, EvaluationError
from churnwise.evaluation.folds import (
    FoldContext,
    build_fold_context,
    fold_labels,
    prepare_fold_matrices,
)
from churnwise.evaluation.metrics import binary_metrics
from churnwise.models.baselines import (
    BASELINE_FAMILIES,
    RANDOM_FOREST_FAMILY,
    LogisticRegressionConfig,
    RandomForestConfig,
    fit_baseline,
)

CV_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier")
CV_THRESHOLD = 0.5


@dataclass
class CrossValidationSummary:
    """Mean and standard deviation of fold metrics at the 0.5 reference threshold."""

    folds_requested: int
    folds_completed: int
    threshold: float = CV_THRESHOLD
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    fold_details: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    status: str = "completed"
    duration_seconds: float = 0.0
    preprocessing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "folds_requested": self.folds_requested,
            "folds_completed": self.folds_completed,
            "threshold": self.threshold,
            "metrics": {key: dict(value) for key, value in self.metrics.items()},
            "fold_details": [dict(item) for item in self.fold_details],
            "notes": list(self.notes),
            "duration_seconds": float(self.duration_seconds),
            "preprocessing": dict(self.preprocessing),
        }


def _summarise(values: list[float | None]) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return {"mean": None, "std": None, "n": 0, "values": list(values)}
    array = np.asarray(defined, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": len(defined),
        "values": list(values),
    }


def cross_validate(
    dataset: Any,
    settings: Settings,
    family: str,
    *,
    folds: int = 5,
    seed: int = 42,
    class_weight: Any = None,
    logistic_config: LogisticRegressionConfig | None = None,
    forest_config: RandomForestConfig | None = None,
    mlp_config: Any = None,
    context: FoldContext | None = None,
) -> CrossValidationSummary:
    """Run stratified, fold-safe CV for one family on train+validation."""
    if folds < 2:
        raise EvaluationError(
            CODE_METRIC_FAILED,
            "cross-validation needs at least two folds",
            "Pass --cv-folds 0 to disable CV or at least 2.",
        )
    summary = CrossValidationSummary(folds_requested=int(folds), folds_completed=0)
    summary.notes.extend(
        [
            "Folds are drawn from train+validation rows only; the test partition is "
            "not used.",
            "Each fold refits the Phase 5 column plan, imputers, scaler and encoder on "
            "its own training rows; validation rows are transform-only.",
            "Fold metrics use the 0.5 reference threshold, not the "
            "validation-selected operating threshold.",
            "Class weights are the Phase 5 training weights in every fold, matching "
            "the weights used for the final family fits.",
        ]
    )
    if family == MLP_FAMILY:
        summary.notes.append(
            "The MLP is refitted from scratch in every fold with an internal "
            "training-only hold-out for early stopping."
        )

    if context is None:
        context, reason = build_fold_context(dataset, settings)
    else:
        reason = None
    if context is None:
        summary.status = "unavailable"
        summary.notes.append(
            reason
            or "fold-specific preprocessing could not be reconstructed for this run"
        )
        summary.preprocessing = {
            "refitted_per_fold": False,
            "verified_against_phase5": False,
            "reason": summary.notes[-1],
        }
        return summary
    summary.preprocessing = {
        "refitted_per_fold": True,
        "verified_against_phase5": True,
        "source": dict(context.source),
    }

    try:
        train_positions = np.asarray(list(dataset.train.row_indices), dtype=np.int64)
        val_positions = np.asarray(list(dataset.val.row_indices), dtype=np.int64)
        positions = np.concatenate([train_positions, val_positions])
        labels = np.concatenate(
            [
                fold_labels(context, train_positions),
                fold_labels(context, val_positions),
            ]
        ).astype(int)
    except (AttributeError, TypeError, ValueError) as exc:
        summary.status = "failed"
        summary.notes.append(f"CV rows could not be read: {type(exc).__name__}: {exc}")
        return summary
    if len(np.unique(labels)) < 2:
        summary.status = "failed"
        summary.notes.append(
            "cross-validation needs both classes in train+validation; only one is present"
        )
        return summary

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    started = time.perf_counter()
    try:
        splits = list(splitter.split(positions.reshape(-1, 1), labels))
    except ValueError as exc:
        summary.status = "failed"
        summary.notes.append(f"cross-validation could not be constructed: {exc}")
        return summary

    collected: dict[str, list[float | None]] = {name: [] for name in CV_METRICS}
    for index, (fit_index, valid_index) in enumerate(splits, start=1):
        fold_started = time.perf_counter()
        fit_positions = positions[fit_index]
        valid_positions = positions[valid_index]
        try:
            X_fit, X_valid, feature_names = prepare_fold_matrices(
                context, fit_positions, valid_positions
            )
            y_fit = fold_labels(context, fit_positions)
            y_valid = fold_labels(context, valid_positions)
            if family in BASELINE_FAMILIES:
                fitted = fit_baseline(
                    family,
                    X_fit,
                    y_fit,
                    config=(
                        forest_config
                        if family == RANDOM_FOREST_FAMILY
                        else logistic_config
                    ),
                    class_weight=class_weight,
                    feature_names=feature_names,
                )
                probabilities = fitted.predict_proba(X_valid)
            elif family == MLP_FAMILY:
                if mlp_config is None:
                    raise EvaluationError(
                        CODE_METRIC_FAILED,
                        "MLP cross-validation requires the Phase 6 MLP configuration",
                        "Reload the model metadata or disable CV.",
                    )
                probabilities = _fit_mlp_fold(
                    X_fit,
                    y_fit,
                    X_valid,
                    config=mlp_config,
                    seed=seed + index,
                    class_weight=class_weight,
                )
            else:
                raise EvaluationError(
                    CODE_METRIC_FAILED,
                    f"unknown family {family!r}",
                    "Use logistic_regression, random_forest or mlp.",
                )
            block = binary_metrics(y_valid, probabilities, CV_THRESHOLD)
        except Exception as exc:  # noqa: BLE001 - one fold must not lose the family
            summary.notes.append(f"fold {index} failed: {type(exc).__name__}: {exc}")
            summary.fold_details.append(
                {"fold": index, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
            for name in CV_METRICS:
                collected[name].append(None)
            continue
        summary.folds_completed += 1
        summary.fold_details.append(
            {
                "fold": index,
                "status": "completed",
                "n_train": int(len(fit_positions)),
                "n_validation": int(len(valid_positions)),
                "feature_count": int(len(feature_names)),
                "positive_rate": float(y_valid.mean()),
                "duration_seconds": float(time.perf_counter() - fold_started),
                **{name: block.get(name) for name in CV_METRICS},
            }
        )
        for name in CV_METRICS:
            collected[name].append(block.get(name))

    summary.metrics = {name: _summarise(collected[name]) for name in CV_METRICS}
    summary.duration_seconds = float(time.perf_counter() - started)
    if summary.folds_completed == 0:
        summary.status = "failed"
    elif summary.folds_completed < summary.folds_requested:
        summary.status = "partial"
    return summary


def _fit_mlp_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    config: Any,
    seed: int,
    class_weight: Any = None,
) -> np.ndarray:
    """Fit a fresh MLP for one CV fold without writing artifacts."""
    from churnwise.models.mlp import build_mlp, set_global_seed

    import tensorflow as tf

    set_global_seed(seed)
    model = build_mlp(int(X_train.shape[1]), config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    # A small internal hold-out keeps early stopping meaningful without touching
    # the outer validation fold; it is drawn from the fold's training rows only.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_train))
    holdout = max(1, min(len(order) // 5, 500))
    if len(order) - holdout < 1:  # degenerate, tiny fold: reuse rows
        inner, outer = order, order
    else:
        inner = order[:holdout]
        outer = order[holdout:]
    weights = (
        {int(key): float(value) for key, value in class_weight.items()}
        if isinstance(class_weight, dict)
        else None
    )
    model.fit(
        X_train[outer],
        y_train[outer].astype(np.float32).reshape(-1, 1),
        validation_data=(
            X_train[inner],
            y_train[inner].astype(np.float32).reshape(-1, 1),
        ),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        class_weight=weights,
        verbose=0,
        callbacks=callbacks,
        shuffle=bool(config.shuffle),
    )
    probabilities = np.asarray(model.predict(X_val, verbose=0), dtype=float).reshape(-1)
    return np.clip(probabilities, 0.0, 1.0)


__all__ = ["CV_METRICS", "CV_THRESHOLD", "CrossValidationSummary", "cross_validate"]
