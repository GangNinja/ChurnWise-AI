"""Phase 7 benchmark orchestration: identical data, frozen thresholds, one test pass.

Design guarantees enforced here:

* all three families consume the same Phase 5 training matrix, validation matrix
  and held-out test matrix (the split fingerprint is verified, not assumed);
* thresholds and the shipped-model decision are computed from validation data only;
* the test partition is evaluated once, after thresholds are frozen;
* a failing family is recorded without altering the other families' results;
* every reported number is persisted and can be regenerated from stored predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from churnwise import __version__
from churnwise.config import Settings
from churnwise.contracts import utc_now_iso
from churnwise.evaluation import plots
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
    precision_recall_curve_data,
    roc_curve_data,
)
from churnwise.evaluation.reporting import render_benchmark_markdown, render_summary
from churnwise.evaluation.thresholds import (
    ThresholdChoice,
    select_threshold,
    threshold_analysis,
)
from churnwise.models.baselines import (
    FittedBaseline,
    LogisticRegressionConfig,
    RandomForestConfig,
    baseline_metadata,
    fit_baseline,
    save_baseline,
)
from churnwise.models.contracts import MLPConfig
from churnwise.models.errors import ModelingError
from churnwise.models.registry import ModelRegistry, validate_model_run_id
from churnwise.preprocessing import PreparedDataset

BENCHMARK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,180}$")
_FAMILY_FAILURES = (EvaluationError, ModelingError, ValueError, RuntimeError)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:110] or "dataset"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            CODE_ARTIFACT_FAILED,
            f"evaluation artifact could not be read: {path}: {type(exc).__name__}: {exc}",
            "Rerun the evaluate command to regenerate the artifact.",
        ) from exc


def validate_benchmark_id(benchmark_run_id: str) -> str:
    if (
        not BENCHMARK_ID_PATTERN.fullmatch(benchmark_run_id)
        or ".." in benchmark_run_id
    ):
        raise EvaluationError(
            CODE_ARTIFACT_FAILED,
            f"invalid benchmark run id {benchmark_run_id!r}",
            "Use the id printed by the evaluate command.",
        )
    return benchmark_run_id


def _matrix(values: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"{name} features are not numeric: {type(exc).__name__}: {exc}",
            "Use the Phase 5 PreparedDataset partitions.",
        ) from exc
    if matrix.ndim != 2 or len(matrix) == 0:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"{name} partition is empty or not 2-D (shape {matrix.shape})",
            "Regenerate the Phase 5 split with a larger dataset.",
        )
    if not np.isfinite(matrix).all():
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"{name} features contain non-finite values",
            "Regenerate the Phase 5 run and inspect the invalid-value report.",
        )
    return matrix


def _labels(values: Any, name: str) -> np.ndarray:
    try:
        labels = np.asarray(values).reshape(-1).astype(int)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"{name} labels are not binary: {type(exc).__name__}: {exc}",
            "Use the Phase 5 0/1 churn labels.",
        ) from exc
    if len(labels) == 0:
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"{name} labels are empty",
            "Regenerate the Phase 5 split.",
        )
    if not np.isin(labels, (0, 1)).all():
        raise EvaluationError(
            CODE_INVALID_CONFIG,
            f"{name} labels must be 0/1 churn indicators",
            "Use the Phase 5 target mapping; positive means churned.",
        )
    return labels


def _validate_dataset(dataset: PreparedDataset) -> dict[str, np.ndarray]:
    """Verify partitions are disjoint, aligned and share one feature space."""
    for attribute in ("train", "val", "test", "split_spec", "schema"):
        if not hasattr(dataset, attribute):
            raise EvaluationError(
                CODE_INCOMPATIBLE_INPUTS,
                f"dataset is missing {attribute!r}; it is not a Phase 5 hand-off",
                "Load the prepared dataset with PreprocessingCoordinator.load().",
            )
    X_train = _matrix(dataset.train.X, "training")
    X_val = _matrix(dataset.val.X, "validation")
    X_test = _matrix(dataset.test.X, "test")
    y_train = _labels(dataset.train.y, "training")
    y_val = _labels(dataset.val.y, "validation")
    y_test = _labels(dataset.test.y, "test")
    widths = {X_train.shape[1], X_val.shape[1], X_test.shape[1]}
    if len(widths) != 1:
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            f"partitions have different feature widths: {sorted(widths)}",
            "Regenerate the Phase 5 run; all partitions must share one matrix.",
        )
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if feature_names and len(feature_names) != X_train.shape[1]:
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            f"schema lists {len(feature_names)} features but the matrix has "
            f"{X_train.shape[1]} columns",
            "Reload the Phase 5 run from the same run id.",
        )
    for name, matrix, labels in (
        ("training", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ):
        if len(matrix) != len(labels):
            raise EvaluationError(
                CODE_INCOMPATIBLE_INPUTS,
                f"{name} has {len(matrix)} feature rows but {len(labels)} labels",
                "Regenerate the Phase 5 run; the partitions are inconsistent.",
            )
    rows = {
        name: {int(index) for index in getattr(dataset, name).row_indices}
        for name in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = rows[left] & rows[right]
        if overlap:
            raise EvaluationError(
                CODE_INCOMPATIBLE_INPUTS,
                f"{left} and {right} partitions share {len(overlap)} source rows",
                "Regenerate the Phase 5 split; partitions must be disjoint.",
            )
    target = dict((dataset.schema or {}).get("target") or {})
    mapping = {str(key): int(value) for key, value in (target.get("mapping") or {}).items()}
    positive = target.get("positive_label")
    if mapping and positive is not None and mapping.get(str(positive)) != 1:
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            f"target mapping puts {positive!r} at {mapping.get(str(positive))}, not 1",
            "Regenerate Phase 4/5 so churn is the positive (1) class.",
        )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def _mlp_config_from_metadata(metadata: Mapping[str, Any]) -> MLPConfig:
    training = dict(metadata.get("training") or {})
    fields = set(MLPConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    payload = {key: value for key, value in training.items() if key in fields}
    for key in ("hidden_units", "dropout"):
        if key in payload and isinstance(payload[key], list):
            payload[key] = tuple(payload[key])
    try:
        config = MLPConfig(**payload)
    except TypeError as exc:
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            f"Phase 6 metadata does not describe a usable MLP configuration: {exc}",
            "Retrain the MLP with this project version.",
        ) from exc
    problems = config.validate()
    if problems:
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            "Phase 6 metadata has an invalid MLP configuration: " + "; ".join(problems),
            "Retrain the MLP with this project version.",
        )
    return config


def _mlp_probabilities(model: Any, matrix: np.ndarray, *, family: str) -> np.ndarray:
    try:
        values = np.asarray(model.predict(matrix, verbose=0), dtype=float).reshape(-1)
    except (ValueError, RuntimeError, TypeError) as exc:
        raise EvaluationError(
            CODE_FAMILY_FAILED,
            f"{family} prediction failed: {type(exc).__name__}: {exc}",
            "Reload the verified Phase 6 model artifact.",
        ) from exc
    if len(values) != len(matrix):
        raise EvaluationError(
            CODE_FAMILY_FAILED,
            f"{family} returned {len(values)} probabilities for {len(matrix)} rows",
            "Reload the verified Phase 6 model artifact.",
        )
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise EvaluationError(
            CODE_FAMILY_FAILED,
            f"{family} produced probabilities outside [0, 1]",
            "Reload the verified Phase 6 model artifact; its output must be sigmoid.",
        )
    return np.clip(values, 0.0, 1.0)


def _class_weights(dataset: PreparedDataset) -> dict[int, float]:
    try:
        weights = {int(key): float(value) for key, value in dataset.class_weights.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            f"class weights are not readable: {type(exc).__name__}: {exc}",
            "Regenerate the Phase 5 class weights from the training split.",
        ) from exc
    if not weights:
        return {0: 1.0, 1: 1.0}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise EvaluationError(
            CODE_INCOMPATIBLE_INPUTS,
            "class weights must be finite and non-negative",
            "Regenerate the Phase 5 class weights from the training split.",
        )
    return weights


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class BenchmarkCoordinator:
    """Fit baselines, score the Phase 6 MLP and evaluate all families once."""

    def __init__(
        self,
        settings: Settings,
        config: EvaluationConfig | None = None,
        *,
        now: Any = utc_now_iso,
        code_version: str = __version__,
    ) -> None:
        self.settings = settings
        self.config = config or EvaluationConfig(seed=settings.seed)
        problems = self.config.validate()
        if problems:
            raise EvaluationError(
                CODE_INVALID_CONFIG,
                "invalid evaluation configuration: " + "; ".join(problems),
                "Correct the evaluate command options and retry.",
            )
        self.artifact_dir = Path(settings.artifact_dir)
        self.evaluation_dir = self.artifact_dir / "evaluation"
        self._now = now
        self.code_version = code_version
        self.registry = ModelRegistry(settings)
        self._fitted_baselines: dict[str, FittedBaseline] = {}
        self._mlp_model: Any | None = None

    # -- public API ---------------------------------------------------------
    def run(
        self,
        dataset: PreparedDataset,
        *,
        mlp_run_id: str | None = None,
        benchmark_run_id: str | None = None,
        persist: bool = True,
    ) -> BenchmarkResult:
        config = self.config
        generated_at = self._now()
        benchmark_run_id = benchmark_run_id or self._default_run_id(
            dataset, generated_at
        )
        validate_benchmark_id(benchmark_run_id)
        partitions = _validate_dataset(dataset)
        class_weight = _class_weights(dataset)
        root = self.evaluation_dir / benchmark_run_id
        paths: dict[str, str] = {}
        warnings: list[str] = []

        train = dataset.resampled_train() if dataset.oversample_indices else dataset.train
        X_train = np.asarray(train.X, dtype=np.float64)
        y_train = np.asarray(train.y).reshape(-1).astype(int)
        if len(X_train) != len(partitions["X_train"]):
            warnings.append(
                "training rows include Phase 5 training-only oversampling; every "
                "family is fitted on the same resampled rows"
            )

        families: dict[str, dict[str, Any]] = {}
        predictions: dict[str, np.ndarray] = {}

        mlp_block, mlp_predictions, mlp_config = self._load_mlp_family(
            dataset, partitions, mlp_run_id
        )
        if mlp_block.get("status") == "completed":
            families[MLP_FAMILY] = mlp_block
            predictions.update(mlp_predictions)
        else:
            families[MLP_FAMILY] = mlp_block

        baseline_configs: dict[str, Any] = {
            LOGISTIC_FAMILY: LogisticRegressionConfig(random_state=config.seed),
            RANDOM_FOREST_FAMILY: RandomForestConfig(random_state=config.seed),
        }
        for family in (LOGISTIC_FAMILY, RANDOM_FOREST_FAMILY):
            block, family_predictions = self._fit_baseline_family(
                family,
                dataset,
                partitions,
                train_rows=(X_train, y_train),
                class_weight=class_weight,
                config=baseline_configs[family],
            )
            families[family] = block
            if block.get("status") == "completed":
                predictions.update(family_predictions)

        completed = [
            name for name in SUPPORTED_FAMILIES if families.get(name, {}).get("status") == "completed"
        ]
        for name in SUPPORTED_FAMILIES:
            block = families.get(name) or {}
            if block.get("status") != "completed":
                error = block.get("error") or {}
                warnings.append(
                    f"{FAMILY_LABELS[name]} could not be evaluated: "
                    f"{error.get('message', 'unknown failure')}"
                )
        if not completed:
            raise EvaluationError(
                CODE_FAMILY_FAILED,
                "no model family could be evaluated",
                "Inspect the recorded family errors, then rerun after fixing them.",
            )

        # -- thresholds: validation only, frozen before any test metric ------
        choices: dict[str, ThresholdChoice] = {}
        sweeps: dict[str, dict[str, Any]] = {}
        for family in completed:
            block = families[family]
            probabilities = predictions[f"val_{family}"]
            choice = select_threshold(
                partitions["y_val"],
                probabilities,
                strategy=config.threshold_strategy,
                fixed_threshold=config.fixed_threshold,
                recall_floor=config.recall_floor,
            )
            choices[family] = choice
            sweeps[family] = threshold_analysis(
                partitions["y_val"], probabilities, choice
            )
            block["threshold"] = {
                **choice.to_dict(),
                "default": float(config.fixed_threshold),
            }
            block["validation_metrics"] = choice.validation_metrics
            block["validation_metrics_at_0_5"] = binary_metrics(
                partitions["y_val"], probabilities, config.fixed_threshold
            )

        # Frozen from here on: this is the only place test rows are scored and
        # test labels are read. No later step changes a threshold or the selection.
        test_calls = 1
        for family in completed:
            block = families[family]
            threshold = choices[family].threshold
            predictions[f"test_{family}"] = self._test_probabilities(
                family, partitions["X_test"]
            )
            block["test_metrics"] = binary_metrics(
                partitions["y_test"], predictions[f"test_{family}"], threshold
            )
            block["test_metrics_at_0_5"] = binary_metrics(
                partitions["y_test"], predictions[f"test_{family}"], config.fixed_threshold
            )
            block["train_metrics"] = binary_metrics(
                y_train, predictions[f"train_{family}"], threshold
            )
            roc = roc_curve_data(partitions["y_test"], predictions[f"test_{family}"])
            pr = precision_recall_curve_data(
                partitions["y_test"], predictions[f"test_{family}"]
            )
            block["curves"] = {
                "roc": {"auc": roc.get("auc"), "points": len(roc.get("fpr") or [])},
                "pr": {"pr_auc": pr.get("pr_auc"), "points": len(pr.get("recall") or [])},
                "file": "curves.json",
            }
            block["generalisation"] = {
                "train_f1_minus_validation_f1": _gap(
                    block["train_metrics"].get("f1"),
                    block["validation_metrics"].get("f1"),
                ),
                "train_roc_auc_minus_validation_roc_auc": _gap(
                    block["train_metrics"].get("roc_auc"),
                    block["validation_metrics"].get("roc_auc"),
                ),
                "test_f1_minus_validation_f1": _gap(
                    block["test_metrics"].get("f1"),
                    block["validation_metrics"].get("f1"),
                ),
                "note": "Train metrics are in-sample; they are reported only as an "
                "overfitting signal, never as performance evidence.",
            }
            block["notes"].extend(
                note
                for note in [
                    *(block["test_metrics"].get("notes") or []),
                    "Test metrics were computed once, after the threshold was frozen.",
                ]
                if note not in block["notes"]
            )

        # -- optional stability estimate (train + validation only) ----------
        cv_enabled = config.cv_folds >= 2
        cv_blocks: dict[str, Any] = {}
        if cv_enabled:
            for family in completed:
                try:
                    summary = cross_validate(
                        dataset,
                        self.settings,
                        family,
                        folds=config.cv_folds,
                        seed=config.seed,
                        class_weight=class_weight,
                        logistic_config=baseline_configs[LOGISTIC_FAMILY],
                        forest_config=baseline_configs[RANDOM_FOREST_FAMILY],
                        mlp_config=mlp_config if family == MLP_FAMILY else None,
                    )
                except _FAMILY_FAILURES as exc:  # pragma: no cover - defensive
                    summary = CrossValidationSummary(
                        folds_requested=config.cv_folds,
                        folds_completed=0,
                        status="failed",
                        notes=[f"cross-validation failed: {type(exc).__name__}: {exc}"],
                    )
                cv_blocks[family] = summary
                families[family]["cv"] = summary.to_dict()

        # -- selection uses validation evidence only -------------------------
        selection = self._select(families, completed, config)

        validity = self._validity(families, completed, test_calls)
        warnings.extend(validity["flags"])
        if not validity["roc_auc_above_chance"]:
            warnings.append(
                "at least one family scored ROC-AUC <= 0.5 on the test partition; "
                "the run is flagged as a defect, not a success"
            )
        if len(partitions["y_test"]) < 200:
            warnings.append(
                f"only {len(partitions['y_test'])} held-out rows; metric uncertainty is large"
            )
        if config.seed < 0:  # unreachable after validation, kept as a guard
            warnings.append("negative seed is not possible here")

        comparison = self._comparison_table(families, completed)
        metrics: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "phase": 7,
            "benchmark_run_id": benchmark_run_id,
            "generated_at": generated_at,
            "code_version": self.code_version,
            "seed": int(config.seed),
            "preprocessing": self._preprocessing_block(dataset),
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.split_spec.dataset_sha256,
                "split_fingerprint": dataset.split_spec.fingerprint,
                "feature_count": len(dataset.feature_names),
                "feature_names": list(dataset.feature_names),
                "n_train": int(len(X_train)),
                "n_val": int(len(partitions["y_val"])),
                "n_test": int(len(partitions["y_test"])),
                "test_positive_count": int(partitions["y_test"].sum()),
                "test_negative_count": int(len(partitions["y_test"]) - partitions["y_test"].sum()),
                "test_positive_rate": float(partitions["y_test"].mean()),
                "validation_positive_rate": float(partitions["y_val"].mean()),
                "train_positive_rate": float(partitions["y_train"].mean()),
                "class_weights": {str(key): float(value) for key, value in class_weight.items()},
                "target": dict((dataset.schema or {}).get("target") or {}),
                "test_rows_used": int(len(partitions["y_test"])),
                "matrix_sha256": {
                    "train": _matrix_digest(partitions["X_train"]),
                    "val": _matrix_digest(partitions["X_val"]),
                    "test": _matrix_digest(partitions["X_test"]),
                },
            },
            "evaluation_config": config.to_dict(),
            "threshold_policy": {
                "primary": {
                    "strategy": config.threshold_strategy,
                    "selected_on": "validation",
                    "frozen_before_test": True,
                    "fixed_threshold": float(config.fixed_threshold),
                    "recall_floor": config.recall_floor,
                },
                "default_threshold": float(config.fixed_threshold),
                "default_threshold_reporting": (
                    "Every family also reports metrics at the 0.5 default; those "
                    "numbers never influenced threshold or model selection."
                ),
            },
            "families": families,
            "comparison_table": comparison,
            "selection": selection,
            "error_analysis": self._error_analysis(
                families, completed, partitions, predictions
            ),
            "cross_validation": {
                "enabled": cv_enabled,
                "folds": int(config.cv_folds) if cv_enabled else 0,
                "scope": "train+validation rows only",
                "status": self._cv_status(cv_enabled, cv_blocks),
                "preprocessing": {
                    "strategy": "refit per fold from the Phase 5 base frame",
                    "fold_safe": all(
                        bool((block.preprocessing or {}).get("refitted_per_fold"))
                        for block in cv_blocks.values()
                    )
                    if cv_blocks
                    else False,
                    "verified_against_phase5": all(
                        bool((block.preprocessing or {}).get("verified_against_phase5"))
                        for block in cv_blocks.values()
                    )
                    if cv_blocks
                    else False,
                    "by_family": {
                        family: dict(block.preprocessing)
                        for family, block in cv_blocks.items()
                    },
                },
                "notes": [
                    "CV metrics are a stability estimate at the 0.5 reference threshold; "
                    "the held-out test result remains the primary evidence.",
                    "Each fold refits the Phase 5 column plan, imputers, scaler and encoder "
                    "on its own training rows; validation rows are transform-only.",
                    "The base frame is reconstructed from the recorded raw artifact and "
                    "accepted only if a full-training refit reproduces the Phase 5 "
                    "matrices exactly; otherwise CV is reported as unavailable.",
                ],
            },
            "plots": {},
            "artifacts": {},
            "environment": self._environment(),
            "validity": validity,
            "complete": len(completed) == len(SUPPORTED_FAMILIES),
            "missing_families": [
                name for name in SUPPORTED_FAMILIES if name not in completed
            ],
            "warnings": warnings,
            "limitations": self._limitations(completed, cv_enabled, cv_blocks),
        }

        result = BenchmarkResult(
            benchmark_run_id=benchmark_run_id,
            metrics=metrics,
            paths=paths,
            warnings=warnings,
        )
        if persist:
            self._persist(result, root, datasets=partitions, predictions=predictions,
                          sweeps=sweeps)
        result.metrics["summary"] = render_summary(result)
        return result

    # -- families -----------------------------------------------------------
    def _load_mlp_family(
        self,
        dataset: PreparedDataset,
        partitions: Mapping[str, np.ndarray],
        mlp_run_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], MLPConfig | None]:
        block: dict[str, Any] = {
            "family": MLP_FAMILY,
            "label": FAMILY_LABELS[MLP_FAMILY],
            "status": "failed",
            "source": "phase6_artifact",
            "notes": [],
            "error": None,
        }
        try:
            run_id = mlp_run_id or self._latest_mlp_for(dataset.run_id)
            if not run_id:
                raise EvaluationError(
                    CODE_NO_MODEL,
                    "no Phase 6 MLP artifact matches this preprocessing run",
                    "Run `python -m churnwise.cli train` first, or pass --mlp-run-id.",
                )
            validate_model_run_id(run_id)
            loaded = self.registry.load(run_id)
            metadata = loaded.metadata
            preprocessing = dict(metadata.get("preprocessing") or {})
            if str(preprocessing.get("run_id", "")) != dataset.run_id:
                raise EvaluationError(
                    CODE_INCOMPATIBLE_INPUTS,
                    f"MLP {run_id} was trained on preprocessing run "
                    f"{preprocessing.get('run_id')!r}, not {dataset.run_id!r}",
                    "Evaluate the MLP artifact produced from this Phase 5 run.",
                )
            if str(preprocessing.get("split_fingerprint", "")) != dataset.split_spec.fingerprint:
                raise EvaluationError(
                    CODE_INCOMPATIBLE_INPUTS,
                    "MLP split fingerprint does not match the Phase 5 split",
                    "Retrain the MLP on the persisted Phase 5 split.",
                )
            feature_names = list(preprocessing.get("feature_names") or [])
            if feature_names and feature_names != list(dataset.feature_names):
                raise EvaluationError(
                    CODE_INCOMPATIBLE_INPUTS,
                    "MLP feature names do not match the Phase 5 transformed matrix",
                    "Evaluate the MLP artifact produced from this Phase 5 run.",
                )
            config = _mlp_config_from_metadata(metadata)
            self._mlp_model = loaded.model
            # Test probabilities are deliberately deferred until thresholds are
            # frozen; only training and validation rows are scored here.
            predictions = {
                f"train_{MLP_FAMILY}": _mlp_probabilities(
                    loaded.model, partitions["X_train"], family=MLP_FAMILY
                ),
                f"val_{MLP_FAMILY}": _mlp_probabilities(
                    loaded.model, partitions["X_val"], family=MLP_FAMILY
                ),
            }
            block.update(
                {
                    "status": "completed",
                    "model": {
                        "model_run_id": run_id,
                        "model_sha256": metadata.get("model_sha256"),
                        "artifact_path": loaded.paths.get("model"),
                        "framework": "tensorflow.keras",
                        "config": config.to_dict(),
                        "parameter_count": (metadata.get("architecture") or {}).get("parameter_count"),
                        "train_rows": (metadata.get("diagnostics") or {}).get("train_rows"),
                        "fit_seconds": (metadata.get("diagnostics") or {}).get("duration_seconds"),
                        "trained_at": metadata.get("created_at"),
                    },
                    "class_weight": dict((metadata.get("training") or {}).get("class_weight") or {}),
                    "notes": [
                        "Loaded from the existing Phase 6 artifact; it was not retrained "
                        "or reselected on the test split."
                    ],
                }
            )
            return block, predictions, config
        except _FAMILY_FAILURES as exc:
            block["error"] = (
                exc.to_dict() if isinstance(exc, EvaluationError) else {"message": str(exc)}
            )
            block["notes"].append(
                f"family failed and was excluded from selection: {type(exc).__name__}: {exc}"
            )
            return block, {}, None

    def _latest_mlp_for(self, preprocessing_run_id: str) -> str | None:
        root = self.artifact_dir / "models"
        if not root.is_dir():
            return None
        matches: list[str] = []
        for path in root.iterdir():
            metadata_path = path / "mlp" / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str((payload.get("preprocessing") or {}).get("run_id", "")) == preprocessing_run_id:
                matches.append(path.name)
        return sorted(matches)[-1] if matches else None

    def _fit_baseline_family(
        self,
        family: str,
        dataset: PreparedDataset,
        partitions: Mapping[str, np.ndarray],
        *,
        train_rows: tuple[np.ndarray, np.ndarray],
        class_weight: dict[int, float],
        config: Any,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        block: dict[str, Any] = {
            "family": family,
            "label": FAMILY_LABELS[family],
            "status": "failed",
            "source": "fitted_in_phase7",
            "notes": [],
            "error": None,
        }
        X_train, y_train = train_rows
        try:
            fitted: FittedBaseline = fit_baseline(
                family,
                X_train,
                y_train,
                config=config,
                class_weight=class_weight,
                feature_names=list(dataset.feature_names),
            )
            self._fitted_baselines[family] = fitted
            predictions = {
                f"train_{family}": fitted.predict_proba(X_train),
                f"val_{family}": fitted.predict_proba(partitions["X_val"]),
            }
            block.update(
                {
                    "status": "completed",
                    "model": {
                        "framework": "scikit-learn",
                        "config": fitted.config,
                        "seed": int(config.random_state),
                        "train_rows": int(fitted.train_rows),
                        "fit_seconds": float(fitted.fit_seconds),
                        "feature_count": len(fitted.feature_names),
                    },
                    "class_weight": {
                        str(key): float(value) for key, value in fitted.class_weight.items()
                    },
                    "notes": [
                        "Fitted in Phase 7 on the Phase 5 training partition with the "
                        "documented fixed hyperparameters; no test row was used."
                    ],
                }
            )
            return block, predictions
        except _FAMILY_FAILURES as exc:
            block["error"] = (
                exc.to_dict() if isinstance(exc, EvaluationError) else {"message": str(exc)}
            )
            block["notes"].append(
                f"family failed and was excluded from selection: {type(exc).__name__}: {exc}"
            )
            return block, {}

    def _test_probabilities(self, family: str, matrix: np.ndarray) -> np.ndarray:
        """Score held-out rows once, after the operating threshold is frozen."""
        if family == MLP_FAMILY:
            model = getattr(self, "_mlp_model", None)
            if model is None:
                raise EvaluationError(
                    CODE_FAMILY_FAILED,
                    "the MLP model is unavailable for test scoring",
                    "Reload the verified Phase 6 artifact and rerun the evaluation.",
                )
            return _mlp_probabilities(model, matrix, family=family)
        fitted = self._fitted_baselines.get(family)
        if fitted is None:
            raise EvaluationError(
                CODE_FAMILY_FAILED,
                f"the fitted {family} model is unavailable for test scoring",
                "Rerun the evaluation so the baseline is fitted again.",
            )
        return fitted.predict_proba(matrix)

    # -- selection ----------------------------------------------------------
    def _select(
        self,
        families: Mapping[str, Mapping[str, Any]],
        completed: Iterable[str],
        config: EvaluationConfig,
    ) -> dict[str, Any]:
        metric = config.selection_metric
        rows: list[dict[str, Any]] = []
        for family in completed:
            validation = families[family].get("validation_metrics") or {}
            rows.append(
                {
                    "family": family,
                    "label": FAMILY_LABELS[family],
                    "metric": metric,
                    "validation_value": _finite_or_none(validation.get(metric)),
                    "validation_recall": _finite_or_none(validation.get("recall")),
                    "validation_f1": _finite_or_none(validation.get("f1")),
                    "validation_roc_auc": _finite_or_none(validation.get("roc_auc")),
                    "test_value": _finite_or_none(
                        (families[family].get("test_metrics") or {}).get(metric)
                    ),
                }
            )
        eligible = [row for row in rows if row["validation_value"] is not None]
        floor = config.selection_min_recall
        floor_note: str | None = None
        if floor is not None:
            constrained = [
                row
                for row in eligible
                if row["validation_recall"] is not None
                and float(row["validation_recall"]) >= float(floor)
            ]
            if constrained:
                eligible = constrained
            else:
                floor_note = (
                    f"no family reached the validation recall floor {float(floor):.2f}; "
                    "the floor was not applied"
                )
        if not eligible:
            return {
                "selection_metric": metric,
                "selected_family": None,
                "rationale": "No family had a defined validation "
                f"{metric}; no shipped model was selected.",
                "ranking": rows,
                "selected_on": "validation",
                "test_used_for_selection": False,
                "recall_floor": floor,
                "notes": [floor_note] if floor_note else [],
            }
        ranked = sorted(
            eligible,
            key=lambda row: (
                -float(row["validation_value"]),
                -(row["validation_roc_auc"] or 0.0),
                SUPPORTED_FAMILIES.index(row["family"]),
            ),
        )
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        rationale = (
            f"Selected {winner['label']} because it had the highest validation "
            f"{metric} ({winner['validation_value']:.4f}) at its validation-selected "
            "operating threshold."
        )
        if runner_up is not None:
            rationale += (
                f" {runner_up['label']} followed with {runner_up['validation_value']:.4f}."
            )
        if winner["family"] != MLP_FAMILY:
            rationale += (
                " The MLP was not the best model on the validation criterion; the "
                "benchmark does not assume it must win."
            )
        rationale += " The test partition was not used to choose the model."
        ranking = sorted(
            rows,
            key=lambda row: (
                -(row["validation_value"] if row["validation_value"] is not None else -1.0),
                SUPPORTED_FAMILIES.index(row["family"]),
            ),
        )
        for position, row in enumerate(ranking, start=1):
            row["rank"] = position
        notes = [
            "Selection uses validation predictions only, after thresholds were frozen.",
            "Accuracy alone is not the selection criterion; the configured metric is "
            f"{metric}.",
        ]
        if floor_note:
            notes.append(floor_note)
        return {
            "selection_metric": metric,
            "selected_family": winner["family"],
            "selected_label": winner["label"],
            "selected_validation_value": winner["validation_value"],
            "threshold": (families[winner["family"]].get("threshold") or {}).get("selected"),
            "selected_on": "validation",
            "test_used_for_selection": False,
            "recall_floor": floor,
            "ranking": ranking,
            "rationale": rationale,
            "notes": notes,
        }

    # -- diagnostics --------------------------------------------------------
    def _validity(
        self,
        families: Mapping[str, Mapping[str, Any]],
        completed: Iterable[str],
        test_calls: int,
    ) -> dict[str, Any]:
        flags: list[str] = []
        aucs: dict[str, float | None] = {}
        for family in completed:
            metrics = families[family].get("test_metrics") or {}
            auc = _finite_or_none(metrics.get("roc_auc"))
            aucs[family] = auc
            if auc is not None and auc <= 0.5:
                flags.append(
                    f"{FAMILY_LABELS[family]} test ROC-AUC is {auc:.4f} (<= 0.5); "
                    "this is a defect to investigate, not a usable model"
                )
            if metrics.get("undefined"):
                flags.append(
                    f"{FAMILY_LABELS[family]} has undefined metrics on the test "
                    f"partition: {', '.join(metrics['undefined'])}"
                )
        defined = [value for value in aucs.values() if value is not None]
        return {
            "roc_auc_above_chance": bool(defined) and all(value > 0.5 for value in defined),
            "test_roc_auc_by_family": aucs,
            "test_evaluation_calls": int(test_calls),
            "test_metric_evaluations": len(list(completed)),
            "test_evaluated_once": test_calls == 1,
            "thresholds_frozen_before_test": True,
            "flags": flags,
        }

    def _comparison_table(
        self, families: Mapping[str, Mapping[str, Any]], completed: Iterable[str]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for family in completed:
            block = families[family]
            metrics = block.get("test_metrics") or {}
            counts = metrics.get("confusion_matrix") or {}
            rows.append(
                {
                    "family": family,
                    "label": FAMILY_LABELS[family],
                    "threshold": (block.get("threshold") or {}).get("selected"),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "brier": metrics.get("brier"),
                    "tn": counts.get("tn"),
                    "fp": counts.get("fp"),
                    "fn": counts.get("fn"),
                    "tp": counts.get("tp"),
                    "n": counts.get("n"),
                }
            )
        return rows

    def _error_analysis(
        self,
        families: Mapping[str, Mapping[str, Any]],
        completed: Iterable[str],
        partitions: Mapping[str, np.ndarray],
        predictions: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        labels = partitions["y_test"]
        rows: dict[str, Any] = {}
        for family in completed:
            metrics = families[family].get("test_metrics") or {}
            counts = metrics.get("confusion_matrix") or {}
            scores = np.asarray(predictions[f"test_{family}"], dtype=float)
            threshold = float(families[family]["threshold"]["selected"])
            predicted = (scores >= threshold).astype(int)
            false_positive_scores = scores[(labels == 0) & (predicted == 1)]
            false_negative_scores = scores[(labels == 1) & (predicted == 0)]
            rows[family] = {
                "false_positives": counts.get("fp"),
                "false_negatives": counts.get("fn"),
                "false_positive_rate_of_negatives": _safe_div(
                    counts.get("fp"), (counts.get("fp") or 0) + (counts.get("tn") or 0)
                ),
                "false_negative_rate_of_positives": _safe_div(
                    counts.get("fn"), (counts.get("fn") or 0) + (counts.get("tp") or 0)
                ),
                "mean_probability_false_positives": (
                    float(false_positive_scores.mean())
                    if len(false_positive_scores)
                    else None
                ),
                "mean_probability_false_negatives": (
                    float(false_negative_scores.mean())
                    if len(false_negative_scores)
                    else None
                ),
                "note": "Aggregate counts and mean probabilities only; no customer "
                "identifiers or raw records are exposed.",
            }
        return {
            "interpretation": (
                "A false positive is a retained customer predicted to churn (wasted "
                "retention spend). A false negative is a churner predicted to stay "
                "(missed save opportunity). Recall drives missed churners; precision "
                "drives outreach cost."
            ),
            "test_rows": int(len(labels)),
            "test_positive_rows": int(labels.sum()),
            "families": rows,
        }

    def _preprocessing_block(self, dataset: PreparedDataset) -> dict[str, Any]:
        schema = dataset.schema or {}
        return {
            "run_id": dataset.run_id,
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.split_spec.dataset_sha256,
            "split_fingerprint": dataset.split_spec.fingerprint,
            "split_sizes": dict(dataset.split_spec.sizes),
            "split_strategy": dataset.split_spec.strategy,
            "split_seed": int(dataset.split_spec.seed),
            "preprocessor_hash": schema.get("preprocessor_hash"),
            "preprocessor_path": schema.get("preprocessor_path"),
            "pipeline_version": schema.get("pipeline_version"),
            "config_fingerprint": (schema.get("config") or {}).get("fingerprint"),
            "target": dict(schema.get("target") or {}),
            "feature_names": list(dataset.feature_names),
            "feature_count": len(dataset.feature_names),
            "provenance": dict(dataset.provenance or {}),
        }

    def _environment(self) -> dict[str, Any]:
        versions: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "code_version": self.code_version,
        }
        try:
            import pandas

            versions["pandas"] = pandas.__version__
        except ImportError:  # pragma: no cover - dependency is declared
            versions["pandas"] = None
        try:
            import sklearn

            versions["scikit_learn"] = sklearn.__version__
        except ImportError:  # pragma: no cover - dependency is declared
            versions["scikit_learn"] = None
        try:
            import tensorflow

            versions["tensorflow"] = tensorflow.__version__
            versions["keras"] = tensorflow.keras.__version__
        except ImportError:  # pragma: no cover - optional for baseline-only use
            versions["tensorflow"] = None
            versions["keras"] = None
        return versions

    @staticmethod
    def _cv_status(cv_enabled: bool, blocks: Mapping[str, Any]) -> str:
        if not cv_enabled:
            return "disabled"
        if not blocks:
            return "unavailable"
        statuses = {str(block.status) for block in blocks.values()}
        if statuses == {"completed"}:
            return "completed"
        if statuses <= {"unavailable", "failed"}:
            return "unavailable" if "unavailable" in statuses else "failed"
        return "partial"

    def _limitations(
        self,
        completed: Iterable[str],
        cv_enabled: bool,
        cv_blocks: Mapping[str, Any] | None = None,
    ) -> list[str]:
        items = [
            "Metrics describe one held-out split; they are not a guarantee of future "
            "churn behaviour or causal retention impact.",
            "A threshold selected to maximise validation F1 is a statistical choice, "
            "not automatically the best business decision; contact costs and capacity "
            "are modeled in later phases.",
            "The benchmark compares model families on one dataset; results do not "
            "transfer automatically to other customer bases.",
            "Test metrics were computed once. Re-running this benchmark on a new split "
            "produces a new evaluation, not a re-tuned score for this one.",
        ]
        if MLP_FAMILY not in completed:
            items.append("The MLP family failed and is absent from the comparison.")
        if not cv_enabled:
            items.append("Cross-validation was disabled for this run.")
        elif any(
            not (block.preprocessing or {}).get("refitted_per_fold")
            for block in (cv_blocks or {}).values()
        ):
            items.append(
                "Fold-safe cross-validation was unavailable for at least one family "
                "because the Phase 5 base frame could not be reconstructed; see the "
                "recorded reason instead of treating the fold spread as evidence."
            )
        else:
            items.append(
                "Cross-validation refits preprocessing inside every fold, but the fold "
                "split is a single seeded 5-fold partition; it does not quantify "
                "split-to-split variance."
            )
        items.append(
            "Probability calibration is summarised by the Brier score only; no "
            "recalibration was fitted."
        )
        return items

    # -- persistence --------------------------------------------------------
    def _persist(
        self,
        result: BenchmarkResult,
        root: Path,
        *,
        datasets: Mapping[str, np.ndarray],
        predictions: Mapping[str, np.ndarray],
        sweeps: Mapping[str, Mapping[str, Any]],
    ) -> None:
        metrics = result.metrics
        root.mkdir(parents=True, exist_ok=True)
        completed = [
            name for name in SUPPORTED_FAMILIES if metrics["families"].get(name, {}).get("status") == "completed"
        ]

        arrays: dict[str, np.ndarray] = {
            "y_train": np.asarray(datasets["y_train"], dtype=np.int8),
            "y_val": np.asarray(datasets["y_val"], dtype=np.int8),
            "y_test": np.asarray(datasets["y_test"], dtype=np.int8),
        }
        for family in completed:
            for split in ("train", "val", "test"):
                arrays[f"proba_{split}_{family}"] = np.asarray(
                    predictions[f"{split}_{family}"], dtype=np.float64
                )
        predictions_path = root / "predictions.npz"
        try:
            np.savez_compressed(predictions_path, **arrays)
        except OSError as exc:
            raise EvaluationError(
                CODE_ARTIFACT_FAILED,
                f"prediction artifact could not be written: {exc}",
                "Check that the artifact directory is writable.",
            ) from exc

        curves: dict[str, Any] = {}
        for family in completed:
            curves[family] = {
                "roc": roc_curve_data(
                    datasets["y_test"], predictions[f"test_{family}"]
                ),
                "pr": precision_recall_curve_data(
                    datasets["y_test"], predictions[f"test_{family}"]
                ),
            }
        curves_path = root / "curves.json"
        _atomic_write_json(curves_path, curves)

        threshold_payload = {
            "selected_on": "validation",
            "strategy": self.config.threshold_strategy,
            "default_threshold": float(self.config.fixed_threshold),
            "families": {
                family: {
                    **sweeps[family],
                    "validation_metrics": metrics["families"][family]["validation_metrics"],
                }
                for family in completed
            },
        }
        thresholds_path = root / "threshold_analysis.json"
        _atomic_write_json(thresholds_path, threshold_payload)

        plot_paths: dict[str, str] = {}
        for family in completed:
            name = f"confusion_matrix_{family}.png"
            plots.plot_confusion_matrix(
                metrics["families"][family]["test_metrics"], family, root / name
            )
            plot_paths[f"confusion_matrix_{family}"] = name
        plots.plot_roc_curves(curves, root / "roc_curves.png")
        plot_paths["roc_curves"] = "roc_curves.png"
        plots.plot_precision_recall_curves(curves, root / "precision_recall_curves.png")
        plot_paths["precision_recall_curves"] = "precision_recall_curves.png"
        plots.plot_metric_comparison(metrics["families"], root / "metric_comparison.png")
        plot_paths["metric_comparison"] = "metric_comparison.png"
        plots.plot_threshold_analysis(sweeps, root / "threshold_analysis.png")
        plot_paths["threshold_analysis"] = "threshold_analysis.png"
        plots.plot_probability_distributions(
            {family: predictions[f"test_{family}"] for family in completed},
            datasets["y_test"],
            root / "probability_distributions.png",
        )
        plot_paths["probability_distributions"] = "probability_distributions.png"

        model_artifacts: dict[str, Any] = {}
        for family in (LOGISTIC_FAMILY, RANDOM_FOREST_FAMILY):
            block = metrics["families"].get(family) or {}
            if block.get("status") != "completed":
                continue
            fitted = getattr(self, "_fitted_baselines", {}).get(family)
            if fitted is None:
                continue
            metadata = baseline_metadata(fitted, seed=self.config.seed)
            metadata["preprocessing"] = metrics["preprocessing"]
            directory = root / "models" / family
            payload = save_baseline(directory, fitted, metadata)
            metadata_path = directory / "metadata.json"
            _atomic_write_json(metadata_path, payload)
            block.setdefault("model", {})
            block["model"]["artifact_path"] = payload["artifacts"]["model"]
            block["model"]["model_sha256"] = payload["model_sha256"]
            model_artifacts[family] = payload["artifacts"]["model"]

        metrics["plots"] = plot_paths
        metrics["artifacts"] = {
            "root": str(root),
            "metrics": str(root / "metrics.json"),
            "benchmark_markdown": str(root / "benchmark.md"),
            "curves": str(curves_path),
            "threshold_analysis": str(thresholds_path),
            "predictions": str(predictions_path),
            "run_manifest": str(root / "run_manifest.json"),
            "plots": {key: str(root / name) for key, name in plot_paths.items()},
            "baseline_models": model_artifacts,
        }
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "phase": 7,
            "benchmark_run_id": result.benchmark_run_id,
            "generated_at": metrics["generated_at"],
            "code_version": self.code_version,
            "preprocessing_run_id": metrics["preprocessing"]["run_id"],
            "dataset_sha256": metrics["preprocessing"]["dataset_sha256"],
            "split_fingerprint": metrics["preprocessing"]["split_fingerprint"],
            "preprocessor_hash": metrics["preprocessing"]["preprocessor_hash"],
            "selected_model_family": metrics["selection"].get("selected_family"),
            "selection_metric": metrics["selection"].get("selection_metric"),
            "selection_rationale": metrics["selection"].get("rationale"),
            "thresholds": {
                family: metrics["families"][family]["threshold"]["selected"]
                for family in completed
            },
            "test_rows_evaluated": metrics["dataset"]["n_test"],
            "test_evaluation_calls": metrics["validity"]["test_evaluation_calls"],
            "complete": metrics["complete"],
            "families": {
                family: {
                    "status": metrics["families"][family]["status"],
                    "source": metrics["families"][family].get("source"),
                }
                for family in SUPPORTED_FAMILIES
            },
            "artifacts": metrics["artifacts"],
            "warnings": list(result.warnings),
        }
        _atomic_write_json(root / "run_manifest.json", manifest)
        _atomic_write_json(root / "metrics.json", metrics)
        (root / "benchmark.md").write_text(
            render_benchmark_markdown(result), encoding="utf-8"
        )
        result.paths.update(
            {
                "root": str(root),
                "metrics": str(root / "metrics.json"),
                "markdown": str(root / "benchmark.md"),
                "predictions": str(predictions_path),
                "curves": str(curves_path),
                "threshold_analysis": str(thresholds_path),
                "run_manifest": str(root / "run_manifest.json"),
            }
        )

    # -- reload / regenerate ------------------------------------------------
    def load(self, benchmark_run_id: str) -> BenchmarkResult:
        validate_benchmark_id(benchmark_run_id)
        root = self.evaluation_dir / benchmark_run_id
        metrics_path = root / "metrics.json"
        if not metrics_path.is_file():
            raise EvaluationError(
                CODE_ARTIFACT_FAILED,
                f"benchmark {benchmark_run_id!r} has no metrics.json",
                "Run the evaluate command first.",
            )
        metrics = _read_json(metrics_path)
        result = BenchmarkResult(
            benchmark_run_id=benchmark_run_id,
            metrics=metrics,
            paths={
                "root": str(root),
                "metrics": str(metrics_path),
                "markdown": str(root / "benchmark.md"),
                "predictions": str(root / "predictions.npz"),
                "curves": str(root / "curves.json"),
                "threshold_analysis": str(root / "threshold_analysis.json"),
                "run_manifest": str(root / "run_manifest.json"),
            },
            warnings=list(metrics.get("warnings") or []),
        )
        return result

    def regenerate(self, benchmark_run_id: str) -> BenchmarkResult:
        """Recompute every metric and figure from stored predictions.

        No model is refitted and no test row is scored again: the saved
        probability arrays are the only source. Recomputed values must match the
        stored ones within ``1e-9`` or the artifact is reported as corrupt.
        """
        result = self.load(benchmark_run_id)
        root = Path(result.paths["root"])
        predictions_path = root / "predictions.npz"
        if not predictions_path.is_file():
            raise EvaluationError(
                CODE_ARTIFACT_FAILED,
                "stored predictions are missing; metrics cannot be regenerated",
                "Rerun the evaluate command to rebuild the benchmark.",
            )
        try:
            stored = np.load(predictions_path, allow_pickle=False)
            arrays = {key: stored[key] for key in stored.files}
        except (OSError, ValueError) as exc:
            raise EvaluationError(
                CODE_ARTIFACT_FAILED,
                f"stored predictions could not be read: {type(exc).__name__}: {exc}",
                "Rerun the evaluate command to rebuild the benchmark.",
            ) from exc
        metrics = result.metrics
        completed = [
            name
            for name in SUPPORTED_FAMILIES
            if metrics["families"].get(name, {}).get("status") == "completed"
        ]
        curves: dict[str, Any] = {}
        for family in completed:
            block = metrics["families"][family]
            threshold = float(block["threshold"]["selected"])
            recomputed = binary_metrics(
                arrays["y_test"], arrays[f"proba_test_{family}"], threshold
            )
            self._assert_unchanged(family, block["test_metrics"], recomputed)
            block["test_metrics"] = recomputed
            block["test_metrics_at_0_5"] = binary_metrics(
                arrays["y_test"],
                arrays[f"proba_test_{family}"],
                float(self.config.fixed_threshold),
            )
            block["validation_metrics"] = binary_metrics(
                arrays["y_val"], arrays[f"proba_val_{family}"], threshold
            )
            block["validation_metrics_at_0_5"] = binary_metrics(
                arrays["y_val"],
                arrays[f"proba_val_{family}"],
                float(self.config.fixed_threshold),
            )
            block["train_metrics"] = binary_metrics(
                arrays["y_train"], arrays[f"proba_train_{family}"], threshold
            )
            curves[family] = {
                "roc": roc_curve_data(arrays["y_test"], arrays[f"proba_test_{family}"]),
                "pr": precision_recall_curve_data(
                    arrays["y_test"], arrays[f"proba_test_{family}"]
                ),
            }
            plots.plot_confusion_matrix(
                block["test_metrics"],
                family,
                root / f"confusion_matrix_{family}.png",
            )
        _atomic_write_json(root / "curves.json", curves)
        plots.plot_roc_curves(curves, root / "roc_curves.png")
        plots.plot_precision_recall_curves(curves, root / "precision_recall_curves.png")
        plots.plot_metric_comparison(metrics["families"], root / "metric_comparison.png")
        plots.plot_probability_distributions(
            {family: arrays[f"proba_test_{family}"] for family in completed},
            arrays["y_test"],
            root / "probability_distributions.png",
        )
        metrics["comparison_table"] = self._comparison_table(metrics["families"], completed)
        metrics["regenerated_at"] = self._now()
        metrics["regeneration"] = {
            "source": str(predictions_path),
            "models_refitted": False,
            "test_rows_rescored": False,
            "verified_against_stored_metrics": True,
        }
        _atomic_write_json(root / "metrics.json", metrics)
        (root / "benchmark.md").write_text(
            render_benchmark_markdown(result), encoding="utf-8"
        )
        result.metrics["summary"] = render_summary(result)
        return result

    def _assert_unchanged(
        self,
        family: str,
        stored: Mapping[str, Any],
        recomputed: Mapping[str, Any],
    ) -> None:
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
            left = _finite_or_none(stored.get(key))
            right = _finite_or_none(recomputed.get(key))
            if left is None and right is None:
                continue
            if left is None or right is None or abs(left - right) > 1e-9:
                raise EvaluationError(
                    CODE_METRIC_FAILED,
                    f"regenerated {family} {key} differs from the stored value "
                    f"({left!r} vs {right!r})",
                    "Treat the benchmark artifact as corrupt and rerun the evaluation.",
                )
        stored_counts = stored.get("confusion_matrix") or {}
        recomputed_counts = recomputed.get("confusion_matrix") or {}
        for key in ("tn", "fp", "fn", "tp"):
            if int(stored_counts.get(key, -1)) != int(recomputed_counts.get(key, -2)):
                raise EvaluationError(
                    CODE_METRIC_FAILED,
                    f"regenerated {family} confusion matrix differs from the stored one",
                    "Treat the benchmark artifact as corrupt and rerun the evaluation.",
                )

    def latest_run_id(self) -> str | None:
        if not self.evaluation_dir.is_dir():
            return None
        candidates = sorted(
            path.name
            for path in self.evaluation_dir.iterdir()
            if (path / "metrics.json").is_file()
        )
        return candidates[-1] if candidates else None

    # -- helpers ------------------------------------------------------------
    def _default_run_id(self, dataset: PreparedDataset, generated_at: str) -> str:
        stamp = generated_at.replace("-", "").replace(":", "")
        if not stamp.endswith("Z"):
            stamp += "Z"
        return (
            f"{stamp}-{_safe_name(dataset.dataset_id)}-bench-"
            f"{self.config.fingerprint()[:8]}"
        )


def _gap(left: Any, right: Any) -> float | None:
    first = _finite_or_none(left)
    second = _finite_or_none(right)
    if first is None or second is None:
        return None
    return float(first - second)


def _matrix_digest(matrix: np.ndarray) -> str:
    """Stable digest of the exact float64 matrix every family consumes."""
    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return None
    if bottom <= 0:
        return None
    return top / bottom


def evaluate(
    dataset: PreparedDataset,
    *,
    settings: Settings,
    config: EvaluationConfig | None = None,
    mlp_run_id: str | None = None,
    benchmark_run_id: str | None = None,
    persist: bool = True,
) -> BenchmarkResult:
    """One-call convenience wrapper around :class:`BenchmarkCoordinator`."""
    return BenchmarkCoordinator(settings, config).run(
        dataset,
        mlp_run_id=mlp_run_id,
        benchmark_run_id=benchmark_run_id,
        persist=persist,
    )


__all__ = [
    "BenchmarkCoordinator",
    "evaluate",
    "validate_benchmark_id",
]
