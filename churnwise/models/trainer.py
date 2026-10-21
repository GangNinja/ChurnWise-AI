"""Phase 6 training workflow: Phase 5 arrays -> fitted, auditable MLP."""

from __future__ import annotations

import platform
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from churnwise import __version__
from churnwise.config import Settings
from churnwise.contracts import utc_now_iso
from churnwise.models.contracts import (
    MODEL_FAMILY,
    MODEL_SCHEMA_VERSION,
    MLPConfig,
    TrainingResult,
)
from churnwise.models.errors import (
    CODE_INVALID_MODEL,
    CODE_TRAINING_FAILED,
    ModelingError,
)
from churnwise.models.mlp import (
    architecture_summary,
    build_mlp,
    parameter_count,
    set_global_seed,
    validate_config,
)
from churnwise.models.registry import ModelRegistry, validate_model_run_id
from churnwise.models.reporting import plot_history, write_reports
from churnwise.preprocessing import PreparedDataset


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:100] or "dataset"


def _default_model_run_id(dataset: PreparedDataset, config: MLPConfig, generated_at: str) -> str:
    stamp = generated_at.replace("-", "").replace(":", "")
    if not stamp.endswith("Z"):
        stamp += "Z"
    return f"{stamp}-{_safe_name(dataset.dataset_id)}-mlp-{config.fingerprint()[:8]}"


def _as_matrix(values: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} features are not a numeric matrix: {type(exc).__name__}: {exc}",
            "Use the Phase 5 PreparedDataset partitions or a numeric matrix.",
        ) from exc
    if matrix.ndim != 2:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} features must be 2-D, got shape {matrix.shape}",
            "Use the Phase 5 PreparedDataset partitions.",
        )
    if not np.isfinite(matrix).all():
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} features contain non-finite values",
            "Regenerate the Phase 5 run and check the invalid-value report.",
        )
    return matrix


def _as_labels(values: Any, name: str) -> np.ndarray:
    try:
        labels = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} labels are not numeric: {type(exc).__name__}: {exc}",
            "Use 0/1 churn labels from the Phase 5 target mapping.",
        ) from exc
    if len(labels) == 0:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} labels are empty",
            "Regenerate the Phase 5 split with a larger dataset.",
        )
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} labels must be 0/1 churn indicators",
            "Use the Phase 5 target mapping; positive means churned.",
        )
    return labels.reshape(-1, 1)


def _validate_partition(dataset: PreparedDataset, name: str, matrix: np.ndarray, labels: np.ndarray) -> None:
    if len(matrix) != len(labels):
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} has {len(matrix)} feature rows but {len(labels)} labels",
            "Regenerate the Phase 5 run; the partitions are inconsistent.",
        )
    if matrix.shape[1] != len(dataset.feature_names):
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"{name} has {matrix.shape[1]} features; schema expects {len(dataset.feature_names)}",
            "Reload the Phase 5 run from the same preprocessing run id.",
        )


def _validate_split_isolation(dataset: PreparedDataset) -> None:
    """Reject overlapping train/validation/test source rows before fitting.

    Phase 5 creates disjoint partitions, but the model boundary should not rely
    on that assumption: an accidental overlap would leak evaluation rows into
    weight updates or early stopping.
    """
    try:
        rows = {
            name: {int(index) for index in getattr(dataset, name).row_indices}
            for name in ("train", "val", "test")
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelingError(
            CODE_INVALID_MODEL,
            f"split row indices are not readable: {type(exc).__name__}: {exc}",
            "Regenerate the Phase 5 split and retry.",
        ) from exc
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = rows[left] & rows[right]
        if overlap:
            raise ModelingError(
                CODE_INVALID_MODEL,
                f"{left} and {right} splits share {len(overlap)} source rows",
                "Regenerate the Phase 5 split; partitions must be disjoint.",
            )


def _epoch_metrics(history: dict[str, list[float]]) -> list[str]:
    return sorted(history)


class MLPTrainer:
    """Fit the churn MLP using training rows only."""

    def __init__(self, settings: Settings, config: MLPConfig | None = None) -> None:
        self.settings = settings
        self.config = config or MLPConfig(seed=settings.seed)
        validate_config(self.config)
        self.registry = ModelRegistry(settings)

    def train(
        self,
        dataset: PreparedDataset,
        *,
        model_run_id: str | None = None,
        persist: bool = True,
    ) -> TrainingResult:
        config = self.config
        validate_config(config)
        generated_at = utc_now_iso()
        model_run_id = model_run_id or _default_model_run_id(dataset, config, generated_at)
        validate_model_run_id(model_run_id)

        train = dataset.resampled_train() if dataset.oversample_indices else dataset.train
        X_train = _as_matrix(train.X, "training")
        y_train = _as_labels(train.y, "training")
        X_val = _as_matrix(dataset.val.X, "validation")
        y_val = _as_labels(dataset.val.y, "validation")
        _validate_partition(dataset, "training", X_train, y_train)
        _validate_partition(dataset, "validation", X_val, y_val)
        _validate_split_isolation(dataset)
        if len(X_val) == 0:
            raise ModelingError(
                CODE_INVALID_MODEL,
                "validation split is empty; early stopping needs validation rows",
                "Regenerate the Phase 5 split with a larger dataset.",
            )

        try:
            # PreparedDataset.class_weights converts schema strings already;
            # catch malformed upstream metadata instead of leaking ValueError.
            raw_class_weight = dict(dataset.class_weights) or {0: 1.0, 1: 1.0}
            class_weight = {
                int(key): float(value) for key, value in raw_class_weight.items()
            }
        except (TypeError, ValueError) as exc:
            raise ModelingError(
                CODE_INVALID_MODEL,
                f"class weights are not numeric: {type(exc).__name__}: {exc}",
                "Regenerate the Phase 5 class weights from the training split.",
            ) from exc
        if any(not np.isfinite(value) or value < 0 for value in class_weight.values()):
            raise ModelingError(
                CODE_INVALID_MODEL,
                "class weights must be finite and non-negative",
                "Regenerate the Phase 5 class weights from the training split.",
            )
        seed_warnings = set_global_seed(config.seed, deterministic=config.deterministic)
        model = build_mlp(X_train.shape[1], config)
        started = time.perf_counter()

        temporary: tempfile.TemporaryDirectory[str] | None = None
        if persist:
            paths = {key: str(value) for key, value in self.registry.paths_for(model_run_id).items()}
            checkpoint = Path(paths["model"]).parent / "best.keras"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            paths["checkpoint"] = str(checkpoint)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="churnwise-mlp-")
            checkpoint = Path(temporary.name) / "best.keras"
            paths = {}

        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
                verbose=0,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        try:
            fitted = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=config.epochs,
                batch_size=config.batch_size,
                class_weight=class_weight,
                shuffle=config.shuffle,
                callbacks=callbacks,
                verbose=config.verbose,
            )
        except KeyboardInterrupt as exc:
            raise ModelingError(
                CODE_TRAINING_FAILED,
                "training was interrupted before completion",
                "No model artifact was published; rerun the train command when ready.",
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise ModelingError(
                CODE_TRAINING_FAILED,
                f"MLP training failed: {type(exc).__name__}: {exc}",
                "Check the training/validation shapes, class weights and TensorFlow installation.",
            ) from exc

        history = {
            str(key): [float(value) for value in values]
            for key, values in (fitted.history or {}).items()
        }
        if not history.get("loss") or not history.get("val_loss"):
            raise ModelingError(
                CODE_TRAINING_FAILED,
                "training produced no loss history",
                "Check TensorFlow/Keras compatibility and retry.",
            )
        lengths = {key: len(values) for key, values in history.items()}
        if len(set(lengths.values())) != 1:
            raise ModelingError(
                CODE_TRAINING_FAILED,
                f"training history is incomplete: {lengths}",
                "Retry the run; interrupted histories are not usable artifacts.",
            )
        if not all(np.isfinite(value) for values in history.values() for value in values):
            raise ModelingError(
                CODE_TRAINING_FAILED,
                "training produced a non-finite loss or metric",
                "Lower the learning rate, reduce the network size, or inspect the input features.",
            )

        val_loss = np.asarray(history["val_loss"], dtype=float)
        best_index = int(np.argmin(val_loss))
        best_epoch = best_index + 1
        best_val_loss = float(val_loss[best_index])
        epochs_completed = len(val_loss)
        stopped_early = epochs_completed < config.epochs

        if not checkpoint.is_file():
            raise ModelingError(
                CODE_TRAINING_FAILED,
                "no best validation checkpoint was written",
                "Check that TensorFlow can write to the artifact directory and retry.",
            )
        try:
            import tensorflow as tf

            best_model = tf.keras.models.load_model(
                str(checkpoint), compile=False, safe_mode=True
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelingError(
                CODE_TRAINING_FAILED,
                f"best validation checkpoint could not be reloaded: {type(exc).__name__}: {exc}",
                "Retry the run and verify the TensorFlow/Keras installation.",
            ) from exc

        duration = time.perf_counter() - started
        diagnostics = self._diagnostics(
            dataset,
            history,
            train_labels=y_train,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            epochs_completed=epochs_completed,
            stopped_early=stopped_early,
            duration=duration,
            class_weight=class_weight,
        )
        warnings = [
            *seed_warnings,
            *self._warnings(history, diagnostics, config),
        ]
        diagnostics["warnings"] = len(warnings)
        metadata = self._metadata(
            dataset,
            model_run_id=model_run_id,
            generated_at=generated_at,
            config=config,
            model=best_model,
            history=history,
            diagnostics=diagnostics,
            warnings=warnings,
            class_weight=class_weight,
            paths=paths,
        )
        result = TrainingResult(
            model=best_model,
            history=history,
            metadata=metadata,
            paths=dict(paths),
            diagnostics=diagnostics,
        )

        if persist:
            model_path, model_hash = self.registry.save_model(model_run_id, best_model)
            history_path = self.registry.save_json(
                Path(paths["history"]), history
            )
            curves_path = Path(paths["curves"])
            try:
                plot_history(history, curves_path)
                paths["curves"] = str(curves_path)
            except (OSError, ValueError) as exc:  # pragma: no cover - plotting backend
                warnings.append(f"learning-curve plot could not be written: {exc}")
                paths.pop("curves", None)
            metadata["model_sha256"] = model_hash
            metadata["model_size_bytes"] = int(Path(model_path).stat().st_size)
            metadata["artifacts"].update(
                {
                    "model": model_path,
                    "checkpoint": paths.get("checkpoint"),
                    "history": history_path,
                    "curves": paths.get("curves"),
                    "metadata": paths["metadata"],
                    "report_json": paths.get("report_json"),
                    "report_md": paths.get("report_md"),
                }
            )
            metadata["warnings"] = warnings
            self.registry.save_json(Path(paths["metadata"]), metadata)
            result.paths.update(
                {
                    "model": model_path,
                    "checkpoint": paths.get("checkpoint", ""),
                    "history": history_path,
                    "curves": paths.get("curves", ""),
                    "metadata": paths["metadata"],
                }
            )
            report_paths = write_reports(result, self.settings.report_dir)
            result.paths.update(report_paths)
            metadata["artifacts"]["report_json"] = report_paths["report_json"]
            metadata["artifacts"]["report_md"] = report_paths["report_md"]
            self.registry.save_json(Path(paths["metadata"]), metadata)
            result.paths.update(report_paths)

        if temporary is not None:
            temporary.cleanup()
        return result

    # -- diagnostics -------------------------------------------------------
    def _diagnostics(
        self,
        dataset: PreparedDataset,
        history: dict[str, list[float]],
        *,
        train_labels: np.ndarray,
        best_epoch: int,
        best_val_loss: float,
        epochs_completed: int,
        stopped_early: bool,
        duration: float,
        class_weight: dict[int, float],
    ) -> dict[str, Any]:
        # Use the labels actually fitted (which include training-only
        # oversampling when Phase 5 configured it), not the raw train split.
        train_y = np.asarray(train_labels, dtype=int).reshape(-1)
        val_y = np.asarray(dataset.val.y, dtype=int)
        return {
            "train_rows": int(len(train_y)),
            "val_rows": int(len(val_y)),
            "test_rows_used": 0,
            "train_positive_count": int(train_y.sum()),
            "val_positive_count": int(val_y.sum()),
            "train_positive_rate": float(train_y.mean()) if len(train_y) else None,
            "val_positive_rate": float(val_y.mean()) if len(val_y) else None,
            "feature_count": len(dataset.feature_names),
            "epochs_completed": int(epochs_completed),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "stopped_early": bool(stopped_early),
            "max_epochs_reached": bool(epochs_completed >= self.config.epochs),
            "duration_seconds": float(duration),
            "class_weight": {str(key): float(value) for key, value in class_weight.items()},
            "history_metrics": _epoch_metrics(history),
        }

    def _warnings(
        self,
        history: dict[str, list[float]],
        diagnostics: dict[str, Any],
        config: MLPConfig,
    ) -> list[str]:
        warnings: list[str] = []
        if diagnostics["train_rows"] < 500:
            warnings.append(
                f"only {diagnostics['train_rows']} training rows; the model may be unstable"
            )
        if diagnostics["val_rows"] < 100:
            warnings.append(
                f"only {diagnostics['val_rows']} validation rows; early-stopping estimates are noisy"
            )
        if diagnostics["val_positive_count"] < 10:
            warnings.append(
                f"only {diagnostics['val_positive_count']} positive validation rows; validation loss is noisy"
            )
        if diagnostics["max_epochs_reached"]:
            warnings.append(
                f"maximum epoch limit ({config.epochs}) reached; early stopping did not trigger"
            )
        if diagnostics["best_epoch"] == 1:
            warnings.append("best validation loss occurred at the first epoch")
        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        if loss and val_loss and val_loss[-1] - loss[-1] > 0.1:
            warnings.append("validation loss is materially higher than training loss; overfitting is possible")
        if diagnostics["train_positive_count"] in (0, diagnostics["train_rows"]):
            warnings.append("training labels contain a single class; the model cannot learn churn")
        if not config.deterministic:
            warnings.append(
                "TensorFlow op determinism is disabled; repeated runs may vary at floating-point precision"
            )
        return warnings

    # -- metadata ----------------------------------------------------------
    def _metadata(
        self,
        dataset: PreparedDataset,
        *,
        model_run_id: str,
        generated_at: str,
        config: MLPConfig,
        model: Any,
        history: dict[str, list[float]],
        diagnostics: dict[str, Any],
        warnings: list[str],
        class_weight: dict[int, float],
        paths: dict[str, str],
    ) -> dict[str, Any]:
        import tensorflow as tf

        schema = dataset.schema
        target = dict(schema.get("target") or {})
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "phase": 6,
            "model_family": MODEL_FAMILY,
            "model_run_id": model_run_id,
            "created_at": generated_at,
            "code_version": __version__,
            "seed": config.seed,
            "architecture": {
                "input_dim": int(len(dataset.feature_names)),
                "hidden_units": [int(value) for value in config.hidden_units],
                "dropout": [float(value) for value in config.dropout],
                "activation": "relu",
                "output_activation": "sigmoid",
                "output_units": 1,
                "kernel_initializer": "he_normal",
                "l2": float(config.l2),
                "parameter_count": int(parameter_count(model)),
                "summary": architecture_summary(model),
            },
            "training": {
                **config.payload(),
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "class_weight": {str(key): float(value) for key, value in class_weight.items()},
                "epochs_requested": config.epochs,
                "epochs_completed": diagnostics["epochs_completed"],
                "early_stopping_patience": config.early_stopping_patience,
                "reduce_lr_factor": config.reduce_lr_factor,
                "reduce_lr_patience": config.reduce_lr_patience,
                "min_learning_rate": config.min_learning_rate,
            },
            "preprocessing": {
                "run_id": dataset.run_id,
                "dataset_id": dataset.dataset_id,
                "dataset_sha256": dataset.split_spec.dataset_sha256,
                "split_fingerprint": dataset.split_spec.fingerprint,
                "preprocessor_hash": schema.get("preprocessor_hash"),
                "preprocessor_path": schema.get("preprocessor_path"),
                "feature_count": len(dataset.feature_names),
                "feature_names": list(dataset.feature_names),
                "feature_sources": dataset.feature_sources,
                "target": target,
                "split_sizes": dict(dataset.split_spec.sizes),
                "provenance": dict(dataset.provenance),
            },
            "diagnostics": diagnostics,
            "reproducibility": {
                "seeds": {
                    "python": config.seed,
                    "numpy": config.seed,
                    "tensorflow": config.seed,
                },
                "deterministic_ops": bool(config.deterministic),
                "note": (
                    "Seeds are fixed; TensorFlow op determinism is enabled."
                    if config.deterministic
                    else "Seeds are fixed; TensorFlow op determinism is disabled, so run-to-run floating-point variation is possible."
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
                "keras": tf.keras.__version__,
                "numpy": np.__version__,
            },
            "history_metrics": sorted(history),
            "warnings": list(warnings),
            "artifacts": {
                "model": paths.get("model", ""),
                "checkpoint": paths.get("checkpoint", ""),
                "history": paths.get("history", ""),
                "curves": paths.get("curves", ""),
                "metadata": paths.get("metadata", ""),
            },
        }


__all__ = ["MLPTrainer"]
