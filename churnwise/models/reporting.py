"""Training diagnostics, learning curves and Phase 6 reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from churnwise.models.contracts import TrainingResult


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def plot_history(history: dict[str, list[float]], path: Path) -> Path:
    """Plot the actual training and validation loss/accuracy curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.get("loss", [])) + 1))
    has_accuracy = bool(history.get("accuracy")) and bool(history.get("val_accuracy"))
    has_lr = bool(history.get("learning_rate"))
    panels = 1 + int(has_accuracy) + int(has_lr)
    figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5))
    axes = axes.tolist() if hasattr(axes, "tolist") else [axes]
    loss_axis = axes[0]
    loss_axis.plot(epochs, history.get("loss", []), label="train loss", color="#1f77b4")
    loss_axis.plot(
        epochs, history.get("val_loss", []), label="validation loss", color="#d62728"
    )
    loss_axis.set_title("Binary cross-entropy loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    next_panel = 1
    if has_accuracy:
        accuracy_axis = axes[next_panel]
        next_panel += 1
        accuracy_axis.plot(
            epochs, history.get("accuracy", []), label="train accuracy", color="#1f77b4"
        )
        accuracy_axis.plot(
            epochs, history.get("val_accuracy", []), label="validation accuracy",
            color="#d62728",
        )
        accuracy_axis.set_title("Binary accuracy")
        accuracy_axis.set_xlabel("Epoch")
        accuracy_axis.set_ylabel("Accuracy")
        accuracy_axis.set_ylim(0.0, 1.0)
        accuracy_axis.grid(alpha=0.25)
        accuracy_axis.legend()
    if has_lr:
        lr_axis = axes[next_panel]
        lr_axis.plot(epochs, history.get("learning_rate", []), label="learning rate", color="#2ca02c")
        lr_axis.set_title("Adam learning rate")
        lr_axis.set_xlabel("Epoch")
        lr_axis.set_ylabel("Learning rate")
        lr_axis.grid(alpha=0.25)
        lr_axis.legend()
    figure.suptitle("ChurnWise MLP — training history", fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def render_summary(result: TrainingResult) -> str:
    metadata = result.metadata
    diagnostics = result.diagnostics
    architecture = metadata.get("architecture", {})
    training = metadata.get("training", {})
    lines = [
        "ChurnWise AI - MLP training",
        "",
        f"Model run: {result.run_id}",
        f"Preprocessing run: {metadata.get('preprocessing', {}).get('run_id', '')}",
        f"Dataset: {metadata.get('preprocessing', {}).get('dataset_id', '')}",
        "Architecture: "
        + " -> ".join(str(units) for units in architecture.get("hidden_units", []))
        + f" -> 1 (input {architecture.get('input_dim', '?')}, sigmoid output, "
        f"{architecture.get('parameter_count', '?')} parameters)",
        f"Training: {training.get('optimizer', 'adam')} loss={training.get('loss')} "
        f"lr={training.get('learning_rate')} batch={training.get('batch_size')} "
        f"epochs={diagnostics.get('epochs_completed')}/{training.get('epochs_requested')}",
        f"Best validation loss: {diagnostics.get('best_val_loss'):.6f} "
        f"at epoch {diagnostics.get('best_epoch')}"
        if diagnostics.get("best_val_loss") is not None
        else "Best validation loss: n/a",
        f"Early stopping: {'yes' if diagnostics.get('stopped_early') else 'no'}"
        f" (patience {training.get('early_stopping_patience')})",
        f"Features: {diagnostics.get('feature_count')} transformed",
        f"Training rows: {diagnostics.get('train_rows')} "
        f"(positives {diagnostics.get('train_positive_count')}, "
        f"rate {diagnostics.get('train_positive_rate'):.4f})"
        if diagnostics.get("train_positive_rate") is not None
        else f"Training rows: {diagnostics.get('train_rows')}",
        f"Class weights: {training.get('class_weight')}",
        f"Validation rows: {diagnostics.get('val_rows')} "
        f"(positives {diagnostics.get('val_positive_count')}, "
        f"test rows used: {diagnostics.get('test_rows_used')})",
        f"Duration: {diagnostics.get('duration_seconds', 0.0):.2f} s",
    ]
    if result.paths.get("model"):
        lines.append(f"Model: {result.paths['model']}")
    if result.paths.get("curves"):
        lines.append(f"Learning curves: {result.paths['curves']}")
    if result.paths.get("report_md"):
        lines.append(f"Report: {result.paths['report_md']}")
    warnings = metadata.get("warnings", [])
    for warning in warnings:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _markdown(result: TrainingResult) -> str:
    metadata = result.metadata
    diagnostics = result.diagnostics
    architecture = metadata.get("architecture", {})
    training = metadata.get("training", {})
    preprocessing = metadata.get("preprocessing", {})
    history = result.history
    lines = [
        f"# MLP training report - {result.run_id}",
        "",
        f"- **Preprocessing run:** `{preprocessing.get('run_id', '')}`",
        f"- **Dataset:** `{preprocessing.get('dataset_id', '')}`",
        f"- **Dataset SHA-256:** `{preprocessing.get('dataset_sha256', '')}`",
        f"- **Split fingerprint:** `{preprocessing.get('split_fingerprint', '')}`",
        f"- **Preprocessor hash:** `{preprocessing.get('preprocessor_hash', '')}`",
        f"- **Created:** {metadata.get('created_at', '')}",
        f"- **Code version:** {metadata.get('code_version', '')}",
        "",
        "## Architecture",
        "",
        f"- Input features: {architecture.get('input_dim')}",
        f"- Hidden layers: {architecture.get('hidden_units')} (ReLU)",
        f"- Dropout: {architecture.get('dropout')}",
        f"- L2: {architecture.get('l2')}",
        f"- Output: sigmoid probability of churn",
        f"- Parameters: {architecture.get('parameter_count')}",
        "",
        "## Training configuration",
        "",
        f"- Optimizer: `{training.get('optimizer')}`",
        f"- Learning rate: {training.get('learning_rate')}",
        f"- Loss: `{training.get('loss')}`",
        f"- Batch size: {training.get('batch_size')}",
        f"- Epoch limit: {training.get('epochs_requested')}",
        f"- Early stopping: patience {training.get('early_stopping_patience')} on `val_loss`",
        f"- ReduceLROnPlateau: factor {training.get('reduce_lr_factor')}, "
        f"patience {training.get('reduce_lr_patience')}, min lr {training.get('min_learning_rate')}",
        f"- Class weights: `{training.get('class_weight')}`",
        f"- Training rows: {diagnostics.get('train_rows')} "
        f"(positive rate {diagnostics.get('train_positive_rate')})",
        f"- Validation rows: {diagnostics.get('val_rows')} "
        f"(positive rate {diagnostics.get('val_positive_rate')})",
        f"- Test rows used: {diagnostics.get('test_rows_used')}",
        "",
        "## Training outcome",
        "",
        f"- Epochs completed: {diagnostics.get('epochs_completed')}",
        f"- Best epoch: {diagnostics.get('best_epoch')}",
        f"- Best validation loss: {diagnostics.get('best_val_loss')}",
        f"- Early stopped: {diagnostics.get('stopped_early')}",
        f"- Duration: {diagnostics.get('duration_seconds'):.2f} s",
        f"- Warnings: {len(metadata.get('warnings', []))}",
        "",
        "## Learning curves",
        "",
    ]
    if result.paths.get("curves"):
        lines.append(f"![Learning curves]({result.paths['curves']})")
    else:
        lines.append("No learning-curve plot was produced.")
    lines.extend(["", "## Per-epoch history", ""])
    keys = [
        key
        for key in ("loss", "val_loss", "accuracy", "val_accuracy", "learning_rate")
        if history.get(key)
    ]
    if keys:
        lines.append("| Epoch | " + " | ".join(keys) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in keys) + " |")
        for index in range(len(history[keys[0]])):
            values = [
                f"{history[key][index]:.6g}" if index < len(history[key]) else ""
                for key in keys
            ]
            lines.append(f"| {index + 1} | " + " | ".join(values) + " |")
    else:
        lines.append("No per-epoch history was recorded.")
    if metadata.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in metadata["warnings"])
    lines.extend(["", "## Artifacts", ""])
    for key, value in result.paths.items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def write_reports(result: TrainingResult, report_dir: Path | str) -> dict[str, str]:
    root = Path(report_dir) / "models"
    json_path = root / f"{result.run_id}.json"
    md_path = root / f"{result.run_id}.md"
    payload = {
        "metadata": result.metadata,
        "history": result.history,
        "diagnostics": result.diagnostics,
        "paths": dict(result.paths),
    }
    _atomic_write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}


__all__ = ["plot_history", "render_summary", "write_reports"]
