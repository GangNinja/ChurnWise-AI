"""Visualizations built only from computed SHAP values.

Every plot takes an existing contribution matrix; nothing here invents values.
Matplotlib uses the Agg backend so the CLI works headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from churnwise.explain.errors import CODE_ARTIFACT_FAILED, ExplainError


def _save(figure: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=140, bbox_inches="tight")
    except OSError as exc:
        raise ExplainError(
            CODE_ARTIFACT_FAILED,
            f"figure could not be written: {path}: {exc}",
            "Check that the artifact directory is writable.",
        ) from exc
    finally:
        plt.close(figure)
    return str(path)


def plot_importance(
    importance: Sequence[dict[str, Any]],
    path: Path | str,
    *,
    title: str,
    output_space: str,
    limit: int = 15,
) -> str:
    rows = list(importance)[: max(1, int(limit))][::-1]
    labels = [str(row["feature"]) for row in rows]
    values = [float(row["mean_abs_shap"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.32 * len(rows) + 1.2)))
    axis.barh(labels, values, color="#2f6f8f")
    axis.set_xlabel(f"mean |SHAP| ({output_space})")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    for position, value in enumerate(values):
        axis.text(value, position, f" {value:.4g}", va="center", fontsize=8)
    figure.tight_layout()
    return _save(figure, Path(path))


def plot_beeswarm(
    values: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    path: Path | str,
    *,
    output_space: str,
    max_display: int = 15,
) -> str:
    try:
        import shap

        shap.summary_plot(
            np.asarray(values, dtype=np.float64),
            np.asarray(features, dtype=np.float64),
            feature_names=[str(name) for name in feature_names],
            max_display=int(max_display),
            show=False,
            plot_size=(9, 6),
        )
        figure = plt.gcf()
        figure.axes[0].set_xlabel(f"SHAP value ({output_space})")
        figure.tight_layout()
        return _save(figure, Path(path))
    except ExplainError:
        raise
    except Exception as exc:  # noqa: BLE001 - optional visualization
        plt.close("all")
        raise ExplainError(
            CODE_ARTIFACT_FAILED,
            f"SHAP summary plot failed: {type(exc).__name__}: {exc}",
            "The explanation JSON is still valid; inspect the SHAP/Matplotlib versions.",
        ) from exc


def plot_dependence(
    feature_name: str,
    feature_values: np.ndarray,
    contributions: np.ndarray,
    path: Path | str,
    *,
    output_space: str,
) -> str:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(
        np.asarray(feature_values, dtype=np.float64),
        np.asarray(contributions, dtype=np.float64),
        s=14,
        alpha=0.7,
        color="#4b7f52",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlabel(f"{feature_name} (transformed value)")
    axis.set_ylabel(f"SHAP contribution ({output_space})")
    axis.set_title(f"Dependence: {feature_name}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return _save(figure, Path(path))


def plot_waterfall(
    row_values: np.ndarray,
    base_value: float,
    row_features: np.ndarray,
    feature_names: Sequence[str],
    path: Path | str,
    *,
    output_space: str,
    max_display: int = 10,
) -> str:
    try:
        import shap

        explanation = shap.Explanation(
            values=np.asarray(row_values, dtype=np.float64),
            base_values=float(base_value),
            data=np.asarray(row_features, dtype=np.float64),
            feature_names=[str(name) for name in feature_names],
        )
        shap.plots.waterfall(explanation, max_display=int(max_display), show=False)
        figure = plt.gcf()
        figure.suptitle(f"Local SHAP contributions ({output_space})", fontsize=10)
        figure.tight_layout()
        return _save(figure, Path(path))
    except ExplainError:
        raise
    except Exception as exc:  # noqa: BLE001 - optional visualization
        plt.close("all")
        raise ExplainError(
            CODE_ARTIFACT_FAILED,
            f"waterfall plot failed: {type(exc).__name__}: {exc}",
            "The local explanation JSON is still valid; inspect the SHAP version.",
        ) from exc


__all__ = [
    "plot_beeswarm",
    "plot_dependence",
    "plot_importance",
    "plot_waterfall",
]
