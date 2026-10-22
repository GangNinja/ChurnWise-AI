"""Read-only selected-model loading for Phase 9.

The loader deliberately reuses :class:`churnwise.explain.ExplainCoordinator`,
which already verifies the Phase 6/7 artifact, the exact Phase 5 preprocessing
run, feature names and split provenance. Phase 9 does not implement a second
prediction path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from churnwise.config import Settings
from churnwise.explain import ExplainConfig, ExplainCoordinator, ExplainError
from churnwise.intelligence.errors import (
    CODE_INVALID_CONFIG,
    CODE_NO_MODEL,
    RiskError,
)
from churnwise.models.errors import ModelingError
from churnwise.preprocessing.errors import PreprocessingError


@dataclass
class LoadedRiskModel:
    """A verified shipped model plus its exact preprocessing run."""

    family: str
    predict_proba: Callable[[np.ndarray], np.ndarray]
    dataset: Any
    threshold: float
    positive_label: str | None
    model_run_id: str
    model_sha256: str | None
    preprocessing_run_id: str
    benchmark_run_id: str
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_path: str | None = None
    preprocessor_path: str | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(getattr(self.dataset, "feature_names", []) or [])


def load_selected_model(
    settings: Settings,
    *,
    benchmark_run_id: str | None = None,
    classification_threshold: float | None = None,
) -> LoadedRiskModel:
    """Load the Phase 7 selected family through the Phase 8 verification path."""
    coordinator = ExplainCoordinator(
        settings,
        ExplainConfig(seed=settings.seed, local_examples=0, sample_size=1),
    )
    try:
        loaded = coordinator.load_model(None, benchmark_run_id=benchmark_run_id)
    except ExplainError as exc:
        raise RiskError(
            exc.code,
            exc.message,
            exc.remediation or "Run `python -m churnwise.cli evaluate` first.",
            dict(exc.details),
        ) from exc
    except (ModelingError, PreprocessingError) as exc:
        raise RiskError(
            exc.code,
            exc.message,
            getattr(exc, "remediation", ""),
        ) from exc

    run_id = str(loaded.provenance.get("benchmark_run_id") or benchmark_run_id or "")
    if not run_id:
        raise RiskError(
            CODE_NO_MODEL,
            "the selected model has no Phase 7 benchmark provenance",
            "Rerun `python -m churnwise.cli evaluate` so the shipped model is recorded.",
        )
    metrics_path = Path(settings.artifact_dir) / "evaluation" / run_id / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RiskError(
            CODE_NO_MODEL,
            f"Phase 7 metrics for {run_id!r} could not be read: {type(exc).__name__}",
            "Rerun the evaluation before scoring.",
        ) from exc
    selection = metrics.get("selection") or {}
    if str(selection.get("selected_family") or loaded.family) != loaded.family:
        raise RiskError(
            CODE_NO_MODEL,
            "the loaded family does not match the Phase 7 selected family",
            "Reload the benchmark artifact and rerun scoring.",
        )

    threshold = loaded.threshold
    if classification_threshold is not None:
        try:
            threshold = float(classification_threshold)
        except (TypeError, ValueError) as exc:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"classification threshold {classification_threshold!r} is not numeric",
                "Pass a probability between 0 and 1.",
            ) from exc
        if not 0.0 <= threshold <= 1.0:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"classification threshold {threshold!r} is outside [0, 1]",
                "Pass a probability between 0 and 1.",
            )

    provenance = dict(loaded.provenance)
    provenance.update(
        {
            "benchmark_run_id": run_id,
            "selected_metric": selection.get("selection_metric"),
            "selected_value": selection.get("selected_validation_value"),
            "dataset_id": loaded.dataset.dataset_id,
            "dataset_sha256": loaded.dataset.split_spec.dataset_sha256,
            "split_fingerprint": loaded.dataset.split_spec.fingerprint,
            "preprocessor_hash": (loaded.dataset.schema or {}).get("preprocessor_hash"),
            "feature_count": len(getattr(loaded.dataset, "feature_names", []) or []),
            "positive_label": loaded.positive_label,
        }
    )
    return LoadedRiskModel(
        family=loaded.family,
        predict_proba=loaded.predict_proba,
        dataset=loaded.dataset,
        threshold=float(threshold),
        positive_label=loaded.positive_label,
        model_run_id=str(loaded.provenance.get("model_run_id") or ""),
        model_sha256=loaded.provenance.get("model_sha256"),
        preprocessing_run_id=str(loaded.provenance.get("preprocessing_run_id") or loaded.dataset.run_id),
        benchmark_run_id=run_id,
        provenance=provenance,
        metrics=metrics,
        model_path=loaded.provenance.get("model_path"),
        preprocessor_path=str(
            (loaded.dataset.paths or {}).get("preprocessor")
            or (loaded.dataset.schema or {}).get("preprocessor_path")
            or ""
        )
        or None,
    )


__all__ = ["LoadedRiskModel", "load_selected_model"]
