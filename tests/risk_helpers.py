"""Network-free Phase 5/7 fixture for Phase 9 tests.

The Random Forest is genuinely fitted on the synthetic Phase 5 training matrix.
Only the persisted benchmark metadata is assembled compactly; the probability
calculations are never mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from churnwise.config import Settings
from churnwise.models.baselines import (
    baseline_metadata,
    fit_baseline,
    save_baseline,
)
from churnwise.preprocessing import PreprocessingCoordinator
from tests.models_helpers import make_prepared_dataset, persist_phase5

BENCHMARK_ID = "20261021T000000Z-fixture-bench"
MODEL_FAMILY = "random_forest"


def build_risk_fixture(
    tmp_path: Path,
    *,
    rows: int = 240,
    seed: int = 9,
    threshold: float = 0.45,
    family: str = MODEL_FAMILY,
) -> tuple[Settings, Any, str]:
    settings = Settings(project_root=tmp_path)
    dataset = persist_phase5(make_prepared_dataset(rows=rows, seed=seed), settings)
    if family == MODEL_FAMILY:
        fitted = fit_baseline(
            family,
            dataset.train.X,
            dataset.train.y,
            feature_names=dataset.feature_names,
        )
    else:
        fitted = fit_baseline(
            family,
            dataset.train.X,
            dataset.train.y,
            feature_names=dataset.feature_names,
        )
    metadata = baseline_metadata(fitted, seed=42)
    metadata["preprocessing"] = {
        "run_id": dataset.run_id,
        "feature_names": list(dataset.feature_names),
        "split_fingerprint": dataset.split_spec.fingerprint,
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.split_spec.dataset_sha256,
        "preprocessor_hash": (dataset.schema or {}).get("preprocessor_hash"),
    }
    evaluation = settings.artifact_dir / "evaluation" / BENCHMARK_ID
    payload = save_baseline(evaluation / "models" / family, fitted, metadata)
    (evaluation / "models" / family / "metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    metrics = {
        "schema_version": "1.0",
        "phase": 7,
        "benchmark_run_id": BENCHMARK_ID,
        "families": {
            family: {
                "status": "completed",
                "threshold": {"selected": threshold, "default": 0.5},
                "model": {
                    "feature_count": len(dataset.feature_names),
                    "model_sha256": payload["model_sha256"],
                    "model_run_id": f"{BENCHMARK_ID}/{family}",
                },
            }
        },
        "preprocessing": {
            "run_id": dataset.run_id,
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.split_spec.dataset_sha256,
            "split_fingerprint": dataset.split_spec.fingerprint,
            "feature_names": list(dataset.feature_names),
            "preprocessor_hash": (dataset.schema or {}).get("preprocessor_hash"),
        },
        "selection": {
            "selected_family": family,
            "selection_metric": "f1",
            "selected_validation_value": 0.75,
            "test_used_for_selection": False,
        },
    }
    (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    # Reload through the real Phase 5 reader so paths/provenance are populated.
    reloaded = PreprocessingCoordinator(settings).load(dataset.run_id)
    return settings, reloaded, BENCHMARK_ID


__all__ = ["BENCHMARK_ID", "MODEL_FAMILY", "build_risk_fixture"]
