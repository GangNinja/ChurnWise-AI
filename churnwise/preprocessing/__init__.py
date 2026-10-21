"""Leakage-safe preprocessing and feature engineering (Phase 5).

Public entry points:

* :class:`PreprocessingCoordinator` - validate the Phase 4 hand-off, split,
  fit, transform and persist one reproducible run.
* :func:`prepare_dataset` - one-call convenience wrapper.
* :class:`PreparedDataset` - the Phase 6 hand-off with train/val/test arrays,
  feature mappings, class weights and ``transform_new`` for inference.
* :class:`PreprocessConfig` - every setting that affects the output.
* :class:`PreprocessingError` - stable error codes and remediation.
"""

from __future__ import annotations

from typing import Any

from churnwise.config import Settings
from churnwise.preprocessing.contracts import (
    ClassBalance,
    PIPELINE_VERSION,
    PreprocessConfig,
    SplitData,
    SplitSpec,
)
from churnwise.preprocessing.coordinator import PreprocessingCoordinator
from churnwise.preprocessing.dataset import PreparedDataset
from churnwise.preprocessing.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_NOT_READY,
    CODE_SPLIT_IMPOSSIBLE,
    CODE_TRANSFORM_FAILED,
    PreprocessingError,
)

__all__ = [
    "CODE_ARTIFACT_FAILED",
    "CODE_NOT_READY",
    "CODE_SPLIT_IMPOSSIBLE",
    "CODE_TRANSFORM_FAILED",
    "ClassBalance",
    "PIPELINE_VERSION",
    "PreprocessConfig",
    "PreparedDataset",
    "PreprocessingCoordinator",
    "PreprocessingError",
    "SplitData",
    "SplitSpec",
    "prepare_dataset",
]


def prepare_dataset(
    validation: Any,
    *,
    settings: Settings,
    config: PreprocessConfig | None = None,
    persist: bool = True,
) -> PreparedDataset:
    """Convenience wrapper around :class:`PreprocessingCoordinator`."""
    return PreprocessingCoordinator(settings).prepare(
        validation, config=config, persist=persist
    )
