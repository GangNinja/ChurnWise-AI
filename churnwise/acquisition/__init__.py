"""Autonomous dataset acquisition and provenance (Phase 3).

Public entry points:

* :class:`AcquisitionCoordinator` — acquire a candidate or ranked list with
  cache reuse and fallback.
* :func:`load_candidates` — read the Phase 2 ranked report.
* :class:`DatasetManifest` — the provenance record consumed by Phase 4.
"""

from churnwise.acquisition.contracts import (
    AcquisitionArtifact,
    AcquisitionAttempt,
    AcquisitionFailure,
    AcquisitionResult,
    DatasetManifest,
)
from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.acquisition.inputs import load_candidates
from churnwise.acquisition.store import DatasetStore

__all__ = [
    "AcquisitionArtifact",
    "AcquisitionAttempt",
    "AcquisitionCoordinator",
    "AcquisitionFailure",
    "AcquisitionResult",
    "DatasetManifest",
    "DatasetStore",
    "load_candidates",
]
