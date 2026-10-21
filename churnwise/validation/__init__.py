"""Dataset validation and data intelligence (Phase 4).

Public entry points:

* :class:`ValidationCoordinator` — verify, load, profile, gate and persist.
* :func:`load_dataset` — load a Phase 3 artifact.
* :func:`detect_target` — autonomous target selection and interpretation.
* :func:`run_checks` / :func:`decide_gate` — checks and severity rollup.
* :class:`ValidationReport`, :class:`ValidationResult` — Phase 5 hand-off.
"""

from churnwise.validation.checks import run_checks
from churnwise.validation.contracts import (
    CheckSeverity,
    GateDecision,
    ValidationAttempt,
    ValidationCheck,
    ValidationFailure,
    ValidationIssue,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)
from churnwise.validation.coordinator import ValidationCoordinator
from churnwise.validation.gate import decide_gate
from churnwise.validation.loader import DatasetLoadError, load_dataset
from churnwise.validation.targets import detect_target, normalize_target

__all__ = [
    "CheckSeverity",
    "DatasetLoadError",
    "GateDecision",
    "ValidationAttempt",
    "ValidationCheck",
    "ValidationCoordinator",
    "ValidationFailure",
    "ValidationIssue",
    "ValidationReport",
    "ValidationResult",
    "ValidationStatus",
    "decide_gate",
    "detect_target",
    "load_dataset",
    "normalize_target",
    "run_checks",
]
