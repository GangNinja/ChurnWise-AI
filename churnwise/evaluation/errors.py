"""Actionable failures for the Phase 7 evaluation and benchmarking stage."""

from __future__ import annotations

from typing import Any

CODE_INVALID_CONFIG = "CW-EVL-001"
CODE_INCOMPATIBLE_INPUTS = "CW-EVL-002"
CODE_METRIC_FAILED = "CW-EVL-003"
CODE_ARTIFACT_FAILED = "CW-EVL-004"
CODE_FAMILY_FAILED = "CW-EVL-005"
CODE_NO_MODEL = "CW-EVL-006"


class EvaluationError(Exception):
    """An evaluation failure with a stable code and a next action."""

    def __init__(self, code: str, message: str, remediation: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
        }


__all__ = [
    "CODE_ARTIFACT_FAILED",
    "CODE_FAMILY_FAILED",
    "CODE_INCOMPATIBLE_INPUTS",
    "CODE_INVALID_CONFIG",
    "CODE_METRIC_FAILED",
    "CODE_NO_MODEL",
    "EvaluationError",
]
