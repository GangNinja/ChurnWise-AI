"""Actionable failures for the Phase 5 preprocessing stage."""

from __future__ import annotations

from typing import Any

CODE_NOT_READY = "CW-PRE-001"
CODE_SPLIT_IMPOSSIBLE = "CW-PRE-002"
CODE_ARTIFACT_FAILED = "CW-PRE-003"
CODE_TRANSFORM_FAILED = "CW-PRE-004"


class PreprocessingError(Exception):
    """A preprocessing failure with a stable code and a next action."""

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
