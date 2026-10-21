"""Actionable failures for Phase 6 model construction, training and inference."""

from __future__ import annotations

from typing import Any

CODE_INVALID_MODEL = "CW-MOD-001"
CODE_TRAINING_FAILED = "CW-MOD-002"
CODE_MODEL_ARTIFACT = "CW-MOD-003"
CODE_PREDICTION_FAILED = "CW-MOD-004"


class ModelingError(Exception):
    """A modeling failure with a stable code and a next action."""

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
