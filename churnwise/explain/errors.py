"""Stable Phase 8 explanation failures with actionable remediation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODE_INVALID_CONFIG = "CW-XAI-001"
CODE_UNSUPPORTED_MODEL = "CW-XAI-002"
CODE_EXPLAIN_FAILED = "CW-XAI-003"
CODE_VALIDATION_FAILED = "CW-XAI-004"
CODE_ARTIFACT_FAILED = "CW-XAI-005"
CODE_NO_MODEL = "CW-XAI-006"


@dataclass
class ExplainError(Exception):
    """A structured explanation failure.

    ``code`` is stable so callers and tests can branch on it; ``remediation``
    tells a user what to do next instead of only what went wrong.
    """

    code: str
    message: str
    remediation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
            "details": dict(self.details),
        }


__all__ = [
    "CODE_ARTIFACT_FAILED",
    "CODE_EXPLAIN_FAILED",
    "CODE_INVALID_CONFIG",
    "CODE_NO_MODEL",
    "CODE_UNSUPPORTED_MODEL",
    "CODE_VALIDATION_FAILED",
    "ExplainError",
]
