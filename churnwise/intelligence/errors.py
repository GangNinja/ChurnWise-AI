"""Stable Phase 9 risk-intelligence failures with actionable remediation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODE_INVALID_CONFIG = "CW-RISK-001"
CODE_NO_MODEL = "CW-RISK-002"
CODE_ARTIFACT_FAILED = "CW-RISK-003"
CODE_INVALID_INPUT = "CW-RISK-004"
CODE_PREDICTION_FAILED = "CW-RISK-005"
CODE_EXPLANATION_FAILED = "CW-RISK-006"
CODE_EXPORT_FAILED = "CW-RISK-007"


@dataclass
class RiskError(Exception):
    """A structured risk-intelligence failure.

    ``details`` is deliberately limited to counts, codes and hashed references.
    Raw customer values and identifiers must not be placed in error payloads or
    logs.
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
    "CODE_EXPLANATION_FAILED",
    "CODE_EXPORT_FAILED",
    "CODE_INVALID_CONFIG",
    "CODE_INVALID_INPUT",
    "CODE_NO_MODEL",
    "CODE_PREDICTION_FAILED",
    "RiskError",
]
