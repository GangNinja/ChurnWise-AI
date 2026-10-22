"""Stable Phase 10 campaign-simulation failures with actionable remediation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODE_INVALID_CONFIG = "CW-SIM-001"
CODE_INVALID_INPUT = "CW-SIM-002"
CODE_NO_RISK_RUN = "CW-SIM-003"
CODE_ARTIFACT_FAILED = "CW-SIM-004"
CODE_EXPORT_FAILED = "CW-SIM-005"


@dataclass
class SimulationError(Exception):
    """A structured simulation failure.

    Details intentionally contain counts, codes and references only; raw
    customer values must never be placed in errors or logs.
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
    "CODE_EXPORT_FAILED",
    "CODE_INVALID_CONFIG",
    "CODE_INVALID_INPUT",
    "CODE_NO_RISK_RUN",
    "SimulationError",
]
