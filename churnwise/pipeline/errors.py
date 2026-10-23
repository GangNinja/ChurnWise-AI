"""Structured failures for the Phase 12 end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODE_INVALID_CONFIG = "CW-RUN-001"
CODE_UPSTREAM_FAILURE = "CW-RUN-002"
CODE_STAGE_FAILED = "CW-RUN-003"
CODE_INCOMPATIBLE_RESUME = "CW-RUN-004"
CODE_VERIFICATION_FAILED = "CW-RUN-005"

PIPELINE_CODES = {
    CODE_INVALID_CONFIG: "invalid pipeline configuration",
    CODE_UPSTREAM_FAILURE: "an upstream phase failed or rejected its input",
    CODE_STAGE_FAILED: "a pipeline stage raised an unexpected error",
    CODE_INCOMPATIBLE_RESUME: "the saved run cannot be resumed with this configuration",
    CODE_VERIFICATION_FAILED: "the completed run is missing a required artifact",
}


@dataclass
class PipelineError(Exception):
    """One actionable, JSON-serialisable pipeline failure.

    ``details`` carries upstream diagnosis (for example a validation gate) so a
    caller never has to parse a prose message to find the root cause.
    """

    code: str
    message: str
    remediation: str = ""
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
            "stage": self.stage,
            "details": self.details,
            "artifacts": self.artifacts,
        }


__all__ = [
    "CODE_INCOMPATIBLE_RESUME",
    "CODE_INVALID_CONFIG",
    "CODE_STAGE_FAILED",
    "CODE_UPSTREAM_FAILURE",
    "CODE_VERIFICATION_FAILED",
    "PIPELINE_CODES",
    "PipelineError",
]
