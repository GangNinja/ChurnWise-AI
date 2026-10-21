"""Stable Phase 4 contracts consumed by the CLI and Phase 5.

A validation report is self-identifying: it records the Phase 3 manifest hash,
the consumed artifact digest, the validator version and every check with its
evidence. The report never embeds the dataframe; Phase 5 receives the loaded
frame through :class:`ValidationResult` and can reload it from the artifact
paths recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


class ValidationStatus(str, Enum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    REJECT = "reject"


class CheckSeverity(str, Enum):
    FATAL = "fatal"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: CheckSeverity = CheckSeverity.WARNING
    columns: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "columns": list(self.columns),
            "evidence": self.evidence,
        }


@dataclass
class ValidationCheck:
    id: str
    name: str
    severity: CheckSeverity
    passed: bool
    detail: str
    issues: list[ValidationIssue] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "passed": self.passed,
            "detail": self.detail,
            "issues": [issue.to_dict() for issue in self.issues],
            "evidence": self.evidence,
        }


@dataclass
class GateDecision:
    status: ValidationStatus
    passed: bool
    can_preprocess: bool
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "can_preprocess": self.can_preprocess,
            "blocking": list(self.blocking),
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
        }


@dataclass
class ValidationAttempt:
    candidate_id: str
    stage: str
    status: str
    duration_ms: int = 0
    error_code: str | None = None
    message: str | None = None
    report_path: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
            "message": self.message,
            "report_path": self.report_path,
            "detail": self.detail,
        }


@dataclass
class ValidationFailure:
    code: str
    message: str
    remediation: str
    causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
            "causes": list(self.causes),
        }


@dataclass
class ValidationReport:
    dataset_id: str
    manifest_sha256: str
    artifact_sha256: str
    artifact_path: str
    generated_at: str
    validator_version: str
    status: ValidationStatus
    checks: list[ValidationCheck]
    summary: dict[str, Any]
    target: dict[str, Any]
    roles: dict[str, Any]
    capabilities: dict[str, bool]
    gate: GateDecision
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    eda_profile_path: str | None = None
    eda_plots: list[str] = field(default_factory=list)
    report_json_path: str | None = None
    report_md_path: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "manifest_sha256": self.manifest_sha256,
            "artifact_sha256": self.artifact_sha256,
            "artifact_path": self.artifact_path,
            "generated_at": self.generated_at,
            "validator_version": self.validator_version,
            "status": self.status.value,
            "checks": [check.to_dict() for check in self.checks],
            "summary": self.summary,
            "target": self.target,
            "roles": self.roles,
            "capabilities": dict(self.capabilities),
            "gate": self.gate.to_dict(),
            "provenance": self.provenance,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "eda_profile_path": self.eda_profile_path,
            "eda_plots": list(self.eda_plots),
            "report_json_path": self.report_json_path,
            "report_md_path": self.report_md_path,
        }


@dataclass
class ValidationResult:
    status: ValidationStatus
    report: ValidationReport | None = None
    profile: Any | None = None
    frame: Any | None = None
    acquisition: Any | None = None
    attempts: list[ValidationAttempt] = field(default_factory=list)
    failure: ValidationFailure | None = None
    report_paths: dict[str, str] = field(default_factory=dict)

    @property
    def can_preprocess(self) -> bool:
        return bool(self.report and self.report.gate.can_preprocess and self.profile is not None)

    @property
    def target(self) -> Any | None:
        return self.profile.target if self.profile is not None else None

    @property
    def features(self) -> list[str]:
        return list(self.profile.roles.features) if self.profile is not None else []

    def normalized_frame(self) -> Any | None:
        """Return a copy with the target mapped to nullable integers 0/1.

        Missing target rows are excluded because they cannot be used for
        supervised training. The original acquired file is never changed.
        """
        if self.frame is None or self.profile is None or not self.profile.target.usable:
            return None
        from churnwise.validation.targets import normalize_target

        frame = self.frame.copy()
        column = self.profile.target.column
        assert column is not None
        frame[column] = normalize_target(frame[column], self.profile.target)
        return frame.loc[frame[column].notna()].reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "report": self.report.to_dict() if self.report else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "failure": self.failure.to_dict() if self.failure else None,
            "can_preprocess": self.can_preprocess,
            "report_paths": dict(self.report_paths),
        }
