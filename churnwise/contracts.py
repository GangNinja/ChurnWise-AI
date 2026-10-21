"""Versioned data contracts for the Phase 2 discovery subsystem.

Nothing here performs network or filesystem IO. These structures are the
stable boundary between source adapters, the discovery coordinator, the CLI and
the Phase 3 acquisition layer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = "1.0"

#: Formats the discovery engine is willing to recommend. ARFF/ZIP are accepted
#: because OpenML publishes ARFF directly and some catalogs archive CSV.
SUPPORTED_FORMATS: tuple[str, ...] = ("csv", "tsv", "arff", "parquet", "xlsx", "zip", "json")

SOURCE_PRIORITY: dict[str, int] = {"openml": 1, "uci": 2, "github-ibm": 3}

CHURN_TARGET_HINTS: tuple[str, ...] = (
    "churn",
    "churned",
    "attrition",
    "cancel",
    "cancelled",
    "canceled",
    "exited",
    "retained",
    "subscription_status",
)

PII_PATTERN = re.compile(
    r"(^|_)(full_?name|first_?name|last_?name|name|email|e_?mail|phone|mobile|"
    r"address|street|ssn|social_security|iban|credit_card|card_number|passport)(_|$)",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    """Current UTC timestamp, second precision, ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_https_url(value: str) -> bool:
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and bool(parts.hostname)
        and not parts.username
        and not parts.password
    )


def _parse_iso(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


class SourceStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class LicenseStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str
    severity: Severity = Severity.ERROR

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class FieldInfo:
    """One column as reported by the source (never inferred from data)."""

    name: str
    data_type: str | None = None
    role: str | None = None  # target | id | feature
    missing_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "role": self.role,
            "missing_count": self.missing_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldInfo":
        return cls(
            name=str(data.get("name", "")),
            data_type=data.get("data_type"),
            role=data.get("role"),
            missing_count=data.get("missing_count"),
        )


@dataclass
class DatasetCandidate:
    """Normalized metadata for one dataset listing.

    Unknown values are ``None`` — never guessed. ``evidence`` keeps a compact
    copy of the source payload fields the candidate was built from so a human
    can verify the claim.
    """

    candidate_id: str
    source_id: str
    source_name: str
    source_dataset_id: str
    title: str
    landing_url: str
    retrieved_at: str
    format: str = "unknown"
    description: str | None = None
    resource_url: str | None = None
    publisher: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    license_status: LicenseStatus = LicenseStatus.UNKNOWN
    published_at: str | None = None
    updated_at: str | None = None
    size_bytes: int | None = None
    row_count: int | None = None
    feature_count: int | None = None
    target_column: str | None = None
    target_class_count: int | None = None
    class_balance: dict[str, int] | None = None
    schema_fields: list[FieldInfo] | None = None
    has_numeric: bool | None = None
    has_categorical: bool | None = None
    access: str = "open"
    limitations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.license_status, LicenseStatus):
            self.license_status = LicenseStatus(str(self.license_status))

    # -- validation ---------------------------------------------------------
    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not self.candidate_id or not re.fullmatch(r"[a-z0-9][a-z0-9._:-]{2,120}", self.candidate_id):
            issues.append(
                ValidationIssue("C001", "candidate_id", "candidate_id must be a stable lowercase identifier")
            )
        if not self.title.strip():
            issues.append(ValidationIssue("C002", "title", "title must not be empty"))
        if not self.source_id.strip():
            issues.append(ValidationIssue("C003", "source_id", "source_id must not be empty"))
        if not _is_https_url(self.landing_url):
            issues.append(
                ValidationIssue("C004", "landing_url", "landing_url must be an absolute https URL")
            )
        if self.resource_url is not None and not _is_https_url(self.resource_url):
            issues.append(
                ValidationIssue("C005", "resource_url", "resource_url must be an absolute https URL")
            )
        if not _parse_iso(self.retrieved_at):
            issues.append(
                ValidationIssue("C006", "retrieved_at", "retrieved_at must be an ISO-8601 timestamp")
            )
        if self.row_count is not None and self.row_count <= 0:
            issues.append(ValidationIssue("C007", "row_count", "row_count must be positive when present"))
        if self.feature_count is not None and self.feature_count <= 0:
            issues.append(
                ValidationIssue("C008", "feature_count", "feature_count must be positive when present")
            )
        if self.size_bytes is not None and self.size_bytes <= 0:
            issues.append(ValidationIssue("C009", "size_bytes", "size_bytes must be positive when present"))
        if self.target_class_count is not None and self.target_class_count < 2:
            issues.append(
                ValidationIssue("C010", "target_class_count", "target_class_count must be at least 2")
            )
        if self.class_balance is not None:
            if set(self.class_balance) != {"majority", "minority"}:
                issues.append(
                    ValidationIssue(
                        "C011", "class_balance", "class_balance must contain majority and minority counts"
                    )
                )
            elif any(int(v) <= 0 for v in self.class_balance.values()):
                issues.append(
                    ValidationIssue("C012", "class_balance", "class_balance counts must be positive")
                )
            elif self.row_count is not None and sum(self.class_balance.values()) != self.row_count:
                issues.append(
                    ValidationIssue(
                        "C013",
                        "class_balance",
                        "class_balance does not sum to row_count; source metadata is inconsistent",
                        Severity.WARNING,
                    )
                )
        if self.format not in SUPPORTED_FORMATS:
            issues.append(
                ValidationIssue(
                    "C014", "format", "format is not a supported tabular format", Severity.WARNING
                )
            )
        if self.license_status is LicenseStatus.VERIFIED and not (self.license_name or "").strip():
            issues.append(
                ValidationIssue(
                    "C015",
                    "license_name",
                    "license_status is verified but no license name was captured",
                    Severity.WARNING,
                )
            )
        for field_info in self.schema_fields or []:
            if not field_info.name.strip():
                issues.append(
                    ValidationIssue("C016", "schema_fields", "schema field names must not be empty")
                )
                break
        return issues

    @property
    def is_valid(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.validate())

    def pii_columns(self) -> list[str]:
        return [
            f.name
            for f in (self.schema_fields or [])
            if f.role != "target" and PII_PATTERN.search(f.name)
        ]

    def target_name_is_churn_like(self) -> bool:
        if not self.target_column:
            return False
        name = self.target_column.lower()
        return any(hint in name for hint in CHURN_TARGET_HINTS)

    def fingerprint(self) -> str | None:
        """Content-ish fingerprint from metadata only; None when too little is known."""
        if self.row_count is None or self.feature_count is None:
            return None
        payload = {
            "rows": self.row_count,
            "features": self.feature_count,
            "target": (self.target_column or "").lower(),
            "balance": self.class_balance,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_dataset_id": self.source_dataset_id,
            "title": self.title,
            "description": self.description,
            "landing_url": self.landing_url,
            "resource_url": self.resource_url,
            "format": self.format,
            "publisher": self.publisher,
            "license": {
                "name": self.license_name,
                "url": self.license_url,
                "status": self.license_status.value,
            },
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "retrieved_at": self.retrieved_at,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "target_column": self.target_column,
            "target_class_count": self.target_class_count,
            "class_balance": self.class_balance,
            "schema_fields": [f.to_dict() for f in self.schema_fields] if self.schema_fields else None,
            "has_numeric": self.has_numeric,
            "has_categorical": self.has_categorical,
            "access": self.access,
            "limitations": list(self.limitations),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetCandidate":
        license_block = data.get("license") or {}
        schema = data.get("schema_fields")
        return cls(
            candidate_id=str(data["candidate_id"]),
            source_id=str(data["source_id"]),
            source_name=str(data.get("source_name", data["source_id"])),
            source_dataset_id=str(data.get("source_dataset_id", "")),
            title=str(data.get("title", "")),
            description=data.get("description"),
            landing_url=str(data.get("landing_url", "")),
            resource_url=data.get("resource_url"),
            format=str(data.get("format", "unknown")),
            publisher=data.get("publisher"),
            license_name=license_block.get("name"),
            license_url=license_block.get("url"),
            license_status=LicenseStatus(license_block.get("status", "unknown")),
            published_at=data.get("published_at"),
            updated_at=data.get("updated_at"),
            retrieved_at=str(data.get("retrieved_at", "")),
            size_bytes=data.get("size_bytes"),
            row_count=data.get("row_count"),
            feature_count=data.get("feature_count"),
            target_column=data.get("target_column"),
            target_class_count=data.get("target_class_count"),
            class_balance=data.get("class_balance"),
            schema_fields=[FieldInfo.from_dict(f) for f in schema] if schema else None,
            has_numeric=data.get("has_numeric"),
            has_categorical=data.get("has_categorical"),
            access=str(data.get("access", "open")),
            limitations=list(data.get("limitations") or []),
            evidence=dict(data.get("evidence") or {}),
        )


@dataclass(frozen=True)
class SearchQuery:
    terms: tuple[str, ...] = ("churn", "customer churn", "attrition")
    formats: tuple[str, ...] = ("csv", "tsv", "arff", "parquet", "xlsx", "zip")
    min_rows: int = 1000
    max_rows: int | None = None
    min_predictors: int = 5
    require_numeric_features: bool = False
    require_categorical_features: bool = False
    require_customer_features: bool = False
    max_results_per_source: int = 10
    enrich_limit: int = 12
    timeout_s: float = 12.0
    max_retries: int = 2
    max_requests_per_source: int = 30
    allow_unverified_license: bool = False

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not (1 <= len(self.terms) <= 6):
            problems.append("terms must contain between 1 and 6 entries")
        seen: set[str] = set()
        for term in self.terms:
            cleaned = term.strip()
            if not (2 <= len(cleaned) <= 64):
                problems.append(f"term {term!r} must be 2-64 characters")
            if any(ord(ch) < 32 for ch in cleaned):
                problems.append(f"term {term!r} contains control characters")
            if cleaned.lower() in seen:
                problems.append(f"duplicate term {term!r}")
            seen.add(cleaned.lower())
        if not self.formats:
            problems.append("formats must not be empty")
        unknown = [f for f in self.formats if f not in SUPPORTED_FORMATS]
        if unknown:
            problems.append(f"unsupported formats requested: {unknown}")
        if self.min_rows < 100:
            problems.append("min_rows must be at least 100")
        if self.max_rows is not None and self.max_rows <= self.min_rows:
            problems.append("max_rows must be greater than min_rows")
        if not (1 <= self.min_predictors <= 100):
            problems.append("min_predictors must be between 1 and 100")
        if not (1 <= self.max_results_per_source <= 100):
            problems.append("max_results_per_source must be between 1 and 100")
        if not (0 <= self.enrich_limit <= 100):
            problems.append("enrich_limit must be between 0 and 100")
        if not (1.0 <= self.timeout_s <= 60.0):
            problems.append("timeout_s must be between 1 and 60")
        if not (0 <= self.max_retries <= 5):
            problems.append("max_retries must be between 0 and 5")
        if not (3 <= self.max_requests_per_source <= 200):
            problems.append("max_requests_per_source must be between 3 and 200")
        return problems

    def fingerprint(self) -> str:
        payload = {
            "terms": list(self.terms),
            "formats": list(self.formats),
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_predictors": self.min_predictors,
            "require_numeric_features": self.require_numeric_features,
            "require_categorical_features": self.require_categorical_features,
            "require_customer_features": self.require_customer_features,
            "max_results_per_source": self.max_results_per_source,
            "enrich_limit": self.enrich_limit,
            "allow_unverified_license": self.allow_unverified_license,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": list(self.terms),
            "formats": list(self.formats),
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_predictors": self.min_predictors,
            "require_numeric_features": self.require_numeric_features,
            "require_categorical_features": self.require_categorical_features,
            "require_customer_features": self.require_customer_features,
            "max_results_per_source": self.max_results_per_source,
            "enrich_limit": self.enrich_limit,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "allow_unverified_license": self.allow_unverified_license,
            "fingerprint": self.fingerprint(),
        }


@dataclass
class SourceResult:
    source_id: str
    status: SourceStatus
    candidates: list[DatasetCandidate] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    remediation: str | None = None
    http_status: int | None = None
    request_count: int = 0
    duration_ms: int = 0
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "status": self.status.value,
            "candidate_count": len(self.candidates),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "remediation": self.remediation,
            "http_status": self.http_status,
            "request_count": self.request_count,
            "duration_ms": self.duration_ms,
            "notes": list(self.notes),
        }


@dataclass
class GateResult:
    gate_id: str
    name: str
    status: GateStatus
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class ScoreComponent:
    name: str
    value: float
    weight: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "weight": self.weight,
            "weighted": round(self.value * self.weight, 6),
            "detail": self.detail,
        }


@dataclass
class CandidateAssessment:
    candidate: DatasetCandidate
    gates: list[GateResult] = field(default_factory=list)
    components: list[ScoreComponent] = field(default_factory=list)
    score: float = 0.0
    eligible: bool = False
    rank: int | None = None
    duplicate_of: str | None = None
    duplicate_reasons: list[str] = field(default_factory=list)

    def gate_map(self) -> dict[str, GateResult]:
        return {g.gate_id: g for g in self.gates}

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "score": round(self.score, 6),
            "score_components": [c.to_dict() for c in self.components],
            "gates": {g.gate_id: g.to_dict() for g in self.gates},
            "eligible": self.eligible,
            "rank": self.rank,
            "duplicate_of": self.duplicate_of,
            "duplicate_reasons": list(self.duplicate_reasons),
        }


@dataclass(frozen=True)
class DuplicateGroup:
    representative_id: str
    member_ids: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "representative_id": self.representative_id,
            "member_ids": list(self.member_ids),
            "reasons": list(self.reasons),
        }


@dataclass
class SelectedCandidate:
    """Stable hand-off contract for Phase 3 acquisition."""

    candidate_id: str
    source_id: str
    source_dataset_id: str
    title: str
    landing_url: str
    resource_url: str | None
    format: str
    license_name: str | None
    license_url: str | None
    license_status: str
    target_column: str | None
    row_count: int | None
    feature_count: int | None
    retrieved_at: str
    limitations: list[str]
    snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "source_dataset_id": self.source_dataset_id,
            "title": self.title,
            "landing_url": self.landing_url,
            "resource_url": self.resource_url,
            "format": self.format,
            "license": {
                "name": self.license_name,
                "url": self.license_url,
                "status": self.license_status,
            },
            "target_column": self.target_column,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "retrieved_at": self.retrieved_at,
            "limitations": list(self.limitations),
            "snapshot": self.snapshot,
        }

    @classmethod
    def from_candidate(cls, candidate: DatasetCandidate) -> "SelectedCandidate":
        return cls(
            candidate_id=candidate.candidate_id,
            source_id=candidate.source_id,
            source_dataset_id=candidate.source_dataset_id,
            title=candidate.title,
            landing_url=candidate.landing_url,
            resource_url=candidate.resource_url,
            format=candidate.format,
            license_name=candidate.license_name,
            license_url=candidate.license_url,
            license_status=candidate.license_status.value,
            target_column=candidate.target_column,
            row_count=candidate.row_count,
            feature_count=candidate.feature_count,
            retrieved_at=candidate.retrieved_at,
            limitations=list(candidate.limitations),
            snapshot=candidate.to_dict(),
        )


@dataclass
class DiscoveryFailure:
    code: str
    message: str
    sources_checked: list[str]
    reasons: list[str]
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "sources_checked": list(self.sources_checked),
            "reasons": list(self.reasons),
            "next_action": self.next_action,
        }


@dataclass
class DiscoveryReport:
    generated_at: str
    query: SearchQuery
    source_results: list[SourceResult]
    assessments: list[CandidateAssessment]
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    selected: SelectedCandidate | None = None
    failure: DiscoveryFailure | None = None
    invalid_candidates: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def ranked(self) -> list[CandidateAssessment]:
        unique = [a for a in self.assessments if a.duplicate_of is None]
        return sorted(unique, key=lambda a: (a.rank if a.rank is not None else 10**6, a.candidate.candidate_id))

    def duplicates(self) -> list[CandidateAssessment]:
        return [a for a in self.assessments if a.duplicate_of is not None]

    def rejections(self) -> list[dict[str, Any]]:
        rejected: list[dict[str, Any]] = []
        for assessment in self.assessments:
            if assessment.eligible and assessment.duplicate_of is None:
                continue
            if assessment.duplicate_of is not None:
                rejected.append(
                    {
                        "candidate_id": assessment.candidate.candidate_id,
                        "stage": "duplicate",
                        "reason": f"duplicate of {assessment.duplicate_of}",
                        "duplicate_reasons": assessment.duplicate_reasons,
                    }
                )
                continue
            failing = [
                {
                    "gate_id": g.gate_id,
                    "status": g.status.value,
                    "reason": g.reason,
                }
                for g in assessment.gates
                if g.status is not GateStatus.PASS
            ]
            rejected.append(
                {
                    "candidate_id": assessment.candidate.candidate_id,
                    "stage": "gates",
                    "reason": "one or more discovery gates did not pass",
                    "failing_gates": failing,
                }
            )
        rejected.extend(self.invalid_candidates)
        return rejected

    def to_dict(self) -> dict[str, Any]:
        ranked = self.ranked()
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": 2,
            "generated_at": self.generated_at,
            "query": self.query.to_dict(),
            "config": self.config,
            "sources": [s.summary_dict() for s in self.source_results],
            "summary": {
                "found": len(self.assessments) + len(self.invalid_candidates),
                "valid": len(self.assessments),
                "duplicates": len(self.duplicates()),
                "unique": len(ranked),
                "eligible": sum(1 for a in ranked if a.eligible),
                "rejected": len(self.rejections()),
                "selected": self.selected.candidate_id if self.selected else None,
            },
            "ranking": [a.to_dict() for a in ranked],
            "duplicates": [a.to_dict() for a in self.duplicates()],
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
            "invalid_candidates": list(self.invalid_candidates),
            "selected": self.selected.to_dict() if self.selected else None,
            "failure": self.failure.to_dict() if self.failure else None,
        }

    def rejections_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": 2,
            "generated_at": self.generated_at,
            "query_fingerprint": self.query.fingerprint(),
            "summary": {
                "rejected": len(self.rejections()),
                "selected": self.selected.candidate_id if self.selected else None,
            },
            "rejections": self.rejections(),
            "failure": self.failure.to_dict() if self.failure else None,
        }
