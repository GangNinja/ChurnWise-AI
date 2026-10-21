"""Stable Phase 3 contracts consumed by the CLI and Phase 4.

The manifest is the provenance record. The acquisition result is the Phase 4
entry point: it distinguishes the discovery candidate, the raw artifact, the
extracted file, integrity status and final success/failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = "1.0"

#: Formats acquisition can store. ZIP is an archive, xlsx is a ZIP container
#: that is stored opaquely (Phase 4 installs its parser).
ACQUISITION_FORMATS: tuple[str, ...] = (
    "csv",
    "tsv",
    "arff",
    "parquet",
    "json",
    "xlsx",
    "zip",
)
ARCHIVE_FORMATS: tuple[str, ...] = ("zip",)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", "com0", "lpt0"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_https_url(value: str) -> bool:
    try:
        parts = urlsplit(value)
    except (TypeError, ValueError):
        return False
    return (
        parts.scheme == "https"
        and bool(parts.hostname)
        and not parts.username
        and not parts.password
    )


def _is_iso(value: str) -> bool:
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def _safe_relative(value: str | None) -> bool:
    """True for a canonical, relative, traversal-free POSIX path.

    Archive members may legitimately contain spaces or non-ASCII characters
    (for example ``churn data.csv``), so those are accepted. Absolute paths,
    drive letters, backslashes, control characters and Windows-reserved names
    are not: they would escape the managed tree or alias another path.
    """
    if value is None:
        return True
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    if re.match(r"^[A-Za-z]:", value):
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    path = PurePosixPath(value)
    if str(path) != value:
        return False
    for part in path.parts:
        if part in ("", ".", "..") or ":" in part:
            return False
        # Windows strips trailing dots/spaces and resolves reserved device
        # names, which could alias a different file than the recorded path.
        if part != part.rstrip(" .") or part.split(".", 1)[0].strip().lower() in _WINDOWS_RESERVED:
            return False
    return True


class AcquisitionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    CACHED = "cached"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PublisherChecksum:
    algorithm: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass
class AcquisitionFailure:
    code: str
    message: str
    category: str
    retryable: bool = False
    remediation: str | None = None
    causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "retryable": self.retryable,
            "remediation": self.remediation,
            "causes": list(self.causes),
        }


@dataclass
class AcquisitionAttempt:
    candidate_id: str
    rank: int
    status: str
    duration_ms: int
    source_id: str | None = None
    format: str | None = None
    url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_category: str | None = None
    retryable: bool = False
    remediation: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "source_id": self.source_id,
            "status": self.status,
            "format": self.format,
            "url": self.url,
            "duration_ms": self.duration_ms,
            "cached": self.cached,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_category": self.error_category,
            "retryable": self.retryable,
            "remediation": self.remediation,
        }


@dataclass
class AcquisitionArtifact:
    dataset_id: str
    filename: str
    format: str
    bytes: int
    sha256: str
    content_type: str | None
    cache_path: str
    raw_path: str
    extracted_path: str | None = None
    extracted_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "format": self.format,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "cache_path": self.cache_path,
            "raw_path": self.raw_path,
            "extracted_path": self.extracted_path,
            "extracted_files": list(self.extracted_files),
        }


@dataclass
class ManifestSource:
    provider: str
    source_id: str
    dataset_id: str
    landing_url: str
    download_url: str
    license_name: str | None
    license_url: str | None
    license_status: str
    publisher: str | None = None
    reported_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "landing_url": self.landing_url,
            "download_url": self.download_url,
            "license": {
                "name": self.license_name,
                "url": self.license_url,
                "status": self.license_status,
            },
            "publisher": self.publisher,
            "reported_version": self.reported_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestSource":
        license_block = data.get("license") or {}
        return cls(
            provider=str(data.get("provider", "")),
            source_id=str(data.get("source_id", "")),
            dataset_id=str(data.get("dataset_id", "")),
            landing_url=str(data.get("landing_url", "")),
            download_url=str(data.get("download_url", "")),
            license_name=license_block.get("name"),
            license_url=license_block.get("url"),
            license_status=str(license_block.get("status", "unknown")),
            publisher=data.get("publisher"),
            reported_version=data.get("reported_version"),
        )


@dataclass
class ManifestFile:
    filename: str
    bytes: int
    sha256: str
    content_type: str | None
    format: str
    cache_path: str
    raw_path: str
    extracted_path: str | None = None
    extracted_sha256: str | None = None
    extracted_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "format": self.format,
            "cache_path": self.cache_path,
            "raw_path": self.raw_path,
            "extracted_path": self.extracted_path,
            "extracted_sha256": self.extracted_sha256,
            "extracted_files": list(self.extracted_files),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestFile":
        return cls(
            filename=str(data.get("filename", "")),
            bytes=int(data.get("bytes", 0)),
            sha256=str(data.get("sha256", "")),
            content_type=data.get("content_type"),
            format=str(data.get("format", "")),
            cache_path=str(data.get("cache_path", "")),
            raw_path=str(data.get("raw_path", "")),
            extracted_path=data.get("extracted_path"),
            extracted_sha256=data.get("extracted_sha256"),
            extracted_files=list(data.get("extracted_files") or []),
        )


@dataclass
class ManifestIntegrity:
    verified: bool
    method: str
    expected_size: int | None = None
    publisher_checksum: PublisherChecksum | None = None
    checksum_status: str = "unavailable"
    archive_members: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "method": self.method,
            "expected_size": self.expected_size,
            "publisher_checksum": (
                self.publisher_checksum.to_dict() if self.publisher_checksum else None
            ),
            "checksum_status": self.checksum_status,
            "archive_members": list(self.archive_members),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestIntegrity":
        checksum = data.get("publisher_checksum")
        verified = data.get("verified")
        if not isinstance(verified, bool):
            raise ValueError("integrity.verified must be a boolean")
        return cls(
            verified=verified,
            method=str(data.get("method", "")),
            expected_size=data.get("expected_size"),
            publisher_checksum=(
                PublisherChecksum(str(checksum.get("algorithm")), str(checksum.get("value")))
                if checksum
                else None
            ),
            checksum_status=str(data.get("checksum_status", "unavailable")),
            archive_members=list(data.get("archive_members") or []),
        )


@dataclass
class ManifestAcquisition:
    method: str
    attempt: int
    duration_ms: int
    status: str = "success"
    fallback_of: str | None = None
    candidate_rank: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "attempt": self.attempt,
            "status": self.status,
            "fallback_of": self.fallback_of,
            "candidate_rank": self.candidate_rank,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestAcquisition":
        return cls(
            method=str(data.get("method", "")),
            attempt=int(data.get("attempt", 1)),
            duration_ms=int(data.get("duration_ms", 0)),
            status=str(data.get("status", "success")),
            fallback_of=data.get("fallback_of"),
            candidate_rank=data.get("candidate_rank"),
            warnings=list(data.get("warnings") or []),
        )


@dataclass
class DatasetManifest:
    dataset_id: str
    title: str
    source: ManifestSource
    retrieved_at: str
    file: ManifestFile
    integrity: ManifestIntegrity
    acquisition: ManifestAcquisition
    discovery: dict[str, Any]
    run_id: str = ""
    dataset_sha256: str = ""
    seed: int = 0
    code_version: str = ""
    notes: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "seed": self.seed,
            "code_version": self.code_version,
            "title": self.title,
            "source": self.source.to_dict(),
            "retrieved_at": self.retrieved_at,
            "file": self.file.to_dict(),
            "integrity": self.integrity.to_dict(),
            "acquisition": self.acquisition.to_dict(),
            "discovery": self.discovery,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=str(data.get("dataset_id", "")),
            title=str(data.get("title", "")),
            source=ManifestSource.from_dict(data.get("source") or {}),
            retrieved_at=str(data.get("retrieved_at", "")),
            file=ManifestFile.from_dict(data.get("file") or {}),
            integrity=ManifestIntegrity.from_dict(data.get("integrity") or {}),
            acquisition=ManifestAcquisition.from_dict(data.get("acquisition") or {}),
            discovery=dict(data.get("discovery") or {}),
            run_id=str(data.get("run_id", "")),
            dataset_sha256=str(data.get("dataset_sha256", "")),
            seed=int(data.get("seed", 0)),
            code_version=str(data.get("code_version", "")),
            notes=list(data.get("notes") or []),
            schema_version=str(data.get("schema_version", "")),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version must be {SCHEMA_VERSION!r}")
        if not self.dataset_id.strip():
            problems.append("dataset_id must not be empty")
        if not self.run_id.strip():
            problems.append("run_id must not be empty")
        if not _SHA256.match(self.dataset_sha256):
            problems.append("dataset_sha256 must be 64 lowercase hex characters")
        if self.dataset_sha256 and self.dataset_sha256 != self.file.sha256:
            problems.append("dataset_sha256 must match file.sha256")
        if self.seed < 0:
            problems.append("seed must not be negative")
        if not self.code_version.strip():
            problems.append("code_version must not be empty")
        if not self.title.strip():
            problems.append("title must not be empty")
        if not self.source.provider.strip() or not self.source.source_id.strip():
            problems.append("source provider and source_id must not be empty")
        if not self.source.dataset_id.strip():
            problems.append("source.dataset_id must not be empty")
        if self.source.license_status not in {
            "verified",
            "unverified",
            "restricted",
            "unknown",
        }:
            problems.append("source.license.status is not recognised")
        if not _is_https_url(self.source.landing_url):
            problems.append("source.landing_url must be an absolute https URL")
        if not _is_https_url(self.source.download_url):
            problems.append("source.download_url must be an absolute https URL")
        if not _is_iso(self.retrieved_at):
            problems.append("retrieved_at must be an ISO-8601 timestamp")
        if not self.file.filename.strip() or "/" in self.file.filename or "\\" in self.file.filename:
            problems.append("file.filename must be a plain filename")
        if self.file.bytes <= 0:
            problems.append("file.bytes must be positive")
        if not _SHA256.match(self.file.sha256):
            problems.append("file.sha256 must be 64 lowercase hex characters")
        if self.file.format not in ACQUISITION_FORMATS:
            problems.append(f"file.format {self.file.format!r} is not supported for acquisition")
        if not _safe_relative(self.file.cache_path):
            problems.append("file.cache_path must be a safe relative path")
        if not _safe_relative(self.file.raw_path):
            problems.append("file.raw_path must be a safe relative path")
        if self.file.extracted_path is not None and not _safe_relative(self.file.extracted_path):
            problems.append("file.extracted_path must be a safe relative path")
        if self.file.extracted_path is not None and self.file.extracted_sha256 is None:
            problems.append("file.extracted_sha256 is required when extracted_path is set")
        if self.file.extracted_sha256 is not None and not _SHA256.match(
            self.file.extracted_sha256
        ):
            problems.append("file.extracted_sha256 must be 64 lowercase hex characters")
        for relative in self.file.extracted_files:
            if not _safe_relative(relative):
                problems.append(f"extracted file {relative!r} is not a safe relative path")
        if self.integrity.verified is not True:
            problems.append("integrity.verified must be true for a consumable manifest")
        if self.integrity.method != "sha256_recomputed":
            problems.append("integrity.method must be 'sha256_recomputed'")
        if self.integrity.expected_size is not None and self.integrity.expected_size != self.file.bytes:
            problems.append("integrity.expected_size does not match file.bytes")
        if self.integrity.checksum_status not in {
            "unavailable",
            "verified",
            "mismatch",
        }:
            problems.append("integrity.checksum_status is not recognised")
        checksum = self.integrity.publisher_checksum
        if checksum is not None:
            expected_length = 32 if checksum.algorithm == "md5" else 64 if checksum.algorithm == "sha256" else 0
            if not expected_length or not re.fullmatch(
                rf"[0-9a-fA-F]{{{expected_length}}}", checksum.value
            ):
                problems.append("integrity.publisher_checksum is not a valid digest")
            if self.integrity.checksum_status != "verified":
                problems.append(
                    "integrity.checksum_status must be 'verified' when a publisher checksum is stored"
                )
        elif self.integrity.checksum_status == "verified":
            problems.append(
                "integrity.checksum_status cannot be 'verified' without a publisher checksum"
            )
        if self.acquisition.attempt < 1:
            problems.append("acquisition.attempt must be >= 1")
        if not self.acquisition.method.strip():
            problems.append("acquisition.method must not be empty")
        if self.acquisition.status != "success":
            problems.append("acquisition.status must be 'success' for a stored manifest")
        return problems


@dataclass
class AcquisitionResult:
    run_id: str
    generated_at: str
    status: str
    attempts: list[AcquisitionAttempt]
    data_dir: str
    candidate: dict[str, Any] | None = None
    artifact: AcquisitionArtifact | None = None
    manifest: DatasetManifest | None = None
    manifest_path: str | None = None
    cached: bool = False
    warnings: list[str] = field(default_factory=list)
    failure: AcquisitionFailure | None = None
    seed: int = 0
    code_version: str = ""
    dataset_sha256: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == AcquisitionStatus.SUCCESS.value and self.artifact is not None

    def to_dict(self) -> dict[str, Any]:
        artifact = self.artifact.to_dict() if self.artifact else None
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": 3,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "status": self.status,
            "cached": self.cached,
            "seed": self.seed,
            "code_version": self.code_version,
            "dataset_sha256": self.dataset_sha256,
            "candidate": self.candidate,
            "artifact": artifact,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "manifest_path": self.manifest_path,
            "storage": {
                "data_dir": self.data_dir,
                "cache_path": artifact["cache_path"] if artifact else None,
                "raw_path": artifact["raw_path"] if artifact else None,
                "extracted_path": artifact["extracted_path"] if artifact else None,
            },
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "warnings": list(self.warnings),
            "failure": self.failure.to_dict() if self.failure else None,
        }
