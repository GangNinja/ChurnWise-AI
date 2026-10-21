"""Structured acquisition failures.

Codes are stable and documented in ``docs/ACQUISITION.md``. Every failure has
a category so the coordinator can decide whether another candidate is worth
trying; only network failures are retryable, and only within the downloader's
bounded retry loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CODE_NO_CANDIDATE = "CW-ACQ-001"
CODE_INTEGRITY = "CW-ACQ-002"
CODE_UNEXPECTED_CONTENT = "CW-ACQ-003"
CODE_NETWORK = "CW-ACQ-004"
CODE_OFFLINE = "CW-ACQ-005"
CODE_ACCESS = "CW-ACQ-006"
CODE_SIZE_LIMIT = "CW-ACQ-007"
CODE_STORAGE = "CW-ACQ-008"
CODE_CORRUPT_ARCHIVE = "CW-ACQ-009"
CODE_UNSAFE_ARCHIVE = "CW-ACQ-010"
CODE_INTERRUPTED = "CW-ACQ-011"
CODE_ALL_FAILED = "CW-ACQ-012"
CODE_CACHE_INVALID = "CW-ACQ-013"

CATEGORY_CANDIDATE = "candidate"
CATEGORY_NETWORK = "network"
CATEGORY_PERMISSION = "permission"
CATEGORY_FORMAT = "format"
CATEGORY_INTEGRITY = "integrity"
CATEGORY_STORAGE = "storage"
CATEGORY_ARCHIVE = "archive"
CATEGORY_OFFLINE = "offline"
CATEGORY_CACHE = "cache"
CATEGORY_INTERRUPTED = "interrupted"

_REMEDIATION = {
    CODE_NO_CANDIDATE: "Run discovery again; no candidate had a usable open resource.",
    CODE_INTEGRITY: "Discard the artifact and retry; if it persists, the source content changed or is corrupted.",
    CODE_UNEXPECTED_CONTENT: "The source returned HTML or a file that does not match its declared format; inspect the landing page.",
    CODE_NETWORK: "Check network access, then retry. Transient failures were already retried within the configured limit.",
    CODE_OFFLINE: "Run once with network access to populate the cache, or run discovery/acquisition without --offline.",
    CODE_ACCESS: "The resource is not openly accessible; choose another candidate or review the source's access policy.",
    CODE_SIZE_LIMIT: "The artifact exceeded the configured cap; raise the cap only after reviewing the source.",
    CODE_STORAGE: "Check free disk space and write permissions for the data directory.",
    CODE_CORRUPT_ARCHIVE: "The ZIP archive is corrupt or truncated; retry from the source.",
    CODE_UNSAFE_ARCHIVE: "The archive contains an unsafe path, symlink or executable; it was not extracted.",
    CODE_INTERRUPTED: "The transfer did not complete; retry the acquisition.",
    CODE_ALL_FAILED: "Every eligible candidate failed; inspect attempts.json for per-candidate reasons.",
    CODE_CACHE_INVALID: "A cached artifact failed verification; it was not reused.",
}


@dataclass
class AcquisitionError(Exception):
    """A failure with an actionable, stable contract."""

    code: str
    message: str
    category: str
    retryable: bool = False
    remediation: str | None = None
    http_status: int | None = None
    causes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.remediation is None:
            self.remediation = _REMEDIATION.get(self.code)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "retryable": self.retryable,
            "remediation": self.remediation,
            "http_status": self.http_status,
            "causes": list(self.causes),
        }


def failure(
    code: str,
    message: str,
    *,
    category: str = CATEGORY_CANDIDATE,
    retryable: bool = False,
    http_status: int | None = None,
    causes: list[str] | None = None,
) -> AcquisitionError:
    return AcquisitionError(
        code=code,
        message=message,
        category=category,
        retryable=retryable,
        http_status=http_status,
        causes=list(causes or []),
    )
