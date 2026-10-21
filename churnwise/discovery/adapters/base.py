"""Shared adapter behaviour: timing, failure mapping and parsing helpers."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from churnwise.contracts import (
    DatasetCandidate,
    SearchQuery,
    SourceResult,
    SourceStatus,
)
from churnwise.discovery.registry import SourceSpec
from churnwise.net import (
    CODE_BUDGET,
    CODE_HTTP,
    CODE_PROTOCOL,
    CODE_RATE_LIMITED,
    CODE_TIMEOUT,
    CODE_TOO_LARGE,
    CODE_UNSAFE_URL,
    DiscoveryHttpError,
    HttpClient,
    UnsafeUrlError,
)

CODE_INTERNAL = "CW-SRC-008"

logger = logging.getLogger("churnwise.discovery")

_FORMAT_ALIASES = {
    "csv": "csv",
    "text/csv": "csv",
    "tsv": "tsv",
    "arff": "arff",
    "parquet": "parquet",
    "xlsx": "xlsx",
    "xls": "xlsx",
    "zip": "zip",
    "json": "json",
    "jsonl": "json",
}

_REMEDIATION = {
    CODE_TIMEOUT: "Check network connectivity or increase --timeout, then retry.",
    CODE_RATE_LIMITED: "Wait before retrying and reduce --limit / the number of search terms.",
    CODE_BUDGET: "Reduce --limit or the number of search terms; the per-source request budget was exhausted.",
    CODE_PROTOCOL: "The source response format may have changed; inspect the adapter and endpoint.",
    CODE_TOO_LARGE: "The source returned an unexpectedly large metadata payload; inspect the endpoint.",
    CODE_UNSAFE_URL: "The adapter produced a URL outside its allowlist; this is a code defect.",
    CODE_HTTP: "The source returned an HTTP error; the endpoint may have moved or be temporarily unavailable.",
    CODE_INTERNAL: "An unexpected adapter error occurred; see the structured logs for the exception type.",
}


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+", text):
            try:
                return int(text)
            except ValueError:
                return None
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else None
    return None


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def canonical_format(raw: Any) -> str:
    text = (as_text(raw) or "").lower()
    if not text:
        return "unknown"
    if text in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[text]
    for alias, canonical in _FORMAT_ALIASES.items():
        if alias in text:
            return canonical
    return "unknown"


def clean_description(value: Any, limit: int = 800) -> str | None:
    text = as_text(value)
    if not text:
        return None
    text = re.sub(r"[`*_#>\[\]]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text or None


def publisher_from(value: Any) -> str | None:
    if isinstance(value, str):
        return as_text(value)
    if isinstance(value, dict):
        return as_text(value.get("name")) or as_text(value.get("title"))
    if isinstance(value, (list, tuple)):
        names = [publisher_from(item) for item in value]
        clean = [name for name in names if name]
        return ", ".join(clean) if clean else None
    return None


class SourceAdapter(ABC):
    """Base class that guarantees one source cannot break a discovery run."""

    def __init__(self, spec: SourceSpec, client: HttpClient) -> None:
        self.spec = spec
        self.client = client

    @abstractmethod
    def _discover(self, query: SearchQuery) -> tuple[list[DatasetCandidate], list[str]]:
        """Return candidates plus human-readable notes about partial failures."""

    def search(self, query: SearchQuery) -> SourceResult:
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            candidates, notes = self._discover(query)
        except UnsafeUrlError as exc:
            return self._failure(CODE_UNSAFE_URL, str(exc), elapsed_ms())
        except DiscoveryHttpError as exc:
            return self._failure(
                exc.code,
                exc.message,
                elapsed_ms(),
                http_status=exc.status,
            )
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            logger.exception(
                "adapter raised an unexpected error",
                extra={"source_id": self.spec.source_id, "status": "error"},
            )
            return self._failure(
                CODE_INTERNAL,
                f"unexpected adapter error: {type(exc).__name__}",
                elapsed_ms(),
            )

        if candidates:
            status = SourceStatus.OK
        elif notes:
            # Search succeeded but every metadata enrichment failed. This is not
            # a successful empty search, so surface it as a source error.
            return SourceResult(
                source_id=self.spec.source_id,
                status=SourceStatus.ERROR,
                error_code=CODE_PROTOCOL,
                error_message=notes[0],
                remediation=_REMEDIATION[CODE_PROTOCOL],
                request_count=self.client.request_count,
                duration_ms=elapsed_ms(),
                notes=notes,
            )
        else:
            status = SourceStatus.EMPTY
        logger.info(
            "source search finished",
            extra={
                "source_id": self.spec.source_id,
                "status": status.value,
                "request_count": self.client.request_count,
                "duration_ms": elapsed_ms(),
            },
        )
        return SourceResult(
            source_id=self.spec.source_id,
            status=status,
            candidates=candidates,
            request_count=self.client.request_count,
            duration_ms=elapsed_ms(),
            notes=notes,
        )

    def _failure(
        self,
        code: str,
        message: str,
        duration_ms: int,
        *,
        http_status: int | None = None,
    ) -> SourceResult:
        if code == CODE_RATE_LIMITED:
            status = SourceStatus.RATE_LIMITED
        elif code == CODE_TIMEOUT:
            status = SourceStatus.UNAVAILABLE
        else:
            status = SourceStatus.ERROR
        logger.warning(
            "source search failed",
            extra={
                "source_id": self.spec.source_id,
                "status": status.value,
                "duration_ms": duration_ms,
                "request_count": self.client.request_count,
            },
        )
        return SourceResult(
            source_id=self.spec.source_id,
            status=status,
            error_code=code,
            error_message=message,
            remediation=_REMEDIATION.get(code),
            http_status=http_status,
            request_count=self.client.request_count,
            duration_ms=duration_ms,
        )

    def _candidate_id(self, dataset_id: str) -> str:
        return f"{self.spec.source_id}:{dataset_id}"
