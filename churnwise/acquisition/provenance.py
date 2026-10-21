"""Provenance helpers: sanitized references, fingerprints and manifest IO."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from churnwise.acquisition.contracts import DatasetManifest, PublisherChecksum
from churnwise.contracts import DatasetCandidate

SENSITIVE_QUERY_MARKERS = (
    "token",
    "key",
    "signature",
    "sig",
    "auth",
    "password",
    "passwd",
    "secret",
    "credential",
    "expires",
    "x-amz-",
    "session",
)
_CHECKSUM_RE = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{64}$")


def sanitize_url(url: str | None) -> str | None:
    """Remove credentials, fragments and sensitive query parameters from a URL."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = parts.hostname or ""
    if not host:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    netloc = host
    if port and port != 443:
        netloc = f"{host}:{port}"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not any(marker in key.lower() for marker in SENSITIVE_QUERY_MARKERS)
    ]
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), ""))


def sanitize_payload(value: Any) -> Any:
    """Recursively sanitize URL-like strings before storing provenance."""
    if isinstance(value, dict):
        return {key: sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        return sanitize_url(value)
    return value


def sanitized_candidate_dict(candidate: DatasetCandidate) -> dict[str, Any]:
    return sanitize_payload(candidate.to_dict())


def candidate_fingerprint(candidate: DatasetCandidate) -> str:
    """Stable fingerprint for cache reuse, excluding retrieval timestamps.

    A publisher checksum or version change invalidates the cache even when the
    resource URL and row counts are unchanged. ``access`` is included so a
    policy change cannot silently reuse bytes that are no longer open.
    """
    checksum = publisher_checksum(candidate)
    evidence = candidate.evidence if isinstance(candidate.evidence, dict) else {}
    version = evidence.get("version")
    payload = {
        "candidate_id": candidate.candidate_id,
        "source_id": candidate.source_id,
        "source_dataset_id": candidate.source_dataset_id,
        "resource_url": sanitize_url(candidate.resource_url),
        "format": candidate.format,
        "size_bytes": candidate.size_bytes,
        "row_count": candidate.row_count,
        "feature_count": candidate.feature_count,
        "target_column": candidate.target_column,
        "license_status": candidate.license_status.value,
        "access": candidate.access,
        "version": str(version) if version is not None else None,
        "publisher_checksum": checksum.to_dict() if checksum else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def publisher_checksum(candidate: DatasetCandidate) -> PublisherChecksum | None:
    """Read a checksum only when the source actually supplied one.

    Discovery currently does not publish checksums, so this normally returns
    ``None``. The parser is intentionally narrow: a value must be a full MD5 or
    SHA-256 hex digest and a recognised evidence key, never a guess.
    """
    evidence = candidate.evidence if isinstance(candidate.evidence, dict) else {}
    for algorithm, keys in (
        ("sha256", ("sha256", "sha256_checksum", "sha_256")),
        ("md5", ("md5", "md5_checksum", "md5sum")),
    ):
        for key in keys:
            value = evidence.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip().lower()
            if _CHECKSUM_RE.match(value) and len(value) == (64 if algorithm == "sha256" else 32):
                return PublisherChecksum(algorithm, value)
    return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_manifest(manifest: DatasetManifest, path: Path | str) -> Path:
    path = Path(path)
    problems = manifest.validate()
    if problems:
        raise ValueError("invalid manifest: " + "; ".join(problems))
    atomic_write_json(path, manifest.to_dict())
    return path


def read_manifest(path: Path | str) -> DatasetManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    try:
        manifest = DatasetManifest.from_dict(payload)
        problems = manifest.validate()
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid manifest shape: {exc}") from exc
    if problems:
        raise ValueError("invalid manifest: " + "; ".join(problems))
    return manifest
