from __future__ import annotations

import json

import pytest

from churnwise.acquisition.provenance import (
    candidate_fingerprint,
    publisher_checksum,
    read_manifest,
    sanitize_url,
)
from tests.acquisition_helpers import make_acquisition_candidate, make_manifest


def test_sanitize_url_removes_credentials_fragment_and_sensitive_parameters():
    value = sanitize_url("https://user:pass@example.org/data?token=abc&foo=bar&X-Amz-Signature=sig#part")
    assert value == "https://example.org/data?foo=bar"


def test_sanitize_url_returns_none_for_malformed_input():
    assert sanitize_url(None) is None
    assert sanitize_url("") is None
    assert sanitize_url("not a url") is None
    assert sanitize_url("https://www.openml.org:not-a-port/data") is None


def test_candidate_fingerprint_ignores_retrieval_timestamp_and_sensitive_token():
    first = make_acquisition_candidate(
        resource_url="https://www.openml.org/data/v1/download/1?token=one",
        retrieved_at="2026-10-20T00:00:00Z",
    )
    second = make_acquisition_candidate(
        resource_url="https://www.openml.org/data/v1/download/1?token=two",
        retrieved_at="2026-10-21T00:00:00Z",
    )
    assert candidate_fingerprint(first) == candidate_fingerprint(second)


def test_manifest_reader_requires_a_boolean_verified_flag(tmp_path):
    candidate = make_acquisition_candidate()
    manifest = make_manifest(
        candidate,
        sha256="a" * 64,
        bytes_=1,
        cache_path="cache/sha256/" + "a" * 64,
        raw_path="raw/openml/46915/2025-04-30/artifact.csv",
    )
    payload = manifest.to_dict()
    payload["integrity"]["verified"] = "false"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="verified must be a boolean"):
        read_manifest(path)


def test_manifest_schema_rejects_an_unverified_artifact():
    candidate = make_acquisition_candidate()
    manifest = make_manifest(
        candidate,
        sha256="a" * 64,
        bytes_=1,
        cache_path="cache/sha256/" + "a" * 64,
        raw_path="raw/openml/46915/2025-04-30/artifact.csv",
    )
    assert manifest.validate() == []

    manifest.integrity.verified = False

    assert any("integrity.verified" in problem for problem in manifest.validate())


def test_publisher_checksum_is_only_read_when_present_and_well_formed():
    assert publisher_checksum(make_acquisition_candidate()) is None
    assert publisher_checksum(make_acquisition_candidate(evidence={"sha256": "abc"})) is None
    digest = "a" * 64
    checksum = publisher_checksum(make_acquisition_candidate(evidence={"sha256": digest}))
    assert checksum is not None
    assert checksum.algorithm == "sha256"
    assert checksum.value == digest
