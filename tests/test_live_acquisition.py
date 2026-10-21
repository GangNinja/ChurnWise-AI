"""Live Phase 3 verification.

Skipped unless ``--run-live`` is passed. This test performs a real Phase 2
discovery, downloads the selected OpenML artifact, recomputes its SHA-256 and
then proves offline cache reuse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.acquisition.integrity import sha256_file
from churnwise.config import Settings
from churnwise.contracts import SearchQuery
from churnwise.discovery import build_adapters, discover

pytestmark = pytest.mark.live


def test_live_openml_artifact_is_acquired_verified_and_cached(tmp_path):
    settings = Settings(project_root=tmp_path)
    query = SearchQuery(
        terms=("churn",),
        max_results_per_source=5,
        enrich_limit=6,
        timeout_s=25,
        max_retries=1,
    )
    report = discover(build_adapters(settings, query), query, settings)
    assert report.selected is not None, report.failure.to_dict() if report.failure else ""

    coordinator = AcquisitionCoordinator(settings)
    result = coordinator.acquire(report)

    assert result.succeeded, result.failure.to_dict() if result.failure else ""
    assert result.artifact is not None
    artifact_path = settings.data_dir / result.artifact.raw_path
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size == result.artifact.bytes
    assert sha256_file(artifact_path) == result.artifact.sha256
    assert result.manifest is not None
    assert result.manifest.validate() == []
    assert result.manifest.integrity.verified is True
    if result.manifest.integrity.publisher_checksum is not None:
        assert result.manifest.integrity.checksum_status == "verified"
    assert result.manifest.file.format in {"csv", "arff", "parquet", "zip", "tsv", "json", "xlsx"}
    assert result.manifest.source.license_status in {"verified", "unverified"}
    assert (settings.data_dir / result.artifact.cache_path).is_file()
    assert result.manifest_path is not None and Path(result.manifest_path).is_file()

    cached = AcquisitionCoordinator(settings).acquire(report, offline=True)
    assert cached.succeeded, cached.failure.to_dict() if cached.failure else ""
    assert cached.cached is True
    assert cached.artifact.sha256 == result.artifact.sha256
    assert cached.attempts[0].status == "cached"
