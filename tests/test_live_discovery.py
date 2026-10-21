"""Live-source verification.

These tests contact OpenML, UCI and GitHub. They are skipped unless
``--run-live`` is passed, and their results are reported honestly in
docs/PHASE2_REPORT.md. Mocked tests prove adapter logic; only these prove
real-source behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from churnwise.config import Settings
from churnwise.contracts import SearchQuery, SourceStatus
from churnwise.discovery import build_adapters, discover
from churnwise.discovery.adapters.openml import OpenMLAdapter
from churnwise.discovery.adapters.uci import UCIAdapter
from churnwise.discovery.registry import get_source
from churnwise.net import HttpClient, RequestsTransport

pytestmark = pytest.mark.live


def live_client(source_id: str) -> HttpClient:
    spec = get_source(source_id)
    return HttpClient(
        spec.hosts,
        timeout_s=25,
        max_retries=1,
        max_requests=25,
        transport=RequestsTransport("ChurnWiseAI/0.2 (live verification)"),
    )


def test_live_openml_returns_enriched_candidates():
    adapter = OpenMLAdapter(get_source("openml"), live_client("openml"))
    result = adapter.search(
        SearchQuery(
            terms=("churn",),
            max_results_per_source=5,
            enrich_limit=3,
            timeout_s=25,
            max_retries=1,
        )
    )
    assert result.status is SourceStatus.OK, result.error_message
    assert result.candidates
    assert any(candidate.target_column for candidate in result.candidates)


def test_live_openml_handles_terms_with_no_results():
    adapter = OpenMLAdapter(get_source("openml"), live_client("openml"))
    result = adapter.search(
        SearchQuery(
            terms=("churn", "customer churn"),
            max_results_per_source=5,
            enrich_limit=2,
            timeout_s=25,
            max_retries=1,
        )
    )
    assert result.status is SourceStatus.OK, result.error_message
    assert result.candidates


def test_live_uci_finds_iranian_churn():
    adapter = UCIAdapter(get_source("uci"), live_client("uci"))
    result = adapter.search(
        SearchQuery(terms=("churn",), enrich_limit=3, timeout_s=25, max_retries=1)
    )
    assert result.status is SourceStatus.OK, result.error_message
    assert any("iranian churn" in candidate.title.lower() for candidate in result.candidates)


def test_live_full_discovery_selects_an_eligible_candidate():
    settings = Settings(project_root=Path.cwd())
    query = SearchQuery(
        terms=("churn",),
        max_results_per_source=5,
        enrich_limit=6,
        timeout_s=25,
        max_retries=1,
    )
    report = discover(build_adapters(settings, query), query, settings)

    assert report.failure is None, report.failure.to_dict() if report.failure else ""
    assert report.selected is not None
    assert report.selected.source_id == "openml"
    assert report.selected.resource_url is not None
    assert report.selected.license_status == "verified"
