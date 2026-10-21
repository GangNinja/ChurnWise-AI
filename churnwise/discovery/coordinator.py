"""Discovery coordinator.

Queries adapters, normalizes and validates candidates, scores them, collapses
duplicates, ranks the result and selects at most one eligible candidate. Every
outcome — including total failure — is represented as a structured report.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from churnwise.config import Settings
from churnwise.contracts import (
    CandidateAssessment,
    DiscoveryFailure,
    DiscoveryReport,
    GateStatus,
    SearchQuery,
    SelectedCandidate,
    SourceResult,
    SourceStatus,
    utc_now_iso,
)
from churnwise.discovery.dedupe import find_duplicate_groups, rank_assessments
from churnwise.discovery.registry import enabled_sources
from churnwise.discovery.selection import select_candidate
from churnwise.discovery.suitability import assess_candidate
from churnwise.discovery.validation import validate_candidates

CODE_ALL_SOURCES_FAILED = "CW-DISC-001"
CODE_NO_CANDIDATES = "CW-DISC-002"
CODE_NO_ELIGIBLE = "CW-DISC-003"
CODE_INVALID_CONFIG = "CW-DISC-004"
CODE_OFFLINE = "CW-DISC-005"
CODE_NO_SOURCES = "CW-DISC-006"

FAILED_SOURCE_STATUSES = frozenset(
    {SourceStatus.ERROR, SourceStatus.UNAVAILABLE, SourceStatus.RATE_LIMITED, SourceStatus.DISABLED}
)

logger = logging.getLogger("churnwise.discovery")


def _failure(
    code: str,
    message: str,
    sources: list[str],
    reasons: list[str],
    next_action: str,
) -> DiscoveryFailure:
    return DiscoveryFailure(
        code=code,
        message=message,
        sources_checked=sources,
        reasons=reasons,
        next_action=next_action,
    )


def _source_failure_reasons(results: Sequence[SourceResult]) -> list[str]:
    reasons = []
    for result in results:
        if result.status in FAILED_SOURCE_STATUSES:
            reasons.append(f"{result.source_id}: {result.error_code} {result.error_message}")
    return reasons


def _adapter_source_id(adapter: Any) -> str:
    """Return the source id an adapter will report, without trusting its class."""
    source_id = getattr(adapter, "source_id", None)
    if isinstance(source_id, str):
        return source_id
    spec = getattr(adapter, "spec", None)
    spec_id = getattr(spec, "source_id", None)
    return spec_id if isinstance(spec_id, str) else "unknown"


def _gate_failure_reasons(ranked: Sequence[CandidateAssessment]) -> list[str]:
    reasons: list[str] = []
    for assessment in ranked:
        if assessment.eligible:
            continue
        blockers = [
            f"{gate.gate_id} {gate.status.value}: {gate.reason}"
            for gate in assessment.gates
            if gate.status is not GateStatus.PASS
        ]
        reasons.append(f"{assessment.candidate.candidate_id}: " + "; ".join(blockers[:4]))
    return reasons[:8]


def discover(
    adapters: Sequence[Any],
    query: SearchQuery,
    settings: Settings,
    *,
    offline: bool | None = None,
    now=utc_now_iso,
) -> DiscoveryReport:
    """Run discovery and always return a report (never raise for a source failure)."""
    generated_at = now()
    effective_offline = settings.offline if offline is None else offline
    source_ids = [spec.source_id for spec in enabled_sources()]

    config: dict[str, Any] = {
        "allow_unverified_license": query.allow_unverified_license,
        "min_rows": query.min_rows,
        "max_rows": query.max_rows,
        "timeout_s": query.timeout_s,
        "max_retries": query.max_retries,
        "max_requests_per_source": query.max_requests_per_source,
        "max_response_bytes": settings.max_response_bytes,
        "seed": settings.seed,
        "enabled_sources": source_ids,
    }

    problems = query.validate() + settings.validate()
    if problems:
        return DiscoveryReport(
            generated_at=generated_at,
            query=query,
            source_results=[],
            assessments=[],
            failure=_failure(
                CODE_INVALID_CONFIG,
                "discovery configuration is invalid",
                source_ids,
                problems,
                "Correct the reported configuration problems and retry.",
            ),
            config=config,
        )

    if effective_offline:
        return DiscoveryReport(
            generated_at=generated_at,
            query=query,
            source_results=[],
            assessments=[],
            failure=_failure(
                CODE_OFFLINE,
                "discovery requires network access but offline mode is enabled",
                source_ids,
                ["offline mode is enabled (--offline or CHURNWISE_OFFLINE=1)"],
                "Re-run without --offline; Phase 3 will support cache-only acquisition.",
            ),
            config=config,
        )

    if not adapters:
        return DiscoveryReport(
            generated_at=generated_at,
            query=query,
            source_results=[],
            assessments=[],
            failure=_failure(
                CODE_NO_SOURCES,
                "no discovery sources are enabled",
                source_ids,
                ["the adapter list is empty"],
                "Enable at least one source in churnwise/discovery/registry.py.",
            ),
            config=config,
        )

    config["enabled_sources"] = [_adapter_source_id(adapter) for adapter in adapters]

    source_results: list[SourceResult] = []
    for adapter in adapters:
        result = adapter.search(query)
        source_results.append(result)
        logger.info(
            "source result",
            extra={
                "source_id": result.source_id,
                "status": result.status.value,
                "request_count": result.request_count,
                "duration_ms": result.duration_ms,
            },
        )

    raw_candidates = [candidate for result in source_results for candidate in result.candidates]
    valid, invalid = validate_candidates(raw_candidates)

    assessments = [assess_candidate(candidate, query) for candidate in valid]
    duplicate_groups, assessments = find_duplicate_groups(assessments)
    ranked = rank_assessments(assessments)

    selected: SelectedCandidate | None = None
    failure: DiscoveryFailure | None = None

    all_sources_failed = bool(source_results) and all(
        result.status in FAILED_SOURCE_STATUSES for result in source_results
    )
    if all_sources_failed:
        failure = _failure(
            CODE_ALL_SOURCES_FAILED,
            "every configured source failed; no discovery result can be trusted",
            [result.source_id for result in source_results],
            _source_failure_reasons(source_results),
            "Check network access and retry. Per-source error codes are in "
            "reports/acquisition/candidates.json.",
        )
    elif not valid:
        empty_note = (
            "no candidates were returned"
            if not raw_candidates
            else "candidates were returned but none passed metadata validation"
        )
        failure = _failure(
            CODE_NO_CANDIDATES,
            f"discovery found no usable candidate ({empty_note})",
            [result.source_id for result in source_results],
            [result.error_message for result in source_results if result.error_message]
            or [empty_note],
            "Broaden --term values or verify source availability. No dataset is fabricated.",
        )
    elif (candidate := select_candidate(ranked)) is None:
        failure = _failure(
            CODE_NO_ELIGIBLE,
            "candidates were found, but none satisfied every blocking discovery gate",
            [result.source_id for result in source_results],
            _gate_failure_reasons(ranked),
            "Inspect reports/acquisition/rejections.json. If licence metadata is the only "
            "blocker, re-run with --allow-unverified-license.",
        )
    else:
        selected = candidate

    if selected is not None:
        logger.info(
            "selected candidate",
            extra={
                "candidate_id": selected.candidate_id,
                "status": "selected",
                "source_id": selected.source_id,
            },
        )

    return DiscoveryReport(
        generated_at=generated_at,
        query=query,
        source_results=source_results,
        assessments=assessments,
        duplicate_groups=duplicate_groups,
        selected=selected,
        failure=failure,
        invalid_candidates=invalid,
        config=config,
    )
