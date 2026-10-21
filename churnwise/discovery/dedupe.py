"""Duplicate detection across source listings.

OpenML in particular hosts several versions/re-uploads of the same corpus.
Counting them as independent candidates overstates choice and diversity, so
they are grouped and only the best representative is ranked.

Rules are metadata-only and deliberately conservative; a false positive is
reported with its reason so a human can challenge it.
"""

from __future__ import annotations

import re

from churnwise.contracts import CandidateAssessment, DuplicateGroup
from churnwise.discovery.registry import source_priority


def _normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _minority_rate(assessment: CandidateAssessment) -> float | None:
    balance = assessment.candidate.class_balance
    if not balance:
        return None
    total = sum(balance.values())
    return balance["minority"] / total if total else None


def duplicate_reason(a: CandidateAssessment, b: CandidateAssessment) -> str | None:
    left, right = a.candidate, b.candidate
    if left.candidate_id == right.candidate_id:
        return "identical candidate id"

    if (
        left.source_id == right.source_id
        and left.row_count is not None
        and left.row_count == right.row_count
        and _normalized_title(left.title) == _normalized_title(right.title)
    ):
        return "same source, same normalized title and row count"

    if (
        left.row_count is not None
        and left.row_count == right.row_count
        and left.feature_count is not None
        and right.feature_count is not None
        and abs(left.feature_count - right.feature_count) <= 1
    ):
        left_rate, right_rate = _minority_rate(a), _minority_rate(b)
        if left_rate is not None and right_rate is not None and abs(left_rate - right_rate) <= 0.01:
            return "same row count, near-identical feature count and class balance"
        if (
            left.feature_count == right.feature_count
            and left.target_column
            and right.target_column
            and left.target_column.lower() == right.target_column.lower()
        ):
            return "same row count, feature count and target name"

    left_fingerprint, right_fingerprint = left.fingerprint(), right.fingerprint()
    if left_fingerprint and left_fingerprint == right_fingerprint:
        return "identical metadata fingerprint"

    return None


def find_duplicate_groups(
    assessments: list[CandidateAssessment],
) -> tuple[list[DuplicateGroup], list[CandidateAssessment]]:
    """Group duplicates in place and return ``(groups, assessments)``.

    The representative of a group is the eligible member with the highest
    score; ties break by source priority then candidate id, so the choice is
    reproducible.
    """
    parent = list(range(len(assessments)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    pair_reasons: list[tuple[int, int, str]] = []
    for i in range(len(assessments)):
        for j in range(i + 1, len(assessments)):
            reason = duplicate_reason(assessments[i], assessments[j])
            if reason:
                union(i, j)
                pair_reasons.append((i, j, reason))

    groups_by_root: dict[int, list[int]] = {}
    for index in range(len(assessments)):
        groups_by_root.setdefault(find(index), []).append(index)

    duplicate_groups: list[DuplicateGroup] = []
    for root, members in sorted(groups_by_root.items()):
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda idx: (
                not assessments[idx].eligible,
                -assessments[idx].score,
                source_priority(assessments[idx].candidate.source_id),
                assessments[idx].candidate.candidate_id,
            ),
        )
        representative = assessments[ordered[0]]
        reasons = sorted(
            {
                reason
                for i, j, reason in pair_reasons
                if find(i) == root and find(j) == root
            }
        )
        for idx in ordered[1:]:
            assessments[idx].duplicate_of = representative.candidate.candidate_id
            assessments[idx].duplicate_reasons = reasons
        duplicate_groups.append(
            DuplicateGroup(
                representative_id=representative.candidate.candidate_id,
                member_ids=sorted(assessments[idx].candidate.candidate_id for idx in ordered),
                reasons=reasons,
            )
        )

    return duplicate_groups, assessments


def rank_assessments(assessments: list[CandidateAssessment]) -> list[CandidateAssessment]:
    """Assign 1-based ranks to unique candidates by score, then priority, then id."""
    unique = [a for a in assessments if a.duplicate_of is None]
    unique.sort(
        key=lambda a: (
            -a.score,
            source_priority(a.candidate.source_id),
            a.candidate.candidate_id,
        )
    )
    for position, assessment in enumerate(unique, start=1):
        assessment.rank = position
    return unique
