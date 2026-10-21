"""Candidate metadata validation.

Separated from orchestration so the contract-level checks (identifier
uniqueness, required metadata, supported formats) can be tested and replaced
independently of source queries and ranking.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from churnwise.contracts import DatasetCandidate, Severity


def validate_candidates(
    candidates: Iterable[DatasetCandidate],
) -> tuple[list[DatasetCandidate], list[dict[str, Any]]]:
    """Split raw candidates into valid candidates and structured rejections.

    The first occurrence of a candidate id wins; repeats are recorded rather
    than merged, because a repeated listing is evidence about the source, not
    about the dataset.
    """
    valid: list[DatasetCandidate] = []
    invalid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            invalid.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "stage": "validation",
                    "reason": "candidate id appeared more than once; repeated listing ignored",
                }
            )
            continue

        issues = candidate.validate()
        errors = [issue for issue in issues if issue.severity is Severity.ERROR]
        if errors:
            invalid.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "stage": "validation",
                    "reason": errors[0].message,
                    "issues": [issue.to_dict() for issue in issues],
                }
            )
            continue

        seen_ids.add(candidate.candidate_id)
        valid.append(candidate)

    return valid, invalid
