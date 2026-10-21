"""Recommended-candidate selection.

Selection is deliberately trivial and isolated: take the highest-ranked
candidate that passed every blocking gate. Keeping it out of the coordinator
makes it obvious that ranking and selection are separate decisions and that
selection never fabricates a candidate when none is eligible.
"""

from __future__ import annotations

from collections.abc import Sequence

from churnwise.contracts import CandidateAssessment, SelectedCandidate


def select_candidate(ranked: Sequence[CandidateAssessment]) -> SelectedCandidate | None:
    """Return the Phase 3 hand-off for the best eligible assessment, else None."""
    for assessment in ranked:
        if assessment.eligible:
            return SelectedCandidate.from_candidate(assessment.candidate)
    return None
