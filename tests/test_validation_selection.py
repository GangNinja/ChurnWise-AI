from __future__ import annotations

from churnwise.contracts import SearchQuery
from churnwise.discovery.selection import select_candidate
from churnwise.discovery.suitability import assess_candidate
from churnwise.discovery.validation import validate_candidates
from tests.helpers import make_candidate

QUERY = SearchQuery()


def test_validation_rejects_malformed_metadata_and_repeated_ids():
    malformed = make_candidate(candidate_id="openml: bad id")
    first = make_candidate(candidate_id="openml:1")
    repeat = make_candidate(candidate_id="openml:1", title="same listing again")

    valid, invalid = validate_candidates([malformed, first, repeat])

    assert [c.candidate_id for c in valid] == ["openml:1"]
    assert len(invalid) == 2
    assert all(entry["stage"] == "validation" for entry in invalid)
    assert invalid[1]["reason"].startswith("candidate id appeared more than once")


def test_selection_returns_the_best_eligible_candidate_or_nothing():
    blocked = assess_candidate(make_candidate(candidate_id="openml:1", format="docx"), QUERY)
    eligible = assess_candidate(make_candidate(candidate_id="openml:2"), QUERY)

    assert blocked.eligible is False
    assert eligible.eligible is True

    selected = select_candidate([blocked, eligible])
    assert selected is not None
    assert selected.candidate_id == "openml:2"
    assert selected.source_dataset_id == "46915"

    assert select_candidate([blocked]) is None
    assert select_candidate([]) is None
