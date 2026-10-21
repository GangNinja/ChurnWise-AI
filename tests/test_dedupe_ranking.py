from __future__ import annotations

from churnwise.contracts import FieldInfo, SearchQuery
from churnwise.discovery.dedupe import find_duplicate_groups, rank_assessments
from churnwise.discovery.suitability import assess_candidate
from tests.helpers import make_candidate

QUERY = SearchQuery()


def assess(**overrides):
    return assess_candidate(make_candidate(**overrides), QUERY)


def test_same_source_title_and_rows_are_duplicates():
    first = assess(candidate_id="openml:1", source_dataset_id="1", title="churn")
    second = assess(candidate_id="openml:2", source_dataset_id="2", title="Churn")
    groups, assessed = find_duplicate_groups([first, second])

    assert len(groups) == 1
    assert groups[0].representative_id == "openml:1"
    assert assessed[1].duplicate_of == "openml:1"
    assert assessed[1].duplicate_reasons


def test_near_identical_shape_and_balance_are_duplicates():
    # Distinct titles so the same-title rule does not mask the shape rule.
    first = assess(
        candidate_id="openml:40701", source_dataset_id="40701", title="churn", feature_count=21
    )
    second = assess(
        candidate_id="openml:46915", source_dataset_id="46915", title="churn v4", feature_count=20
    )
    groups, _ = find_duplicate_groups([first, second])
    assert len(groups) == 1
    assert "class balance" in " ".join(groups[0].reasons)


def test_different_rows_are_not_duplicates():
    first = assess(candidate_id="openml:1", source_dataset_id="1", row_count=5000)
    second = assess(candidate_id="openml:2", source_dataset_id="2", row_count=7043)
    groups, assessed = find_duplicate_groups([first, second])
    assert groups == []
    assert all(a.duplicate_of is None for a in assessed)


def test_representative_prefers_eligible_member_over_lower_id():
    ineligible = assess(
        candidate_id="openml:1",
        source_dataset_id="1",
        title="churn",
        schema_fields=[
            FieldInfo("CustomerChurned", "nominal", "target"),
            FieldInfo("email", "string", "feature"),
        ],
    )
    eligible = assess(candidate_id="openml:2", source_dataset_id="2", title="churn")
    assert ineligible.eligible is False
    assert eligible.eligible is True

    groups, assessed = find_duplicate_groups([ineligible, eligible])
    assert groups[0].representative_id == "openml:2"
    assert assessed[0].duplicate_of == "openml:2"


def test_ranking_orders_by_score_first():
    higher = assess(candidate_id="openml:9", source_dataset_id="9")
    lower = assess(candidate_id="openml:3", source_dataset_id="3", class_balance=None)
    ranked = rank_assessments([lower, higher])
    assert [a.candidate.candidate_id for a in ranked] == ["openml:9", "openml:3"]
    assert [a.rank for a in ranked] == [1, 2]


def test_equal_scores_break_ties_by_id():
    first = assess(candidate_id="openml:9", source_dataset_id="9")
    second = assess(candidate_id="openml:3", source_dataset_id="3")
    ranked = rank_assessments([first, second])
    assert [a.candidate.candidate_id for a in ranked] == ["openml:3", "openml:9"]


def test_duplicate_members_are_excluded_from_ranking():
    first = assess(candidate_id="openml:1", source_dataset_id="1", title="churn")
    second = assess(candidate_id="openml:2", source_dataset_id="2", title="churn")
    find_duplicate_groups([first, second])
    ranked = rank_assessments([first, second])
    assert [a.candidate.candidate_id for a in ranked] == ["openml:1"]
