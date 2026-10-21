from __future__ import annotations

import json

import pytest

from churnwise.acquisition.inputs import load_candidates
from tests.acquisition_helpers import make_acquisition_candidate


def write_report(tmp_path, ranking):
    directory = tmp_path / "reports" / "acquisition"
    directory.mkdir(parents=True)
    path = directory / "candidates.json"
    path.write_text(json.dumps({"ranking": ranking, "selected": None}), encoding="utf-8")
    return path


def entry(rank, candidate):
    payload = candidate.to_dict()
    payload.update({"eligible": True, "duplicate_of": None, "rank": rank, "score": 0.9})
    return payload


def test_candidates_are_loaded_in_rank_order_not_array_order(tmp_path):
    first = make_acquisition_candidate(
        candidate_id="openml:1", source_dataset_id="1", resource_url="https://www.openml.org/data/v1/download/1"
    )
    second = make_acquisition_candidate(
        candidate_id="openml:2", source_dataset_id="2", resource_url="https://www.openml.org/data/v1/download/2"
    )
    path = write_report(tmp_path, [entry(2, second), entry(1, first)])

    candidates = load_candidates(path)

    assert [candidate.candidate_id for candidate in candidates] == ["openml:1", "openml:2"]


def test_ineligible_and_duplicate_entries_are_skipped(tmp_path):
    candidate = make_acquisition_candidate()
    ineligible = candidate.to_dict()
    ineligible.update({"eligible": False, "duplicate_of": None, "rank": 1})
    duplicate = candidate.to_dict()
    duplicate.update({"eligible": True, "duplicate_of": "openml:46915", "rank": 2})
    path = write_report(tmp_path, [ineligible, duplicate])

    assert load_candidates(path) == []


def test_malformed_ranking_entry_is_a_value_error(tmp_path):
    path = write_report(tmp_path, [{"eligible": True, "duplicate_of": None, "rank": 1}])

    with pytest.raises(ValueError, match="ranking entry is invalid"):
        load_candidates(path)
