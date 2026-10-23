"""Validation and formatting used by the dashboard record-assessment page."""

from __future__ import annotations

import pytest

from churnwise.dashboard import actions
from churnwise.dashboard.pages import assessment


class FakeUpload:
    def __init__(self, data: bytes, name: str = "input.csv") -> None:
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def test_read_upload_accepts_a_valid_csv():
    frame, problem = assessment._read_upload(FakeUpload(b"a,b\n1,2\n3,4\n"))
    assert problem == ""
    assert frame is not None and len(frame) == 2
    assert list(frame.columns) == ["a", "b"]


def test_read_upload_rejects_oversized_files():
    frame, problem = assessment._read_upload(FakeUpload(b"a\n" + b"1\n" * (assessment.MAX_UPLOAD_BYTES // 2 + 1)))
    assert frame is None
    assert "10 MiB" in problem


def test_read_upload_rejects_unparseable_and_empty_files():
    frame, problem = assessment._read_upload(FakeUpload(b"\x00\xff\xfe"))
    assert frame is None
    assert "could not be parsed" in problem
    frame, problem = assessment._read_upload(FakeUpload(b"a,b\n"))
    assert frame is None
    assert "no data rows" in problem


def test_read_upload_rejects_too_many_rows(monkeypatch):
    monkeypatch.setattr(assessment, "MAX_BATCH_ROWS", 2)
    frame, problem = assessment._read_upload(FakeUpload(b"a\n1\n2\n3\n"))
    assert frame is None
    assert "at most 2" in problem


def test_column_kinds_and_numeric_parsing():
    schema = {
        "required_columns": ["age", "plan", "active"],
        "preprocessing": {
            "numeric": ["age"],
            "categorical": ["plan"],
            "boolean": ["active"],
            "categories": {"plan": ["basic", "pro"]},
        },
    }
    required, kinds, categories = assessment._column_kinds(schema)
    assert required == ["age", "plan", "active"]
    assert kinds == {"age": "numeric", "plan": "categorical", "active": "boolean"}
    assert categories["plan"] == ["basic", "pro"]

    record, problems = assessment._parse_numeric(
        {"age": "41.5", "plan": "pro", "active": "true"},
        required,
        kinds,
    )
    assert problems == []
    assert record == {"age": 41.5, "plan": "pro", "active": True}

    record, problems = assessment._parse_numeric(
        {"age": "", "plan": assessment.NOT_PROVIDED, "active": "false"},
        required,
        kinds,
    )
    assert record == {"age": None, "plan": None, "active": False}

    _, problems = assessment._parse_numeric(
        {"age": "not-a-number", "plan": "pro", "active": "true"},
        required,
        kinds,
    )
    assert problems and "age" in problems[0]


def test_batch_frame_keeps_input_positions_and_safe_columns():
    payload = {
        "records": [
            {
                "input_position": 1,
                "record_reference": "risk_input_000001",
                "churn_probability": 0.75,
                "risk_tier": "critical",
                "predicted_label": "Yes",
                "classification_threshold": 0.45,
                "explanation_status": "unavailable",
            },
            {
                "input_position": 0,
                "record_reference": "risk_input_000000",
                "churn_probability": 0.25,
                "risk_tier": "low",
                "predicted_label": "No",
                "classification_threshold": 0.45,
                "explanation_status": "unavailable",
            },
        ]
    }
    frame = assessment._batch_frame(payload)
    assert list(frame.columns) == list(assessment.RESULT_COLUMNS)
    assert list(frame["input_position"]) == [1, 0]
    assert frame["churn_probability"].round(2).tolist() == [0.75, 0.25]


def test_acquisition_fallback_passes_lower_ranked_candidates(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from churnwise.config import Settings

    settings = Settings(project_root=tmp_path)
    candidates_file = settings.report_dir / "acquisition" / "candidates.json"
    candidates_file.parent.mkdir(parents=True)
    candidates_file.write_text("{}", encoding="utf-8")
    candidates = [SimpleNamespace(candidate_id=name) for name in ("one", "two", "three")]
    captured: list[list[str]] = []

    class FakeCoordinator:
        def __init__(self, settings):
            self.settings = settings

        def acquire(self, source, *, offline, origin):
            captured.append([item.candidate_id for item in source])
            return SimpleNamespace(
                succeeded=True,
                failure=None,
                to_dict=lambda: {"status": "success", "cached": False},
            )

    monkeypatch.setattr("churnwise.acquisition.load_candidates", lambda path: candidates)
    monkeypatch.setattr("churnwise.acquisition.AcquisitionCoordinator", FakeCoordinator)
    monkeypatch.setattr(
        "churnwise.acquisition.reporting.write_attempts",
        lambda result, report_dir, offline: settings.report_dir / "acquisition" / "attempts.json",
    )

    actions.acquire_candidate(settings, "two")
    actions.acquire_candidate(settings, "two", allow_fallback=False)
    assert captured == [["two", "three"], ["two"]]


def test_training_configuration_is_validated_before_fitting(tmp_path):
    from churnwise.config import Settings

    settings = Settings(project_root=tmp_path)
    prepared = settings.artifact_dir / "prepared" / "20260101T000000Z-fixture-prep"
    prepared.mkdir(parents=True)
    (prepared / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(actions.DashboardActionError) as caught:
        actions.train_mlp(settings, {"hidden_units": "0,abc"})
    assert caught.value.code == "CW-UI-007"
    assert "hidden units" in caught.value.message

    with pytest.raises(actions.DashboardActionError) as caught:
        actions.train_mlp(settings, {"hidden_units": "8,4", "dropout": "0.2"})
    assert caught.value.code == "CW-UI-007"
    assert "dropout" in caught.value.message
