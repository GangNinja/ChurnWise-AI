"""Artifact resolution rules used by the dashboard."""

from __future__ import annotations

import json
import os

from churnwise.config import Settings
from churnwise.dashboard import data_access


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_latest_validation_ignores_newer_attempt_log(tmp_path):
    settings = Settings(project_root=tmp_path)
    directory = settings.report_dir / "validation"
    report = directory / "openml_46915.json"
    attempts = directory / "attempts.json"
    _write(report, {"dataset_id": "openml:46915", "status": "pass"})
    _write(attempts, {"attempts": []})
    os.utime(attempts, (report.stat().st_atime + 10, report.stat().st_mtime + 10))

    assert data_access.latest_validation_path(settings) == report
    payload, path = data_access.validation_report(settings)
    assert path == report
    assert payload["dataset_id"] == "openml:46915"


def test_latest_validation_is_none_when_only_attempts_exist(tmp_path):
    settings = Settings(project_root=tmp_path)
    _write(settings.report_dir / "validation" / "attempts.json", {"attempts": []})
    assert data_access.latest_validation_path(settings) is None
