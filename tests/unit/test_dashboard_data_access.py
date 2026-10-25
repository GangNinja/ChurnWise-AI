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


def test_population_risk_run_is_preferred_over_a_newer_batch(tmp_path):
    settings = Settings(project_root=tmp_path)
    root = settings.artifact_dir / "predictions"
    population = root / "20260101T010000Z-population-risk"
    batch = root / "20260101T020000Z-batch-risk"
    _write(
        population / "risk_summary.json",
        {"run_id": population.name, "n_rows": 1000, "population": {"split": "val"}},
    )
    _write(
        batch / "risk_summary.json",
        {"run_id": batch.name, "n_rows": 6, "population": {"split": "input"}},
    )

    assert data_access.latest_risk_dir(settings) == batch
    assert data_access.latest_population_risk_dir(settings) == population
    assert data_access.default_risk_run_id(settings) == population.name
    payload, path = data_access.risk_summary(settings, prefer_population=True)
    assert path == population / "risk_summary.json"
    assert payload["n_rows"] == 1000


def test_local_explanation_is_scoped_to_family_and_returns_same_run_plot(tmp_path):
    settings = Settings(project_root=tmp_path)
    root = settings.artifact_dir / "explanations"
    older = root / "20260101T000000Z-rf-xai"
    newer = root / "20260102T000000Z-mlp-xai"
    for run in (older, newer):
        _write(run / "global.json", {"run_id": run.name})
        _write(run / "run_manifest.json", {"run_id": run.name})
    _write(
        older / "local" / "row_00007.json",
        {
            "family": "random_forest",
            "row_position": 7,
            "probability": 0.75,
            "summary": "matching family",
        },
    )
    (older / "local" / "row_00007.png").write_bytes(b"png")
    _write(
        newer / "local" / "row_00007.json",
        {
            "family": "mlp",
            "row_position": 7,
            "probability": 0.75,
            "summary": "different model family",
        },
    )
    (newer / "local" / "row_00007.png").write_bytes(b"png")

    payload, path, plot = data_access.local_explanation(settings, 7, family="random_forest")

    assert payload is not None and payload["summary"] == "matching family"
    assert path == older / "local" / "row_00007.json"
    assert plot == older / "local" / "row_00007.png"


def test_local_explanation_returns_nothing_when_no_family_matches(tmp_path):
    settings = Settings(project_root=tmp_path)
    run = settings.artifact_dir / "explanations" / "20260101T000000Z-mlp-xai"
    _write(run / "global.json", {"run_id": run.name})
    _write(run / "local" / "row_00001.json", {"family": "mlp", "row_position": 1})

    assert data_access.local_explanation(settings, 1, family="random_forest") == (
        None,
        None,
        None,
    )


def _write_customer_fixture(tmp_path):
    """Minimal provenance chain for a verified score-to-source join."""

    import pandas as pd

    settings = Settings(project_root=tmp_path)
    digest = "a" * 64
    fingerprint = "fixture-fingerprint"
    run_id = "20260101T000000Z-fixture-population"
    run = settings.artifact_dir / "predictions" / run_id
    run.mkdir(parents=True)

    _write(
        run / "risk_summary.json",
        {
            "run_id": run_id,
            "n_rows": 2,
            "population": {"split": "val"},
            "classification_threshold": 0.45,
            "model": {"preprocessing_run_id": "fixture-prep"},
            "provenance": {"dataset_sha256": digest, "split_fingerprint": fingerprint},
        },
    )
    pd.DataFrame(
        {
            "record_reference": ["risk_val_00000", "risk_val_00001"],
            "input_position": [0, 1],
            "churn_probability": [0.10, 0.90],
            "predicted_label": ["retained", "Yes"],
            "risk_tier": ["low", "critical"],
            "observed_churn": [0, 0],
        }
    ).to_parquet(run / "churn_scores.parquet", index=False)
    _write(
        settings.artifact_dir / "splits" / "fixture-prep.json",
        {
            "fingerprint": fingerprint,
            "dataset_sha256": digest,
            "row_indices": {"train": [1, 3], "val": [0, 2], "test": []},
        },
    )
    raw = settings.data_dir / "cache" / "sha256" / digest
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "@RELATION fixture\n"
        "@ATTRIBUTE account_length NUMERIC\n"
        "@ATTRIBUTE international_plan NUMERIC\n"
        "@ATTRIBUTE CustomerChurned {No,Yes}\n"
        "@DATA\n"
        "10,0,No\n"
        "20,1,Yes\n"
        "30,0,No\n"
        "40,1,Yes\n",
        encoding="utf-8",
    )
    _write(
        settings.report_dir / "validation" / "fixture.json",
        {
            "dataset_id": "fixture",
            "target": {
                "column": "CustomerChurned",
                "mapping": {"No": 0, "Yes": 1},
                "positive_rate": 0.5,
            },
            "summary": {
                "n_rows": 4,
                "usable_features": ["account_length", "international_plan"],
            },
        },
    )
    return settings, run_id, digest, fingerprint


def test_customer_population_joins_only_through_a_verified_split(tmp_path):
    settings, run_id, digest, fingerprint = _write_customer_fixture(tmp_path)

    population = data_access.customer_population(settings, run_id)

    assert population.frame is not None, population.reason
    assert population.reason.startswith("Verified 2-row join")
    assert population.split == "val"
    assert population.split_fingerprint == fingerprint
    assert list(population.frame["record_reference"]) == ["risk_val_00000", "risk_val_00001"]
    assert list(population.frame["account_length"]) == [10.0, 30.0]
    assert "CustomerChurned" not in population.raw_columns
    assert set(population.raw_columns) == {"account_length", "international_plan"}


def test_customer_population_refuses_a_mismatched_fingerprint(tmp_path):
    settings, run_id, _, _ = _write_customer_fixture(tmp_path)
    path = settings.artifact_dir / "splits" / "fixture-prep.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fingerprint"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")

    population = data_access.customer_population(settings, run_id)

    assert population.frame is None
    assert "fingerprint does not match" in population.reason
