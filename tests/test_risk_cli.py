"""CLI tests for the Phase 9 score and risk-export commands."""

from __future__ import annotations

import json

import pytest

from churnwise.cli import main
from tests.risk_helpers import BENCHMARK_ID, build_risk_fixture


def test_score_invalid_tier_configuration_is_structured(settings, capsys) -> None:
    code = main(
        ["score", "--population", "val", "--risk-tiers", "0.5,0.3"],
        settings=settings,
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "CW-RISK-001" in captured.err
    assert "Traceback" not in captured.err


def test_score_without_artifacts_is_structured(settings, capsys) -> None:
    code = main(["score"], settings=settings)
    captured = capsys.readouterr()
    assert code == 2
    assert "CW-XAI-006" in captured.err
    assert "Traceback" not in captured.err


def test_score_population_persists_exports_and_json(tmp_path, capsys) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    code = main(
        [
            "score",
            "--population",
            "val",
            "--benchmark-run-id",
            BENCHMARK_ID,
            "--explain",
            "none",
            "--run-id",
            "20261022T000000Z-cli-risk",
            "--json",
        ],
        settings=settings,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["run_id"] == "20261022T000000Z-cli-risk"
    assert payload["summary"]["population"]["scored"] > 0
    assert payload["export"]["paths"]["csv"]
    assert (settings.artifact_dir / "predictions" / "20261022T000000Z-cli-risk" / "churn_scores.parquet").is_file()


def test_score_input_csv_keeps_row_correspondence(tmp_path, capsys) -> None:
    import pandas as pd

    from churnwise.intelligence import RiskConfig, RiskService
    from tests.models_helpers import synthetic_frame

    settings, _, _ = build_risk_fixture(tmp_path)
    raw = synthetic_frame(rows=9, seed=73).drop(columns=["churn"])
    source = tmp_path / "customers.csv"
    raw.to_csv(source, index=False)
    code = main(
        [
            "score",
            "--input",
            str(source),
            "--benchmark-run-id",
            BENCHMARK_ID,
            "--explain",
            "none",
            "--run-id",
            "20261022T000000Z-cli-input",
        ],
        settings=settings,
    )
    assert code == 0, capsys.readouterr().err
    frame = pd.read_parquet(
        settings.artifact_dir
        / "predictions"
        / "20261022T000000Z-cli-input"
        / "churn_scores.parquet"
    ).sort_values("input_position")
    expected = RiskService(
        settings, RiskConfig(), benchmark_run_id=BENCHMARK_ID
    ).assess_batch(raw, explain="none")
    assert frame["input_position"].tolist() == list(range(len(raw)))
    assert frame["churn_probability"].tolist() == pytest.approx(
        [item.churn_probability for item in expected.records], abs=1e-12
    )


def test_score_input_blank_reference_column_uses_positions(tmp_path, capsys) -> None:
    import pandas as pd

    from tests.models_helpers import synthetic_frame

    settings, _, _ = build_risk_fixture(tmp_path)
    raw = synthetic_frame(rows=5, seed=97).drop(columns=["churn"])
    raw["customer_id"] = ""
    source = tmp_path / "blank_ids.csv"
    raw.to_csv(source, index=False)
    code = main(
        [
            "score",
            "--input",
            str(source),
            "--benchmark-run-id",
            BENCHMARK_ID,
            "--reference-column",
            "customer_id",
            "--explain",
            "none",
            "--run-id",
            "20261022T000000Z-cli-blank-reference",
            "--json",
        ],
        settings=settings,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["summary"]["population"]["scored"] == 5
    frame = pd.read_parquet(
        settings.artifact_dir
        / "predictions"
        / "20261022T000000Z-cli-blank-reference"
        / "churn_scores.parquet"
    )
    references = frame["record_reference"].tolist()
    assert all(reference.startswith("risk_input_") for reference in references)
    assert len(set(references)) == len(references)


def test_risk_export_filters_and_reports_rows(tmp_path, capsys) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    first = main(
        [
            "score",
            "--population",
            "val",
            "--benchmark-run-id",
            BENCHMARK_ID,
            "--explain",
            "none",
            "--run-id",
            "20261022T000000Z-cli-export",
        ],
        settings=settings,
    )
    assert first == 0
    capsys.readouterr()
    code = main(
        [
            "risk-export",
            "--run-id",
            "20261022T000000Z-cli-export",
            "--tier",
            "high",
            "--tier",
            "critical",
            "--limit",
            "10",
            "--format",
            "csv",
            "--json",
        ],
        settings=settings,
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["page"]["rows"] <= 10
    assert payload["export"]["row_count"] == payload["page"]["rows"]
    assert all(
        item["risk_tier"] in {"high", "critical"}
        for item in payload["page"]["records"]
    )
