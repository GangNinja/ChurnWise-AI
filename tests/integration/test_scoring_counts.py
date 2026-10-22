"""Phase 9 integration tests against a genuinely fitted synthetic model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnwise.cli import main
from churnwise.intelligence import (
    RiskConfig,
    RiskError,
    RiskFilter,
    RiskService,
    RiskThresholds,
)
from tests.models_helpers import synthetic_frame
from tests.risk_helpers import BENCHMARK_ID, build_risk_fixture


def _service(settings, *, max_driver_rows: int = 100, **overrides) -> RiskService:
    config = RiskConfig(
        thresholds=RiskThresholds.from_value((0.30, 0.50, 0.70)),
        top_k=3,
        batch_size=5,
        max_driver_rows=max_driver_rows,
        **overrides,
    )
    return RiskService(settings, config, benchmark_run_id=BENCHMARK_ID)


def test_population_scores_every_row_with_unique_ranks_and_tier_counts(tmp_path) -> None:
    settings, dataset, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(split="val", explain="none", run_id="20261022T000000Z-risk-fixture")

    records = run.table.records
    assert len(records) == len(dataset.val.X)
    assert all(0.0 <= item.churn_probability <= 1.0 for item in records)
    assert [item.rank for item in records] == list(range(1, len(records) + 1))
    assert len({item.rank for item in records}) == len(records)
    counts = run.table.tier_counts()
    assert sum(value["count"] for value in counts.values()) == len(records)
    assert sum(value["pct"] for value in counts.values()) == pytest.approx(1.0)
    assert run.summary["population"]["scored"] == len(records)
    assert run.summary["population"]["rejected"] == 0
    assert path_exists(run.paths["churn_scores"])
    assert path_exists(run.paths["risk_summary"])
    assert path_exists(run.paths["tiers"])

    loaded = service.load_run(run.run_id)
    assert len(loaded.table.records) == len(records)
    assert loaded.table.records[0].record_reference == records[0].record_reference
    assert loaded.table.records[0].model_sha256 == records[0].model_sha256


def path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()


def test_individual_and_batch_use_the_same_probability_path(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=12, seed=17).drop(columns=["churn"])
    batch = service.assess_batch(raw, batch_size=5)
    assert batch.total_received == 12
    assert batch.scored == 12
    assert batch.rejected == 0
    assert batch.counts_reconcile
    assert [item.input_position for item in batch.records] == list(range(12))
    assert [item.record_reference for item in batch.records] == [
        f"risk_input_{index:06d}" for index in range(12)
    ]
    for item in batch.records:
        assert item.predicted_class == int(item.churn_probability >= 0.45)

    single = service.assess_customer(raw.iloc[3].to_dict())
    same = batch.records[3]
    assert single.churn_probability == pytest.approx(same.churn_probability)
    assert single.risk_tier == same.risk_tier
    assert single.predicted_class == same.predicted_class


def test_batch_size_does_not_change_probabilities(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=17, seed=23).drop(columns=["churn"])
    small = service.assess_batch(raw, batch_size=3)
    large = service.assess_batch(raw, batch_size=100)
    assert [item.churn_probability for item in small.records] == pytest.approx(
        [item.churn_probability for item in large.records], abs=1e-12
    )


def test_partial_batch_failure_isolates_the_bad_record(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=6, seed=29).drop(columns=["churn"]).reset_index(drop=True)
    raw.at[2, "contract"] = {"not": "a scalar"}
    batch = service.assess_batch(raw, batch_size=6)
    assert batch.total_received == 6
    assert batch.scored == 5
    assert batch.rejected == 1
    assert batch.counts_reconcile
    assert batch.rejections[0].input_position == 2
    assert "unsupported value type" in batch.rejections[0].message
    assert [item.input_position for item in batch.records] == [0, 1, 3, 4, 5]
    with pytest.raises(RiskError):
        service.assess_customer(raw.iloc[2].to_dict())


def test_empty_and_missing_schema_batches_are_structured(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    empty = service.assess_batch(pd.DataFrame(columns=["tenure_months"]))
    assert empty.total_received == 0
    assert empty.scored == 0
    assert empty.rejected == 0
    assert "empty batch" in empty.warnings[0]

    raw = synthetic_frame(rows=4, seed=31).drop(columns=["churn", "contract"])
    missing = service.assess_batch(raw)
    assert missing.total_received == 4
    assert missing.scored == 0
    assert missing.rejected == 4
    assert missing.counts_reconcile
    assert "contract" in missing.batch_error


def test_unexpected_columns_are_reported_and_ignored(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=3, seed=57).drop(columns=["churn"])
    raw["not_a_model_feature"] = ["a", "b", "c"]
    batch = service.assess_batch(raw)
    assert batch.scored == 3
    assert any(
        "not used by the model" in warning and "not_a_model_feature" in warning
        for warning in batch.warnings
    )


def test_missing_values_reuse_phase5_handling_and_are_reported(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=3, seed=59).drop(columns=["churn"])
    raw.loc[0, "monthly_charges"] = np.nan
    batch = service.assess_batch(raw)
    assert batch.scored == 3
    assert any("missing value" in warning for warning in batch.records[0].warnings)
    assert all(0.0 <= item.churn_probability <= 1.0 for item in batch.records)


def test_duplicate_references_remain_distinct(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=4, seed=37).drop(columns=["churn"])
    raw["customer_id"] = ["same", "other", "same", "third"]
    batch = service.assess_batch(raw, reference_column="customer_id")
    references = [item.record_reference for item in batch.records]
    assert len(set(references)) == 4
    assert any(reference.endswith("_dup2") for reference in references)
    assert any("duplicate" in warning.lower() for warning in batch.warnings)


def test_selected_explanation_subset_is_position_bounded(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings, max_driver_rows=100)
    raw = synthetic_frame(rows=12, seed=79).drop(columns=["churn"])
    batch = service.assess_batch(
        raw,
        explain="selected",
        explain_positions=(1, 7),
        batch_size=4,
    )
    available = {
        item.input_position
        for item in batch.records
        if item.explanation_status == "available"
    }
    assert available == {1, 7}
    assert all(
        "selected explanation subset" in item.explanation_reason
        for item in batch.records
        if item.input_position not in available
    )
    with pytest.raises(RiskError):
        service.assess_batch(
            raw,
            explain="selected",
            explain_positions=(99,),
        )
    with pytest.raises(RiskError):
        service.assess_batch(raw, explain="selected")


def test_population_top_mode_uses_row_count_not_feature_top_k(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings, max_driver_rows=100)
    run = service.score_population(
        split="val",
        explain="top",
        explain_top_n=3,
        run_id="20261022T000000Z-risk-population-top",
    )
    available = [
        item for item in run.table.records if item.explanation_status == "available"
    ]
    assert len(available) == 3
    assert all(
        len(item.top_positive) <= 3 for item in available
    )


def test_explanations_are_attached_and_never_claim_causality(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings, max_driver_rows=100)
    run = service.score_population(
        split="val", explain="all", run_id="20261022T000000Z-risk-explain"
    )
    assert all(item.explanation_status == "available" for item in run.table.records)
    assert all(item.explanation_summary for item in run.table.records)
    assert all(
        "causal" in item.explanation_summary.lower()
        or "not guaranteed" in item.explanation_summary.lower()
        for item in run.table.records
    )
    assert path_exists(run.paths["drivers"])
    explanation = service.get_explanation(run.run_id, run.table.records[0].record_reference)
    assert explanation["explanation_status"] == "available"
    assert explanation["top_positive"] or explanation["top_negative"]
    assert any("not causal" in line.lower() for line in explanation["limitations"])


def test_top_explanation_mode_marks_unselected_rows_unavailable(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings, max_driver_rows=100)
    batch = service.assess_batch(
        synthetic_frame(rows=12, seed=41).drop(columns=["churn"]),
        explain="top",
        explain_top_n=3,
        batch_size=4,
    )
    available = [item for item in batch.records if item.explanation_status == "available"]
    assert len(available) == 3
    assert all(
        "highest-risk" in item.explanation_reason
        for item in batch.records
        if item.explanation_status == "unavailable"
    )


def test_missing_explanation_is_explicit_not_fabricated(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    batch = service.assess_batch(
        synthetic_frame(rows=4, seed=43).drop(columns=["churn"]),
        explain="all",
        explain_run_id="20261022T000000Z-does-not-exist",
    )
    assert batch.scored == 4
    assert all(item.explanation_status == "unavailable" for item in batch.records)
    assert all(item.explanation_reason for item in batch.records)
    assert any("unavailable" in warning.lower() for warning in batch.warnings)


def test_export_round_trip_and_privacy(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=10, seed=47).drop(columns=["churn"])
    raw["customer_id"] = [f"raw-{index}@example.com" for index in range(len(raw))]
    raw["note"] = "=1+1"
    raw["contract"] = "monthly"
    run = service.score_frame(
        raw,
        explain="none",
        reference_column="customer_id",
        segment_fields=("contract", "note"),
        run_id="20261022T000000Z-risk-export",
    )
    stored = service.ranked(
        run_id=run.run_id,
        filters=RiskFilter(segment_equals={"contract": ("monthly",)}),
        limit=100,
    )
    assert len(stored.records) == len(raw)
    exported = service.export(
        run.run_id,
        formats=("csv", "json", "parquet"),
        include_segment=True,
    )
    assert exported.row_count == len(raw)
    csv_frame = pd.read_csv(exported.paths["csv"])
    parquet_frame = pd.read_parquet(exported.paths["parquet"])
    payload = json.loads(open(exported.paths["json"], encoding="utf-8").read())
    assert len(csv_frame) == len(parquet_frame) == payload["row_count"] == len(raw)
    assert "churn_probability" in csv_frame.columns
    assert csv_frame["churn_probability"].between(0, 1).all()
    assert pd.api.types.is_integer_dtype(parquet_frame["input_position"])
    assert pd.api.types.is_integer_dtype(parquet_frame["predicted_class"])
    assert pd.api.types.is_float_dtype(parquet_frame["churn_probability"])
    assert "customer_id" not in csv_frame.columns
    assert "raw-" not in open(exported.paths["csv"], encoding="utf-8").read()
    assert payload["records"][0]["record_reference"].startswith("cust_")


def test_all_rejected_batch_persists_counts_and_rejection_details(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=3, seed=61).drop(columns=["churn", "contract"])
    run = service.score_frame(
        raw,
        explain="none",
        run_id="20261022T000000Z-risk-rejections",
    )
    assert run.total_received == 3
    assert run.table.records == []
    assert len(run.rejections) == 3
    assert run.summary["population"] == {
        "split": "input",
        "received": 3,
        "scored": 0,
        "rejected": 3,
    }
    assert run.paths["churn_scores"]
    assert path_exists(run.paths["rejections"])
    loaded = service.load_run(run.run_id)
    assert loaded.table.records == []
    assert loaded.total_received == 3
    assert len(loaded.rejections) == 3

    exported = service.export(run.run_id, formats=("csv",))
    csv_frame = pd.read_csv(exported.paths["csv"])
    assert len(csv_frame) == 0
    assert "risk_thresholds" in csv_frame.columns
    assert "segment" in csv_frame.columns

    Path(run.paths["rejections"]).unlink()
    with pytest.raises(RiskError) as raised:
        service.load_run(run.run_id)
    assert raised.value.code == "CW-RISK-003"
    assert "rejection details" in raised.value.message


def test_invalid_value_proxy_is_reported_without_raw_value(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=2, seed=67).drop(columns=["churn"])
    raw["monthly_value"] = [10.0, "private-not-a-number"]
    with pytest.raises(RiskError) as raised:
        service.score_frame(raw, value_column="monthly_value", explain="none")
    assert raised.value.code == "CW-RISK-004"
    assert "input position 1" in raised.value.message
    assert "private-not-a-number" not in raised.value.message


def test_export_cannot_escape_the_predictions_directory(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(split="val", explain="none", run_id="20261022T000000Z-risk-path")
    outside = tmp_path / "outside"
    with pytest.raises(RiskError) as raised:
        service.export(run.run_id, output_dir=outside)
    assert raised.value.code == "CW-RISK-007"
    assert "inside this risk run" in raised.value.message

    other_run = settings.artifact_dir / "predictions" / "another-run"
    with pytest.raises(RiskError):
        service.export(run.run_id, output_dir=other_run)


def test_stale_model_artifact_is_rejected_on_reload(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(split="val", explain="none", run_id="20261022T000000Z-risk-stale")
    model_path = run.manifest["model_path"]
    with open(model_path, "ab") as handle:
        handle.write(b"stale")
    with pytest.raises(RiskError) as raised:
        service.load_run(run.run_id)
    assert raised.value.code == "CW-RISK-003"
    assert "stale" in raised.value.message.lower()


def test_mismatched_model_and_preprocessing_are_rejected(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    metadata_path = (
        settings.artifact_dir
        / "evaluation"
        / BENCHMARK_ID
        / "models"
        / "random_forest"
        / "metadata.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["preprocessing"]["split_fingerprint"] = "0" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    service = RiskService(
        settings,
        RiskConfig(),
        benchmark_run_id=BENCHMARK_ID,
    )
    with pytest.raises(RiskError) as raised:
        _ = service.model
    assert "split_fingerprint" in raised.value.message


def test_missing_model_artifacts_are_actionable(tmp_path) -> None:
    from churnwise.config import Settings

    service = RiskService(Settings(project_root=tmp_path))
    with pytest.raises(RiskError) as raised:
        _ = service.model
    assert raised.value.code
    assert raised.value.remediation


def test_ranked_filters_and_value_at_risk_are_consistent(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=9, seed=53).drop(columns=["churn"])
    raw["monthly_value"] = np.arange(50, 50 + len(raw) * 10, 10, dtype=float)
    run = service.score_frame(
        raw,
        explain="none",
        value_column="monthly_value",
        run_id="20261022T000000Z-risk-value",
    )
    high = service.ranked(
        run_id=run.run_id,
        filters=RiskFilter(tiers=("high", "critical")),
        limit=100,
    )
    assert all(item.risk_tier in {"high", "critical"} for item in high.records)
    assert [item.rank for item in high.records] == list(range(1, len(high.records) + 1))
    by_value = service.ranked(
        run_id=run.run_id,
        sort_by="value_at_risk",
        limit=100,
    )
    losses = [item.value_at_risk for item in by_value.records]
    assert all(loss is not None for loss in losses)
    assert losses == sorted(losses, reverse=True)
    assert run.summary["value_at_risk"]["available"] is True
    assert run.summary["value_at_risk"]["column"] == "monthly_value"
    assert run.manifest["value_proxy_column"] == "monthly_value"


def test_blank_reference_column_falls_back_to_positional_references(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=4, seed=83).drop(columns=["churn"])
    raw["customer_id"] = np.nan
    batch = service.assess_batch(raw, reference_column="customer_id")
    assert batch.scored == 4
    assert batch.rejected == 0
    assert [item.record_reference for item in batch.records] == [
        f"risk_input_{index:06d}" for index in range(4)
    ]
    assert any("customer_id" in warning for warning in batch.warnings)
    assert any("empty" in warning for warning in batch.warnings)


def test_partial_missing_references_use_positions_without_collisions(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=4, seed=85).drop(columns=["churn"])
    raw["customer_id"] = ["alpha", np.nan, "alpha", "beta"]
    batch = service.assess_batch(raw, reference_column="customer_id")
    references = [item.record_reference for item in batch.records]
    assert batch.scored == 4
    assert len(set(references)) == 4
    assert references[1] == "risk_input_000001"
    assert references[0].startswith("cust_")
    assert references[2].endswith("_dup2")
    assert any("1 missing value" in warning for warning in batch.warnings)


def test_absent_reference_column_is_reported_not_silently_ignored(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=3, seed=87).drop(columns=["churn"])
    batch = service.assess_batch(raw, reference_column="not_a_column")
    assert batch.scored == 3
    assert all(item.record_reference.startswith("risk_input_") for item in batch.records)
    assert any("not present" in warning for warning in batch.warnings)


def test_ignored_column_missing_value_is_not_attributed_to_the_pipeline(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=3, seed=89).drop(columns=["churn"])
    raw["note"] = np.nan
    batch = service.assess_batch(raw, segment_fields=("note",))
    assert batch.scored == 3
    assert not any(
        "missing value" in warning
        for item in batch.records
        for warning in item.warnings
    )


def test_malformed_scored_table_is_a_structured_artifact_error(tmp_path, capsys) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(
        split="val", explain="none", run_id="20261022T000000Z-risk-bad-columns"
    )
    path = Path(run.paths["churn_scores"])
    frame = pd.read_parquet(path).drop(columns=["risk_tier"])
    frame.to_parquet(path, index=False)
    with pytest.raises(RiskError) as raised:
        service.load_run(run.run_id)
    assert raised.value.code == "CW-RISK-003"
    assert "missing required column" in raised.value.message

    code = main(["risk-export", "--run-id", run.run_id], settings=settings)
    captured = capsys.readouterr()
    assert code == 2
    assert "CW-RISK-003" in captured.err
    assert "Traceback" not in captured.err


def test_stored_row_count_mismatch_is_rejected(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(
        split="val", explain="none", run_id="20261022T000000Z-risk-bad-count"
    )
    path = Path(run.paths["churn_scores"])
    pd.read_parquet(path).head(3).to_parquet(path, index=False)
    with pytest.raises(RiskError) as raised:
        service.load_run(run.run_id)
    assert raised.value.code == "CW-RISK-003"
    assert "row(s)" in raised.value.message


def test_changed_preprocessor_artifact_is_rejected_on_reload(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(
        split="val", explain="none", run_id="20261022T000000Z-risk-stale-preprocessor"
    )
    preprocessor = Path(run.manifest["preprocessor_path"])
    with preprocessor.open("ab") as handle:
        handle.write(b"stale")
    with pytest.raises(RiskError) as raised:
        service.load_run(run.run_id)
    assert raised.value.code == "CW-RISK-003"
    assert "stale" in raised.value.message.lower()


def test_malformed_explanation_file_keeps_scores_valid(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=4, seed=91).drop(columns=["churn"])
    service.assess_batch(raw, explain="all", batch_size=4)
    explain_root = Path(settings.artifact_dir) / "explanations"
    run_dirs = sorted(path for path in explain_root.iterdir() if path.is_dir())
    local = sorted((run_dirs[-1] / "local").glob("*.json"))[0]
    payload = json.loads(local.read_text(encoding="utf-8"))
    payload["top_positive"] = [{"feature": "x", "contribution": "not-a-number"}]
    local.write_text(json.dumps(payload), encoding="utf-8")

    batch = service.assess_batch(
        raw, explain="all", explain_run_id=run_dirs[-1].name
    )
    assert batch.scored == 4
    assert all(it.churn_probability >= 0.0 for it in batch.records)
    assert all(item.explanation_status == "unavailable" for item in batch.records)
    assert all(item.explanation_reason for item in batch.records)
    assert any("malformed" in warning for warning in batch.warnings)


def test_invalid_explain_run_id_keeps_scores_valid(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    raw = synthetic_frame(rows=3, seed=93).drop(columns=["churn"])
    batch = service.assess_batch(raw, explain="all", explain_run_id="../escape")
    assert batch.scored == 3
    assert all(item.explanation_status == "unavailable" for item in batch.records)
    assert any("invalid explanation run id" in warning for warning in batch.warnings)


def test_json_export_keeps_stable_schema_and_records_applied_filters(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings)
    run = service.score_population(
        split="val", explain="none", run_id="20261022T000000Z-risk-json-schema"
    )
    reference = run.table.records[0].record_reference
    exported = service.export(
        run.run_id,
        formats=("json", "csv"),
        filters=RiskFilter(record_references=(reference,)),
        limit=10,
    )
    payload = json.loads(Path(exported.paths["json"]).read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    record = payload["records"][0]
    assert "segment" in record and record["segment"] == {}
    assert payload["filters"]["record_references"] == [reference]
    assert payload["sort_by"] == "probability"
    assert payload["limit"] == 10
    csv_frame = pd.read_csv(exported.paths["csv"])
    assert "segment" in csv_frame.columns
    assert csv_frame["segment"].tolist() == ["{}"]


def test_classification_threshold_override_does_not_move_tiers(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    default = _service(settings)
    override = RiskService(
        settings,
        RiskConfig(
            thresholds=RiskThresholds.from_value((0.30, 0.50, 0.70)),
            classification_threshold=0.0,
            top_k=3,
            batch_size=5,
            max_driver_rows=100,
        ),
        benchmark_run_id=BENCHMARK_ID,
    )
    raw = synthetic_frame(rows=8, seed=105).drop(columns=["churn"])
    base = default.assess_batch(raw)
    changed = override.assess_batch(raw)
    assert [item.churn_probability for item in changed.records] == pytest.approx(
        [item.churn_probability for item in base.records], abs=1e-12
    )
    assert [item.risk_tier for item in changed.records] == [
        item.risk_tier for item in base.records
    ]
    assert all(item.predicted_class == 1 for item in changed.records)
    assert all(item.classification_threshold == 0.0 for item in changed.records)


def test_driver_cap_marks_excess_rows_unavailable(tmp_path) -> None:
    settings, _, _ = build_risk_fixture(tmp_path)
    service = _service(settings, max_driver_rows=2)
    run = service.score_population(
        split="val", explain="all", run_id="20261022T000000Z-risk-driver-cap"
    )
    available = [
        item for item in run.table.records if item.explanation_status == "available"
    ]
    assert len(available) == 2
    highest = sorted(
        run.table.records,
        key=lambda item: (-item.churn_probability, item.record_reference),
    )[:2]
    assert {item.record_reference for item in available} == {
        item.record_reference for item in highest
    }
    assert all(
        "highest-risk" in item.explanation_reason
        for item in run.table.records
        if item.explanation_status == "unavailable"
    )
    assert any("max_driver_rows=2" in warning for warning in run.warnings)
