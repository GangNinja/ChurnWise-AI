"""Phase 12 end-to-end workflow tests against the committed fixture.

These tests run the complete orchestration with explicit fast settings. The
resulting model metrics are workflow evidence only: the fixture is small and
the fast profile deliberately shortens training.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from churnwise.config import Settings
from churnwise.contracts import LicenseStatus
from churnwise.intelligence.service import RiskService
from churnwise.pipeline import (
    CODE_INVALID_CONFIG,
    CODE_INCOMPATIBLE_RESUME,
    CODE_UPSTREAM_FAILURE,
    PipelineError,
    PipelineRunner,
    build_pipeline_config,
)
from churnwise.preprocessing.coordinator import PreprocessingCoordinator
from churnwise.simulation.service import SimulationService
from tests.helpers import FakeAdapter, make_candidate, make_result
from tests.pipeline_contract_helpers import validate_json_artifact

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_churn.csv"
SEED = 13
RUN_ID = "phase12-fixture-e2e"
SCENARIO_NAME = "Phase 12 baseline"


def _load(path: Path | str, root: Path) -> dict:
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    return json.loads(target.read_text(encoding="utf-8"))


def _artifact(completed: dict, key: str) -> Path:
    target = Path(completed["result"].artifacts[key])
    if not target.is_absolute():
        target = completed["root"] / target
    assert target.is_file(), f"missing artifact {key}: {target}"
    return target


def _stage(manifest: dict, name: str) -> dict:
    return next(item for item in manifest["stages"] if item["name"] == name)


@pytest.fixture(scope="module")
def completed(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("phase12-pipeline")
    settings = Settings(project_root=root, seed=SEED)
    scenario = build_pipeline_config(settings, seed=SEED).scenario
    assert scenario is not None
    scenario = replace(scenario, name=SCENARIO_NAME)
    config = build_pipeline_config(
        settings,
        dataset_path=FIXTURE,
        fast=True,
        seed=SEED,
        run_id=RUN_ID,
        scenario=scenario,
    )
    before = FIXTURE.read_bytes()
    result = PipelineRunner(settings, config).run()
    assert result.succeeded, result.manifest_path.read_text(encoding="utf-8")
    return {
        "root": root,
        "settings": settings,
        "config": config,
        "result": result,
        "before": before,
        "manifest": _load(result.manifest_path, root),
    }


def test_pipeline_runs_every_stage_and_preserves_provenance(completed) -> None:
    manifest = completed["manifest"]
    assert manifest["status"] == "succeeded"
    assert manifest["phase"] == 12
    assert manifest["config_digest"] == completed["result"].config_digest
    assert [item["name"] for item in manifest["stages"]] == [
        "source",
        "preprocess",
        "train",
        "evaluate",
        "explain",
        "score",
        "simulate",
    ]
    assert all(item["status"] == "succeeded" for item in manifest["stages"])
    assert all(item["duration_ms"] is not None for item in manifest["stages"])

    # The committed source is never modified; the managed raw copy is byte-identical.
    assert FIXTURE.read_bytes() == completed["before"]
    source_manifest = _load(_artifact(completed, "source.manifest"), completed["root"])
    raw_path = completed["root"] / "data" / source_manifest["file"]["raw_path"]
    assert raw_path.read_bytes() == completed["before"]
    assert source_manifest["dataset_sha256"] == hashlib.sha256(completed["before"]).hexdigest()
    assert source_manifest["integrity"]["verified"] is True
    assert source_manifest["source"]["license"]["status"] == "project_owned"


def test_target_mapping_and_split_isolation_are_consistent(completed) -> None:
    settings: Settings = completed["settings"]
    manifest = completed["manifest"]
    validation = _load(_artifact(completed, "source.validation_report_json"), completed["root"])
    assert validation["target"]["column"] == "churn"
    assert validation["target"]["mapping"] == {"No": 0, "Yes": 1}
    assert validation["target"]["positive_label"] == "Yes"
    assert validation["gate"]["can_preprocess"] is True

    dataset = PreprocessingCoordinator(settings).load(
        manifest["artifacts"]["preprocessing_run_id"]
    )
    assert dataset.dataset_id == manifest["artifacts"]["dataset_id"]
    assert dataset.schema["target"]["mapping"] == {"No": 0, "Yes": 1}
    assert set(dataset.train.row_indices).isdisjoint(dataset.val.row_indices)
    assert set(dataset.train.row_indices).isdisjoint(dataset.test.row_indices)
    assert set(dataset.val.row_indices).isdisjoint(dataset.test.row_indices)
    assert len(dataset.train.row_indices) == 96
    assert len(dataset.val.row_indices) == 32
    assert len(dataset.test.row_indices) == 32
    assert "churn" not in dataset.feature_names
    assert "customerID" not in dataset.feature_names
    assert all(name not in dataset.as_frame("test").columns for name in ("churn", "customerID"))


def test_model_benchmark_explanations_and_risk_share_the_same_artifacts(completed) -> None:
    root: Path = completed["root"]
    manifest = completed["manifest"]
    ids = manifest["artifacts"]

    metadata = _load(_artifact(completed, "train.metadata"), root)
    assert metadata["preprocessing"]["run_id"] == ids["preprocessing_run_id"]
    assert metadata["preprocessing"]["dataset_id"] == ids["dataset_id"]
    assert metadata["architecture"]["hidden_units"] == [32, 16]
    assert metadata["architecture"]["activation"] == "relu"
    assert metadata["architecture"]["output_activation"] == "sigmoid"
    assert metadata["training"]["optimizer"] == "adam"
    assert metadata["training"]["loss"] == "binary_crossentropy"
    assert metadata["training"]["epochs_completed"] <= metadata["training"]["epochs_requested"]

    train_summary = _stage(manifest, "train")["summary"]
    assert train_summary["hidden_units"] == [32, 16]
    assert train_summary["threshold"] == pytest.approx(0.5)
    assert train_summary["epochs_trained"] >= 1
    assert train_summary["best_epoch"] is not None
    assert train_summary["best_epoch"] <= train_summary["epochs_trained"]

    metrics = _load(_artifact(completed, "evaluate.metrics"), root)
    evaluate_summary = _stage(manifest, "evaluate")["summary"]
    assert evaluate_summary["selected_threshold"] == pytest.approx(
        metrics["selection"]["threshold"]
    )
    assert evaluate_summary["selection_metric"] == metrics["selection"]["selection_metric"]
    assert metrics["preprocessing"]["run_id"] == ids["preprocessing_run_id"]
    assert metrics["preprocessing"]["dataset_id"] == ids["dataset_id"]
    assert metrics["selection"]["test_used_for_selection"] is False
    assert metrics["selection"]["selected_family"] == ids["selected_family"]
    assert metrics["complete"] is True
    assert metrics["families"]["mlp"]["status"] == "completed"

    explain = _load(_artifact(completed, "explain.run_manifest"), root)
    risk = _load(_artifact(completed, "score.run_manifest"), root)
    assert explain["family"] == ids["selected_family"]
    assert explain["preprocessing_run_id"] == ids["preprocessing_run_id"]
    assert explain["model"]["model_sha256"] == risk["model_sha256"]
    assert risk["model_family"] == ids["selected_family"]
    assert risk["preprocessing_run_id"] == ids["preprocessing_run_id"]
    assert risk["benchmark_run_id"] == ids["benchmark_run_id"]

    risk_summary = _load(_artifact(completed, "score.risk_summary"), root)
    assert risk_summary["n_rows"] == 32
    assert risk_summary["classification_threshold"] not in risk_summary["risk_thresholds"]["boundaries"]
    assert risk_summary["explanations"]["available"] > 0
    assert risk_summary["model"]["model_sha256"] == explain["model"]["model_sha256"]


def test_simulation_uses_the_risk_run_and_reconciles_its_arithmetic(completed) -> None:
    result = _load(_artifact(completed, "simulate.result"), completed["root"])
    manifest = completed["manifest"]
    assert result["hypothetical"] is True
    assert result["input"]["risk_run_id"] == manifest["artifacts"]["risk_run_id"]
    assert result["provenance"]["risk_run_id"] == manifest["artifacts"]["risk_run_id"]
    assert result["input"]["model"]["model_sha256"] == _load(
        _artifact(completed, "score.run_manifest"), completed["root"]
    )["model_sha256"]
    assert result["assumptions"]["effectiveness"] == pytest.approx(0.30)
    assert result["assumptions"]["financial_metrics_available"] is False
    assert result["projection"]["observed_retention_result"] is False
    assert result["projection"]["expected_saved_value"] is None
    assert result["allocation"]["counts_reconcile"] is True
    assert result["population"]["counts_reconcile"] is True

    selected = result["selected_customers"]
    selected_probability_mass = sum(item["churn_probability"] for item in selected)
    assert result["projection"]["expected_churn_count"] == pytest.approx(
        selected_probability_mass
    )
    assert result["projection"]["expected_prevented_churn"] == pytest.approx(
        sum(item["expected_prevented_churn"] for item in selected)
    )
    assert result["projection"]["expected_prevented_churn"] == pytest.approx(
        selected_probability_mass * 0.30
    )


def test_resume_reuses_upstream_stages_and_rebuilds_missing_downstream(completed, tmp_path) -> None:
    copied_root = tmp_path / "resumed"
    shutil.copytree(completed["root"], copied_root)
    settings = Settings(project_root=copied_root, seed=SEED)
    config = build_pipeline_config(
        settings,
        dataset_path=FIXTURE,
        fast=True,
        seed=SEED,
        run_id=RUN_ID,
        scenario=completed["config"].scenario,
        resume=True,
    )
    original_manifest = completed["manifest"]
    original_score = _stage(original_manifest, "score")
    original_source = _stage(original_manifest, "source")

    risk_manifest = copied_root / Path(original_manifest["artifacts"]["score.run_manifest"])
    if not risk_manifest.is_absolute():
        risk_manifest = copied_root / risk_manifest
    risk_manifest.unlink()

    resumed = PipelineRunner(settings, config).run()
    assert resumed.succeeded
    resumed_manifest = _load(resumed.manifest_path, copied_root)
    new_source = _stage(resumed_manifest, "source")
    new_score = _stage(resumed_manifest, "score")
    assert new_source["started_at"] == original_source["started_at"]
    assert new_source["duration_ms"] == original_source["duration_ms"]
    assert new_score["started_at"] != original_score["started_at"]
    assert new_score["status"] == "succeeded"
    assert _stage(resumed_manifest, "simulate")["status"] == "succeeded"
    assert FIXTURE.read_bytes() == completed["before"]


def test_resume_rejects_a_changed_configuration(completed) -> None:
    changed = build_pipeline_config(
        completed["settings"],
        dataset_path=FIXTURE,
        fast=True,
        seed=SEED + 1,
        run_id=RUN_ID,
        scenario=completed["config"].scenario,
        resume=True,
    )
    with pytest.raises(PipelineError) as caught:
        PipelineRunner(completed["settings"], changed).run()
    assert caught.value.code == CODE_INCOMPATIBLE_RESUME
    # The existing successful manifest must not be overwritten by a rejected resume.
    manifest = _load(completed["result"].manifest_path, completed["root"])
    assert manifest["status"] == "succeeded"
    assert manifest["config_digest"] == completed["result"].config_digest


def test_rescoring_and_resimulation_are_deterministic(completed) -> None:
    settings: Settings = completed["settings"]
    manifest = completed["manifest"]
    risk_config = completed["config"].risk
    assert risk_config is not None
    service = RiskService(
        settings,
        risk_config,
        benchmark_run_id=manifest["artifacts"]["benchmark_run_id"],
    )
    original = service.load_run(manifest["artifacts"]["risk_run_id"]).table
    second = service.score_population(
        split="val",
        explain="top",
        explain_top_n=10,
        run_id=f"{RUN_ID}-risk-repeat",
        persist=False,
    ).table
    assert [item.record_reference for item in second.records] == [
        item.record_reference for item in original.records
    ]
    assert [item.churn_probability for item in second.records] == pytest.approx(
        [item.churn_probability for item in original.records]
    )
    assert [item.risk_tier for item in second.records] == [
        item.risk_tier for item in original.records
    ]

    scenario = completed["config"].scenario
    assert scenario is not None
    first_simulation = SimulationService(settings).run_scenario(
        scenario,
        table=original,
        scenario_id=f"{RUN_ID}-sim-original",
        include_sensitivity=False,
        persist=False,
    )
    second_simulation = SimulationService(settings).run_scenario(
        scenario,
        table=second,
        scenario_id=f"{RUN_ID}-sim-repeat",
        include_sensitivity=False,
        persist=False,
    )
    assert second_simulation.result.projection.expected_prevented_churn == pytest.approx(
        first_simulation.result.projection.expected_prevented_churn
    )
    assert second_simulation.result.allocation == first_simulation.result.allocation


def test_dashboard_renders_the_pipeline_artifacts(completed, monkeypatch) -> None:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    root: Path = completed["root"]
    monkeypatch.setenv("CHURNWISE_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("CHURNWISE_ARTIFACT_DIR", str(root / "artifacts"))
    monkeypatch.setenv("CHURNWISE_REPORT_DIR", str(root / "reports"))
    monkeypatch.setenv("CHURNWISE_OFFLINE", "true")
    st.cache_data.clear()
    try:
        at = AppTest.from_file(
            str(ROOT / "churnwise" / "dashboard" / "app.py"), default_timeout=120
        ).run()
        assert not at.exception
        at.radio[0].set_value("Retention Workbench").run()
        assert not at.exception
        assert len(at.dataframe[0].value) == 32
        at.radio[0].set_value("Campaign Simulator").run()
        assert not at.exception
        risk_run = next(item for item in at.selectbox if item.label == "Risk run")
        assert risk_run.value == completed["manifest"]["artifacts"]["risk_run_id"]
        body = "\n".join(str(item.value) for item in at.markdown)
        assert "hypothetical" in body.lower()
    finally:
        st.cache_data.clear()


def test_two_identical_runs_with_op_determinism_are_identical(tmp_path) -> None:
    runs = []
    for suffix in ("a", "b"):
        root = tmp_path / suffix
        settings = Settings(project_root=root, seed=SEED)
        config = build_pipeline_config(
            settings,
            dataset_path=FIXTURE,
            fast=True,
            seed=SEED,
            run_id=f"phase12-determinism-{suffix}",
        )
        assert config.mlp is not None
        config = replace(config, mlp=replace(config.mlp, deterministic=True))
        result = PipelineRunner(settings, config).run()
        assert result.succeeded, result.manifest_path.read_text(encoding="utf-8")
        runs.append((settings, config, result))

    (settings_a, config_a, result_a), (settings_b, config_b, result_b) = runs
    assert config_a.mlp.fingerprint() == config_b.mlp.fingerprint()
    assert result_a.config_digest == result_b.config_digest
    metrics_a = _load(result_a.artifacts["evaluate.metrics"], settings_a.project_root)
    metrics_b = _load(result_b.artifacts["evaluate.metrics"], settings_b.project_root)
    assert metrics_a["selection"] == metrics_b["selection"]
    for family in ("mlp", "logistic_regression", "random_forest"):
        assert (
            metrics_a["families"][family]["test_metrics"]
            == metrics_b["families"][family]["test_metrics"]
        )
    split_a = _load(result_a.artifacts["preprocess.split"], settings_a.project_root)
    split_b = _load(result_b.artifacts["preprocess.split"], settings_b.project_root)
    assert split_a["fingerprint"] == split_b["fingerprint"]
    assert split_a["row_indices"] == split_b["row_indices"]


def test_every_json_artifact_has_a_registered_contract(completed) -> None:
    root: Path = completed["root"]
    json_files = sorted(path for path in root.rglob("*.json") if path.is_file())
    assert len(json_files) >= 30, "the fixture run must emit the full JSON artifact set"
    for path in json_files:
        relative = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            pytest.fail(f"invalid JSON artifact {relative}: {exc}")
        validate_json_artifact(relative, payload)


def test_targetless_dataset_fails_at_the_validation_boundary(tmp_path) -> None:
    root = tmp_path / "targetless"
    root.mkdir()
    frame = pd.read_csv(FIXTURE).drop(columns=["churn"])
    bad_dataset = root / "no_target.csv"
    frame.to_csv(bad_dataset, index=False)
    settings = Settings(project_root=root, seed=SEED)
    config = build_pipeline_config(
        settings,
        dataset_path=bad_dataset,
        fast=True,
        seed=SEED,
        run_id="phase12-no-target",
    )
    runner = PipelineRunner(settings, config)
    with pytest.raises(PipelineError) as caught:
        runner.run()
    error = caught.value
    assert error.code == CODE_UPSTREAM_FAILURE
    assert error.stage == "source"
    manifest = _load(runner.manifest_path, root)
    assert manifest["status"] == "failed"
    assert _stage(manifest, "source")["status"] == "failed"
    assert not (root / "artifacts" / "models").exists()


def test_licence_blocked_candidate_stops_before_acquisition(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, seed=SEED)
    candidate = make_candidate(license_status=LicenseStatus.UNVERIFIED)
    adapter = FakeAdapter(make_result("openml", [candidate]))
    config = build_pipeline_config(
        settings,
        fast=True,
        seed=SEED,
        run_id="phase12-licence-block",
        refresh_discovery=True,
    )
    runner = PipelineRunner(settings, config, adapters=[adapter])
    with pytest.raises(PipelineError) as caught:
        runner.run()
    error = caught.value
    assert error.code == CODE_UPSTREAM_FAILURE
    assert error.stage == "source"
    assert "CW-DISC-003" in str(error.details.get("upstream_code"))
    assert adapter.calls == 1
    assert not (tmp_path / "artifacts" / "models").exists()


def test_rejected_dataset_fails_at_the_validation_boundary(tmp_path) -> None:
    root = tmp_path / "rejected"
    root.mkdir()
    frame = pd.read_csv(FIXTURE)
    frame["churn"] = "No"
    bad_dataset = root / "single_class.csv"
    frame.to_csv(bad_dataset, index=False)
    settings = Settings(project_root=root, seed=SEED)
    config = build_pipeline_config(
        settings,
        dataset_path=bad_dataset,
        fast=True,
        seed=SEED,
        run_id="phase12-single-class",
    )
    runner = PipelineRunner(settings, config)
    with pytest.raises(PipelineError) as caught:
        runner.run()
    error = caught.value
    assert error.code == CODE_UPSTREAM_FAILURE
    assert error.stage == "source"
    assert "V4" in error.details.get("blocking_checks", [])

    manifest = _load(runner.manifest_path, root)
    assert manifest["status"] == "failed"
    source = _stage(manifest, "source")
    assert source["status"] == "failed"
    assert source["error"]["code"] == CODE_UPSTREAM_FAILURE
    assert _stage(manifest, "train")["status"] == "pending"
    assert not (root / "artifacts" / "models").exists()


def test_invalid_pipeline_inputs_are_rejected_before_work_starts(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    with pytest.raises(PipelineError) as offline:
        PipelineRunner(settings, build_pipeline_config(settings, offline=True))
    assert offline.value.code == CODE_INVALID_CONFIG

    with pytest.raises(PipelineError) as missing:
        PipelineRunner(
            settings,
            build_pipeline_config(
                settings,
                dataset_path=tmp_path / "does-not-exist.csv",
                run_id="phase12-missing-file",
            ),
        )
    assert missing.value.code == CODE_INVALID_CONFIG

    runner = PipelineRunner(
        settings,
        build_pipeline_config(
            settings,
            dataset_path=FIXTURE,
            fast=True,
            run_id="phase12-missing-run",
            resume=True,
        ),
    )
    with pytest.raises(PipelineError) as absent:
        runner.run()
    assert absent.value.code == CODE_INCOMPATIBLE_RESUME
