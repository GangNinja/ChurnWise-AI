from __future__ import annotations

import json
from pathlib import Path

import pytest

from churnwise.config import Settings
from churnwise.pipeline import (
    CODE_INCOMPATIBLE_RESUME,
    CODE_VERIFICATION_FAILED,
    PIPELINE_SCHEMA_VERSION,
    STAGE_NAMES,
    PipelineError,
    PipelineRunner,
    build_pipeline_config,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_churn.csv"
RUN_ID = "phase12-resume-validation"


def _runner(tmp_path: Path, *, seed: int = 13) -> PipelineRunner:
    settings = Settings(project_root=tmp_path, seed=seed)
    config = build_pipeline_config(
        settings,
        dataset_path=FIXTURE,
        fast=True,
        seed=seed,
        run_id=RUN_ID,
        resume=True,
    )
    return PipelineRunner(settings, config)


def _stages() -> list[dict]:
    return [
        {"name": name, "status": "pending", "artifacts": {}, "outputs": {}}
        for name in STAGE_NAMES
    ]


def _manifest(runner: PipelineRunner, **overrides) -> dict:
    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "phase": 12,
        "run_id": runner.run_id,
        "status": "running",
        "config_digest": runner.config_digest,
        "stages": _stages(),
        "artifacts": {},
    }
    payload.update(overrides)
    return payload


def _write(runner: PipelineRunner, payload: object) -> None:
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _expect_error(runner: PipelineRunner) -> PipelineError:
    with pytest.raises(PipelineError) as caught:
        runner.run()
    return caught.value


def test_non_object_manifest_is_refused(tmp_path) -> None:
    runner = _runner(tmp_path)
    _write(runner, [])
    error = _expect_error(runner)
    assert error.code == CODE_INCOMPATIBLE_RESUME
    assert "JSON object" in error.message


def test_malformed_stage_artifact_map_is_refused(tmp_path) -> None:
    runner = _runner(tmp_path)
    stages = _stages()
    stages[0]["artifacts"] = []
    _write(runner, _manifest(runner, stages=stages))
    error = _expect_error(runner)
    assert error.code == CODE_INCOMPATIBLE_RESUME
    assert "artifact map" in error.message


def test_success_manifest_with_pending_stages_is_refused(tmp_path) -> None:
    runner = _runner(tmp_path)
    _write(runner, _manifest(runner, status="succeeded"))
    error = _expect_error(runner)
    assert error.code == CODE_INCOMPATIBLE_RESUME
    assert "claims success" in error.message


def test_succeeded_stage_without_outputs_is_refused(tmp_path) -> None:
    runner = _runner(tmp_path)
    stages = _stages()
    stages[0]["status"] = "succeeded"
    stages[0]["artifacts"] = {"manifest": str(tmp_path / "manifest.json")}
    _write(runner, _manifest(runner, stages=stages))
    error = _expect_error(runner)
    assert error.code == CODE_INCOMPATIBLE_RESUME
    assert "incomplete" in error.message


def test_unknown_stage_status_is_refused(tmp_path) -> None:
    runner = _runner(tmp_path)
    stages = _stages()
    stages[1]["status"] = "skipped"
    _write(runner, _manifest(runner, stages=stages))
    error = _expect_error(runner)
    assert error.code == CODE_INCOMPATIBLE_RESUME
    assert "invalid status" in error.message


def test_output_verification_includes_source_and_preprocess(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner._records = {
        name: {"status": "succeeded", "artifacts": {}} for name in STAGE_NAMES
    }
    runner._verify_outputs()
    runner._records["source"]["artifacts"] = {
        "missing": str(tmp_path / "missing-source.json")
    }
    with pytest.raises(PipelineError) as caught:
        runner._verify_outputs()
    assert caught.value.code == CODE_VERIFICATION_FAILED
    assert "source.missing" in caught.value.message


def test_runner_uses_the_config_seed_for_phase_services(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, seed=42)
    config = build_pipeline_config(
        settings,
        dataset_path=FIXTURE,
        fast=True,
        seed=7,
        run_id=RUN_ID,
    )
    runner = PipelineRunner(settings, config)
    assert runner.settings.seed == 7
    assert runner.config.seed == 7
    assert runner._config_payload()["seed"] == 7
