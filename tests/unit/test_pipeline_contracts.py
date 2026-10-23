from __future__ import annotations

import ast
from pathlib import Path

import pytest

from churnwise.config import Settings
from churnwise.pipeline import (
    CODE_INVALID_CONFIG,
    PipelineError,
    PipelineRunner,
    build_pipeline_config,
)
from churnwise.pipeline.errors import PIPELINE_CODES


def test_pipeline_error_is_structured_and_serialisable() -> None:
    error = PipelineError(
        CODE_INVALID_CONFIG,
        "bad input",
        "fix it",
        stage="source",
        details={"problem": "x"},
        artifacts={"report": "reports/x.json"},
    )
    payload = error.to_dict()
    assert payload["code"] == CODE_INVALID_CONFIG
    assert payload["stage"] == "source"
    assert payload["details"] == {"problem": "x"}
    assert payload["artifacts"] == {"report": "reports/x.json"}
    assert CODE_INVALID_CONFIG in PIPELINE_CODES


def test_configuration_digest_is_stable_and_changes_with_semantics(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    first = PipelineRunner(settings, build_pipeline_config(settings, seed=3, run_id="phase12-a"))
    second = PipelineRunner(settings, build_pipeline_config(settings, seed=3, run_id="phase12-b"))
    changed = PipelineRunner(settings, build_pipeline_config(settings, seed=4, run_id="phase12-c"))
    assert first.config_digest == second.config_digest
    assert first.config_digest != changed.config_digest


def test_fast_profile_is_explicit_and_bounded(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    config = build_pipeline_config(settings, fast=True, seed=3)
    assert config.mlp is not None and config.mlp.epochs < 150
    assert config.evaluation is not None and config.evaluation.cv_folds == 3
    assert config.explain is not None and config.explain.sample_size <= 40
    assert config.risk is not None and config.risk.max_driver_rows <= 10
    assert config.scenario is not None and config.scenario.budget == 1_000.0


def test_pipeline_run_id_rejects_path_components(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    with pytest.raises(PipelineError) as caught:
        PipelineRunner(
            settings,
            build_pipeline_config(settings, dataset_path=tmp_path / "x.csv", run_id="../escape"),
        )
    assert caught.value.code == CODE_INVALID_CONFIG


def test_phase_modules_do_not_import_the_orchestrator(tmp_path) -> None:
    """Dependency direction stays one-way: phases never import Phase 12."""
    project = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    # The CLI is the composition root and may call the orchestrator; phase
    # packages and the dashboard must not acquire that upward dependency.
    allowed = {project / "churnwise" / "cli.py"}
    for path in (project / "churnwise").rglob("*.py"):
        if path.is_relative_to(project / "churnwise" / "pipeline") or path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "churnwise.pipeline":
                offenders.append(f"{path.relative_to(project)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and (
                node.module or ""
            ).startswith("churnwise.pipeline."):
                offenders.append(f"{path.relative_to(project)}:{node.lineno}")
    assert offenders == []


def test_orchestrator_does_not_import_the_dashboard() -> None:
    path = Path(__file__).resolve().parents[2] / "churnwise" / "pipeline" / "runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        module.startswith("churnwise.dashboard") or module == "streamlit"
        for module in imported
    )
