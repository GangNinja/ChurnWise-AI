from __future__ import annotations

import json
from pathlib import Path

import pytest

from churnwise.config import Settings
from churnwise.pipeline import CODE_INVALID_CONFIG, PipelineError
from churnwise.pipeline.cleanup import apply_cleanup, plan_cleanup

RUN_ID = "phase12-cleanup-test"


def _make_run(root: Path) -> tuple[Settings, Path]:
    settings = Settings(project_root=root)
    run_dir = Path(settings.artifact_dir) / "pipelines" / RUN_ID
    run_dir.mkdir(parents=True)
    stage_dir = Path(settings.artifact_dir) / "evaluation" / f"{RUN_ID}-bench"
    stage_dir.mkdir(parents=True)
    metrics = stage_dir / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    report = Path(settings.report_dir) / "preprocessing" / f"{RUN_ID}-prep.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    shared_cache = Path(settings.data_dir) / "cache" / "shared.bin"
    shared_cache.parent.mkdir(parents=True)
    shared_cache.write_text("shared", encoding="utf-8")
    unrelated = Path(settings.artifact_dir) / "evaluation" / "unrelated-run"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.json").write_text("{}", encoding="utf-8")
    prefix_collision = Path(settings.artifact_dir) / "models" / f"{RUN_ID}-extra"
    prefix_collision.mkdir(parents=True)
    (prefix_collision / "keep.txt").write_text("keep", encoding="utf-8")

    manifest = {
        "run_id": RUN_ID,
        "status": "succeeded",
        "artifacts": {
            "evaluate.metrics": metrics.relative_to(root).as_posix(),
            "evaluate.root": stage_dir.relative_to(root).as_posix(),
            "source.validation_report_json": report.relative_to(root).as_posix(),
            "data.cache": shared_cache.relative_to(root).as_posix(),
            "other.run": unrelated.relative_to(root).as_posix(),
            "prefix.collision": prefix_collision.relative_to(root).as_posix(),
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return settings, run_dir


def test_cleanup_plan_is_confined_and_dry_by_default(tmp_path) -> None:
    settings, run_dir = _make_run(tmp_path)
    plan = plan_cleanup(settings, RUN_ID)

    paths = {entry.path for entry in plan.entries}
    assert run_dir in paths
    assert any(path.name == f"{RUN_ID}-bench" for path in paths)
    assert any(path.name == f"{RUN_ID}-prep.json" for path in paths)
    assert all("cache" not in path.parts for path in paths)
    assert all("unrelated-run" not in path.parts for path in paths)
    assert any("data.cache" in message for message in plan.skipped)
    assert any("other.run" in message for message in plan.skipped)
    assert any("prefix.collision" in message for message in plan.skipped)

    # Planning must not touch the filesystem.
    assert run_dir.is_dir()
    assert (run_dir / "run.json").is_file()


def test_cleanup_apply_removes_run_artifacts_but_keeps_shared_data(tmp_path) -> None:
    settings, run_dir = _make_run(tmp_path)
    plan = plan_cleanup(settings, RUN_ID)
    deleted = apply_cleanup(plan)

    assert deleted
    assert not run_dir.exists()
    assert not (Path(settings.artifact_dir) / "evaluation" / f"{RUN_ID}-bench").exists()
    assert not (Path(settings.report_dir) / "preprocessing" / f"{RUN_ID}-prep.json").exists()
    assert (Path(settings.data_dir) / "cache" / "shared.bin").is_file()
    assert (Path(settings.artifact_dir) / "evaluation" / "unrelated-run" / "keep.json").is_file()
    assert (Path(settings.artifact_dir) / "models" / f"{RUN_ID}-extra" / "keep.txt").is_file()


def test_cleanup_rejects_a_path_outside_the_managed_roots(tmp_path) -> None:
    settings, run_dir = _make_run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["escape"] = outside.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = plan_cleanup(settings, RUN_ID)
    assert outside.is_file()
    assert any("escape" in message for message in plan.skipped)


def test_cleanup_requires_a_matching_manifest(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    with pytest.raises(PipelineError) as missing:
        plan_cleanup(settings, RUN_ID)
    assert missing.value.code == CODE_INVALID_CONFIG

    run_dir = Path(settings.artifact_dir) / "pipelines" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "different-run", "artifacts": {}}), encoding="utf-8"
    )
    with pytest.raises(PipelineError) as mismatch:
        plan_cleanup(settings, RUN_ID)
    assert mismatch.value.code == CODE_INVALID_CONFIG
