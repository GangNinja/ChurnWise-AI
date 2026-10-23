"""Safe cleanup of one persisted Phase 12 pipeline run.

The command never deletes shared acquisition data. It removes only files and
directories recorded in ``artifacts/pipelines/<run-id>/run.json`` that resolve
inside the configured artifact or report roots, plus the pipeline run directory
itself. It is a dry run unless ``--delete`` is passed.

Usage::

    python -m churnwise.pipeline.cleanup --run-id phase12-example
    python -m churnwise.pipeline.cleanup --run-id phase12-example --delete
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from churnwise.config import Settings
from churnwise.pipeline.errors import CODE_INVALID_CONFIG, PipelineError

_RUN_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
_STAGE_SUFFIXES = ("prep", "mlp", "bench", "xai", "risk", "risk-drivers", "sim")


@dataclass(frozen=True)
class CleanupEntry:
    path: Path
    kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path.as_posix(),
            "kind": self.kind,
            "reason": self.reason,
            "exists": str(self.path.exists()).lower(),
        }


@dataclass
class CleanupPlan:
    run_id: str
    manifest_path: Path
    entries: list[CleanupEntry] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "manifest_path": self.manifest_path.as_posix(),
            "entries": [entry.to_dict() for entry in self.entries],
            "skipped": list(self.skipped),
        }


def _validate_run_id(run_id: str) -> None:
    if (
        len(run_id) < 3
        or len(run_id) > 111
        or run_id[0] in "._-"
        or run_id[0] not in _RUN_ID_CHARS
        or any(char not in _RUN_ID_CHARS for char in run_id)
    ):
        raise PipelineError(
            CODE_INVALID_CONFIG,
            f"invalid run id: {run_id!r}",
            "Use the exact run id recorded in artifacts/pipelines/<run-id>/run.json.",
        )


def _resolve(settings: Settings, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(settings.project_root) / path).resolve()


def _is_inside(target: Path, root: Path) -> bool:
    return target == root or target.is_relative_to(root)


def _names_run(run_id: str, part: str) -> bool:
    """Whether a path component is one of this run's known artifact names.

    A plain ``startswith(run_id)`` would let run ``demo`` delete artifacts from
    ``demo-extra``. Only the exact run id and its known stage suffixes qualify.
    """
    allowed = {run_id} | {f"{run_id}-{suffix}" for suffix in _STAGE_SUFFIXES}
    return part in allowed or Path(part).stem in allowed


def plan_cleanup(settings: Settings, run_id: str) -> CleanupPlan:
    """Build a deletion plan for one run without modifying anything."""
    _validate_run_id(run_id)
    artifact_root = Path(settings.artifact_dir).resolve()
    report_root = Path(settings.report_dir).resolve()
    run_dir = (artifact_root / "pipelines" / run_id).resolve()
    manifest_path = run_dir / "run.json"

    if not _is_inside(run_dir, artifact_root) or run_dir == artifact_root / "pipelines":
        raise PipelineError(
            CODE_INVALID_CONFIG,
            "the run directory escapes the configured artifact root",
            "Check artifact_dir and the run id; no files were deleted.",
        )
    if not manifest_path.is_file():
        raise PipelineError(
            CODE_INVALID_CONFIG,
            f"no pipeline manifest exists for run {run_id!r}",
            "Use the exact run id recorded under artifacts/pipelines/.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(
            CODE_INVALID_CONFIG,
            f"pipeline manifest could not be read: {exc}",
            "Inspect the manifest manually; no files were deleted.",
        ) from exc
    if not isinstance(manifest, dict):
        raise PipelineError(
            CODE_INVALID_CONFIG,
            "pipeline manifest is not a JSON object",
            "Inspect the manifest manually; no files were deleted.",
        )
    if manifest.get("run_id") != run_id:
        raise PipelineError(
            CODE_INVALID_CONFIG,
            "pipeline manifest run_id does not match the requested run",
            "Use the run id recorded inside the manifest; no files were deleted.",
        )

    candidates: list[CleanupEntry] = []
    skipped: list[str] = []
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raw_artifacts = {}
    for key, value in raw_artifacts.items():
        if not isinstance(value, str) or not value:
            continue
        target = _resolve(settings, value)
        if not _is_inside(target, artifact_root) and not _is_inside(
            target, report_root
        ):
            skipped.append(f"{key}: outside artifact/report roots ({target})")
            continue
        if target in {artifact_root, report_root}:
            skipped.append(f"{key}: refusing to delete a configured root")
            continue
        if target == run_dir or target.is_relative_to(run_dir):
            continue
        # Only artifacts whose path names this run may be removed. This keeps a
        # corrupt or hand-edited manifest from wiping a shared stage directory.
        relative_parts = target.relative_to(
            artifact_root if _is_inside(target, artifact_root) else report_root
        ).parts
        if not any(_names_run(run_id, part) for part in relative_parts):
            skipped.append(f"{key}: path does not name run {run_id!r} ({target})")
            continue
        candidates.append(
            CleanupEntry(target, "directory" if target.is_dir() else "file", str(key))
        )

    # The run directory is removed last, after every external stage artifact.
    candidates.sort(key=lambda entry: (len(entry.path.parts), entry.path.as_posix()))
    unique: list[CleanupEntry] = []
    seen: set[Path] = set()
    for entry in candidates:
        if entry.path in seen:
            continue
        if any(parent.path == entry.path or entry.path.is_relative_to(parent.path) for parent in unique):
            continue
        seen.add(entry.path)
        unique.append(entry)
    unique.append(CleanupEntry(run_dir, "directory", "pipeline run directory"))
    return CleanupPlan(run_id, manifest_path, unique, skipped)


def apply_cleanup(plan: CleanupPlan) -> list[Path]:
    """Delete the planned paths; return those that existed before deletion."""
    deleted: list[Path] = []
    for entry in plan.entries:
        if not entry.path.exists():
            continue
        if entry.path.is_dir():
            shutil.rmtree(entry.path)
        else:
            entry.path.unlink()
        deleted.append(entry.path)
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely clean one persisted pipeline run (dry run by default)."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="perform deletion; without this flag the plan is printed only",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)

    settings = Settings.load(Path(args.project_root))
    try:
        plan = plan_cleanup(settings, args.run_id)
        deleted = apply_cleanup(plan) if args.delete else []
    except PipelineError as exc:
        payload = exc.to_dict()
        if args.as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"{payload['code']}: {payload['message']}", file=sys.stderr)
            print(payload["remediation"], file=sys.stderr)
        return 2

    if args.as_json:
        payload = plan.to_dict()
        payload["deleted"] = [path.as_posix() for path in deleted]
        payload["dry_run"] = not args.delete
        print(json.dumps(payload, indent=2))
    else:
        print(f"Pipeline run {plan.run_id}")
        for entry in plan.entries:
            print(f"  {entry.kind:<9} {entry.path}")
        for message in plan.skipped:
            print(f"  skipped   {message}")
        if args.delete:
            print(f"Deleted {len(deleted)} path(s).")
        else:
            print("Dry run: no files deleted. Re-run with --delete to remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
