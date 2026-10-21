"""Acquisition attempt persistence and human-readable output."""

from __future__ import annotations

from pathlib import Path

from churnwise.acquisition.contracts import AcquisitionResult
from churnwise.acquisition.provenance import atomic_write_json


def write_attempts(
    result: AcquisitionResult,
    report_dir: Path | str,
    *,
    offline: bool,
) -> Path:
    path = Path(report_dir) / "acquisition" / "attempts.json"
    payload = {
        "schema_version": "1.0",
        "phase": 3,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "dataset_sha256": result.dataset_sha256,
        "seed": result.seed,
        "code_version": result.code_version,
        "offline": offline,
        "attempts": [attempt.to_dict() for attempt in result.attempts],
        "result": {
            "status": result.status,
            "cached": result.cached,
            "failure": result.failure.to_dict() if result.failure else None,
        },
    }
    atomic_write_json(path, payload)
    return path


def render_summary(result: AcquisitionResult, attempts_path: Path | None = None) -> str:
    lines = ["ChurnWise AI — dataset acquisition", ""]
    if not result.succeeded or result.artifact is None:
        lines.append("Acquisition failed")
        if result.failure:
            lines.append(f"  code: {result.failure.code}")
            lines.append(f"  {result.failure.message}")
            for cause in result.failure.causes:
                lines.append(f"  - {cause}")
            if result.failure.remediation:
                lines.append(f"  next: {result.failure.remediation}")
        for attempt in result.attempts:
            if attempt.error_code:
                lines.append(
                    f"  attempt {attempt.rank}: {attempt.candidate_id} "
                    f"{attempt.error_code}: {attempt.error_message}"
                )
    else:
        artifact = result.artifact
        state = "cached" if result.cached else "acquired"
        lines.append(f"Selected: {artifact.dataset_id} ({state})")
        lines.append(f"  file: {artifact.filename} · {artifact.bytes} bytes · {artifact.format}")
        lines.append(f"  sha256: {artifact.sha256}")
        if artifact.extracted_path:
            lines.append(f"  extracted: {artifact.extracted_path}")
        lines.append(f"  raw: {artifact.raw_path}")
        lines.append(f"  cache: {artifact.cache_path}")
        if result.manifest_path:
            lines.append(f"  manifest: {result.manifest_path}")
        for warning in result.warnings:
            lines.append(f"  warning: {warning}")
    if attempts_path is not None:
        lines.append("")
        lines.append(f"Attempts: {attempts_path}")
    return "\n".join(lines)
