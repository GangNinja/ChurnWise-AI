"""Discovery report persistence and human-readable rendering."""

from __future__ import annotations

import json
import os
from pathlib import Path

from churnwise.contracts import DiscoveryReport


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_reports(report: DiscoveryReport, report_dir: Path | str) -> dict[str, Path]:
    """Write ``candidates.json`` and ``rejections.json`` under ``<report_dir>/acquisition``."""
    directory = Path(report_dir) / "acquisition"
    candidates_path = directory / "candidates.json"
    rejections_path = directory / "rejections.json"
    _atomic_write_json(candidates_path, report.to_dict())
    _atomic_write_json(rejections_path, report.rejections_dict())
    return {"candidates": candidates_path, "rejections": rejections_path}


def render_summary(report: DiscoveryReport, paths: dict[str, Path] | None = None) -> str:
    lines: list[str] = ["ChurnWise AI — dataset discovery", ""]

    lines.append("Sources")
    if not report.source_results:
        lines.append("  (none queried)")
    for result in report.source_results:
        status = result.status.value
        detail = f"{len(result.candidates)} candidate(s)"
        if result.error_code:
            detail = f"{result.error_code}: {result.error_message}"
        lines.append(
            f"  {result.source_id:<12} {status:<12} {detail}"
            f"  [{result.request_count} req, {result.duration_ms} ms]"
        )
        for note in result.notes[:3]:
            lines.append(f"      note: {note}")
    lines.append("")

    unique = report.ranked()
    duplicates = report.duplicates()
    lines.append(
        f"Result: {len(report.assessments) + len(report.invalid_candidates)} found, "
        f"{len(unique)} unique, {sum(1 for a in unique if a.eligible)} eligible, "
        f"{len(duplicates)} duplicate listing(s)"
    )
    if duplicates:
        for group in report.duplicate_groups:
            lines.append(
                f"  duplicate group: {group.representative_id} <- "
                f"{', '.join(m for m in group.member_ids if m != group.representative_id)}"
            )
    lines.append("")

    if report.selected is None:
        failure = report.failure
        lines.append("Discovery failed")
        if failure:
            lines.append(f"  code: {failure.code}")
            lines.append(f"  {failure.message}")
            for reason in failure.reasons[:6]:
                lines.append(f"  - {reason}")
            lines.append(f"  next: {failure.next_action}")
    else:
        selected = report.selected
        best = next(
            (a for a in unique if a.candidate.candidate_id == selected.candidate_id), None
        )
        score_text = f"{best.score:.3f}" if best else "n/a"
        lines.append(f"Selected: {selected.candidate_id} — {selected.title} (score {score_text})")
        lines.append(
            f"  licence: {selected.license_name or 'unknown'} ({selected.license_status})"
        )
        lines.append(
            f"  target: {selected.target_column or 'unknown'} · "
            f"{selected.row_count or '?'} rows × {selected.feature_count or '?'} columns · "
            f"format {selected.format}"
        )
        if selected.resource_url:
            lines.append(f"  resource: {selected.resource_url}")
        if selected.limitations:
            for limitation in selected.limitations:
                lines.append(f"  limitation: {limitation}")
    lines.append("")

    if paths:
        lines.append(f"Report: {paths['candidates']}")
        lines.append(f"Rejections: {paths['rejections']}")
    return "\n".join(lines)
