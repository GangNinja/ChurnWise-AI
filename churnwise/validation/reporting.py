"""Human-readable Phase 4 output."""

from __future__ import annotations

from pathlib import Path

from churnwise.validation.contracts import ValidationResult


def render_summary(result: ValidationResult) -> str:
    # ASCII only: the summary is printed through whatever codec the console has.
    lines = ["ChurnWise AI - dataset validation", ""]
    if result.report is None:
        lines.append("Validation failed")
        if result.failure:
            lines.append(f"  code: {result.failure.code}")
            lines.append(f"  {result.failure.message}")
            for cause in result.failure.causes:
                lines.append(f"  - {cause}")
            if result.failure.remediation:
                lines.append(f"  next: {result.failure.remediation}")
        for attempt in result.attempts:
            lines.append(
                f"  attempt {attempt.candidate_id}: {attempt.stage} {attempt.status}"
                + (f" {attempt.error_code}" if attempt.error_code else "")
            )
        return "\n".join(lines)

    report = result.report
    lines.append(f"Dataset: {report.dataset_id}")
    lines.append(f"Status: {report.status.value}")
    lines.append(f"Gate: {report.gate.status.value} (can preprocess: {str(report.gate.can_preprocess).lower()})")
    target = report.target
    lines.append(
        f"Target: {target.get('column')} - classes {target.get('classes')} - "
        f"positive {target.get('positive_label')!r} - rate {target.get('positive_rate')}"
    )
    summary = report.summary
    lines.append(
        f"Shape: {summary.get('n_rows')} rows x {summary.get('n_cols')} columns - "
        f"{len(summary.get('usable_features', []))} usable features"
    )
    if report.gate.blocking:
        lines.append(f"Blocking checks: {', '.join(report.gate.blocking)}")
    if report.gate.warnings:
        lines.append(f"Warning checks: {', '.join(report.gate.warnings)}")
    for check in report.checks:
        if not check.passed:
            lines.append(f"  {check.id} {check.name}: {check.detail}")
    if report.report_md_path:
        lines.append(f"Report: {report.report_md_path}")
    if report.eda_profile_path:
        lines.append(f"Profile: {report.eda_profile_path}")
    if report.eda_plots:
        plot_dir = Path(report.eda_plots[0]).parent.as_posix()
        lines.append(f"Plots: {len(report.eda_plots)} in {plot_dir}")
    if result.attempts:
        lines.append(f"Attempts: {len(result.attempts)} candidate(s) considered")
    return "\n".join(lines)
