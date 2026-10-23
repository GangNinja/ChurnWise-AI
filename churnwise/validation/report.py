"""Validation report and EDA artifact writers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from churnwise.eda.plots import generate_plots
from churnwise.eda.profile import DatasetProfile
from churnwise.validation.contracts import (
    GateDecision,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
    SCHEMA_VERSION,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:120] or "dataset"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_summary(profile: DatasetProfile) -> dict[str, Any]:
    target = profile.target
    return {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "missing_cells": profile.missing_cells,
        "missing_cell_pct": profile.missing_cell_pct,
        "duplicate_rows": profile.duplicate_rows,
        "duplicate_row_pct": profile.duplicate_row_pct,
        "target_rate": target.positive_rate,
        "target_classes": list(target.classes),
        "constant_columns": [
            column.name for column in profile.columns if column.constant
        ],
        "near_constant_columns": [
            column.name for column in profile.columns if column.near_constant
        ],
        "id_like_columns": [
            column.name for column in profile.columns if "id_like" in column.flags
        ],
        "high_cardinality_columns": [
            column.name
            for column in profile.columns
            if "high_cardinality" in column.flags
        ],
        "duplicate_columns": list(profile.duplicate_columns),
        "leakage_risks": [
            {"column": column.name, "reasons": column.reasons}
            for column in profile.columns
            if "leakage_risk" in column.flags
        ],
        "usable_features": profile.roles.features,
        "usable_numeric": list(profile.roles.numeric),
        "usable_categorical": list(profile.roles.categorical),
        "id_column": profile.roles.id,
        "revenue_proxy": profile.roles.revenue_proxy,
    }


def build_report(
    *,
    dataset_id: str,
    manifest_sha256: str,
    artifact_sha256: str,
    artifact_path: str,
    generated_at: str,
    validator_version: str,
    checks: list[ValidationCheck],
    gate: GateDecision,
    profile: DatasetProfile,
    provenance: dict[str, Any],
    warnings: list[str] | None = None,
    notes: list[str] | None = None,
) -> ValidationReport:
    return ValidationReport(
        dataset_id=dataset_id,
        manifest_sha256=manifest_sha256,
        artifact_sha256=artifact_sha256,
        artifact_path=artifact_path,
        generated_at=generated_at,
        validator_version=validator_version,
        status=gate.status,
        checks=checks,
        summary=build_summary(profile),
        target=profile.target.to_dict(),
        roles=profile.roles.to_dict(),
        capabilities=dict(profile.capabilities),
        gate=gate,
        provenance=provenance,
        warnings=list(profile.warnings) + list(warnings or []),
        notes=list(notes or []),
    )


def _markdown(report: ValidationReport) -> str:
    lines = [
        f"# Validation report — {report.dataset_id}",
        "",
        f"- **Status:** `{report.status.value}`",
        f"- **Gate:** `{report.gate.status.value}` (can preprocess: "
        f"`{str(report.gate.can_preprocess).lower()}`)",
        f"- **Generated:** {report.generated_at}",
        f"- **Validator:** {report.validator_version}",
        f"- **Artifact:** `{report.artifact_path}`",
        f"- **Artifact SHA-256:** `{report.artifact_sha256}`",
        f"- **Manifest SHA-256:** `{report.manifest_sha256}`",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in (
        "n_rows",
        "n_cols",
        "missing_cells",
        "duplicate_rows",
        "target_rate",
        "id_column",
        "revenue_proxy",
    ):
        lines.append(f"| {key} | {report.summary.get(key)} |")
    lines.extend(
        [
            f"| usable_features | {len(report.summary.get('usable_features', []))} |",
            "",
            "## Target",
            "",
            f"- Column: `{report.target.get('column')}`",
            f"- Classes: {report.target.get('classes')}",
            f"- Mapping: {report.target.get('mapping')}",
            f"- Positive label (churn): `{report.target.get('positive_label')}`",
            f"- Positive rate: {report.target.get('positive_rate')}",
            f"- Confidence: `{report.target.get('confidence')}`",
            f"- Method: {report.target.get('method')}",
            "",
            "## Checks",
            "",
            "| ID | Check | Result | Severity | Detail |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for check in report.checks:
        result = "PASS" if check.passed else "FAIL"
        detail = check.detail.replace("|", "\\|")
        lines.append(
            f"| {check.id} | {check.name} | {result} | {check.severity.value} | {detail} |"
        )
    lines.extend(["", "## Roles", ""])
    roles = report.roles
    lines.append(f"- Target: `{roles.get('target')}`")
    lines.append(f"- ID: `{roles.get('id')}`")
    lines.append(f"- Numeric: {roles.get('numeric')}")
    lines.append(f"- Categorical: {roles.get('categorical')}")
    if roles.get("dropped"):
        lines.append("- Dropped:")
        for item in roles["dropped"]:
            lines.append(f"  - `{item['column']}` — {item['reason']}")
    lines.extend(["", "## Capabilities", "", "| Capability | Available | Source |", "| --- | --- | --- |"])
    for name, available in report.capabilities.items():
        source = report.roles.get("capability_sources", {}).get(name)
        lines.append(f"| {name} | {available} | {source or '—'} |")
    lines.extend(["", "## Provenance", ""])
    for key in (
        "run_id",
        "dataset_id",
        "retrieved_at",
        "download_url",
        "license",
        "manifest_path",
        "acquisition_attempt",
        "candidate_rank",
    ):
        if key in report.provenance:
            lines.append(f"- {key}: `{report.provenance[key]}`")
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    if report.eda_profile_path or report.eda_plots:
        lines.extend(["", "## EDA artifacts", ""])

        def relative_link(target: str) -> str:
            if not report.report_md_path:
                return target
            try:
                return Path(
                    os.path.relpath(Path(target).resolve(), Path(report.report_md_path).parent.resolve())
                ).as_posix()
            except (OSError, ValueError):
                return target

        if report.eda_profile_path:
            lines.append(
                f"- Profile: [{report.eda_profile_path}]({relative_link(report.eda_profile_path)})"
            )
        for plot in report.eda_plots:
            lines.append(f"- Plot: [{plot}]({relative_link(plot)})")
    return "\n".join(lines) + "\n"


def write_outputs(
    report: ValidationReport,
    profile: DatasetProfile | None,
    frame: pd.DataFrame | None,
    *,
    report_dir: Path | str,
    data_dir: Path | str,
    generate_eda: bool = True,
) -> dict[str, str]:
    """Write validation JSON/Markdown, the EDA profile and plots."""
    root = Path(report_dir)
    validation_dir = root / "validation"
    eda_dir = root / "eda"
    stem = _safe_name(report.dataset_id)
    paths: dict[str, str] = {}

    if profile is not None:
        profile_payload = profile.to_dict()
        profile_payload["generated_at"] = report.generated_at
        profile_path = eda_dir / f"{stem}.json"
        _atomic_write_json(profile_path, profile_payload)
        archive_profile = Path(data_dir) / "manifests" / f"{stem}.profile.json"
        _atomic_write_json(archive_profile, profile_payload)
        report.eda_profile_path = _display_path(profile_path)
        paths["profile"] = report.eda_profile_path
        paths["archived_profile"] = _display_path(archive_profile)
        if generate_eda and frame is not None and not frame.empty:
            plots = generate_plots(frame, profile, eda_dir)
            report.eda_plots = plots
            for index, plot in enumerate(plots):
                paths[f"plot_{index}"] = plot

    report_json = validation_dir / f"{stem}.json"
    report_md = validation_dir / f"{stem}.md"
    report.report_json_path = _display_path(report_json)
    report.report_md_path = _display_path(report_md)
    report.schema_version = SCHEMA_VERSION
    _atomic_write_json(report_json, report.to_dict())
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_markdown(report), encoding="utf-8")
    paths["report_json"] = report.report_json_path
    paths["report_md"] = report.report_md_path
    return paths
