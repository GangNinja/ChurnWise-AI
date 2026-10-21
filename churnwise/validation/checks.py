"""Validation checks V1–V12.

Each check has one stable id and name, a severity, a pass/fail result, a human
detail and machine-readable evidence. Thresholds are declared here and repeated
in ``docs/VALIDATION.md`` and the Phase 4 report.
"""

from __future__ import annotations

from typing import Any

from churnwise.eda.profile import DatasetProfile, TargetInterpretation
from churnwise.validation.contracts import (
    CheckSeverity,
    ValidationCheck,
    ValidationIssue,
)

MIN_ROWS = 100
WARN_ROWS = 1_000
MIN_POSITIVE_RATE = 0.02
MAX_POSITIVE_RATE = 0.50
SEVERE_MISSING_PCT = 0.50
TARGET_MISSING_FATAL_PCT = 0.20
MAX_CATEGORICAL_LEVELS = 30
MIN_NUMERIC_FEATURES = 3
NEAR_CONSTANT_SHARE = 0.995

CHECK_NAMES: dict[str, str] = {
    "V1": "file_integrity_and_readability",
    "V2": "structural_integrity",
    "V3": "target_identified",
    "V4": "target_binary",
    "V5": "target_interpretation",
    "V6": "target_balance",
    "V7": "duplicate_columns",
    "V8": "duplicate_rows",
    "V9": "missingness",
    "V10": "low_information_features",
    "V11": "identifier_and_cardinality",
    "V12": "target_leakage",
}


def _issue(
    code: str,
    message: str,
    *,
    severity: CheckSeverity = CheckSeverity.WARNING,
    columns: tuple[str, ...] = (),
    evidence: dict[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        columns=columns,
        evidence=evidence or {},
    )


def _check(
    check_id: str,
    *,
    issues: list[ValidationIssue],
    detail: str,
    evidence: dict[str, Any] | None = None,
    default_severity: CheckSeverity = CheckSeverity.WARNING,
) -> ValidationCheck:
    if not issues:
        severity = default_severity
    elif any(issue.severity is CheckSeverity.FATAL for issue in issues):
        severity = CheckSeverity.FATAL
    else:
        # A non-fatal issue makes the check a warning even when the check's
        # declared (passing) severity is fatal, e.g. V2 on a small dataset.
        severity = CheckSeverity.WARNING
    return ValidationCheck(
        id=check_id,
        name=CHECK_NAMES[check_id],
        severity=severity,
        passed=not issues,
        detail=detail,
        issues=issues,
        evidence=evidence or {},
    )


def _not_evaluated(check_id: str, reason: str) -> ValidationCheck:
    return _check(
        check_id,
        issues=[
            _issue(
                f"{check_id}.not_evaluated",
                f"not evaluated: {reason}",
                severity=CheckSeverity.WARNING,
            )
        ],
        detail=f"not evaluated: {reason}",
        default_severity=CheckSeverity.WARNING,
    )


def run_checks(
    profile: DatasetProfile | None,
    target: TargetInterpretation | None,
    *,
    integrity_ok: bool,
    integrity_detail: str = "",
    load_error: str | None = None,
) -> list[ValidationCheck]:
    """Run all twelve checks. The returned order is stable for report diffing."""
    if not integrity_ok:
        reason = integrity_detail or "artifact failed integrity verification"
        checks = [
            _check(
                "V1",
                issues=[
                    _issue(
                        "V1.digest_mismatch",
                        reason,
                        severity=CheckSeverity.FATAL,
                        evidence={"integrity_ok": False},
                    )
                ],
                detail=reason,
                evidence={"integrity_ok": False},
                default_severity=CheckSeverity.FATAL,
            )
        ]
        checks.append(
            _check(
                "V2",
                issues=[
                    _issue(
                        "V2.not_evaluated",
                        "structure cannot be inspected because the artifact failed integrity verification",
                        severity=CheckSeverity.FATAL,
                    )
                ],
                detail="artifact was not loaded",
                default_severity=CheckSeverity.FATAL,
            )
        )
        checks.extend(
            _not_evaluated(check_id, "artifact failed integrity verification")
            for check_id in ("V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12")
        )
        return checks

    if load_error is not None:
        checks = [
            _check(
                "V1",
                issues=[
                    _issue(
                        "V1.load_failed",
                        load_error,
                        severity=CheckSeverity.FATAL,
                        evidence={"integrity_ok": True},
                    )
                ],
                detail=load_error,
                evidence={"integrity_ok": True},
                default_severity=CheckSeverity.FATAL,
            ),
            _check(
                "V2",
                issues=[
                    _issue(
                        "V2.load_failed",
                        "structure could not be inspected because parsing failed",
                        severity=CheckSeverity.FATAL,
                    )
                ],
                detail="dataset could not be parsed",
                default_severity=CheckSeverity.FATAL,
            ),
        ]
        checks.extend(
            _not_evaluated(check_id, "dataset could not be parsed")
            for check_id in ("V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12")
        )
        return checks

    if profile is None:
        checks = [
            _check(
                "V1",
                issues=[
                    _issue(
                        "V1.no_profile",
                        "dataset could not be loaded into a profile",
                        severity=CheckSeverity.FATAL,
                    )
                ],
                detail="no profile",
                default_severity=CheckSeverity.FATAL,
            )
        ]
        checks.extend(
            _not_evaluated(check_id, "no profile was produced")
            for check_id in ("V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12")
        )
        return checks

    interpretation = target or profile.target
    checks = [
        _check(
            "V1",
            issues=[],
            detail="artifact exists, is readable and matches its recorded digest",
            evidence={
                "integrity_ok": True,
                "source_path": profile.source_path,
                "format": profile.format,
            },
            default_severity=CheckSeverity.FATAL,
        )
    ]

    structure_issues: list[ValidationIssue] = []
    if profile.n_rows == 0:
        structure_issues.append(
            _issue(
                "V2.empty",
                "dataset has zero rows",
                severity=CheckSeverity.FATAL,
                evidence={"n_rows": 0},
            )
        )
    elif profile.n_rows < MIN_ROWS:
        structure_issues.append(
            _issue(
                "V2.too_small",
                f"{profile.n_rows} rows is below the validation minimum of {MIN_ROWS}",
                severity=CheckSeverity.FATAL,
                evidence={"n_rows": profile.n_rows, "minimum": MIN_ROWS},
            )
        )
    elif profile.n_rows < WARN_ROWS:
        structure_issues.append(
            _issue(
                "V2.small",
                f"{profile.n_rows} rows is below the recommended {WARN_ROWS}",
                severity=CheckSeverity.WARNING,
                evidence={"n_rows": profile.n_rows, "recommended": WARN_ROWS},
            )
        )
    if profile.n_cols < 2:
        structure_issues.append(
            _issue(
                "V2.too_few_columns",
                f"dataset has {profile.n_cols} columns; a target and at least one predictor are required",
                severity=CheckSeverity.FATAL,
                evidence={"n_cols": profile.n_cols},
            )
        )
    checks.append(
        _check(
            "V2",
            issues=structure_issues,
            detail=(
                f"{profile.n_rows} rows x {profile.n_cols} columns"
                if not structure_issues
                else f"structural problems in {profile.n_rows} rows x {profile.n_cols} columns"
            ),
            evidence={"n_rows": profile.n_rows, "n_cols": profile.n_cols},
            default_severity=CheckSeverity.FATAL,
        )
    )

    target_issues: list[ValidationIssue] = []
    if interpretation.column is None:
        target_issues.append(
            _issue(
                "V3.missing_target",
                interpretation.reason or "no plausible churn target column was found",
                severity=CheckSeverity.FATAL,
                evidence={"candidates": interpretation.candidates},
            )
        )
    elif interpretation.ambiguous:
        target_issues.append(
            _issue(
                "V3.ambiguous_target",
                interpretation.reason or "multiple plausible target columns",
                severity=CheckSeverity.FATAL,
                columns=(interpretation.column,),
                evidence={"candidates": interpretation.candidates},
            )
        )
    checks.append(
        _check(
            "V3",
            issues=target_issues,
            detail=(
                f"selected target {interpretation.column!r}"
                if not target_issues
                else "target selection failed or is ambiguous"
            ),
            evidence={
                "column": interpretation.column,
                "ambiguous": interpretation.ambiguous,
                "candidates": interpretation.candidates,
            },
            default_severity=CheckSeverity.FATAL,
        )
    )

    binary_issues: list[ValidationIssue] = []
    if interpretation.column is None:
        binary_issues.append(
            _issue(
                "V4.no_target",
                "target cardinality cannot be checked without a target",
                severity=CheckSeverity.FATAL,
            )
        )
    elif len(interpretation.classes) != 2:
        binary_issues.append(
            _issue(
                "V4.not_binary",
                f"target has {len(interpretation.classes)} classes; exactly two are required",
                severity=CheckSeverity.FATAL,
                columns=(interpretation.column,),
                evidence={
                    "classes": interpretation.classes,
                    "counts": interpretation.counts,
                },
            )
        )
    checks.append(
        _check(
            "V4",
            issues=binary_issues,
            detail=(
                f"target has two classes: {interpretation.classes}"
                if not binary_issues
                else "target is not binary"
            ),
            evidence={
                "classes": interpretation.classes,
                "counts": interpretation.counts,
            },
            default_severity=CheckSeverity.FATAL,
        )
    )

    interpretation_issues: list[ValidationIssue] = []
    if interpretation.column is None:
        interpretation_issues.append(
            _issue(
                "V5.no_target",
                "positive class cannot be interpreted without a target",
                severity=CheckSeverity.FATAL,
            )
        )
    elif not interpretation.usable:
        interpretation_issues.append(
            _issue(
                "V5.low_confidence",
                interpretation.reason or "positive class could not be interpreted confidently",
                severity=CheckSeverity.FATAL,
                columns=(interpretation.column,),
                evidence={
                    "confidence": interpretation.confidence,
                    "mapping": interpretation.mapping,
                    "method": interpretation.method,
                },
            )
        )
    checks.append(
        _check(
            "V5",
            issues=interpretation_issues,
            detail=(
                f"1 = churn maps from {interpretation.positive_label!r}; "
                f"confidence {interpretation.confidence}"
                if not interpretation_issues
                else "positive-class interpretation failed"
            ),
            evidence={
                "column": interpretation.column,
                "mapping": interpretation.mapping,
                "positive_label": interpretation.positive_label,
                "negative_label": interpretation.negative_label,
                "confidence": interpretation.confidence,
                "method": interpretation.method,
            },
            default_severity=CheckSeverity.FATAL,
        )
    )

    balance_issues: list[ValidationIssue] = []
    rate = interpretation.positive_rate
    if rate is None:
        balance_issues.append(
            _issue(
                "V6.unavailable",
                "positive rate is unavailable",
                severity=CheckSeverity.WARNING,
            )
        )
    elif rate < MIN_POSITIVE_RATE or rate > MAX_POSITIVE_RATE:
        balance_issues.append(
            _issue(
                "V6.outside_range",
                f"positive rate {rate:.3%} is outside the documented 2%-50% range",
                severity=CheckSeverity.WARNING,
                columns=(interpretation.column,) if interpretation.column else (),
                evidence={
                    "positive_rate": rate,
                    "minimum": MIN_POSITIVE_RATE,
                    "maximum": MAX_POSITIVE_RATE,
                },
            )
        )
    checks.append(
        _check(
            "V6",
            issues=balance_issues,
            detail=(
                f"positive rate {rate:.3%}" if rate is not None else "positive rate unavailable"
            ),
            evidence={
                "positive_rate": rate,
                "positive_count": interpretation.positive_count,
                "observed_count": interpretation.observed_count,
                "minimum": MIN_POSITIVE_RATE,
                "maximum": MAX_POSITIVE_RATE,
            },
            default_severity=CheckSeverity.WARNING,
        )
    )

    duplicate_issues: list[ValidationIssue] = []
    for name in profile.duplicate_headers:
        duplicate_issues.append(
            _issue(
                "V7.duplicate_header",
                f"header {name!r} appears more than once",
                severity=CheckSeverity.WARNING,
                columns=(name,),
            )
        )
    for pair in profile.duplicate_columns:
        duplicate_issues.append(
            _issue(
                "V7.duplicate_content",
                f"column {pair['column']!r} duplicates {pair['duplicate_of']!r}",
                severity=CheckSeverity.WARNING,
                columns=(pair["column"], pair["duplicate_of"]),
            )
        )
    checks.append(
        _check(
            "V7",
            issues=duplicate_issues,
            detail=(
                "no duplicate headers or columns"
                if not duplicate_issues
                else f"{len(duplicate_issues)} duplicate header/column issue(s)"
            ),
            evidence={
                "duplicate_headers": profile.duplicate_headers,
                "duplicate_columns": profile.duplicate_columns,
            },
            default_severity=CheckSeverity.WARNING,
        )
    )

    duplicate_row_issues: list[ValidationIssue] = []
    if profile.duplicate_rows > 0:
        duplicate_row_issues.append(
            _issue(
                "V8.duplicate_rows",
                f"{profile.duplicate_rows} duplicate rows ({profile.duplicate_row_pct:.1%})",
                severity=CheckSeverity.WARNING,
                evidence={
                    "duplicate_rows": profile.duplicate_rows,
                    "duplicate_row_pct": profile.duplicate_row_pct,
                },
            )
        )
    checks.append(
        _check(
            "V8",
            issues=duplicate_row_issues,
            detail=(
                "no duplicate rows"
                if not duplicate_row_issues
                else f"{profile.duplicate_rows} duplicate rows"
            ),
            evidence={
                "duplicate_rows": profile.duplicate_rows,
                "duplicate_row_pct": profile.duplicate_row_pct,
            },
            default_severity=CheckSeverity.WARNING,
        )
    )

    missing_issues: list[ValidationIssue] = []
    if interpretation.missing_pct > TARGET_MISSING_FATAL_PCT:
        missing_issues.append(
            _issue(
                "V9.target_missing_fatal",
                f"target {interpretation.column!r} is missing in "
                f"{interpretation.missing_pct:.1%} of rows (limit {TARGET_MISSING_FATAL_PCT:.0%})",
                severity=CheckSeverity.FATAL,
                columns=(interpretation.column,) if interpretation.column else (),
                evidence={
                    "target_missing_count": interpretation.missing_count,
                    "target_missing_pct": interpretation.missing_pct,
                    "limit": TARGET_MISSING_FATAL_PCT,
                },
            )
        )
    elif interpretation.missing_count > 0:
        missing_issues.append(
            _issue(
                "V9.target_missing",
                f"target {interpretation.column!r} is missing in "
                f"{interpretation.missing_pct:.1%} of rows; those rows must be excluded",
                severity=CheckSeverity.WARNING,
                columns=(interpretation.column,) if interpretation.column else (),
                evidence={
                    "target_missing_count": interpretation.missing_count,
                    "target_missing_pct": interpretation.missing_pct,
                },
            )
        )
    if profile.missing_cells > 0:
        missing_issues.append(
            _issue(
                "V9.cells_missing",
                f"{profile.missing_cells} cells are missing ({profile.missing_cell_pct:.2%})",
                severity=CheckSeverity.WARNING,
                evidence={
                    "missing_cells": profile.missing_cells,
                    "missing_cell_pct": profile.missing_cell_pct,
                },
            )
        )
    for column in profile.columns:
        if column.name == interpretation.column:
            continue
        if column.missing_pct > SEVERE_MISSING_PCT and column.role not in {"dropped", "id"}:
            missing_issues.append(
                _issue(
                    "V9.column_missing",
                    f"column {column.name!r} is {column.missing_pct:.1%} missing",
                    severity=CheckSeverity.WARNING,
                    columns=(column.name,),
                    evidence={"missing_pct": column.missing_pct},
                )
            )
    non_finite_columns = [
        column
        for column in profile.columns
        if column.role == "numeric"
        and column.numeric
        and column.numeric.get("infinite_count", 0)
    ]
    for column in non_finite_columns:
        infinite_count = int(column.numeric["infinite_count"])
        missing_issues.append(
            _issue(
                "V9.non_finite",
                f"column {column.name!r} contains {infinite_count} infinite value(s); "
                "they must be handled before scaling",
                severity=CheckSeverity.WARNING,
                columns=(column.name,),
                evidence={
                    "infinite_count": infinite_count,
                    "finite_count": column.numeric.get("count"),
                },
            )
        )
    checks.append(
        _check(
            "V9",
            issues=missing_issues,
            detail=(
                "no missing or non-finite values"
                if not missing_issues
                else f"{profile.missing_cells} missing cells"
                if profile.missing_cells
                else f"{len(missing_issues)} data-quality issue(s)"
            ),
            evidence={
                "missing_cells": profile.missing_cells,
                "missing_cell_pct": profile.missing_cell_pct,
                "target_missing_count": interpretation.missing_count,
                "target_missing_pct": interpretation.missing_pct,
                "non_finite_columns": [column.name for column in non_finite_columns],
            },
            default_severity=CheckSeverity.WARNING,
        )
    )

    low_information_issues: list[ValidationIssue] = []
    for column in profile.columns:
        if column.role == "target":
            continue
        if column.constant:
            low_information_issues.append(
                _issue(
                    "V10.constant",
                    f"column {column.name!r} is constant and must not be used as a feature",
                    severity=CheckSeverity.WARNING,
                    columns=(column.name,),
                    evidence={"unique_count": column.unique_count},
                )
            )
        elif column.near_constant:
            low_information_issues.append(
                _issue(
                    "V10.near_constant",
                    f"column {column.name!r} is nearly constant (dominant value >= {NEAR_CONSTANT_SHARE:.1%})",
                    severity=CheckSeverity.WARNING,
                    columns=(column.name,),
                    evidence={"unique_count": column.unique_count},
                )
            )
    checks.append(
        _check(
            "V10",
            issues=low_information_issues,
            detail=(
                "no constant or near-constant columns"
                if not low_information_issues
                else f"{len(low_information_issues)} low-information column(s)"
            ),
            evidence={"columns": [column.name for column in profile.columns if column.constant or column.near_constant]},
            default_severity=CheckSeverity.WARNING,
        )
    )

    identity_issues: list[ValidationIssue] = []
    if profile.roles.id:
        identity_issues.append(
            _issue(
                "V11.identifier",
                f"identifier-like column {profile.roles.id!r} is excluded from features",
                severity=CheckSeverity.WARNING,
                columns=(profile.roles.id,),
                evidence={"role": "id"},
            )
        )
    for column in profile.columns:
        if column.role in {"id", "target"}:
            continue
        if "high_cardinality" in column.flags:
            identity_issues.append(
                _issue(
                    "V11.high_cardinality",
                    f"column {column.name!r} has {column.unique_count} levels (limit {MAX_CATEGORICAL_LEVELS})",
                    severity=CheckSeverity.WARNING,
                    columns=(column.name,),
                    evidence={"unique_count": column.unique_count, "limit": MAX_CATEGORICAL_LEVELS},
                )
            )
        elif "text_like" in column.flags:
            identity_issues.append(
                _issue(
                    "V11.text_like",
                    f"column {column.name!r} is free-text or near-unique and was excluded",
                    severity=CheckSeverity.WARNING,
                    columns=(column.name,),
                )
            )
    if not profile.roles.features:
        identity_issues.append(
            _issue(
                "V11.no_features",
                "no usable numeric or categorical predictor remains after role mapping",
                severity=CheckSeverity.FATAL,
                evidence={
                    "numeric": profile.roles.numeric,
                    "categorical": profile.roles.categorical,
                },
            )
        )
    elif not profile.roles.numeric:
        identity_issues.append(
            _issue(
                "V11.no_numeric",
                "no numeric predictor remains; at least one numeric feature is required",
                severity=CheckSeverity.WARNING,
                evidence={"categorical": profile.roles.categorical},
            )
        )
    numeric_only = bool(profile.roles.numeric) and not profile.roles.categorical
    if numeric_only and len(profile.roles.numeric) < MIN_NUMERIC_FEATURES:
        identity_issues.append(
            _issue(
                "V11.insufficient_features",
                f"only {len(profile.roles.numeric)} numeric predictor(s) remain; "
                f"at least {MIN_NUMERIC_FEATURES} numeric or one categorical feature is required",
                severity=CheckSeverity.FATAL,
                evidence={
                    "numeric": profile.roles.numeric,
                    "minimum_numeric": MIN_NUMERIC_FEATURES,
                },
            )
        )
    checks.append(
        _check(
            "V11",
            issues=identity_issues,
            detail=(
                (
                    f"{len(identity_issues)} identifier/cardinality warning(s); "
                    if identity_issues
                    else ""
                )
                + (
                    f"{len(profile.roles.numeric)} numeric and "
                    f"{len(profile.roles.categorical)} categorical usable features"
                    if profile.roles.features
                    else "no usable features"
                )
            ),
            evidence={
                "id": profile.roles.id,
                "numeric": profile.roles.numeric,
                "categorical": profile.roles.categorical,
                "dropped": profile.roles.dropped,
            },
            default_severity=CheckSeverity.WARNING,
        )
    )

    leakage_issues: list[ValidationIssue] = []
    for column in profile.columns:
        if column.role == "target" or not column.reasons:
            continue
        if "leakage_risk" in column.flags or any(
            word in reason
            for reason in column.reasons
            for word in ("target", "outcome", "retention", "cancellation", "churn")
        ):
            leakage_issues.append(
                _issue(
                    "V12.leakage_risk",
                    f"column {column.name!r} is a suspected leakage source; it was excluded",
                    severity=CheckSeverity.WARNING,
                    columns=(column.name,),
                    evidence={"reasons": column.reasons},
                )
            )
    checks.append(
        _check(
            "V12",
            issues=leakage_issues,
            detail=(
                "no suspected target leakage"
                if not leakage_issues
                else f"{len(leakage_issues)} suspected leakage column(s)"
            ),
            evidence={
                "columns": [column.name for column in profile.columns if "leakage_risk" in column.flags]
            },
            default_severity=CheckSeverity.WARNING,
        )
    )
    return checks
