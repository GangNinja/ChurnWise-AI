"""Severity rollup for Phase 4 validation checks."""

from __future__ import annotations

from churnwise.validation.contracts import (
    CheckSeverity,
    GateDecision,
    ValidationCheck,
    ValidationStatus,
)


def decide_gate(checks: list[ValidationCheck]) -> GateDecision:
    """Convert check results into an explicit, documented gate decision.

    Fatal failures reject the dataset. Any warning-level failure is usable but
    requires attention. A clean dataset is ``PASS``. The gate is deliberately
    independent of the profile so tests can exercise it directly.
    """
    blocking = [
        check.id
        for check in checks
        if not check.passed and check.severity is CheckSeverity.FATAL
    ]
    warnings = [
        check.id
        for check in checks
        if not check.passed and check.severity is CheckSeverity.WARNING
    ]
    reasons = [
        f"{check.id} {check.name}: {check.detail}"
        for check in checks
        if not check.passed
    ]
    if blocking:
        status = ValidationStatus.REJECT
    elif warnings:
        status = ValidationStatus.PASS_WITH_WARNINGS
    else:
        status = ValidationStatus.PASS
    return GateDecision(
        status=status,
        passed=status is not ValidationStatus.REJECT,
        can_preprocess=status is not ValidationStatus.REJECT,
        blocking=blocking,
        warnings=warnings,
        reasons=reasons,
    )
