"""Human-readable Phase 9 summaries; all numbers come from persisted results."""

from __future__ import annotations

from typing import Any

from churnwise.intelligence.contracts import BatchAssessment, RiskRun, RiskTable


def render_summary(result: RiskRun) -> str:
    table = result.table
    summary = result.summary or {}
    lines = [
        f"Risk run {result.run_id}",
        f"Model: {table.model_family} ({table.model_run_id})",
        (
            f"Classification threshold: {table.classification_threshold:.3f}; "
            f"risk tiers: {', '.join(table.thresholds.labels)}"
        ),
    ]
    population = summary.get("population") or {}
    if population:
        lines.append(
            f"Population: {population.get('split', '?')} split, "
            f"{population.get('received', len(table.records))} received, "
            f"{population.get('scored', len(table.records))} scored, "
            f"{population.get('rejected', 0)} rejected"
        )
    counts = table.tier_counts()
    lines.append("Tiers:")
    for label, data in counts.items():
        lines.append(
            f"  {label:>8}: {data['count']:>6}  ({float(data['pct']):.2%})"
        )
    lines.append(
        f"Predicted churn: {(summary.get('predicted_class_counts') or {}).get('churn', 0)}; "
        f"drivers available: {(summary.get('explanations') or {}).get('available', 0)}"
    )
    value = summary.get("value_at_risk") or {}
    if value.get("available"):
        lines.append(
            f"Value-at-risk: available over {value.get('horizon_months')} months"
        )
    else:
        lines.append("Value-at-risk: unavailable (no revenue proxy supplied)")
    if table.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in table.warnings)
    if result.rejections:
        lines.append("Rejections:")
        lines.extend(
            f"  - position {item.input_position}: {item.code}: {item.message}"
            for item in result.rejections[:10]
        )
        remaining = len(result.rejections) - 10
        if remaining > 0:
            lines.append(f"  - {remaining} more rejection(s) in rejections.json")
    if result.paths:
        lines.append("Artifacts:")
        for name in ("churn_scores", "risk_summary", "tiers", "drivers", "rejections"):
            if name in result.paths:
                lines.append(f"  {name}: {result.paths[name]}")
    return "\n".join(lines)


def render_batch_summary(batch: BatchAssessment) -> str:
    lines = [
        f"Batch assessment — {batch.scored} scored, {batch.rejected} rejected, "
        f"{batch.total_received} received",
        f"Model: {batch.model_family} ({batch.model_run_id}); "
        f"classification threshold {batch.classification_threshold:.3f}",
    ]
    if batch.records:
        counts: dict[str, int] = {}
        for item in batch.records:
            counts[item.risk_tier] = counts.get(item.risk_tier, 0) + 1
        lines.append(
            "Tiers: "
            + ", ".join(f"{label}={counts.get(label, 0)}" for label in batch.risk_thresholds.labels)
        )
        explained = sum(1 for item in batch.records if item.explanation_status == "available")
        lines.append(f"Drivers available: {explained}/{batch.scored}")
    if batch.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in batch.warnings)
    if batch.rejections:
        lines.append("Rejections:")
        lines.extend(
            f"  - position {item.input_position}: {item.code}: {item.message}"
            for item in batch.rejections[:10]
        )
    return "\n".join(lines)


def render_table(table: RiskTable, *, limit: int = 10) -> str:
    lines = [
        f"Top {min(limit, len(table.records))} of {len(table.records)} assessed customers",
        f"{'rank':>4}  {'reference':<24} {'probability':>11} {'class':>8} {'tier':<9}",
    ]
    for item in table.records[: max(0, limit)]:
        lines.append(
            f"{item.rank or 0:>4}  {item.record_reference:<24} "
            f"{item.churn_probability:>11.4f} {item.predicted_class:>8} {item.risk_tier:<9}"
        )
    return "\n".join(lines)


__all__ = ["render_batch_summary", "render_summary", "render_table"]
