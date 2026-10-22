"""Human-readable summaries for Phase 8 explanation runs."""

from __future__ import annotations

from typing import Any

from churnwise.explain.contracts import ExplainResult


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def render_summary(result: ExplainResult) -> str:
    """Render a short, evidence-only CLI summary."""
    lines = [
        "ChurnWise AI - Phase 8 explanations",
        "",
        f"Explanation run: {result.run_id}",
        f"Model family: {result.family}",
    ]
    block = result.global_explanation
    if block is not None:
        sample = block.sample or {}
        lines.extend(
            [
                f"Explainer: {block.explainer.get('name')} "
                f"(output space: {block.output_space})",
                f"Sample: {sample.get('sampled_rows')} of {sample.get('available_rows')} "
                f"{sample.get('split')} rows (seed {sample.get('seed')}); "
                f"test rows used: {sample.get('test_rows_used', 0)}",
                f"Max additivity error: {block.additivity_error:.3g}",
                "",
                "Top drivers by mean |SHAP| (source features):",
            ]
        )
        for row in block.feature_importance[:10]:
            lines.append(
                f"  {row['rank']:>2}. {row['feature']:<32} "
                f"{row['mean_abs_shap']:.4f}  ({_pct(row['share'])})  "
                f"{row['direction']}"
            )
        if block.base_value is not None:
            lines.append("")
            lines.append(
                f"Baseline model output ({block.output_space}): "
                f"{block.base_value:.4f}"
            )
    if result.local_explanations:
        lines.extend(["", "Local examples:"])
        for item in result.local_explanations:
            drivers = ", ".join(
                f"{row.feature} {row.contribution:+.3f}"
                for row in item.top_positive[:2]
            )
            lines.append(
                f"  row {item.row_position}: P(churn)={item.probability:.3f} "
                f"class={item.predicted_class}; space={item.output_space}"
                + (f"; top upward: {drivers}" if drivers else "")
            )
    if result.paths:
        lines.extend(["", "Artifacts:"])
        for key in ("global_json", "summary_png", "importance_png", "run_manifest"):
            if result.paths.get(key):
                lines.append(f"  {key}: {result.paths[key]}")
        local_json = [value for key, value in result.paths.items() if key.startswith("local_json")]
        if local_json:
            lines.append(f"  local explanations: {len(local_json)} JSON files")
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in result.warnings)
    limits = (result.manifest or {}).get("limitations") or []
    if limits:
        lines.extend(["", "Limitations:"])
        lines.extend(f"  - {item}" for item in limits)
    return "\n".join(lines) + "\n"


__all__ = ["render_summary"]
