"""Overview: verified project state, not a decorative landing page."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    format_count,
    format_money,
    format_percent,
    format_probability,
    nested_get,
)
from churnwise.dashboard.components import charts, layout, tables


def _model_card(metrics: Mapping[str, Any] | None, path) -> str:
    if not metrics:
        return "No benchmark artifact"
    selection = metrics.get("selection") or {}
    family = selection.get("selected_family")
    family_metrics = (metrics.get("families") or {}).get(family) or {}
    test = family_metrics.get("test_metrics") or {}
    if not test:
        return "Benchmark present, test metrics unavailable"
    return (
        f"{family_metrics.get('label') or family} · F1 {test.get('f1', 0):.4f} · "
        f"ROC-AUC {test.get('roc_auc', 0):.4f} · Brier {test.get('brier', 0):.4f}"
    )


def render(settings: Settings) -> None:
    validation, validation_path = access.validation_report(settings)
    eda, _ = access.eda_profile(settings, str((validation or {}).get("dataset_id") or "") or None)
    prepared, _ = access.prepared_summary(settings)
    metadata, metadata_path = access.model_metadata(settings)
    metrics, metrics_path = access.benchmark_metrics(settings)
    explanation, explanation_path = access.explanation_global(settings)
    risk, risk_path = access.risk_summary(settings)
    risk_manifest, _ = access.risk_manifest(settings)
    simulation, simulation_path = access.simulation_result(settings)

    identity = (
        access.identity_from_risk(risk, risk_manifest)
        if risk
        else access.identity_from_benchmark(metrics)
        if metrics
        else access.identity_from_validation(validation)
    )
    layout.page_header(
        "Intelligence overview",
        "A verified snapshot of the dataset, model, risk population and latest campaign projection. "
        "Every value below is loaded from a persisted artifact; missing evidence is shown as unavailable.",
        identity,
    )

    if not any((validation, metrics, risk, simulation, metadata)):
        layout.empty_state(
            "No project artifacts are available yet",
            "Run the pipeline in order, then refresh this page. The dashboard never substitutes sample numbers.",
            "python -m churnwise.cli discover && python -m churnwise.cli acquire && "
            "python -m churnwise.cli validate && python -m churnwise.cli prepare && "
            "python -m churnwise.cli train && python -m churnwise.cli evaluate",
        )
        return

    summary = (validation or {}).get("summary") or {}
    gate = (validation or {}).get("gate") or {}
    target = (validation or {}).get("target") or {}
    selected_family = (metrics or {}).get("selection") or {}

    layout.section("Dataset", "Provenance, shape and the Phase 4 validation decision.")
    layout.status_line(
        "Validation gate",
        str(gate.get("status") or "not available").replace("_", " "),
        "ok" if gate.get("passed") else "warn",
        "; ".join(gate.get("reasons") or [])[:220],
    )
    layout.metric_row(
        [
            ("Records", format_count(summary.get("n_rows")), "Rows in the validated artifact"),
            ("Features", format_count(summary.get("n_cols")), "Columns before preprocessing"),
            (
                "Churn rate",
                format_percent(target.get("positive_rate")),
                f"Positive class: {target.get('positive_label') or 'unknown'}",
            ),
            (
                "Usable predictors",
                format_count(len(summary.get("usable_features") or [])),
                "Features retained by the validation roles",
            ),
        ]
    )

    layout.section("Selected model", "Phase 7 model selection was made on validation F1, before test scoring.")
    if metrics:
        family = selected_family.get("selected_family")
        family_metrics = (metrics.get("families") or {}).get(family) or {}
        test = family_metrics.get("test_metrics") or {}
        layout.metric_row(
            [
                ("Family", family_metrics.get("label") or family or "not available", "Selected on validation"),
                ("Test F1", f"{test.get('f1', float('nan')):.4f}" if test else "not available", "Held-out test split"),
                (
                    "Test ROC-AUC",
                    f"{test.get('roc_auc', float('nan')):.4f}" if test else "not available",
                    "Held-out test split",
                ),
                (
                    "Threshold",
                    f"{selected_family.get('threshold', float('nan')):.2f}"
                    if selected_family.get("threshold") is not None
                    else "not available",
                    "Frozen on validation before test scoring",
                ),
            ]
        )
        st.caption(
            f"Benchmark `{metrics.get('benchmark_run_id')}` generated {metrics.get('generated_at')}; "
            f"test rows: {format_count(nested_get(metrics, 'dataset', 'test_rows_used'))}."
        )
    elif metadata:
        st.info(
            f"An MLP artifact exists (`{metadata.get('model_run_id')}`) but no benchmark has been run, "
            "so no performance metric is claimed here."
        )
    else:
        layout.empty_state(
            "No trained or evaluated model",
            "Train the MLP and run the comparative benchmark to populate this section.",
            "python -m churnwise.cli prepare && python -m churnwise.cli train && "
            "python -m churnwise.cli evaluate",
        )

    layout.section("Risk population", "Phase 9 scores, tiers and driver coverage from the latest risk run.")
    if risk:
        tier_counts = risk.get("tier_counts") or {}
        predicted = risk.get("predicted_class_counts") or {}
        high_critical = int((tier_counts.get("high") or {}).get("count", 0)) + int(
            (tier_counts.get("critical") or {}).get("count", 0)
        )
        explanations = risk.get("explanations") or {}
        layout.metric_row(
            [
                ("Scored customers", format_count(risk.get("n_rows")), "Rows receiving a probability"),
                ("Predicted churn", format_count(predicted.get("churn")), "At the frozen classification threshold"),
                ("High + critical", format_count(high_critical), "Risk tiers needing attention"),
                (
                    "With drivers",
                    format_count(explanations.get("available")),
                    "Rows with Phase 8 SHAP driver attachment",
                ),
            ]
        )
        st.plotly_chart(charts.tier_chart(tier_counts), width="stretch")
        st.caption(f"Risk run `{risk.get('run_id')}` generated {risk.get('generated_at')}.")
    else:
        layout.empty_state(
            "No risk run exists",
            "Score the validation population with Phase 9; the dashboard cannot derive risk scores itself.",
            "python -m churnwise.cli score --population val --explain all",
        )

    layout.section("Latest campaign projection", "Phase 10 output is hypothetical and assumption-driven.")
    if simulation:
        allocation = simulation.get("allocation") or {}
        projection = simulation.get("projection") or {}
        unit = nested_get(simulation, "assumptions", "value_unit", default="value_units")
        layout.disclaimer(
            "These are hypothetical projections, not observed retention results. Assumed effectiveness is not measured uplift."
        )
        layout.metric_row(
            [
                ("Selected", format_count(allocation.get("selected_count")), "Customers contacted under the scenario"),
                ("Allocated spend", format_money(allocation.get("allocated_spend"), unit), "Sum of selected intervention costs"),
                (
                    "Projected prevented churn",
                    format_probability(projection.get("expected_prevented_churn")),
                    "Fractional expectation Σ(p × assumed effectiveness)",
                ),
                ("Scenario", str(nested_get(simulation, "scenario", "name", default="unnamed")), "Persisted scenario"),
            ]
        )
        source_risk = nested_get(simulation, "input", "risk_run_id", default="not recorded")
        st.caption(
            f"Scenario `{simulation.get('scenario_id')}` generated {simulation.get('generated_at')} "
            f"from risk run `{source_risk}` (not necessarily the risk population shown above); "
            "open Retention simulator to compute a new projection."
        )
    else:
        layout.empty_state(
            "No campaign simulation has been persisted",
            "Open the retention simulator to compute a projection, or run the Phase 10 command.",
            "python -m churnwise.cli simulate --budget 1000 --cost-per-contact 5 --effectiveness 0.30",
        )

    layout.section("Evidence inventory")
    inventory = access.artifact_inventory(settings)
    frame = []
    for entry in inventory:
        frame.append(
            {
                "Artifact": entry.kind,
                "Run / id": entry.run_id,
                "Generated": entry.generated_at or "not recorded",
                "Status": "present" if entry.exists else "missing",
                "Detail": entry.detail,
            }
        )
    if frame:
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.caption("No artifact inventory is available.")

    warnings = list((validation or {}).get("warnings") or [])
    if risk:
        warnings.extend(risk.get("warnings") or [])
    if metrics:
        warnings.extend(metrics.get("warnings") or [])
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
