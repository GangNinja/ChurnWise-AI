"""Campaign Simulator: hypothetical projections under explicit assumptions.

The simulator combines stored churn probabilities with user-supplied economics.
It never claims measured uplift, causal effects or realised revenue.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    dict_to_json_bytes,
    format_count,
    format_money,
    format_percent,
    format_probability,
    nested_get,
)
from churnwise.dashboard.components import charts, layout, tables

STRATEGIES = {
    "highest_risk": "Highest risk",
    "risk_threshold": "Risk threshold",
    "risk_category": "Risk category",
    "budget_constrained": "Budget constrained",
}


def _normalize(payload: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    payload = payload or {}
    if "result" in payload:
        return payload.get("result") or {}, payload.get("comparison"), payload.get("sensitivity")
    return dict(payload), None, None


def _persisted_siblings(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    root = path.parent
    return access.load_json(root / "comparison.json"), access.load_json(root / "sensitivity.json")


def _with_default(value: Any, fallback: Any) -> Any:
    """Use a fallback only when a persisted value is absent, never when it is zero."""

    return fallback if value is None else value


def _defaults(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    result = payload or {}
    assumptions = result.get("assumptions") or {}
    targeting = result.get("targeting") or {}
    return {
        "budget": float(_with_default(result.get("budget"), 1000.0)),
        "cost_per_contact": float(_with_default(assumptions.get("cost_per_contact"), 5.0)),
        "effectiveness": float(_with_default(assumptions.get("effectiveness"), 0.30)),
        "monthly_value": assumptions.get("monthly_value"),
        "horizon_months": float(_with_default(assumptions.get("horizon_months"), 6.0)),
        "value_unit": str(_with_default(assumptions.get("value_unit"), "value_units")),
        "strategy": str(_with_default(targeting.get("strategy"), "highest_risk")),
        "min_risk_probability": targeting.get("min_risk_probability"),
        "risk_tiers": tuple(targeting.get("risk_tiers") or ()),
        "max_contacts": targeting.get("max_contacts"),
    }


def _autofill_example() -> None:
    """Fill a varied, plausible example while leaving scoring explicit."""
    rng = random.SystemRandom()
    budget = float(rng.randrange(500, 5001, 100))
    previous = st.session_state.get("cw_sim_autofill_last") or {}
    if budget == previous.get("budget"):
        budget = 500.0 if budget >= 5000.0 else budget + 100.0
    cost = float(rng.choice([3, 5, 8, 10, 12, 15]))
    effectiveness = round(rng.uniform(0.20, 0.65), 2)
    tier_options = ["low", "medium", "high", "critical"]
    tiers = rng.sample(tier_options, k=rng.randint(1, 3))
    values = {
        "cw_sim_budget": budget,
        "cw_sim_cost": cost,
        "cw_sim_effectiveness": effectiveness,
        "cw_sim_horizon": float(rng.choice([3, 6, 9, 12])),
        "cw_sim_strategy": rng.choice(list(STRATEGIES)),
        "cw_sim_min_risk": round(rng.uniform(0.20, 0.60), 2),
        "cw_sim_tiers": tiers,
        "cw_sim_max_contacts": rng.choice([0, 50, 100, 250, 500]),
        "cw_sim_use_value": True,
        "cw_sim_monthly_value": float(rng.randrange(25, 251, 5)),
        "cw_sim_value_unit": str(st.session_state.get("cw_sim_value_unit") or "value_units"),
        "cw_sim_compare": True,
        "cw_sim_alt_budget": round(budget * rng.uniform(1.25, 2.0), 2),
        "cw_sim_alt_effectiveness": min(round(effectiveness + rng.uniform(0.03, 0.15), 2), 1.0),
        "cw_sim_compare_strategies": bool(rng.getrandbits(1)),
    }
    st.session_state["cw_sim_autofill_last"] = {
        "budget": values["cw_sim_budget"],
        "cost": values["cw_sim_cost"],
        "effectiveness": values["cw_sim_effectiveness"],
        "horizon": values["cw_sim_horizon"],
        "monthly_value": values["cw_sim_monthly_value"],
    }
    for key, value in values.items():
        st.session_state[key] = value


def _compute_projection(settings: Settings, risk_run: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return actions.run_campaign_projection(
        settings,
        risk_run_id=risk_run,
        budget=float(values["budget"]),
        cost_per_contact=float(values["cost_per_contact"]),
        effectiveness=float(values["effectiveness"]),
        monthly_value=values.get("monthly_value"),
        horizon_months=float(values["horizon_months"]),
        value_unit=str(values["value_unit"]),
        strategy=str(values["strategy"]),
        min_risk_probability=float(values["min_risk_probability"]),
        risk_tiers=tuple(values.get("risk_tiers") or ()),
        max_contacts=int(values.get("max_contacts") or 0) or None,
        comparison=(
            {
                "budget": float(values["alt_budget"]),
                "effectiveness": float(values["alt_effectiveness"]),
                "cost_per_contact": float(values["cost_per_contact"]),
                "monthly_value": values.get("monthly_value"),
                "horizon_months": float(values["horizon_months"]),
                "value_unit": str(values["value_unit"]),
                "strategy": str(values["strategy"]),
                "min_risk_probability": float(values["min_risk_probability"]),
                "risk_tiers": tuple(values.get("risk_tiers") or ()),
                "max_contacts": int(values.get("max_contacts") or 0) or None,
            }
            if values.get("compare")
            else None
        ),
        comparisons=(
            [
                {
                    "strategy": candidate,
                    "budget": float(values["budget"]),
                    "cost_per_contact": float(values["cost_per_contact"]),
                    "effectiveness": float(values["effectiveness"]),
                    "monthly_value": values.get("monthly_value"),
                    "horizon_months": float(values["horizon_months"]),
                    "value_unit": str(values["value_unit"]),
                    "min_risk_probability": float(values["min_risk_probability"]),
                    "risk_tiers": tuple(values.get("risk_tiers") or ()),
                    "max_contacts": int(values.get("max_contacts") or 0) or None,
                }
                for candidate in STRATEGIES
                if candidate != str(values["strategy"])
            ]
            if values.get("compare_strategies")
            else []
        ),
    )


def _metric_block(result: Mapping[str, Any]) -> None:
    allocation = result.get("allocation") or {}
    projection = result.get("projection") or {}
    assumptions = result.get("assumptions") or {}
    unit = str(assumptions.get("value_unit") or "value_units")
    layout.metric_row(
        [
            ("Selected customers", format_count(allocation.get("selected_count")), "Contacts under the scenario"),
            (
                f"Allocated spend ({unit})",
                f"{float(allocation.get('allocated_spend') or 0):,.2f}",
                "Sum of selected costs",
            ),
            (
                f"Remaining budget ({unit})",
                f"{float(allocation.get('remaining_budget') or 0):,.2f}",
                "Budget minus spend",
            ),
            (
                "Projected prevented churn",
                format_probability(projection.get("expected_prevented_churn")),
                "Σ(probability × assumed effectiveness); a fractional expectation",
            ),
        ]
    )
    financial = projection.get("financial_metrics_available")
    if not financial:
        st.caption(
            "Financial projection is unavailable until a monthly value assumption is supplied."
        )
        return
    with st.expander("Financial projection · based on supplied assumptions", expanded=True):
        layout.metric_row(
            [
                (
                    f"Expected saved value ({unit})",
                    f"{float(projection.get('expected_saved_value') or 0):,.2f}",
                    "Hypothetical",
                ),
                (
                    f"Net benefit ({unit})",
                    f"{float(projection.get('net_benefit') or 0):,.2f}",
                    "Saved value minus spend",
                ),
                (
                    "ROI",
                    format_percent(projection.get("roi"), digits=2)
                    if projection.get("roi") is not None
                    else "undefined",
                    "Net benefit / spend",
                ),
                (
                    "Break-even effectiveness",
                    format_percent(projection.get("break_even_effectiveness"), digits=2)
                    if projection.get("break_even_effectiveness") is not None
                    else "undefined",
                    "Assumed effectiveness needed to cover spend",
                ),
            ]
        )
        undefined = projection.get("undefined_metrics") or {}
        if undefined:
            st.markdown("**Undefined values**")
            for key, reason in undefined.items():
                st.write(f"- **{key}**: {reason}")


def _decision_brief(settings: Settings, result: Mapping[str, Any]) -> None:
    """Show run-specific population context and evidence-backed campaign ideas."""
    run_id = str(nested_get(result, "input", "risk_run_id", default="") or "")
    if not run_id:
        return
    scores, _ = access.risk_table(settings, run_id)
    layout.section("Campaign planning brief", "Use the selected scoring run to ground targeting and intervention tests.")
    if scores is None or scores.empty:
        st.info("The selected run has no readable score table, so population insights are unavailable.")
        return

    tiers = scores.get("risk_tier", pd.Series(dtype=str)).astype(str).str.lower()
    tier_counts = tiers.value_counts()
    columns = st.columns(4)
    for column, tier in zip(columns, ("critical", "high", "medium", "low")):
        column.metric(f"{tier.title()} risk", format_count(tier_counts.get(tier, 0)))

    allocation = result.get("allocation") or {}
    assumptions = result.get("assumptions") or {}
    budget = float(result.get("budget") or 0)
    cost = float(assumptions.get("cost_per_contact") or 0)
    affordable = int(budget // cost) if cost > 0 else int(allocation.get("selected_count") or 0)
    selected = result.get("selected_customers") or []
    st.markdown("**Budget levers**")
    st.write(
        f"At {format_money(cost, assumptions.get('value_unit', 'value_units'))} per contact, "
        f"the budget can cover up to {format_count(affordable)} contacts before eligibility and contact limits. "
        f"This scenario selects {format_count(allocation.get('selected_count'))} customers; "
        f"the projection above shows the resulting spend and assumption-based value."
    )
    st.caption("Try lowering cost per contact, increasing budget, or changing targeting, then compare the projected trade-offs.")

    driver_payload, _ = access.risk_drivers(settings, run_id)
    selected_refs = {str(row.get("record_reference")) for row in selected}
    driver_counts: dict[str, list[float]] = {}
    if isinstance(driver_payload, Mapping):
        for reference, item in driver_payload.items():
            if not selected_refs or str(reference) not in selected_refs:
                continue
            if not isinstance(item, Mapping) or item.get("status") != "available":
                continue
            for driver in item.get("top_positive") or []:
                if not isinstance(driver, Mapping):
                    continue
                feature = str(driver.get("feature") or "").strip()
                contribution = pd.to_numeric(driver.get("contribution"), errors="coerce")
                if feature and pd.notna(contribution) and float(contribution) > 0:
                    driver_counts.setdefault(feature, []).append(float(contribution))

    st.markdown("**Intervention ideas from this run**")
    if driver_counts:
        rows = sorted(driver_counts.items(), key=lambda pair: (-len(pair[1]), -sum(pair[1]) / len(pair[1]), pair[0]))[:5]
        ideas = []
        for feature, values in rows:
            ideas.append({
                "Model-associated factor": feature,
                "Selected customers with this driver": len(values),
                "Test idea": f"Pilot a tailored outreach or service check for customers flagged by {feature}; compare against standard outreach.",
            })
        st.dataframe(pd.DataFrame(ideas), width="stretch", hide_index=True)
        st.caption("Factors summarize stored positive SHAP drivers for selected customers. They describe model associations, not causes; test interventions before scaling.")
    else:
        st.info("This scoring run has no stored local explanations. Use the risk tiers to target a pilot, or rescore with explanations to see model-associated factors here.")

    strategy_ideas = pd.DataFrame([
        {"Strategy": "Highest risk", "Use when": "Budget is tight and you want to contact the largest estimated risks first."},
        {"Strategy": "Risk threshold", "Use when": "You want a minimum churn-probability floor before spending."},
        {"Strategy": "Risk category", "Use when": "Your team has capacity for selected risk tiers only."},
        {"Strategy": "Budget constrained", "Use when": "Contact costs vary and the next customer must fit the remaining budget."},
    ])
    with st.expander("Choose a targeting technique"):
        st.dataframe(strategy_ideas, width="stretch", hide_index=True)


def render(settings: Settings) -> None:
    runs = access.risk_ids(settings)
    run_labels = access.risk_run_labels(settings, runs)
    if not runs:
        layout.page_header(
            "Campaign Simulator",
            "Campaign projections require a scored population. No risk run is available.",
            access.RunIdentity(source="campaign simulation"),
        )
        layout.empty_state(
            "No risk run to simulate against",
            "Score a population first; the simulator only combines stored probabilities with your assumptions.",
            "python -m churnwise.cli score --population val --explain none",
        )
        return

    persisted, persisted_path = access.simulation_result(settings)
    persisted_result, persisted_comparison, persisted_sensitivity = _normalize(persisted)
    sibling_comparison, sibling_sensitivity = _persisted_siblings(persisted_path)
    persisted_comparison = persisted_comparison or sibling_comparison
    persisted_sensitivity = persisted_sensitivity or sibling_sensitivity
    computed = st.session_state.get("cw_simulation_run")
    computed_result, computed_comparison, computed_sensitivity = _normalize(computed)
    active = computed_result or persisted_result
    # A newly computed run owns its comparison/sensitivity payload, even when
    # either is None. Falling back here could pair current KPIs with stale plots.
    active_comparison = computed_comparison if computed else persisted_comparison
    active_sensitivity = computed_sensitivity if computed else persisted_sensitivity
    defaults = _defaults(active)
    source_risk = str(nested_get(active, "input", "risk_run_id", default="") or "")
    if source_risk in runs:
        default_run_index = runs.index(source_risk)
    else:
        fallback = access.default_risk_run_id(settings, runs)
        default_run_index = runs.index(fallback) if fallback in runs else 0
    default_risk_run = runs[default_run_index]
    initial_values = {
        "cw_sim_risk_run": default_risk_run,
        "cw_sim_budget": defaults["budget"],
        "cw_sim_cost": defaults["cost_per_contact"],
        "cw_sim_effectiveness": defaults["effectiveness"],
        "cw_sim_horizon": defaults["horizon_months"],
        "cw_sim_strategy": defaults["strategy"] if defaults["strategy"] in STRATEGIES else "highest_risk",
        "cw_sim_min_risk": defaults["min_risk_probability"] if defaults["min_risk_probability"] is not None else 0.30,
        "cw_sim_tiers": list(defaults["risk_tiers"]),
        "cw_sim_max_contacts": int(defaults["max_contacts"] or 0),
        "cw_sim_use_value": defaults["monthly_value"] is not None,
        "cw_sim_monthly_value": float(defaults["monthly_value"] if defaults["monthly_value"] is not None else 50.0),
        "cw_sim_value_unit": defaults["value_unit"],
        "cw_sim_compare": False,
        "cw_sim_alt_budget": float(defaults["budget"]) * 1.5,
        "cw_sim_alt_effectiveness": min(float(defaults["effectiveness"]) + 0.05, 1.0),
        "cw_sim_compare_strategies": False,
    }
    for key, value in initial_values.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("cw_sim_risk_run") not in runs:
        st.session_state["cw_sim_risk_run"] = default_risk_run
    st.session_state.setdefault("cw_sim_applied_inputs", {
        "risk_run_id": default_risk_run,
        "budget": defaults["budget"],
        "cost_per_contact": defaults["cost_per_contact"],
        "effectiveness": defaults["effectiveness"],
        "monthly_value": defaults["monthly_value"],
        "horizon_months": defaults["horizon_months"],
        "value_unit": defaults["value_unit"],
        "strategy": defaults["strategy"],
        "min_risk_probability": initial_values["cw_sim_min_risk"],
        "risk_tiers": defaults["risk_tiers"],
        "max_contacts": defaults["max_contacts"] or 0,
        "compare": False,
        "alt_budget": initial_values["cw_sim_alt_budget"],
        "alt_effectiveness": initial_values["cw_sim_alt_effectiveness"],
        "compare_strategies": False,
    })
    identity = access.identity_from_simulation(computed_result if computed else persisted_result)
    if not identity.run_id and active:
        identity = access.identity_from_simulation(active)

    layout.page_header(
        "Campaign Simulator",
        "Compare hypothetical retention scenarios built from stored churn probabilities and your own assumptions. "
        "Every retention and financial value is a projection, not an observed campaign result.",
        identity,
    )
    if st.session_state.get("cw_simulation_error"):
        st.error(st.session_state.pop("cw_simulation_error"))
    if computed:
        st.info("Showing a projection computed in this session. It is not persisted.")
    elif persisted:
        st.caption(f"Showing the latest persisted scenario `{persisted_result.get('scenario_id')}`.")
    layout.disclaimer(
        "Hypothetical projection. Assumed effectiveness is an input, not measured uplift. "
        "No causal model is fitted and no campaign is executed."
    )

    with st.expander("Assumptions and targeting", expanded=not bool(active)):
        st.button("Autofill example assumptions", on_click=_autofill_example, key="cw_sim_autofill")
        st.caption("Autofill generates varied example assumptions; review them before running a projection.")
        risk_run = st.selectbox(
            "Scoring run", runs, key="cw_sim_risk_run",
            format_func=lambda value: run_labels.get(value, value),
            help="Changing this run refreshes the outcome using the last applied assumptions.",
        )
        applied_inputs = dict(st.session_state.get("cw_sim_applied_inputs") or {})
        if risk_run != applied_inputs.get("risk_run_id"):
            try:
                st.session_state["cw_simulation_run"] = _compute_projection(settings, risk_run, applied_inputs)
            except actions.DashboardActionError as exc:
                st.session_state["cw_simulation_error"] = (
                    f"{exc.code}: {exc.message} The displayed projection is still for the previous scoring run."
                    + (f" — {exc.remediation}" if exc.remediation else "")
                )
                applied_inputs["risk_run_id"] = risk_run
                st.session_state["cw_sim_applied_inputs"] = applied_inputs
            else:
                applied_inputs["risk_run_id"] = risk_run
                st.session_state["cw_sim_applied_inputs"] = applied_inputs
            st.rerun()

        compare = st.checkbox(
            "Compare with a higher-budget alternative",
            key="cw_sim_compare",
            help="Turn this on to edit the alternative budget and effectiveness below.",
        )
        with st.form("cw_simulation_form"):
            cols = st.columns(3)
            with cols[0]:
                budget = st.number_input("Campaign budget", min_value=0.0, step=100.0, key="cw_sim_budget")
                cost = st.number_input(
                    "Cost per contacted customer", min_value=0.0, step=1.0, key="cw_sim_cost"
                )
                effectiveness = st.slider(
                    "Assumed retention effectiveness (input, not measured uplift)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key="cw_sim_effectiveness",
                )
                horizon = st.number_input(
                    "Value horizon (months)", min_value=1.0, step=1.0, key="cw_sim_horizon"
                )
            with cols[1]:
                strategy = st.selectbox(
                    "Targeting strategy",
                    list(STRATEGIES),
                    format_func=lambda value: STRATEGIES[value],
                    key="cw_sim_strategy",
                )
                min_risk = st.number_input(
                    "Minimum risk probability",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key="cw_sim_min_risk",
                )
                tiers = st.multiselect(
                    "Targeted risk tiers",
                    options=["low", "medium", "high", "critical"],
                    help="Leave empty to target every tier (subject to the probability floor).",
                    key="cw_sim_tiers",
                )
                max_contacts = st.number_input(
                    "Maximum contacts (0 = no cap)",
                    min_value=0,
                    max_value=10_000_000,
                    step=10,
                    key="cw_sim_max_contacts",
                )
            with cols[2]:
                use_value = st.checkbox(
                    "Supply a monthly value",
                    help="Without a value, saved value, net benefit, ROI and break-even stay undefined.",
                    key="cw_sim_use_value",
                )
                monthly_value = st.number_input(
                    "Assumed monthly value per retained customer",
                    min_value=0.0,
                    step=5.0,
                    disabled=not use_value,
                    key="cw_sim_monthly_value",
                )
                value_unit = st.text_input("Value unit", max_chars=24, key="cw_sim_value_unit")
                alt_budget = st.number_input(
                    "Alternative budget",
                    min_value=0.0,
                    step=100.0,
                    disabled=not compare,
                    key="cw_sim_alt_budget",
                )
                alt_effectiveness = st.slider(
                    "Alternative assumed effectiveness",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    disabled=not compare,
                    key="cw_sim_alt_effectiveness",
                )
                compare_strategies = st.checkbox("Compare all targeting strategies", key="cw_sim_compare_strategies")
            submitted = st.form_submit_button("Run projection", type="primary")
        if submitted:
            try:
                scenario = actions.run_campaign_projection(
                    settings,
                    risk_run_id=risk_run,
                    budget=budget,
                    cost_per_contact=cost,
                    effectiveness=effectiveness,
                    monthly_value=monthly_value if use_value else None,
                    horizon_months=horizon,
                    value_unit=value_unit.strip() or "value_units",
                    strategy=strategy,
                    min_risk_probability=min_risk,
                    risk_tiers=tuple(tiers),
                    max_contacts=int(max_contacts) or None,
                    comparison=(
                        {
                            "budget": alt_budget,
                            "effectiveness": alt_effectiveness,
                            "cost_per_contact": cost,
                            "monthly_value": monthly_value if use_value else None,
                            "horizon_months": horizon,
                            "value_unit": value_unit.strip() or "value_units",
                            "strategy": strategy,
                            "min_risk_probability": min_risk,
                            "risk_tiers": tuple(tiers),
                            "max_contacts": int(max_contacts) or None,
                        }
                        if compare
                        else None
                    ),
                    comparisons=(
                        [
                            {
                                "strategy": candidate,
                                "budget": budget,
                                "cost_per_contact": cost,
                                "effectiveness": effectiveness,
                                "monthly_value": monthly_value if use_value else None,
                                "horizon_months": horizon,
                                "value_unit": value_unit.strip() or "value_units",
                                "min_risk_probability": min_risk,
                                "risk_tiers": tuple(tiers),
                                "max_contacts": int(max_contacts) or None,
                            }
                            for candidate in STRATEGIES
                            if candidate != strategy
                        ]
                        if compare_strategies
                        else []
                    ),
                )
            except actions.DashboardActionError as exc:
                st.session_state["cw_simulation_error"] = (
                    f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else "")
                )
                st.rerun()
            else:
                st.session_state["cw_simulation_run"] = scenario
                st.session_state["cw_sim_applied_inputs"] = {
                    "risk_run_id": risk_run,
                    "budget": budget,
                    "cost_per_contact": cost,
                    "effectiveness": effectiveness,
                    "monthly_value": monthly_value if use_value else None,
                    "horizon_months": horizon,
                    "value_unit": value_unit.strip() or "value_units",
                    "strategy": strategy,
                    "min_risk_probability": min_risk,
                    "risk_tiers": tuple(tiers),
                    "max_contacts": int(max_contacts) or 0,
                    "compare": compare,
                    "alt_budget": alt_budget,
                    "alt_effectiveness": alt_effectiveness,
                    "compare_strategies": compare_strategies,
                }
                st.rerun()

    if not active:
        layout.empty_state(
            "No simulation has been computed or persisted",
            "Set the assumptions above and run a projection, or use the Phase 10 CLI.",
            "python -m churnwise.cli simulate --budget 1000 --cost-per-contact 5 --effectiveness 0.30 --monthly-value 50",
        )
        return

    scenario_name = nested_get(active, "scenario", "name", default="unnamed scenario")
    layout.section(
        "Projected outcome",
        f"Scenario `{scenario_name}` · strategy {nested_get(active, 'targeting', 'strategy', default='not recorded')}.",
    )
    _metric_block(active)
    layout.section(
        "Projection charts",
        "Visuals below use the selected scoring run and the assumptions applied to this projection.",
    )
    visual_columns = st.columns(2)
    projection = active.get("projection") or {}
    allocation = active.get("allocation") or {}
    assumptions = active.get("assumptions") or {}
    unit = str(assumptions.get("value_unit") or "value_units")
    if projection.get("financial_metrics_available"):
        with visual_columns[0]:
            st.markdown("**Value bridge**")
            st.plotly_chart(
                charts.financial_bridge_chart(projection, allocation, unit),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption("Expected saved value minus allocated spend; both depend on your assumptions.")
    else:
        with visual_columns[0]:
            st.markdown("**Budget allocation**")
            st.plotly_chart(
                charts.allocation_chart(allocation, unit),
                width="stretch",
                config={"displayModeBar": False},
            )

    comparison_rows = (active_comparison or {}).get("rows") or []
    if comparison_rows:
        with visual_columns[1]:
            st.markdown("**Scenario comparison · projected prevented churn**")
            st.plotly_chart(
                charts.comparison_chart(comparison_rows, unit),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(
                "Scenarios use the selected scoring run; differences come from their listed assumptions. "
                "All values are projections."
            )
    else:
        run_id = str(nested_get(active, "input", "risk_run_id", default="") or "")
        score_frame, _ = access.risk_table(settings, run_id) if run_id else (None, None)
        if score_frame is not None and "risk_tier" in score_frame:
            counts = score_frame["risk_tier"].astype(str).str.lower().value_counts().to_dict()
            with visual_columns[1]:
                st.markdown("**Scored population by risk tier**")
                st.plotly_chart(
                    charts.tier_chart(counts),
                    width="stretch",
                    config={"displayModeBar": False},
                )
    _decision_brief(settings, active)
    population = active.get("population") or {}
    count_rows = [
        {"Measure": "Assessed", "Value": population.get("assessed")},
        {"Measure": "Eligible", "Value": population.get("eligible")},
        {"Measure": "Selected", "Value": population.get("selected")},
        {"Measure": "Excluded", "Value": population.get("excluded")},
        {"Measure": "Untargeted (budget)", "Value": (active.get("allocation") or {}).get("unaffordable_count")},
        {"Measure": "Untargeted (contact cap)", "Value": (active.get("allocation") or {}).get("contact_limited_count")},
    ]
    with st.expander("Population allocation details"):
        st.caption("Counts reconcile" if population.get("counts_reconcile") else "Counts did not reconcile")
        st.dataframe(pd.DataFrame(count_rows), width="stretch", hide_index=True)

    assumptions = active.get("assumptions") or {}
    targeting = active.get("targeting") or {}
    unit = assumptions.get("value_unit", "value_units")
    assumption_rows = [
        ("Budget (input)", format_money(active.get("budget"), unit)),
        ("Cost per contact (input)", format_money(assumptions.get("cost_per_contact"), unit)),
        ("Assumed effectiveness (input)", format_percent(assumptions.get("effectiveness"))),
        (
            "Monthly value (input)",
            format_money(assumptions.get("monthly_value"), unit)
            if assumptions.get("monthly_value") is not None
            else "not supplied",
        ),
        ("Horizon (input)", f"{assumptions.get('horizon_months', '?')} months"),
        ("Minimum risk (input)", format_probability(targeting.get("min_risk_probability"))),
        ("Risk tiers (input)", ", ".join(targeting.get("risk_tiers") or []) or "all tiers"),
        ("Contact cap (input)", targeting.get("max_contacts") or "none"),
    ]
    with st.expander("Inputs applied to this projection"):
        st.caption(
            "Every row below is a user-supplied input or a direct reference to the stored risk run. "
            "None of these values is a measured campaign result."
        )
        st.dataframe(
            pd.DataFrame(assumption_rows, columns=["Assumption", "Value"]),
            width="stretch",
            hide_index=True,
        )

    series = (active_sensitivity or {}).get("series") or []
    if series:
        with st.expander("Sensitivity", expanded=False):
            st.caption("Each point is a genuine recomputation with one assumption changed.")
            tab_chart, tab_table = st.tabs(["Chart", "Table"])
            with tab_chart:
                st.plotly_chart(charts.sensitivity_chart(series), width="stretch")
            with tab_table:
                st.dataframe(tables.sensitivity_frame(series), width="stretch", hide_index=True)
            warnings = (active_sensitivity or {}).get("warnings") or []
            if warnings:
                st.caption(" ".join(str(item) for item in warnings))
    else:
        st.caption("No sensitivity series is attached to this result.")

    if active_comparison and active_comparison.get("rows"):
        layout.section("Scenario comparison", "Assumption differences stay visible; neither row is evidence of real performance.")
        comparison_frame = pd.DataFrame(active_comparison["rows"])
        keep = [
            column
            for column in (
                "name",
                "targeting_strategy",
                "budget",
                "cost_per_contact",
                "assumed_effectiveness",
                "selected_count",
                "allocated_spend",
                "expected_prevented_churn",
                "expected_saved_value",
                "net_benefit",
                "roi",
            )
            if column in comparison_frame
        ]
        display = comparison_frame[keep].rename(
            columns={
                "name": "Scenario",
                "targeting_strategy": "Targeting strategy",
                "budget": "Budget",
                "cost_per_contact": "Cost / contact",
                "assumed_effectiveness": "Assumed effectiveness",
                "selected_count": "Selected",
                "allocated_spend": "Spend",
                "expected_prevented_churn": "Prevented churn",
                "expected_saved_value": f"Saved value ({unit})",
                "net_benefit": f"Net benefit ({unit})",
                "roi": "ROI",
            }
        )
        if "ROI" in display:
            roi = pd.to_numeric(display["ROI"], errors="coerce")
            display["ROI"] = roi.map(lambda value: f"{value:.1%}" if pd.notna(value) else "undefined")
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption(f"Budget, cost per contact and spend use `{unit}`. Prevented churn is an expected value.")
        if active_comparison.get("shared_inputs") is False:
            st.warning("The scenarios do not share all risk inputs; see the comparison notes.")
        for text in active_comparison.get("notes") or []:
            st.caption(str(text))

    selected = active.get("selected_customers") or []
    if selected:
        with st.expander(f"Selected customers ({format_count(len(selected))})"):
            st.dataframe(tables.selected_customers_frame(selected), width="stretch", hide_index=True)

    warnings = active.get("warnings") or []
    if warnings:
        with st.expander(f"Projection warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
    limitations = active.get("limitations") or []
    if limitations:
        with st.expander("Projection limitations"):
            for item in limitations:
                st.write(f"- {item}")

    download_payload = computed if computed else persisted
    if download_payload:
        st.download_button(
            "Download result JSON",
            data=dict_to_json_bytes(download_payload),
            file_name=f"simulation_{identity.run_id or 'result'}.json",
            mime="application/json",
            key="cw_simulation_download",
        )
