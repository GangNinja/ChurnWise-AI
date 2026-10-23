"""Data discovery: inspect Phase 2 rankings and trigger audited actions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard import data_access as access
from churnwise.dashboard.components import layout


def _candidate_frame(ranking: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in ranking:
        gates = item.get("gates") or {}
        gate_values = list(gates.values()) if isinstance(gates, dict) else list(gates)
        failed = [
            gate.get("gate_id")
            for gate in gate_values
            if isinstance(gate, dict) and gate.get("status") != "pass"
        ]
        description = str(item.get("description") or "").replace("\n", " ")
        rows.append(
            {
                "Rank": item.get("rank"),
                "Candidate": item.get("candidate_id"),
                "Title": item.get("title") or item.get("candidate", {}).get("title"),
                "Source": item.get("source_name") or item.get("source_id"),
                "Publisher": item.get("publisher"),
                "Description": description[:120] + ("…" if len(description) > 120 else ""),
                "Format": item.get("format"),
                "Rows": item.get("row_count"),
                "Features": item.get("feature_count"),
                "Target": item.get("target_column"),
                "Access": item.get("access"),
                "Score": item.get("score"),
                "License": (item.get("license") or {}).get("status")
                if isinstance(item.get("license"), dict)
                else item.get("license"),
                "Eligible": "yes" if item.get("eligible") else "no",
                "Failed gates": ", ".join(failed) if failed else "none",
            }
        )
    return pd.DataFrame(rows)


def render(settings: Settings) -> None:
    report, report_path = access.acquisition_candidates(settings)
    attempts, attempts_path = access.acquisition_attempts(settings)
    identity = access.identity_from_candidates(report)
    layout.page_header(
        "Data discovery",
        "Autonomous discovery ranks public churn datasets with source evidence and suitability gates. "
        "Discovery uses the network only when you explicitly start it; acquisition reuses verified cached artifacts.",
        identity,
    )
    if st.session_state.get("discovery_error"):
        st.error(st.session_state.pop("discovery_error"))
    if st.session_state.get("discovery_notice"):
        st.success(st.session_state.pop("discovery_notice"))
    if st.session_state.get("acquisition_error"):
        st.error(st.session_state.pop("acquisition_error"))
    if st.session_state.get("acquisition_notice"):
        st.success(st.session_state.pop("acquisition_notice"))

    with st.expander("Run discovery", expanded=report is None):
        st.caption(
            "This performs live HTTP requests to the enabled public sources. It can take up to the configured "
            "timeout and writes reports/acquisition/{candidates,rejections}.json."
        )
        with st.form("cw_discovery_form", clear_on_submit=False):
            terms = st.text_input(
                "Search terms",
                value="",
                placeholder="churn, customer churn, telecom churn",
                help="Comma-separated terms. Leave blank to use project defaults.",
            )
            cols = st.columns(2)
            with cols[0]:
                min_rows = st.number_input("Minimum rows", min_value=100, max_value=1_000_000, value=1000, step=100)
            with cols[1]:
                max_results = st.number_input("Max results per source", min_value=1, max_value=50, value=10, step=1)
            submitted = st.form_submit_button("Start discovery", type="primary")
        if submitted:
            try:
                with st.spinner("Querying public sources and evaluating gates…"):
                    actions.discover_datasets(
                        settings,
                        terms=terms,
                        min_rows=int(min_rows),
                        max_results=int(max_results),
                    )
            except actions.DashboardActionError as exc:
                st.session_state["discovery_error"] = (
                    f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else "")
                )
                st.rerun()
            else:
                st.session_state["discovery_notice"] = "Discovery completed. The ranking below is refreshed."
                st.cache_data.clear()
                st.rerun()

    if not report:
        layout.empty_state(
            "No discovery report exists",
            "Start discovery above or run the Phase 2 command. No candidate is fabricated in the meantime.",
            "python -m churnwise.cli discover",
        )
        return

    summary = report.get("summary") or {}
    layout.section("Discovery summary", f"Report generated {report.get('generated_at')}.")
    layout.metric_row(
        [
            ("Found", summary.get("found", "not available"), "Candidates returned by the sources"),
            ("Eligible", summary.get("eligible", "not available"), "Candidates that passed all gates"),
            ("Rejected", summary.get("rejected", "not available"), "Ineligible or duplicate candidates"),
            ("Selected", summary.get("selected") or "none", "Chosen by the documented ranking"),
        ]
    )

    ranking = report.get("ranking") or []
    if ranking:
        layout.section("Ranked candidates", "Gate failures and licence status are shown beside each candidate.")
        st.dataframe(_candidate_frame(ranking), width="stretch", hide_index=True)
        options = {
            f"{item.get('rank')}. {item.get('candidate_id')} — {item.get('title') or ''}": item.get(
                "candidate_id"
            )
            for item in ranking
            if item.get("eligible")
        }
        if options:
            label = st.selectbox("Preferred candidate", list(options), key="cw_acquire_candidate")
            allow_fallback = st.checkbox(
                "Try lower-ranked eligible candidates if this one fails",
                value=True,
                help="Phase 3 records every attempt and stops at the first verified acquisition.",
            )
            st.caption(
                "Acquisition verifies size and SHA-256, records provenance and writes attempts. "
                "Existing verified cache entries are reused."
            )
            if st.button("Acquire selected candidate", key="cw_acquire_button"):
                try:
                    with st.spinner("Acquiring and verifying the candidate…"):
                        result = actions.acquire_candidate(
                            settings, str(options[label]), allow_fallback=allow_fallback
                        )
                except actions.DashboardActionError as exc:
                    st.session_state["acquisition_error"] = (
                        f"{exc.code}: {exc.message}"
                        + (f" — {exc.remediation}" if exc.remediation else "")
                    )
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.session_state["acquisition_notice"] = (
                        f"Acquisition finished: status={result.get('status', 'success')}, "
                        f"cached={result.get('cached', 'unknown')}. Run validation or the pipeline next."
                    )
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.warning("No candidate passed the discovery gates, so there is nothing to acquire.")
    else:
        st.warning("The discovery report contains no ranked candidates.")

    if report.get("selected"):
        selected = report["selected"]
        with st.expander("Selected candidate detail", expanded=False):
            st.json(selected)

    layout.section("Acquisition attempts", "Every attempt is recorded with its status and error category.")
    if attempts and attempts.get("attempts"):
        frame = pd.DataFrame(attempts["attempts"])
        keep = [
            column
            for column in ("candidate_id", "rank", "status", "cached", "duration_ms", "error_code", "error_message")
            if column in frame
        ]
        st.dataframe(frame[keep], width="stretch", hide_index=True)
        st.caption(f"Attempt log: {attempts_path}")
    else:
        layout.empty_state(
            "No acquisition attempts recorded",
            "Acquire a ranked candidate to create the provenance chain consumed by validation.",
            "python -m churnwise.cli acquire",
        )
