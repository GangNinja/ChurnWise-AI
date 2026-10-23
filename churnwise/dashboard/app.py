"""ChurnWise AI interactive dashboard entry point (Phase 11).

Run locally::

    python -m streamlit run churnwise/dashboard/app.py
"""

from __future__ import annotations

import importlib

import streamlit as st

from churnwise.dashboard.components import layout
from churnwise.dashboard.data_access import ArtifactError, current_settings, risk_ids, simulation_ids

PAGES: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("discovery", "Data discovery"),
    ("quality", "Data quality"),
    ("training", "Model training"),
    ("evaluation", "Model evaluation"),
    ("explainability", "Explainability"),
    ("risk", "Risk intelligence"),
    ("assessment", "Assess records"),
    ("customer", "Customer detail"),
    ("simulator", "Retention simulator"),
    ("reliability", "Reliability & limits"),
)
PAGE_KEYS = dict(PAGES)
PAGE_LABELS = {label: key for key, label in PAGES}


def _sidebar(settings) -> str:
    with st.sidebar:
        layout.brand()
        st.divider()
        label = st.radio(
            "Navigate",
            options=[item[1] for item in PAGES],
            key="cw_navigation",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Persisted runs")
        st.caption(f"Risk runs: {len(risk_ids(settings))}")
        st.caption(f"Simulations: {len(simulation_ids(settings))}")
        if st.button("Refresh artifact cache", key="cw_refresh_cache", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption(
            "Read-only over artifacts. Discovery, acquisition, training and "
            "simulation run only when an explicit action button is used."
        )
    return PAGE_LABELS[label]


def main() -> None:
    st.set_page_config(
        page_title="ChurnWise AI — Intelligence Dashboard",
        page_icon="◔",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    layout.inject_styles()
    settings = current_settings()
    page_key = _sidebar(settings)
    module = importlib.import_module(f"churnwise.dashboard.pages.{page_key}")
    try:
        module.render(settings)
    except ArtifactError as exc:
        st.error(f"{exc.code}: {exc.message}")
        if exc.path:
            st.caption(f"Artifact: {exc.path}")
        if exc.remediation:
            st.info(exc.remediation)
    layout.footer()


if __name__ == "__main__":
    main()
