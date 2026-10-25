"""ChurnWise AI interactive dashboard entry point.

Five product pages over persisted artifacts:

    Executive Dashboard -> Customer Risk Explorer -> Model Laboratory
    -> Retention Workbench -> Campaign Simulator

Run locally::

    python -m streamlit run churnwise/dashboard/app.py
"""

from __future__ import annotations

import importlib

import streamlit as st

from churnwise.dashboard.components import layout
from churnwise.dashboard.data_access import ArtifactError, current_settings

PAGES: tuple[tuple[str, str], ...] = (
    ("executive", "Executive Dashboard"),
    ("explorer", "Customer Risk Explorer"),
    ("laboratory", "Model Laboratory"),
    ("workbench", "Retention Workbench"),
    ("simulator", "Campaign Simulator"),
)
PAGE_KEYS = dict(PAGES)
PAGE_LABELS = {label: key for key, label in PAGES}


def _sidebar() -> str:
    with st.sidebar:
        layout.brand()
        st.divider()
        pending = st.session_state.pop("cw_pending_navigation", None)
        if pending in PAGE_LABELS:
            st.session_state["cw_navigation"] = pending
        label = st.radio(
            "Navigate",
            options=[item[1] for item in PAGES],
            key="cw_navigation",
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("Refresh artifact cache", key="cw_refresh_cache", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption(
            "Read-only over persisted artifacts. Pipeline actions run only from the explicit "
            "controls in Model Laboratory."
        )
    return PAGE_LABELS[label]


def main() -> None:
    st.set_page_config(
        page_title="ChurnWise AI — Retention Intelligence",
        page_icon="◔",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    layout.inject_styles()
    settings = current_settings()
    page_key = _sidebar()
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
