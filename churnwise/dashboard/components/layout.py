"""Page shell, identity headers and reusable state components."""

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

import streamlit as st

from churnwise.dashboard.data_access import RunIdentity

NAV_HINT = "python -m churnwise.cli"

_CSS = """
<style>
:root {
  --cw-ink: #102536;
  --cw-muted: #617286;
  --cw-line: #dfe7ee;
  --cw-panel: #ffffff;
  --cw-canvas: #f5f7f9;
  --cw-teal: #0b7a70;
  --cw-teal-soft: #e8f5f2;
  --cw-amber: #b45b0b;
  --cw-red: #b42318;
  --cw-blue: #2563eb;
  --cw-sidebar: #0c2233;
}
html, body, [class*="css"] {
  font-family: "Aptos", "Segoe UI", system-ui, -apple-system, sans-serif;
  color: var(--cw-ink);
}
.stApp { background: var(--cw-canvas); }
.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 2.25rem; }
h1, h2, h3 { color: var(--cw-ink); letter-spacing: -0.02em; }
h2 { margin-top: 0.4rem; }
[data-testid="stMetric"] {
  background: var(--cw-panel);
  border: 1px solid var(--cw-line);
  border-radius: 11px;
  padding: 0.75rem 0.9rem;
  box-shadow: 0 1px 2px rgba(16, 38, 58, 0.035);
}
[data-testid="stMetricLabel"] p {
  color: var(--cw-muted);
  font-size: 0.7rem;
  letter-spacing: 0.065em;
  text-transform: uppercase;
  white-space: normal !important;
  line-height: 1.3;
}
[data-testid="stMetricLabel"] .stMarkdownEllipsis,
[data-testid="stMetricValue"] .stMarkdownEllipsis {
  overflow: visible !important;
  white-space: normal !important;
  text-overflow: clip !important;
}
[data-testid="stMetricValue"] { color: var(--cw-ink); font-weight: 650; font-size: 1.62rem !important; }
[data-testid="stMetricValue"] p {
  font-size: inherit !important;
  white-space: normal !important;
  overflow-wrap: anywhere;
  line-height: 1.22;
}
[data-testid="stDataFrame"] { border: 1px solid var(--cw-line); border-radius: 10px; }
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { font-size: 0.82rem; }
[data-testid="stSidebar"] { background: var(--cw-sidebar); }
[data-testid="stSidebar"] * { color: #e8eef5; }
[data-testid="stSidebar"] .stRadio label p { color: #dbe6f0; }
[data-testid="stSidebar"] div[role="radiogroup"] > div { width: 100%; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
  display: flex;
  width: 100%;
  box-sizing: border-box;
  padding: 0.32rem 0.55rem;
  border-radius: 7px;
  border-left: 3px solid transparent;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div { width: 100%; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {
  display: none;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {
  background: rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {
  background: rgba(255,255,255,0.10);
  border-left-color: #2dd4bf;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p {
  color: #ffffff;
  font-weight: 600;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.16); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code { color: #f4d7a1; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * { color: #102a43 !important; }
.cw-brand { padding: 0.1rem 0 0.45rem 0; }
.cw-brand .cw-brand-name { font-size: 1.16rem; font-weight: 700; letter-spacing: 0.01em; color: #ffffff; }
.cw-brand .cw-brand-sub { font-size: 0.72rem; color: #a9bdd1; line-height: 1.45; margin-top: 0.18rem; }
.cw-brand .cw-brand-mark { display:inline-block; width:0.42rem; height:0.42rem; border-radius:50%; background:#2dd4bf; margin-right:0.42rem; vertical-align:middle; }
.cw-hero {
  background: linear-gradient(180deg, #ffffff 0%, #fbfdfd 100%);
  border: 1px solid var(--cw-line);
  border-left: 4px solid var(--cw-teal);
  border-radius: 12px;
  padding: 0.72rem 1.05rem 0.62rem 1.05rem;
  margin-bottom: 0.45rem;
}
.cw-eyebrow {
  color: var(--cw-teal);
  font-size: 0.68rem;
  font-weight: 750;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  margin-bottom: 0.2rem;
}
.cw-title { font-size: 1.5rem; line-height: 1.12; font-weight: 680; margin: 0; color: var(--cw-ink); }
.cw-subtitle { color: var(--cw-muted); font-size: 0.82rem; line-height: 1.35; margin-top: 0.2rem; max-width: 92ch; }
.cw-context-line { display:flex; flex-wrap:wrap; align-items:center; gap:0.35rem; margin-top:0.38rem; color:var(--cw-muted); font-size:0.7rem; }
.cw-context-line strong { color:var(--cw-ink); font-weight:650; }
.cw-context-dot { display:inline-block; width:0.42rem; height:0.42rem; border-radius:50%; background:var(--cw-teal); }
.cw-identity { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.45rem; margin: 0.25rem 0 0.35rem 0; }
.cw-id-card {
  background: var(--cw-panel);
  border: 1px solid var(--cw-line);
  border-radius: 9px;
  padding: 0.5rem 0.65rem;
  min-height: 3.2rem;
}
.cw-id-label { display: block; color: var(--cw-muted); font-size: 0.64rem; letter-spacing: 0.07em; text-transform: uppercase; }
.cw-id-value { display: block; color: var(--cw-ink); font-size: 0.78rem; margin-top: 0.18rem; word-break: break-word; }
.cw-id-value.missing { color: var(--cw-amber); font-style: italic; }
.cw-section { margin: 1.2rem 0 0.4rem 0; display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.cw-section h3 { font-size: 1.02rem; margin: 0; color: var(--cw-ink) !important; letter-spacing:-0.01em; }
.cw-section h3 span { color: inherit !important; }
.cw-section p { color: var(--cw-muted); font-size: 0.8rem; margin: 0.15rem 0 0.25rem 0; }
.cw-insight { background:var(--cw-panel); border:1px solid var(--cw-line); border-radius:10px; padding:0.65rem 0.8rem; height:100%; }
.cw-insight .cw-insight-label { color:var(--cw-teal); font-size:0.62rem; font-weight:750; letter-spacing:0.09em; text-transform:uppercase; margin-bottom:0.2rem; }
.cw-insight .cw-insight-value { color:var(--cw-ink); font-size:0.88rem; font-weight:600; line-height:1.35; }
.cw-insight .cw-insight-detail { color:var(--cw-muted); font-size:0.75rem; line-height:1.4; margin-top:0.2rem; }
.cw-model-strip { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; background:var(--cw-panel); border:1px solid var(--cw-line); border-radius:11px; padding:0.7rem 0.9rem; }
.cw-model-strip .cw-model-name { font-size:0.94rem; font-weight:650; }
.cw-model-strip .cw-model-metrics { display:flex; gap:1.2rem; flex-wrap:wrap; }
.cw-model-strip .cw-model-metric { display:flex; flex-direction:column; gap:0.08rem; }
.cw-model-strip .cw-model-metric span { color:var(--cw-muted); font-size:0.6rem; letter-spacing:0.07em; text-transform:uppercase; }
.cw-model-strip .cw-model-metric strong { font-size:0.82rem; font-weight:650; }
.cw-risk-card { background:var(--cw-panel); border:1px solid var(--cw-line); border-left:5px solid var(--cw-teal); border-radius:10px; padding:0.8rem 0.95rem; height:100%; }
.cw-risk-card.critical { border-left-color:var(--cw-red); }
.cw-risk-card.high { border-left-color:var(--cw-amber); }
.cw-risk-card.medium { border-left-color:var(--cw-blue); }
.cw-risk-card.low { border-left-color:#94a3b8; }
.cw-risk-label { color:var(--cw-muted); font-size:0.64rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:750; }
.cw-risk-probability { color:var(--cw-ink); font-size:2.15rem; font-weight:680; line-height:1.1; margin:0.18rem 0 0.35rem 0; }
.cw-risk-meta { display:flex; flex-wrap:wrap; gap:0.3rem 0.9rem; color:var(--cw-muted); font-size:0.75rem; }
.cw-risk-meta strong { color:var(--cw-ink); font-weight:650; }
.cw-profile { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.45rem; }
.cw-profile-item { background:var(--cw-panel); border:1px solid var(--cw-line); border-radius:8px; padding:0.45rem 0.6rem; }
.cw-profile-item .cw-profile-label { display:block; color:var(--cw-muted); font-size:0.6rem; letter-spacing:0.07em; text-transform:uppercase; }
.cw-profile-item .cw-profile-value { display:block; color:var(--cw-ink); font-size:0.84rem; font-weight:600; margin-top:0.12rem; }
.cw-filter-line { display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; color:var(--cw-muted); font-size:0.78rem; }
.cw-chip { display:inline-flex; align-items:center; border:1px solid #c8e5df; background:var(--cw-teal-soft); color:#075e56; border-radius:999px; padding:0.12rem 0.5rem; font-size:0.7rem; font-weight:650; }
.cw-empty {
  background: #fffaf0;
  border: 1px solid #f0d7a8;
  border-left: 5px solid var(--cw-amber);
  border-radius: 10px;
  padding: 0.95rem 1.1rem;
  margin: 0.35rem 0 0.8rem 0;
}
.cw-empty strong { color: var(--cw-ink); }
.cw-empty p { color: var(--cw-muted); margin: 0.25rem 0 0.1rem 0; font-size: 0.9rem; }
.cw-empty code { background: #fdf3dd; color: #6b4a12; padding: 0.08rem 0.28rem; border-radius: 4px; }
.cw-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 0.13rem 0.55rem;
  font-size: 0.7rem;
  font-weight: 650;
  letter-spacing: 0.03em;
}
.cw-pill.ok { background: #e7f6ec; color: #166534; border: 1px solid #b7e2c3; }
.cw-pill.warn { background: #fff4e5; color: #8a4b08; border: 1px solid #efd0a4; }
.cw-pill.risk { background: #feeceb; color: #a1281d; border: 1px solid #f2b8b2; }
.cw-pill.info { background: #eaf1ff; color: #1e40af; border: 1px solid #c3d5fb; }
.cw-note { color: var(--cw-muted); font-size: 0.8rem; line-height: 1.5; }
.cw-disclaimer {
  border: 1px dashed #d9b26a;
  background: #fffbeb;
  border-radius: 9px;
  padding: 0.65rem 0.8rem;
  color: #6b4a12;
  font-size: 0.82rem;
}
.cw-footer { color: var(--cw-muted); font-size: 0.72rem; border-top: 1px solid var(--cw-line); margin-top: 2rem; padding-top: 0.7rem; }
.stButton > button, .stDownloadButton > button { border-radius:8px; font-size:0.8rem; font-weight:600; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:var(--cw-teal); border-color:var(--cw-teal); }
.stButton > button[kind="tertiary"] { color:var(--cw-teal); padding-left:0; }
[data-testid="stExpander"] details { border-color:var(--cw-line); border-radius:9px; }
@media (max-width: 900px) {
  .block-container { padding-top: 1.25rem; padding-bottom: 1.5rem; }
  .cw-identity { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cw-title { font-size: 1.4rem; }
  .cw-profile { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        """
        <div class="cw-brand">
          <div class="cw-brand-name"><span class="cw-brand-mark"></span>ChurnWise AI</div>
          <div class="cw-brand-sub">Retention intelligence over audited artifacts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escaped(value: Any) -> str:
    return html.escape(str(value)) if value not in (None, "") else ""


def _short(value: Any, limit: int = 20) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def page_header(
    title: str,
    subtitle: str,
    identity: RunIdentity,
    *,
    eyebrow: str = "ChurnWise AI",
) -> None:
    """Render a compact page header with full run details on demand."""

    context = ""
    if identity.run_id or identity.family:
        run = f'<strong>{html.escape(_short(identity.run_id, 54))}</strong>' if identity.run_id else "Run not recorded"
        family = f' · {html.escape(_short(identity.family, 22))}' if identity.family else ""
        context = f'<div class="cw-context-line"><span class="cw-context-dot"></span>Selected scoring run · {run}{family}</div>'
    st.markdown(
        f"""
        <div class="cw-hero">
          <div class="cw-eyebrow">{html.escape(eyebrow)}</div>
          <div class="cw-title">{html.escape(title)}</div>
          <div class="cw-subtitle">{html.escape(subtitle)}</div>
          {context}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Run details", expanded=False):
        values = {
            "Run id": identity.run_id,
            "Model family": identity.family,
            "Dataset hash": identity.dataset_sha256,
            "Generated at": identity.generated_at,
        }
        cards = []
        for label, value in values.items():
            display = str(value) if value else "not available"
            css = "" if value else " missing"
            cards.append(
                f'<div class="cw-id-card"><span class="cw-id-label">{html.escape(label)}</span>'
                f'<span class="cw-id-value{css}">{html.escape(display)}</span></div>'
            )
        st.markdown('<div class="cw-identity">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        if identity.source:
            st.caption(f"Evidence source: {identity.source}.")
        if identity.missing:
            st.caption(
                "Missing identity fields: "
                + ", ".join(identity.missing)
                + ". The page still shows only the artifacts it can verify."
            )


def section(title: str, description: str | None = None) -> None:
    body = f'<p>{html.escape(description)}</p>' if description else ""
    st.markdown(
        f'<div class="cw-section"><div><h3>{html.escape(title)}</h3>{body}</div></div>',
        unsafe_allow_html=True,
    )


def navigate_to(page_label: str) -> None:
    """Queue a sidebar change for the next run; the label must match ``app.PAGES``.

    The sidebar radio is instantiated before page render, so its widget state
    cannot be changed from a button inside the page. The app consumes this
    queued label before creating the radio on the next run.
    """

    st.session_state["cw_pending_navigation"] = page_label
    st.rerun()


def insight_cards(items: Sequence[tuple[str, str, str]]) -> None:
    """Small labelled observations, each tied to a value the page can verify."""

    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value, detail) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="cw-insight"><div class="cw-insight-label">{html.escape(label)}</div>'
                f'<div class="cw-insight-value">{html.escape(value)}</div>'
                f'<div class="cw-insight-detail">{html.escape(detail)}</div></div>',
                unsafe_allow_html=True,
            )


def profile_grid(items: Sequence[tuple[str, str]]) -> None:
    """Compact original-customer attributes; values are already formatted."""

    if not items:
        return
    cards = "".join(
        f'<div class="cw-profile-item"><span class="cw-profile-label">{html.escape(label)}</span>'
        f'<span class="cw-profile-value">{html.escape(value)}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="cw-profile">{cards}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    safe_kind = kind if kind in {"ok", "warn", "risk", "info"} else "info"
    return f'<span class="cw-pill {safe_kind}">{html.escape(text)}</span>'


def status_line(label: str, status: str, kind: str = "info", detail: str = "") -> None:
    extra = f" — {html.escape(detail)}" if detail else ""
    st.markdown(
        f'{pill(status, kind)} <strong>{html.escape(label)}</strong>{extra}',
        unsafe_allow_html=True,
    )


def risk_card(
    probability: Any,
    tier: Any,
    predicted: Any,
    threshold: Any,
    reference: Any,
) -> None:
    """Prominent assessment card; tier text carries the meaning, not colour alone."""

    tier_text = str(tier or "unclassified").lower()
    safe_tier = tier_text if tier_text in {"low", "medium", "high", "critical"} else "unclassified"
    predicted_text = str(predicted or "not available")
    st.markdown(
        f'<div class="cw-risk-card {safe_tier}">'
        f'<div class="cw-risk-label">Predicted churn probability</div>'
        f'<div class="cw-risk-probability">{html.escape(str(probability))}</div>'
        f'<div class="cw-risk-meta">'
        f'<span>Risk tier <strong>{html.escape(tier_text.title())}</strong></span>'
        f'<span>Predicted class <strong>{html.escape(predicted_text)}</strong></span>'
        f'<span>Threshold <strong>{html.escape(str(threshold))}</strong></span>'
        f'<span>Customer <strong>{html.escape(str(reference or "not available"))}</strong></span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def metric_row(items: Sequence[tuple[str, Any, str | None]], *, columns: int | None = None) -> None:
    count = columns or len(items)
    cols = st.columns(count)
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, str(value) if value not in (None, "") else "not available", help=help_text)


def empty_state(title: str, detail: str, command: str) -> None:
    """Actionable missing-artifact state. Never includes a fabricated value."""

    st.markdown(
        f"""
        <div class="cw-empty">
          <strong>{html.escape(title)}</strong>
          <p>{html.escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(command, language="bash")


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="cw-disclaimer">{html.escape(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="cw-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="cw-footer">Read-only over persisted artifacts except where an '
        "explicit action is labelled. No metric on this page is sampled or invented.</div>",
        unsafe_allow_html=True,
    )


__all__ = [
    "NAV_HINT",
    "brand",
    "disclaimer",
    "empty_state",
    "footer",
    "inject_styles",
    "insight_cards",
    "metric_row",
    "navigate_to",
    "note",
    "page_header",
    "pill",
    "profile_grid",
    "risk_card",
    "section",
    "status_line",
]
