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
  --cw-ink: #12263a;
  --cw-muted: #5b6b7f;
  --cw-line: #dfe7ef;
  --cw-panel: #ffffff;
  --cw-canvas: #f6f8fa;
  --cw-teal: #0f766e;
  --cw-amber: #b45309;
  --cw-red: #b42318;
  --cw-blue: #1d4ed8;
}
html, body, [class*="css"] {
  font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  color: var(--cw-ink);
}
.stApp { background: var(--cw-canvas); }
.block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--cw-ink); letter-spacing: -0.02em; }
h2 { margin-top: 0.4rem; }
[data-testid="stMetric"] {
  background: var(--cw-panel);
  border: 1px solid var(--cw-line);
  border-radius: 10px;
  padding: 0.8rem 0.95rem;
  box-shadow: 0 1px 2px rgba(16, 38, 58, 0.04);
}
[data-testid="stMetricLabel"] p {
  color: var(--cw-muted);
  font-size: 0.78rem;
  letter-spacing: 0.04em;
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
[data-testid="stMetricValue"] { color: var(--cw-ink); font-weight: 650; font-size: 1.6rem !important; }
[data-testid="stMetricValue"] p {
  font-size: inherit !important;
  white-space: normal !important;
  overflow-wrap: anywhere;
  line-height: 1.25;
}
[data-testid="stDataFrame"] { border: 1px solid var(--cw-line); border-radius: 10px; }
[data-testid="stSidebar"] { background: #102a43; }
[data-testid="stSidebar"] * { color: #e8eef5; }
[data-testid="stSidebar"] .stRadio label p { color: #dbe6f0; }
[data-testid="stSidebar"] div[role="radiogroup"] > div { width: 100%; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
  display: flex;
  width: 100%;
  box-sizing: border-box;
  padding: 0.3rem 0.55rem;
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
.cw-brand { padding: 0.1rem 0 0.6rem 0; }
.cw-brand .cw-brand-name { font-size: 1.18rem; font-weight: 700; letter-spacing: 0.01em; color: #ffffff; }
.cw-brand .cw-brand-sub { font-size: 0.74rem; color: #a9bdd1; line-height: 1.45; margin-top: 0.2rem; }
.cw-hero {
  background: var(--cw-panel);
  border: 1px solid var(--cw-line);
  border-left: 5px solid var(--cw-teal);
  border-radius: 12px;
  padding: 1.15rem 1.3rem 1.05rem 1.3rem;
  margin-bottom: 0.9rem;
}
.cw-eyebrow {
  color: var(--cw-teal);
  font-size: 0.73rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.25rem;
}
.cw-title { font-size: 1.7rem; line-height: 1.18; font-weight: 680; margin: 0; color: var(--cw-ink); }
.cw-subtitle { color: var(--cw-muted); font-size: 0.92rem; line-height: 1.45; margin-top: 0.35rem; max-width: 78ch; }
.cw-identity { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.55rem; margin: 0.65rem 0 1rem 0; }
.cw-id-card {
  background: var(--cw-panel);
  border: 1px solid var(--cw-line);
  border-radius: 9px;
  padding: 0.55rem 0.7rem;
  min-height: 3.5rem;
}
.cw-id-label { display: block; color: var(--cw-muted); font-size: 0.68rem; letter-spacing: 0.07em; text-transform: uppercase; }
.cw-id-value { display: block; color: var(--cw-ink); font-size: 0.82rem; margin-top: 0.2rem; word-break: break-word; }
.cw-id-value.missing { color: var(--cw-amber); font-style: italic; }
.cw-section { margin: 1.1rem 0 0.35rem 0; }
.cw-section h3 { font-size: 1.08rem; margin: 0; color: var(--cw-ink) !important; }
.cw-section h3 span { color: inherit !important; }
.cw-section p { color: var(--cw-muted); font-size: 0.86rem; margin: 0.18rem 0 0.35rem 0; }
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
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.03em;
}
.cw-pill.ok { background: #e7f6ec; color: #166534; border: 1px solid #b7e2c3; }
.cw-pill.warn { background: #fff4e5; color: #8a4b08; border: 1px solid #efd0a4; }
.cw-pill.risk { background: #feeceb; color: #a1281d; border: 1px solid #f2b8b2; }
.cw-pill.info { background: #eaf1ff; color: #1e40af; border: 1px solid #c3d5fb; }
.cw-note { color: var(--cw-muted); font-size: 0.82rem; line-height: 1.5; }
.cw-disclaimer {
  border: 1px dashed #d9b26a;
  background: #fffbeb;
  border-radius: 9px;
  padding: 0.65rem 0.8rem;
  color: #6b4a12;
  font-size: 0.84rem;
}
.cw-footer { color: var(--cw-muted); font-size: 0.74rem; border-top: 1px solid var(--cw-line); margin-top: 2rem; padding-top: 0.7rem; }
@media (max-width: 900px) {
  .cw-identity { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cw-title { font-size: 1.4rem; }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        """
        <div class="cw-brand">
          <div class="cw-brand-name">ChurnWise AI</div>
          <div class="cw-brand-sub">Customer intelligence over audited artifacts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escaped(value: Any) -> str:
    return html.escape(str(value)) if value not in (None, "") else ""


def page_header(
    title: str,
    subtitle: str,
    identity: RunIdentity,
    *,
    eyebrow: str = "ChurnWise AI",
) -> None:
    """Render the page title and the four required run-identity fields."""

    st.markdown(
        f"""
        <div class="cw-hero">
          <div class="cw-eyebrow">{html.escape(eyebrow)}</div>
          <div class="cw-title">{html.escape(title)}</div>
          <div class="cw-subtitle">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    values = {
        "Run id": identity.run_id,
        "Model family": identity.family,
        "Dataset hash": identity.dataset_prefix,
        "Generated at": identity.generated_at,
    }
    cards = []
    for label, value in values.items():
        if value:
            cards.append(
                f'<div class="cw-id-card"><span class="cw-id-label">{label}</span>'
                f'<span class="cw-id-value">{html.escape(str(value))}</span></div>'
            )
        else:
            cards.append(
                f'<div class="cw-id-card"><span class="cw-id-label">{label}</span>'
                '<span class="cw-id-value missing">not available</span></div>'
            )
    st.markdown('<div class="cw-identity">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    if identity.missing:
        st.caption(
            "Missing identity fields: "
            + ", ".join(identity.missing)
            + ". The page still shows only the artifacts it can verify."
        )


def section(title: str, description: str | None = None) -> None:
    body = f'<p>{html.escape(description)}</p>' if description else ""
    st.markdown(
        f'<div class="cw-section"><h3>{html.escape(title)}</h3>{body}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info") -> str:
    safe_kind = kind if kind in {"ok", "warn", "risk", "info"} else "info"
    return f'<span class="cw-pill {safe_kind}">{html.escape(text)}</span>'


def status_line(label: str, status: str, kind: str = "info", detail: str = "") -> None:
    extra = f" — {html.escape(detail)}" if detail else ""
    st.markdown(
        f'{pill(status, kind)} <strong>{html.escape(label)}</strong>{extra}',
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
    "metric_row",
    "note",
    "page_header",
    "pill",
    "section",
    "status_line",
]
