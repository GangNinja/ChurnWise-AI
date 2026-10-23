"""Assess records: score new customers through the Phase 9 service.

The dashboard supplies schema-driven inputs and file validation only. Model
loading, preprocessing, probability calculation, tier assignment, rejection
handling and identifier hashing all remain in :class:`RiskService`.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    dict_to_json_bytes,
    format_count,
    format_probability,
    frame_to_csv_bytes,
    nested_get,
)
from churnwise.dashboard.components import charts, layout

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 20_000
NOT_PROVIDED = "— not provided —"
RESULT_COLUMNS = (
    "input_position",
    "record_reference",
    "churn_probability",
    "risk_tier",
    "predicted_label",
    "classification_threshold",
    "explanation_status",
)


def _thresholds(settings: Settings) -> tuple[float, ...]:
    summary, _ = access.risk_summary(settings)
    values = ((summary or {}).get("risk_thresholds") or {}).get("boundaries")
    if values and len(values) in (2, 3):
        return tuple(float(item) for item in values)
    return tuple(float(item) for item in settings.risk_tiers)


def _schema_context(settings: Settings) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Any]:
    metrics, metrics_path = access.benchmark_metrics(settings)
    run_id = nested_get(metrics, "preprocessing", "run_id")
    schema, schema_path = access.preprocessing_schema(settings, str(run_id) if run_id else None)
    return metrics, schema, schema_path or metrics_path


def _column_kinds(schema: Mapping[str, Any]) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    required = [str(name) for name in schema.get("required_columns") or []]
    preprocessing = schema.get("preprocessing") or {}
    numeric = {str(name) for name in preprocessing.get("numeric") or []}
    categorical = {str(name) for name in preprocessing.get("categorical") or []}
    boolean = {str(name) for name in preprocessing.get("boolean") or []}
    categories = {
        str(name): [str(value) for value in values]
        for name, values in (preprocessing.get("categories") or {}).items()
    }
    kinds = {}
    for name in required:
        if name in numeric:
            kinds[name] = "numeric"
        elif name in boolean:
            kinds[name] = "boolean"
        elif name in categorical:
            kinds[name] = "categorical"
        else:
            kinds[name] = "text"
    return required, kinds, categories


def _parse_numeric(values: Mapping[str, str], required: list[str], kinds: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    record: dict[str, Any] = {}
    problems: list[str] = []
    for name in required:
        raw = values.get(name)
        if kinds.get(name) == "numeric":
            text = str(raw or "").strip()
            if not text:
                record[name] = None
                continue
            try:
                record[name] = float(text)
            except ValueError:
                problems.append(f"`{name}` must be a number or left blank.")
        elif raw in (None, "", NOT_PROVIDED):
            record[name] = None
        elif kinds.get(name) == "boolean":
            record[name] = str(raw).strip().lower() == "true"
        else:
            record[name] = raw
    return record, problems


def _individual_form(settings: Settings, schema: Mapping[str, Any]) -> None:
    required, kinds, categories = _column_kinds(schema)
    if not required:
        st.error("The preprocessing schema does not declare required input columns.")
        return

    st.caption(
        "All fields come from the fitted Phase 5 schema. Blank values are passed to the existing "
        "pipeline, which applies its recorded imputation; the dashboard does not replace them with zeros."
    )
    with st.form("cw_individual_form"):
        values: dict[str, str] = {}
        columns = st.columns(3)
        for index, name in enumerate(required):
            with columns[index % 3]:
                if kinds[name] == "numeric":
                    values[name] = st.text_input(
                        name,
                        value="",
                        placeholder="number",
                        key=f"cw_ind_{name}",
                    )
                elif kinds[name] == "boolean":
                    choice = st.selectbox(
                        name,
                        [NOT_PROVIDED, "true", "false"],
                        key=f"cw_ind_{name}",
                    )
                    values[name] = "" if choice == NOT_PROVIDED else choice
                elif kinds[name] == "categorical":
                    options = [NOT_PROVIDED] + categories.get(name, [])
                    choice = st.selectbox(name, options, key=f"cw_ind_{name}")
                    values[name] = "" if choice == NOT_PROVIDED else choice
                else:
                    values[name] = st.text_input(name, value="", key=f"cw_ind_{name}")
        explain = st.checkbox(
            "Compute local SHAP drivers (may take several seconds)",
            value=False,
            help="Uses the Phase 8 explanation engine for this record only. Off by default.",
        )
        submitted = st.form_submit_button("Assess this customer", type="primary")

    if submitted:
        record, problems = _parse_numeric(values, required, kinds)
        if problems:
            for problem in problems:
                st.error(problem)
            return
        try:
            with st.spinner("Scoring the record with the selected model…"):
                result = actions.assess_customer_record(
                    settings,
                    record,
                    thresholds=_thresholds(settings),
                    explain=explain,
                )
        except actions.DashboardActionError as exc:
            st.error(f"{exc.code}: {exc.message}")
            if exc.remediation:
                st.info(exc.remediation)
        else:
            st.session_state["cw_individual_assessment"] = result
            st.success(f"Assessment completed for `{result.get('record_reference')}`.")

    result = st.session_state.get("cw_individual_assessment")
    if not result:
        return
    layout.section("Individual assessment", "The probability is a model estimate, not a guaranteed outcome.")
    layout.metric_row(
        [
            ("Churn probability", format_probability(result.get("churn_probability")), "Selected model"),
            ("Risk tier", str(result.get("risk_tier") or "unavailable").title(), "Configured tier boundaries"),
            ("Predicted class", result.get("predicted_label") or "not available", "Frozen classification threshold"),
            ("Threshold", format_probability(result.get("classification_threshold")), "Classification, separate from tiers"),
        ]
    )
    st.caption(
        f"Record `{result.get('record_reference')}` · model `{result.get('model_run_id')}` · "
        f"assessed {result.get('assessed_at')}. The raw values you entered are not included in this result."
    )
    for warning in result.get("warnings") or []:
        st.warning(str(warning))

    positives = result.get("top_positive_drivers") or []
    negatives = result.get("top_negative_drivers") or []
    if positives or negatives:
        layout.section("Local drivers", "SHAP contributions in probability space; they describe the model, not causality.")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Increases churn**")
            st.dataframe(pd.DataFrame(positives), width="stretch", hide_index=True) if positives else st.caption("None stored.")
        with cols[1]:
            st.markdown("**Decreases churn**")
            st.dataframe(pd.DataFrame(negatives), width="stretch", hide_index=True) if negatives else st.caption("None stored.")
    elif explain:
        st.info("The model was scored, but no local driver payload was returned. Probabilities remain valid.")
    else:
        st.caption("Local SHAP drivers were not requested for this assessment.")

    st.download_button(
        "Download this assessment (JSON)",
        data=dict_to_json_bytes(result),
        file_name=f"assessment_{result.get('record_reference') or 'record'}.json",
        mime="application/json",
        key="cw_individual_download",
    )
    if st.button("Clear individual assessment", key="cw_individual_clear"):
        st.session_state.pop("cw_individual_assessment", None)
        st.rerun()


def _read_upload(upload) -> tuple[pd.DataFrame | None, str]:
    data = upload.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        return None, "The file exceeds the 10 MiB upload limit."
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return None, f"The CSV could not be parsed ({exc.__class__.__name__}); no rows were scored."
    if frame.empty:
        return None, "The uploaded file contains no data rows."
    if len(frame) > MAX_BATCH_ROWS:
        return None, f"The file has {len(frame):,} rows; the dashboard scores at most {MAX_BATCH_ROWS:,} per request."
    return frame, ""


def _batch_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("records") or []:
        rows.append(
            {
                "input_position": item.get("input_position"),
                "record_reference": item.get("record_reference"),
                "churn_probability": item.get("churn_probability"),
                "risk_tier": item.get("risk_tier"),
                "predicted_label": item.get("predicted_label"),
                "classification_threshold": item.get("classification_threshold"),
                "explanation_status": item.get("explanation_status"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and "churn_probability" in frame:
        frame["churn_probability"] = pd.to_numeric(frame["churn_probability"], errors="coerce").round(6)
    return frame


def _batch_section(settings: Settings, schema: Mapping[str, Any]) -> None:
    required, kinds, _ = _column_kinds(schema)
    st.caption(
        "Upload a CSV with one row per customer. Required columns and expected types are listed below. "
        "A churn target column is not required and is ignored if present."
    )
    template = pd.DataFrame(columns=required).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download header-only CSV template",
        data=template,
        file_name="churnwise_input_template.csv",
        mime="text/csv",
        key="cw_batch_template",
    )
    st.dataframe(
        pd.DataFrame([{"Column": name, "Expected type": kinds.get(name, "text")} for name in required]),
        width="stretch",
        hide_index=True,
    )
    upload = st.file_uploader("Customer records (CSV)", type=["csv"], key="cw_batch_upload")

    frame = None
    if upload is not None:
        fingerprint = hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("cw_batch_fingerprint") not in (None, fingerprint):
            st.session_state.pop("cw_batch_assessment", None)
        st.session_state["cw_batch_fingerprint"] = fingerprint
        frame, problem = _read_upload(upload)
        if problem:
            st.error(problem)
        else:
            assert frame is not None
            missing = [name for name in required if name not in frame.columns]
            extra = [str(name) for name in frame.columns if str(name) not in required]
            st.caption(f"Parsed {format_count(len(frame))} rows and {format_count(len(frame.columns))} columns.")
            if missing:
                st.error("Missing required column(s): " + ", ".join(missing) + ". No rows were scored.")
                frame = None
            elif extra:
                st.warning("Column(s) the model does not consume will be ignored: " + ", ".join(extra))

    reference_column = None
    if frame is not None:
        reference_choice = st.selectbox(
            "Optional identifier column",
            ["None"] + [str(name) for name in frame.columns],
            help="If selected, values are hashed by Phase 9 before storage. Raw identifiers are never shown.",
            key="cw_batch_reference",
        )
        reference_column = None if reference_choice == "None" else reference_choice
        if st.button("Assess uploaded records", type="primary", key="cw_batch_score"):
            try:
                with st.spinner("Scoring the uploaded records with the selected model…"):
                    result = actions.assess_batch_records(
                        settings,
                        frame,
                        thresholds=_thresholds(settings),
                        reference_column=reference_column,
                    )
            except actions.DashboardActionError as exc:
                st.error(f"{exc.code}: {exc.message}")
                if exc.remediation:
                    st.info(exc.remediation)
            else:
                st.session_state["cw_batch_assessment"] = result
                st.session_state["cw_batch_fingerprint"] = fingerprint
                st.success(
                    f"Assessed {format_count(result.get('scored'))} of "
                    f"{format_count(result.get('total_received'))} rows."
                )

    result = st.session_state.get("cw_batch_assessment")
    if not result:
        return
    layout.section("Batch assessment", "Counts reconcile scored and rejected rows; input positions preserve correspondence.")
    if upload is None:
        st.caption("Showing the last batch assessed in this session; upload the file again to rerun it.")
    st.caption("Batch scoring does not compute SHAP. Stored local explanations remain available on Customer detail.")
    layout.metric_row(
        [
            ("Received", format_count(result.get("total_received")), "Rows in the uploaded file"),
            ("Scored", format_count(result.get("scored")), "Rows with a probability"),
            ("Rejected", format_count(result.get("rejected")), "Rows that failed validation"),
            (
                "High + critical",
                format_count(
                    sum(
                        1
                        for item in result.get("records") or []
                        if item.get("risk_tier") in {"high", "critical"}
                    )
                ),
                "Configured display tiers",
            ),
        ]
    )
    if result.get("batch_error"):
        st.error(str(result["batch_error"]))
    for warning in result.get("warnings") or []:
        st.warning(str(warning))

    batch = _batch_frame(result)
    if not batch.empty:
        counts = {
            label: int((batch["risk_tier"] == label).sum())
            for label in ("low", "medium", "high", "critical")
        }
        if any(counts.values()):
            st.plotly_chart(charts.tier_chart(counts), width="stretch")
        st.dataframe(batch.head(200), width="stretch", hide_index=True)
        if len(batch) > 200:
            st.caption(f"Showing the first 200 of {len(batch):,} scored rows; exports contain every row.")
        st.download_button(
            "Download scored rows (CSV)",
            data=frame_to_csv_bytes(batch),
            file_name="churnwise_scored_records.csv",
            mime="text/csv",
            key="cw_batch_csv",
        )
        st.download_button(
            "Download full assessment (JSON)",
            data=dict_to_json_bytes(result),
            file_name="churnwise_batch_assessment.json",
            mime="application/json",
            key="cw_batch_json",
        )
    rejections = result.get("rejections") or []
    if rejections:
        layout.section("Rejected records", "Only positions, hashed references and reasons are shown; raw values are omitted.")
        st.dataframe(pd.DataFrame(rejections), width="stretch", hide_index=True)
    if not result.get("counts_reconcile", True):
        st.error("Scored and rejected counts do not reconcile; treat this batch as incomplete.")


def render(settings: Settings) -> None:
    metrics, schema, schema_path = _schema_context(settings)
    identity = access.identity_from_benchmark(metrics)
    layout.page_header(
        "Assess records",
        "Score individual customers or an uploaded CSV with the same selected model, preprocessing pipeline "
        "and tier configuration used by the persisted risk runs. The dashboard never fits an ad-hoc model.",
        identity,
    )
    if not metrics:
        layout.empty_state(
            "No evaluated model is available",
            "Record assessment needs the Phase 7 selected model and its Phase 5 preprocessing artifacts.",
            "python -m churnwise.cli evaluate --cv-folds 5",
        )
        return
    if not schema:
        layout.empty_state(
            "No preprocessing schema was found",
            "The benchmark points at a preprocessing run whose schema artifact is missing. Regenerate preparation before scoring new records.",
            "python -m churnwise.cli prepare",
        )
        st.caption(f"Benchmark artifact: {schema_path}")
        return

    required = schema.get("required_columns") or []
    layout.section(
        "Model and thresholds",
        f"{format_count(len(required))} required input columns from preprocessing run "
        f"`{nested_get(metrics, 'preprocessing', 'run_id', default='unknown')}`.",
    )
    boundaries = _thresholds(settings)
    layout.metric_row(
        [
            ("Model family", identity.family or "not available", "Phase 7 selection"),
            ("Classification threshold", format_probability(nested_get(metrics, "selection", "threshold")), "Frozen before test scoring"),
            ("Tier boundaries", " / ".join(f"{value:.2f}" for value in boundaries), "Lower-inclusive boundaries"),
            ("Output scope", "New records", "No upload is required to use the rest of the dashboard"),
        ]
    )
    tabs = st.tabs(["Individual customer", "Batch CSV"])
    with tabs[0]:
        _individual_form(settings, schema)
    with tabs[1]:
        _batch_section(settings, schema)

    layout.disclaimer(
        "Probabilities are model estimates, not guarantees. The churn label and thresholds came from the "
        "project's persisted artifacts; this page does not retrain, recalibrate or explain causal effects."
    )


__all__ = ["render"]
