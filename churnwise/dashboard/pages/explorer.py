"""Customer Risk Explorer: one stored assessment, plus explicit new scoring.

The page combines three closely related jobs: inspect a persisted assessment,
score a single customer through the fitted pipeline, and batch-score a CSV.
Model loading, preprocessing, tiering and rejection handling stay in the
existing Phase 9 service; this module only prepares inputs and renders results.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import random
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
    nested_get,
)
from churnwise.dashboard.components import charts, layout, tables

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 20_000
NOT_PROVIDED = "— not provided —"


# ---------------------------------------------------------------------------
# Stored assessment


def _parse_driver_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _driver_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keep = [column for column in ("feature", "active_level", "direction", "contribution", "value") if column in frame]
    frame = frame[keep].rename(
        columns={
            "feature": "Feature",
            "active_level": "Level",
            "direction": "Direction",
            "contribution": "Contribution",
            "value": "Value",
        }
    )
    if "Contribution" in frame:
        frame["Contribution"] = pd.to_numeric(frame["Contribution"], errors="coerce").round(4)
    if "Value" in frame:
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce").round(4)
    if "Level" in frame:
        frame["Level"] = frame["Level"].where(frame["Level"].notna(), "—")
    if "Direction" in frame:
        frame["Direction"] = frame["Direction"].astype(str).str.replace("_", " ", regex=False)
    return frame


def _priority(tier: str) -> tuple[str, str]:
    return {
        "critical": ("Immediate review", "Highest-probability customers; verify contactability and owner."),
        "high": ("Prioritise outreach", "High-probability customers worth a retention intervention."),
        "medium": ("Monitor / test", "Ambiguous risk; use a measured pilot before broad spend."),
        "low": ("Standard service", "No special retention action is indicated by the model alone."),
    }.get(str(tier), ("Unclassified", "No tier was available for this assessment."))


def _contribution_rows(driver_entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("top_positive", "top_negative"):
        for item in _parse_driver_rows(driver_entry.get(key)):
            if not isinstance(item, Mapping) or "contribution" not in item or "feature" not in item:
                continue
            contribution = pd.to_numeric(item.get("contribution"), errors="coerce")
            if pd.isna(contribution):
                continue
            rows.append({**dict(item), "contribution": float(contribution)})
    return rows


def _predicted_text(row: pd.Series) -> str:
    value = row.get("predicted_class")
    if pd.notna(value):
        try:
            return "Churn (1)" if int(float(value)) == 1 else "Retained (0)"
        except (TypeError, ValueError):
            pass
    label = str(row.get("predicted_label") or "not available")
    return "Churn" if label.lower() == "yes" else label.title()


def _stored_assessment(
    settings: Settings,
    run_id: str | None,
    frame: pd.DataFrame,
    summary: Mapping[str, Any] | None,
    *,
    raw_columns: tuple[str, ...],
    population_reason: str = "",
) -> None:
    if frame is None or frame.empty:
        layout.empty_state(
            "No scored customers are available",
            "Create a Phase 9 risk run before inspecting individual assessments.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    drivers_map, _ = access.risk_drivers(settings, run_id)
    ordered = frame.sort_values("rank", ascending=True) if "rank" in frame else frame
    options = {
        str(row.get("record_reference")): int(row.get("input_position", -1))
        for row in ordered[["record_reference", "input_position"]].to_dict(orient="records")
    }
    state_reference = st.session_state.get("cw_customer_reference")
    if state_reference is not None and state_reference not in options:
        st.session_state.pop("cw_customer_reference", None)
    index = list(options).index(state_reference) if state_reference in options else 0
    reference = st.selectbox("Customer reference", list(options), index=index, key="cw_customer_reference")
    position = options[reference]
    selected = ordered[ordered["record_reference"].astype(str) == str(reference)]
    row = selected.iloc[0]
    probability = float(row.get("churn_probability"))
    tier = str(row.get("risk_tier") or "unclassified").lower()
    priority, priority_detail = _priority(tier)
    predicted_text = _predicted_text(row)
    threshold = row.get("classification_threshold")
    threshold_text = format_probability(threshold)

    top_cols = st.columns([1.05, 1])
    with top_cols[0]:
        st.markdown(f"**Customer profile** · `{reference}`")
        profile = tables.profile_items(selected, raw_columns) if raw_columns else []
        if profile:
            layout.profile_grid(profile)
            st.caption("Original account attributes, joined through the verified split manifest.")
        else:
            st.caption(
                "Original account attributes are unavailable for this run. "
                + (population_reason or "The stored risk artifacts contain opaque references only.")
            )
        layout.status_line(priority, tier, "risk" if tier in {"critical", "high"} else "info", priority_detail)
    with top_cols[1]:
        layout.risk_card(
            format_probability(probability),
            tier,
            predicted_text,
            threshold_text,
            reference,
        )
        st.caption(
            f"Rank {row.get('rank', 'not available')} · assessed {row.get('assessed_at', 'not recorded')} · "
            "stored model estimate, not an observed outcome."
        )

    driver_entry = (drivers_map or {}).get(str(reference)) or {}
    layout.section(
        "Why this prediction",
        "Stored SHAP contributions for this customer. Red is associated with higher predicted churn; "
        "blue with lower. Associations are not causes.",
    )
    contributions = _contribution_rows(driver_entry) if driver_entry else []
    if contributions:
        st.plotly_chart(
            charts.contribution_chart(contributions),
            width="stretch",
            config={"displayModeBar": False},
        )
        if driver_entry.get("summary"):
            st.caption(str(driver_entry.get("summary")))
        st.caption(
            "Contributions describe the model output, not the effect of changing a business attribute."
        )
    else:
        status = str(row.get("explanation_status") or "unavailable")
        reason = str(row.get("explanation_reason") or "No driver payload was stored for this customer.")
        if status == "available":
            reason = "The run summary says drivers are available, but no per-customer payload was found."
        st.warning(f"Drivers unavailable ({status}): {reason}")

    local_payload, _, local_plot = access.local_explanation(
        settings,
        position,
        family=str(row.get("model_family") or "") or None,
    )
    matched = False
    if local_payload:
        local_probability = local_payload.get("probability")
        try:
            matched = local_probability is not None and abs(float(local_probability) - probability) < 1e-9
        except (TypeError, ValueError):
            matched = False

    predicted_class = None
    if pd.notna(row.get("predicted_class")):
        try:
            predicted_class = int(float(row.get("predicted_class")))
        except (TypeError, ValueError):
            predicted_class = None
    rank_value = row.get("rank")
    payload = {
        "risk_run_id": (summary or {}).get("run_id"),
        "record_reference": reference,
        "input_position": position,
        "churn_probability": probability,
        "risk_tier": tier,
        "predicted_class": predicted_class,
        "rank": None if pd.isna(rank_value) else int(rank_value),
        "drivers": driver_entry or None,
        "local_explanation": local_payload if matched else None,
    }

    with st.expander("Attribution details and artifact metadata", expanded=False):
        st.markdown("**Stored driver rows**")
        if driver_entry:
            for title, key in (("Associated with higher churn", "top_positive"), ("Associated with lower churn", "top_negative")):
                st.markdown(f"{title}")
                drivers = _driver_frame(_parse_driver_rows(driver_entry.get(key)))
                if drivers.empty:
                    st.caption("None stored.")
                else:
                    st.dataframe(drivers, width="stretch", hide_index=True)
            st.caption(
                "`contribution` is in probability space; `value` is the transformed model input, "
                "not an original business attribute."
            )
        else:
            st.caption("No driver rows are stored for this assessment.")

        if local_payload and matched:
            st.markdown("**Stored local SHAP explanation**")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Base value", format_probability(local_payload.get("base_value")))
            with cols[1]:
                st.metric("Model output", format_probability(local_payload.get("model_output")))
            with cols[2]:
                st.metric("Output space", local_payload.get("output_space", "not available"))
            st.write(local_payload.get("summary") or "No summary sentence stored.")
            local_frame = tables.local_contributions_frame(local_payload)
            if not local_frame.empty:
                st.dataframe(local_frame, width="stretch", hide_index=True)
            if local_plot:
                st.image(str(local_plot), caption=f"Local SHAP waterfall · row {position}")
        elif local_payload:
            st.warning(
                "A local explanation exists at this matrix position but its probability does not match "
                "this assessment, so it is not shown."
            )
        else:
            st.caption("No local explanation was persisted at this matrix position; SHAP was not run on render.")

        st.markdown("**Assessment metadata**")
        st.caption(
            f"Run `{(summary or {}).get('run_id') or run_id or 'not recorded'}` · input position {position} · "
            f"model `{row.get('model_run_id', 'not recorded')}` · assessed {row.get('assessed_at', 'not recorded')}."
        )
        observed = row.get("observed_churn")
        if pd.notna(observed):
            st.caption(f"Stored observed label for this row: {int(float(observed))} (evaluation evidence only).")
        limitations = (summary or {}).get("limitations") or []
        for item in limitations:
            st.write(f"- {item}")
        st.download_button(
            "Download this assessment (JSON)",
            data=dict_to_json_bytes(payload),
            file_name=f"assessment_{reference}.json",
            mime="application/json",
            key="cw_customer_download",
        )


# ---------------------------------------------------------------------------
# New scoring


def _thresholds(settings: Settings) -> tuple[float, ...]:
    summary, _ = access.risk_summary(settings, prefer_population=True)
    values = ((summary or {}).get("risk_thresholds") or {}).get("boundaries")
    if values and len(values) in (2, 3):
        return tuple(float(item) for item in values)
    return tuple(float(item) for item in settings.risk_tiers)


def _schema(settings: Settings) -> tuple[dict[str, Any] | None, Any]:
    metrics, metrics_path = access.benchmark_metrics(settings)
    run_id = nested_get(metrics, "preprocessing", "run_id")
    schema, schema_path = access.preprocessing_schema(settings, str(run_id) if run_id else None)
    return schema, schema_path or metrics_path


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


def _autofill_individual_form(schema: Mapping[str, Any]) -> None:
    """Populate the individual form with a random, schema-compatible demo row."""
    required, kinds, categories = _column_kinds(schema)
    rng = random.Random()
    minutes = {
        "total_day_minutes": round(rng.uniform(70, 300), 1),
        "total_eve_minutes": round(rng.uniform(80, 320), 1),
        "total_night_minutes": round(rng.uniform(70, 320), 1),
        "total_intl_minutes": round(rng.uniform(0, 22), 1),
    }
    charge_rates = {
        "total_day_charge": ("total_day_minutes", 0.17),
        "total_eve_charge": ("total_eve_minutes", 0.085),
        "total_night_charge": ("total_night_minutes", 0.045),
        "total_intl_charge": ("total_intl_minutes", 0.27),
    }
    for name in required:
        kind = kinds.get(name)
        if kind == "numeric":
            if name in minutes:
                value: Any = minutes[name]
            elif name in charge_rates:
                source, rate = charge_rates[name]
                value = round(minutes[source] * rate, 2)
            elif name == "account_length":
                value = rng.randint(1, 243)
            elif name == "number_customer_service_calls":
                value = rng.randint(0, 8)
            elif name == "number_vmail_messages":
                value = rng.randint(0, 40)
            elif "_calls" in name and "intl" in name:
                value = rng.randint(1, 18)
            elif "_calls" in name:
                value = rng.randint(45, 155)
            else:
                value = round(rng.uniform(1, 100), 2)
            st.session_state[f"cw_ind_{name}"] = str(value)
        elif kind == "boolean":
            st.session_state[f"cw_ind_{name}"] = rng.choice(["true", "false"])
        elif kind == "categorical" and categories.get(name):
            st.session_state[f"cw_ind_{name}"] = rng.choice(categories[name])
        else:
            st.session_state[f"cw_ind_{name}"] = NOT_PROVIDED
    st.session_state.pop("cw_individual_assessment", None)


def _individual_form(settings: Settings, schema: Mapping[str, Any]) -> None:
    required, kinds, categories = _column_kinds(schema)
    if not required:
        st.error("The preprocessing schema does not declare required input columns.")
        return

    st.caption(
        "Fields come from the fitted Phase 5 schema. Blank values are passed to the existing pipeline, which "
        "applies its recorded imputation; the dashboard does not replace them with zeros."
    )
    st.button(
        "Autofill",
        key="cw_individual_autofill",
        help="Fill the fields with one random, schema-compatible example. This does not score it.",
        on_click=_autofill_individual_form,
        args=(schema,),
    )
    with st.form("cw_individual_form"):
        values: dict[str, str] = {}
        columns = st.columns(3)
        for index, name in enumerate(required):
            with columns[index % 3]:
                if kinds[name] == "numeric":
                    values[name] = st.text_input(name, value="", placeholder="number", key=f"cw_ind_{name}")
                elif kinds[name] == "boolean":
                    choice = st.selectbox(name, [NOT_PROVIDED, "true", "false"], key=f"cw_ind_{name}")
                    values[name] = "" if choice == NOT_PROVIDED else choice
                elif kinds[name] == "categorical":
                    choice = st.selectbox(name, [NOT_PROVIDED] + categories.get(name, []), key=f"cw_ind_{name}")
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
        st.markdown("**Local drivers** — SHAP contributions in probability space; they describe the model, not causality.")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("Increases churn")
            if positives:
                st.dataframe(pd.DataFrame(positives), width="stretch", hide_index=True)
            else:
                st.caption("None stored.")
        with cols[1]:
            st.markdown("Decreases churn")
            if negatives:
                st.dataframe(pd.DataFrame(negatives), width="stretch", hide_index=True)
            else:
                st.caption("None stored.")
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


def _sample_datasets() -> dict[str, Any]:
    """Return test CSVs available for trying the batch scoring flow."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "testing_datasets"
    samples: dict[str, Any] = {}
    for folder, label in (("proper_datasets", "Proper"), ("improper_datasets", "Improper")):
        directory = root / folder
        if directory.is_dir():
            for path in sorted(directory.glob("*.csv")):
                samples[f"{label} · {path.name}"] = path
    return samples


def _batch_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("records") or []:
        status = str(item.get("explanation_status") or "unavailable")
        reason = str(item.get("explanation_reason") or "")
        if status == "available":
            status_label = "Available"
        elif "only the highest-risk rows were explained" in reason or "outside the selected explanation subset" in reason:
            status_label = "Not generated for this row"
        elif reason == "explanation was not requested":
            status_label = "Not requested"
        else:
            status_label = "Unavailable"
        explanation = str(item.get("explanation_summary") or "")
        positive = [str(driver.get("feature")) for driver in item.get("top_positive_drivers") or [] if driver.get("feature")]
        negative = [str(driver.get("feature")) for driver in item.get("top_negative_drivers") or [] if driver.get("feature")]
        if not explanation and (positive or negative):
            pieces = []
            if positive:
                pieces.append("Higher churn: " + ", ".join(positive))
            if negative:
                pieces.append("Lower churn: " + ", ".join(negative))
            explanation = " · ".join(pieces)
        if not explanation and status != "available":
            explanation = reason
        rows.append(
            {
                "input_position": item.get("input_position"),
                "record_reference": item.get("record_reference"),
                "churn_probability": item.get("churn_probability"),
                "risk_tier": item.get("risk_tier"),
                "predicted_label": item.get("predicted_label"),
                "classification_threshold": item.get("classification_threshold"),
                "explanation_status": status_label,
                "model_explanation": explanation,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and "churn_probability" in frame:
        frame["churn_probability"] = pd.to_numeric(frame["churn_probability"], errors="coerce").round(6)
    return frame


def _batch_section(settings: Settings, schema: Mapping[str, Any]) -> None:
    required, kinds, _ = _column_kinds(schema)
    st.markdown("#### Upload a customer dataset")
    st.caption("Select a CSV from your computer. The model checks its columns before scoring; an outcome column is optional and ignored.")
    template = pd.DataFrame(columns=required).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download header-only CSV template",
        data=template,
        file_name="churnwise_input_template.csv",
        mime="text/csv",
        key="cw_batch_template",
    )
    with st.expander("Expected columns"):
        st.dataframe(
            pd.DataFrame([{"Column": name, "Expected type": kinds.get(name, "text")} for name in required]),
            width="stretch",
            hide_index=True,
        )
    upload = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="cw_batch_upload",
        help="CSV files up to 10 MiB and 20,000 customer rows are accepted.",
    )
    samples = _sample_datasets()
    sample_names = ["Choose a bundled dataset…", *samples]
    selected_sample = st.selectbox(
        "Or try a bundled dataset",
        sample_names,
        key="cw_batch_sample_v2",
        help="Testing datasets are organized by valid and invalid model input schema.",
    )
    dataset_name = upload.name if upload is not None else ""
    if upload is None and selected_sample != sample_names[0]:
        dataset_name = selected_sample
    source_label = (
        upload.name
        if upload is not None
        else samples[dataset_name].name
        if dataset_name in samples
        else None
    )

    frame = None
    fingerprint = None
    if upload is not None or (dataset_name and dataset_name in samples):
        data = upload.getvalue() if upload is not None else samples[dataset_name].read_bytes()
        fingerprint = hashlib.sha256(data).hexdigest()
        if st.session_state.get("cw_batch_fingerprint") not in (None, fingerprint):
            st.session_state.pop("cw_batch_assessment", None)
        st.session_state["cw_batch_fingerprint"] = fingerprint
        from types import SimpleNamespace

        frame, problem = _read_upload(SimpleNamespace(getvalue=lambda: data))
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

    if frame is not None:
        reference_choice = st.selectbox(
            "Optional identifier column",
            ["None"] + [str(name) for name in frame.columns],
            help="If selected, values are hashed by Phase 9 before storage. Raw identifiers are never shown.",
            key="cw_batch_reference",
        )
        reference_column = None if reference_choice == "None" else reference_choice
        explanation_choice = st.selectbox(
            "Generate explanations for",
            [
                "Top 10 highest-risk customers",
                "Top 25 highest-risk customers",
                "Top 50 highest-risk customers",
                "All eligible rows (up to 1,000)",
            ],
            help="Top-row explanations show the strongest model associations. This can take longer; at most 1,000 highest-risk rows receive explanations.",
            key="cw_batch_explanation_coverage",
        )
        if explanation_choice.startswith("All"):
            explanation_mode, explanation_top_n = "all", 10
        else:
            explanation_mode = "top"
            explanation_top_n = int(explanation_choice.split()[1])
        if st.button("Score this dataset", type="primary", key="cw_batch_score"):
            try:
                with st.spinner("Scoring the uploaded records with the selected model…"):
                    result = actions.assess_batch_records(
                        settings,
                        frame,
                        thresholds=_thresholds(settings),
                        reference_column=reference_column,
                        explain=explanation_mode,
                        explain_top_n=explanation_top_n,
                        source_label=source_label,
                    )
            except actions.DashboardActionError as exc:
                st.error(f"{exc.code}: {exc.message}")
                if exc.remediation:
                    st.info(exc.remediation)
            else:
                result["dataset_name"] = dataset_name
                st.session_state["cw_batch_assessment"] = result
                st.session_state["cw_batch_fingerprint"] = fingerprint
                if result.get("run_id"):
                    st.session_state["cw_exec_run"] = result["run_id"]
                st.success(
                    f"Scored and saved {format_count(result.get('scored'))} of "
                    f"{format_count(result.get('total_received'))} rows."
                )

    result = st.session_state.get("cw_batch_assessment")
    if not result:
        return
    if result.get("dataset_name"):
        st.caption(f"Scoring output for input dataset: {result['dataset_name']}")
    elif upload is None:
        st.caption("Showing the last batch assessed in this session; select the dataset again to rerun it.")
    layout.metric_row(
        [
            ("Received", format_count(result.get("total_received")), "Rows in the uploaded file"),
            ("Scored", format_count(result.get("scored")), "Rows with a stored probability"),
            ("Rejected", format_count(result.get("rejected")), "Rows that failed schema validation"),
            ("Tier available", "yes" if result.get("records") else "no", "Tier boundaries came from the risk run"),
        ]
    )
    frame = _batch_frame(result)
    if frame.empty:
        st.info("No rows were scored; see the batch error below.")
    else:
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Download batch results (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="churnwise_batch_results.csv",
            mime="text/csv",
            key="cw_batch_download",
        )
    if result.get("batch_error"):
        st.error(str(result["batch_error"]))
    records = result.get("records") or []
    explained = sum(str(item.get("explanation_status")) == "available" for item in records)
    if explained:
        st.caption(f"Generated model explanations for {format_count(explained)} of {format_count(result.get('scored'))} scored customers. Rows outside the selected set are marked “Not generated for this row.”")
    elif result.get("scored"):
        st.warning("No explanations were generated. The model scores are available; review the explanation status or scoring warnings below.")
    rejections = result.get("rejections") or []
    if rejections:
        with st.expander(f"Rejected rows ({len(rejections)})"):
            st.dataframe(pd.DataFrame(rejections), width="stretch", hide_index=True)
    if result.get("run_id"):
        st.caption(f"Saved scoring run `{result['run_id']}` is selected for the Executive Dashboard.")
        if st.button("Open Executive Dashboard insights", key="cw_batch_open_executive"):
            st.session_state["cw_pending_navigation"] = "Executive Dashboard"
            st.rerun()
    st.caption("Batch scoring does not compute SHAP; stored local explanations remain on the stored assessment above.")


# ---------------------------------------------------------------------------
# Page


def render(settings: Settings) -> None:
    runs = access.risk_ids(settings)
    run_labels = access.risk_run_labels(settings, runs)
    chosen = st.session_state.get("cw_customer_run")
    if chosen not in runs:
        st.session_state.pop("cw_customer_run", None)
        chosen = access.default_risk_run_id(settings, runs)
    run_id = chosen if runs else None
    summary, _ = access.risk_summary(settings, run_id)
    manifest, _ = access.risk_manifest(settings, run_id)
    population = access.customer_population(settings, run_id)
    if population.frame is not None and not population.frame.empty:
        frame = population.frame.copy()
        raw_columns = population.raw_columns
    else:
        frame, _ = access.risk_table(settings, run_id)
        raw_columns = ()
    identity = access.identity_from_risk(summary, manifest)
    identity = replace(identity, run_id=run_labels.get(run_id, identity.run_id))
    layout.page_header(
        "Customer Risk Explorer",
        "Inspect one customer's stored assessment, understand the model's strongest associations, "
        "or score a customer through the fitted pipeline.",
        identity,
    )

    if runs:
        selector_cols = st.columns([1.8, 2])
        with selector_cols[0]:
            selected_run = st.selectbox(
                "Scoring run",
                runs,
                index=runs.index(run_id) if run_id in runs else 0,
                key="cw_customer_run",
                format_func=lambda value: run_labels.get(value, value),
            )
        with selector_cols[1]:
            split = str(
                nested_get(summary, "population", "split", default="not recorded") or "not recorded"
            )
            st.caption(
                f"{'Uploaded dataset' if split == 'input' else f'Population {split} split'} · "
                f"{format_count((summary or {}).get('n_rows'))} scored · "
                f"threshold {format_probability((summary or {}).get('classification_threshold'))}"
            )
        if selected_run != run_id:
            st.rerun()

    tab_customer, tab_score, tab_batch = st.tabs(
        ["Stored customers", "Score a customer", "Upload & predict dataset"]
    )
    with tab_customer:
        if not runs:
            layout.empty_state(
                "No scored customers are available",
                "Create a Phase 9 risk run to browse stored assessments. Individual customers can still "
                "be scored on the other tabs.",
                "python -m churnwise.cli score --population val --explain all",
            )
        else:
            _stored_assessment(
                settings,
                run_id,
                frame,
                summary,
                raw_columns=raw_columns,
                population_reason=population.reason,
            )

    with tab_score:
        st.caption(
            "Scoring is an explicit action. Nothing is scored while the page renders, and blank values use "
            "the fitted pipeline's recorded imputation."
        )
        schema, _ = _schema(settings)
        if not schema:
            layout.empty_state(
                "No preprocessing schema is available",
                "Scoring needs the fitted Phase 5 pipeline. Train and prepare a model first.",
                "python -m churnwise.cli prepare && python -m churnwise.cli train",
            )
        else:
            _individual_form(settings, schema)

    with tab_batch:
        st.caption(
            "Upload a customer CSV or try a synthetic dataset, preview its schema, then choose Predict churn risk. "
            "Stored assessments are never recomputed on render."
        )
        schema, _ = _schema(settings)
        if not schema:
            layout.empty_state(
                "No preprocessing schema is available",
                "Batch scoring needs the fitted Phase 5 pipeline.",
                "python -m churnwise.cli prepare && python -m churnwise.cli train",
            )
        else:
            _batch_section(settings, schema)
