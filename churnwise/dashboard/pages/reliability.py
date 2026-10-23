"""Reliability & limits: artifact freshness, identity checks and honest boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import format_count
from churnwise.dashboard.components import layout


def _resolve_path(settings: Settings, value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = Path(settings.project_root) / path
    return path


def _digest_check(settings: Settings, report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    rows = []
    model_sha = report.get("model_sha256")
    model_path = _resolve_path(settings, report.get("model_path"))
    if model_path.is_file() and model_sha:
        actual = access.file_sha256(model_path)
        rows.append(
            {
                "Artifact": "Selected model",
                "Declared": str(model_sha)[:16],
                "Actual": actual[:16],
                "Match": actual == str(model_sha),
            }
        )
    preprocessor_path = _resolve_path(settings, report.get("preprocessor_path"))
    declared = report.get("preprocessor_hash")
    if preprocessor_path.is_file() and declared:
        actual = access.file_sha256(preprocessor_path)
        rows.append(
            {
                "Artifact": "Fitted preprocessor",
                "Declared": str(declared)[:16],
                "Actual": actual[:16],
                "Match": actual == str(declared),
            }
        )
    return rows


def render(settings: Settings) -> None:
    inventory = access.artifact_inventory(settings)
    validation, _ = access.validation_report(settings)
    metrics, _ = access.benchmark_metrics(settings)
    risk, _ = access.risk_summary(settings)
    risk_manifest, _ = access.risk_manifest(settings)
    simulation, _ = access.simulation_result(settings)
    identity = (
        access.identity_from_risk(risk, risk_manifest)
        if risk
        else access.identity_from_benchmark(metrics)
        if metrics
        else access.identity_from_validation(validation)
    )
    layout.page_header(
        "Reliability & limits",
        "Artifact presence, recorded digests, failed checks, privacy rules and the limitations that must travel "
        "with every number. This page reports what is saved, not what would be convenient.",
        identity,
    )
    if not inventory:
        layout.empty_state(
            "No artifacts to inspect",
            "Run the pipeline to create the artifacts this page audits.",
            "python -m churnwise.cli discover && python -m churnwise.cli acquire && "
            "python -m churnwise.cli validate && python -m churnwise.cli prepare",
        )
        return

    present = sum(1 for entry in inventory if entry.exists)
    layout.metric_row(
        [
            ("Artifact families", format_count(len(inventory)), "Latest artifact for each pipeline stage"),
            ("Present", format_count(present), "Files found on disk"),
            ("Missing", format_count(len(inventory) - present), "Referenced but not found"),
            ("Risk rows", format_count((risk or {}).get("n_rows")), "Scored population"),
        ]
    )

    layout.section("Artifact inventory", "The dashboard resolves the newest run per stage.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Stage": entry.kind,
                    "Run / id": entry.run_id,
                    "Generated": entry.generated_at or "not recorded",
                    "Status": "present" if entry.exists else "missing",
                    "Detail": entry.detail,
                    "Path": str(entry.path),
                }
                for entry in inventory
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    layout.section("Identity and digest checks", "Declared digests come from the Phase 9 run manifest.")
    if risk_manifest:
        model = risk_manifest.get("model") or {}
        risk_cols = st.columns(3)
        with risk_cols[0]:
            st.metric("Model family", risk_manifest.get("model_family") or "not recorded")
        with risk_cols[1]:
            st.metric("Dataset hash", str(risk_manifest.get("dataset_sha256") or "")[:12] or "not recorded")
        with risk_cols[2]:
            st.metric("Model SHA-256", str(risk_manifest.get("model_sha256") or "")[:12] or "not recorded")
        st.caption(
            f"Model run: {risk_manifest.get('model_run_id')}; preprocessing run: "
            f"{risk_manifest.get('preprocessing_run_id')}; benchmark: {risk_manifest.get('benchmark_run_id')}."
        )
        if st.button("Verify model and preprocessor digests", key="cw_verify_digests"):
            rows = _digest_check(settings, risk_manifest)
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                if all(row["Match"] for row in rows):
                    st.success("Declared digests match the files on disk.")
                else:
                    st.error("At least one digest does not match the declared value; reload the run before trusting it.")
            else:
                st.warning(
                    "The manifest does not expose both a readable model path and a readable preprocessor path, "
                    "so the digests could not be verified."
                )
    else:
        st.info("No risk manifest exists, so there are no model digests to verify.")

    layout.section("Validation failures and warnings")
    checks = (validation or {}).get("checks") or []
    failed = [check for check in checks if not check.get("passed")]
    if failed:
        st.error(f"{len(failed)} validation check(s) failed or warned.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": check.get("id"),
                        "Check": str(check.get("name") or "").replace("_", " "),
                        "Severity": check.get("severity"),
                        "Detail": check.get("detail"),
                    }
                    for check in failed
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("All recorded validation checks passed.")
    gate = (validation or {}).get("gate") or {}
    if gate.get("blocking"):
        st.error("Blocking gates: " + ", ".join(str(item) for item in gate["blocking"]))
    if gate.get("warnings"):
        st.warning("Gate warnings: " + ", ".join(str(item) for item in gate["warnings"]))

    warnings: list[str] = []
    for source in (validation, metrics, risk, simulation):
        warnings.extend((source or {}).get("warnings") or [])
    warnings = list(dict.fromkeys(str(item) for item in warnings))
    if warnings:
        with st.expander(f"Recorded warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")

    limitations: list[str] = []
    for source in (metrics, risk, simulation):
        limitations.extend((source or {}).get("limitations") or [])
    limitations = list(dict.fromkeys(str(item) for item in limitations))
    layout.section("Stated limitations")
    if limitations:
        for item in limitations:
            st.write(f"- {item}")
    else:
        st.caption("No limitation strings were recorded by the available artifacts.")

    layout.section("Privacy and cache behaviour")
    st.markdown(
        "- Risk artifacts contain opaque record references and transformed values only; raw customer attributes "
        "are not copied into the dashboard.\n"
        "- Loads are cached by file path, modification time and size, so a rewritten artifact invalidates its cache "
        "entry on the next render.\n"
        "- Discovery, acquisition and training run only from labelled actions; a page rerun never re-executes them.\n"
        "- No page computes SHAP, fits a model or downloads a dataset during rendering."
    )
    st.caption("Use the sidebar 'Refresh artifact cache' button after an external pipeline run.")


__all__ = ["render"]
