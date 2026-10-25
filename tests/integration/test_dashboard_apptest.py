"""Phase 11 dashboard tests: AppTest rendering, states, filters and actions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard.actions import DashboardActionError
from tests.dashboard_helpers import (
    APP_PATH,
    BENCHMARK_ID,
    RISK_RUN_ID,
    apply_environment,
    install_dashboard_project,
)

LABELS = [
    "Executive Dashboard",
    "Customer Risk Explorer",
    "Model Laboratory",
    "Retention Workbench",
    "Campaign Simulator",
]


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    root = tmp_path / "dashboard-project"
    root.mkdir()
    info = install_dashboard_project(root)
    apply_environment(monkeypatch, root)
    return info


@pytest.fixture
def empty_dashboard(tmp_path, monkeypatch):
    root = tmp_path / "empty-project"
    root.mkdir()
    apply_environment(monkeypatch, root)
    return root


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def _go(at: AppTest, label: str) -> AppTest:
    at.radio[0].set_value(label).run()
    return at


def test_all_dashboard_pages_render_with_real_artifact_fixture(dashboard):
    at = _app()
    for label in LABELS:
        _go(at, label)
        assert not at.exception, f"{label}: {[str(item.value) for item in at.exception]}"
    markdown = "\n".join(str(item.value) for item in at.markdown)
    assert "Run id" in markdown
    assert "Model family" in markdown
    assert "Dataset hash" in markdown
    assert "Generated at" in markdown


def test_missing_artifacts_are_actionable_on_every_page(empty_dashboard):
    at = _app()
    for label in LABELS:
        _go(at, label)
        assert not at.exception, f"{label}: {[str(item.value) for item in at.exception]}"
        assert at.code, f"{label} did not expose a next-step command"
        assert any("churnwise.cli" in str(item.value) for item in at.code)
    body = "\n".join(str(item.value) for item in at.markdown) + "\n" + "\n".join(
        str(item.value) for item in at.warning
    )
    assert "not available" in body.lower()


def test_risk_tier_filter_changes_the_stored_table(dashboard):
    at = _go(_app(), "Retention Workbench")
    assert len(at.dataframe[0].value) == 6
    at.multiselect[0].set_value(["critical"]).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 2
    assert set(at.dataframe[0].value["Risk tier"]) == {"Critical"}


def test_risk_table_cache_invalidates_when_parquet_changes(dashboard):
    at = _go(_app(), "Retention Workbench")
    assert len(at.dataframe[0].value) == 6
    path = dashboard["risk_frame"]
    frame = pd.read_parquet(path).head(5).copy()
    frame.loc[:, "rank"] = range(1, len(frame) + 1)
    frame.to_parquet(path, index=False)
    at.run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 5


def test_customer_detail_matches_the_stored_probability(dashboard):
    at = _go(_app(), "Customer Risk Explorer")
    reference = at.selectbox[1]
    assert reference.label == "Customer reference"
    reference.set_value("fixture_ref_000").run()
    assert not at.exception
    body = "\n".join(str(item.value) for item in at.markdown)
    assert "0.9000" in body
    assert "Critical" in body
    assert "fixture_ref_000" in body
    assert any("Fixture local explanation summary" in str(item.value) for item in at.markdown)


def test_laboratory_renders_the_stored_global_explanation(dashboard):
    at = _go(_app(), "Model Laboratory")
    assert not at.exception
    explain_run = next(item for item in at.selectbox if item.label == "Explanation run")
    assert explain_run.options == ["20260101T010000Z-fixture-xai"]
    captions = "\n".join(str(item.value) for item in at.caption)
    assert "aggregated into 4 feature groups" in captions
    bodies = [frame.value for frame in at.dataframe if "feature" in getattr(frame.value, "columns", [])]
    assert bodies and set(bodies[0]["feature"]) == {"support_calls", "tenure_months"}


def test_training_success_runs_the_service_once(dashboard, monkeypatch):
    calls: list[dict] = []

    def fake_train(settings, config):
        calls.append(dict(config))
        return {
            "run_id": "20260101T040000Z-fixture-trained",
            "metadata": {
                "model_run_id": "20260101T040000Z-fixture-trained",
                "model_family": "mlp",
                "created_at": "2026-01-01T04:00:00Z",
                "architecture": {"hidden_units": [8], "parameter_count": 65},
                "seed": 42,
            },
            "history": {"loss": [0.7, 0.6], "val_loss": [0.72, 0.65], "accuracy": [0.5, 0.6]},
            "paths": {},
            "diagnostics": {},
        }

    monkeypatch.setattr(actions, "train_mlp", fake_train)
    at = _go(_app(), "Model Laboratory")
    button = next(item for item in at.button if item.label == "Train MLP")
    button.click().run()
    assert not at.exception
    assert len(calls) == 1
    assert any("Training completed" in str(item.value) for item in at.success)
    at.run()
    assert len(calls) == 1


def test_training_failure_is_structured_and_privacy_safe(dashboard, monkeypatch):
    def fail_train(settings, config):
        raise DashboardActionError("invalid MLP configuration: epochs must be positive", code="CW-UI-007")

    monkeypatch.setattr(actions, "train_mlp", fail_train)
    at = _go(_app(), "Model Laboratory")
    next(item for item in at.button if item.label == "Train MLP").click().run()
    assert not at.exception
    errors = "\n".join(str(item.value) for item in at.error)
    assert "CW-UI-007" in errors
    assert "Traceback" not in errors


def test_discovery_success_is_not_repeated_by_a_rerun(dashboard, monkeypatch):
    calls: list[dict] = []

    def fake_discovery(settings, **kwargs):
        calls.append(dict(kwargs))
        return {"selected": {"candidate_id": "fixture:churn"}}

    monkeypatch.setattr(actions, "discover_datasets", fake_discovery)
    at = _go(_app(), "Model Laboratory")
    next(item for item in at.button if item.label == "Start discovery").click().run()
    assert not at.exception
    assert len(calls) == 1
    assert any("Discovery completed" in str(item.value) for item in at.success)
    at.run()
    assert len(calls) == 1


def test_acquisition_refreshes_the_attempt_log_and_keeps_the_notice(dashboard, monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_acquire(settings, candidate_id, *, allow_fallback=True):
        calls.append((candidate_id, allow_fallback))
        path = dashboard["root"] / "reports" / "acquisition" / "attempts.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["attempts"].append(
            {
                "candidate_id": candidate_id,
                "rank": 1,
                "status": "success",
                "cached": False,
                "duration_ms": 7,
                "error_code": None,
                "error_message": None,
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "success", "cached": False}

    monkeypatch.setattr(actions, "acquire_candidate", fake_acquire)
    at = _go(_app(), "Model Laboratory")
    next(item for item in at.button if item.label == "Acquire selected candidate").click().run()
    assert not at.exception
    assert len(calls) == 1
    assert calls[0][1] is True
    assert any("Acquisition finished" in str(item.value) for item in at.success)
    attempts = [
        frame.value for frame in at.dataframe if "candidate_id" in getattr(frame.value, "columns", [])
    ]
    assert attempts and len(attempts[0]) == 2
    at.run()
    assert len(calls) == 1


def test_simulator_defaults_to_the_persisted_source_risk_run(dashboard):
    source = dashboard["risk_run_id"]
    artifact_dir = dashboard["artifact_dir"]
    newer = artifact_dir / "predictions" / "20260101T040000Z-fixture-newer-risk"
    shutil.copytree(artifact_dir / "predictions" / source, newer)
    summary_path = newer / "risk_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_id"] = newer.name
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    at = _go(_app(), "Campaign Simulator")
    assert not at.exception
    risk_run = next(item for item in at.selectbox if item.label == "Risk run")
    assert risk_run.value == source


def test_discovery_failure_is_shown_without_a_traceback(dashboard, monkeypatch):
    def fail_discovery(settings, **kwargs):
        raise DashboardActionError("source request failed", code="CW-UI-004", remediation="Retry later.")

    monkeypatch.setattr(actions, "discover_datasets", fail_discovery)
    at = _go(_app(), "Model Laboratory")
    next(item for item in at.button if item.label == "Start discovery").click().run()
    assert not at.exception
    errors = "\n".join(str(item.value) for item in at.error)
    assert "CW-UI-004" in errors
    assert "Retry later" in errors
    assert "Traceback" not in errors


def test_page_headers_show_all_identity_fields(dashboard):
    at = _go(_app(), "Executive Dashboard")
    markdown = "\n".join(str(item.value) for item in at.markdown)
    for label in ("Run id", "Model family", "Dataset hash", "Generated at"):
        assert label in markdown


def test_laboratory_shows_stored_test_metrics(dashboard):
    at = _go(_app(), "Model Laboratory")
    assert not at.exception
    captions = "\n".join(str(item.value) for item in at.caption)
    assert "Test F1 0.8800" in captions
    assert "Test ROC-AUC 0.9400" in captions
    assert "Selection used validation F1 only" in captions


def test_laboratory_reports_fold_safe_metadata_from_stored_cv(dashboard):
    at = _go(_app(), "Model Laboratory")
    frames = [
        frame.value
        for frame in at.dataframe
        if "Refit per fold" in getattr(frame.value, "columns", [])
    ]
    assert frames, "cross-validation metadata table was not rendered"
    assert set(frames[0]["Refit per fold"]) == {"yes"}
    assert set(frames[0]["Verified against Phase 5"]) == {"yes"}


def test_laboratory_presents_warning_checks_as_non_blocking(dashboard):
    at = _go(_app(), "Model Laboratory")
    body = "\n".join(str(item.value) for item in at.markdown)
    captions = "\n".join(str(item.value) for item in at.caption)
    warnings = "\n".join(str(item.value) for item in at.warning)
    assert "cw-pill warn" in body
    assert "1 passed, 1 warning-level not passed, 0 fatal not passed" in captions
    assert "warning-level check(s) did not pass" in warnings
    assert not at.error


def test_executive_derives_high_risk_counts_from_stored_scores(dashboard):
    path = dashboard["risk_summary"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("tier_counts", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    at = _go(_app(), "Executive Dashboard")
    assert not at.exception
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Customers Scored"] == "6"
    assert metrics["Predicted Churn"].startswith("3 ")
    assert metrics["High-Risk Customers"].startswith("3 ")
    captions = "\n".join(str(item.value) for item in at.caption)
    assert "different populations" in captions
    assert "stored classification threshold" in captions


def test_assessment_section_shows_schema_driven_inputs(dashboard):
    at = _go(_app(), "Customer Risk Explorer")
    assert not at.exception
    labels = {item.label for item in at.text_input}
    assert {"tenure_months", "monthly_charges", "support_calls"} <= labels
    assert any(item.label == "contract" for item in at.selectbox)
    assert any("header-only CSV template" in item.label for item in at.download_button)


def test_assessment_actions_delegate_to_phase_9(tmp_path):
    from churnwise.intelligence import RiskConfig, RiskService, RiskThresholds
    from tests.models_helpers import FEATURES, synthetic_frame
    from tests.risk_helpers import BENCHMARK_ID as FIXTURE_BENCHMARK, build_risk_fixture

    settings, _, _ = build_risk_fixture(tmp_path, rows=120)
    frame = synthetic_frame(rows=12, seed=21)
    record = frame.iloc[0][FEATURES].to_dict()
    thresholds = (0.3, 0.5, 0.7)
    payload = actions.assess_customer_record(settings, record, thresholds=thresholds)
    direct = RiskService(
        settings,
        RiskConfig(thresholds=RiskThresholds(boundaries=thresholds)),
        benchmark_run_id=FIXTURE_BENCHMARK,
    ).assess_customer(record)
    assert payload["churn_probability"] == pytest.approx(direct.churn_probability, abs=1e-12)
    assert payload["risk_tier"] == direct.risk_tier
    assert payload["predicted_label"] == direct.predicted_label

    batch = actions.assess_batch_records(settings, frame, thresholds=thresholds)
    assert batch["scored"] == len(frame)
    assert batch["rejected"] == 0
    assert [item["input_position"] for item in batch["records"]] == list(range(len(frame)))

    missing = frame.drop(columns=["contract"])
    rejected = actions.assess_batch_records(settings, missing, thresholds=thresholds)
    assert rejected["scored"] == 0
    assert rejected["rejected"] == len(missing)
    assert "contract" in str(rejected.get("batch_error"))
    assert all(item["code"] == "CW-RISK-004" for item in rejected["rejections"])


def test_simulation_computed_by_action_matches_direct_engine(tmp_path):
    from churnwise.intelligence import RiskService
    from churnwise.simulation import (
        CampaignAssumptions,
        CampaignScenario,
        SimulationService,
        TargetingConfig,
    )
    from tests.risk_helpers import BENCHMARK_ID as FIXTURE_BENCHMARK, build_risk_fixture

    settings, _, _ = build_risk_fixture(tmp_path, rows=120)
    risk_run = RiskService(settings, benchmark_run_id=FIXTURE_BENCHMARK).score_population(
        split="val",
        explain="none",
        run_id="20260101T050000Z-dashboard-sim-risk",
    )
    payload = actions.run_campaign_projection(
        settings,
        risk_run_id=risk_run.run_id,
        budget=150.0,
        cost_per_contact=5.0,
        effectiveness=0.3,
        monthly_value=50.0,
        horizon_months=6.0,
        value_unit="value_units",
        strategy="highest_risk",
        min_risk_probability=0.3,
        risk_tiers=(),
        max_contacts=None,
    )
    direct = SimulationService(settings).run_scenario(
        CampaignScenario(
            name="Dashboard scenario",
            budget=150.0,
            assumptions=CampaignAssumptions(
                cost_per_contact=5.0,
                effectiveness=0.3,
                monthly_value=50.0,
                horizon_months=6.0,
                value_unit="value_units",
            ),
            targeting=TargetingConfig(strategy="highest_risk", min_risk_probability=0.3),
        ),
        run_id=risk_run.run_id,
        include_sensitivity=False,
        persist=False,
    )
    page_result = payload["result"]
    direct_result = direct.to_dict()["result"]
    assert page_result["allocation"]["selected_count"] == direct_result["allocation"]["selected_count"]
    assert page_result["allocation"]["allocated_spend"] == pytest.approx(
        direct_result["allocation"]["allocated_spend"]
    )
    assert page_result["projection"]["expected_prevented_churn"] == pytest.approx(
        direct_result["projection"]["expected_prevented_churn"]
    )


def test_executive_filters_restrict_the_customer_preview(dashboard):
    at = _go(_app(), "Executive Dashboard")
    assert not at.exception
    tier = next(item for item in at.pills if item.label == "Risk tier")
    tier.set_value("Critical").run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 2
    assert set(at.dataframe[0].value["Risk tier"]) == {"Critical"}
    body = "\n".join(str(item.value) for item in at.markdown)
    assert "tier: Critical" in body

    probability = next(item for item in at.selectbox if item.label == "Probability range")
    probability.set_value("0.00–0.20").run()
    assert not at.exception
    assert at.info
    assert "No customers match" in str(at.info[0].value)
