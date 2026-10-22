"""Phase 10 integration: real Phase 9 scores, assumptions, artifacts and CLI.

The Random Forest and the Phase 9 risk table are genuinely produced by the
existing pipeline; only this test module's synthetic dataset is artificial.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from churnwise.cli import main as cli_main
from churnwise.config import Settings
from churnwise.intelligence import RiskConfig, RiskService
from churnwise.simulation import (
    CampaignAssumptions,
    CampaignScenario,
    RiskContext,
    TargetingConfig,
    SimulationError,
    SimulationService,
    assume_scenario,
    run_campaign,
)
from tests.risk_helpers import BENCHMARK_ID, build_risk_fixture

SCENARIO_ID = "20261022T000000Z-sim-fixture"


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    root = tmp_path_factory.mktemp("simulation")
    settings, _dataset, _ = build_risk_fixture(root, rows=180, seed=11)
    service = RiskService(
        settings,
        RiskConfig(top_k=2, batch_size=20, max_driver_rows=0),
        benchmark_run_id=BENCHMARK_ID,
    )
    run = service.score_population(
        split="val", explain="none", run_id=SCENARIO_ID
    )
    return settings, run


def _assert_artifact(path: str) -> Path:
    resolved = Path(path)
    assert resolved.is_file(), path
    assert resolved.stat().st_size > 0, path
    return resolved


def test_simulation_uses_actual_phase9_probabilities_and_persists(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    scenario = assume_scenario(
        "Fixture pilot",
        budget=40.0,
        cost_per_contact=2.0,
        effectiveness=0.30,
        monthly_value=30.0,
        horizon_months=6.0,
        min_risk_probability=0.20,
    )
    run = service.run_scenario(
        scenario, run_id=risk_run.run_id, scenario_id="20261022T010000Z-fixture-pilot"
    )

    table = risk_run.table
    assert run.result.context.run_id == risk_run.run_id
    assert run.result.context.model_family == "random_forest"
    assert run.result.assessed_count == len(table.records)
    assert run.result.eligible_count == sum(
        1 for item in table.records if item.churn_probability >= 0.20
    )
    for selected in run.result.selected:
        source = table.get(selected.record_reference)
        assert source is not None
        assert selected.churn_probability == source.churn_probability
    selected_references = [item.record_reference for item in run.result.selected]
    expected_top = [
        item.record_reference
        for item in sorted(
            table.records, key=lambda item: (-item.churn_probability, item.record_reference)
        )
        if item.churn_probability >= 0.20
    ][:20]
    assert selected_references == expected_top
    assert run.result.counts_reconcile

    for name in (
        "result",
        "report",
        "selection",
        "sensitivity",
        "sensitivity_csv",
        "run_manifest",
        "plot_allocation",
        "plot_sensitivity_effectiveness",
    ):
        _assert_artifact(run.paths[name])
    for name, path in run.paths.items():
        if name.startswith("plot_"):
            assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    report = Path(run.paths["report"]).read_text(encoding="utf-8")
    for section in (
        "Campaign configuration and assumptions",
        "Targeting, eligibility and allocation",
        "Hypothetical outcome projection",
        "Sensitivity analysis",
        "Warnings",
        "Inputs, provenance and limitations",
    ):
        assert section in report
    assert "not observed retention results" in report

    payload = service.load_result("20261022T010000Z-fixture-pilot")
    assert payload["scenario_id"] == "20261022T010000Z-fixture-pilot"
    assert payload["assumptions"]["effectiveness"] == pytest.approx(0.30)
    assert payload["assumptions"]["cost_per_contact"] == pytest.approx(2.0)
    assert payload["budget"] == pytest.approx(40.0)
    assert payload["disclaimers"]
    assert payload["projection"]["projection"] is True
    assert payload["hypothetical"] is True


def test_sensitivity_points_equal_fresh_recomputation(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    scenario = assume_scenario(
        "Recompute", budget=30.0, cost_per_contact=2.0, effectiveness=0.25,
        monthly_value=25.0,
    )
    run = service.run_scenario(
        scenario,
        run_id=risk_run.run_id,
        scenario_id="20261022T010100Z-recompute",
        effectiveness_values="0.1,0.25,0.5",
        budget_values="10,30",
        cost_values="1,2",
    )
    context = RiskContext.from_table(risk_run.table)
    analysis = run.sensitivity
    assert analysis is not None
    assert {series.dimension for series in analysis.series} == {
        "effectiveness",
        "budget",
        "cost_per_contact",
    }
    for series in analysis.series:
        for point in series.points:
            if series.dimension == "effectiveness":
                varied = replace(
                    scenario,
                    assumptions=replace(
                        scenario.assumptions, effectiveness=point.value
                    ),
                )
            elif series.dimension == "budget":
                varied = replace(scenario, budget=point.value)
            else:
                varied = replace(
                    scenario,
                    assumptions=replace(
                        scenario.assumptions, cost_per_contact=point.value
                    ),
                )
            fresh = run_campaign(risk_run.table.records, varied, context=context)
            assert point.selected_count == fresh.allocation.selected_count
            assert point.allocated_spend == pytest.approx(
                fresh.allocation.allocated_spend
            )
            assert point.expected_prevented_churn == pytest.approx(
                fresh.projection.expected_prevented_churn
            )

    csv_path = Path(run.paths["sensitivity_csv"])
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("dimension,unit,value,is_baseline")
    assert "selection_fingerprint" in header
    baseline_rows = [
        line for line in csv_path.read_text(encoding="utf-8").splitlines()
        if ",true," in line
    ]
    assert len(baseline_rows) == 3


def test_comparison_persists_rows_and_discloses_assumptions(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    primary = assume_scenario(
        "Compare baseline",
        budget=40.0,
        cost_per_contact=2.0,
        effectiveness=0.30,
        monthly_value=30.0,
    )
    alternative = CampaignScenario(
        name="Compare targeted",
        budget=20.0,
        description="High-risk only, lower assumed effectiveness",
        assumptions=CampaignAssumptions(
            cost_per_contact=4.0,
            effectiveness=0.15,
            monthly_value=30.0,
        ),
        targeting=TargetingConfig(
            strategy="risk_category", risk_tiers=("high", "critical")
        ),
    )
    run = service.run_scenario(
        primary,
        run_id=risk_run.run_id,
        scenario_id="20261022T010200Z-comparison",
        include_sensitivity=False,
        compare_with=(alternative,),
    )
    comparison = run.comparison
    assert comparison is not None
    assert len(comparison.rows) == 2
    assert comparison.shared_inputs is True
    assert set(comparison.differing_assumptions) >= {
        "budget",
        "cost_per_contact",
        "effectiveness",
        "strategy",
        "risk_tiers",
    }
    assert any("does not" in note for note in comparison.notes)
    assert comparison.disclaimers

    payload = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2
    assert payload["rows"][1]["assumed_effectiveness"] == pytest.approx(0.15)
    csv_lines = Path(run.paths["comparison_csv"]).read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 3
    assert "risk_tiers" in csv_lines[0]
    assert Path(run.paths["report"]).read_text(encoding="utf-8").count(
        "Scenario comparison"
    ) >= 1


def test_zero_cost_and_zero_budget_on_real_scores(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    free = service.run_scenario(
        assume_scenario(
            "Free interventions", budget=100.0, cost_per_contact=0.0,
            effectiveness=0.5, monthly_value=10.0,
        ),
        run_id=risk_run.run_id,
        scenario_id="20261022T010300Z-free",
        include_sensitivity=False,
    )
    assert free.result.allocation.zero_cost is True
    assert len(free.result.selected) == free.result.eligible_count
    assert free.result.projection.roi is None
    assert "campaign cost is zero" in free.result.projection.undefined_metrics["roi"]

    broke = service.run_scenario(
        assume_scenario(
            "No budget", budget=0.0, cost_per_contact=2.0, effectiveness=0.4
        ),
        run_id=risk_run.run_id,
        scenario_id="20261022T010400Z-no-budget",
        include_sensitivity=False,
    )
    assert broke.result.selected == []
    assert broke.result.projection.expected_prevented_churn == pytest.approx(0.0)
    assert broke.result.projection.cost_per_expected_prevented_churn is None
    assert broke.result.allocation.utilization is None


def test_report_and_csv_carry_projections_and_guard_formulas(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    scenario = assume_scenario(
        "=Injected name",
        budget=20.0,
        cost_per_contact=2.0,
        effectiveness=0.2,
        monthly_value=15.0,
        description="+cmd payload",
    )
    run = service.run_scenario(
        scenario,
        run_id=risk_run.run_id,
        scenario_id="20261022T010500Z-injection",
        include_sensitivity=False,
        compare_with=(assume_scenario("Plain", budget=10.0, cost_per_contact=2.0),),
    )
    report = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "hypothetical projections" in report.lower()
    assert risk_run.run_id in report
    assert "Assumed effectiveness" in report
    assert "20.00%" in report
    assert "+cmd payload" in report
    csv_text = Path(run.paths["comparison_csv"]).read_text(encoding="utf-8")
    assert "'=Injected name" in csv_text
    result_payload = json.loads(Path(run.paths["result"]).read_text(encoding="utf-8"))
    assert result_payload["scenario"]["name"] == "=Injected name"


def test_cli_simulate_end_to_end(scored, tmp_path) -> None:
    settings, risk_run = scored
    output = tmp_path / "cli-simulations"
    code = cli_main(
        [
            "simulate",
            "--artifact-dir",
            str(settings.artifact_dir),
            "--output-dir",
            str(output),
            "--run-id",
            risk_run.run_id,
            "--scenario-id",
            "cli-fixture-run",
            "--name",
            "CLI pilot",
            "--budget",
            "20",
            "--cost-per-contact",
            "4",
            "--effectiveness",
            "0.4",
            "--monthly-value",
            "12",
            "--no-sensitivity",
        ]
    )
    assert code == 0
    payload = json.loads(
        (output / "cli-fixture-run" / "result.json").read_text(encoding="utf-8")
    )
    assert payload["phase"] == 10
    assert payload["scenario"]["name"] == "CLI pilot"
    assert payload["allocation"]["allocated_spend"] <= 20.0
    assert (output / "cli-fixture-run" / "report.md").is_file()

    invalid = cli_main(
        [
            "simulate",
            "--artifact-dir",
            str(settings.artifact_dir),
            "--output-dir",
            str(output),
            "--effectiveness",
            "1.5",
        ]
    )
    assert invalid == 3


def test_degenerate_cost_grid_is_skipped_and_reported(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    run = service.run_scenario(
        assume_scenario(
            "Free sensitivity", budget=20.0, cost_per_contact=0.0, effectiveness=0.5
        ),
        run_id=risk_run.run_id,
        scenario_id="20261022T010800Z-free-sensitivity",
    )
    assert run.sensitivity is not None
    dimensions = [series.dimension for series in run.sensitivity.series]
    assert "cost_per_contact" not in dimensions
    assert any("skipped" in warning for warning in run.sensitivity.warnings)
    report = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "Sensitivity warnings" in report
    assert "skipped" in report


def test_service_comparison_and_sensitivity_helpers(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    primary = assume_scenario(
        "Helper A", budget=20.0, cost_per_contact=2.0, effectiveness=0.2
    )
    alternative = assume_scenario(
        "Helper B", budget=10.0, cost_per_contact=2.0, effectiveness=0.4
    )
    run = service.run_comparison(
        [primary, alternative],
        run_id=risk_run.run_id,
        scenario_id="20261022T010700Z-helpers",
        persist=False,
    )
    assert run.comparison is not None
    assert len(run.comparison.rows) == 2
    assert run.sensitivity is None
    assert run.result.scenario.name == "Helper A"

    analysis = service.sensitivity(
        primary,
        table=risk_run.table,
        scenario_id="helper-sensitivity",
        dimensions=("effectiveness",),
        effectiveness_values="0.1,0.2",
    )
    assert len(analysis.series) == 1
    assert [point.value for point in analysis.series[0].points] == [0.1, 0.2]
    assert analysis.series[0].points[-1].is_baseline is True


def test_missing_and_broken_risk_runs_are_structured(tmp_path) -> None:
    empty = Settings(project_root=tmp_path)
    service = SimulationService(empty)
    with pytest.raises(SimulationError) as missing:
        service.run_scenario(assume_scenario("No run", budget=10.0))
    assert missing.value.code == "CW-SIM-003"
    assert "score" in missing.value.remediation

    with pytest.raises(SimulationError) as broken:
        service.load_table("20261022T000000Z-does-not-exist")
    assert broken.value.code == "CW-SIM-004"
    assert broken.value.details.get("upstream_code")


def test_artifact_identity_and_path_guards(scored) -> None:
    settings, risk_run = scored
    service = SimulationService(settings)
    service.run_scenario(
        assume_scenario("Guarded", budget=10.0, cost_per_contact=2.0),
        run_id=risk_run.run_id,
        scenario_id="20261022T010600Z-guarded",
        include_sensitivity=False,
    )
    with pytest.raises(SimulationError):
        service.load_result("../outside")
    with pytest.raises(SimulationError):
        service.run_scenario(
            assume_scenario("Escape", budget=10.0, cost_per_contact=2.0),
            run_id=risk_run.run_id,
            scenario_id="../escape",
        )
    assert not (settings.artifact_dir / "escape").exists()
    path = settings.artifact_dir / "simulations" / "20261022T010600Z-guarded" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenario_id"] = "20261022T010600Z-other"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SimulationError) as tampered:
        service.load_result("20261022T010600Z-guarded")
    assert "does not match" in tampered.value.message
