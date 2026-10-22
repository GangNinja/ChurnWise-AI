"""Stable Phase 10 interface: load Phase 9 scores, simulate, compare, persist.

The service accepts **actual Phase 9 outputs** through :class:`RiskService`
(``load_run``) or an already-loaded :class:`RiskTable`, and writes every
artifact beneath ``artifacts/simulations/<scenario_id>/``. Probabilities are
never re-estimated here and assumptions are never inferred from the data.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from churnwise.config import Settings
from churnwise.intelligence.contracts import RiskTable
from churnwise.intelligence.errors import RiskError
from churnwise.intelligence.service import RiskService, sanitize_csv_value
from churnwise.simulation.campaign import (
    compare_scenarios,
    run_campaign,
    run_sensitivity,
)
from churnwise.simulation.contracts import (
    DISCLAIMERS,
    SIMULATION_SCHEMA_VERSION,
    CampaignScenario,
    RiskContext,
    SensitivityAnalysis,
    SimulationRun,
)
from churnwise.simulation.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_EXPORT_FAILED,
    CODE_INVALID_CONFIG,
    CODE_INVALID_INPUT,
    CODE_NO_RISK_RUN,
    SimulationError,
)
from churnwise.simulation.plots import (
    plot_budget_allocation,
    plot_scenario_comparison,
    plot_sensitivity_series,
)
from churnwise.simulation.reporting import (
    render_comparison_rows,
    render_report,
    render_summary,
)

SCENARIO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,180}$")


def validate_scenario_id(scenario_id: str) -> str:
    """Reject path traversal and separator characters in a scenario id."""
    if (
        not isinstance(scenario_id, str)
        or not SCENARIO_ID_PATTERN.fullmatch(scenario_id)
        or ".." in scenario_id
    ):
        raise SimulationError(
            CODE_INVALID_CONFIG,
            f"invalid scenario id {scenario_id!r}",
            "Use an id printed by the simulate command or a name-safe identifier.",
        )
    return scenario_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise SimulationError(
            CODE_EXPORT_FAILED,
            f"could not write simulation artifact {path}: {exc}",
            "Check that the artifact directory is writable.",
        ) from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SimulationError(
            CODE_EXPORT_FAILED,
            f"simulation payload for {path.name} is not JSON-serializable: {exc}",
            "Report this defect; do not persist the artifact.",
        ) from exc
    _atomic_write_text(path, text + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {name: sanitize_csv_value(row.get(name, "")) for name in columns}
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise SimulationError(
            CODE_EXPORT_FAILED,
            f"could not write simulation CSV {path}: {exc}",
            "Check that the artifact directory is writable.",
        ) from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass


SELECTION_COLUMNS = (
    "record_reference",
    "input_position",
    "risk_rank",
    "churn_probability",
    "risk_tier",
    "intervention_cost",
    "expected_prevented_churn",
)
COMPARISON_COLUMNS = (
    "scenario_id",
    "name",
    "targeting_strategy",
    "budget",
    "cost_per_contact",
    "assumed_effectiveness",
    "monthly_value",
    "horizon_months",
    "value_unit",
    "min_risk_probability",
    "risk_tiers",
    "max_contacts",
    "eligible_count",
    "selected_count",
    "allocated_spend",
    "remaining_budget",
    "budget_utilization",
    "expected_prevented_churn",
    "cost_per_expected_prevented_churn",
    "expected_saved_value",
    "net_benefit",
    "roi",
    "break_even_effectiveness",
    "targeting_lift_vs_random",
)
SENSITIVITY_COLUMNS = (
    "dimension",
    "unit",
    "value",
    "is_baseline",
    "eligible_count",
    "selected_count",
    "selection_fingerprint",
    "allocated_spend",
    "remaining_budget",
    "expected_prevented_churn",
    "expected_saved_value",
    "cost_per_expected_prevented_churn",
    "budget_utilization",
)


class SimulationService:
    """Run, compare and persist retention-campaign simulations."""

    def __init__(
        self,
        settings: Settings,
        risk_service: RiskService | None = None,
        *,
        simulations_dir: Path | str | None = None,
        now: Callable[[], str] = _utc_now,
    ) -> None:
        self.settings = settings
        self.artifact_dir = Path(settings.artifact_dir)
        self.risk_service = risk_service or RiskService(settings)
        self.simulations_dir = Path(
            simulations_dir or (self.artifact_dir / "simulations")
        )
        self._now = now

    # -- inputs ------------------------------------------------------------
    def load_table(
        self, run_id: str | None = None, *, verify_freshness: bool = True
    ) -> RiskTable:
        """Load a persisted Phase 9 risk table, failing with clear remediation."""
        resolved = run_id or self.risk_service.latest_run_id()
        if not resolved:
            raise SimulationError(
                CODE_NO_RISK_RUN,
                "no Phase 9 risk run is available to simulate",
                "Run `python -m churnwise.cli score --population val` first.",
            )
        try:
            run = self.risk_service.load_run(
                resolved, verify_freshness=verify_freshness
            )
        except RiskError as exc:
            raise SimulationError(
                CODE_ARTIFACT_FAILED,
                f"Phase 9 risk run {resolved!r} could not be used: {exc.message}",
                exc.remediation or "Rerun scoring and retry.",
                {"upstream_code": exc.code},
            ) from exc
        return run.table

    # -- runs --------------------------------------------------------------
    def run_scenario(
        self,
        scenario: CampaignScenario,
        *,
        table: RiskTable | None = None,
        run_id: str | None = None,
        scenario_id: str | None = None,
        include_sensitivity: bool = True,
        sensitivity_dimensions: Sequence[str] | None = None,
        effectiveness_values: Any | None = None,
        budget_values: Any | None = None,
        cost_values: Any | None = None,
        compare_with: Sequence[CampaignScenario] = (),
        persist: bool = True,
    ) -> SimulationRun:
        """Run one primary scenario, optionally comparing extra scenarios."""
        scenario.validate_or_raise()
        started = time.perf_counter()
        resolved_table = table if table is not None else self.load_table(run_id)
        context = RiskContext.from_table(resolved_table)
        resolved_id = (
            validate_scenario_id(scenario_id)
            if scenario_id
            else self._next_scenario_id(scenario)
        )

        primary = run_campaign(
            resolved_table.records,
            scenario,
            context=context,
            scenario_id=resolved_id,
        )
        results = [primary]
        for index, extra in enumerate(compare_with, start=1):
            extra.validate_or_raise()
            results.append(
                run_campaign(
                    resolved_table.records,
                    extra,
                    context=context,
                    scenario_id=f"{resolved_id}-cmp{index}",
                )
            )
        comparison = compare_scenarios(results) if results[1:] else None

        sensitivity: SensitivityAnalysis | None = None
        if include_sensitivity:
            sensitivity = run_sensitivity(
                resolved_table.records,
                scenario,
                context=context,
                scenario_id=resolved_id,
                dimensions=sensitivity_dimensions,
                effectiveness_values=effectiveness_values,
                budget_values=budget_values,
                cost_values=cost_values,
            )

        run = SimulationRun(
            scenario_id=resolved_id,
            result=primary,
            comparison=comparison,
            sensitivity=sensitivity,
            duration_seconds=time.perf_counter() - started,
            warnings=primary.warnings,
        )
        if persist:
            run.paths = self._persist(run)
        return run

    def run_comparison(
        self,
        scenarios: Sequence[CampaignScenario],
        *,
        table: RiskTable | None = None,
        run_id: str | None = None,
        scenario_id: str | None = None,
        include_sensitivity: bool = False,
        persist: bool = True,
    ) -> SimulationRun:
        """Run several scenarios against one risk table and compare them."""
        if not scenarios:
            raise SimulationError(
                CODE_INVALID_INPUT,
                "scenario comparison requires at least one scenario",
                "Pass one or more validated scenarios.",
            )
        return self.run_scenario(
            scenarios[0],
            table=table,
            run_id=run_id,
            scenario_id=scenario_id,
            include_sensitivity=include_sensitivity,
            compare_with=scenarios[1:],
            persist=persist,
        )

    def sensitivity(
        self,
        scenario: CampaignScenario,
        *,
        table: RiskTable | None = None,
        run_id: str | None = None,
        scenario_id: str = "",
        dimensions: Sequence[str] | None = None,
        effectiveness_values: Any | None = None,
        budget_values: Any | None = None,
        cost_values: Any | None = None,
    ) -> SensitivityAnalysis:
        """Standalone sensitivity analysis for dashboards and notebooks."""
        scenario.validate_or_raise()
        resolved_table = table if table is not None else self.load_table(run_id)
        return run_sensitivity(
            resolved_table.records,
            scenario,
            context=RiskContext.from_table(resolved_table),
            scenario_id=scenario_id,
            dimensions=dimensions,
            effectiveness_values=effectiveness_values,
            budget_values=budget_values,
            cost_values=cost_values,
        )

    # -- persistence -------------------------------------------------------
    def _next_scenario_id(self, scenario: CampaignScenario) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"{stamp}-{scenario.slug}"
        candidate = base
        counter = 2
        while (self.simulations_dir / candidate).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _persist(self, run: SimulationRun) -> dict[str, str]:
        root = self.simulations_dir / run.scenario_id
        paths: dict[str, str] = {"root": str(root)}

        selection_rows = [
            {
                "record_reference": item.record_reference,
                "input_position": item.input_position,
                "risk_rank": "" if item.rank is None else item.rank,
                "churn_probability": repr(float(item.churn_probability)),
                "risk_tier": item.risk_tier,
                "intervention_cost": repr(float(item.intervention_cost)),
                "expected_prevented_churn": repr(
                    float(item.expected_prevented_churn)
                ),
            }
            for item in run.result.selected
        ]
        selection_path = root / "selection.csv"
        _write_csv(selection_path, selection_rows, SELECTION_COLUMNS)
        paths["selection"] = str(selection_path)

        plots: list[dict[str, str]] = []
        try:
            allocation_path = plot_budget_allocation(
                run.result.allocation,
                root / "plots" / "budget_allocation.png",
                unit=run.result.scenario.assumptions.value_unit,
                scenario_name=run.result.scenario.name,
            )
            paths["plot_allocation"] = allocation_path
            plots.append(
                {
                    "label": "Budget allocation",
                    "path": "plots/budget_allocation.png",
                }
            )
        except SimulationError as exc:
            run.warnings.append(f"Budget-allocation chart unavailable: {exc.message}")

        if run.comparison is not None:
            comparison_path = root / "comparison.json"
            _write_json(comparison_path, run.comparison.to_dict())
            paths["comparison"] = str(comparison_path)
            comparison_csv = root / "comparison.csv"
            _write_csv(
                comparison_csv,
                [
                    {
                        **row,
                        "risk_tiers": ",".join(row["risk_tiers"]),
                    }
                    for row in render_comparison_rows(run.comparison)
                ],
                COMPARISON_COLUMNS,
            )
            paths["comparison_csv"] = str(comparison_csv)
            if len(run.comparison.rows) > 1:
                try:
                    comparison_plot = plot_scenario_comparison(
                        run.comparison,
                        root / "plots" / "scenario_comparison.png",
                        unit=run.result.scenario.assumptions.value_unit,
                    )
                    paths["plot_comparison"] = comparison_plot
                    plots.append(
                        {
                            "label": "Scenario comparison",
                            "path": "plots/scenario_comparison.png",
                        }
                    )
                except SimulationError as exc:
                    run.warnings.append(
                        f"Scenario-comparison chart unavailable: {exc.message}"
                    )

        if run.sensitivity is not None:
            sensitivity_path = root / "sensitivity.json"
            _write_json(sensitivity_path, run.sensitivity.to_dict())
            paths["sensitivity"] = str(sensitivity_path)
            sensitivity_csv = root / "sensitivity.csv"
            _write_csv(
                sensitivity_csv,
                run.sensitivity.csv_rows(),
                SENSITIVITY_COLUMNS,
            )
            paths["sensitivity_csv"] = str(sensitivity_csv)
            for series in run.sensitivity.series:
                try:
                    series_path = plot_sensitivity_series(
                        series,
                        root / "plots" / f"sensitivity_{series.dimension}.png",
                        unit=run.result.scenario.assumptions.value_unit,
                        scenario_name=run.result.scenario.name,
                    )
                except SimulationError as exc:
                    run.warnings.append(
                        f"Sensitivity chart for {series.dimension} unavailable: "
                        f"{exc.message}"
                    )
                    continue
                paths[f"plot_sensitivity_{series.dimension}"] = series_path
                plots.append(
                    {
                        "label": f"Sensitivity: {series.dimension}",
                        "path": f"plots/sensitivity_{series.dimension}.png",
                    }
                )

        result_path = root / "result.json"
        _write_json(result_path, run.result.to_dict())
        paths["result"] = str(result_path)

        report_path = root / "report.md"
        _atomic_write_text(
            report_path,
            render_report(
                run.result,
                comparison=run.comparison,
                sensitivity=run.sensitivity,
                plots=plots,
                manifest_path="run_manifest.json",
            ),
        )
        paths["report"] = str(report_path)

        manifest_path = root / "run_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": SIMULATION_SCHEMA_VERSION,
                "phase": 10,
                "scenario_id": run.scenario_id,
                "created_at": run.result.generated_at or self._now(),
                "duration_seconds": float(run.duration_seconds),
                "code_version": run.result.provenance.get("code_version"),
                "seed": None,
                "scenario": run.result.scenario.to_dict(),
                "scenario_fingerprint": run.result.scenario.fingerprint(),
                "risk": {
                    "run_id": run.result.context.run_id,
                    "generated_at": run.result.context.generated_at,
                    "model": run.result.context.to_dict()["model"],
                    "tier_boundaries": list(run.result.context.tier_boundaries),
                    "dataset_sha256": run.result.context.dataset_sha256,
                    "split_fingerprint": run.result.context.split_fingerprint,
                    "classification_threshold": float(
                        run.result.context.classification_threshold
                    ),
                },
                "assumptions": run.result.scenario.assumptions.to_dict(),
                "comparison": (
                    None
                    if run.comparison is None
                    else {
                        "scenarios": [row.name for row in run.comparison.rows],
                        "shared_risk_inputs": run.comparison.shared_inputs,
                    }
                ),
                "sensitivity": (
                    None
                    if run.sensitivity is None
                    else {
                        "dimensions": [
                            series.dimension for series in run.sensitivity.series
                        ],
                        "points": sum(
                            len(series.points) for series in run.sensitivity.series
                        ),
                    }
                ),
                "artifacts": paths,
                "warnings": list(run.warnings),
                "disclaimers": list(DISCLAIMERS),
            },
        )
        paths["run_manifest"] = str(manifest_path)
        return paths

    # -- retrieval ---------------------------------------------------------
    def latest_scenario_id(self) -> str | None:
        if not self.simulations_dir.is_dir():
            return None
        candidates = sorted(
            path.name
            for path in self.simulations_dir.iterdir()
            if path.is_dir() and (path / "result.json").is_file()
        )
        return candidates[-1] if candidates else None

    def load_result(self, scenario_id: str) -> dict[str, Any]:
        """Load a persisted ``result.json`` and verify its identity fields."""
        validate_scenario_id(scenario_id)
        path = self.simulations_dir / scenario_id / "result.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SimulationError(
                CODE_ARTIFACT_FAILED,
                f"simulation {scenario_id!r} could not be read",
                "Rerun the simulate command; the artifact may be missing or corrupt.",
            ) from exc
        if not isinstance(payload, dict):
            raise SimulationError(
                CODE_ARTIFACT_FAILED,
                f"simulation {scenario_id!r} is not a JSON object",
                "Rerun the simulate command; do not edit artifacts by hand.",
            )
        if payload.get("scenario_id") != scenario_id or payload.get("phase") != 10:
            raise SimulationError(
                CODE_ARTIFACT_FAILED,
                f"simulation {scenario_id!r} metadata does not match its directory",
                "Rerun the simulate command; do not edit artifacts by hand.",
            )
        required = ("assumptions", "projection", "allocation", "population", "disclaimers")
        missing = [name for name in required if name not in payload]
        if missing:
            raise SimulationError(
                CODE_ARTIFACT_FAILED,
                f"simulation {scenario_id!r} is missing field(s): {', '.join(missing)}",
                "Rerun the simulate command with this project version.",
            )
        return payload

    def render(self, run: SimulationRun) -> str:
        """Console summary for a completed run."""
        text = render_summary(run.result)
        if run.sensitivity is not None and run.sensitivity.warnings:
            text += "\nSensitivity warnings:"
            for warning in run.sensitivity.warnings:
                text += f"\n  - {warning}"
        if run.paths:
            text += "\nArtifacts:"
            for name in ("result", "report", "selection", "sensitivity_csv"):
                if name in run.paths:
                    text += f"\n  {name}: {run.paths[name]}"
        return text


__all__ = [
    "COMPARISON_COLUMNS",
    "SELECTION_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "SimulationService",
    "validate_scenario_id",
]
