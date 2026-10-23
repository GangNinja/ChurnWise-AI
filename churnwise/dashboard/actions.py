"""Explicit side-effecting actions triggered from the dashboard.

Ordinary page rendering never imports these services. Each action is behind a
labelled button and delegates to the existing Phase 2/3/6/10 implementation; no
business logic is duplicated in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from churnwise.config import Settings
from churnwise.dashboard.data_access import latest_prepared_dir


class DashboardActionError(Exception):
    """A structured action failure that is safe to show in the UI."""

    def __init__(self, message: str, *, code: str = "CW-UI-003", remediation: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation


def _split_terms(terms: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(terms, str):
        values = tuple(part.strip() for part in terms.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in terms if str(part).strip())
    return values or ("churn", "customer churn", "telecom churn")


def discover_datasets(
    settings: Settings,
    *,
    terms: str | Sequence[str] = (),
    min_rows: int = 1000,
    max_results: int = 10,
) -> dict[str, Any]:
    """Run Phase 2 discovery and persist the ranked reports."""

    from churnwise.contracts import SearchQuery
    from churnwise.discovery import build_adapters, discover
    from churnwise.discovery.reporting import write_reports

    query = SearchQuery(
        terms=_split_terms(terms),
        min_rows=int(min_rows),
        max_results_per_source=int(max_results),
    )
    report = discover(build_adapters(settings, query), query, settings)
    try:
        write_reports(report, settings.report_dir)
    except OSError as exc:
        raise DashboardActionError(
            f"discovery results could not be written: {exc.__class__.__name__}",
            remediation="Check that the reports directory is writable.",
        ) from exc
    payload = report.to_dict()
    if report.selected is None:
        failure = report.failure
        detail = failure.message if failure is not None else "no candidate passed the discovery gates"
        raise DashboardActionError(
            f"discovery produced no acquirable candidate: {detail}",
            code="CW-UI-004",
            remediation="Inspect the rejection reasons in the report and retry with different terms.",
        )
    return payload


def acquire_candidate(
    settings: Settings, candidate_id: str, *, allow_fallback: bool = True
) -> dict[str, Any]:
    """Acquire one ranked candidate through Phase 3, reusing the cache.

    With ``allow_fallback`` the coordinator also receives the lower-ranked
    eligible candidates, so its documented fallback loop can try them if the
    preferred candidate fails. The UI never implements that loop itself.
    """

    from churnwise.acquisition import AcquisitionCoordinator, load_candidates
    from churnwise.acquisition.reporting import write_attempts
    from churnwise.dashboard.data_access import candidates_path

    path = candidates_path(settings)
    if not path.is_file():
        raise DashboardActionError(
            "no discovery report exists yet",
            code="CW-UI-005",
            remediation="Run discovery first.",
        )
    try:
        candidates = load_candidates(path)
    except (OSError, ValueError) as exc:
        raise DashboardActionError(
            f"the discovery report could not be read: {exc.__class__.__name__}",
            remediation="Rerun discovery to regenerate the report.",
        ) from exc
    ids = [str(getattr(item, "candidate_id", "")) for item in candidates]
    if str(candidate_id) not in ids:
        raise DashboardActionError(
            f"candidate {candidate_id!r} is not in the current ranking",
            code="CW-UI-005",
            remediation="Refresh the page and choose a candidate from the current list.",
        )
    start = ids.index(str(candidate_id))
    selected = candidates[start:] if allow_fallback else candidates[start : start + 1]
    result = AcquisitionCoordinator(settings).acquire(
        selected, offline=settings.offline, origin="dashboard"
    )
    try:
        attempt_path = write_attempts(result, settings.report_dir, offline=settings.offline)
    except OSError as exc:
        raise DashboardActionError(
            f"acquisition attempts could not be written: {exc.__class__.__name__}",
            remediation="Check that the reports directory is writable.",
        ) from exc
    if not result.succeeded:
        failure = result.failure
        raise DashboardActionError(
            f"acquisition failed: {failure.message if failure is not None else 'unknown failure'}",
            code="CW-UI-006",
            remediation="Inspect the attempt log and try the next candidate.",
        )
    payload = result.to_dict()
    payload["attempts_path"] = str(attempt_path)
    return payload


def _parse_ints(value: str | Sequence[int], name: str) -> tuple[int, ...]:
    try:
        if isinstance(value, str):
            parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
            numbers = tuple(int(part.strip()) for part in parts)
        else:
            numbers = tuple(int(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise DashboardActionError(
            f"{name} must be a comma-separated list of integers",
            code="CW-UI-007",
        ) from exc
    if not numbers or any(number < 1 for number in numbers):
        raise DashboardActionError(f"{name} must contain positive integers", code="CW-UI-007")
    return numbers


def _parse_floats(value: str | Sequence[float], name: str) -> tuple[float, ...]:
    try:
        if isinstance(value, str):
            parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
            numbers = tuple(float(part.strip()) for part in parts)
        else:
            numbers = tuple(float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise DashboardActionError(
            f"{name} must be a comma-separated list of numbers",
            code="CW-UI-007",
        ) from exc
    return numbers


def train_mlp(settings: Settings, config: Mapping[str, Any]) -> dict[str, Any]:
    """Train the Phase 6 MLP on the latest prepared dataset.

    This call is synchronous and may take minutes. The dashboard reports that
    honestly with a spinner; it never pretends to provide background progress.
    """

    from churnwise.models import MLPConfig, MLPTrainer
    from churnwise.preprocessing import PreprocessingCoordinator, PreprocessingError
    from churnwise.models import ModelingError

    prepared = latest_prepared_dir(settings)
    if prepared is None:
        raise DashboardActionError(
            "no prepared Phase 5 dataset exists",
            code="CW-UI-008",
            remediation="Run python -m churnwise.cli prepare first.",
        )
    hidden = _parse_ints(config.get("hidden_units", "64,32"), "hidden units")
    dropout = _parse_floats(config.get("dropout", ""), "dropout") if str(config.get("dropout", "")).strip() else tuple(0.2 for _ in hidden)
    if len(dropout) != len(hidden):
        raise DashboardActionError(
            "dropout must have one value per hidden layer",
            code="CW-UI-007",
        )
    try:
        options = MLPConfig(
            seed=int(config.get("seed", settings.seed)),
            hidden_units=hidden,
            dropout=dropout,
            l2=float(config.get("l2", 1e-4)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            batch_size=int(config.get("batch_size", 64)),
            epochs=int(config.get("epochs", 150)),
            early_stopping_patience=int(config.get("early_stopping_patience", 12)),
            threshold=float(config.get("threshold", 0.5)),
            deterministic=bool(config.get("deterministic", False)),
        )
        problems = options.validate()
        if problems:
            raise DashboardActionError(
                "invalid MLP configuration: " + "; ".join(problems),
                code="CW-UI-007",
            )
        dataset = PreprocessingCoordinator(settings).load(prepared.name)
        result = MLPTrainer(settings, options).train(dataset, persist=True)
    except DashboardActionError:
        raise
    except (PreprocessingError, ModelingError) as exc:
        raise DashboardActionError(
            f"{getattr(exc, 'code', 'CW-MOD')}: {getattr(exc, 'message', str(exc))}",
            code=str(getattr(exc, "code", "CW-UI-009")),
            remediation=str(getattr(exc, "remediation", "")),
        ) from exc
    except (OSError, ValueError) as exc:
        raise DashboardActionError(
            f"training failed: {exc.__class__.__name__}",
            remediation="Inspect the prepared dataset and model settings, then retry.",
        ) from exc
    return {
        "run_id": result.run_id,
        "metadata": dict(result.metadata),
        "history": dict(result.history),
        "paths": dict(result.paths),
        "diagnostics": dict(result.diagnostics),
    }


def _risk_service(settings: Settings, thresholds: Sequence[float] | None = None):
    from churnwise.intelligence import RiskConfig, RiskThresholds, RiskService

    config = RiskConfig(
        thresholds=RiskThresholds(boundaries=tuple(thresholds) if thresholds else RiskThresholds().boundaries)
    )
    return RiskService(settings, config)


def assess_customer_record(
    settings: Settings,
    record: Mapping[str, Any],
    *,
    thresholds: Sequence[float] | None = None,
    explain: bool = False,
) -> dict[str, Any]:
    """Score one raw record through the Phase 9 service (no dashboard math)."""

    from churnwise.intelligence import RiskError

    try:
        item = _risk_service(settings, thresholds).assess_customer(record, explain=explain)
    except RiskError as exc:
        raise DashboardActionError(
            exc.message,
            code=str(exc.code),
            remediation=str(exc.remediation),
        ) from exc
    except (OSError, ValueError) as exc:
        raise DashboardActionError(
            f"assessment failed ({exc.__class__.__name__}); no probability was produced",
            code="CW-UI-010",
            remediation="Check the model and preprocessing artifacts, then retry.",
        ) from exc
    return item.to_dict()


def assess_batch_records(
    settings: Settings,
    frame: Any,
    *,
    thresholds: Sequence[float] | None = None,
    reference_column: str | None = None,
) -> dict[str, Any]:
    """Score uploaded records through Phase 9, preserving input positions."""

    from churnwise.intelligence import RiskError

    try:
        batch = _risk_service(settings, thresholds).assess_batch(
            frame,
            explain="none",
            reference_column=reference_column,
        )
    except RiskError as exc:
        raise DashboardActionError(
            exc.message,
            code=str(exc.code),
            remediation=str(exc.remediation),
        ) from exc
    except (OSError, ValueError) as exc:
        raise DashboardActionError(
            f"batch assessment failed ({exc.__class__.__name__}); no scores were produced",
            code="CW-UI-011",
            remediation="Check the model and preprocessing artifacts, then retry.",
        ) from exc
    return batch.to_dict()


def run_campaign_projection(
    settings: Settings,
    *,
    risk_run_id: str,
    budget: float,
    cost_per_contact: float,
    effectiveness: float,
    monthly_value: float | None,
    horizon_months: float,
    value_unit: str,
    strategy: str,
    min_risk_probability: float | None,
    risk_tiers: Sequence[str] = (),
    max_contacts: int | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the Phase 10 engine in memory (``persist=False``) for the UI."""

    from churnwise.simulation import (
        CampaignAssumptions,
        CampaignScenario,
        SimulationError,
        SimulationService,
        TargetingConfig,
    )

    def scenario_from(values: Mapping[str, Any], name: str, description: str) -> Any:
        return CampaignScenario(
            name=name,
            description=description,
            budget=float(values["budget"]),
            assumptions=CampaignAssumptions(
                cost_per_contact=float(values["cost_per_contact"]),
                effectiveness=float(values["effectiveness"]),
                monthly_value=None if values.get("monthly_value") in (None, "") else float(values["monthly_value"]),
                horizon_months=float(values["horizon_months"]),
                value_unit=str(values["value_unit"]),
            ),
            targeting=TargetingConfig(
                strategy=str(values["strategy"]),
                min_risk_probability=None if values.get("min_risk_probability") in (None, "") else float(values["min_risk_probability"]),
                risk_tiers=tuple(values.get("risk_tiers") or ()),
                max_contacts=None if values.get("max_contacts") in (None, "") else int(values["max_contacts"]),
            ),
        )

    primary_values = {
        "budget": budget,
        "cost_per_contact": cost_per_contact,
        "effectiveness": effectiveness,
        "monthly_value": monthly_value,
        "horizon_months": horizon_months,
        "value_unit": value_unit,
        "strategy": strategy,
        "min_risk_probability": min_risk_probability,
        "risk_tiers": tuple(risk_tiers),
        "max_contacts": max_contacts,
    }
    extra: list[Any] = []
    if comparison:
        merged = dict(primary_values)
        merged.update(comparison)
        extra.append(scenario_from(merged, "Comparison", "Dashboard comparison scenario"))
    try:
        service = SimulationService(settings)
        primary = scenario_from(primary_values, "Dashboard scenario", "Projection computed in the dashboard")
        run = service.run_scenario(
            primary,
            run_id=risk_run_id,
            include_sensitivity=True,
            compare_with=tuple(extra),
            persist=False,
        )
    except SimulationError as exc:
        raise DashboardActionError(
            f"{exc.code}: {exc.message}",
            code=str(exc.code),
            remediation=str(exc.remediation),
        ) from exc
    payload = run.to_dict()
    payload["rendered_report"] = service.render(run)
    return payload


__all__ = [
    "DashboardActionError",
    "acquire_candidate",
    "assess_batch_records",
    "assess_customer_record",
    "discover_datasets",
    "run_campaign_projection",
    "train_mlp",
]
