"""Charts drawn only from computed simulation results.

Every axis, legend and title names the unit and states that values are
hypothetical projections. Matplotlib uses the Agg backend so the CLI works
headless; nothing here interpolates or fabricates a data point.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt

from churnwise.simulation.contracts import (
    BudgetAllocation,
    ScenarioComparison,
    SensitivitySeries,
)
from churnwise.simulation.errors import CODE_ARTIFACT_FAILED, SimulationError

PROJECTION_NOTE = "hypothetical projection from stated assumptions"


def _save(figure: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=140, bbox_inches="tight")
    except OSError as exc:
        raise SimulationError(
            CODE_ARTIFACT_FAILED,
            f"simulation figure could not be written: {path}: {exc}",
            "Check that the artifact directory is writable.",
        ) from exc
    finally:
        plt.close(figure)
    return str(path)


def plot_budget_allocation(
    allocation: BudgetAllocation,
    path: Path | str,
    *,
    unit: str,
    scenario_name: str,
) -> str:
    """Allocated spend versus remaining budget for one scenario."""
    spend = float(allocation.allocated_spend)
    remaining = float(allocation.remaining_budget)
    figure, axis = plt.subplots(figsize=(7.5, 2.6))
    axis.barh(["Budget"], [spend], color="#2f6f8f", label="allocated spend")
    axis.barh(
        ["Budget"],
        [remaining],
        left=[spend],
        color="#c9d8df",
        label="remaining budget",
    )
    axis.set_xlabel(f"{unit}")
    axis.set_title(
        f"{scenario_name}: budget allocation "
        f"({allocation.selected_count} selected of {allocation.eligible_count} eligible; "
        f"{PROJECTION_NOTE})"
    )
    utilization = (
        "n/a" if allocation.utilization is None else f"{allocation.utilization:.1%}"
    )
    axis.text(
        0.99,
        0.05,
        f"utilisation {utilization}",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return _save(figure, Path(path))


def plot_sensitivity_series(
    series: SensitivitySeries,
    path: Path | str,
    *,
    unit: str,
    scenario_name: str,
) -> str:
    """Expected prevented churn and spend across one varied assumption."""
    points = list(series.points)
    if not points:
        raise SimulationError(
            CODE_ARTIFACT_FAILED,
            f"sensitivity series {series.dimension!r} has no points to plot",
            "Provide at least two values for the grid.",
        )
    x_values = [float(point.value) for point in points]
    prevented = [float(point.expected_prevented_churn) for point in points]
    spends = [float(point.allocated_spend) for point in points]
    baseline_positions = [
        index for index, point in enumerate(points) if point.is_baseline
    ]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 5.4), sharex=True
    )
    top.plot(
        x_values,
        prevented,
        marker="o",
        color="#2f6f8f",
        linewidth=1.8,
        label="expected prevented churn (hypothetical)",
    )
    for index in baseline_positions:
        top.axvline(
            x_values[index],
            color="#999999",
            linestyle="--",
            linewidth=0.9,
            label="scenario value",
        )
    top.set_ylabel("expected prevented\nchurn (customers)")
    top.set_title(
        f"{scenario_name}: {series.dimension} sensitivity ({PROJECTION_NOTE})"
    )
    top.grid(alpha=0.25)
    top.legend(frameon=False, fontsize=8, loc="upper left")

    bottom.plot(
        x_values,
        spends,
        marker="s",
        color="#4b7f52",
        linewidth=1.8,
        label="allocated spend",
    )
    bottom.set_xlabel(f"{series.dimension} ({series.unit})")
    bottom.set_ylabel(f"allocated spend ({unit})")
    bottom.grid(alpha=0.25)
    bottom.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return _save(figure, Path(path))


def plot_scenario_comparison(
    comparison: ScenarioComparison,
    path: Path | str,
    *,
    unit: str,
) -> str:
    """Selected customers and expected prevented churn per scenario."""
    rows: Sequence[Any] = list(comparison.rows)
    if not rows:
        raise SimulationError(
            CODE_ARTIFACT_FAILED,
            "scenario comparison has no rows to plot",
            "Run at least one scenario before generating comparison charts.",
        )
    labels = [row.name for row in rows]
    selected = [int(row.selected_count) for row in rows]
    prevented = [float(row.expected_prevented_churn) for row in rows]
    spends = [float(row.allocated_spend) for row in rows]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
    positions = list(range(len(rows)))
    top.bar(
        [position - 0.2 for position in positions],
        selected,
        width=0.4,
        color="#6d8fa3",
        label="selected customers",
    )
    top.set_ylabel("selected customers")
    top.set_title(
        f"Scenario comparison ({PROJECTION_NOTE}); budget, cost and value in {unit}"
    )
    top.grid(axis="y", alpha=0.25)
    top.legend(frameon=False, fontsize=8, loc="upper right")

    bottom.bar(
        positions,
        prevented,
        width=0.55,
        color="#2f6f8f",
        label="expected prevented churn",
    )
    bottom.set_ylabel("expected prevented\nchurn (customers)")
    bottom.grid(axis="y", alpha=0.25)
    bottom.legend(frameon=False, fontsize=8, loc="upper right")
    for position, value, spend in zip(positions, prevented, spends):
        bottom.text(
            position,
            value,
            f"{value:.3f}\nspend {spend:,.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    figure.tight_layout()
    return _save(figure, Path(path))


__all__ = [
    "plot_budget_allocation",
    "plot_scenario_comparison",
    "plot_sensitivity_series",
]
