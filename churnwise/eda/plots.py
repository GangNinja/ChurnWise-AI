"""Diagnostic EDA plots for Phase 4.

Plotting is read-only: it never fills, drops or rewrites data. The functions use
the non-interactive Agg backend so the validator works in headless runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churnwise.eda.profile import DatasetProfile, normalize_label  # noqa: E402


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:120] or "dataset"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return _display_path(path)


def generate_plots(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    output_dir: Path | str,
    *,
    max_numeric: int = 6,
    max_categories: int = 3,
) -> list[str]:
    """Generate target balance, distributions, correlation and conditional plots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(profile.dataset_id)
    paths: list[str] = []

    target = profile.target
    if target.column and target.counts:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        labels = list(target.counts)
        counts = [target.counts[label] for label in labels]
        colors = [
            "#c44e52" if label == target.positive_label else "#4c72b0"
            for label in labels
        ]
        ax.bar(labels, counts, color=colors)
        ax.set_title(f"Target balance: {target.column}")
        ax.set_ylabel("rows")
        for index, count in enumerate(counts):
            ax.text(index, count, str(count), ha="center", va="bottom", fontsize=8)
        if target.positive_rate is not None:
            ax.set_xlabel(f"positive (churn) rate = {target.positive_rate:.2%}")
        paths.append(_save(fig, output / f"{stem}.png"))

    numeric = [name for name in profile.roles.numeric if name in frame.columns][:max_numeric]
    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce") if numeric else None
    if numeric_frame is not None and not numeric_frame.empty:
        columns = min(3, len(numeric))
        rows = int(np.ceil(len(numeric) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
        for index, name in enumerate(numeric):
            axis = axes[index // columns][index % columns]
            values = numeric_frame[name].replace([np.inf, -np.inf], np.nan).dropna()
            axis.hist(values, bins=30, color="#4c72b0", alpha=0.85)
            axis.set_title(name, fontsize=9)
            axis.tick_params(labelsize=7)
        for index in range(len(numeric), rows * columns):
            axes[index // columns][index % columns].axis("off")
        paths.append(_save(fig, output / f"{stem}_numeric.png"))

    if numeric_frame is not None and numeric_frame.shape[1] >= 2:
        correlation = numeric_frame.corr()
        fig, ax = plt.subplots(figsize=(min(10, 1.2 + 0.55 * len(correlation)),) * 2)
        image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.columns)))
        ax.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(correlation.columns, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Numeric feature correlation")
        paths.append(_save(fig, output / f"{stem}_correlation.png"))

    if numeric_frame is not None and profile.target.usable and profile.target.column in frame.columns:
        from churnwise.validation.targets import normalize_target

        target_binary = normalize_target(frame[profile.target.column], profile.target)
        conditional = numeric[:4]
        if conditional:
            fig, axes = plt.subplots(
                1, len(conditional), figsize=(3.8 * len(conditional), 3.2), squeeze=False
            )
            for index, name in enumerate(conditional):
                axis = axes[0][index]
                groups = [
                    pd.to_numeric(frame.loc[target_binary == value, name], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy()
                    for value in (0, 1)
                ]
                axis.boxplot(groups, showfliers=False)
                axis.set_xticks([1, 2])
                axis.set_xticklabels(["not churned", "churned"])
                axis.set_title(name, fontsize=9)
                axis.tick_params(labelsize=7)
            paths.append(_save(fig, output / f"{stem}_conditional.png"))

    categorical = [
        name
        for name in profile.roles.categorical
        if name in frame.columns and not profile.column(name).constant
    ][:max_categories]
    if categorical:
        fig, axes = plt.subplots(
            1, len(categorical), figsize=(4.2 * len(categorical), 3.2), squeeze=False
        )
        for index, name in enumerate(categorical):
            axis = axes[0][index]
            labels = frame[name].map(normalize_label).dropna()
            counts = labels.value_counts().head(10)
            axis.barh([str(value) for value in counts.index[::-1]], counts.to_numpy()[::-1], color="#55a868")
            axis.set_title(name, fontsize=9)
            axis.tick_params(labelsize=7)
        paths.append(_save(fig, output / f"{stem}_categorical.png"))

    return paths
