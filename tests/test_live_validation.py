"""Live Phase 4 verification on a real acquired dataset.

Skipped unless ``--run-live`` is passed. Discovery, acquisition and validation
all run against public sources; the report is then cross-checked with pandas and
the on-disk artifact digest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churnwise.acquisition.integrity import sha256_file
from churnwise.config import Settings
from churnwise.contracts import SearchQuery
from churnwise.discovery import build_adapters, discover
from churnwise.validation import ValidationCoordinator, ValidationStatus

pytestmark = pytest.mark.live


def test_live_openml_dataset_is_validated_and_reported(tmp_path):
    settings = Settings(project_root=tmp_path)
    query = SearchQuery(
        terms=("churn",),
        max_results_per_source=5,
        enrich_limit=6,
        timeout_s=25,
        max_retries=1,
    )
    discovery = discover(build_adapters(settings, query), query, settings)
    assert discovery.selected is not None, (
        discovery.failure.to_dict() if discovery.failure else ""
    )

    result = ValidationCoordinator(settings).validate_candidates(discovery)

    assert result.status in {
        ValidationStatus.PASS,
        ValidationStatus.PASS_WITH_WARNINGS,
    }
    assert result.can_preprocess is True
    report = result.report
    assert report is not None
    assert [check.id for check in report.checks] == [f"V{index}" for index in range(1, 13)]
    assert report.target["column"]
    assert report.target["positive_label"] is not None

    artifact = Path(report.artifact_path)
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    assert artifact.is_file()
    assert sha256_file(artifact) == report.artifact_sha256

    frame = pd.read_csv(artifact) if artifact.suffix.lower() == ".csv" else None
    if frame is not None:
        target = report.target["column"]
        recomputed = float(
            (frame[target].astype(str).str.lower() == str(report.target["positive_label"]).lower()).mean()
        )
        assert report.target["positive_rate"] == pytest.approx(recomputed, abs=1e-9)

    assert (settings.report_dir / "validation" / f"{report.dataset_id.replace(':', '_')}.json").is_file()
    assert (settings.report_dir / "eda" / f"{report.dataset_id.replace(':', '_')}.json").is_file()
    assert len(report.eda_plots) >= 3
    for plot in report.eda_plots:
        plot_path = Path(plot)
        if not plot_path.is_absolute():
            plot_path = Path.cwd() / plot_path
        assert plot_path.is_file()

    normalized = result.normalized_frame()
    assert normalized is not None
    assert set(normalized[report.target["column"]].unique()) <= {0, 1}
