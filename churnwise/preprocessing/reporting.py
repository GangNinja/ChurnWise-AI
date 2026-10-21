"""Human-readable and machine-readable Phase 5 reports (ASCII console output)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from churnwise.preprocessing.artifacts import validate_run_id
from churnwise.preprocessing.dataset import PreparedDataset


def render_summary(dataset: PreparedDataset) -> str:
    spec = dataset.split_spec
    balance = dataset.class_balance
    lines = [
        "ChurnWise AI - preprocessing",
        "",
        f"Run: {dataset.run_id}",
        f"Dataset: {dataset.dataset_id}",
        f"Split: {spec.strategy} (seed {spec.seed}) - "
        f"{spec.sizes['train']}/{spec.sizes['val']}/{spec.sizes['test']} rows",
        f"Fingerprint: {spec.fingerprint}",
        f"Features: {len(dataset.feature_names)} transformed from "
        f"{len(dataset.schema.get('preprocessing', {}).get('numeric', []))} numeric, "
        f"{len(dataset.schema.get('preprocessing', {}).get('categorical', []))} categorical, "
        f"{len(dataset.schema.get('preprocessing', {}).get('boolean', []))} boolean columns",
        f"Imbalance: {balance.strategy} - positive rate "
        f"{_pct(balance.rates.get('train'))} train, {_pct(balance.rates.get('val'))} val, "
        f"{_pct(balance.rates.get('test'))} test",
    ]
    if balance.class_weights:
        lines.append(
            f"Class weights: 0={balance.class_weights.get('0'):.3f} "
            f"1={balance.class_weights.get('1'):.3f}"
        )
    derived = dataset.schema.get("derived", [])
    lines.append(f"Derived features: {len(derived)}" + (f" ({', '.join(d['name'] for d in derived)})" if derived else ""))
    skipped = dataset.schema.get("derived_skipped", [])
    if skipped:
        lines.append(f"Derived features skipped: {len(skipped)}")
    excluded = dataset.schema.get("excluded", [])
    if excluded:
        lines.append(f"Excluded columns: {len(excluded)}")
    for warning in dataset.warnings:
        lines.append(f"Warning: {warning}")
    if dataset.paths.get("report_md"):
        lines.append(f"Report: {dataset.paths['report_md']}")
    if dataset.paths.get("preprocessor"):
        lines.append(f"Preprocessor: {dataset.paths['preprocessor']}")
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown(dataset: PreparedDataset) -> str:
    spec = dataset.split_spec
    schema = dataset.schema
    balance = dataset.class_balance
    lines = [
        f"# Preprocessing report - {dataset.run_id}",
        "",
        f"- **Dataset:** `{dataset.dataset_id}`",
        f"- **Dataset SHA-256:** `{spec.dataset_sha256}`",
        f"- **Split fingerprint:** `{spec.fingerprint}`",
        f"- **Strategy:** `{spec.strategy}` (seed `{spec.seed}`)",
        f"- **Constraint:** `test_touched: false`, `fitted_on: train_only`",
        f"- **Transformed features:** {len(dataset.feature_names)}",
        "",
        "## Splits",
        "",
        "| Split | Rows | Positive | Rate |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("train", "val", "test"):
        counts = spec.class_counts[name]
        lines.append(
            f"| {name} | {spec.sizes[name]} | {counts.get('1', 0)} | "
            f"{_pct(spec.class_rates.get(name))} |"
        )
    lines.extend(
        [
            "",
            f"Duplicate rows dropped before splitting: {spec.duplicate_rows_dropped}",
            f"Rows dropped for a missing target: {spec.target_missing_dropped}",
            "",
            "## Imbalance",
            "",
            f"- Strategy: `{balance.strategy}`",
            f"- Imbalance ratio: {balance.ratio if balance.ratio is None else round(balance.ratio, 3)}",
            f"- Oversampled training rows: {balance.resampled_train_rows}",
            f"- Class weights: `{balance.class_weights}`",
            "",
            "## Derived features",
            "",
        ]
    )
    derived = schema.get("derived", [])
    if derived:
        lines.extend(
            [
                "| Feature | Capability | Sources | Role | Description |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in derived:
            lines.append(
                f"| {item['name']} | {item['capability']} | "
                f"{', '.join(item['sources'])} | {item['dtype']} | {item['description']} |"
            )
    else:
        lines.append("No derived feature was built; every skip is listed below.")
    skipped = schema.get("derived_skipped", [])
    if skipped:
        lines.extend(["", "### Skipped", ""])
        for item in skipped:
            lines.append(f"- `{item['feature']}`: {item['reason']}")
    lines.extend(["", "## Excluded columns", ""])
    excluded = schema.get("excluded", [])
    if excluded:
        lines.extend(["| Column | Stage | Reason |", "| --- | --- | --- |"])
        for item in excluded:
            lines.append(f"| {item['column']} | {item['stage']} | {item['reason']} |")
    else:
        lines.append("No column was excluded.")
    invalid = schema.get("invalid_numeric", {})
    lines.extend(["", "## Invalid numeric values (training split)", ""])
    if invalid:
        lines.extend(
            [
                "| Column | Invalid | Share | Unparseable | Infinite | Treatment |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, item in invalid.items():
            lines.append(
                f"| {name} | {item['invalid']} | {_pct(item['pct'])} | "
                f"{item['unparseable']} | {item['infinite']} | {item['treatment']} |"
            )
        lines.append("")
        lines.append("Each invalid value became missing and was then imputed from the training split.")
    else:
        lines.append("No non-null numeric value was invalid or non-finite.")
    outliers = schema.get("outliers", {})
    lines.extend(["", "## Outliers (training split, IQR)", ""])
    if outliers:
        lines.extend(["| Column | Count | Share | Lower | Upper | Treated |", "| --- | --- | --- | --- | --- | --- |"])
        for name, item in outliers.items():
            lines.append(
                f"| {name} | {item['count']} | {_pct(item['pct'])} | "
                f"{item['lower']:.4g} | {item['upper']:.4g} | {item['treated']} |"
            )
        lines.append("")
        lines.append("Outliers are reported and never deleted in Phase 5.")
    else:
        lines.append("No numeric feature had IQR outliers.")
    if dataset.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in dataset.warnings)
    lines.extend(["", "## Artifacts", ""])
    for key, value in dataset.paths.items():
        if key.endswith("_hash"):
            continue
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def write_reports(dataset: PreparedDataset, report_dir: Path | str) -> dict[str, str]:
    root = Path(report_dir) / "preprocessing"
    validate_run_id(dataset.run_id)
    json_path = root / f"{dataset.run_id}.json"
    md_path = root / f"{dataset.run_id}.md"
    payload = {
        "schema_version": dataset.schema.get("schema_version", "1.0"),
        "phase": 5,
        "run_id": dataset.run_id,
        "dataset_id": dataset.dataset_id,
        "split": dataset.split_spec.to_dict(),
        "preprocessing": dataset.schema.get("preprocessing", {}),
        "target": dataset.schema.get("target", {}),
        "derived": dataset.schema.get("derived", []),
        "derived_skipped": dataset.schema.get("derived_skipped", []),
        "excluded": dataset.schema.get("excluded", []),
        "output_features": dataset.schema.get("output_features", []),
        "class_balance": dataset.class_balance.to_dict(),
        "invalid_numeric": dataset.schema.get("invalid_numeric", {}),
        "outliers": dataset.schema.get("outliers", {}),
        "warnings": list(dataset.warnings),
        "paths": {key: value for key, value in dataset.paths.items()},
        "provenance": dict(dataset.provenance),
    }
    _atomic_write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(dataset), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}
