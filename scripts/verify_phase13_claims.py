"""Verify that Phase 13 documentation numbers match saved evidence.

This is deliberately narrow: it checks the public facing benchmark, risk and
simulation claims against the post-fix live artifacts. It does not replace the
phase reports or the acceptance matrix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "phase13-live2"
BENCHMARK = EVIDENCE / "benchmark_metrics.json"
RISK = EVIDENCE / "risk_summary.json"
SIMULATION = EVIDENCE / "simulation_summary.json"
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "PROJECT_PRESENTATION.md",
    ROOT / "docs" / "PROJECT_SHOWCASE.md",
    ROOT / "docs" / "RESUME_BULLETS.md",
    ROOT / "docs" / "FINAL_VERIFICATION_REPORT.md",
]
STALE = ("0.5585", "0.6417", "0.7943", "0.5448", "0.7718", "0.5030", "0.7383")

# Numbers that must appear somewhere in the public documentation set.
EXPECTED = {
    "0.8718": "RF test F1",
    "0.9383": "RF test ROC-AUC",
    "0.8973": "RF test PR-AUC",
    "0.0390": "RF test Brier",
    "0.6350": "LR test F1",
    "0.8741": "LR test ROC-AUC",
    "0.5717": "LR test PR-AUC",
    "0.7208": "full-profile MLP test F1",
    "0.9266": "MLP test ROC-AUC",
    "0.7801": "MLP test PR-AUC",
    "0.7068": "fast-profile MLP test F1",
    "858": "low tier count",
    "31": "medium tier count",
    "40": "high tier count",
    "71": "critical tier count",
    "121": "predicted churn count",
    "0.45": "classification threshold",
    "109.4446": "expected churn mass",
    "32.8334": "expected prevented churn",
    "9.1385": "random-contact baseline",
    "3.593": "targeting lift",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    failures: list[str] = []
    benchmark = _load(BENCHMARK)
    risk = _load(RISK)
    simulation = _load(SIMULATION)

    rf = benchmark["families"]["random_forest"]["test_metrics"]
    lr = benchmark["families"]["logistic_regression"]["test_metrics"]
    mlp = benchmark["families"]["mlp"]["test_metrics"]
    if (round(rf["f1"], 4), round(rf["roc_auc"], 4), round(rf["pr_auc"], 4)) != (
        0.8718,
        0.9383,
        0.8973,
    ):
        failures.append("live RF metrics no longer match the documented values")
    if (round(lr["f1"], 4), round(mlp["f1"], 4)) != (0.6350, 0.7068):
        failures.append("live baseline metrics no longer match the documented values")
    if risk["n_rows"] != 1000 or risk["tier_counts"]["critical"]["count"] != 71:
        failures.append("risk summary no longer matches the documented population")
    projection = simulation["projection"]
    if round(projection["expected_prevented_churn"], 4) != 32.8334:
        failures.append("simulation projection no longer matches 32.8334")

    texts: dict[str, str] = {}
    for path in DOCS:
        if not path.is_file():
            failures.append(f"missing document: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        texts[path.name] = text
        if "TODO" in text or "TBD" in text:
            failures.append(f"{path.name}: placeholder marker remains")
        for token in STALE:
            if token in text:
                failures.append(f"{path.name}: stale metric {token} still present")

    combined = "\n".join(texts.values())
    for token, label in EXPECTED.items():
        if token not in combined:
            failures.append(f"documentation set: missing {label} ({token})")

    # Each headline document must carry the selected model's core result.
    for name in (
        "README.md",
        "PROJECT_PRESENTATION.md",
        "PROJECT_SHOWCASE.md",
        "RESUME_BULLETS.md",
        "FINAL_VERIFICATION_REPORT.md",
    ):
        text = texts.get(name, "")
        if "0.8718" not in text or "0.9383" not in text:
            failures.append(f"{name}: selected Random Forest result is not stated")

    if failures:
        print("Phase 13 claim verification: FAIL")
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("Phase 13 claim verification: PASS")
    print(f"  checked {len(texts)} documents and {len(EXPECTED)} evidence claims")
    print("  benchmark, risk and simulation values match post-fix evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
