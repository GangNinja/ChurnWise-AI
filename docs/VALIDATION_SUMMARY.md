# ChurnWise AI — Validation Summary

**Validation date:** 2026-10-24
**Scope:** four post-Phase-13 validation workstreams
**Final verdict:** **VALIDATION PASSED WITH LIMITATIONS**

This page is the curated, version-controlled summary of the validation evidence.
The full generated reports and run trees remain in the working tree under
`reports/post_phase13/` and are intentionally excluded from Git by the
repository's `reports/*` policy. Small, verified extracts and a source map are
tracked in [docs/evidence/post_phase13](evidence/post_phase13/README.md).

## 1. What was validated

| Workstream | Result | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Full-profile pipeline | **PASS WITH LIMITATIONS** | The documented non-`--fast` pipeline completed all seven stages on OpenML `46915` and reproduced the committed benchmark metrics. | Repeated-split variance, production SLAs or a second population. |
| Second-dataset robustness | **PASS WITH LIMITATIONS** | The pipeline could be retrained on OpenML `42178` after a documented ARFF-to-CSV format adaptation; validation-only selection and held-out testing still worked. | Transfer of the primary model, or generalisation to other customer bases. |
| Probability calibration | **PASS WITH LIMITATIONS** | Sigmoid calibration improved held-out probability-quality metrics without changing ranking. | Integration into the production candidate, subgroup calibration or calibration-drift monitoring. |
| Dashboard browser E2E | **PASS WITH LIMITATIONS** | A real browser exercised all dashboard pages and key workflows against real local Streamlit servers. | Live discovery/training, optional local SHAP, responsive layouts, accessibility or visual regression. |

No critical product defect was found. The production model, threshold, registry,
dashboard and generated evidence were not replaced.

## 2. Primary dataset and committed benchmark

The reference dataset is OpenML `46915`: 5,000 rows, target `CustomerChurned`,
14.14% positive, 27 transformed features and a stratified 3,000 / 1,000 / 1,000
train / validation / test split. Models and thresholds were selected on the
validation split; the test split was scored once.

| Family | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.590476 | 0.635015 | 0.874107 | 0.571705 | 0.145961 | 0.64 |
| Random Forest **(selected)** | **0.798479** | **0.871795** | **0.938342** | **0.897321** | **0.039015** | **0.45** |
| MLP `[64, 32]` | 0.693727 | 0.720848 | 0.926642 | 0.780120 | 0.094439 | 0.73 |

The selected Random Forest's held-out confusion matrix was TN 846, FP 13,
FN 22, TP 119. These are **test** results. The selection decision used validation
F1 only; test performance did not choose the model.

Committed artifacts:

- [Benchmark metrics](evidence/benchmark/metrics.json)
- [Benchmark report](evidence/benchmark/benchmark.md)
- [Phase 7 report](PHASE7_REPORT.md)

## 3. Workstream 1 — full-profile pipeline

Command (as executed, from the repository root):

```bash
python -m churnwise.cli run --seed 42 --run-id post13-full-20261024c \
  --data-dir reports/post_phase13/workstream1/data \
  --artifact-dir reports/post_phase13/workstream1/artifacts \
  --report-dir reports/post_phase13/workstream1/reports
```

Observed result:

- All seven stages succeeded: `source`, `preprocess`, `train`, `evaluate`,
  `explain`, `score`, `simulate`.
- Wall time: **328.259 s**.
- Peak process-tree working set: **613.9 MiB**.
- Selected model: Random Forest at threshold **0.45**.
- Test F1 **0.871795**, ROC-AUC **0.938342**, PR-AUC **0.897321** and Brier
  **0.039015**, matching the committed benchmark.
- 1,000 customers scored: 858 low / 31 medium / 40 high / 71 critical.
- Simulation: budget 1,000, cost per contact 5, 200 contacted, expected
  prevented churn **32.8334** at assumed effectiveness 0.30.
- The MLP trained 43 epochs (best epoch 31) and stopped early.

The run used live OpenML discovery and download. Three successful full-profile
runs produced identical reported metrics, but the MLP weight bytes differed
because TensorFlow op determinism is disabled. This is statistical
reproducibility, not bitwise reproducibility.

## 4. Workstream 2 — second-dataset robustness

A separately trained experiment used OpenML `42178` (IBM Telco sample):
7,043 rows, 21 columns, `Churn` target and 26.54% positive. The project's native
ARFF reader could not parse the file's OpenML `STRING` attributes, so the
experiment used an isolated ARFF-to-CSV adapter that preserved rows, columns,
values and order. The production reader was not changed.

The split was 4,225 / 1,409 / 1,409 with 49 transformed features. Family and
threshold were selected on validation F1, then frozen.

| Family | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression **(selected)** | **0.659574** | **0.619552** | **0.837960** | **0.651048** | **0.169109** | **0.61** |
| MLP `[64, 32]` | 0.653302 | 0.615569 | 0.834238 | 0.626959 | 0.169632 | 0.59 |
| Random Forest | 0.633028 | 0.606200 | 0.821257 | 0.618607 | 0.149806 | 0.39 |

**Interpretation:** these are new models trained on a different dataset. The
result supports the pipeline's ability to handle a changed schema and domain; it
does **not** show that the primary Random Forest transfers to another customer
population. The OpenML licence field says `Public` but supplies no licence URL,
so upstream reuse terms remain unverified.

## 5. Workstream 3 — probability calibration

Protocol: compare uncalibrated, sigmoid and isotonic candidates with out-of-fold
predictions inside the 1,000-row validation partition only; select by Brier
score with log loss as the tie-break; fit the chosen calibrator on validation;
freeze the method and threshold; score test once.

Sigmoid was selected. At the unchanged production threshold of 0.45:

| Metric | Uncalibrated | Sigmoid | Change |
| --- | ---: | ---: | ---: |
| Brier ↓ | 0.039015 | 0.030555 | **−0.008442** (95% bootstrap CI 0.006475–0.010313 improvement) |
| Log loss ↓ | 0.168357 | 0.133686 | **−0.034609** (95% bootstrap CI 0.027241–0.041723 improvement) |
| ECE ↓ | 0.068607 | 0.030073 | **−0.038534** |
| ROC-AUC | 0.938342 | 0.938342 | unchanged |
| PR-AUC | 0.897321 | 0.897321 | unchanged |
| F1 @ 0.45 | 0.871795 | 0.872727 | +0.000932 |

The calibrated analysis threshold was 0.46, but the production threshold remains
0.45. The calibrated estimator serializes, reloads in a clean process and
returns finite probabilities in `[0, 1]`.

**Important:** calibration is an evaluated experiment, not a production change.
It is not wired into the model registry, CLI risk service, API or dashboard. The
evidence is one validation split with 142 positives and one test split with 141
positives; subgroup calibration and drift were not evaluated.

## 6. Workstream 4 — dashboard browser E2E

The opt-in Playwright suite launched real Streamlit servers over temporary
artifact trees and used a real Chromium-based browser. Result: **10 passed, 0
failed in 49.89 s** in the five-page redesign rerun (an earlier post-audit run
took 78.33 s and the original workstream run 80.67 s).

Covered journeys:

1. All five product pages render without Streamlit exceptions.
2. Navigation inventory and identity fields.
3. Laboratory held-out metrics match the controlled fixture (`F1 0.8800`,
   `ROC-AUC 0.9400`).
4. Workbench filtering from 6 to 2 critical rows and a real CSV download.
5. Customer Risk Explorer (`0.9000`, `Critical`, stored explanation).
6. Campaign Simulator projection through the real simulation engine.
7. Missing-model assessment returns `CW-XAI-006` with remediation.
8. Individual assessment through the real risk service.
9. Batch CSV upload scores 2/2 rows and exports the result.
10. Empty artifact trees produce actionable commands on all five pages.

The suite uses one browser engine at one desktop viewport. It does not exercise
live network discovery/acquisition, background training, optional local SHAP,
responsive layouts, accessibility or visual regression.

## 7. Test-suite evidence

| Suite | Command | Result |
| --- | --- | --- |
| Default/offline suite | `python -m pytest -q` | **634 passed, 17 skipped** (post-audit rerun, 293.81 s) |
| Live-network tests | `python -m pytest -q -m live --run-live` | **7 passed** (network required) |
| Browser E2E | `CHURNWISE_BROWSER_E2E=1 python -m pytest tests/e2e -m browser -q` | **10 passed**, 78.33 s |

The 17 default skips are the 7 opt-in live-network tests and the 10 opt-in
browser tests. No existing test was disabled or weakened. The browser suite is
collected by default and skipped cleanly without the environment flag.

The Phase 13 clean-environment result remains **624 passed, 7 skipped** in
326.77 s before the browser tests were added. The post-audit rerun added ten
regression tests for dashboard lineage and status semantics; it did not remove
or weaken any test.

## 8. Defects and unresolved risks

| Finding | Severity | Status |
| --- | --- | --- |
| The native ARFF reader cannot parse OpenML `STRING` attributes; dataset 42178 required an isolated adapter. | Medium interoperability limitation; the primary dataset path is unaffected. | Deferred; production reader unchanged. |
| The calibrated estimator is serializable but not integrated. | Medium deployment gap. | Deferred by design; requires a separate integration and re-validation. |
| OpenML 42178 reports `Public` without a licence URL. | Provenance/licensing ambiguity. | Documented; formal reuse review outstanding. |
| Full-profile MLP has statistical but not bitwise reproducibility. | Reproducibility limitation. | Documented; TensorFlow op determinism is disabled. |
| One split per experiment; no repeated holdout, subgroup analysis, confidence intervals or drift monitoring. | Statistical limitation. | Not addressed in this validation. |
| Browser suite covers one engine and one viewport; live actions and optional SHAP are not covered. | Test coverage limitation. | Documented; suite remains opt-in. |

No defect changed the selected model or invalidated the reported test metrics.

## 9. Reproduction

Use the repository root and the existing environment:

```bash
# Full profile (network required; about 5.5 minutes, 614 MiB peak)
python -m churnwise.cli run --seed 42 --run-id reproduce-full

# Fast offline fixture (no network; about 1 minute)
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv \
  --fast --seed 13 --run-id reproduce-fixture

# Standard suite
python -m pytest -q

# Optional browser suite
CHURNWISE_BROWSER_E2E=1 python -m pytest tests/e2e -m browser -q
```

On Windows PowerShell, set the browser flag with
`$env:CHURNWISE_BROWSER_E2E="1"` before the pytest command.

The full second-dataset and calibration experiment tooling lives under
`reports/post_phase13/tools/` in the working tree. Those experiments require
network access and generated artifacts; they are not part of the default test
suite and were not rerun during this documentation pass.

## 10. Evidence map

| Evidence | Tracked location | Original generated location |
| --- | --- | --- |
| This summary | `docs/VALIDATION_SUMMARY.md` | — |
| Curated extracts and provenance | `docs/evidence/post_phase13/` | `reports/post_phase13/` |
| Primary benchmark | `docs/evidence/benchmark/` | `artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/` |
| Phase 13 live pipeline | `docs/evidence/phase13-live2/` | generated run tree, Git-ignored |
| Phase 13 test logs | `docs/evidence/phase13-tests/` | generated logs, Git-ignored |

The complete post-Phase-13 reports are `POST_PHASE13_VALIDATION_SUMMARY.md`,
`FULL_PIPELINE_VALIDATION_REPORT.md`, `SECOND_DATASET_ROBUSTNESS_REPORT.md`,
`PROBABILITY_CALIBRATION_REPORT.md` and `DASHBOARD_E2E_REPORT.md` under
`reports/post_phase13/`. They remain authoritative for details; this page is the
reviewable summary.

## 11. Bottom line

The system is verified end to end on one public dataset with a full-profile run,
has separately evaluated robustness and calibration evidence, and its dashboard
is covered by a real-browser suite. It is **not** production-deployed, causally
validated, externally generalised or equipped with calibrated production
probabilities. Those boundaries are deliberate and are reflected in the
dashboard, reports and documentation.
