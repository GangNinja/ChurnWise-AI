# ChurnWise AI

**Customer-churn intelligence from raw data to a defensible retention decision —
with leakage-safe modelling, traceable artifacts and honest evidence boundaries.**

ChurnWise AI is a local-first Python platform that discovers a public churn
dataset, verifies its provenance, validates it, builds a leakage-safe feature
pipeline, compares three model families, explains predictions with SHAP, scores a
customer population and simulates budget-constrained retention campaigns. A
Streamlit dashboard exposes the persisted results; every reported number is tied
to a saved artifact.

> **Status:** portfolio and research system, verified through Phase 13 and four
> post-Phase-13 validation workstreams. The public Streamlit app is a demo, not a
> production service; it is not causally validated or proven to generalise to
> other customer populations. See [VALIDATION_SUMMARY.md](docs/VALIDATION_SUMMARY.md)
> for the evidence boundary.

**Live demo:** [ChurnWise AI on Streamlit](https://gangninja-churnwise-ai-churnwisedashboardapp-lanwl3.streamlit.app)

![ChurnWise AI Executive Dashboard](docs/screenshots/redesign-executive.png)

## The problem

A churn score is not an operational decision. Analysts still need to answer:

- Is the dataset suitable, and where did it come from?
- Was preprocessing fitted only on training data?
- Which model was selected, and on what validation criterion?
- Why does a particular customer receive a high risk score?
- Which customers fit a limited campaign budget, and what is a projection versus
  an observed outcome?

ChurnWise AI is designed for ML/data-engineering portfolio review and for
analysts exploring a reproducible public-data workflow. It makes those questions
explicit instead of hiding them behind a single accuracy number.

## At a glance

| Item | Verified result |
| --- | --- |
| Primary dataset | Instructor-provided IBM Telco Customer Churn CSV, 7,043 rows, 21 columns; 1,869 observed churn labels (26.54%) |
| Primary model | Multilayer Perceptron (MLP), selected and scored on the matching Telco dataset digest |
| Held-out test | MLP F1 **0.6156**, accuracy **0.7722**, precision **0.5575**, recall **0.6872** at the validation-selected threshold **0.59** (1,409 test rows) |
| Pipeline | Seven stages: source → preprocess → train → evaluate → explain → score → simulate |
| Dashboard | 5 Streamlit pages over persisted artifacts |
| Default suite | **634 passed, 17 skipped** (post-audit rerun) |
| Browser E2E | **10 passed** across all 5 pages (opt-in) |
| Full-profile run | All seven stages in 328.259 s, 613.9 MiB peak |

The test split was scored once. The MLP and decision threshold were selected
using validation evidence; held-out test results were not used to choose either.

## What it does

1. **Discovery** — queries reviewed adapters for OpenML, UCI and a pinned IBM
   GitHub sample; applies nine suitability gates and ranks candidates
   deterministically.
2. **Acquisition** — downloads over allowlisted HTTPS, verifies size, format and
   SHA-256, stores content-addressed bytes and records provenance.
3. **Validation and EDA** — infers the target and positive class, runs 12 quality
   checks, produces a profile and plots, and records a pass/warn/block gate.
4. **Preprocessing** — makes a seeded stratified train/validation/test split,
   fits imputers, encoders, scalers and derived features on training rows only,
   and persists the pipeline, schema and fingerprints.
5. **Modelling and evaluation** — trains a Keras MLP and compares it with
   Logistic Regression and Random Forest; selects a family and threshold on
   validation, then evaluates test once.
6. **Explainability** — uses TreeSHAP, LinearSHAP and DeepSHAP, groups one-hot
   contributions back to source features, checks additivity and persists global
   and local explanations.
7. **Risk intelligence and simulation** — ranks customers by configurable risk
   tiers, validates batch CSVs, exports safely, and projects campaign outcomes
   under explicit budget, cost and effectiveness assumptions.

The single-command orchestrator records every stage, timing, warning and artifact
in `artifacts/pipelines/<run-id>/run.json`. Resume reuses only verified stages
whose configuration and artifacts still match.

## Verified evaluation

Reference benchmark `20261021T145336Z-openml_46915-bench-04595d7b`:

| Family | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5905 | 0.6350 | 0.8741 | 0.5717 | 0.1460 |
| **Random Forest (selected)** | **0.7985** | **0.8718** | **0.9383** | **0.8973** | **0.0390** |
| MLP (Keras, `[64, 32]`) | 0.6937 | 0.7208 | 0.9266 | 0.7801 | 0.0944 |

These are held-out test results from one split. Five-fold CV F1 was RF 0.818
± 0.023, MLP 0.713 ± 0.038 and Logistic Regression 0.552 ± 0.025. The benchmark
metrics are committed in [docs/evidence/benchmark](docs/evidence/benchmark/);
the machine-readable table is in [metrics.json](docs/evidence/benchmark/metrics.json).

The **full-profile** run `post13-full-20261024c` later completed all seven stages
and reproduced these test metrics exactly. The **fast profile** used in Phase 13
live demonstrations is a laptop profile, not production hyperparameter evidence.

### Calibration is evaluated, not integrated

A separate validation-only experiment selected sigmoid calibration:
held-out Brier improved **0.0390 → 0.0306**, log loss **0.1684 → 0.1337** and ECE
**0.0686 → 0.0301**, while ROC-AUC and PR-AUC were unchanged. The calibrated
model serializes and reloads, but it is **not** wired into the production
registry, CLI, API or dashboard. This describes the historical OpenML
experiment; the current primary dashboard uses the Telco MLP run below.

### Earlier OpenML work is historical evidence

The project previously used OpenML `46915` / Random Forest as its primary
demonstration. That work remains in the artifact history and can still be
selected, but it is not the current primary dataset or model. The current
instructor-provided file is normalized, hashed, split, fitted, evaluated, and
scored as one linked evidence chain. `customerID` is a record reference, not a
predictor; `Churn` is the binary target. Blank `TotalCharges` values are handled
by the fitted preprocessing pipeline.

The current test metrics above come from the held-out test split and are kept
separate from the scored validation population. A dashboard's “observed churn”
count refers only to labels for the rows in that scored cohort; “predicted churn”
is the MLP output at the saved decision threshold. The metrics are evidence for
this split, not proof of causal impact or performance on a new population.

## Dashboard

The product has five pages, organised as one workflow — understand churn →
investigate a customer → evaluate the model → prioritise retention → simulate a
campaign:

1. **Executive Dashboard** — defaults to the successful, SHA-256-matched Telco
MLP scoring run; uses the screenshot-matched green Churn Analytics layout with
five segment filters, four data-backed KPI cards, seven customer/revenue charts,
a retention-priority list, the customer-attention table, selected-model test
metrics and SHAP. **Upload & predict** sits beside the scoring-run selector.
2. **Customer Risk Explorer** — a stored customer profile and assessment with
  stored SHAP drivers, plus explicit individual scoring. Batch CSV and preloaded
  dataset scoring is available from **Upload & predict** beside the Executive
  Dashboard scoring-run selector.
3. **Model Laboratory** — held-out performance, CV stability, global SHAP, data
quality and provenance; discovery, acquisition and MLP training live only in a
collapsed Operations tab.
4. **Retention Workbench** — filter, rank, fit customers to a budget, select and
export a contact list.
5. **Campaign Simulator** — hypothetical scenario comparison from explicit
assumptions, with a financial value bridge and a chart of prevented churn across
scenarios attached to the selected scoring run.

The dashboard:

- reads persisted artifacts and never substitutes placeholder metrics;
- shows actionable empty states with the command needed to create evidence;
- caches by path, modification time and size;
- performs discovery, training, scoring and simulation only from explicit
  actions, never on ordinary page render;
- scores individual records and batch CSVs through the same risk service used by
  the CLI;
- includes six synthetic Telco CSVs with the full MLP feature schema and six
  matching schema-invalid CSVs
  under [`testing_datasets/`](testing_datasets/README.md) for upload demos;
- labels simulations as hypothetical and leaves unsupported financial fields
  undefined.

To try an upload, open **Executive Dashboard → Analyze another customer
dataset → Upload & predict**. Choose **Proper · testing4.csv** (1,800 rows) in
the bundled dataset menu, or upload
`testing_datasets/proper_datasets/testing4.csv`. The saved MLP run becomes the
active dashboard population. For schema validation, choose **Improper ·
testing4.csv** or upload `testing_datasets/improper_datasets/testing4.csv`; it
is rejected because it is missing `InternetService`. These examples are
synthetic. Their `Churn` labels are kept for audit and never enter the model.

Uploads use the teacher-provided Telco MLP and preprocessing schema. The
offline `mini_churn.csv` fixture uses a different schema; see
[docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for the matching model workflow.

In Campaign Simulator, **Autofill example assumptions** creates varied draft
values; click **Run projection** to apply them. The value bridge and scenario
chart use the selected scoring run and applied assumptions. Amounts use the
scenario's configured value unit (the bundled examples use generic
`value_units`), and all outcomes are hypothetical rather than measured results.

The opt-in Playwright suite covers all five pages, risk filtering and CSV
download, stored customer detail, simulator projection, individual assessment,
batch upload and export, missing-model handling and empty-artifact states.

## Architecture

```mermaid
flowchart LR
  A[Public source] --> B[Discovery + suitability gates]
  B --> C[Acquisition + SHA-256 provenance]
  C --> D[Validation + EDA]
  D --> E[Leakage-safe preprocessing]
  E --> F[LR / Random Forest / MLP]
  F --> G[Validation-only selection]
  G --> H[Held-out test evaluation]
  G --> I[SHAP explanations]
  H --> J[Risk scoring + exports]
  I --> J
  J --> K[Retention simulation]
  H --> L[Streamlit dashboard]
  I --> L
  J --> L
  K --> L
```

**Stack:** Python 3.13, NumPy, pandas, scikit-learn, TensorFlow/Keras, SHAP,
Streamlit, Plotly, Matplotlib, joblib, PyArrow, pytest and Playwright. Data and
artifacts are local files; there is no database, backend service or cloud
dependency.

The dataset, preprocessing pipeline, split fingerprint and model digest are
carried through evaluation, explanation, scoring and simulation. Artifacts from
different runs are rejected rather than silently combined.

## Quick start

Requires CPython 3.13. The verified environment is Windows 11. Run from the
repository root.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt

# Offline fixture: populates a real artifact tree in about 1 minute
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13 --run-id getting-started

# Open the dashboard at http://localhost:8501
python -m streamlit run churnwise/dashboard/app.py
```

The fixture is a 160-row workflow test. Its metrics are **not** model
performance evidence. For the real dataset:

```powershell
# Network required; full profile, about 5.5 minutes and ~614 MiB peak
python -m churnwise.cli run --seed 42 --run-id reference-full
```

On macOS/Linux, use `python3.13 -m venv .venv` and `source .venv/bin/activate`;
that path is documented but not verified by this project. `requirements.lock.txt`
is Windows/CPython 3.13-specific; use `requirements.txt` on other platforms.

See **[docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)** for prerequisites, configuration,
all workflows, outputs and troubleshooting.

## Tests

```bash
python -m pytest -q                         # 634 passed, 17 skipped
python -m pytest -q -m live --run-live      # 7 passed, network required

# Browser E2E, opt-in
CHURNWISE_BROWSER_E2E=1 python -m pytest tests/e2e -m browser -q   # 10 passed
```

On Windows PowerShell, set `$env:CHURNWISE_BROWSER_E2E="1"` first. The default 17
skips are the 7 live-network tests and 10 browser tests; both suites pass when
explicitly enabled. See [HOW_TO_RUN.md](docs/HOW_TO_RUN.md#i-run-the-tests) for
focused suites and browser setup.

## Data provenance

The primary dashboard uses the teacher-provided IBM Telco Customer Churn CSV:
7,043 rows, 21 columns, and 1,869 observed churn labels (26.54%). Its SHA-256 is
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`. Churn is
retained for audit and dashboard reporting; it is excluded from model inputs.
OpenML `46915` remains a separate, 5,000-row reference dataset used by earlier
experiments.

Generated run artifacts remain ignored except for the small, curated default
dashboard snapshot: the matching MLP and preprocessor, evaluation metrics,
global/local SHAP explanations, and the 7,043-customer scoring run. Other local
run histories remain out of Git. The model uses a validation-selected 0.59
classification threshold; its scores are predictions, while the observed churn
labels are shown separately for audit.

## Streamlit Cloud

**Live dashboard:** [https://gangninja-churnwise-ai-churnwisedashboardapp-lanwl3.streamlit.app/](https://gangninja-churnwise-ai-churnwisedashboardapp-lanwl3.streamlit.app/)

Streamlit Cloud deploys `churnwise/dashboard/app.py` from the `main` branch.
The committed dashboard snapshot opens on all 7,043 rows of the teacher-provided
Telco Customer Churn dataset, scored by the MLP at its validation-selected
0.59 threshold. The default population run is
`telco-primary-full-7043-20261106`; the stored global SHAP explanation is
`telco-teacher-mlp-global-shap-20261106`. The Executive Dashboard includes the
observed churn audit, customer and revenue charts, attention list, model metrics,
and SHAP drivers. After a new push, Streamlit Cloud rebuilds the app; refresh the
live URL when that rebuild completes.

## Documentation

**Start here**

- [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) — install, configure, run, test and troubleshoot
- [PROJECT_CASE_STUDY.md](docs/PROJECT_CASE_STUDY.md) — full technical narrative
- [VALIDATION_SUMMARY.md](docs/VALIDATION_SUMMARY.md) — curated post-Phase-13 evidence
- [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) — 5–8 minute reviewer walkthrough
- [RESUME_BULLETS.md](docs/RESUME_BULLETS.md) — verified, ATS-friendly claims
- [PROJECT_SHOWCASE.md](docs/PROJECT_SHOWCASE.md) and [PROJECT_PRESENTATION.md](docs/PROJECT_PRESENTATION.md) — portfolio summaries

**Technical references**

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) · [ROADMAP.md](docs/ROADMAP.md) · [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md)
- [DISCOVERY.md](docs/DISCOVERY.md) · [ACQUISITION.md](docs/ACQUISITION.md) · [VALIDATION.md](docs/VALIDATION.md) · [PREPROCESSING.md](docs/PREPROCESSING.md)
- [EVALUATION.md](docs/EVALUATION.md) · [EXPLAINABILITY.md](docs/EXPLAINABILITY.md) · [RISK_INTELLIGENCE.md](docs/RISK_INTELLIGENCE.md) · [SIMULATION.md](docs/SIMULATION.md)
- [DASHBOARD.md](docs/DASHBOARD.md) · [RUNBOOK.md](docs/RUNBOOK.md) · [DEPENDENCIES.md](docs/DEPENDENCIES.md) · [PERFORMANCE.md](docs/PERFORMANCE.md)
- [Evidence index](docs/evidence/README.md) · [Post-Phase-13 evidence](docs/evidence/post_phase13/README.md)
- Phase reports: [1](docs/PHASE1_REPORT.md) · [2](docs/PHASE2_REPORT.md) · [3](docs/PHASE3_REPORT.md) · [4](docs/PHASE4_REPORT.md) · [5](docs/PHASE5_REPORT.md) · [6](docs/PHASE6_REPORT.md) · [7](docs/PHASE7_REPORT.md) · [8](docs/PHASE8_REPORT.md) · [9](docs/PHASE9_REPORT.md) · [10](docs/PHASE10_REPORT.md) · [11](docs/PHASE11_REPORT.md) · [12](docs/PHASE12_REPORT.md)
- [Phase 13 final verification](docs/FINAL_VERIFICATION_REPORT.md) (historical) · [Phase 13 acceptance matrix](docs/FINAL_ACCEPTANCE_MATRIX.md) (historical) · [Integration defects](docs/INTEGRATION_DEFECTS.md)

## Repository layout

```text
churnwise/
  discovery/       source adapters, G1–G9 suitability gates, ranking
  acquisition/     allowlisted download, integrity, cache, provenance
  validation/      V1–V12 checks, target inference, gate, reports
  eda/             dataset profiles and plots
  preprocessing/   split, train-only fitting, persisted pipeline
  models/          Keras MLP, baselines, registry, inference
  evaluation/      metrics, thresholds, fold-safe CV, figures
  explain/         SHAP explainers, grouping, global/local artifacts
  intelligence/    risk service, tiers, ranking, drivers, exports
  simulation/      campaign assumptions, budget, projections
  dashboard/       Streamlit app, cached loaders, pages, actions
  pipeline/        orchestration, manifest, resume, cleanup
  cli.py           command-line entry points
docs/              guides, case study, validation, evidence, phase reports
tests/             unit, integration, opt-in live and browser tests
scripts/           documentation checker
```

## Limitations

- Metrics describe **one primary dataset and one held-out split**; there are no
  confidence intervals or repeated-holdout variance estimates.
- The Random Forest has a train/validation overfitting gap (train F1 ≈ 0.99).
- Calibration was evaluated but **not integrated** into the production candidate.
- The second-dataset experiment is robustness evidence, not external
  generalisation or model transfer.
- SHAP values are model attributions, not causal effects. Logistic Regression
  values are log-odds and are not comparable with tree/MLP values.
- Risk tiers are configurable business categories, not optimized or calibrated
  decision boundaries.
- Campaign projections are hypothetical: no causal model, no measured uplift and
  no observed retention outcome.
- MLP results are not bitwise reproducible while TensorFlow op determinism is
  disabled; splits, tree models, scoring and simulation are deterministic for
  fixed artifacts.
- The dashboard is local and single-user; training is synchronous. The browser
  suite covers one engine and one desktop viewport and does not test
  responsive/accessibility/visual regressions.
- The exact lock is Windows/CPython 3.13-specific and has no hash pins.

See [VALIDATION_SUMMARY.md](docs/VALIDATION_SUMMARY.md) and
[PROJECT_CASE_STUDY.md](docs/PROJECT_CASE_STUDY.md) for the complete discussion.

## Reproducing the evidence

The default test suite and fixture pipeline are safe to run locally. The full
live pipeline, second-dataset experiment and calibration experiment require
network access and generated artifacts; their commands and limitations are
documented in [VALIDATION_SUMMARY.md](docs/VALIDATION_SUMMARY.md#9-reproduction).

The live dashboard is a public demonstration backed by committed, curated
artifacts. Uploaded datasets are session inputs and are not persisted to the
public repository.
