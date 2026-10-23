# ChurnWise AI

Autonomous customer-churn intelligence: discover a public churn dataset, acquire
and verify it, validate it, build a leakage-safe feature pipeline, train a Keras
MLP, benchmark it against scikit-learn baselines on one held-out split, explain
its predictions with SHAP, rank customers by configurable churn-risk tiers, and
simulate retention campaigns under explicit, user-supplied assumptions — with
every number traceable to a saved artifact.

The system is built in phases. **All 13 phases are implemented and verified**,
with the evidence and remaining limitations recorded in
[FINAL_VERIFICATION_REPORT.md](docs/FINAL_VERIFICATION_REPORT.md) and
[FINAL_ACCEPTANCE_MATRIX.md](docs/FINAL_ACCEPTANCE_MATRIX.md).

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Repository, architecture, requirements | Complete |
| 2 | Autonomous dataset discovery | Complete |
| 3 | Acquisition and provenance | Complete |
| 4 | Validation and data intelligence | Complete |
| 5 | Preprocessing and feature engineering | Complete |
| 6 | MLP development and training | Complete |
| 7 | Comparative benchmarking and evaluation | Complete |
| 8 | Explainable AI and SHAP | Complete |
| 9 | Customer churn risk intelligence and exports | Complete |
| 10 | Retention campaign simulation | Complete |
| 11 | Interactive intelligence dashboard | Complete |
| 12 | End-to-end orchestration, reliability and performance | Complete |
| 13 | Final verification, documentation and presentation | Complete |

## Measured results

Reference run `20261021T145336Z-openml_46915-bench-04595d7b` on OpenML dataset
46915 (5,000 rows, 27 transformed features, stratified 60/20/20 split, 1,000
held-out test rows, 14.1% churn). Thresholds were selected on validation and
frozen; the test split was scored once. The later fast-profile live runs
reproduced the selected Random Forest and Logistic Regression test values; the
fast-profile MLP is reported separately (test F1 0.7068) because it trains fewer
epochs than the benchmark profile.

| Family | Test F1 | Test ROC-AUC | Test PR-AUC | Brier |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6350 | 0.8741 | 0.5717 | 0.1460 |
| Random Forest **(selected)** | 0.8718 | 0.9383 | 0.8973 | 0.0390 |
| MLP (Keras) | 0.7208 | 0.9266 | 0.7801 | 0.0944 |

Random Forest was selected on validation F1 (0.7985), not on test performance.
Full tables, curves, confusion matrices and CV stability estimates are in
[PHASE7_REPORT.md](docs/PHASE7_REPORT.md); the underlying `metrics.json`,
`benchmark.md`, predictions and figures are regenerated locally by the commands
below into the git-ignored `artifacts/` tree. A committed copy of the benchmark
metrics is in [docs/evidence/benchmark](docs/evidence/benchmark).

Two separate fast-profile live Phase 13 runs completed all seven stages on the
same OpenML dataset: 66 s initially and 105 s after the determinism fix. Both
produced identical metrics, tier counts and projections; evidence is in
[docs/evidence/phase13-live2](docs/evidence/phase13-live2) and
[docs/evidence/phase13-live](docs/evidence/phase13-live).

## Quick start

Requires Python 3.13. Run from the repository root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.lock.txt   # exact audited versions
# Cross-platform alternative: python -m pip install -r requirements.txt

# One command: discovery → acquisition → validation → preprocessing →
# MLP training → benchmark → SHAP → risk scoring → retention simulation.
python -m churnwise.cli run --fast --seed 42

python -m streamlit run churnwise/dashboard/app.py   # interactive dashboard
```

`--fast` is a laptop demonstration profile; its metrics are not production
hyperparameter evidence. The offline equivalent uses the committed fixture and
needs no network:

```bash
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13
```

The individual stage commands remain available for diagnosis and focused work:

```bash
python -m churnwise.cli discover          # rank public churn datasets
python -m churnwise.cli acquire           # download + verify + record provenance
python -m churnwise.cli validate          # schema/quality gate + EDA report
python -m churnwise.cli prepare           # split, fit preprocessing, persist matrices
python -m churnwise.cli train             # train the MLP on the training split
python -m churnwise.cli evaluate --cv-folds 5
python -m churnwise.cli explain             # SHAP for the selected family
python -m churnwise.cli score --population val --explain all
python -m churnwise.cli simulate --budget 1000 --cost-per-contact 5 --effectiveness 0.30
```

Useful variants:

```bash
python -m churnwise.cli run --run-id demo --resume          # reuse verified stages
python -m churnwise.pipeline.cleanup --run-id demo          # dry-run artifact cleanup
python -m churnwise.pipeline.cleanup --run-id demo --delete
python -m churnwise.cli validate --offline --json   # cached artifacts, JSON output
python -m churnwise.cli evaluate --regenerate <benchmark-run-id>
python -m churnwise.cli explain --family mlp
python -m churnwise.cli explain --input customers.csv --family random_forest
python -m churnwise.cli predict --input customers.csv
python -m churnwise.cli score --input customers.csv --explain top --explain-top-n 20
python -m churnwise.cli risk-export --run-id <risk-run-id> --tier high --tier critical --limit 100
python -m churnwise.cli simulate --monthly-value 50 --horizon-months 6 \
  --strategy risk_threshold --min-risk 0.30 --name "Q4 pilot"
python -m churnwise.cli simulate --strategy risk_category --risk-tier critical \
  --budget 600 --cost-per-contact 12 --effectiveness 0.45 --json
```

Exit codes: `0` success, `2` structured rejection/failure, `3` invalid
configuration or report-write failure. Missing prerequisites fail with an
actionable code rather than a traceback.

## What each command produces

| Command | Output |
| --- | --- |
| `discover` | `reports/acquisition/{candidates,rejections}.json` — ranked candidates with evidence |
| `acquire` | verified file under `data/raw/`, manifest, content-addressed cache, `attempts.json` |
| `validate` | `reports/validation/<dataset>.{json,md}`, EDA profile and plots, gate decision |
| `prepare` | `artifacts/prepared/<run_id>/` matrices, split fingerprint, `pipeline.joblib`, schema |
| `train` | `artifacts/models/<run_id>/mlp/{model.keras,best.keras,metadata.json,history.json,...}` |
| `predict` | CSV of probabilities and classes for raw customer records |
| `evaluate` | `artifacts/evaluation/<run_id>/{metrics.json,benchmark.md,run_manifest.json,predictions.npz,curves.json,threshold_analysis.json,*.png}` |
| `explain` | `artifacts/explanations/<run_id>/{global.json,importance.png,summary.png,dependence_*.png,run_manifest.json,local/*.json,local/*.png}` |
| `score` | `artifacts/predictions/<run_id>/{churn_scores.parquet,risk_summary.json,tiers.csv,run_manifest.json,drivers.json,exports/*}` |
| `risk-export` | filtered CSV/JSON/Parquet under the risk run's `exports/` directory |
| `simulate` | `artifacts/simulations/<scenario_id>/{result.json,selection.csv,sensitivity.json,sensitivity.csv,comparison.*,report.md,run_manifest.json,plots/*.png}` |
| `run` | all of the above plus `artifacts/pipelines/<run-id>/run.json` stage manifest |
| dashboard | no writes on render; explicit actions run discovery/acquisition/training/assessment/simulation |

All generated data, models, reports and artifacts are reproducible and
git-ignored; only source, tests and documentation are committed.

## Repository layout

```text
churnwise/
  discovery/       source adapters, suitability gates, ranking (Phase 2)
  acquisition/     allowlisted download, integrity, cache, provenance (Phase 3)
  validation/      schema/quality checks, target/roles, gate, reports (Phase 4)
  eda/             dataset profile and plots (Phase 4)
  preprocessing/   leakage-safe split, pipeline, artifacts (Phase 5)
  models/          Keras MLP, training, registry, inference, baselines (Phase 6/7)
  evaluation/      metrics, thresholds, fold-safe CV, figures, reports (Phase 7)
  explain/         SHAP explainers, grouping, global/local explanations (Phase 8)
  intelligence/    risk service, tiers, ranking, drivers, exports (Phase 9)
  simulation/      campaign assumptions, targeting, budget, projections (Phase 10)
  dashboard/       Streamlit app, cached artifact loaders, pages, actions (Phase 11)
  pipeline/        end-to-end orchestration, manifest, resume, cleanup (Phase 12)
  cli.py           command-line entry points
docs/              architecture, roadmap, acceptance criteria, per-phase reports
tests/             unit, integration and opt-in live-network tests
scripts/           documentation checker
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — module boundaries, contracts, guarantees
- [ROADMAP.md](docs/ROADMAP.md) — all 13 phases, dependencies and exit gates
- [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md) — verifiable criteria per phase
- [DISCOVERY.md](docs/DISCOVERY.md) · [ACQUISITION.md](docs/ACQUISITION.md) · [VALIDATION.md](docs/VALIDATION.md) · [PREPROCESSING.md](docs/PREPROCESSING.md) · [MODELING.md](docs/MODELING.md) · [EVALUATION.md](docs/EVALUATION.md) · [EXPLAINABILITY.md](docs/EXPLAINABILITY.md) · [RISK_INTELLIGENCE.md](docs/RISK_INTELLIGENCE.md) · [SIMULATION.md](docs/SIMULATION.md) · [DASHBOARD.md](docs/DASHBOARD.md) · [RUNBOOK.md](docs/RUNBOOK.md)
- Phase reports: [1](docs/PHASE1_REPORT.md) · [2](docs/PHASE2_REPORT.md) · [3](docs/PHASE3_REPORT.md) · [4](docs/PHASE4_REPORT.md) · [5](docs/PHASE5_REPORT.md) · [6](docs/PHASE6_REPORT.md) · [7](docs/PHASE7_REPORT.md) · [8](docs/PHASE8_REPORT.md) · [9](docs/PHASE9_REPORT.md) · [10](docs/PHASE10_REPORT.md) · [11](docs/PHASE11_REPORT.md) · [12](docs/PHASE12_REPORT.md)
- Final verification: [FINAL_VERIFICATION_REPORT.md](docs/FINAL_VERIFICATION_REPORT.md) · [FINAL_ACCEPTANCE_MATRIX.md](docs/FINAL_ACCEPTANCE_MATRIX.md) · [INTEGRATION_DEFECTS.md](docs/INTEGRATION_DEFECTS.md)
- Portfolio: [PROJECT_SHOWCASE.md](docs/PROJECT_SHOWCASE.md) · [PROJECT_PRESENTATION.md](docs/PROJECT_PRESENTATION.md) · [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) · [RESUME_BULLETS.md](docs/RESUME_BULLETS.md)
- Operations and measurements: [RUNBOOK.md](docs/RUNBOOK.md) · [PERFORMANCE.md](docs/PERFORMANCE.md) · [DEPENDENCIES.md](docs/DEPENDENCIES.md)
- Integration audit: [INTEGRATION_AUDIT.md](docs/INTEGRATION_AUDIT.md) · [INTEGRATION_DEFECTS.md](docs/INTEGRATION_DEFECTS.md) · [Evidence index](docs/evidence/README.md)

## Tests

```bash
python -m pytest -q                 # unit + integration; live-network tests skip
python -m pytest -q --run-live      # adds real OpenML/UCI/HTTP checks
```

Latest verified state: **624 passed, 7 skipped** (268.70 s), with the seven skips
being the opt-in live-network tests. Those seven also passed when run alone with
`python -m pytest -q -m live --run-live`. The same suite passed in a clean
environment installed only from `requirements.lock.txt` (**624 passed, 7
skipped** in 326.77 s).

Focused suites: Phase 12 pipeline **24 passed** (14 integration, 6 contract and
boundary, 4 cleanup); Phase 11 dashboard **42 passed**; Phase 10 simulation **57
passed** (47 unit + 10 integration); Phase 9 risk **58 passed**; Phase 8
explanation **42 passed**; Phase 7 evaluation **57 passed**. The preserved test
evidence is in [docs/evidence/phase13-tests](docs/evidence/phase13-tests/README.md).

## Guarantees and limitations

Guaranteed by tests and artifacts:

- The test split is never used for threshold, hyperparameter or model selection;
  it is scored once (`test_evaluation_calls = 1`).
- Split and preprocessing fingerprints are reproducible; transformed columns have
  one order for every consumer.
- Cross-validation is fold-safe: preprocessing is refitted on each fold's
  training rows after the reconstructed base frame is verified against the Phase 5
  matrices.
- Every metric in the report is recomputable from saved predictions
  (`evaluate --regenerate`).
- SHAP explanations are tied to the exact dataset hash, split fingerprint,
  preprocessing run and model digest; additivity is checked before values are
  reported, and the test split is never used.
- Every risk score comes from the shipped Phase 7 model through the verified
  Phase 8 loader. Tier counts reconcile to the scored population, ranks are
  unique and deterministic, and stale model/preprocessor artifacts are rejected.
- CSV exports neutralise spreadsheet formulas, exclude raw identifiers and are
  confined to the risk run's artifact directory.
- Campaign selections are deterministic (probability, then record reference),
  every budget is enforced against actual selected costs, and population counts
  reconcile to the assessed rows.
- Simulation results embed the assumptions used, label outcomes as projections,
  return `null` with a reason for undefined ROI/break-even/cost-efficiency, and
  never infer customer value or intervention effectiveness from data.
- One command runs discovery through simulation and records every stage in a
  durable manifest; cross-phase checks reject a run whose model, explanation,
  risk and simulation artifacts do not belong to the same run.
- Resume reuses a stage only when its configuration matches and every recorded
  artifact still exists; missing artifacts invalidate that stage and all later
  stages. A corrupt manifest is refused, not overwritten.
- The cleanup tool is a dry run by default and deletes only run-owned paths
  inside the managed artifact/report roots; shared acquisition data and
  dataset-level reports are retained.
- Every dashboard page reads persisted artifacts; missing evidence is an
  actionable empty state, never a placeholder number. New-record scoring calls
  the Phase 9 service rather than duplicating preprocessing or model logic.
- Dashboard caches are keyed by file path, modification time and size; a written
  artifact invalidates the affected load, and a changed batch upload clears the
  previous batch result.

Limitations, stated rather than hidden:

- Metrics describe one dataset and one held-out split; no confidence intervals.
- Thresholds optimise validation F1, not contact cost or campaign capacity.
- Probability calibration is summarised by Brier score; no recalibration is fitted.
- The Random Forest shows a train/validation overfitting gap (train F1 ≈ 0.99).
- Cross-machine bitwise determinism is not guaranteed (BLAS/TensorFlow versions).
- SHAP rankings are sample estimates (seed 42; 200 LR/RF rows, 60 MLP rows), not
  population values; Logistic Regression values are log-odds and are not
  comparable with tree/MLP probability-space values. SHAP is not causal.
- Risk-tier defaults are business categories, not optimised thresholds; the
  model probabilities are not recalibrated.
- Value-at-risk is available only when a real revenue proxy is supplied. The
  reference OpenML dataset has none, so no monetary value is invented.
- Campaign projections are hypothetical: assumed effectiveness is not measured
  uplift, there is no causal model, and no observed retention outcome exists to
  validate the projections. Without a supplied `monthly_value`, saved-value,
  net-benefit, ROI and break-even are unavailable.
- Dashboard training runs synchronously inside Streamlit; there is no
  background job or live epoch log. Batch CSV scoring does not compute SHAP.
- No automated browser end-to-end suite is included; Phase 11 screenshots are
  stored in [docs/screenshots](docs/screenshots/).
- Phase 12 live evidence used the fast profile (24 MLP epochs, 3-fold CV, 40-row
  SHAP sample), not the full production profile. It is not a hyperparameter-search
  claim.
- MLP floats are not bitwise reproducible while TensorFlow op determinism is
  disabled. Random Forest now defaults to serial inference (`n_jobs=1`) so its
  predictions are bitwise stable; explicitly selecting `n_jobs=-1` can change the
  last floating-point bit. Splits, Logistic Regression, scoring and simulation
  are deterministic for fixed artifacts.
- Artifact validation is distributed by phase plus principal cross-phase
  assertions; there is no universal JSON-Schema registry.
- Exact dependency reproduction uses the platform-specific
  `requirements.lock.txt`; `requirements.txt` remains the cross-platform range
  manifest. See [DEPENDENCIES.md](docs/DEPENDENCIES.md).
- Artifact cleanup assumes one user/process at a time and has no lock. It never
  deletes shared acquisition data.
- Phase 13 verification used the fast profile for its live run; the full
  production training/SHAP profile was not executed end to end.
