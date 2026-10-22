# ChurnWise AI

Autonomous customer-churn intelligence: discover a public churn dataset, acquire
and verify it, validate it, build a leakage-safe feature pipeline, train a Keras
MLP, benchmark it against scikit-learn baselines on one held-out split, explain
its predictions with SHAP, rank customers by configurable churn-risk tiers, and
simulate retention campaigns under explicit, user-supplied assumptions — with
every number traceable to a saved artifact.

The system is built in phases. **Phases 1–10 are implemented and verified.**
Phases 11–13 (dashboard, end-to-end orchestration, final verification) are
planned, not built; this README does not claim them.

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
| 11–13 | Dashboard, E2E, final docs | Not started |

## Measured results

Reference run `20261021T145336Z-openml_46915-bench-04595d7b` on OpenML dataset
46915 (5,000 rows, 27 transformed features, stratified 60/20/20 split, 1,000
held-out test rows, 14.1% churn). Thresholds were selected on validation and
frozen; the test split was scored once.

| Family | Test F1 | Test ROC-AUC | Brier |
| --- | --- | --- | --- |
| Logistic Regression | 0.6350 | 0.8741 | 0.1460 |
| Random Forest **(selected)** | 0.8718 | 0.9383 | 0.0390 |
| MLP (Keras) | 0.7208 | 0.9266 | 0.0944 |

Random Forest was selected on validation F1 (0.7985), not on test performance.
Full tables, curves, confusion matrices and CV stability estimates are in
[PHASE7_REPORT.md](docs/PHASE7_REPORT.md); the underlying `metrics.json`,
`benchmark.md`, predictions and figures are regenerated locally by the commands
below into the git-ignored `artifacts/` tree. Run IDs are recorded in every
report so a result can be traced to its artifact.

## Quick start

Requires Python 3.13. Run from the repository root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.txt

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
  cli.py           command-line entry points
docs/              architecture, roadmap, acceptance criteria, per-phase reports
tests/             unit, integration and opt-in live-network tests
scripts/           documentation checker
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — module boundaries, contracts, guarantees
- [ROADMAP.md](docs/ROADMAP.md) — all 13 phases, dependencies and exit gates
- [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md) — verifiable criteria per phase
- [DISCOVERY.md](docs/DISCOVERY.md) · [ACQUISITION.md](docs/ACQUISITION.md) · [VALIDATION.md](docs/VALIDATION.md) · [PREPROCESSING.md](docs/PREPROCESSING.md) · [MODELING.md](docs/MODELING.md) · [EVALUATION.md](docs/EVALUATION.md) · [EXPLAINABILITY.md](docs/EXPLAINABILITY.md) · [RISK_INTELLIGENCE.md](docs/RISK_INTELLIGENCE.md) · [SIMULATION.md](docs/SIMULATION.md)
- Phase reports: [1](docs/PHASE1_REPORT.md) · [2](docs/PHASE2_REPORT.md) · [3](docs/PHASE3_REPORT.md) · [4](docs/PHASE4_REPORT.md) · [5](docs/PHASE5_REPORT.md) · [6](docs/PHASE6_REPORT.md) · [7](docs/PHASE7_REPORT.md) · [8](docs/PHASE8_REPORT.md) · [9](docs/PHASE9_REPORT.md) · [10](docs/PHASE10_REPORT.md)

## Tests

```bash
python -m pytest -q                 # unit + integration; live-network tests skip
python -m pytest -q --run-live      # adds real OpenML/UCI/HTTP checks
```

Latest verified state: **547 passed, 7 skipped**, with the seven skips being the
opt-in live-network tests. Phase 10 simulation tests: **57 passed** (47 unit +
10 integration); Phase 9 risk tests: **58 passed**; Phase 8 explanation tests:
**42 passed**; Phase 7 evaluation tests: **57 passed**.

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
- No dashboard or orchestration code exists yet (Phases 11–13).
