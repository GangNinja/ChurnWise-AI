# ChurnWise AI

Autonomous customer-churn intelligence: discover a public churn dataset, acquire
and verify it, validate it, build a leakage-safe feature pipeline, train a Keras
MLP, and benchmark it against scikit-learn baselines on one held-out split — with
every number traceable to a saved artifact.

The system is built in phases. **Phases 1–7 are implemented and verified.**
Phases 8–13 (SHAP explainability, risk intelligence, campaign simulation,
dashboard, end-to-end orchestration, final verification) are planned, not built;
this README does not claim them.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Repository, architecture, requirements | Complete |
| 2 | Autonomous dataset discovery | Complete |
| 3 | Acquisition and provenance | Complete |
| 4 | Validation and data intelligence | Complete |
| 5 | Preprocessing and feature engineering | Complete |
| 6 | MLP development and training | Complete |
| 7 | Comparative benchmarking and evaluation | Complete |
| 8–13 | SHAP, risk tiers, simulation, dashboard, E2E, final docs | Not started |

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
```

Useful variants:

```bash
python -m churnwise.cli validate --offline --json   # cached artifacts, JSON output
python -m churnwise.cli evaluate --regenerate <benchmark-run-id>
python -m churnwise.cli predict --input customers.csv
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
  cli.py           command-line entry points
docs/              architecture, roadmap, acceptance criteria, per-phase reports
tests/             unit, integration and opt-in live-network tests
scripts/           documentation checker
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — module boundaries, contracts, guarantees
- [ROADMAP.md](docs/ROADMAP.md) — all 13 phases, dependencies and exit gates
- [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md) — verifiable criteria per phase
- [DISCOVERY.md](docs/DISCOVERY.md) · [ACQUISITION.md](docs/ACQUISITION.md) · [VALIDATION.md](docs/VALIDATION.md) · [PREPROCESSING.md](docs/PREPROCESSING.md) · [MODELING.md](docs/MODELING.md) · [EVALUATION.md](docs/EVALUATION.md)
- Phase reports: [1](docs/PHASE1_REPORT.md) · [2](docs/PHASE2_REPORT.md) · [3](docs/PHASE3_REPORT.md) · [4](docs/PHASE4_REPORT.md) · [5](docs/PHASE5_REPORT.md) · [6](docs/PHASE6_REPORT.md) · [7](docs/PHASE7_REPORT.md)

## Tests

```bash
python -m pytest -q                 # unit + integration; live-network tests skip
python -m pytest -q --run-live      # adds real OpenML/UCI/HTTP checks
```

Latest verified state: **390 passed, 7 skipped**, with the seven skips being the
opt-in live-network tests. Phase 7 evaluation tests: **57 passed**.

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

Limitations, stated rather than hidden:

- Metrics describe one dataset and one held-out split; no confidence intervals.
- Thresholds optimise validation F1, not contact cost or campaign capacity.
- Probability calibration is summarised by Brier score; no recalibration is fitted.
- The Random Forest shows a train/validation overfitting gap (train F1 ≈ 0.99).
- Cross-machine bitwise determinism is not guaranteed (BLAS/TensorFlow versions).
- No SHAP, risk tiers, simulation or dashboard code exists yet (Phases 8–13).
