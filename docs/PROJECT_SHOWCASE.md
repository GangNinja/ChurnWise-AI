# ChurnWise AI — Project Showcase

**Customer churn intelligence from raw dataset to retention decision, with the
evidence kept honest.**

ChurnWise AI is a local-first Python platform that autonomously finds a public
customer-churn dataset, validates it, learns a churn model, explains it, scores a
customer population and explores retention-campaign scenarios. Every stage
persists auditable artifacts and can be rerun through a resumable pipeline or an
five-page Streamlit retention workbench.

> **Status:** implementation and verification complete through Phase 13, plus
> four post-Phase-13 validation workstreams: a full-profile pipeline, a
> second-dataset experiment, probability calibration and browser E2E. This is a
> portfolio/research system, not a production service. See
> [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md) for the exact evidence boundary.

---

## The problem

Customer-success teams can usually obtain churn data but struggle to turn it into
defensible action:

- Which public dataset is suitable, and can its provenance be trusted?
- Was preprocessing fitted only on training data, or did test information leak?
- Which model should be selected, and on which validation criterion?
- Why does the model rank a particular customer as high risk?
- Given a limited budget, which customers should be contacted — and what is the
  difference between a model probability and a business outcome?

ChurnWise AI addresses the full chain while keeping those questions explicit.
It is deliberately not a black-box “churn score API.”

---

## What the system does

| Capability | Implementation |
| --- | --- |
| Dataset discovery | Reviewed OpenML, UCI and pinned-GitHub adapters; deterministic ranking; nine gates covering target, size, licence, PII and schema |
| Acquisition | Bounded HTTPS download, SHA-256 integrity, archive safety, managed storage and provenance manifests |
| Validation | Target/positive-class detection, 12 quality checks, EDA profile and plots, explicit pass/fail gate |
| Preprocessing | Leakage-safe train/validation/test split; train-only imputation, encoding and scaling; persisted pipeline and schema |
| Modelling | Keras MLP (ReLU hidden layers, sigmoid output, BCE, Adam, early stopping, checkpoint restore) plus Logistic Regression and Random Forest baselines |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier, log-loss, confusion matrices, threshold analysis and calibration views |
| Explainability | Exact TreeSHAP for forests, DeepSHAP for the MLP, grouped original-feature contributions, global and local artifacts |
| Risk intelligence | Model probabilities, tiers independent of the 0.45 threshold, ranking/filtering, batch CSV validation and safe exports |
| Retention simulation | Budget/cost/effectiveness constraints, expected prevented churn, sensitivity analysis, undefined financial metrics kept undefined |
| Dashboard | 11 Streamlit pages over persisted artifacts; explicit actions only; cached reads; honest empty states |
| Orchestration | Seven-stage resumable pipeline with config digests, stage timing and failure codes |

---

## Architecture at a glance

```text
OpenML / local fixture
        │
        ▼
 Discovery ──► Acquisition ──► Validation + EDA
                                    │
                                    ▼
                         Leakage-safe preprocessing
                         (train / validation / test)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Logistic Reg.    Random Forest       MLP
                    └───────────────┼───────────────┘
                          Validation-only selection
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                 Test evaluation            SHAP explanations
                       │                  (global + local)
                       ▼                         │
                 Risk scoring ◄──────────────────┘
                       │
                       ▼
             Retention simulation
             (hypothetical projection)
                       │
                       ▼
              Streamlit dashboard
```

All stages write JSON/Parquet/PNG artifacts with run manifests and digests under
`artifacts/`, `data/` and `reports/`. The pipeline manifest is the durable source
of truth for what ran, with which configuration and for how long.

---

## Measured evidence

### End-to-end live run (Phase 13)

| Item | Value |
| --- | --- |
| Dataset | OpenML `46915` (customer churn, 5,000 rows) |
| SHA-256 | `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6` |
| Run ID | `phase13-final-live2` (post-fix) |
| Wall time | **105 s** (fast demonstration profile) |
| Stages | 7/7 succeeded |
| Split | 3,000 train / 1,000 validation / 1,000 test |
| Selected model | Random Forest, threshold 0.45, selected on validation F1 |
| Test metrics | accuracy 0.9650, precision 0.9015, recall 0.8440, F1 0.8718, ROC-AUC 0.9383, PR-AUC 0.8973 |
| Risk population | 1,000 scored; 858 low / 31 medium / 40 high / 71 critical |
| Simulation | 200 contacted at budget 1,000 / cost 5; expected prevented churn 32.83 at assumed effectiveness 0.30 |

Evidence: [phase13-live2/README.md](evidence/phase13-live2/README.md) and
[phase13-live2/run.json](evidence/phase13-live2/run.json). The earlier pre-fix run
(`phase13-final-live`, 66 s) is preserved in
[phase13-live](evidence/phase13-live); it produced identical metrics, tiers and
projections. The full generated trees are reproducible with the commands in
those READMEs; bulky artifacts are git-ignored.

> The Phase 13 live run used `--fast`. Its test metrics are valid for that
> configuration but are not a tuned production benchmark.

### Full-profile validation (post-Phase 13)

Run `post13-full-20261024c` used the documented non-`--fast` profile:

| Item | Result |
| --- | --- |
| Stages | **7/7 succeeded** (`source`, `preprocess`, `train`, `evaluate`, `explain`, `score`, `simulate`) |
| Wall time / peak memory | **328.259 s** / **613.9 MiB** |
| Selected model | Random Forest at threshold 0.45, selected on validation F1 |
| Test metrics | F1 **0.871795**, ROC-AUC **0.938342**, PR-AUC **0.897321**, Brier **0.039015** |
| Risk tiers | 1,000 scored; 858 low / 31 medium / 40 high / 71 critical |
| Simulation | 200 contacted at budget 1,000 / cost 5; expected prevented churn 32.8334 at assumed effectiveness 0.30 |

The full-profile run reproduced the committed benchmark metrics exactly. The
MLP weights were not bitwise identical across repeated runs because TensorFlow
op determinism is disabled; reported metrics were identical.

### Benchmark profile (committed)

On the full benchmark profile, validation-only selection chose Random Forest:

| Model | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC |
| --- | --- | --- | --- | --- |
| Random Forest (selected) | **0.7985** | **0.8718** | **0.9383** | **0.8973** |
| Logistic Regression | 0.5905 | 0.6350 | 0.8741 | 0.5717 |
| MLP | 0.6937 | 0.7208 | 0.9266 | 0.7801 |

Five-fold CV F1: Random Forest 0.818 ± 0.023, MLP 0.713 ± 0.038, Logistic
Regression 0.552 ± 0.025. The test partition was not used for selection.

Exact values and threshold policy: [benchmark metrics](evidence/benchmark/metrics.json)
and [benchmark report](evidence/benchmark/benchmark.md).

### Post-Phase-13 experiments

- **Calibration:** validation-only sigmoid selection improved test Brier from
  0.0390 to 0.0306, log loss from 0.1684 to 0.1337 and ECE from 0.0686 to 0.0301,
  with unchanged ROC/PR. It is **not integrated** into the production candidate.
- **Second dataset:** OpenML `42178` (IBM Telco, 7,043 rows) was used to retrain
  new models after an isolated ARFF-to-CSV format adapter. Logistic Regression
  reached test ROC-AUC 0.8380 / PR-AUC 0.6510 / F1 0.6196. This is pipeline
  robustness evidence, not transfer of the primary model.
- **Browser E2E:** 10 tests across all five pages passed with a real
  Chromium-based browser and real Streamlit servers.

Details and limits: [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md).

### Test suite

```bash
python -m pytest -q
# 634 passed, 17 skipped (7 live-network + 10 browser)

python -m pytest -q -m live --run-live
# 7 passed (requires network)

CHURNWISE_BROWSER_E2E=1 python -m pytest tests/e2e -m browser -q
# 10 passed (real browser and Streamlit servers)
```

### Explainability

- Global ranking: [global_random_forest.json](evidence/explainability/global_random_forest.json)
- Full 1,000-row local contribution matrix (a separate Phase 9 risk run, not the
  Phase 13 fast run): the summary is
  [matrix_1000_summary.json](evidence/explainability/matrix_1000_summary.json);
  `artifacts/explanations/20261022T155236Z-openml_46915-risk-final-drivers/local/`
  holds the 1,000 per-row files in the working tree.
- The Phase 13 fast pipeline stores drivers for the top 10 customers only
  (`drivers_available: 10`); this is a configured limit, not full-population SHAP.
- Additivity and feature-grouping checks are covered by unit and integration
  tests; SHAP values are model attributions, **not causal effects**.

### Dashboard

- Five pages: Executive Dashboard, Customer Risk Explorer, Model Laboratory,
  Retention Workbench and Campaign Simulator.
- Verified with Streamlit `AppTest`: every page renders without exceptions; the
  documented first render was **1.937 s** in this environment.
- Opt-in Playwright E2E: **10 passed**, covering all five pages plus risk
  filtering/download, stored customer detail, simulator, individual assessment,
  batch scoring/export, missing-model handling and empty states.
- Screenshot: [Executive Dashboard](screenshots/redesign-executive.png)

---

## Engineering highlights

- **Leakage prevention by construction** — the split happens before any learned
  transformation, and every transformer is fitted on the training partition only.
  Tests assert that validation/test distributions are not used for fitting.
- **Selection discipline** — the test split is never used for model selection or
  threshold tuning; the benchmark records the selection metric and policy.
- **Artifact provenance** — dataset digests, model hashes, split fingerprints,
  configuration digests and explicit schema versions accompany persisted outputs.
- **Failure semantics** — typed error codes, resumable stage manifests, refusal to
  overwrite a mismatched run, and dry-run cleanup by default.
- **Simulation honesty** — the simulator separates probability mass from customer
  counts, labels projections as hypothetical, and returns `null` with reasons for
  financial metrics that lack a value assumption.
- **Privacy-aware exports** — risk CSV exports exclude raw identifiers by default;
  explanation artifacts store transformed values and contributions, not raw
  customer records.

---

## Technology stack

Python 3.13 · pandas / NumPy / PyArrow · scikit-learn · TensorFlow/Keras ·
SHAP · Matplotlib · Streamlit · pytest · OpenML REST API · joblib · Parquet.

Full dependency policy: [DEPENDENCIES.md](DEPENDENCIES.md). Platform note: the
committed lock was generated on Windows/CPython 3.13; other platforms should use
the range manifest or regenerate the lock.

---

## Run it

```bash
python -m venv .venv
# activate the environment
python -m pip install -r requirements.lock.txt

# offline, synthetic, ~1 minute
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv \
  --fast --seed 13 --run-id demo

# live OpenML, fast profile
python -m churnwise.cli run --fast --seed 42 --run-id demo-live

# full reference profile (network required, about 5.5 minutes)
python -m churnwise.cli run --seed 42 --run-id reference-full

python -m streamlit run churnwise/dashboard/app.py
```

See [README.md](../README.md), [HOW_TO_RUN.md](HOW_TO_RUN.md) and
[DEMO_SCRIPT.md](DEMO_SCRIPT.md) for the full walkthrough. For the complete
technical narrative, see [PROJECT_CASE_STUDY.md](PROJECT_CASE_STUDY.md).

---

## Limitations and responsible use

1. **Projections are not outcomes.** Expected prevented churn is
   `sum(selected probabilities) × assumed effectiveness`; it is not measured
   uplift, revenue or ROI.
2. **SHAP is not causality.** Attributions describe model behaviour on transformed
   inputs.
3. **Public benchmark, not production.** OpenML `46915` is a public dataset; no
   real customer data, deployment, monitoring or A/B test is included.
4. **The Phase 13 live metrics came from a fast profile.** A later full-profile
   validation run completed all seven stages and reproduced the committed
   benchmark, but the evidence remains one dataset and one split.
5. **Calibration is not in production.** Sigmoid improved probability-quality
   metrics in a separate validation-only experiment, but the shipped candidate
   remains the uncalibrated Random Forest at threshold 0.45.
6. **The second dataset is not transfer evidence.** OpenML `42178` used newly
   trained models on a different schema and population.
7. **MLP reproducibility is statistical, not bitwise.** Splits, tree models,
   scoring, ranking and simulation are deterministic for fixed artifacts; MLP
   float results can differ across runs.
8. **Local single-user design.** No authentication, multi-tenancy, database or
   cloud deployment; cleanup assumes one process at a time.
9. **Artifact compatibility is phase-specific.** Validation is implemented per
   artifact contract, not as a universal JSON-Schema registry.
10. **Model risk requires governance.** Churn scores should be reviewed for bias,
    calibrated for the target population, and paired with human decisions before
    any customer-facing use.

---

## Documentation map

- [README.md](../README.md) — setup, usage, testing
- [HOW_TO_RUN.md](HOW_TO_RUN.md) — installation, configuration and troubleshooting
- [PROJECT_CASE_STUDY.md](PROJECT_CASE_STUDY.md) — full technical narrative
- [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md) — curated validation evidence
- [ARCHITECTURE.md](ARCHITECTURE.md) — design and data flow
- [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) — phase criteria
- [FINAL_ACCEPTANCE_MATRIX.md](FINAL_ACCEPTANCE_MATRIX.md) — criterion-level status
- [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md) — evidence and limits
- [PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md) — presentation outline
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md) — reproducible demo
- [RESUME_BULLETS.md](RESUME_BULLETS.md) — verified resume claims
