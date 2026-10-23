# ChurnWise AI — Final Verification Report

**Project:** ChurnWise AI customer-churn intelligence platform  
**Phase:** 13 — final verification, documentation and presentation  
**Verification date:** 2026-10-23  
**Repository state:** final Phase 13 working tree, after the Random Forest
determinism fix (D-12)  
**Overall status:** **VERIFIED WITH DOCUMENTED LIMITATIONS** — the supported
workflow is implemented and exercised end to end; this is not a production
deployment and no business outcome is claimed.

Companion documents: [FINAL_ACCEPTANCE_MATRIX.md](FINAL_ACCEPTANCE_MATRIX.md),
[INTEGRATION_DEFECTS.md](INTEGRATION_DEFECTS.md),
[PROJECT_SHOWCASE.md](PROJECT_SHOWCASE.md),
[PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md),
[DEMO_SCRIPT.md](DEMO_SCRIPT.md), [RESUME_BULLETS.md](RESUME_BULLETS.md).

---

## 1. Final project status

| Dimension | Status | Evidence |
| --- | --- | --- |
| Implementation | Complete for all 13 phases | Source, phase reports, 13-page acceptance structure |
| Automated tests | **624 passed, 7 skipped** (268.70 s) | `pytest -q --disable-warnings`, post-fix audit environment |
| Live network tests | **7 passed** | `pytest -q -m live --run-live` |
| Clean environment | **624 passed, 7 skipped** in 326.77 s | `.venv_clean` from `requirements.lock.txt` |
| Live end-to-end run | **7/7 stages succeeded** | `phase13-final-live2`, 105 s wall / 93 s manifest |
| Dashboard | **11 pages**, 42 focused tests, first render 1.937 s | AppTest + live HTTP/browser evidence |
| Documentation checks | **114 passed, 0 failed** | `scripts/verify_phase1_docs.py` |
| Acceptance matrix | **13 VERIFIED, 0 blocked** | [FINAL_ACCEPTANCE_MATRIX.md](FINAL_ACCEPTANCE_MATRIX.md) |
| Critical open defects | **None** | D-08/D-09 are documented non-critical limits |

**What is genuinely complete:** a first-time user can install the lock, run live
discovery through simulation in one command, start the dashboard, inspect the
real artifacts, score new records, and create a clearly labelled retention
projection.

**What is not claimed:** production deployment, real customer retention, causal
effects, measured uplift, state-of-the-art accuracy, or bitwise reproducibility
of default MLP floats.

---

## 2. Verified capabilities and evidence

| Capability | Verified behavior | Evidence |
| --- | --- | --- |
| Autonomous discovery | OpenML, UCI and pinned GitHub candidates; deterministic ranking; nine suitability gates; duplicate handling | [PHASE2_REPORT.md](PHASE2_REPORT.md); live discovery tests |
| Autonomous acquisition | Allowlisted HTTPS, size/type/parser checks, SHA-256, content-addressed cache, provenance, fallback | [PHASE3_REPORT.md](PHASE3_REPORT.md); `test_live_acquisition.py` |
| Validation and EDA | Target/positive-label detection, 12 quality checks, role assignment, EDA profile and four plots, gate | [PHASE4_REPORT.md](PHASE4_REPORT.md); live validation test |
| Leakage-safe preprocessing | Split before fitting; train-only imputers/encoders/scaler; persisted `ColumnTransformer`; feature names and fingerprints | [PHASE5_REPORT.md](PHASE5_REPORT.md); leakage suite |
| MLP training | ReLU hidden layers, sigmoid output, BCE, Adam, backpropagation, early stopping, checkpoint restore, history | [PHASE6_REPORT.md](PHASE6_REPORT.md); MLP suite |
| Fair benchmark | LR/RF/MLP on identical rows; validation-only selection; test scored once; metrics, curves, CV, error analysis | [PHASE7_REPORT.md](PHASE7_REPORT.md); benchmark artifact |
| Explainability | TreeSHAP, LinearSHAP, DeepSHAP; grouped original features; global and local artifacts; additivity checks | [PHASE8_REPORT.md](PHASE8_REPORT.md); SHAP suite |
| Risk intelligence | Genuine probabilities, tiers distinct from threshold, ranking/filtering, batch validation, exports | [PHASE9_REPORT.md](PHASE9_REPORT.md); risk suite |
| Campaign simulation | Budget/cost/effectiveness constraints, expected prevented churn, sensitivity, comparison, undefined financial metrics kept undefined | [PHASE10_REPORT.md](PHASE10_REPORT.md); 57 tests |
| Dashboard | 11 pages over persisted artifacts; cached reads; explicit action buttons; honest empty states | [PHASE11_REPORT.md](PHASE11_REPORT.md); 42 tests |
| Orchestration | Seven resumable stages; config digest; stage timings; structured failures; dry-run cleanup | [PHASE12_REPORT.md](PHASE12_REPORT.md); 24 tests |
| Documentation | README, architecture, roadmap, criteria, phase reports, runbook, final matrix/report, portfolio set | 114/114 doc checks |

---

## 3. Partially verified and unverified capabilities

These items are deliberately separated from the verified workflow:

1. **Full production profile was not run live end to end.** The Phase 13 live
   runs used `--fast` (MLP `[32, 16]`, 24 epochs; 3-fold CV; 40-row SHAP sample).
   The committed benchmark artifact used the full profile (MLP `[64, 32]`,
   150-epoch limit; 5-fold CV; 200-row LR/RF and 60-row MLP SHAP samples). Both are real, but the fast live
   rerun is not a production-hyperparameter claim.
2. **No committed browser end-to-end suite.** The dashboard was exercised with
   Streamlit `AppTest`, HTTP checks and manual browser screenshots. Upload,
   download and multi-page navigation are not guarded by a CI browser test.
3. **Some dashboard action paths were tested at the service boundary, not each
   in a live browser.** Training, discovery/acquisition and the SHAP checkbox
   have service tests; the individual-score and batch-upload paths were executed
   against the running app in Phase 11.
4. **Live acquisition is date-dependent.** It passed on 2026-10-23; upstream
   URLs, licences and availability can change. Cached artifacts are reproducible.
5. **One primary dataset.** All real explanations and held-out results use OpenML
   `46915`. Cross-dataset robustness is not measured.
6. **Global SHAP rankings are sample estimates.** One seed and bounded samples;
   no confidence intervals or seed-stability study.
7. **Full-population local explanations come from a separate risk run.** The
   Phase 13 fast pipeline stores drivers for 10 customers. The committed
   `20261022T155236Z-...-risk-final-drivers` artifact contains 1,000 per-row
   explanations for the 1,000-row risk population; probabilities match the risk
   parquet within `4.44e-16`.
8. **MLP float identity is not guaranteed.** Default training does not enable
   TensorFlow op determinism. Deterministic mode is tested, but cross-machine
   equality is not promised.
9. **No universal JSON-Schema registry.** Validation is distributed by phase plus
   principal cross-phase assertions (D-09).
10. **The lock is platform-specific.** `requirements.lock.txt` was generated on
    Windows/CPython 3.13 and has no hash pins; other platforms should use
    `requirements.txt` or regenerate the lock.
11. **No production infrastructure.** No authentication, multi-tenancy,
    database, cloud deployment, monitoring, drift detection or A/B test exists.
12. **D-07 legacy manifests.** The code now writes portable paths and resolves
    relative paths; older absolute-path manifests remain readable only on the
    machine that created them.

---

## 4. Actual model-training and benchmark results

### 4.1 Post-fix live end-to-end run (fast profile)

| Field | Value |
| --- | --- |
| Run ID | `phase13-final-live2` |
| Started / finished | 2026-10-23T01:32:55Z / 01:34:28Z (93 s in manifest; 105 s wall) |
| Dataset | OpenML `46915`, 5,000 rows, 20 columns, 14.14% churn |
| Dataset SHA-256 | `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6` |
| Split | 3,000 train / 1,000 validation / 1,000 test, seed 42 |
| Features | 27 transformed |
| Class weights | `{0: 0.5823, 1: 3.5377}` |
| MLP (fast) | `[32, 16]`, dropout `[0.2, 0.1]`, 24 epochs, best epoch 24, no early stop |
| Selected model | Random Forest, validation F1 **0.7985**, threshold **0.45** |
| Selection rule | validation F1 only; `test_used_for_selection: false` |

Test-set results (1,000 rows, 141 positives):

| Family | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.64 | 0.8770 | 0.5459 | 0.7589 | 0.6350 | 0.8741 | 0.5717 | 0.1460 |
| **Random Forest (selected)** | **0.45** | **0.9650** | **0.9015** | **0.8440** | **0.8718** | **0.9383** | **0.8973** | **0.0390** |
| MLP (fast) | 0.73 | 0.9220 | 0.7520 | 0.6667 | 0.7068 | 0.9221 | 0.7637 | 0.1091 |

The MLP's fast-profile F1 (0.7068) differs from its full-profile F1 (0.7208);
the selected Random Forest's test results are identical across the two Phase 13
live runs and the committed benchmark.

### 4.2 Full benchmark profile (committed reference)

Run `20261021T145336Z-openml_46915-bench-04595d7b`, same dataset and split,
test scored once after validation-only selection:

| Family | Validation F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5905 | 0.6350 | 0.8741 | 0.5717 | 0.1460 |
| **Random Forest (selected)** | **0.7985** | **0.8718** | **0.9383** | **0.8973** | **0.0390** |
| MLP (Keras) | 0.6937 | 0.7208 | 0.9266 | 0.7801 | 0.0944 |

Five-fold CV F1: Random Forest **0.818 ± 0.023**, MLP **0.713 ± 0.038**,
Logistic Regression **0.552 ± 0.025** at the 0.5 reference threshold.

Evidence: [benchmark metrics](evidence/benchmark/metrics.json),
[benchmark report](evidence/benchmark/benchmark.md),
[live benchmark metrics](evidence/phase13-live2/benchmark_metrics.json), and the
full manifest [run.json](evidence/phase13-live2/run.json).

### 4.3 Reproducibility evidence

Two live runs (`phase13-final-live`, 66 s, and `phase13-final-live2`, 105 s) and
the committed benchmark produced identical selected-model metrics, tier counts
and simulation projections. During final verification, two otherwise identical
pipeline runs differed by one ULP in Random Forest `log_loss`; D-12 was fixed by
making serial execution the default for the forest. The existing determinism
test and a new focused regression test pass. MLP float identity remains
explicitly out of scope unless deterministic mode is enabled.

---

## 5. Dataset source and provenance

| Field | Value |
| --- | --- |
| Dataset | OpenML `46915`, title `churn` |
| Provider / landing | OpenML / `https://www.openml.org/d/46915` |
| Download | `https://www.openml.org/data/v1/download/22125226` |
| Licence | MIT (verified by discovery metadata) |
| Retrieved | 2026-10-23T00:46:40Z (live run 1); live run 2 re-acquired live |
| File | `22125226.arff`, **475,249 bytes** |
| SHA-256 | `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6` |
| Shape | 5,000 rows × 20 columns; 707 positives (14.14%) |
| Target | `CustomerChurned`, mapped `No: 0`, `Yes: 1`; positive label `Yes` |
| Validation status | `pass_with_warnings`, `can_preprocess: true` |

The source was found through live discovery; no dataset or URL was supplied by
the user. Provenance, licence and digest are in
[the validation report](evidence/phase13-live2/validation_report.json).
Acquisition wrote a manifest, a content-addressed cache entry and a verified raw
file; the fixture command separately proves the offline path.

**No synthetic fallback was used for the reported live results.**

---

## 6. Explainability and risk-assessment verification

### SHAP

- **Selected model:** Random Forest, TreeSHAP in probability space.
- **Other models:** Logistic Regression (LinearSHAP, log-odds space) and MLP
  (DeepSHAP, probability space); output spaces are never mixed.
- **Global live ranking (fast run):** `international_plan`, `total_day_charge`,
  `number_customer_service_calls`, `total_day_minutes`,
  `support_calls_per_usage_hour`.
- **Additivity:** maximum error `8.06e-15` on the fast live global sample.
- **Privacy:** explanations store transformed values and contributions; raw
  records and identifiers are not included.
- **Local matrix:** 1,000 per-row JSON files for the 1,000-customer
  `20261022T155236Z...risk-final` run; 1,000/1,000 predictions match the risk
  parquet within `4.44e-16`.
- **Not causal:** every explanation carries that warning in the UI and report.

### Risk intelligence

- 1,000 validation customers scored, 0 rejected, threshold **0.45**.
- Tiers: low **858**, medium **31**, high **40**, critical **71**;
  predictions: churn **121**, retained **879**.
- Tier boundaries 0.3 / 0.5 / 0.7 are business categories and are independent of
  the classification threshold.
- Ranking is deterministic by probability then record reference; counts reconcile.
- Batch CSV validation rejects malformed rows; exports omit raw identifiers and
  neutralise spreadsheet formulas.
- The final fast pipeline records **10** available driver entries and **990**
  unavailable, rather than implying every customer has a stored explanation.
  The separate 1,000-row matrix is the full-population evidence.

---

## 7. Retention-simulation verification

Live scenario `phase13-final-live2-sim`:

| Input / output | Value |
| --- | --- |
| Population | 1,000 assessed customers |
| Budget | 1,000 value units |
| Cost per contact | 5 |
| Assumed effectiveness | 0.30 |
| Selected / contacted | 200 |
| Spend | 1,000 |
| Expected churn mass | 109.444617 |
| Expected prevented churn | **32.833385** |
| Random-contact baseline | 9.138484 |
| Targeting lift | **3.5929×** |
| Monthly value | not supplied |
| Saved value / net benefit / ROI / break-even | `null` with explicit reasons |
| Projection marker | `projection: true`, `observed_retention_result: false` |

Independent arithmetic: `109.444617 × 0.30 = 32.833385`. The contact count is
consistent with `floor(1000 / 5) = 200`. The result is a hypothetical projection;
there is no measured uplift, causal model or observed retention outcome.

Tests cover zero budgets, zero-cost interventions, undefined cost-efficiency,
comparison consistency, sensitivity calculations and target-lift edge cases.

---

## 8. Dashboard launch and workflow results

Start command:

```bash
CHURNWISE_DATA_DIR=.phase13_final2/data \
CHURNWISE_ARTIFACT_DIR=.phase13_final2/artifacts \
CHURNWISE_REPORT_DIR=.phase13_final2/reports \
python -m streamlit run churnwise/dashboard/app.py
```

Verified:

- all **11 pages** rendered without exceptions against real artifacts;
- first real-artifact render **1.937 s**, below the 5 s budget;
- HTTP **200** in under 2 s on the local server;
- sidebar navigation, risk filters, customer detail, simulator controls and
  reliability inventory work on persisted runs;
- page renders do **not** train, download, score or compute SHAP; explicit
  buttons invoke services;
- cache keys include path, `st_mtime_ns` and size; refresh clears the cache;
- missing-artifact states were checked page by page;
- one individual record was genuinely scored through the running app
  (`0.0611`, low, retained);
- a six-row batch CSV was uploaded and assessed (`6 of 6` scored);
- no automated browser E2E suite is committed (documented limitation).

Evidence: [PHASE11_REPORT.md](PHASE11_REPORT.md),
[screenshots](screenshots/), and the 42-test focused suite.

---

## 9. Test commands and exact outcomes

### Final audit environment (`.venv`, Windows, CPython 3.13)

| Command | Exit | Result |
| --- | ---: | --- |
| `python -m pytest -q --disable-warnings` | 0 | **624 passed, 7 skipped** in 268.70 s |
| `python -m pytest -q -m live --run-live` | 0 | **7 passed** |
| Dashboard-focused suite | 0 | **42 passed** in 24.61 s |
| `python scripts/verify_phase1_docs.py` | 0 | **114 checks passed, 0 failed** |
| Fixture demo run | 0 | 7/7 stages, 160 rows, 53 s |
| Live run 1 (`phase13-final-live`) | 0 | 7/7 stages, 66 s |
| Live run 2 (`phase13-final-live2`) | 0 | 7/7 stages, 105 s wall / 93 s manifest |

### Failure found and corrected during this audit

The first full-suite run was **1 failed, 622 passed, 7 skipped**. The failure was
not suppressed: two deterministic pipeline runs produced Random Forest
`log_loss` values differing by one ULP. Root cause: `n_jobs=-1` parallel tree
averaging. Fix: serial Random Forest default (`n_jobs=1`) for stable training and
prediction; explicit `-1` remains available as a speed trade-off. Regression
coverage: the existing cross-run determinism test plus
`test_random_forest_default_is_serial_for_bitwise_stable_predictions`. The full
suite was then rerun: **624 passed**.

No tests were deleted, skipped or weakened. The seven skips are the documented
opt-in live-network tests. The repository has no ruff/mypy/flake8 configuration,
so no separate lint or static-type command was available to run; imports and
runtime behavior are exercised by the suite.

### Clean environment

`.venv_clean` was created only from `requirements.lock.txt`; the post-fix full
suite result is recorded here after execution:

| Command | Result |
| --- | --- |
| install from lock | exit 0; dependency set importable |
| `.venv_clean/Scripts/python -m pytest -q` | **624 passed, 7 skipped** in 326.77 s (exit 0) |

The clean environment was created from CPython 3.13 and installed only from the
lock. Its post-fix result matches the audit environment, so the suite no longer
depends on the primary venv's package history. See
[Phase 13 test evidence](evidence/phase13-tests/README.md) for the preserved
suite log and summary.

---

## 10. Resolved and unresolved defects

| ID | Status | Outcome |
| --- | --- | --- |
| D-01 | Fixed | One-command orchestration with manifest |
| D-02 | Fixed | Managed offline fixture path with provenance |
| D-03 | Fixed | CLI exit-code shadowing |
| D-04 | Fixed | Safe resume with config digest and artifact checks |
| D-05 | Fixed | Probability-mass labeling corrected |
| D-06 | Fixed | Stage summaries read correct metadata keys |
| D-07 | Fixed | Portable risk paths; relative-path resolution |
| D-08 | Open, documented | MLP floats not bitwise reproducible by default |
| D-09 | Mitigated | No universal JSON-Schema registry |
| D-10 | Fixed | EDA profile timestamps |
| D-11 | Fixed | Project-owned fixture licence vocabulary |
| D-12 | Fixed | Random Forest one-ULP drift; serial default |

No critical or high-severity defect remains open. D-08 and D-09 are visible in
the UI/docs, do not break the supported workflow, and have explicit next steps.

---

## 11. Environment limitations

- Verified on Windows 11, CPython 3.13.0, CPU-only TensorFlow; no CUDA or GPU
  validation.
- The exact lock is Windows/CPython 3.13-specific and unpinned by hash.
- Live tests depend on external OpenML/UCI/GitHub availability and network.
- Browser evidence is manual/Playwright-assisted rather than a committed E2E
  suite.
- The dashboard is a local single-process application; training blocks the
  Streamlit run and there is no background job queue.
- Cleanup has no cross-process lock and assumes one user/process.
- The full production training/SHAP profile was not executed end to end in
  Phase 13; only the fast profile was.
- No cloud deployment, authentication, database, monitoring or CI pipeline is
  included.

---

## 12. Security, privacy and responsible-use limitations

**Verified safeguards**

- Downloads are allowlisted to configured HTTPS sources; redirects, content type,
  size, archive traversal, symlinks, executables and parser failures are checked.
- Dataset and model artifacts are digest-verified; corrupted/stale artifacts are
  rejected rather than silently used.
- Risk exports omit raw identifiers and neutralise spreadsheet formulas.
- Explanation artifacts store transformed values and contributions, not raw
  customer records.
- No secrets, credentials or private customer data are required or committed.
- The dashboard does not execute arbitrary user code or shell commands.

**Responsible-use limits**

- Churn probabilities are model estimates, not guaranteed outcomes.
- SHAP values are model attributions, **not causal effects**; acting on a feature
  does not guarantee a change in churn.
- Campaign results are hypothetical projections based on an assumed
  effectiveness; they are not measured uplift, revenue or ROI.
- Tier boundaries are business categories, not calibrated decision thresholds.
- The model has not been tested for demographic fairness, drift, adversarial
  inputs or regulatory suitability.
- Any real deployment needs human review, calibration, bias assessment,
  monitoring and a privacy/legal review.

---

## 13. Final acceptance-matrix summary

| Phase | Status |
| --- | --- |
| 1 Foundation | VERIFIED |
| 2 Discovery | VERIFIED |
| 3 Acquisition | VERIFIED |
| 4 Validation/EDA | VERIFIED |
| 5 Preprocessing | VERIFIED |
| 6 MLP | VERIFIED |
| 7 Benchmark | VERIFIED |
| 8 Explainability | VERIFIED |
| 9 Risk intelligence | VERIFIED |
| 10 Simulation | VERIFIED |
| 11 Dashboard | VERIFIED |
| 12 Integration/reliability | VERIFIED |
| 13 Final verification | VERIFIED |

Full evidence per criterion: [FINAL_ACCEPTANCE_MATRIX.md](FINAL_ACCEPTANCE_MATRIX.md).

---

## 14. Recommended next steps, ordered by importance

1. **Run the full profile end to end** (150 epochs, 5-fold CV, larger SHAP sample)
   on the reference machine and record timing and memory; do not replace the
   fast-profile evidence.
2. **Add a browser E2E suite** for navigation, individual scoring, batch upload,
   downloads and training actions, using a disposable artifact tree.
3. **Add a universal artifact-schema registry** (or generated schema checks) and
   keep the existing cross-phase identity assertions.
4. **Add CI** that installs both dependency manifests, runs the offline suite,
   runs opt-in live tests on a schedule, and checks documentation claims.
5. **Evaluate calibration and fairness** before any customer-facing use; add
   reliability curves, subgroup metrics and drift monitoring.
6. **Test a second dataset** with different categorical levels, missingness and
   class balance to validate the capability-driven interfaces.
7. **Measure seed stability** of global SHAP rankings and report uncertainty.
8. **Add a cross-platform lock strategy** (per-platform locks or hash-pinned
   universal constraints).
9. **Add process locking and retention policy** for cleanup in multi-user or
   long-running environments.
10. **If a real deployment is intended**, add authentication, storage, audit
    logging, monitoring, rollback and a human decision workflow.

---

## Appendix A — Commands used

```bash
python -m pytest -q
python -m pytest -q -m live --run-live
python scripts/verify_phase1_docs.py

python -m churnwise.cli run \
  --dataset tests/fixtures/mini_churn.csv --fast --seed 13 \
  --run-id demo-fixture --data-dir .phase13_demo/data \
  --artifact-dir .phase13_demo/artifacts --report-dir .phase13_demo/reports

python -m churnwise.cli run --fast --seed 42 \
  --run-id phase13-final-live2 \
  --data-dir .phase13_final2/data \
  --artifact-dir .phase13_final2/artifacts \
  --report-dir .phase13_final2/reports
```

## Appendix B — Evidence map

- [Evidence directory guide](evidence/README.md)
- [Phase 13 test evidence](evidence/phase13-tests/README.md)
- [Phase 13 post-fix live evidence](evidence/phase13-live2/README.md)
- [Phase 13 pre-fix live evidence](evidence/phase13-live/README.md)
- [Committed benchmark metrics](evidence/benchmark/metrics.json)
- [SHAP global ranking](evidence/explainability/global_random_forest.json)
- [1,000-row SHAP matrix summary](evidence/explainability/matrix_1000_summary.json)
- [Defect register](INTEGRATION_DEFECTS.md)
- [Screenshots](screenshots/)
