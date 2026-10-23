# ChurnWise AI — Acceptance Criteria

| Field | Value |
| --- | --- |
| Version | 1.0 (Phase 1 baseline) |
| Date | 2026-10-20 |
| Applies to | Phases 1–13 |
| Companion documents | [README.md](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [PHASE1_REPORT.md](PHASE1_REPORT.md) |

---

## 1. How acceptance works

A phase is **accepted** only when every completion criterion for that phase is checked *and* backed by a named artifact or command output. A criterion is never satisfied by assertion, by a plausible-looking number, or by "it worked when I ran it earlier".

Three rules govern every phase:

1. **Evidence rule.** Each checked criterion references a file path, a test ID with its result, or a command with its output. In the final verification (Phase 13) each row carries that reference.
2. **Honesty rule.** No fabricated metric, no predetermined performance target, no synthetic rows presented as real data, no success reported while a gate failed. Missing evidence is reported as a failure, not omitted.
3. **Preservation rule.** Scope may be deferred with a documented risk, never removed. A deferred criterion stays visible as `deferred — <risk>`.

### 1.1 Global Definition of Done

These apply to all 13 phases and are re-checked at Phase 13.

| ID | Global criterion |
| --- | --- |
| GD1 | Works on the audited machine (Windows 11, Python 3.13) from a clean project-local `.venv` created from the pinned `requirements.txt` |
| GD2 | No secret, token or credential appears in source, docs, logs or artifacts |
| GD3 | Every generated artifact records `run_id`, `dataset_sha256`, `seed` and `code_version` |
| GD4 | `pytest -q` exits 0 with no skipped acceptance-critical test; skips require a documented reason |
| GD5 | All documentation links resolve, with no unresolved placeholder markers or unmeasured results |
| GD6 | No infrastructure beyond the single local process + Streamlit UI (no database, backend service, auth, cloud) |
| GD7 | Downloaded content is never executed; only allowlisted HTTPS hosts are contacted |
| GD8 | Any failure path produces an error code plus a remediation message; nothing fails silently |
| GD9 | Repeat runs with the same seed and dataset produce identical split fingerprints and metrics within 1e-6 |
| GD10 | The dashboard never displays a number that is not read from an artifact |

---

## 2. Phase 1 — Repository, Architecture and Requirements

**Objectives and scope.** Audit the workspace and environment; define module boundaries, data contracts, autonomous-data requirements, ML/evaluation requirements and a 13-phase plan with acceptance criteria. Explicitly out of scope: application code, dependency installation, any model training.

**Inputs and outputs.** Inputs: working directory, environment, source reachability. Outputs: architecture, roadmap, acceptance criteria, phase report, pinned requirements, doc-verification script.

**Expected artifacts.**

| Artifact | Purpose |
| --- | --- |
| `docs/ARCHITECTURE.md` | Modules, contracts, data flow, ML/autonomy design, decisions |
| `docs/ROADMAP.md` | 13 phases, dependencies, five-day schedule, risks |
| `docs/ACCEPTANCE_CRITERIA.md` | This document |
| `docs/PHASE1_REPORT.md` | Audit evidence, findings, verification, open decisions |
| `requirements.txt` | Pinned dependency manifest (not installed in Phase 1) |
| `.gitignore` | Keeps data, artifacts, venv and caches out of version control |
| `scripts/verify_phase1_docs.py` | Automated consistency check for the documentation set |

**Dependencies.** None.

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T1.1 | All 13 phase headings present in roadmap and acceptance docs | `python scripts/verify_phase1_docs.py` |
| T1.2 | Each acceptance phase has all seven required subsections | Same script, structural parse |
| T1.3 | Phase dependency graph covers every phase and every declared dependency | Script parses the `Dependencies` line per phase |
| T1.4 | Autonomy stated as mandatory (discovery + acquisition, no user dataset) | Script keyword assertions in architecture + roadmap |
| T1.5 | Mermaid diagram present and syntactically fenced | Script checks a `mermaid` block in both architecture and roadmap |
| T1.6 | No placeholder tokens (`TODO`, `TBD`, `lorem`, `FIXME`) | Script scans prose outside inline code in all docs |
| T1.7 | Internal doc links resolve to existing files/anchors | Script resolves relative links |

**Completion criteria.**

- [ ] Workspace audit recorded with actual command output (not paraphrased).
- [ ] Environment table states each required package as present or missing with version evidence.
- [ ] All 13 phases present, numbered 1–13, none merged or dropped.
- [ ] Each phase documents objective/scope, inputs/outputs, artifacts, dependencies, testing, completion criteria, risks.
- [ ] Autonomous discovery and acquisition are stated as **mandatory** requirements with failure behaviour and an explicit ban on synthetic fallback.
- [ ] MLP specification covers ReLU, sigmoid, binary cross-entropy, Adam, backpropagation, regularisation and early stopping.
- [ ] Fair-comparison protocol with Logistic Regression and Random Forest is defined, including leakage controls.
- [ ] Metric set explicitly includes accuracy, precision, recall, F1, ROC-AUC and confusion matrices, measured on a held-out test set.
- [ ] No performance number is stated as an expectation or target anywhere in the docs.
- [ ] Five-day plan fits all 13 phases and names the tight constraints.
- [ ] `python scripts/verify_phase1_docs.py` exits 0.

**Risks and unresolved decisions.** No Git repository exists (recommend `git init`; not performed). Primary dataset not yet chosen. `unverified` licence policy pending. TF/SHAP not installed, only confirmed installable.

---

## 3. Phase 2 — Autonomous Dataset Discovery

**Objectives and scope.** Discover suitable public churn datasets with no user-supplied file or URL. In scope: source registry, OpenML/UCI/GitHub adapters, multi-term query plan, suitability gates G1–G9, deterministic scoring and ranking, rejection logging, `discover` CLI. Out of scope: payload download (Phase 3), schema validation of parsed data (Phase 4).

**Inputs and outputs.** Inputs: registry, discovery criteria (`terms`, formats, row bounds, `min_predictors`, numeric/categorical/customer-feature requirements, target rules, licence policy). Outputs: `reports/acquisition/candidates.json` (ranked, with gate evidence) and `reports/acquisition/rejections.json` (gate-by-gate reasons).

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/discovery/registry.py` | Declarative source entries with provider, endpoint, auth, licence posture, trust tier |
| `churnwise/discovery/adapters/openml.py`, `uci.py`, `github.py` | `search(query) -> SourceResult` with normalized candidates, notes and error codes |
| `churnwise/discovery/suitability.py` | Gates G1–G9, scoring formula, tie-breaking |
| `churnwise/discovery/validation.py`, `selection.py`, `dedupe.py`, `coordinator.py` | Metadata rejection, eligible selection, duplicate collapse/ranking, orchestration |
| `reports/acquisition/candidates.json` | Ranked candidates with per-gate results and score components |
| `tests/test_gates.py`, `test_dedupe_ranking.py`, `test_validation_selection.py` | Gate, ranking, validation and selection tests |

**Dependencies.** Phase 1 (contracts, registry and gate design).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T2.1 | Search returns ≥1 candidate for the seed query set | Live run, output count recorded |
| T2.2 | Every gate G1–G9 is exercised; gates that can fail have pass and fail cases (G4/G8 also unknown). G9 is a metadata-access checkpoint that cannot fail once a candidate exists | `pytest tests/test_gates.py` |
| T2.3 | An incompatible candidate (no binary target / non-tabular) is rejected with a reason | Fixture candidate in test + recorded real rejection |
| T2.4 | Ranking is deterministic across runs | Run twice, compare order, scores and gate outcomes (report timestamps are expected to differ) |
| T2.5 | Score components sum correctly and lie in [0,1] | Unit test with hand-computed example |
| T2.6 | Adapters degrade gracefully on HTTP error/timeout | Mocked 500/timeout tests, error code asserted |

**Completion criteria.**

- [ ] A first-time run finds at least one eligible dataset without any user-supplied input.
- [ ] All nine gates are implemented, individually evidenced, and persisted per candidate.
- [ ] Rejected candidates appear in the rejection log with reasons — the log is never empty on a real run.
- [ ] Ranking is reproducible: two runs produce identical ordering.
- [ ] Each candidate records provider, landing URL, download URL, licence field and trust tier.
- [ ] No network payload is written to disk in this phase.
- [ ] Discovery never claims success without a candidate that passed all gates.

**Risks and unresolved decisions.** OpenML's name-based search is narrow (observed: `telco` → 0 hits, `churn` → 3). Mitigation: multiple query terms, explicit seed IDs, metadata lookups. Whether Kaggle is supported at all depends on the licence policy decision; it must never be required.

---

## 4. Phase 3 — Autonomous Data Acquisition and Provenance

**Objectives and scope.** Download the selected dataset safely, verify integrity, cache it content-addressed, and record complete provenance; fall back across candidates and fail with actionable errors. In scope: allowlisted HTTPS fetcher, size caps, redirect re-validation, content sniffing, parser probe, sha256, archive safety, cache, manifest, attempt log, offline mode. Out of scope: quality/schema validation (Phase 4).

**Inputs and outputs.** Inputs: ranked candidates from Phase 2, cache state, licence policy. Outputs: raw file, cache blob, `DatasetManifest`, attempt log.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/acquisition/downloader.py` | Streamed HTTPS fetch with allowlist, caps, retries |
| `churnwise/acquisition/integrity.py` | sha256, archive inspection, content/parser checks |
| `churnwise/acquisition/store.py` | Content-addressed store with pointer manifests, cache lookup and invalidation |
| `churnwise/acquisition/provenance.py` | Manifest read/write, URL sanitization, attempt log and fallback lineage |
| `churnwise/acquisition/coordinator.py`, `inputs.py`, `reporting.py` | Fallback orchestration, ranked-report input, attempts/summary |
| `data/raw/<provider>/<dataset>/<version>/<file>` | Human-browsable raw copy |
| `data/manifests/<dataset_id>.json` | Provenance record |
| `reports/acquisition/attempts.json` | Every attempt with error code and duration |
| `tests/test_acquisition_*.py`, `tests/test_live_acquisition.py` | Mocked and live acquisition coverage |

**Dependencies.** Phase 2 (ranked candidate, gates); Phase 1 (manifest contract).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T3.1 | Successful acquisition writes file + manifest with UTC timestamp and matching sha256 | Run, recompute hash, compare |
| T3.2 | Tampered payload is rejected (`CW-ACQ-002`) | Corrupt the byte stream in a test double |
| T3.3 | HTML error page served as HTTP 200 is rejected (`CW-ACQ-003`) | Fixture server returns HTML |
| T3.4 | Archive with path traversal or executable entry is rejected | Crafted zip fixture |
| T3.5 | Oversized download aborts and leaves no partial file | Capped-response fixture |
| T3.6 | Warm-cache offline run succeeds; cold-cache offline run fails with `CW-ACQ-005` and remediation | `CHURNWISE_OFFLINE=1` both ways |
| T3.7 | Fallback chain proceeds to the next candidate and records `fallback_of` | Primary forced to fail |
| T3.8 | No synthetic or generated data is created anywhere | Assert `data/` contains only fetched bytes; grep for generator calls outside tests |
| T3.9 | A real source artifact is acquired, rehashed and reused offline | Live test + CLI, actual SHA-256 and cache hit recorded |

**Completion criteria.**

- [ ] A verified dataset and manifest exist, produced without user-supplied input.
- [ ] Manifest contains source URL, retrieval timestamp, sha256, byte size, content type when supplied (otherwise explicitly null), licence status and fallback lineage.
- [ ] Every rejected attempt is recorded with one of the documented error codes.
- [ ] Cache hit avoids the network (verified by running with network blocked).
- [ ] Downloaded files are never executed; archive entries are inspected, not trusted.
- [ ] Exhausted fallbacks stop the run with a remediation block listing every attempted source and its failure reason.
- [ ] Zero occurrences of synthetic-data generation in non-test code.

**Risks and unresolved decisions.** Source availability and licence ambiguity for the IBM mirror; `unverified` licence default still open. File-hash values observed in Phase 1 are evidence, not pins — the acquisition layer records whatever hash it stores and must not hardcode a Phase 1 observation.

---

## 5. Phase 4 — Dataset Validation and Data Intelligence

**Objectives and scope.** Prove the dataset is usable and understand it before modelling. In scope: schema checks, target detection and normalisation to `1 = churned`, role mapping, missingness/duplicate/constant/cardinality analysis, ID-like detection, leakage-risk flags, class balance, capability matrix, EDA profile and plots, JSON + Markdown reports, severity gate. Out of scope: imputation/scaling (Phase 5).

**Inputs and outputs.** Inputs: verified raw file + manifest. Outputs: `ValidationReport`, `DatasetProfile` (roles + capabilities), EDA figures, gate decision.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/validation/checks.py` | Checks V1–V12 with severity and evidence |
| `churnwise/validation/gate.py` | Severity rollup and blocking list |
| `churnwise/eda/profile.py` | Role mapping, capability matrix, statistics |
| `churnwise/eda/plots.py` | Distributions, correlation, target-conditional plots |
| `reports/validation/<dataset_id>.{json,md}` | Full report + gate |
| `reports/eda/<dataset_id>.{json,png}` | EDA outputs |
| `tests/fixtures/mini_churn.csv` | Small synthetic fixture, clearly labelled, test-only |
| `tests/unit/test_validation_checks.py` | Check-by-check tests |

**Dependencies.** Phase 3 (verified file + manifest).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T4.1 | Every check has a passing and a failing fixture | `pytest tests/unit/test_validation_checks.py` |
| T4.2 | Fatal issue (missing target / single class) blocks the gate | Fixture-driven |
| T4.3 | Target normalisation maps `Yes/No`, `0/1`, `True/False` to `1/0` correctly | Parametrised test |
| T4.4 | `customerID`-style unique keys are flagged and excluded from features | Asserted on real dataset profile |
| T4.5 | Gate blocks training when a fatal check fails (end-to-end) | `pytest tests/integration/test_validation_gate.py` |
| T4.6 | Report values match a pandas recomputation | Cross-check script |

**Completion criteria.**

- [ ] Validation report exists for the acquired dataset with every check individually evidenced.
- [ ] Target identified, normalised and its positive rate reported from the actual data.
- [ ] Role mapping assigns target, id (if any), numeric and categorical columns explicitly.
- [ ] IDs, constants, high-cardinality and leakage-risk columns are flagged with reasons.
- [ ] Capability matrix declares which derived features Phase 5 may build.
- [ ] A fatal check demonstrably prevents training.
- [ ] EDA figures and profile are generated from the real dataset, not a fixture.

**Risks and unresolved decisions.** Ambiguous target names across sources; ARFF categorical formatting; imbalance near the 2%/50% gate boundary; `TotalCharges`-style blank strings.

---

## 6. Phase 5 — Preprocessing and Feature Engineering

**Objectives and scope.** Build one leakage-safe, reproducible pipeline shared by all models. In scope: stratified splits with fingerprint, role-driven ColumnTransformer, imputation, scaling, one-hot encoding with unknown handling, capability-guarded derived features, persistence of fitted artifacts, imbalance strategy, leakage checklist enforcement. Out of scope: model fitting.

**Inputs and outputs.** Inputs: validated frame, dataset profile, seed. Outputs: `SplitSpec`, fitted `ColumnTransformer` + schema, transformed matrices for train/val/test.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/preprocessing/split.py` | Stratified 60/20/20 splits with fingerprint |
| `churnwise/preprocessing/pipeline.py` | ColumnTransformer assembly |
| `churnwise/preprocessing/features.py` | Capability-guarded derived features |
| `churnwise/preprocessing/artifacts.py` | Save/load preprocessor + schema |
| `artifacts/splits/<run_id>.json` | Split spec + fingerprint |
| `artifacts/preprocessors/<run_id>/{pipeline.joblib,schema.json}` | Fitted preprocessor + output schema |
| `tests/unit/test_preprocessing_leakage.py`, `tests/unit/test_split_determinism.py` | Leakage and determinism tests |

**Dependencies.** Phase 4 (validated frame + profile).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T5.1 | Identical seed → identical split fingerprint and row assignment | Two runs, compare JSON |
| T5.2 | Fitted scaler statistics come from train only | Inject an extreme outlier in test; assert statistics unchanged |
| T5.3 | Unseen categories at transform time do not raise | Encode a category absent from train |
| T5.4 | Missing values handled per documented rule (median / most-frequent / indicator) | Synthetic missingness fixture |
| T5.5 | Transformed matrix has stable column order and names across runs | Compare `schema.json` |
| T5.6 | SMOTE, when enabled, never touches validation/test rows | Assert sample counts per split |
| T5.7 | Leakage checklist L1–L8 each has an executable assertion | `pytest tests/unit/test_preprocessing_leakage.py` |

**Completion criteria.**

- [ ] Split proportions ≈ 60/20/20 and stratification preserved (class rate within 1 percentage point per split).
- [ ] Test set marked untouched at this stage (`test_touched: false`).
- [ ] Preprocessor artifact reloads and produces byte-identical transforms for the same input.
- [ ] `schema.json` records input columns, output feature names, dtypes and categories.
- [ ] Derived features appear only when their source columns exist, with a recorded reason otherwise.
- [ ] Imbalance strategy is recorded in config and applied identically to all consumers.
- [ ] No fitted statistic is computed from validation or test data.

**Risks and unresolved decisions.** High-cardinality categoricals may need `min_frequency`; derived features risk overreach on sparse schemas; class-weight vs SMOTE default is documented but its effect is only measurable in Phase 7.

---

## 7. Phase 6 — MLP Neural Network Development and Training

**Objectives and scope.** Implement and train the specified Keras MLP. In scope: architecture (Dense 64/32 ReLU, dropout, L2, sigmoid output), BCE loss, Adam, class weights, callbacks (early stopping, LR reduction, checkpointing), training loop, history capture, artifact persistence, seed control. Out of scope: baseline comparison (Phase 7), SHAP (Phase 8).

**Inputs and outputs.** Inputs: preprocessor + splits. Outputs: trained `.keras` model, metadata, history, reloadable predictions.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/models/mlp.py` | Model builder with documented hyperparameters |
| `churnwise/models/trainer.py` | Fit loop, callbacks, class weights, history |
| `churnwise/models/registry.py` | Model file + metadata registry |
| `artifacts/models/<run_id>/mlp/{model.keras,metadata.json,history.json}` | Trained model and evidence |
| `tests/unit/test_mlp_architecture.py` | Layer/activation/loss/optimiser assertions |
| `tests/integration/test_mlp_trains.py` | Fixture-size end-to-end training |

**Dependencies.** Phase 5 (preprocessor + splits); Phase 2/3 for the real dataset.

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T6.1 | Architecture matches spec exactly (2 hidden ReLU layers, sigmoid output, dropout, L2) | Introspect `model.layers`, assert units/activations/regularisers |
| T6.2 | Loss is binary cross-entropy and optimiser is Adam with documented LR | Assert compiled config |
| T6.3 | Early stopping restores best weights and reports `best_epoch` | Train with forced overfit fixture; assert restore |
| T6.4 | Class weights derived from the train split only | Compare to computed weights |
| T6.5 | Reloaded model reproduces predictions within 1e-6 | Save/load/compare |
| T6.6 | Training history captured per epoch | Assert `history.json` length equals epochs run |
| T6.7 | Training completes within the 3-minute budget on the audited machine | Timed run recorded |

**Completion criteria.**

- [ ] MLP trains end-to-end on the real preprocessed dataset and saves a loadable `.keras` artifact.
- [ ] `metadata.json` records every hyperparameter, seed, dataset hash and preprocessor hash.
- [ ] Early stopping configured and demonstrably active (epochs run < max epochs, or max-epoch completion recorded).
- [ ] Architecture assertions pass — ReLU hidden layers, sigmoid output, BCE, Adam.
- [ ] Model output is a probability in [0,1] for every input row.
- [ ] No claim about model quality is made in this phase.

**Risks and unresolved decisions.** TensorFlow installation on cp313 Windows; nondeterminism across runs; CPU training time. If TF cannot be installed, this is a blocker to escalate, not a reason to swap frameworks silently.

**Implementation status (2026-10-21, verified in [PHASE6_REPORT.md](PHASE6_REPORT.md)).** Architecture, BCE/Adam, callbacks, history, persistence and inference are implemented and tested. The reference run trained 43/150 epochs (best validation loss 0.365427 at epoch 31, 11.75 s); the saved model reloads with identical predictions and the test partition was not used for fitting or selection. T6.1–T6.5 are covered by the focused suite; T6.6 is covered by real `history.json`; T6.7 is evidenced by the 11.75 s CPU run (well inside the 3-minute budget). Model quality is deliberately unmeasured until Phase 7.

---

## 8. Phase 7 — Comparative Model Benchmarking and Evaluation

**Objectives and scope.** Evaluate MLP, Logistic Regression and Random Forest fairly and report real held-out metrics. In scope: baseline estimators, shared metric module, single test evaluation, 5-fold CV stability, confusion matrices, ROC/PR curves, benchmark table (JSON + Markdown), error slices, shipped-model decision. Out of scope: extended hyperparameter search, SHAP.

**Inputs and outputs.** Inputs: trained models, identical preprocessor and split fingerprint. Outputs: per-family metrics, benchmark report, chosen shipped model.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/models/baselines.py` | LR and RF with documented hyperparameters |
| `churnwise/evaluation/metrics.py` | Metric computation validated against sklearn |
| `churnwise/evaluation/benchmark.py` | Comparison table and figures |
| `artifacts/evaluation/<run_id>/metrics.json` | All families, all metrics, threshold, `n_test` |
| `artifacts/evaluation/<run_id>/benchmark.md` | Human-readable benchmark |
| `artifacts/evaluation/<run_id>/*.png` | ROC, PR, confusion matrices |
| `tests/unit/test_metrics.py` | Reference comparison |
| `tests/integration/test_benchmark_parity.py` | Same rows, same matrix for all families |

**Dependencies.** Phase 6 (MLP), Phase 5 (shared preprocessing + splits).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T7.1 | Metrics match sklearn reference within 1e-9 on fixed inputs | `pytest tests/unit/test_metrics.py` |
| T7.2 | Confusion matrix reconciles with per-row predictions (tn+fp+fn+tp = n_test) | Assertion |
| T7.3 | All three families evaluated on identical test indices and feature matrix | Compare split fingerprint + hashed matrix |
| T7.4 | Test set touched exactly once (no earlier reference) | Audit `test_touched` flag + code review of calls |
| T7.5 | Threshold used is recorded and applied consistently | Assert threshold in metrics vs predictions |
| T7.6 | ROC-AUC > 0.5 on test, else the run is flagged as a defect | Automated validity check |
| T7.7 | Benchmark regenerates deterministically | Re-run comparison from stored predictions |

**Implementation status (2026-10-21, verified in [PHASE7_REPORT.md](PHASE7_REPORT.md)).** All seven checks are implemented and evidenced by the reference run `20261021T145336Z-openml_46915-bench-04595d7b`: sklearn parity within 1e-9, confusion counts summing to 1000 test rows, identical train/validation/test matrix SHA-256 values in `metrics.json`, `test_evaluation_calls = 1`, frozen thresholds recorded per family, test ROC-AUC 0.874/0.938/0.927 (all above chance), and exact regeneration from saved predictions. LR, RF and the Phase 6 MLP were evaluated on the same 27-feature matrix and split fingerprint; 5-fold CV mean ± std is reported for each family. Random Forest was selected on validation F1 0.7985 (LR 0.5905, MLP 0.6937) and recorded in `run_manifest.json`; the MLP was not the best model, and the measured test results are unchanged by that outcome. Focused tests 57/57; full suite 390 passed with 7 pre-existing live-network skips. CV is fold-safe: each fold refits the Phase 5 plan, imputers, scaler and encoder on its own training rows after verifying the reconstructed base frame against the persisted matrices (`cross_validation.preprocessing.fold_safe = true`). Remaining limitation: one seeded 5-fold partition, so split-to-split variance is not quantified.

**Completion criteria.**

- [ ] Accuracy, precision, recall, F1, ROC-AUC and a 2×2 confusion matrix reported for **each** of the three families.
- [ ] Every number comes from `metrics.json` produced on the held-out test split, with `n_test` and threshold stated.
- [ ] Logistic Regression and Random Forest use the same transformed matrix and split as the MLP.
- [ ] CV mean ± std reported alongside the single test result.
- [ ] Curves and confusion-matrix figures exist for all families.
- [ ] The shipped model family is recorded in `run_manifest.json` with its selection rationale.
- [ ] No target metric was set in advance; results are reported as measured even if the MLP loses.

**Risks and unresolved decisions.** Accidental leakage through preprocessing; threshold confusion between validation and test; RF/MLP probability comparability (Brier score reported to surface calibration differences).

---

## 9. Phase 8 — Explainable AI and SHAP Integration

**Objectives and scope.** Explain global behaviour and individual predictions across families. In scope: explainer selection (Linear/Tree/Deep/Kernel), background sampling, additivity verification, one-hot grouping to source columns, global ranking and beeswarm, per-customer top-k signed drivers, precomputed JSON, runtime measurement. Out of scope: counterfactuals, causal claims.

**Inputs and outputs.** Inputs: trained models, preprocessor, feature names, reference rows. Outputs: `global.json`, `importance.png`, `summary.png`, `dependence_*.png`, `run_manifest.json`, `local/row_<position>.json`, `local/row_<position>.png`, additivity evidence.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/explain/explainers.py` | Per-family explainer construction and additivity validation |
| `churnwise/explain/features.py` | One-hot → source-column rollup and model-order labels |
| `churnwise/explain/engine.py` | Sampling, chunking, global/local explanation values |
| `churnwise/explain/coordinator.py` | Model loading, persistence, provenance and raw-record batches |
| `churnwise/explain/plots.py` | Importance, beeswarm, dependence and waterfall figures |
| `artifacts/explanations/<run_id>/{global.json,importance.png,summary.png,dependence_*.png,run_manifest.json,local/*}` | Explanations |
| `tests/unit/test_shap_{features,explainers,local,coordinator}.py` | Grouping, explainers, local semantics and end-to-end coordinator |
| `tests/integration/test_shap_mlp.py` | Real Keras MLP additivity and fallback |

**Dependencies.** Phase 6 and 7 (trained models), Phase 5 (feature names).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T8.1 | `sum(shap) + base_value ≈ model_output` within tolerance | Assertion per explained row |
| T8.2 | Grouped contributions sum to raw one-hot contributions | Unit test with a multi-level category |
| T8.3 | Top-k drivers are ordered by absolute contribution | Assertion |
| T8.4 | Explanation covers every customer present in the demo/prediction set | Set comparison of IDs |
| T8.5 | Explainer fallback works when the primary explainer fails | Forced-failure test |
| T8.6 | Generation completes within the 5-minute budget | Timed run |

**Completion criteria.**

- [ ] SHAP values produced for all three model families (or a documented, evidenced reason why one family uses a different explainer).
- [ ] Global feature ranking exists with mean |SHAP| values from the real model.
- [ ] Per-customer signed top-5 drivers exist and are human-readable (source column names, not one-hot names).
- [ ] Additivity check passes and its tolerance is documented.
- [ ] Explanations are cached artifacts; the dashboard does not recompute SHAP.
- [ ] No causal or "reason for churn" language is used for what are model attributions.

T8.4 is now **verified**: the Phase 9 risk run `20261022T155236Z-openml_46915-risk-final` was explained with `explain=all`, producing 1,000 local JSON explanations for all 1,000 scored customers (`artifacts/explanations/20261022T155236Z-openml_46915-risk-final-drivers/local/`); all 1,000 driver keys are present in the risk table. T8.1–T8.3 and T8.5–T8.6 are covered by the tests listed in [PHASE8_REPORT.md](PHASE8_REPORT.md); T8.6 measured 12.93 s (RF), 2.52 s (LR) and 7.19 s (MLP), within the 5-minute budget. Bounded sampling remains the default for routine global runs; explain-all is an explicit option.

**Verified evidence (Phase 8).** All three real families were explained against the OpenML `46915` Phase 5–7 artifacts: RF `20261022T131420-openml_46915-random_forest-xai-84eecf5d`, LR `20261022T131432-openml_46915-logistic_regression-xai-c160a8be`, MLP `20261022T131446-openml_46915-mlp-xai-4c391837`. Focused tests 42/42; full suite 432 passed / 7 live-network skips. Interface and limitations: [EXPLAINABILITY.md](EXPLAINABILITY.md), [PHASE8_REPORT.md](PHASE8_REPORT.md).

**Risks and unresolved decisions.** `DeepExplainer` compatibility with the Keras artifact was resolved in this environment (Deep SHAP succeeded); the bounded permutation fallback exists and is synthetically tested but was not needed for the real MLP. Permutation runtime remains a risk on larger data. SHAP/numpy version friction is pinned by `requirements.txt`. Phase 8 records the selected explainer per family in every manifest.

---

## 10. Phase 9 — Customer Churn Risk Intelligence

**Objectives and scope.** Convert probabilities into an actionable ranked risk list. In scope: individual and batch scoring, tiering from configurable thresholds, deterministic ranking/filtering/pagination, value-at-risk ordering where a revenue proxy exists, segment filters, driver attachment, CSV/JSON/Parquet export, record-level rejection reconciliation and artifact freshness. Out of scope: campaign economics (Phase 10), UI (Phase 11).

**Inputs and outputs.** Inputs: shipped model, preprocessor, raw records or persisted split, Phase 8 local explanations. Outputs: `churn_scores.parquet`, `risk_summary.json`, `tiers.csv`, `run_manifest.json`, optional `drivers.json` / `rejections.json`, and filtered exports.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/intelligence/service.py` | Individual/batch/population scoring, persistence, retrieval and exports |
| `churnwise/intelligence/model.py` | Read-only selected-model loading through the Phase 8 verifier |
| `churnwise/intelligence/drivers.py` | Phase 8 driver attachment and explanation provenance checks |
| `churnwise/intelligence/contracts.py` | Assessment, threshold, filter, table, run and export contracts |
| `churnwise/intelligence/ranking.py` | Tiers, ranks, filters, pagination and value-at-risk ordering |
| `churnwise/intelligence/reporting.py` | Evidence-only console summaries |
| `artifacts/predictions/<run_id>/{churn_scores.parquet,risk_summary.json,tiers.csv,run_manifest.json,drivers.json,rejections.json,exports/*}` | Ranked output, drivers, rejection log and exports |
| `tests/unit/test_ranking_tiers.py`, `tests/integration/test_scoring_counts.py`, `tests/test_risk_cli.py` | Tier/rank/count, scoring, failure and CLI tests |

**Dependencies.** Phase 7 (shipped model, threshold), Phase 8 (drivers), Phase 4 (revenue proxy availability).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T9.1 | Every scored row has probability in [0,1], a tier and a unique rank | Assertions |
| T9.2 | Tier counts sum to the scored population | Assertion |
| T9.3 | Boundary probabilities map to the documented tier | Parametrised boundary test |
| T9.4 | Value-at-risk ordering equals `p × value × horizon` ordering | Recompute and compare |
| T9.5 | Filters return exactly the matching rows | Predicate tests |
| T9.6 | Exports round-trip without losing rows or dtypes | Read back Parquet/CSV |
| T9.7 | Missing revenue proxy does not fabricate values | Null-handling test |
| T9.8 | Batch probabilities equal the same records scored in a population run | Real-model comparison (max abs difference) |
| T9.9 | CSV formula prefixes are neutralised and raw identifiers are excluded | Export round-trip assertions |
| T9.10 | Stale model/preprocessor artifacts are rejected; export paths cannot escape | Hash-mismatch and path-escape tests |
| T9.11 | Partial and all-rejected batches retain counts and rejection details | Rejection-summary assertions |

**Completion criteria.**

- [ ] Every input row receives exactly one score; counts reconcile (input rows = scored rows).
- [ ] Tiers derived from config thresholds, with counts and percentages reported.
- [ ] Ranking is deterministic and documented (tie-breaking rule stated).
- [ ] Top drivers attached to each customer from precomputed SHAP.
- [ ] Value-at-risk ordering present when a revenue proxy exists, absent and explained when it does not.
- [ ] Exports usable by the dashboard without recomputation.

**Implementation status (2026-10-22, verified in [PHASE9_REPORT.md](PHASE9_REPORT.md)).** Individual and batch assessment, configured tiers, ranking, filtering, pagination, optional value-at-risk, Phase 8 drivers and CSV/JSON/Parquet exports are implemented. The real OpenML `46915` validation population produced 1,000 scores in 16.96 s with 1,000 drivers: low 858, medium 31, high 40, critical 71; 121 predicted churn. A six-row real raw batch reproduced the stored validation probabilities exactly. A filtered export returned exactly 111 high/critical rows. Focused risk tests 58/58; full suite 490 passed / 7 live-network skips. A follow-up verification pass fixed blank-identifier fallback, malformed/incompatible artifact handling, tab/CR CSV neutralisation, JSON export schema/filter metadata, misleading missing-value warnings and value-proxy provenance; regression tests are recorded in [PHASE9_REPORT.md](PHASE9_REPORT.md#104-post-implementation-verification-and-fixes). All model and preprocessing artifacts were opened read-only; no model was retrained. Default tier boundaries are not claimed to be business-optimal and probabilities are not recalibrated.

**Risks and unresolved decisions.** Rank ties (resolved by documented reference/position tie-breaks); threshold drift between config and UI (mitigated by recording thresholds in every manifest and export); value-at-risk misleading when charges are a poor revenue proxy (only available when an explicit proxy is supplied, and not exercised on the real dataset).

---

## 11. Phase 10 — Retention Campaign Simulation

**Objectives and scope.** Let a user size a campaign and see projected saves, cost, ROI, break-even and sensitivity, with assumptions explicit. In scope: targeting selection, assumption schema/defaults, projection math, break-even guard, random-targeting baseline, sensitivity grid, disclaimer propagation, scenario persistence. Out of scope: uplift modelling, real dispatch.

**Inputs and outputs.** Inputs: ranked scores, scenario assumptions. Outputs: `result.json`, `sensitivity.csv`.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/simulation/campaign.py` | Scenario engine |
| `churnwise/simulation/assumptions.py` | Defaults, validation ranges, sensitivity grids |
| `artifacts/simulations/<scenario_id>/{result.json,sensitivity.csv}` | Scenario output |
| `tests/unit/test_campaign_math.py` | Hand-computed cases |
| `tests/integration/test_simulation_assumptions.py` | Assumption propagation |

**Dependencies.** Phase 9 (ranked scores), Phase 4 (revenue proxy availability).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T10.1 | Three hand-computed scenarios match engine output exactly | `pytest tests/unit/test_campaign_math.py` |
| T10.2 | Zero at-risk population yields no division error and a clear message | Edge-case test |
| T10.3 | Zero campaign cost handled without infinite ROI | Edge-case test |
| T10.4 | Assumptions embedded in `result.json` equal the inputs used | Assertion |
| T10.5 | Sensitivity grid spans documented ranges and is internally consistent | Recompute grid point |
| T10.6 | Every result carries a projection disclaimer | String assertion in JSON and UI |
| T10.7 | Targeting lift vs random contact computed correctly | Hand-computed comparison |

**Completion criteria.**

- [ ] Simulation inputs include monthly value, horizon, cost per contact and save rate, all visible and editable.
- [ ] Outputs include contacted count, expected saves, expected saved value, cost, net benefit, ROI and break-even save rate.
- [ ] Outputs are labelled as projections in JSON and in the UI.
- [ ] A sensitivity grid accompanies every scenario.
- [ ] Missing revenue proxy requires a user value instead of inventing one.
- [ ] Math verified against independent hand calculations.

**Verified evidence (Phase 10, report: [PHASE10_REPORT.md](PHASE10_REPORT.md)).** The JSON/CSV/Markdown/CLI half of the projection-label criterion is tested; the UI half stays open until Phase 11. `tests/unit/test_campaign_math.py` (47 passed) covers three hand-computed scenarios, all four strategies, threshold boundaries, tie-breaking, allocation, zero-cost/zero-budget, contact limits, per-customer costs, effectiveness 0/1, undefined metrics, configuration rejection, assumption embedding, comparison (including a value-unit mismatch warning) and sensitivity grids. `tests/integration/test_simulation_assumptions.py` (10 passed) runs against a genuinely fitted Phase 9 Random Forest, matches selected probabilities to the risk table, recomputes every sensitivity point, exercises the CLI and guards artifact identity. A real-data run on risk run `20261022T155236Z-openml_46915-risk-final` selected 142 of 1,000 assessed customers, spent 710 of 1,000 value units, projected 28.8787 prevented churns at assumed effectiveness 0.30 (ROI 1120.22%, break-even 2.46%) and persisted `artifacts/simulations/20261022T160000Z-real-baseline/` and `.../20261022T162000Z-real-comparison/`. A post-implementation audit additionally fixed tier-label derivation for three-tier runs, degenerate sensitivity grids, all-strategy budget warnings, chart projection labels, ASCII scenario ids, report escaping and empty sensitivity selections; see [PHASE10_REPORT.md](PHASE10_REPORT.md#12-post-implementation-verification-audit).

**Risks and unresolved decisions.** Projections mistaken for measured uplift; default assumptions may not match the user's business; revenue proxy quality (the reference dataset has none, so all financial projections require a supplied value).

---

## 12. Phase 11 — Interactive Intelligence Dashboard

**Objectives and scope.** Deliver the Streamlit experience over stored artifacts and existing phase services. In scope: navigation shell, cached artifact access, all required areas (Overview, Data discovery, Data quality, Model training, Model evaluation, Explainability, Risk intelligence, Assess records, Customer detail, Retention simulator, Reliability & limits), empty/error states, filters, exports, run-identity headers, and explicit button-gated actions for discovery, acquisition, training, record assessment and simulation. Ordinary rendering must not download, train, score or compute SHAP.

**Inputs and outputs.** Inputs: all prior artifacts and user-entered action parameters. Outputs: running app, action results and persisted artifacts written by the underlying services, AppTest coverage.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/dashboard/app.py` | Entry point + navigation |
| `churnwise/dashboard/data_access.py` | Cached loaders with availability checks |
| `churnwise/dashboard/actions.py` | Explicit discovery/acquisition/training/assessment/simulation actions |
| `churnwise/dashboard/pages/*` | Eleven pages |
| `churnwise/dashboard/components/*` | Layout, tables, charts |
| `tests/unit/test_dashboard_{analytics,assessment,data_access}.py` | Pure helpers, upload validation, artifact resolution |
| `tests/integration/test_dashboard_apptest.py` | AppTest coverage and service cross-checks |
| `docs/DASHBOARD.md` | Architecture and usage |
| `docs/PHASE11_REPORT.md` | Verified evidence and limitations |

**Dependencies.** Phase 2/3 (discovery/acquisition), Phase 4 (overview/quality), Phase 6 (training), Phase 7 (performance), Phase 8 (explainability), Phase 9 (risk/assessment), Phase 10 (simulation).

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T11.1 | App imports and starts headless without exceptions | `streamlit run` + HTTP probe |
| T11.2 | All eleven pages render with real artifacts | `AppTest` per page, assert no exception |
| T11.3 | Missing-artifact state is actionable, contains no fabricated numbers | Empty project tree, assert command per page |
| T11.4 | Tier filter changes the table row count consistently | AppTest interaction |
| T11.5 | Customer detail shows score matching the Parquet value | Cross-check |
| T11.6 | Simulator output matches the engine result for identical inputs | Direct Phase 10 cross-check |
| T11.7 | Cold load under 5 s | Timed first real-artifact `AppTest` render: 1.937 s |
| T11.8 | Training and discovery actions are not repeated by reruns | AppTest submit + rerun |
| T11.9 | Cache invalidates after an artifact change | Rewrite Parquet, assert new row count |
| T11.10 | Individual/batch assessment matches Phase 9 | Direct service cross-check |
| T11.11 | Upload validation handles size, parse, empty and row-limit failures | Unit tests |
| T11.12 | Validation report resolution ignores acquisition attempt logs | Unit test |

**Completion criteria.**

- [x] All eleven pages exist and render from real artifacts.
- [x] Every page header shows run ID, model family, dataset hash prefix and generated-at time.
- [x] Page rendering never downloads, trains, scores or computes SHAP; labelled actions delegate to the existing services.
- [x] Empty states tell the user the exact command to run.
- [x] Filters and exports work on real scored data.
- [x] `AppTest` covers every page and passes.
- [x] Screenshots captured as evidence.
- [x] Cold-load timing measured against the budget (1.937 s first render).

**Verified evidence (Phase 11, report: [PHASE11_REPORT.md](PHASE11_REPORT.md)).** Focused dashboard suite **42 passed**; full suite **624 passed / 7 live-network skips** after Phase 13. All eleven pages rendered against the real artifact tree; missing-artifact states were checked on every page. The app was started headless; one individual record was genuinely scored through the running app (`0.0611`, low, retained) and a six-row batch CSV was uploaded and assessed through the browser (`6 of 6` scored). Screenshots of all eleven pages, plus the batch result, are in [docs/screenshots](screenshots/). Phase 13 measured **1.937 s** for the first real-artifact render and **HTTP 200 in under 2 s**; live discovery/acquisition, live training, SHAP checkbox execution and an automated browser E2E suite remain outside the automated suite.

**Risks and unresolved decisions.** Streamlit API drift (1.66.1 verified); cache-key mistakes serving stale metrics; synchronous training blocking the process; large SHAP payloads; no automated browser E2E suite.

---

## 13. Phase 12 — Integration, Reliability and End-to-End Testing

**Objectives and scope.** Prove the pipeline runs reproducibly from an empty data directory and behaves honestly under failure. In scope: `run` orchestrator, offline end-to-end test on a committed fixture, contract tests, determinism test, failure-path tests, import-boundary test, performance measurement, cleanup tooling. Out of scope: new features.

**Inputs and outputs.** Inputs: all stages. Outputs: single reproducible command, test suite, performance report, runbook.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `churnwise/pipeline/**` | Stage graph, provenance, manifest, resume, output verification, cleanup |
| `tests/integration/test_pipeline_workflow.py` | Offline full run, cross-phase identity, resume, rejection, determinism, dashboard |
| `tests/unit/test_pipeline_contracts.py` | Error/config contracts and import boundaries |
| `tests/unit/test_pipeline_cleanup.py` | Dry-run, path confinement, deletion and manifest safety |
| `docs/PERFORMANCE.md` | Measured timings and memory vs budget |
| `docs/DEPENDENCIES.md` | Dependency and reproducibility audit |
| `requirements.lock.txt` | Exact audited Windows/CPython 3.13 dependency pins |
| `docs/RUNBOOK.md` | Operate, resume, recover, inspect and clean |

**Dependencies.** Phases 1–11.

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T12.1 | Full pipeline runs offline on the fixture and produces every artifact | `pytest tests/integration/test_pipeline_workflow.py` |
| T12.2 | Artifact contracts validate | Phase contract suites plus principal cross-phase identity assertions in the pipeline test |
| T12.3 | Determinism of reusable outputs | Split/model/risk/simulation tests; MLP floats are explicitly excluded |
| T12.4 | Failure paths emit documented error codes | Single-class rejection, invalid inputs, missing/corrupt manifest, changed resume, missing artifacts, bad hash/cold cache in acquisition suites |
| T12.5 | Import boundaries hold (dashboard/orchestrator direction) | `pytest tests/unit/test_pipeline_contracts.py` |
| T12.6 | Measured stage timings recorded against budget | `docs/PERFORMANCE.md` |
| T12.7 | `pytest -q` green from a clean `.venv` | Fresh `.venv` installed only from `requirements.lock.txt`; post-fix suite result recorded in the final report |

**Completion criteria.**

- [x] One documented command reproduces the full artifact set from fixture data (network disabled) and from live discovery into project-local empty directories.
- [x] Principal artifact contracts and cross-phase identity are tested; a universal JSON-Schema registry remains absent (D-09).
- [x] Reusable outputs are deterministic under the stated method: splits, baselines, scoring, ranking and simulation, plus MLP metrics with op determinism enabled. Default MLP float identity is explicitly excluded (D-08).
- [x] Covered failure paths yield `CW-RUN-*` codes and remediation; a rejected dataset leaves only diagnostic acquisition/validation artifacts.
- [x] Import boundaries are enforced by AST tests.
- [x] Measured performance is reported honestly against the budget; no measured live stage exceeded its budget.
- [x] The runbook covers operation, resume, failure diagnosis, inspection and cleanup.

**Verified evidence (Phase 12, report: [PHASE12_REPORT.md](PHASE12_REPORT.md)).** Phase 12 baseline: **623 passed / 7 live-network skips**; reproduced in a fresh `.venv_clean` installed only from `requirements.lock.txt` (295.46 s). After the Phase 13 determinism fix and added regression test, the final suite is **624 passed / 7 skipped**; the post-fix clean-environment rerun is recorded in [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md). The live-only suite passed **7/7** in 102.44 s. The live pipeline run `phase12-live-final` completed all seven stages in 58.0 s and selected Random Forest on validation F1. Fixture workflow completed in 16.668 s. Cleanup, resume, rejection, manifest verification and import boundaries have dedicated tests. The fast profile was used for both measured runs; the full production profile remains unmeasured.

**Risks and unresolved decisions.** TF/Streamlit startup cost in tests; fixture vs real-data divergence; Windows path handling; the exact lock is Windows/CPython 3.13-specific; default MLP float non-determinism; distributed artifact schema validation.

---

## 14. Phase 13 — Final Verification, Documentation and Presentation

**Objectives and scope.** Verify every criterion with evidence and produce final docs and demo. In scope: criteria-by-criteria verification table, measured metrics summary with run IDs, limitations and ethics statement, README quick start, refreshed architecture/roadmap, demo script and screenshots, dead-code cleanup. Out of scope: scope expansion.

**Inputs and outputs.** Inputs: the complete system and all prior evidence. Outputs: verification record, README, presentation, demo.

**Expected artifacts.**

| Artifact | Contents |
| --- | --- |
| `docs/FINAL_ACCEPTANCE_MATRIX.md` | Phase-by-phase status, evidence and limitations |
| `docs/FINAL_VERIFICATION_REPORT.md` | Full verification narrative, metrics, defects, limits and next steps |
| `docs/PROJECT_PRESENTATION.md` | Presentation narrative and measured results |
| `README.md` | Quick start, architecture summary, command reference |
| `docs/screenshots/*` | Final UI evidence |

**Dependencies.** Phase 12.

**Testing and verification.**

| ID | Check | Method |
| --- | --- | --- |
| T13.1 | Every criterion in this document has a status and an evidence reference | Manual + scripted checklist parse |
| T13.2 | No placeholder or unverified claim remains | Grep + review |
| T13.3 | Reported metrics equal `metrics.json` values exactly | Automated comparison of doc numbers vs artifact |
| T13.4 | Clean-checkout reproduction follows README without undocumented steps | Reviewer walkthrough |
| T13.5 | Demo script runs start-to-finish on the audited machine | Timed run |

**Completion criteria.**

- [x] All 13 phases marked pass (or `deferred` with documented risk) in the verification table.
- [x] Every displayed metric traceable to a run ID and artifact path.
- [x] Limitations, assumptions and non-goals stated plainly (no causal claims, no uplift measurement).
- [x] README quick start works from a clean checkout.
- [x] Presentation includes the architecture, the pipeline, measured benchmark results, SHAP insights, risk ranking and a live simulation.
- [x] No `TODO`/`TBD` remains; dead code and stale artifacts removed.
- [x] Scope preservation statement re-verified against [ROADMAP.md](ROADMAP.md#5-scope-preservation-statement).

**Risks and unresolved decisions.** Documentation drift late in the schedule; temptation to over-claim; insufficient time for slides — mitigated by building the presentation from existing artifacts rather than new analysis.

---

## 15. Traceability matrix

| Phase | Primary requirement | Key artifact | Decisive tests |
| --- | --- | --- | --- |
| 1 | Foundation and plan | `docs/*` | T1.1–T1.7 |
| 2 | Autonomous discovery (mandatory) | `reports/acquisition/candidates.json` | T2.1–T2.6 |
| 3 | Autonomous acquisition + provenance (mandatory) | `data/manifests/*.json` | T3.1–T3.8 |
| 4 | Data integrity and understanding | `reports/validation/*` | T4.1–T4.6 |
| 5 | Leakage-safe preprocessing | `artifacts/preprocessors/*` | T5.1–T5.7 |
| 6 | MLP (ReLU/sigmoid/BCE/Adam/regularisation/early stop) | `artifacts/models/*/mlp/*` | T6.1–T6.7 |
| 7 | Fair benchmark + held-out metrics | `artifacts/evaluation/*` | T7.1–T7.7 |
| 8 | SHAP global + local | `artifacts/explanations/*` | T8.1–T8.6 |
| 9 | Risk ranking and filtering | `artifacts/predictions/*` | T9.1–T9.7 |
| 10 | Campaign scenario simulation | `artifacts/simulations/*` | T10.1–T10.7 |
| 11 | Eleven-page dashboard | `churnwise/dashboard/*` | T11.1–T11.12 |
| 12 | Reliability and end-to-end | `tests/integration/test_pipeline_workflow.py` | T12.1–T12.7 |
| 13 | Verification and presentation | `docs/FINAL_ACCEPTANCE_MATRIX.md` · `docs/FINAL_VERIFICATION_REPORT.md` | T13.1–T13.5 |

---

## 16. Final sign-off gate

The project is complete only when all of the following hold simultaneously:

1. All 13 phases are `pass`, or explicitly `deferred` with a risk and owner recorded — never silently absent.
2. A first-time user obtains a working dataset and dashboard without supplying data or a URL.
3. All three models are trained and benchmarked on identical data, with real held-out metrics.
4. SHAP explanations exist globally and per customer.
5. Risk ranking and campaign simulation are functional, with assumptions visible and results labelled as projections.
6. The dashboard's eleven pages read only stored artifacts; service actions run only from explicit buttons.
7. `pytest -q` passes, determinism holds, and every failure path is documented.
8. No fabricated, predetermined or unsupported number appears anywhere.
9. No unnecessary infrastructure was introduced.
10. No secret or credential appears in the repository or artifacts.
