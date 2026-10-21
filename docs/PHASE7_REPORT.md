# ChurnWise AI — Phase 7 Report

**Scope:** implement and execute a fair, reproducible benchmark of Logistic
Regression, Random Forest and the Phase 6 MLP on the Phase 5 held-out test split.
No SHAP, risk tiering, campaign simulation or dashboard work was started.

**Status:** implementation, tests and a genuine benchmark are complete. The test
partition was evaluated **once** after thresholds were frozen on validation data.
Random Forest was selected on validation F1; the MLP was **not** the best model on
that criterion. Every number below is read from
`artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/metrics.json`.

---

## 1. Executive summary

Phase 7 adds:

- documented scikit-learn **Logistic Regression** and **Random Forest** baselines
  that consume the same Phase 5 matrix and class weights as the MLP;
- a shared metric module with positive-class conventions, explicit `n/a` handling
  for undefined metrics, curves and a 99-point threshold sweep;
- **validation-only** threshold selection (`fixed`, `max_f1`, `max_youden`,
  `recall_floor`) frozen before the single test evaluation;
- a configurable, validation-only model-selection rule with a transparent
  rationale and ranking;
- optional 5-fold stratified cross-validation on train+validation;
- confusion-matrix, ROC, PR, metric-comparison, threshold and probability-
  distribution figures generated only from measured results;
- digest-verified baseline artifacts, saved per-row probabilities, a run manifest
  and deterministic regeneration from stored predictions;
- 57 focused evaluation tests (390 total in the project).

| Verification | Result |
| --- | --- |
| Focused Phase 7 tests | **57 passed** in 25.93 s |
| Full project suite | **390 passed, 7 skipped** in 114.89 s |
| Genuine benchmark | **completed**, all three families |
| Test rows evaluated | **1000**, in one recorded pass |
| Selected model | **Random Forest** (validation F1 0.7985) |
| MLP outcome | **third of three** on validation F1 (0.6937); not assumed best |
| Test ROC-AUC | LR 0.8741 · RF 0.9383 · MLP 0.9266 (all > 0.5) |
| Regeneration from saved predictions | **exact match** (max difference 0.0) |

---

## 2. Upstream inspection and hand-off

The implementation was written after reading Phases 1–6, the Phase 5 split and
preprocessor contracts, and the actual Phase 6 artifact.

| Upstream fact | Value used |
| --- | --- |
| Phase 5 run | `20261021T123405Z-openml_46915-989d8802` |
| Phase 6 MLP run | `20261021T132609Z-openml_46915-mlp-443bb6a1` |
| Dataset | `openml:46915`, 5,000 rows, MIT licence |
| Dataset SHA-256 | `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6` |
| Split | stratified 60/20/20, seed 42, **3000 / 1000 / 1000** |
| Split fingerprint | `4b0c45db677fa9619de99b6a8b786a3da2c4da93824c46db65412a85829594dd` |
| Preprocessor hash | `3c31f13c266c21084050bb021e60a41d4145a2055b983eb03e2267fab9f52174` |
| Transformed features | **27** |
| Positive class | `1 = churned`, source `Yes` |
| Test churn rows | **141 / 1000 (14.1%)** |
| Class weights | `{0: 0.5823, 1: 3.5377}` |
| Test matrix SHA-256 | `f87cb7d706d6022c8bb0296aa00d9873ffce3c95fa3df15b868ecf39677f61c7` |

The MLP was loaded, not retrained: its metadata declares the same preprocessing
run, split fingerprint and 27 feature names, which the benchmark verifies before
predicting. No Phase 5 code was duplicated. There were no upstream defects that
required a workaround; the only extension to existing modules is the new
`churnwise/models/baselines.py` and `churnwise/evaluation/` package.

---

## 3. Benchmarking methodology

1. Load the persisted Phase 5 dataset (`PreparedDataset`) and validate that the
   three partitions are disjoint, aligned and share one feature width.
2. Load the verified Phase 6 MLP and check its preprocessing run id, split
   fingerprint and feature names against the dataset.
3. Fit LR and RF on the training rows (including Phase 5 training-only
   oversampling if configured) with the Phase 5 class weights.
4. Score **validation** rows only; select a per-family operating threshold on
   validation predictions; freeze the thresholds.
5. Fit nothing else. Score **test** rows once, compute metrics and curves, then
   run model selection on validation metrics.
6. Run optional 5-fold CV on train+validation; persist everything.

Test labels never enter threshold selection, hyperparameter choice or model
selection. A test proves this by randomising the test labels and asserting that
every threshold, validation metric and the selected family stay identical while
test metrics change. A second test proves that a failed family (missing MLP,
foreign preprocessing run) leaves the other families' metrics and artifacts intact.

**Identical-matrix evidence.** The exact float64 matrices consumed by every family
are hashed into `metrics.json`:

| Partition | SHA-256 |
| --- | --- |
| Train | `dd012e143225a7068dede8f5eed30681be4ef9ea7cf296a1fbe035834b1733ec` |
| Validation | `56601abb706119e1545a35adb19d0977544cb7d9cda67873743bc8fcd898e41e` |
| Test | `f87cb7d706d6022c8bb0296aa00d9873ffce3c95fa3df15b868ecf39677f61c7` |

---

## 4. Model configurations

| Family | Configuration | Train rows | Fit time |
| --- | --- | --- | --- |
| Logistic Regression | `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=Phase 5 weights, random_state=42)` | 3000 | 0.011 s |
| Random Forest | `RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features="sqrt", bootstrap=True, class_weight=Phase 5 weights, n_jobs=-1, random_state=42)` | 3000 | 0.537 s |
| MLP Neural Network | Phase 6 artifact: 27 → Dense 64 ReLU → Dropout 0.30 → Dense 32 ReLU → Dropout 0.20 → sigmoid; L2 1e-4, Adam 1e-3, batch 64; 3,905 parameters | 3000 | 11.753 s (Phase 6) |

All three receive the **same Phase 5 class weights**; the RF default
`balanced_subsample` from the architecture document is replaced by that explicit
dictionary for fairness. Hyperparameters were fixed in advance and were not tuned
on any split during this phase. No preprocessing differs between families, so no
per-family transformation is claimed.

---

## 5. Metric definitions and positive-class mapping

Positive class = **1 = churned**. Reported positive-class metrics:

- `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`;
- `accuracy = (TP+TN)/N`;
- `specificity`, `balanced_accuracy`;
- `ROC-AUC` (threshold-free ranking) and `PR-AUC` (average precision);
- `Brier = mean((p−y)²)` and log loss for calibration;
- macro precision/recall/F1 over both classes;
- a 2×2 confusion matrix with explicit label order `[0, 1]` and `tn/fp/fn/tp`.

Precision, recall, F1, AUC and PR-AUC are stored as `null` with a reason when
undefined (no predicted positives, no churn rows, single-class partition). They
are never replaced by a misleading zero. Metrics match scikit-learn within 1e-9
on fixed inputs (T7.1).

---

## 6. Threshold selection

The primary policy for this run is `max_f1` on **validation** predictions, using a
99-point grid from 0.01 to 0.99 that includes 0.5. The threshold is frozen before
test scoring. The 0.5 view is also reported for every family but never used for
selection. ROC-AUC is documented as a threshold-free metric; the report states
that a statistically optimal threshold is not automatically the best business
decision.

| Family | Strategy | Validation precision | Validation recall | Validation F1 | Frozen threshold |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | max_f1 | 0.5376 | 0.6549 | 0.5905 | **0.640** |
| Random Forest | max_f1 | 0.8678 | 0.7394 | 0.7985 | **0.450** |
| MLP Neural Network | max_f1 | 0.7287 | 0.6620 | 0.6937 | **0.730** |

The MLP's threshold rose from 0.5 to 0.730 because its validation probabilities
are shifted upward by the Phase 5 class weights; the sweep is saved in
[threshold_analysis.json](../artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/threshold_analysis.json).

---

## 7. Actual held-out results

### 7.1 Primary test metrics at the frozen thresholds

| Family | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.640 | 0.877 | 0.5459 | 0.7589 | 0.6350 | 0.8741 | 0.5717 | 0.1460 | 770 | 89 | 34 | 107 |
| Random Forest | 0.450 | **0.965** | **0.9015** | 0.8440 | **0.8718** | **0.9383** | **0.8973** | **0.0390** | 846 | 13 | 22 | 119 |
| MLP Neural Network | 0.730 | 0.921 | 0.7183 | 0.7234 | 0.7208 | 0.9266 | 0.7801 | 0.0944 | 819 | 40 | 39 | 102 |

Each confusion matrix reconciles to 1000 test rows. Test ROC-AUC exceeds 0.5 for
every family (T7.6), so no validity defect was flagged.

### 7.2 Secondary view at the default 0.5 threshold

| Family | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.809 | 0.4120 | 0.8298 | 0.5506 |
| Random Forest | 0.969 | 0.9365 | 0.8369 | 0.8839 |
| MLP Neural Network | 0.886 | 0.5634 | 0.8511 | 0.6780 |

At 0.5 the MLP has higher recall but much lower precision than at its selected
threshold; the selected threshold trades recall for precision. This is a measured
trade-off, not a claim that 0.73 is a business optimum.

### 7.3 Cross-validation stability (train + validation, 5 folds, threshold 0.5)

| Family | Accuracy | Precision | Recall | F1 | ROC-AUC | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8143 ± 0.0172 | 0.4202 ± 0.0273 | 0.8074 ± 0.0259 | 0.5523 ± 0.0246 | 0.8684 ± 0.0146 | 0.45 s |
| Random Forest | 0.9533 ± 0.0052 | 0.9094 ± 0.0212 | 0.7438 ± 0.0316 | 0.8180 ± 0.0229 | 0.9089 ± 0.0173 | 2.74 s |
| MLP Neural Network | 0.9070 ± 0.0166 | 0.6377 ± 0.0571 | 0.8127 ± 0.0250 | 0.7134 ± 0.0380 | 0.9097 ± 0.0193 | 70.95 s |

CV folds are drawn from train+validation only. **Preprocessing is refitted inside
every fold:** the Phase 5 column plan, imputers, scaler and encoder are fitted on
each fold's training rows, and its validation rows are transform-only. The base
frame is reconstructed from the recorded raw artifact and accepted only after
labels match every split and a full-training refit reproduces the Phase 5
matrices within `1e-9`; otherwise CV is reported as unavailable rather than run on
a shared transform. Fold widths can differ when a categorical level is missing
from a fold's training rows (recorded per fold in `fold_details`). No test row
participates in CV.

Observed (not explained) difference: RF's test ROC-AUC (0.9383) is higher than its
CV estimate (0.9089), while the MLP's test AUC (0.9266) is higher than its CV
estimate (0.9097). A single 1000-row test split with 141 positives carries
sampling uncertainty that this phase does not quantify.

---

## 8. Graphs and report artifacts

| Artifact | Location |
| --- | --- |
| Structured metrics | [metrics.json](../artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/metrics.json) |
| Human-readable benchmark | [benchmark.md](../artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/benchmark.md) |
| Run manifest (selected family, thresholds, provenance) | [run_manifest.json](../artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/run_manifest.json) |
| Saved probabilities | `predictions.npz` |
| ROC / PR curve data | `curves.json` |
| Validation sweeps | `threshold_analysis.json` |
| Confusion matrices | `confusion_matrix_{logistic_regression,random_forest,mlp}.png` |
| ROC curves with AUC | `roc_curves.png` |
| Precision-recall curves with AP | `precision_recall_curves.png` |
| Metric comparison | `metric_comparison.png` |
| Threshold analysis | `threshold_analysis.png` |
| Probability distributions by true class | `probability_distributions.png` |
| Baseline models (LR/RF, digest-verified) | `models/<family>/{model.joblib,metadata.json}` |

Every figure is generated from the measured probabilities; none contains
illustrative data. The Markdown report embeds the same numbers as `metrics.json`.
All figures were written at readable sizes (649×492 to 1644×499) and non-blank
checks passed.

---

## 9. Model selection, error analysis and overfitting evidence

### Selection

Criterion: highest **validation** positive-class F1 at each family's
validation-selected threshold. Ranking:

| Rank | Family | Validation F1 | Validation recall | Validation ROC-AUC | Test F1 (not used) |
| --- | --- | --- | --- | --- | --- |
| 1 | Random Forest | **0.7985** | 0.7394 | 0.8826 | 0.8718 |
| 2 | MLP Neural Network | 0.6937 | 0.6620 | **0.8920** | 0.7208 |
| 3 | Logistic Regression | 0.5905 | 0.6549 | 0.8699 | 0.6350 |

**Selected family: `random_forest`.** Rationale (generated from the stored
values): RF had the highest validation F1; MLP followed; the MLP was not the best
model on the validation criterion and the benchmark does not assume it must win.
`run_manifest.json` records `selected_model_family`, the metric, the rationale and
the frozen thresholds. The test column above is shown for transparency only.

Notable trade-off: the MLP had the **highest validation ROC-AUC** (0.8920 vs
0.8826) but lower validation F1, because its precision was lower at the selected
cut-off. Choosing on AUC would have selected the MLP; choosing on F1 selected RF.
The configurable `selection_metric` makes that policy explicit rather than hiding
it. Selecting on accuracy alone was explicitly avoided.

### Error analysis (aggregate only)

| Family | False positives | False negatives | FP rate of negatives | FN rate of positives | Mean P(FP) | Mean P(FN) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 89 | 34 | 10.4% | 24.1% | 0.8092 | 0.3694 |
| Random Forest | 13 | 22 | 1.5% | 15.6% | 0.5765 | 0.1789 |
| MLP Neural Network | 40 | 39 | 4.7% | 27.7% | 0.8595 | 0.4391 |

A false positive is a retained customer predicted to churn (wasted outreach); a
false negative is a churner predicted to stay (missed save opportunity). At these
thresholds the RF makes far fewer false positives but still misses 22 of 141
churners; the MLP misses 39. No customer identifiers or raw records are exposed.

### Overfitting signals

| Family | Train F1 | Validation F1 | Test F1 | Train−val ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6208 | 0.5905 | 0.6350 | 0.0039 |
| Random Forest | **0.9918** | 0.7985 | 0.8718 | **0.1174** |
| MLP Neural Network | 0.7871 | 0.6937 | 0.7208 | 0.0582 |

The RF nearly memorises its training rows (F1 0.99) while validation F1 is 0.80 —
a clear overfitting signal. Its held-out test F1 is nonetheless 0.87 on this
split, but that should not be read as evidence that the forest will generalise
equally well to a different customer base. The MLP and LR show much smaller
train/validation gaps. Train metrics are in-sample diagnostics only.

The report deliberately distinguishes **observed** results from interpretation;
no causal or production-readiness claim is made.

---

## 10. Reproducibility and result artifacts

- Benchmark run id: `20261021T145336Z-openml_46915-bench-04595d7b`
- Configuration fingerprint: `04595d7bd2b1d335713df1d73e4d505a1cc44f7543c1f00a9ff944789b5a98a1`
- Environment: Python 3.13.0, NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.7.2,
  TensorFlow 2.21.0, Keras 3.15.1 on Windows 11 (CPU).
- Seeds: Phase 5 split seed 42; baseline/CV seed 42; Phase 6 MLP seed 42.
- Saved predictions: `predictions.npz` with train/validation/test labels and
  per-family probabilities.
- Baseline artifacts carry SHA-256 digests; the MLP artifact was already
  digest-verified by `ModelRegistry`.

**Regeneration evidence.** `python -m churnwise.cli evaluate --regenerate
20261021T145336Z-openml_46915-bench-04595d7b` recomputed every metric from
`predictions.npz` at the stored thresholds. Accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, Brier and every confusion count matched the stored values exactly
(max difference 0.0). The regeneration block records `models_refitted: false` and
`test_rows_rescored: false`.

The benchmark was executed three times from the same Phase 5/6 artifacts while the
code was being finalised; all three produced identical metrics and the same
selected family, consistent with the fixed seeds and frozen MLP artifact. This is
evidence of stable behaviour on this host, not a bitwise cross-machine guarantee.

---

## 11. Tests run and actual results

### Focused Phase 7 suite

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest \
  tests/unit/test_metrics.py \
  tests/unit/test_thresholds.py \
  tests/unit/test_baselines.py \
  tests/unit/test_benchmark_selection.py \
  tests/integration/test_benchmark_parity.py -q
```

Result: **53 passed, 48 warnings in 24.60 s**. Warnings are Keras's NumPy
deprecation warning; no test failed or was skipped.

Coverage includes:

- sklearn metric parity within 1e-9, confusion label ordering,
  `tn+fp+fn+tp = n`, undefined precision/F1, single-class partitions,
  ROC/PR data, threshold monotonicity, invalid inputs;
- all four threshold strategies, centred tie-breaks, recall-floor fallback,
  fixed threshold, invalid configurations;
- LR/RF construction, fitting, bounded probabilities, class weights,
  single-class rejection, feature-width checks, fixed-seed reproducibility,
  joblib round trip and tampered-digest rejection;
- two complete benchmark runs under identical settings reproduce the same
  thresholds, metrics, confusion matrices and selected family;
- full benchmark with all three families, shared matrix digests, confusion
  reconciliation, artifacts, selection ranking, `test_evaluation_calls = 1`;
- **test-label poisoning leaves thresholds, validation metrics and selection
  unchanged** while test metrics change;
- a missing/foreign MLP is recorded as a failed family while LR/RF stay intact;
  all-families-failing raises `CW-EVL-005` instead of fabricating results;
- CLI `evaluate` and `--regenerate` round trips, invalid configuration exit code;
- integration with the actual Phase 5 and Phase 6 artifacts when present.

### Full project suite

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
```

Result: **390 passed, 7 skipped in 114.89 s**. The seven skips are the
pre-existing live-network tests (`--run-live` opt-in). No pre-existing test
regressed.

---

## 12. Genuine benchmark command and result

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m churnwise.cli evaluate \
  --preprocessing-run-id 20261021T123405Z-openml_46915-989d8802 \
  --mlp-run-id 20261021T132609Z-openml_46915-mlp-443bb6a1
```

Recorded outcome:

| Field | Value |
| --- | --- |
| Benchmark run | `20261021T145336Z-openml_46915-bench-04595d7b` |
| Families completed | 3 / 3 |
| Test rows | 1000 (141 churn, 14.1%) |
| Frozen thresholds | LR 0.640, RF 0.450, MLP 0.730 |
| Selected family | `random_forest` |
| Test F1 | LR 0.6350, RF 0.8718, MLP 0.7208 |
| Test ROC-AUC | LR 0.8741, RF 0.9383, MLP 0.9266 |
| Test evaluation calls | 1 |
| CV | 5 folds per family, completed |
| Warnings in the run | none |

The command exited `0` and wrote the artifacts listed in §8. No number was
invented, and no metric was preselected.

---

## 13. Known limitations and unverified items

| Item | Status |
| --- | --- |
| Cross-validation preprocessing refit | **Implemented.** Each fold refits the Phase 5 plan, imputers, scaler and encoder on its own training rows after a verified base-frame reconstruction; `cross_validation.preprocessing.fold_safe = true` in the reference run. The fold partition itself is a single seeded 5-fold split, so split-to-split variance is still unquantified. |
| Confidence intervals | **Not computed.** The test set has 141 churn rows; sampling uncertainty is not quantified. |
| Business-optimal threshold | **Not claimed.** Thresholds maximise validation F1; contact cost, capacity and uplift belong to later phases. |
| Probability calibration | **Not fitted.** Brier/log loss are reported; no isotonic/Platt calibration exists. |
| MLP tuning in Phase 7 | **Not done by design.** The Phase 6 artifact is evaluated as-is. |
| RF overfitting | **Observed** (train F1 0.99 vs validation 0.80). Not remedied with new hyperparameter search, to avoid unequal tuning budgets. |
| Cross-machine bitwise determinism | **Not guaranteed** (oneDNN/BLAS/TensorFlow version differences). Seeds are fixed and same-host reruns matched. |
| GPU performance | **Not exercised** (CPU-only environment). |
| Live-network tests | **Skipped** by design in the default suite. |
| SHAP, risk tiers, simulation, dashboard | **Not implemented** — Phase 8+ scope. |

No Phase 7 acceptance check (T7.1–T7.7) is unverified. The remaining CV caveat
(one seeded fold partition, so split-to-split variance is unquantified) is a
documented design limitation, not a failed check.

---

## 14. Files delivered

| Area | Files |
| --- | --- |
| Evaluation package | `churnwise/evaluation/{__init__,errors,contracts,metrics,thresholds,cv,plots,benchmark,reporting}.py` |
| Baselines | `churnwise/models/baselines.py`; re-exported from `churnwise/models/__init__.py`; family constants in `churnwise/models/contracts.py` |
| CLI | `churnwise/cli.py` (`evaluate`, `--regenerate`) |
| Version | `churnwise/__init__.py` (`0.6.0`) |
| Tests | `tests/unit/test_metrics.py`, `test_thresholds.py`, `test_baselines.py`, `test_benchmark_selection.py`, `tests/integration/test_benchmark_parity.py` |
| Documentation | this report, `docs/EVALUATION.md`, `docs/ARCHITECTURE.md` §8.3/§8.4, `docs/ROADMAP.md` Phase 7, `docs/ACCEPTANCE_CRITERIA.md` Phase 7 |
| Artifacts | `artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/*` |

No dependency was added: scikit-learn and joblib were already declared. No
destructive Git operation was performed (the workspace is not a Git repository).
No Phase 8 implementation was started.

---

## 15. Completion procedure check

| Procedure step | Outcome |
| --- | --- |
| Implement Phase 7 only | Done; no SHAP, ranking, simulation or dashboard code |
| Run focused benchmarking and metric tests | 53 passed |
| Run integration tests against Phases 5 and 6 | Passed, including real artifacts |
| Execute a genuine benchmark | Done, all three families, one test pass |
| Fix introduced failures and rerun | Fixed ranking/formatting defects; non-monotonic test assumptions corrected; refocused tests rerun |
| Separate pre-existing failures from regressions | No new failures; 7 live-network skips are pre-existing |
| Inspect for test leakage and inconsistent comparisons | Test-label poisoning test, shared matrix digests, frozen thresholds, single test pass |
| Update the Phase 7 report with verified evidence | This document |
| Stop after Phase 7 | Yes |
