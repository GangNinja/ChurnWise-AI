# ChurnWise AI — Benchmarking and Evaluation Reference

| Field | Value |
| --- | --- |
| Scope | Phase 7 implementation reference (comparative model benchmarking) |
| Code | `churnwise/evaluation/`, `churnwise/models/baselines.py` |
| Artifacts | `artifacts/evaluation/<benchmark_run_id>/` |
| Related | [PHASE7_REPORT.md](PHASE7_REPORT.md) · [ARCHITECTURE.md](ARCHITECTURE.md#84-evaluation-protocol) · [MODELING.md](MODELING.md) · [PREPROCESSING.md](PREPROCESSING.md) |

This document describes how ChurnWise benchmarks Logistic Regression, Random
Forest and the Phase 6 MLP. It contains **no performance claims**; measured
results live in [PHASE7_REPORT.md](PHASE7_REPORT.md) and in each run's
`metrics.json`.

---

## 1. Quick start

```bash
# Full benchmark: fit LR/RF, load the Phase 6 MLP, 5-fold CV, one test pass
python -m churnwise.cli evaluate \
  --preprocessing-run-id <phase5-run-id> \
  --mlp-run-id <phase6-model-run-id>

# Faster: skip cross-validation
python -m churnwise.cli evaluate --cv-folds 0

# Rebuild metrics and figures from saved probabilities (no refit, no rescoring)
python -m churnwise.cli evaluate --regenerate <benchmark-run-id>
```

The command exits `0` when all three families were evaluated, `2` when the
benchmark is incomplete or an artifact/model failure occurs, and `3` for an
invalid configuration. If the MLP artifact is missing, the run still records
LR/RF results, marks `complete: false`, and exits `2` rather than inventing an
MLP score.

---

## 2. Fairness contract

The benchmark enforces seven conditions:

1. **One dataset, one matrix.** LR, RF and the MLP consume the same Phase 5
   transformed `train`/`val`/`test` matrices. The MLP's recorded
   `split_fingerprint`, `preprocessing.run_id` and `feature_names` must match the
   loaded Phase 5 run or the MLP family is rejected.
2. **Disjoint partitions.** Overlapping train/validation/test source indices are
   rejected before any fit.
3. **Training-only fitting.** Baselines are fitted on the same rows the MLP used
   (including Phase 5 training-only oversampling when configured). Validation and
   test rows never enter `.fit()`.
4. **Identical imbalance treatment.** All three families receive the Phase 5
   training class weights. The baseline defaults in `docs/ARCHITECTURE.md` are
   overridden with that explicit dictionary for parity.
5. **Validation-only threshold selection.** Every threshold is chosen from
   validation predictions and frozen before the test partition is scored.
6. **Validation-only model selection.** The shipped family is chosen by the
   configured validation metric (default positive-class F1). Test metrics are
   reported but never consulted by the selection code.
7. **One recorded test pass.** Test probabilities are produced after the freeze,
   inside a single evaluation loop; `test_evaluation_calls = 1` and
   `test_metric_evaluations = 3` are persisted in the validity block.

Hyperparameters were fixed in advance. No grid search touches the test split; if
one is added later, it must run inside the training partition under an equal
budget for every family.

---

## 3. Public interface

```python
from churnwise.config import Settings
from churnwise.evaluation import (
    BenchmarkCoordinator, EvaluationConfig, EvaluationError, evaluate,
)
from churnwise.preprocessing import PreprocessingCoordinator

settings = Settings.load()
dataset = PreprocessingCoordinator(settings).load("<phase5-run-id>")

result = evaluate(
    dataset,
    settings=settings,
    config=EvaluationConfig(cv_folds=5, threshold_strategy="max_f1"),
    mlp_run_id="<phase6-model-run-id>",
)

result.selected_family          # "logistic_regression" | "random_forest" | "mlp"
result.threshold("random_forest")  # frozen operating threshold
result.test_metrics("mlp")      # accuracy, precision, recall, F1, ROC-AUC, ...
result.family("mlp")["cv"]      # per-fold and mean ± std CV payload
result.paths["metrics"]         # artifact locations
```

`BenchmarkCoordinator` also exposes:

| Method | Purpose |
| --- | --- |
| `run(dataset, mlp_run_id=None, benchmark_run_id=None, persist=True)` | Produce a benchmark; `persist=False` keeps results in memory. |
| `load(benchmark_run_id)` | Reload structured metrics without recomputing anything. |
| `regenerate(benchmark_run_id)` | Recompute metrics/figures from `predictions.npz`; verifies values against the stored metrics. |
| `latest_run_id()` | Most recent persisted benchmark. |

`EvaluationConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `seed` | 42 | Seed for baseline estimators and CV folds. |
| `threshold_strategy` | `max_f1` | `fixed`, `max_f1`, `max_youden` or `recall_floor`. |
| `fixed_threshold` | 0.5 | Default cut-off; also reported as a secondary view. |
| `recall_floor` | `None` | Required by `recall_floor`. |
| `selection_metric` | `f1` | Validation metric used to choose the shipped family. |
| `selection_min_recall` | `None` | Optional validation recall floor for selection. |
| `cv_folds` | 5 | `0` disables CV; `1` is invalid. |

---

## 4. Models

### Logistic Regression (`churnwise/models/baselines.py`)

`LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
class_weight=<Phase 5 weights>, random_state=seed)` on the shared transformed
matrix. `predict_proba(...)[:, 1]` is `P(churn = 1)`.

### Random Forest

`RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2,
min_samples_leaf=2, max_features="sqrt", bootstrap=True,
class_weight=<Phase 5 weights>, n_jobs=-1, random_state=seed)`. The forest is
deliberately bounded for CPU-only laptops; `min_samples_leaf=2` limits tree
memorisation while remaining a genuine ensemble.

### MLP

The **existing Phase 6 artifact** is loaded through `ModelRegistry`: digest
verified, metadata run id checked, feature width checked, and predictions taken
from the fitted Keras model. It is not retrained or reselected during the
benchmark. Baseline artifacts are saved with joblib plus SHA-256 metadata so
Phase 8 can explain exactly the models that produced these metrics.

---

## 5. Metrics

Positive class: **1 = churned** (Phase 4/5 mapping; source `Yes`).

| Metric | Definition | Notes |
| --- | --- | --- |
| `accuracy` | `(TP+TN)/N` | Reported, but never the sole selection criterion. |
| `precision` | `TP/(TP+FP)` | Positive class. |
| `recall` | `TP/(TP+FN)` | Positive class; drives missed churners. |
| `f1` | harmonic mean of precision and recall | Positive class. |
| `specificity` | `TN/(TN+FP)` | Negative class. |
| `balanced_accuracy` | mean of recall and specificity. | Robust to imbalance. |
| `macro_precision/recall/f1` | unweighted mean over both classes. | Undefined when either class metric is undefined. |
| `roc_auc` | threshold-free ranking quality. | `n/a` for a single-class partition. |
| `pr_auc` | average precision for the positive class. | `n/a` when there are no churn rows. |
| `brier` | `mean((p − y)²)`. | Calibration-oriented; lower is better. |
| `log_loss` | binary cross-entropy. | `n/a` for a single-class partition. |

Every metric block records `n`, `n_positive`, `positive_rate`,
`predicted_positive`, `threshold`, `undefined` (metric names) and `notes`
(human-readable reasons). `precision`, `recall`, `F1` and AUC are returned as
`null` — never silently replaced by `0.0` — when their denominator is undefined.

The confusion matrix stores an explicit label order:

```json
{"labels": [0, 1], "matrix": [[tn, fp], [fn, tp]], "tn": 0, "fp": 0, "fn": 0, "tp": 0}
```

`tn + fp + fn + tp` is asserted equal to `n_test` in tests and in the report.

---

## 6. Threshold selection

Threshold sweeps use a deterministic 99-point grid (`0.01 … 0.99`) that always
contains `0.5`.

| Strategy | Rule |
| --- | --- |
| `fixed` | Use the configured fixed threshold (default 0.5). |
| `max_f1` | Highest positive-class F1 on validation. |
| `max_youden` | Highest `recall + specificity − 1` on validation. |
| `recall_floor` | Highest threshold whose validation recall ≥ floor; fallback to `max_f1` (recorded) if none. |

Ties break toward the threshold closest to 0.5. The chosen threshold is written
into every metric block and into `run_manifest.json`; the 0.5 view is reported
separately as `test_metrics_at_0_5`. ROC-AUC is explicitly labelled as a
threshold-free metric, and the report states that a statistically optimal
threshold is not automatically the best business decision.

---

## 7. Cross-validation

CV is optional (`--cv-folds`, default 5; `0` disables). It uses stratified folds
drawn from **train + validation only** and reports mean ± std for accuracy,
precision, recall, F1, ROC-AUC, PR-AUC and Brier at the 0.5 reference threshold.

**Fold-safe preprocessing:** every fold refits the Phase 5 column plan, imputers,
scaler and one-hot encoder on its own training rows; that fold's validation rows
are transform-only. No fold sees its own validation statistics. Because the Phase
5 hand-off persists transformed matrices rather than raw rows, the base frame is
reconstructed from the recorded raw artifact and used only after two checks: the
labels at every split's recorded row indices must match, and a full-training
refit must reproduce the Phase 5 train/validation/test matrices within `1e-9`. If
either check fails, CV is reported as `unavailable` with the reason instead of
being run against a shared transform. Per-fold feature counts can differ when a
categorical level is absent from a fold's training rows; each fold's count is
recorded in `fold_details`, and the aggregate flags are in
`cross_validation.preprocessing`.

The MLP CV path refits a fresh Keras model per fold with the Phase 6
configuration, an internal training-only hold-out for early stopping, and its own
seed. It is the slowest part of a full run (~70 s for 5 folds on the audited
laptop, measured 71–107 s across reruns); use `--cv-folds 0` for a quicker
benchmark.

---

## 8. Artifacts

```text
artifacts/evaluation/<benchmark_run_id>/
├── metrics.json                 # everything below, structured and JSON-safe
├── benchmark.md                 # human-readable report with the same numbers
├── run_manifest.json            # selected family, thresholds, provenance
├── predictions.npz              # y_train/y_val/y_test + proba_{split}_{family}
├── curves.json                  # ROC/PR points and AUC/AP per family
├── threshold_analysis.json      # validation sweeps + frozen choices
├── confusion_matrix_<family>.png
├── roc_curves.png
├── precision_recall_curves.png
├── metric_comparison.png
├── threshold_analysis.png
├── probability_distributions.png
└── models/<family>/{model.joblib,metadata.json}   # LR and RF only
```

`metrics.json` top-level keys:

| Key | Contents |
| --- | --- |
| `preprocessing` | Phase 5 run id, dataset hash, split fingerprint, preprocessor hash, target, feature names. |
| `dataset` | Train/validation/test sizes, test churn count/rate, class weights, `test_rows_used`. |
| `evaluation_config` | Threshold/selection/CV settings and fingerprint. |
| `threshold_policy` | Primary strategy, freeze ordering, default-threshold reporting. |
| `families` | Per-family status, model config, threshold, validation/test/train metrics, curves summary, CV, notes or error. |
| `comparison_table` | One row per completed family with counts and metrics. |
| `selection` | Selection metric, winner, ranking, validation evidence, rationale, `test_used_for_selection: false`. |
| `error_analysis` | Aggregate FP/FN counts and mean error probabilities; no customer records. |
| `cross_validation` | Enabled flag, folds, scope and caveats. |
| `plots`, `artifacts` | Figure and file locations. |
| `environment` | Python, NumPy, pandas, scikit-learn, TensorFlow/Keras versions. |
| `validity` | ROC-AUC-by-family, test evaluation counts, freeze flag, defect flags. |
| `complete`, `missing_families`, `warnings`, `limitations` | Honest completeness record. |

`predictions.npz` keys: `y_train`, `y_val`, `y_test` and
`proba_{train,val,test}_{family}` for every completed family. Regeneration reads
only these arrays and the stored thresholds.

---

## 9. Regeneration and reproducibility

`evaluate --regenerate <id>`:

1. loads `metrics.json` and `predictions.npz`;
2. recomputes every metric at the **stored** thresholds;
3. raises `CW-EVL-003` if a recomputed value differs from the stored one by more
   than `1e-9`, or if a confusion count changes;
4. rewrites `metrics.json`, `curves.json`, every figure and `benchmark.md` with
   `regeneration.models_refitted = false` and
   `regeneration.test_rows_rescored = false`.

Baseline estimators are seeded (`random_state`), the MLP artifact is frozen, and
thresholds are deterministic, so a fresh run on the same artefacts reproduces the
same metrics. Bitwise reproducibility across machines is not guaranteed
(TensorFlow/oneDNN and BLAS differences), and the report says so.

---

## 10. Failure semantics

| Code | Meaning | Behaviour |
| --- | --- | --- |
| `CW-EVL-001` | Invalid evaluation configuration. | Raised before any fit; CLI exit 3. |
| `CW-EVL-002` | Incompatible dataset/model/preprocessing artifacts. | Family rejected or run rejected with remediation. |
| `CW-EVL-003` | Metric computation or regeneration mismatch. | Raised; the stored artifact is treated as corrupt. |
| `CW-EVL-004` | Evaluation artifact read/write failure. | Raised with the offending path. |
| `CW-EVL-005` | Every family failed, or a family cannot be scored. | Run fails rather than fabricating a comparison. |
| `CW-EVL-006` | No matching Phase 6 MLP artifact. | MLP recorded as failed; LR/RF continue; incomplete run. |

A single family failure (missing MLP, bad class balance, estimator error) is
recorded under that family with `status: "failed"` and an error dict. Other
families keep their metrics, plots and artifacts.

---

## 11. Contract for Phase 8 and later

Phase 8 (SHAP) and Phase 9 (risk scoring) can rely on:

1. `BenchmarkCoordinator(settings).load(run_id)` for structured metrics.
2. `result.selected_family`, `result.threshold(family)`, `result.test_metrics(...)`.
3. `metrics.json` → `families.<family>.model` for the exact model configuration
   and, for LR/RF, the digest-verified `model.joblib` path.
4. `predictions.npz` for reproducible per-row probabilities without rescoring.
5. `curves.json` / `threshold_analysis.json` for curve and sweep data.
6. `run_manifest.json` for the shipped family, thresholds and provenance chain
   (benchmark → preprocessing run → split fingerprint → dataset SHA-256 →
   preprocessor hash).

Phase 8 should explain the **selected** family recorded in `run_manifest.json`
unless a comparison is explicitly requested. It must reuse the frozen threshold;
it must not re-evaluate the test split to choose a different model.

---

## 12. Limitations

- One held-out split per evaluation; no confidence intervals are computed.
- Thresholds optimise a statistical criterion, not contact cost or capacity.
- CV refits preprocessing per fold but draws a single seeded fold partition, so
  it does not quantify split-to-split variance (see §7).
- Calibration is summarised by Brier score; no recalibration is fitted.
- The benchmark compares families on one dataset; results do not transfer
  automatically to another customer base.
- No SHAP, campaign simulation or dashboard code exists in Phase 7.
