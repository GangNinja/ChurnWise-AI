# ChurnWise AI — Phase 7 benchmark `telco-mlp-primary-evaluation`

All metrics below were measured on the held-out test partition created in Phase 5 and scored once, after thresholds were frozen on validation data. No number is illustrative.

| Field | Value |
| --- | --- |
| Benchmark run | `telco-mlp-primary-evaluation` |
| Preprocessing run | `20261030T171452Z-wa-fn-usec-telco-customer-churn-066103c0-prep` |
| Dataset | `fixture-wa-fn-usec-telco-customer-churn-88be4b93` |
| Dataset SHA-256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |
| Split fingerprint | `5368065bd03180ded359a09e3df7c74d3e36406553ddad9cd280614e6cb20113` |
| Preprocessor hash | `64094e20109f86e2aa23c2469d7d14cfb452d2afaa9f237a6d3ce63341922764` |
| Transformed features | 49 |
| Train / validation / test rows | 4225 / 1409 / 1409 |
| Test churn rows | 374 (26.5%) |
| Generated | 2026-10-30T17:24:43Z |
| Code version | `0.10.0` |
| Complete (all three families) | yes |

## 1. Methodology

- **Shared data:** Logistic Regression, Random Forest and the Phase 6 MLP all consume the same Phase 5 transformed matrices and the same split fingerprint recorded above.
- **Fitting scope:** baselines are fitted on the training partition only. The MLP artifact was trained on that same partition in Phase 6 and is loaded, not retrained.
- **Imbalance treatment:** all families receive the Phase 5 training class weights `{'0': 0.6805734536082474, '1': 1.884478144513827}`.
- **Threshold policy:** each family's operating threshold is selected on validation predictions (`max_f1`) and frozen before the test partition is scored. The 0.5 default is also reported but never used for selection.
- **Selection criterion:** `f1` measured on validation, with the optional recall floor `None`. Test metrics are not an input.
- **Test isolation:** the test matrix was evaluated once (`test_evaluation_calls = 1`); no threshold, hyperparameter or model choice used test labels.
- **Cross-validation:** disabled (0 folds on train+validation).

## 2. Model configurations

| Family | Source | Configuration | Train rows | Fit time (s) | Threshold |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | fitted_in_phase7 | LogisticRegression(C=1.0, solver=lbfgs, penalty=l2, max_iter=2000) | 4225 | 0.02 | 0.610 |
| Random Forest | fitted_in_phase7 | RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features=sqrt) | 4225 | 1.10 | 0.390 |
| MLP Neural Network | phase6_artifact | MLP 64/32 ReLU, dropout [0.3, 0.2], L2 0.0001, Adam 0.001, batch 64, epochs <= 150 | 4225 | 12.30 | 0.590 |

Hyperparameters were fixed in advance from the architecture document and were not searched on the test set. The MLP configuration is the one recorded in its Phase 6 metadata.

## 3. Metric definitions and positive class

The positive class is **1 = churned** (Phase 4/5 target mapping; source label `Yes`).

- `accuracy = (TP + TN) / N`
- `precision = TP / (TP + FP)` — of customers flagged for retention, the share who would actually churn
- `recall = TP / (TP + FN)` — of customers who churn, the share flagged
- `F1 = 2 · precision · recall / (precision + recall)`
- `ROC-AUC` and `PR-AUC` are threshold-free ranking metrics; PR-AUC here is average precision for the positive (churn) class
- `Brier = mean((p − y)²)` — lower is better; a calibration-oriented summary

Precision, recall, F1 and AUC are reported as `n/a` when their denominator is undefined (for example no predicted positives, or a single-class partition) rather than being replaced with a misleading zero. Macro precision/recall/F1 are also stored in `metrics.json` for both classes.

## 4. Threshold selection

Thresholds are chosen from validation predictions only. ROC-AUC ranks across all thresholds; the operating threshold is a separate decision. A statistically optimal threshold is not automatically the best business decision — contact costs and capacity belong to later phases.

| Family | Strategy | Objective | Threshold | Validation precision | Validation recall | Validation F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | max_f1 | maximum validation positive-class F1 | 0.610 | 0.5911 | 0.7460 | 0.6596 |
| Random Forest | max_f1 | maximum validation positive-class F1 | 0.390 | 0.5542 | 0.7380 | 0.6330 |
| MLP Neural Network | max_f1 | maximum validation positive-class F1 | 0.590 | 0.5844 | 0.7406 | 0.6533 |

- Logistic Regression: The threshold was chosen on validation predictions and frozen before the test partition was evaluated.
- Logistic Regression: ROC-AUC ranks across all thresholds; the threshold above is a separate operating point, not the AUC.
- Random Forest: The threshold was chosen on validation predictions and frozen before the test partition was evaluated.
- Random Forest: ROC-AUC ranks across all thresholds; the threshold above is a separate operating point, not the AUC.
- MLP Neural Network: The threshold was chosen on validation predictions and frozen before the test partition was evaluated.
- MLP Neural Network: ROC-AUC ranks across all thresholds; the threshold above is a separate operating point, not the AUC.

Threshold sweeps and the frozen choice are stored in [threshold_analysis.json](threshold_analysis.json) and plotted in `threshold_analysis.png`.

## 5. Held-out test results

Positive-class metrics at each family's frozen threshold:

| Family | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.610 | 0.7708 | 0.5537 | 0.7032 | 0.6196 | 0.8380 | 0.6510 | 0.1691 | 823 | 212 | 111 | 263 |
| Random Forest | 0.390 | 0.7566 | 0.5312 | 0.7059 | 0.6062 | 0.8213 | 0.6186 | 0.1498 | 802 | 233 | 110 | 264 |
| MLP Neural Network | 0.590 | 0.7722 | 0.5575 | 0.6872 | 0.6156 | 0.8342 | 0.6270 | 0.1696 | 831 | 204 | 117 | 257 |

Confusion-matrix counts reconcile to the number of test rows (1409).

Secondary view at the default 0.5 threshold (not used for selection):

| Family | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7388 | 0.5050 | 0.8075 | 0.6214 | 0.8380 |
| Random Forest | 0.7700 | 0.5672 | 0.5642 | 0.5657 | 0.8213 |
| MLP Neural Network | 0.7495 | 0.5184 | 0.7914 | 0.6265 | 0.8342 |

### Cross-validation stability

| Family | Folds | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| not run | 0 | — | — | — | — | — |

CV values are means ± standard deviations across folds drawn from train+validation rows at the 0.5 reference threshold. They are a stability estimate, not the primary evidence.

Cross-validation was disabled for this run.

## 6. Figures

| Figure | File |
| --- | --- |
| Confusion matrix per family | [Logistic Regression](confusion_matrix_logistic_regression.png), [Random Forest](confusion_matrix_random_forest.png), [MLP Neural Network](confusion_matrix_mlp.png) |
| ROC curves | [roc_curves.png](roc_curves.png) |
| Precision-recall curves | [precision_recall_curves.png](precision_recall_curves.png) |
| Metric comparison | [metric_comparison.png](metric_comparison.png) |
| Threshold analysis | [threshold_analysis.png](threshold_analysis.png) |
| Probability distributions | [probability_distributions.png](probability_distributions.png) |

## 7. Model selection

**Selected family:** MLP Neural Network (via validation `f1`, threshold 0.590).

Selected the required primary model MLP Neural Network using validation f1 (0.6533); other families are comparison baselines. The test partition was not used to choose the model.

- The configured primary family is fixed by the product objective.
- Its threshold is chosen using validation rows only.

| Rank | Family | Validation metric | Validation recall | Validation ROC-AUC | Test metric |
| --- | --- | --- | --- | --- | --- |
| 1 | Logistic Regression | 0.6596 | 0.7460 | 0.8598 | 0.6196 |
| 2 | MLP Neural Network | 0.6533 | 0.7406 | 0.8556 | 0.6156 |
| 3 | Random Forest | 0.6330 | 0.7380 | 0.8473 | 0.6062 |

The `Test metric` column is shown for transparency only; it did not participate in the choice.

## 8. Error analysis and practical interpretation

A false positive is a retained customer predicted to churn (wasted retention spend). A false negative is a churner predicted to stay (missed save opportunity). Recall drives missed churners; precision drives outreach cost.

| Family | False positives | False negatives | FP rate of negatives | FN rate of positives | Mean P(FP) | Mean P(FN) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 212 | 111 | 20.5% | 29.7% | 0.7450 | 0.4012 |
| Random Forest | 233 | 110 | 22.5% | 29.4% | 0.5956 | 0.2384 |
| MLP Neural Network | 204 | 117 | 19.7% | 31.3% | 0.7457 | 0.4041 |

No customer identifiers or raw customer records are included in this report; the analysis is aggregate.

### Observed strengths and weaknesses

- **Logistic Regression** (threshold 0.610): observed strengths — highest test F1 (0.6196); highest test ROC-AUC (0.8380). Observed weaknesses — no trailing test metric.
- **Random Forest** (threshold 0.390): observed strengths — highest test recall (0.7059); best probability calibration by Brier score (0.1498). Observed weaknesses — lowest test precision (0.5312); lowest test F1 (0.6062); lowest test ROC-AUC (0.8213).
- **MLP Neural Network** (threshold 0.590): observed strengths — highest test precision (0.5575). Observed weaknesses — lowest test recall (0.6872); weakest Brier score (0.1696).
- These relative statements describe this held-out split only; a different customer base or period may reorder them.

### Overfitting signals

| Family | Train F1 | Validation F1 | Test F1 | Train−val ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6391 | 0.6596 | 0.6196 | -0.0090 |
| Random Forest | 0.8726 | 0.6330 | 0.6062 | 0.1441 |
| MLP Neural Network | 0.6716 | 0.6533 | 0.6156 | 0.0189 |

Train scores are in-sample and are shown only to surface overfitting; they are not performance evidence.

## 9. Limitations and warnings

- Metrics describe one held-out split; they are not a guarantee of future churn behaviour or causal retention impact.
- A threshold selected to maximise validation F1 is a statistical choice, not automatically the best business decision; contact costs and capacity are modeled in later phases.
- The benchmark compares model families on one dataset; results do not transfer automatically to other customer bases.
- Test metrics were computed once. Re-running this benchmark on a new split produces a new evaluation, not a re-tuned score for this one.
- Cross-validation was disabled for this run.
- Probability calibration is summarised by the Brier score only; no recalibration was fitted.

## 10. Reproducibility and artifacts

```bash
python -m churnwise.cli evaluate /
  --preprocessing-run-id 20261030T171452Z-wa-fn-usec-telco-customer-churn-066103c0-prep /
  --mlp-run-id 20261030T171452Z-wa-fn-usec-telco-customer-churn-066103c0-mlp
```

| Artifact | Path |
| --- | --- |
| Structured metrics | `artifacts/evaluation/telco-mlp-primary-evaluation/metrics.json` |
| Curve data | `artifacts/evaluation/telco-mlp-primary-evaluation/curves.json` |
| Threshold analysis | `artifacts/evaluation/telco-mlp-primary-evaluation/threshold_analysis.json` |
| Saved predictions | `artifacts/evaluation/telco-mlp-primary-evaluation/predictions.npz` |
| Run manifest | `artifacts/evaluation/telco-mlp-primary-evaluation/run_manifest.json` |

`python -m churnwise.cli evaluate --regenerate telco-mlp-primary-evaluation` recomputes every metric and figure from the saved probabilities without refitting a model or rescoring test rows.

Environment: python 3.13.0, numpy 2.5.4, pandas 2.3.3, scikit_learn 1.7.2, tensorflow 2.21.0, keras 3.15.1
