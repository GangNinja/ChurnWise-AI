# ChurnWise AI — Phase 7 benchmark `phase13-final-live2-bench`

All metrics below were measured on the held-out test partition created in Phase 5 and scored once, after thresholds were frozen on validation data. No number is illustrative.

| Field | Value |
| --- | --- |
| Benchmark run | `phase13-final-live2-bench` |
| Preprocessing run | `phase13-final-live2-prep` |
| Dataset | `openml:46915` |
| Dataset SHA-256 | `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6` |
| Split fingerprint | `4b0c45db677fa9619de99b6a8b786a3da2c4da93824c46db65412a85829594dd` |
| Preprocessor hash | `3c31f13c266c21084050bb021e60a41d4145a2055b983eb03e2267fab9f52174` |
| Transformed features | 27 |
| Train / validation / test rows | 3000 / 1000 / 1000 |
| Test churn rows | 141 (14.1%) |
| Generated | 2026-10-23T01:33:35Z |
| Code version | `0.10.0` |
| Complete (all three families) | yes |

## 1. Methodology

- **Shared data:** Logistic Regression, Random Forest and the Phase 6 MLP all consume the same Phase 5 transformed matrices and the same split fingerprint recorded above.
- **Fitting scope:** baselines are fitted on the training partition only. The MLP artifact was trained on that same partition in Phase 6 and is loaded, not retrained.
- **Imbalance treatment:** all families receive the Phase 5 training class weights `{'0': 0.5822981366459627, '1': 3.5377358490566038}`.
- **Threshold policy:** each family's operating threshold is selected on validation predictions (`max_f1`) and frozen before the test partition is scored. The 0.5 default is also reported but never used for selection.
- **Selection criterion:** `f1` measured on validation, with the optional recall floor `None`. Test metrics are not an input.
- **Test isolation:** the test matrix was evaluated once (`test_evaluation_calls = 1`); no threshold, hyperparameter or model choice used test labels.
- **Cross-validation:** completed (3 folds on train+validation, preprocessing refitted per fold).

## 2. Model configurations

| Family | Source | Configuration | Train rows | Fit time (s) | Threshold |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | fitted_in_phase7 | LogisticRegression(C=1.0, solver=lbfgs, penalty=l2, max_iter=2000) | 3000 | 0.03 | 0.640 |
| Random Forest | fitted_in_phase7 | RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features=sqrt) | 3000 | 3.33 | 0.450 |
| MLP Neural Network | phase6_artifact | MLP 32/16 ReLU, dropout [0.2, 0.1], L2 0.0001, Adam 0.001, batch 64, epochs <= 24 | 3000 | 9.97 | 0.800 |

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
| Logistic Regression | max_f1 | maximum validation positive-class F1 | 0.640 | 0.5376 | 0.6549 | 0.5905 |
| Random Forest | max_f1 | maximum validation positive-class F1 | 0.450 | 0.8678 | 0.7394 | 0.7985 |
| MLP Neural Network | max_f1 | maximum validation positive-class F1 | 0.800 | 0.7615 | 0.5845 | 0.6614 |

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
| Logistic Regression | 0.640 | 0.8770 | 0.5459 | 0.7589 | 0.6350 | 0.8741 | 0.5717 | 0.1460 | 770 | 89 | 34 | 107 |
| Random Forest | 0.450 | 0.9650 | 0.9015 | 0.8440 | 0.8718 | 0.9383 | 0.8973 | 0.0390 | 846 | 13 | 22 | 119 |
| MLP Neural Network | 0.800 | 0.9220 | 0.7520 | 0.6667 | 0.7068 | 0.9221 | 0.7637 | 0.1091 | 828 | 31 | 47 | 94 |

Confusion-matrix counts reconcile to the number of test rows (1000).

Secondary view at the default 0.5 threshold (not used for selection):

| Family | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8090 | 0.4120 | 0.8298 | 0.5506 | 0.8741 |
| Random Forest | 0.9690 | 0.9365 | 0.8369 | 0.8839 | 0.9383 |
| MLP Neural Network | 0.8660 | 0.5143 | 0.8936 | 0.6528 | 0.9221 |

### Cross-validation stability

| Family | Folds | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 3/3 | 0.8128 ± 0.0136 | 0.4173 ± 0.0219 | 0.8074 ± 0.0028 | 0.5500 ± 0.0192 | 0.8685 ± 0.0133 |
| Random Forest | 3/3 | 0.9530 ± 0.0015 | 0.8978 ± 0.0250 | 0.7544 ± 0.0180 | 0.8196 ± 0.0045 | 0.9121 ± 0.0103 |
| MLP Neural Network | 3/3 | 0.8480 ± 0.0096 | 0.4789 ± 0.0185 | 0.8074 ± 0.0245 | 0.6007 ± 0.0080 | 0.8850 ± 0.0107 |

CV values are means ± standard deviations across folds drawn from train+validation rows at the 0.5 reference threshold. They are a stability estimate, not the primary evidence.

Each fold refits the Phase 5 column plan, imputers, scaler and encoder on its own training rows; validation rows are transform-only, so no fold sees its own validation statistics.

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

**Selected family:** Random Forest (via validation `f1`, threshold 0.450).

Selected Random Forest because it had the highest validation f1 (0.7985) at its validation-selected operating threshold. MLP Neural Network followed with 0.6614. The MLP was not the best model on the validation criterion; the benchmark does not assume it must win. The test partition was not used to choose the model.

- Selection uses validation predictions only, after thresholds were frozen.
- Accuracy alone is not the selection criterion; the configured metric is f1.

| Rank | Family | Validation metric | Validation recall | Validation ROC-AUC | Test metric |
| --- | --- | --- | --- | --- | --- |
| 1 | Random Forest | 0.7985 | 0.7394 | 0.8826 | 0.8718 |
| 2 | MLP Neural Network | 0.6614 | 0.5845 | 0.8913 | 0.7068 |
| 3 | Logistic Regression | 0.5905 | 0.6549 | 0.8699 | 0.6350 |

The `Test metric` column is shown for transparency only; it did not participate in the choice.

## 8. Error analysis and practical interpretation

A false positive is a retained customer predicted to churn (wasted retention spend). A false negative is a churner predicted to stay (missed save opportunity). Recall drives missed churners; precision drives outreach cost.

| Family | False positives | False negatives | FP rate of negatives | FN rate of positives | Mean P(FP) | Mean P(FN) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 89 | 34 | 10.4% | 24.1% | 0.8092 | 0.3694 |
| Random Forest | 13 | 22 | 1.5% | 15.6% | 0.5765 | 0.1789 |
| MLP Neural Network | 31 | 47 | 3.6% | 33.3% | 0.8871 | 0.5249 |

No customer identifiers or raw customer records are included in this report; the analysis is aggregate.

### Observed strengths and weaknesses

- **Logistic Regression** (threshold 0.640): observed strengths — no leading test metric. Observed weaknesses — lowest test precision (0.5459); lowest test F1 (0.6350); lowest test ROC-AUC (0.8741); weakest Brier score (0.1460).
- **Random Forest** (threshold 0.450): observed strengths — highest test recall (0.8440); highest test precision (0.9015); highest test F1 (0.8718); highest test ROC-AUC (0.9383); best probability calibration by Brier score (0.0390). Observed weaknesses — no trailing test metric.
- **MLP Neural Network** (threshold 0.800): observed strengths — no leading test metric. Observed weaknesses — lowest test recall (0.6667).
- These relative statements describe this held-out split only; a different customer base or period may reorder them.

### Overfitting signals

| Family | Train F1 | Validation F1 | Test F1 | Train−val ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6208 | 0.5905 | 0.6350 | 0.0039 |
| Random Forest | 0.9918 | 0.7985 | 0.8718 | 0.1174 |
| MLP Neural Network | 0.7289 | 0.6614 | 0.7068 | 0.0375 |

Train scores are in-sample and are shown only to surface overfitting; they are not performance evidence.

## 9. Limitations and warnings

- Metrics describe one held-out split; they are not a guarantee of future churn behaviour or causal retention impact.
- A threshold selected to maximise validation F1 is a statistical choice, not automatically the best business decision; contact costs and capacity are modeled in later phases.
- The benchmark compares model families on one dataset; results do not transfer automatically to other customer bases.
- Test metrics were computed once. Re-running this benchmark on a new split produces a new evaluation, not a re-tuned score for this one.
- Cross-validation refits preprocessing inside every fold, but the fold split is a single seeded 5-fold partition; it does not quantify split-to-split variance.
- Probability calibration is summarised by the Brier score only; no recalibration was fitted.

## 10. Reproducibility and artifacts

```bash
python -m churnwise.cli evaluate \
  --preprocessing-run-id phase13-final-live2-prep \
  --mlp-run-id phase13-final-live2-mlp
```

| Artifact | Path |
| --- | --- |
| Structured metrics | `.phase13_final2\artifacts\evaluation\phase13-final-live2-bench\metrics.json` |
| Curve data | `.phase13_final2\artifacts\evaluation\phase13-final-live2-bench\curves.json` |
| Threshold analysis | `.phase13_final2\artifacts\evaluation\phase13-final-live2-bench\threshold_analysis.json` |
| Saved predictions | `.phase13_final2\artifacts\evaluation\phase13-final-live2-bench\predictions.npz` |
| Run manifest | `.phase13_final2\artifacts\evaluation\phase13-final-live2-bench\run_manifest.json` |

`python -m churnwise.cli evaluate --regenerate phase13-final-live2-bench` recomputes every metric and figure from the saved probabilities without refitting a model or rescoring test rows.

Environment: python 3.13.0, numpy 2.5.4, pandas 2.3.3, scikit_learn 1.7.2, tensorflow 2.21.0, keras 3.15.1

