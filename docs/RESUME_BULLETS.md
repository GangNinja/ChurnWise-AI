# ChurnWise AI — Resume Bullets

Use only the bullets that match the role. Every metric below comes from the
committed benchmark, the post-Phase-13 validation reports or the recorded test
runs summarized in [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md).

The reported model metrics are **held-out test results from one split**, not
validation results and not production outcomes.

---

## Primary bullets

- **Built an end-to-end customer-churn ML platform in Python** covering dataset
  discovery and provenance, validation, leakage-safe preprocessing, model
  comparison, SHAP explainability, risk scoring, retention simulation and an
  five-page Streamlit retention product, orchestrated as a resumable seven-stage pipeline.

- **Implemented leakage-safe ML engineering** with seeded 60/20/20 stratified
  splits, train-only imputation/encoding/scaling, fold-safe cross-validation,
  persisted preprocessing pipelines and model artifacts, and validation-only
  model/threshold selection.

- **Benchmarked three model families on OpenML 46915** (5,000 rows) and selected
  Random Forest on validation F1 0.7985; it achieved held-out test F1 **0.8718**,
  ROC-AUC **0.9383**, PR-AUC **0.8973** and Brier **0.0390**, versus Logistic
  Regression 0.6350 / 0.8741 / 0.5717 and an MLP 0.7208 / 0.9266 / 0.7801.

- **Delivered model explainability with SHAP** using TreeSHAP, LinearSHAP and
  DeepSHAP, grouped transformed contributions back to source features, verified
  additivity, and generated 1,000 per-customer explanation artifacts in a
  separate Phase 9 risk run.

- **Built risk and retention-simulation services** with probability tiers
  separate from the 0.45 classification threshold, batch CSV validation, safe
  exports and provenance manifests; the simulator computes budget-constrained
  expected prevented churn from model probabilities while marking financial
  results as assumptions, not observed outcomes.

- **Validated the full-profile pipeline end to end**: all seven stages completed
  in 328.259 s with a 613.9 MiB peak, and the selected Random Forest reproduced
  the committed held-out benchmark metrics. A separately trained OpenML 42178
  experiment reached test ROC-AUC **0.8380** with Logistic Regression, without
  claiming model transfer.

- **Evaluated probability calibration without test leakage**: validation-only
  sigmoid selection improved held-out Brier **0.0390 → 0.0306**, log loss
  **0.1684 → 0.1337** and ECE **0.0686 → 0.0301** with unchanged ROC/PR; the
  calibrated estimator was serialized and reloaded but deliberately not
  integrated into the production candidate.

- **Designed and tested a focused five-page Streamlit retention product** over
  persisted artifacts, with cached reads, explicit action controls, honest empty
  states and no automatic downloads or training on page render.

- **Established reliability evidence**: a 634-passing default suite plus 7
  opt-in live-network tests and a 10-test Playwright suite covering all five
  pages, risk filtering/CSV download, stored customer detail, simulator,
  individual assessment, batch scoring/export and empty-artifact states.

---

## Short version (2 bullets)

- Built a tested, end-to-end churn ML platform (discovery → validation →
  leakage-safe preprocessing → MLP/baselines → SHAP → risk scoring → campaign
  simulation) with a Streamlit dashboard and reproducible artifact/provenance
  tracking.

- Benchmarked Random Forest, Logistic Regression and an MLP on a 5,000-row public
  churn dataset; selected on validation F1 and reported held-out test F1 0.8718 /
  ROC-AUC 0.9383, with per-customer SHAP explanations and explicit simulation
  assumptions.

---

## Strongest single-line variant

> Built and validated an end-to-end churn ML platform in Python with
> leakage-safe preprocessing, three-model benchmarking (held-out RF F1 0.8718 /
> ROC-AUC 0.9383), SHAP explanations, risk scoring, campaign simulation and an
> five-page Streamlit retention product tested with a 10-case Playwright suite.

---

## Do not claim

- Production deployment, real users, measured retention uplift or business
  impact.
- That calibration is part of the production model; it was evaluated separately.
- That the primary model transfers to another customer population; the second
  dataset used separately trained models.
- Causal drivers from SHAP values.
- State-of-the-art, tuned or externally validated model performance.
- Bitwise-reproducible MLP training.
- That the default test suite includes the live-network or browser tests; both
  are opt-in and skipped by default.
