# ChurnWise AI — Resume Bullets

Use only the bullets that match the role. Every number below is traceable to a
committed artifact or test run recorded in
[FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md).

---

## Primary bullets

- **Built an end-to-end customer-churn intelligence platform in Python** that
  discovers and acquires a public dataset, validates it, performs leakage-safe
  preprocessing, trains an MLP plus two baselines, computes SHAP explanations,
  scores a customer population and simulates retention campaigns — orchestrated
  as a resumable seven-stage pipeline with a Streamlit dashboard.

- **Implemented reproducible ML engineering**: seeded train/validation/test
  splits, train-only fitting of imputers/encoders/scalers, a Keras MLP with ReLU
  hidden layers, sigmoid output, binary cross-entropy, Adam, early stopping and
  checkpoint restore, plus persisted preprocessing pipelines and model artifacts.
  Verified with 624 offline tests and 7 opt-in live-network tests.

- **Benchmarked three models on OpenML dataset 46915** (5,000 rows, benchmark
  profile) with a validation-only selection rule: Random Forest was selected at
  validation F1 0.7985 and achieved **test F1 0.8718 / ROC-AUC 0.9383 / PR-AUC
  0.8973**, versus Logistic Regression 0.6350 / 0.8741 / 0.5717 and MLP 0.7208 /
  0.9266 / 0.7801. Reported test metrics separately from validation metrics.

- **Delivered model explainability with SHAP**: exact TreeSHAP for the selected
  Random Forest, DeepSHAP for the MLP, grouped transformed contributions back to
  original features, verified additivity, and generated 1,000 per-customer local
  explanation artifacts for a 1,000-customer risk population.

- **Built churn-risk and retention-simulation services** with probability tiers
  distinct from the 0.45 classification threshold, batch CSV validation, safe
  exports and provenance manifests; the simulator computes budget-constrained
  expected prevented churn from real model probabilities while explicitly
  labelling financial results as assumptions, not observed outcomes.

- **Engineered for auditability**: SHA-256 dataset integrity checks, content-
  addressed artifact paths, run manifests with configuration digests, explicit
  failure codes, dry-run cleanup, and live end-to-end runs (66 s initial, 105 s
  post-fix) that reproduced identical metrics on a real public dataset.

- **Built and tested an 11-page Streamlit dashboard** over persisted artifacts,
  with cached reads, explicit action buttons for training/scoring/simulation,
  honest empty states, and no automatic downloads or training on page render.

---

## Short version (2 bullets)

- Built a tested, end-to-end churn ML platform (data acquisition → validation →
  preprocessing → MLP/baselines → SHAP → risk scoring → campaign simulation) with
  a Streamlit dashboard and reproducible artifact/provenance tracking.

- Benchmarked Random Forest, Logistic Regression and an MLP on a 5,000-row public
  churn dataset; selected on validation PR-AUC and reported separate test
  PR-AUC/ROC-AUC, with per-customer SHAP explanations and explicit simulation
  assumptions.

---

## Do not claim

- Production deployment, real users, or business impact.
- Causal drivers from SHAP values.
- Measured retention uplift or revenue from the simulator.
- Bitwise-reproducible MLP training.
- State-of-the-art or tuned model performance.
