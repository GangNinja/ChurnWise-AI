# Telco MLP Learning Dashboard — Design Spec

**Status:** Approved in chat; awaiting user review of this written spec
**Date:** 2026-10-30
**Scope:** Make the supplied Telco Customer Churn CSV the default local training/evaluation dataset and make the MLP, its learning objectives, and its held-out evaluation the center of ChurnWise AI. Preserve customer exploration, retention workbench, and campaign simulation as supporting experiences.

## User goals

The application must demonstrate, with the actual model and run artifacts, how a multilayer perceptron predicts Telco customer churn. A student should be able to point to the input layer, hidden layers, ReLU activations, sigmoid output, binary cross-entropy loss, backpropagation, Adam optimizer, and held-out accuracy, precision, recall, F1, and confusion matrix. The supplied CSV, not the previous OpenML dataset, is the primary source for the default training/evaluation run.

## Source data and preprocessing

- Expected local source: `data/source/WA_Fn-UseC_-Telco-Customer-Churn.csv`. The raw source stays in the existing ignored `data/` directory and is not committed; setup instructions explain how to place the teacher-provided file there.
- The verified input has 7,043 rows, 21 columns, and target `Churn` with 5,174 `No` and 1,869 `Yes` rows. `TotalCharges` is textual and has 11 whitespace-only values; parsing must turn these into missing numeric values so the existing training-fitted imputer can handle them.
- Drop `customerID` from predictive features while retaining it only as a customer reference where needed. Normalize the target to binary `No=0`, `Yes=1`; do not use target labels as features.
- Use a seeded, stratified, disjoint train/validation/test split. Fit all imputation, encoding, scaling, and any class balancing on training rows only. Select any classification threshold with validation rows only. Touch the held-out test split once for final reporting.
- Record the source path label, SHA-256, target mapping, row/column counts, dropped identifier, missing-value handling, split sizes/fingerprint, and preprocessing/model configuration in run artifacts.

## Model and prediction contract

- The shipped predictor is the existing TensorFlow/Keras MLP, not the currently benchmark-selected Random Forest. The operational pipeline must train, evaluate, explain where supported, and score with that MLP. Other model families may remain as explicitly labelled optional baselines, but may not silently become the primary predictor or supply the main dashboard scoring run.
- The instructional architecture view is generated from persisted model metadata: transformed feature input → configurable Dense hidden layers with ReLU → one sigmoid churn-probability output. Training configuration and diagrams must agree with the serialized model.
- Training uses binary cross-entropy and Adam. TensorFlow/Keras autodifferentiation performs backpropagation. The interface explains these steps accurately; conceptual diagrams are explanatory and must not imply that individual gradients were recorded if they were not.
- A predicted churn class is determined from the probability and the saved validation-selected threshold. A prediction is a model estimate, not an observed outcome.

## Dashboard experience

Keep the current product identity and visual system; reshape navigation and content around an instructional workflow:

1. **Executive Dashboard:** primary Telco source/run, population summary, churn probability/risk distributions, key MLP evaluation indicators, and clear navigation into model details. Historical `Churn` prevalence and model-predicted churn must be visually distinguished.
2. **MLP Learning Studio:** new primary page showing the saved network architecture, layer dimensions/activations, ReLU and sigmoid behavior, BCE loss intuition, backpropagation flow, Adam’s role, actual training/validation loss and metric curves, and the matching model/run configuration. Distinguish explanatory content from measured run evidence.
3. **Model Evaluation:** make test accuracy, precision, recall, F1 and a labeled confusion matrix easy to find. Include class counts, threshold, split provenance, and relevant curves when present. Mark the test set as held out and never use it to tune the threshold.
4. **Customer Risk Explorer:** retain individual and batch scoring, explanations where the MLP explainer is available, and direct selection of persisted MLP scoring runs.
5. **Retention Workbench and Campaign Simulator:** retain these secondary operational tools, wired to the same MLP scoring run. Continue labeling projected campaign effects as assumptions, not causal or measured lift.

Navigation should present the learning/evaluation pages before the applied tools. Avoid displaying obsolete RF-selected run data as if it belongs to the Telco MLP run; all visible values must share a dataset/model/run lineage.

## Run and artifact behavior

- Provide a repeatable default command and dashboard action using the local teacher-provided source. If the file is absent or malformed, show the expected path and a direct remediation message rather than silently falling back to discovery, fixtures, or another dataset.
- A new training run creates a new immutable run ID and does not overwrite earlier evaluation/model artifacts. Run selectors and labels identify dataset and model family clearly.
- Default artifact reads should select the latest compatible Telco MLP run, and should not mix scores, explanations, evaluation metrics, or campaign scenarios from other dataset hashes or model digests.
- The previous benchmark runs remain historical artifacts. The redesigned interface must identify their dataset/model family accurately and keep them out of the default Telco MLP experience.

## Documentation and validation

Update the README and relevant run/dashboard/model documentation to make the supplied Telco CSV the default source, state the expected local path, show training and dashboard commands, explain the requested MLP concepts, and distinguish held-out metrics from customer-population predictions. Retire claims that the previous OpenML Random Forest is the current primary model while preserving accurate historical benchmark notes.

Focused automated checks should cover source validation and normalization, customerID exclusion, whitespace `TotalCharges` coercion and train-only imputation, stratified split isolation, MLP family selection and scoring, architecture metadata, required evaluation metrics/confusion matrix, artifact/run lineage, and dashboard empty/error states. Run relevant unit/integration tests and a full default pipeline run against the provided CSV; inspect that dashboard selectors resolve to the resulting Telco MLP run.

## Non-goals and guardrails

- Do not claim causal retention uplift, production deployment, or model generalization beyond this dataset.
- Do not use customerID as a predictive signal.
- Do not tune using held-out test metrics or describe validation results as test results.
- Do not delete historical artifacts or unrelated dashboard capabilities.
- Do not track the teacher-provided raw CSV in Git; it remains a local input under ignored `data/`.
- Do not fabricate gradient values or claim TensorFlow’s internal gradients were persisted.

## Design review checklist

- [x] The MLP, rather than a baseline, is the operational predictor.
- [x] All requested learning objectives map to visible concepts or measured evidence.
- [x] Dataset-specific missing values and identifier leakage are addressed.
- [x] Split/threshold/test boundaries are explicit and auditable.
- [x] Existing customer/retention pages remain in scope as secondary pages.
- [x] Missing-data, artifact-lineage, and historical-run behavior are specified.
- [x] Raw dataset is local and excluded from Git.
