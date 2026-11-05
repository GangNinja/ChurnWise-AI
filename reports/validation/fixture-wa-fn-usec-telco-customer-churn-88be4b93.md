# Validation report — fixture-wa-fn-usec-telco-customer-churn-88be4b93

- **Status:** `pass_with_warnings`
- **Gate:** `pass_with_warnings` (can preprocess: `true`)
- **Generated:** 2026-10-30T17:33:55Z
- **Validator:** 0.10.0
- **Artifact:** `data/raw/teacher_provided/fixture-wa-fn-usec-telco-customer-churn-88be4b93/2026-10-30/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Artifact SHA-256:** `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- **Manifest SHA-256:** `20c7c5d29b43258b9443e2847b4e7d765c385826ab9bfb0dee5dbc126d3b716d`

## Summary

| Field | Value |
| --- | --- |
| n_rows | 7043 |
| n_cols | 21 |
| missing_cells | 11 |
| duplicate_rows | 0 |
| target_rate | 0.2653698707936959 |
| id_column | customerID |
| revenue_proxy | MonthlyCharges |
| usable_features | 19 |

## Target

- Column: `Churn`
- Classes: ['No', 'Yes']
- Mapping: {'No': 0, 'Yes': 1}
- Positive label (churn): `Yes`
- Positive rate: 0.2653698707936959
- Confidence: `high`
- Method: churn-style target name: conventional positive label is churn

## Checks

| ID | Check | Result | Severity | Detail |
| --- | --- | --- | --- | --- |
| V1 | file_integrity_and_readability | PASS | fatal | artifact exists, is readable and matches its recorded digest |
| V2 | structural_integrity | PASS | fatal | 7043 rows x 21 columns |
| V3 | target_identified | PASS | fatal | selected target 'Churn' |
| V4 | target_binary | PASS | fatal | target has two classes: ['No', 'Yes'] |
| V5 | target_interpretation | PASS | fatal | 1 = churn maps from 'Yes'; confidence high |
| V6 | target_balance | PASS | warning | positive rate 26.537% |
| V7 | duplicate_columns | PASS | warning | no duplicate headers or columns |
| V8 | duplicate_rows | PASS | warning | no duplicate rows |
| V9 | missingness | FAIL | warning | 11 missing cells |
| V10 | low_information_features | PASS | warning | no constant or near-constant columns |
| V11 | identifier_and_cardinality | FAIL | warning | 1 identifier/cardinality warning(s); 3 numeric and 16 categorical usable features |
| V12 | target_leakage | PASS | warning | no suspected target leakage |

## Roles

- Target: `Churn`
- ID: `customerID`
- Numeric: ['tenure', 'MonthlyCharges', 'TotalCharges']
- Categorical: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

## Capabilities

| Capability | Available | Source |
| --- | --- | --- |
| has_tenure | True | tenure |
| has_monthly_charge | True | MonthlyCharges |
| has_total_charges | True | TotalCharges |
| has_contract | True | Contract |
| has_payment_method | True | PaymentMethod |
| has_internet_service | True | InternetService |
| has_support_calls | False | — |
| has_usage_minutes | False | — |
| has_plan | False | — |
| has_senior_citizen | True | SeniorCitizen |
| has_gender | True | gender |
| has_revenue_proxy | True | MonthlyCharges |

## Provenance

- run_id: `20261030T173355Z-wa-fn-usec-telco-customer-churn-775cb55c`
- dataset_id: `fixture-wa-fn-usec-telco-customer-churn-88be4b93`
- retrieved_at: `2026-10-30T17:33:55Z`
- download_url: `https://fixtures.churnwise.invalid/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- license: `Teacher-provided dataset; upstream licence not recorded (unverified)`
- manifest_path: `data/manifests/fixture-wa-fn-usec-telco-customer-churn-88be4b93.json`
- acquisition_attempt: `1`
- candidate_rank: `None`

## Warnings

- Telco source normalization trimmed text, mapped Churn labels to canonical No/Yes, and coerced 11 blank TotalCharges values to missing.

## Notes

- Validation is descriptive only; no imputation, scaling, encoding, split or model fitting was performed.
- Missing target rows are excluded only in the in-memory normalized frame; the acquired artifact is untouched.
- Identifier columns are retained for output and excluded from predictive features.

## EDA artifacts

- Profile: [reports/eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93.json](../eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93.json)
- Plot: [reports/eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93.png](../eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93.png)
- Plot: [reports/eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_numeric.png](../eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_numeric.png)
- Plot: [reports/eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_correlation.png](../eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_correlation.png)
- Plot: [reports/eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_conditional.png](../eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_conditional.png)
- Plot: [reports/eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_categorical.png](../eda/fixture-wa-fn-usec-telco-customer-churn-88be4b93_categorical.png)
