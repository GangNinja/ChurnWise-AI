# ChurnWise upload test datasets

These generated telecom customer CSVs demonstrate the dashboard's batch upload
and schema validation. They contain no real customer records. A fitted
preprocessing pipeline/model for the matching telecom schema (OpenML `46915`)
is required; the small `tests/fixtures/mini_churn.csv` workflow has a different
schema. See [How to Run](../docs/HOW_TO_RUN.md#try-the-bundled-customer-csvs)
for the model prerequisite and dashboard steps.

Each file includes a generated `CustomerChurned` example label. The upload
scorer ignores this outcome column: predictions are model outputs, not measured
outcomes or accuracy evidence.

| Dataset | Data rows | Valid schema | Deliberately missing required field |
| --- | ---: | --- | --- |
| `testing1.csv` | 1,000 | `proper_datasets/` | `improper_datasets/`: `total_day_charge` |
| `testing2.csv` | 1,200 | `proper_datasets/` | `improper_datasets/`: `international_plan` |
| `testing3.csv` | 1,500 | `proper_datasets/` | `improper_datasets/`: `total_eve_charge` |
| `testing4.csv` | 1,800 | `proper_datasets/` | `improper_datasets/`: `voice_mail_plan` |
| `testing5.csv` | 2,200 | `proper_datasets/` | `improper_datasets/`: `number_customer_service_calls` |
| `testing6.csv` | 3,000 | `proper_datasets/` | `improper_datasets/`: `total_night_charge` |

Use `proper_datasets/testing4.csv` for a larger valid upload demonstration. Use
`improper_datasets/testing4.csv` to show that a missing required field is
reported before any rows are scored. The same files are available in the
dashboard's bundled-dataset selector as `Proper · testingN.csv` and
`Improper · testingN.csv`.
