# Telco MLP upload examples

These files are deterministic synthetic customer examples for demonstrating
the Telco MLP upload workflow. They do not copy customer rows from the teacher
dataset. `Churn` is an illustrative label and is retained only for audit;
the trained model never receives it as an input feature.

| File | Rows | Folder | Upload behavior |
| --- | ---: | --- | --- |
| `testing1.csv` | 1,000 | `proper_datasets/` | Complete Telco feature schema |
| `testing2.csv` | 1,200 | `proper_datasets/` | Complete Telco feature schema |
| `testing3.csv` | 1,500 | `proper_datasets/` | Complete Telco feature schema |
| `testing4.csv` | 1,800 | `proper_datasets/` | Complete Telco feature schema |
| `testing5.csv` | 2,200 | `proper_datasets/` | Complete Telco feature schema |
| `testing6.csv` | 3,000 | `proper_datasets/` | Complete Telco feature schema |

`improper_datasets/testingN.csv` has the same synthetic rows and row count as
its matching proper file, but omits one required MLP feature. The dashboard
should identify the missing column and stop before scoring.

Choose a bundled file from **Executive Dashboard → Upload & predict → Or try a
bundled dataset**. The same CSVs can be uploaded from disk. Regenerate all
examples with `python scripts/generate_testing_datasets.py`.
