# Dashboard screenshots

These are genuine captures of the local Streamlit application reading persisted
artifacts. No metrics, labels or values were edited.

## Five-page redesign (current)

Captured at a 1440 × 1100 desktop viewport; narrow checks use 1024 × 1000.

| File | Page |
| --- | --- |
| `redesign-executive.png` | Executive Dashboard, header and KPI scope |
| `redesign-executive-2.png` | Executive Dashboard, interactive charts and insights |
| `redesign-executive-3.png` | Executive Dashboard, customer table and model strip |
| `redesign-executive-filtered.png` | Executive Dashboard after a tier-chart selection |
| `redesign-executive-row-selected.png` | Executive Dashboard with a table row selected |
| `redesign-explorer.png` | Customer Risk Explorer, stored customer profile and assessment |
| `redesign-explorer-2.png` | Customer Risk Explorer, SHAP contributions and metadata |
| `redesign-laboratory.png` | Model Laboratory, family comparison, confusion matrix and ROC |
| `redesign-laboratory-2.png` | Model Laboratory, cross-validation stability |
| `redesign-laboratory-3.png` | Model Laboratory, cross-validation and benchmark limitations |
| `redesign-workbench.png` | Retention Workbench, filters and budget fit |
| `redesign-workbench-table.png` | Retention Workbench, prioritised list and exports |
| `redesign-simulator.png` | Campaign Simulator, header and projected outcome |
| `redesign-simulator-2.png` | Campaign Simulator, allocation and scenario comparison |
| `redesign-executive-narrow.png`, `redesign-explorer-narrow.png`, `redesign-laboratory-narrow.png`, `redesign-workbench-narrow.png`, `redesign-simulator-narrow.png` | 1024 px layout checks of all five pages |

Captured against the local project artifacts (`20261022T155236Z-openml_46915-risk-final`,
benchmark `20261021T145336Z-openml_46915-bench-04595d7b`). At both widths the
browser reported no page errors and no page-level horizontal overflow
(`main.scrollWidth == main.clientWidth`). The Executive drill-down and row
selection were exercised interactively before capture. These are documentation
evidence, not a substitute for the AppTest or browser suites.

## Historical captures (Phase 11–13)

| Files | Run shown | Source |
| --- | --- | --- |
| `phase13-overview.png`, `phase13-risk.png`, `phase13-simulator.png` | `phase13-final-live` (pre-fix) | Phase 13 live artifact tree; metrics, tiers and projection are identical to the post-fix run |
| `phase11-*.png` | Phase 11 fixture/live verification tree | Phase 11 report |

The post-fix evidence in
[`docs/evidence/phase13-live2`](../evidence/phase13-live2/README.md) is the
authoritative machine-readable record for those captures (RF test F1 0.8718,
ROC-AUC 0.9383; 1,000 scored; expected prevented churn 32.8334).
