# Interactive Dashboard — Architecture and Usage

Status: **implemented and verified for Phase 11**, integrated with the Phase 12
pipeline, and re-verified in Phase 13. AppTest fixtures, the real OpenML `46915`
artifact tree and browser screenshots cover all eleven pages. Phase 12 measured
the first render at 1.48 s; Phase 13 measured 1.937 s against the final pipeline
artifacts, with server HTTP readiness under 2 s. See
[PERFORMANCE.md](PERFORMANCE.md), [PHASE12_REPORT.md](PHASE12_REPORT.md) and
[FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md).

> The dashboard reads persisted artifacts. It does not invent metrics, sample
> data, training curves or campaign outcomes. Where an artifact is missing, the
> page states what is unavailable and names the command that produces it.

## 1. Stack and startup

| Component | Verified version |
| --- | --- |
| Python | 3.13 |
| Streamlit | 1.66.1 |
| Plotly | 6.9.0 |
| pandas | 2.3.3 |
| NumPy | 2.5.4 |

Install into the project virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the app from the repository root:

```bash
python -m streamlit run churnwise/dashboard/app.py
```

For a headless local check:

```bash
python -m streamlit run churnwise/dashboard/app.py \
  --server.headless true --server.address 127.0.0.1 --server.port 8507
curl -I http://127.0.0.1:8507/
```

Streamlit prints the local URL. The app is a single process; it does not start a
database, API, worker or remote service.

### Configuration

The dashboard uses the same `Settings` object and environment variables as the
CLI (`CHURNWISE_DATA_DIR`, `CHURNWISE_ARTIFACT_DIR`, `CHURNWISE_REPORT_DIR`,
`CHURNWISE_OFFLINE`, `CHURNWISE_RISK_TIERS`, …). No secrets are required.

## 2. Page map

| # | Navigation label | Module | Primary artifacts |
| --- | --- | --- | --- |
| 1 | Overview | `pages/overview.py` | validation, EDA, benchmark, risk, simulation, model metadata |
| 2 | Data discovery | `pages/discovery.py` | `reports/acquisition/{candidates,attempts}.json` |
| 3 | Data quality | `pages/quality.py` | validation report, EDA profile |
| 4 | Model training | `pages/training.py` | `artifacts/prepared/*`, `artifacts/models/*/mlp/*` |
| 5 | Model evaluation | `pages/evaluation.py` | `artifacts/evaluation/<run>/{metrics,curves}.json` |
| 6 | Explainability | `pages/explainability.py` | `artifacts/explanations/<run>/{global,run_manifest}.json`, local files |
| 7 | Risk intelligence | `pages/risk.py` | `artifacts/predictions/<run>/churn_scores.parquet` + summary |
| 8 | Assess records | `pages/assessment.py` | benchmark-selected model, Phase 5 schema, Phase 9 service |
| 9 | Customer detail | `pages/customer.py` | risk table, `drivers.json`, matched local explanation |
| 10 | Retention simulator | `pages/simulator.py` | `artifacts/simulations/<run>/`; live Phase 10 engine for new projections |
| 11 | Reliability & limits | `pages/reliability.py` | all metadata, validation checks, warnings, limitations |

Every page header renders four identity fields: run id, model family, dataset
hash prefix and generated-at timestamp. Missing fields are shown as
`not available` rather than omitted.

## 3. Service integration boundary

The dashboard is a presentation layer. Model loading, preprocessing, scoring,
tier assignment, SHAP, ranking, campaign allocation and projection math are
called from the existing phase services; they are never reimplemented in a page.

| Area | Existing interface used |
| --- | --- |
| Discovery | `churnwise.discovery.discover` + adapters + `reporting.write_reports` |
| Acquisition | `AcquisitionCoordinator.acquire` with ranked candidates and fallback |
| Validation | `reports/validation/*.json`, EDA JSON |
| Training | `PreprocessingCoordinator.load`, `MLPTrainer.train` (`persist=True`) |
| Evaluation | metrics/curves JSON (read-only) |
| Explanations | `global.json`, `run_manifest.json`, stored local JSON/PNG |
| Record scoring | `RiskService.assess_customer`, `RiskService.assess_batch` |
| Population explorer | stored Phase 9 Parquet + summary |
| Simulation | `SimulationService.run_scenario(..., persist=False)` for in-session projections; persisted results are read-only |

`churnwise/dashboard/actions.py` contains the only side-effecting calls. Each is
behind an explicitly labelled button. Importing a page or rerunning it does not
start discovery, download, training, scoring or a simulation.

### Acquisition fallback

The preferred candidate is selected from the current ranking. With “Try
lower-ranked eligible candidates if this one fails” enabled (the default), the
page passes that candidate **and every lower-ranked eligible candidate** to
`AcquisitionCoordinator`. Phase 3 performs the attempts, cache checks and
fallback ordering; the dashboard only displays the returned attempts.

### Individual and batch assessment

The **Assess records** page builds its fields from
`artifacts/preprocessors/<run>/schema.json` and uses the model selected by the
latest Phase 7 benchmark.

- Individual: text fields for numeric columns, schema categories for
  categoricals. Blank values are passed to the fitted pipeline, which applies
  its recorded imputation. The page never substitutes zero or a modal value.
- Optional local SHAP: an unchecked-by-default checkbox calls the Phase 8
  explanation engine for that one record.
- Batch: CSV upload, 10 MiB / 20,000-row limits, required-column validation,
  optional identifier column hashed by Phase 9, row-level rejection reasons,
  and CSV/JSON exports. Batch scoring does not compute SHAP.
- Raw input values are not written to the result payload or displayed in error
  messages. Phase 9 references are positional or hashed.

## 4. State, caching and artifact freshness

`data_access.py` uses:

- `st.cache_data` keyed by `(path, st_mtime_ns, st_size)` for JSON and Parquet.
  A rewritten artifact changes the token and invalidates its cached value.
- `st.session_state` only for UI-owned results: the last training outcome, last
  individual/batch assessment, last in-session simulation, notices and errors.
  These are not written back as project artifacts.
- An explicit **Refresh artifact cache** sidebar button for external pipeline
  runs.
- Success paths call `st.cache_data.clear()` before rerunning so the page sees
  the artifact just written.

Expensive training and simulation results are held in session state so a
routine rerun does not resubmit them. A new batch upload with a different
SHA-256 clears the previous batch result.

## 5. Error handling

User-facing failures use stable codes:

| Code range | Source |
| --- | --- |
| `CW-UI-001` | unreadable/corrupt artifact |
| `CW-UI-003…006` | discovery, candidate, acquisition or attempt-write failure |
| `CW-UI-007` | training configuration invalid |
| `CW-UI-008` | prepared dataset missing |
| `CW-UI-009` | training execution failure |
| `CW-UI-010/011` | individual/batch assessment wrapper failure |
| `CW-RISK-*` | Phase 9 input, model, configuration or prediction failure |
| `CW-SIM-*` | Phase 10 scenario/assumption failure |

Messages contain schema names, counts and error classes, not raw customer rows
or tracebacks. Missing artifacts render an amber empty state with the exact CLI
command; they never render a zero, placeholder chart or synthetic metric.

## 6. Exports and privacy

- Risk and assessment CSV exports apply the Phase 9 spreadsheet-formula guard
  (`=`, `+`, `-`, `@`, tab and carriage return are prefixed with `'`).
- JSON exports contain only the stored/returned fields.
- Risk rows use opaque references and transformed values; raw customer
  attributes are not stored in the risk artifacts.
- Batch rejection tables contain position, hashed reference, code and message.

## 7. Testing

Focused dashboard tests (42):

```bash
python -m pytest -q \
  tests/unit/test_dashboard_analytics.py \
  tests/unit/test_dashboard_assessment.py \
  tests/unit/test_dashboard_data_access.py \
  tests/integration/test_dashboard_apptest.py
```

Coverage highlights:

- `AppTest` renders all 11 pages, against synthetic fixture artifacts and
  against the real project artifact tree.
- Missing artifacts produce actionable commands on every page.
- Tier filtering changes stored-table rows; cache invalidates after Parquet is
  rewritten; probabilities displayed for a customer match the stored value.
- Training action runs once per submit, not on rerun; failures and discovery
  failures are structured and contain no traceback.
- Assessment actions are compared directly with `RiskService` for one record and
  a batch; missing-column batches return rejection rows.
- Upload size/parse/empty/row-limit validation and schema typing.
- Simulation action output matches a direct Phase 10 engine run.
- Analytics helpers: tier boundaries, formatting, export formula guard,
  nested access.
- Validation report discovery ignores a newer Phase 3 `attempts.json` in the
  same directory.

The tests do not require the network or the production dataset. Live discovery,
acquisition and training are not run by the test suite.

Browser evidence is stored in [docs/screenshots](screenshots/): all eleven pages
at 1440 px wide, plus a batch assessment screenshot produced by uploading the
six-row Phase 9 CSV through the real file picker. The batch action scored 6 of 6
rows with no fabricated values. Start with the
[overview](screenshots/phase11-overview.png),
[risk table](screenshots/phase11-risk.png) and
[simulator](screenshots/phase11-simulator.png).

## 8. Known limitations

- Training is synchronous inside the Streamlit process. The page states this;
  there is no background job or streaming epoch log.
- Discovery and acquisition perform live HTTP only when explicitly started.
  They were not exercised during Phase 11 verification.
- Batch assessment does not compute SHAP; stored local explanations are
  available only for stored risk runs on Customer detail.
- The retention simulator computes in memory and does not persist; it labels
  every output as hypothetical and does not claim causal uplift.
- No automated browser end-to-end test is committed. Verification used
  `streamlit.testing.v1.AppTest`, a real HTTP probe, direct browser DOM
  inspection and Playwright-driven screenshots with system Chrome.
- Cold-load timing meets the 5 s target in a first-render `AppTest`
  measurement (1.937 s), but there is still no browser performance trace. The
  first real-data render includes pandas/Plotly/TensorFlow-adjacent imports.
