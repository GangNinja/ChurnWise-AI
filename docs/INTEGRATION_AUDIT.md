# ChurnWise AI — Phase 12 Integration Audit

**Date:** 2026-10-23  
**Audited tree:** working checkout, uncommitted Phase 1–11 changes included  
**Auditor:** Phase 12 implementation pass

This audit is based on the repository itself, not on the earlier phase reports
alone. Each phase report was treated as a claim to verify against code, tests and
artifacts.

---

## 1. Scope reviewed

| Area | Files / evidence inspected |
| --- | --- |
| Requirements | `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/ACCEPTANCE_CRITERIA.md`, `README.md` |
| Phase reports | `docs/PHASE2_REPORT.md` through `docs/PHASE11_REPORT.md` where present |
| Entry points | `churnwise/cli.py`, `churnwise/dashboard/app.py` |
| Phase services | `discovery/`, `acquisition/`, `validation/`, `preprocessing/`, `models/`, `evaluation/`, `explain/`, `intelligence/`, `simulation/` |
| Dashboard | `churnwise/dashboard/{app,data_access,actions,analytics}.py`, `pages/`, `components/` |
| Artifacts | `data/manifests/openml_46915.json`, `reports/validation/openml_46915.json`, previous evaluation/explanation/risk/simulation artifacts |
| Tests | `pytest.ini`, `tests/conftest.py`, all existing unit/integration suites, live-network markers |
| Configuration | `churnwise/config.py`, `requirements.txt`, `.gitignore`, environment handling |

## 2. Baseline verification

The Phase 11 report claimed 589 passing tests and 7 live-network skips. The
baseline suite was re-run during this phase, not assumed:

```bash
.venv/Scripts/python.exe -m pytest -q
```

The first run exposed one CLI regression introduced by the new `run` command:
the pipeline's `CODE_INVALID_CONFIG` import shadowed the discovery coordinator's
constant of the same name, changing `discover --min-rows 1` from exit code 3 to
2. It was fixed (D-03) and the existing test now passes again.

The live-network tests were skipped by default, as designed. They were executed
separately during Phase 12; see §4.

## 3. Phase-by-phase capability audit

| Phase | Claimed capability | Audit result before Phase 12 changes |
| --- | --- | --- |
| 1 | Architecture, contracts, environment | Documents and packages present. The roadmap named `churnwise/pipeline.py` and a `run` command that did not exist. |
| 2 | Autonomous discovery | Implemented; adapters and gates present. A fresh live run found 5 listings, 3 unique candidates and 1 eligible selection (`openml:46915`). |
| 3 | Acquisition and provenance | Implemented serial path with cache, integrity checks, fallback and manifest. No path accepted a committed local fixture into the same provenance contract. |
| 4 | Validation and EDA | Implemented; target detection, V1–V12 checks, gate and reports present. |
| 5 | Preprocessing | Implemented; deterministic split and fitted preprocessing artifacts present. |
| 6 | MLP training | Implemented; architecture, early stopping, checkpointing and metadata present. |
| 7 | Benchmarking | Implemented; all three families, validation-only selection and frozen test evaluation present. |
| 8 | SHAP explainability | Implemented; global/local artifacts and model/preprocessor provenance present. |
| 9 | Risk intelligence | Implemented; individual/batch/population scoring, tiers, drivers, exports and freshness checks present. |
| 10 | Retention simulation | Implemented; budget enforcement, allocations, projections, comparison and sensitivity present. |
| 11 | Dashboard | Implemented; eleven pages and AppTest suite present. It reads persisted artifacts and gates actions behind buttons. |
| 12 | Integration | **Missing.** There was no single command, no run manifest, no resume, no end-to-end test and no cross-phase artifact verification. |

## 4. Gaps and integration risks

### 4.1 Missing end-to-end composition

Every phase service existed, but users had to run eight commands in order and
manually carry run IDs. Nothing verified that the Phase 7 selected family is the
one explained in Phase 8 and scored in Phase 9, or that Phase 10 consumed that
same risk run.

**Resolution:** `churnwise/pipeline/` now composes the services, records every
stage in `artifacts/pipelines/<run-id>/run.json`, and supports safe resume.

### 4.2 Fixture ingestion bypassed provenance

Tests used in-memory fixtures or handcrafted acquisitions. A committed CSV could
not be run through the real Phase 4–10 workflow without a network discovery.

**Resolution:** `run --dataset <path>` copies the file into the managed
content-addressed store, writes a `DatasetManifest` with an explicit
`offline_fixture` acquisition method and SHA-256, then runs the normal validation
and modelling path. The source file is never modified.

### 4.3 CLI constant collision

The new pipeline package exported `CODE_INVALID_CONFIG`; importing it into the CLI
replaced the discovery constant and broke the documented exit-code contract.

**Resolution:** pipeline constants are imported under distinct aliases; the
existing CLI regression test was retained and now passes.

### 4.4 Misleading simulation label

`OutcomeProjection.aggregate_churn_probability` stores the sum of individual
probabilities (probability mass), not a fraction between zero and one. The
simulation report labelled it “Aggregate churn probability”, which invites the
wrong interpretation.

**Resolution:** the value is retained for backward-compatible JSON, but now has an
explicit `expected_churn_count` alias and a
`aggregate_churn_probability_is_probability_mass` marker. The Markdown report
labels it “Expected churn count in selected set … probability mass, not a
fraction”.

### 4.5 Stage summaries read the wrong metadata keys

The first complete live run produced `best_epoch: null`, `hidden_units: []` and
`selected_threshold: null` in the run manifest because the runner looked at the
wrong nested metadata keys.

**Resolution:** summaries read `metadata["training"]`,
`metadata["architecture"]`, benchmark `selection["threshold"]` and
`selection["selection_metric"]`; a pipeline test asserts the corrected values.

### 4.6 Path portability in Phase 9 manifests

The Phase 9 `run_manifest.json` can contain a relative `model_path` and an
absolute `preprocessor_path` in the same document. The paths are verified by hash
on load, so this does not corrupt results, but moving a run tree to another
machine may produce a freshness warning. This predates Phase 12 and was not
changed because the public artifact schema is already consumed by the dashboard
and prior reports. It is recorded in the defect register as an unresolved low
severity portability issue.

### 4.7 Unverified areas found by the audit

- No single generic JSON-Schema registry covers every artifact file; validation is
  distributed across phase-specific contract tests and pipeline assertions.
- Full MLP retraining is not bitwise reproducible while TensorFlow op determinism
  is disabled. Splits, model selection inputs, risk scoring and simulation are
  deterministic with fixed artifacts.
- The full production MLP profile (150 epochs, larger SHAP samples, 5-fold CV)
  was not run end-to-end on the live dataset during this phase; the live run used
  the documented `--fast` profile.
- No automated browser suite is added beyond Streamlit `AppTest`; the live server
  was still exercised through the browser panel for navigation and data display.

## 5. Artifacts and contracts

The live network run produced a complete artifact set from empty project-local
directories:

| Stage | Output |
| --- | --- |
| Discovery | `reports/acquisition/candidates.json` and `rejections.json` |
| Acquisition | `data/manifests/openml_46915.json`, content-addressed ARFF blob, raw copy |
| Validation | `reports/validation/openml_46915.{json,md}`, EDA profile and plots |
| Preprocessing | split spec, fitted preprocessor, matrices, summary and reports |
| Training | MLP model/checkpoint, metrics/history, metadata and reports |
| Evaluation | `metrics.json`, predictions, curves, threshold analysis and run manifest |
| Explanation | `global.json`, plots, local examples and run manifest |
| Risk | `churn_scores.parquet`, `risk_summary.json`, tiers, drivers and run manifest |
| Simulation | `result.json`, sensitivity, comparison-ready selection, report and manifest |
| Pipeline | `artifacts/pipelines/<run-id>/run.json` |
| Cleanup | dry-run/delete plan confined to run-owned artifact and report paths; shared acquisition data retained |

## 6. Conclusion

The individual phases were implemented, but the application was not integrated
before Phase 12. The central gap was composition and evidence: no command proved
that a real candidate could travel from discovery to a hypothetical campaign, and
no test compared artifact identities across phases.

Phase 12 adds that composition, exercises it offline and against the real OpenML
source, and records the remaining limitations rather than treating phase reports
as proof.
