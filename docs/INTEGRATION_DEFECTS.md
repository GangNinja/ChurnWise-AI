# ChurnWise AI — Integration and Final Defect Register

**Date:** 2026-10-23  
**Scope:** integration, reliability, end-to-end composition  
**Status vocabulary:** Fixed / Mitigated / Open

Severity reflects impact on correctness, data integrity or the supported user
workflow. “Pre-existing” means the defect existed before the Phase 12 changes.

---

## Register

| ID | Severity | Phase / component | Reproduction or evidence | Root cause | Fix | Regression test | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-01 | High | Integration / CLI | No `run` subcommand; eight manual commands required and run IDs had to be carried by hand. | Phases were implemented as isolated services; no composition root existed. | Added `churnwise/pipeline/` with `run`, stage manifest, resume and structured errors; added `python -m churnwise.cli run`. | `tests/integration/test_pipeline_workflow.py::test_pipeline_runs_every_stage_and_preserves_provenance`; `tests/test_cli.py` run tests | Fixed |
| D-02 | High | Acquisition / preprocessing | A committed fixture could not enter the managed workflow without a network discovery or handcrafted acquisition objects. | No documented path for a local dataset that still produced provenance and passed the real validation gate. | `run --dataset` copies into the content-addressed store, writes an `offline_fixture` manifest, then validates/preprocesses normally; source file is never modified. | Same end-to-end test; provenance and byte-identity assertions | Fixed |
| D-03 | High | CLI | `discover --min-rows 1` returned exit 2 instead of the documented 3. | The new pipeline export `CODE_INVALID_CONFIG` shadowed the discovery coordinator constant in `churnwise/cli.py`. | Imported pipeline constants under distinct aliases. | `tests/test_cli.py::test_invalid_configuration_returns_three` | Fixed (Phase 12 regression) |
| D-04 | High | Pipeline | A failed run could not be resumed; artifacts could be partially overwritten. | No manifest, no stage status, no compatibility fingerprint. | `run.json` records stages, timings, outputs and artifacts; `--resume` checks config digest, invalidates a stage whose artifacts are missing and all later stages. | `test_resume_reuses_upstream_stages_and_rebuilds_missing_downstream`; `test_resume_rejects_a_changed_configuration` | Fixed |
| D-05 | Medium | Simulation / reporting | `aggregate_churn_probability` was 109.44 for 200 contacts; the report called this a probability. | The field stores probability mass (sum of probabilities), not a fraction. The schema predates Phase 12. | Retained the legacy JSON key; added `expected_churn_count` and an explicit `aggregate_churn_probability_is_probability_mass` marker; corrected the Markdown label. | `test_simulation_uses_the_risk_run_and_reconciles_its_arithmetic`; existing `tests/unit/test_campaign_math.py` | Fixed (label/alias only) |
| D-06 | Medium | Pipeline manifest | Live run stage summaries showed `hidden_units: []`, `threshold: null`, `best_epoch: null`, `selected_threshold: null`. | Runner read `metadata["config"]` / `selection["selected_threshold"]` instead of the actual nested keys. | Read `metadata["architecture"]`, `metadata["training"]`, `diagnostics`, `selection["threshold"]` and `selection["selection_metric"]`. | `test_model_benchmark_explanations_and_risk_share_the_same_artifacts` | Fixed |
| D-07 | Low | Phase 9 artifacts | Risk `run_manifest.json` can contain a relative `model_path` and an absolute `preprocessor_path`. | Paths originate from different layers; consumers assumed the current working directory. | Phase 13 writes both paths portably (project-relative when inside the project) and resolves relative paths against `Settings.project_root` in the risk loader and reliability page. Legacy absolute manifests remain readable. | `test_stale_model_artifact_is_rejected_on_reload`, `test_changed_preprocessor_artifact_is_rejected_on_reload` | Fixed |
| D-08 | Medium | Model training | Two MLP trainings can differ at floating-point precision. | TensorFlow op determinism is disabled by default (`MLPConfig.deterministic=False`); the trainer warns about it. | Split fingerprints, RF/LR training, risk scoring, ranking and simulation are deterministic. No silent claim of bitwise MLP reproducibility; users can set `deterministic=True` in a custom config. | Split/model/risk determinism suites; pipeline rescoring/simulation test | Open (documented) |
| D-09 | Medium | Artifact contracts | No single JSON-Schema registry validates every artifact. | Each phase owns its schema and tests; the pipeline did not assert cross-phase identity. | Added principal-artifact contract assertions (target mapping, split isolation, model→explain→risk hash equality, simulation arithmetic) while retaining phase contract suites. | Pipeline artifact test plus existing phase contract tests | Mitigated |
| D-10 | Low | Dashboard | Evidence inventory displays “not recorded” in the Generated column for EDA profiles. | EDA profile JSON had no `generated_at` field. | New validation runs write `generated_at` into the EDA profile and archived profile; the dashboard inventory reads it. Older artifacts still show “not recorded”. | `test_validation_writes_report_profile_and_plots`; dashboard AppTest suite | Fixed |
| D-11 | Low | Local fixture provenance | Local fixture manifest uses `license_status=unverified` even though the data is a project-owned synthetic fixture. | The fixture has no external licence document; the manifest vocabularies did not include “project-owned”. | Added `LicenseStatus.PROJECT_OWNED`, accepted by discovery G6, acquisition policy and manifest validation; local fixture manifests now record it explicitly. | `test_licence_policy_requires_opt_in_for_unverified_sources`; pipeline source-manifest assertion | Fixed |
| D-12 | Medium | Baseline inference | Two identical deterministic pipeline runs differed by one ULP in Random Forest `log_loss` (`0.7423580474286281` vs `...82`). | `n_jobs=-1` parallel tree averaging produced non-deterministic floating-point accumulation during `predict_proba`; the MLP and LR predictions were bitwise identical. | Random Forest now defaults to `n_jobs=1`, making training and prediction bitwise stable on the reference environment; `n_jobs=-1` remains an explicit speed trade-off. | `test_two_identical_runs_with_op_determinism_are_identical`; `test_random_forest_default_is_serial_for_bitwise_stable_predictions` | Fixed (Phase 13) |

## Fixed defect details

### D-01 / D-04 — orchestration and resume

The runner has seven stages, each persisted with:

- `status`: pending / running / succeeded / failed
- `started_at`, `finished_at`, `duration_ms`
- `outputs` (IDs consumed by later stages)
- `artifacts` (paths to files/directories)
- `summary`, `warnings`, `error`

The run manifest also stores the config fingerprint, settings, aggregate outputs,
warnings and terminal error. A resume with a different fingerprint is refused
before any artifact is touched.

### D-03 — CLI regression caught by the existing suite

This defect was introduced during Phase 12 and found by the full-suite baseline.
It is exactly the class of issue this phase is meant to catch: an apparently
harmless import changed a documented exit code. The test was not modified; the
implementation was corrected.

### D-05 — simulation semantics

Example from the live run:

- selected contacts: 200
- sum of churn probabilities: 109.4446 expected churners
- assumed effectiveness: 0.30
- expected prevented churn: 32.8334

Calling 109.44 a “probability” is wrong. The legacy key remains for compatibility;
new consumers should use `expected_churn_count` and read
`aggregate_churn_probability_is_probability_mass`.

## Unresolved items carried forward

1. **D-08 MLP bitwise determinism** — the pipeline records the warning and does not
   claim exact reproducibility of MLP floats. Any future comparison of repeated
   MLP runs must state its tolerance. Deterministic mode exists and is tested.
   Random Forest prediction nondeterminism (D-12) is fixed by its serial default.
2. **D-09 generic schemas** — artifact validation remains distributed by phase
   rather than centralized in a JSON-Schema registry.

D-07, D-10 and D-11 were fixed in Phase 13; their regression evidence is in the
register above.

## Not defects

- Live-network tests are marked and skipped by default; this is intentional and
  documented, not a hidden pass.
- A rejected dataset leaves its acquired blob, manifest and validation report for
  diagnosis. It creates no model, benchmark, risk or simulation success artifact.
  This is a deliberate safety choice, not a “partial success”.
- `targeting_lift_vs_random = 1.0` when every eligible customer is contacted is
  correct: there is no targeting lift if selection is not selective.
