# Post-Phase-13 evidence (curated)

This directory preserves the small, reviewable evidence behind
[docs/VALIDATION_SUMMARY.md](../../VALIDATION_SUMMARY.md). It is versioned;
the full generated experiments remain in the Git-ignored
`reports/post_phase13/` working tree.

## Files

| File | Content | Original source |
| --- | --- | --- |
| `full_pipeline.json` | Full-profile run identity, seven stage statuses, test metrics, risk tiers and simulation summary | `reports/post_phase13/workstream1/artifacts/pipelines/post13-full-20261024c/run.json`; timing/memory from `workstream1/resource_c.json` |
| `second_dataset.json` | OpenML 42178 dataset, adapter, split, validation gate and family metrics | `reports/post_phase13/workstream2/experiment_summary.json`; adapter details from `workstream2/adapter.json` |
| `calibration.json` | Calibration protocol, validation comparison, test metrics, bootstrap intervals, serialization and limitations | `reports/post_phase13/workstream3/calibration_summary.json` |
| `browser_e2e.log` | Exact pytest browser-suite terminal result | `reports/post_phase13/workstream4/browser-suite.log` |
| `regression_tests.txt` | Default-suite rerun during the documentation pass: **624 passed, 17 skipped** | console output, 2026-10-24 |

These are **curated extracts**, not replacements for the source reports. Numeric
values, identifiers, thresholds and limitations were copied without changing
their meaning. The original generated reports provide the full context and
reproduction details. `regression_tests.txt` is a concise record of a fresh
run; the full pytest console (including 1,484 library warnings) was not copied.

## Source checksums

The originals used for the extracts had these SHA-256 digests when curated on
2026-10-24:

```text
f380a6c06686dd347ecfaf91b0d9ed43f8265e583510f1fa8b759a1d07d37db7  workstream1/artifacts/pipelines/post13-full-20261024c/run.json
3865165b2bbc29bebe15ef4dcd0a6a19f4582456073e33f3fc9e0f61e8c0a1c3  workstream1/resource_c.json
ace4a3c4198be1ad18706765fd8cab19110eae99b7e74892e28bdb3196ef1fcf  workstream2/experiment_summary.json
1596afa1f10f372ecca8551308d07be6a9ffa648a7be82290895727fbe978f45  workstream2/adapter.json
5ef091a9bdde93ecbb0791889cc37141d3cd302c1988d40989ce7abeedfef4dc  workstream3/calibration_summary.json
1d5a8470b0bdbdf772c15ac6e07c471a6ad156958ce2f26e5377bd00af38f87a  workstream4/browser-suite.log
```

`full_pipeline.json` and `second_dataset.json` record the source paths and
digests they were derived from. `calibration.json` does the same for its source
summary.

## What was not copied

- raw datasets (OpenML downloads, including the ARFF and derived CSV);
- trained model binaries, NumPy prediction arrays and Parquet score files;
- full pipeline artifact trees, plots, per-row SHAP files and browser logs;
- the full Markdown reports under `reports/post_phase13/`;
- the local absolute paths and username embedded in the generated run trees.

These files are reproducible with the commands in
[HOW_TO_RUN.md](../../HOW_TO_RUN.md) and the run-specific READMEs. They are
intentionally not committed: the repository policy keeps generated data, models
and reports out of Git.

## Evidence boundary

The curated files support the claims in `VALIDATION_SUMMARY.md`: one full-profile
primary pipeline run, a separately trained second-dataset experiment, a
validation-only calibration experiment, and an opt-in 10-test browser suite. They
do not establish production deployment, causal impact, measured business
savings or external model transfer.

The full reports remain:

- `reports/post_phase13/POST_PHASE13_VALIDATION_SUMMARY.md`
- `reports/post_phase13/FULL_PIPELINE_VALIDATION_REPORT.md`
- `reports/post_phase13/SECOND_DATASET_ROBUSTNESS_REPORT.md`
- `reports/post_phase13/PROBABILITY_CALIBRATION_REPORT.md`
- `reports/post_phase13/DASHBOARD_E2E_REPORT.md`
