# Evidence directory

This directory preserves the small machine-readable artifacts behind the final
documentation. Full models, NumPy arrays, Parquet scores, plots and per-row SHAP
files remain in the git-ignored working trees and are reproducible with the
commands in the run-specific READMEs.

## What is here

| Directory | Content |
| --- | --- |
| `benchmark/` | Full-profile benchmark metrics and Markdown report |
| `explainability/` | Global SHAP ranking, RF run manifest and 1,000-row matrix summary |
| `phase13-live/` | Pre-fix live run: pipeline manifest, validation, metrics, risk and simulation |
| `phase13-live2/` | Post-fix live run: same artifact set; authoritative for final numbers |
| `phase13-tests/` | Preserved test logs and observed outcomes |
| `post_phase13/` | Curated extracts for the full-profile, second-dataset, calibration and browser E2E workstreams, with source checksums |

## Integrity note

Numeric values, IDs, thresholds, tier counts, digests and simulation fields are
copied from the generated artifacts without editing. Absolute paths containing
the local username/project location were replaced with project-relative paths
during the final repository-hygiene pass; this changes only path strings. The
run-specific manifests therefore may show a relative artifact path where the
original working tree had an absolute one.

The complete generated trees are intentionally git-ignored. Recreate them with
the commands in [phase13-live2/README.md](phase13-live2/README.md).
