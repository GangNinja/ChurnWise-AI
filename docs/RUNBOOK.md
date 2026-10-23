# ChurnWise AI — Phase 12 Runbook

This runbook covers the single-command pipeline, resume, failure diagnosis,
artifact verification and safe cleanup. It does not replace the phase documents
for discovery, acquisition, validation, modelling, risk or simulation.

## 1. Prerequisites

- Python 3.13 in an isolated `.venv`.
- The dependencies installed from `requirements.lock.txt` for an exact
  reproduction of the audited environment, or from `requirements.txt` for a
  cross-platform range install; see [DEPENDENCIES.md](DEPENDENCIES.md).
- Run from the repository root. All paths below are relative to it.

Windows PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
```

The live workflow needs network access. The fixture workflow does not; it copies
a committed local dataset into the managed store and runs the same validation and
modelling path.

## 2. Normal runs

Offline fixture (small, deliberate `--fast` profile):

```bash
python -m churnwise.cli run \
  --dataset tests/fixtures/mini_churn.csv \
  --fast --seed 13 --run-id demo-fixture
```

Live network discovery and full workflow:

```bash
python -m churnwise.cli run --fast --seed 42 --run-id demo-live
```

Without `--fast`, the configured MLP (150 epochs by default), 5-fold benchmark
and larger explanation samples are used. Do not present fast-profile metrics as
production evidence.

Useful options:

| Option | Effect |
| --- | --- |
| `--dataset PATH` | Offline workflow from a committed CSV/TSV/ARFF/Parquet/JSON/XLSX file; copied, never modified |
| `--run-id ID` | Stable run identity; required for resume |
| `--resume` | Reuse succeeded stages whose recorded artifacts still exist |
| `--fast` | Shorter MLP/benchmark/explanation profile |
| `--seed N` | Seed for the split, models, sampling and scenario |
| `--budget`, `--cost-per-contact`, `--effectiveness` | Hypothetical campaign assumptions |
| `--data-dir`, `--artifact-dir`, `--report-dir` | Project-local storage overrides |
| `--json` | Machine-readable terminal result |

The manifest is always `artifacts/pipelines/<run-id>/run.json` (or the same path
under `--artifact-dir`). It records each stage's status, timestamps, duration,
outputs, artifact paths, summary, warnings and error. A successful manifest has
`"status": "succeeded"` and seven succeeded stages.

## 3. Resume and recovery

Resume the **same configuration** with the **same run id**:

```bash
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv \
  --fast --seed 13 --run-id demo-fixture --resume
```

Resume never silently mixes old and new work:

1. The saved `config_digest` must equal the current configuration; a different
   seed, dataset bytes, profile or scenario is refused with `CW-RUN-004`.
2. A `running` or `failed` stage invalidates itself and every later stage.
3. A succeeded stage is reused only if every artifact recorded for it still
   exists. A missing artifact invalidates that stage and all downstream stages.
4. A corrupt or unreadable manifest is refused; it is never overwritten.
5. A rejected dataset leaves its acquisition and validation diagnostics for
   inspection, but no model, benchmark, risk or simulation success artifacts.

A completed run returns its existing manifest. To run again from scratch, use a
new `--run-id`; `--resume` is not a reset command.

### Error codes

| Code | Meaning | Action |
| --- | --- | --- |
| `CW-RUN-001` | Invalid pipeline configuration | Read the reported problem; fix the path, seed, profile or scenario |
| `CW-RUN-002` | An upstream phase failed or rejected its input | Read `error.details` and the source stage diagnostics; do not force later stages |
| `CW-RUN-003` | A stage raised an unexpected exception | Inspect the manifest stage error and the phase report; preserve the run for diagnosis |
| `CW-RUN-004` | The saved run cannot be resumed with this configuration | Use the original configuration or a new run id |
| `CW-RUN-005` | A completed stage is missing a required artifact | Do not trust the run; delete it and rerun |

`CW-RUN-002` details can include the validation gate's blocking checks. For
example, a single-class target fails V4 in the `source` stage; that is a correct
stop, not a reason to skip validation.

## 4. Inspect before deleting

Choose the exact run id from the manifest; do not guess from a directory name.
The cleanup tool is a dry run by default:

```bash
python -m churnwise.pipeline.cleanup --run-id demo-fixture
```

To inspect machine-readable output:

```bash
python -m churnwise.pipeline.cleanup --run-id demo-fixture --json
```

The plan lists only paths recorded in the manifest that resolve inside the
configured artifact or report roots and whose path names the run. It refuses to
delete a configured root, shared dataset reports, acquisition manifests, or a
path belonging to another run. It skips non-path output identifiers such as
`dataset_id` and `selected_family`. The shared acquisition cache is never
deleted; it is content-addressed and can serve later runs.

Common reasons a path is skipped:

- `data/manifests/...` and `data/cache/...` are shared provenance/cache data.
- `reports/acquisition/...` and `reports/validation/<dataset>.json` describe the
  dataset, not one pipeline run, and may be reused.
- A hand-edited manifest points outside the managed roots or at another run.

## 5. Delete a run

After reviewing the plan and confirming no process is using the run:

```bash
python -m churnwise.pipeline.cleanup --run-id demo-fixture --delete
```

This removes the run's stage artifacts, its report files that name the run, and
finally `artifacts/pipelines/demo-fixture/`. It does **not** delete the raw
dataset, the acquisition cache, dataset-level manifests or dataset-level
validation reports. Remove those only with a separate, explicit data-retention
decision; they are not owned by a single pipeline run.

Do not delete a run while another process is writing it. There is no distributed
lock; this is a single-machine, single-user workflow.

## 6. Verification after a change or failed run

```bash
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv \
  --fast --seed 13 --run-id verify-fixture

python -m pytest -q tests/integration/test_pipeline_workflow.py
python -m pytest -q tests/unit/test_pipeline_contracts.py \
  tests/unit/test_pipeline_cleanup.py
```

The first command produces a fresh artifact tree. The integration tests run the
workflow, check cross-phase artifact identity, resume behavior, deterministic
rescoring/simulation and dashboard rendering. They do not prove the live network
path; use the live run and `--run-live` tests for that, and record failures
rather than citing a skipped test as a pass.

For timings and resource measurements, see [PERFORMANCE.md](PERFORMANCE.md).
