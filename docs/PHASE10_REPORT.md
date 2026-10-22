# Phase 10 Report — Retention Campaign Simulation

Status: **implemented and verified** against synthetic hand-computed fixtures and
the real Phase 9 OpenML `46915` risk run. Scope is limited to Phase 10; no
dashboard (Phase 11) code exists.

> **The simulator estimates hypothetical outcomes.** It does not establish
> causal intervention effectiveness, does not observe retention, and does not
> claim that a campaign will prevent churn. Every projected figure below is the
> arithmetic consequence of user-supplied assumptions applied to measured model
> probabilities.

## 1. What was built

| Module | Responsibility |
| --- | --- |
| `churnwise/simulation/contracts.py` | Validated scenario/assumption/targeting contracts, selected/excluded records, allocation, projection, comparison, sensitivity and run records |
| `churnwise/simulation/assumptions.py` | Defaults, validation ranges, sensitivity grids and dimension units |
| `churnwise/simulation/campaign.py` | Eligibility, deterministic selection, budget allocation, projection math, comparison and one-at-a-time sensitivity |
| `churnwise/simulation/service.py` | Phase 9 loading, scenario execution, persistence, artifact identity checks and path guards |
| `churnwise/simulation/reporting.py` | Console summary, Markdown report and comparison/sensitivity tables |
| `churnwise/simulation/plots.py` | Allocation, sensitivity and comparison PNG charts from computed points only |
| `churnwise/simulation/errors.py` | `CW-SIM-001…005` structured failures with remediation |
| `churnwise/cli.py` (`simulate`) | CLI entry point with explicit assumptions, grids, comparison files and JSON output |

No database, cloud service, network call or separate API was added; the
subsystem reads local Phase 9 artifacts and writes local artifacts.

## 2. Campaign configuration and validation rules

`CampaignScenario` = name/description + `budget` + `CampaignAssumptions` +
`TargetingConfig` + optional per-customer `cost_overrides`.

Defaults (all assumptions, never measurements): budget `1000`, cost per contact
`5`, effectiveness `0.30`, `monthly_value` absent, horizon `6` months, value unit
`value_units`, strategy `highest_risk`, no threshold, no tiers, no contact cap.

| Rule | Behaviour |
| --- | --- |
| Budget | finite, `>= 0`, `<= 1e12` |
| Cost per contact / overrides | finite, `>= 0`; unknown references ignored with a warning |
| Effectiveness | finite, `[0, 1]`; required definition captured in the payload |
| Monthly value | `None` or finite `>= 0`; never inferred |
| Horizon | finite, `> 0`, `<= 1200` months |
| Value unit | non-empty label, `<= 24` characters |
| Strategy | one of four; `risk_threshold` requires a threshold, `risk_category` requires tiers |
| Tiers | validated against the loaded Phase 9 run's tier labels |
| Contacts | `None` or integer `[1, 10_000_000]` |

All violations are collected, not short-circuited, and raise
`SimulationError` `CW-SIM-001` with every problem plus a remediation string.
`CampaignScenario.from_dict` rejects unknown keys, so a scenario file cannot
silently acquire an unsupported assumption. The scenario payload is
fingerprinted with SHA-256 and embedded in each result and manifest.

## 3. Targeting and allocation strategies

Eligibility is applied first, with no budget/count limit: optional tier filter
and lower-inclusive `p_i >= min_risk_probability`.

| Strategy | Behaviour |
| --- | --- |
| `highest_risk` | all tier/threshold-eligible customers, descending probability |
| `risk_threshold` | requires a minimum probability, then descending |
| `risk_category` | requires tier labels (e.g. high + critical), then descending |
| `budget_constrained` | descending, stops at the first customer the remaining budget cannot cover (strict risk-order prefix) |

Selection key: `(-churn_probability, record_reference, input_position)`, so ties
are stable and repeat runs are identical. Tier labels are taken from the loaded
Phase 9 run (or derived from the assessments when a context is not supplied), so
a three-tier run cannot silently accept a `critical` rule. `highest_risk`,
`risk_threshold` and `risk_category` skip an unaffordable customer when
per-customer cost overrides make later customers affordable;
`budget_constrained` does not. Every strategy warns when budget or the contact
limit leaves eligible customers untargeted.

Allocation rules:

- `max_affordable = floor(budget / cost_per_contact)` for a uniform positive
  cost; `None` for zero or varying costs.
- Spend is the sum of actual selected intervention costs, compared against the
  budget with a `1e-9`-scaled tolerance; remaining budget is clamped at zero.
- `max_contacts` is enforced before the budget, and the two exclusion reasons
  are counted separately.
- **Zero-cost** interventions ignore the budget count, report `max_affordable =
  None`, ROI `null` (not infinite) and cost per prevented churn `0.0`.
- **Zero budget** with paid interventions selects nobody, reports every
  eligible customer as `budget_exhausted`, and warns.
- Population accounting reconciles: `assessed = selected + threshold + tier +
  budget + contact limit`; an internal mismatch raises `CW-SIM-002`.

## 4. Expected-outcome formulas and assumptions

```text
expected_prevented_churn = Σ_{i∈S} p_i · e
expected_saved_value     = expected_prevented_churn · monthly_value · horizon_months
campaign_cost            = Σ_{i∈S} intervention_cost_i
net_benefit              = expected_saved_value − campaign_cost
roi                      = net_benefit / campaign_cost
cost_per_prevented       = campaign_cost / expected_prevented_churn
break_even_effectiveness = campaign_cost / (Σ_{i∈S} p_i · monthly_value · horizon_months)
random_contact           = mean(p over eligible) · |S| · e
targeting_lift           = expected_prevented_churn / random_contact
```

`e` is the assumed probability that an intervention prevents churn for a
customer who would otherwise churn. Expected prevented churn is fractional and
is never rounded or described as retained customers. Every result JSON sets
`"projection": true`, `"observed_retention_result": false`, `"hypothetical":
true` and carries three disclaimers. Undefined metrics are `null` **and** named
in `projection.undefined_metrics` with a reason; break-even above `1.0` warns
that no admissible effectiveness recovers the cost.

## 5. Scenario comparison methodology

`compare_scenarios` builds one assumption-visible row per scenario (budget,
cost, effectiveness, value unit, threshold, tiers, contact cap, selected, spend,
remaining, utilisation, expected prevented churn, cost per prevented churn,
saved value, net benefit, ROI, break-even, lift). It:

- lists `differing_assumptions` so a projection cannot be read as a like-for-like
  performance comparison;
- sets `shared_inputs = true` only when risk run id, model digest, dataset
  digest, assessed population size and tier boundaries match, otherwise
  enumerates each difference;
- flags a value-unit mismatch because monetary columns in different units are
  not directly comparable;
- states that spend is simulated spend, not the configured budget, and that a
  larger estimate under optimistic assumptions is not a better campaign;
- carries the standard limitations and projection disclaimers.

## 6. Sensitivity methodology

`run_sensitivity` varies one assumption at a time, recomputing selection,
allocation and projection for every value; nothing is interpolated.

| Dimension | Default grid | Unit |
| --- | --- | --- |
| effectiveness | `0.10, 0.20, 0.30, 0.40, 0.50` | probability |
| budget | `0.5×, 1×, 1.5×, 2×` scenario budget | value unit |
| cost per contact | `0.5×, 1×, 2×` scenario cost | value unit per contact |

Grids require 2–50 finite values inside their ranges (effectiveness `[0,1]`,
budgets `>= 0`, costs `>= 0`); duplicates collapse, and a grid that collapses
to one value (zero cost or zero budget with the default multipliers) is skipped
with a warning instead of producing a meaningless point. Each point records
`is_baseline`, counts, a 16-hex fingerprint of the selected reference set,
spend, remaining budget, expected prevented churn, saved value, cost per
prevented churn and utilisation. `fixed_parameters` records what was held
constant. Interactions are explicitly out of scope.

## 7. Artifacts

Written to `artifacts/simulations/<scenario_id>/` (git-ignored):

`result.json`, `selection.csv`, `sensitivity.json`, `sensitivity.csv`,
`comparison.json`, `comparison.csv` (when comparisons were requested),
`report.md`, `run_manifest.json`, and `plots/budget_allocation.png`,
`plots/sensitivity_<dimension>.png`, `plots/scenario_comparison.png`.

`run_manifest.json` and `result.json.provenance` record the scenario fingerprint,
package version, risk run id, model digest, preprocessing run id, dataset SHA-256,
split fingerprint and `seed: null` (the engine samples nothing). `load_result`
verifies that the directory name, `scenario_id` and phase match and that
required blocks exist; scenario ids are regex-restricted with no `..`.
CSV cells pass through the Phase 9 formula-injection guard. Simulation outputs
contain Phase 9 opaque references only — no raw segments or customer attributes.

## 8. Tests actually run

```bash
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest tests/unit/test_campaign_math.py -q
# 47 passed

PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest tests/integration/test_simulation_assumptions.py -q
# 10 passed

PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest tests/test_cli.py tests/test_risk_cli.py \
  tests/unit/test_ranking_tiers.py tests/integration/test_scoring_counts.py -q
# 63 passed in 46.13s (Phase 9/CLI regression after the CLI change)

PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest -q
# 547 passed, 7 skipped in 180.12s (0:03:00)
# The 7 skips are the opt-in live-network tests (`--run-live`).
```

Coverage highlights, all with hand-computed expectations rather than shape
checks:

- three fully hand-computed scenarios (selection, spend, remainder,
  utilisation, prevented churn, saved value, net benefit, ROI, cost efficiency,
  break-even, baseline, lift);
- each strategy, threshold inclusivity at `p = 0.30`, stable tie-breaking,
  determinism under input reordering;
- contact limits, budget exhaustion under every strategy, per-customer cost
  overrides and their documented strategy difference;
- zero cost with positive and zero budgets, zero budget with paid
  interventions, zero risk, zero expected prevented churn, empty population;
- effectiveness exactly `0` and `1`;
- undefined ROI / cost-per-prevented / break-even / lift with reasons;
- invalid budgets, costs, values, probabilities, horizons, unit labels,
  strategies, contact caps, tiers and cost overrides; strategy-specific errors;
- scenario payload round-trip and unknown-key rejection;
- assumptions embedded in `result.json` equal the inputs used;
- comparison disclosure, mismatched risk-input detection and a differing
  value-unit warning;
- tier-label derivation for two-boundary (three-tier) runs, degenerate sensitivity grids,
  the selection fingerprint and budget-limit warnings;
- sensitivity grids against independently recomputed results, including a
  baseline-flagged point for every dimension and rejection of invalid grids;
- non-assessment records are rejected with `CW-SIM-002` and an index, while
  impossible probabilities remain impossible to construct at the Phase 9
  boundary;
- result JSON does not contain segment values;
- real Phase 9 integration: selected probabilities equal the risk table,
  selected set equals the top-ranked eligible set, sensitivity points equal
  fresh recomputations, artifacts and CSV schema exist, CLI success and
  invalid-configuration exit code `3`, missing/broken run codes, path and
  identity guards, and CSV formula sanitization.

## 9. Integration with Phase 9

The real Phase 9 run `20261022T155236Z-openml_46915-risk-final` (Random Forest,
1,000 assessed validation customers, tier boundaries 0.30/0.50/0.70, no revenue
proxy) was loaded through `RiskService.load_run`, including model SHA-256
freshness checks. No probability was recomputed by this phase.

Cross-checks:

- selected probabilities equal the stored `churn_scores.parquet` values exactly;
- exclusion/tier counts reconcile with the risk table's tier labels and
  probabilities;
- the integration test independently asserts that the selected references are
  the top-ranked eligible rows from the risk table;
- an input-difference comparison test proves the service refuses to present
  mismatched risk runs/models as shared inputs.

## 10. Genuine simulations

### 10.1 Baseline (real data)

Command:

```bash
PYTHONPATH=. ./.venv/Scripts/python.exe -m churnwise.cli simulate \
  --run-id 20261022T155236Z-openml_46915-risk-final \
  --name "Baseline retention pilot" --budget 1000 --cost-per-contact 5 \
  --effectiveness 0.30 --monthly-value 50 --horizon-months 6 --min-risk 0.30 \
  --scenario-id 20261022T160000Z-real-baseline
```

Result (`artifacts/simulations/20261022T160000Z-real-baseline/`): 1,000
assessed, 142 eligible at `p >= 0.30`, 142 selected (858 excluded below the
threshold), spend **710** of 1,000 value units (71% utilisation, 290 remaining,
200 affordable). Aggregate selected probability **96.2622**; expected prevented
churn **28.8787** at assumed `e = 0.30`; assumed value 50/month × 6 months = 300
per prevented churn; expected saved value **8,663.60**, net benefit **7,953.60**,
ROI **1,120.22%**, cost per expected prevented churn **24.5856**, break-even
effectiveness **2.46%**, random-contact baseline **28.8787** and lift **1.000×**
(no budget truncation, so the eligible set is the selected set).

Effectiveness sensitivity (spend fixed at 710): `0.10 → 9.6262`, `0.20 →
19.2524`, `0.30 → 28.8787`, `0.40 → 38.5049`, `0.50 → 48.1311` expected
prevented churns. Budget sensitivity: 500 → 100 selected / 500 spent / 23.3357
prevented; 1,000 → 142 / 710 / 28.8787; 1,500 and 2,000 → unchanged at 142 /
710 / 28.8787 because the eligible population is exhausted. Cost sensitivity:
2.5 → 142 selected / 355 spent; 5 → 142 / 710; 10 → 100 / 1,000 / 23.3357.

### 10.2 Two-scenario comparison (real data)

`artifacts/simulations/20261022T162000Z-real-comparison/` compares the baseline
with a premium outreach scenario (budget 600, cost 12, assumed effectiveness
0.45, `critical` tier only). Both used the same risk run and population.

| Scenario | Selected | Spend | Expected prevented | Saved value | ROI | Cost/prevented | Lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline threshold pilot | 142 | 710 | 28.8787 | 8,663.60 | 1,120.22% | 24.5856 | 1.000× |
| Critical-only premium outreach | 50 | 600 | 19.7744 | 5,932.31 | 888.72% | 30.3423 | 1.049× |

The premium scenario has a marginally higher targeting lift but a **worse** cost
per expected prevented churn and a lower total projection, because it assumes a
higher effectiveness on a narrower, more expensive segment. The comparison
records six differing assumptions (`budget`, `cost_per_contact`, `effectiveness`,
`min_risk_probability`, `risk_tiers`, `strategy`) and states that neither row is
evidence of real-world campaign performance.

## 11. Limitations and unverified acceptance criteria

- **No observed outcome.** The Phase 9 dataset has no retention experiment, so
  projections cannot be scored against realised saves. `e` is an assumption.
- **Uniform effectiveness and value.** No segment-specific uplift, saturation,
  contact fatigue, discounting or currency conversion is modelled.
- **No causal inference.** No uplift model, A/B test or causal adjustment exists.
- **Random-contact baseline** assumes a same-size independent draw from the
  eligible pool; it is a comparison device, not a control group.
- **Sensitivity is one-at-a-time**; interactions between budget, cost and
  effectiveness are not explored.
- **Dashboard label unverified.** The acceptance criterion "labelled as
  projections in JSON and in the UI" is satisfied for JSON, the Markdown report
  and the CLI and tested there; CSV rows carry `expected_*` columns under a
  projection-labelled scenario artifact, and the UI does not exist until Phase
  11, so that half remains open.
- **Reference run has no revenue proxy**, so the real-data financial figures
  above use a supplied illustrative value of 50 value units per month. They are
  not claims about the dataset's economics.
- **Artifacts are git-ignored**, as for every other phase; the numbers above are
  reproducible from the documented commands.

## 12. Post-implementation verification audit

A requirement-by-requirement audit was run after the first green suite. It
exercised edge cases directly and re-derived the real artifacts independently.
The following defects were found and fixed, with regression tests added or
strengthened:

1. **Tier-label hole.** With no explicit `RiskContext`, a three-tier risk run
   was validated against the default four-tier labels, so a `critical` rule
   passed validation and silently selected nobody. Tier labels and boundaries
   are now derived from the assessments; the case is covered by a test.
2. **Degenerate sensitivity grids.** Zero cost or zero budget made the default
   multiplier grid collapse to a single point, producing a meaningless
   sensitivity series and chart. Such dimensions are now skipped with an
   explicit warning that appears in `sensitivity.json`, the Markdown report and
   the CLI summary.
3. **Missing budget-limit warning.** Only `budget_constrained` warned when the
   budget left eligible customers untargeted; every strategy now reports it.
4. **Chart labelling.** The allocation chart lacked the hypothetical-projection
   label, and some single-series panels lacked legends. All charts now carry
   the projection note, labelled axes with units and legends.
5. **Unsafe generated scenario ids.** `slug` accepted non-ASCII alphanumerics,
   so a scenario named `Café` produced an id that `load_result` rejects. Slugs
   are now ASCII-only.
6. **Report robustness.** User-supplied names containing `|` or newlines could
   break Markdown tables; labels are escaped. Sensitivity warnings are now
   surfaced in the report and CLI, and exclusion counts are broken out by
   reason with a total.
7. **Sensitivity membership audit.** Each point now records a 16-hex
   fingerprint of the selected reference set, so set changes are verifiable
   without duplicating every customer id.
8. **Unit semantics and provenance.** `roi_is_ratio` and
   `budget_utilization_is_fraction` make the JSON units explicit; the manifest
   now shares the result timestamp and records duration.
9. **Error taxonomy.** The unused `CODE_MISSING_VALUE` code was removed and the
   artifact/export codes renumbered contiguously (`001`–`005`), with docs and
   tests updated.
10. **Empty sensitivity selection.** Passing an explicit empty dimension list
    fell back to varying every dimension, contradicting the documented "at
    least one dimension" rule. An explicit empty list now raises `CW-SIM-001`;
    omitting the argument still selects all dimensions. Covered by a test.

Independent re-derivation from the real artifacts confirmed: assumptions in
`result.json` equal the manifest; spend equals the sum of `selection.csv`
costs; remaining budget equals budget minus spend; the 142 selected
references equal the stored top-142 eligible Phase 9 rows; and every budget
sensitivity point recomputes exactly from its recorded configuration.

After the fixes: focused simulation tests **57/57** (47 unit + 10 integration)
and the full suite result recorded in §8.

## 13. Decisions and upstream gaps

- **Per-customer cost overrides** were added (optional) so
  `budget_constrained` has a genuine, testable difference from the other
  strategies; with a uniform cost the two are identical and the documentation
  says so.
- **No revenue inference.** The reference dataset has no charges column, so the
  simulator requires `monthly_value` and reports financial metrics as undefined
  otherwise. It never estimates customer value from tenure or contract fields.
- **Abstract unit default.** `value_units` avoids implying a currency when the
  caller supplies none; the unit travels with every metric and chart label.
- **Phase 9 gaps used as-is:** tier labels are the run's own (two or three
  boundaries), and Phase 9 does not record an observed retention outcome; the
  simulator consumes only probabilities, tiers, references and provenance.
