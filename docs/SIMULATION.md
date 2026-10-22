# Campaign Simulation (Phase 10)

The simulator combines **measured Phase 9 risk probabilities** with **explicit,
user-supplied campaign assumptions** to compare hypothetical retention
strategies. It is a planning calculator, not an experiment.

> Every saved-customer, saved-value, net-benefit, ROI and break-even number is a
> projection. Assumed effectiveness is an input, not a measured uplift. The
> simulator does not estimate causal intervention effectiveness and does not
> prove that a campaign will prevent churn.

## 1. Inputs

| Input | Source | Notes |
| --- | --- | --- |
| `churn_probability` | Phase 9 `RiskAssessment` | Never re-estimated; the pipeline's model is not invoked |
| `risk_tier` | Phase 9 configurable tiers | Tier names are validated against the loaded run |
| `record_reference` | Phase 9 opaque reference | Used for tie-breaking, selection CSVs and exclusion reporting |
| `risk_rank` | Phase 9 ranking | Retained for traceability only; selection re-orders deterministically |
| Model/run provenance | Phase 9 `RiskTable` | Copied into `result.json` and `run_manifest.json` |
| Budget, cost, effectiveness, value, horizon | Caller (CLI or Python) | Recorded verbatim in every artifact |

The reference OpenML `46915` dataset has **no revenue proxy**, so monthly value
is absent unless the caller supplies it. No revenue, lifetime value or customer
attribute is invented.

## 2. Campaign configuration

`CampaignScenario` bundles the budget, `CampaignAssumptions` and
`TargetingConfig`. All fields are validated before any calculation.

| Field | Default | Validation |
| --- | --- | --- |
| `name` | — | non-empty, ≤ 120 characters |
| `description` | `""` | string |
| `budget` | `1000.0` | finite, `>= 0`, `<= 1e12` |
| `cost_per_contact` | `5.0` | finite, `>= 0` |
| `effectiveness` | `0.30` | finite, in `[0, 1]` |
| `monthly_value` | `None` | `None` or finite `>= 0` |
| `horizon_months` | `6.0` | finite, `> 0`, `<= 1200` |
| `value_unit` | `value_units` | non-empty label, ≤ 24 characters |
| `strategy` | `highest_risk` | one of the four documented strategies |
| `min_risk_probability` | `None` | `None` or in `[0, 1]`; required for `risk_threshold` |
| `risk_tiers` | `()` | known tier labels; required for `risk_category` |
| `max_contacts` | `None` | `None` or integer in `[1, 10_000_000]` |
| `cost_overrides` | `{}` | record reference → finite cost `>= 0` |

`value_unit` defaults to the abstract `value_units` rather than assuming a
currency. Supply `USD`, `EUR` or another label if the numbers are real money;
all budget, cost and value fields then share that unit.

**Effectiveness definition.** `effectiveness = e` is the assumed probability
that an intervention prevents churn for a customer who would otherwise churn.
It is applied uniformly to every selected customer. Heterogeneous or
segment-specific uplift is out of scope.

Validation failures raise `SimulationError` with code `CW-SIM-001`, a message
naming every problem and a remediation string. Negative budgets, negative
costs, negative values, non-finite numbers, probabilities outside `[0, 1]`,
zero/negative horizons, unknown strategies or tiers, impossible strategy
combinations and `max_contacts < 1` are all rejected.

## 3. Targeting and eligibility

Eligibility is applied first, without any budget or count limit:

1. if `risk_tiers` is configured, the customer's tier must be in that set;
2. if `min_risk_probability` is configured, `p_i >= threshold`
   (lower-inclusive, matching Phase 9 tier semantics);
3. every other customer is recorded with an exclusion reason and detail.

Strategies:

| Strategy | Eligibility | Selection order |
| --- | --- | --- |
| `highest_risk` | configured threshold/tier rules | descending `p_i` |
| `risk_threshold` | requires `min_risk_probability` | descending `p_i` |
| `risk_category` | requires `risk_tiers` | descending `p_i` |
| `budget_constrained` | configured threshold/tier rules | descending `p_i`, stops at the first customer the remaining budget cannot cover |

**Stable tie-breaking.** Ordering is `(-churn_probability, record_reference,
input_position)`, so identical inputs and configurations always produce the
same selection, selection CSV and metrics. `budget_constrained` preserves a
strict risk-order prefix; the other strategies skip an unaffordable customer
when per-customer cost overrides make later, cheaper customers affordable.

Population accounting is explicit: `assessed = selected + below-threshold +
tier-not-targeted + budget-exhausted + contact-limited`. The engine raises
`CW-SIM-002` rather than returning a result whose counts do not reconcile.

## 4. Budget allocation

For each scenario:

```text
affordable_count (uniform positive cost) = floor(budget / cost_per_contact)
spend                                   = Σ actual intervention costs of selected customers
remaining_budget                        = max(0, budget - spend)
utilization                             = spend / budget        (None when budget == 0)
```

- Budget enforcement uses a `1e-9` tolerance scaled to the budget to avoid
  float-drift false rejections; `spend` never exceeds `budget` by more than
  that tolerance.
- **Zero-cost interventions:** the budget does not constrain the count,
  `max_affordable` is `None`, `utilization` is `0%` when the budget is positive
  and `None` when it is zero. ROI is undefined rather than infinite;
  `cost_per_expected_prevented_churn` is `0.0`.
- **Zero budget with paid interventions:** nothing is selected, every eligible
  customer is reported as `budget_exhausted`, and a warning states the budget
  is zero. No division by zero occurs.
- **Per-customer overrides:** when costs vary, `max_affordable` is `None` and a
  note explains that allocation is greedy in risk order.
- `max_contacts` is enforced before the budget check, so contact-limited
  customers are distinguishable from unaffordable ones.

## 5. Expected outcomes

For a selected set `S`, with `p_i` the Phase 9 probability and `e` the assumed
effectiveness:

```text
expected_prevented_churn    = Σ_{i∈S} p_i · e            (fractional expectation)
expected_saved_value        = expected_prevented_churn · monthly_value · horizon_months
campaign_cost               = Σ_{i∈S} intervention_cost_i
net_benefit                 = expected_saved_value - campaign_cost
roi                         = net_benefit / campaign_cost
cost_per_prevented_churn    = campaign_cost / expected_prevented_churn
break_even_effectiveness    = campaign_cost / (Σ_{i∈S} p_i · monthly_value · horizon_months)
random_contact_prevented    = mean(p over eligible) · |S| · e
targeting_lift              = expected_prevented_churn / random_contact_prevented
```

Undefined metrics are returned as `null` **and** listed in
`projection.undefined_metrics` with the reason:

| Metric | Undefined when |
| --- | --- |
| `expected_saved_value`, `net_benefit`, `roi`, `break_even_effectiveness` | `monthly_value` was not supplied |
| `roi` | campaign cost is zero |
| `cost_per_expected_prevented_churn` | expected prevented churn is zero |
| `break_even_effectiveness` | at-risk mass or average value is zero |
| `targeting_lift_vs_random` | the random-contact baseline is zero |

`break_even_effectiveness` above `1.0` is reported as-is and raises a warning:
no admissible effectiveness assumption recovers the cost. Fractional expected
counts are never rounded into claimed retained customers; every result carries
`"projection": true` and `"observed_retention_result": false`.

## 6. Scenario comparison

`compare_scenarios(results)` produces one row per scenario with budget, cost,
effectiveness, value unit, threshold, tiers, contact limit, selected count,
spend, remaining budget, utilisation, expected prevented churn, cost per
expected prevented churn, saved value, net benefit, ROI, break-even and
targeting lift.

- The comparison is **assumption-visible**: every input is a column, and
  `differing_assumptions` names the fields that vary.
- A **value-unit mismatch** is flagged in `differing_assumptions` and the notes,
  because monetary columns from different units are not directly comparable.
- `shared_inputs` is `true` only when the risk run id, model digest, assessed
  population size and tier boundaries all match. Otherwise each difference is
  listed in `input_differences`.
- The notes state that a larger projection under more optimistic assumptions
  does not make a scenario better, and that spend is simulated spend, not the
  configured budget. The comparison also carries the standard limitations and
  projection disclaimers.

## 7. Sensitivity analysis

`run_sensitivity` varies **one assumption at a time** while holding the rest of
the scenario fixed, and recomputes the full selection, allocation and
projection for each value.

| Dimension | Default grid | Unit |
| --- | --- | --- |
| `effectiveness` | `0.10, 0.20, 0.30, 0.40, 0.50` | probability |
| `budget` | `0.5×, 1×, 1.5×, 2×` the scenario budget | value unit |
| `cost_per_contact` | `0.5×, 1×, 2×` the scenario cost | value unit per contact |

Grids accept at least two and at most fifty finite values within their ranges;
duplicates are collapsed. A dimension whose grid collapses to a single value
(for example the default cost multipliers when the scenario cost is zero) is
skipped with a warning rather than rendered as a meaningless single point. Each
point records the selected count, a short fingerprint of the selected reference
set (so membership changes are auditable without duplicating customer lists),
spend, remaining budget, expected prevented churn, saved value, cost per
prevented churn and utilisation, plus an `is_baseline` flag. `fixed_parameters`
records everything that was held constant. Interactions between assumptions are
not explored, and sensitivity is never evidence of real-world effectiveness.

## 8. Artifacts

`artifacts/simulations/<scenario_id>/` (git-ignored):

| File | Contents |
| --- | --- |
| `result.json` | Full scenario, assumptions, population, allocation, projection, selected customers, exclusions, warnings, limitations, disclaimers, provenance |
| `selection.csv` | Selected references with probability, tier, Phase 9 rank, cost and expected prevented churn |
| `sensitivity.json` / `sensitivity.csv` | One-at-a-time series with fixed parameters |
| `comparison.json` / `comparison.csv` | Comparison rows (only when extra scenarios were supplied) |
| `report.md` | Markdown report with assumptions, projections, sensitivity tables and chart links |
| `plots/budget_allocation.png` | Spend versus remaining budget |
| `plots/sensitivity_<dimension>.png` | Expected prevented churn and spend across one dimension |
| `plots/scenario_comparison.png` | Selected counts and projected prevented churn per scenario |
| `run_manifest.json` | Scenario fingerprint, risk run/model provenance, artifact paths, disclaimers |

Every JSON/CSV artifact includes the assumptions that produced it. Provenance
records the risk run id, model digest, preprocessing run id, dataset SHA-256,
split fingerprint, scenario fingerprint and package version; `seed` is `null`
because selection and arithmetic are deterministic rather than sampled. CSV
cells are formula-injection sanitized. Customer references are Phase 9 opaque
identifiers; raw customer attributes and segments are never copied into
simulation outputs.

## 9. Stable interface for Phase 11

```python
from churnwise.config import Settings
from churnwise.simulation import (
    SimulationService, assume_scenario, CampaignScenario, SimulationError,
)

service = SimulationService(Settings.load())
scenario = assume_scenario(
    "Q4 retention pilot",
    budget=1000, cost_per_contact=5, effectiveness=0.30,
    monthly_value=50, horizon_months=6, min_risk_probability=0.30,
)

run = service.run_scenario(scenario, run_id="<phase 9 run id>")
run.result.to_dict()          # structured campaign metrics
run.comparison                # None unless compare_with was supplied
run.sensitivity.to_dict()     # one-at-a-time series
run.paths                     # artifact locations

service.load_result(run.scenario_id)   # read back the persisted result.json
service.latest_scenario_id()           # newest persisted scenario
```

`run_scenario` accepts an already-loaded `RiskTable` via `table=` (no re-read),
`compare_with=` for scenario comparisons, explicit sensitivity grids, and
`persist=False` for in-memory use. `SimulationService.load_table(run_id)` wraps
Phase 9 loading and freshness checks, raising `CW-SIM-004` if the run is missing,
stale or corrupt, and `CW-SIM-003` when no risk run exists. Scenario ids are
restricted to `[A-Za-z0-9][A-Za-z0-9._-]{2,180}` with no `..`, so artifact paths
cannot escape the simulations directory.

The dashboard must render these stored values, label them as projections and
never recompute risk probabilities in the UI process.

## 10. CLI

```bash
python -m churnwise.cli simulate \
  --budget 1000 --cost-per-contact 5 --effectiveness 0.30 \
  --monthly-value 50 --horizon-months 6 --min-risk 0.30 \
  --strategy risk_threshold --name "Q4 pilot"

python -m churnwise.cli simulate --budget 600 --cost-per-contact 12 \
  --effectiveness 0.45 --strategy risk_category --risk-tier critical \
  --sensitivity-effectiveness 0.1,0.2,0.3 --json
```

`--compare FILE` runs additional scenario JSON files against the same risk rows
and writes comparison artifacts. Exit codes follow the project convention:
`0` success, `2` structured failure (missing/stale run), `3` invalid
configuration or artifact-write failure.

## 11. Limitations

- Projections are not causal and are not measured uplift; assumed effectiveness
  is never validated against an experiment.
- A single uniform effectiveness and value is applied to every selected
  customer; no segment heterogeneity, saturation or contact fatigue.
- The random-contact baseline is a same-size draw from the eligible pool and
  assumes independence between contacts.
- Sensitivity is one-at-a-time; interactions between budget, cost and
  effectiveness are not explored.
- Budget and value share one unit label; the model does not convert currencies
  or discount cash flows over the horizon.
- No observed retention outcome exists in the Phase 9 dataset, so the
  projection cannot be scored against reality.
