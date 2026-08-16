# Project Canary — External Modeling Context

## Purpose of this document

This is a platform-neutral context brief for an independent modeling review in Claude Code, Cursor, Codex, or another agentic coding environment.

It describes the business problem, available data, intended prediction outcomes, known data limitations, current benchmark, and expected evidence. It deliberately does **not** prescribe a model family, target transformation, feature set, validation design, or training recipe. The reviewing agent should inspect the evidence and propose its own defensible approach.

## 1. What Project Canary is

Project Canary is an early-warning and decision-support prototype for a commercial broiler farm. Its business purpose is to make production outcomes more consistent by giving management earlier visibility into buildings that are starting to go off-track.

The working hypothesis is:

> Earlier visibility → earlier investigation and action → more consistent recovery and growth outcomes.

Canary is intended to answer:

> Which buildings appear to be going off-track, what recorded evidence explains the concern, what harvest recovery and Day 35 bodyweight are currently expected, and what should management inspect next?

The two owner-facing targets are:

- **Harvest recovery:** 95% at the end of the recorded cycle.
- **Average bodyweight:** 1,800 grams on production Day 35.

Day 14 is the principal early-warning checkpoint from the original capstone proposal. Forecasts should also be useful later in the cycle as new observations become available.

Project Canary has three separate business-logic layers:

1. Deterministic operational risk scoring.
2. Predictive forecasts for recovery and Day 35 average bodyweight.
3. Deterministic recommendations mapped from recorded triggers to Doc Raymond's playbook.

This external review concerns the **two predictive forecasts**. Operational recommendations must not be inferred as causal treatments from model importance alone.

## 2. Canonical project location and relevant files

When this brief is used from the Project Canary repository, paths below are relative to `canary_app/`.

### Primary source

#### `data/FARM HARVEST DATA.xlsx`

This is the corrected primary source and should be independently inspected.

Known sheets:

- `Farm Harvest Data (Daily)` — daily production records at building/cycle/date level after source aggregation logic.
- `Temperature` — environmental source records; later cycles may include Zone A and Zone B rows.
- `Target Weights` — the farm's approved age-specific target-weight reference.
- `Farm Harvest Data (Weekly)` — weekly summaries or checkpoint-oriented records.
- `Farm Harvest Data (By Cycle)` — one-row-per-building-cycle summaries and outcomes.
- `Harvest Report (2)` — present in the workbook but may contain little or no usable data.

The daily source contains approximately 43 fields, including identifiers and dates; beginning and current population; mortality; feed; observed bodyweight; target weight and weight deficits; and minimum, maximum, and average temperature and humidity measures plus daily ranges.

### Transparent modeling and audit companion

#### `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx`

This workbook documents how the existing Canary implementation reconciled outcomes and constructed leakage-safe historical decision snapshots. It is an audit/reference artifact, not a required training design.

Known sheets:

- `Data Dictionary` — definitions, units, roles, sources, missing-data notes, and leakage warnings.
- `How to Read` — guide to the workbook.
- `Building Outcomes` — one row per building-cycle with reconciled outcome fields.
- `Recovery Training` — existing checkpoint/latest recovery snapshots.
- `Recovery Schedule` — existing recovery forecast schedule/checkpoint reference.
- `Weight Training` — existing Day 7/14/21/28 bodyweight snapshots.
- `Latest Cycle Weight Audit` — later-cycle bodyweight audit observations.
- `Latest Recovery Audit` — later-cycle recovery audit observations.
- `Recovery Daily Audit` — all eligible historical daily recovery snapshots before balancing.

Machine-readable companion exports:

- `outputs/model_ready/recovery_training.csv`
- `outputs/model_ready/day35_weight_training.csv`
- `outputs/model_ready/latest_cycle_recovery_audit.csv`
- `outputs/model_ready/latest_cycle_day35_audit.csv`

An independent reviewer may train from the raw workbook, use these prepared tables, rebuild different tables, or compare multiple representations. The prepared sheets should not be treated as proof that the current formulation is optimal.

### Other supporting data

#### `data/Project_Canary_Canonical_Building_Day_1666.xlsx`

An auditable historical canonical building-day export. Its filename preserves an earlier row count, while the newest cleaned source currently contains **1,624** unique building-day rows after unsupported forward-filled rows were removed. Treat the manifest/current source as authoritative for the newest count.

#### `data/Aggregated Temperature Data.xlsx`

Reference used to investigate the environmental grain. Earlier cycles generally have one building-day environmental record; later cycles may have Zone A and Zone B measurements. Multiple zone rows must not silently duplicate a flock-day during analysis.

#### `data/Weights Cleaned.xlsx`

Contains cleaned weight information and the latest interpolated in-house target curve. Observed flock weights must remain distinguishable from targets and any interpolated reference values.

#### `data/Project_Canary_Revised_Target_Weights.xlsx`

Daily target-weight reference based on Doc Raymond's checkpoints:

| Production day | Target weight |
|---:|---:|
| 7 | 170 g |
| 14 | 380 g |
| 21 | 800 g |
| 28 | 1,200 g |
| 35 | 1,800 g |

Intermediate daily targets are estimated reference values, not observed bird measurements.

#### `data/Farm Performance Summary.xlsx`

Used as a source of historical final average weights where applicable. Its recovery fields are not the agreed source for Canary's recovery proxy.

## 3. Dataset size and analytical grain

The source is longitudinal: the same building and harvest cycle appears on multiple dates. Daily rows are repeated observations of a single flock, not additional independent completed outcomes.

Current reconciled counts are:

| Item | Current count |
|---|---:|
| Cleaned unique building-days | 1,624 |
| Total recorded building-cycles including 2026-3 | 34 |
| Historical development building-cycles | 31 across six cycles |
| Existing recovery training snapshots | 151 |
| Existing Day 35 weight training snapshots | 124 |
| Later 2026-3 building outcomes | 3 |

The six historical development cycles are:

- 2025-2
- 2025-3
- 2025-4
- 2025-5
- 2026-1
- 2026-2

The latest 2026-3 data contains three Tags buildings and observed Day 35 weights. In the current Canary evaluation it is separated as a later-cycle audit rather than being used to choose the approach. Its recovery endpoint is currently based on the last recorded Day 35 population and remains subject to farm-owner confirmation as the temporary cycle endpoint.

Snapshot row counts must not be described as independent sample counts. The independent biological/business unit is the **building-cycle**.

## 4. Prediction problem A — harvest recovery

### Business outcome to estimate

The requested owner-facing output is the building's final harvest-recovery proxy, expressed as a percentage and compared with the 95% goal.

For this capstone, the agreed proxy is:

`final recovery proxy = last recorded population ÷ beginning population`

The source does not yet contain a consistently verified harvest-event flag or fully audited transfer/cull/partial-harvest reconciliation. Therefore, this is a proxy based on the last recorded population, not a claim of biologically perfect final recovery.

### Target variable

The **business Y** is:

`final_recovery_proxy_y`

or the same result expressed in percentage points.

The current Canary implementation instead fits an internal target:

`additional_population_loss_y = current percentage alive − final recovery proxy`

and reconstructs:

`predicted final recovery = current percentage alive − predicted additional population loss`

That formulation is the current benchmark, not a requirement. An independent review may compare direct final-recovery prediction, remaining-loss prediction, or another defensible representation.

### Information potentially available by a review date

The raw data may support identifiers and timing; current survival/population loss; recent and cumulative mortality; observed bodyweight and gap from the age target; measurement freshness; feed fields; temperature and humidity levels, ranges, deviations, and exposure histories; building group; and missingness/freshness indicators.

These are candidate evidence sources, not a mandated feature list. Their units, completeness, timing, redundancy, and leakage risk must be checked before interpretation.

### Essential outcome and timing facts

- Current percentage alive is known at the review date; final population and final recovery are not.
- A forecast made on Day 14 cannot use Day 21/28/35 measurements or later mortality.
- Ending population, final recovery, future mortalities, and future-derived quantities are outcomes or leakage for an earlier forecast.
- The endpoint date can vary by building-cycle; it is not necessarily Day 48.
- The 95% threshold is a management goal. A continuous forecast is not automatically a calibrated probability of hitting that goal.

## 5. Prediction problem B — Day 35 average bodyweight

### Business outcome to estimate

The owner-facing output is average building bodyweight on production Day 35, expressed in grams and compared with the 1,800 g milestone.

### Target variable

The **business Y** is:

`actual_day35_weight_kg_y`

or the equivalent observed Day 35 average weight in grams.

The current Canary implementation uses remaining growth as its internal prediction target:

`remaining_gain_to_day35_y = observed Day 35 weight − latest observed weight`

and reconstructs:

`predicted Day 35 weight = latest observed weight + predicted remaining gain`

Again, this is the current benchmark rather than a required formulation. Direct Day 35 prediction, remaining-growth prediction, biological curve approaches, longitudinal methods, or other defensible formulations may be evaluated independently.

### Measurement pattern

- Bodyweight is not observed every day.
- Some cycles have weekly checkpoints around Days 7, 14, 21, 28, and 35.
- Later cycles may have more frequent observations during Days 1–14.
- The same observation may remain the latest known weight on subsequent days, but it must not be presented as a newly measured daily value.
- Target and interpolated curve values are reference standards, not observed flock weights.
- A Day 14 forecast cannot use Day 21, Day 28, or Day 35 weight.

Potentially relevant evidence includes all observed weight history available by the review date, growth rate, gap or ratio to the target curve, measurement timing and staleness, mortality/population loss, survival, feed, and environmental exposure. Whether these improve out-of-sample forecasting is an empirical question.

## 6. Known data-quality and interpretation issues

These issues should be made visible in any independent review:

1. **Limited independent outcomes.** There are 31 historical development building-cycles, even though longitudinal snapshots create more rows.
2. **Mixed environmental grain.** Later cycles may contain Zone A/B records, while earlier cycles commonly have one row per building-day.
3. **Uneven weight measurement.** Checkpoints and measurement frequency differ across cycles.
4. **Observed versus reference weights.** Target, interpolated, and extrapolated weights must not be mistaken for observed flock weights.
5. **Endpoint uncertainty.** Recovery uses the last recorded population because a consistently verified harvest event is unavailable.
6. **Feed-unit uncertainty.** Feed fields exist, but their exact units and interpretation remain pending confirmation.
7. **Temporal leakage risk.** Daily records contain fields that become available only later; an earlier forecast must not use them.
8. **Grouped dependence.** Rows from the same building-cycle and cycles exposed to shared conditions are correlated.
9. **Outcome imbalance.** Most historical buildings fall below the 1,800 g or 95% targets, so plain target-side accuracy can be misleading.
10. **Observational evidence.** Feature importance and SHAP describe model reliance or association, not proof that a feature caused the outcome.

## 7. Current benchmark—not a required design

The current implementation is a validated experimental prototype and provides a benchmark for comparison.

### Existing recovery benchmark

- Version: `recovery-3.2.0`
- Current internal target: additional population loss after the review date.
- Current selected model: ordinary linear remaining-loss regression.
- Historical development: 31 building-cycles, 151 snapshots.
- Held-out snapshot MAE: approximately **1.74 recovery percentage points**.
- Cycle-balanced MAE: approximately **1.76 percentage points**.
- RMSE: approximately **2.57 percentage points**.
- Held-out R²: approximately **0.054**.

Interpretation: the continuous estimate has directional value but explains little unseen-cycle variation and should not be presented as a reliable guarantee of 95% attainment.

### Existing Day 35 weight benchmark

- Version: `day35-weight-2.2.0`
- Current internal target: remaining gain from latest measurement to Day 35.
- Current operational method: historical remaining-gain baseline.
- Historical development: 31 building-cycles, 124 checkpoint snapshots.
- Held-out MAE: approximately **178 g**.
- Cycle-balanced MAE: approximately **182 g**.
- RMSE: approximately **242 g**.
- Held-out R²: approximately **0.126**.
- Within 100 g: approximately **40%**.
- Within 200 g: approximately **65%**.

Interpretation: it is an approximate outlook with substantial uncertainty. The existing learned challengers have not yet demonstrated a sufficiently stable improvement to replace the transparent baseline operationally.

The external review should treat these as honest comparison points, not scores that must be preserved or beaten through a particular method.

## 8. Expected outputs from an independent modeling review

For **each prediction problem**, the desired deliverables are:

1. A reproducible notebook or code pipeline that begins from source inspection.
2. A concise description of the analytical grain and the resulting independent outcome count.
3. A transparent data dictionary showing:
   - Outcome Y.
   - Candidate X variables actually used.
   - Units and transformations.
   - When each variable becomes available.
   - Missingness treatment.
   - Leakage assessment.
4. Relevant EDA covering outcome distribution, coverage, missingness, cycle/building differences, measurement timing, outliers, and important relationships.
5. A fair comparison of **at least five** reasonable approaches, including a simple benchmark.
6. A model-comparison table with, at minimum:
   - MAE.
   - RMSE.
   - R².
   - Bias.
   - Stability or variability across evaluation groups/time periods.
   - Outcome-specific business metrics.
7. Recovery-specific metrics:
   - Errors expressed in recovery percentage points.
   - Performance by review age/checkpoint.
   - 95% target-side confusion matrix, class recall, and comparison with the majority rule.
8. Weight-specific metrics:
   - Errors expressed in grams.
   - Performance by review age/checkpoint.
   - Percentage within 100 g and 200 g.
   - 1,800 g target-side results and comparison with the majority rule.
9. Actual-versus-predicted, residual, error-by-review-age, and stability visualizations.
10. Feature importance and SHAP for suitable finalists, with both global and individual-building explanations.
11. An explicit warning that feature importance and SHAP show predictive association, not causation.
12. Uncertainty estimates or prediction intervals, including an empirical coverage assessment.
13. A separately labelled later-cycle audit after the approach is finalized.
14. Exported champion artifact/pipeline, feature schema, hyperparameters, predictions, metrics, and supporting figures.
15. A plain-language conclusion covering:
    - What the model can be trusted to do.
    - What it cannot yet do.
    - Whether it improves credibly on a simple benchmark.
    - Which additional farm measurements would most improve reliability.

No particular algorithm, feature-engineering method, split strategy, or target transformation is mandated by this document. The independent agent is expected to propose and justify those choices from the data and intended use.

## 9. Success standard

The goal is not to maximize an in-sample score. The goal is to identify the strongest model whose reported performance remains credible for future, unseen farm data and whose outputs can be explained honestly to both a farm owner and a technically knowledgeable capstone panel.

Any improvement should be judged against simple baselines, uncertainty, stability, data leakage, the small number of independent outcomes, and operational interpretability.

## 10. Open domain decisions

The following require farm-owner confirmation and should be shown as unresolved assumptions where relevant:

1. Whether Day 35 population is an acceptable temporary recovery endpoint for 2026-3.
2. What forecast error is acceptable for management use.
3. The exact units and operational meaning of feed-intake fields.
4. Final approval of environmental thresholds and interventions.
5. A standardized bodyweight sampling procedure across buildings and cycles.

