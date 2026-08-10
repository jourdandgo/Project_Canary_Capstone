# Project Canary Engine Crash Course

## Where to see the exact modeling rows

Open `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx`. It turns the behind-the-scenes transformation into a visible audit trail:

- **Building Outcomes:** one row per historical building-cycle outcome
- **Recovery Training:** the exact 122 decision snapshots compared by the recovery candidates
- **Weight Training:** the exact 124 checkpoint rows compared by the Day 35 candidates
- **Recovery Daily Audit:** all 1,122 leakage-safe daily candidates before balancing
- **Data Dictionary:** X/Y/identifier roles, units, missing-data rules, and leakage warnings

The two executed notebooks load these CSV exports and assert that their rows match the regenerated canonical training data. Metrics in the app come from the same versioned manifests, including cycle-by-cycle performance.

The teammate comparison is not complete until the teammate’s notebook and trusted artifact are supplied. A pickle alone is not enough: the workflow must reproduce on Canary’s data and pass identical complete-cycle holdouts.

## 1. The one-minute explanation

Project Canary is an early-warning and decision-support system for broiler farms.

For every active building, it answers:

1. Is this building operationally at risk?
2. Why was it flagged?
3. What recovery and average weight are currently projected?
4. What should management check next?

Canary has three separate engines:

| Engine | Input | Process | Output |
|---|---|---|---|
| Rules-based risk | Recorded weight, population, mortality, temperature, humidity | Apply transparent 0–3 point rules | Low / Medium / High / Critical and reasons |
| Predictive models | Historical building-cycle snapshots | Compare candidates using unseen-cycle validation | Recovery forecast and Day 35 weight forecast |
| Recommendation playbook | Identified problem + risk severity | Match to Doc Raymond's rule table | Action, urgency, checklist, escalation trigger |

The risk label is **not** a probability and the predictive models do **not** change it.

## 2. What the review date means

The review date is an **as-of date**.

If the review date is 5 July 2026, Canary asks:

> What would we have known and recommended using only records dated on or before 5 July 2026?

Canary then:

1. Removes later rows from the calculation.
2. Finds the latest eligible observation for each building as of that date.
3. Recalculates all four risk dimensions.
4. Rebuilds both predictive-model inputs.
5. Recomputes forecasts, reasons, actions, and business value.

This is useful for daily operations and historical backtesting. It does not alter the source workbook.

## 3. Engine 1 — Rules-based operational risk

### Goal

Prioritize buildings using evidence management can verify directly.

### Four dimensions

| Dimension | Calculation | 0 points | 1 point | 2 points | 3 points |
|---|---|---:|---:|---:|---:|
| Weight gap | `max((target for weighing day − measured weight) ÷ target, 0)` | ≤5% | >5–10% | >10–30% | >30% |
| Population loss | `(beginning birds − current birds) ÷ beginning birds` | ≤3% | >3–5% | >5–7% | >7% |
| Daily mortality | `latest daily mortality ÷ beginning birds` | ≤0.1% | >0.1–0.2% | >0.2–0.3% | >0.3% |
| Environmental conditions | Worse of temperature or humidity distance outside the age range | Within both ranges | Temp ≤1°C or humidity ≤5 points outside | Temp >1–2°C or humidity >5–10 points outside | Temp >2°C or humidity >10 points outside |

Missing or stale evidence is **not scored**. It is never silently converted to zero.

### Tropical operating bands used

| Age | Temperature | Humidity |
|---|---:|---:|
| Days 1–6 | 29–33°C | 60–70% |
| Day 7 | 26–29°C | 60–70% |
| Days 8–13 | 26–29°C | 50–65% |
| Days 14–20 | 25–28°C | 50–65% |
| Days 21–27 | 24–27°C | 50–65% |
| Days 28–35 | 24–26°C | 50–65% |
| Day 36 onward | Day 35 bands carried provisionally | Day 35 bands carried provisionally |

The environmental score uses the **higher** temperature or humidity score so one environmental event is not counted twice. Daily temperature swing remains a supporting diagnostic, not a formal score.

### Total score to label

| Total | Label |
|---:|---|
| 0–1 | Low |
| 2–3 | Medium |
| 4–5 | High |
| 6–12 | Critical |

### Example

Suppose a Day 14 building has:

- 300 g recorded versus the 380 g target: 21.1% shortfall → 2 points.
- 4% population loss → 1 point.
- 0.12% daily mortality → 1 point.
- 30.5°C average temperature versus the 25–28°C Day 14 band: 2.5°C high → 3 points.

Total: `2 + 1 + 1 + 3 = 7` → **Critical**.

The main problem is **High Temperature**, because it has the highest individual score. Canary then selects the high-temperature action rule.

### Limits

- Cutoffs are expert rules, not learned causal thresholds.
- Tropical bands were supplied, but severity distances and Day 36+ carry-forward still need Doc Raymond's sign-off.
- A Low rating with missing evidence is not an all-clear.

## 4. Engine 2A — Harvest-recovery prediction

### Business question

Using only information known today, what last-recorded recovery proxy should we expect for this building, compared with 95%?

### Y target variable

`last recorded population ÷ beginning population`

This is the agreed capstone proxy. The source workbook does not contain a verified harvest-event flag.

### X input variables used by the champion

- Production day.
- Current percentage alive.
- Latest daily mortality per 1,000 beginning birds.
- Recent three-day mortality per 1,000.
- Mortality trend change.
- Daily and cumulative feed per 1,000 birds.
- Recent average temperature.
- Recent average humidity.
- Missing-value indicators added during preprocessing where relevant.

The champion deliberately excludes building identity and raw beginning-population size.

### Preprocessing

1. Convert 1,785 source rows to 1,666 unique building-day records.
2. Aggregate zone rows to one building-day before modeling.
3. Build leakage-safe as-of snapshots.
4. Retain Days 7, 14, 21, 28 and the latest eligible snapshot per building-cycle.
5. Median-impute numeric missing values inside each training fold.
6. Add missingness indicators and standardize Ridge inputs.

### Validation

- Five historical cycles.
- 25 independent building-cycle outcomes.
- 122 balanced training snapshots.
- Leave-one-complete-cycle-out cross-validation.
- Primary metric: cycle-macro MAE.
- Selection rule: choose the simplest candidate within 10% of the best cycle-macro MAE.

### Candidate results

| Candidate | MAE | Cycle-macro MAE | RMSE |
|---|---:|---:|---:|
| Ridge core | 1.32 points | 1.42 points | 1.76 points |
| Ridge without weight | 1.34 | 1.38 | 1.69 |
| Full Ridge | 1.50 | 1.56 | 1.92 |
| Random forest | 1.45 | 1.33 | 1.83 |
| Historical mean | 1.67 | 1.73 | 2.16 |
| Trend-naïve | 3.55 | 3.34 | 4.32 |

### Champion

**Compact Ridge (`ridge_core`)**.

Why:

- Lowest overall MAE.
- Within the 10% tolerance of the best cycle-balanced result.
- More stable and easier to explain than a small-data random forest.
- Avoids building identity and raw inventory size.

### Interpretation

- Held-out MAE: **1.32 percentage points**.
- Held-out RMSE: **1.76 points**.
- Empirical 80% error half-width: about **±2.25 points**.
- Day 14 MAE: **1.43 points** across 25 building outcomes.
- Target-side accuracy: **84%**, but this only equals the always-below majority baseline.
- The model correctly warned on below-95% outcomes but recognized **0 of 4** at-or-above-95% Day 14 outcomes.

Use the forecast as a continuous estimate with uncertainty. Do not describe it as a proven classifier of 95% target attainment.

### Day 14 actual-versus-projected example

For `2025-2 / Tags 1`:

1. Recreate the snapshot using only records available through Day 14.
2. Hold the entire `2025-2` cycle out of training.
3. Train the champion on the other cycles.
4. Projected recovery = **92.82%**.
5. Last-recorded actual proxy = **94.28%**.
6. Error = `92.82% − 94.28% = −1.46 percentage points`.

The negative error means Canary underestimated recovery by 1.46 points.

## 5. Engine 2B — Day 35 average-weight prediction

### Business question

Using weights recorded so far, what building average weight should we expect on Day 35, compared with 1,800 g?

### Y target variable

Observed building average bodyweight on production **Day 35**.

The interpolated target curve is not the Y label. It is an input/reference only.

### X input variables used by Ridge

- Measurement day.
- Current measured weight.
- Current weight ÷ target weight for that age.
- Recent average daily gain and whether it is available.
- Cumulative average daily gain since Day 7.
- Day 7, Day 14, Day 21 and Day 28 weights available by that checkpoint.

Future checkpoint weights are hidden.

### Preprocessing and validation

- Six historical cycles.
- 31 independent building Day 35 outcomes.
- 124 checkpoint rows: up to four views of each outcome.
- The latest current cycle is excluded from training.
- Leave-one-complete-cycle-out cross-validation.
- Median imputation and standardization inside training folds.
- Primary metric: cycle-macro MAE in kilograms.
- Selection rule: simplest candidate within 5% of the best.

### Candidate results

| Candidate | MAE | Cycle-macro MAE | RMSE | Within 200 g |
|---|---:|---:|---:|---:|
| Ridge regression | 172 g | 170 g | 232 g | 65% |
| Historical remaining gain | 178 | 182 | 242 | 65% |
| Random forest | 176 | 178 | 238 | 66% |
| Gradient boosting | 206 | 209 | 268 | 59% |
| Historical Day 35 mean | 210 | 213 | 276 | 52% |
| Target-curve ratio | 322 | 327 | 382 | 35% |
| Recent linear ADG | 432 | 445 | 541 | 31% |

### Champion

**Ridge regression**.

Why:

- Best cycle-macro MAE.
- Regularization is appropriate for small, correlated growth features.
- More stable and explainable than tree ensembles.
- Clearly beats the simple target-curve and recent-ADG approaches.

### Interpretation

- Overall held-out MAE: **172 g**.
- Overall held-out RMSE: **232 g**.
- Bias: **+7 g**, so there is little average over/under prediction.
- 65% of predictions are within 200 g.
- Day 14 MAE: **167 g** across 31 outcomes.
- Only 5 of 31 historical outcomes hit 1,800 g. Target-side accuracy is therefore affected by imbalance; at Day 14 the model recognizes 1 of 5 hits.

### Historical remaining gain

At each checkpoint:

1. For every eligible historical building, compute `observed Day 35 weight − checkpoint weight`.
2. Average those gains using training cycles only.
3. Add the average to the current building's checkpoint weight.

It remains a transparent benchmark and fallback. It is not the live champion while Ridge remains better.

## 6. Engine 3 — Recommendation playbook

### Inputs

- Identified highest-scoring problem pattern.
- Final risk severity.
- Evidence freshness.

### Process

1. Match the problem pattern to one deterministic playbook rule.
2. Match risk level to urgency.
3. Display action, inspection checklist, possible causes, responsible person, and escalation trigger.

### Examples

| Problem | Action shown |
|---|---|
| High temperature | Verify reading; check ventilation, airflow, cooling, heater state, and water within 6 hours |
| Low temperature | Verify reading; check heater condition/timing and drafts within 6 hours |
| High humidity | Check ventilation, litter moisture, leaks, drinkers, and water-pump timing |
| Low humidity | Review sensor, ventilation schedule, dust, and air movement |
| Low body weight | Reweigh; inspect health, feeder allocation, feed/water access, and temperature |
| High mortality | Confirm count; inspect flock condition and ventilation; escalate continuing or clinical concerns |

The source ideas come from Doc Raymond's Farmer Validation Workbook. Canary expanded them into dashboard wording and traceability. Final wording remains pending approval.

## 7. Completed-cycle farm KPI

The cycle-level final harvest recovery card is inventory-weighted:

`sum of estimated ending birds ÷ sum of beginning birds`

Equivalently, ending birds per building are recovered as:

`building actual recovery × building beginning inventory`

This is preferable to a simple average of six percentages because larger flocks contribute proportionally to the farm result. Buildings without a valid beginning/recovery pair are excluded and the card states how many buildings were included.

## 8. Why Canary does not use SMOTE

- Both forecasts are regression problems; standard SMOTE is for classification.
- Synthetic rows do not create new independent building-cycle outcomes.
- They may create biologically implausible flock histories.
- They can inflate apparent sample size and understate error.
- If applied before grouped validation, they can leak cycle information.

The defensible small-data approach is:

- Complete-cycle holdouts.
- Simple regularized models and naïve baselines.
- Cycle-balanced sampling.
- Metrics by forecast horizon.
- Empirical uncertainty ranges.
- Cycle-level bootstrap sensitivity intervals.
- Transparent limitations.
- More standardized completed cycles with verified harvest events.

## 9. What we can and cannot claim

### We can claim

- Canary is explainable and leakage-aware.
- It compares several forecasting candidates on unseen cycles.
- It exposes score calculations, model inputs, model reliance, uncertainty, and action provenance.
- It supports daily prioritization and historical Day 14 backtesting.

### We cannot yet claim

- A causal effect of temperature, humidity, feed, or weight on final outcomes.
- A clinically validated disease or heat-stress diagnosis.
- Guaranteed target attainment.
- Production-grade generalization beyond this farm.
- Verified final-harvest recovery until a harvest-event field is collected.

## 10. Where to show panelists

- **Home:** owner decision view and completed-cycle KPI.
- **Building View:** exact risk-score table, forecast inputs, local drivers, and action provenance.
- **Canary Methodology:** candidate tables, champion logic, feature reliance, and limitations.
- **EDA & Insights:** evidence behind the early-warning story.
- **These notebooks:** executable end-to-end model audit trails.
