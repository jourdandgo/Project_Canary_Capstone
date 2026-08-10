# Project Canary — Capstone Defense Cheat Sheet

## New transparent evidence package

Canary no longer asks the panel to trust an in-memory transformation. The repository includes:

- `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx`
  - Data Dictionary
  - 31 union building-outcome rows, including 25 recovery-eligible outcomes and 31 Day 35 weight outcomes
  - the exact 122 recovery-training snapshots
  - the exact 124 Day 35 weight-training rows
  - all 1,122 leakage-safe daily recovery candidates before checkpoint balancing
- `outputs/model_ready/recovery_training.csv`
- `outputs/model_ready/day35_weight_training.csv`
- two executed notebooks that rebuild the workflow and verify their rows against these exports

One building-outcome row is the final historical evidence for one building-cycle. A training snapshot is an earlier decision point containing only what management could have known on that review date. The snapshot design is what lets Canary evaluate Day 7, 14, 21, and 28 forecasts without exposing later measurements.

The versioned model manifests are the single source for metrics displayed in the app, notebooks, and defense material. The app now shows overall, Day 14, forecast-horizon, and held-out-cycle results.

### Teammate-model comparison

The comparison framework is ready, but the actual comparison is pending the teammate’s notebook, trusted model artifact, environment file, and preprocessing/feature specification. Canary first inventories the pickle without executing it, then reproduces the notebook approach on the canonical exports. The model must pass the same complete-cycle holdouts and leakage rules before it can replace a champion.

## 1. The 20-second answer

**Project Canary is an early-warning and decision-support system for a broiler farm.** It combines daily farm records into one view per building, assigns an explainable operational risk rating, forecasts Day 35 average weight and final harvest recovery, and recommends what management should inspect next.

The two farm goals are:

- **1,800 g average weight on Day 35**
- **95% harvest recovery**, defined for this capstone as ending or surviving birds divided by beginning population

Days 1–14 are the early-warning window. Monitoring continues through the active cycle.

## 2. Business question

> Which buildings are at risk, what Day 35 weight and harvest-recovery outcomes are currently expected, why are they at risk, and what should management check next?

## 3. The three components

| Component | What it answers | Output |
|---|---|---|
| Rules-based risk scoring | Which building needs attention and why? | Low, Medium, High, or Critical; 0–12 score; dimension evidence |
| Predictive models | What outcome is likely from what is known today? | Projected Day 35 average weight and projected final recovery, with ranges |
| Recommendation playbook | What should management inspect next? | Pattern-based action, urgency, checklist, and escalation trigger |

**Important:** the risk score and predictions are separate. A forecast does not add points to the risk score.

## 4. Data preparation

### One reliable building-day table

- Source rows: **1,785**
- Unique building-day rows: **1,666**
- Repeated building-days consolidated: **119**
- Production-value conflicts in repeated rows: **0**
- Recorded weight-measurement days: **210**

The canonical key is:

`Harvest cycle + building + production day`

### Zone A and Zone B

In 2026-2 and 2026-3, some building-days have one environmental row for Zone A and another for Zone B. Canary does **not** treat these as two independent flock observations.

For each affected building-day:

- average temperature/humidity = unweighted average of the two zone averages
- minimum = minimum across zones
- maximum = maximum across zones
- zone spread = absolute difference between zone averages
- production fields such as population, mortality, feed, and weight are retained once

Why unweighted? Zone-level flock counts or floor-area weights are unavailable. This is transparent and provisional.

### Corrected weight records

The updated workbook now contains the corrected checkpoint weights:

- 2026-1: Days 7, 14, 21, 28, and 35 for all six buildings
- 2026-2: Days 7, 14, 21, 28, and 35 for all six buildings
- 2026-3: Days 7, 14, 21, 28, and 35 for Tags 1–3

The latest cycle, 2026-3, is excluded from model training so current/future information cannot leak into validation.

## 5. Revised target-weight curve

Doc Raymond’s approved checkpoints are:

| Day | Target |
|---:|---:|
| 7 | 170 g |
| 14 | 380 g |
| 21 | 800 g |
| 28 | 1,200 g |
| 35 | 1,800 g |

Canary creates two daily curves:

- **Linear:** spreads each weekly gain evenly across seven days.
- **Smoothed, used by Canary:** preserves the previous farm curve’s proportional within-week growth shape, then rescales the segment to hit the revised weekly endpoints exactly.

Canary also tested a three-parameter **Gompertz growth curve**, because broiler growth is nonlinear. The fitted curve had about **20 g checkpoint MAE**, but missed individual approved checkpoints by as much as **42 g**. It is therefore shown as a scientific comparison, not used as the operating target. For daily risk comparisons, hitting every farm-approved checkpoint exactly is more important than forcing one global biological curve.

The Day 0 anchor remains a **40 g working assumption from the former curve**; it was not part of Doc Raymond's revised checkpoint list. Targets stay at 1,800 g after Day 35 for milestone comparisons.

The smoothed curve is used for:

- the rules-based weight-gap calculation
- current-weight-to-age-target progress in the Day 35 model
- dashboard target charts

It is **not** the model target. The model target is the actual recorded Day 35 weight.

## 6. Component 1 — Risk scoring

### Four dimensions

Each dimension scores 0–3 points.

| Dimension | Question | Evidence |
|---|---|---|
| Weight gap | How far below the smoothed target was the latest measured weight at that measurement age? | Actual weight versus target for the weighing day |
| Population loss | How much of the beginning flock has already been lost? | `(beginning − current population) ÷ beginning` |
| Daily mortality | Is there an urgent current loss? | Latest daily mortality ÷ beginning population |
| Environmental conditions | Is the recorded house environment outside the age-specific tropical range? | Worse of average-temperature deviation and humidity deviation |

### Weight-gap thresholds

| Gap below age target | Score |
|---:|---:|
| 5% or less | 0 |
| Above 5% to 10% | 1 |
| Above 10% to 30% | 2 |
| Above 30% | 3 |

### Other starting thresholds from the Farmer Validation Workbook

| Check | 0 points up to | 1 point up to | 2 points up to | 3 points above |
|---|---:|---:|---:|---:|
| Population loss | 3% | 5% | 7% | 7% |
| Daily mortality | 0.1% | 0.2% | 0.3% | 0.3% |
| Temperature outside age range | 0°C | 1°C | 2°C | 2°C |
| Humidity outside age range | 0 points | 5 points | 10 points | 10 points |

Temperature reference bands are 29–33°C for Days 1–6; 26–29°C for Days 7–13; 25–28°C for Days 14–20; 24–27°C for Days 21–27; and 24–26°C for Days 28–35. Humidity is 60–70% for Days 1–7 and 50–65% from Day 8 onward. The Day 35 bands are carried after Day 35 provisionally. The environmental score is the **higher** of the temperature and humidity scores, so related environmental evidence is not counted twice.

### Total score to label

| Total | Label |
|---:|---|
| 0–1 | Low |
| 2–3 | Medium |
| 4–5 | High |
| 6–12 | Critical |

### Why mortality trend and peer points were removed

- A worsening-versus-baseline trend can look safe once a high mortality level becomes steady.
- Peer points can repeat the same weight or mortality problem already scored elsewhere.
- The replacement checks use simpler building-level values that management can verify directly.

Peer comparison remains useful context, but it no longer adds points. Feed remains an alert only because some cycles contain inconsistent feed-per-bird magnitudes. Water and THI remain deferred until reliable inputs and approved formulas exist.

### Environmental freshness and honest limitation

- Direct environmental readings cover **706 of 1,666 building-days (42.4%)**.
- With Canary's maximum two-day carry-forward rule, **768 building-days (46.1%)** have sufficiently current environmental evidence.
- A stale value is never scored as safe. The dashboard shows the last environmental measurement day, its age, and why it was excluded.
- Example: on 2026-3 Day 22, the latest environment reading is from Day 17. At five days old it exceeds the two-day limit, so the dimension is **Not scored**.

Risk thresholds are provisional until Doc Raymond signs them off. The tropical operating bands have been supplied; the remaining validation question is how far outside a band should count as mild, moderate, or severe, and whether the Day 35 range should be carried forward. The score represents operational concern, not a statistical probability of missing a goal.

## 7. Component 2A — Day 35 weight model

### Business question and target

**Question:** Given the checkpoint weights known today, what average building weight should we expect on Day 35?

**Y:** observed building average bodyweight on production Day 35.

### Training unit

- **31 independent building-cycle Day 35 outcomes**
- **6 historical cycles:** 2025-2 through 2026-2
- **124 as-of checkpoint rows:** 31 outcomes × Days 7, 14, 21, and 28

The 124 rows are not 124 independent final outcomes. They are four decision snapshots for each of 31 outcomes.

### Leakage-safe inputs

At each checkpoint, only weights already known are included:

- Day 7 row: Day 7 weight only
- Day 14 row: Days 7 and 14
- Day 21 row: Days 7, 14, and 21
- Day 28 row: Days 7, 14, 21, and 28

Additional inputs:

- current measurement day
- current measured weight
- current weight divided by the smoothed target for that age
- recent average daily gain when a previous measurement exists
- cumulative gain from Day 7
- indicators for whether growth-history inputs are available

### Models compared

| Candidate | Held-out MAE | Result |
|---|---:|---|
| Historical Day 35 mean | about 210 g | Baseline |
| Target-curve ratio | about 322 g | Not selected |
| Recent linear ADG | about 432 g | Not selected |
| Historical remaining gain | about 178 g | Strong transparent benchmark |
| **Ridge regression** | **about 172 g** | **Selected** |
| Random Forest | about 176 g | Not selected |
| Gradient boosting | about 206 g | Not selected |

### Validation and selection

Canary uses leave-one-cycle-out cross-validation. Every row from a held-out cycle stays out of training.

Primary selection metric: **cycle-balanced MAE in kilograms**, reported to the business in grams.

Rule: choose the simplest candidate within 5% of the best cycle-balanced MAE. Historical remaining gain was more than 5% worse than Ridge, so Ridge became the champion.

### Selected-model results

- Overall MAE: **about 172 g**
- RMSE: **about 232 g**
- Within 200 g: **about 65%**
- Correct side of the revised 1.8 kg target: **about 86%**
- Day 14 MAE: **about 167 g**
- Day 14 correct target side: **about 87%**

There are only **5 historical 1.8 kg hits** and 26 misses. Across all four checkpoints, the model catches misses well but recognizes only about 15% of the small hit group; at Day 14 it recognizes about 20%. Do not oversell target-hit classification.

### Feature importance

Ridge importance is based on absolute standardized coefficients. The leading fitted inputs are target progress, Day 14 weight, recent gain, Day 21 weight, and cumulative gain.

These are model associations—not proof that changing one input causes the final result. Correlated growth variables can share importance or have counterintuitive signs.

## 8. Historical average remaining gain — full example

This method is still important because it is the strongest simple benchmark.

### How it is computed

For every eligible historical building-cycle at the same checkpoint:

`Remaining gain = actual Day 35 weight − checkpoint weight`

At Day 14, Canary averages this across the 31 historical building outcomes:

`Average Day 14-to-35 remaining gain = 1.254981 kg ≈ 1,255 g`

### End-to-end illustrative forecast

Suppose a current building weighs **380 g on Day 14**.

1. Current measured weight = 380 g.
2. Historical average remaining gain from Day 14 = 1,255 g.
3. Baseline projection = `380 + 1,255 = 1,635 g`.
4. Revised Day 35 goal = 1,800 g.
5. Gap = `1,635 − 1,800 = −165 g`.

Interpretation: the simple benchmark projects 1,635 g, or 165 g below goal.

During cross-validation, the held-out cycle is excluded before calculating the average remaining gain. This prevents the answer from using its own future result.

### Do we still use it?

- **Yes, as a benchmark and possible fallback.**
- **No, not as the live champion while Ridge remains better than the 5% tolerance.**

If future retraining shows Ridge no longer beats it reliably, Canary can safely fall back to this simpler formula.

## 9. Component 2B — Harvest-recovery model

### Business question and target

**Question:** Given the current daily flock evidence, what final recovery should we expect?

**Y:** last recorded population divided by beginning population for an eligible historical completed-record cycle.

This is the agreed capstone proxy. It is not a verified harvest-event count.

### Evidence and inputs

- **25 independent building-cycle outcomes**
- **5 eligible historical cycles:** 2025-2 through 2026-1
- **122 balanced as-of snapshots**

Selected compact inputs:

- cycle day
- current percentage alive
- latest daily mortality per 1,000 birds
- recent 3-day mortality per 1,000
- mortality trend versus baseline
- latest and cumulative feed per 1,000 birds
- recent temperature
- recent humidity

Raw beginning population and building identity were removed from the selected model. Corrected bodyweight did not materially improve recovery validation and is not in the champion.

**Why current survival belongs:** final recovery is ending population ÷ beginning population, while current survival is today’s population ÷ beginning population. It is the best known starting point for estimating how much more loss may occur. It is not future leakage because today’s population is already known at prediction time. Raw beginning population itself was removed because flock size should not mechanically raise or lower the recovery percentage.

### Models compared

| Candidate | Held-out MAE |
|---|---:|
| Current-survival trend projection | about 3.55 percentage points |
| Historical mean | about 1.67 points |
| Ridge with all tested inputs | about 1.50 points |
| Random Forest | about 1.46 points |
| Ridge without weight | about 1.34 points |
| **Compact Ridge** | **about 1.32 points** |

### Selection rule and result

Primary metric: cycle-balanced MAE, with a 10% simplicity tolerance for recovery.

Compact Ridge was selected because it had the lowest overall MAE, remained close to the best cycle-balanced candidate, and avoided questionable identity and raw-inventory features.

Random Forest had the best cycle-balanced MAE at about **1.33 points**. Compact Ridge was about **1.42 points**, which is within the predeclared 10% simplicity tolerance, and it had the best overall row-level MAE at about **1.32 points**. Ridge was therefore selected for stability and interpretability on a small dataset—not because it won every metric.

- Overall MAE: **about 1.32 percentage points**
- RMSE: **about 1.76 points**
- Day 14 MAE: **about 1.43 points**
- Empirical 80% half-width: **about ±2.25 points**

### Critical interpretation

Target-side accuracy is about 84%, but that equals the always-below majority baseline. At Day 14, all 25 forecasts were below 95%, including four flocks that later finished at or above 95%.

Therefore:

- useful as a continuous planning estimate
- useful for ranking likely recovery gaps
- **not proven as a classifier of who will hit 95%**

### Feature importance

Current survival has the strongest fitted reliance at about 27%. Recent three-day mortality lowers the estimate in the fitted model; cumulative feed, age, temperature, humidity, and missing-data indicators also contribute. One mortality-trend coefficient has a counterintuitive positive sign because correlated mortality variables share signal in a small dataset. Feature importance is therefore for model explanation—not a causal intervention rule.

## 10. Component 3 — Recommendations

Canary maps the leading scored trigger to Doc Raymond’s deterministic playbook. The action is traceable by rule ID, response time, possible causes to verify, and approval status.

| Pattern | Plain meaning | Recommended focus |
|---|---|---|
| No Material Concern | No scored warning is above the current limits | Continue normal monitoring |
| Low Body Weight | Weight is behind the age target | Confirm weight; check bird health, feeder allocation, feed/water access, and temperature |
| High Mortality | Latest daily mortality exceeds the limit | Confirm count; inspect bird condition and ventilation; escalate health concerns |
| Rapid Population Loss | Cumulative loss exceeds the limit | Reconcile population and mortality records; inspect for continuing loss |
| Abnormal Temperature Fluctuation | Daily max-minus-min range is too large | Verify sensors; check ventilation, fans, controller, heaters, cooling, and air leaks |
| High Humidity | Humidity is above the age range | Check ventilation, litter, leaks, drinkers, cooling pads, and pump timing |
| Low Humidity | Humidity is below the age range | Check sensor, ventilation schedule, air speed, dust, and weather |
| Low Feed Intake / Rapid Feed Drop | Feed evidence is low or falling | Verify unit and reading; check feed system, quality, access, water, heat, and bird condition |
| Poor Recovery Prediction | Forecast is materially below 95% | Review health, current loss, continuing mortality, and condemn information when available |
| Missing or Stale Evidence | Required evidence is absent or old | Collect the missing measurement before a major decision |

Urgency follows the risk label:

- Low: routine monitoring
- Medium: within 24 hours
- High: current shift
- Critical: immediate inspection and escalation where appropriate

Possible causes are hypotheses to inspect, not diagnoses. Recommendations do not prescribe medication or automatically change house settings.

**Provenance:** DOC-002 through DOC-010 retain trigger and action concepts from Doc Raymond's Farmer Validation Workbook. The Canary team expanded them into clearer dashboard wording, inspection checklists, and escalation guidance. DOC-001 and DOC-011 are team-authored safety fallbacks. All wording remains marked **Pending Review** until Doc Raymond approves it and an approval date is recorded.

## 11. Environment and actionability

Average-temperature deviation and humidity deviation now contribute through one combined environmental risk dimension. Specific operating alerts remain visible in the building investigation view.

Current tropical temperature references are:

- Days 1–6: 29–33°C
- Days 7–13: 26–29°C
- Days 14–20: 25–28°C
- Days 21–27: 24–27°C
- Days 28–35: 24–26°C

The Day 28–35 band is carried forward after Day 35 so monitoring does not stop; this carry-forward and the 1/2/3-point severity distances require farm approval. Humidity uses 60–70% on Days 1–7 and 50–65% from Day 8 onward. Feed alerts remain provisional. Water and THI require reliable inputs and approved formulas.

Canary should say “temperature is above the approved range; inspect ventilation/cooling and verify the sensor,” not automatically “lower temperature by 3°C.” A specific adjustment must depend on the verified sensor, bird behavior, equipment, housing, weather, and farm protocol.

## 12. Day 14 hypothesis — what the data supports

### Do higher Day 14 weights relate to higher Day 35 weights?

Yes, **directionally**:

- 25 paired building-cycles
- raw Pearson correlation: **r = 0.50**, p = 0.012
- within-cycle correlation: **r = 0.18**, p = 0.436
- interpretation: the overall relationship is moderate, but much weaker after comparing buildings within the same cycle; cycle conditions explain part of the pattern

Only **1 of 25** historical building-cycles met the revised 380 g Day 14 target. That flock recorded 1.81 kg on Day 35. This is encouraging, but far too little evidence for a reliable “met versus missed” claim.

### Does Day 14 weight relate to harvest recovery?

The relationship points upward but is not conclusive:

- raw correlation: **r = 0.25**, p = 0.230
- within-cycle correlation: **r = 0.29**, p = 0.207

### Do we need a separate Day 14 prediction?

No. Canary already:

- uses Days 1–14 as the early-warning window
- compares every measured weight with the daily target curve
- forecasts Day 35 weight from a Day 14 checkpoint
- reports Day 14 forecast error separately

A second “predict Day 14” model would add complexity without a new management decision. Before Day 14, the daily target gap already answers whether the flock is on track.

## 13. Worked building-card example — 2026-3 Tags 2 on Day 22

### A. Risk score

| Dimension | Evidence | Points |
|---|---|---:|
| Weight gap | 519.9 g measured on Day 21 versus 800 g target: 35.0% below | 3 |
| Population loss | `(7,114 − 6,763) ÷ 7,114 = 4.93%` | 1 |
| Daily mortality | latest daily mortality = 0.197% of beginning birds | 1 |
| Environment | last reading Day 17; five days old, above the two-day freshness limit | Not scored |

Total = `3 + 1 + 1 = 5` → **High risk**. Leading trigger: **Low Body Weight**.

### B. Projected harvest recovery

- Compact Ridge uses evidence available by Day 22.
- Projection: **91.87%**.
- Gap to the 95% goal: **3.13 percentage points below**.
- This is a model output, not a component of the 5-point risk score.

### C. Projected Day 35 weight

- Ridge uses the known checkpoint weights and growth features available by Day 22.
- Projection: **1,465 g**.
- Gap to the 1,800 g goal: **335 g below**.

### D. Problem and next action

- Identified problem: **Low Body Weight** because the 3-point weight gap is the largest scored trigger.
- Rule: **DOC-002**.
- Action: confirm the weight, then check bird health, feeder allocation, feed availability/quality, water access, and house conditions.

### E. Gross revenue at risk

Default planning assumptions: 2.0 kg sale weight and PHP120/kg.

- Birds at risk = `7,114 × (95% − 91.87%) = about 222 birds`
- Gross value per bird = `2.0 kg × PHP120 = PHP240`
- Gross revenue at risk = `222.47 × PHP240 = about PHP53,392`

This is an editable scenario estimate, not guaranteed recoverable profit.

## 14. Business value estimator

For a recovery shortfall:

`Birds at risk = beginning population × recovery gap`

`Gross revenue at risk = birds at risk × assumed sale weight × PHP per kg`

Example:

- beginning population = 10,000 birds
- projected recovery = 93%
- goal = 95%
- gap = 2 percentage points
- birds at risk = `10,000 × 0.02 = 200`
- assumed sale weight = 2.0 kg (an editable commercial assumption, separate from the 1.8 kg Day 35 milestone)
- price = PHP 120/kg
- revenue at risk = `200 × 2.0 × 120 = PHP 48,000`

This is a scenario estimate, not guaranteed recoverable profit. It excludes costs and assumes recovery improvement is achievable.

## 15. Expected panel questions

### “Why not train one model per building?”

Each building has too few completed outcomes. Canary pools all eligible buildings and cycles, then holds out whole cycles. A per-building model would be unstable and impossible to validate credibly.

### “Why are there 124 weight rows but only 31 outcomes?”

Each building-cycle is recreated at four decision checkpoints. Cross-validation groups the entire cycle so those related rows never split between train and test.

### “Why Ridge rather than Random Forest?”

Ridge had the best cycle-balanced error for Day 35 weight, is more stable on a small dataset, and is easier to explain. Random Forest was close but not better. Gradient boosting was worse.

### “Why not just use average daily gain?”

Straight-line ADG assumes the recent growth rate continues unchanged. In held-out testing its MAE was about 432 g, far worse than Ridge at about 172 g.

### “Why keep historical remaining gain?”

It is transparent, strong, and easy to audit. It is the benchmark the ML model must beat. Ridge beat it beyond the 5% tolerance, so Ridge is champion today.

### “Can we trust the recovery model?”

Trust it as a limited-data continuous estimate with about 1.3 points MAE—not as proof that a building will hit 95%. It did not beat the majority baseline for target classification.

### “Why did you select Ridge when Random Forest had the best cycle-balanced recovery MAE?”

Random Forest's cycle-balanced MAE was about 1.33 points. Compact Ridge was about 1.42 points, within the predeclared 10% simplicity tolerance, while having the best overall MAE at about 1.32 points. On only 25 building outcomes, Ridge is easier to explain and less likely to overfit. We disclose both results instead of claiming Ridge won every metric.

### “Do temperature and humidity drive the forecast?”

They are recovery-model inputs and operating checks. Average-temperature and humidity deviations also form one provisional environmental risk dimension. Sparse coverage prevents causal claims: use them to guide inspection, not to promise that one adjustment will change the outcome.

### “Is the app production ready?”

No. It is capstone-prototype ready after final farm validation. Production use still needs stronger data governance, confirmed harvest events, threshold approval, secure deployment, monitoring, and more cycles.

## 16. What we can defend strongly

- one-row-per-building-day data preparation
- correct handling of Zone A/B records without double-counting production
- corrected checkpoint-weight coverage
- rules-based score traceability
- cycle-held-out model validation
- three ML candidates plus transparent baselines for Day 35 weight
- compact, leakage-safe recovery model
- visible uncertainty and limitations
- deterministic recommendation mapping
- replacement of opaque trend/peer points with direct population, mortality, and environmental evidence

## 17. What we must not overclaim

- risk score is not a probability
- feature importance is not causality
- recovery target classification is not proven
- five Day 35 target hits are a small group
- unweighted zone averaging assumes equal relevance of Zones A and B
- the recovery target is a last-recorded proxy, not a verified harvest-event result
- recommended actions support inspection; they do not guarantee recovered revenue or target attainment
- environmental thresholds are expert starting rules, not historically calibrated causal cutoffs

## 18. Final one-sentence defense

> Project Canary converts daily building data into transparent operational priorities, leakage-safe outcome forecasts, and practical inspection guidance, while clearly separating what the evidence supports from what still requires farm validation.
