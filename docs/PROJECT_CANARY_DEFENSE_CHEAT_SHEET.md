# Project Canary — Capstone Defense Cheat Sheet

**Purpose:** A concise reference for explaining what Canary is, how it works, what the evidence supports, and where its limits are.

## 1. The 30-second answer

**What is Project Canary?**

- A daily early-warning and decision-support system for a broiler farm.
- It shows which buildings need attention, why they were flagged, what Day 35 weight and harvest recovery are currently expected, and what management should inspect next.
- It uses Days 1–14 as the early-warning window but continues monitoring the current flock after Day 14.

**Business question**

> Based on everything known today, which buildings need attention, what Day 35 weight and harvest recovery should we expect, why are they being flagged, and what should management check next?

**Two targets, one decision workflow**

- **Growth target:** at least **2,000 g average weight on Day 35**.
- **Survival target:** at least **95% harvest recovery**.
- These represent different outcomes: growth performance and birds remaining.

## 2. The one-minute workflow

1. Load the farm workbook.
2. Select the current harvest cycle and an as-of review date.
3. Use only records available on or before that date.
4. Score four operational warning signs for every building.
5. Estimate harvest recovery and project Day 35 average weight.
6. Explain the risk-score components and the model drivers.
7. Check current temperature, humidity, feed, and mortality against provisional farm thresholds.
8. Lead with the most specific supported operating alert and match it to a management action.

**Important distinction**

- The **risk score** prioritizes attention.
- The **models** estimate the two outcomes.
- The **operational alert layer** identifies a measurable condition management can investigate.
- The **playbook** recommends the next inspection or adjustment toward an approved target range.
- The risk label is **not** a probability of missing either target.

## 2A. How every current-building card is produced

| Card output | How Canary produces it |
|---|---|
| Current state | Latest population and weight observations recorded on or before the selected review date |
| Risk rating | Sum of the available 0–3 point weight, survival, mortality-trend, and peer checks; total mapped to Low/Medium/High/Critical |
| Main reason | Most specific current operational alert when supported; otherwise the strongest rules-based risk driver |
| Projected harvest recovery | Selected compact Ridge model using age, current survival, mortality, feed, and available temperature/humidity inputs |
| Projected Day 35 weight | Latest measured weight plus historically expected remaining gain from that measurement age to Day 35 |
| Identified problem | Deterministic pattern assigned from which risk dimensions are elevated and whether the concern is localized or farm-wide |
| Next action | Specific operational-alert action when supported; otherwise the deterministic playbook rule for the identified pattern, with urgency from the risk rating |
| Gross revenue at risk | Beginning birds × recovery gap to 95% × assumed sale weight × assumed PHP price per kg |

### Worked current-cycle example — Tags 2 on Day 22

This is the result using records available through **25 July 2026** for cycle **2026-3**.

### A. Risk rating — Medium, 3/12

| Risk check | Current calculation | Day 22+ threshold and score |
|---|---|---|
| Weight gap | No measured building weight in the current cycle | Not scored; shown as missing rather than zero |
| Survival reference gap | Beginning 7,114; latest population 6,763; currently alive = 6,763 ÷ 7,114 = **95.07%**. Provisional reference = 1 − (1 − 95%) × 22 ÷ 35 = **96.86%**. Gap = **1.79 points**. | 0: ≤0.75; 1: >0.75–2; 2: >2–4; 3: >4 points. **Score 1.** |
| Mortality trend | Recent three-day mortality = 1.36/1,000; preceding baseline = 1.51/1,000. Worsening amount = max(1.36 − 1.51, 0) = **0.00/1,000**. | 0: ≤0.15; 1: >0.15–0.50; 2: >0.50–0.95; 3: >0.95/1,000. **Score 0.** |
| Peer comparison | Recent mortality is **0.38/1,000 worse** than the median of Tags 1 and Tags 3. | 0: ≤0.10; 1: >0.10–0.30; 2: >0.30–0.60; 3: >0.60/1,000. **Score 2.** |

**Total:** survival 1 + mortality 0 + peer 2 = **3/12**. Weight is not scored. Rating bands are Low 0–1, Medium 2–3, High 4–5, and Critical 6–12. Therefore **3 = Medium risk**, with **3 of 4 checks supported by data**.

**How to say this:** “By Day 22, the temporary straight-line reference allows 3.14% cumulative loss, so 96.86% would remain. The record shows 95.07% remaining. The 1.79-point shortfall receives 1 point.” This reference is a configurable management benchmark, not a learned biological curve.

### B. Projected harvest recovery — 91.9%

1. Canary builds the Day 22 input row: age, current survival, latest and recent mortality, mortality trend, latest and cumulative feed, and recent temperature and humidity when available.
2. Missing values are imputed from training data and represented by explicit missing-data flags; numeric inputs are standardized.
3. The selected compact Ridge model calculates an unconstrained estimate of **91.8728%**.
4. Canary applies the accounting safeguard: final recovery cannot exceed survival already recorded today. `min(91.8728%, 95.0661%) = 91.8728%`.
5. Displayed prediction: **91.9%**. Gap to the 95% goal: **3.13 percentage points below**.
6. The empirical 80% likely range is approximately **89.6%–94.1%**.

The forecast is a model estimate, not part of the 3/12 risk-score calculation and not proof that a particular input caused the result.

### C. Projected Day 35 average weight — unavailable for this current card

- Tags 2 has **no measured building weight in cycle 2026-3**.
- Canary therefore displays **Not available / Needs a measured weight** instead of giving every building a generic prediction.
- When a measurement exists, the calculation is:

> Projected Day 35 weight = latest measured weight + historical remaining gain from that measurement age

**Illustrative recorded example:** Lags 2 in cycle 2026-2 measured **172.5 g on Day 14**. The training history's average remaining Day 14-to-35 gain is **1,264.3 g**. The projection is therefore **172.5 + 1,264.3 = 1,436.8 g**, or **563.2 g below** the 2,000 g goal. Its empirical range is approximately **1,197.7–1,675.9 g**. This example demonstrates inference; 2026-2 cannot improve training until a reliable Day 35 outcome is supplied.

### D. Identified problem

- Formal risk pattern: **Localized Building Drift**.
- Reason: the peer-comparison check is elevated, while the evidence does not meet the farm-wide-drift rule.
- More specific current operating alert: **Feed intake is 43% below the provisional Day 22 target**—74 g/bird recorded versus 130 g/bird proposed.
- The operating alert is displayed as the practical main reason because it is more actionable, but it does **not** add points to or alter the 3/12 risk score.

### E. Proposed solution

- Medium risk sets the urgency to **within 24 hours**.
- The current feed alert supplies the specific action: confirm the unit and reading; check feed availability and quality, feeder allocation and line operation, bird access, water access, house temperature, and flock condition.
- The fallback pattern rule is **ACT-005**, which calls for comparison with better-performing peers and inspection for a building-specific problem.
- These rules remain preliminary until Doc Raymond approves them.

### F. Gross revenue at risk — approximately PHP 53,392

Default planning assumptions are **2.0 kg sale weight** and **PHP 120/kg**, so gross revenue per additional surviving bird is `2.0 × 120 = PHP 240`.

1. Recovery gap = `95.0% − 91.8728% = 3.1272 percentage points`.
2. Birds associated with the gap = `7,114 × 3.1272% = 222.47 birds`.
3. Gross revenue at risk = `222.47 × PHP 240 = PHP 53,392.15`.
4. Displayed rounded value = **PHP 53,392**.

This is gross revenue associated with closing the modeled recovery gap—not profit, guaranteed savings, or proof that Canary caused an improvement. Price and sale-weight assumptions are editable in **Business Value**.

### Other card details to defend

- **Current versus projected:** current recovery is observed so far; projected recovery estimates the later outcome.
- **Likely range:** derived from held-out historical errors, not a guarantee.
- **Data coverage:** “3/4 risk checks” warns that weight evidence is missing.
- **Model/rule versions:** retained in Building View and Canary Methodology for reproducibility.
- **Review date:** freezes all calculations to information recorded on or before that date.
- **Separation of layers:** the risk score, forecasts, operational alerts, recommendation rules, and business-value assumptions are intentionally calculated independently and then combined only for presentation.

## 3. Data foundation

- Primary daily source: **FARM HARVEST DATA.xlsx**.
- Final average weight for completed-cycle display: **Farm Performance Summary.xlsx**.
- Age-specific weight targets: farm **Target Weights** sheet.
- Early detailed weights: **Weights Cleaned.xlsx**, available only for limited buildings/cycles.
- Farm thresholds and interventions: **Farmer Validation Workbook.xlsx**, excluding its Risk Scoring Matrix as instructed.

**Cleaning performed**

- Started with **1,785 source rows**.
- Consolidated **119 repeated rows**.
- Produced **1,666 unique building-day records**.
- Found **0 blocking duplicate conflicts**.
- Normalized dates, building names, age, population, mortality, feed, weight, temperature, and humidity.
- Kept missing values as missing; blanks were not silently changed to zero.
- Auditable output: **data/Project_Canary_Canonical_Building_Day_1666.xlsx**.

**What the 119 repeated rows were**

- They are 119 cycle-building-age keys that each occurred twice, or **238 source rows in total**.
- The repeated days are 2026-2 Lags 1 Days 1–30, Lags 2 Days 1–22, Lags 3 Days 1–16, and 2026-3 Tags 1–3 Days 1–17.
- Production values matched within every repeated key; the rows contained different environmental readings, consistent with multiple environmental sections or zones.
- Example: 2026-2 Lags 1 Day 1 has identical beginning population, ending population, mortality, feed, and weight values in both source rows. The average-temperature readings are 31.85°C and 33.12°C, so the canonical average is their mean; minimum and maximum readings retain the overall minimum and maximum.

**Leakage protection**

- Every prediction uses only information available by the review date.
- Model validation holds out entire harvest cycles.
- Daily rows from the same cycle never appear in both training and validation.

## 4. Component 1 — Rules-based risk score

### What it answers

- Which building should management inspect first?
- Which recorded warning signs explain the concern?

### Four dimensions

1. **Weight gap:** latest measured weight versus the target for the actual weighing day.
2. **Survival reference gap:** current percentage alive versus a provisional straight-line path to 95% on Day 35.
3. **Mortality momentum:** recent three-day mortality versus the preceding baseline.
4. **Peer context:** performance versus similar-age buildings in the same cycle.

### Point system

- Each available dimension receives **0–3 points**.
- Total score: **0–12**.
- Missing evidence is labelled **not scored**, not zero.

| Total | Rating |
|---:|---|
| 0–1 | Low |
| 2–3 | Medium |
| 4–5 | High |
| 6–12 | Critical |

### Main thresholds

**Weight shortfall at all ages**

- 0 points: up to 5% below target.
- 1 point: over 5% through 15%.
- 2 points: over 15% through 30%.
- 3 points: over 30%.

**Survival and mortality thresholds**

- Become more or less tolerant by production-age band.
- Full threshold tables and editable settings are available in **Data & Settings**.
- Peer scoring requires at least three comparable buildings within two days of age.

**Survival-reference formula**

> Reference survival on Day d = 100% − (5% × min(d, 35) ÷ 35)

- Day 22: `100% − (5% × 22 ÷ 35) = 96.86%`.
- This simply spreads the allowed 5% loss evenly through Day 35.
- It is not learned from the historical data and is not claimed to be a biological mortality curve.

### How to explain a High/Critical label

> “The building is High risk because its total operational-attention score is 5/12: weight gap contributed 3 points and worsening mortality contributed 2 points. Survival and peer checks contributed zero.”

### Can we trust it?

**Trust it for:**

- Consistent operational prioritization.
- Exact, traceable calculations.
- Showing which measurements and thresholds produced the label.

**Do not claim:**

- It is the statistical probability of missing 2,000 g or 95%.
- Its current thresholds are fully farm-validated.
- A higher score independently proves a worse final outcome.
- A performance-risk dimension identifies the physical root cause by itself.

**Historical audit**

- Day 14 snapshots reviewed: **31**.
- Score correlation with last-recorded recovery: **+0.19**.
- Score correlation with recorded Day 35 weight: **+0.04**.
- Risk bands were not consistently ordered by final outcomes.
- Conclusion: retain the score as an **operational attention tool**, not an outcome-prediction model.

## 5. Component 2A — Harvest-recovery model

### What it predicts

- Expected recovery at harvest, compared with the **95% goal**.

### Historical target variable (Y)

- **Last recorded population ÷ beginning population** for each completed building-cycle.
- This is the agreed capstone recovery formula.
- It is a proxy because the source does not contain a verified harvest-event flag.

### Input features (X)

- Flock age.
- Current percentage alive.
- Latest and recent mortality.
- Mortality trend.
- Daily and cumulative feed per 1,000 birds.
- Recent temperature and humidity when available.
- Missing-data indicators.

Weight inputs were tested but excluded from the winning recovery model because held-out error did not improve.

### Models compared

| Candidate | Overall MAE | Result |
|---|---:|---|
| Current-survival projection | 3.55 pts | Transparent but weak |
| Historical mean | 1.67 pts | Useful baseline |
| Ridge with weight | 1.40 pts | Competitive |
| Random forest | 1.48 pts | Did not justify added complexity |
| Ridge without weight | 1.34 pts | Competitive |
| **Compact Ridge without weight, inventory, or building identity** | **1.32 pts** | **Selected** |

### Why compact Ridge won

- Lowest overall held-out MAE.
- Within 5% of the best cycle-balanced candidate.
- More compact and explainable than the tree model.
- Removed weight, raw beginning inventory, and Tags/Lagundi identity because those inputs did not strengthen the defensible held-out result.
- Kept current survival because final recovery uses the same beginning-population denominator: final recovery cannot exceed the share still alive today.

### Validation results

- Complete cycles used: **5**.
- Distinct building outcomes: **25**.
- Balanced decision snapshots: **122**, sampled from 1,122 eligible daily snapshots.
- Overall MAE: **1.32 recovery points**.
- Overall RMSE: **1.76 recovery points**.
- Day 14 MAE: **1.43 recovery points**.
- Prototype likely range: approximately prediction ± the 80th-percentile held-out error.

### Important classification limitation

- Target-side accuracy: **84%**.
- Always predicting the historically common “below 95%” result also gives **84%**.
- At/above-95% recall: **0%**.
- Use it as a **directional point estimate and gap estimate**, not a proven hit/miss classifier.

### Top five recorded model inputs

These are model-wide standardized coefficient magnitudes, not causal effects.

| Recorded input | Relative reliance | Fitted direction |
|---|---:|---|
| Current survival | 26.9% | Higher raises the estimate |
| Recent 3-day mortality | 9.8% | Higher lowers the estimate |
| Cumulative feed per 1,000 birds | 9.4% | Higher raises the estimate |
| Flock age | 9.4% | Higher raises the fitted estimate |
| Mortality-trend feature | 9.1% | Fitted positive; interpret cautiously because mortality features overlap |

**Interpret carefully**

- Temperature and humidity values together represent about **4.6%** of coefficient magnitude.
- Missing temperature/humidity flags represent about **24.4%**.
- This means data availability is entangled with historical patterns.
- It does not prove that changing one input will produce the coefficient-sized outcome change.

## 6. Component 2B — Day 35 weight model

### What it predicts

- Expected average building weight specifically on **production Day 35**.
- It is not final weight at an unknown harvest date.

### Historical target variable (Y)

- Actual recorded average bodyweight on **Day 35** in FARM HARVEST DATA.xlsx.
- The model does not convert a later final weight by multiplying by 35/49.

### What was one training example?

- One example represented **one building at one standard checkpoint**: Day 7, 14, 21, or 28.
- Its Y value was that same building's observed Day 35 average weight.
- The data were pooled across every eligible building and historical cycle; Canary did **not** train six separate building-specific models.
- There were **19 building-cycles across four complete cycles**. Pooling was necessary because each building alone had far too few completed outcomes.
- Validation held out an entire cycle at a time, so checkpoints from a cycle under evaluation never appeared in its training data.

### Were Day 7, 14, 21, and 28 four simultaneous X variables?

- **No.** A Day 14 forecast did not require Day 7, Day 21, and Day 28 weights to be present in one row.
- Each checkpoint was a separate as-of prediction opportunity using only information known by that checkpoint.
- This avoided using a future Day 21 or Day 28 weight to make a supposed Day 14 prediction.
- Standard checkpoints made cycles with weekly weighing comparable with cycles that had daily early weights.
- Daily measurements between checkpoints remain useful for the live latest-weight calculation, target-gap monitoring, and recent-growth features in candidate models.
- They do not create useful supervised training examples unless that building-cycle also has a reliable observed Day 35 outcome.

### Direct inputs

1. Latest measured building weight.
2. Measurement day, used to select the historical remaining gain to Day 35.

### What was trained and compared?

- Canary trained or fitted **seven pooled candidate methods** using the historical building-checkpoint rows.
- The candidates included simple baselines, a recent-ADG projection, and a regularized Ridge machine-learning regression.
- The Ridge candidate used measurement day, current weight, current-to-target ratio, recent ADG, and an indicator for whether recent ADG was available.
- Candidate selection used unseen-cycle error, not performance on the same rows used to fit the method.

### Models compared

| Candidate | Overall MAE |
|---|---:|
| Historical Day 35 mean | 211 g |
| Target-curve ratio | 317 g |
| Recent straight-line ADG | 460 g |
| **Historical remaining gain** | **198 g** |
| Ridge regression | 202 g |
| Random Forest | 253 g |
| Gradient-boosted trees | 293 g |

### Why historical remaining gain won

- Best overall row-level MAE.
- Within 5% of the best cycle-balanced candidate.
- More transparent than Ridge.
- Much more accurate than projecting recent straight-line ADG.
- Responds to each building’s latest measured weight instead of giving every building the same result.

### Is the deployed weight forecast machine learning?

- **Not in the strict sense.** A Ridge machine-learning model was trained and evaluated, but it did not win.
- The deployed champion is an age-aware historical-growth formula because it had slightly lower validation error and was easier to explain.
- For a live building, Canary uses:

> Projected Day 35 weight = latest measured weight + historical average remaining gain from that measurement age to Day 35

- For each eligible historical building, Canary calculates `observed Day 35 weight − weight at the checkpoint`, then averages those remaining gains across the training buildings.
- With all eligible historical labels fitted for deployment, the mean allowances are approximately **1,420 g from Day 7**, **1,264 g from Day 14**, **983 g from Day 21**, and **637 g from Day 28**.
- Example: if the latest weight is 1,100 g on Day 21, the projection is `1,100 + 983 = 2,083 g`.
- During validation, the gain allowance is recalculated without the held-out cycle. The 983 g value is the final deployment fit, not a value leaked into its own test cycle.
- For a measurement between checkpoints, Canary interpolates the remaining-growth allowance between the surrounding checkpoint ages.
- Before Day 7, it uses an explicitly labelled target-curve fallback with wider uncertainty.
- If no weight has been measured, Canary says the projection is unavailable.

### Standard checkpoint versus farm target curve

- The farm target curve answers: **"What should the bird weigh at this age?"**
- The weight-gap risk check answers: **"How far is the latest measured weight from that age-specific target?"**
- The forecast answers: **"Given the latest measured weight and historical remaining growth, where might this building reach by Day 35?"**
- These are related but separate calculations. Being below target creates a present-day warning; the growth projection estimates the future Day 35 outcome.

### Validation results

- Complete cycles: **4**.
- Day 35 building outcomes: **19**.
- Overall MAE: **198 g**.
- Overall RMSE: **273 g**.
- Within 200 g: **61.8%**.
- Day 14 MAE: **183 g**.
- Day 14 within 200 g: **57.9%**.

### Important limitation

- All 19 historical Day 35 outcomes were below 2,000 g.
- Error in grams can be evaluated.
- Ability to distinguish target hits from misses cannot yet be evaluated.
- No weight projection is shown when a building has no measured weight.

### Feature importance

- The selected method is a transparent two-input formula, not a multi-feature fitted model.
- Its drivers are latest measured weight and measurement day.
- Do not invent five feature importances for this method.
- Target progress, recent ADG, and Ridge features were tested in competing methods but were not used by the selected winner.

### Best defense wording

> "We did not assume machine learning must win. We compared seven approaches, including Ridge, Random Forest, and gradient-boosted trees, using complete-cycle-held-out validation. The simpler historical remaining-gain method produced the best overall validated error and stayed within 5% of the best cycle-balanced result, so Canary deploys it. The method pools eligible buildings because the sample is too small for reliable building-specific models, while every forecast still starts from that building's own latest measured weight."

## 7. Component 3 — Recommendation playbook

### What it answers

- What should management inspect next?
- How urgently should the building be reviewed?
- When should management escalate?

### How it works

- Deterministic rule lookup, not generative AI.
- Inputs: current operating alert when supported; otherwise identified performance pattern + risk severity.
- Outputs: management focus, inspection checklist, urgency, and escalation condition.
- Every recommendation shows its rule ID, version, and approval status.
- Owner cards prefer a specific recorded alert over a generic peer-performance label.
- When no causal measurement is available, Canary says **cause not confirmed** and names the measurements to collect.

### Main problem patterns

| Pattern | Primary management focus |
|---|---|
| No material drift | Continue normal monitoring |
| Weight lag only | Confirm weight; check feed and water access |
| Survival concern only | Reconcile counts and investigate mortality/survival |
| Growth + survival drift | Complete a focused combined flock assessment |
| Localized building drift | Compare with better peers; inspect building-specific equipment and conditions |
| Farm-wide drift | Investigate shared feed, water, controller, weather, source, or management factors |
| Missing or stale evidence | Obtain current measurements before major decisions |

### Urgency

- Low: routine monitoring.
- Medium: within 24 hours.
- High: current shift.
- Critical: immediate inspection and appropriate escalation.

### Can we trust it?

**Strengths**

- Simple, deterministic, editable, and traceable.
- Based on industry-management references and farm-provided intervention mappings.
- Does not invent medication or diagnose disease.

**Limitations**

- Wording and thresholds still require Doc Raymond’s approval.
- Canary has not measured the causal effect of each intervention.
- Recommendations are inspection guidance, not guaranteed solutions.

## 8. Temperature, humidity, feed, water, and THI

### What Canary can responsibly do now

- Compare recorded temperature and humidity with provisional age-specific ranges.
- Flag daily mortality against provisional limits.
- Compare recorded daily feed per bird with provisional age-specific targets, pending unit confirmation.
- State the current value, target, acceptable range, and size of the gap.
- Recommend practical checks such as sensors, fans, inlets, curtains, cooling pads, heaters, litter, feed access, and water availability.
- Direct management toward the target range without prescribing an unapproved equipment setting.

### Example owner explanation

> “Temperature is above the approved age-based range. Verify the reading at bird height, then check fans, inlets or curtains, cooling pads, airflow, and water availability. Bring conditions toward the approved range. Canary does not claim heat caused the performance result.”

### What Canary cannot claim yet

- “Heat caused the mortality.”
- “Reducing temperature by 1°C will improve recovery by X points.”
- Water-intake risk when water data is absent.
- THI risk until one formula and age-specific bands are approved.
- A direct command such as “lower the controller by 3°C” unless the sensor, housing response, bird behavior, and farm SOP have been verified.

### Why environment is not a fifth risk dimension yet

- It may double-count the result of the same problem.
- Environmental coverage is incomplete.
- Water is unavailable and the daily feed-per-bird unit needs confirmation.
- Farm-approved thresholds are incomplete.
- Current model importance does not establish causation.

**Current architecture verdict:** keep environment out of the 0–12 outcome-warning score for this capstone, but use supported operating alerts as the owner-facing reason and action. This is more actionable without double-counting the same outcome or pretending that an association is a proven cause.

## 9. Business-value estimator

### What it estimates

- Birds represented by a recovery-rate improvement.
- Estimated gross revenue represented by those birds.

### Formula

- Birds represented = beginning population × recovery improvement.
- Gross revenue per bird = assumed sale weight × assumed price per kg.
- Estimated gross revenue = birds represented × gross revenue per bird.

### Required defense wording

- This is a scenario estimate, not proven incremental profit.
- It excludes intervention cost, feed, labor, electricity, treatment, mortality timing, and price changes.
- It does not prove Canary caused the improvement.

## 10. Actual versus predicted

### Current/latest cycle

- Shows current risk rating, predicted recovery, projected Day 35 weight when weight is available, and recommended next check.

### Previous cycles

- Shown as completed under the capstone convention.
- Completion date = each building’s last recorded daily date.
- Actual recovery = ending recorded population ÷ beginning population.
- Actual final average weight = matched Farm Performance Summary value when available.
- Historical Day 14 backtest compares what Canary would have predicted using only Day 14 data with what was later observed.

### Why backtesting matters

- Demonstrates performance on unseen cycles.
- Makes errors visible instead of showing only successful examples.
- Supports discussion of when the model performs well or poorly.

## 11. Expected panel questions

### “Why are there two targets?”

- Weight measures growth performance.
- Recovery measures birds remaining.
- Both affect farm output and require different forecasts.

### “Why is Day 14 important?”

- It is the agreed early-warning checkpoint while management still has time to investigate.
- Canary continues updating after Day 14.
- Association between Day 14 and later results is exploratory, not causal proof.

### “Why Day 35?”

- The farm’s key management milestone is at least 2,000 g on Day 35.
- Day 35 is not automatically the harvest date.

### “Why not scale final weight by 35/49?”

- Broiler growth is not linear.
- That conversion would create an artificial label.
- Canary uses actual recorded Day 35 weights instead.

### “Why split validation by cycle?”

- Daily rows from one flock are highly related.
- Randomly splitting daily rows would leak flock information and exaggerate accuracy.
- Holding out complete cycles better represents future use.

### “Why choose a simple model?”

- The dataset is small.
- Complex models can overfit.
- Canary selects the simplest candidate within 5% of the best cycle-balanced error.

### “Is feature importance causal?”

- No. It shows what the fitted model relied on.
- A farm intervention still requires operational verification and expert judgment.

### “Can Canary diagnose disease or prescribe treatment?”

- No. It supports inspection, escalation, and management decisions.
- Veterinary diagnosis and treatment remain outside scope.

### “What is the biggest model limitation?”

- Recovery uses only 25 distinct building outcomes and a last-recorded recovery proxy.
- Weight uses 19 Day 35 outcomes, all below 2,000 g.
- More standardized completed cycles are needed.

### “Is Canary production-ready?”

- It is capstone-ready as a local prototype with transparent limitations.
- It is not yet a validated production control system.

## 12. Three-minute demo path

1. **Home:** explain the two targets and show all six buildings.
2. **Review first:** identify the highest-priority current building.
3. **Building View:** show total risk score and the scored reasons.
4. **Most actionable signal:** show the current temperature, humidity, feed, or mortality gap and the target-based next check.
5. **Forecasts:** compare current recorded state with predicted recovery and projected Day 35 weight.
6. **Drivers:** show the five strongest recovery-model contributions and the two direct weight drivers.
7. **Recommendation:** show urgency, inspection focus, escalation trigger, and rule ID.
8. **Completed cycle:** show a Day 14 prediction-versus-actual backtest.
9. **Methodology:** show candidate-model comparison, held-out performance, and limitations.

## 13. What is capstone-ready versus future work?

### Capstone-ready

- Standardized building-day dataset.
- Six-building decision dashboard.
- Explainable 0–12 risk score.
- Separate recovery and Day 35 weight outlooks.
- Cycle-held-out model evaluation.
- Global and building-specific model-driver explanations.
- Deterministic recommendation playbook.
- Historical Day 14 backtests.
- Business-value scenario estimator.

### Future improvements

- Verified harvest-event flag.
- More Day 14 and Day 35 weights across cycles.
- More examples that achieve 2,000 g and 95%.
- Reliable daily water intake.
- Farm-approved environmental and THI thresholds.
- Prospective validation on new cycles.
- Measurement of intervention effectiveness and net profit.
- Live integrations, authentication, and production deployment.

## 14. Final defense posture

**Strongest claim**

> Canary turns daily farm records into a consistent, explainable workflow for prioritizing buildings, estimating two important outcomes, and deciding what management should inspect next.

**Claims to avoid**

- “Canary proves what caused the problem.”
- “Canary guarantees the flock will hit or miss the target.”
- “The risk score is the probability of missing the target.”
- “The recommendations have proven causal financial impact.”

**Best closing line**

> The capstone demonstrates a defensible decision-support prototype: transparent where the rules are expert-defined, quantitative where historical validation is possible, and explicit where more farm data or expert approval is still needed.
