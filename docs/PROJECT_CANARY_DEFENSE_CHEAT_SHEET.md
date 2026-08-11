# Project Canary — Capstone Defense Cheat Sheet

> Updated 11 August 2026. Use this as the shared answer key for the dashboard, notebooks, model files, and presentation.

## One-page “memorize this” summary

### What is Project Canary?

Project Canary is an early-warning and decision-support system for a broiler farm. It identifies buildings that need attention, explains the recorded warning signs, estimates harvest recovery and Day 35 average weight, and links confirmed operating patterns to Doc Raymond’s action playbook.

### What business question does it answer?

**Which buildings need attention today, why, what harvest recovery and Day 35 weight should we expect, and what should management check next?**

### What are the three engines?

1. **Risk scoring:** transparent rules score four observed dimensions from 0–3 points each. Total 0–12 becomes Low, Medium, High, or Critical.
2. **Predictive outlooks:** a recovery estimator and a Day 35 weight method are evaluated independently using data available by the selected review date.
3. **Recommendation playbook:** deterministic triggers map to Doc Raymond’s recorded checks, urgency, and escalation. It does not diagnose disease or automatically treat birds.

### What data do we have?

- 1,785 source rows became **1,666 unique building-day records** after repeated Zone A/B environment rows were consolidated.
- Recovery: **25 independent building-cycle outcomes**, represented by **122 balanced decision snapshots**; 1,122 leakage-safe daily snapshots are retained for audit.
- Day 35 weight: **31 independent building-cycle outcomes**, represented by **124 Day 7/14/21/28 checkpoint rows**.
- Recovery Y is the agreed proxy: last-recorded population ÷ beginning population.
- Weight Y is observed average bodyweight on production Day 35.
- Day 35 goal: **1,800 g**. Recovery goal: **95%**.

### What models are operational?

- **Recovery:** ordinary linear regression is the continuous estimator. Whole-cycle MAE is **1.37 percentage points**; cycle-balanced MAE is **1.48 points**; RMSE is **1.84 points**; R² is **0.189**. It improves cycle-balanced MAE by **14.5%** versus the historical-mean baseline. It is not validated as a 95% hit/miss classifier.
- **Day 35 weight:** **historical remaining gain** is the operational fallback. MAE is **178 g**; cycle-balanced MAE is **182 g**; RMSE is **242 g**; R² is **0.126**; **65.3%** are within 200 g. Ridge is the best learned linear challenger, but it is worse than the baseline and fails the replacement gates.

### What is the honest claim?

Canary is a **validated prototype**, not a production guarantee. Recovery is useful directionally as a continuous estimate. Day 35 weight uses the more defensible transparent fallback because the learned models do not yet improve enough on unseen cycles.

## Five-step explanation: Engine 1 — Risk scoring

1. **Question:** Which building shows the most operational concern today?
2. **Inputs:** weight gap versus the approved age target; survival/population-loss evidence; recent mortality; and environmental evidence against age-specific temperature/humidity bands. Missing or stale evidence is disclosed.
3. **Process:** each dimension receives 0–3 points using configurable thresholds. Available points are added; the trace shows each observation, rule, score, and missing dimension.
4. **Output:** total score, Low/Medium/High/Critical label, leading problem pattern, and “why now” evidence.
5. **Limitation:** the score is an operational-priority rule, not a probability of missing 95% recovery or 1,800 g. Environmental thresholds remain subject to farm approval.

### Why rules rather than a risk classifier?

- Only 25–31 independent outcomes are available.
- The owner needs to see the exact rule and measurement behind every flag.
- Rules keep operating concern separate from statistical forecasts.
- Thresholds can be reviewed and changed without retraining a model.

## Five-step explanation: Engine 2A — Recovery forecast

1. **Question:** Using only what is known today, what final recovery proxy should we expect?
2. **Inputs (X):** cycle day; current percentage alive; recent mortality and mortality change; latest weight gap and freshness; temperature/humidity band deviations; recent out-of-band environment days; and environment freshness. Feed is withheld until its unit is confirmed.
3. **Process:** create Day 7/14/21/28/latest snapshots; give every building-cycle equal total weight; impute, scale, tune, and compare models inside nested whole-cycle folds.
4. **Output:** point estimate, target gap, empirical range, model version, forecast date, and predictive drivers.
5. **Limitation:** Y is a last-recorded recovery proxy, only five historical cycles are available, and the model cannot yet distinguish 95% hits from misses better than the majority rule.

### Recovery workflow — explain it from start to finish

| Stage | What Canary does |
|---|---|
| **Data** | Starts from the corrected farm workbook, consolidates Zone A/B rows, and produces one building-day record. |
| **Y output** | Calculates the completed-cycle recovery proxy: last-recorded population ÷ beginning population. |
| **X inputs** | Uses only evidence known on the review date: age, current survival, mortality signals, latest weight gap/freshness, and environmental deviations/freshness. |
| **Feature engineering** | Converts raw readings into per-bird rates, gaps from approved targets, rolling mortality signals, days outside environmental bands, and staleness flags. |
| **Pre-processing** | Inside each training fold only: median-impute missing numeric values, add missingness indicators, and standardize inputs for linear models. |
| **Validation** | Removes one complete harvest cycle at a time; the inner loop selects settings using only the remaining cycles. Repeated snapshots are weighted so each building-cycle has equal total influence. |
| **Selection** | Compares five declared methods on cycle-balanced MAE, then checks R², cycle stability, target-side performance, and champion gates. |
| **Live output** | Returns an estimated final recovery, likely range, gap to 95%, model version, and non-causal driver explanation. |

### Recovery Y target

`final recovery proxy = last-recorded population ÷ beginning population`

The value is used only after prediction for evaluation. Current survival may be an X because it is already known on the review date and logically constrains what final recovery can be. Future ending population is never an X.

### Recovery model comparison — primary prospective test

| Candidate | MAE (pts) | Cycle-balanced MAE (pts) | RMSE (pts) | R² | Decision |
|---|---:|---:|---:|---:|---|
| Historical mean | 1.66 | 1.73 | 2.17 | -0.132 | Baseline |
| **Ordinary linear regression** | **1.37** | **1.48** | **1.84** | **0.189** | **Operational continuous estimator** |
| Compact Ridge | 1.55 | 1.59 | 1.97 | 0.070 | Compared |
| Gradient Boosting | 1.57 | 1.66 | 2.08 | -0.041 | Compared |
| XGBoost | — | — | — | — | Declared challenger; unavailable locally because macOS lacks `libomp` |

### Why ordinary linear regression instead of Ridge?

We did test both. Under the redesigned nested whole-cycle test, ordinary linear regression improved the historical-mean baseline by **14.5%** and kept positive R². Ridge was more regularized, but improved the baseline by only **8.1%**, below the approved 10% gate. The evidence therefore favors OLS for the continuous estimate in this release.

### Recovery classification warning

- Target-side accuracy: **82.8%**.
- Majority baseline: **84.4%** overall.
- Day 14: **80.0%** versus an **84.0%** majority baseline.
- At/above-95% recall: **0%**.

Therefore, do not say “the model accurately classifies whether recovery will hit 95%.” Say: **it estimates the continuous recovery level, but target-side classification is not validated.**

### Recovery top held-out drivers

The largest out-of-fold permutation importances are humidity-band deviation, current survival, flock age, recent mortality, and mortality trend. These are associations that helped prediction on unseen cycles; they do not prove causal treatment effects.

## Five-step explanation: Engine 2B — Day 35 weight

1. **Question:** Given recorded growth so far, what average building weight should we expect on Day 35 against the 1,800 g goal?
2. **Inputs (X):** latest/checkpoint weights; weighing day; weight-to-target ratio; recent and cumulative average daily gain; earlier checkpoint weights available by that date; current survival; environmental-band exposure and freshness.
3. **Process:** use only observed Day 7/14/21/28 weights; hide future checkpoints; give each building-cycle equal total influence; run nested whole-cycle validation and apply strict replacement gates.
4. **Output:** projected Day 35 weight, gram and percent gap to 1,800 g, range, method, measurement day, and staleness.
5. **Limitation:** only 31 independent Day 35 outcomes exist; 26 are below target and five are at/above target; no learned model clears every gate.

### Day 35 weight workflow — explain it from start to finish

| Stage | What Canary does |
|---|---|
| **Data** | Uses corrected, observed building weights and keeps observed values separate from target-curve interpolation. |
| **Y output** | Uses the observed building average bodyweight on production Day 35. |
| **X inputs** | At each Day 7/14/21/28 checkpoint, exposes only weights and operating evidence already recorded by that checkpoint. Later weights remain blank. |
| **Feature engineering** | Creates measurement age, target ratio/deficit, average daily gain, available prior checkpoint weights, survival, environmental exposure, missingness, and staleness. |
| **Pre-processing** | Performs imputation, missingness handling, scaling, feature filtering, and tuning inside training folds—not before the holdout. |
| **Validation** | Removes a complete cycle, trains on the other cycles, predicts the unseen cycle, and repeats. Every building-cycle has equal total influence. |
| **Selection** | Compares the transparent remaining-gain baseline with OLS, Ridge, Gradient Boosting, and declared XGBoost. A learned model must improve cycle-balanced MAE by 10% and reach 70% within 200 g. |
| **Live output** | Because no learned model passes, adds the training-only historical remaining gain for the measurement age to the latest observed building weight. |

### Day 35 weight Y target

`Y = observed average building bodyweight on production Day 35`

The approved target curve is a known reference used to calculate age-specific progress. It never replaces a missing observed Y.

### Weight model comparison — primary prospective test

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² | Within 200 g | Decision |
|---|---:|---:|---:|---:|---:|---|
| **Historical remaining gain** | **178 g** | **182 g** | **242 g** | **0.126** | **65.3%** | **Operational fallback** |
| Ordinary linear regression | 217 g | 209 g | 273 g | -0.114 | 55.6% | Compared |
| Ridge regression | 197 g | 193 g | 252 g | 0.048 | 60.5% | Best learned linear challenger |
| Gradient Boosting | 189 g | 189 g | 256 g | 0.018 | 58.1% | Compared |
| XGBoost | — | — | — | — | — | Declared challenger; unavailable locally because macOS lacks `libomp` |

### How historical remaining gain works

For each checkpoint age in the training cycles:

`remaining gain = observed Day 35 weight − observed checkpoint weight`

Average those gains using training cycles only, then:

`projected Day 35 weight = current observed weight + average historical remaining gain for that age`

The held-out cycle is excluded when validating, so its Day 35 result cannot influence its own prediction.

### Why not use Ridge as the live model?

Ridge is explainable and has positive R², but it is **5.7% worse** than historical remaining gain on cycle-balanced MAE and reaches only **60.5% within 200 g**. The approved gate requires at least 10% improvement and at least 70% within 200 g. Using the baseline is therefore the honest, defensible choice.

### What drives the live weight projection?

The live formula uses three direct items: the latest measured weight, its measurement day, and the average historical remaining gain for that checkpoint age. Ridge coefficients may be shown as research evidence about the best learned challenger, but they do **not** drive the live forecast.

### Day 14 weight proof

- 31 building outcomes tested.
- Day 14 MAE: **181 g**.
- Day 14 RMSE: **237 g**.
- Within 200 g: **58.1%**.
- At/above-1,800 g recall: **0%**.

Day 14 remains the main early-warning checkpoint because it is early enough to act and its observed weight is used in both target-gap monitoring and the later-growth forecast. It is a useful signal, not proof of causality.

## Five-step explanation: Engine 3 — Recommendations

1. **Question:** What should management inspect next, given the confirmed warning pattern?
2. **Inputs:** deterministic risk/alert pattern, severity, recorded evidence, measurement freshness, and approval status.
3. **Process:** match the trigger to Doc Raymond’s playbook—for example low bodyweight, high mortality, high/low temperature, high/low humidity, abnormal fluctuation, low feed intake, or poor recovery outlook.
4. **Output:** plain-language check, urgency, escalation condition, and applied rule.
5. **Limitation:** recommendations are inspection/management guidance. They do not diagnose disease, estimate causal treatment effects, or prescribe automatic treatment.

## Why nested whole-cycle validation?

- **Outer loop:** remove every row from one harvest cycle; train and tune without it; predict it; repeat for all cycles.
- **Inner loop:** choose hyperparameters using only the remaining training cycles.
- Imputation, scaling, and tuning never see the outer test cycle.
- Repeated snapshots are weighted so one building-cycle has the same total influence as another.

This answers the deployment-like question: **can Canary generalize to a future unseen harvest cycle?**

The colleague’s building-cycle LOGO test answers an easier question: can it predict one building when other buildings from the same cycle remain in training? Its higher R² is not directly comparable.

## What we learned from the teammate model

### Incorporated

- A structured and reproducible cleaning pipeline.
- Zone A/B aggregation before feature creation.
- Environmental, growth, missingness, and freshness features.
- Group-aware cross-validation and comparison with boosted trees.
- Explicit model comparison and feature-importance reporting.

### Not adopted as the primary Canary test

- Treating repeated daily rows as new independent flock outcomes.
- Leaving out one building while keeping other buildings from the same cycle in training.
- Using interpolated early weights that were created with later observations.
- Selecting features on the full dataset before cross-validation.
- Using a pickle or headline score that cannot be reproduced from the corrected canonical workbook.

**Defense answer:** We took the useful engineering ideas, but retained stricter whole-cycle validation because Canary must generalize to a future cycle, not merely another building in a cycle it has already seen.

## Why no SMOTE or oversampling?

- These are regression problems, while standard SMOTE is primarily for classification.
- The scarce evidence is independent building-cycles, not spreadsheet rows.
- Synthetic rows do not create new cycles and can make uncertainty look falsely small.
- Synthetic poultry trajectories may be biologically implausible.

## How to interpret MAE, RMSE, R², and bias

- **MAE:** typical absolute miss in business units; primary selection metric.
- **RMSE:** penalizes large misses more strongly.
- **R²:** fraction of variance explained on held-out data; useful diagnostic, not the sole decision rule.
- **Bias:** whether predictions are high or low on average.
- **Target-side accuracy/recall:** whether a model distinguishes the two sides of 95% or 1,800 g; always compare with the majority baseline.

Low R² means much variation remains unexplained. It does not erase all usefulness, but it requires prototype positioning, wider uncertainty, and conservative claims.

## Seven EDA questions Canary answers

1. **Coverage:** Which cycles, buildings, days, and measures are complete enough to analyze?
2. **Early growth:** Is Day 14 weight associated with Day 35 weight?
3. **Early recovery signal:** Is Day 14 weight associated with the final recovery proxy?
4. **Environment:** Are recorded temperature and humidity within the provisional age-specific bands, and is the evidence balanced enough to compare outcomes?
5. **Survival paths:** When do buildings begin to drift below the expected survival path?
6. **Model accuracy:** How large are forecast errors overall, by checkpoint, and by held-out cycle?
7. **Target attainment:** How often have completed building-cycles reached 1,800 g, 95% recovery, or both?

**Defense answer:** EDA is used to test the business story, expose data limitations, inform transparent features, and prevent overclaiming. It does not prove that changing one factor will cause a better outcome.

## Common panel questions

### “Did you train on daily rows as if they were independent flocks?”

No. The independent outcome is one building-cycle. Repeated snapshots give multiple historical decision points, but they receive equal building-cycle weighting and are always held out together by cycle.

### “Did you leak future weights?”

No. At a Day 14 snapshot, Day 21/28/35 inputs are blank. The Day 35 value is used only as Y after prediction. Target-curve interpolation is a known reference, not an observed weight.

### “Why is current survival allowed in recovery X?”

It is known today and directly limits what final recovery can be under the agreed formula. Future ending population is never used as X. Raw beginning population is excluded unless the farm later approves a meaningful density feature.

### “Can feature importance tell the owner to lower temperature by 3°C?”

No. Importance shows association and predictive reliance. The action table must be triggered by an observed threshold and approved farm guidance, not by a causal claim from observational data.

### “Are these models reliable?”

They are useful as validated prototype estimates, not guarantees. Recovery passes its continuous-estimate gate but not classification. Weight learned models fail the replacement gates, so Canary uses the transparent baseline.

### “What would most improve the models?”

Collect more completed cycles with verified harvest dates/populations, consistent bodyweight sampling, confirmed feed units, complete environment readings, and more outcomes on both sides of the targets.

## Where to show proof in the deliverables

- **Canary Methodology page:** five-model tables, gates, timing and cycle performance, drivers, and limitations.
- **Building View:** current evidence, forecast trace, and building-specific linear contribution view where available.
- **EDA & Insights:** data coverage, early-weight relationships, target attainment, and limitations.
- `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx`: outcomes, exact training rows, data dictionary, and daily audit.
- `notebooks/Project_Canary_Harvest_Recovery_Model.ipynb`
- `notebooks/Project_Canary_Day35_Weight_Model.ipynb`
- `models/harvest_recovery_champion.pkl`
- `models/day35_weight_champion.pkl`

## Outstanding farm validation

1. Approve environmental thresholds and interventions.
2. Confirm feed-intake units and targets.
3. Define a consistent bodyweight sampling procedure.
4. Confirm the eventual verified harvest-date and ending-population source.
5. Define acceptable forecast-error limits for management use.
