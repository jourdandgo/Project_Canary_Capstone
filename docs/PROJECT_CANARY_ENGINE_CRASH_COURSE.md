# Project Canary — Three-Engine Crash Course

## Start here

Project Canary answers one practical question:

**Which buildings need attention today, why, what recovery and Day 35 weight should we expect, and what should management check next?**

It answers through three separate engines. Keeping them separate makes the product explainable.

| Engine | Inputs | Process | Output |
|---|---|---|---|
| Risk scoring | Current recorded operating evidence | Transparent 0–3 point rules | 0–12 score, Low/Medium/High/Critical, and why |
| Predictive outlooks | Historical leakage-safe building-cycle snapshots | Nested whole-cycle comparison of five declared methods | Recovery estimate and Day 35 weight projection |
| Recommendations | Confirmed pattern, severity, and rule approval | Deterministic Doc Raymond playbook mapping | Check, urgency, escalation, and applied rule |

## Data pipeline in one minute

1. Read the corrected farm workbook.
2. Normalize cycle, building, date, population, mortality, feed, weight, temperature, and humidity.
3. Aggregate Zone A/B environment rows before feature calculation.
4. Preserve missing values and distinguish observed from target/interpolated weights.
5. Produce 1,666 unique building-day records from 1,785 source rows.
6. Build 25 recovery outcomes and 31 observed Day 35 weight outcomes.
7. Create 122 balanced recovery snapshots and 124 weight checkpoint rows.
8. Hold out complete cycles during testing.

Open `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx` to inspect every outcome, X row, Y value, definition, and leakage guard.

## Engine 1 — Risk scoring

### Question

Which building deserves management attention first?

### Inputs

- Weight gap versus the approved age target.
- Survival/population-loss evidence.
- Recent mortality evidence.
- Environment versus the age-specific temperature/humidity bands.
- Freshness and missingness of those measurements.

### Process

Each available dimension receives 0–3 points. The total is 0–12 and maps to Low, Medium, High, or Critical. The building trace displays the measurement, threshold, points, missing checks, and final sum.

### Output

Risk score and label, leading problem pattern, “why now,” and a link to the relevant action rule.

### Limitation

It is an operational concern score—not a probability of missing the two goals. Thresholds remain editable and require farm approval.

## Engine 2A — Recovery forecast

### Y target

`last-recorded population ÷ beginning population`

This is the agreed capstone proxy, not a verified harvest-event outcome.

### X inputs

- Flock age.
- Current percentage alive.
- Recent mortality and mortality change.
- Latest weight gap and days since weighing.
- Temperature/humidity band deviations.
- Recent recorded environment days outside the bands.
- Days since the latest environment reading.

Feed is withheld until its unit is confirmed.

### Validation

- Outer loop: leave one complete cycle unseen.
- Inner loop: tune using only the remaining cycles.
- Imputation and scaling occur inside training folds.
- Repeated snapshots receive equal building-cycle weighting.
- MAE is primary; RMSE, R², bias, confidence intervals, and target-side metrics are secondary.

### Five-model table

| Method | MAE | Cycle MAE | RMSE | R² | Role |
|---|---:|---:|---:|---:|---|
| Historical mean | 1.66 pts | 1.73 pts | 2.17 pts | -0.132 | Baseline |
| Ordinary linear regression | 1.37 pts | 1.48 pts | 1.84 pts | 0.189 | Operational continuous estimator |
| Ridge | 1.55 pts | 1.59 pts | 1.97 pts | 0.070 | Compared |
| Gradient Boosting | 1.57 pts | 1.66 pts | 2.08 pts | -0.041 | Compared |
| XGBoost | — | — | — | — | Declared; unavailable locally without `libomp` |

### Verdict

OLS improves cycle-balanced MAE by 14.5% versus the baseline and has positive R², so the continuous-estimate gate passes. It does not beat the majority baseline for 95% hit/miss classification, so that classification is not validated.

## Engine 2B — Day 35 weight

### Y target

Observed building average bodyweight on production Day 35.

### X inputs tested

- Measurement day and latest/checkpoint weight.
- Current weight divided by the approved age target.
- Recent and cumulative average daily gain.
- Day 7/14/21/28 weights available by the checkpoint.
- Current survival and environment-band exposure/freshness.

Future checkpoints remain blank. The target curve is a reference, never a substitute Y.

### Five-model table

| Method | MAE | Cycle MAE | RMSE | R² | Within 200 g | Role |
|---|---:|---:|---:|---:|---:|---|
| Historical remaining gain | 178 g | 182 g | 242 g | 0.126 | 65.3% | Operational fallback |
| OLS | 217 g | 209 g | 273 g | -0.114 | 55.6% | Compared |
| Ridge | 197 g | 193 g | 252 g | 0.048 | 60.5% | Best learned linear challenger |
| Gradient Boosting | 189 g | 189 g | 256 g | 0.018 | 58.1% | Compared |
| XGBoost | — | — | — | — | — | Declared; unavailable locally without `libomp` |

### How the fallback works

For a checkpoint age, calculate `Day 35 weight − checkpoint weight` in training cycles, average that remaining gain, then add it to the current building’s observed weight. The held-out cycle never contributes to its own average.

### Verdict

No learned model beats the baseline by 10% and reaches 70% within 200 g. Canary therefore uses the transparent fallback. Day 14 MAE is 181 g; Day 14 remains an early-warning checkpoint, not a guarantee.

## Engine 3 — Recommendation playbook

### Inputs

Confirmed operational trigger, severity, evidence, freshness, and rule approval.

### Process

Map low bodyweight, high mortality, temperature/humidity problems, abnormal fluctuation, low feed intake, or poor recovery outlook to Doc Raymond’s approved inspection guidance.

### Output

What to inspect, response time, escalation condition, and applied rule.

### Limitation

The recommendation is not a diagnosis, causal model, or automatic treatment. Feature importance adds context but cannot justify statements such as “lower temperature 3°C and recovery will rise 1%.”

## Why the teammate’s R² can be higher

The teammate’s primary split leaves out one building-cycle while other buildings from the same harvest cycle may stay in training. Canary removes the entire cycle. The teammate’s test measures within-cycle generalization; Canary’s primary test measures prospective future-cycle generalization. The latter is harder and closer to deployment.

### What Canary adopted from the teammate pipeline

- Structured cleaning and Zone A/B consolidation.
- Environmental, growth, freshness, and missingness features.
- Group-aware validation and boosted-tree challengers.
- Side-by-side model metrics and driver reporting.

### What Canary did not adopt

- Daily rows counted as independent outcomes.
- A primary split that allows the test cycle to remain partly in training.
- Future-informed interpolation of early bodyweights.
- Feature selection before the validation split.
- Unreproducible scores from an older workbook or a pickle alone.

The practical rule is: **borrow useful feature and workflow ideas, but never weaken the leakage-safe future-cycle test.**

## What to say about low R²

“R² shows that much variation remains unexplained, so we position Canary as a prototype. We select on business-unit MAE, require positive R², report uncertainty and target-side metrics, and fall back to a transparent method whenever learned models do not improve enough.”

## Do not claim

- Production-ready accuracy.
- Causal feature effects.
- Reliable classification of 95% recovery hits.
- A learned Day 35 weight model that beats the baseline.
- Disease diagnosis or automatic treatment.

## Do claim

- Leakage-safe, nested whole-cycle evaluation.
- Transparent model comparisons and replacement gates.
- Honest operational fallback behavior.
- Full traceability from source rows to predictions and actions.
