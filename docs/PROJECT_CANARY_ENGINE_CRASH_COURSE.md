# Project Canary — Three-Engine Crash Course

## Start here

Project Canary’s business purpose is more consistent production outcomes. It creates earlier visibility of off-track buildings so management can investigate and act before weak daily signals become final outcomes.

It answers one practical question:

**How can the farm make production outcomes more consistent? Which buildings are going off-track, what recovery and Day 35 weight should we expect, and what should management check first?**

Canary supports management decisions; it does not diagnose disease, prescribe treatment, or guarantee recovery or growth outcomes.

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
5. Verify the newest workbook contains 1,624 unique building-day records with no duplicate building-day keys. Unsupported forward-filled Days 36–49 were removed from 2026-3.
6. Build 31 historical recovery outcomes and 31 observed Day 35 weight outcomes; keep the three later 2026-3 outcomes for prospective audit only.
7. Create 151 balanced recovery snapshots and 124 historical weight checkpoint rows.
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

`additional loss after review date = current percentage alive − last-recorded recovery proxy`

Live output: `predicted final recovery = current percentage alive − predicted additional loss`.

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

The six recovery holdout folds are **2025-2, 2025-3, 2025-4, 2025-5, 2026-1, and 2026-2**. For each fold, Canary removes every building from that cycle before fitting or preprocessing. The latest 2026-3 cycle is excluded from model selection and scored afterward as a prospective audit. Risk alerts are deterministic rules, so they are replayed at historical dates rather than trained with a holdout.

### Day 14 retrospective example

For each eligible completed building-cycle, Canary freezes the inputs at Day 14, excludes that entire cycle from training, estimates the loss still expected after Day 14, and compares the resulting projection with the last-recorded recovery proxy.

Example — **2025-2 Tags 1**:

- Day 14 projected recovery: **92.75%**.
- Last-recorded recovery proxy: **94.28%**.
- Error: **−1.56 percentage points**; the projection was lower than the recorded result.

Across the 31 eligible outcomes, Day 14 MAE is **1.95 points**, RMSE is **2.71 points**, bias is **+0.49 points**, R² is **0.025**, and the empirical 80% range is approximately **±2.70 points**. The 95% target-side accuracy is **87.1%**, equal to the always-below majority baseline, with **0% recall** for actual target hitters. Use it as a directional continuous estimate—not a reliable hit/miss classifier.

### Live dates between checkpoints

The selected ordinary-linear model can score any live review date using only the evidence available on that date. The Day 7/14/21/28 remaining-loss interpolation remains the transparent operational fallback if the fitted model is unavailable. Extra Trees is retained only as a nonlinear sensitivity and held-out SHAP challenger.

### Five-model table

| Method | MAE | Cycle MAE | RMSE | R² | Role |
|---|---:|---:|---:|---:|---|
| Age-band remaining-loss baseline | 2.00 pts | 2.09 pts | 2.66 pts | -0.010 | Transparent baseline |
| Linear remaining-loss | 1.74 pts | 1.76 pts | 2.57 pts | 0.054 | Selected continuous estimate |
| Ridge remaining-loss | 1.74 pts | 1.76 pts | 2.57 pts | 0.055 | Compared |
| Gradient Boosting remaining-loss | 1.76 pts | 1.84 pts | 2.47 pts | 0.129 | Compared |
| Extra Trees remaining-loss | 1.73 pts | 1.80 pts | 2.47 pts | 0.130 | Nonlinear/SHAP challenger |

### Verdict

Ordinary linear regression improves cycle-balanced MAE by **16.1%** versus the refreshed age-band baseline, has positive held-out R², and is effectively tied with Ridge while remaining easier to explain. Its at/above-95% recall is only **21.1%**, so it is not presented as a target-hit classifier or probability of success. Extra Trees remains a nonlinear sensitivity challenger for held-out SHAP, not the live champion.

The later 2026-3 recovery audit is materially weaker: **4.08 percentage-point MAE**, with all predictions biased low. The weight method's later audit is more encouraging at **63 g MAE**, but it contains only three independent buildings. Neither small audit overrides the whole-cycle champion selection; both are shown as evidence of real deployment uncertainty.

### SHAP interpretation

- Global SHAP is calculated on complete held-out cycles, not the training fit.
- Mean absolute SHAP ranks how strongly each feature moved recovery estimates.
- Signed SHAP for one building shows which recorded inputs moved its estimate up or down.
- SHAP describes association and model behavior, not a causal treatment effect.
- Management actions still require a recorded rule trigger and an approved playbook response.

The business-value output is downstream of this forecast, not a separate model:

`gross revenue at risk = beginning birds × gap to 95% × assumed sale weight × price/kg`

An optimistic recovery projection therefore makes revenue at risk smaller; a more conservative projection makes it larger.

## Engine 2B — Day 35 weight

### Y target

`remaining gain = observed Day 35 weight − current checkpoint weight`

Live output: `projected Day 35 weight = current weight + predicted remaining gain`.

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
| Checkpoint linear | 216 g | 208 g | 273 g | -0.118 | 56.5% | Compared |
| Ridge remaining-gain | 207 g | 200 g | 264 g | -0.045 | 56.5% | Compared |
| Robust Huber remaining-gain | 226 g | 223 g | 276 g | -0.144 | 51.6% | Compared |
| Gradient Boosting remaining-gain | 203 g | 207 g | 262 g | -0.026 | 52.4% | Best nonlinear challenger |

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
