# Boosted-model audit - 11 August 2026

## Decision after the final nested audit

Use **ordinary linear regression** for the continuous recovery estimate and **historical remaining gain** as the operational Day 35 weight fallback. Recovery passes the continuous-regression gate but fails the 95% target-classification gate. No learned weight candidate passes all champion gates. The colleague's XGBoost pickle is not adopted because its reported validation is not comparable to Canary's prospective whole-cycle protocol.

## Why the colleague's XGBoost result is not comparable

- It holds out one **building-cycle**, while other buildings from the same harvest cycle remain in training. Canary holds out the **entire cycle**.
- It uses a wide, interpolated feature set (79 predictors for roughly 34 independent building outcomes). Some interpolation and lag features can carry information from later measurements.
- It uses an earlier workbook; Canary includes the corrected 2026-1 to 2026-3 bodyweights.
- Its reported MAE is on a 0-1 recovery scale (0.0107 = about 1.07 percentage points), not automatically evidence of better future-cycle performance.

## Fair Canary comparison

All results use leave-one-complete-cycle-out validation, leakage-safe as-of inputs, and cycle-balanced MAE as the primary selection metric.

### Harvest recovery (percentage points)

| Candidate | MAE | Cycle-balanced MAE | RMSE | R2 | Decision |
|---|---:|---:|---:|---:|---|
| Historical mean | 1.66 | 1.73 | 2.17 | -0.132 | Baseline |
| Ordinary linear regression | **1.37** | **1.48** | **1.84** | **0.189** | **Selected continuous estimate** |
| Compact Ridge | 1.55 | 1.59 | 1.97 | 0.070 | Compared |
| Gradient Boosting | 1.57 | 1.66 | 2.08 | -0.041 | Compared |
| XGBoost | — | — | — | — | Unavailable locally |

### Day 35 bodyweight (grams)

| Candidate | MAE | Cycle-balanced MAE | RMSE | R2 | Decision |
|---|---:|---:|---:|---:|---|
| Historical remaining gain | **178** | **182** | **242** | **0.126** | **Operational fallback** |
| Ordinary linear regression | 217 | 209 | 273 | -0.114 | Compared |
| Ridge regression | 197 | 193 | 252 | 0.048 | Best learned challenger |
| Gradient Boosting | 189 | 189 | 256 | 0.018 | Compared |
| XGBoost | — | — | — | — | Unavailable locally |

## XGBoost status

XGBoost is declared as an optional challenger in the code, but this Mac cannot load it because the required OpenMP runtime (`libomp`) is absent and Homebrew is not installed. It is explicitly shown as unavailable in the model manifests; no copied metric is presented as a fair Canary result.

## Defense-ready interpretation

The R2 values are modest. That is expected with only five historical cycles and 25 recovery / 31 Day 35 outcome labels. The models are decision-support estimates, not guarantees. Nested whole-cycle validation deliberately produces a harder, more prospective test than random or building-level splits. Improvement requires more complete cycles, standardized sampling, verified harvest events, and controlled environmental records—not oversampling or SMOTE, which would invent regression outcomes.
