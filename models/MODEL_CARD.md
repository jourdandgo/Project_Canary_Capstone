# Project Canary Model Card

## Decision use

Canary forecasts harvest recovery and projects average liveweight on Day 35. These outputs do not set or change the independent rules-based risk rating, diagnose disease, or prescribe treatment.

Day 35 is the farm-approved 1.8 kg weight milestone. The primary weight output is a Day 35 projection, not final liveweight at an unknown harvest date.

## Selected methods

| Outcome | Version | Selected method | Cycles | Distinct building outcomes | Validation MAE | Status |
|---|---|---:|---:|---:|---:|---|
| Predicted harvest recovery | recovery-0.7.0 | ridge_core | 5 | 25 | 1.32 points | Prototype; trained on last-recorded recovery proxy |
| Projected Day 35 weight | day35-weight-0.4.0 | ridge_regression | 6 | 31 | 0.172 kg | Prototype; 5 historical 1.8 kg hits |

Validation holds out one complete recorded cycle at a time. Recovery training is balanced to Days 7, 14, 21, 28, and the latest eligible checkpoint for each building-cycle. The Day 35 comparison uses one building checkpoint at Days 7, 14, 21, and 28.

## Day 14 recovery backtest

For every eligible building-cycle, Canary recreated the forecast using only information available on Day 14, then compared it with last-recorded recovery.

- Building-cycles evaluated: 25
- Day 14 MAE: 1.43 percentage points
- Day 14 RMSE: 1.89 percentage points
- Actual at/above 95%: 4; correctly recognized: 0.0%
- Actual below 95%: 21; warned below target: 100.0%
- Target-side accuracy: 84.0%; always-below majority baseline: 84.0%
- Interpretation: target-side accuracy does not beat the majority baseline and must not be presented as discrimination proof.

## Recovery model reliance

Standardized Ridge coefficients describe association and direction in the fitted model; they do not prove causality.

| Model input | Relative reliance | Direction | Standardized effect |
|---|---:|---|---:|
| percentage_alive | 26.9% | Raises estimate | +1.42 recovery points |
| missing__temperature_recent_avg_c | 16.3% | Raises estimate | +0.86 recovery points |
| mortality_recent_3d_per_1000 | 9.8% | Lowers estimate | -0.52 recovery points |
| feed_cumulative_per_1000_birds | 9.4% | Raises estimate | +0.50 recovery points |
| cycle_day | 9.4% | Raises estimate | +0.49 recovery points |
| mortality_trend_delta_per_1000 | 9.1% | Raises estimate | +0.48 recovery points |
| missing__humidity_recent_avg_pct | 8.1% | Lowers estimate | -0.43 recovery points |
| mortality_daily_per_1000 | 6.1% | Lowers estimate | -0.32 recovery points |
| temperature_recent_avg_c | 3.4% | Lowers estimate | -0.18 recovery points |
| humidity_recent_avg_pct | 1.2% | Lowers estimate | -0.06 recovery points |
| feed_daily_per_1000_birds | 0.3% | Lowers estimate | -0.02 recovery points |

## Day 35 candidate comparison

| Candidate | MAE | RMSE | Within 200 g |
|---|---:|---:|---:|
| historical_day35_mean | 0.210 kg | 0.276 kg | 51.6% |
| target_curve_ratio | 0.322 kg | 0.382 kg | 34.7% |
| recent_linear_adg | 0.432 kg | 0.541 kg | 30.6% |
| historical_remaining_gain | 0.178 kg | 0.242 kg | 65.3% |
| ridge_regression | 0.172 kg | 0.232 kg | 65.3% |
| random_forest | 0.176 kg | 0.238 kg | 66.1% |
| gradient_boosting | 0.206 kg | 0.268 kg | 58.9% |

## Day 14 to Day 35 weight backtest

- Building-cycles evaluated: 31
- MAE: 167 g
- RMSE: 219 g
- Bias: +19 g
- Within 200 g: 64.5%
- Correct side of the 1.8 kg target: 87.1%
- Historical Day 35 results at/above 1.8 kg: 5; below: 26

## Important limitations

- Recovery is trained on five recorded cycle histories and 25 building outcomes. The label is last-recorded population divided by beginning population, not confirmed actual-harvest recovery.
- Day 35 weight uses 31 building outcomes across 6 cycles. The current cycle is excluded from training.
- Recovery selection uses cycle-balanced MAE with a 10% simplicity tolerance; Day 35 weight uses a 5% tolerance. This avoids promoting complexity for a trivial gain.
- Uncertainty ranges use the 80th percentile of held-out absolute errors. They are empirical prototype ranges, not formal clinical or statistical guarantees.
- Risk thresholds remain provisional until farm experts approve them. Recommendations remain pending Doc Raymond's action table.

## Day 35 weight improvement plan

The current champion is ridge_regression. Historical remaining gain remains a required transparent benchmark; it is not used for live forecasts when Ridge remains the validated winner.

1. Standardize weights near Days 7, 14, 21, 28, and 35, including sample size and zone.
2. Continue comparing historical remaining gain with target-curve, recent-ADG, Ridge, Random Forest, and gradient-boosting candidates as new data arrives.
3. Keep one building record per checkpoint and hold out complete unseen cycles.
4. Report MAE in grams, bias, within-100 g / within-200 g rates, and target-hit usefulness once target hits exist.

## Retraining

Run `uv run python -m scripts.train_models <workbook.xlsx> --performance-summary <summary.xlsx> --output models`. Daily dashboard use performs inference only and never retrains a model.
