# Project Canary Model Card

## Decision use

Canary forecasts harvest recovery and projects average liveweight on Day 35. These outputs do not set or change the independent rules-based risk rating, diagnose disease, or prescribe treatment.

Day 35 is the farm-approved 1.8 kg weight milestone. The primary weight output is a Day 35 projection, not final liveweight at an unknown harvest date.

## Selected methods

| Outcome | Version | Selected method | Cycles | Distinct building outcomes | Validation MAE | Status |
|---|---|---:|---:|---:|---:|---|
| Predicted harvest recovery | recovery-1.0.0 | linear_regression | 5 | 25 | 1.37 points | Prototype; trained on last-recorded recovery proxy |
| Projected Day 35 weight | day35-weight-1.0.0 | historical_remaining_gain | 6 | 31 | 0.178 kg | Prototype; 5 historical 1.8 kg hits |

Validation is nested: the outer loop holds out one complete recorded cycle, while the inner loop tunes only within the remaining cycles. Repeated snapshots receive equal building-cycle weight. Recovery uses Days 7, 14, 21, 28, and the latest eligible checkpoint; Day 35 weight uses checkpoints at Days 7, 14, 21, and 28.

Recovery learned challenger: linear_regression; operational method: linear_regression. Continuous-estimate gate passed: True; 95% classification gate passed: False.
Weight learned challenger: ridge_regression; operational method: historical_remaining_gain. Learned-model regression gate passed: False; target-classification gate passed: True.

## Day 14 recovery backtest

For every eligible building-cycle, Canary recreated the forecast using only information available on Day 14, then compared it with last-recorded recovery.

- Building-cycles evaluated: 25
- Day 14 MAE: 1.65 percentage points
- Day 14 RMSE: 2.18 percentage points
- Actual at/above 95%: 4; correctly recognized: 0.0%
- Actual below 95%: 21; warned below target: 95.2%
- Target-side accuracy: 80.0%; always-below majority baseline: 84.0%
- Interpretation: target-side accuracy does not beat the majority baseline and must not be presented as discrimination proof.

## Recovery candidate comparison

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| historical_mean | 1.66 pts | 1.73 pts | 2.17 pts | -0.132 |
| linear_regression | 1.37 pts | 1.48 pts | 1.84 pts | 0.189 |
| ridge_core | 1.55 pts | 1.59 pts | 1.97 pts | 0.070 |
| gradient_boosting | 1.57 pts | 1.66 pts | 2.08 pts | -0.041 |

## Recovery model reliance

Held-out permutation importance describes predictive reliance on unseen cycles; it does not prove causality.

| Model input | Relative reliance | Held-out MAE increase |
|---|---:|---:|
| humidity_deviation_from_band_pp | 28.3% | 0.959 recovery points |
| percentage_alive | 25.7% | 0.872 recovery points |
| cycle_day | 11.7% | 0.397 recovery points |
| mortality_recent_3d_per_1000 | 10.3% | 0.350 recovery points |
| mortality_trend_delta_per_1000 | 6.3% | 0.214 recovery points |
| temperature_deviation_from_band_c | 5.2% | 0.175 recovery points |
| environment_staleness_days | 4.6% | 0.155 recovery points |
| environment_out_of_band_days_7d | 4.5% | 0.151 recovery points |
| weight_gap_pct | 2.7% | 0.092 recovery points |
| weight_staleness_days | 0.7% | 0.025 recovery points |

## Day 35 candidate comparison

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² | Within 200 g |
|---|---:|---:|---:|---:|---:|
| historical_remaining_gain | 0.178 kg | 0.182 kg | 0.242 kg | 0.126 | 65.3% |
| linear_regression | 0.217 kg | 0.209 kg | 0.273 kg | -0.114 | 55.6% |
| ridge_regression | 0.197 kg | 0.193 kg | 0.252 kg | 0.048 | 60.5% |
| gradient_boosting | 0.189 kg | 0.189 kg | 0.256 kg | 0.018 | 58.1% |

## Day 14 to Day 35 weight backtest

- Building-cycles evaluated: 31
- MAE: 181 g
- RMSE: 237 g
- Bias: +2 g
- Within 200 g: 58.1%
- Correct side of the 1.8 kg target: 83.9%
- Historical Day 35 results at/above 1.8 kg: 5; below: 26

## Important limitations

- Recovery is trained on five recorded cycle histories and 25 building outcomes. The label is last-recorded population divided by beginning population, not confirmed actual-harvest recovery.
- Day 35 weight uses 31 building outcomes across 6 cycles. The current cycle is excluded from training.
- Both comparisons use nested whole-cycle validation and cycle-balanced MAE as the primary metric. RMSE and R² are secondary checks; target-side metrics describe decision usefulness.
- Uncertainty ranges use the 80th percentile of held-out absolute errors. They are empirical prototype ranges, not formal clinical or statistical guarantees.
- Risk thresholds remain provisional until farm experts approve them. Recommendations remain pending Doc Raymond's action table.

## Day 35 weight improvement plan

The best learned challenger is ridge_regression. The operational method is historical_remaining_gain because no learned challenger cleared every approved gate.

1. Standardize weights near Days 7, 14, 21, 28, and 35, including sample size and zone.
2. Continue comparing historical remaining gain, ordinary linear regression, Ridge, constrained Gradient Boosting, and XGBoost where its runtime is available.
3. Keep one building record per checkpoint and hold out complete unseen cycles.
4. Report MAE in grams, bias, within-100 g / within-200 g rates, and target-hit usefulness once target hits exist.

## Retraining

Run `uv run python -m scripts.train_models <workbook.xlsx> --performance-summary <summary.xlsx> --output models`. Daily dashboard use performs inference only and never retrains a model.
