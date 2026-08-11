# Project Canary Model Card

## Decision use

Canary forecasts harvest recovery and projects average liveweight on Day 35. These outputs do not set or change the independent rules-based risk rating, diagnose disease, or prescribe treatment.

Day 35 is the farm-approved 1.8 kg weight milestone. The primary weight output is a Day 35 projection, not final liveweight at an unknown harvest date.

## Selected methods

| Outcome | Version | Selected method | Cycles | Distinct building outcomes | Validation MAE | Status |
|---|---|---:|---:|---:|---:|---|
| Predicted harvest recovery | recovery-2.0.0 | age_band_remaining_loss | 5 | 25 | 1.39 points | Prototype; trained on last-recorded recovery proxy |
| Projected Day 35 weight | day35-weight-2.0.0 | historical_remaining_gain | 6 | 31 | 0.178 kg | Prototype; 5 historical 1.8 kg hits |

Validation is nested: the outer loop holds out one complete recorded cycle, while the inner loop tunes only within the remaining cycles. Repeated snapshots receive equal building-cycle weight. Recovery uses Days 7, 14, 21, 28, and the latest eligible checkpoint; Day 35 weight uses checkpoints at Days 7, 14, 21, and 28.

Recovery learned challenger: remaining_loss_huber; operational method: age_band_remaining_loss. Continuous-estimate gate passed: True; 95% classification gate passed: False.
Weight learned challenger: checkpoint_linear_remaining_gain; operational method: historical_remaining_gain. Learned-model regression gate passed: False; target-classification gate passed: True.

## Day 14 recovery backtest

For every eligible building-cycle, Canary recreated the forecast using only information available on Day 14, then compared it with last-recorded recovery.

- Building-cycles evaluated: 25
- Day 14 MAE: 1.40 percentage points
- Day 14 RMSE: 1.89 percentage points
- Actual at/above 95%: 4; correctly recognized: 0.0%
- Actual below 95%: 21; warned below target: 100.0%
- Target-side accuracy: 84.0%; always-below majority baseline: 84.0%
- Interpretation: target-side accuracy does not beat the majority baseline and must not be presented as discrimination proof.

## Recovery candidate comparison

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| age_band_remaining_loss | 1.39 pts | 1.50 pts | 1.80 pts | 0.222 |
| remaining_loss_linear | 1.29 pts | 1.40 pts | 1.80 pts | 0.224 |
| remaining_loss_ridge | 1.31 pts | 1.43 pts | 1.82 pts | 0.207 |
| remaining_loss_huber | 1.24 pts | 1.33 pts | 1.75 pts | 0.260 |
| remaining_loss_gradient_boosting | 1.25 pts | 1.37 pts | 1.77 pts | 0.243 |

## Recovery model reliance

Held-out permutation importance describes predictive reliance on unseen cycles; it does not prove causality.

| Model input | Relative reliance | Held-out MAE increase |
|---|---:|---:|
| humidity_deviation_from_band_pp | 31.6% | 0.843 recovery points |
| cycle_day | 12.6% | 0.337 recovery points |
| mortality_recent_3d_per_1000 | 9.8% | 0.261 recovery points |
| temperature_deviation_from_band_c | 8.3% | 0.221 recovery points |
| environment_out_of_band_days_7d | 7.8% | 0.207 recovery points |
| environment_staleness_days | 7.4% | 0.199 recovery points |
| percentage_alive | 7.1% | 0.190 recovery points |
| weight_gap_pct | 7.0% | 0.188 recovery points |
| mortality_trend_delta_per_1000 | 5.4% | 0.145 recovery points |
| weight_staleness_days | 3.1% | 0.082 recovery points |

## Day 35 candidate comparison

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² | Within 200 g |
|---|---:|---:|---:|---:|---:|
| historical_remaining_gain | 0.178 kg | 0.182 kg | 0.242 kg | 0.126 | 65.3% |
| checkpoint_linear_remaining_gain | 0.216 kg | 0.208 kg | 0.273 kg | -0.118 | 56.5% |
| ridge_remaining_gain | 0.207 kg | 0.200 kg | 0.264 kg | -0.045 | 56.5% |
| huber_remaining_gain | 0.226 kg | 0.223 kg | 0.276 kg | -0.144 | 51.6% |
| gradient_boosting_remaining_gain | 0.203 kg | 0.207 kg | 0.262 kg | -0.026 | 52.4% |

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

The best learned challenger is checkpoint_linear_remaining_gain. The operational method is historical_remaining_gain because no learned challenger cleared every approved gate.

1. Standardize weights near Days 7, 14, 21, 28, and 35, including sample size and zone.
2. Continue comparing historical remaining gain, checkpoint-calibrated linear regression, Ridge, robust Huber regression, and constrained Gradient Boosting.
3. Keep one building record per checkpoint and hold out complete unseen cycles.
4. Report MAE in grams, bias, within-100 g / within-200 g rates, and target-hit usefulness once target hits exist.

## Retraining

Run `uv run python -m scripts.train_models <workbook.xlsx> --performance-summary <summary.xlsx> --output models`. Daily dashboard use performs inference only and never retrains a model.
