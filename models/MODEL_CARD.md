# Project Canary Model Card

## Decision use

Canary forecasts harvest recovery and projects average liveweight on Day 35. These outputs do not set or change the independent rules-based risk rating, diagnose disease, or prescribe treatment.

Day 35 is the 2.0 kg weight milestone. The primary weight output is a Day 35 projection, not final liveweight at an unknown harvest date.

## Selected methods

| Outcome | Version | Selected method | Cycles | Distinct building outcomes | Validation MAE | Status |
|---|---|---:|---:|---:|---:|---|
| Predicted harvest recovery | recovery-0.6.0 | ridge_core | 5 | 25 | 1.32 points | Prototype; trained on last-recorded recovery proxy |
| Projected Day 35 weight | day35-weight-0.3.0 | historical_remaining_gain | 4 | 19 | 0.198 kg | Prototype; no historical 2.0 kg hits |

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
| historical_day35_mean | 0.211 kg | 0.289 kg | 57.9% |
| target_curve_ratio | 0.317 kg | 0.388 kg | 35.5% |
| recent_linear_adg | 0.460 kg | 0.569 kg | 27.6% |
| historical_remaining_gain | 0.198 kg | 0.273 kg | 61.8% |
| ridge_regression | 0.202 kg | 0.279 kg | 64.5% |
| random_forest | 0.253 kg | 0.314 kg | 42.1% |
| gradient_boosting | 0.293 kg | 0.350 kg | 30.3% |

## Day 14 to Day 35 weight backtest

- Building-cycles evaluated: 19
- MAE: 183 g
- RMSE: 254 g
- Bias: +7 g
- Within 200 g: 57.9%
- All evaluated Day 35 outcomes were below 2.0 kg, so target-hit discrimination cannot be tested.

## Important limitations

- Recovery is trained on five recorded cycle histories and 25 building outcomes. The label is last-recorded population divided by beginning population, not confirmed actual-harvest recovery.
- Day 35 weight uses 19 building outcomes across four cycles. All 19 are below 2.0 kg, so target-hit classification cannot yet be evaluated.
- Selection uses cycle-balanced MAE, then chooses the simplest candidate within 5% of the best result to avoid promoting complexity for a trivial gain.
- Uncertainty ranges use the 80th percentile of held-out absolute errors. They are empirical prototype ranges, not formal clinical or statistical guarantees.
- Risk thresholds remain provisional until farm experts approve them. Recommendations remain pending Doc Raymond's action table.

## Day 35 weight improvement plan

The current age-aware baseline adds the historically observed remaining gain from the measurement age to the latest building weight. It is building-responsive, but still limited-data.

1. Standardize weights near Days 7, 14, 21, 28, and 35, including sample size and zone.
2. Continue comparing the age-aware remaining-gain baseline with target-curve, recent-ADG, and compact Ridge candidates as new data arrives.
3. Keep one building record per checkpoint and hold out complete unseen cycles.
4. Report MAE in grams, bias, within-100 g / within-200 g rates, and target-hit usefulness once target hits exist.

## Retraining

Run `uv run python -m scripts.train_models <workbook.xlsx> --performance-summary <summary.xlsx> --output models`. Daily dashboard use performs inference only and never retrains a model.
