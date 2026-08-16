# Project Canary Model Validation Review

Status: **PASS**

## Executive verdict

- Recovery: Ordinary linear regression is the refreshed continuous estimator. Whole-cycle MAE is 1.74 points, cycle-balanced MAE is 1.76 points, RMSE is 2.57 points, and R² is 0.054. Day 14 MAE is 1.95 points. It improves cycle-balanced MAE by 16.1% over the refreshed age-band baseline, but recall of actual outcomes at or above 95% is only 21.1%.
- Day 35 weight: No learned model cleared the champion gates. Historical remaining gain is the transparent operational fallback at about 178 g overall MAE and 181 g from Day 14; uncertainty remains material.
- Risk: retain as a transparent operational-priority score, not as a probability model. Thresholds still require farm validation.

## Data foundation

The refreshed source is already consolidated to 1,666 unique building-day records with 0 blocking conflicts. The prior source had 1,785 rows and 119 repeated Zone A/B rows; those duplicates were aggregated before the refreshed workbook was issued.

## Recovery model

The selected continuous-estimate method is ordinary linear regression trained/evaluated across 6 cycles and 31 distinct building outcomes. The 151 retained training snapshots are repeated decision points, not independent flock outcomes, and each building-cycle receives equal total weight. They comprise Days 7, 14, 21 and 28 plus one separately labelled latest pre-outcome snapshot per building-cycle. Current survival remains valid because it is known on the review date and constrains possible final recovery. Raw beginning inventory and exact building identity are excluded; a compact Tags/Lags group indicator is retained.

Across all retained snapshots, below-target recall is 100%, at/above-target recall is 21.1%, and balanced accuracy is 60.5%. At Day 14, MAE is 1.95 points and R² is 0.025; target-side accuracy remains equal to the majority baseline and at/above-target recall is 0%. Linear coefficients and out-of-fold permutation importance describe the live champion. Held-out SHAP is shown only for the constrained Extra Trees challenger and is interpreted as association, not causation.

## Day 35 weight model

The operational method is historical remaining gain, validated across 6 historical cycles and 31 Day 35 building outcomes using 124 leakage-safe checkpoint rows. Overall, about 65% of projections were within 200 g. At Day 14, MAE is about 181 g and target-side accuracy is about 84%. Five outcomes reached the revised 1.8 kg goal and at/above-target recall remains 0%, so this is an experimental point estimate rather than a reliable target classifier. Ridge is documented as the best learned challenger but is not deployed.

## Required interpretation

Feature reliance and per-building contributions are statistical associations, not proof of cause. Temperature, humidity, feed, and mortality exceptions are shown in a separate operating-condition layer so management receives a specific next check. Water and THI remain unavailable until the field, units, formula, and age-based thresholds are approved.
