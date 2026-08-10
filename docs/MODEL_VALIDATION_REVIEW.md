# Project Canary Model Validation Review

Status: **PASS**

## Executive verdict

- Recovery: Directional point estimate only; target-side discrimination is not established. Held-out MAE is 1.32 points overall and 1.43 points at Day 14.
- Day 35 weight: Useful age-aware baseline; uncertainty is material and target-hit discrimination cannot be tested. Held-out MAE is 198 g overall and 183 g from Day 14.
- Risk: retain as a transparent operational-priority score, not as a probability model. Thresholds still require farm validation.

## Data foundation

The source contained 1,785 rows. Canary consolidated 119 repeated rows into 1,666 unique building-day records with 0 blocking conflicts.

## Recovery model

The selected method is `ridge_core`, a compact Ridge regression trained/evaluated across 5 cycles and 25 distinct building outcomes. The 122 checkpoint rows are repeated time snapshots, not independent flock outcomes. Current survival remains a valid input because it is the current numerator of the same recovery ratio; raw beginning inventory and building identity were removed because they did not strengthen the defensible held-out result.

At Day 14, target-side accuracy is 84.0%, equal to the 84.0% majority baseline; at/above-target recall is 0.0%.

## Day 35 weight model

The selected method is `historical_remaining_gain`, validated across 4 cycles and 19 Day 35 building outcomes. At Day 14, 57.9% of projections were within 200 g. Historical Day 35 target hits: 0.

## Required interpretation

Feature reliance and per-building contributions are statistical associations, not proof of cause. Temperature, humidity, feed, and mortality exceptions are shown in a separate operating-condition layer so management receives a specific next check. Water and THI remain unavailable until the field, units, formula, and age-based thresholds are approved.
