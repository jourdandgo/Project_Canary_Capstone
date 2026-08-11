# Project Canary Model Validation Review

Status: **PASS**

## Executive verdict

- Recovery: Ordinary linear regression is a validated prototype for a continuous estimate. Nested whole-cycle MAE is 1.37 points overall and 1.65 points at Day 14. Target-side discrimination is not established.
- Day 35 weight: No learned model cleared the champion gates. Historical remaining gain is the transparent operational fallback at about 178 g overall MAE and 181 g from Day 14; uncertainty remains material.
- Risk: retain as a transparent operational-priority score, not as a probability model. Thresholds still require farm validation.

## Data foundation

The source contained 1,785 rows. Canary consolidated 119 repeated rows into 1,666 unique building-day records with 0 blocking conflicts.

## Recovery model

The selected continuous-estimate method is ordinary linear regression trained/evaluated across 5 cycles and 25 distinct building outcomes. The 122 checkpoint rows are repeated time snapshots, not independent flock outcomes, and each building-cycle receives equal total weight. Current survival remains a valid input because it is known on the review date and constrains possible final recovery; raw beginning inventory and building identity are excluded.

At Day 14, target-side accuracy is 80.0%, below the 84.0% majority baseline; at/above-target recall is 0.0%.

## Day 35 weight model

The operational method is historical remaining gain, validated across 6 historical cycles and 31 Day 35 building outcomes using 124 leakage-safe checkpoint rows. Overall, about 65% of projections were within 200 g. At Day 14, MAE is about 181 g and target-side accuracy is about 84%. Five outcomes reached the revised 1.8 kg goal and at/above-target recall remains 0%, so this is an experimental point estimate rather than a reliable target classifier. Ridge is documented as the best learned challenger but is not deployed.

## Required interpretation

Feature reliance and per-building contributions are statistical associations, not proof of cause. Temperature, humidity, feed, and mortality exceptions are shown in a separate operating-condition layer so management receives a specific next check. Water and THI remain unavailable until the field, units, formula, and age-based thresholds are approved.
