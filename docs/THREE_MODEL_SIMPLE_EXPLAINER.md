# Project Canary: Three-Model Explainer

## 1. What we are trying to do

Project Canary gives farm management earlier visibility into two possible end-of-cycle outcomes:

1. the **end-of-cycle recovery proxy**; and
2. the **average bodyweight on Day 35**.

Canary uses three model engines to answer those two questions. Model 1 handles recovery. Model 3 and the checkpoint method are two alternative ways to project Day 35 bodyweight. The app always names the model being shown, and a sidebar control lets the user switch between the two bodyweight methods.

These are planning outlooks. They do not change Canary's separate rules-based review-priority score, diagnose disease, prescribe treatment, or guarantee the final result.

## 2. The three models

| Engine | What it predicts | When it can predict | Simple description |
|---|---|---|---|
| **Model 1 · Extra Trees** | End-of-cycle recovery proxy | Day 7 and Day 14 | Combines 85 engineered inputs across 500 decision trees. Day 14 is held afterward. |
| **Model 3 · XGBoost** | Day 35 bodyweight | Day 21 | Combines 88 engineered inputs through Day 21 using 250 boosted trees. |
| **Checkpoint model · Historical remaining gain** | Day 35 bodyweight | Days 7, 14, 21, and 28 | Adds the historical average remaining growth for that checkpoint to the latest actual weight. |

## 3. How each model was built, end to end

### Common data preparation

1. Electronic files and recovered physical records were consolidated.
2. Farm, building, cycle, date, flock age, population, mortality, feed, bodyweight, temperature, and humidity fields were standardized.
3. Bodyweight units and percentage calculations were standardized.
4. Duplicate building-day records were checked and removed.
5. Missing measurements remained missing. They were not treated as zero or invented through forward filling.
6. The recovery target was defined as last recorded population divided by beginning population. This is a proxy because independently verified harvest counts are unavailable.
7. The bodyweight target was the recorded average bodyweight on production Day 35.
8. Thirty-one building-cycles from 2025-2 through 2026-2 were used for development. The three buildings from 2026-3 were kept together for a later audit.

### Model 1 · Extra Trees

- **Target:** end-of-cycle recovery proxy.
- **Evidence points:** Day 7 and Day 14.
- **Inputs:** 85 locked engineered variables covering survival, mortality, weight progress, feed patterns, temperature, humidity, THI, population, stocking density, building history, downtime, missingness, and freshness.
- **Training:** 500 Extra Trees regression trees were fitted. Each tree learned different decision splits; the final forecast is their average.
- **Validation:** one complete production cycle was held out at a time. Every building and repeated observation from the held-out cycle stayed outside training.
- **Current app behavior:** recalculate at Day 7 and Day 14, then hold the Day 14 estimate.

### Model 3 · XGBoost

- **Target:** recorded Day 35 bodyweight in grams.
- **Evidence point:** Day 21 only.
- **Inputs:** 88 locked engineered variables through Day 21. Important groups include weight trajectory, projected Day 35 trajectory, temperature, humidity, THI, mortality, feed, population, housing context, missingness, and freshness.
- **Training:** 250 shallow boosted trees were trained sequentially. Each new tree focused on errors left by the earlier trees.
- **Validation:** one complete production cycle was held out at a time, with all preprocessing and fitting restricted to the remaining cycles.
- **Current app behavior:** produce a Day 21 forecast, then hold it afterward.

### Checkpoint model · Historical remaining gain

- **Target:** recorded Day 35 bodyweight in grams.
- **Evidence points:** Days 7, 14, 21, and 28.
- **Direct inputs:** latest actual checkpoint weight, checkpoint day, measurement freshness, and the historical remaining gain calculated inside each training fold.
- **Calculation:** latest measured weight + historical average gain remaining from that checkpoint to Day 35.
- **Model comparison:** linear, regularized, robust, tree-based, boosting, CatBoost, XGBoost, and transparent baselines were compared. The historical remaining-gain method was selected under the predefined simplicity and stability rule.
- **Validation:** one complete production cycle was held out at a time. The remaining-gain average for the held-out cycle was calculated only from the other cycles.
- **Current app behavior:** refresh after an actual Day 7, 14, 21, or 28 measurement and hold the latest result between weigh-ins.

## 4. Performance comparison

| Engine | Development performance | Later 2026-3 audit | What the result means |
|---|---:|---:|---|
| Model 1 · Extra Trees | 2.47 percentage-point pooled MAE; 2.76-point cycle-macro MAE | 4.55-point MAE | Experimental. Its transparent baseline had slightly lower error, so Model 1 has not cleared the deployment gate. |
| Model 3 · XGBoost | 146 g Day 21 MAE; 132 g cycle-macro MAE | 116 g MAE | Useful Day 21 shadow benchmark, but it arrives late. |
| Checkpoint model | 127 g pooled MAE; 121 g cycle-macro MAE | 78 g MAE | Provides earlier forecasts and the best Day 21 error among the two bodyweight options. |

Checkpoint model error by forecast point:

| Forecast point | Held-out MAE | Lead time to Day 35 |
|---|---:|---:|
| Day 7 | 155 g | 28 days |
| Day 14 | 138 g | 21 days |
| Day 21 | 112 g | 14 days |
| Day 28 | 103 g | 7 days |

At the same Day 21 evidence point, the checkpoint method's MAE was approximately **34 g lower** than Model 3's: 112 g versus 146 g.

The 2026-3 audit contains three buildings from one production cycle. It is useful as a later-time check, but it is not three independent cycles and should not be treated as final proof of generalization.

## 5. When to use which model

- Use **Model 1** when demonstrating the original recovery-model approach or reviewing its experimental recovery outlook at Day 7 or Day 14. Do not describe it as the proven champion; it did not beat the recovery baseline.
- Use the **checkpoint model** as the default Day 35 bodyweight outlook. It offers earlier visibility, refreshes four times, is easier to audit, and had lower Day 21 error.
- Use **Model 3** as a Day 21 comparison or shadow benchmark when the team wants to test whether its richer feature set adds insight. Do not use it before Day 21.
- If the two bodyweight forecasts disagree, investigate the difference. Do not average them automatically and do not let either forecast change the risk score.

### Recommended operating position

Keep all three visible for the capstone evaluation. Use Model 1 for recovery, make the checkpoint method the default bodyweight option, and retain Model 3 behind the bodyweight toggle as a clearly labelled Day 21 shadow benchmark.

## 6. Easy speaker notes

### Big picture

> Project Canary is trying to give farm management an earlier view of where a flock may be heading. We forecast two outcomes: the end-of-cycle recovery proxy and average bodyweight on Day 35. We use three engines because two different bodyweight approaches are being compared. Forecasts remain separate from the risk score and never trigger an automatic action.

### Data and validation

> We consolidated the farm records into one building-day dataset, standardized names, dates, units, and percentages, removed duplicates, and preserved missing values. We developed the models on 31 building-cycles from six production cycles. During validation, we held out one complete production cycle at a time, so related records from the same cycle could not leak into both training and testing. We opened 2026-3 only after the methods were frozen.

### Model 1

> Model 1 predicts the end-of-cycle recovery proxy. It is an Extra Trees model made up of 500 decision trees and 85 engineered inputs. It produces outlooks at Day 7 and Day 14. Its historical error was about 2.47 percentage points, but it did not beat the transparent recovery baseline, so we label it experimental. We show it because it is our reconstructed original recovery approach, not because we want to overstate its performance.

### Model 3

> Model 3 predicts Day 35 bodyweight using information available through Day 21. It is an XGBoost model with 250 boosted trees and 88 engineered inputs, including growth trajectory, mortality, feed, temperature, humidity, and THI. Its Day 21 historical error was about 146 grams. It is useful as a richer benchmark, but it provides only two weeks of lead time.

### Checkpoint model

> The checkpoint model also predicts Day 35 bodyweight, but it works at Days 7, 14, 21, and 28. It takes the latest measured weight and adds the average historical growth that remained from that checkpoint to Day 35. It is much easier to explain. Its overall error was about 127 grams, and at Day 21 its error was about 112 grams, compared with 146 grams for Model 3.

### Why the toggle exists

> The toggle lets us compare the two bodyweight approaches without showing conflicting numbers at the same time. The selected model is written directly on the card. The checkpoint method is our default because it is earlier, more accurate at Day 21, and easier to audit. Model 3 remains available as a shadow benchmark.

### THI and feature importance

> Temperature, humidity, and THI were retained in the engineered feature sets for Models 1 and 3. Feature importance tells us what the model relied on statistically; it does not prove that changing THI or another variable will cause the outcome to improve. We use those features as inspection context and research leads, not automatic treatment rules.

### Trust boundary

> These models are reliable enough for a controlled shadow pilot and capstone comparison. They are not reliable enough to diagnose disease, prescribe treatment, or promise a production improvement. Model 1 remains experimental, Model 3 remains a shadow benchmark, and the checkpoint method is the provisional default bodyweight outlook.

