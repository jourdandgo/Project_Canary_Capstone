# Project Canary Model Card — Trish v19 Final Handoff

## Decision use

Canary presents two planning outlooks. Model 1 estimates the end-of-cycle recovery proxy. Model 3 estimates average bodyweight on Day 35. Neither forecast changes the independent 0–12 observed-condition risk score, diagnoses a cause, prescribes treatment, or guarantees an outcome.

| Model | Outcome | Algorithm | Features | Overall held-out MAE | Held-out R² | Display schedule |
|---|---|---|---:|---:|---:|---|
| Model 1 | Last-recorded-population recovery proxy | Extra Trees | 85 | 1.58 percentage points | 0.373 | Daily through Day 14; held afterward |
| Model 3 | Recorded Day 35 average bodyweight | CatBoost | 85 | 122.1 g | 0.241 | Days 7, 14, and 21; held between and afterward |

## Validation and replay

The application displays Trish's saved leave-one-building-flock-out predictions. All repeated daily rows from the held-out building-flock remain outside that fold's fit. Other buildings from the same production cycle may remain in training, so this should not be described as leave-one-complete-cycle-out validation.

The final serialized artifacts were fitted using all 34 building-flocks. To avoid showing in-sample fitted values for historical or 2026-3 screens, Canary replays the saved held-out prediction rows instead. The bundle manifest records the artifact SHA-256 hashes and MLflow run IDs.

## Checkpoint performance

| Model | Checkpoint | Held-out MAE | 80th-percentile absolute error |
|---|---:|---:|---:|
| Model 1 | Day 7 | 1.62 percentage points | 1.89 percentage points |
| Model 1 | Day 14 | 1.55 percentage points | 2.10 percentage points |
| Model 3 | Day 7 | 122.6 g | 187.2 g |
| Model 3 | Day 14 | 105.7 g | 189.4 g |
| Model 3 | Day 21 | 105.7 g | 158.2 g |

The displayed range is the point prediction plus or minus the checkpoint's 80th-percentile held-out absolute error. It is an empirical error reference, not a formal probabilistic confidence interval.

## Refresh rules

- Model 1 uses daily rows and may refresh daily through Day 14. The Day 14 value is held afterward.
- Model 3 is shown only at Days 7, 14, and 21 because bodyweight is measured mainly at weekly checkpoints. It is held between weigh-ins and after Day 21.
- A Day 28 bodyweight observation may change the rules-based weight-gap score, but it does not create a Model 3 forecast.
- A recorded Day 35 measurement replaces the weight forecast.
- A current record without a measured checkpoint weight does not receive a Model 3 outlook.

## Explainability

Each app detail panel exposes the exact 85-feature model-ready row, evidence cutoff, algorithm, MLflow run, artifact version, prediction, recorded outcome for completed replays, checkpoint MAE, and error-band calculation. Global leave-one-feature-out results show which inputs improved held-out accuracy. They are predictive associations, not causal effects or guaranteed management levers.

## Limitations

- The sample contains only 34 building-flocks.
- Recovery is last recorded population divided by beginning population; it is not independently verified harvest or sales recovery.
- Environmental, feed, health, and commercial records remain incomplete or inconsistently measured.
- The handoff does not package the full raw-data-to-85-feature transformer. Arbitrary future-flock scoring is therefore unavailable in this release.
- The current app is a pilot-stage replay and decision-support prototype, not a production-approved autonomous system.
