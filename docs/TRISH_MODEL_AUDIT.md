# Trish Model Audit — Project Canary

## Executive verdict

**Do not deploy the submitted pickle directly.** Trish's work contains strong engineering ideas, but its headline score does not survive Canary's stricter unseen-cycle test. Canary adopted its cleaning layers, zone aggregation, environmental feature catalog, missingness flags, group-aware evaluation, and tree-ensemble challenger. On the latest corrected workbook, ordinary linear regression is the recovery continuous-estimate champion; constrained Extra Trees remains a nonlinear sensitivity/SHAP challenger. Historical remaining gain remains the Day 35 operational fallback.

## What Trish built

- Target (Y): `harvest_recovery`, repeated across Days 1–14 for each building-cycle.
- Training table: 476 daily snapshots representing 34 building-cycle outcomes across seven cycles.
- Predictors: 79 encoded inputs spanning survival/population, mortality, feed, environment, THI, bodyweight, stocking-density/farm proxies, and vaccine observations.
- Submitted model: default `ExtraTreesRegressor`, 100 trees, unlimited depth, minimum leaf size 1, all features considered at each split.
- Reported validation: leave one **building-cycle** out at a time.
- Saved test result: 0.85 percentage-point MAE, 1.04-point RMSE, R² 0.77 on 112 rows from eight building outcomes.

## What was done well

1. Clear bronze/silver/gold data layers and reproducible Python pipeline.
2. Group-aware splitting instead of a random daily-row split.
3. Multiple candidate models and a naïve baseline.
4. Strong environmental feature catalog, including temperature range, humidity range, THI, stress-day counts, and lag/trend features.
5. Interpolation and fallback flags were retained, making synthetic values auditable.
6. Zone A/B rows were collapsed before modeling.

## Material issues

### High — same-cycle information remains in training

Trish's LOGO fold holds out one `(harvest_cycle, building)` group. Other buildings from the same harvest cycle remain in training. Canary's decision standard is stricter: **every building from the held-out harvest cycle must stay out of training**.

When the submitted 79-feature approach was rerun with whole-cycle holdouts, Extra Trees changed from a strong positive R² to a negative R²:

| Validation design | Extra Trees MAE | RMSE | R² |
|---|---:|---:|---:|
| Submitted 80/20 grouped test | 0.85 pts | 1.04 pts | 0.77 |
| Submitted-data whole-cycle holdout | 2.04 pts pooled / 2.11 pts cycle-balanced | 2.41 pts | -0.16 |

### High — early weight snapshots can use later measurements

The biological growth builder interpolates a building's entire bodyweight trajectory before Day 1–14 snapshots are created. An interpolated Day 8–13 weight can therefore use the later Day 14 anchor. That is look-ahead leakage for an earlier forecast.

- 23.95% of Trish's Day 1–14 weight values are interpolated.
- 59.45% are target-curve fallback values rather than measurements.
- Only about 16.6% are neither interpolated nor target fallback.

Canary instead carries only the latest weight actually recorded by the review date and marks staleness explicitly.

### High — outcomes and source data do not match the corrected Canary evidence

Trish's workbook is a different, older file. It includes 34 labeled recovery outcomes through 2026-3, while Canary's verified training set retains 25 historical recovery outcomes through 2026-1. The Trish file also lacks the corrected bodyweight recordings later supplied for 2026-1 to 2026-3.

### Medium — model capacity is too high for the effective sample

The submitted Extra Trees model uses 79 features with only 34 independent outcomes, unlimited tree depth, and one-sample leaves. The 476 rows are repeated views of 34 outcomes, not 476 independent flocks. This configuration can memorize small-sample patterns.

### Medium — feature selection occurs before cross-validation

Zero-variance, duplicate, and correlation filtering are run on the full modeling table before LOGO validation. Any data-driven selection should ideally be fitted inside each training fold.

### Medium — artifact/version and result drift

The pickle contains scikit-learn 1.8 estimators, while the current environment is 1.9. The folder also contains more than one Extra Trees LOGO result (about 1.01 and 1.11 points), so the exact model-to-report lineage is not singular.

## Superseded comparison and refreshed decision

All models below use the same compact, as-of feature set and nested leave-one-complete-cycle-out validation.

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² | Within 2 pts |
|---|---:|---:|---:|---:|---:|
| Age-band remaining-loss baseline | 1.39 pts | 1.50 pts | 1.80 pts | 0.222 | Transparent baseline |
| Ordinary linear regression | 1.47 | 1.55 | 1.90 | 0.135 | Compared |
| Compact Ridge | 1.28 | 1.33 | 1.71 | 0.299 | Compared |
| Gradient Boosting | 1.32 | 1.39 | 1.78 | 0.243 | Compared |
| **Constrained Extra Trees** | **1.17** | **1.27** | **1.57** | **0.409** | **Selected continuous estimate** |

The table above records the earlier five-cycle audit and is retained for lineage. After 2026-2 endpoint data were corrected, Canary retrained on 31 outcomes across six cycles and 151 balanced snapshots. Under the refreshed whole-cycle protocol, ordinary linear regression has MAE 1.74 points, cycle-balanced MAE 1.76 points, RMSE 2.57 points and R² 0.054. It improves cycle-balanced MAE by 16.1% over the refreshed age-band baseline and is simpler than the nearly tied Ridge model. It remains an experimental continuous estimate because at/above-95% recall is only 21.1%.

## Final comparison on corrected Canary Day 35 weight data

| Candidate | MAE | Cycle-balanced MAE | RMSE | R² | Within 200 g |
|---|---:|---:|---:|---:|---:|
| **Historical remaining gain** | **178 g** | **182 g** | **242 g** | **0.126** | **65.3%** |
| Checkpoint linear regression | 216 g | 208 g | 273 g | -0.118 | 56.5% |
| Ridge | 207 g | 200 g | 264 g | -0.045 | 56.5% |
| Robust Huber | 226 g | 223 g | 276 g | -0.144 | 51.6% |
| Gradient Boosting | 203 g | 207 g | 262 g | -0.026 | 52.4% |

**Decision:** retain historical remaining gain. No learned candidate improves its cycle-balanced MAE by the required 10%, keeps positive R², and reaches the 70% within-200 g gate. The apparently perfect earlier linear result was rejected after an audit found that the derived remaining-gain label had accidentally entered X; the corrected comparison above contains no such leakage.

## What Canary should reuse

1. Keep the teammate approach as a documented secondary audit, not as the live model.
2. Reuse the environmental feature taxonomy for future research and EDA, but only add features with reliable as-of coverage.
3. Preserve explicit flags for observed, interpolated, carried-forward, and unavailable measurements.
4. Preserve the layered data-pipeline pattern and model artifact lineage.
5. Add R² to MAE, RMSE, bias, variability, uncertainty, and target-side metrics—but never use R² alone for selection.

## Final recommendation

Use Trish's work as a valuable engineering reference. Do not deploy the submitted pickle. Canary's current choices are driven by nested complete-cycle holdouts and predeclared gates: ordinary linear regression for continuous recovery estimation, constrained Extra Trees as the nonlinear SHAP challenger, and historical remaining gain for the experimental Day 35 outlook.

Reproducible audit outputs are stored under `analysis/trish_model_audit/`.
