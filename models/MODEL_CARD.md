# Project Canary Model Card — Final Capstone Baselines

> **Superseded for the 2026-3 dashboard:** the operational outlook now uses the Trish v18 prospective deployment described in `docs/TRISH_V18_INTEGRATION.md`. The transparent methods below remain documented fallbacks and comparison baselines.

## Decision use

Canary forecasts final harvest recovery and average Day 35 bodyweight for each building. Forecasts do not change the independent 0–12 observed-condition risk score, diagnose disease, or prescribe treatment.

| Outcome | Version | Selected method | Development cycles | Building outcomes | Held-out MAE | Held-out R² | Status |
|---|---|---|---:|---:|---:|---:|---|
| Harvest recovery | recovery-capstone-4.1.0 | age-band remaining loss | 6 | 31 | 2.45 pp | 0.056 | Capstone operational; not production-approved |
| Day 35 bodyweight | day35-weight-capstone-3.1.0 | historical remaining gain | 6 | 31 | 127.1 g | 0.310 | Capstone operational; not production-approved |

## Validation

Primary evaluation uses nested leave-one-complete-harvest-cycle-out cross-validation. Each outer fold represents a future unseen cycle; cleaning, feature engineering, expected paths, feature selection and tuning use only the remaining cycles. Days 7, 14, 21 and 28 are the principal validated checkpoints. Daily estimates between checkpoints are available and explicitly labeled.

The authoritative source contains 1,624 unique building-days and 34 building-cycles. Thirty-one outcomes across six cycles are development evidence. The three 2026-3 buildings remain a locked later-cycle audit, and their recovery endpoint is provisional.

## Recovery formula

`projected final recovery = current recorded survival − expected remaining loss for flock age`

Expected remaining loss from the full development data is approximately 7.52 pp at Day 7, 6.50 pp at Day 14, 5.62 pp at Day 21 and 4.03 pp at Day 28. Daily application values are linearly interpolated between validated checkpoints. Final recovery is constrained not to exceed current survival.

Performance: 3.09 pp cycle-macro RMSE, 3.12 pp pooled RMSE, 2.45 pp MAE, R² 0.056 and -0.04 pp bias. Residual LightGBM and XGBoost tied the baseline exactly because their learned corrections collapsed to zero; the simpler baseline was retained.

## Bodyweight formula

`projected Day 35 weight = latest actually observed weight + expected remaining gain for measurement age`

Expected remaining gain from the full development data is approximately 1.373 kg at Day 7, 1.226 kg at Day 14, 0.943 kg at Day 21 and 0.513 kg at Day 28. Target/interpolated weights are never substituted for measurements.

Performance: 149.9 g cycle-macro RMSE, 163.9 g pooled RMSE, 127.1 g MAE, R² 0.310, -0.6 g bias, 50.8% within 100 g and 75.8% within 200 g. This transparent baseline outperformed all learned challengers in the latest balanced refresh.

Checkpoint cycle-macro RMSE is 163.7 g at Day 7, 141.7 g at Day 14, 124.0 g at Day 21 and 129.3 g at Day 28. Day 28 R² is 0.585; this is checkpoint-specific and must not be presented as the overall R².

## Explainability

The selected baselines are explained by their explicit formulas. SHAP is generated only for compatible learned shadow models and represents predictive association, not causation. Many SHAP directions are unstable across held-out cycles and must not drive management recommendations.

## Limitations and promotion

- Six independent development cycles are insufficient for production approval.
- Recovery uses a last-recorded endpoint proxy; the later audit endpoint is provisional.
- Environmental history is incomplete, feed units remain unresolved, and important flock metadata are absent.
- Only one of 31 development building-cycles reached 1,800 g.
- Empirical 80% interval coverage is 75.0% for recovery and 72.6% for bodyweight.

Any learned challenger requires at least three new complete prospective cycles and must retain the prespecified RMSE, bias, worst-cycle, checkpoint and coverage gates before production promotion.
