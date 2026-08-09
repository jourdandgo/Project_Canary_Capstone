# Project Canary — Sprint 5 Validation Report

> **Archived and superseded by the Phase 0 correction.** This report documents the earlier build. The current application no longer treats the workbook's maximum daily date as confirmed harvest, no longer substitutes the 2.03 kg farm baseline as the primary weight prediction, and instead projects building-level Day 35 average weight when a measured weight exists. Use `OPERATING_GUIDE.md`, `OPEN_ITEMS.md`, and the current in-app Model proof for defense claims.

## Executive Summary

- **Overall assessment: Share with caveats.** The capstone prototype passes 45 of 45 historical scenario checks and 36 of 36 automated tests. Later health and model-proof audits also corrected misleading building-coverage wording, an unnecessary weight-baseline withholding rule, and actual-versus-predicted traceability.
- **The five required outputs are implemented for all six buildings.** Risk, Why, predicted recovery, experimental weight outlook, and a traceable recommended action behave correctly across Day 14, Day 22, Day 48, staggered placements, missing daily data, inactive buildings, and completed harvests.
- **The remaining constraints are approvals, sparse evidence, and source definitions—not hidden calculation failures.** Risk thresholds and the seven action rules remain provisional. Final-weight labels now come from Farm Performance Summary.xlsx, but the selected model remains a non-personalized farm baseline trained on only 17 accepted building-cycle labels.

## What was tested

The source was `FARM HARVEST DATA.xlsx`, converted without changing the source into 1,666 unique building-day records from 1,785 source rows. The validation used only observations on or before each scenario’s review date.

| Scenario | Evidence covered | Result |
|---|---|---|
| Day 14, cycle 2025-5 | Agreed early-warning boundary; active Tags buildings and inactive Lags buildings | Pass |
| Day 22, cycle 2026-3 | Continued daily operation after Day 14; missing bodyweight handled explicitly | Pass |
| June 12, cycle 2026-2 | Harvested Tags, active Lags 1–2, and inactive Lags 3 in one six-building view | Pass |
| Day 48, cycle 2025-5 | Continued operation after Day 35; Day 35 weight shown as 13 days stale rather than current | Pass |
| Missing daily entry, cycle 2026-3 | Incomplete state uses the latest recorded evidence and labels the delayed-data forecast | Pass after correction |

Detailed, building-level evidence is saved in `artifacts/capstone_validation.json` and can be regenerated with `uv run python -m scripts.validate_capstone`.

## Acceptance against the five required outputs

| Required output | Validation result | Important qualification |
|---|---|---|
| A. Risk rating | Pass | Rules-based, age-aware, independent from forecasts; thresholds remain provisional pending farm approval. |
| B. Why | Pass | Shows primary/supporting drivers, raw values, targets, peer evidence, freshness, dimension scores, equation, and label mapping. |
| C. Predicted final harvest recovery | Pass as limited-data prototype | Ridge model, five completed cycles, cycle-held-out MAE 1.26 percentage points, 80% empirical half-width about 2.00 percentage points. At the exact Day 14 checkpoint, held-out MAE is 1.36 points and target-side accuracy is 80%. |
| D. Predicted final average liveweight | Pass as experimental farm baseline | Uses 17 accepted final-harvest labels from Farm Performance Summary.xlsx. The historical-mean baseline validates best (MAE 0.093 kg) and is available for eligible operating buildings, but is not personalized and has weak target-side accuracy (32.2%). |
| E. Recommended action | Pass as preliminary guidance | Every action traces to a deterministic rule and severity. The seven rules remain pending Doc Raymond’s approval. |

## Correction made during validation

On an incomplete production day, the risk and recommendation layers correctly continued from the latest known observations, but the forecast layer incorrectly displayed “Waiting for placement.” The forecast logic now treats **Active** and **Incomplete** buildings as placed. When enough prior data exists, it updates the estimate and says **“Forecast available — latest recorded data used.”**

The visual review then found that the page summary still said “0 Active buildings” and “No active building” even though three Incomplete buildings were placed and receiving outputs. The summary now counts **Placed buildings** (Active + Incomplete), retains a starting priority, and places a visible delayed-data notice on each incomplete card. Regression tests protect both corrections.

A subsequent top-to-bottom health audit found two additional presentation/logic gaps. First, the dashboard did not explain that some cycles contain fewer than six recorded building-flocks and that staggered placement or harvest can leave only part of the farm operating on a selected date. It now reports both source coverage and as-of operational states. Second, the selected historical-mean weight method was being withheld when no weight was measured even though that method does not use current building weight. Eligible buildings now receive the baseline, explicitly labeled **Farm baseline—not personalized**. Missing weight remains missing in the rules-based risk evidence.

## Data and calculation spot-checks

- **Building-day grain:** Verified. 1,785 source rows become 1,666 unique cycle-building-day rows; 119 duplicate environmental sections are consolidated and no conflicting production duplicates remain.
- **Recovery denominator:** Verified against the agreed definition: ending inventory / beginning inventory.
- **Rules-versus-model separation:** Verified in every scenario. Forecasting and recommendation steps do not change risk scores.
- **Inactive behavior:** Verified. Inactive buildings remain visible and receive neither risk nor forecasts.
- **Harvested behavior:** Verified. Completed buildings stop receiving live forecasts and show completed recovery status.
- **Weight freshness:** Verified. A Day 35 weight used on Day 48 is explicitly shown as 13 days stale.
- **Missing current-day entry:** Verified after correction. The state remains Incomplete while eligible estimates use only prior recorded data.
- **Day 14 recovery backtest:** Verified across 25 completed building-cycles using only information available by Day 14. MAE is 1.36 percentage points, RMSE is 1.70 points, mean error is +0.41 points, and 80% finish on the correctly predicted side of the 95% goal.

## Recommended next steps

1. Have Doc Raymond approve or revise the provisional risk thresholds.
2. Complete the seven-rule action-playbook review in the supplied approval workbook.
3. Confirm the exact denominator and operational definition of the Farm Performance Summary average-liveweight field, and add 2026-2/2026-3 labels when available.
4. Confirm whether workbook End Date always means actual harvest completion.
5. Run a stakeholder walkthrough using the five validated scenarios, then record sign-off decisions in `docs/OPEN_ITEMS.md`.

## Further questions and caveats

The durable open list is maintained in `docs/OPEN_ITEMS.md`. The highest-impact unresolved questions are risk-threshold approval, action-rule approval, building-cycle coverage, final-weight definition, End Date meaning, and the treatment of culls/transfers/partial harvests within recovery.

The prototype is suitable for a capstone demonstration with these caveats visible. It is not yet suitable to present as validated farm policy, a verified final-weight prediction system, a disease-diagnosis tool, or a production deployment.
