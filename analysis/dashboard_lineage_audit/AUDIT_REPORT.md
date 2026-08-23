# Project Canary dashboard lineage audit

## Overall assessment

**Needs revision before defense.** The demo files, model-ready inputs, deployed predictions, and dashboard are internally aligned to the workbook's Daily tab. However, the source workbook contains a material Day 7 and Day 14 bodyweight disagreement between its Daily tab and its detailed weighing tab.

## Verified lineage

- Demo CSV files checked: **6**; all match the Daily tab: **True**.
- Deployed model bundle: **trish-v18-prospective-2026-3**.
- Model 1 target: **final_harvest_recovery** (last-recorded-population recovery proxy, not uniformly Day 35).
- Models 2 and 3 target: **bodyweight_at_day_35** (recorded Day 35 bodyweight).
- 2026-3 prospective scoring uses only Tags 1-3; Lags 1-3 have no 2026-3 records.

## Material source-data issue

- On Day 7, the Daily tab records **114.74 g for Tags 1, Tags 2, and Tags 3**.
- The detailed aggregated weighing tab records **100.28 g, 94.65 g, and 98.04 g**, respectively.
- On Day 14, the Daily tab again repeats **242.76 g** across all three buildings, while the aggregated table records distinct values.
- Canary must retain the Daily-tab values for the defense replay because Trish's feature tables and deployed models were built from that source. The discrepancy must be disclosed and corrected upstream before production use.

## Forecast refresh logic

- Recovery M1 recalculates only through Day 14. Day 21 and Day 28 are held from Day 14; they are not new accuracy observations.
- Day 35 bodyweight uses M2 through Day 14 and M3 through Day 21. Day 28 holds the Day 21 estimate.
- The checkpoint chart now leaves held checkpoints blank and labels the hold explicitly.

## Metric definition

`final_harvest_recovery` equals population on each building's **last recorded day** divided by beginning population. Historical buildings commonly end on Day 49, while 2026-3 currently ends on Day 35. It is therefore an ending-population recovery proxy—not a uniformly Day 35 recovery target and not a reconciled sale count.
