# Project Canary — Capstone Health Audit

## Verdict

**Ready for a capstone demonstration with clearly disclosed limitations; not ready for production use.**

> The workbook's maximum daily date is not a verified harvest event. The current build treats every earlier cycle as completed for the capstone display convention, uses each building's last recorded date as the displayed completion date, and clearly labels recovery as the agreed last-recorded population proxy.

The implemented decision flow matches the latest agreed scope: every current building receives a rules-based risk rating and explanation, a harvest-recovery estimate, a Day 35 average-weight outlook when a measured weight exists, and a deterministic recommended action. Earlier cycles show completed outcomes instead of current risk or forecasts. Canary uses Days 1–14 as the early-warning window and Day 35 as the revised 1.8 kg milestone.

## What the audit corrected

1. **Building coverage was misleading.** “Placed buildings” looked like missing dashboard rows. The dashboard now shows how many of the six physical buildings are recorded in the selected cycle and separately counts operating, harvested, not-yet-placed, and unrecorded buildings as of the review date.
2. **Weight outlooks were previously unresponsive.** Corrected checkpoint weights increased the training set to 31 Day 35 building outcomes. A final nested whole-cycle audit found that no learned model cleared the champion gates, so the transparent historical remaining-gain method is the operational fallback and Ridge is retained only as the best learned challenger.
3. **Reduced risk evidence needed stronger disclosure.** Cards now show how many of the four risk dimensions were scored, and the overview warns when an operating building has incomplete risk evidence.
4. **Handoff documents were stale.** The operating guide and validation report now reflect the accepted Farm Performance Summary labels and the current baseline behavior.

## Acceptance against the agreed product

| Requirement | Health | Evidence / qualification |
|---|---|---|
| A. Rules-based risk rating | Implemented | Four age-aware dimensions; 0–12 score; Low/Medium/High/Critical; independent from models. Thresholds remain provisional. |
| B. Why | Implemented | Raw observation, target/peer comparison, freshness, dimension score, equation, label rule, and problem pattern are inspectable. |
| C. Predicted harvest recovery | Validated experimental prototype | Current survival minus predicted remaining loss; 6 historical cycles, 31 building outcomes, and 151 balanced as-of snapshots. Ordinary linear remaining-loss regression is selected for the continuous estimate with nested whole-cycle MAE 1.74 points, cycle-balanced MAE 1.76 points, RMSE 2.57 points, and R² 0.054. It has weak at/above-95% recall, so the app presents an estimate and range—not a target-hit probability. |
| D. Projected Day 35 average weight | Experimental transparent fallback | 6 historical cycles, 31 observed Day 35 outcomes, and 124 leakage-safe checkpoint snapshots. Historical remaining gain remains operational at 178 g pooled MAE and 182 g cycle-balanced MAE because Ridge and the other learned candidates failed the predeclared gates. |
| E. Recommended action | Implemented as preliminary guidance | Seven deterministic problem-pattern rules with severity, inspection checklist, escalation trigger, version, and approval status. Doc Raymond approval remains open. |
| Day 1 through actual harvest | Implemented | Historical validation covers Day 14, Day 22, Day 48, incomplete days, mixed states, and completed harvest. |
| Six-building view | Implemented with explicit source-state handling | Six physical buildings always appear; only recorded and operating flocks receive live outputs. |
| Day 35 milestone | Implemented | 1.8 kg is treated as a management milestone, not an assumed harvest date or an artificially transformed final-harvest label. |
| Traceability and explainability | Implemented | Risk, forecast, and action evidence are separate and versioned in the building detail view. |

## Source coverage by cycle

| Cycle | Buildings recorded in source | Important interpretation |
|---|---:|---|
| 2025-2 | 3 of 6 | Tags 1–3 only |
| 2025-3 | 5 of 6 | No Lags 3 record |
| 2025-4 | 5 of 6 | No Lags 3 record |
| 2025-5 | 6 of 6 | Full physical-building coverage |
| 2026-1 | 6 of 6 | Full coverage; corrected checkpoint weights are available through Day 35 |
| 2026-2 | 6 of 6 | Full coverage; corrected checkpoint weights are available, plus detailed Day 1–14 zone weights for Lags 1–3 |
| 2026-3 | 3 of 6 | Tags 1–3 only; corrected checkpoint weights are available through the latest recorded checkpoint |

`Weights Cleaned.xlsx` is already represented correctly in `FARM HARVEST DATA.xlsx`: its zone A/B samples aggregate exactly to the 40 corresponding Lags building-day weights for cycle 2026-2. It is useful for those records but too narrow to establish a general zone-level model.

## Remaining issues ranked by defense impact

1. **Approve the risk thresholds and rating bands.** Calculations are reproducible, but the cutoffs are not yet farm policy.
2. **Approve or revise the seven recommended-action rules.** Until then the interface correctly calls them preliminary guidance.
3. **Confirm missing building-cycle records.** The farm should say whether absent buildings were unused or whether their data is missing.
4. **Treat the Day 35 projection honestly.** Historical remaining gain is the operational fallback; Ridge is only the best learned challenger. With 31 outcomes and only five target hits, more completed cycles are required before a learned model or target-hit classifier can be trusted.
5. **Continue recording comparable checkpoint weights.** Use a consistent weighing method on Days 7, 14, 21, 28, and 35 and record sample size and zone where possible.
6. **Confirm End Date and recovery accounting.** Establish whether End Date is actual harvest and how transfers, culls, missing birds, and partial harvests affect survived birds.
7. **Agree on a bodyweight sampling protocol.** Sampling frequency, bird count, and representativeness affect both rules and future model quality.
8. **Agree on acceptable forecast error.** The app now reports overall and Day 14 performance, but the farm has not yet decided what error is acceptable for a management decision.

## Verification result

- Automated tests: rerun as part of the final release audit
- Historical acceptance checks: **45 of 45 passed**
- Visual QA: latest partial cycle (2026-3) and full-coverage cycle (2026-2) reviewed
- Latest-cycle eligible weight outlooks: generated by the historical remaining-gain fallback when a measured checkpoint exists

The prototype is defensible if the presentation distinguishes implemented mechanics from validated farm policy, presents recovery as the agreed last-recorded proxy, and discloses the small target-hit sample for the Day 35 model.
