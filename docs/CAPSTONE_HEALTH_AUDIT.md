# Project Canary — Capstone Health Audit

## Verdict

**Ready for a capstone demonstration with clearly disclosed limitations; not ready for production or a claim of personalized weight prediction.**

> **Superseded note (Phase 0 correction):** the workbook's maximum daily date is not a confirmed harvest date. The current build uses **Records ended**, keeps recovery as a disclosed last-recorded proxy, and makes projected Day 35 average weight the primary weight output. Any historical statement below that calls the maximum date “harvested” or describes the 2.03 kg farm baseline as the primary prediction is retained only as an audit of the earlier build.

The implemented decision flow matches the latest agreed scope: every current or records-ended building receives a rules-based risk rating and explanation, a harvest-recovery estimate, a Day 35 average-weight outlook when a measured weight exists, and a deterministic recommended action. Canary uses Days 1–14 as the early-warning window and Day 35 as the 2.0 kg milestone.

## What the audit corrected

1. **Building coverage was misleading.** “Placed buildings” looked like missing dashboard rows. The dashboard now shows how many of the six physical buildings are recorded in the selected cycle and separately counts operating, harvested, not-yet-placed, and unrecorded buildings as of the review date.
2. **Weight outlooks were unnecessarily withheld.** The selected final-weight method is a historical-mean model that does not depend on current building weight. Eligible buildings now receive that estimate even without a weighing, but it is explicitly labeled **Farm baseline—not personalized**.
3. **Reduced risk evidence needed stronger disclosure.** Cards now show how many of the four risk dimensions were scored, and the overview warns when an operating building has incomplete risk evidence.
4. **Handoff documents were stale.** The operating guide and validation report now reflect the accepted Farm Performance Summary labels and the current baseline behavior.

## Acceptance against the agreed product

| Requirement | Health | Evidence / qualification |
|---|---|---|
| A. Rules-based risk rating | Implemented | Four age-aware dimensions; 0–12 score; Low/Medium/High/Critical; independent from models. Thresholds remain provisional. |
| B. Why | Implemented | Raw observation, target/peer comparison, freshness, dimension score, equation, label rule, and problem pattern are inspectable. |
| C. Predicted final harvest recovery | Prototype-ready | Ridge model; 5 completed cycles, 25 building-cycles; cycle-held-out MAE 1.26 percentage points. An exact Day 14 backtest has MAE 1.36 points and correctly identifies the final side of the 95% target in 80% of cases. |
| D. Predicted final average liveweight | Experimental baseline | 17 accepted final-weight labels across 5 cycles; historical-mean MAE 0.093 kg. It is not personalized and target-side accuracy is only 32.2%. |
| E. Recommended action | Implemented as preliminary guidance | Seven deterministic problem-pattern rules with severity, inspection checklist, escalation trigger, version, and approval status. Doc Raymond approval remains open. |
| Day 1 through actual harvest | Implemented | Historical validation covers Day 14, Day 22, Day 48, incomplete days, mixed states, and completed harvest. |
| Six-building view | Implemented with explicit source-state handling | Six physical buildings always appear; only recorded and operating flocks receive live outputs. |
| Day 35 milestone | Implemented | 2.0 kg is treated as a management milestone, not an assumed harvest date or linearly derived label. |
| Traceability and explainability | Implemented | Risk, forecast, and action evidence are separate and versioned in the building detail view. |

## Source coverage by cycle

| Cycle | Buildings recorded in source | Important interpretation |
|---|---:|---|
| 2025-2 | 3 of 6 | Tags 1–3 only |
| 2025-3 | 5 of 6 | No Lags 3 record |
| 2025-4 | 5 of 6 | No Lags 3 record |
| 2025-5 | 6 of 6 | Full physical-building coverage |
| 2026-1 | 6 of 6 | Full coverage, but very limited bodyweight measurements |
| 2026-2 | 6 of 6 | Full coverage; detailed Day 1–14 zone weights exist only for Lags 1–3 |
| 2026-3 | 3 of 6 | Tags 1–3 only; no bodyweight measurements yet |

`Weights Cleaned.xlsx` is already represented correctly in `FARM HARVEST DATA.xlsx`: its zone A/B samples aggregate exactly to the 40 corresponding Lags building-day weights for cycle 2026-2. It is useful for those records but too narrow to establish a general zone-level model.

## Remaining issues ranked by defense impact

1. **Approve the risk thresholds and rating bands.** Calculations are reproducible, but the cutoffs are not yet farm policy.
2. **Approve or revise the seven recommended-action rules.** Until then the interface correctly calls them preliminary guidance.
3. **Confirm missing building-cycle records.** The farm should say whether absent buildings were unused or whether their data is missing.
4. **Treat final-weight output honestly.** It satisfies the required output as the best validated naïve baseline, but it does not yet answer building-specific target risk well. More completed cycles with frequent weights and trusted final labels are required for personalization.
5. **Confirm the average-liveweight definition and provide 2026-2/2026-3 final labels when available.** Do not use the suspicious duplicated 2026-1 Lagundi summary rows unless validated.
6. **Confirm End Date and recovery accounting.** Establish whether End Date is actual harvest and how transfers, culls, missing birds, and partial harvests affect survived birds.
7. **Agree on a bodyweight sampling protocol.** Sampling frequency, bird count, and representativeness affect both rules and future model quality.
8. **Agree on acceptable forecast error.** The app now reports overall and Day 14 performance, but the farm has not yet decided what error is acceptable for a management decision.

## Verification result

- Automated tests: **36 passed**
- Historical acceptance checks: **45 of 45 passed**
- Visual QA: latest partial cycle (2026-3) and full-coverage cycle (2026-2) reviewed
- Latest-cycle eligible weight outlooks: **3 of 3**, all explicitly labeled farm baseline

The prototype is defensible if the presentation distinguishes implemented mechanics from validated farm policy and distinguishes a farm baseline from a personalized predictive model.
