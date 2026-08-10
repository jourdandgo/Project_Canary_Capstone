# Message for Doc Raymond — Project Canary validation

Hi Doc Raymond! We now have a working Project Canary capstone prototype that reviews the farm's six buildings using the records available on a selected date.

For each current flock, Canary shows:

- A rules-based risk level and the reason for it
- An estimated recovery outcome
- A preliminary Day 35 average-weight projection against the 2.0 kg milestone
- The main problem pattern
- A recommended inspection or management action

Thank you for already confirming these three items:

1. **400 grams on Day 14 is the correct target**, and the same weighing method should be used across buildings.
2. The workbook's **End Date is only the latest or maximum date found in the daily records**. It is not necessarily the actual harvest or completion date.
3. For this capstone, **recovery = ending/surviving population divided by beginning population**. We do not need to adjust for transfers, culls, partial harvests, or missing birds.

## Simple summary of the current risk score

The risk score is separate from the forecasts. Canary checks four warning signs and gives each one **0 to 3 points**:

1. **Weight gap:** how far the latest measured weight is below the target for that bird age. Current point boundaries are 5%, 15%, and 30% below target.
2. **Survival reference gap:** how far the percentage alive is below a temporary straight-line reference toward 95% on Day 35. For example, the Day 22 reference is 96.86%. This is a configurable management assumption—not a biological curve—and we need your approval or correction.
3. **Mortality trend:** whether the latest three days of mortality are worse than the preceding seven-day pattern, with age-adjusted thresholds.
4. **Peer comparison:** whether the building is performing worse than other buildings of a similar age.

The four scores total **0 to 12 points**:

- **Low:** 0–1
- **Medium:** 2–3
- **High:** 4–5
- **Critical:** 6–12

If a measurement is missing, Canary shows that the evidence is incomplete rather than silently giving it zero points. These thresholds are still marked provisional until you approve or revise them.

## Current problem patterns and recommendations

- **No Material Drift:** Continue normal monitoring.
- **Weight Lag Only:** Confirm the weight, then check feed, water, access, temperature, ventilation, lighting, and stocking conditions.
- **Survival Concern Only:** Reconcile counts and mortality, inspect flock condition and the environment, and escalate if mortality continues rising or clinical signs appear.
- **Growth + Survival Drift:** Verify the records, check water and feed first, then inspect environment and flock condition as a combined performance concern.
- **Localized Building Drift:** Compare the building with better-performing peers and inspect building-specific equipment, environment, access, and records.
- **Farm-Wide Drift:** Check shared causes such as feed batch, water source, controller settings, weather, chick source, vaccination history, or a common operational change.
- **Missing or Stale Evidence:** Obtain a fresh weight or correct the missing records before making a major decision.

Current response timing is:

- **Low:** normal flock rounds
- **Medium:** inspect within 24 hours
- **High:** inspect during the current shift
- **Critical:** inspect immediately and notify the farm manager; involve the veterinarian or service technician when health, welfare, mortality, or equipment concerns are present

## Items we still need your guidance on

1. Since End Date is only the last daily record, may we use the **population on the last recorded day as the capstone's “ending/surviving population”**, while clearly calling it a last-recorded result rather than confirmed actual harvest recovery?
2. If we should show a true harvest-complete status, **where should Canary get the actual harvest/completion date or status**?
3. Are the four risk checks, current point thresholds, and **Low/Medium/High/Critical bands** sensible? Which ones would you change?
4. Are the seven problem patterns and recommended checks above appropriate for the farm? Which action or wording should be revised?
5. Are the response times—normal rounds, within 24 hours, current shift, and immediate—appropriate? Who should own each response: farm owner, manager, technician, nutritionist, or veterinarian?
6. When a building is absent from a harvest cycle, does that mean it was not used, or could its data be missing?
7. How is **Ave Live Weight (kg)** calculated, particularly when harvesting happens in stages? Can we validate the two repeated Lagundi 2026-1 entries and add 2026-2/2026-3 final weights when available?
8. For routine weights beyond Day 14, what weighing days, number of birds, and zone/sample approach should be the farm standard?
9. Are the proposed age-based temperature and humidity ranges appropriate? For feed, how many kilograms are in one bag, and are the workbook entries daily or cumulative?
10. For a forecast to be useful, what average error would you accept for recovery and Day 35 weight? Would you prefer Canary to prioritize catching possible misses, even if that creates more false alarms?

Canary's recommendations are only inspection and escalation guidance. It does not diagnose disease, automatically prescribe treatment, or guarantee that an action will improve the result.

Thank you! Your answers will help us clearly separate farm-approved operating rules from prototype assumptions in the final capstone.
