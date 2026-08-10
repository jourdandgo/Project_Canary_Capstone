# Message for Doc Raymond — Project Canary validation

Hi Doc Raymond! We now have a working Project Canary capstone prototype that reviews the farm's six buildings using the records available on a selected date.

For each current flock, Canary shows:

- A rules-based risk level and the reason for it
- An estimated recovery outcome
- A Day 35 average-weight projection against the revised 1.8 kg milestone
- The main problem pattern
- A recommended inspection or management action

Thank you for already confirming these items:

1. The revised checkpoints are **170 g on Day 7, 380 g on Day 14, 800 g on Day 21, 1,200 g on Day 28, and 1,800 g on Day 35**.
2. The workbook's **End Date is only the latest or maximum date found in the daily records**. It is not necessarily the actual harvest or completion date.
3. For this capstone, **recovery = ending/surviving population divided by beginning population**. We do not need to adjust for transfers, culls, partial harvests, or missing birds.
4. The same weighing method should be used across buildings.

Because only weekly weight targets were provided, Canary estimates the missing daily targets by preserving the former curve's within-week shape and rescaling it to hit every revised checkpoint exactly. The existing 40 g Day 0 value is retained only as a working placement anchor.

## Simple summary of the current risk score

The risk score is separate from the forecasts. Canary checks four warning signs and gives each one **0 to 3 points**:

1. **Weight gap:** how far the latest measured weight is below the target for that bird age. Current point boundaries are 5%, 10%, and 30% below target.
2. **Population loss:** beginning birds minus current birds, divided by beginning birds. Starting point boundaries are 3%, 5%, and 7%.
3. **Daily mortality:** latest daily mortality divided by beginning birds. Starting boundaries are 0.1%, 0.2%, and 0.3%.
4. **Environmental conditions:** the worse of (a) daily maximum temperature minus minimum temperature, and (b) humidity outside the age-specific range. The proposed temperature-range boundaries are 2°C, 3°C, and 5°C.

The four scores total **0 to 12 points**:

- **Low:** 0–1
- **Medium:** 2–3
- **High:** 4–5
- **Critical:** 6–12

If a measurement is missing, Canary shows that the evidence is incomplete rather than silently giving it zero points. These thresholds are still marked provisional until you approve or revise them.

Peer comparisons remain visible for context but no longer add points. Missing evidence is shown as missing rather than silently receiving zero.

Important: the current 2/3/5°C temperature-range rule would place most recorded days above 5°C, so this rule needs your calibration before farm use.

## Current problem triggers and recommendations

- **Low Body Weight:** Confirm the weight; check bird condition, feeder allocation, feed/water access, and temperature.
- **High Mortality:** Confirm the count; check sick birds and whether ventilation is adequate.
- **Rapid Population Loss:** Reconcile population and mortality; inspect the flock and escalate unexplained continuing loss.
- **Abnormal Temperature Fluctuation:** Verify the readings; check ventilation, fans, controllers, heaters, cooling equipment, and air leaks.
- **High Humidity:** Check ventilation, litter, leaks, drinkers, cooling pads, and water-pump timing.
- **Low Humidity:** Check the sensor, ventilation schedule, air speed, dust, and weather.
- **Low Feed Intake / Rapid Feed Drop:** Verify the unit and reading; check feed system, quality, access, water, heat, and bird condition. This remains an alert—not a risk point—until feed units are confirmed.
- **Poor Recovery Prediction:** Check flock health, continuing mortality, and any available condemn information. This is supplemental and does not add risk points.
- **Missing or Stale Evidence:** Obtain a fresh weight or correct the missing records before making a major decision.

Current response timing is:

- **Low:** normal flock rounds
- **Medium:** inspect within 24 hours
- **High:** inspect during the current shift
- **Critical:** inspect immediately and notify the farm manager; involve the veterinarian or service technician when health, welfare, mortality, or equipment concerns are present

## Items we still need your guidance on

1. Please approve age-specific **absolute temperature ranges** for this farm and confirm where sensors are positioned. The old target sheet appears too cold for Philippine conditions.
2. Does “temperature deviation” mean **daily maximum minus daily minimum**? Are 2°C / 3°C / 5°C sensible, given that most historical recorded days exceed 5°C?
3. Are the humidity ranges—60–70% on Days 1–7, 55–65% on Days 8–14, and 50–60% after Day 14—appropriate? What should count as warning versus critical?
4. Is **Daily FI/bird** consistently grams or kilograms per bird per day? Should a feed shortfall be compared with the target for that age, and what gaps should trigger warning and critical alerts?
5. Are the triggers, recommended checks, possible causes, and response times above correct? Who owns each response, and when is veterinarian or technician escalation mandatory?
6. Is the checkpoint-calibrated smoothed daily weight curve, including the 40 g Day 0 working anchor, acceptable for non-checkpoint comparisons? A Gompertz curve was tested but not selected because it missed approved checkpoints by up to 42 g.
7. Since End Date is only the last daily record, may we use the last population as the capstone ending-population proxy, and where should a true harvest-complete status come from later?
8. When a building is absent from a cycle, was it not used or could its data be missing?
9. For Zone A / Zone B records, should Canary keep an equal average, or is one zone larger?
10. What average forecast error would be acceptable for farm use? Current held-out errors are about 1.3 recovery points and 172 g for Day 35 weight.

Canary's recommendations are only inspection and escalation guidance. It does not diagnose disease, automatically prescribe treatment, or guarantee that an action will improve the result.

Thank you! Your answers will help us clearly separate farm-approved operating rules from prototype assumptions in the final capstone.
