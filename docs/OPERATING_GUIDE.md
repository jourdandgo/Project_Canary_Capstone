# Project Canary — Operating Guide

## What Canary answers

For the latest cycle, Canary answers for each of the six buildings:

1. How operationally concerning is this building today?
2. Why did it receive that risk rating?
3. What harvest recovery is currently expected against the 95% goal?
4. Is the building projected to reach 2.0 kg on Day 35?
5. What should management inspect or focus on next?

Risk is a rules-based operating signal. It is not a probability and is not changed by either forecast.

For every earlier cycle, Canary instead shows a completed-results summary: **harvest completed on**, **actual harvest recovery**, and **actual final average weight** when a defensible weight record is available. Historical screens do not show risk ratings, predictions, or recommendations.

## Day 14, Day 35, and harvest recovery

- **Day 35 is a management milestone:** the average bird should weigh at least 2.0 kg.
- **Days 1–14 are the early-warning window:** Day 14 has a confirmed 400 g target and provides time to investigate emerging gaps.
- **Day 35 is not automatically the end of the cycle:** it is the primary 2.0 kg weight milestone.
- **Harvest recovery is a separate outcome:** for the latest cycle, Canary forecasts it against 95%. For earlier cycles, actual recovery uses ending recorded population divided by beginning population.
- Canary does not estimate a Day 35 outcome by multiplying a later harvest weight by `35/49`.

Before Day 35, the dashboard projects whether the flock is on track for 2.0 kg. On Day 35 it uses the recorded Day 35 weight when available. After Day 35 it keeps the Day 35 milestone visible and continues the operational assessment for as long as daily records exist.

## Start the prototype

From the `canary_app` folder:

```bash
uv sync --dev
uv run streamlit run app.py
```

Canary automatically looks for `FARM HARVEST DATA.xlsx` one folder above the app. A user can also upload a compatible workbook from the sidebar.

## Daily owner workflow

1. Select the harvest cycle.
2. If it is the latest cycle, select the review date. Treat this as “everything known as of this day.” Earlier cycles automatically open as completed summaries, so they do not ask for a review date.
3. Begin on **Home**. The six cards always stay in the same order: Tags 1–3, then Lags 1–3.
4. Use **Review first** to identify the current flock with the highest rules-based concern.
5. Read the risk score, main issue, predicted outcomes, and recommended next step on each current-flock card.
6. Select **View [building] details** to open the dedicated **Building View** page.
7. Follow its fixed sequence: decision summary and next check; risk-score table; forecast deep dive; operational checks; then outlook history.
8. Expand **See why this action was selected** for the problem pattern, severity, rule ID, version, inspection list, and approval status.
9. Expand **See the forecast evidence and model proof** for the prediction-time input audit and building-specific model trace.
10. Use **Business Value** to adjust price, sale-weight, improvement, and cycle assumptions and see the estimated gross revenue represented by the recovery gap.
11. Use **EDA & Insights** for question-led historical evidence and **Canary Methodology** for the complete data, scoring, model, validation, and recommendation logic.

## How to interpret each result

- **Risk rating:** Low, Medium, High, or Critical operational concern from the four rules-based dimensions. Missing dimensions are not silently scored as good.
- **Why:** Deterministic evidence showing the actual value, comparison, score, freshness, and problem pattern.
- **Predicted harvest recovery:** Estimated last-recorded population / beginning population, compared with the 95% target. This is a disclosed capstone proxy for true harvest recovery.
- **Projected Day 35 weight:** Latest measured building weight plus the average remaining gain historically observed from that age to Day 35. It is compared with 2.0 kg. No measured weight means no building projection.
- **Recommended action:** A deterministic inspection or management response. Until Doc Raymond approves the playbook, it remains preliminary guidance.
- **Estimated gross revenue at risk:** Beginning birds × predicted recovery gap to 95% × assumed sale weight × assumed selling price. It is not profit or guaranteed savings.

## Business-value estimator

The Business Value page uses four editable planning assumptions: live-chicken price in PHP/kg, sale weight per bird, recovery improvement in percentage points, and production cycles per year.

- One recovery point represents `beginning population × 1%` birds.
- Gross revenue per recovered bird is `assumed sale weight × assumed price/kg`.
- Gross revenue at risk uses the predicted recovery gap to 95%.
- The selected improvement scenario is capped at each building's gap to 95%.

The defaults are clearly labeled placeholders. The output excludes feed, labor, electricity, treatment, intervention cost, mortality timing, and price changes. It must never be called profit or a guaranteed benefit.

## Current-versus-historical behavior

| Cycle selected | What Canary shows |
|---|---|
| Latest cycle | Review-date selector, current risk ratings, explanations, harvest-recovery predictions, Day 35 weight projections, and recommended next actions. |
| Any earlier cycle | All six buildings in fixed positions, each recorded building's last daily date as **Harvest completed on**, calculated actual recovery, and actual final average weight when available. No historical risk, forecast, or recommendation output. |

This is a deliberate capstone convention. The source workbook's `End Date` is the maximum daily-record date rather than a verified harvest-event flag. The interface clearly states that limitation while using the agreed convention for simple historical review.

## Current-cycle data-state behavior

| State | What Canary does |
|---|---|
| Active | Uses observations available on the review date and produces eligible risk, forecast, and action outputs. |
| Incomplete | Clearly marks missing current-day data, retains the latest known observations, continues risk and recovery forecasting when enough history exists, and labels that delayed data was used. |
| Inactive | Shows the building but does not calculate risk or forecasts before placement. |
| Records ended | Within the latest cycle only, shows the last available current-cycle assessment and forecasts using available records. |
| No measured weight | Shows the Day 35 projection as unavailable; the risk view also marks the weight dimension as missing. |
| Stale measured weight | Shows the actual measurement day and staleness; it never presents the carried-forward value as a current measurement. |

## Recorded versus predicted outcomes

- In the latest cycle, Canary shows estimates using only records available by the selected review date.
- In every earlier cycle, Canary shows completed actuals under the documented last-recorded-date convention.
- Historical actual recovery is `ending recorded population ÷ beginning population`.
- Historical actual final average weight comes only from a defensible building-cycle match in `Farm Performance Summary.xlsx`; otherwise the dashboard says **Not available**.
- On Day 35, a recorded building weight is shown as the observed milestone result. Before Day 35, Canary shows a projection.
- Farm Performance Summary supplies historical final average weights but is not the target source for the current-cycle Day 35 projection.

### Day 14 model check

Open **Canary Methodology → 2A · Recovery model** and review **Day 14 prediction versus last-recorded recovery**. Canary recreates one forecast per building history using only information available by Day 14, holds that entire cycle out of training, and compares the estimate with the last-recorded recovery proxy. Most historical outcomes were below 95%, and the model predicted all of them below target; therefore its target-side accuracy does not prove it can recognize a true target hitter. Point-error metrics remain useful but limited.

## Reviewing and approving action rules

Use **Action playbook** to inspect the seven rules. A rule change requires explicit confirmation before it is saved. An approved rule also requires an approval date. The overall playbook is treated as farm-approved only when all rules are approved.

The editable review workbook is `docs/Project_Canary_Preliminary_Action_Playbook.xlsx`. The application reads the version-controlled rules from `config/recommendation_playbook_draft.json`.

## Reproduce the capstone checks

```bash
uv run python -m scripts.validate_capstone
uv run pytest
```

The first command runs five historical walkthroughs covering Day 14, Day 22, staggered building states, Day 48 with a stale weight, and a missing current-day entry. It writes the detailed evidence to `artifacts/capstone_validation.json`.

## Safety boundaries

- Canary supports management attention; it does not diagnose disease.
- Recommendations are not automatic treatment instructions.
- An old weight is never relabeled as a current weight.
- A missing dimension is disclosed and does not add a zero-risk claim.
- Daily upload performs inference only. It never retrains a model.
- Model limitations and rule-approval status must remain visible until their open items are resolved.

## Troubleshooting

- **Workbook rejected:** Review **Data & Settings** for missing sheets, required columns, conflicting duplicate values, or incomplete keys.
- **Day 35 weight is unavailable:** No measured building weight exists by the review date. Record a representative bodyweight sample; Canary does not substitute the same farm average for every building.
- **Recovery forecast unavailable:** Confirm placement and that current or previous flock population/mortality observations exist.
- **Recommendation marked preliminary:** Complete Doc Raymond’s playbook review; this is expected until all rules are approved.
- **Missing historical final weight:** No defensible building-cycle weight match was found in Farm Performance Summary. Canary leaves it unavailable instead of imputing a value.
