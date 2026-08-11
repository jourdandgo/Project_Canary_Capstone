# Project Canary — Product, UX, Analytics, and Model Roadmap

## Executive summary

Project Canary is directionally aligned with the capstone: it gives six-building visibility, a rules-based risk rating and explanation, two outcome outlooks, and deterministic next-action guidance. The build is technically traceable and suitable for a prototype demonstration.

The next iteration should **simplify before expanding**. Three integrity corrections come first:

1. `End Date` is the maximum date in the daily records, not confirmed harvest completion. The app must stop calling it an actual harvest date or automatically treating the flock as harvested solely for that reason.
2. The recovery label must be described as **last-recorded recovery** (last recorded population / beginning population) unless a true harvest-completion source is added or the farm explicitly approves the last record as the capstone harvest proxy.
3. Day 14 target-side accuracy is not evidence that the recovery model distinguishes target hitters. Only 4 of 25 historical building outcomes were at or above 95%; the Day 14 model predicted all 25 below target. Its 84% accuracy equals the always-below majority baseline.

After those corrections, reorganize the app around the farm owner's daily decision, simplify the cards, add a transparent business-value calculator, and present EDA and methodology as separate question-led pages.

## Current fit against the agreed objectives

| Objective | Current status | Assessment |
|---|---|---|
| Show all six buildings every day | Implemented | Strong, with explicit missing-building states. |
| Rules-based risk rating | Implemented | Reproducible and explainable; thresholds remain provisional. |
| Explain why a building was flagged | Implemented | Strong traceability, but too much detail is exposed on the default path. |
| Predict recovery | Prototype | Point-error performance is promising, but the endpoint name and target-miss usefulness need correction. |
| Project Day 35 average weight | Experimental transparent fallback | Building-specific when a measured checkpoint exists; 31 historical Day 35 outcomes across six cycles support the audit. No learned candidate cleared the champion gates, so historical remaining gain remains operational and Ridge is retained only as the best learned challenger. Five outcomes reached 1.8 kg, so target-hit recognition remains limited. |
| Recommend the next action | Implemented as preliminary | Sensible deterministic playbook; farm-owner approval remains required. |
| Continue after Day 14 and Day 35 | Mechanically implemented | Must no longer use maximum daily date as proof of actual harvest completion. |
| Measure business value | Narrative only | Source presentations contain ROI assumptions; the app needs an adjustable, explicitly estimated calculator. |
| Support a defensible capstone | Partially achieved | Strong engineering and traceability; endpoint semantics, evaluation design, and UX density need one more disciplined pass. |

## UX audit: what currently feels overwhelming

The rendered desktop review showed that the six building cards begin below several navigation rows, explanatory tabs, metric definitions, denominator notes, and a long weight-baseline warning. The owner must read too much before reaching the daily decision.

The active building cards then repeat recovery current value, prediction, change, goal gap, interval, weight baseline, missing-weight explanation, goal gap, interval, model scope, and a full action sentence. This is accurate but visually flat and difficult to scan across buildings.

### Simplified information architecture

Use five primary destinations:

1. **Today** — the six-building operating view and business value at risk.
2. **Building** — one-building decision and evidence drill-down.
3. **Insights** — EDA questions and historical findings.
4. **Canary Methodology** — risk score, predictive models, and recommendations.
5. **Settings & Data** — workbook validation, assumptions, business levers, and rule administration.

Move “What Canary is,” the business question, and limitations into a compact About panel or the Methodology page. Do not place four explanatory tabs above the daily building view.

### New default-page hierarchy

1. Compact cycle/date context.
2. Three headline metrics: current flocks, buildings needing attention, estimated revenue at risk.
3. “Review first” action strip.
4. Six building cards above the fold where screen size permits.
5. One short assumptions note below the cards.

### Simplified six-building card

Each active card should show only:

- Building, production day, and risk pill.
- Risk score with a small four-part evidence indicator.
- Predicted recovery with gap to 95%.
- Final-weight output with a clear badge: `Personalized forecast`, `Farm baseline`, or `Unavailable`.
- Estimated gross revenue at risk versus the 95% recovery goal.
- One-line main reason.
- One-line next action and response timing.
- `View details` button.

Move uncertainty ranges, current-to-final calculation details, full model provenance, individual dimension evidence, and inspection checklists into the Building page.

Use muted compact cards for `No cycle record`, `Not yet placed`, and `Records ended`; do not give these the same visual weight as an active decision.

## Business-value calculator

### Owner-controlled assumptions

- Live chicken selling price in PHP per kg.
- Expected sale weight in kg per bird.
- Recovery improvement scenario in percentage points (default 1.0 point).
- Production cycles per year (default 5, editable).

Keep these in a single assumptions panel. Show the active values near every business-value result and provide a reset-to-default button.

### Calculations

For a building:

`Birds represented by 1 percentage point = beginning population × 0.01`

`Gross revenue per recovered bird = assumed sale weight × selling price per kg`

`Value of selected recovery improvement = beginning population × improvement percentage points / 100 × assumed sale weight × selling price per kg`

`Estimated gross revenue at risk = beginning population × max(95% - predicted recovery, 0) × assumed sale weight × selling price per kg`

Annualize only when explicitly requested:

`Estimated annual value = per-cycle value × cycles per year`

### Required wording

Call the result **estimated gross revenue opportunity** or **estimated gross revenue at risk**, never profit or guaranteed savings. It excludes feed, labor, electricity, treatment, mortality timing, price changes, and the cost or effectiveness of an intervention. Recommendations do not claim to cause the modeled recovery improvement.

## Predictive-model simplification and strengthening

### Recovery model

**Current strength:** leakage-safe leave-one-cycle-out validation and an overall point-estimate MAE of about 1.22 percentage points.

**Current weaknesses:**

- The target is recovery on the last recorded daily date, not confirmed harvest recovery.
- There are only five historical cycles and 25 building outcomes.
- Daily snapshots repeat the same final outcome many times and can overweight longer-recorded building-cycles.
- `percentage_alive` and `cumulative_mortality_rate` are mathematically redundant.
- Target-side accuracy is dominated by the below-95% majority class.

**Recommended streamlined comparison:**

1. A historical mean baseline.
2. An age-aware survival-decay baseline: current survival adjusted by the historical remaining mortality from the same age/horizon.
3. One regularized Ridge model with a short, non-redundant feature set.

Evaluate at fixed decision horizons (Days 7, 14, 21, 28, and latest eligible day), with one record per building per horizon. Report macro-average performance so every building-cycle and held-out cycle has balanced influence.

Select the simplest model unless a more complex model shows a meaningful and stable improvement. Random forest can remain a comparison candidate but should not be promoted merely because it is machine learning.

**Metrics to show in business language:**

- MAE: typical size of the recovery estimate error in percentage points.
- RMSE: whether occasional large misses are a concern.
- Bias: whether Canary tends to be optimistic or pessimistic.
- Miss-target recall: how many actual below-95% outcomes Canary warned about.
- Target-hit recall: how many actual at-or-above-95% outcomes Canary correctly recognized.
- Confusion matrix and majority-class baseline.
- Performance at Days 7, 14, 21, and later horizons.

### Day 35 weight model

**Current strength:** the primary target is now the recorded building average weight on Day 35 from Farm Harvest Data, matching the simplified defense storyline.

**Current weakness:** only 31 trusted Day 35 outcomes across six cycles exist, and only five are at or above 1.8 kg. No learned candidate cleared the final champion gates, so historical remaining gain remains the transparent operational fallback and target-hit discrimination remains unvalidated.

**Recommended streamlined comparison:**

1. Historical Day 35 mean.
2. Age-target-ratio projection using the latest measured weight and the farm target curve.
3. Recent linear average-daily-gain projection.
4. Historical remaining-gain projection from the measurement age to Day 35.

Compact Ridge is currently selected by cycle-held-out MAE. Historical remaining gain remains the strongest transparent benchmark and operational fallback. The output must remain labeled a limited-data Day 35 projection, not a guaranteed target-hit probability. Continue using the age-specific weight-gap risk check as the independent operational signal.

### Minimum data improvements

- Record weights using the same method on Days 7, 14, 21, 28, and 35 where practical.
- Record sample size and zone for each weight measurement.
- Add trusted final average weight for every completed building-cycle.
- Add a true harvest/completion status or date if the intended forecast endpoint is actual harvest.
- Preserve last-recorded population as the simple recovery numerator for the capstone, but label the endpoint honestly.

## Insights page: question-led EDA

The EDA page should answer questions, not display a collection of charts. Each question gets: a one-sentence answer, one visual, sample size, and a visible limitation.

### Five required questions

1. **How complete and fresh is the data by cycle, building, day, and variable?**
2. **Is Day 14 weight associated with Day 35 weight and final average weight?**
3. **Is Day 14 weight associated with last-recorded recovery?**
4. **How early do mortality and survival trajectories begin to separate between better and worse outcomes?**
5. **How well do the recovery and weight methods perform at each decision horizon?**

### High-value additions

6. **How much variation comes from the production cycle versus the individual building?**
7. **Do temperature or humidity deviations align with mortality or recovery where environment data exists?**
8. **How often is weight missing or stale, and how does freshness affect risk evidence and forecast error?**
9. **Which buildings and cycles show the largest estimated gross revenue opportunity?**

Use observational language: `associated with`, `aligned with`, or `observed alongside`; do not use causal language such as `caused` or `improved` without a controlled intervention study.

## Canary Methodology page

Start with three large visual steps:

1. **Risk scoring — Where should we inspect?**
2. **Predictive outlooks — What result appears likely?**
3. **Recommendations — What should we check next?**

Each step opens a short, approachable tab:

### Risk scoring

- Four dimensions, each 0–3.
- Total score and label bands.
- One worked example.
- Missing-evidence behavior.
- Clear statement that forecasts do not alter the risk score.

### Predictive outlooks

- What the target label means.
- Which historical cycles/outcomes were eligible.
- A five-step training and held-out-cycle validation flow.
- Winner versus baselines.
- Performance by horizon and a confusion matrix.
- Plain-language limitations and permitted business use.

### Recommendations

- How problem patterns are identified.
- The seven pattern-to-action mappings.
- How risk level changes response timing.
- What guidance is farm-approved versus preliminary.
- Safety boundary: inspection and escalation guidance, not diagnosis or automatic treatment.

## Phased build plan

### Phase 0 — Correct definitions before adding features

**Status: Superseded by the stakeholder-approved simple lifecycle convention.**

- Rename `End Date` to `Latest recorded date` throughout the code and interface.
- Replace `Harvest complete` with `Records ended` unless a true harvest flag exists.
- Rename the recovery endpoint to `last-recorded recovery` or document an explicitly approved capstone proxy.
- Stop replacing predictions with an “actual harvest result” solely because the review date reaches the maximum daily date.
- Correct model-proof claims about target-side performance.
- Update tests, model card, operating guide, open items, and defense wording.

**Updated decision:** only the latest cycle receives risk ratings, predictions, and recommendations. Every earlier cycle is displayed as completed at each building's maximum daily-record date and shows actual recovery plus final average weight when available. The dashboard discloses that this date is a capstone convention rather than a verified source event.

### Phase 1 — Simplify the daily owner experience

**Status: Complete in the Phase B build.**

- Implement the six-section navigation.
- Put six cards and priorities above explanatory material.
- Redesign active and inactive cards using progressive disclosure.
- Move detailed comparison rows and intervals into Building tabs.
- Add plain labels and consistent color/icon semantics.

**Gate:** a first-time, non-technical reviewer can name the priority building, reason, forecast gaps, estimated value at risk, and next action within 30 seconds.

### Phase 2 — Rebuild evaluation before retraining

**Status: Complete in the Phase B build.**

- Create fixed-horizon, one-building-one-row evaluation snapshots.
- Add cycle-macro MAE, bias, confusion matrix, miss-target recall, and majority baseline.
- Remove redundant recovery features.
- Recompute distinct-outcome weight baselines.
- Compare only the short candidate list and use a simple-winner rule.

**Gate:** every headline model claim beats the appropriate naïve baseline or is explicitly labeled baseline-only.

### Phase 3 — Add business value and question-led EDA

**Status: Complete for the five required EDA questions; four additional questions remain optional extensions.**

- Implement editable value assumptions and formulas.
- Add estimated gross revenue at risk to active building cards.
- Build the nine-question Insights page, with the first five required.
- Add assumptions and non-causal interpretation notes near the values.

**Gate:** every peso result can be reproduced from visible inputs and is never presented as profit or guaranteed uplift.

### Phase 4 — Build the presentation-ready Methodology experience

**Status: Complete in the Phase B build.**

- Add the three-component Canary Methodology page.
- Turn Model Proof into the predictive-model subsection rather than a separate dense destination.
- Add a 60-second summary, worked example, validation flow, and simple metric glossary.
- Keep technical tables behind expanders.

**Gate:** the capstone team can explain the complete approach in three minutes and answer technical follow-ups from the same page.

### Phase 5 — Apply farm validation and rehearse

- Apply Doc Raymond's approved thresholds, playbook wording, owners, and escalation timing.
- Update final-weight data and endpoint definitions if new records arrive.
- Run desktop/narrow-width UX checks, accessibility checks, data checks, model tests, and historical acceptance scenarios.
- Rehearse the defense using one early, one late, one missing-data, and one records-ended scenario.

**Gate:** all approved decisions are versioned, every remaining caveat is visible, and the demo contains no unsupported “actual harvest” or causal claim.

## Recommended next action

Proceed to **Phase 5: farm validation and defense rehearsal**. The application structure, balanced model comparison, business-value estimator, five required EDA questions, and presentation-ready methodology are implemented and tested. The remaining work is stakeholder approval, final data clarification, and defense rehearsal—not another major feature build.
