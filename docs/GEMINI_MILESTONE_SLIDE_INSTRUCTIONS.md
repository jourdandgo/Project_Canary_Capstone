# Gemini revision prompts — Project Canary milestone update

These prompts refer to **Project Canary - Milestone Update (3)**. Apply them in order. Preserve the existing dark-green/cream visual system, 16:9 layout, large typography, and concise owner-friendly language. Do not invent metrics.

## Prompt 1 — clean the deck and tighten the storyline

> Edit this presentation into a mentor-ready milestone update of approximately 20–24 slides. Keep the strongest current visual style. Open with this storyline: “The business problem is inconsistent production outcomes: recovery can fall below the 95% goal and birds do not consistently reach the 1,800 g Day 35 weight milestone. Canary’s value is earlier visibility of off-track buildings, enabling earlier investigation and action.” Make clear that Canary supports management decisions rather than directly controlling mortality or growth. Remove the divider titled “OLD SLIDES” and every slide after it. Remove repeated milestone-progress slides, repeated data-quality slides, and repeated impact slides. Keep one copy of each idea. Organize the remaining deck into: 1) problem and business question, 2) Milestone 1 — strategy and EDA, 3) Milestone 2 — dashboard and three logic engines, 4) model evidence and limitations, 5) Milestone 3 — pilot roadmap and mentor asks. Add a small footer to evidence slides: “Validated prototype; association is not causation.”

## Prompt 2 — correct the Day 14 evidence slide

> Replace the Day 14 evidence slide with the title “Day 14 is an early-warning checkpoint—not a guarantee.” Use two side-by-side evidence cards. Card 1: “Day 14 weight vs Day 35 weight: n=31, Pearson r=0.601, p=0.00035. After removing cycle averages: r=0.298, p=0.14.” Add: “The pooled association is meaningful, but the weaker within-cycle result shows that cycle-wide conditions explain part of it.” Card 2: “Day 14 weight vs final recovery proxy: n=31, Pearson r=0.472, p=0.0074; within-cycle r=0.378, p=0.057.” Add: “Directionally useful, but observational and not proof that higher Day 14 weight causes recovery.” Footer: “Canary therefore uses Day 14 as an actionable checkpoint and continues updating through Day 35 and harvest.”

## Prompt 3 — correct the data-foundation slides

> Replace the current data-preparation and data-summary slides with two slides. Slide A title: “The refreshed workbook contains 1,666 unique building-day records.” Explain that the earlier 1,785-row source had 119 Zone A/B duplicates that were aggregated before the refreshed workbook was issued. Show the lineage: refreshed daily workbook → standardized identifiers and units → duplicate-key verification → missing values retained as missing → 1,666 analysis-ready building-days. Slide B title: “The models learn from outcomes recreated at historical decision points.” Show: recovery = 31 independent completed building-cycle outcomes, 151 balanced training snapshots, plus 1,355 daily audit snapshots; Day 35 weight = 31 independent Day 35 outcomes and 124 Day 7/14/21/28 checkpoint rows; 2026-3 = three genuinely later weight outcomes kept for prospective audit only. Explain that repeated snapshots are not independent flocks and are weighted equally by building-cycle. State that future checkpoint weights are hidden and the recovery label is a last-recorded-population proxy, not a verified harvest event.

## Prompt 4 — make the modeling workflow explicit

> Replace the model-workflow slide with a six-step horizontal process: 1) clean and consolidate the corrected farm workbook; 2) define Y and freeze each historical review date; 3) engineer only review-date-safe features; 4) inside each training fold, impute missing values, add missingness flags, scale linear-model inputs, and filter redundant features; 5) outer leave-one-complete-cycle-out validation plus inner whole-cycle hyperparameter tuning; 6) compare five declared methods and apply predeclared champion gates. Add a callout: “No random row-level 80/20 split: every row from the held-out cycle stays out of cleaning, tuning, training, and feature selection.”

## Prompt 5 — replace the recovery-model result

> Create a slide titled “Recovery: predict remaining loss, then apply strict deployment gates.” State training Y: additional population loss after the review date; final forecast = current survival − predicted additional loss. Use this table: refreshed age-band baseline — MAE 2.00 pts, cycle MAE 2.09, RMSE 2.66, R² -0.010; ordinary linear — 1.74, 1.76, 2.57, 0.054; Ridge — 1.74, 1.76, 2.57, 0.055; Gradient Boosting — 1.76, 1.84, 2.47, 0.129; constrained Extra Trees — 1.73, 1.80, 2.47, 0.129. Highlight ordinary linear regression as the selected live continuous-estimate model because it improves cycle-balanced MAE by 16.1% over the baseline, is effectively tied with Ridge, and is easier to explain. State clearly that at/above-95% recall is only 21.1%, so the output remains an experimental estimate and range—not a probability of hitting target.

## Prompt 5A — add held-out SHAP proof

> Add a technical model-proof slide titled “What drives the recovery estimate?” First show standardized coefficients and held-out permutation importance for the selected ordinary-linear model. Then show a separate horizontal SHAP chart for the constrained Extra Trees nonlinear challenger. Explain that every explanation is generated only after removing the complete test cycle. Add two guardrails: “Importance is predictive association, not causal proof” and “Management actions still require a recorded threshold violation and Doc Raymond’s approved playbook.” Do not describe Extra Trees as the live champion.

## Prompt 6 — replace the Day 35 weight-model result

> Create a slide titled “Day 35 weight: the transparent baseline remains the honest winner.” State training Y: remaining gain from the checkpoint to Day 35; final forecast = latest measured weight + predicted remaining gain; goal 1,800 g. Use this table: historical remaining gain — MAE 178 g, cycle MAE 182 g, RMSE 242 g, R² 0.126, 65.3% within 200 g; checkpoint linear — 216, 208, 273, -0.118, 56.5%; Ridge — 207, 200, 264, -0.045, 56.5%; robust Huber — 226, 223, 276, -0.144, 51.6%; Gradient Boosting — 203, 207, 262, -0.026, 52.4%. State: “All learned challengers are worse on unseen cycles, so the transparent baseline remains operational.”

## Prompt 7 — address low R² directly

> Add one slide titled “R² changes the claim—it does not replace business error.” Use a large central statement: “Recovery R² = 0.054; weight R² = 0.126. Most variation across unseen cycles remains unexplained.” Below it, show three columns: What is useful — recovery MAE is 1.74 points, forecasts update with current evidence, and rules/actions remain traceable; Why uncertainty remains — only six historical cycles and 31 independent outcomes per target, proxy recovery labels, inconsistent weight sampling, missing health/feed/intervention data; Honest positioning — validated experimental estimates, ranges not guarantees, transparent weight fallback remains operational, collect more standardized cycles before production deployment.

## Prompt 8 — refresh the dashboard example

> Replace the current worked-example numbers and screenshot with a fresh capture from the latest Project Canary app. Show one current-cycle building only. Use this sequence: risk score and four dimension scores → leading recorded trigger → projected recovery and gap to 95% → projected Day 35 weight and gap to 1,800 g → recommended check and urgency → estimated gross revenue at risk under editable assumptions. Label predictions “projected,” observations “recorded,” and completed-cycle outcomes “actual proxy.” Do not hard-code old values such as a 2,000 g target or 1,465 g projection.

## Prompt 9 — correct the business-value claim

> Wherever the deck says “value protected,” replace it with “estimated potential gross-revenue opportunity under assumptions.” For the annual scenario, show the formula explicitly: beginning birds × assumed recovery improvement × sale weight × price per kg × cycles per year. If retaining the approximately ₱2.75M example, label every assumption beside it and add: “Scenario estimate—not measured causal impact, profit, or guaranteed savings.”

## Prompt 10 — update the roadmap and mentor asks

> Replace “backtest next” with a three-stage roadmap. Now: capstone prototype with traceable risk rules, two validated forecast methods, seven-question EDA, and Doc Raymond’s preliminary playbook. Next farm cycle: pilot with consistent Day 7/14/21/28/35 weight sampling, verified harvest event and ending population, confirmed feed units, approved environmental thresholds, and recorded actions/outcomes. Later: retrain only after enough new complete cycles, monitor MAE/RMSE/R² and target-side performance, and promote a learned model only if it clears the predeclared gates. End with five mentor asks: validate thresholds, confirm feed units, approve weighing protocol, define verified harvest labels, and define acceptable forecast error.

## Optional final quality-control prompt

> Audit the entire revised deck for internal consistency. The only approved targets are 95% recovery and 1,800 g on Day 35. Day 14 is the early-warning checkpoint, not the end of monitoring. The current recovery method is ordinary linear regression predicting remaining loss; the current weight method is historical remaining gain. Use only strict whole-cycle metrics from the current model manifests. Delete claims of 15 g error, recovery R² around 0.60, guaranteed revenue, causal interventions, or the old mortality-trend/peer-comparison risk system. Keep the final deck concise and readable from a projector.
