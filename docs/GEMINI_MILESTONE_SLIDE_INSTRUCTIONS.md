# Gemini revision prompts — Project Canary milestone update

These prompts refer to **Project Canary - Milestone Update (3)**. Apply them in order. Preserve the existing dark-green/cream visual system, 16:9 layout, large typography, and concise owner-friendly language. Do not invent metrics.

## Prompt 1 — clean the deck and tighten the storyline

> Edit this presentation into a mentor-ready milestone update of approximately 20–24 slides. Keep the strongest current visual style. Remove the divider titled “OLD SLIDES” and every slide after it. Remove repeated milestone-progress slides, repeated data-quality slides, and repeated impact slides. Keep one copy of each idea. Organize the remaining deck into: 1) problem and business question, 2) Milestone 1 — strategy and EDA, 3) Milestone 2 — dashboard and three logic engines, 4) model evidence and limitations, 5) Milestone 3 — pilot roadmap and mentor asks. Add a small footer to evidence slides: “Validated prototype; association is not causation.”

## Prompt 2 — correct the Day 14 evidence slide

> Replace the Day 14 evidence slide with the title “Day 14 is an early-warning checkpoint—not a guarantee.” Use two side-by-side evidence cards. Card 1: “Day 14 weight vs Day 35 weight: n=25, Pearson r=0.495, p=0.0119. After removing cycle averages: r=0.180, p=0.436.” Add: “A 100 g higher Day 14 weight is associated with about 210 g higher Day 35 weight in the pooled sample, but much of the pattern is between cycles.” Card 2: “Day 14 weight vs final recovery proxy: n=25, Pearson r=0.249, p=0.230; within-cycle r=0.287, p=0.207.” Add: “Directionally positive, but not statistically conclusive.” Footer: “Canary therefore uses Day 14 as an actionable checkpoint and continues updating through Day 35 and harvest.” Do not claim that Day 14 weight causes recovery.

## Prompt 3 — correct the data-foundation slides

> Replace the current data-preparation and data-summary slides with two slides. Slide A title: “1,785 source rows became 1,666 unique building-day records.” Show the lineage: raw daily workbook → standardized identifiers and units → Zone A/B environmental rows aggregated → duplicate building-day records consolidated → missing values retained as missing → 1,666 analysis-ready building-days. Slide B title: “The models learn from outcomes, recreated at historical decision points.” Show: recovery = 25 independent completed building-cycle outcomes, 122 balanced checkpoint snapshots, plus 1,122 daily audit snapshots; Day 35 weight = 31 independent Day 35 outcomes and 124 Day 7/14/21/28 checkpoint rows. Explain that repeated snapshots are not independent flocks and are weighted equally by building-cycle. State that future checkpoint weights are hidden and the recovery label is a last-recorded-population proxy, not a verified harvest event.

## Prompt 4 — make the modeling workflow explicit

> Replace the model-workflow slide with a six-step horizontal process: 1) clean and consolidate the corrected farm workbook; 2) define Y and freeze each historical review date; 3) engineer only review-date-safe features; 4) inside each training fold, impute missing values, add missingness flags, scale linear-model inputs, and filter redundant features; 5) outer leave-one-complete-cycle-out validation plus inner whole-cycle hyperparameter tuning; 6) compare five declared methods and apply predeclared champion gates. Add a callout: “No random row-level 80/20 split: every row from the held-out cycle stays out of cleaning, tuning, training, and feature selection.”

## Prompt 5 — replace the recovery-model result

> Create a slide titled “Recovery: useful continuous estimate, weak target classifier.” State Y: final recovery proxy = last-recorded population ÷ beginning population. Use this comparison table: Historical mean — MAE 1.66 pts, cycle-balanced MAE 1.73 pts, RMSE 2.17 pts, R² -0.132; Ordinary linear regression — MAE 1.37 pts, cycle-balanced MAE 1.48 pts, RMSE 1.84 pts, R² 0.189; Ridge — MAE 1.55 pts, cycle-balanced MAE 1.59 pts, RMSE 1.97 pts, R² 0.070; Gradient Boosting — MAE 1.57 pts, cycle-balanced MAE 1.66 pts, RMSE 2.08 pts, R² -0.041; XGBoost — declared challenger, not executed in the local release environment. Highlight OLS as the continuous estimator because it improves cycle-balanced MAE by 14.5% and retains positive held-out R². Add a warning: target-side accuracy 82.8% is below the 84.4% majority baseline and at/above-95% recall is 0%; do not call it a validated 95% hit/miss classifier.

## Prompt 6 — replace the Day 35 weight-model result

> Create a slide titled “Day 35 weight: the transparent baseline remains the honest winner.” State Y: observed average building bodyweight on production Day 35; goal 1,800 g. Use this comparison table: Historical remaining gain — MAE 178 g, cycle-balanced MAE 182 g, RMSE 242 g, R² 0.126, 65.3% within 200 g; Ordinary linear regression — 217 g, 209 g, 273 g, -0.114, 55.6%; Ridge — 197 g, 193 g, 252 g, 0.048, 60.5%; Gradient Boosting — 189 g, 189 g, 256 g, 0.018, 58.1%; XGBoost — declared challenger, not executed in the local release environment. Explain the live formula: latest observed weight + average historical remaining gain from the same checkpoint age, calculated from training cycles only. State: “No learned model achieved the required 10% MAE improvement and 70% within-200 g gate, so the transparent baseline remains operational.”

## Prompt 7 — address low R² directly

> Add one slide titled “Low R² changes the claim—it does not erase all value.” Use a large central statement: “Recovery R² = 0.189; weight R² = 0.126. Most outcome variation remains unexplained.” Below it, show three columns: What is still useful — MAE is in business units, forecasts update with current evidence, rules and actions remain traceable; Why R² is low — only 5–6 historical cycles, 25–31 independent outcomes, proxy recovery labels, inconsistent weight sampling, missing/unrecorded health, feed and management events; Honest positioning — validated prototype, ranges not guarantees, recovery target classification not validated, weight uses a transparent fallback, collect more standardized cycles before production deployment. Do not describe either forecast as highly accurate.

## Prompt 8 — refresh the dashboard example

> Replace the current worked-example numbers and screenshot with a fresh capture from the latest Project Canary app. Show one current-cycle building only. Use this sequence: risk score and four dimension scores → leading recorded trigger → projected recovery and gap to 95% → projected Day 35 weight and gap to 1,800 g → recommended check and urgency → estimated gross revenue at risk under editable assumptions. Label predictions “projected,” observations “recorded,” and completed-cycle outcomes “actual proxy.” Do not hard-code old values such as a 2,000 g target or 1,465 g projection.

## Prompt 9 — correct the business-value claim

> Wherever the deck says “value protected,” replace it with “estimated potential gross-revenue opportunity under assumptions.” For the annual scenario, show the formula explicitly: beginning birds × assumed recovery improvement × sale weight × price per kg × cycles per year. If retaining the approximately ₱2.75M example, label every assumption beside it and add: “Scenario estimate—not measured causal impact, profit, or guaranteed savings.”

## Prompt 10 — update the roadmap and mentor asks

> Replace “backtest next” with a three-stage roadmap. Now: capstone prototype with traceable risk rules, two validated forecast methods, seven-question EDA, and Doc Raymond’s preliminary playbook. Next farm cycle: pilot with consistent Day 7/14/21/28/35 weight sampling, verified harvest event and ending population, confirmed feed units, approved environmental thresholds, and recorded actions/outcomes. Later: retrain only after enough new complete cycles, monitor MAE/RMSE/R² and target-side performance, and promote a learned model only if it clears the predeclared gates. End with five mentor asks: validate thresholds, confirm feed units, approve weighing protocol, define verified harvest labels, and define acceptable forecast error.

## Optional final quality-control prompt

> Audit the entire revised deck for internal consistency. The only approved targets are 95% recovery and 1,800 g on Day 35. Day 14 is the early-warning checkpoint, not the end of monitoring. The current recovery estimator is ordinary linear regression; the current weight method is historical remaining gain. Use only the approved model metrics in the revised result slides. Delete any remaining claims of 15 g error, R² around 0.60, guaranteed/protected revenue, causal interventions, or the old mortality-trend/peer-comparison risk system. Keep the final deck concise and readable from a projector.
