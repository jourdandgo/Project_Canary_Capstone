# Project Canary Defense Demo Runbook

## One-minute story

Project Canary is an early-warning and decision-support system for broiler farms. It brings current records into one building-level view so management can see a developing concern earlier, understand the recorded evidence, review two outcome outlooks, and document the next inspection step. It does not diagnose disease, prescribe treatment, or promise a production improvement.

## Recommended live route

1. Open **Home** and select cycle 2026-3.
2. Choose a review date such as Day 14 or upload the source-backed Day 14 replay CSV.
3. Point out that all six physical buildings remain visible; only Tags 1–3 are present in 2026-3.
4. Open one Tags building through **See how this building's predictions were made**.
5. In **Building View**, show:
   - **View risk calculation · input → rule → score → label**;
   - **View calculation · Model 1 · End-of-cycle recovery proxy**;
   - **View calculation · Model 3 · Day 35 bodyweight**;
   - the preliminary inspection and documentation guidance.
6. Open **Defense tools → Model Evidence Explorer** and replay one held-out row.
7. Open **Defense tools → How Canary Works** and close with the separation between observed risk, the two forecast models, and inspection guidance.

## What to say about the two models

### Model 1

> Model 1 is an Extra Trees regression model with 85 engineered inputs. It estimates the end-of-cycle recovery proxy, defined as last recorded population divided by beginning population. Because its source rows are daily, Canary can refresh it daily through Day 14 and holds the Day 14 result afterward. Its overall held-out MAE was 1.58 percentage points. The output is an outlook, not verified harvest recovery.

> Canary also watches for persistent recorded conditions. Temperature and humidity require three consecutive daily readings outside the flock-age range in the same direction. Because bodyweight is generally measured weekly, a weight watch means the latest checkpoint deficit has remained unresolved for at least three review days; it does not claim three separate low-weight readings. Open “View 3-day signal calculation” to show every source value, range comparison, rule, and the explicit statement that the watch adds no risk points.

> Canary shows one overall operational-priority label so management has one urgency decision. It can show several detected problem patterns underneath that label. Each pattern is matched to its own preliminary inspection guide, while the highest-scoring safety-prioritized concern remains the primary action. Open “View problem-pattern criteria” and “See why this action was selected” to trace every match.

### Model 3

> Model 3 is a CatBoost regression model with 85 engineered inputs. It estimates average Day 35 bodyweight. Canary shows it at Days 7, 14, and 21, then holds the latest outlook because bodyweight is measured mainly weekly. Its overall held-out MAE was 122.1 grams. At the official checkpoints, MAE was 122.6 grams at Day 7, 105.7 grams at Day 14, and 105.7 grams at Day 21. We do not claim a Day 28 model refresh.

## Why forecasts do not affect risk points

> Risk points summarize conditions already observed in the building. Forecasts estimate uncertain future outcomes. Keeping them separate prevents a forecast error from silently adding or removing an alert, and it allows management to audit both independently.

## How to explain the validation

> The displayed replay values are saved leave-one-building-flock-out predictions. The building-flock being evaluated was not used to fit its fold's model, including all of its repeated daily rows. Other buildings from the same production cycle may still be in training, so we do not call this leave-one-complete-cycle-out validation. The sample has 34 building-flocks, which is why both models remain pilot-stage.

## Important boundaries

- The error band is the point estimate plus or minus the 80th percentile of held-out absolute error. It is not a formal confidence interval.
- LOFO feature results describe predictive association, not causation.
- The current package replays saved model-ready rows. It still needs Trish's full 85-feature transformer before it can score any arbitrary future raw CSV.
- Risk thresholds and inspection guidance remain proposed for farm review.
- A recorded Day 35 measurement replaces the weight forecast.

## If asked why not predict bodyweight daily

> The records contain daily mortality and population but mainly weekly bodyweight measurements. Changing a bodyweight projection every day without a new weighing would imply precision the data do not contain. Canary therefore refreshes the weight outlook only at validated measurement checkpoints and clearly labels held values between them.
