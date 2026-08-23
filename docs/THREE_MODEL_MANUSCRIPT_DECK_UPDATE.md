# Copy-Ready Manuscript and Deck Alignment — Three Models

This file prepares changes for the live manuscript and deck. It does not edit the Google assets.

## Shared project wording

> Project Canary provides two planning outlooks through three evaluated model engines. Model 1 estimates the end-of-cycle recovery proxy at Day 7 and Day 14. For Day 35 bodyweight, the app compares a dedicated Day 21 Model 3 XGBoost outlook with a checkpoint method that refreshes after actual Day 7, 14, 21, and 28 weigh-ins. The selected bodyweight engine is always labelled. Forecasts remain separate from Canary's observed-condition risk score.

## Model result wording

> Model 1 retained the original Extra Trees approach and 85 engineered inputs. Under corrected complete-cycle validation, it produced a pooled mean absolute error of 2.47 percentage points, compared with 2.43 points for its transparent baseline. It therefore remains experimental rather than being presented as the recovery champion.
>
> Model 3 retained the original Day 21 XGBoost approach and 88 engineered inputs. Its Day 21 mean absolute error was approximately 146 g. The checkpoint historical-remaining-gain method produced outlooks at Days 7, 14, 21, and 28, with errors of approximately 155 g, 138 g, 112 g, and 103 g, respectively. At the common Day 21 evidence point, the checkpoint method reduced error by approximately 34 g. It is therefore the provisional default bodyweight outlook, while Model 3 remains visible as a shadow benchmark.

## Deck changes

- Title the model slide **Two Outcomes, Three Model Engines**.
- Show Model 1 → recovery proxy; Model 3 → Day 35 bodyweight at Day 21; checkpoint method → Day 35 bodyweight at Days 7, 14, 21, and 28.
- Show the bodyweight toggle as an evaluation control, not as two simultaneous owner recommendations.
- Show 146 g versus 112 g at the common Day 21 evidence point.
- Label Model 1 experimental because it did not beat the recovery baseline.
- Keep THI and feature importance as predictive-association evidence, not causal proof.
- State that Models 1 and 3 currently support the prepared 2026-3 audit replay; generic future-cycle scoring requires packaging their raw-feature transformer.

For the simplest complete explanation and speaker notes, use [THREE_MODEL_SIMPLE_EXPLAINER.md](THREE_MODEL_SIMPLE_EXPLAINER.md).
