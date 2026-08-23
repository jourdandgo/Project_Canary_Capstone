# Archived Trish v18 Integration — Project Canary

> Historical audit document only. These specialist routes are no longer loaded by the owner-facing application. The current release uses `config/forecast_registry.json` and is explained in `HOW_PROJECT_CANARY_FORECASTS_WORK.md`.

## Deployment decision

Project Canary uses Trish's six v18 champion definitions. The dashboard keeps the risk score independent and uses the models only as supporting outlooks.

The supplied final pickles were trained on all 34 building-cycles, including 2026-3. To avoid presenting in-sample results as a live forecast, the deployable bundle refits the same champion algorithms and exact saved feature lists on cycles 2025-2 through 2026-2. Cycle 2026-3 is used only for inference.

| Model | Outcome | Champion | Window | Dashboard role |
|---|---|---|---|---|
| 1 | Final harvest recovery | Extra Trees | Days 1–14 | Primary recovery outlook |
| 2 | Day 35 bodyweight | CatBoost | Days 1–14 | Early weight outlook |
| 3 | Day 35 bodyweight | XGBoost | Days 1–21 | Later weight update |
| 4 | Estimated age to 1.8 kg | Gradient Boosting | Days 1–14 | Secondary harvest planning |
| 5 | Estimated age to 2.0 kg | CatBoost | Days 1–14 | Secondary harvest planning |
| 6 | Recovery at the sale-ready milestone | CatBoost | Days 1–14 | Secondary harvest planning |

Model 2 is used through Day 14. Model 3 replaces it from Day 15 onward and is held at its Day 21 update after the early window. Models 1 and 4–6 are held at Day 14 after their early windows. A recorded Day 35 bodyweight replaces the weight forecast once it exists.

## Safeguards

- Model 1 recovery cannot exceed currently recorded survival.
- Model 5's 2.0 kg timing cannot appear earlier than Model 4's 1.8 kg timing.
- Model 6 recovery cannot exceed currently recorded survival.
- Forecasts never change the rules-based 0–12 risk score.
- Local SHAP explanations are shown as associations, not causes.
- Models 4–6 remain secondary because their timing targets are partly curve-derived.

## Current limitation

The v18 folder contains the full reproducible training pipeline but no standalone scorer for an arbitrary newly uploaded workbook. The packaged bundle supports the authoritative 2026-3 workbook and its as-of replay dates. A future data refresh requires rerunning the v18 feature pipeline and rebuilding the bundle with:

```bash
python -m scripts.build_trish_v18_bundle --source ../capstone_FINAL_v18
```

The app retains its earlier transparent forecast as a labeled fallback when the selected cycle is not covered by the v18 bundle. This fallback must not be described as a Trish model result.
