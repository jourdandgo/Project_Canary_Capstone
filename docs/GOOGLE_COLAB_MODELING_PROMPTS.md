# Project Canary — Independent Google Colab / Gemini Model Review

## Recommended upload package

Upload both workbooks to the same Colab session:

1. `FARM HARVEST DATA.xlsx`
   - Treat this as the primary modeling source.
   - It preserves the daily fields that an independent reviewer may want to explore, including population, mortality, feed, observed bodyweight, target gaps, and temperature and humidity summaries.
2. `Project_Canary_Model_Ready_Data.xlsx`
   - Treat this as a reference and audit companion.
   - It documents Project Canary's current outcome definitions, data lineage, historical snapshots, and known leakage boundaries.
   - Gemini is not required to train from its prepared training sheets or reproduce Canary's current feature engineering.

This two-file setup is intentional. The raw workbook gives Gemini enough freedom to propose a genuinely different approach, while the model-ready workbook makes definitions and reconciliations transparent and easier to audit.

## Essential guardrails

These are data-validity requirements rather than instructions about which model to build:

- Daily rows from the same building and cycle are repeated observations of one flock, not independent outcomes.
- A historical forecast may use only information recorded on or before its review date.
- Final outcomes, ending population, future mortality, future weights, and future-informed interpolation cannot be used as earlier predictors.
- Target-weight curves are reference standards, not observed flock weights.
- Zone A/B records must be reconciled to the intended building-day grain before they are joined or summarized.
- Keep 2026-3 untouched during approach design, preprocessing decisions, feature selection, and tuning; use it only as a later-cycle audit after the pipeline is frozen.
- Feed units remain subject to farm confirmation and must not be used without documenting the interpretation.

## Prompt 1 — Harvest recovery

Use the complete prompt in:

`docs/GEMINI_COLAB_PROMPT_RECOVERY.txt`

The brief asks Gemini to independently determine the analytical grain, target formulation, cleaning rules, feature engineering, validation design, algorithms, tuning strategy, and uncertainty method. It supplies the business objective and non-negotiable leakage boundaries without prescribing Canary's current training design.

## Prompt 2 — Day 35 bodyweight

Use the complete prompt in:

`docs/GEMINI_COLAB_PROMPT_DAY35_WEIGHT.txt`

The brief asks Gemini to independently evaluate direct Day 35 prediction, remaining-growth formulations, biological growth curves, tabular approaches, or other defensible alternatives. It does not require Gemini to reproduce Canary's current historical remaining-gain method.

## Expected second-opinion outputs

For either outcome, ask Gemini to return:

- A reproducible notebook beginning with workbook inspection.
- A transparent X/Y and leakage table.
- Relevant EDA and data-quality findings.
- At least five justified approaches, including a simple benchmark.
- Out-of-sample MAE, RMSE, R², bias, stability, and outcome-specific business metrics.
- Actual-versus-predicted and error-by-review-age visuals.
- A separately labelled untouched 2026-3 audit.
- Feature importance and SHAP where technically appropriate.
- Exported champion pipeline, feature list, hyperparameters, predictions, and comparison table.
- An honest conclusion about what the model can and cannot be trusted to do.

The goal is not to force a more impressive score. The goal is to discover whether a different defensible approach performs better on genuinely unseen flocks or cycles.
