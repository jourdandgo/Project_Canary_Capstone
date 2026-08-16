# Project Canary — New Chat Handoff

## Start here

Canonical working folder:

`/Users/jourdan.go/Downloads/PROJECT CANARY/canary_app/`

GitHub-ready copy:

`/Users/jourdan.go/Downloads/PROJECT CANARY/Project_Canary_GitHub_Ready/`

Use the files in those folders. Do not resume work from an older temporary Codex mirror.

## Current business narrative

Project Canary is an early-warning and decision-support prototype for more consistent broiler production. It identifies off-track buildings, explains recorded warning signs, projects the 95% harvest-recovery proxy and the 1,800 g Day 35 weight milestone, and maps observed triggers to Doc Raymond's approved checks. Day 14 is the primary early-warning checkpoint; monitoring continues throughout the cycle.

## Latest source-data refresh

- `FARM HARVEST DATA.xlsx` contains corrected 2026-2 Lagundi mortality/population records after July 9.
- 2026-2 is now eligible for recovery training.
- 2026-3 contains observed Day 35 weights for Tags 1–3.
- `Weights Cleaned.xlsx` and the farm target sheet use the revised in-house curve ending at 1,800 g on Day 35.
- Newest cleaned daily table: 1,624 unique building-day rows; unsupported forward-filled Days 36–49 were removed from 2026-3.

## Modeling datasets

- Recovery: 31 independent outcomes across six cycles; 151 balanced checkpoint/latest snapshots; 1,355 daily audit snapshots.
- Day 35 weight training: 31 independent historical outcomes across six cycles; 124 Day 7/14/21/28 checkpoint rows.
- Latest-cycle weight audit: 2026-3 Tags 1–3; 12 checkpoint rows; excluded from fitting and selection.

The model-ready workbook is `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx` with matching CSV exports.

## Non-negotiable validation rules

- Hold out entire harvest cycles, never random daily rows.
- Fit imputation, scaling, feature selection and tuning inside each training fold.
- Repeated snapshots receive equal building-cycle weighting.
- No future weights, ending population, recovery label or later records may enter an earlier snapshot.
- MAE/cycle-balanced MAE is primary. RMSE, R², bias, stability and target-side metrics are guardrails.
- No SMOTE or synthetic outcomes.
- Model importance is association, not causation.

## Three business engines

1. Rules-based risk: weight gap, survival path, daily mortality/population loss and environmental condition/freshness. Deterministic and inspectable.
2. Forecasting: predict future loss for recovery and remaining gain for Day 35 weight. Always show uncertainty and limitations.
3. Recommendations: observed triggers map to Doc Raymond's playbook. Forecast importance does not automatically prescribe an intervention.

## Key deliverables

- App: `app.py`
- Recovery notebook: `notebooks/Project_Canary_Recovery_Model.ipynb`
- Weight notebook: `notebooks/Project_Canary_Day35_Weight_Model.ipynb`
- Model manifests/artifacts: `models/`
- Defense guide: `docs/PROJECT_CANARY_DEFENSE_CHEAT_SHEET.md` plus DOCX/PDF
- Colab prompts: `docs/GOOGLE_COLAB_MODELING_PROMPTS.md`
- Model-ready workbook/CSVs: `outputs/model_ready/`

## Outstanding farm decisions

1. Acceptable forecast errors for management use.
2. Feed-intake units.
3. Verified harvest date and ending-population source.
4. Standard bodyweight sampling method.
5. Final environmental thresholds and interventions.

## First action in the new chat

Read this handoff, inspect the current manifests and test results, then continue only from the canonical folder. Re-run leakage, model, app and release-sync checks after any modeling change.
