# Project Canary — Compact Standalone Handoff

**Snapshot date:** 20 August 2026  
**Full canonical handoff:** `../canary_app/docs/` when the parent workspace is available.

## Mission and boundaries

Project Canary is a capstone prototype for JJ Agriventures. It gives Doc Raymond a building-level view of observed risk, planning outlooks, and suggested inspections while there is still time to investigate. It is not a disease diagnosis, treatment recommender, automated-control system, causal proof, or guaranteed-outcome predictor.

The app, capstone manuscript, and defense deck must all tell one story:

**early visibility → earlier investigation and action → repeatable management learning.**

## Three deliverables

1. **Canary app:** owner-facing Streamlit prototype in this folder.
2. **Manuscript:** formal capstone argument; reference: `https://docs.google.com/document/d/1lpUbkdyBZWesKXWu-HFTONyDrFWy-S6IDp8-6wlyRqI/edit`
3. **Defense deck:** clear, evidence-led, human-made presentation; reference: `https://docs.google.com/presentation/d/1CsmO_48kxOPwidWNanOugqh88fVBXxFGpAbSVADB9OA/edit`

Do not edit Google assets unless explicitly asked. Operational/model facts override narrative claims.

## Source hierarchy

1. Operational actuals: parent `FARM HARVEST DATA.xlsx`, or this folder’s bundled app data.
2. Forecast evidence: `models/three_model/legacy/` for reconstructed Models 1 and 3, and `models/three_model/checkpoint_champion/` for the checkpoint bodyweight method.
3. Observed-risk authority: `config/risk_rules.json` and `docs/RISK_SYSTEM_GOVERNANCE.md`.
4. Recommendation rules: deterministic inspection prompts only; never treatment directions.

## Three engines — keep separate

| Engine | Function | Rule |
|---|---|---|
| Observed risk | Ranks buildings for inspection using recorded conditions | Never uses forecasts |
| Predictive outlooks | Two planning outcomes supported by three model engines | Show the selected method, evidence date, calculation or feature trace, uncertainty, status, and limitation |
| Recommendation playbook | Maps recorded patterns to a suggested inspection | Human can accept, modify, defer, or override; preserve history |

## Two outcomes, three model engines

- **Model 1 · Extra Trees:** recovery-proxy outlook at Day 7 and Day 14; 2.47 percentage-point pooled MAE; experimental because it did not beat its baseline.
- **Model 3 · XGBoost:** Day 35 bodyweight at Day 21; 146 g MAE; shadow benchmark.
- **Checkpoint historical remaining gain:** Day 35 bodyweight at Days 7, 14, 21 and 28; 127 g pooled MAE and 112 g at Day 21; default bodyweight option.

The app toggle explicitly selects Model 3 or the checkpoint bodyweight method. It never silently substitutes one for the other. Models 1 and 3 currently score the prepared 2026-3 audit replay; generic future-cycle inference still requires their original raw-feature transformer. THI remains outside operating rules until its formula and age-specific action bands are approved.

## Current observed-risk governance

Current version: `risk-rules-0.5.0-banded-hybrid`, proposed for farm shadow-pilot validation, not routine adoption.

Four observed 0–3 point checks: weight gap against age target; cumulative population loss; latest daily mortality; and the worse of temperature/humidity deviation from an age band. Base labels: Low 0–2, Medium 3–5, High 6–8, Critical 9–12. Acute survivability and multi-domain safeguards can elevate the base label; sparse evidence displays as insufficient rather than silently scoring zero. Exact thresholds, evidence rules, and provenance: `docs/RISK_SYSTEM_GOVERNANCE.md` and `config/risk_rules.json`.

## Current app state

- Three-model provenance, a bodyweight model toggle, calculation/feature traceability, checkpoint accuracy evidence, 2026-3 replay checkpoints, reset, risk traceability, management-decision history, compact Home, and About Canary are implemented.
- 2026-3 is a prospective replay/audit cycle: Tags 1–3 only. Never invent Lags data.
- Completed cycles show actual outcomes, not active risk cards.
- Arbitrary raw future-cycle model scoring is not yet a validated claim; known 2026-3 replay is source-backed.
- Risk thresholds need Doc Raymond’s shadow-pilot approval.
- Deck needs continued human-design polish and source verification; manuscript needs final model/risk alignment.

## Panel feedback still governing the work

1. Deck should feel team-designed, not AI-generated: use real visuals/data, concise language, varied layouts, and no invented charts.
2. Risk system must visibly defend dimensions, thresholds, labels, safeguards, missingness, provenance, and worked examples.
3. Every displayed score, prediction, pattern, and recommendation must be traceable.
4. Human judgment must be clear: decision/override with rationale, status, follow-up, original evidence preserved.
5. Model performance/SHAP must be proportionate and honest about small-sample/prospective limits.

## Session discipline

1. Read this file and task-specific source documents before work.
2. Test the canonical `../canary_app/` first whenever it exists; mirror targeted changes here afterward.
3. After a material change, update `../canary_app/docs/CURRENT_STATE.md` and this file if its content is affected. Record changed files, evidence impacted, tests, sync state, and remaining issue.
4. Never rely on chat history as project memory.
