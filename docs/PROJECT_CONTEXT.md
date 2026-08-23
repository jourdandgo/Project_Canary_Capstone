# Project Canary — Shared Project Context

**Purpose:** durable handoff for Codex, Claude Code, and human teammates. Read with `../../CLAUDE.md` and `CURRENT_STATE.md`.

## 1. The project in one paragraph

JJ Agriventures operates a multi-building broiler farm. Performance varies across cycles and buildings; management records exist but are difficult to consolidate into a timely building-level view. Project Canary helps Doc Raymond identify which building deserves attention, understand the recorded evidence, see planning outlooks, and choose what to inspect next while time remains. The strategic proposition is not “perfect prediction”; it is **earlier visibility, disciplined investigation, and repeatable organizational learning**.

## 2. Product boundary and owner workflow

Canary's owner workflow is:

`Upload/validate records → observe current building conditions → rank review needs → inspect → record decision/override → follow up → learn.`

The Home page must remain operational and compact. It should answer:

1. Which building needs attention now?
2. What observed evidence caused the rating?
3. What outcomes are currently plausible?
4. What should management inspect next?

The app must never diagnose disease, prescribe treatment, automatically act on equipment, substitute veterinary/farm judgment, or guarantee recovery/weight improvement.

## 3. Deliverables and links

| Deliverable | Current role | Reference |
|---|---|---|
| Canary app | Demoable owner-facing prototype | `../app.py` |
| Manuscript | Formal capstone report; Chapters I–VI must align | [Google Doc](https://docs.google.com/document/d/1lpUbkdyBZWesKXWu-HFTONyDrFWy-S6IDp8-6wlyRqI/edit) |
| Defense deck | Mock/final defense presentation | [Working Google Slides deck](https://docs.google.com/presentation/d/1CsmO_48kxOPwidWNanOugqh88fVBXxFGpAbSVADB9OA/edit) |

Treat Google assets as collaboration references. Do not edit them unless the user explicitly asks. Before changing any statement in them, validate operational/model numbers against the sources below.

## 4. Canonical folders

```text
PROJECT CANARY/
├── FARM HARVEST DATA.xlsx               # operational-data authority
├── canary v19/                          # Trish's final modeling handoff
├── canary_app/                          # canonical active Streamlit app
│   ├── app.py
│   ├── canary/                          # business logic
│   ├── config/risk_rules.json           # observed-risk authority
│   ├── models/trish_v19/manifest.json   # deployed v19-model authority
│   ├── demo_data/2026-3/                # checkpoint replay CSVs
│   ├── docs/                            # handoffs/governance/runbooks
│   └── tests/
└── Project_Canary_GitHub_Ready/          # mirrored distribution copy
```

Important specialist documents:

- `TRISH_V18_INTEGRATION.md` — integration boundary and routing.
- `RISK_SYSTEM_GOVERNANCE.md` — risk rationale, safeguards, and approval needs.
- `Project_Canary_Defense_Demo_Runbook.md` — live demo route.
- `POST_MOCK_DEFENSE_DECK_ADVISORY.md` — deck feedback/history.

## 5. Data and metrics that must not be conflated

- **Day 35 bodyweight** is a management milestone. It is not necessarily sale weight.
- **Harvest recovery** is the project’s ending-population/last-recorded-population proxy unless confirmed harvest reconciliation is available. Do not call it a confirmed sale-count recovery without evidence.
- **2026-3** is a prospective replay / audit cycle. The demo has Tags 1–3 only. Never fabricate current-cycle Lags records.
- Complete outcomes and bodyweight-label availability differ; always state the denominator/sample size that applies to a claim.
- The Farm Harvest Data workbook is the factual source. Trish’s input tables are feature-engineered transformations for modeling, not replacements for raw operational facts.

## 6. Three-engine architecture

| Engine | Answers | Inputs | Boundary |
|---|---|---|---|
| Observed-risk engine | Where should management look first, and why? | Recorded growth, population, mortality, environmental conditions | Deterministic, explainable, independent of forecasts |
| Predictive-outlook engine | What outcome is currently plausible? | Trish v19 saved held-out rows routed to Model 1 and Model 3 | Planning reference; show model, evidence cutoff, and held-out error |
| Recommendation playbook | What should management inspect next? | Recorded patterns/rules | Deterministic inspection guidance, never diagnosis or treatment |

## 7. Trish v19 models: authoritative routing

| ID | Outcome | Algorithm | Evidence window | App role |
|---|---|---|---|---|
| M1 | End-of-cycle recovery proxy | Extra Trees | Daily through Day 14 | Primary recovery-proxy outlook; held afterward |
| M3 | Day 35 bodyweight | CatBoost | Days 7, 14, and 21 | Primary bodyweight outlook; held between and afterward |

Operational routing:

- Days 1–6: M1 may refresh; M3 is unavailable before its first validated bodyweight checkpoint.
- Days 7–14: M1 refreshes daily; M3 refreshes on Days 7 and 14 and is held between them.
- Days 15–21: M1 holds its Day 14 outlook; M3 holds Day 14 then refreshes on Day 21.
- Days 22–34: observed-risk score continues to update; both model outlooks are held and labelled with their evidence day. Day 28 weight may affect observed risk but does not create a model refresh.
- Day 35: recorded Day 35 bodyweight replaces the weight forecast; recovery is still an outlook until ending population is confirmed.

The app displays saved leave-one-building-flock-out predictions from Trish's final v19 handoff. The final artifacts were fitted on 34 building-flocks. Arbitrary new-cycle CSV inference requires the upstream 85-feature transformer to be packaged and validated; do not overstate this capability.

## 8. Observed-risk system

Current authority: `config/risk_rules.json`, version `risk-rules-0.5.0-banded-hybrid`, **proposed for farm validation—not approved for routine use**.

Four 0–3 point dimensions:

1. Weight gap versus the age-specific target.
2. Cumulative population loss.
3. Latest daily mortality.
4. Environmental deviation: the worse of temperature or humidity versus an age-specific reference band, to avoid double counting.

Base labels: Low 0–2; Medium 3–5; High 6–8; Critical 9–12. Acute daily mortality, acute population loss, or two distinct severe domains can elevate the label under documented safeguards. With fewer than three scored dimensions, display **Insufficient evidence** unless an acute survivability override applies. Show raw value, target/band, calculation, points, rule ID/version, evidence freshness, total, label, and override rationale on the Building View.

Threshold provenance: farm-validation candidate bands for growth/population/mortality; supplied tropical age reference bands for temperature/humidity; severity distances and overrides remain shadow-pilot proposals pending Doc Raymond approval.

## 9. Panel and mentor feedback to address

1. **Deck looked AI-generated.** Use fewer generic claims, more source-backed visuals, varied but simple compositions, concise human wording, and real charts/tables. Do not use decorative fake data or repetitive large-card layouts.
2. **Risk score must be defensible.** Explain why four dimensions, why those thresholds, why labels and overrides, why environmental factors are combined, what is missing, and what governs change control.
3. **Make traceability visible.** Every displayed score, label, model outlook, pattern, and recommendation should be reconstructable from source values, rules, model artifact/feature window, and version.
4. **Keep the human in the loop.** Support accept/modify/defer/override decisions with reasons, follow-up timing, original recommendation preserved, and append-only action history.
5. **Model evidence should be honest.** Show selected metrics and representative actual-versus-predicted/SHAP evidence in defense material; state small-sample and prospective-replay limits.

## 10. Narrative rules for manuscript and deck

- Lead with the farm’s production goals, performance variability, and decision gap—not the models.
- EDA should establish: targets were inconsistently achieved; underperformance moved by cycle/building; Day 14 weight helps growth monitoring; early mortality is a separate survivability view; environment/feed provide context rather than standalone causal proof.
- Explain strategic choice: Canary combines low-capex, available-data, explainability, and near-term value; it does not claim that competing options are useless.
- Introduce Canary as the response to design requirements from EDA: current-cycle building visibility, separate growth/survivability monitoring, explainable inspection guidance, and human judgment.
- Show model proof in a proportionate main-deck segment and detailed appendix. Never present SHAP as causality.

## 11. Required update protocol

After every meaningful work session, update `CURRENT_STATE.md`. Update this file as well when scope, model routing, risk governance, data authority, panel feedback, or cross-deliverable strategy changes. Do not rely on chat history as project memory.
