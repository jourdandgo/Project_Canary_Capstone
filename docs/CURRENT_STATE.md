# Project Canary — Current State and Next Actions

**Last updated:** 23 August 2026
**Update owner:** the agent or teammate completing the most recent meaningful change.

## 2026-08-23 — Trish v19 two-model integration and calculation trace

- **Changed:** `app.py`, `canary/forecast.py`, `canary/trish_v19.py`, `models/trish_v19/`, `config/trish_model_registry.json`, tests, model card, README, and project context.
- **What changed:** replaced the live v18 six-model story with final Model 1 recovery-proxy and Model 3 Day 35 bodyweight held-out replay; added artifact hashes, checkpoint error bands, exact 85-feature input tables, step-by-step risk/forecast calculations, and explicit no-fallback behavior outside the packaged replay lineage.
- **Evidence affected:** Model 1 overall held-out MAE 1.58 percentage points; Model 3 overall held-out MAE 122.1 g. Model 3 is shown at Days 7, 14, and 21 only and is held afterward.
- **Verified:** both artifact hashes match; canonical full suite passed (155 tests); GitHub-ready full suite passed (165 tests); a final 10-test v19/UI smoke check passed in both copies; Home, Building View, Model Evidence Explorer, and How Canary Works were visually inspected on a fresh local server.
- **Sync:** targeted runtime, model, documentation, and test changes mirrored to GitHub-ready and verified.
- **Open item:** arbitrary new-flock inference requires Trish's raw-data-to-85-feature transformer. Current release is pilot-stage replay, not autonomous production inference.

## 2026-08-20 — Cross-tool handoff pack

- **Changed:** `../../CLAUDE.md`, `PROJECT_CONTEXT.md`, `MODEL_REGISTRY.md`, `DEFENSE_NARRATIVE.md`, `HANDOVER_MAINTENANCE.md`, `../../Project_Canary_GitHub_Ready/CLAUDE.md`, `../../Project_Canary_GitHub_Ready/docs/HANDOFF.md`
- **What changed:** created durable Codex/Claude Code operating guidance, including product boundaries, source hierarchy, current app state, model routing and validation references, panel feedback, deck/manuscript alignment, and a mandatory update protocol.
- **Evidence affected:** no operational, risk, model, manuscript, deck, or app behavior changed.
- **Verified:** reviewed against `models/trish_v18/manifest.json`, `RISK_SYSTEM_GOVERNANCE.md`, `risk_rules.json`, and current source/app paths.
- **Sync:** canonical handoff is in `canary_app/docs/`; a compact standalone mirror was added to `Project_Canary_GitHub_Ready/docs/HANDOFF.md`.
- **Open item:** keep both handoff layers current after future material changes.

## Current headline

The canonical Canary prototype uses the final Trish v19 two-model handoff, current observed-risk governance, source-backed replay workflow, and end-to-end calculation traces. It is capstone-demo ready as a pilot-stage replay. It is not yet a generic future-flock scoring service.

## Completed and verified recently

### App

- Integrated Trish v19 Model 1 and Model 3 provenance into Command Center and Building View.
- Added forecast trace fields: model ID, algorithm, evidence cutoff, state (recalculated/held/observed), target, typical historical error, R², version, lineage status, and technical audit inputs.
- Replaced non-reproducible local-SHAP claims with exact held-out input rows, model/run identity, error-band construction, and global LOFO association evidence.
- Added 2026-3 replay checkpoint data: Day 7, 14, 15, 21, 28, and 35 CSVs for Tags 1–3; reset returns to historical baseline through 2026-2.
- Corrected completed-cycle presentation: completed cycles show actual outcomes rather than current critical-risk cards.
- Implemented observed-risk governance version `risk-rules-0.5.0-banded-hybrid`: base bands 0–2/3–5/6–8/9–12 plus documented acute/evidence overrides.
- Added traceable management decisions and Action History: accept, modify, defer, or override the suggested next check while preserving original evidence/recommendation.
- Simplified Home to a compact “What needs attention today?” operational header.
- Added **About Canary** under Farm owner; retained detailed **How Canary Works** under Defense tools.

### Cross-copy sync

The current Home/About change and the previous traceability upgrade were mirrored to:

- `canary_app/`
- `Project_Canary_GitHub_Ready/`

Do not assume future changes are synced; check both copies after every change. The GitHub-ready working tree already has unrelated user/project edits—mirror targeted files only.

### Recent verification

Focused Streamlit AppTest checks passed in both app copies on 20 August 2026:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_app.py::test_dashboard_renders_without_streamlit_errors \
  tests/test_app.py::test_home_prioritizes_today_decision_without_repeating_product_brief \
  tests/test_demo.py::test_defense_pages_render
```

Results: **5 passed** in `canary_app`; **5 passed** in `Project_Canary_GitHub_Ready`. SHAP emitted known upstream deprecation warnings only.

The local app was visually checked at `http://127.0.0.1:8501/`: Home reaches the cycle context, KPIs, and review-first content immediately; About Canary renders as a short orientation page.

## Current known limitations / open decisions

1. **Risk system approval:** thresholds, severity distances, and emergency overrides are proposed for farm shadow-pilot validation; not yet approved for routine use.
2. **Arbitrary future-cycle model inference:** the demo validates known 2026-3 prefixes. A full reusable Trish feature-engineering transformer/scorer must be packaged before claiming general raw-CSV scoring for future cycles.
3. **Data quality:** confirm sampling protocol, feed units, environmental-sensor freshness, ending-population source, and true harvest reconciliation.
4. **Outcome semantics:** use “ending-population recovery proxy” unless confirmed harvest recovery is available.
5. **Deck:** still needs systematic human-design polish, final source verification for every chart, and updated appendix/risk-modeling slides.
6. **Manuscript:** still needs final alignment after model/risk-system changes, especially Chapters III–VI, Executive Summary, front matter, references, and appendices.
7. **Model presentation:** M1 refreshes daily through Day 14 and holds afterward. M3 refreshes only at Days 7, 14, and 21 and holds afterward. Do not claim a Day 28 model forecast.
8. **Persistent-condition watch:** Canary separately flags temperature or humidity that remains on the same side of its age-specific range for three consecutive recorded days. A bodyweight watch means that one measured checkpoint deficit above 10% has remained unresolved for at least three review days; it must never be described as three daily weight measurements. Every signal exposes its daily/checkpoint trace and adds no risk points.
9. **Multiple problem patterns:** Canary retains one overall Low/Medium/High/Critical operational-priority label, but now preserves every supported observed problem pattern. The primary matched inspection guide is shown first; additional rule-matched guides remain visible with their rule IDs, checklists, and escalation triggers. Missing evidence is additional guidance and cannot suppress a confirmed acute mortality or population-loss pattern.
10. **Immutable calculation and override history:** Action History can save or backfill seven-day, 30-day, or full-cycle building-date calculations; display six-building score trends and a priority heatmap; and export snapshots. Manual overrides are linked to a saved snapshot and record system value, management value, rationale, responsible person, timestamp, and follow-up date without changing the original calculation. Local CSV persistence is suitable for the zipped pilot; durable database-backed storage remains required for routine Streamlit Cloud use.

## Priority next actions

1. **App audit:** run full test suite in both copies; complete a Day 7 → Day 14 → Day 21 → Day 35 demo replay and audit every displayed actual/forecast against source data.
2. **Human-in-the-loop audit:** verify Building View decision controls, original recommendation retention, override reasons, statuses, follow-up dates, and Action History filters visually.
3. **Risk-system proof:** prepare/revise deck appendix slides for why four checks, threshold provenance, primary-pattern selection, worked scoring example, and recommendation handoff.
4. **Model proof:** use only source-backed held-out predictions and LOFO association evidence; verify every MAE/R²/model label against the v19 manifest and saved OOF files.
5. **Deck polish:** replace generic/AI-looking language and layouts with real evidence, simple visual hierarchy, concise speaker notes, and defense-safe claims.
6. **Manuscript revision:** update the manuscript only after agreeing the final risk/model story; retain early-visibility positioning and avoid unsupported financial, causal, or accuracy claims.

## Required handoff entry template

Copy this at the top of this file after a material change:

```markdown
## YYYY-MM-DD — Short task title

- **Changed:** `path/to/file`, `other/file`
- **What changed:** concise behavior/narrative summary.
- **Evidence affected:** data/model/risk/deck/manuscript claim, if any.
- **Verified:** exact commands, visual workflow, and result.
- **Sync:** canonical only / mirrored to GitHub-ready / pending.
- **Open item:** any limitation, approval, or next action.
```
