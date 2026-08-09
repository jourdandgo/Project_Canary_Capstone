# Project Canary Implementation Plan

The capstone is built in small, review-gated sprints. A sprint advances only after its calculations, user-facing behavior, and known limitations have been reviewed.

## Cross-cutting traceability contract

Every building decision must be explorable from output back to evidence. The detail view must preserve and display:

1. Source workbook, cycle, building, as-of date, production age, and measurement freshness.
2. Raw observations used by every available dimension.
3. Formula, comparison baseline, age band, and exact thresholds applied.
4. Individual dimension scores and explicit not-scored reasons.
5. Score equation, total score, and the score-to-label mapping used.
6. Primary and supporting drivers plus the identified problem pattern.
7. Risk-rule, target, model, feature-schema, and recommendation-rule versions as each layer becomes available.
8. Predicted outcomes, uncertainty, target differences, and model provenance once Sprint 3 is complete.
9. Recommended action and the exact Doc Raymond-approved rule that fired once Sprint 4 is complete.

No recommendation may be invented while the approved action mapping is unavailable. The interface must show a visible pending state instead.

## Sprint 1 — Data foundation and daily building state

**Status:** Complete

- Read the farm workbook without modifying it.
- Validate required sheets and fields.
- Consolidate repeated environmental readings into one reliable building-day record.
- Preserve true missing observations rather than converting blanks to zero.
- Provide cycle and as-of-date selection.
- Show Tags 1–3 and Lags 1–3 as Active, Incomplete, Inactive, or Records ended. A maximum daily-record date is not treated as proof of harvest.
- Show observed population, percentage alive, measured weight, age-specific target, and measurement freshness.
- Expose a data-quality report.

**Gate:** Automated calculation tests, application smoke test, and visual browser review pass.

## Sprint 2 — Rules-based risk rating and “Why”

**Status:** Complete with provisional thresholds pending farm validation

- Implement four independently explainable dimensions: weight gap, survival, mortality trend, and peer comparison.
- Keep thresholds in editable, versioned configuration rather than application code.
- Produce dimension scores, total score, Low/Medium/High/Critical label, primary drivers, supporting evidence, and problem pattern.
- Show reduced-evidence states when a dimension cannot be scored.
- Add risk history and rule evidence to the building detail view.

**Gate:** Hand-calculated examples reconcile exactly; no forecast value affects the risk score; provisional thresholds are visibly marked until farm-approved.

## Sprint 3 — Outcome forecasting

**Status:** Complete as a limited-data prototype using the simplified Day 35 storyline

- Build leakage-safe daily training snapshots grouped by complete recorded cycle.
- Train and compare naïve baselines with simple regularized and tree-based candidates.
- Produce separate outlooks for harvest recovery and Day 35 average liveweight.
- Report target gaps, uncertainty, validation results by forecast horizon, and model versions.
- Use the naïve model whenever ML does not demonstrate better validation performance.

**Gate review:** Cycle-level validation and future-information exclusion pass. Recovery is released as a limited-data prototype whose historical target is last-recorded population divided by beginning population—not confirmed harvest recovery. The Day 35 method uses 19 recorded building-level Day 35 outcomes across four cycles. Its cycle-held-out MAE is approximately 0.198 kg. All 19 outcomes were below 2.0 kg, so the data cannot yet validate whether it distinguishes target hitters from misses.

## Sprint 4 — Recommendations and integrated decision view

**Status:** Implemented with preliminary seven-rule guidance; formal operational approval remains pending Doc Raymond review

- Encode Doc Raymond’s approved problem-pattern action table.
- Add editable recommendation-rule administration with explicit save confirmation.
- Integrate risk, Why, both forecasts, confidence, and recommended action into the six-building overview.
- Rank current buildings by management priority while retaining incomplete, inactive, and records-ended buildings.

**Gate review:** Every displayed recommendation traces to a deterministic rule and no diagnosis or automated treatment language is introduced. The application clearly labels the rules preliminary, so the final approval gate remains open until Doc Raymond reviews all seven rules.

**Approval artifact:** `docs/Project_Canary_Preliminary_Action_Playbook.xlsx` provides one review row per rule, editable approval status, owner comments, approved wording, severity guidance, inspection checklists, safety boundaries, and research sources. The synchronized machine-readable draft is `config/recommendation_playbook_draft.json`.

## Sprint 5 — Capstone validation and handoff

**Status:** Complete; Phase 0 terminology and Day 35 corrections applied afterward

- Run historical walkthroughs and representative as-of-date scenarios, including after Day 14 and after Day 35.
- Validate missing-data, stale-weight, partial-cycle, and records-ended states.
- Finalize the demo workbook, operating guide, model card, limitations, and reproducible run instructions.
- Conduct a final stakeholder acceptance review against the PRD’s five required outputs.

**Gate review:** Five representative historical/as-of scenarios pass 45 of 45 acceptance checks, and the automated suite passes 32 tests. Validation corrected an incomplete-day forecast-status gap and an owner-facing summary that excluded incomplete but already-placed buildings. The reproducible evidence is stored in `artifacts/capstone_validation.json`; the plain-language handoff is in `docs/OPERATING_GUIDE.md`, `docs/SPRINT5_VALIDATION_REPORT.md`, and `docs/OPEN_ITEMS.md`.

**Gate:** The demonstration is reproducible on a clean local setup and all unresolved assumptions are either approved or clearly disclosed.

## Inputs still needed after the capstone prototype

- Farm-approved thresholds for the four risk dimensions. Provisional documented thresholds can be used to build Sprint 2, but not represented as validated farm policy.
- Confirmation of the final average-liveweight calculation and the two suspect Lagundi 2026-1 summary rows; final weights for 2026-2 and 2026-3 when available.
- Doc Raymond’s approval or revision of the preliminary recommendation playbook for Sprint 4.

The maintained source of truth for these and other follow-ups is `docs/OPEN_ITEMS.md`.

## Phase 0 revision — Metric clarity and Day 35 storyline

**Status:** Complete

- Treat Day 35 as the 2.0 kg management milestone, not the assumed harvest day.
- Project each building's Day 35 average weight against 2.0 kg when a measured weight exists.
- Keep recovery separate and disclose last-recorded recovery as its historical proxy until true harvest status is available.
- Prohibit the unsupported `final weight ÷ 49 × 35` conversion.
- Reconcile metric definitions, suspect-label exclusions, and outstanding expert approvals in one traceable register.

**Gate:** The app, operating guide, model card, and defense wording use the same definitions; unresolved decisions remain visible and are not silently assumed.

## Phase 1 — Daily operating experience

- Put active priorities first and visually quiet inactive or completed buildings.
- Add a concise “what changed since yesterday?” summary for risk, forecasts, mortality, survival, and data freshness.
- Preserve one-click movement from the priority view into the selected building’s evidence.

**Gate:** A non-technical owner can identify what changed, where to focus, why, and what to do next without opening the technical audit layer.

## Later phase — Action follow-through

- Allow recommendations to be marked checked, escalated, or resolved.
- Capture the responsible person, optional owner note, and action timestamp.
- Preserve a simple building-level action history without turning the prototype into a full farm-management system.

**Gate:** Action state is clearly separate from the recommendation rule, is auditable, and cannot alter historical risk or forecast results.

## Sprint 9 — Capstone Defense Mode and model comparison

- Add guided scenarios covering early warning, missing data, late-cycle disagreement, and completed outcomes.
- Compare naïve, Ridge, and tree-based candidates overall and by forecast horizon.
- Explain leakage prevention, validation splits, label sources, uncertainty, and limitations in panel-ready language.

**Gate:** Every claim shown in Defense Mode links to reproducible evidence and no model is presented as stronger or more personalized than validation supports.

## Sprint 10 — Final hardening and rehearsal

- Apply completed farm approvals and refresh all reports.
- Test desktop and narrower layouts, accessibility, empty states, and deployment setup.
- Run the full acceptance suite, deployment rehearsal, and defense dry run.

**Gate:** The hosted or local demonstration is reproducible, stakeholder-approved items are versioned, and every remaining caveat is disclosed in the defense materials.
