# Project Canary

Local Streamlit prototype for daily broiler-farm decision support.

Product definition: **Day 35 is the primary 1.8 kg management milestone.** Canary uses Days 1–14 as the early-warning window, projects each building's Day 35 average weight when a measured weight exists, and separately forecasts harvest recovery against 95%.

## Completed capstone scope through Sprint 5

- Read `FARM HARVEST DATA.xlsx` without modifying the uploaded file; apply the farm-approved revised target curve in the ingestion layer.
- Consolidate multiple environmental-section readings to one building-day.
- Preserve missing mortality and feed observations as missing, not zero.
- Select a harvest cycle; the latest cycle also accepts an as-of review date.
- Separate the latest-cycle decision experience from earlier-cycle completed results.
- Show all six physical buildings in a fixed position for every cycle.
- For earlier cycles, show harvest completed on, actual harvest recovery, and actual final average weight when available—without historical risk ratings, predictions, or recommendations.
- Show latest observed population, percentage alive, bodyweight measurement, target, and freshness.
- Show an inspectable data-quality report.
- Calculate four separate rules-based dimensions: weight gap, cumulative population loss, latest daily mortality, and combined environmental conditions.
- Map the available-dimension total to Low, Medium, High, or Critical using the agreed 0–12 structure.
- Generate deterministic primary and supporting explanations plus problem-pattern classifications.
- Preserve missing dimensions as not scored and disclose reduced evidence.
- Show risk priority, dimension evidence, daily score history, rule version, and provisional-threshold status.
- Expose an end-to-end decision trace with raw observations, calculations, applied thresholds, dimension scores, score equation, label mapping, problem pattern, action-rule status, and audit metadata.
- Build leakage-safe daily modeling snapshots using only complete recorded cycles and information available as of each day.
- Compare a trend baseline, historical mean, Ridge regression, and random forest with leave-one-cycle-out validation.
- Forecast harvest recovery with a point estimate, target gap, empirical uncertainty range, model version, and plain-language confidence note. The historical target is explicitly disclosed as last-recorded recovery because confirmed harvest status is not available in the source workbook.
- Project Day 35 average weight with compact Ridge regression using only checkpoint evidence known by the review date. Historical remaining gain remains the transparent benchmark and fallback candidate.
- Keep forecasting fully independent of the rules-based risk score.
- Present a simple owner-first view: how many buildings need attention and which ones, the inventory-weighted projected harvest recovery and target gap, estimated gross revenue at risk, and the first building/action to review.
- Use a multipage sidebar shell: owner pages (Home, Building View, Business Value), capstone-evidence pages (EDA & Insights, Canary Methodology), and administration pages (Action Playbook, Data & Settings).
- Show current recovery and latest measured weight beside predicted recovery and the projected Day 35 result.
- Constrain predicted final recovery so it never exceeds survival already recorded today under the agreed accounting rule.
- Compare Day 35 historical mean, target-curve, recent-ADG, historical remaining-gain, Ridge, Random Forest, and gradient-boosting candidates; select Ridge because it has the lowest cycle-balanced held-out MAE and beats the simple benchmark beyond the 5% tolerance.
- Treat each eligible building checkpoint as a separate as-of training example and pool examples across buildings; do not fit unreliable building-specific models or use later checkpoint weights in an earlier forecast.
- Provide a dedicated adjustable Business Value page and card-level estimated gross revenue at risk, clearly separated from profit or guaranteed savings.
- Provide dedicated question-led EDA and detailed Canary Methodology pages for capstone defense, separate from the owner-facing Home dashboard.
- Map each identified problem pattern to one deterministic recommendation rule.
- Apply a separate Low/Medium/High/Critical urgency guide without changing the risk score.
- Show the recommended action, response timing, inspection checklist, escalation condition, rule ID, version, and approval status.
- Provide a simple in-app action-playbook screen with explicit confirmation before local rule changes are saved.
- Provide a Data & Settings risk-rule screen that documents all four dimensions and safely edits age-based cutoffs, peer cutoffs, survival target assumptions, rating bands, version, and approval status with validation and explicit confirmation.
- Preserve a reproducible risk-system audit in `analysis/risk_scoring_audit.json`; disclose that the provisional Day 14 rule score is operational prioritization, not a validated outcome classifier.
- Continue forecasts on incomplete production days when enough earlier observations exist, while clearly disclosing that the latest recorded data was used.
- Reproduce five acceptance scenarios covering Day 14, Day 22, staggered building states, Day 48 with a stale weight, and a missing current-day record.
- Maintain a stakeholder validation report, plain-language operating guide, and durable open-items register.

The current prototype implements the Day 35 storyline and a simple lifecycle split suitable for a capstone demonstration with disclosed caveats. Only the latest cycle receives live risk, forecast, and action outputs. Every earlier cycle is shown as completed using each building's last recorded daily date, and the dashboard explicitly identifies this as a capstone convention because the source has no verified harvest-event flag. Unapproved current-cycle actions remain visibly preliminary until Doc Raymond completes the review.
See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the gated build sequence.

Handoff documents:

- `docs/SPRINT5_VALIDATION_REPORT.md`
- `docs/OPERATING_GUIDE.md`
- `docs/OPEN_ITEMS.md`
- `docs/PROJECT_CANARY_DEFENSE_CHEAT_SHEET.md`
- `docs/Project_Canary_Defense_Cheat_Sheet.docx`
- `outputs/model_ready/Project_Canary_Model_Ready_Data.xlsx`
- `outputs/model_ready/recovery_training.csv`
- `outputs/model_ready/day35_weight_training.csv`
- `notebooks/Project_Canary_Harvest_Recovery_Model.ipynb`
- `notebooks/Project_Canary_Day35_Weight_Model.ipynb`
- `docs/TEAMMATE_MODEL_COMPARISON_PROTOCOL.md`

A researched seven-rule preliminary action playbook is available for Doc Raymond's review. The editable approval workbook is stored in `docs/Project_Canary_Preliminary_Action_Playbook.xlsx`, and its synchronized version-controlled source is `config/recommendation_playbook_draft.json`. The dashboard uses these as visibly preliminary inspection and escalation guidance—not approved farm policy or treatment advice.

The initial scoring thresholds are stored in `config/risk_rules.json`. They are intentionally versioned and visibly marked provisional until reviewed with farm experts. Farm-owner-approved changes can be made through **Data & Settings → Risk score rules**, where unsafe cutoffs or incomplete score bands are rejected before saving.

Model artifacts and full validation results are stored in `models/`; see `models/MODEL_CARD.md` for limitations and candidate comparisons.

## Run locally

From this directory:

```bash
uv sync --dev
uv run streamlit run app.py
```

When the app lives inside the Project Canary folder, it automatically finds `../FARM HARVEST DATA.xlsx`. A different default workbook can be supplied with `CANARY_DEFAULT_WORKBOOK`.

The GitHub/Streamlit capstone package may bundle an approved current daily workbook and final-weight summary under `data/` so the dashboard opens immediately. A newer daily workbook can be supplied through **Update daily farm data (optional)**, and a newer final-weight summary through **Update final-weight data (optional)**. An upload replaces the bundled file only for that session. See [DEPLOYMENT.md](DEPLOYMENT.md).

## Test

```bash
uv run pytest
```

## Reproduce capstone acceptance checks

```bash
uv run python -m scripts.validate_capstone
```

The detailed building-level evidence is written to `artifacts/capstone_validation.json`.

## Rebuild forecast models

```bash
uv run python -m scripts.train_models "../FARM HARVEST DATA.xlsx" --output models
```

The dashboard only performs inference. It never retrains models during a daily workbook upload.
