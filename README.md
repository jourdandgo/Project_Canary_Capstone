# Project Canary

Local Streamlit prototype for daily broiler-farm decision support.

Current model release: **Trish v19 final handoff**. Canary uses Model 1 (Extra Trees) for the end-of-cycle recovery proxy and Model 3 (CatBoost) for Day 35 bodyweight. Historical and 2026-3 screens use Trish's saved held-out replay predictions; arbitrary future-cycle scoring remains unavailable until the 85-feature transformer is packaged.

Product definition: **Day 35 is the primary 1.8 kg management milestone.** Model 1 refreshes daily through Day 14 and is held afterward. Model 3 refreshes at Days 7, 14, and 21 and is held between and after checkpoints. The recovery target is a last-recorded-population proxy, not independently verified harvest recovery.

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
- Show the final Trish v19 Model 1 recovery-proxy outlook with its point estimate, held-out error band, evidence cutoff, model version, and explicit proxy limitation.
- Show the final Trish v19 Model 3 Day 35 bodyweight outlook only when an eligible Day 7, 14, or 21 checkpoint record exists; otherwise state that no validated forecast is available.
- Keep forecasting fully independent of the rules-based risk score.
- Present a simple owner-first view: how many buildings need attention and which ones, the inventory-weighted projected harvest recovery and target gap, estimated gross revenue at risk, and the first building/action to review.
- Use a multipage sidebar shell: owner pages (Home, Building View, Harvest Analysis, Business Value), capstone-evidence pages (EDA & Insights, Canary Methodology), and administration pages (Action Playbook, Data & Settings).
- Provide an all-cycle Harvest Analysis page with recovery and Day 35 weight trends, building comparisons, target lines, target-specific model eligibility, and a downloadable cycle-building table. Historical proxies and current projections remain visibly separate.
- Show current recovery and latest measured weight beside predicted recovery and the projected Day 35 result.
- Constrain predicted final recovery so it never exceeds survival already recorded today under the agreed accounting rule.
- Keep Trish's packaged v19 held-out replay lineage separate from arbitrary future-cycle scoring; the app does not silently substitute a local fallback model when the required 85-feature model-ready row is unavailable.
- Provide a dedicated adjustable Business Value page and card-level estimated gross revenue at risk, clearly separated from profit or guaranteed savings.
- Provide dedicated question-led EDA and detailed Canary Methodology pages for capstone defense, separate from the owner-facing Home dashboard.
- Retain every supported observed problem pattern, keep one overall priority label, and map each detected pattern to its own deterministic inspection rule while preserving one primary action.
- Apply a separate Low/Medium/High/Critical urgency guide without changing the risk score.
- Show the recommended action, response timing, inspection checklist, escalation condition, rule ID, version, and approval status.
- Provide a simple in-app action-playbook screen with explicit confirmation before local rule changes are saved.
- Save immutable, fingerprinted building-date calculation snapshots; review six-building score and priority history over seven days, 30 days, or a complete cycle; and export the ledger.
- Record management overrides separately from Canary calculations, including system value, management value, reason, responsible person, timestamp, follow-up date, and linked snapshot ID.
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
- `docs/Project_Canary_Defense_Cheat_Sheet.pdf`
- `docs/Project_Canary_Team_Walkthrough.pptx`
- `docs/TRISH_MODEL_AUDIT.md`
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

Calculation snapshots and management overrides default to `outputs/audit_ledger/`. Set `CANARY_AUDIT_DIR` to a durable writable location when one is available. Streamlit Community Cloud does not guarantee that local application files survive every restart or redeployment, so routine cloud use requires durable mounted storage or a persistent database. All history pages provide CSV exports for pilot backup.

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

## Forecast runtime boundary

The dashboard performs inference from the versioned Trish v19 handoff artifacts. It never retrains models during a workbook upload. A generic future-flock deployment still requires the handoff's raw-record-to-85-feature transformer; until that is packaged, Canary shows an explicit unavailable state rather than changing model families.
## Defense demo workflow

1. Start the app with `streamlit run app.py`.
2. Click **Reset demo** to return to the historical baseline through cycle 2026-2.
3. Upload one source-backed checkpoint from `demo_data/2026-3/`.
4. Review **Home**, **Building View**, and **Harvest Analysis** as Doc Raymond.
5. Open **Defense tools → Model Evidence Explorer** to replay an exact held-out prediction from input row to output.
6. Open **Defense tools → How Canary Works** for the separation between observed risk, the two forecast models, and inspection guidance.

The prepared CSVs cover Days 7, 14, 15, 21, 28, and 35 for Tags 1–3. The ZIP in the same folder contains all six files plus their lineage manifest. These are validated prospective replays of the actual 2026-3 records; they are not synthetic data and do not prove generic future-cycle feature engineering.
