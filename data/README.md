# Bundled capstone data

The GitHub-ready capstone package includes:

- `FARM HARVEST DATA.xlsx` — daily production records, targets, and current-cycle inputs.
- `Farm Performance Summary.xlsx` — used only for completed-cycle final average weights.
- `Project_Canary_Canonical_Building_Day_1666.xlsx` — auditable cleaned output showing the 1,666 canonical building-days, duplicate summary, and all repeated source rows. The app rebuilds this structure in memory from the source workbook; this file is supplied for review rather than used as a second source of truth.

These files let Project Canary open with preliminary results. Uploading a newer workbook in the app replaces the corresponding bundled file for the current Streamlit session and recalculates the applicable outputs.

Do not add other farm files or publish this repository/app more broadly without farm-owner approval.
