# Bundled capstone data

The GitHub-ready capstone package includes:

- `FARM HARVEST DATA.xlsx` — daily production records, targets, and current-cycle inputs.
- `Farm Performance Summary.xlsx` — used only for completed-cycle final average weights.
- `Project_Canary_Canonical_Building_Day_1666.xlsx` — auditable cleaned output showing the 1,666 canonical building-days, duplicate summary, and all repeated source rows. The app rebuilds this structure in memory from the source workbook; this file is supplied for review rather than used as a second source of truth.
- `Project_Canary_Revised_Target_Weights.xlsx` — auditable daily target curve. It fixes Doc Raymond's revised Days 7/14/21/28/35 checkpoints at 170/380/800/1,200/1,800 g and estimates the missing days using a rescaled version of the former within-week curve shape. Day 0 remains a clearly labeled 40 g working anchor.
- `Aggregated Temperature Data.xlsx` — source reference used to verify the Zone A / Zone B duplication pattern. The app uses the matching temperature fields already embedded in `FARM HARVEST DATA.xlsx`.

These files let Project Canary open with preliminary results. Uploading a newer workbook in the app replaces the corresponding bundled file for the current Streamlit session and recalculates the applicable outputs.

Do not add other farm files or publish this repository/app more broadly without farm-owner approval.
