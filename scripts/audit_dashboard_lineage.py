"""Reconcile Project Canary demo inputs, model outputs, and dashboard labels.

This is a defense-facing audit. It does not rewrite source data. It records
where the authoritative Daily tab agrees with the app and where the workbook's
underlying weighing table disagrees with the Daily tab used by Trish's models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
V18 = ROOT / "capstone_FINAL_v18"
WORKBOOK = V18 / "data" / "raw" / "FARM HARVEST DATA (with connected temp).xlsx"
MASTER = V18 / "data" / "intermediate" / "master_dataset.csv"
BUNDLE = APP / "models" / "trish_v18"
DEMO = APP / "demo_data" / "2026-3"
OUTPUT = APP / "analysis" / "dashboard_lineage_audit"


def _daily_source() -> pd.DataFrame:
    daily = pd.read_excel(WORKBOOK, sheet_name="Farm Harvest Data (Daily)").rename(
        columns={
            "Harvest Cycle": "cycle_id",
            "Bldg.": "building_id",
            "Age": "age_day",
            "Date": "record_date",
            "Bodyweight (kgs)": "bodyweight_kg",
        }
    )
    daily["record_date"] = pd.to_datetime(daily["record_date"]).dt.normalize()
    return daily


def _weight_source_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    aggregate = pd.read_excel(
        WORKBOOK, sheet_name="Days 1 to 14 Weights Aggregated"
    ).rename(
        columns={
            "Harvest Cycle": "cycle_id",
            "Blg": "building_id",
            "Day": "age_day",
            "Average Bodyweight (in KG)": "aggregated_weight_kg",
        }
    )
    left = daily.loc[
        daily["cycle_id"].astype(str).eq("2026-3")
        & daily["age_day"].isin([1, 7, 14]),
        ["cycle_id", "building_id", "age_day", "bodyweight_kg"],
    ]
    right = aggregate.loc[
        aggregate["cycle_id"].astype(str).eq("2026-3")
        & aggregate["age_day"].isin([1, 7, 14]),
        ["cycle_id", "building_id", "age_day", "aggregated_weight_kg"],
    ]
    comparison = left.merge(
        right,
        on=["cycle_id", "building_id", "age_day"],
        how="outer",
        validate="one_to_one",
    )
    comparison["difference_g"] = (
        comparison["bodyweight_kg"] - comparison["aggregated_weight_kg"]
    ) * 1000
    comparison["matches"] = comparison["difference_g"].abs().le(0.01)
    return comparison.sort_values(["age_day", "building_id"])


def _demo_validation(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = daily.loc[daily["cycle_id"].astype(str).eq("2026-3")].copy()
    for path in sorted(DEMO.glob("Project_Canary_2026-3_Day_*.csv")):
        demo = pd.read_csv(path, parse_dates=["record_date"])
        cutoff = int(demo["age_day"].max())
        expected = source.loc[
            source["age_day"].le(cutoff),
            [
                "cycle_id",
                "building_id",
                "age_day",
                "record_date",
                "Population",
                "bodyweight_kg",
            ],
        ].rename(columns={"Population": "population"})
        merged = demo.merge(
            expected,
            on=["cycle_id", "building_id", "age_day"],
            how="outer",
            suffixes=("_demo", "_source"),
            indicator=True,
        )
        numeric_match = (
            pd.to_numeric(merged["population_demo"], errors="coerce")
            .fillna(-1)
            .eq(pd.to_numeric(merged["population_source"], errors="coerce").fillna(-1))
            & pd.to_numeric(merged["bodyweight_kg_demo"], errors="coerce")
            .fillna(-1)
            .sub(pd.to_numeric(merged["bodyweight_kg_source"], errors="coerce").fillna(-1))
            .abs()
            .le(1e-9)
        )
        rows.append(
            {
                "file": path.name,
                "cutoff_day": cutoff,
                "row_count": len(demo),
                "buildings": ", ".join(sorted(demo["building_id"].unique())),
                "continuous_days": all(
                    group["age_day"].tolist() == list(range(1, cutoff + 1))
                    for _, group in demo.sort_values("age_day").groupby("building_id")
                ),
                "matches_daily_source": bool(
                    merged["_merge"].eq("both").all() and numeric_match.all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _endpoint_summary() -> pd.DataFrame:
    master = pd.read_csv(MASTER, parse_dates=["date"])
    rows = []
    for (cycle, building), group in master.loc[
        master["harvest_cycle"].astype(str).isin(["2026-2", "2026-3"])
    ].groupby(["harvest_cycle", "bldg"]):
        group = group.sort_values("age")
        last = group.iloc[-1]
        rows.append(
            {
                "cycle_id": cycle,
                "building_id": building,
                "first_date": group["date"].min().date().isoformat(),
                "last_date": group["date"].max().date().isoformat(),
                "last_recorded_day": int(last["age"]),
                "beginning_population": int(last["beginning_inventory"]),
                "last_recorded_population": int(last["population"]),
                "ending_recovery_proxy_pct": float(last["final_harvest_recovery"]) * 100,
                "day35_weight_g": group.loc[group["age"].eq(35), "bodyweight_g"].dropna().iloc[-1]
                if group.loc[group["age"].eq(35), "bodyweight_g"].notna().any()
                else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _prospective_accuracy() -> pd.DataFrame:
    snapshot = pd.read_csv(BUNDLE / "prediction_snapshot.csv")
    master = pd.read_csv(MASTER)
    recovery = master.loc[
        master["harvest_cycle"].astype(str).eq("2026-3"),
        ["bldg", "final_harvest_recovery"],
    ].drop_duplicates()
    weight = master.loc[
        master["harvest_cycle"].astype(str).eq("2026-3") & master["age"].eq(35),
        ["bldg", "bodyweight_g"],
    ].drop_duplicates()
    rows = []
    for model_id, days, actuals, actual_col, unit_scale in (
        ("model_1", [7, 14], recovery, "final_harvest_recovery", 100),
        ("model_2", [7, 14], weight, "bodyweight_g", 1),
        ("model_3", [21], weight, "bodyweight_g", 1),
    ):
        scored = snapshot.loc[
            snapshot["model_id"].eq(model_id)
            & snapshot["prediction_day"].isin(days)
        ].merge(actuals, on="bldg", validate="many_to_one")
        scored["absolute_error"] = (
            scored["prediction"] - scored[actual_col]
        ).abs() * unit_scale
        for day, group in scored.groupby("prediction_day"):
            rows.append(
                {
                    "model_id": model_id,
                    "evidence_day": int(day),
                    "buildings": len(group),
                    "mean_absolute_error": group["absolute_error"].mean(),
                    "unit": "percentage points" if model_id == "model_1" else "grams",
                    "source": "Deployed Canary v18 bundle",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily = _daily_source()
    comparisons = _weight_source_comparison(daily)
    demos = _demo_validation(daily)
    endpoints = _endpoint_summary()
    accuracy = _prospective_accuracy()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())

    comparisons.to_csv(OUTPUT / "2026-3_weight_source_comparison.csv", index=False)
    demos.to_csv(OUTPUT / "demo_file_reconciliation.csv", index=False)
    endpoints.to_csv(OUTPUT / "cycle_endpoint_reconciliation.csv", index=False)
    accuracy.to_csv(OUTPUT / "deployed_2026-3_accuracy.csv", index=False)

    report = f"""# Project Canary dashboard lineage audit

## Overall assessment

**Needs revision before defense.** The demo files, model-ready inputs, deployed predictions, and dashboard are internally aligned to the workbook's Daily tab. However, the source workbook contains a material Day 7 and Day 14 bodyweight disagreement between its Daily tab and its detailed weighing tab.

## Verified lineage

- Demo CSV files checked: **{len(demos)}**; all match the Daily tab: **{bool(demos['matches_daily_source'].all())}**.
- Deployed model bundle: **{manifest['bundle_version']}**.
- Model 1 target: **{manifest['models']['model_1']['target']}** (last-recorded-population recovery proxy, not uniformly Day 35).
- Models 2 and 3 target: **{manifest['models']['model_2']['target']}** (recorded Day 35 bodyweight).
- 2026-3 prospective scoring uses only Tags 1-3; Lags 1-3 have no 2026-3 records.

## Material source-data issue

- On Day 7, the Daily tab records **114.74 g for Tags 1, Tags 2, and Tags 3**.
- The detailed aggregated weighing tab records **100.28 g, 94.65 g, and 98.04 g**, respectively.
- On Day 14, the Daily tab again repeats **242.76 g** across all three buildings, while the aggregated table records distinct values.
- Canary must retain the Daily-tab values for the defense replay because Trish's feature tables and deployed models were built from that source. The discrepancy must be disclosed and corrected upstream before production use.

## Forecast refresh logic

- Recovery M1 recalculates only through Day 14. Day 21 and Day 28 are held from Day 14; they are not new accuracy observations.
- Day 35 bodyweight uses M2 through Day 14 and M3 through Day 21. Day 28 holds the Day 21 estimate.
- The checkpoint chart now leaves held checkpoints blank and labels the hold explicitly.

## Metric definition

`final_harvest_recovery` equals population on each building's **last recorded day** divided by beginning population. Historical buildings commonly end on Day 49, while 2026-3 currently ends on Day 35. It is therefore an ending-population recovery proxy—not a uniformly Day 35 recovery target and not a reconciled sale count.
"""
    (OUTPUT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
