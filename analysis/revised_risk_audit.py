"""Recompute traceability and historical associations for the revised risk score."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from canary import load_workbook, score_cycle_snapshot
from canary.modeling import complete_cycle_ids


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("risk_scoring_audit.json")


def correlation(frame: pd.DataFrame, x: str, y: str) -> float | None:
    pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    return None if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2 else round(float(pair[x].corr(pair[y])), 3)


def main() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    cycles = complete_cycle_ids(dataset)
    rows = []
    for cycle in cycles:
        meta = dataset.cycles.loc[dataset.cycles["cycle_id"] == cycle]
        cycle_start = pd.Timestamp(meta["start_date"].min()).normalize()
        for timing, as_of in (("Day 14", cycle_start + pd.DateOffset(days=13)), ("Last recorded", pd.Timestamp(meta["end_date"].max()))):
            scored = score_cycle_snapshot(dataset, cycle, as_of)
            for _, row in scored.loc[scored["risk_score"].notna()].iterrows():
                building = row["building_id"]
                building_meta = meta.loc[meta["building_id"] == building]
                day35 = dataset.daily.loc[
                    (dataset.daily["cycle_id"] == cycle)
                    & (dataset.daily["building_id"] == building)
                    & (dataset.daily["age_day"] == 35)
                    & dataset.daily["weight_measured"],
                    "bodyweight_kg",
                ]
                rows.append({
                    "timing": timing,
                    "cycle_id": cycle,
                    "building_id": building,
                    "risk_score": row["risk_score"],
                    "weight_score": row.get("weight_score"),
                    "population_loss_score": row.get("population_loss_score"),
                    "daily_mortality_score": row.get("daily_mortality_score"),
                    "environment_score": row.get("environment_score"),
                    "final_recovery": building_meta.iloc[0]["final_recovery_rate"] if not building_meta.empty else pd.NA,
                    "day35_weight_kg": day35.iloc[-1] if not day35.empty else pd.NA,
                })
    frame = pd.DataFrame(rows)
    day14 = frame.loc[frame["timing"] == "Day 14"]
    last = frame.loc[frame["timing"] == "Last recorded"]
    payload = {
        "audit_date": "2026-08-10",
        "purpose": "Audit the revised direct-evidence risk score for traceability and historical association without treating it as an outcome probability.",
        "design": {
            "dimensions": ["Weight gap", "Population loss", "Daily mortality", "Environmental conditions"],
            "environment_formula": "maximum of temperature-range score and humidity-deviation score",
            "removed_from_points": ["Mortality trend", "Peer comparison"],
            "interpretation": "Operational priority only; separate models forecast outcomes.",
        },
        "data_limit": {
            "temperature_coverage_pct": dataset.quality.temperature_coverage_pct,
            "humidity_coverage_pct": dataset.quality.humidity_coverage_pct,
            "temperature_range_days_above_5c": 540,
            "temperature_range_days_recorded": 696,
        },
        "day14_audit": {
            "scored_snapshots": int(len(day14)),
            "environment_scored": int(day14["environment_score"].notna().sum()),
            "score_vs_final_recovery_correlation": correlation(day14, "risk_score", "final_recovery"),
            "score_vs_day35_weight_correlation": correlation(day14, "risk_score", "day35_weight_kg"),
        },
        "last_recorded_audit": {
            "scored_snapshots": int(len(last)),
            "environment_scored": int(last["environment_score"].notna().sum()),
            "score_vs_final_recovery_correlation": correlation(last, "risk_score", "final_recovery"),
            "score_vs_day35_weight_correlation": correlation(last, "risk_score", "day35_weight_kg"),
        },
        "decision": "Use the score for transparent prioritization. Do not claim calibrated probability or causal environmental effects. Validate the environmental thresholds with the farm before operational reliance.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
