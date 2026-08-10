"""Reproducible audit for the corrected weights and Zone A/Zone B records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from canary import load_workbook, train_outcome_model
from canary.day35 import CHECKPOINT_DAYS, build_day35_training_rows, train_day35_weight_baseline


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "FARM HARVEST DATA.xlsx"
TEMPERATURE_WORKBOOK = ROOT / "data" / "Aggregated Temperature Data.xlsx"
OUTPUT = ROOT / "analysis" / "updated_workbook_model_audit.json"


def _plain(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if pd.isna(value):
        return None
    return value


def main() -> None:
    dataset = load_workbook(WORKBOOK)
    checkpoints = dataset.daily.loc[
        dataset.daily["weight_measured"]
        & dataset.daily["age_day"].isin([*CHECKPOINT_DAYS, 35]),
        ["cycle_id", "building_id", "age_day", "bodyweight_kg"],
    ].drop_duplicates(["cycle_id", "building_id", "age_day"])
    coverage = (
        checkpoints.assign(available=True)
        .pivot_table(
            index=["cycle_id", "building_id"],
            columns="age_day",
            values="available",
            aggfunc="max",
            fill_value=False,
        )
        .reset_index()
    )
    for day in [*CHECKPOINT_DAYS, 35]:
        if day not in coverage:
            coverage[day] = False
    coverage = coverage[["cycle_id", "building_id", *CHECKPOINT_DAYS, 35]]
    coverage.columns = [
        "cycle_id",
        "building_id",
        "day_7",
        "day_14",
        "day_21",
        "day_28",
        "day_35",
    ]

    trajectories = checkpoints.pivot(
        index=["cycle_id", "building_id"], columns="age_day", values="bodyweight_kg"
    ).reset_index()
    trajectory_columns = [day for day in [*CHECKPOINT_DAYS, 35] if day in trajectories]
    duplicate_trajectories = trajectories.loc[
        trajectories.duplicated(trajectory_columns, keep=False)
    ].sort_values(trajectory_columns)

    source_temperature = pd.read_excel(
        WORKBOOK, sheet_name="Temperature", engine="openpyxl"
    )
    external_temperature = pd.read_excel(
        TEMPERATURE_WORKBOOK, sheet_name="Temperature", engine="openpyxl"
    )
    compare_columns = sorted(set(source_temperature.columns) & set(external_temperature.columns))

    def _normalize_temperature(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame[compare_columns].copy()
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        for column in normalized.select_dtypes(include="object"):
            normalized[column] = normalized[column].fillna("").astype(str).str.strip()
        return normalized.sort_values(compare_columns, na_position="first").reset_index(drop=True)

    source_compare = _normalize_temperature(source_temperature)
    external_compare = _normalize_temperature(external_temperature)
    temperature_files_match = source_compare.equals(external_compare)

    zone_summary = (
        dataset.daily.groupby("cycle_id", as_index=False)
        .agg(
            building_days=("age_day", "size"),
            zoned_days=("zone_aggregated", "sum"),
            max_sections=("environment_section_count", "max"),
            temperature_coverage_pct=("temperature_avg_c", lambda values: values.notna().mean() * 100),
            humidity_coverage_pct=("humidity_avg_pct", lambda values: values.notna().mean() * 100),
        )
        .sort_values("cycle_id")
    )

    day35_manifest = train_day35_weight_baseline(dataset)
    recovery = train_outcome_model(dataset, "recovery").manifest
    training_rows = build_day35_training_rows(dataset)
    payload = {
        "source": str(WORKBOOK),
        "quality": dataset.quality.__dict__,
        "weight_checkpoint_coverage": coverage.to_dict(orient="records"),
        "weight_checkpoint_summary": {
            "eligible_historical_building_cycles": int(
                training_rows[["cycle_id", "building_id"]].drop_duplicates().shape[0]
            ),
            "as_of_checkpoint_rows": int(len(training_rows)),
            "training_cycles": sorted(training_rows["cycle_id"].unique().tolist()),
            "current_cycle_excluded": "2026-3",
        },
        "duplicate_weight_trajectories_for_review": duplicate_trajectories.to_dict(
            orient="records"
        ),
        "environment": {
            "temperature_workbooks_match": bool(temperature_files_match),
            "aggregation_rule": "Unweighted mean of section averages; minimum of section minima; maximum of section maxima; preserve section spread",
            "zone_summary": zone_summary.to_dict(orient="records"),
        },
        "day35_weight_model": {
            "selected_model": day35_manifest["selected_model"],
            "selected_metrics": day35_manifest["selected_metrics"],
            "candidate_metrics": day35_manifest["candidate_metrics"],
            "day14_metrics": day35_manifest["day14_backtest_metrics"],
            "feature_importance": day35_manifest.get("ridge_feature_importance", []),
        },
        "recovery_model": {
            "selected_model": recovery["selected_model"],
            "training_cycles": recovery["training_cycles"],
            "training_building_cycles": recovery["training_building_cycles"],
            "training_snapshot_rows": recovery["training_snapshot_rows"],
            "selected_metrics": recovery["selected_metrics"],
            "candidate_metrics": recovery["metrics"],
            "day14_metrics": recovery["day14_backtest_metrics"],
            "feature_importance": recovery.get("global_feature_importance", []),
        },
    }
    OUTPUT.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")
    print(
        f"Canonical rows: {len(dataset.daily):,}; historical Day 35 outcomes: "
        f"{payload['weight_checkpoint_summary']['eligible_historical_building_cycles']}; "
        f"checkpoint rows: {len(training_rows)}."
    )
    print(
        f"Selected weight model: {day35_manifest['selected_model']} "
        f"({day35_manifest['selected_metrics']['mae_kg'] * 1000:.0f} g MAE)."
    )
    print(
        f"Selected recovery model: {recovery['selected_model']} "
        f"({recovery['selected_metrics']['mae'] * 100:.2f} percentage-point MAE)."
    )


if __name__ == "__main__":
    main()
