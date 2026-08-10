"""Prepare JSON inputs for the artifact-tool canonical-data workbook."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from canary import load_workbook


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "FARM HARVEST DATA.xlsx"
OUTPUT = ROOT / "analysis" / "canonical_workbook_inputs.json"


def _json_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[list[object]]:
    return [[_json_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]


def _building(value: object) -> object:
    if pd.isna(value):
        return None
    compact = "".join(str(value).split()).lower()
    return {
        "tags1": "Tags 1",
        "tags2": "Tags 2",
        "tags3": "Tags 3",
        "lags1": "Lags 1",
        "lags2": "Lags 2",
        "lags3": "Lags 3",
    }.get(compact, str(value).strip())


def main() -> None:
    dataset = load_workbook(SOURCE)
    canonical = dataset.daily.copy()
    canonical["record_date"] = canonical["record_date"].dt.strftime("%Y-%m-%d")

    raw = pd.read_excel(SOURCE, sheet_name="Farm Harvest Data (Daily)", engine="openpyxl")
    raw.insert(0, "source_excel_row", raw.index + 2)
    raw["normalized_cycle_id"] = raw["Harvest Cycle"].astype("string").str.strip()
    raw["normalized_building_id"] = raw["Bldg."].map(_building)
    raw["normalized_age_day"] = pd.to_numeric(raw["Age"], errors="coerce")
    key = ["normalized_cycle_id", "normalized_building_id", "normalized_age_day"]
    valid = raw[key].notna().all(axis=1)
    duplicate_mask = raw.loc[valid].duplicated(key, keep=False)
    duplicate_indices = raw.loc[valid].index[duplicate_mask]
    duplicates = raw.loc[duplicate_indices].copy()
    duplicates["duplicate_group"] = (
        duplicates["normalized_cycle_id"].astype(str)
        + " | "
        + duplicates["normalized_building_id"].astype(str)
        + " | Day "
        + duplicates["normalized_age_day"].astype(int).astype(str)
    )
    audit_columns = [
        "duplicate_group",
        "source_excel_row",
        "normalized_cycle_id",
        "normalized_building_id",
        "normalized_age_day",
        "Date",
        "Beginning Inventory",
        "Population",
        "mortality_daily",
        "mortality_cum(hds)",
        "feedconsumption_daily",
        "feedconsumption_cummulative",
        "Daily FI/bird",
        "Bodyweight (kgs)",
        "Min Temperature",
        "Max Temperature",
        "Average Temperature",
        "Min Humidity ",
        "Max Humidity",
        "Average Humidity",
    ]
    duplicates = duplicates[audit_columns].sort_values(
        ["normalized_cycle_id", "normalized_building_id", "normalized_age_day", "source_excel_row"]
    )
    duplicates["Date"] = pd.to_datetime(duplicates["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    duplicate_summary = (
        duplicates.groupby(["normalized_cycle_id", "normalized_building_id"], as_index=False)
        .agg(
            duplicate_building_days=("duplicate_group", "nunique"),
            source_rows=("source_excel_row", "count"),
            first_age=("normalized_age_day", "min"),
            last_age=("normalized_age_day", "max"),
        )
        .sort_values(["normalized_cycle_id", "normalized_building_id"])
    )

    targets = dataset.targets.loc[
        dataset.targets["age_day"].le(35),
        [
            "age_day",
            "target_weight_scaled_g",
            "daily_gain_scaled_g",
            "target_weight_linear_g",
            "daily_gain_linear_g",
            "target_source",
        ],
    ].copy()
    checkpoint_coverage = (
        dataset.daily.loc[
            dataset.daily["weight_measured"]
            & dataset.daily["age_day"].isin([7, 14, 21, 28, 35]),
            ["cycle_id", "building_id", "age_day"],
        ]
        .assign(recorded="Yes")
        .pivot_table(
            index=["cycle_id", "building_id"],
            columns="age_day",
            values="recorded",
            aggfunc="first",
        )
        .reindex(columns=[7, 14, 21, 28, 35])
        .fillna("No")
        .reset_index()
        .rename(
            columns={
                7: "Day 7",
                14: "Day 14",
                21: "Day 21",
                28: "Day 28",
                35: "Day 35",
            }
        )
        .sort_values(["cycle_id", "building_id"])
    )

    payload = {
        "quality": dataset.quality.__dict__,
        "canonical": {
            "headers": canonical.columns.tolist(),
            "rows": _records(canonical),
        },
        "duplicate_rows": {
            "headers": duplicates.columns.tolist(),
            "rows": _records(duplicates),
        },
        "duplicate_summary": {
            "headers": duplicate_summary.columns.tolist(),
            "rows": _records(duplicate_summary),
        },
        "targets": {
            "headers": targets.columns.tolist(),
            "rows": _records(targets),
        },
        "checkpoint_coverage": {
            "headers": checkpoint_coverage.columns.tolist(),
            "rows": _records(checkpoint_coverage),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Prepared {len(canonical):,} canonical rows and {len(duplicates):,} duplicate source rows.")


if __name__ == "__main__":
    main()
