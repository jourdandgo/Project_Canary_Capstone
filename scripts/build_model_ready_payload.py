"""Create auditable model-ready tables for the spreadsheet/CSV export builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from canary import (
    build_day35_feature_rows,
    build_day35_training_rows,
    build_modeling_snapshots,
    build_recovery_training_snapshots,
    load_workbook,
)
from canary.modeling import RECOVERY_CORE_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "data" / "FARM HARVEST DATA.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs" / "model_ready" / "model_ready_payload.json"


def _serializable_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    clean = clean.replace({np.nan: None, pd.NaT: None})
    return clean.to_dict(orient="records")


def _building_outcomes(dataset, recovery: pd.DataFrame, weight: pd.DataFrame) -> pd.DataFrame:
    recovery_outcomes = (
        recovery[
            [
                "cycle_id",
                "building_id",
                "beginning_inventory",
                "target",
                "label_date",
            ]
        ]
        .drop_duplicates(["cycle_id", "building_id"])
        .rename(
            columns={
                "target": "final_recovery_proxy",
                "label_date": "recovery_label_date",
            }
        )
    )
    recovery_outcomes["ending_population"] = (
        recovery_outcomes["beginning_inventory"]
        * recovery_outcomes["final_recovery_proxy"]
    ).round()
    recovery_outcomes["eligible_recovery_model"] = True

    checkpoint_rows = build_day35_training_rows(dataset)
    weight_outcomes = weight[
        ["cycle_id", "building_id", "actual_day35_weight_kg_y"]
    ].drop_duplicates(["cycle_id", "building_id"])
    checkpoint_wide = (
        checkpoint_rows.pivot_table(
            index=["cycle_id", "building_id"],
            columns="measurement_day",
            values="current_weight_kg",
            aggfunc="first",
        )
        .rename(columns={day: f"day_{day}_weight_g" for day in (7, 14, 21, 28)})
        .reset_index()
    )
    for column in [f"day_{day}_weight_g" for day in (7, 14, 21, 28)]:
        if column in checkpoint_wide:
            checkpoint_wide[column] = checkpoint_wide[column] * 1000
    weight_outcomes = weight_outcomes.merge(
        checkpoint_wide, on=["cycle_id", "building_id"], how="left"
    )
    weight_outcomes["day_35_weight_g"] = (
        weight_outcomes.pop("actual_day35_weight_kg_y") * 1000
    )
    weight_outcomes["eligible_day35_weight_model"] = True

    keys = pd.concat(
        [
            recovery_outcomes[["cycle_id", "building_id"]],
            weight_outcomes[["cycle_id", "building_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    outcomes = (
        keys.merge(recovery_outcomes, on=["cycle_id", "building_id"], how="left")
        .merge(weight_outcomes, on=["cycle_id", "building_id"], how="left")
        .sort_values(["cycle_id", "building_id"])
        .reset_index(drop=True)
    )
    outcomes["eligible_recovery_model"] = outcomes["eligible_recovery_model"].eq(True)
    outcomes["eligible_day35_weight_model"] = outcomes[
        "eligible_day35_weight_model"
    ].eq(True)
    ordered = [
        "cycle_id",
        "building_id",
        "beginning_inventory",
        "ending_population",
        "final_recovery_proxy",
        "recovery_label_date",
        "day_7_weight_g",
        "day_14_weight_g",
        "day_21_weight_g",
        "day_28_weight_g",
        "day_35_weight_g",
        "eligible_recovery_model",
        "eligible_day35_weight_model",
    ]
    return outcomes.reindex(columns=ordered)


def _recovery_training(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["validation_cycle"] = result["cycle_id"].astype(str)
    result["forecast_horizon_days"] = (
        pd.to_datetime(result["label_date"]) - pd.to_datetime(result["as_of_date"])
    ).dt.days
    result["row_selection"] = np.where(
        result["cycle_day"].isin([7, 14, 21, 28]),
        "Standard checkpoint",
        "Latest eligible pre-outcome",
    )
    result = result.rename(columns={"target": "final_recovery_proxy_y"})
    identifiers = [
        "cycle_id",
        "building_id",
        "as_of_date",
        "cycle_day",
        "label_date",
        "forecast_horizon_days",
        "validation_cycle",
        "row_selection",
    ]
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"cycle_id", "building_id", "as_of_date", "label_date", "target", "cycle_day"}
    ]
    return result[identifiers + feature_columns + ["final_recovery_proxy_y"]]


def _dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    definitions = {
        "cycle_id": ("Identifier", "text", "Farm workbook", "Harvest-cycle identifier"),
        "building_id": ("Identifier", "text", "Farm workbook", "Physical building identifier"),
        "as_of_date": ("Identifier", "date", "Derived", "Latest date allowed into this snapshot"),
        "cycle_day": ("X", "day", "Derived", "Production age known on the review date"),
        "label_date": ("Audit", "date", "Derived", "Date from which the recovery proxy label was taken"),
        "recovery_label_date": ("Audit", "date", "Derived", "Last recorded date used for the recovery proxy label"),
        "validation_cycle": ("Validation group", "text", "Derived", "Entire cycle held out together during validation"),
        "forecast_horizon_days": ("Audit", "days", "Derived", "Days between the snapshot and label date"),
        "row_selection": ("Audit", "text", "Derived", "Why this recovery snapshot was retained"),
        "beginning_inventory": ("X / audit", "birds", "Farm workbook", "Beginning population"),
        "ending_population": ("Outcome evidence", "birds", "Farm workbook", "Last recorded population used by the recovery proxy"),
        "percentage_alive": ("X", "proportion", "Derived", "Current population divided by beginning population"),
        "final_recovery_proxy": ("Outcome", "proportion", "Derived", "Last recorded population divided by beginning population"),
        "final_recovery_proxy_y": ("Y", "proportion", "Derived", "Recovery-model target; unavailable to the model at prediction time"),
        "actual_day35_weight_kg_y": ("Y", "kg", "Farm workbook", "Observed average bodyweight recorded on production Day 35"),
        "day_35_weight_g": ("Outcome", "g", "Farm workbook", "Observed average bodyweight on production Day 35"),
        "temperature_recent_avg_c": ("X", "°C", "Derived", "Mean of available recent temperature readings known by the review date"),
        "humidity_recent_avg_pct": ("X", "%", "Derived", "Mean of available recent humidity readings known by the review date"),
        "current_to_target_ratio": ("X", "ratio", "Derived", "Current measured weight divided by the age-specific farm target"),
        "recent_adg_kg_day": ("X", "kg/day", "Derived", "Gain since the previous recorded checkpoint divided by elapsed days"),
        "cumulative_adg_kg_day": ("X", "kg/day", "Derived", "Average gain since the Day 7 checkpoint"),
    }
    rows: list[dict[str, object]] = []
    for sheet, frame in tables.items():
        for column in frame.columns:
            fallback_role = "X" if column in RECOVERY_CORE_FEATURE_COLUMNS or column.startswith("weight_day_") else "Audit"
            fallback_unit = (
                "g" if column.startswith("day_") and column.endswith("_weight_g")
                else "kg" if column.endswith("_kg")
                else "recorded unit"
            )
            role, unit, source, definition = definitions.get(
                column,
                (
                    fallback_role,
                    fallback_unit,
                    "Farm workbook or derived",
                    column.replace("_", " ").capitalize(),
                ),
            )
            missing = (
                "Kept missing; median-imputed inside each training fold"
                if role == "X"
                else "Blank means the outcome or field is not available/eligible"
            )
            leakage = (
                "Never included in X; used only after prediction for evaluation"
                if role in {"Y", "Outcome", "Outcome evidence"}
                else "Built using records on or before the snapshot date"
                if role == "X"
                else "Identifier or audit field; excluded from fitted X unless explicitly documented"
            )
            rows.append(
                {
                    "sheet": sheet,
                    "column": column,
                    "role": role,
                    "unit": unit,
                    "source": source,
                    "plain_language_definition": definition,
                    "missing_value_handling": missing,
                    "leakage_guard": leakage,
                }
            )
    return pd.DataFrame(rows)


def build_payload(workbook: Path) -> dict[str, object]:
    dataset = load_workbook(workbook)
    recovery_daily = build_modeling_snapshots(dataset, "recovery")
    recovery_training = _recovery_training(build_recovery_training_snapshots(dataset))
    weight_training = build_day35_feature_rows(dataset)
    outcomes = _building_outcomes(dataset, recovery_daily, weight_training)
    tables = {
        "Building Outcomes": outcomes,
        "Recovery Training": recovery_training,
        "Weight Training": weight_training,
        "Recovery Daily Audit": recovery_daily.rename(columns={"target": "final_recovery_proxy_y"}),
    }
    dictionary = _dictionary(tables)
    return {
        "summary": {
            "source_workbook": dataset.source_name,
            "canonical_building_day_rows": int(len(dataset.daily)),
            "recovery_building_outcomes": int(
                outcomes["eligible_recovery_model"].sum()
            ),
            "day35_weight_building_outcomes": int(
                outcomes["eligible_day35_weight_model"].sum()
            ),
            "recovery_training_rows": int(len(recovery_training)),
            "weight_training_rows": int(len(weight_training)),
            "recovery_daily_audit_rows": int(len(recovery_daily)),
            "recovery_champion_features": list(RECOVERY_CORE_FEATURE_COLUMNS),
        },
        "tables": {
            "Data Dictionary": _serializable_records(dictionary),
            **{name: _serializable_records(frame) for name, frame in tables.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
