"""All-cycle owner reporting for Project Canary.

The harvest-analysis table deliberately keeps three concepts separate:

* recorded historical outcome proxies;
* current-cycle observations and projections; and
* eligibility for model training.

This prevents a recorded workbook endpoint from being presented as a verified
harvest event and prevents repeated decision snapshots from being mistaken for
independent flock outcomes.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .data import CanaryDataset
from .outcomes import build_historical_outcomes, latest_cycle_id
from .state import CANONICAL_BUILDINGS


RECOVERY_TARGET = 0.95
DAY35_WEIGHT_TARGET_KG = 1.8


def _training_pairs(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    """Return the exact independent building-cycle outcomes in a manifest."""

    return {
        (str(row["cycle_id"]), str(row["building_id"]))
        for row in manifest.get("day14_backtest", [])
    }


def _recorded_day35_weights(dataset: CanaryDataset) -> pd.DataFrame:
    """Return one observed Day 35 weight and contemporaneous population per flock."""

    rows = dataset.daily.loc[
        dataset.daily["weight_measured"]
        & dataset.daily["age_day"].eq(35),
        ["cycle_id", "building_id", "record_date", "bodyweight_kg", "population"],
    ].sort_values("record_date")
    rows = rows.drop_duplicates(["cycle_id", "building_id"], keep="last")
    return rows.rename(
        columns={
            "record_date": "day35_measurement_date",
            "bodyweight_kg": "recorded_day35_weight_kg",
            "population": "day35_population",
        }
    )


def build_harvest_analysis_rows(
    dataset: CanaryDataset,
    final_weight_labels: pd.DataFrame | None,
    current_snapshot: pd.DataFrame,
    recovery_manifest: dict[str, Any],
    day35_manifest: dict[str, Any],
    *,
    include_latest_as_current: bool = True,
) -> pd.DataFrame:
    """Return one transparent reporting row for every cycle and physical building."""

    cycle_starts = (
        dataset.cycles.groupby("cycle_id")["start_date"].min().sort_values()
    )
    cycles = cycle_starts.index.astype(str).tolist()
    current_cycle = latest_cycle_id(dataset)
    grid = pd.DataFrame(
        product(cycles, CANONICAL_BUILDINGS),
        columns=["cycle_id", "building_id"],
    )
    grid["building_order"] = grid["building_id"].map(
        {building: order for order, building in enumerate(CANONICAL_BUILDINGS)}
    )
    grid["cycle_order"] = grid["cycle_id"].map(
        {cycle: order for order, cycle in enumerate(cycles)}
    )

    metadata = dataset.cycles[
        [
            "cycle_id",
            "building_id",
            "start_date",
            "end_date",
            "beginning_inventory",
            "ending_inventory",
            "final_recovery_rate",
        ]
    ].rename(
        columns={
            "end_date": "last_recorded_date",
            "ending_inventory": "recorded_ending_population",
            "final_recovery_rate": "recorded_recovery_proxy",
        }
    )
    result = grid.merge(
        metadata, on=["cycle_id", "building_id"], how="left", validate="one_to_one"
    )

    historical = build_historical_outcomes(dataset, final_weight_labels)[
        [
            "cycle_id",
            "building_id",
            "actual_final_average_weight_kg",
            "actual_final_weight_status",
        ]
    ]
    result = result.merge(
        historical,
        on=["cycle_id", "building_id"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        _recorded_day35_weights(dataset),
        on=["cycle_id", "building_id"],
        how="left",
        validate="one_to_one",
    )

    current_columns = [
        "cycle_id",
        "building_id",
        "state",
        "as_of_date",
        "cycle_day",
        "latest_population",
        "percentage_alive",
        "latest_weight_kg",
        "weight_measurement_day",
        "predicted_final_recovery",
        "recovery_target_gap_pp",
        "projected_day35_weight_kg",
        "day35_weight_target_gap_kg",
        "recovery_forecast_status",
        "day35_weight_status",
    ]
    available = [column for column in current_columns if column in current_snapshot]
    current = current_snapshot[available].copy()
    current = current.rename(columns={"state": "current_source_state"})
    result = result.merge(
        current,
        on=["cycle_id", "building_id"],
        how="left",
        validate="one_to_one",
    )
    # A reset baseline supplies a historical snapshot rather than a forecast-
    # enriched current snapshot. Keep the reporting schema stable and leave
    # current-only fields empty instead of treating history as live output.
    for column in (
        "percentage_alive",
        "latest_population",
        "latest_weight_kg",
        "predicted_final_recovery",
        "projected_day35_weight_kg",
        "recovery_target_gap_pp",
        "day35_weight_target_gap_kg",
    ):
        if column not in result:
            result[column] = pd.NA

    has_record = result["start_date"].notna()
    is_current = result["cycle_id"].eq(current_cycle) & include_latest_as_current
    result["reporting_status"] = "No building data"
    result.loc[has_record & ~is_current, "reporting_status"] = "Historical records ended"
    active_state = result["current_source_state"].isin(["Active", "Incomplete"])
    result.loc[has_record & is_current & active_state, "reporting_status"] = "Current flock"
    result.loc[
        has_record & is_current & result["current_source_state"].eq("Records ended"),
        "reporting_status",
    ] = "Current records ended"
    result.loc[
        has_record & is_current & result["current_source_state"].eq("Inactive"),
        "reporting_status",
    ] = "Not started on review date"

    # Historical rows display recorded evidence only. Current rows display current
    # observations and forecasts only, even when the workbook contains future rows.
    historical_record = has_record & ~is_current
    result["historical_recovery_proxy"] = result["recorded_recovery_proxy"].where(
        historical_record
    )
    result["historical_day35_weight_kg"] = result["recorded_day35_weight_kg"].where(
        historical_record
    )
    result["historical_final_average_weight_kg"] = result[
        "actual_final_average_weight_kg"
    ].where(historical_record)
    result["current_survival"] = result["percentage_alive"].where(is_current)
    result["current_population"] = result["latest_population"].where(is_current)
    result["current_latest_weight_kg"] = result["latest_weight_kg"].where(is_current)
    result["projected_recovery"] = result["predicted_final_recovery"].where(is_current)
    result["projected_day35_weight_kg"] = result[
        "projected_day35_weight_kg"
    ].where(is_current)

    recovery_pairs = _training_pairs(recovery_manifest)
    weight_pairs = _training_pairs(day35_manifest)
    pair_series = list(zip(result["cycle_id"], result["building_id"]))
    result["recovery_training_eligible"] = [pair in recovery_pairs for pair in pair_series]
    result["weight_training_eligible"] = [pair in weight_pairs for pair in pair_series]
    result["model_training_eligibility"] = np.select(
        [
            result["recovery_training_eligible"] & result["weight_training_eligible"],
            result["recovery_training_eligible"],
            result["weight_training_eligible"],
        ],
        ["Recovery and weight", "Recovery only", "Weight only"],
        default="Not used",
    )

    notes: list[str] = []
    for _, row in result.iterrows():
        messages: list[str] = []
        if row["reporting_status"] == "No building data":
            messages.append("No building data for this cycle")
        elif row["cycle_id"] == "2026-2" and not row["recovery_training_eligible"]:
            messages.append(
                "Recorded recovery endpoint is incomplete; excluded from recovery training"
            )
        if (
            row["reporting_status"] == "Historical records ended"
            and pd.isna(row["historical_day35_weight_kg"])
        ):
            messages.append("No observed Day 35 weight")
        if (
            row["reporting_status"] == "Current flock"
            and pd.isna(row["projected_day35_weight_kg"])
        ):
            messages.append("No current Day 35 weight projection")
        notes.append("; ".join(messages) if messages else "No material data warning")
    result["data_quality_note"] = notes

    result["recovery_gap_to_95_pp"] = np.where(
        historical_record,
        (result["historical_recovery_proxy"] - RECOVERY_TARGET) * 100,
        result["recovery_target_gap_pp"],
    )
    result["weight_gap_to_1800_g"] = np.where(
        historical_record,
        (result["historical_day35_weight_kg"] - DAY35_WEIGHT_TARGET_KG) * 1000,
        result["day35_weight_target_gap_kg"] * 1000,
    )

    return result.sort_values(["cycle_order", "building_order"]).reset_index(drop=True)


def summarize_harvest_analysis(rows: pd.DataFrame) -> dict[str, object]:
    """Calculate owner-facing KPIs from a filtered harvest-analysis table."""

    recorded = rows.loc[rows["start_date"].notna()]
    historical = recorded.loc[recorded["reporting_status"].eq("Historical records ended")]
    recovery = historical.dropna(
        subset=["beginning_inventory", "recorded_ending_population"]
    )
    recovery = recovery.loc[recovery["beginning_inventory"] > 0]
    historical_recovery = (
        float(recovery["recorded_ending_population"].sum() / recovery["beginning_inventory"].sum())
        if not recovery.empty and recovery["beginning_inventory"].sum() > 0
        else np.nan
    )

    weights = historical.dropna(subset=["historical_day35_weight_kg"])
    if not weights.empty:
        weight_denominator = weights["day35_population"].fillna(
            weights["beginning_inventory"]
        )
        valid = weight_denominator.notna() & weight_denominator.gt(0)
        historical_weight = (
            float(
                (weights.loc[valid, "historical_day35_weight_kg"] * weight_denominator[valid]).sum()
                / weight_denominator[valid].sum()
            )
            if valid.any()
            else float(weights["historical_day35_weight_kg"].mean())
        )
    else:
        historical_weight = np.nan

    current = recorded.loc[recorded["reporting_status"].eq("Current flock")]
    current_recovery_rows = current.dropna(
        subset=["projected_recovery", "beginning_inventory"]
    )
    current_recovery = (
        float(
            (
                current_recovery_rows["projected_recovery"]
                * current_recovery_rows["beginning_inventory"]
            ).sum()
            / current_recovery_rows["beginning_inventory"].sum()
        )
        if not current_recovery_rows.empty
        and current_recovery_rows["beginning_inventory"].sum() > 0
        else np.nan
    )
    current_weight_rows = current.dropna(
        subset=["projected_day35_weight_kg", "current_population"]
    )
    current_weight = (
        float(
            (
                current_weight_rows["projected_day35_weight_kg"]
                * current_weight_rows["current_population"]
            ).sum()
            / current_weight_rows["current_population"].sum()
        )
        if not current_weight_rows.empty
        and current_weight_rows["current_population"].sum() > 0
        else np.nan
    )
    return {
        "cycles": int(recorded["cycle_id"].nunique()),
        "building_records": int(len(recorded)),
        "historical_recovery": historical_recovery,
        "historical_recovery_buildings": int(len(recovery)),
        "historical_day35_weight_kg": historical_weight,
        "historical_day35_weight_buildings": int(len(weights)),
        "current_projected_recovery": current_recovery,
        "current_recovery_buildings": int(len(current_recovery_rows)),
        "current_projected_day35_weight_kg": current_weight,
        "current_weight_buildings": int(len(current_weight_rows)),
    }


def recovery_cycle_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Return inventory-weighted recovery by cycle for charting."""

    records: list[dict[str, object]] = []
    for cycle_id, group in rows.groupby("cycle_id", sort=False):
        historical = group.loc[
            group["reporting_status"].eq("Historical records ended")
            & group["beginning_inventory"].notna()
            & group["recorded_ending_population"].notna()
        ]
        current = group.loc[
            group["reporting_status"].eq("Current flock")
            & group["beginning_inventory"].notna()
            & group["projected_recovery"].notna()
        ]
        if not historical.empty and historical["beginning_inventory"].sum() > 0:
            value = historical["recorded_ending_population"].sum() / historical[
                "beginning_inventory"
            ].sum()
            records.append(
                {
                    "cycle_id": cycle_id,
                    "recovery": float(value),
                    "result_type": "Recorded historical proxy",
                    "buildings": int(len(historical)),
                }
            )
        elif not current.empty and current["beginning_inventory"].sum() > 0:
            value = (
                current["projected_recovery"] * current["beginning_inventory"]
            ).sum() / current["beginning_inventory"].sum()
            records.append(
                {
                    "cycle_id": cycle_id,
                    "recovery": float(value),
                    "result_type": "Current projection",
                    "buildings": int(len(current)),
                }
            )
    return pd.DataFrame(records)


def weight_cycle_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Return bird-count-weighted Day 35 weight by cycle for charting."""

    records: list[dict[str, object]] = []
    for cycle_id, group in rows.groupby("cycle_id", sort=False):
        historical = group.loc[
            group["reporting_status"].eq("Historical records ended")
            & group["historical_day35_weight_kg"].notna()
        ].copy()
        current = group.loc[
            group["reporting_status"].eq("Current flock")
            & group["projected_day35_weight_kg"].notna()
        ].copy()
        if not historical.empty:
            historical["_weight"] = historical["day35_population"].fillna(
                historical["beginning_inventory"]
            )
            valid = historical["_weight"].notna() & historical["_weight"].gt(0)
            value = (
                (historical.loc[valid, "historical_day35_weight_kg"] * historical.loc[valid, "_weight"]).sum()
                / historical.loc[valid, "_weight"].sum()
                if valid.any()
                else historical["historical_day35_weight_kg"].mean()
            )
            records.append(
                {
                    "cycle_id": cycle_id,
                    "weight_kg": float(value),
                    "result_type": "Recorded Day 35",
                    "buildings": int(len(historical)),
                }
            )
        elif not current.empty:
            current["_weight"] = current["current_population"].fillna(
                current["beginning_inventory"]
            )
            valid = current["_weight"].notna() & current["_weight"].gt(0)
            value = (
                (current.loc[valid, "projected_day35_weight_kg"] * current.loc[valid, "_weight"]).sum()
                / current.loc[valid, "_weight"].sum()
                if valid.any()
                else current["projected_day35_weight_kg"].mean()
            )
            records.append(
                {
                    "cycle_id": cycle_id,
                    "weight_kg": float(value),
                    "result_type": "Current projection",
                    "buildings": int(len(current)),
                }
            )
    return pd.DataFrame(records)
