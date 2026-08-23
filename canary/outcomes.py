"""Owner-facing historical outcome summaries for completed capstone cycles."""

from __future__ import annotations

import pandas as pd

from .data import CanaryDataset
from .modeling import _eligible_final_weight_labels


def latest_cycle_id(dataset: CanaryDataset) -> str:
    """Return the cycle with the latest placement date in the workbook."""

    starts = dataset.cycles.groupby("cycle_id")["start_date"].min().sort_values()
    if starts.empty:
        raise ValueError("The farm workbook does not contain any harvest cycles.")
    return str(starts.index[-1])


def build_historical_outcomes(
    dataset: CanaryDataset,
    final_weight_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one actual-outcome row per recorded historical building-cycle.

    For the capstone, the last recorded building date is treated as the
    completion date and recovery is ending population divided by beginning
    population. Final sale weight remains missing unless Farm Performance
    Summary contains a defensible building-cycle match. A separate display
    field falls back to the recorded Day 35 weight (or, if Day 35 is absent,
    the last recorded weight) without relabeling it as a final sale weight.
    """

    outcomes = dataset.cycles[
        [
            "cycle_id",
            "building_id",
            "start_date",
            "end_date",
            "beginning_inventory",
            "ending_inventory",
            "final_recovery_rate",
        ]
    ].copy()
    outcomes = outcomes.rename(
        columns={
            "end_date": "completion_date",
            "ending_inventory": "actual_ending_population",
            "final_recovery_rate": "actual_harvest_recovery",
        }
    )
    outcomes["actual_final_average_weight_kg"] = pd.NA
    outcomes["actual_final_weight_status"] = "Not available in Farm Performance Summary"
    outcomes["actual_final_weight_source"] = pd.NA

    measured = dataset.daily.loc[
        dataset.daily["weight_measured"].fillna(False).astype(bool)
        & dataset.daily["bodyweight_kg"].notna(),
        ["cycle_id", "building_id", "age_day", "record_date", "bodyweight_kg"],
    ].copy()
    if not measured.empty:
        measured = measured.sort_values(
            ["cycle_id", "building_id", "age_day", "record_date"]
        )
        latest = measured.drop_duplicates(
            ["cycle_id", "building_id"], keep="last"
        ).rename(
            columns={
                "age_day": "last_recorded_weight_day",
                "record_date": "last_recorded_weight_date",
                "bodyweight_kg": "last_recorded_weight_kg",
            }
        )
        day35 = measured.loc[measured["age_day"].eq(35)].drop_duplicates(
            ["cycle_id", "building_id"], keep="last"
        ).rename(
            columns={
                "record_date": "recorded_day35_weight_date",
                "bodyweight_kg": "recorded_day35_weight_kg",
            }
        )
        outcomes = outcomes.merge(
            latest[
                [
                    "cycle_id",
                    "building_id",
                    "last_recorded_weight_day",
                    "last_recorded_weight_date",
                    "last_recorded_weight_kg",
                ]
            ],
            on=["cycle_id", "building_id"],
            how="left",
            validate="one_to_one",
        )
        outcomes = outcomes.merge(
            day35[
                [
                    "cycle_id",
                    "building_id",
                    "recorded_day35_weight_date",
                    "recorded_day35_weight_kg",
                ]
            ],
            on=["cycle_id", "building_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        for column in (
            "last_recorded_weight_day",
            "last_recorded_weight_date",
            "last_recorded_weight_kg",
            "recorded_day35_weight_date",
            "recorded_day35_weight_kg",
        ):
            outcomes[column] = pd.NA

    if final_weight_labels is not None and not final_weight_labels.empty:
        matched = _eligible_final_weight_labels(dataset, final_weight_labels)[
            [
                "cycle_id",
                "building_id",
                "final_average_weight_kg",
                "weight_label_source",
                "weight_label_valid",
            ]
        ]
        outcomes = outcomes.merge(
            matched,
            on=["cycle_id", "building_id"],
            how="left",
            validate="one_to_one",
        )
        valid = outcomes["weight_label_valid"].fillna(False).astype(bool)
        suspicious = outcomes["final_average_weight_kg"].notna() & ~valid
        outcomes.loc[valid, "actual_final_average_weight_kg"] = outcomes.loc[
            valid, "final_average_weight_kg"
        ]
        outcomes.loc[valid, "actual_final_weight_status"] = "Recorded final average weight"
        outcomes.loc[valid, "actual_final_weight_source"] = outcomes.loc[
            valid, "weight_label_source"
        ]
        outcomes.loc[
            suspicious, "actual_final_weight_status"
        ] = "Source value excluded — building-cycle date needs validation"
        outcomes = outcomes.drop(
            columns=["final_average_weight_kg", "weight_label_source", "weight_label_valid"]
        )

    outcomes["display_recorded_weight_kg"] = outcomes[
        "actual_final_average_weight_kg"
    ]
    outcomes["display_recorded_weight_label"] = "Actual final average weight (g)"
    outcomes["display_recorded_weight_status"] = outcomes[
        "actual_final_weight_status"
    ]

    no_final = outcomes["actual_final_average_weight_kg"].isna()
    has_day35 = no_final & outcomes["recorded_day35_weight_kg"].notna()
    outcomes.loc[has_day35, "display_recorded_weight_kg"] = outcomes.loc[
        has_day35, "recorded_day35_weight_kg"
    ]
    outcomes.loc[
        has_day35, "display_recorded_weight_label"
    ] = "Recorded Day 35 weight (g)"
    outcomes.loc[
        has_day35, "display_recorded_weight_status"
    ] = "Day 35 management milestone; not a final sale-weight record"

    has_latest = (
        no_final
        & ~has_day35
        & outcomes["last_recorded_weight_kg"].notna()
    )
    outcomes.loc[has_latest, "display_recorded_weight_kg"] = outcomes.loc[
        has_latest, "last_recorded_weight_kg"
    ]
    outcomes.loc[has_latest, "display_recorded_weight_label"] = outcomes.loc[
        has_latest, "last_recorded_weight_day"
    ].map(lambda day: f"Last recorded weight — Day {int(day)} (g)")
    outcomes.loc[
        has_latest, "display_recorded_weight_status"
    ] = "Latest recorded milestone weight; not a final sale-weight record"

    return outcomes.sort_values(["cycle_id", "building_id"]).reset_index(drop=True)


def attach_historical_day14_backtests(
    outcomes: pd.DataFrame,
    recovery_manifest: dict,
    day35_manifest: dict,
) -> pd.DataFrame:
    """Attach cycle-held-out Day 14 backtests to completed building results."""

    result = outcomes.copy()
    recovery = pd.DataFrame(recovery_manifest.get("day14_backtest", []))
    if not recovery.empty:
        recovery = recovery.rename(
            columns={
                "predicted": "day14_projected_recovery",
                "actual": "day14_actual_recovery_proxy",
                "error": "day14_recovery_error",
                "absolute_error": "day14_recovery_absolute_error",
            }
        ).drop(columns=["as_of_date"], errors="ignore")
        result = result.merge(
            recovery,
            on=["cycle_id", "building_id"],
            how="left",
            validate="one_to_one",
        )

    weight = pd.DataFrame(day35_manifest.get("day14_backtest", []))
    if not weight.empty:
        weight = weight.rename(
            columns={
                "current_weight_kg": "day14_measured_weight_kg",
                "predicted_day35_weight_kg": "day14_projected_day35_weight_kg",
                "actual_day35_weight_kg": "day14_actual_day35_weight_kg",
                "error_kg": "day14_weight_error_kg",
                "absolute_error_kg": "day14_weight_absolute_error_kg",
            }
        )
        result = result.merge(
            weight,
            on=["cycle_id", "building_id"],
            how="left",
            validate="one_to_one",
        )
    return result
