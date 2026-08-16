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
    extract_feature_row,
    load_model_bundle,
    load_workbook,
)
from canary.forecast import _predict
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

    # Keep the genuinely later cycle visible as an outcome-level audit even
    # though it is deliberately excluded from fitting and champion selection.
    # Previously the export only built its key set from the two training
    # matrices, which made 2026-3 look as if it had been overlooked.
    latest_start = pd.to_datetime(dataset.cycles["start_date"]).max()
    latest_cycles = set(
        dataset.cycles.loc[
            pd.to_datetime(dataset.cycles["start_date"]).eq(latest_start),
            "cycle_id",
        ].astype(str)
    )
    latest_cycle_rows = dataset.cycles.loc[
        dataset.cycles["cycle_id"].astype(str).isin(latest_cycles)
    ].copy()
    if not latest_cycle_rows.empty:
        latest_weights = dataset.daily.loc[
            dataset.daily["cycle_id"].astype(str).isin(latest_cycles)
            & dataset.daily["weight_measured"].eq(True)
            & dataset.daily["age_day"].isin([7, 14, 21, 28, 35]),
            ["cycle_id", "building_id", "age_day", "bodyweight_kg"],
        ].copy()
        latest_wide = (
            latest_weights.pivot_table(
                index=["cycle_id", "building_id"],
                columns="age_day",
                values="bodyweight_kg",
                aggfunc="last",
            )
            .rename(columns={day: f"day_{day}_weight_g" for day in (7, 14, 21, 28, 35)})
            .reset_index()
        )
        for column in [f"day_{day}_weight_g" for day in (7, 14, 21, 28, 35)]:
            if column in latest_wide:
                latest_wide[column] = latest_wide[column] * 1000
        latest_outcomes = latest_cycle_rows[
            [
                "cycle_id",
                "building_id",
                "beginning_inventory",
                "ending_inventory",
                "final_recovery_rate",
                "end_date",
            ]
        ].rename(
            columns={
                "ending_inventory": "ending_population",
                "final_recovery_rate": "final_recovery_proxy",
                "end_date": "recovery_label_date",
            }
        )
        latest_outcomes = latest_outcomes.merge(
            latest_wide, on=["cycle_id", "building_id"], how="left"
        )
        latest_outcomes["eligible_recovery_model"] = False
        latest_outcomes["eligible_day35_weight_model"] = False
        existing = set(zip(outcomes["cycle_id"].astype(str), outcomes["building_id"]))
        latest_outcomes = latest_outcomes.loc[
            ~latest_outcomes.apply(
                lambda row: (str(row["cycle_id"]), row["building_id"]) in existing,
                axis=1,
            )
        ]
        outcomes = pd.concat([outcomes, latest_outcomes], ignore_index=True, sort=False)

    outcomes["recovery_role"] = np.where(
        outcomes["eligible_recovery_model"],
        "Historical training outcome",
        "Prospective audit candidate - excluded from fitting; endpoint remains a last-recorded proxy",
    )
    outcomes["day35_weight_role"] = np.where(
        outcomes["eligible_day35_weight_model"],
        "Historical training outcome",
        "Prospective audit only - excluded from fitting and champion selection",
    )
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
        "recovery_role",
        "day35_weight_role",
    ]
    return outcomes.reindex(columns=ordered).sort_values(
        ["cycle_id", "building_id"]
    ).reset_index(drop=True)


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
    result["additional_population_loss_y"] = (
        result["percentage_alive"] - result["final_recovery_proxy_y"]
    ).clip(lower=0)
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
        if column not in {*identifiers, "target"}
    ]
    targets = ["additional_population_loss_y", "final_recovery_proxy_y"]
    return result[identifiers + targets + feature_columns]


def _weight_training(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the Day 35 label beside the identifiers so its X/Y role is unmistakable."""

    result = frame.copy()
    result["prediction_day"] = 35
    result["snapshot_role"] = (
        "X values known at checkpoint; Day 35 weight is Y and is never included in X"
    )
    identifiers = [
        "cycle_id",
        "building_id",
        "measurement_day",
        "prediction_day",
        "validation_cycle",
        "snapshot_role",
    ]
    targets = ["actual_day35_weight_kg_y", "remaining_gain_to_day35_kg_y"]
    remaining = [
        column for column in result.columns if column not in {*identifiers, *targets}
    ]
    return result[identifiers + targets + remaining]


def _latest_recovery_audit(dataset) -> pd.DataFrame:
    """Score the later cycle once with the already-frozen recovery pipeline."""

    latest_start = pd.to_datetime(dataset.cycles["start_date"]).max()
    latest = dataset.cycles.loc[
        pd.to_datetime(dataset.cycles["start_date"]).eq(latest_start)
    ]
    manifest, model = load_model_bundle("recovery")
    rows: list[dict[str, object]] = []
    for outcome in latest.itertuples(index=False):
        for day in (7, 14, 21, 28):
            as_of = pd.Timestamp(outcome.start_date) + pd.Timedelta(days=day - 1)
            feature = extract_feature_row(
                dataset, str(outcome.cycle_id), str(outcome.building_id), as_of
            )
            if feature is None:
                continue
            predicted = _predict(feature, "recovery", manifest, model)
            actual = float(outcome.final_recovery_rate)
            audit_row = {
                    "cycle_id": str(outcome.cycle_id),
                    "building_id": str(outcome.building_id),
                    "review_day": day,
                    "as_of_date": as_of,
                    "current_percentage_alive": float(feature["percentage_alive"]),
                    "additional_population_loss_y": float(
                        feature["percentage_alive"] - actual
                    ),
                    "predicted_final_recovery": predicted,
                    "last_recorded_recovery_proxy_y": actual,
                    "error_percentage_points": (predicted - actual) * 100,
                    "absolute_error_percentage_points": abs(predicted - actual) * 100,
                    "audit_role": "Prospective audit only - excluded from fitting, preprocessing, tuning and champion selection",
                    "endpoint_warning": "Last-recorded population proxy; not a verified harvest-event endpoint",
                }
            audit_row.update(
                {
                    column: feature.get(column, np.nan)
                    for column in RECOVERY_CORE_FEATURE_COLUMNS
                }
            )
            rows.append(audit_row)
    return pd.DataFrame(rows)


def _how_to_read() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sheet": "Building Outcomes",
                "one_row_represents": "One recorded building-cycle, including the later 2026-3 audit cycle",
                "inputs_x": "Not a fitted training matrix; outcome-level audit view",
                "target_y": "Final recovery proxy and/or recorded Day 35 weight",
                "why_this_shape": "Shows 31 historical fitting outcomes plus three later 2026-3 audit outcomes. Repeated snapshots do not create new independent flocks.",
            },
            {
                "sheet": "Recovery Training",
                "one_row_represents": "One building-cycle as seen at a historical review date",
                "inputs_x": "Only operational values known by the as-of date",
                "target_y": "Additional loss after that date; final recovery proxy is retained for audit",
                "why_this_shape": "Days 7/14/21/28 test repeatable checkpoints. The separate latest pre-outcome row tests a near-end forecast and may occur on Day 23–48 depending on source coverage.",
            },
            {
                "sheet": "Weight Training",
                "one_row_represents": "One building-cycle at Day 7, 14, 21, or 28",
                "inputs_x": "Weights and operating signals available at that checkpoint; future checkpoint weights are blank",
                "target_y": "Recorded Day 35 average weight and remaining gain to Day 35",
                "why_this_shape": "Day 35 is deliberately the answer, not an input. Including it among X features would leak the result the model is meant to predict.",
            },
            {
                "sheet": "Latest Cycle Weight Audit",
                "one_row_represents": "One 2026-3 building at a historical checkpoint",
                "inputs_x": "Checkpoint information available in the latest cycle",
                "target_y": "Now-recorded Day 35 weight",
                "why_this_shape": "Prospective audit only; 2026-3 remains outside fitting and champion selection.",
            },
            {
                "sheet": "Latest Recovery Audit",
                "one_row_represents": "One 2026-3 building at Day 7, 14, 21 or 28",
                "inputs_x": "Only information recorded by that checkpoint",
                "target_y": "The later cycle's last-recorded recovery proxy",
                "why_this_shape": "A genuinely later prospective score of the frozen model. The endpoint remains provisional because no verified harvest event is recorded.",
            },
            {
                "sheet": "Recovery Daily Audit",
                "one_row_represents": "Every eligible building-day before the proxy endpoint",
                "inputs_x": "Daily leakage-safe features",
                "target_y": "Final recovery proxy",
                "why_this_shape": "Provides full lineage and lets reviewers inspect dates not retained in the balanced training sheet.",
            },
        ]
    )


def _recovery_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    schedule = (
        frame.groupby(["row_selection", "cycle_day"], as_index=False)
        .agg(
            training_rows=("building_id", "size"),
            independent_building_cycles=("building_id", "nunique"),
            earliest_as_of_date=("as_of_date", "min"),
            latest_as_of_date=("as_of_date", "max"),
        )
        .sort_values(["row_selection", "cycle_day"])
    )
    schedule["plain_language_purpose"] = np.where(
        schedule["row_selection"].eq("Standard checkpoint"),
        "Repeatable management checkpoint used to compare forecast accuracy by age",
        "Last leakage-safe row before the recorded recovery-proxy endpoint; age varies by building",
    )
    return schedule


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
        "additional_population_loss_y": ("Y", "proportion", "Derived", "Current percentage alive minus the completed-cycle recovery proxy; redesigned recovery target"),
        "actual_day35_weight_kg_y": ("Y", "kg", "Farm workbook", "Observed average bodyweight recorded on production Day 35"),
        "remaining_gain_to_day35_kg_y": ("Y", "kg", "Derived", "Observed Day 35 weight minus current checkpoint weight; redesigned weight target"),
        "day_35_weight_g": ("Outcome", "g", "Farm workbook", "Observed average bodyweight on production Day 35"),
        "temperature_recent_avg_c": ("X", "°C", "Derived", "Mean of available recent temperature readings known by the review date"),
        "humidity_recent_avg_pct": ("X", "%", "Derived", "Mean of available recent humidity readings known by the review date"),
        "weight_gap_pct": ("X", "%", "Derived", "Latest observed weight shortfall versus the approved age target"),
        "weight_staleness_days": ("X", "days", "Derived", "Days since the latest observed weight"),
        "temperature_deviation_from_band_c": ("X", "°C", "Derived", "Distance outside the approved tropical age band; zero when inside"),
        "humidity_deviation_from_band_pp": ("X", "percentage points", "Derived", "Distance outside the approved humidity age band; zero when inside"),
        "environment_out_of_band_days_7d": ("X", "recorded days", "Derived", "Number of the latest seven recorded environment days outside either approved band"),
        "environment_staleness_days": ("X", "days", "Derived", "Days since the latest recorded temperature or humidity reading"),
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
    weight_training_raw = build_day35_feature_rows(dataset)
    weight_training_raw["remaining_gain_to_day35_kg_y"] = (
        weight_training_raw["actual_day35_weight_kg_y"]
        - weight_training_raw["current_weight_kg"]
    )
    all_weight_rows = build_day35_feature_rows(
        dataset,
        include_latest_cycle=True,
    )
    training_cycles = set(weight_training_raw["cycle_id"].astype(str))
    latest_weight_audit = all_weight_rows.loc[
        ~all_weight_rows["cycle_id"].astype(str).isin(training_cycles)
    ].copy()
    if not latest_weight_audit.empty:
        latest_weight_audit["remaining_gain_to_day35_kg_y"] = (
            latest_weight_audit["actual_day35_weight_kg_y"]
            - latest_weight_audit["current_weight_kg"]
        )
        latest_weight_audit["training_role"] = (
            "Prospective audit only - excluded from model fitting and champion selection"
        )
        latest_weight_audit = _weight_training(latest_weight_audit)
    weight_training = _weight_training(weight_training_raw)
    latest_recovery_audit = _latest_recovery_audit(dataset)
    outcomes = _building_outcomes(dataset, recovery_daily, weight_training_raw)
    tables = {
        "How to Read": _how_to_read(),
        "Building Outcomes": outcomes,
        "Recovery Training": recovery_training,
        "Recovery Schedule": _recovery_schedule(recovery_training),
        "Weight Training": weight_training,
        "Latest Cycle Weight Audit": latest_weight_audit,
        "Latest Recovery Audit": latest_recovery_audit,
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
            "latest_cycle_day35_audit_rows": int(len(latest_weight_audit)),
            "latest_cycle_day35_audit_outcomes": int(
                latest_weight_audit[["cycle_id", "building_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "total_recorded_building_outcomes": int(len(outcomes)),
            "latest_cycle_recovery_audit_candidates": int(
                (
                    ~outcomes["eligible_recovery_model"]
                    & outcomes["final_recovery_proxy"].notna()
                ).sum()
            ),
            "latest_cycle_recovery_audit_rows": int(len(latest_recovery_audit)),
            "latest_cycle_recovery_audit_mae_pp": float(
                latest_recovery_audit["absolute_error_percentage_points"].mean()
            ),
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
