"""Independent capstone-readiness audit for Canary's data and model choices."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from canary import load_workbook
from canary.day35 import (
    CHECKPOINT_DAYS,
    _ridge_feature_frame,
    build_day35_training_rows,
)
from canary.modeling import (
    FEATURE_COLUMNS,
    RECOVERY_NO_WEIGHT_FEATURE_COLUMNS,
    _decision_checkpoint_snapshots,
    build_modeling_snapshots,
)


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "FARM HARVEST DATA.xlsx"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "bias": float(np.mean(predicted - actual)),
    }


def _recovery_pipeline(kind: str) -> object:
    if kind == "ridge":
        regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    if kind == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def recovery_ablation(dataset) -> dict[str, object]:
    snapshots = _decision_checkpoint_snapshots(build_modeling_snapshots(dataset, "recovery"))
    groups = snapshots["cycle_id"].astype(str).to_numpy()
    actual = snapshots["target"].to_numpy(float)
    no_inventory = [
        column for column in RECOVERY_NO_WEIGHT_FEATURE_COLUMNS
        if column not in {"beginning_inventory", "is_lags_building"}
    ]
    core = [
        "cycle_day",
        "percentage_alive",
        "mortality_daily_per_1000",
        "mortality_recent_3d_per_1000",
        "mortality_trend_delta_per_1000",
        "feed_daily_per_1000_birds",
        "feed_cumulative_per_1000_birds",
        "temperature_recent_avg_c",
        "humidity_recent_avg_pct",
    ]
    outcome_only = [
        "cycle_day",
        "percentage_alive",
        "mortality_daily_per_1000",
        "mortality_recent_3d_per_1000",
        "mortality_trend_delta_per_1000",
    ]
    no_environment = [
        *outcome_only,
        "feed_daily_per_1000_birds",
        "feed_cumulative_per_1000_birds",
    ]
    candidate_specs = {
        "current_survival": (None, ["percentage_alive"]),
        "ridge_current": ("ridge", RECOVERY_NO_WEIGHT_FEATURE_COLUMNS),
        "ridge_no_inventory_or_group": ("ridge", no_inventory),
        "ridge_core": ("ridge", core),
        "ridge_no_environment": ("ridge", no_environment),
        "ridge_outcome_only": ("ridge", outcome_only),
        "random_forest_core": ("random_forest", core),
    }
    predictions = {name: np.full(len(snapshots), np.nan) for name in candidate_specs}
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(snapshots, actual, groups):
        for name, (kind, columns) in candidate_specs.items():
            if name == "current_survival":
                prediction = snapshots.iloc[test_index]["percentage_alive"].to_numpy(float)
            else:
                model = _recovery_pipeline(str(kind))
                model.fit(snapshots.iloc[train_index][columns], actual[train_index])
                prediction = model.predict(snapshots.iloc[test_index][columns])
            predictions[name][test_index] = np.clip(prediction, 0.0, 1.0)

    results: dict[str, object] = {}
    for name, prediction in predictions.items():
        cycle_mae = [
            mean_absolute_error(actual[groups == cycle], prediction[groups == cycle])
            for cycle in np.unique(groups)
        ]
        day14 = snapshots["cycle_day"].eq(14).to_numpy()
        results[name] = {
            **_metrics(actual, prediction),
            "cycle_macro_mae": float(np.mean(cycle_mae)),
            "day14_mae": float(mean_absolute_error(actual[day14], prediction[day14])),
            "target_side_accuracy": float(np.mean((prediction >= 0.95) == (actual >= 0.95))),
            "actual_target_hits": int(np.sum(actual >= 0.95)),
            "predicted_target_hits": int(np.sum(prediction >= 0.95)),
        }
    return {
        "building_outcomes": int(snapshots[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "checkpoint_rows": int(len(snapshots)),
        "cycles": sorted(snapshots["cycle_id"].unique().tolist()),
        "candidates": results,
    }


def weight_tree_benchmark(dataset) -> dict[str, object]:
    rows = build_day35_training_rows(dataset)
    target_by_age = dataset.targets.set_index("age_day")["target_weight_kg"]
    features = _ridge_feature_frame(rows, target_by_age)
    actual = rows["actual_day35_weight_kg"].to_numpy(float)
    groups = rows["cycle_id"].astype(str).to_numpy()
    candidate_specs = {
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    predictions = {name: np.full(len(rows), np.nan) for name in candidate_specs}
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(features, actual, groups):
        for name, estimator in candidate_specs.items():
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                    ("model", estimator),
                ]
            )
            model.fit(features.iloc[train_index], actual[train_index])
            predictions[name][test_index] = np.clip(model.predict(features.iloc[test_index]), 0.1, 3.5)

    results: dict[str, object] = {}
    for name, prediction in predictions.items():
        cycle_mae = [
            mean_absolute_error(actual[groups == cycle], prediction[groups == cycle])
            for cycle in np.unique(groups)
        ]
        horizon = {}
        for checkpoint in CHECKPOINT_DAYS:
            mask = rows["measurement_day"].eq(checkpoint).to_numpy()
            horizon[f"Day {checkpoint}"] = _metrics(actual[mask], prediction[mask])
        results[name] = {
            **_metrics(actual, prediction),
            "cycle_macro_mae": float(np.mean(cycle_mae)),
            "within_200g_rate": float(np.mean(np.abs(prediction - actual) <= 0.2)),
            "horizon": horizon,
        }
    return {
        "building_outcomes": int(rows[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "checkpoint_rows": int(len(rows)),
        "cycles": sorted(rows["cycle_id"].unique().tolist()),
        "candidates": results,
    }


def duplicate_audit(dataset) -> dict[str, object]:
    raw = pd.read_excel(WORKBOOK, sheet_name="Farm Harvest Data (Daily)", engine="openpyxl")
    keys = ["Harvest Cycle", "Bldg.", "Age"]
    sizes = raw.groupby(keys, dropna=False).size().rename("source_rows").reset_index()
    duplicates = sizes.loc[sizes["source_rows"] > 1].copy()
    summary = (
        duplicates.groupby(["Harvest Cycle", "Bldg."], as_index=False)
        .agg(duplicate_days=("Age", "count"), first_day=("Age", "min"), last_day=("Age", "max"))
        .to_dict(orient="records")
    )
    examples = raw.merge(duplicates[keys].head(3), on=keys, how="inner")[
        [
            "Harvest Cycle",
            "Bldg.",
            "Age",
            "Date",
            "Beginning Inventory",
            "Population",
            "mortality_daily",
            "feedconsumption_daily",
            "Bodyweight (kgs)",
            "Min Temperature",
            "Max Temperature",
            "Average Temperature",
            "Min Humidity ",
            "Max Humidity",
            "Average Humidity",
        ]
    ]
    return {
        "source_rows": dataset.quality.source_rows,
        "canonical_rows": dataset.quality.canonical_rows,
        "duplicate_keys": dataset.quality.duplicate_keys,
        "rows_consolidated": dataset.quality.duplicate_rows_consolidated,
        "production_conflicts": dataset.quality.production_conflict_keys,
        "summary": summary,
        "examples": examples.astype(object).where(pd.notna(examples), None).to_dict(orient="records"),
    }


def main() -> None:
    dataset = load_workbook(WORKBOOK)
    payload = {
        "report": "Project Canary engine review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "survival_path_formula": "1 - (1 - 0.95) * min(age, 35) / 35",
        "data_quality": {
            "temperature_coverage_pct": dataset.quality.temperature_coverage_pct,
            "humidity_coverage_pct": dataset.quality.humidity_coverage_pct,
            "weight_measurement_days": dataset.quality.weight_measurement_days,
            "operationally_missing_days": dataset.quality.operationally_missing_days,
        },
        "duplicates": duplicate_audit(dataset),
        "recovery_ablation": recovery_ablation(dataset),
        "weight_tree_benchmark": weight_tree_benchmark(dataset),
    }
    destination = ROOT / "analysis" / "capstone_engine_review.json"
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(destination)
    print(json.dumps(payload["recovery_ablation"], indent=2))
    print(json.dumps(payload["weight_tree_benchmark"], indent=2))


if __name__ == "__main__":
    main()
