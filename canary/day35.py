"""Transparent Day 35 liveweight projection for Project Canary."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from .data import CanaryDataset


DAY35_TARGET_KG = 1.8
CHECKPOINT_DAYS = (7, 14, 21, 28)
DEFAULT_DAY35_MANIFEST = (
    Path(__file__).resolve().parent.parent / "models" / "day35_weight_manifest.json"
)
RIDGE_FEATURES = (
    "measurement_day",
    "current_weight_kg",
    "current_to_target_ratio",
    "recent_adg_kg_day",
    "has_recent_adg",
    "cumulative_adg_kg_day",
    "weight_day_7_kg",
    "weight_day_14_kg",
    "weight_day_21_kg",
    "weight_day_28_kg",
)


@lru_cache(maxsize=4)
def load_day35_manifest(
    path: str | Path = DEFAULT_DAY35_MANIFEST,
) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_day35_training_rows(dataset: CanaryDataset) -> pd.DataFrame:
    """Return one leakage-safe checkpoint row for each observed Day 35 weight."""

    cycle_starts = dataset.cycles.groupby("cycle_id")["start_date"].min()
    latest_cycle = str(cycle_starts.idxmax())
    completed = set(cycle_starts.index.astype(str)) - {latest_cycle}
    weights = dataset.daily.loc[
        dataset.daily["cycle_id"].isin(completed)
        & dataset.daily["weight_measured"]
        & dataset.daily["age_day"].isin([*CHECKPOINT_DAYS, 35]),
        ["cycle_id", "building_id", "age_day", "bodyweight_kg"],
    ].drop_duplicates(["cycle_id", "building_id", "age_day"])
    pivot = weights.pivot(
        index=["cycle_id", "building_id"],
        columns="age_day",
        values="bodyweight_kg",
    ).reset_index()
    if 35 not in pivot:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, record in pivot.dropna(subset=[35]).iterrows():
        for checkpoint in CHECKPOINT_DAYS:
            if checkpoint not in pivot or pd.isna(record.get(checkpoint)):
                continue
            previous = checkpoint - 7
            rows.append(
                {
                    "cycle_id": str(record["cycle_id"]),
                    "building_id": str(record["building_id"]),
                    "measurement_day": checkpoint,
                    "current_weight_kg": float(record[checkpoint]),
                    "previous_weight_kg": (
                        float(record[previous])
                        if previous in pivot and pd.notna(record.get(previous))
                        else np.nan
                    ),
                    "actual_day35_weight_kg": float(record[35]),
                    **{
                        f"weight_day_{day}_kg": (
                            float(record[day])
                            if day <= checkpoint
                            and day in pivot
                            and pd.notna(record.get(day))
                            else np.nan
                        )
                        for day in CHECKPOINT_DAYS
                    },
                }
            )
    return pd.DataFrame(rows)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - actual
    actual_hit = actual >= DAY35_TARGET_KG
    predicted_hit = predicted >= DAY35_TARGET_KG
    return {
        "rows": int(len(actual)),
        "mae_kg": float(mean_absolute_error(actual, predicted)),
        "rmse_kg": float(mean_squared_error(actual, predicted) ** 0.5),
        "bias_kg": float(np.mean(residual)),
        "within_100g_rate": float(np.mean(np.abs(residual) <= 0.1)),
        "within_200g_rate": float(np.mean(np.abs(residual) <= 0.2)),
        "target_side_accuracy": float(np.mean(actual_hit == predicted_hit)),
        "below_target_recall": float(
            np.mean(~predicted_hit[~actual_hit]) if (~actual_hit).any() else np.nan
        ),
        "at_or_above_target_recall": float(
            np.mean(predicted_hit[actual_hit]) if actual_hit.any() else np.nan
        ),
    }


def _ridge_feature_frame(
    rows: pd.DataFrame, target_by_age: pd.Series
) -> pd.DataFrame:
    features = pd.DataFrame(index=rows.index)
    features["measurement_day"] = rows["measurement_day"].astype(float)
    features["current_weight_kg"] = rows["current_weight_kg"].astype(float)
    features["current_to_target_ratio"] = [
        float(weight) / float(target_by_age.loc[int(day)])
        for weight, day in zip(rows["current_weight_kg"], rows["measurement_day"])
    ]
    features["recent_adg_kg_day"] = (
        rows["current_weight_kg"] - rows["previous_weight_kg"]
    ) / 7
    features["has_recent_adg"] = rows["previous_weight_kg"].notna().astype(float)
    first_weight = rows["weight_day_7_kg"]
    features["cumulative_adg_kg_day"] = np.where(
        rows["measurement_day"].gt(7) & first_weight.notna(),
        (rows["current_weight_kg"] - first_weight)
        / (rows["measurement_day"] - 7),
        np.nan,
    )
    for checkpoint in CHECKPOINT_DAYS:
        features[f"weight_day_{checkpoint}_kg"] = rows[
            f"weight_day_{checkpoint}_kg"
        ].astype(float)
    return features[list(RIDGE_FEATURES)]


def build_day35_feature_rows(dataset: CanaryDataset) -> pd.DataFrame:
    """Return the exact raw engineered X rows and observed Day 35 Y used in validation."""

    rows = build_day35_training_rows(dataset)
    if rows.empty:
        return rows
    target_by_age = dataset.targets.set_index("age_day")["target_weight_kg"]
    features = _ridge_feature_frame(rows, target_by_age)
    audit = rows[["cycle_id", "building_id"]].copy()
    audit["validation_cycle"] = rows["cycle_id"].astype(str)
    audit = pd.concat([audit, features], axis=1)
    audit["actual_day35_weight_kg_y"] = rows["actual_day35_weight_kg"].astype(float)
    return audit


def _fit_ridge_parameters(
    features: pd.DataFrame, target: np.ndarray
) -> tuple[Ridge, pd.Series, pd.Series, pd.Series]:
    medians = features.median().fillna(0.0)
    filled = features.fillna(medians)
    means = filled.mean()
    scales = filled.std(ddof=0).replace(0, 1.0)
    model = Ridge(alpha=10.0)
    model.fit((filled - means) / scales, target)
    return model, medians, means, scales


def train_day35_weight_baseline(dataset: CanaryDataset) -> dict[str, Any]:
    """Compare simple projections using leave-one-complete-cycle-out validation."""

    rows = build_day35_training_rows(dataset)
    if rows.empty or rows["cycle_id"].nunique() < 3:
        raise ValueError("At least three cycles with Day 35 weights are required.")

    target_by_age = dataset.targets.set_index("age_day")["target_weight_kg"]
    candidates = (
        "historical_day35_mean",
        "target_curve_ratio",
        "recent_linear_adg",
        "historical_remaining_gain",
        "ridge_regression",
        "random_forest",
        "gradient_boosting",
    )
    predictions = {candidate: np.full(len(rows), np.nan) for candidate in candidates}
    groups = rows["cycle_id"].to_numpy(str)
    splitter = LeaveOneGroupOut()
    ridge_features = _ridge_feature_frame(rows, target_by_age)

    for train_index, test_index in splitter.split(rows, groups=groups):
        train = rows.iloc[train_index]
        test = rows.iloc[test_index]
        fallback = float(train["actual_day35_weight_kg"].mean())
        for position, (_, test_row) in zip(test_index, test.iterrows()):
            age = int(test_row["measurement_day"])
            current = float(test_row["current_weight_kg"])
            same_age = train.loc[train["measurement_day"] == age]
            remaining_gain = float(
                (same_age["actual_day35_weight_kg"] - same_age["current_weight_kg"]).mean()
            )
            predictions["historical_day35_mean"][position] = fallback
            predictions["target_curve_ratio"][position] = (
                current / float(target_by_age.loc[age]) * DAY35_TARGET_KG
            )
            previous = test_row["previous_weight_kg"]
            predictions["recent_linear_adg"][position] = (
                current + (current - float(previous)) / 7 * (35 - age)
                if pd.notna(previous)
                else fallback
            )
            predictions["historical_remaining_gain"][position] = current + remaining_gain
        ridge, medians, means, scales = _fit_ridge_parameters(
            ridge_features.iloc[train_index],
            rows.iloc[train_index]["actual_day35_weight_kg"].to_numpy(float),
        )
        ridge_test = ridge_features.iloc[test_index].fillna(medians)
        predictions["ridge_regression"][test_index] = ridge.predict(
            (ridge_test - means) / scales
        )
        tree_candidates = {
            "random_forest": RandomForestRegressor(
                n_estimators=500,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=1,
            ),
            "gradient_boosting": HistGradientBoostingRegressor(
                max_iter=150,
                max_leaf_nodes=7,
                min_samples_leaf=8,
                l2_regularization=1.0,
                random_state=42,
            ),
        }
        for name, estimator in tree_candidates.items():
            model = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                            keep_empty_features=True,
                        ),
                    ),
                    ("model", estimator),
                ]
            )
            model.fit(ridge_features.iloc[train_index], rows.iloc[train_index]["actual_day35_weight_kg"])
            predictions[name][test_index] = model.predict(ridge_features.iloc[test_index])

    actual = rows["actual_day35_weight_kg"].to_numpy(float)
    candidate_metrics: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        predicted = np.clip(predictions[candidate], 0.1, 3.5)
        cycle_metrics = {
            str(cycle): {
                **_metrics(actual[groups == cycle], predicted[groups == cycle]),
                "bias_kg": float(np.mean(predicted[groups == cycle] - actual[groups == cycle])),
            }
            for cycle in np.unique(groups)
        }
        cycle_mae = [float(values["mae_kg"]) for values in cycle_metrics.values()]
        horizon: dict[str, dict[str, float | int]] = {}
        for checkpoint in CHECKPOINT_DAYS:
            mask = rows["measurement_day"].eq(checkpoint).to_numpy()
            horizon[f"Day {checkpoint}"] = _metrics(actual[mask], predicted[mask])
        candidate_metrics[candidate] = {
            **_metrics(actual, predicted),
            "cycle_macro_mae_kg": float(np.mean(cycle_mae)),
            "horizon": horizon,
            "cycle": cycle_metrics,
        }

    best_macro_mae = min(
        float(candidate_metrics[name]["cycle_macro_mae_kg"])
        for name in candidates
    )
    eligible = {
        name
        for name in candidates
        if float(candidate_metrics[name]["cycle_macro_mae_kg"])
        <= best_macro_mae * 1.05
    }
    simplicity_order = (
        "historical_day35_mean",
        "target_curve_ratio",
        "historical_remaining_gain",
        "recent_linear_adg",
        "ridge_regression",
        "random_forest",
        "gradient_boosting",
    )
    selected = next(name for name in simplicity_order if name in eligible)
    selected_predictions = np.clip(predictions[selected], 0.1, 3.5)
    backtest_rows = rows[
        [
            "cycle_id",
            "building_id",
            "measurement_day",
            "current_weight_kg",
            "actual_day35_weight_kg",
        ]
    ].copy()
    backtest_rows["predicted_day35_weight_kg"] = selected_predictions
    backtest_rows["error_kg"] = selected_predictions - actual
    backtest_rows["absolute_error_kg"] = np.abs(backtest_rows["error_kg"])
    day14_rows = backtest_rows.loc[backtest_rows["measurement_day"].eq(14)].copy()
    day14_metrics = candidate_metrics[selected]["horizon"]["Day 14"] | {
        "building_cycles": int(len(day14_rows)),
        "mean_error_kg": float(day14_rows["error_kg"].mean()),
        "actual_at_or_above_target": int(
            day14_rows["actual_day35_weight_kg"].ge(DAY35_TARGET_KG).sum()
        ),
        "actual_below_target": int(
            day14_rows["actual_day35_weight_kg"].lt(DAY35_TARGET_KG).sum()
        ),
    }
    remaining_gain_by_day: dict[str, float] = {}
    uncertainty_by_day: dict[str, float] = {}
    for checkpoint in CHECKPOINT_DAYS:
        mask = rows["measurement_day"].eq(checkpoint).to_numpy()
        checkpoint_rows = rows.loc[mask]
        remaining_gain_by_day[str(checkpoint)] = float(
            (
                checkpoint_rows["actual_day35_weight_kg"]
                - checkpoint_rows["current_weight_kg"]
            ).mean()
        )
        uncertainty_by_day[str(checkpoint)] = float(
            np.quantile(np.abs(actual[mask] - selected_predictions[mask]), 0.80)
        )
    remaining_gain_by_day["35"] = 0.0
    uncertainty_by_day["35"] = 0.0

    day35_outcomes = rows[
        ["cycle_id", "building_id", "actual_day35_weight_kg"]
    ].drop_duplicates()
    ridge, medians, means, scales = _fit_ridge_parameters(ridge_features, actual)
    ridge_parameters = {
        "features": list(RIDGE_FEATURES),
        "medians": {key: float(value) for key, value in medians.items()},
        "means": {key: float(value) for key, value in means.items()},
        "scales": {key: float(value) for key, value in scales.items()},
        "coefficients": {
            key: float(value) for key, value in zip(RIDGE_FEATURES, ridge.coef_)
        },
        "intercept": float(ridge.intercept_),
        "alpha": 10.0,
    }
    return {
        "outcome": "day35_average_liveweight",
        "model_version": "day35-weight-0.4.0",
        "selected_model": selected,
        "model_kind": "fitted" if selected == "ridge_regression" else "formula",
        "target_day": 35,
        "target_weight_kg": DAY35_TARGET_KG,
        "label_definition": "Observed building average bodyweight recorded on production Day 35",
        "training_source": dataset.source_name,
        "training_cycles": sorted(rows["cycle_id"].unique().tolist()),
        "training_building_cycles": int(len(day35_outcomes)),
        "training_checkpoint_rows": int(len(rows)),
        "actual_target_hits": int((day35_outcomes["actual_day35_weight_kg"] >= DAY35_TARGET_KG).sum()),
        "actual_target_misses": int((day35_outcomes["actual_day35_weight_kg"] < DAY35_TARGET_KG).sum()),
        "candidate_metrics": candidate_metrics,
        "selected_metrics": candidate_metrics[selected],
        "selection_metric": "cycle_macro_mae_kg_within_5pct_then_simplest",
        "selection_tolerance_pct": 5.0,
        "selected_method_drivers": (
            [
                {
                    "driver": item["feature"],
                    "role": f"{item['absolute_importance_pct']:.1f}% of absolute standardized Ridge reliance",
                    "direction": item["direction"],
                }
                for item in sorted(
                    [
                        {
                            "feature": feature,
                            "coefficient": float(coefficient),
                            "absolute_importance_pct": float(
                                abs(coefficient)
                                / max(float(np.abs(ridge.coef_).sum()), 1e-12)
                                * 100
                            ),
                            "direction": "Raises projection" if coefficient > 0 else "Lowers projection",
                        }
                        for feature, coefficient in zip(RIDGE_FEATURES, ridge.coef_)
                    ],
                    key=lambda item: item["absolute_importance_pct"],
                    reverse=True,
                )[:5]
            ]
            if selected == "ridge_regression"
            else [
                {
                    "driver": "Latest measured building weight",
                    "role": "Direct starting point",
                    "direction": "A higher current weight raises the Day 35 projection.",
                },
                {
                    "driver": "Production day",
                    "role": "Selects the average historical remaining gain",
                    "direction": "Earlier measurements have more growth remaining.",
                },
            ]
        ),
        "feature_importance_interpretation": (
            "Ridge reliance uses absolute standardized coefficients. It shows which inputs most influence the fitted forecast after accounting for other inputs; it is not causal evidence. Correlated growth inputs can share importance or show counterintuitive signs."
            if selected == "ridge_regression"
            else "The selected formula has two direct drivers: latest measured weight and measurement day."
        ),
        "historical_remaining_gain_definition": (
            "For each checkpoint age, subtract the measured checkpoint weight from the observed Day 35 weight for every eligible historical building-cycle, average those gains, then add that average to the current building's measured weight. During validation, the held-out cycle is excluded from the average."
        ),
        "day14_backtest_metrics": day14_metrics,
        "day14_backtest": [
            {
                "cycle_id": str(record["cycle_id"]),
                "building_id": str(record["building_id"]),
                "current_weight_kg": float(record["current_weight_kg"]),
                "predicted_day35_weight_kg": float(record["predicted_day35_weight_kg"]),
                "actual_day35_weight_kg": float(record["actual_day35_weight_kg"]),
                "error_kg": float(record["error_kg"]),
                "absolute_error_kg": float(record["absolute_error_kg"]),
            }
            for record in day14_rows.to_dict(orient="records")
        ],
        "ridge_parameters": ridge_parameters,
        "ridge_feature_importance": sorted(
            [
                {
                    "feature": feature,
                    "coefficient_kg_per_standard_deviation": float(coefficient),
                    "absolute_importance_pct": float(
                        abs(coefficient) / max(float(np.abs(ridge.coef_).sum()), 1e-12) * 100
                    ),
                    "direction": "Raises projection" if coefficient > 0 else "Lowers projection",
                }
                for feature, coefficient in zip(RIDGE_FEATURES, ridge.coef_)
            ],
            key=lambda item: item["absolute_importance_pct"],
            reverse=True,
        ),
        "remaining_gain_by_measurement_day_kg": remaining_gain_by_day,
        "uncertainty_half_width_by_measurement_day_kg": uncertainty_by_day,
        "early_day_fallback": "target_curve_ratio",
        "status": "Prototype — current cycle excluded; leave-one-complete-cycle-out validation",
    }


def save_day35_manifest(
    manifest: dict[str, Any], path: str | Path = DEFAULT_DAY35_MANIFEST
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _interpolate(mapping: dict[str, float], age: int) -> float:
    points = sorted((int(day), float(value)) for day, value in mapping.items())
    ages = np.asarray([day for day, _ in points], dtype=float)
    values = np.asarray([value for _, value in points], dtype=float)
    return float(np.interp(age, ages, values))


def project_day35_weight(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
    manifest: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Project Day 35 from the latest eligible measured weight."""

    manifest = manifest or load_day35_manifest()
    as_of = pd.Timestamp(as_of).normalize()
    weights = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & dataset.daily["weight_measured"]
        & (dataset.daily["record_date"] <= as_of)
        & (dataset.daily["age_day"] <= 35)
    ].sort_values(["record_date", "age_day"])
    if weights.empty:
        return {
            "prediction": np.nan,
            "status": "A measured weight is needed for a Day 35 projection",
            "scope": "Unavailable",
            "confidence": "No building weight is available",
            "measurement_day": np.nan,
            "interval_low": np.nan,
            "interval_high": np.nan,
        }

    exact_day35 = weights.loc[weights["age_day"] == 35]
    if not exact_day35.empty:
        observed = float(exact_day35.iloc[-1]["bodyweight_kg"])
        return {
            "prediction": observed,
            "status": "Observed Day 35 weight",
            "scope": "Recorded Day 35 result",
            "confidence": "Observed measurement — no projection model used",
            "measurement_day": 35,
            "interval_low": np.nan,
            "interval_high": np.nan,
        }

    latest = weights.iloc[-1]
    age = int(latest["age_day"])
    current = float(latest["bodyweight_kg"])
    if age < 7:
        target_match = dataset.targets.loc[
            dataset.targets["age_day"] == age, "target_weight_kg"
        ]
        if target_match.empty or float(target_match.iloc[0]) <= 0:
            return {
                "prediction": np.nan,
                "status": "No age target is available for this measurement",
                "scope": "Unavailable",
                "confidence": "Projection cannot be calculated",
                "measurement_day": age,
                "interval_low": np.nan,
                "interval_high": np.nan,
            }
        prediction = current / float(target_match.iloc[0]) * DAY35_TARGET_KG
        width = 0.4
        status = "Early Day 35 projection — target-curve fallback"
        confidence = "Early estimate · historical validation begins on Day 7"
        scope = "Early building projection"
    elif manifest["selected_model"] == "ridge_regression":
        previous_rows = weights.loc[weights["age_day"] < age]
        previous = previous_rows.iloc[-1] if not previous_rows.empty else None
        recent_adg = (
            (current - float(previous["bodyweight_kg"]))
            / (age - int(previous["age_day"]))
            if previous is not None and age > int(previous["age_day"])
            else np.nan
        )
        target_match = dataset.targets.loc[
            dataset.targets["age_day"] == age, "target_weight_kg"
        ]
        current_target = (
            float(target_match.iloc[0]) if not target_match.empty else np.nan
        )
        values = {
            "measurement_day": float(age),
            "current_weight_kg": current,
            "current_to_target_ratio": (
                current / current_target
                if pd.notna(current_target) and current_target > 0
                else np.nan
            ),
            "recent_adg_kg_day": recent_adg,
            "has_recent_adg": 0.0 if pd.isna(recent_adg) else 1.0,
            "cumulative_adg_kg_day": np.nan,
            **{f"weight_day_{checkpoint}_kg": np.nan for checkpoint in CHECKPOINT_DAYS},
        }
        for checkpoint in CHECKPOINT_DAYS:
            checkpoint_rows = weights.loc[weights["age_day"].eq(checkpoint)]
            if not checkpoint_rows.empty and checkpoint <= age:
                values[f"weight_day_{checkpoint}_kg"] = float(
                    checkpoint_rows.iloc[-1]["bodyweight_kg"]
                )
        day7_weight = values.get("weight_day_7_kg", np.nan)
        if pd.notna(day7_weight) and age > 7:
            values["cumulative_adg_kg_day"] = (current - float(day7_weight)) / (age - 7)
        parameters = manifest["ridge_parameters"]
        standardized: dict[str, float] = {}
        for feature_name in parameters["features"]:
            value = values.get(feature_name, np.nan)
            if pd.isna(value):
                value = parameters["medians"][feature_name]
            standardized[feature_name] = (
                float(value) - parameters["means"][feature_name]
            ) / parameters["scales"][feature_name]
        prediction = parameters["intercept"] + sum(
            standardized[name] * parameters["coefficients"][name]
            for name in parameters["features"]
        )
        width = _interpolate(
            manifest["uncertainty_half_width_by_measurement_day_kg"], age
        )
        status = "Day 35 projection available"
        confidence = f"Compact Ridge projection · latest measured weight from Day {age}"
        scope = "Building projection"
    else:
        gain = _interpolate(manifest["remaining_gain_by_measurement_day_kg"], age)
        width = _interpolate(
            manifest["uncertainty_half_width_by_measurement_day_kg"], age
        )
        prediction = current + gain
        status = "Day 35 projection available"
        confidence = (
            f"Building projection · latest measured weight from Day {age}"
        )
        scope = "Building projection"
    prediction = float(np.clip(prediction, 0.1, 3.5))
    return {
        "prediction": prediction,
        "status": status,
        "scope": scope,
        "confidence": confidence,
        "measurement_day": age,
        "interval_low": max(0.1, prediction - width),
        "interval_high": min(3.5, prediction + width),
    }
