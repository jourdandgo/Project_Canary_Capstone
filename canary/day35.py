"""Transparent Day 35 liveweight projection for Project Canary."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:  # Optional challenger; macOS needs the separate libomp runtime.
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - environment-dependent optional package.
    XGBRegressor = None  # type: ignore[assignment,misc]

from .data import CanaryDataset
from .modeling import extract_feature_row


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
    "percentage_alive",
    "temperature_deviation_from_band_c",
    "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d",
    "environment_staleness_days",
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
            feature_as_of = extract_feature_row(
                dataset,
                str(record["cycle_id"]),
                str(record["building_id"]),
                pd.Timestamp(
                    dataset.daily.loc[
                        (dataset.daily["cycle_id"] == str(record["cycle_id"]))
                        & (dataset.daily["building_id"] == str(record["building_id"]))
                        & (dataset.daily["age_day"] <= checkpoint),
                        "record_date",
                    ].max()
                ),
            )
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
                    **{
                        feature: (
                            feature_as_of.get(feature, np.nan)
                            if feature_as_of is not None
                            else np.nan
                        )
                        for feature in (
                            "percentage_alive",
                            "mortality_recent_3d_per_1000",
                            "feed_daily_kg_per_bird",
                            "temperature_deviation_from_band_c",
                            "humidity_deviation_from_band_pp",
                            "environment_out_of_band_days_7d",
                            "environment_staleness_days",
                        )
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
        "r2": float(r2_score(actual, predicted)),
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
    for feature in (
        "percentage_alive",
        "mortality_recent_3d_per_1000",
        "feed_daily_kg_per_bird",
        "temperature_deviation_from_band_c",
        "humidity_deviation_from_band_pp",
        "environment_out_of_band_days_7d",
        "environment_staleness_days",
    ):
        features[feature] = pd.to_numeric(rows[feature], errors="coerce")
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
    features: pd.DataFrame, target: np.ndarray, alpha: float = 10.0
) -> tuple[Ridge, pd.Series, pd.Series, pd.Series]:
    medians = features.median().fillna(0.0)
    filled = features.fillna(medians)
    means = filled.mean()
    scales = filled.std(ddof=0).replace(0, 1.0)
    model = Ridge(alpha=alpha)
    model.fit((filled - means) / scales, target)
    return model, medians, means, scales


def _row_weights(rows: pd.DataFrame) -> np.ndarray:
    keys = rows["cycle_id"].astype(str) + "::" + rows["building_id"].astype(str)
    counts = keys.map(keys.value_counts()).to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.mean()


def _day35_pipeline(candidate: str) -> Pipeline:
    if candidate == "linear_regression":
        estimator: object = LinearRegression()
    elif candidate == "ridge_regression":
        estimator = Ridge(alpha=10.0)
    elif candidate == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=75,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=4,
            random_state=42,
        )
    elif candidate == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost requires the operating-system OpenMP runtime.")
        estimator = XGBRegressor(
            n_estimators=75,
            max_depth=2,
            learning_rate=0.04,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
            n_jobs=1,
        )
    else:
        raise ValueError(candidate)
    steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        )
    ]
    if candidate in {"linear_regression", "ridge_regression"}:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _day35_options(candidate: str) -> list[dict[str, object]]:
    if candidate == "ridge_regression":
        return [{"model__alpha": alpha} for alpha in (10.0, 25.0, 50.0, 100.0)]
    if candidate == "gradient_boosting":
        return [
            {
                "model__n_estimators": n_estimators,
                "model__learning_rate": rate,
                "model__max_depth": depth,
                "model__min_samples_leaf": leaf,
            }
            for n_estimators, rate, depth, leaf in (
                (50, 0.03, 1, 4),
                (75, 0.04, 2, 4),
                (100, 0.03, 2, 5),
                (75, 0.06, 1, 3),
            )
        ]
    if candidate == "xgboost":
        return [
            {
                "model__n_estimators": n_estimators,
                "model__learning_rate": rate,
                "model__max_depth": depth,
                "model__min_child_weight": child,
                "model__reg_lambda": regularization,
            }
            for n_estimators, rate, depth, child, regularization in (
                (50, 0.03, 1, 4, 10.0),
                (75, 0.04, 2, 4, 5.0),
                (100, 0.03, 2, 6, 10.0),
            )
        ]
    return [{}]


def _tune_day35(
    candidate: str,
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[Pipeline, dict[str, object]]:
    base = _day35_pipeline(candidate)
    options = _day35_options(candidate)
    scores: list[tuple[float, dict[str, object]]] = []
    if len(options) > 1 and len(np.unique(groups)) >= 3:
        for params in options:
            fold_mae: list[float] = []
            for train_index, valid_index in LeaveOneGroupOut().split(
                features, target, groups
            ):
                model = clone(base).set_params(**params)
                model.fit(
                    features.iloc[train_index],
                    target[train_index],
                    model__sample_weight=weights[train_index],
                )
                fold_mae.append(
                    float(
                        mean_absolute_error(
                            target[valid_index], model.predict(features.iloc[valid_index])
                        )
                    )
                )
            scores.append((float(np.mean(fold_mae)), params))
        best = min(scores, key=lambda item: item[0])[1]
    else:
        best = options[0]
    fitted = clone(base).set_params(**best)
    fitted.fit(features, target, model__sample_weight=weights)
    return fitted, best


def _bootstrap_cycle_mae(
    actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray, repeats: int = 2000
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    cycles = np.unique(groups)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(cycles, size=len(cycles), replace=True)
        values.append(
            float(
                np.mean(
                    [
                        mean_absolute_error(actual[groups == cycle], predicted[groups == cycle])
                        for cycle in sampled
                    ]
                )
            )
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return {"lower_kg": float(low), "upper_kg": float(high), "confidence": 0.95}


def train_day35_weight_baseline(dataset: CanaryDataset) -> dict[str, Any]:
    """Compare simple projections using leave-one-complete-cycle-out validation."""

    rows = build_day35_training_rows(dataset)
    if rows.empty or rows["cycle_id"].nunique() < 3:
        raise ValueError("At least three cycles with Day 35 weights are required.")

    target_by_age = dataset.targets.set_index("age_day")["target_weight_kg"]
    all_candidates = (
        "historical_remaining_gain",
        "linear_regression",
        "ridge_regression",
        "gradient_boosting",
        "xgboost",
    )
    candidates = all_candidates
    if XGBRegressor is None:
        candidates = tuple(candidate for candidate in candidates if candidate != "xgboost")
    predictions = {candidate: np.full(len(rows), np.nan) for candidate in candidates}
    groups = rows["cycle_id"].to_numpy(str)
    sample_weights = _row_weights(rows)
    splitter = LeaveOneGroupOut()
    ridge_features = _ridge_feature_frame(rows, target_by_age)
    fold_parameters: dict[str, list[dict[str, object]]] = {
        candidate: [] for candidate in candidates
    }

    for train_index, test_index in splitter.split(rows, groups=groups):
        train = rows.iloc[train_index]
        test = rows.iloc[test_index]
        for position, (_, test_row) in zip(test_index, test.iterrows()):
            age = int(test_row["measurement_day"])
            current = float(test_row["current_weight_kg"])
            same_age = train.loc[train["measurement_day"] == age]
            remaining_gain = float(
                (same_age["actual_day35_weight_kg"] - same_age["current_weight_kg"]).mean()
            )
            predictions["historical_remaining_gain"][position] = current + remaining_gain
        fold_parameters["historical_remaining_gain"].append({})
        for candidate in candidates:
            if candidate == "historical_remaining_gain":
                continue
            model, params = _tune_day35(
                candidate,
                ridge_features.iloc[train_index],
                rows.iloc[train_index]["actual_day35_weight_kg"].to_numpy(float),
                groups[train_index],
                sample_weights[train_index],
            )
            fold_parameters[candidate].append(params)
            predictions[candidate][test_index] = model.predict(
                ridge_features.iloc[test_index]
            )

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
            "outer_fold_best_parameters": fold_parameters[candidate],
        }

    learned_candidates = [
        name for name in candidates if name != "historical_remaining_gain"
    ]
    best_macro_mae = min(
        float(candidate_metrics[name]["cycle_macro_mae_kg"])
        for name in learned_candidates
    )
    eligible = {
        name
        for name in learned_candidates
        if float(candidate_metrics[name]["cycle_macro_mae_kg"])
        <= best_macro_mae * 1.05
    }
    simplicity_order = (
        "ridge_regression",
        "linear_regression",
        "gradient_boosting",
        "xgboost",
    )
    research_champion = next(name for name in simplicity_order if name in eligible)
    baseline_metrics = candidate_metrics["historical_remaining_gain"]
    research_metrics = candidate_metrics[research_champion]
    improvement_pct = (
        (
            float(baseline_metrics["cycle_macro_mae_kg"])
            - float(research_metrics["cycle_macro_mae_kg"])
        )
        / float(baseline_metrics["cycle_macro_mae_kg"])
        * 100
    )
    majority_accuracy = max(
        float(np.mean(actual >= DAY35_TARGET_KG)),
        float(np.mean(actual < DAY35_TARGET_KG)),
    )
    regression_gate = bool(
        improvement_pct >= 10.0
        and float(research_metrics["r2"]) > 0
        and float(research_metrics["within_200g_rate"]) >= 0.70
    )
    classification_gate = bool(
        float(research_metrics["target_side_accuracy"]) > majority_accuracy
        and float(research_metrics["below_target_recall"]) > 0
        and float(research_metrics["at_or_above_target_recall"]) > 0
    )
    selected = (
        research_champion
        if regression_gate and classification_gate
        else "historical_remaining_gain"
    )
    selected_predictions = np.clip(predictions[selected], 0.1, 3.5)

    final_research_model, final_research_parameters = _tune_day35(
        research_champion,
        ridge_features,
        actual,
        groups,
        sample_weights,
    )
    within_cycle_groups = (
        rows["cycle_id"].astype(str) + "::" + rows["building_id"].astype(str)
    ).to_numpy()
    secondary_metrics: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        secondary_prediction = np.full(len(rows), np.nan)
        for train_index, test_index in LeaveOneGroupOut().split(
            rows, actual, within_cycle_groups
        ):
            if candidate == "historical_remaining_gain":
                train = rows.iloc[train_index]
                for position, (_, test_row) in zip(test_index, rows.iloc[test_index].iterrows()):
                    age = int(test_row["measurement_day"])
                    gain = (
                        train.loc[train["measurement_day"] == age, "actual_day35_weight_kg"]
                        - train.loc[train["measurement_day"] == age, "current_weight_kg"]
                    ).mean()
                    secondary_prediction[position] = float(test_row["current_weight_kg"]) + float(gain)
            else:
                model = _day35_pipeline(candidate)
                if candidate == research_champion:
                    model.set_params(**final_research_parameters)
                model.fit(
                    ridge_features.iloc[train_index],
                    actual[train_index],
                    model__sample_weight=sample_weights[train_index],
                )
                secondary_prediction[test_index] = model.predict(
                    ridge_features.iloc[test_index]
                )
        secondary_prediction = np.clip(secondary_prediction, 0.1, 3.5)
        secondary_metrics[candidate] = {
            "mae_kg": float(mean_absolute_error(actual, secondary_prediction)),
            "rmse_kg": float(mean_squared_error(actual, secondary_prediction) ** 0.5),
            "r2": float(r2_score(actual, secondary_prediction)),
        }

    accumulated_importance: dict[str, list[float]] = {
        feature: [] for feature in RIDGE_FEATURES
    }
    for train_index, test_index in splitter.split(rows, groups=groups):
        model, _ = _tune_day35(
            research_champion,
            ridge_features.iloc[train_index],
            actual[train_index],
            groups[train_index],
            sample_weights[train_index],
        )
        importance = permutation_importance(
            model,
            ridge_features.iloc[test_index],
            actual[test_index],
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=42,
        )
        for feature, value in zip(RIDGE_FEATURES, importance.importances_mean):
            accumulated_importance[feature].append(float(max(0.0, value)))
    mean_importance = {
        feature: float(np.mean(values))
        for feature, values in accumulated_importance.items()
    }
    importance_total = sum(mean_importance.values())
    held_out_importance = sorted(
        [
            {
                "feature": feature,
                "mean_mae_increase_kg": value,
                "relative_importance_pct": value / importance_total * 100
                if importance_total
                else 0.0,
            }
            for feature, value in mean_importance.items()
        ],
        key=lambda item: item["mean_mae_increase_kg"],
        reverse=True,
    )
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
    ridge_alpha = float(
        final_research_parameters.get("model__alpha", 10.0)
        if research_champion == "ridge_regression"
        else 10.0
    )
    ridge, medians, means, scales = _fit_ridge_parameters(
        ridge_features, actual, alpha=ridge_alpha
    )
    ridge_parameters = {
        "features": list(RIDGE_FEATURES),
        "medians": {key: float(value) for key, value in medians.items()},
        "means": {key: float(value) for key, value in means.items()},
        "scales": {key: float(value) for key, value in scales.items()},
        "coefficients": {
            key: float(value) for key, value in zip(RIDGE_FEATURES, ridge.coef_)
        },
        "intercept": float(ridge.intercept_),
        "alpha": ridge_alpha,
    }
    return {
        "outcome": "day35_average_liveweight",
        "model_version": "day35-weight-1.0.0",
        "selected_model": selected,
        "research_champion": research_champion,
        "operational_model": selected,
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
        "research_champion_metrics": candidate_metrics[research_champion],
        "selection_metric": "nested_leave_one_complete_cycle_out_cycle_macro_mae_kg",
        "selection_tolerance_pct": 5.0,
        "nested_validation": {
            "outer_split": "Leave one complete harvest cycle out",
            "inner_split": "Leave one complete remaining harvest cycle out",
            "optimization_metric": "Cycle-balanced mean absolute error in kilograms",
            "preprocessing_scope": "Imputation, scaling and tuning are fitted inside each training fold",
            "independent_unit": "Building-cycle; four checkpoint rows receive equal total training weight",
        },
        "champion_gates": {
            "baseline": "historical_remaining_gain",
            "baseline_improvement_pct": improvement_pct,
            "requires_at_least_10pct_mae_improvement": improvement_pct >= 10.0,
            "requires_positive_r2": float(research_metrics["r2"]) > 0,
            "requires_at_least_70pct_within_200g": float(
                research_metrics["within_200g_rate"]
            )
            >= 0.70,
            "regression_gate_passed": regression_gate,
            "majority_target_side_accuracy": majority_accuracy,
            "requires_better_than_majority_target_side_accuracy": float(
                research_metrics["target_side_accuracy"]
            )
            > majority_accuracy,
            "requires_recall_for_both_target_sides": (
                float(research_metrics["below_target_recall"]) > 0
                and float(research_metrics["at_or_above_target_recall"]) > 0
            ),
            "target_classification_gate_passed": classification_gate,
            "operational_fallback_applied": selected == "historical_remaining_gain",
        },
        "primary_whole_cycle_bootstrap_mae_95ci": _bootstrap_cycle_mae(
            actual, selected_predictions, groups
        ),
        "secondary_within_cycle_metrics": secondary_metrics,
        "secondary_validation_note": (
            "Diagnostic only: leaves out one building-cycle while other buildings from the same harvest cycle can remain in training. This is easier and more optimistic than prospective whole-cycle validation."
        ),
        "candidate_registry": [
            {
                "model": candidate,
                "available": candidate in candidates,
                "reason": (
                    "Evaluated under nested whole-cycle validation"
                    if candidate in candidates
                    else "Unavailable locally because the required OpenMP runtime is missing; run in Linux/CI before release if XGBoost evidence is required"
                ),
            }
            for candidate in all_candidates
        ],
        "final_research_fit_parameters": final_research_parameters,
        "unavailable_candidates": (
            {"xgboost": "Not run in this local build because the required macOS OpenMP runtime (libomp) is unavailable."}
            if XGBRegressor is None
            else {}
        ),
        "selected_method_drivers": (
            [
                {
                    "driver": "Latest measured building weight",
                    "role": "Direct starting point in the operational formula",
                    "direction": "A higher recorded weight raises the Day 35 projection by the same amount.",
                },
                {
                    "driver": "Measurement day",
                    "role": "Selects the historical remaining-gain average",
                    "direction": "Earlier measurements use a larger expected remaining gain because more growth days remain.",
                },
                {
                    "driver": "Historical remaining gain",
                    "role": "Average Day 35 minus checkpoint gain from training cycles only",
                    "direction": "A larger historical gain for that checkpoint age raises the projection.",
                },
            ]
            if selected == "historical_remaining_gain"
            else
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
        "research_champion_drivers": (
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
            if research_champion == "ridge_regression"
            else []
        ),
        "feature_importance_interpretation": (
            "Held-out permutation importance and Ridge coefficients show predictive reliance, not causal effects. The operational baseline is used whenever the learned challenger fails the approved gates."
        ),
        "held_out_permutation_importance": held_out_importance,
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
        "early_day_fallback": "historical_remaining_gain",
        "status": (
            "Validated prototype — champion gates passed"
            if selected == research_champion
            else "Experimental learned forecast — transparent historical remaining-gain baseline used operationally because the learned challenger failed at least one champion gate"
        ),
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
        operational = extract_feature_row(
            dataset, cycle_id, building_id, as_of
        ) or {}
        for feature_name in (
            "percentage_alive",
            "temperature_deviation_from_band_c",
            "humidity_deviation_from_band_pp",
            "environment_out_of_band_days_7d",
            "environment_staleness_days",
        ):
            values[feature_name] = operational.get(feature_name, np.nan)
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
