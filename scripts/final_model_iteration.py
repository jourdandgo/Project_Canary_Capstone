"""Leakage-safe final model iteration for Project Canary.

This research script compares compact and colleague-inspired feature sets on
the corrected, exported Canary training tables.  Complete harvest cycles are
held out in the outer loop.  Hyperparameters, imputation, scaling, historical
building priors, and correlation filtering are learned only from each outer
training fold.

The script does not promote a live model.  It writes reproducible evidence
used to decide whether a challenger is strong enough to replace the current
transparent fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_READY = ROOT / "outputs" / "model_ready"
OUT = ROOT / "analysis" / "final_model_iteration"
OUT.mkdir(parents=True, exist_ok=True)

RECOVERY_ID = {
    "cycle_id", "building_id", "as_of_date", "label_date", "validation_cycle",
    "row_selection", "final_recovery_proxy_y", "additional_population_loss_y",
    "naive_recovery_projection", "naive_weight_projection",
}
WEIGHT_ID = {
    "cycle_id", "building_id", "validation_cycle",
    "actual_day35_weight_kg_y", "remaining_gain_to_day35_kg_y",
}

RECOVERY_COMPACT = [
    "cycle_day", "percentage_alive", "population_loss_pct",
    "mortality_recent_3d_per_1000", "mortality_trend_delta_per_1000",
    "weight_gap_pct", "weight_staleness_days",
    "temperature_deviation_from_band_c", "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d", "environment_staleness_days",
    "is_lags_building",
]
RECOVERY_EXPANDED = [
    "cycle_day", "forecast_horizon_days", "percentage_alive",
    "survival_gap_pp", "population_loss_pct", "cumulative_mortality_rate",
    "mortality_daily_per_1000", "mortality_recent_3d_per_1000",
    "mortality_trend_delta_per_1000", "feed_daily_per_1000_birds",
    "feed_cumulative_per_1000_birds", "feed_daily_kg_per_bird",
    "latest_weight_kg", "weight_gap_pct", "weight_measurement_day",
    "weight_staleness_days", "temperature_recent_avg_c",
    "temperature_recent_min_c", "temperature_recent_max_c",
    "temperature_recent_range_c", "temperature_deviation_from_band_c",
    "humidity_recent_avg_pct", "humidity_recent_min_pct",
    "humidity_recent_max_pct", "humidity_recent_range_pp",
    "humidity_deviation_from_band_pp", "environment_out_of_band_days_7d",
    "environment_staleness_days", "is_lags_building",
]
WEIGHT_COMPACT = [
    "measurement_day", "current_weight_kg", "current_to_target_ratio",
    "recent_adg_kg_day", "has_recent_adg", "cumulative_adg_kg_day",
    "weight_day_7_kg", "weight_day_14_kg", "weight_day_21_kg",
    "weight_day_28_kg", "percentage_alive", "mortality_recent_3d_per_1000",
    "temperature_deviation_from_band_c", "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d", "environment_staleness_days",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_set: str
    target_form: str


def cycle_key(value: object) -> tuple[int, int]:
    year, sequence = str(value).replace("_", "-").split("-")[:2]
    return int(year), int(sequence)


def unit_weights(frame: pd.DataFrame) -> np.ndarray:
    units = frame["cycle_id"].astype(str) + "::" + frame["building_id"].astype(str)
    count = units.map(units.value_counts()).to_numpy(float)
    result = 1.0 / count
    return result / result.mean()


def historical_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
    name: str,
) -> tuple[pd.Series, pd.Series]:
    outcomes = train[["cycle_id", "building_id", label]].drop_duplicates(
        ["cycle_id", "building_id"]
    )
    overall = float(outcomes[label].mean())
    mapping = outcomes.groupby("building_id")[label].mean().to_dict()
    train_prior = train["building_id"].map(mapping).fillna(overall).rename(name)
    test_prior = test["building_id"].map(mapping).fillna(overall).rename(name)
    return train_prior, test_prior


def contextual_features(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["is_lags_group"] = frame["building_id"].astype(str).str.startswith("Lags").astype(float)
    if outcome == "recovery":
        group_cols = ["cycle_id", "cycle_day"]
        for column in ("percentage_alive", "population_loss_pct", "weight_gap_pct"):
            median = frame.groupby(group_cols)[column].transform("median")
            result[f"peer_delta_{column}"] = frame[column] - median
    else:
        group_cols = ["cycle_id", "measurement_day"]
        for column in ("current_weight_kg", "current_to_target_ratio", "percentage_alive"):
            median = frame.groupby(group_cols)[column].transform("median")
            result[f"peer_delta_{column}"] = frame[column] - median
    return result


def prepare_fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    outcome: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[columns].copy()
    test_x = test[columns].copy()
    label = "final_recovery_proxy_y" if outcome == "recovery" else "actual_day35_weight_kg_y"
    prior_name = "historical_building_recovery" if outcome == "recovery" else "historical_building_day35_weight"
    train_prior, test_prior = historical_prior(train, test, label, prior_name)
    train_x[prior_name] = train_prior.to_numpy()
    test_x[prior_name] = test_prior.to_numpy()
    train_context = contextual_features(train, outcome)
    test_context = contextual_features(test, outcome)
    for column in train_context:
        train_x[column] = train_context[column].to_numpy()
        test_x[column] = test_context[column].to_numpy()

    # Fold-local coverage and redundancy control.  Never inspect the held-out
    # cycle while deciding which columns survive.
    keep = [column for column in train_x if train_x[column].notna().mean() >= 0.40]
    train_x = train_x[keep]
    test_x = test_x[keep]
    numeric = train_x.astype(float)
    correlation = numeric.corr().abs()
    drop: set[str] = set()
    for i, column in enumerate(correlation.columns):
        for other in correlation.columns[i + 1 :]:
            if correlation.loc[column, other] > 0.97:
                protected = {"percentage_alive", "current_weight_kg", "current_to_target_ratio"}
                if other not in protected:
                    drop.add(other)
                elif column not in protected:
                    drop.add(column)
    keep = [column for column in train_x if column not in drop]
    return train_x[keep].astype(float), test_x[keep].astype(float)


def pipeline(family: str, params: dict[str, Any]) -> Pipeline:
    if family == "linear":
        model: object = LinearRegression()
    elif family == "ridge":
        model = Ridge(alpha=float(params.get("alpha", 10.0)))
    elif family == "huber":
        model = HuberRegressor(
            alpha=float(params.get("alpha", 0.01)),
            epsilon=float(params.get("epsilon", 1.35)),
            max_iter=4000,
        )
    elif family == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(params.get("trees", 75)),
            learning_rate=float(params.get("rate", 0.04)),
            max_depth=int(params.get("depth", 1)),
            min_samples_leaf=int(params.get("leaf", 4)),
            loss="huber",
            random_state=42,
        )
    elif family == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=params.get("depth", 4),
            min_samples_leaf=int(params.get("leaf", 4)),
            max_features=params.get("max_features", 0.7),
            random_state=42,
            n_jobs=1,
        )
    else:
        raise ValueError(family)
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if family in {"linear", "ridge", "huber"}:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def options(family: str) -> list[dict[str, Any]]:
    if family == "ridge":
        return [{"alpha": value} for value in (0.3, 1.0, 3.0, 10.0, 30.0)]
    if family == "huber":
        return [
            {"alpha": alpha, "epsilon": epsilon}
            for alpha, epsilon in ((0.001, 1.2), (0.01, 1.35), (0.1, 1.5))
        ]
    if family == "gradient_boosting":
        return [
            {"trees": trees, "rate": rate, "depth": depth, "leaf": leaf}
            for trees, rate, depth, leaf in (
                (50, 0.03, 1, 4), (75, 0.04, 1, 4),
                (75, 0.03, 2, 5), (100, 0.03, 1, 5),
            )
        ]
    if family == "extra_trees":
        return [
            {"depth": depth, "leaf": leaf, "max_features": max_features}
            for depth, leaf, max_features in (
                (3, 3, 0.6), (4, 4, 0.7), (5, 5, 1.0), (None, 6, 0.7),
            )
        ]
    return [{}]


def raw_target(frame: pd.DataFrame, outcome: str, target_form: str) -> np.ndarray:
    if outcome == "recovery":
        final = frame["final_recovery_proxy_y"].to_numpy(float)
        if target_form == "remaining":
            return np.clip(frame["percentage_alive"].to_numpy(float) - final, 0.0, 1.0)
        return final
    final = frame["actual_day35_weight_kg_y"].to_numpy(float)
    if target_form == "remaining":
        return final - frame["current_weight_kg"].to_numpy(float)
    if target_form == "ratio":
        return final / frame["current_weight_kg"].to_numpy(float)
    return final


def to_outcome(frame: pd.DataFrame, raw: np.ndarray, outcome: str, target_form: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if outcome == "recovery":
        if target_form == "remaining":
            current = frame["percentage_alive"].to_numpy(float)
            return current - np.clip(raw, 0.0, current)
        return np.clip(raw, 0.0, 1.0)
    if target_form == "remaining":
        raw = frame["current_weight_kg"].to_numpy(float) + raw
    elif target_form == "ratio":
        raw = frame["current_weight_kg"].to_numpy(float) * raw
    return np.clip(raw, 0.1, 3.5)


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame, outcome: str) -> np.ndarray:
    if outcome == "recovery":
        loss = np.clip(
            train["percentage_alive"].to_numpy(float)
            - train["final_recovery_proxy_y"].to_numpy(float),
            0.0,
            1.0,
        )
        band = np.select(
            [train["cycle_day"] <= 7, train["cycle_day"] <= 14, train["cycle_day"] <= 21],
            [7, 14, 21], default=28,
        )
        mapping = pd.Series(loss).groupby(band).mean().to_dict()
        test_band = np.select(
            [test["cycle_day"] <= 7, test["cycle_day"] <= 14, test["cycle_day"] <= 21],
            [7, 14, 21], default=28,
        )
        predicted_loss = np.asarray([mapping.get(value, float(np.mean(loss))) for value in test_band])
        return to_outcome(test, predicted_loss, outcome, "remaining")
    gain = train["actual_day35_weight_kg_y"] - train["current_weight_kg"]
    mapping = gain.groupby(train["measurement_day"]).mean().to_dict()
    fallback = float(gain.mean())
    predicted_gain = np.asarray([mapping.get(day, fallback) for day in test["measurement_day"]])
    return to_outcome(test, predicted_gain, outcome, "remaining")


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: Candidate,
    outcome: str,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, Pipeline, list[str]]:
    if outcome == "recovery":
        columns = RECOVERY_COMPACT if candidate.feature_set == "compact" else RECOVERY_EXPANDED
    else:
        columns = WEIGHT_COMPACT
    train_x, test_x = prepare_fold_features(train, test, columns, outcome)
    model = pipeline(candidate.family, parameters)
    weights = unit_weights(train)
    model.fit(train_x, raw_target(train, outcome, candidate.target_form), model__sample_weight=weights)
    raw = model.predict(test_x)
    return to_outcome(test, raw, outcome, candidate.target_form), model, list(train_x.columns)


def tune(
    train: pd.DataFrame,
    candidate: Candidate,
    outcome: str,
) -> dict[str, Any]:
    groups = train["cycle_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 3 or len(options(candidate.family)) == 1:
        return options(candidate.family)[0]
    scored = []
    for parameters in options(candidate.family):
        fold_errors = []
        for inner_train, inner_valid in LeaveOneGroupOut().split(train, groups=groups):
            predicted, _, _ = fit_predict(
                train.iloc[inner_train], train.iloc[inner_valid], candidate, outcome, parameters
            )
            label = "final_recovery_proxy_y" if outcome == "recovery" else "actual_day35_weight_kg_y"
            fold_errors.append(mean_absolute_error(train.iloc[inner_valid][label], predicted))
        scored.append((float(np.mean(fold_errors)), parameters))
    return min(scored, key=lambda item: item[0])[1]


def metrics(frame: pd.DataFrame, predicted: np.ndarray, outcome: str) -> dict[str, Any]:
    label = "final_recovery_proxy_y" if outcome == "recovery" else "actual_day35_weight_kg_y"
    actual = frame[label].to_numpy(float)
    groups = frame["cycle_id"].astype(str).to_numpy()
    factor = 100.0 if outcome == "recovery" else 1000.0
    cycle_mae = {
        group: float(mean_absolute_error(actual[groups == group], predicted[groups == group]) * factor)
        for group in np.unique(groups)
    }
    target = 0.95 if outcome == "recovery" else 1.8
    result = {
        "rows": int(len(frame)),
        "independent_outcomes": int(frame[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "cycle_macro_mae": float(np.mean(list(cycle_mae.values()))),
        "pooled_mae": float(mean_absolute_error(actual, predicted) * factor),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5 * factor),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual) * factor),
        "cycle_mae": cycle_mae,
        "target_side_accuracy": float(np.mean((actual >= target) == (predicted >= target))),
        "majority_target_side_accuracy": float(max(np.mean(actual >= target), np.mean(actual < target))),
    }
    if outcome == "weight":
        result["within_100g"] = float(np.mean(np.abs(actual - predicted) <= 0.1))
        result["within_200g"] = float(np.mean(np.abs(actual - predicted) <= 0.2))
    return result


def evaluate(
    frame: pd.DataFrame,
    outcome: str,
    candidates: list[Candidate],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    groups = frame["cycle_id"].astype(str).to_numpy()
    all_predictions: dict[str, np.ndarray] = {"Transparent baseline": np.full(len(frame), np.nan)}
    all_parameters: dict[str, list[dict[str, Any]]] = {"Transparent baseline": []}
    all_features: dict[str, list[list[str]]] = {"Transparent baseline": []}
    for candidate in candidates:
        all_predictions[candidate.name] = np.full(len(frame), np.nan)
        all_parameters[candidate.name] = []
        all_features[candidate.name] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        all_predictions["Transparent baseline"][test_idx] = baseline_predict(train, test, outcome)
        all_parameters["Transparent baseline"].append({})
        all_features["Transparent baseline"].append([])
        for candidate in candidates:
            selected = tune(train, candidate, outcome)
            predicted, _, used = fit_predict(train, test, candidate, outcome, selected)
            all_predictions[candidate.name][test_idx] = predicted
            all_parameters[candidate.name].append(selected)
            all_features[candidate.name].append(used)
    evidence: dict[str, Any] = {}
    rows = []
    for name, predicted in all_predictions.items():
        result = metrics(frame, predicted, outcome)
        result["outer_fold_parameters"] = all_parameters[name]
        result["outer_fold_feature_sets"] = all_features[name]
        evidence[name] = result
        rows.append({"model": name, **{k: v for k, v in result.items() if not isinstance(v, (dict, list))}})
    return pd.DataFrame(rows).sort_values("cycle_macro_mae"), evidence


def checkpoint_results(frame: pd.DataFrame, outcome: str, candidates: list[Candidate]) -> dict[str, Any]:
    field = "cycle_day" if outcome == "recovery" else "measurement_day"
    output: dict[str, Any] = {}
    for day in (7, 14, 21, 28):
        subset = frame[frame[field].eq(day)].reset_index(drop=True)
        if subset.empty:
            continue
        comparison, evidence = evaluate(subset, outcome, candidates)
        output[str(day)] = {
            "comparison": comparison.to_dict(orient="records"),
            "evidence": evidence,
        }
    return output


def _checkpoint_weight_templates(day: int) -> list[list[str]]:
    observed = [f"weight_day_{value}_kg" for value in (7, 14, 21, 28) if value <= day]
    return [
        ["current_weight_kg"],
        observed,
        observed + ["percentage_alive"],
        observed + [
            "temperature_deviation_from_band_c",
            "humidity_deviation_from_band_pp",
        ],
    ]


def checkpoint_ridge_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Nested checkpoint-specific Ridge model for the Day 35 outcome.

    Each checkpoint gets its own compact model.  The feature template and
    regularisation strength are selected using only the outer training cycles.
    """

    predicted = np.full(len(frame), np.nan)
    audit: list[dict[str, Any]] = []
    outer_groups = frame["cycle_id"].astype(str).to_numpy()
    for outer_train, outer_test in LeaveOneGroupOut().split(frame, groups=outer_groups):
        train_all = frame.iloc[outer_train]
        test_all = frame.iloc[outer_test]
        for day in (7, 14, 21, 28):
            train = train_all[train_all["measurement_day"].eq(day)].copy()
            test = test_all[test_all["measurement_day"].eq(day)].copy()
            if test.empty:
                continue
            inner_groups = train["cycle_id"].astype(str).to_numpy()
            scored: list[tuple[float, float, list[str]]] = []
            for columns in _checkpoint_weight_templates(day):
                for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
                    errors = []
                    for inner_train, inner_valid in LeaveOneGroupOut().split(
                        train, groups=inner_groups
                    ):
                        model = Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                                ("scale", StandardScaler()),
                                ("model", Ridge(alpha=alpha)),
                            ]
                        )
                        model.fit(
                            train.iloc[inner_train][columns],
                            train.iloc[inner_train]["actual_day35_weight_kg_y"],
                        )
                        inner_prediction = model.predict(train.iloc[inner_valid][columns])
                        errors.append(
                            mean_absolute_error(
                                train.iloc[inner_valid]["actual_day35_weight_kg_y"],
                                inner_prediction,
                            )
                        )
                    scored.append((float(np.mean(errors)), alpha, columns))
            _, alpha, columns = min(scored, key=lambda item: item[0])
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=alpha)),
                ]
            )
            model.fit(train[columns], train["actual_day35_weight_kg_y"])
            test_positions = test.index.to_numpy()
            predicted[test_positions] = np.clip(model.predict(test[columns]), 0.1, 3.5)
            audit.append(
                {
                    "held_out_cycle": str(test["cycle_id"].iloc[0]),
                    "checkpoint_day": day,
                    "alpha": alpha,
                    "features": columns,
                }
            )
    return predicted, audit


def recovery_shap_oof(frame: pd.DataFrame) -> pd.DataFrame:
    """Fold-local SHAP values for the best nonlinear recovery challenger."""

    candidate = Candidate(
        "Extra Trees - compact remaining loss", "extra_trees", "compact", "remaining"
    )
    groups = frame["cycle_id"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_index], frame.iloc[test_index]
        selected = tune(train, candidate, "recovery")
        train_x, test_x = prepare_fold_features(train, test, RECOVERY_COMPACT, "recovery")
        fitted = pipeline(candidate.family, selected)
        fitted.fit(
            train_x,
            raw_target(train, "recovery", "remaining"),
            model__sample_weight=unit_weights(train),
        )
        imputer = fitted.named_steps["imputer"]
        tree = fitted.named_steps["model"]
        transformed_test = imputer.transform(test_x)
        feature_names = list(imputer.get_feature_names_out(train_x.columns))
        # Tree output is additional loss.  Negate SHAP so positive values mean
        # the feature raises predicted final recovery.
        values = -np.asarray(shap.TreeExplainer(tree).shap_values(transformed_test))
        transformed_frame = pd.DataFrame(transformed_test, columns=feature_names)
        for row_position, source_index in enumerate(test.index):
            for column_position, feature in enumerate(feature_names):
                records.append(
                    {
                        "source_index": int(source_index),
                        "cycle_id": str(test.loc[source_index, "cycle_id"]),
                        "building_id": str(test.loc[source_index, "building_id"]),
                        "cycle_day": int(test.loc[source_index, "cycle_day"]),
                        "feature": feature,
                        "feature_value": float(transformed_frame.iloc[row_position, column_position]),
                        "shap_recovery_effect": float(values[row_position, column_position]),
                    }
                )
    return pd.DataFrame(records)


def weight_quality_sensitivity(frame: pd.DataFrame) -> dict[str, Any]:
    outcome = (
        frame.groupby(["cycle_id", "building_id"])
        .agg(
            day28=("weight_day_28_kg", "max"),
            day35=("actual_day35_weight_kg_y", "max"),
        )
        .reset_index()
    )
    outcome["day28_to_35_adg_g"] = (outcome["day35"] - outcome["day28"]) / 7 * 1000
    flagged = outcome[outcome["day28_to_35_adg_g"] > 120].copy()
    flagged_units = set(zip(flagged["cycle_id"], flagged["building_id"]))
    clean = frame[
        ~frame.apply(lambda row: (row["cycle_id"], row["building_id"]) in flagged_units, axis=1)
    ].reset_index(drop=True)
    comparison, evidence = evaluate(
        clean,
        "weight",
        [
            Candidate("Ordinary linear - final weight", "linear", "compact", "final"),
            Candidate("Ridge - remaining gain", "ridge", "compact", "remaining"),
            Candidate("Gradient boosting - final weight", "gradient_boosting", "compact", "final"),
            Candidate("Extra Trees - final weight", "extra_trees", "compact", "final"),
        ],
    )
    return {
        "rule": "Flag Day 28 to Day 35 average gain above 120 g/day for expert review; do not automatically delete it.",
        "flagged_outcomes": flagged.to_dict(orient="records"),
        "remaining_independent_outcomes": int(clean[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "comparison": comparison.to_dict(orient="records"),
        "evidence": evidence,
    }


def save_proof_charts(
    recovery_comparison: pd.DataFrame,
    weight_comparison: pd.DataFrame,
    recovery_shap: pd.DataFrame,
    weight_checkpoints: dict[str, Any],
) -> None:
    sns.set_theme(style="whitegrid")
    green = "#174f3b"
    lime = "#91c529"
    for name, frame, unit in (
        ("recovery", recovery_comparison.head(5), "percentage points"),
        ("weight", weight_comparison.head(5), "grams"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        ordered = frame.sort_values("cycle_macro_mae", ascending=True)
        sns.barplot(data=ordered, x="cycle_macro_mae", y="model", color=green, ax=axes[0])
        axes[0].set_title("Whole-cycle MAE")
        axes[0].set_xlabel(unit)
        axes[0].set_ylabel("")
        sns.barplot(data=ordered, x="r2", y="model", color=lime, ax=axes[1])
        axes[1].axvline(0, color="#7b8794", linewidth=1)
        axes[1].set_title("Held-out R²")
        axes[1].set_xlabel("variance explained")
        axes[1].set_ylabel("")
        fig.suptitle(f"{name.title()} model comparison — complete-cycle holdouts", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_model_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    summary = (
        recovery_shap.groupby("feature")["shap_recovery_effect"]
        .agg(mean_abs=lambda values: float(np.mean(np.abs(values))), mean_signed="mean")
        .sort_values("mean_abs", ascending=False)
        .head(12)
        .reset_index()
    )
    summary.to_csv(OUT / "recovery_oof_shap_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=summary, x="mean_abs", y="feature", color=green, ax=ax)
    ax.set_title("Recovery model — held-out SHAP importance")
    ax.set_xlabel("mean absolute effect on projected recovery")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "recovery_oof_shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    checkpoint_rows = []
    for day, payload in weight_checkpoints.items():
        for row in payload["comparison"]:
            if row["model"] in {"Transparent baseline", "Ordinary linear - final weight", "Ridge - remaining gain"}:
                checkpoint_rows.append({"day": int(day), **row})
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=checkpoint_frame, x="day", y="cycle_macro_mae", hue="model", marker="o", ax=axes[0])
    axes[0].set_title("Weight MAE by checkpoint")
    axes[0].set_ylabel("grams")
    sns.lineplot(data=checkpoint_frame, x="day", y="r2", hue="model", marker="o", ax=axes[1], legend=False)
    axes[1].axhline(0, color="#7b8794", linewidth=1)
    axes[1].set_title("Weight R² by checkpoint")
    axes[1].set_ylabel("variance explained")
    fig.suptitle("Day 35 weight reliability improves as later weights become available", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "weight_checkpoint_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    recovery = pd.read_csv(MODEL_READY / "recovery_training.csv")
    weight = pd.read_csv(MODEL_READY / "day35_weight_training.csv")
    recovery_candidates = [
        Candidate("Ordinary linear - remaining loss", "linear", "compact", "remaining"),
        Candidate("Ridge - compact remaining loss", "ridge", "compact", "remaining"),
        Candidate("Gradient boosting - compact remaining loss", "gradient_boosting", "compact", "remaining"),
        Candidate("Extra Trees - compact remaining loss", "extra_trees", "compact", "remaining"),
    ]
    weight_candidates = [
        Candidate("Ordinary linear - final weight", "linear", "compact", "final"),
        Candidate("Ridge - remaining gain", "ridge", "compact", "remaining"),
        Candidate("Gradient boosting - final weight", "gradient_boosting", "compact", "final"),
        Candidate("Extra Trees - final weight", "extra_trees", "compact", "final"),
    ]
    recovery_comparison, recovery_evidence = evaluate(recovery, "recovery", recovery_candidates)
    weight_comparison, weight_evidence = evaluate(weight, "weight", weight_candidates)
    checkpoint_ridge, checkpoint_ridge_audit = checkpoint_ridge_predictions(weight)
    checkpoint_ridge_metrics = metrics(weight, checkpoint_ridge, "weight")
    # Keep the owner/panel comparison to exactly five declared methods
    # (baseline plus four challengers).  The checkpoint-specific Ridge audit is
    # retained as a sensitivity check, not promoted into a sixth headline row.
    recovery_checkpoints = checkpoint_results(recovery, "recovery", recovery_candidates)
    weight_checkpoints = checkpoint_results(weight, "weight", weight_candidates)
    recovery_shap = recovery_shap_oof(recovery)
    recovery_shap.to_csv(OUT / "recovery_oof_shap_values.csv", index=False)
    weight_sensitivity = weight_quality_sensitivity(weight)
    save_proof_charts(recovery_comparison, weight_comparison, recovery_shap, weight_checkpoints)
    recovery_comparison.to_csv(OUT / "recovery_whole_cycle_comparison.csv", index=False)
    weight_comparison.to_csv(OUT / "weight_whole_cycle_comparison.csv", index=False)
    payload = {
        "validation": "nested leave-one-complete-cycle-out",
        "recovery": {
            "comparison": recovery_comparison.to_dict(orient="records"),
            "evidence": recovery_evidence,
            "checkpoint_results": recovery_checkpoints,
        },
        "weight": {
            "comparison": weight_comparison.to_dict(orient="records"),
            "evidence": weight_evidence,
            "checkpoint_results": weight_checkpoints,
            "checkpoint_ridge_sensitivity": {
                **checkpoint_ridge_metrics,
                "outer_fold_selection_audit": checkpoint_ridge_audit,
            },
            "data_quality_sensitivity": weight_sensitivity,
        },
        "xgboost_status": "Not run locally because libomp is unavailable; Extra Trees and Gradient Boosting provide leakage-safe nonlinear challenger evidence.",
    }
    (OUT / "iteration_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print("RECOVERY")
    print(recovery_comparison.to_string(index=False))
    print("\nWEIGHT")
    print(weight_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
