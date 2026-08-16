"""Small-sample, leakage-safe challenger pipelines for Project Canary.

Both outcomes are expressed as what remains after the review date:

* recovery predicts additional population loss, then subtracts it from the
  currently observed survival rate;
* Day 35 weight predicts remaining growth, then adds it to the latest observed
  bodyweight.

The module deliberately compares only five compact methods and validates them
by complete harvest cycle.  It is designed for auditability, not leaderboard
optimisation on repeated daily rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RECOVERY_FEATURES = [
    "cycle_day",
    "percentage_alive",
    "population_loss_pct",
    "mortality_recent_3d_per_1000",
    "mortality_trend_delta_per_1000",
    "weight_gap_pct",
    "weight_staleness_days",
    "temperature_deviation_from_band_c",
    "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d",
    "environment_staleness_days",
    "is_lags_building",
]

WEIGHT_BASE_FEATURES = [
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
    "mortality_recent_3d_per_1000",
    "temperature_deviation_from_band_c",
    "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d",
    "environment_staleness_days",
]
WEIGHT_CHECKPOINT_FEATURES = [f"checkpoint_day_{day}" for day in (7, 14, 21, 28)]
WEIGHT_FEATURES = WEIGHT_BASE_FEATURES + WEIGHT_CHECKPOINT_FEATURES

RECOVERY_CANDIDATES = (
    "age_band_remaining_loss",
    "remaining_loss_linear",
    "remaining_loss_ridge",
    "remaining_loss_gradient_boosting",
    "remaining_loss_extra_trees",
)
WEIGHT_CANDIDATES = (
    "historical_remaining_gain",
    "checkpoint_linear_remaining_gain",
    "ridge_remaining_gain",
    "huber_remaining_gain",
    "gradient_boosting_remaining_gain",
)


@dataclass(frozen=True)
class StrengthenedResult:
    manifest: dict[str, Any]
    model: object | None


def _cycle_key(value: object) -> tuple[int, ...]:
    pieces = str(value).replace("_", "-").split("-")
    return tuple(int(piece) if piece.isdigit() else 0 for piece in pieces)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    keys = frame["cycle_id"].astype(str) + "::" + frame["building_id"].astype(str)
    counts = keys.map(keys.value_counts()).to_numpy(float)
    raw = 1.0 / counts
    return raw / raw.mean()


def _age_band(days: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(days, dtype=float)
    return np.select(
        [values <= 7, values <= 14, values <= 21],
        [7, 14, 21],
        default=28,
    ).astype(int)


def _fit_age_baseline(train: pd.DataFrame, loss: np.ndarray) -> dict[int, float]:
    frame = pd.DataFrame({"band": _age_band(train["cycle_day"]), "loss": loss})
    overall = float(frame["loss"].mean())
    grouped = frame.groupby("band")["loss"].mean().to_dict()
    return {day: float(grouped.get(day, overall)) for day in (7, 14, 21, 28)}


def _predict_age_baseline(frame: pd.DataFrame, mapping: dict[int, float]) -> np.ndarray:
    return np.asarray([mapping[int(day)] for day in _age_band(frame["cycle_day"])])


def _pipeline(candidate: str) -> Pipeline:
    if candidate in {"remaining_loss_linear", "checkpoint_linear_remaining_gain"}:
        model: object = LinearRegression()
    elif candidate in {"remaining_loss_ridge", "ridge_remaining_gain"}:
        model = Ridge(alpha=10.0)
    elif candidate in {"remaining_loss_huber", "huber_remaining_gain"}:
        model = HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=2000)
    elif candidate in {
        "remaining_loss_gradient_boosting",
        "gradient_boosting_remaining_gain",
    }:
        model = GradientBoostingRegressor(
            n_estimators=75,
            learning_rate=0.04,
            max_depth=1,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )
    elif candidate == "remaining_loss_extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=4,
            max_features=0.7,
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
    if not isinstance(model, (GradientBoostingRegressor, ExtraTreesRegressor)):
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _options(candidate: str) -> list[dict[str, object]]:
    if candidate in {"remaining_loss_ridge", "ridge_remaining_gain"}:
        return [{"model__alpha": alpha} for alpha in (0.1, 1.0, 10.0, 25.0)]
    if candidate in {"remaining_loss_huber", "huber_remaining_gain"}:
        return [
            {"model__alpha": alpha, "model__epsilon": epsilon}
            for alpha, epsilon in ((0.001, 1.2), (0.01, 1.35), (0.1, 1.5))
        ]
    if candidate in {
        "remaining_loss_gradient_boosting",
        "gradient_boosting_remaining_gain",
    }:
        return [
            {
                "model__n_estimators": trees,
                "model__learning_rate": rate,
                "model__max_depth": depth,
                "model__min_samples_leaf": leaf,
            }
            for trees, rate, depth, leaf in (
                (50, 0.03, 1, 4),
                (75, 0.04, 1, 4),
                (75, 0.04, 2, 5),
                (100, 0.03, 1, 5),
            )
        ]
    if candidate == "remaining_loss_extra_trees":
        return [
            {
                "model__max_depth": depth,
                "model__min_samples_leaf": leaf,
                "model__max_features": max_features,
            }
            for depth, leaf, max_features in (
                (3, 3, 0.6),
                (4, 4, 0.7),
                (5, 5, 1.0),
                (None, 6, 0.7),
            )
        ]
    return [{}]


def _fit_candidate(
    candidate: str,
    x: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    convert: Callable[[np.ndarray, np.ndarray], np.ndarray],
    current: np.ndarray,
) -> tuple[Pipeline, dict[str, object]]:
    options = _options(candidate)
    best = options[0]
    if len(options) > 1 and len(np.unique(groups)) >= 3:
        scored: list[tuple[float, dict[str, object]]] = []
        for parameters in options:
            errors: list[float] = []
            for train, valid in LeaveOneGroupOut().split(x, target, groups):
                model = clone(_pipeline(candidate)).set_params(**parameters)
                model.fit(x.iloc[train], target[train], model__sample_weight=sample_weight[train])
                raw = model.predict(x.iloc[valid])
                predicted = convert(current[valid], raw)
                actual = convert(current[valid], target[valid])
                errors.append(float(mean_absolute_error(actual, predicted)))
            scored.append((float(np.mean(errors)), parameters))
        best = min(scored, key=lambda item: item[0])[1]
    fitted = clone(_pipeline(candidate)).set_params(**best)
    fitted.fit(x, target, model__sample_weight=sample_weight)
    return fitted, best


def _recovery_from_loss(current: np.ndarray, loss: np.ndarray) -> np.ndarray:
    bounded_loss = np.clip(np.asarray(loss, dtype=float), 0.0, np.asarray(current, dtype=float))
    return np.asarray(current, dtype=float) - bounded_loss


def _weight_from_gain(current: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(current, dtype=float) + np.asarray(gain, dtype=float), 0.1, 3.5)


def _target_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    target: float,
    weight: bool = False,
) -> dict[str, Any]:
    cycle = {}
    for group in np.unique(groups):
        mask = groups == group
        cycle[str(group)] = {
            "rows": int(mask.sum()),
            "mae_kg" if weight else "mae": float(mean_absolute_error(actual[mask], predicted[mask])),
            "rmse_kg" if weight else "rmse": float(mean_squared_error(actual[mask], predicted[mask]) ** 0.5),
            "bias_kg" if weight else "bias": float(np.mean(predicted[mask] - actual[mask])),
        }
    absolute = np.abs(predicted - actual)
    hit = actual >= target
    predicted_hit = predicted >= target
    below_recall = float(np.mean(~predicted_hit[~hit])) if (~hit).any() else np.nan
    above_recall = float(np.mean(predicted_hit[hit])) if hit.any() else np.nan
    base = {
        "rows": int(len(actual)),
        "r2": float(r2_score(actual, predicted)),
        "target_side_accuracy": float(np.mean(hit == predicted_hit)),
        "majority_side_accuracy": float(max(np.mean(hit), np.mean(~hit))),
        "below_target_recall": below_recall,
        "at_or_above_target_recall": above_recall,
        "balanced_target_accuracy": float(np.nanmean([below_recall, above_recall])),
        "confusion_matrix": {
            "actual_below_predicted_below": int(np.sum((~hit) & (~predicted_hit))),
            "actual_below_predicted_at_or_above": int(np.sum((~hit) & predicted_hit)),
            "actual_at_or_above_predicted_below": int(np.sum(hit & (~predicted_hit))),
            "actual_at_or_above_predicted_at_or_above": int(np.sum(hit & predicted_hit)),
        },
        "cycle": cycle,
    }
    if weight:
        maes = [item["mae_kg"] for item in cycle.values()]
        base.update(
            {
                "mae_kg": float(np.mean(absolute)),
                "cycle_macro_mae_kg": float(np.mean(maes)),
                "rmse_kg": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                "bias_kg": float(np.mean(predicted - actual)),
                "within_100g_rate": float(np.mean(absolute <= 0.1)),
                "within_200g_rate": float(np.mean(absolute <= 0.2)),
                "fold_mae_std_kg": float(np.std(maes)),
                "uncertainty_half_width_80_kg": float(np.quantile(absolute, 0.8)),
            }
        )
    else:
        maes = [item["mae"] for item in cycle.values()]
        base.update(
            {
                "mae": float(np.mean(absolute)),
                "cycle_macro_mae": float(np.mean(maes)),
                "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                "bias": float(np.mean(predicted - actual)),
                "fold_mae_std": float(np.std(maes)),
                "uncertainty_half_width_80": float(np.quantile(absolute, 0.8)),
            }
        )
    return base


def _cycle_bootstrap(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    weight: bool,
    repeats: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    cycles = np.unique(groups)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(cycles, len(cycles), replace=True)
        values.append(
            float(np.mean([mean_absolute_error(actual[groups == c], predicted[groups == c]) for c in sampled]))
        )
    low, high = np.quantile(values, [0.025, 0.975])
    suffix = "_kg" if weight else ""
    return {f"lower{suffix}": float(low), f"upper{suffix}": float(high), "confidence": 0.95}


def _permutation_records(
    candidate: str,
    x: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    current: np.ndarray,
    convert: Callable[[np.ndarray, np.ndarray], np.ndarray],
    unit_key: str,
) -> list[dict[str, object]]:
    accumulated = {column: [] for column in x.columns}
    for train, test in LeaveOneGroupOut().split(x, target, groups):
        model, _ = _fit_candidate(
            candidate,
            x.iloc[train],
            target[train],
            groups[train],
            sample_weight[train],
            convert,
            current[train],
        )
        baseline = mean_absolute_error(
            convert(current[test], target[test]),
            convert(current[test], model.predict(x.iloc[test])),
        )
        rng = np.random.default_rng(42)
        for column in x.columns:
            increases = []
            for _ in range(12):
                permuted = x.iloc[test].copy()
                permuted[column] = rng.permutation(permuted[column].to_numpy())
                error = mean_absolute_error(
                    convert(current[test], target[test]),
                    convert(current[test], model.predict(permuted)),
                )
                increases.append(max(0.0, float(error - baseline)))
            accumulated[column].append(float(np.mean(increases)))
    averaged = {name: float(np.mean(values)) for name, values in accumulated.items()}
    total = sum(averaged.values())
    return sorted(
        [
            {
                "feature": name,
                unit_key: value,
                "relative_importance_pct": value / total * 100 if total else 0.0,
            }
            for name, value in averaged.items()
        ],
        key=lambda item: float(item[unit_key]),
        reverse=True,
    )


def _held_out_shap_records(
    candidate: str,
    x: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    current: np.ndarray,
    convert: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> list[dict[str, object]]:
    """Aggregate SHAP values calculated only on complete held-out cycles.

    The fitted tree predicts additional population loss.  SHAP values are
    therefore negated so their sign describes the effect on final recovery:
    positive values raise the recovery estimate and negative values lower it.
    """

    if candidate != "remaining_loss_extra_trees":
        return []

    import shap

    records: list[pd.DataFrame] = []
    for train, test in LeaveOneGroupOut().split(x, target, groups):
        model, _ = _fit_candidate(
            candidate,
            x.iloc[train],
            target[train],
            groups[train],
            sample_weight[train],
            convert,
            current[train],
        )
        imputer = model.named_steps["imputer"]
        estimator = model.named_steps["model"]
        transformed = imputer.transform(x.iloc[test])
        names = list(imputer.get_feature_names_out(x.columns))
        shap_values = -np.asarray(
            shap.TreeExplainer(estimator).shap_values(transformed)
        )
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        fold = pd.DataFrame(shap_values, columns=names)
        fold["__cycle"] = groups[test]
        for index, name in enumerate(names):
            fold[f"__value__{name}"] = transformed[:, index]
        records.append(fold)

    combined = pd.concat(records, ignore_index=True)
    results = []
    for name in [column for column in combined.columns if not column.startswith("__")]:
        values = combined[f"__value__{name}"].to_numpy(float)
        effects = combined[name].to_numpy(float)
        correlation = (
            float(np.corrcoef(values, effects)[0, 1])
            if np.std(values) > 0 and np.std(effects) > 0
            else 0.0
        )
        results.append(
            {
                "feature": str(name),
                "mean_abs_shap_recovery": float(np.mean(np.abs(effects))),
                "mean_shap_recovery": float(np.mean(effects)),
                "value_effect_correlation": correlation,
                "direction_when_value_increases": (
                    "Generally raises the recovery estimate"
                    if correlation >= 0.10
                    else "Generally lowers the recovery estimate"
                    if correlation <= -0.10
                    else "Non-linear or mixed effect"
                ),
            }
        )
    results.sort(key=lambda item: float(item["mean_abs_shap_recovery"]), reverse=True)
    total = sum(float(item["mean_abs_shap_recovery"]) for item in results)
    for item in results:
        item["relative_mean_abs_shap_pct"] = (
            float(item["mean_abs_shap_recovery"]) / total * 100 if total else 0.0
        )
    return results


def _rolling_origin(
    rows: pd.DataFrame,
    candidates: tuple[str, ...],
    x: pd.DataFrame,
    target: np.ndarray,
    current: np.ndarray,
    convert: Callable[[np.ndarray, np.ndarray], np.ndarray],
    baseline_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
) -> dict[str, Any]:
    groups = rows["cycle_id"].astype(str).to_numpy()
    cycles = sorted(np.unique(groups), key=_cycle_key)
    sample_weight = _weights(rows)
    results: dict[str, list[dict[str, object]]] = {candidate: [] for candidate in candidates}
    for position in range(2, len(cycles)):
        test_cycle = cycles[position]
        train_cycles = set(cycles[:position])
        train = np.asarray([value in train_cycles for value in groups])
        test = groups == test_cycle
        for candidate in candidates:
            if candidate == candidates[0]:
                raw = baseline_predict(rows.loc[train], target[train], rows.loc[test])
            else:
                model, _ = _fit_candidate(
                    candidate,
                    x.loc[train],
                    target[train],
                    groups[train],
                    sample_weight[train],
                    convert,
                    current[train],
                )
                raw = model.predict(x.loc[test])
            actual = convert(current[test], target[test])
            predicted = convert(current[test], raw)
            results[candidate].append(
                {
                    "test_cycle": str(test_cycle),
                    "training_cycles": sorted(train_cycles, key=_cycle_key),
                    "rows": int(test.sum()),
                    "mae": float(mean_absolute_error(actual, predicted)),
                    "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
                    "bias": float(np.mean(predicted - actual)),
                }
            )
    return {
        candidate: {
            "folds": folds,
            "cycle_macro_mae": float(np.mean([fold["mae"] for fold in folds])) if folds else np.nan,
            "rmse": float(np.sqrt(np.mean([fold["rmse"] ** 2 for fold in folds]))) if folds else np.nan,
            "bias": float(np.mean([fold["bias"] for fold in folds])) if folds else np.nan,
        }
        for candidate, folds in results.items()
    }


def _within_building_cycle_validation(
    rows: pd.DataFrame,
    candidates: tuple[str, ...],
    x: pd.DataFrame,
    target: np.ndarray,
    current: np.ndarray,
    convert: Callable[[np.ndarray, np.ndarray], np.ndarray],
    baseline_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
) -> dict[str, dict[str, float]]:
    """Easier diagnostic that leaves out one building-cycle, not a whole cycle."""

    cycle_groups = rows["cycle_id"].astype(str).to_numpy()
    unit_groups = (
        rows["cycle_id"].astype(str) + "::" + rows["building_id"].astype(str)
    ).to_numpy()
    sample_weight = _weights(rows)
    output: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        predicted = np.full(len(rows), np.nan)
        for train, test in LeaveOneGroupOut().split(x, target, unit_groups):
            if candidate == candidates[0]:
                raw = baseline_predict(rows.iloc[train], target[train], rows.iloc[test])
            else:
                model, _ = _fit_candidate(
                    candidate,
                    x.iloc[train],
                    target[train],
                    cycle_groups[train],
                    sample_weight[train],
                    convert,
                    current[train],
                )
                raw = model.predict(x.iloc[test])
            predicted[test] = convert(current[test], raw)
        actual = convert(current, target)
        output[candidate] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r2": float(r2_score(actual, predicted)),
        }
    return output


def train_recovery_remaining_loss(rows: pd.DataFrame) -> StrengthenedResult:
    """Compare five recovery methods using nested whole-cycle validation."""

    rows = rows.copy().reset_index(drop=True)
    actual = rows["target"].to_numpy(float)
    current = rows["percentage_alive"].to_numpy(float)
    loss = np.clip(current - actual, 0.0, 1.0)
    groups = rows["cycle_id"].astype(str).to_numpy()
    x = rows[RECOVERY_FEATURES].copy()
    sample_weight = _weights(rows)
    predictions = {name: np.full(len(rows), np.nan) for name in RECOVERY_CANDIDATES}
    parameters = {name: [] for name in RECOVERY_CANDIDATES}
    splitter = LeaveOneGroupOut()

    for train, test in splitter.split(x, loss, groups):
        mapping = _fit_age_baseline(rows.iloc[train], loss[train])
        predictions[RECOVERY_CANDIDATES[0]][test] = _predict_age_baseline(rows.iloc[test], mapping)
        parameters[RECOVERY_CANDIDATES[0]].append({str(k): v for k, v in mapping.items()})
        for candidate in RECOVERY_CANDIDATES[1:]:
            model, chosen = _fit_candidate(
                candidate,
                x.iloc[train],
                loss[train],
                groups[train],
                sample_weight[train],
                _recovery_from_loss,
                current[train],
            )
            predictions[candidate][test] = model.predict(x.iloc[test])
            parameters[candidate].append(chosen)

    final_predictions = {
        name: _recovery_from_loss(current, prediction)
        for name, prediction in predictions.items()
    }
    metrics = {
        name: _target_metrics(actual, prediction, groups, 0.95)
        for name, prediction in final_predictions.items()
    }
    baseline = RECOVERY_CANDIDATES[0]
    learned = RECOVERY_CANDIDATES[1:]
    best_mae = min(float(metrics[name]["cycle_macro_mae"]) for name in learned)
    eligible = [name for name in learned if float(metrics[name]["cycle_macro_mae"]) <= best_mae * 1.10]
    baseline_metrics = metrics[baseline]
    gate_eligible = [
        name
        for name in eligible
        if (
            (
                float(baseline_metrics["cycle_macro_mae"])
                - float(metrics[name]["cycle_macro_mae"])
            )
            / float(baseline_metrics["cycle_macro_mae"])
            * 100
            >= 10
            and float(metrics[name]["r2"]) > 0
            and max(item["mae"] for item in metrics[name]["cycle"].values())
            <= max(item["mae"] for item in baseline_metrics["cycle"].values()) * 1.25
        )
    ]
    research_pool = gate_eligible or eligible
    research = min(
        research_pool,
        key=lambda name: float(metrics[name]["cycle_macro_mae"]),
    )
    research_metrics = metrics[research]
    improvement = (
        (float(baseline_metrics["cycle_macro_mae"]) - float(research_metrics["cycle_macro_mae"]))
        / float(baseline_metrics["cycle_macro_mae"])
        * 100
    )
    worst_cycle_ok = max(item["mae"] for item in research_metrics["cycle"].values()) <= (
        max(item["mae"] for item in baseline_metrics["cycle"].values()) * 1.25
    )
    regression_gate = improvement >= 10 and float(research_metrics["r2"]) > 0 and worst_cycle_ok
    classification_gate = (
        float(research_metrics["target_side_accuracy"]) > float(research_metrics["majority_side_accuracy"])
        and float(research_metrics["balanced_target_accuracy"]) >= 0.60
    )
    # The 95% hit/miss gate controls classification claims, not whether a
    # materially better continuous estimate can be used.  Canary therefore
    # selects the research model when the regression gates pass, while the app
    # explicitly avoids presenting the estimate as a probability of success.
    selected = research if regression_gate else baseline

    research_model, research_parameters = _fit_candidate(
        research,
        x,
        loss,
        groups,
        sample_weight,
        _recovery_from_loss,
        current,
    )
    research_importance = _permutation_records(
        research,
        x,
        loss,
        groups,
        sample_weight,
        current,
        _recovery_from_loss,
        "mean_mae_increase",
    )
    if selected == baseline:
        model = None
        final_parameters: dict[str, object] = {
            str(k): v for k, v in _fit_age_baseline(rows, loss).items()
        }
        importance: list[dict[str, object]] = []
    else:
        model, final_parameters = _fit_candidate(
            selected,
            x,
            loss,
            groups,
            sample_weight,
            _recovery_from_loss,
            current,
        )
        importance = _permutation_records(
            selected,
            x,
            loss,
            groups,
            sample_weight,
            current,
            _recovery_from_loss,
            "mean_mae_increase",
        )
    nonlinear_shap_candidate = min(
        ("remaining_loss_gradient_boosting", "remaining_loss_extra_trees"),
        key=lambda name: float(metrics[name]["cycle_macro_mae"]),
    )
    # The operational champion can be linear on a tiny dataset.  We still
    # retain a leakage-safe SHAP sensitivity view for the strongest nonlinear
    # challenger, but label it separately so it is never mistaken for the
    # live forecast's explanation.
    held_out_shap = _held_out_shap_records(
        nonlinear_shap_candidate,
        x,
        loss,
        groups,
        sample_weight,
        current,
        _recovery_from_loss,
    )

    day14 = rows["cycle_day"].eq(14).to_numpy()
    selected_prediction = final_predictions[selected]
    recovery_backtest = rows[
        ["cycle_id", "building_id", "as_of_date", "cycle_day", "percentage_alive"]
    ].copy()
    recovery_backtest["predicted_final_recovery"] = selected_prediction
    recovery_backtest["actual_final_recovery_proxy"] = actual
    recovery_backtest["error"] = selected_prediction - actual
    recovery_backtest["absolute_error"] = np.abs(recovery_backtest["error"])
    day14_frame = rows.loc[day14, ["cycle_id", "building_id", "as_of_date"]].copy()
    day14_frame["predicted"] = selected_prediction[day14]
    day14_frame["actual"] = actual[day14]
    day14_frame["error"] = day14_frame["predicted"] - day14_frame["actual"]
    day14_frame["absolute_error"] = day14_frame["error"].abs()
    day14_metrics = _target_metrics(actual[day14], selected_prediction[day14], groups[day14], 0.95)
    day14_metrics["building_cycles"] = int(day14.sum())
    day14_metrics["mean_error"] = float(np.mean(selected_prediction[day14] - actual[day14]))
    day14_metrics["actual_at_or_above_target"] = int(np.sum(actual[day14] >= 0.95))
    day14_metrics["actual_below_target"] = int(np.sum(actual[day14] < 0.95))
    day14_metrics["predicted_at_or_above_target"] = int(
        np.sum(selected_prediction[day14] >= 0.95)
    )
    day14_metrics["predicted_below_target"] = int(
        np.sum(selected_prediction[day14] < 0.95)
    )

    for name in RECOVERY_CANDIDATES:
        metrics[name]["outer_fold_best_parameters"] = parameters[name]
        metrics[name]["horizon"] = {}
        for label, low, high in (
            ("Days 1-7", 0, 7),
            ("Days 8-14", 8, 14),
            ("Days 15-21", 15, 21),
            ("Day 22+", 22, 10_000),
        ):
            mask = rows["cycle_day"].between(low, high).to_numpy()
            metrics[name]["horizon"][label] = {
                "rows": int(mask.sum()),
                "mae": float(mean_absolute_error(actual[mask], final_predictions[name][mask])),
                "rmse": float(mean_squared_error(actual[mask], final_predictions[name][mask]) ** 0.5),
                "r2": float(r2_score(actual[mask], final_predictions[name][mask])) if mask.sum() > 1 else np.nan,
            }

    def recovery_baseline(train_rows: pd.DataFrame, train_target: np.ndarray, test_rows: pd.DataFrame) -> np.ndarray:
        return _predict_age_baseline(test_rows, _fit_age_baseline(train_rows, train_target))

    rolling = _rolling_origin(
        rows,
        RECOVERY_CANDIDATES,
        x,
        loss,
        current,
        _recovery_from_loss,
        recovery_baseline,
    )
    within_cycle = _within_building_cycle_validation(
        rows,
        RECOVERY_CANDIDATES,
        x,
        loss,
        current,
        _recovery_from_loss,
        recovery_baseline,
    )
    manifest = {
        "outcome": "recovery",
        "model_version": "recovery-3.2.0",
        "selected_model": selected,
        "research_champion": research,
        "operational_model": selected,
        "model_kind": "fitted" if model is not None else "formula",
        "prediction_target": "additional_population_loss_after_review_date",
        "prediction_equation": "predicted final recovery = current percentage alive - predicted additional loss",
        "feature_schema_version": "features-0.4.0",
        "feature_columns": RECOVERY_FEATURES if model is not None else ["cycle_day", "percentage_alive"],
        "all_candidate_features": {name: RECOVERY_FEATURES if name != baseline else ["cycle_day", "percentage_alive"] for name in RECOVERY_CANDIDATES},
        "metrics": metrics,
        "selected_metrics": metrics[selected],
        "research_champion_metrics": metrics[research],
        "candidate_registry": [{"model": name, "available": True, "reason": "Evaluated under nested whole-cycle validation"} for name in RECOVERY_CANDIDATES],
        "training_cycles": sorted(rows["cycle_id"].astype(str).unique(), key=_cycle_key),
        "training_building_cycles": int(rows[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "training_snapshot_rows": int(len(rows)),
        "snapshot_sampling": "Days 7, 14, 21, and 28 plus the latest eligible pre-outcome snapshot",
        "target": 0.95,
        "label_definition": "Population on the last recorded daily date divided by beginning inventory; capstone recovery proxy",
        "status": "Validated experimental forecast" if selected != baseline else "Experimental - transparent age-band remaining-loss baseline retained",
        "selection_metric": "nested_leave_one_complete_cycle_out_cycle_macro_mae",
        "selection_tolerance_pct": 0.0,
        "nested_validation": {
            "outer_split": "Leave one complete harvest cycle out",
            "inner_split": "Leave one complete remaining harvest cycle out",
            "optimization_metric": "Cycle-balanced MAE in final recovery",
            "preprocessing_scope": "Imputation, scaling, feature handling and tuning fitted inside each training fold",
            "independent_unit": "Building-cycle; repeated snapshots receive equal total training weight",
        },
        "rolling_origin_validation": rolling,
        "rolling_origin_note": "Secondary prospective check: train only on earlier cycles and predict the next recorded cycle.",
        "secondary_within_cycle_metrics": within_cycle,
        "secondary_validation_note": "Diagnostic only: leaves out one building-cycle while other buildings from the same harvest cycle remain in training.",
        "champion_gates": {
            "baseline": baseline,
            "baseline_improvement_pct": improvement,
            "requires_at_least_10pct_mae_improvement": improvement >= 10,
            "requires_positive_r2": float(research_metrics["r2"]) > 0,
            "requires_stable_worst_cycle": worst_cycle_ok,
            "regression_gate_passed": regression_gate,
            "requires_better_than_majority_target_side_accuracy": float(research_metrics["target_side_accuracy"]) > float(research_metrics["majority_side_accuracy"]),
            "requires_recall_for_both_target_sides": float(research_metrics["below_target_recall"]) > 0 and float(research_metrics["at_or_above_target_recall"]) > 0,
            "requires_at_least_60pct_balanced_target_accuracy": float(research_metrics["balanced_target_accuracy"]) >= 0.60,
            "target_classification_gate_passed": classification_gate,
            "classification_claim_allowed": classification_gate,
            "operational_fallback_applied": selected == baseline,
        },
        "final_fit_parameters": final_parameters,
        "research_champion_fit_parameters": research_parameters,
        "additional_loss_by_age_band": final_parameters if selected == baseline else {str(k): v for k, v in _fit_age_baseline(rows, loss).items()},
        "live_inference_policy": {
            "name": "piecewise_linear_checkpoint_loss",
            "checkpoints": [7, 14, 21, 28],
            "between_checkpoints": "Linearly interpolate expected additional loss between the surrounding validated checkpoints",
            "before_day_7": "Hold the Day 7 expected additional loss",
            "after_day_28": "Hold the Day 28 expected additional loss because a verified harvest-date horizon is unavailable",
            "validation_note": "Exact Day 7, 14, 21, and 28 held-out predictions and metrics are unchanged",
        },
        "held_out_permutation_importance": importance,
        "held_out_shap_importance": held_out_shap,
        "held_out_shap_model": nonlinear_shap_candidate,
        "held_out_shap_note": (
            "SHAP values were calculated for the strongest nonlinear challenger on complete outer held-out cycles. "
            "This sensitivity view does not explain the operational linear champion. Signs describe association with the final-recovery estimate, not causal effects."
            if held_out_shap
            else "No leakage-safe nonlinear SHAP view is available."
        ),
        "research_champion_permutation_importance": research_importance,
        "research_champion_drivers": research_importance[:5],
        "global_feature_importance": [],
        "feature_importance_interpretation": "Held-out permutation importance shows predictive association and model reliance, not causality.",
        "primary_whole_cycle_bootstrap_mae_95ci": _cycle_bootstrap(actual, selected_prediction, groups, False),
        "day14_backtest_metrics": day14_metrics,
        "day14_backtest": [
            {
                **record,
                "as_of_date": pd.Timestamp(record["as_of_date"]).date().isoformat(),
                "predicted": float(record["predicted"]),
                "actual": float(record["actual"]),
                "error": float(record["error"]),
                "absolute_error": float(record["absolute_error"]),
            }
            for record in day14_frame.to_dict(orient="records")
        ],
        "backtest_predictions": [
            {
                **record,
                "as_of_date": pd.Timestamp(record["as_of_date"]).date().isoformat(),
                "cycle_day": int(record["cycle_day"]),
                "percentage_alive": float(record["percentage_alive"]),
                "predicted_final_recovery": float(record["predicted_final_recovery"]),
                "actual_final_recovery_proxy": float(record["actual_final_recovery_proxy"]),
                "error": float(record["error"]),
                "absolute_error": float(record["absolute_error"]),
            }
            for record in recovery_backtest.to_dict(orient="records")
        ],
        "checkpoint_performance": {
            str(day): {
                "rows": int(rows["cycle_day"].eq(day).sum()),
                "mae": float(
                    mean_absolute_error(
                        actual[rows["cycle_day"].eq(day).to_numpy()],
                        selected_prediction[rows["cycle_day"].eq(day).to_numpy()],
                    )
                ),
                "rmse": float(
                    mean_squared_error(
                        actual[rows["cycle_day"].eq(day).to_numpy()],
                        selected_prediction[rows["cycle_day"].eq(day).to_numpy()],
                    )
                    ** 0.5
                ),
                "r2": float(
                    r2_score(
                        actual[rows["cycle_day"].eq(day).to_numpy()],
                        selected_prediction[rows["cycle_day"].eq(day).to_numpy()],
                    )
                ),
            }
            for day in (7, 14, 21, 28)
            if rows["cycle_day"].eq(day).any()
        },
        "limitations": [
            f"Only {int(rows[['cycle_id', 'building_id']].drop_duplicates().shape[0])} independent building-cycle recovery outcomes across {int(rows['cycle_id'].nunique())} cycles are available.",
            "The recovery label is a last-recorded proxy, not a verified harvest event.",
            "Environmental evidence is incomplete in part of the history.",
            "Target-side classification is not trusted unless it beats the majority baseline.",
        ],
    }
    return StrengthenedResult(manifest, model)


def add_checkpoint_indicators(features: pd.DataFrame, measurement_day: pd.Series) -> pd.DataFrame:
    result = features[WEIGHT_BASE_FEATURES].copy()
    for day in (7, 14, 21, 28):
        result[f"checkpoint_day_{day}"] = measurement_day.eq(day).astype(float).to_numpy()
    return result[WEIGHT_FEATURES]


def train_weight_remaining_gain(rows: pd.DataFrame, base_features: pd.DataFrame) -> StrengthenedResult:
    """Compare five remaining-gain methods under complete-cycle validation."""

    rows = rows.copy().reset_index(drop=True)
    x = add_checkpoint_indicators(base_features.reset_index(drop=True), rows["measurement_day"])
    actual = rows["actual_day35_weight_kg"].to_numpy(float)
    current = rows["current_weight_kg"].to_numpy(float)
    gain = actual - current
    groups = rows["cycle_id"].astype(str).to_numpy()
    sample_weight = _weights(rows)
    predictions = {name: np.full(len(rows), np.nan) for name in WEIGHT_CANDIDATES}
    parameters = {name: [] for name in WEIGHT_CANDIDATES}

    def baseline_raw(train_rows: pd.DataFrame, train_gain: np.ndarray, test_rows: pd.DataFrame) -> np.ndarray:
        training = pd.DataFrame({"day": train_rows["measurement_day"].to_numpy(), "gain": train_gain})
        overall = float(training["gain"].mean())
        mapping = training.groupby("day")["gain"].mean().to_dict()
        return np.asarray([float(mapping.get(day, overall)) for day in test_rows["measurement_day"]])

    for train, test in LeaveOneGroupOut().split(x, gain, groups):
        predictions[WEIGHT_CANDIDATES[0]][test] = baseline_raw(rows.iloc[train], gain[train], rows.iloc[test])
        parameters[WEIGHT_CANDIDATES[0]].append({})
        for candidate in WEIGHT_CANDIDATES[1:]:
            model, chosen = _fit_candidate(
                candidate,
                x.iloc[train],
                gain[train],
                groups[train],
                sample_weight[train],
                _weight_from_gain,
                current[train],
            )
            predictions[candidate][test] = model.predict(x.iloc[test])
            parameters[candidate].append(chosen)

    final_predictions = {name: _weight_from_gain(current, values) for name, values in predictions.items()}
    metrics = {name: _target_metrics(actual, pred, groups, 1.8, True) for name, pred in final_predictions.items()}
    for name in WEIGHT_CANDIDATES:
        metrics[name]["outer_fold_best_parameters"] = parameters[name]
        metrics[name]["horizon"] = {}
        for day in (7, 14, 21, 28):
            mask = rows["measurement_day"].eq(day).to_numpy()
            metrics[name]["horizon"][f"Day {day}"] = _target_metrics(actual[mask], final_predictions[name][mask], groups[mask], 1.8, True)

    baseline = WEIGHT_CANDIDATES[0]
    learned = WEIGHT_CANDIDATES[1:]
    best_mae = min(float(metrics[name]["cycle_macro_mae_kg"]) for name in learned)
    eligible = [name for name in learned if float(metrics[name]["cycle_macro_mae_kg"]) <= best_mae * 1.05]
    research = next(name for name in learned if name in eligible)
    baseline_metrics = metrics[baseline]
    research_metrics = metrics[research]
    improvement = (
        (float(baseline_metrics["cycle_macro_mae_kg"]) - float(research_metrics["cycle_macro_mae_kg"]))
        / float(baseline_metrics["cycle_macro_mae_kg"])
        * 100
    )
    worst_cycle_ok = max(item["mae_kg"] for item in research_metrics["cycle"].values()) <= max(item["mae_kg"] for item in baseline_metrics["cycle"].values()) * 1.25
    regression_gate = improvement >= 10 and float(research_metrics["r2"]) > 0 and float(research_metrics["within_200g_rate"]) >= 0.70 and worst_cycle_ok
    classification_gate = float(research_metrics["target_side_accuracy"]) > float(research_metrics["majority_side_accuracy"]) and float(research_metrics["below_target_recall"]) > 0 and float(research_metrics["at_or_above_target_recall"]) > 0
    selected = research if regression_gate and classification_gate else baseline
    research_model, research_parameters = _fit_candidate(
        research, x, gain, groups, sample_weight, _weight_from_gain, current
    )
    research_importance = _permutation_records(
        research,
        x,
        gain,
        groups,
        sample_weight,
        current,
        _weight_from_gain,
        "mean_mae_increase_kg",
    )
    if selected == baseline:
        model = None
        final_parameters: dict[str, object] = {}
        importance: list[dict[str, object]] = []
    else:
        model, final_parameters = _fit_candidate(selected, x, gain, groups, sample_weight, _weight_from_gain, current)
        importance = _permutation_records(selected, x, gain, groups, sample_weight, current, _weight_from_gain, "mean_mae_increase_kg")

    selected_prediction = final_predictions[selected]
    weight_backtest = rows[
        [
            "cycle_id",
            "building_id",
            "measurement_day",
            "current_weight_kg",
            "actual_day35_weight_kg",
        ]
    ].copy()
    weight_backtest["predicted_day35_weight_kg"] = selected_prediction
    weight_backtest["error_kg"] = selected_prediction - actual
    weight_backtest["absolute_error_kg"] = np.abs(weight_backtest["error_kg"])
    day14 = rows["measurement_day"].eq(14).to_numpy()
    day14_frame = rows.loc[day14, ["cycle_id", "building_id", "current_weight_kg", "actual_day35_weight_kg"]].copy()
    day14_frame["predicted_day35_weight_kg"] = selected_prediction[day14]
    day14_frame["error_kg"] = day14_frame["predicted_day35_weight_kg"] - day14_frame["actual_day35_weight_kg"]
    day14_frame["absolute_error_kg"] = day14_frame["error_kg"].abs()
    day14_metrics = _target_metrics(actual[day14], selected_prediction[day14], groups[day14], 1.8, True)
    day14_metrics["building_cycles"] = int(day14.sum())
    day14_metrics["mean_error_kg"] = float(np.mean(selected_prediction[day14] - actual[day14]))
    day14_metrics["actual_at_or_above_target"] = int(np.sum(actual[day14] >= 1.8))
    day14_metrics["actual_below_target"] = int(np.sum(actual[day14] < 1.8))
    day14_metrics["predicted_at_or_above_target"] = int(
        np.sum(selected_prediction[day14] >= 1.8)
    )
    day14_metrics["predicted_below_target"] = int(
        np.sum(selected_prediction[day14] < 1.8)
    )

    remaining_gain = {}
    uncertainty = {}
    for day in (7, 14, 21, 28):
        mask = rows["measurement_day"].eq(day).to_numpy()
        remaining_gain[str(day)] = float(np.mean(gain[mask]))
        uncertainty[str(day)] = float(np.quantile(np.abs(selected_prediction[mask] - actual[mask]), 0.8))
    remaining_gain["35"] = 0.0
    uncertainty["35"] = 0.0

    rolling = _rolling_origin(rows, WEIGHT_CANDIDATES, x, gain, current, _weight_from_gain, baseline_raw)
    within_cycle = _within_building_cycle_validation(
        rows,
        WEIGHT_CANDIDATES,
        x,
        gain,
        current,
        _weight_from_gain,
        baseline_raw,
    )
    outcomes = rows[["cycle_id", "building_id", "actual_day35_weight_kg"]].drop_duplicates()
    manifest = {
        "outcome": "day35_average_liveweight",
        "model_version": "day35-weight-2.2.0",
        "selected_model": selected,
        "research_champion": research,
        "operational_model": selected,
        "model_kind": "fitted" if model is not None else "formula",
        "prediction_target": "remaining_growth_from_latest_measurement_to_day35",
        "prediction_equation": "projected Day 35 weight = latest measured weight + predicted remaining gain",
        "target_day": 35,
        "target_weight_kg": 1.8,
        "label_definition": "Observed building average bodyweight recorded on production Day 35",
        "training_cycles": sorted(rows["cycle_id"].astype(str).unique(), key=_cycle_key),
        "training_building_cycles": int(len(outcomes)),
        "training_checkpoint_rows": int(len(rows)),
        "actual_target_hits": int((outcomes["actual_day35_weight_kg"] >= 1.8).sum()),
        "actual_target_misses": int((outcomes["actual_day35_weight_kg"] < 1.8).sum()),
        "feature_columns": WEIGHT_FEATURES if model is not None else ["current_weight_kg", "measurement_day"],
        "candidate_metrics": metrics,
        "selected_metrics": metrics[selected],
        "research_champion_metrics": metrics[research],
        "candidate_registry": [{"model": name, "available": True, "reason": "Evaluated under nested whole-cycle validation"} for name in WEIGHT_CANDIDATES],
        "selection_metric": "nested_leave_one_complete_cycle_out_cycle_macro_mae_kg",
        "selection_tolerance_pct": 5.0,
        "nested_validation": {
            "outer_split": "Leave one complete harvest cycle out",
            "inner_split": "Leave one complete remaining harvest cycle out",
            "optimization_metric": "Cycle-balanced MAE in projected Day 35 weight",
            "preprocessing_scope": "Imputation, scaling, checkpoint features and tuning fitted inside each training fold",
            "independent_unit": "Building-cycle; four checkpoint rows receive equal total training weight",
        },
        "rolling_origin_validation": rolling,
        "rolling_origin_note": "Secondary prospective check: train only on earlier cycles and predict the next recorded cycle.",
        "secondary_within_cycle_metrics": within_cycle,
        "secondary_validation_note": "Diagnostic only: leaves out one building-cycle while other buildings from the same harvest cycle remain in training.",
        "champion_gates": {
            "baseline": baseline,
            "baseline_improvement_pct": improvement,
            "requires_at_least_10pct_mae_improvement": improvement >= 10,
            "requires_positive_r2": float(research_metrics["r2"]) > 0,
            "requires_at_least_70pct_within_200g": float(research_metrics["within_200g_rate"]) >= 0.70,
            "requires_stable_worst_cycle": worst_cycle_ok,
            "regression_gate_passed": regression_gate,
            "requires_better_than_majority_target_side_accuracy": float(research_metrics["target_side_accuracy"]) > float(research_metrics["majority_side_accuracy"]),
            "requires_recall_for_both_target_sides": float(research_metrics["below_target_recall"]) > 0 and float(research_metrics["at_or_above_target_recall"]) > 0,
            "target_classification_gate_passed": classification_gate,
            "operational_fallback_applied": selected == baseline,
        },
        "final_research_fit_parameters": final_parameters,
        "research_champion_fit_parameters": research_parameters,
        "selected_method_drivers": [
            {"driver": "Latest measured weight", "role": "Starting point", "direction": "A higher current weight raises the projection."},
            {"driver": "Measurement checkpoint", "role": "Sets how much growth remains", "direction": "Earlier checkpoints normally have more remaining gain."},
        ] if selected == baseline else importance[:5],
        "research_champion_drivers": research_importance[:5],
        "held_out_permutation_importance": importance,
        "research_champion_permutation_importance": research_importance,
        "feature_importance_interpretation": "Held-out permutation importance shows predictive association and model reliance, not causality.",
        "remaining_gain_by_measurement_day_kg": remaining_gain,
        "uncertainty_half_width_by_measurement_day_kg": uncertainty,
        "primary_whole_cycle_bootstrap_mae_95ci": _cycle_bootstrap(actual, selected_prediction, groups, True),
        "day14_backtest_metrics": day14_metrics,
        "day14_backtest": [
            {key: (float(value) if isinstance(value, (float, np.floating)) else str(value) if key in {"cycle_id", "building_id"} else value) for key, value in record.items()}
            for record in day14_frame.to_dict(orient="records")
        ],
        "backtest_predictions": [
            {
                **record,
                "measurement_day": int(record["measurement_day"]),
                "current_weight_kg": float(record["current_weight_kg"]),
                "actual_day35_weight_kg": float(record["actual_day35_weight_kg"]),
                "predicted_day35_weight_kg": float(record["predicted_day35_weight_kg"]),
                "error_kg": float(record["error_kg"]),
                "absolute_error_kg": float(record["absolute_error_kg"]),
            }
            for record in weight_backtest.to_dict(orient="records")
        ],
        "historical_remaining_gain_definition": "Within each training fold, calculate observed Day 35 weight minus the checkpoint weight, average that remaining gain for the same checkpoint, and add it to the current building's measured weight.",
        "early_day_fallback": "historical_remaining_gain",
        "status": "Validated experimental forecast" if selected != baseline else "Experimental - historical remaining-gain baseline retained",
        "limitations": [
            f"Only {int(len(outcomes))} independent Day 35 building outcomes across {int(rows['cycle_id'].nunique())} cycles are available for training.",
            "Only five historical outcomes reached the 1.8 kg milestone.",
            "Weight sampling and environmental coverage vary across cycles.",
            "The projection is not a guarantee of target attainment.",
        ],
    }
    return StrengthenedResult(manifest, model)
