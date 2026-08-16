"""Research-only pooled versus checkpoint versus hybrid architecture test.

This module deliberately writes outside ``models/``.  It rebuilds daily
Day 7--34 landmarks from the authoritative workbook, compares all candidates
on identical held-out checkpoint rows, and retains daily evidence for the
architectures that can genuinely score intervening days.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .biology_aware_modeling import (
    DAY35_TARGET_G,
    LANDMARK_DAYS,
    RECOVERY_FEATURES,
    WEIGHT_FEATURES,
    _curve_predict,
    _fit_curve,
    _fit_negative_binomial,
    _kalman_predict,
    _predict_negative_binomial,
    build_daily_landmarks,
)
from .bodyweight_modeling_review import CHECKPOINTS, SEED
from .data import CanaryDataset, load_workbook
from .farmwide_features import assert_primary_schema_has_no_identity
from .farmwide_modeling import build_source_quality_audit


ROUND_VERSION = "robust-architecture-1.0.0"
RECOVERY_TARGET = 0.95
ARCHITECTURES = ("pooled", "checkpoint", "hybrid")


@dataclass(frozen=True)
class ArchitectureCandidate:
    name: str
    outcome: str
    architecture: str
    family: str
    target_form: str
    feature_block: str
    complexity: int
    daily_capable: bool
    description: str


def _candidate(
    name: str,
    outcome: str,
    architecture: str,
    family: str,
    target_form: str,
    feature_block: str,
    complexity: int,
    daily_capable: bool,
    description: str,
) -> ArchitectureCandidate:
    return ArchitectureCandidate(name, outcome, architecture, family, target_form, feature_block, complexity, daily_capable, description)


def _learned_candidates(outcome: str) -> tuple[ArchitectureCandidate, ...]:
    prefix = "recovery" if outcome == "recovery" else "weight"
    rows: list[ArchitectureCandidate] = []
    families = (
        ("ridge", 3), ("elastic_net", 4), ("huber", 4),
        ("random_forest", 6), ("extra_trees", 6), ("hist_gradient", 6),
        ("xgboost", 7), ("lightgbm", 7), ("catboost", 7),
    )
    for architecture in ARCHITECTURES:
        for family, complexity in families:
            # Direct and remaining formulations are compared for Ridge and the
            # three principal boosting families. Other families use the more
            # biologically constrained remaining formulation.
            target_forms = ("direct", "remaining") if family in {"ridge", "hist_gradient", "xgboost", "lightgbm", "catboost"} else ("remaining",)
            for target_form in target_forms:
                rows.append(_candidate(
                    f"{architecture}_{target_form}_{family}", prefix, architecture, family,
                    target_form, "full", complexity + (1 if architecture == "hybrid" else 0),
                    architecture != "checkpoint",
                    f"{architecture.title()} {family.replace('_', ' ')} using the {target_form} target.",
                ))
    return tuple(rows)


RECOVERY_BASELINES = (
    _candidate("current_survival", "recovery", "pooled", "persistence", "direct", "core", 0, True, "Current survival persists to harvest."),
    _candidate("daily_age_remaining_loss", "recovery", "pooled", "age_baseline", "remaining", "core", 1, True, "Fold-local remaining loss by daily age."),
    _candidate("pooled_negative_binomial_hazard", "recovery", "pooled", "negative_binomial", "remaining", "full", 3, True, "Population-at-risk negative-binomial loss model."),
)
WEIGHT_BASELINES = (
    _candidate("historical_remaining_gain", "weight", "pooled", "age_baseline", "remaining", "core", 0, True, "Fold-local historical remaining gain by daily age."),
    _candidate("target_curve_ratio", "weight", "pooled", "target_ratio", "direct", "core", 1, True, "Latest observed target-relative pace projected to Day 35."),
    _candidate("pooled_target_anchored_kalman", "weight", "pooled", "kalman", "direct", "full", 3, True, "Target-anchored state-space growth update."),
    _candidate("pooled_gompertz_partial_pooling", "weight", "pooled", "gompertz", "direct", "full", 4, True, "Fold-local Gompertz growth curve with shrinkage."),
    _candidate("pooled_logistic_partial_pooling", "weight", "pooled", "logistic", "direct", "full", 5, True, "Fold-local logistic growth curve with shrinkage."),
)
RECOVERY_CANDIDATES = RECOVERY_BASELINES + _learned_candidates("recovery")
WEIGHT_CANDIDATES = WEIGHT_BASELINES + _learned_candidates("weight")


SCREENING_FAMILIES = {"persistence", "age_baseline", "target_ratio", "negative_binomial", "kalman", "gompertz", "logistic", "ridge", "hist_gradient", "xgboost", "lightgbm", "catboost"}
BIOLOGICAL_FAMILIES = {"negative_binomial", "kalman", "gompertz", "logistic"}


class FoldWinsorizer(BaseEstimator, TransformerMixin):
    """Fold-local robust clipping with sklearn-compatible feature names."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, values: Any, y: Any = None) -> "FoldWinsorizer":
        array = np.asarray(values, dtype=float)
        self.lower_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, values: Any) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(list(input_features) if input_features is not None else [], dtype=object)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _cycle_key(value: str) -> tuple[int, int, str]:
    try:
        year, cycle = str(value).split("-", 1)
        return int(year), int(cycle), str(value)
    except (TypeError, ValueError):
        return 0, 0, str(value)


def _hybrid_columns(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Add regularized age-phase effects shared across all building-cycles."""
    result = frame.copy()
    age = result["review_day"].astype(float)
    result["age_scaled"] = age / 35.0
    result["age_squared"] = result["age_scaled"] ** 2
    phase_starts = (7, 14, 21, 28)
    for start in phase_starts:
        end = start + 6 if start < 28 else 34
        indicator = age.between(start, end).astype(float)
        result[f"phase_{start}_indicator"] = indicator
        signal = result["percentage_alive"] if outcome == "recovery" else result["current_ratio_to_target"]
        result[f"phase_{start}_signal"] = indicator * signal
        result[f"phase_{start}_horizon"] = indicator * result["days_to_day35"]
    return result


HYBRID_FEATURES = [
    "age_scaled", "age_squared",
    *[f"phase_{day}_indicator" for day in CHECKPOINTS],
    *[f"phase_{day}_signal" for day in CHECKPOINTS],
    *[f"phase_{day}_horizon" for day in CHECKPOINTS],
]


def feature_columns(candidate: ArchitectureCandidate) -> list[str]:
    if candidate.outcome == "recovery":
        core = [
            "review_day", "days_to_day35", "beginning_inventory", "percentage_alive",
            "population_loss_pct", "mortality_recent_3d_per_1000",
            "mortality_recent_7d_per_1000", "mortality_ewma_per_1000",
            "weight_ratio_to_target", "weight_staleness_days", "record_completeness_ratio",
        ]
        columns = core if candidate.feature_block == "core" else list(RECOVERY_FEATURES)
    else:
        core = [
            "review_day", "days_to_day35", "current_weight_g", "current_target_g",
            "current_ratio_to_target", "current_gap_to_target_g", "latest_measurement_day",
            "measurement_staleness_days", "weight_measurement_count", "robust_weight_slope_g_day",
            "recent_adg_all_g_day", "survival_pct", "mortality_recent_7d_per_1000",
        ]
        columns = core if candidate.feature_block == "core" else list(WEIGHT_FEATURES)
    if candidate.feature_block == "no_environment":
        columns = [column for column in columns if not any(token in column for token in ("temperature", "humidity", "environment", "heat", "cold"))]
    elif candidate.feature_block == "no_peer":
        columns = [column for column in columns if "peer" not in column]
    if candidate.architecture == "hybrid":
        columns += HYBRID_FEATURES
    columns = list(dict.fromkeys(columns))
    assert_primary_schema_has_no_identity(columns)
    lowered = " ".join(columns).lower()
    if "feed" in lowered:
        raise AssertionError("Feed entered a primary architecture schema.")
    return columns


def candidate_grid(candidate: ArchitectureCandidate) -> list[dict[str, Any]]:
    if candidate.family in {"persistence", "age_baseline", "target_ratio"}:
        return [{}]
    if candidate.family in {"negative_binomial", "gompertz", "logistic"}:
        return [{}]
    if candidate.family == "kalman":
        return [{"q": q, "r": r} for q, r in ((0.0002, 0.0025), (0.0005, 0.005), (0.001, 0.01))]
    if candidate.family == "ridge":
        return [{"alpha": value, "clip": clip} for value, clip in ((1.0, (0.0, 1.0)), (10.0, (0.01, 0.99)), (100.0, (0.05, 0.95)))]
    if candidate.family == "elastic_net":
        values = ((0.0001, 0.1), (0.001, 0.5), (0.01, 0.9)) if candidate.outcome == "recovery" else ((0.1, 0.1), (1.0, 0.5), (10.0, 0.9))
        return [{"alpha": alpha, "l1_ratio": ratio, "clip": (0.01, 0.99)} for alpha, ratio in values]
    if candidate.family == "huber":
        return [{"epsilon": epsilon, "alpha": alpha, "clip": (0.01, 0.99)} for epsilon, alpha in ((1.2, 0.001), (1.35, 0.01), (1.5, 0.1))]
    if candidate.family in {"random_forest", "extra_trees"}:
        return [{"depth": depth, "leaf": leaf, "features": features, "clip": (0.0, 1.0)} for depth, leaf, features in ((2, 4, 0.7), (3, 6, 0.8), (4, 8, 1.0))]
    if candidate.family in {"hist_gradient", "xgboost", "lightgbm", "catboost"}:
        return [
            {"trees": 80, "rate": 0.03, "depth": 1, "leaf": 8, "l2": 20.0, "clip": (0.0, 1.0)},
            {"trees": 120, "rate": 0.025, "depth": 2, "leaf": 12, "l2": 40.0, "clip": (0.01, 0.99)},
            {"trees": 160, "rate": 0.02, "depth": 2, "leaf": 16, "l2": 80.0, "clip": (0.05, 0.95)},
        ]
    raise ValueError(candidate.family)


def _pipeline(candidate: ArchitectureCandidate, parameters: dict[str, Any], seed: int) -> Pipeline:
    clip = parameters.get("clip", (0.01, 0.99))
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("winsor", FoldWinsorizer(*clip)),
    ]
    if candidate.family in {"ridge", "elastic_net", "huber"}:
        steps.append(("scale", StandardScaler()))
    if candidate.family == "ridge":
        model: Any = Ridge(alpha=float(parameters["alpha"]))
    elif candidate.family == "elastic_net":
        model = ElasticNet(alpha=float(parameters["alpha"]), l1_ratio=float(parameters["l1_ratio"]), max_iter=100000, tol=1e-3, random_state=seed)
    elif candidate.family == "huber":
        model = HuberRegressor(epsilon=float(parameters["epsilon"]), alpha=float(parameters["alpha"]), max_iter=20000, tol=1e-4)
    elif candidate.family == "random_forest":
        model = RandomForestRegressor(n_estimators=400, max_depth=int(parameters["depth"]), min_samples_leaf=int(parameters["leaf"]), max_features=float(parameters["features"]), random_state=seed, n_jobs=1)
    elif candidate.family == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=400, max_depth=int(parameters["depth"]), min_samples_leaf=int(parameters["leaf"]), max_features=float(parameters["features"]), random_state=seed, n_jobs=1)
    elif candidate.family == "hist_gradient":
        model = HistGradientBoostingRegressor(max_iter=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_leaf_nodes=max(3, 2 ** int(parameters["depth"]) + 1), min_samples_leaf=int(parameters["leaf"]), l2_regularization=float(parameters["l2"]), random_state=seed)
    elif candidate.family == "xgboost":
        model = XGBRegressor(n_estimators=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_depth=int(parameters["depth"]), min_child_weight=float(parameters["leaf"]), reg_lambda=float(parameters["l2"]), reg_alpha=1.0, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=1, verbosity=0)
    elif candidate.family == "lightgbm":
        model = LGBMRegressor(n_estimators=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_depth=int(parameters["depth"]), num_leaves=max(2, 2 ** int(parameters["depth"]) - 1), min_child_samples=int(parameters["leaf"]), reg_lambda=float(parameters["l2"]), reg_alpha=1.0, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    elif candidate.family == "catboost":
        model = CatBoostRegressor(iterations=int(parameters["trees"]), learning_rate=float(parameters["rate"]), depth=int(parameters["depth"]), l2_leaf_reg=float(parameters["l2"]), random_seed=seed, loss_function="RMSE", random_strength=0.5, allow_writing_files=False, thread_count=1, verbose=False)
    else:
        raise ValueError(candidate.family)
    steps.append(("model", model))
    return Pipeline(steps)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _baseline(train: pd.DataFrame) -> dict[str, Any]:
    mapping = {
        int(day): _weighted_mean(group["remaining_target"], group["sample_weight"])
        for day, group in train.groupby("review_day")
    }
    return {"mapping": mapping, "fallback": _weighted_mean(train["remaining_target"], train["sample_weight"])}


def _baseline_predict(frame: pd.DataFrame, baseline: dict[str, Any], outcome: str) -> np.ndarray:
    remaining = np.asarray([baseline["mapping"].get(int(day), baseline["fallback"]) for day in frame["review_day"]], dtype=float)
    if outcome == "recovery":
        return frame["current_value"].to_numpy(float) - np.maximum(remaining, 0.0)
    return frame["current_value"].to_numpy(float) + remaining


def _target(train: pd.DataFrame, candidate: ArchitectureCandidate) -> np.ndarray:
    return train["remaining_target"].to_numpy(float) if candidate.target_form == "remaining" else train["actual"].to_numpy(float)


def _fit_single(train: pd.DataFrame, candidate: ArchitectureCandidate, parameters: dict[str, Any], seed: int) -> dict[str, Any]:
    train = _hybrid_columns(train, candidate.outcome) if candidate.architecture == "hybrid" else train.copy()
    baseline = _baseline(train)
    if candidate.family in {"persistence", "age_baseline", "target_ratio"}:
        return {"baseline": baseline, "family": candidate.family}
    if candidate.family == "negative_binomial":
        return {"baseline": baseline, "family": candidate.family, "biological": _fit_negative_binomial(train)}
    if candidate.family in {"gompertz", "logistic"}:
        return {"baseline": baseline, "family": candidate.family, "biological": _fit_curve(train, candidate.family)}
    if candidate.family == "kalman":
        return {"baseline": baseline, "family": candidate.family, "parameters": parameters}
    columns = feature_columns(candidate)
    model = _pipeline(candidate, parameters, seed)
    fit_kwargs: dict[str, Any] = {}
    if candidate.family != "elastic_net":
        fit_kwargs["model__sample_weight"] = train["sample_weight"].to_numpy(float)
    model.fit(train[columns], _target(train, candidate), **fit_kwargs)
    return {"baseline": baseline, "family": candidate.family, "model": model, "features": columns}


def fit_candidate(train: pd.DataFrame, candidate: ArchitectureCandidate, parameters: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    if candidate.architecture == "checkpoint":
        checkpoint_train = train.loc[train["review_day"].isin(CHECKPOINTS)]
        entries = {int(day): _fit_single(group.reset_index(drop=True), candidate, parameters, seed) for day, group in checkpoint_train.groupby("review_day")}
        return {"architecture": "checkpoint", "entries": entries, "candidate": asdict(candidate), "parameters": parameters}
    return {"architecture": candidate.architecture, "entry": _fit_single(train.reset_index(drop=True), candidate, parameters, seed), "candidate": asdict(candidate), "parameters": parameters}


def _predict_single(entry: dict[str, Any], frame: pd.DataFrame, candidate: ArchitectureCandidate) -> np.ndarray:
    prepared = _hybrid_columns(frame, candidate.outcome) if candidate.architecture == "hybrid" else frame.copy()
    if candidate.family == "persistence":
        prediction = prepared["current_value"].to_numpy(float)
    elif candidate.family == "age_baseline":
        prediction = _baseline_predict(prepared, entry["baseline"], candidate.outcome)
    elif candidate.family == "target_ratio":
        prediction = prepared["current_weight_g"].to_numpy(float) / prepared["current_target_g"].clip(lower=1).to_numpy(float) * DAY35_TARGET_G
    elif candidate.family == "negative_binomial":
        prediction = _predict_negative_binomial(entry["biological"], prepared)
    elif candidate.family == "kalman":
        prediction, _ = _kalman_predict(prepared, float(entry["parameters"]["q"]), float(entry["parameters"]["r"]))
    elif candidate.family in {"gompertz", "logistic"}:
        prediction = _curve_predict(entry["biological"], prepared)
    else:
        raw = np.asarray(entry["model"].predict(prepared[entry["features"]]), dtype=float).reshape(-1)
        if candidate.target_form == "remaining":
            if candidate.outcome == "recovery":
                prediction = prepared["current_value"].to_numpy(float) - np.maximum(raw, 0.0)
            else:
                prediction = prepared["current_value"].to_numpy(float) + raw
        else:
            prediction = raw
    if candidate.outcome == "recovery":
        return np.minimum(np.clip(prediction, 0.0, 1.0), prepared["current_value"].to_numpy(float))
    return np.clip(prediction, 100.0, 3500.0)


def predict_candidate(bundle: dict[str, Any], frame: pd.DataFrame, candidate: ArchitectureCandidate) -> np.ndarray:
    if bundle["architecture"] != "checkpoint":
        return _predict_single(bundle["entry"], frame, candidate)
    if not frame["review_day"].isin(CHECKPOINTS).all():
        raise ValueError("Checkpoint-specific candidates cannot score intervening days.")
    output = pd.Series(index=frame.index, dtype=float)
    for day, group in frame.groupby("review_day"):
        output.loc[group.index] = _predict_single(bundle["entries"][int(day)], group, candidate)
    return output.loc[frame.index].to_numpy(float)


def _metric_factor(outcome: str) -> float:
    return 100.0 if outcome == "recovery" else 1.0


def _cycle_macro_rmse(actual: np.ndarray, predicted: np.ndarray, cycles: np.ndarray, outcome: str) -> float:
    factor = _metric_factor(outcome)
    return float(np.mean([
        mean_squared_error(actual[cycles == cycle], predicted[cycles == cycle]) ** 0.5 * factor
        for cycle in pd.unique(cycles)
    ]))


def _candidate_frame(frame: pd.DataFrame, candidate: ArchitectureCandidate) -> pd.DataFrame:
    if candidate.architecture == "checkpoint":
        return frame.loc[frame["review_day"].isin(CHECKPOINTS)].reset_index(drop=True)
    return frame.reset_index(drop=True)


def tune_candidate(train: pd.DataFrame, candidate: ArchitectureCandidate, seed: int = SEED) -> dict[str, Any]:
    train = _candidate_frame(train, candidate)
    grid = candidate_grid(candidate)
    if len(grid) == 1 or train["cycle_id"].nunique() < 2:
        return grid[0]
    groups = train["cycle_id"].astype(str).to_numpy()
    actual = train["actual"].to_numpy(float)
    scores: list[tuple[float, int, dict[str, Any]]] = []
    for order, parameters in enumerate(grid):
        predicted = np.full(len(train), np.nan)
        for fit_index, valid_index in LeaveOneGroupOut().split(train, groups=groups):
            fitted = fit_candidate(train.iloc[fit_index], candidate, parameters, seed)
            predicted[valid_index] = predict_candidate(fitted, train.iloc[valid_index], candidate)
        checkpoint = train["review_day"].isin(CHECKPOINTS).to_numpy()
        scores.append((_cycle_macro_rmse(actual[checkpoint], predicted[checkpoint], groups[checkpoint], candidate.outcome), order, parameters))
    return min(scores, key=lambda item: (item[0], item[1]))[2]


def evaluate_nested_logo(
    frame: pd.DataFrame,
    candidate: ArchitectureCandidate,
    *,
    view: str = "cycle",
    seed: int = SEED,
    fixed_parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    working = _candidate_frame(frame, candidate)
    if view == "cycle":
        groups = working["cycle_id"].astype(str).to_numpy()
    elif view == "building_label":
        groups = working["building_id"].astype(str).to_numpy()
    elif view == "building_cycle":
        groups = (working["cycle_id"].astype(str) + "::" + working["building_id"].astype(str)).to_numpy()
    else:
        raise ValueError(view)
    rows: list[dict[str, Any]] = []
    settings: list[dict[str, Any]] = []
    for train_index, test_index in LeaveOneGroupOut().split(working, groups=groups):
        train = working.iloc[train_index].reset_index(drop=True)
        test = working.iloc[test_index]
        parameters = fixed_parameters or tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        held = str(groups[test_index][0])
        settings.append({"held_out_group": held, "validation_view": view, **parameters})
        for source, value in zip(test.to_dict("records"), predicted):
            rows.append({
                "candidate": candidate.name, "architecture": candidate.architecture,
                "family": candidate.family, "target_form": candidate.target_form,
                "validation_view": view, "held_out_group": held,
                "cycle_id": source["cycle_id"], "building_id": source["building_id"],
                "review_day": int(source["review_day"]), "as_of_date": source["as_of_date"],
                "actual": float(source["actual"]), "predicted": float(value),
                "sample_weight": float(source["sample_weight"]),
                "checkpoint_status": source["checkpoint_status"],
            })
    return pd.DataFrame(rows), settings


def summarize(predictions: pd.DataFrame, outcome: str, *, bootstrap: bool = True, seed: int = SEED) -> dict[str, Any]:
    factor = _metric_factor(outcome)
    actual = predictions["actual"].to_numpy(float)
    predicted = predictions["predicted"].to_numpy(float)
    error = (predicted - actual) * factor
    cycles = predictions["cycle_id"].astype(str).to_numpy()
    cycle_values = {
        cycle: mean_squared_error(actual[cycles == cycle], predicted[cycles == cycle]) ** 0.5 * factor
        for cycle in pd.unique(cycles)
    }
    fold_values = list(cycle_values.values())
    result: dict[str, Any] = {
        "cycle_macro_rmse": float(np.mean(list(cycle_values.values()))),
        "pooled_rmse": float(mean_squared_error(actual, predicted) ** 0.5 * factor),
        "mae": float(mean_absolute_error(actual, predicted) * factor),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(error)),
        "fold_sd_rmse": float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0,
        "worst_cycle_rmse": float(max(cycle_values.values())),
        "n_predictions": int(len(predictions)),
    }
    if outcome == "weight":
        absolute = np.abs(error)
        result.update({"within_100g": float(np.mean(absolute <= 100)), "within_200g": float(np.mean(absolute <= 200))})
    target = RECOVERY_TARGET if outcome == "recovery" else DAY35_TARGET_G
    actual_positive = actual >= target
    predicted_positive = predicted >= target
    tn = int((~actual_positive & ~predicted_positive).sum())
    fp = int((~actual_positive & predicted_positive).sum())
    fn = int((actual_positive & ~predicted_positive).sum())
    tp = int((actual_positive & predicted_positive).sum())
    below_recall = tn / (tn + fp) if (tn + fp) else np.nan
    above_recall = tp / (tp + fn) if (tp + fn) else np.nan
    available_recalls = [value for value in (below_recall, above_recall) if np.isfinite(value)]
    majority_accuracy = float(max(actual_positive.mean(), 1.0 - actual_positive.mean()))
    result.update({
        "target_tn": tn, "target_fp": fp, "target_fn": fn, "target_tp": tp,
        "below_target_recall": float(below_recall) if np.isfinite(below_recall) else np.nan,
        "at_or_above_target_recall": float(above_recall) if np.isfinite(above_recall) else np.nan,
        "balanced_target_accuracy": float(np.mean(available_recalls)) if available_recalls else np.nan,
        "majority_target_accuracy": majority_accuracy,
    })
    if bootstrap:
        rng = np.random.default_rng(seed)
        unique = np.asarray(pd.unique(cycles))
        draws = []
        for _ in range(2000):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            draws.append(float(np.mean([cycle_values[item] for item in chosen])))
        result["cycle_macro_rmse_ci_low"] = float(np.quantile(draws, 0.025))
        result["cycle_macro_rmse_ci_high"] = float(np.quantile(draws, 0.975))
    return result


def metrics_by_day(predictions: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for day, group in predictions.groupby("review_day"):
        row = summarize(group, outcome, bootstrap=False)
        rows.append({"review_day": int(day), **row})
    return pd.DataFrame(rows).sort_values("review_day")


def _finite_quantile(values: np.ndarray, level: float) -> float:
    clean = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(clean):
        return np.nan
    rank = int(np.ceil((len(clean) + 1) * level))
    return float(clean[min(max(rank, 1), len(clean)) - 1])


def conformal_logo(frame: pd.DataFrame, candidate: ArchitectureCandidate, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidate.daily_capable:
        raise ValueError("Daily conformal intervals require a pooled or hybrid candidate.")
    groups = frame["cycle_id"].astype(str).to_numpy()
    rows: list[pd.DataFrame] = []
    factor = _metric_factor(candidate.outcome)
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].copy()
        parameters = tune_candidate(train, candidate, seed)
        inner, _ = evaluate_nested_logo(train, candidate, seed=seed, fixed_parameters=parameters)
        residual_frame = inner[["review_day"]].copy()
        residual_frame["absolute_residual"] = np.abs(inner["actual"].to_numpy(float) - inner["predicted"].to_numpy(float))
        residual = residual_frame["absolute_residual"].to_numpy(float)
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        block = test[["cycle_id", "building_id", "review_day", "as_of_date", "actual"]].copy()
        block["candidate"] = candidate.name
        block["predicted"] = predicted
        for level in (0.8, 0.9):
            suffix = int(level * 100)
            pooled_q = _finite_quantile(residual, level)
            age_quantiles = {
                int(day): _finite_quantile(group["absolute_residual"].to_numpy(float), level)
                for day, group in residual_frame.groupby("review_day")
            }
            age_q = np.asarray([age_quantiles.get(int(day), pooled_q) for day in test["review_day"]], dtype=float)
            block[f"lower_{suffix}"] = predicted - age_q
            block[f"upper_{suffix}"] = predicted + age_q
            block[f"pooled_lower_{suffix}"] = predicted - pooled_q
            block[f"pooled_upper_{suffix}"] = predicted + pooled_q
        if candidate.outcome == "recovery":
            current = test["current_value"].to_numpy(float)
            for level in (80, 90):
                block[f"lower_{level}"] = np.clip(block[f"lower_{level}"], 0.0, current)
                block[f"upper_{level}"] = np.clip(block[f"upper_{level}"], 0.0, current)
                block[f"pooled_lower_{level}"] = np.clip(block[f"pooled_lower_{level}"], 0.0, current)
                block[f"pooled_upper_{level}"] = np.clip(block[f"pooled_upper_{level}"], 0.0, current)
        else:
            for level in (80, 90):
                block[f"lower_{level}"] = block[f"lower_{level}"].clip(100.0, 3500.0)
                block[f"upper_{level}"] = block[f"upper_{level}"].clip(100.0, 3500.0)
                block[f"pooled_lower_{level}"] = block[f"pooled_lower_{level}"].clip(100.0, 3500.0)
                block[f"pooled_upper_{level}"] = block[f"pooled_upper_{level}"].clip(100.0, 3500.0)
        for level in (80, 90):
            block[f"covered_{level}"] = block["actual"].between(block[f"lower_{level}"], block[f"upper_{level}"])
            block[f"width_{level}"] = (block[f"upper_{level}"] - block[f"lower_{level}"]) * factor
            block[f"pooled_covered_{level}"] = block["actual"].between(block[f"pooled_lower_{level}"], block[f"pooled_upper_{level}"])
            block[f"pooled_width_{level}"] = (block[f"pooled_upper_{level}"] - block[f"pooled_lower_{level}"]) * factor
        rows.append(block)
    intervals = pd.concat(rows, ignore_index=True)
    calibration = intervals.groupby("review_day", as_index=False).agg(
        coverage_80=("covered_80", "mean"), coverage_90=("covered_90", "mean"),
        mean_width_80=("width_80", "mean"), mean_width_90=("width_90", "mean"), n=("actual", "size"),
        pooled_coverage_80=("pooled_covered_80", "mean"), pooled_coverage_90=("pooled_covered_90", "mean"),
        pooled_mean_width_80=("pooled_width_80", "mean"), pooled_mean_width_90=("pooled_width_90", "mean"),
    )
    calibration["checkpoint_support_warning"] = calibration["n"].lt(20)
    return intervals, calibration


def temporal_stress(frame: pd.DataFrame, candidate: ArchitectureCandidate, seed: int = SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = _candidate_frame(frame, candidate)
    cycles = sorted(working["cycle_id"].astype(str).unique(), key=_cycle_key)
    rows = []
    for position in range(2, len(cycles)):
        train_cycles, test_cycle = cycles[:position], cycles[position]
        train = working.loc[working["cycle_id"].isin(train_cycles)].reset_index(drop=True)
        test = working.loc[working["cycle_id"].eq(test_cycle)]
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        block = test[["cycle_id", "building_id", "review_day", "as_of_date", "actual"]].copy()
        block["candidate"] = candidate.name
        block["predicted"] = predicted
        rows.append(block)
    predictions = pd.concat(rows, ignore_index=True)
    checkpoints = predictions.loc[predictions["review_day"].isin(CHECKPOINTS)]
    return predictions, summarize(checkpoints, candidate.outcome, bootstrap=False)


def _one_se_select(comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ordered = comparison.sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
    best = ordered.iloc[0]
    threshold = float(best["cycle_macro_rmse"] + best["fold_sd_rmse"] / np.sqrt(6))
    eligible = ordered.loc[ordered["cycle_macro_rmse"].le(threshold)].sort_values(["complexity", "cycle_macro_rmse", "candidate"])
    selected = str(eligible.iloc[0]["candidate"])
    return selected, {
        "lowest_error_candidate": str(best["candidate"]),
        "lowest_cycle_macro_rmse": float(best["cycle_macro_rmse"]),
        "one_standard_error_threshold": threshold,
        "selected_candidate": selected,
    }


def _feature_importance(
    frame: pd.DataFrame, candidate: ArchitectureCandidate, seed: int
) -> pd.DataFrame:
    if candidate.family in {"persistence", "age_baseline", "target_ratio", *BIOLOGICAL_FAMILIES}:
        return pd.DataFrame(columns=["candidate", "feature", "mean_rmse_increase", "folds"])
    groups = frame["cycle_id"].astype(str).to_numpy()
    # Hybrid age/phase columns are deterministic transformations. Permute the
    # source as-of features and allow the transformations to be recomputed;
    # shuffling a derived column before it exists would be invalid.
    columns = [column for column in feature_columns(candidate) if column in frame.columns]
    rng = np.random.default_rng(seed)
    rows = []
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index].copy()
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        base = predict_candidate(fitted, test, candidate)
        checkpoint = test["review_day"].isin(CHECKPOINTS).to_numpy()
        base_rmse = mean_squared_error(test.loc[checkpoint, "actual"], base[checkpoint]) ** 0.5 * _metric_factor(candidate.outcome)
        for column in columns:
            shuffled = test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            value = predict_candidate(fitted, shuffled, candidate)
            rmse = mean_squared_error(test.loc[checkpoint, "actual"], value[checkpoint]) ** 0.5 * _metric_factor(candidate.outcome)
            rows.append({"candidate": candidate.name, "held_out_cycle": str(groups[test_index][0]), "feature": column, "rmse_increase": float(rmse - base_rmse)})
    detail = pd.DataFrame(rows)
    return detail.groupby(["candidate", "feature"], as_index=False).agg(mean_rmse_increase=("rmse_increase", "mean"), folds=("held_out_cycle", "nunique")).sort_values("mean_rmse_increase", ascending=False)


def _tree_shap(frame: pd.DataFrame, candidate: ArchitectureCandidate, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if candidate.architecture == "checkpoint" or candidate.family not in {"random_forest", "extra_trees", "hist_gradient", "xgboost", "lightgbm", "catboost"}:
        return pd.DataFrame(), pd.DataFrame(), np.empty((0, 0)), np.empty((0, 0)), []
    groups = frame["cycle_id"].astype(str).to_numpy()
    shap_parts, value_parts, local_rows = [], [], []
    feature_names: list[str] = []
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].copy()
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        entry = fitted["entry"]
        prepared = _hybrid_columns(test, candidate.outcome) if candidate.architecture == "hybrid" else test
        pipeline: Pipeline = entry["model"]
        transformed = pipeline[:-1].transform(prepared[entry["features"]])
        try:
            feature_names = list(pipeline[:-1].get_feature_names_out(entry["features"]))
        except Exception:
            feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        values = np.asarray(explainer.shap_values(transformed), dtype=float)
        if values.ndim == 3:
            values = values[:, :, 0]
        shap_parts.append(values)
        value_parts.append(np.asarray(transformed, dtype=float))
        for row_index, source in enumerate(test.to_dict("records")):
            for column_index, feature in enumerate(feature_names):
                local_rows.append({
                    "candidate": candidate.name, "cycle_id": source["cycle_id"], "building_id": source["building_id"],
                    "review_day": int(source["review_day"]), "feature": feature,
                    "feature_value": float(transformed[row_index, column_index]), "shap_value": float(values[row_index, column_index]),
                })
    shap_values = np.vstack(shap_parts)
    feature_values = np.vstack(value_parts)
    local = pd.DataFrame(local_rows)
    global_frame = local.groupby("feature", as_index=False).agg(
        mean_abs_shap=("shap_value", lambda values: float(np.mean(np.abs(values)))),
        mean_shap=("shap_value", "mean"),
        positive_share=("shap_value", lambda values: float(np.mean(values > 0))),
        cycles=("cycle_id", "nunique"),
    ).sort_values("mean_abs_shap", ascending=False)
    return global_frame, local, shap_values, feature_values, feature_names


def _promotion_gate(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate_name: str,
    baseline_name: str,
    outcome: str,
    intervals: pd.DataFrame,
    audit_metrics: dict[str, Any],
    baseline_audit_metrics: dict[str, Any],
) -> dict[str, Any]:
    table = comparison.set_index("candidate")
    candidate, baseline = table.loc[candidate_name], table.loc[baseline_name]
    blocks = predictions.loc[predictions["candidate"].isin([candidate_name, baseline_name])]
    cycle_wins = 0
    for cycle in blocks["cycle_id"].unique():
        current = blocks.loc[(blocks["candidate"].eq(candidate_name)) & (blocks["cycle_id"].eq(cycle))]
        reference = blocks.loc[(blocks["candidate"].eq(baseline_name)) & (blocks["cycle_id"].eq(cycle))]
        if mean_squared_error(current["actual"], current["predicted"]) < mean_squared_error(reference["actual"], reference["predicted"]):
            cycle_wins += 1
    coverage = float(intervals["covered_80"].mean())
    checkpoint_stable = True
    for day in CHECKPOINTS:
        current = blocks.loc[blocks["candidate"].eq(candidate_name) & blocks["review_day"].eq(day)]
        reference = blocks.loc[blocks["candidate"].eq(baseline_name) & blocks["review_day"].eq(day)]
        current_rmse = mean_squared_error(current["actual"], current["predicted"]) ** .5
        reference_rmse = mean_squared_error(reference["actual"], reference["predicted"]) ** .5
        checkpoint_stable = checkpoint_stable and current_rmse <= reference_rmse * 1.05
    bias_limit = 0.5 if outcome == "recovery" else 50.0
    improvement_pct = float(
        100.0 * (baseline["cycle_macro_rmse"] - candidate["cycle_macro_rmse"])
        / baseline["cycle_macro_rmse"]
    )
    gate = {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "cycle_macro_rmse_improvement_pct": improvement_pct,
        "at_least_10pct_better_than_baseline": bool(candidate["cycle_macro_rmse"] <= baseline["cycle_macro_rmse"] * 0.90),
        "positive_held_out_r2": bool(candidate["r2"] > 0),
        "mae_not_worse_than_5pct": bool(candidate["mae"] <= baseline["mae"] * 1.05),
        "acceptable_bias": bool(abs(candidate["bias"]) <= bias_limit),
        "worst_cycle_not_worse_than_10pct": bool(candidate["worst_cycle_rmse"] <= baseline["worst_cycle_rmse"] * 1.10),
        "cycle_wins": cycle_wins,
        "wins_at_least_four_cycles": cycle_wins >= 4,
        "principal_checkpoints_stable": bool(checkpoint_stable),
        "interval_80_coverage": coverage,
        "credible_interval_coverage": bool(0.72 <= coverage <= 0.95),
        "later_cycle_audit_rmse": audit_metrics["cycle_macro_rmse"],
        "audit_not_materially_worse": bool(audit_metrics["cycle_macro_rmse"] <= baseline_audit_metrics["cycle_macro_rmse"] * 1.10),
    }
    boolean_keys = [key for key, value in gate.items() if isinstance(value, (bool, np.bool_))]
    gate["retrospective_gate_passed"] = bool(all(gate[key] for key in boolean_keys))
    return gate


def _score_audit(
    development: pd.DataFrame,
    audit: pd.DataFrame,
    candidate: ArchitectureCandidate,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    parameters = tune_candidate(development, candidate, seed)
    fitted = fit_candidate(development, candidate, parameters, seed)
    working_audit = _candidate_frame(audit, candidate)
    predicted = predict_candidate(fitted, working_audit, candidate)
    block = working_audit[["cycle_id", "building_id", "review_day", "as_of_date", "actual"]].copy()
    block["candidate"] = candidate.name
    block["predicted"] = predicted
    checkpoints = block.loc[block["review_day"].isin(CHECKPOINTS)]
    return block, summarize(checkpoints, candidate.outcome, bootstrap=False), {"parameters": parameters, "bundle": fitted}


def _plot_outputs(
    output: Path,
    outcome: str,
    comparison: pd.DataFrame,
    oof: pd.DataFrame,
    daily_metrics_frame: pd.DataFrame,
    intervals: pd.DataFrame,
    shap_global: pd.DataFrame,
    shap_values: np.ndarray,
    shap_features: np.ndarray,
    shap_names: list[str],
    temporal_predictions: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unit = "percentage points" if outcome == "recovery" else "grams"
    palette = {"pooled": "#285e47", "checkpoint": "#d99a2b", "hybrid": "#3f6fa6"}
    top = comparison.head(10).sort_values("cycle_macro_rmse")
    fig, axis = plt.subplots(figsize=(10, 6.5))
    labels = top["candidate"].str.replace("_", " ").tolist()
    values = top["cycle_macro_rmse"].to_numpy(float)
    positions = np.arange(len(top))
    axis.scatter(values, positions, s=92, color=[palette[value] for value in top["architecture"]], zorder=3)
    for value, position in zip(values, positions):
        axis.text(value, position, f"  {value:.2f}", va="center", fontsize=9)
    spread = max(float(values.max() - values.min()), float(values.max()) * .03)
    axis.set_xlim(max(0.0, float(values.min() - spread * .35)), float(values.max() + spread * .7))
    axis.set_yticks(positions, labels); axis.invert_yaxis(); axis.grid(axis="x", alpha=.2)
    axis.set_xlabel(f"Cycle-macro RMSE ({unit}; focused scale, lower is better)"); axis.set_title(f"{outcome.title()} architecture comparison")
    fig.tight_layout(); fig.savefig(output / "architecture_comparison.png", dpi=220); fig.savefig(output / "architecture_comparison.svg"); plt.close(fig)

    best_names = comparison.groupby("architecture", as_index=False).first()["candidate"].tolist()
    selected = oof.loc[oof["candidate"].isin(best_names) & oof["review_day"].isin(CHECKPOINTS)].copy()
    fig, axis = plt.subplots(figsize=(8.5, 7))
    markers = {7: "o", 14: "s", 21: "^", 28: "D"}
    for day, group in selected.groupby("review_day"):
        axis.scatter(group["actual"] * _metric_factor(outcome), group["predicted"] * _metric_factor(outcome), alpha=.68, s=32, marker=markers[int(day)], label=f"Day {day}")
    low = min(axis.get_xlim()[0], axis.get_ylim()[0]); high = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([low, high], [low, high], color="#333333", linestyle="--", linewidth=1.2, label="Perfect prediction")
    axis.set(xlabel=f"Actual ({unit})", ylabel=f"Predicted ({unit})", title=f"{outcome.title()} actual versus predicted by checkpoint")
    axis.legend(ncol=2); fig.tight_layout(); fig.savefig(output / "actual_vs_predicted_by_checkpoint.png", dpi=220); fig.savefig(output / "actual_vs_predicted_by_checkpoint.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    line_colors = ("#285e47", "#3f6fa6", "#d99a2b", "#8a4f7d")
    line_styles = ("-", "--", "-.", ":")
    for index, (candidate_name, group) in enumerate(daily_metrics_frame.groupby("candidate", sort=False)):
        style = {"color": line_colors[index % len(line_colors)], "linestyle": line_styles[index % len(line_styles)]}
        axes[0].plot(group["review_day"], group["cycle_macro_rmse"], label=candidate_name.replace("_", " "), linewidth=2, **style)
        axes[1].plot(group["review_day"], group["mae"], label=candidate_name.replace("_", " "), linewidth=2, **style)
    for axis, metric in zip(axes, ("RMSE", "MAE")):
        for day in CHECKPOINTS:
            axis.axvline(day, color="#8a8a8a", linestyle=":", linewidth=.8)
            axis.annotate(f"D{day}", (day, axis.get_ylim()[1]), xytext=(0, -2), textcoords="offset points", ha="center", va="top", fontsize=8)
        axis.set_ylabel(f"{metric} ({unit})"); axis.grid(alpha=.18)
    axes[0].set_title(f"{outcome.title()} error as evidence accumulates")
    axes[1].set_xlabel("Forecast day"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "daily_rmse_mae_learning_curve.png", dpi=220); fig.savefig(output / "daily_rmse_mae_learning_curve.svg"); plt.close(fig)

    daily_candidates = comparison.loc[comparison["daily_capable"]].head(3)["candidate"].tolist()
    residual = oof.loc[oof["candidate"].isin(daily_candidates)].copy()
    residual["error"] = (residual["predicted"] - residual["actual"]) * _metric_factor(outcome)
    fig, axis = plt.subplots(figsize=(9, 6))
    for name, group in residual.groupby("candidate"):
        axis.scatter(group["predicted"] * _metric_factor(outcome), group["error"], s=18, alpha=.35, label=name.replace("_", " "))
    axis.axhline(0, color="#333333", linestyle="--"); axis.set(xlabel=f"Predicted ({unit})", ylabel=f"Residual ({unit})", title=f"{outcome.title()} held-out residuals")
    axis.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "residuals_vs_predicted.png", dpi=220); fig.savefig(output / "residuals_vs_predicted.svg"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(residual["error"], bins=18, color="#285e47", edgecolor="white", alpha=.85)
    axis.axvline(0, color="#333333", linestyle="--"); axis.set(xlabel=f"Residual ({unit})", ylabel="Held-out predictions", title=f"{outcome.title()} residual distribution")
    fig.tight_layout(); fig.savefig(output / "residual_distribution.png", dpi=220); fig.savefig(output / "residual_distribution.svg"); plt.close(fig)

    daily_name = str(comparison.loc[comparison["daily_capable"]].iloc[0]["candidate"])
    cycle_rows = []
    champion_oof = oof.loc[oof["candidate"].eq(daily_name) & oof["review_day"].isin(CHECKPOINTS)]
    for cycle, group in champion_oof.groupby("cycle_id"):
        cycle_rows.append((str(cycle), mean_squared_error(group["actual"], group["predicted"]) ** .5 * _metric_factor(outcome)))
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar([item[0] for item in cycle_rows], [item[1] for item in cycle_rows], color="#3f6fa6")
    axis.set(xlabel="Held-out harvest cycle", ylabel=f"RMSE ({unit})", title=f"{outcome.title()} fold stability: {daily_name.replace('_', ' ')}")
    fig.tight_layout(); fig.savefig(output / "error_by_harvest_cycle.png", dpi=220); fig.savefig(output / "error_by_harvest_cycle.svg"); plt.close(fig)

    if not temporal_predictions.empty:
        temporal_rows = []
        for (name, cycle), group in temporal_predictions.loc[temporal_predictions["review_day"].isin(CHECKPOINTS)].groupby(["candidate", "cycle_id"]):
            temporal_rows.append({"candidate": name, "cycle_id": str(cycle), "rmse": mean_squared_error(group["actual"], group["predicted"]) ** .5 * _metric_factor(outcome)})
        temporal_frame = pd.DataFrame(temporal_rows)
        fig, axis = plt.subplots(figsize=(10, 5.8))
        for name, group in temporal_frame.groupby("candidate"):
            group = group.sort_values("cycle_id", key=lambda values: values.map(_cycle_key))
            axis.plot(group["cycle_id"], group["rmse"], marker="o", linewidth=1.8, label=name.replace("_", " "))
        axis.set(xlabel="Later development cycle", ylabel=f"Expanding-window RMSE ({unit})", title=f"{outcome.title()} temporal stress test")
        axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(output / "temporal_validation.png", dpi=220); fig.savefig(output / "temporal_validation.svg"); plt.close(fig)

    target = RECOVERY_TARGET if outcome == "recovery" else DAY35_TARGET_G
    actual_positive = champion_oof["actual"].to_numpy(float) >= target
    predicted_positive = champion_oof["predicted"].to_numpy(float) >= target
    matrix = np.asarray([
        [(~actual_positive & ~predicted_positive).sum(), (~actual_positive & predicted_positive).sum()],
        [(actual_positive & ~predicted_positive).sum(), (actual_positive & predicted_positive).sum()],
    ], dtype=int)
    fig, axis = plt.subplots(figsize=(6, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13, color="#172d25")
    axis.set_xticks([0, 1], ["Predicted below", "Predicted meets"]); axis.set_yticks([0, 1], ["Actual below", "Actual meets"])
    axis.set_title(f"{outcome.title()} target-side confusion matrix"); fig.colorbar(image, ax=axis, shrink=.75)
    fig.tight_layout(); fig.savefig(output / "target_confusion_matrix.png", dpi=220); fig.savefig(output / "target_confusion_matrix.svg"); plt.close(fig)

    coverage = intervals.groupby("review_day", as_index=False).agg(coverage_80=("covered_80", "mean"), width_80=("width_80", "mean"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(coverage["review_day"], coverage["coverage_80"], color="#285e47", linewidth=2); axes[0].axhline(.8, color="#333333", linestyle="--"); axes[0].set_ylabel("80% coverage")
    axes[1].plot(coverage["review_day"], coverage["width_80"], color="#3f6fa6", linewidth=2); axes[1].set_ylabel(f"Mean width ({unit})"); axes[1].set_xlabel("Forecast day")
    axes[0].set_title(f"{outcome.title()} interval calibration and width"); fig.tight_layout(); fig.savefig(output / "interval_coverage_width.png", dpi=220); fig.savefig(output / "interval_coverage_width.svg"); plt.close(fig)

    if not shap_global.empty:
        top_shap = shap_global.head(10).sort_values("mean_abs_shap")
        fig, axis = plt.subplots(figsize=(9, 6))
        axis.barh(top_shap["feature"].str.replace("_", " "), top_shap["mean_abs_shap"], color="#285e47")
        axis.set(xlabel="Mean absolute SHAP value", title=f"{outcome.title()} top held-out SHAP signals")
        fig.tight_layout(); fig.savefig(output / "shap_top10.png", dpi=220); fig.savefig(output / "shap_top10.svg"); plt.close(fig)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, shap_features, feature_names=shap_names, max_display=12, show=False)
        plt.title(f"{outcome.title()} SHAP direction (predictive association)")
        plt.tight_layout(); plt.savefig(output / "shap_beeswarm.png", dpi=220, bbox_inches="tight"); plt.savefig(output / "shap_beeswarm.svg", bbox_inches="tight"); plt.close()


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(output: Path, outcome: str, manifest: dict[str, Any], comparison: pd.DataFrame, checkpoint: pd.DataFrame, shap_global: pd.DataFrame) -> Path:
    title = "Harvest Recovery" if outcome == "recovery" else "Day 35 Bodyweight"
    unit = "percentage points" if outcome == "recovery" else "grams"
    top = comparison.head(5)[["candidate", "architecture", "target_form", "cycle_macro_rmse", "mae", "r2", "bias", "worst_cycle_rmse"]]
    selected_name = manifest["selection"]["daily_capable_champion"]
    shadow_name = manifest["selection"]["daily_capable_lowest_error"]
    selected_row = comparison.loc[comparison["candidate"].eq(selected_name)].iloc[0]
    shadow_row = comparison.loc[comparison["candidate"].eq(shadow_name)].iloc[0]
    daily_checkpoint = checkpoint.loc[checkpoint["candidate"].eq(shadow_name), ["review_day", "cycle_macro_rmse", "mae", "r2"]]
    drivers = ", ".join(shap_global.head(10)["feature"].str.replace("_", " ").tolist()) if not shap_global.empty else "No compatible tree model produced stable SHAP output."
    gate = manifest["promotion_gate"]
    shadow_status = "shadow-eligible after retrospective testing" if gate["retrospective_gate_passed"] else "research-only shadow challenger"
    improvement = float(gate["cycle_macro_rmse_improvement_pct"])
    text = f"""# Project Canary Robust Architecture Test: {title}

## Executive conclusion

The one-standard-error rule selected **{selected_name.replace('_', ' ')}** at **{selected_row['cycle_macro_rmse']:.2f} {unit}** cycle-macro RMSE. The lowest-error daily model was **{shadow_name.replace('_', ' ')}** at **{shadow_row['cycle_macro_rmse']:.2f} {unit}**, a **{improvement:.1f}%** improvement over the transparent baseline. It is a **{shadow_status}**. Operational inference remains unchanged.

## Top five matched nested cycle-LOGO results

{_markdown_table(top)}

All architectures are ranked on the same held-out Days 7, 14, 21, and 28 rows. Checkpoint-specific models do not receive fabricated predictions for intervening days.

![Architecture comparison](figures/architecture_comparison.png)

## Does accuracy improve with more days of evidence?

The learning curve below follows the lowest-error daily challenger, **{shadow_name.replace('_', ' ')}**.

{_markdown_table(daily_checkpoint)}

![Daily learning curve](figures/daily_rmse_mae_learning_curve.png)

## Actual versus predicted

![Actual versus predicted](figures/actual_vs_predicted_by_checkpoint.png)

## Predictive explanations

Leading held-out SHAP signals for the best compatible daily tree challenger were: {drivers}

![SHAP importance](figures/shap_top10.png)

![SHAP direction](figures/shap_beeswarm.png)

SHAP and permutation importance describe predictive association, not causation. They cannot justify an intervention without biological and management review.

## Trust and use

The primary validation holds out one complete harvest cycle and tunes only on the remaining cycles. Daily landmarks are weighted so every building-cycle contributes equal total influence. Exact building identity and feed are excluded. Six independent development cycles remain a fundamental limitation, so this is capstone/shadow evidence rather than production approval.
"""
    path = output / f"PROJECT_CANARY_{outcome.upper()}_ROBUST_ARCHITECTURE_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def _profile_candidates(outcome: str, architecture: str, profile: str) -> tuple[ArchitectureCandidate, ...]:
    source = RECOVERY_CANDIDATES if outcome == "recovery" else WEIGHT_CANDIDATES
    selected = [candidate for candidate in source if architecture == "all" or candidate.architecture == architecture or candidate.family in {"persistence", "age_baseline", "target_ratio"}]
    if profile == "screening":
        selected = [candidate for candidate in selected if candidate.family in SCREENING_FAMILIES and (candidate.target_form == "remaining" or candidate.family in {"persistence", "age_baseline", "target_ratio", *BIOLOGICAL_FAMILIES})]
    elif profile != "full":
        raise ValueError("profile must be 'screening' or 'full'")
    return tuple(selected)


def _data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": frame.columns,
        "dtype": [str(frame[column].dtype) for column in frame.columns],
        "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
        "description": [
            "As-of feature or metadata; observed weights remain distinct from target references."
            if column not in {"actual", "remaining_target", "role"} else
            {"actual": "Final held-out outcome.", "remaining_target": "Final outcome minus/current remaining process target.", "role": "Development or locked audit role."}[column]
            for column in frame.columns
        ],
    })


def _run_outcome(
    dataset: CanaryDataset,
    output_root: Path,
    outcome: str,
    development_cycles: tuple[str, ...],
    audit_cycle: str,
    profile: str,
    architecture: str,
    seed: int,
    reuse_primary: bool = False,
) -> dict[str, Any]:
    output = output_root / ("recovery" if outcome == "recovery" else "bodyweight")
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    snapshots = build_daily_landmarks(dataset, outcome, development_cycles, audit_cycle)
    development = snapshots.loc[snapshots["role"].eq("development")].reset_index(drop=True)
    audit = snapshots.loc[snapshots["role"].eq("later_cycle_audit")].reset_index(drop=True)
    candidates = _profile_candidates(outcome, architecture, profile)
    development.to_csv(output / "daily_model_ready_landmarks.csv", index=False)
    audit.to_csv(output / "locked_later_cycle_landmarks.csv", index=False)
    _data_dictionary(snapshots).to_csv(output / "data_dictionary.csv", index=False)
    pd.DataFrame([{**asdict(candidate), "grid": json.dumps(candidate_grid(candidate)), "grid_size": len(candidate_grid(candidate))} for candidate in candidates]).to_csv(output / "experiment_registry.csv", index=False)

    primary_path = output / "all_nested_logo_predictions.csv"
    comparison_path = output / "candidate_comparison.csv"
    settings_path = output / "nested_hyperparameters.json"
    if reuse_primary and primary_path.exists() and comparison_path.exists() and settings_path.exists():
        print(f"[{outcome}] reusing completed primary architecture search", flush=True)
        oof = pd.read_csv(primary_path)
        comparison = pd.read_csv(comparison_path)
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        existing = set(comparison["candidate"].astype(str))
        missing = [candidate for candidate in candidates if candidate.name not in existing]
        if missing:
            additions, addition_rows = [], []
            for position, candidate in enumerate(missing, start=1):
                print(f"[{outcome}] incremental architecture LOGO {position}/{len(missing)}: {candidate.name}", flush=True)
                predictions, parameters = evaluate_nested_logo(development, candidate, seed=seed)
                additions.append(predictions)
                checkpoint_predictions = predictions.loc[predictions["review_day"].isin(CHECKPOINTS)]
                addition_rows.append({
                    "candidate": candidate.name, "architecture": candidate.architecture, "family": candidate.family,
                    "target_form": candidate.target_form, "feature_block": candidate.feature_block,
                    "complexity": candidate.complexity, "daily_capable": candidate.daily_capable,
                    "description": candidate.description, **summarize(checkpoint_predictions, outcome, seed=seed),
                })
                settings[candidate.name] = parameters
            oof = pd.concat([oof, *additions], ignore_index=True)
            comparison = pd.concat([comparison.drop(columns=["rank"], errors="ignore"), pd.DataFrame(addition_rows)], ignore_index=True, sort=False)
            comparison = comparison.sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
            comparison["rank"] = np.arange(1, len(comparison) + 1)
            oof.to_csv(primary_path, index=False)
            comparison.to_csv(comparison_path, index=False)
    else:
        all_predictions, comparison_rows, settings = [], [], {}
        for position, candidate in enumerate(candidates, start=1):
            print(f"[{outcome}] architecture LOGO {position}/{len(candidates)}: {candidate.name}", flush=True)
            predictions, parameters = evaluate_nested_logo(development, candidate, seed=seed)
            all_predictions.append(predictions)
            checkpoint_predictions = predictions.loc[predictions["review_day"].isin(CHECKPOINTS)]
            comparison_rows.append({
                "candidate": candidate.name, "architecture": candidate.architecture, "family": candidate.family,
                "target_form": candidate.target_form, "feature_block": candidate.feature_block,
                "complexity": candidate.complexity, "daily_capable": candidate.daily_capable,
                "description": candidate.description, **summarize(checkpoint_predictions, outcome, seed=seed),
            })
            settings[candidate.name] = parameters
        oof = pd.concat(all_predictions, ignore_index=True)
        comparison = pd.DataFrame(comparison_rows).sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
        comparison["rank"] = np.arange(1, len(comparison) + 1)
        oof.to_csv(primary_path, index=False)
        comparison.to_csv(comparison_path, index=False)
    comparison.head(5).to_csv(output / "top_five_models.csv", index=False)
    settings_path.write_text(json.dumps(settings, indent=2, default=_json_default), encoding="utf-8")

    selected_name, selection = _one_se_select(comparison)
    daily_comparison = comparison.loc[comparison["daily_capable"]].reset_index(drop=True)
    daily_selected_name, daily_selection = _one_se_select(daily_comparison)
    selection["daily_capable_champion"] = daily_selected_name
    selection["daily_capable_lowest_error"] = daily_selection["lowest_error_candidate"]
    selection["shadow_challenger"] = daily_selection["lowest_error_candidate"]
    baseline_name = "daily_age_remaining_loss" if outcome == "recovery" else "historical_remaining_gain"
    lowest_name = selection["lowest_error_candidate"]
    daily_champion = next(candidate for candidate in candidates if candidate.name == daily_selected_name)
    daily_shadow = next(candidate for candidate in candidates if candidate.name == daily_selection["lowest_error_candidate"])
    lowest = next(candidate for candidate in candidates if candidate.name == lowest_name)

    checkpoint_tables, daily_tables = [], []
    for name, group in oof.groupby("candidate"):
        candidate = next(item for item in candidates if item.name == name)
        metrics = metrics_by_day(group, outcome)
        metrics.insert(0, "candidate", name); metrics.insert(1, "architecture", candidate.architecture)
        if candidate.daily_capable:
            daily_tables.append(metrics)
        checkpoint_tables.append(metrics.loc[metrics["review_day"].isin(CHECKPOINTS)])
    daily_metrics_frame = pd.concat(daily_tables, ignore_index=True)
    checkpoint_metrics_frame = pd.concat(checkpoint_tables, ignore_index=True)
    daily_metrics_frame.to_csv(output / "daily_metrics.csv", index=False)
    checkpoint_metrics_frame.to_csv(output / "checkpoint_metrics.csv", index=False)

    finalist_names = list(dict.fromkeys([baseline_name, selected_name, daily_selected_name, daily_shadow.name, lowest_name]))
    secondary_rows, temporal_rows, temporal_predictions = [], [], []
    for name in finalist_names:
        candidate = next(item for item in candidates if item.name == name)
        prediction, _ = evaluate_nested_logo(development, candidate, view="building_label", seed=seed)
        secondary_rows.append({"candidate": name, "validation_view": "building_label", **summarize(prediction.loc[prediction["review_day"].isin(CHECKPOINTS)], outcome, bootstrap=False)})
        temporal_prediction, temporal_metric = temporal_stress(development, candidate, seed)
        temporal_predictions.append(temporal_prediction)
        temporal_rows.append({"candidate": name, **temporal_metric})
    pd.DataFrame(secondary_rows).to_csv(output / "building_label_logo_metrics.csv", index=False)
    pd.concat(temporal_predictions, ignore_index=True).to_csv(output / "temporal_predictions.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(output / "temporal_metrics.csv", index=False)

    selected_intervals, selected_calibration = conformal_logo(development, daily_champion, seed)
    if daily_shadow.name == daily_champion.name:
        shadow_intervals, shadow_calibration = selected_intervals.copy(), selected_calibration.copy()
    else:
        shadow_intervals, shadow_calibration = conformal_logo(development, daily_shadow, seed)
    selected_intervals.to_csv(output / "selected_grouped_conformal_predictions.csv", index=False)
    selected_calibration.to_csv(output / "selected_interval_calibration_by_day.csv", index=False)
    shadow_intervals.to_csv(output / "shadow_grouped_conformal_predictions.csv", index=False)
    shadow_calibration.to_csv(output / "shadow_interval_calibration_by_day.csv", index=False)
    # Backward-compatible aliases point to the research challenger used by the
    # promotion gate, not silently to the simpler one-SE selection.
    shadow_intervals.to_csv(output / "grouped_conformal_predictions.csv", index=False)
    shadow_calibration.to_csv(output / "interval_calibration_by_day.csv", index=False)

    learned_daily = comparison.loc[comparison["daily_capable"] & ~comparison["family"].isin(["persistence", "age_baseline", "target_ratio", *BIOLOGICAL_FAMILIES])]
    importance_candidate = next(item for item in candidates if item.name == str(learned_daily.iloc[0]["candidate"])) if not learned_daily.empty else daily_champion
    importance = _feature_importance(development, importance_candidate, seed)
    importance.to_csv(output / "held_out_permutation_importance.csv", index=False)
    tree_daily = learned_daily.loc[learned_daily["family"].isin(["random_forest", "extra_trees", "hist_gradient", "xgboost", "lightgbm", "catboost"])]
    explanation_candidate = next(item for item in candidates if item.name == str(tree_daily.iloc[0]["candidate"])) if not tree_daily.empty else None
    if explanation_candidate:
        shap_global, shap_local, shap_values, shap_features, shap_names = _tree_shap(development, explanation_candidate, seed)
    else:
        shap_global, shap_local, shap_values, shap_features, shap_names = pd.DataFrame(), pd.DataFrame(), np.empty((0, 0)), np.empty((0, 0)), []
    shap_global.to_csv(output / "held_out_shap_global.csv", index=False)
    shap_local.to_csv(output / "held_out_shap_local.csv", index=False)

    ablation_predictions, ablation_rows = [], []
    if explanation_candidate:
        for block in ("full", "core", "no_environment", "no_peer"):
            ablation_candidate = replace(explanation_candidate, name=f"{explanation_candidate.name}__{block}", feature_block=block)
            if block == "full":
                prediction = oof.loc[oof["candidate"].eq(explanation_candidate.name)].copy()
                prediction["candidate"] = ablation_candidate.name
            else:
                print(f"[{outcome}] matched feature ablation: {block}", flush=True)
                prediction, _ = evaluate_nested_logo(development, ablation_candidate, seed=seed)
            ablation_predictions.append(prediction)
            checkpoint_prediction = prediction.loc[prediction["review_day"].isin(CHECKPOINTS)]
            ablation_rows.append({"feature_block": block, **summarize(checkpoint_prediction, outcome, bootstrap=False)})
    pd.concat(ablation_predictions, ignore_index=True).to_csv(output / "feature_ablation_predictions.csv", index=False) if ablation_predictions else pd.DataFrame().to_csv(output / "feature_ablation_predictions.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(output / "feature_ablation_comparison.csv", index=False)

    audit_predictions, audit_metrics, final_fit = _score_audit(development, audit, daily_champion, seed)
    shadow_audit_predictions, shadow_audit_metrics, shadow_fit = _score_audit(development, audit, daily_shadow, seed)
    if baseline_name == daily_champion.name:
        baseline_audit_metrics = audit_metrics
    else:
        baseline_candidate = next(candidate for candidate in candidates if candidate.name == baseline_name)
        _, baseline_audit_metrics, _ = _score_audit(development, audit, baseline_candidate, seed)
    audit_predictions.to_csv(output / "selected_later_cycle_audit_predictions.csv", index=False)
    shadow_audit_predictions.to_csv(output / "shadow_later_cycle_audit_predictions.csv", index=False)
    shadow_audit_predictions.to_csv(output / "later_cycle_audit_predictions.csv", index=False)
    artifact_path = output / "daily_capable_research_champion.joblib"
    artifact = {
        "round_version": ROUND_VERSION, "deployment_status": "research_shadow",
        "candidate": asdict(daily_champion), "fitted": final_fit,
        "source_schema": feature_columns(daily_champion) if daily_champion.family not in {"persistence", "age_baseline", "target_ratio", *BIOLOGICAL_FAMILIES} else [],
    }
    joblib.dump(artifact, artifact_path)
    restored = joblib.load(artifact_path)
    parity_frame = development.head(50)
    parity = float(np.max(np.abs(
        predict_candidate(final_fit["bundle"], parity_frame, daily_champion)
        - predict_candidate(restored["fitted"]["bundle"], parity_frame, daily_champion)
    )))
    if parity > 1e-10:
        raise AssertionError("Reloaded architecture artifact changed predictions.")

    shadow_artifact_path = output / "lowest_error_shadow_challenger.joblib"
    shadow_artifact = {
        "round_version": ROUND_VERSION, "deployment_status": "research_shadow",
        "candidate": asdict(daily_shadow), "fitted": shadow_fit,
        "source_schema": feature_columns(daily_shadow) if daily_shadow.family not in {"persistence", "age_baseline", "target_ratio", *BIOLOGICAL_FAMILIES} else [],
    }
    joblib.dump(shadow_artifact, shadow_artifact_path)
    restored_shadow = joblib.load(shadow_artifact_path)
    shadow_parity = float(np.max(np.abs(
        predict_candidate(shadow_fit["bundle"], parity_frame, daily_shadow)
        - predict_candidate(restored_shadow["fitted"]["bundle"], parity_frame, daily_shadow)
    )))
    if shadow_parity > 1e-10:
        raise AssertionError("Reloaded shadow artifact changed predictions.")

    gate = _promotion_gate(comparison, oof.loc[oof["review_day"].isin(CHECKPOINTS)], daily_shadow.name, baseline_name, outcome, shadow_intervals, shadow_audit_metrics, baseline_audit_metrics)
    selection["checkpoint_only_best"] = lowest_name if not lowest.daily_capable else None
    manifest = {
        "round_version": ROUND_VERSION, "outcome": outcome, "seed": seed,
        "profile": profile, "architectures": list(ARCHITECTURES),
        "development_cycles": list(development_cycles), "locked_audit_cycle": audit_cycle,
        "development_rows": int(len(development)),
        "development_building_cycles": int(development[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "daily_landmarks": list(LANDMARK_DAYS), "validated_checkpoints": list(CHECKPOINTS),
        "selection": selection, "baseline": baseline_name,
        "explanation_model": explanation_candidate.name if explanation_candidate else None,
        "promotion_gate": gate, "later_cycle_audit_metrics": shadow_audit_metrics,
        "selected_later_cycle_audit_metrics": audit_metrics,
        "baseline_later_cycle_audit_metrics": baseline_audit_metrics,
        "artifact": {"path": str(artifact_path), "sha256": _sha(artifact_path), "prediction_parity_max_abs": parity},
        "shadow_artifact": {"path": str(shadow_artifact_path), "sha256": _sha(shadow_artifact_path), "prediction_parity_max_abs": shadow_parity},
        "operational_models_changed": False,
        "shap_warning": "SHAP and importance describe predictive association, not causation.",
    }
    _plot_outputs(figures, outcome, comparison, oof, daily_metrics_frame.loc[daily_metrics_frame["candidate"].isin(list(dict.fromkeys([baseline_name, daily_selected_name, daily_selection["lowest_error_candidate"]])))], shadow_intervals, shap_global, shap_values, shap_features, shap_names, pd.concat(temporal_predictions, ignore_index=True))
    report = _write_report(output, outcome, manifest, comparison, checkpoint_metrics_frame, shap_global)
    manifest["technical_report"] = str(report)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest


def _package_versions() -> dict[str, str]:
    values = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not-installed"
    return values


def _write_combined_report(output_root: Path, outcomes: dict[str, Any]) -> Path:
    sections = []
    for outcome, folder in (("recovery", "recovery"), ("bodyweight", "bodyweight")):
        if outcome not in outcomes:
            continue
        comparison = pd.read_csv(output_root / folder / "candidate_comparison.csv")
        architecture_best = comparison.sort_values("cycle_macro_rmse").groupby("architecture", as_index=False).first()
        fields = architecture_best[["architecture", "candidate", "cycle_macro_rmse", "mae", "r2", "bias"]]
        manifest = outcomes[outcome]
        unit = "percentage points" if outcome == "recovery" else "grams"
        selected_name = manifest["selection"]["daily_capable_champion"]
        shadow_name = manifest["selection"]["daily_capable_lowest_error"]
        selected_row = comparison.loc[comparison["candidate"].eq(selected_name)].iloc[0]
        shadow_row = comparison.loc[comparison["candidate"].eq(shadow_name)].iloc[0]
        gate = manifest["promotion_gate"]
        gate_text = "passed retrospectively but still requires three prospective cycles" if gate["retrospective_gate_passed"] else "did not pass every retrospective gate"
        sections.append(f"""## {outcome.title()}

{_markdown_table(fields)}

**Capstone selection:** **{selected_name.replace('_', ' ')}**, chosen by the one-standard-error rule at **{selected_row['cycle_macro_rmse']:.2f} {unit}** cycle-macro RMSE.

**Lowest-error daily challenger:** **{shadow_name.replace('_', ' ')}** at **{shadow_row['cycle_macro_rmse']:.2f} {unit}**, **{gate['cycle_macro_rmse_improvement_pct']:.1f}%** better than the transparent baseline. Its promotion gate {gate_text}. Operational inference was not changed.
""")
    passed = [outcome for outcome, manifest in outcomes.items() if manifest["promotion_gate"]["retrospective_gate_passed"]]
    gate_summary = (
        f"The retrospective gate passed for {', '.join(passed)}, but this authorizes shadow evaluation only—not deployment."
        if passed else
        "Neither outcome passed every prespecified retrospective promotion gate."
    )
    text = f"""# Project Canary Robust Model-Architecture Test

## Executive summary

This research round compared one pooled farm-wide model, four separate checkpoint models, and a partially pooled hybrid under identical nested leave-one-harvest-cycle-out validation. All architectures were ranked on the same held-out Days 7, 14, 21, and 28 records. Pooled and hybrid designs were additionally tested on every day from Day 7 through Day 34.

Operational models remain unchanged. {gate_summary} Canary can produce intervening-day forecasts with pooled or hybrid models while clearly labelling the four principal validation checkpoints. Separate checkpoint models are used only at their intended days.

{''.join(sections)}

## Capstone interpretation

The architecture question was tested rather than assumed. With only six independent development cycles, checkpoint-specific models reduce effective training support. The one-standard-error rule protects against selecting a complex model for an uncertain apparent gain; the promotion gate separately determines whether the lowest-error challenger is ready even for prospective shadow evaluation. Learned candidates remain research evidence and operational models were not changed.
"""
    path = output_root / "PROJECT_CANARY_ROBUST_ARCHITECTURE_EXECUTIVE_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_robust_architecture_round(
    workbook: str | Path,
    output: str | Path,
    *,
    outcome: str = "both",
    architecture: str = "all",
    profile: str = "screening",
    audit_cycle: str = "2026-3",
    seed: int = SEED,
    reuse: bool = True,
    reuse_primary: bool = False,
) -> dict[str, Any]:
    """Execute the frozen architecture comparison without touching operations."""
    if outcome not in {"recovery", "bodyweight", "both"}:
        raise ValueError("outcome must be recovery, bodyweight, or both")
    if architecture not in {*ARCHITECTURES, "all"}:
        raise ValueError("architecture must be pooled, checkpoint, hybrid, or all")
    workbook_path = Path(workbook).resolve()
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if reuse and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        # ``latest`` is a user-facing alias. A frozen manifest records the
        # resolved cycle name, so compare against that name when deciding
        # whether an unchanged workbook can safely reuse its completed run.
        requested_audit_cycle = previous.get("locked_audit_cycle") if audit_cycle == "latest" else audit_cycle
        if (
            previous.get("source_sha256") == _sha(workbook_path)
            and previous.get("profile") == profile
            and previous.get("architecture_filter") == architecture
            and previous.get("outcome_filter") == outcome
            and previous.get("seed") == seed
            and previous.get("locked_audit_cycle") == requested_audit_cycle
        ):
            print(f"Reusing matching frozen architecture run at {output_root}", flush=True)
            return previous

    dataset = load_workbook(workbook_path)
    cycles = tuple(sorted(dataset.cycles["cycle_id"].astype(str).unique(), key=_cycle_key))
    if audit_cycle == "latest":
        audit_cycle = cycles[-1]
    if audit_cycle not in cycles:
        raise ValueError(f"Audit cycle {audit_cycle!r} is absent from the workbook")
    development_cycles = tuple(cycle for cycle in cycles if cycle != audit_cycle)
    source_profile, source_checks = build_source_quality_audit(workbook_path, dataset)
    previews = {name: build_daily_landmarks(dataset, name, development_cycles, audit_cycle) for name in ("recovery", "weight")}
    expected_units = int(dataset.cycles.loc[dataset.cycles["cycle_id"].astype(str).isin(development_cycles), ["cycle_id", "building_id"]].drop_duplicates().shape[0])
    extra_checks = pd.DataFrame([
        {"check": "recovery_daily_landmarks", "severity": "critical", "failed_rows": int(len(previews["recovery"].query("role == 'development'")) != expected_units * len(LANDMARK_DAYS)), "detail": "Every development building-cycle must have Day 7-34 landmarks."},
        {"check": "bodyweight_daily_landmarks", "severity": "critical", "failed_rows": int(len(previews["weight"].query("role == 'development'")) != expected_units * len(LANDMARK_DAYS)), "detail": "Every eligible development building-cycle must have Day 7-34 landmarks."},
        {"check": "future_evidence", "severity": "critical", "failed_rows": int(sum((frame["max_source_day_used"] > frame["review_day"]).sum() for frame in previews.values())), "detail": "No landmark may use evidence after its review day."},
        {"check": "audit_exclusion", "severity": "critical", "failed_rows": int(sum((frame.loc[frame["role"].eq("development"), "cycle_id"].astype(str) == audit_cycle).sum() for frame in previews.values())), "detail": "The locked audit cycle must not enter development."},
        {"check": "equal_building_cycle_weight", "severity": "critical", "failed_rows": int(any(frame.query("role == 'development'").groupby(["cycle_id", "building_id"])["sample_weight"].sum().round(10).nunique() != 1 for frame in previews.values())), "detail": "Each building-cycle must contribute equal total landmark weight."},
    ])
    checks = pd.concat([source_checks, extra_checks], ignore_index=True)
    checks.to_csv(output_root / "data_quality_checks.csv", index=False)
    (output_root / "source_audit.json").write_text(json.dumps(source_profile, indent=2, default=_json_default), encoding="utf-8")
    if checks.loc[checks["severity"].eq("critical"), "failed_rows"].sum() > 0:
        raise AssertionError("Critical architecture-round data checks failed.")

    outcomes: dict[str, Any] = {}
    if outcome in {"recovery", "both"}:
        outcomes["recovery"] = _run_outcome(dataset, output_root, "recovery", development_cycles, audit_cycle, profile, architecture, seed, reuse_primary)
    if outcome in {"bodyweight", "both"}:
        outcomes["bodyweight"] = _run_outcome(dataset, output_root, "weight", development_cycles, audit_cycle, profile, architecture, seed, reuse_primary)
    combined_frames = []
    for key, folder in (("recovery", "recovery"), ("bodyweight", "bodyweight")):
        path = output_root / folder / "top_five_models.csv"
        if path.exists():
            combined_frames.append(pd.read_csv(path).assign(outcome=key))
    if combined_frames:
        pd.concat(combined_frames, ignore_index=True, sort=False).to_csv(output_root / "top_five_models_by_outcome.csv", index=False)
    executive_report = _write_combined_report(output_root, outcomes)
    manifest = {
        "round_version": ROUND_VERSION, "created": pd.Timestamp.now(tz="Asia/Manila").isoformat(),
        "source_workbook": str(workbook_path), "source_sha256": _sha(workbook_path), "source": source_profile,
        "profile": profile, "outcome_filter": outcome, "architecture_filter": architecture, "seed": seed,
        "development_cycles": list(development_cycles), "locked_audit_cycle": audit_cycle,
        "package_versions": _package_versions(), "outcomes": outcomes,
        "primary_comparison_scope": "Identical held-out Days 7, 14, 21, and 28 under nested harvest-cycle LOGO.",
        "daily_evaluation_scope": "Pooled and hybrid candidates only, Days 7 through 34.",
        "operational_models_changed": False, "research_only": True,
        "executive_report": str(executive_report),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest
