"""Final isolated optimization round for Project Canary's farm-wide models.

The module rebuilds leakage-safe snapshots from the corrected workbook, runs
nested leave-one-harvest-cycle-out model selection, and writes research-only
artifacts.  It never writes to ``models/`` or changes application inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from .bodyweight_modeling_review import (
    AUDIT_CYCLE,
    CHECKPOINTS,
    DAY35_TARGET_G,
    DEVELOPMENT_CYCLES,
    SEED,
    build_snapshots as build_weight_snapshots,
    feature_columns as weight_feature_columns,
)
from .data import CanaryDataset, load_workbook
from .external_modeling_review import (
    RECOVERY_COMPACT_FEATURES,
    RECOVERY_FEATURES,
    build_snapshots as build_external_snapshots,
)
from .farmwide_features import assert_primary_schema_has_no_identity
from .farmwide_modeling import build_source_quality_audit


ROUND_VERSION = "farmwide-finalization-2.0.0"
FIXED_SEEDS = (20260812, 20260813, 20260814, 20260815, 20260816)
RECOVERY_TARGET = 0.95

BALANCED_RECOVERY_NAMES = {
    "current_survival", "age_band_remaining_loss", "remaining_ridge_peer",
    "remaining_huber_peer", "remaining_random_forest_peer",
    "remaining_extra_trees_peer", "residual_hist_gradient_peer",
    "residual_xgboost_peer", "residual_lightgbm_peer", "residual_catboost_peer",
}

BALANCED_WEIGHT_NAMES = {
    "historical_remaining_gain", "target_gap_preserving", "target_curve_ratio",
    "direct_trajectory_pls", "residual_biological_ridge",
    "direct_huber_biological", "residual_random_forest_peer",
    "residual_extra_trees_peer", "residual_hist_gradient_peer",
    "residual_xgboost_peer", "residual_lightgbm_peer", "residual_catboost_peer",
}


@dataclass(frozen=True)
class OptimizationCandidate:
    name: str
    outcome: str
    family: str
    target_form: str
    feature_set: str
    checkpoint_specific: bool
    complexity: int
    description: str


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Fold-local column clipping that preserves sklearn feature names."""

    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        self.lower = lower
        self.upper = upper

    def fit(self, values: Any, y: Any = None) -> "QuantileWinsorizer":
        array = np.asarray(values, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, values: Any) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(list(input_features) if input_features is not None else [], dtype=object)


class SafeVarianceFilter(BaseEstimator, TransformerMixin):
    """Drop constant columns unless that would leave a fold with no features."""

    def fit(self, values: Any, y: Any = None) -> "SafeVarianceFilter":
        array = np.asarray(values, dtype=float)
        variance = np.nanvar(array, axis=0)
        self.keep_ = variance > 0
        if not self.keep_.any():
            self.keep_ = np.ones(array.shape[1], dtype=bool)
        return self

    def transform(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = np.asarray(list(input_features) if input_features is not None else [], dtype=object)
        return names[self.keep_]


class SafePercentileFilter(BaseEstimator, TransformerMixin):
    """Fold-local univariate filtering that always retains at least one feature."""

    def __init__(self, percentile: int = 100):
        self.percentile = percentile

    def fit(self, values: Any, y: Any = None) -> "SafePercentileFilter":
        array = np.asarray(values, dtype=float)
        count = array.shape[1]
        keep_count = max(1, int(np.ceil(count * float(self.percentile) / 100.0)))
        if keep_count >= count:
            self.keep_ = np.ones(count, dtype=bool)
        else:
            target = np.asarray(y, dtype=float)
            centered_x = array - array.mean(axis=0)
            centered_y = target - target.mean()
            denominator = np.sqrt(np.sum(centered_x ** 2, axis=0) * np.sum(centered_y ** 2))
            scores = np.divide(np.abs(centered_x.T @ centered_y), denominator, out=np.zeros(count), where=denominator > 0)
            indices = np.argsort(scores, kind="stable")[-keep_count:]
            self.keep_ = np.zeros(count, dtype=bool)
            self.keep_[indices] = True
        return self

    def transform(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = np.asarray(list(input_features) if input_features is not None else [], dtype=object)
        return names[self.keep_]


RECOVERY_ENGINEERED = [
    "peer_survival_mean", "survival_vs_peer", "peer_mortality_7d_mean",
    "mortality_7d_vs_peer", "peer_latest_weight_mean", "weight_vs_peer",
    "peer_building_count", "mortality_horizon_pressure", "survival_horizon_interaction",
    "environment_missing_both", "environment_coverage_x_horizon",
    "weight_gap_x_staleness", "recent_mortality_share", "reconciliation_flag",
]

WEIGHT_ENGINEERED = [
    "peer_current_weight_mean_g", "current_weight_vs_peer_g", "peer_survival_mean_pct",
    "survival_vs_peer_pp", "peer_building_count", "log_current_weight_g",
    "log_measurement_day", "target_ratio_log", "growth_fraction_2500g",
    "gompertz_linear_component", "adg_x_days_remaining", "mortality_x_days_remaining",
    "environment_missing_both", "measurement_irregularity_flag", "implausible_adg_flag",
]


def _c(name: str, outcome: str, family: str, target: str, features: str, checkpoint: bool, complexity: int, description: str) -> OptimizationCandidate:
    return OptimizationCandidate(name, outcome, family, target, features, checkpoint, complexity, description)


RECOVERY_CANDIDATES = (
    _c("current_survival", "recovery", "persistence", "direct", "core", False, 0, "Current survival persists to endpoint"),
    _c("age_band_remaining_loss", "recovery", "baseline", "remaining", "core", False, 1, "Fold-local historical remaining loss by checkpoint"),
    _c("remaining_linear", "recovery", "linear", "remaining", "core", False, 2, "Linear remaining-loss model"),
    _c("remaining_bayesian_ridge", "recovery", "bayesian_ridge", "remaining", "core", False, 3, "Shrinkage remaining-loss regression"),
    _c("residual_bayesian_peer", "recovery", "bayesian_ridge", "baseline_residual", "peer", False, 4, "Bayesian residual correction to the age baseline"),
    _c("remaining_ridge_peer", "recovery", "ridge", "remaining", "peer", False, 4, "Regularized remaining-loss model with peer context"),
    _c("remaining_elastic_peer", "recovery", "elastic_net", "remaining", "peer", False, 5, "Sparse remaining-loss model with peer context"),
    _c("remaining_huber_peer", "recovery", "huber", "remaining", "peer", False, 5, "Robust remaining-loss model with peer context"),
    _c("direct_spline_core", "recovery", "spline", "direct", "core", False, 6, "Regularized nonlinear direct model"),
    _c("remaining_random_forest_peer", "recovery", "random_forest", "remaining", "peer", False, 7, "Conservative random forest on remaining loss"),
    _c("remaining_extra_trees_peer", "recovery", "extra_trees", "remaining", "peer", False, 7, "Conservative Extra Trees on remaining loss"),
    _c("residual_gradient_boosting_peer", "recovery", "gradient_boosting", "baseline_residual", "peer", False, 7, "Gradient boosting correction to the age baseline"),
    _c("residual_hist_gradient_peer", "recovery", "hist_gradient_boosting", "baseline_residual", "peer", False, 7, "Histogram boosting correction to the age baseline"),
    _c("residual_hist_gradient_core", "recovery", "hist_gradient_boosting", "baseline_residual", "core", False, 7, "Histogram boosting correction without peer context"),
    _c("residual_hist_gradient_no_environment", "recovery", "hist_gradient_boosting", "baseline_residual", "peer_no_environment", False, 7, "Peer-context histogram boosting without environment"),
    _c("residual_xgboost_peer", "recovery", "xgboost", "baseline_residual", "peer", False, 8, "XGBoost correction to the age baseline"),
    _c("residual_lightgbm_peer", "recovery", "lightgbm", "baseline_residual", "peer", False, 8, "LightGBM correction to the age baseline"),
    _c("residual_catboost_peer", "recovery", "catboost", "baseline_residual", "peer", False, 8, "CatBoost correction to the age baseline"),
    _c("direct_gradient_boosting_full", "recovery", "gradient_boosting", "direct", "full", False, 7, "Direct final-recovery gradient boosting using the full as-of feature set"),
    _c("remaining_gradient_boosting_full", "recovery", "gradient_boosting", "remaining", "full", False, 7, "Remaining-loss gradient boosting using the full as-of feature set"),
    _c("direct_hist_gradient_full", "recovery", "hist_gradient_boosting", "direct", "full", False, 7, "Direct final-recovery histogram boosting"),
    _c("remaining_hist_gradient_full", "recovery", "hist_gradient_boosting", "remaining", "full", False, 7, "Remaining-loss histogram boosting"),
    _c("remaining_xgboost_full", "recovery", "xgboost", "remaining", "full", False, 8, "Regularized XGBoost on remaining loss"),
    _c("remaining_lightgbm_full", "recovery", "lightgbm", "remaining", "full", False, 8, "Regularized LightGBM on remaining loss"),
    _c("remaining_catboost_full", "recovery", "catboost", "remaining", "full", False, 8, "Regularized CatBoost on remaining loss"),
)


WEIGHT_CANDIDATES = (
    _c("historical_remaining_gain", "weight", "baseline", "remaining", "current", True, 0, "Fold-local historical remaining gain"),
    _c("target_gap_preserving", "weight", "target_curve", "direct", "current", True, 0, "Target-curve remaining gain preserving current deficit"),
    _c("target_curve_ratio", "weight", "target_ratio", "direct", "current", True, 0, "Current observed weight compounded by the approved target-curve ratio to Day 35"),
    _c("historical_growth_ratio", "weight", "historical_ratio", "direct", "current", True, 1, "Current observed weight multiplied by the fold-local historical checkpoint-to-Day-35 ratio"),
    _c("recent_adg_projection", "weight", "recent_adg", "direct", "trajectory", True, 1, "Fold-tuned blend of recent observed ADG and historical expected remaining daily gain"),
    _c("direct_trajectory_pls", "weight", "pls", "direct", "trajectory", True, 2, "Low-rank observed growth trajectory"),
    _c("blend_baseline_pls", "weight", "blend_pls", "direct", "trajectory", True, 3, "Fold-tuned baseline and PLS blend"),
    _c("direct_trajectory_bayesian", "weight", "bayesian_ridge", "direct", "trajectory", True, 3, "Bayesian Ridge on observed trajectory"),
    _c("residual_trajectory_bayesian", "weight", "bayesian_ridge", "baseline_residual", "trajectory", True, 4, "Bayesian correction to historical remaining gain"),
    _c("residual_biological_ridge", "weight", "ridge", "baseline_residual", "biological", True, 4, "Biological and trajectory residual correction"),
    _c("direct_huber_biological", "weight", "huber", "direct", "biological", True, 5, "Robust biological trajectory model"),
    _c("direct_spline_biological", "weight", "spline", "direct", "biological", True, 6, "Regularized nonlinear biological trajectory"),
    _c("residual_random_forest_peer", "weight", "random_forest", "baseline_residual", "peer", True, 7, "Random forest correction with leave-self-out peer context"),
    _c("residual_extra_trees_peer", "weight", "extra_trees", "baseline_residual", "peer", True, 7, "Extra Trees correction with leave-self-out peer context"),
    _c("residual_gradient_boosting_peer", "weight", "gradient_boosting", "baseline_residual", "peer", True, 7, "Gradient boosting correction with peer context"),
    _c("residual_hist_gradient_peer", "weight", "hist_gradient_boosting", "baseline_residual", "peer", True, 7, "Histogram boosting correction with peer context"),
    _c("residual_hist_gradient_core", "weight", "hist_gradient_boosting", "baseline_residual", "biological", True, 7, "Histogram boosting correction without peer context"),
    _c("residual_hist_gradient_no_environment", "weight", "hist_gradient_boosting", "baseline_residual", "peer_no_environment", True, 7, "Peer-context histogram boosting without environment"),
    _c("residual_xgboost_peer", "weight", "xgboost", "baseline_residual", "peer", True, 8, "XGBoost correction with peer context"),
    _c("residual_lightgbm_peer", "weight", "lightgbm", "baseline_residual", "peer", True, 8, "LightGBM correction with peer context"),
    _c("residual_catboost_peer", "weight", "catboost", "baseline_residual", "peer", True, 8, "CatBoost correction with peer context"),
    _c("pooled_direct_ridge_biological", "weight", "ridge", "direct", "biological", False, 4, "One pooled checkpoint-aware Ridge model"),
    _c("pooled_direct_hist_gradient_peer", "weight", "hist_gradient_boosting", "direct", "peer", False, 7, "Pooled checkpoint-aware direct histogram boosting"),
    _c("pooled_remaining_hist_gradient_peer", "weight", "hist_gradient_boosting", "remaining", "peer", False, 7, "Pooled checkpoint-aware remaining-gain histogram boosting"),
    _c("pooled_direct_xgboost_peer", "weight", "xgboost", "direct", "peer", False, 8, "Pooled checkpoint-aware direct XGBoost"),
    _c("pooled_remaining_xgboost_peer", "weight", "xgboost", "remaining", "peer", False, 8, "Pooled checkpoint-aware remaining-gain XGBoost"),
    _c("pooled_direct_lightgbm_peer", "weight", "lightgbm", "direct", "peer", False, 8, "Pooled checkpoint-aware direct LightGBM"),
    _c("pooled_remaining_lightgbm_peer", "weight", "lightgbm", "remaining", "peer", False, 8, "Pooled checkpoint-aware remaining-gain LightGBM"),
    _c("pooled_direct_catboost_peer", "weight", "catboost", "direct", "peer", False, 8, "Pooled checkpoint-aware direct CatBoost"),
    _c("pooled_remaining_catboost_peer", "weight", "catboost", "remaining", "peer", False, 8, "Pooled checkpoint-aware remaining-gain CatBoost"),
)


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


def _leave_self_out(frame: pd.DataFrame, source: str, prefix: str) -> pd.DataFrame:
    grouped = frame.groupby(["cycle_id", "review_day"])[source]
    count = grouped.transform("count")
    total = grouped.transform("sum")
    frame[f"peer_{prefix}_mean"] = np.where(count > 1, (total - frame[source]) / (count - 1), np.nan)
    frame["peer_building_count"] = np.maximum(frame.get("peer_building_count", 0), count - 1)
    return frame


def add_optimization_features(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Add deterministic, as-of-safe feature recipes without using outcomes."""
    result = frame.copy()
    result["peer_building_count"] = 0
    if outcome == "recovery":
        for source, prefix in (
            ("percentage_alive", "survival"),
            ("mortality_recent_7d_per_1000", "mortality_7d"),
            ("latest_weight_kg", "latest_weight"),
        ):
            result = _leave_self_out(result, source, prefix)
        result["survival_vs_peer"] = result["percentage_alive"] - result["peer_survival_mean"]
        result["mortality_7d_vs_peer"] = result["mortality_recent_7d_per_1000"] - result["peer_mortality_7d_mean"]
        result["weight_vs_peer"] = result["latest_weight_kg"] - result["peer_latest_weight_mean"]
        result["mortality_horizon_pressure"] = result["mortality_recent_7d_per_1000"] * result["days_to_day35"]
        result["survival_horizon_interaction"] = result["percentage_alive"] * result["days_to_day35"]
        result["environment_missing_both"] = result["temperature_missing"] * result["humidity_missing"]
        result["environment_coverage_x_horizon"] = result["environment_coverage_ratio"] * result["days_to_day35"]
        result["weight_gap_x_staleness"] = result["weight_gap_pct"] * result["weight_staleness_days"]
        cumulative = result["mortality_recent_7d_per_1000"] + result["mortality_recent_vs_early_per_1000"].abs()
        result["recent_mortality_share"] = result["mortality_recent_7d_per_1000"] / cumulative.replace(0, np.nan)
        result["reconciliation_flag"] = result["population_mortality_reconciliation_gap_per_1000"].abs().gt(0.01).astype(float)
    else:
        for source, prefix in (("current_weight_g", "current_weight"), ("survival_pct", "survival_pct")):
            result = _leave_self_out(result, source, prefix)
        result = result.rename(columns={"peer_survival_pct_mean": "peer_survival_mean_pct"})
        result["current_weight_vs_peer_g"] = result["current_weight_g"] - result["peer_current_weight_mean"]
        result = result.rename(columns={"peer_current_weight_mean": "peer_current_weight_mean_g"})
        result["survival_vs_peer_pp"] = result["survival_pct"] - result["peer_survival_mean_pct"]
        result["log_current_weight_g"] = np.log(result["current_weight_g"].clip(lower=1))
        result["log_measurement_day"] = np.log(result["latest_measurement_day"].clip(lower=1))
        result["target_ratio_log"] = np.log(result["current_ratio_to_target"].clip(lower=0.05))
        fraction = (result["current_weight_g"] / 2500.0).clip(lower=0.01, upper=0.99)
        result["growth_fraction_2500g"] = fraction
        result["gompertz_linear_component"] = np.log(-np.log(fraction))
        days_remaining = 35 - result["review_day"]
        result["adg_x_days_remaining"] = result["last_interval_adg_g_day"] * days_remaining
        result["mortality_x_days_remaining"] = result["mortality_recent_7d_per_1000"] * days_remaining
        result["environment_missing_both"] = ((result["temperature_coverage"] == 0) & (result["humidity_coverage"] == 0)).astype(float)
        result["measurement_irregularity_flag"] = result["measurement_staleness_days"].gt(0).astype(float)
        result["implausible_adg_flag"] = (result["last_interval_adg_g_day"].lt(0) | result["last_interval_adg_g_day"].gt(120)).astype(float)
    return result


def build_optimization_snapshots(
    dataset: CanaryDataset,
    outcome: str,
    development_cycles: tuple[str, ...] = DEVELOPMENT_CYCLES,
    audit_cycle: str = AUDIT_CYCLE,
) -> pd.DataFrame:
    if outcome == "recovery":
        frame = build_external_snapshots(dataset, "recovery", development_cycles, audit_cycle)
        frame["actual"] = frame["outcome_y"]
        frame["current_value"] = frame["percentage_alive"]
        frame["remaining_target"] = frame["additional_loss_y"]
    else:
        frame = build_weight_snapshots(dataset, development_cycles, audit_cycle)
        frame["actual"] = frame["outcome_day35_weight_g"]
        frame["current_value"] = frame["current_weight_g"]
        frame["remaining_target"] = frame["outcome_day35_weight_g"] - frame["current_weight_g"]
    frame = add_optimization_features(frame, outcome)
    if not frame["max_source_day_used"].le(frame["review_day"]).all():
        raise AssertionError("Post-review evidence entered optimization snapshots.")
    return frame


def _features(candidate: OptimizationCandidate, review_day: int | None = None) -> list[str]:
    if candidate.outcome == "recovery":
        base = list(RECOVERY_COMPACT_FEATURES if candidate.feature_set == "core" else RECOVERY_FEATURES)
        if candidate.feature_set.startswith("peer"):
            base += RECOVERY_ENGINEERED
        if candidate.feature_set.endswith("no_environment"):
            base = [name for name in base if not any(token in name for token in ("temperature", "humidity", "environment"))]
    else:
        day = int(review_day if review_day is not None else 28)
        if candidate.feature_set == "current":
            base = weight_feature_columns(day, "current")
        elif candidate.feature_set == "trajectory":
            base = weight_feature_columns(day, "trajectory")
        elif candidate.feature_set == "biological":
            base = weight_feature_columns(day, "compact") + [
                "log_current_weight_g", "log_measurement_day", "target_ratio_log",
                "growth_fraction_2500g", "gompertz_linear_component",
                "adg_x_days_remaining", "mortality_x_days_remaining",
                "measurement_irregularity_flag", "implausible_adg_flag",
            ]
        else:
            base = weight_feature_columns(day, "poultry_core") + WEIGHT_ENGINEERED
        if candidate.feature_set.endswith("no_environment"):
            base = [name for name in base if not any(token in name for token in ("temperature", "humidity", "thi", "environment", "heat", "cold"))]
    columns = list(dict.fromkeys(base))
    assert_primary_schema_has_no_identity(columns)
    return columns


def candidate_grid(candidate: OptimizationCandidate) -> list[dict[str, Any]]:
    common = {"clip": (0.0, 1.0), "percentile": 100}
    if candidate.family in {"persistence", "baseline", "target_curve", "target_ratio", "historical_ratio", "linear"}:
        return [common]
    if candidate.family == "recent_adg":
        return [{**common, "recent_weight": weight} for weight in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if candidate.family == "blend_pls":
        return [{**common, "blend_weight": weight} for weight in (0.25, 0.5, 0.75)]
    if candidate.family == "ridge":
        return [{"alpha": alpha, "clip": clip, "percentile": percentile} for alpha, clip, percentile in ((1.0, (0.0, 1.0), 100), (10.0, (0.01, 0.99), 75), (50.0, (0.05, 0.95), 50), (100.0, (0.01, 0.99), 100))]
    if candidate.family == "bayesian_ridge":
        return [{"alpha_1": value, "clip": clip, "percentile": percentile} for value, clip, percentile in ((1e-6, (0.0, 1.0), 100), (1e-5, (0.01, 0.99), 75), (1e-4, (0.05, 0.95), 50))]
    if candidate.family == "elastic_net":
        return [{"alpha": alpha, "l1_ratio": ratio, "clip": (0.01, 0.99), "percentile": 75} for alpha, ratio in ((0.001, 0.1), (0.01, 0.5), (0.1, 0.9))]
    if candidate.family == "huber":
        return [{"epsilon": epsilon, "alpha": alpha, "clip": clip, "percentile": 75} for epsilon, alpha, clip in ((1.2, 0.001, (0.01, 0.99)), (1.35, 0.01, (0.01, 0.99)), (1.5, 0.1, (0.05, 0.95)))]
    if candidate.family == "pls":
        # Day 7 can collapse to one effective trajectory feature after fold-local
        # variance filtering, so a two-component PLS model is not always valid.
        return [{"components": 1, "clip": clip, "percentile": 100} for clip in ((0.0, 1.0), (0.01, 0.99))]
    if candidate.family == "spline":
        return [{"alpha": alpha, "knots": knots, "clip": (0.01, 0.99), "percentile": percentile} for alpha, knots, percentile in ((10.0, 3, 50), (50.0, 3, 75), (100.0, 4, 50))]
    if candidate.family in {"random_forest", "extra_trees"}:
        return [{"depth": depth, "leaf": leaf, "max_features": features, "clip": (0.0, 1.0), "percentile": 100} for depth, leaf, features in ((2, 3, 0.7), (3, 4, 0.7), (4, 5, 1.0))]
    if candidate.family in {"gradient_boosting", "hist_gradient_boosting"}:
        return [{"trees": trees, "rate": rate, "depth": depth, "leaf": leaf, "l2": l2, "clip": (0.0, 1.0), "percentile": percentile} for trees, rate, depth, leaf, l2, percentile in ((75, 0.04, 1, 5, 10.0, 100), (120, 0.025, 2, 8, 30.0, 75), (160, 0.02, 2, 10, 50.0, 50))]
    if candidate.family in {"xgboost", "lightgbm", "catboost"}:
        return [{"trees": trees, "rate": rate, "depth": depth, "leaf": leaf, "l2": l2, "clip": (0.0, 1.0), "percentile": percentile} for trees, rate, depth, leaf, l2, percentile in ((75, 0.03, 1, 5, 20.0, 100), (120, 0.025, 2, 8, 30.0, 75), (160, 0.02, 2, 10, 50.0, 50))]
    raise ValueError(candidate.family)


def _pipeline(candidate: OptimizationCandidate, parameters: dict[str, Any], seed: int) -> Pipeline:
    clip = parameters.get("clip", (0.0, 1.0))
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("winsor", QuantileWinsorizer(*clip)),
        ("variance", SafeVarianceFilter()),
        ("select", SafePercentileFilter(percentile=int(parameters.get("percentile", 100)))),
    ]
    linear = candidate.family in {"linear", "ridge", "bayesian_ridge", "elastic_net", "huber", "pls", "blend_pls", "spline"}
    if linear:
        if candidate.family == "spline":
            steps.append(("spline", SplineTransformer(n_knots=int(parameters.get("knots", 3)), degree=2, include_bias=False)))
        steps.append(("scale", StandardScaler()))
    if candidate.family == "linear": model: Any = LinearRegression()
    elif candidate.family == "ridge" or candidate.family == "spline": model = Ridge(alpha=float(parameters.get("alpha", 10.0)))
    elif candidate.family == "bayesian_ridge": model = BayesianRidge(alpha_1=float(parameters.get("alpha_1", 1e-6)), alpha_2=float(parameters.get("alpha_1", 1e-6)))
    elif candidate.family == "elastic_net": model = ElasticNet(alpha=float(parameters["alpha"]), l1_ratio=float(parameters["l1_ratio"]), max_iter=20000, random_state=seed)
    elif candidate.family == "huber": model = HuberRegressor(epsilon=float(parameters["epsilon"]), alpha=float(parameters["alpha"]), max_iter=6000)
    elif candidate.family in {"pls", "blend_pls"}: model = PLSRegression(n_components=int(parameters.get("components", 1)), scale=False, max_iter=2000)
    elif candidate.family == "random_forest": model = RandomForestRegressor(n_estimators=400, max_depth=parameters["depth"], min_samples_leaf=int(parameters["leaf"]), max_features=parameters["max_features"], random_state=seed, n_jobs=1)
    elif candidate.family == "extra_trees": model = ExtraTreesRegressor(n_estimators=400, max_depth=parameters["depth"], min_samples_leaf=int(parameters["leaf"]), max_features=parameters["max_features"], random_state=seed, n_jobs=1)
    elif candidate.family == "gradient_boosting": model = GradientBoostingRegressor(n_estimators=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_depth=int(parameters["depth"]), min_samples_leaf=int(parameters["leaf"]), loss="huber", random_state=seed)
    elif candidate.family == "hist_gradient_boosting": model = HistGradientBoostingRegressor(max_iter=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_leaf_nodes=max(3, 2 ** int(parameters["depth"]) + 1), min_samples_leaf=int(parameters["leaf"]), l2_regularization=float(parameters["l2"]), random_state=seed)
    elif candidate.family == "xgboost": model = XGBRegressor(n_estimators=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_depth=int(parameters["depth"]), min_child_weight=float(parameters["leaf"]), reg_lambda=float(parameters["l2"]), reg_alpha=1.0, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=1, verbosity=0)
    elif candidate.family == "lightgbm": model = LGBMRegressor(n_estimators=int(parameters["trees"]), learning_rate=float(parameters["rate"]), max_depth=int(parameters["depth"]), num_leaves=max(2, 2 ** int(parameters["depth"]) - 1), min_child_samples=int(parameters["leaf"]), reg_lambda=float(parameters["l2"]), reg_alpha=1.0, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    elif candidate.family == "catboost": model = CatBoostRegressor(iterations=int(parameters["trees"]), learning_rate=float(parameters["rate"]), depth=int(parameters["depth"]), l2_leaf_reg=float(parameters["l2"]), random_seed=seed, loss_function="RMSE", random_strength=0.5, allow_writing_files=False, thread_count=1, verbose=False)
    else: raise ValueError(candidate.family)
    steps.append(("model", model))
    return Pipeline(steps)


def _baseline_map(train: pd.DataFrame) -> dict[str, Any]:
    values = train.groupby("review_day")["remaining_target"].mean().to_dict()
    ratio = (train["actual"] / train["current_value"].replace(0, np.nan)).groupby(train["review_day"]).mean().to_dict()
    horizon = (35 - train["review_day"]).replace(0, np.nan)
    daily_gain = (train["remaining_target"] / horizon).groupby(train["review_day"]).mean().to_dict()
    return {
        "mapping": {int(day): float(value) for day, value in values.items()},
        "fallback": float(train["remaining_target"].mean()),
        "ratio_mapping": {int(day): float(value) for day, value in ratio.items()},
        "ratio_fallback": float((train["actual"] / train["current_value"].replace(0, np.nan)).mean()),
        "daily_gain_mapping": {int(day): float(value) for day, value in daily_gain.items()},
        "daily_gain_fallback": float((train["remaining_target"] / horizon).mean()),
    }


def _baseline_prediction(frame: pd.DataFrame, baseline: dict[str, Any], outcome: str) -> np.ndarray:
    remaining = np.asarray([baseline["mapping"].get(int(day), baseline["fallback"]) for day in frame["review_day"]], dtype=float)
    if outcome == "recovery":
        return frame["current_value"].to_numpy(float) - np.maximum(remaining, 0.0)
    return frame["current_value"].to_numpy(float) + remaining


def _training_target(train: pd.DataFrame, candidate: OptimizationCandidate, baseline: dict[str, Any]) -> np.ndarray:
    if candidate.target_form == "remaining":
        return train["remaining_target"].to_numpy(float)
    if candidate.target_form == "baseline_residual":
        return train["actual"].to_numpy(float) - _baseline_prediction(train, baseline, candidate.outcome)
    return train["actual"].to_numpy(float)


def _fit_entry(train: pd.DataFrame, candidate: OptimizationCandidate, parameters: dict[str, Any], seed: int) -> dict[str, Any]:
    baseline = _baseline_map(train)
    if candidate.family in {"persistence", "baseline", "target_curve", "target_ratio", "historical_ratio", "recent_adg"}:
        return {"family": candidate.family, "baseline": baseline, "parameters": parameters}
    review_day = int(train["review_day"].iloc[0]) if candidate.checkpoint_specific else None
    columns = _features(candidate, review_day)
    model = _pipeline(candidate, parameters, seed)
    model.fit(train[columns], _training_target(train, candidate, baseline))
    return {"family": candidate.family, "baseline": baseline, "model": model, "features": columns, "parameters": parameters}


def _predict_entry(entry: dict[str, Any], frame: pd.DataFrame, candidate: OptimizationCandidate) -> np.ndarray:
    baseline = _baseline_prediction(frame, entry["baseline"], candidate.outcome)
    if candidate.family == "persistence":
        prediction = frame["current_value"].to_numpy(float)
    elif candidate.family == "baseline":
        prediction = baseline
    elif candidate.family == "target_curve":
        target_now = frame["current_target_g"].to_numpy(float)
        prediction = frame["current_weight_g"].to_numpy(float) + (DAY35_TARGET_G - target_now)
    elif candidate.family == "target_ratio":
        target_now = frame["current_target_g"].to_numpy(float)
        prediction = frame["current_weight_g"].to_numpy(float) * DAY35_TARGET_G / np.maximum(target_now, 1.0)
    elif candidate.family == "historical_ratio":
        mapping = entry["baseline"]["ratio_mapping"]
        fallback = entry["baseline"]["ratio_fallback"]
        ratios = np.asarray([mapping.get(int(day), fallback) for day in frame["review_day"]], dtype=float)
        prediction = frame["current_weight_g"].to_numpy(float) * ratios
    elif candidate.family == "recent_adg":
        mapping = entry["baseline"]["daily_gain_mapping"]
        fallback = entry["baseline"]["daily_gain_fallback"]
        expected = np.asarray([mapping.get(int(day), fallback) for day in frame["review_day"]], dtype=float)
        recent = frame["last_interval_adg_g_day"].to_numpy(float)
        recent = np.where(np.isfinite(recent), recent, expected)
        weight = float(entry["parameters"].get("recent_weight", 0.5))
        blended = weight * recent + (1.0 - weight) * expected
        prediction = frame["current_weight_g"].to_numpy(float) + blended * (35 - frame["review_day"].to_numpy(float))
    else:
        raw = np.asarray(entry["model"].predict(frame[entry["features"]]), dtype=float).reshape(-1)
        if candidate.target_form == "remaining":
            prediction = frame["current_value"].to_numpy(float) - np.maximum(raw, 0) if candidate.outcome == "recovery" else frame["current_value"].to_numpy(float) + raw
        elif candidate.target_form == "baseline_residual":
            prediction = baseline + raw
        else:
            prediction = raw
        if candidate.family == "blend_pls":
            weight = float(entry["parameters"].get("blend_weight", 0.5))
            prediction = weight * prediction + (1.0 - weight) * baseline
    if candidate.outcome == "recovery":
        current = frame["current_value"].to_numpy(float)
        return np.minimum(np.clip(prediction, 0.0, 1.0), current)
    return np.clip(prediction, 100.0, 3500.0)


def fit_candidate(train: pd.DataFrame, candidate: OptimizationCandidate, parameters: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    if candidate.checkpoint_specific:
        entries = {int(day): _fit_entry(group.reset_index(drop=True), candidate, parameters, seed) for day, group in train.groupby("review_day")}
        return {"checkpoint_specific": True, "entries": entries, "candidate": asdict(candidate)}
    return {"checkpoint_specific": False, "entry": _fit_entry(train.reset_index(drop=True), candidate, parameters, seed), "candidate": asdict(candidate)}


def predict_candidate(bundle: dict[str, Any], frame: pd.DataFrame, candidate: OptimizationCandidate) -> np.ndarray:
    if not bundle["checkpoint_specific"]:
        return _predict_entry(bundle["entry"], frame, candidate)
    output = pd.Series(index=frame.index, dtype=float)
    for day, group in frame.groupby("review_day"):
        output.loc[group.index] = _predict_entry(bundle["entries"][int(day)], group, candidate)
    return output.loc[frame.index].to_numpy(float)


def _cycle_macro_rmse(actual: np.ndarray, predicted: np.ndarray, cycles: np.ndarray, factor: float) -> float:
    values = []
    for cycle in pd.unique(cycles):
        selected = cycles == cycle
        values.append(mean_squared_error(actual[selected], predicted[selected]) ** 0.5 * factor)
    return float(np.mean(values))


def tune_candidate(train: pd.DataFrame, candidate: OptimizationCandidate, seed: int = SEED) -> dict[str, Any]:
    options = candidate_grid(candidate)
    if len(options) == 1:
        return options[0]
    groups = train["cycle_id"].astype(str).to_numpy()
    if len(pd.unique(groups)) < 2:
        return options[0]
    actual = train["actual"].to_numpy(float)
    factor = 100.0 if candidate.outcome == "recovery" else 1.0
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for order, parameters in enumerate(options):
        predictions = np.full(len(train), np.nan)
        for fit_index, valid_index in LeaveOneGroupOut().split(train, groups=groups):
            fitted = fit_candidate(train.iloc[fit_index], candidate, parameters, seed)
            predictions[valid_index] = predict_candidate(fitted, train.iloc[valid_index], candidate)
        scored.append((_cycle_macro_rmse(actual, predictions, groups, factor), order, parameters))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def _group_values(frame: pd.DataFrame, view: str) -> np.ndarray:
    if view == "cycle": return frame["cycle_id"].astype(str).to_numpy()
    if view == "building_label": return frame["building_id"].astype(str).to_numpy()
    if view == "building_cycle": return (frame["cycle_id"].astype(str) + "::" + frame["building_id"].astype(str)).to_numpy()
    raise ValueError(view)


def evaluate_logo(frame: pd.DataFrame, candidate: OptimizationCandidate, view: str = "cycle", seed: int = SEED, fixed_parameters: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    groups = _group_values(frame, view)
    rows: list[dict[str, Any]] = []
    fold_parameters: list[dict[str, Any]] = []
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index]
        parameters = fixed_parameters or tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        held_group = str(groups[test_index][0])
        fold_parameters.append({"held_out_group": held_group, "view": view, **parameters})
        for source, value in zip(test.to_dict("records"), predicted):
            rows.append({
                "candidate": candidate.name, "validation_view": view, "held_out_group": held_group,
                "cycle_id": source["cycle_id"], "building_id": source["building_id"],
                "review_day": int(source["review_day"]), "as_of_date": source["as_of_date"],
                "actual": float(source["actual"]), "predicted": float(value),
                "error": float(value - source["actual"]), "seed": seed,
            })
    return pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True), fold_parameters


def _target_metrics(actual: np.ndarray, predicted: np.ndarray, outcome: str) -> dict[str, Any]:
    threshold = RECOVERY_TARGET if outcome == "recovery" else DAY35_TARGET_G
    actual_hit, predicted_hit = actual >= threshold, predicted >= threshold
    below = float(np.mean(~predicted_hit[~actual_hit])) if (~actual_hit).any() else np.nan
    above = float(np.mean(predicted_hit[actual_hit])) if actual_hit.any() else np.nan
    return {
        "target_side_accuracy": float(np.mean(actual_hit == predicted_hit)),
        "majority_side_accuracy": float(max(np.mean(actual_hit), np.mean(~actual_hit))),
        "below_target_recall": below, "at_or_above_target_recall": above,
        "balanced_target_accuracy": float(np.nanmean([below, above])),
        "actual_target_hits": int(actual_hit.sum()),
    }


def summarize(predictions: pd.DataFrame, outcome: str, bootstrap: bool = True) -> dict[str, Any]:
    actual = predictions["actual"].to_numpy(float)
    predicted = predictions["predicted"].to_numpy(float)
    factor = 100.0 if outcome == "recovery" else 1.0
    errors = predicted - actual
    by_cycle = predictions.assign(abs_error=np.abs(errors), sq_error=errors ** 2).groupby("cycle_id").agg(mae=("abs_error", "mean"), mse=("sq_error", "mean"))
    cycle_rmse = np.sqrt(by_cycle["mse"]) * factor
    result = {
        "rows": int(len(predictions)),
        "independent_building_cycles": int(predictions[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5 * factor),
        "cycle_macro_rmse": float(cycle_rmse.mean()),
        "mae": float(mean_absolute_error(actual, predicted) * factor),
        "cycle_macro_mae": float(by_cycle["mae"].mean() * factor),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(errors.mean() * factor),
        "worst_cycle_rmse": float(cycle_rmse.max()),
        "fold_rmse_sd": float(cycle_rmse.std(ddof=1)),
        "fold_rmse_se": float(cycle_rmse.std(ddof=1) / np.sqrt(len(cycle_rmse))),
    }
    if outcome == "weight":
        result["within_100g_rate"] = float(np.mean(np.abs(errors) <= 100))
        result["within_200g_rate"] = float(np.mean(np.abs(errors) <= 200))
    result.update(_target_metrics(actual, predicted, outcome))
    if bootstrap:
        rng = np.random.default_rng(SEED)
        cycles = pd.unique(predictions["cycle_id"])
        rmses, r2s = [], []
        for _ in range(3000):
            sampled = rng.choice(cycles, len(cycles), replace=True)
            pieces = [predictions.loc[predictions["cycle_id"].eq(cycle)] for cycle in sampled]
            boot = pd.concat(pieces, ignore_index=True)
            rmses.append(mean_squared_error(boot["actual"], boot["predicted"]) ** 0.5 * factor)
            r2s.append(r2_score(boot["actual"], boot["predicted"]) if boot["actual"].nunique() > 1 else np.nan)
        result["rmse_95ci_low"], result["rmse_95ci_high"] = [float(value) for value in np.nanquantile(rmses, [0.025, 0.975])]
        result["r2_95ci_low"], result["r2_95ci_high"] = [float(value) for value in np.nanquantile(r2s, [0.025, 0.975])]
    return result


def checkpoint_metrics(predictions: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for day, group in predictions.groupby("review_day"):
        rows.append({"review_day": int(day), **summarize(group, outcome, bootstrap=False)})
    return pd.DataFrame(rows)


def temporal_stress(frame: pd.DataFrame, candidate: OptimizationCandidate, seed: int = SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    cycles = sorted(frame["cycle_id"].astype(str).unique(), key=_cycle_key)
    for position in range(2, len(cycles)):
        train = frame.loc[frame["cycle_id"].isin(cycles[:position])].reset_index(drop=True)
        test = frame.loc[frame["cycle_id"].eq(cycles[position])]
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        prediction = predict_candidate(fitted, test, candidate)
        for source, value in zip(test.to_dict("records"), prediction):
            rows.append({"candidate": candidate.name, "cycle_id": source["cycle_id"], "building_id": source["building_id"], "review_day": source["review_day"], "actual": source["actual"], "predicted": value, "error": value - source["actual"]})
    predictions = pd.DataFrame(rows)
    return predictions, summarize(predictions, candidate.outcome, bootstrap=False)


def conformal_logo(frame: pd.DataFrame, candidate: OptimizationCandidate, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, calibration_rows = [], []
    groups = frame["cycle_id"].astype(str).to_numpy()
    for outer_train, outer_test in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[outer_train].reset_index(drop=True), frame.iloc[outer_test]
        parameters = tune_candidate(train, candidate, seed)
        train_groups = train["cycle_id"].astype(str).to_numpy()
        calibration_prediction = np.full(len(train), np.nan)
        for inner_train, inner_test in LeaveOneGroupOut().split(train, groups=train_groups):
            fitted = fit_candidate(train.iloc[inner_train], candidate, parameters, seed)
            calibration_prediction[inner_test] = predict_candidate(fitted, train.iloc[inner_test], candidate)
        residual = np.abs(calibration_prediction - train["actual"].to_numpy(float))
        normalized = residual / np.sqrt(np.maximum(1.0, 35 - train["review_day"].to_numpy(float)))
        n = len(residual)
        q80 = float(np.quantile(residual, min(1.0, np.ceil((n + 1) * 0.8) / n), method="higher"))
        q90 = float(np.quantile(residual, min(1.0, np.ceil((n + 1) * 0.9) / n), method="higher"))
        nq80 = float(np.quantile(normalized, min(1.0, np.ceil((n + 1) * 0.8) / n), method="higher"))
        nq90 = float(np.quantile(normalized, min(1.0, np.ceil((n + 1) * 0.9) / n), method="higher"))
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        for source, value in zip(test.to_dict("records"), predicted):
            scale = np.sqrt(max(1.0, 35 - float(source["review_day"])))
            width80, width90 = min(q80, nq80 * scale), min(q90, nq90 * scale)
            lower80, upper80 = value - width80, value + width80
            lower90, upper90 = value - width90, value + width90
            actual = float(source["actual"])
            rows.append({"cycle_id": source["cycle_id"], "building_id": source["building_id"], "review_day": source["review_day"], "actual": actual, "predicted": value, "lower_80": lower80, "upper_80": upper80, "lower_90": lower90, "upper_90": upper90, "covered_80": lower80 <= actual <= upper80, "covered_90": lower90 <= actual <= upper90, "checkpoint_coverage_reliable": bool((train["review_day"] == source["review_day"]).sum() >= 20)})
        calibration_rows.append({"held_out_cycle": str(test["cycle_id"].iloc[0]), "calibration_rows": n, "q80": q80, "q90": q90, "normalized_q80": nq80, "normalized_q90": nq90})
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def permutation_importance_logo(frame: pd.DataFrame, candidate: OptimizationCandidate, seed: int = SEED, repeats: int = 5) -> pd.DataFrame:
    groups = frame["cycle_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    union_features = sorted(set().union(*[_features(candidate, int(day) if candidate.checkpoint_specific else None) for day in CHECKPOINTS]))
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index].copy()
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        base = predict_candidate(fitted, test, candidate)
        base_rmse = mean_squared_error(test["actual"], base) ** 0.5
        factor = 100.0 if candidate.outcome == "recovery" else 1.0
        for feature in union_features:
            if feature not in test:
                continue
            increases = []
            for _ in range(repeats):
                permuted = test.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                prediction = predict_candidate(fitted, permuted, candidate)
                increases.append((mean_squared_error(test["actual"], prediction) ** 0.5 - base_rmse) * factor)
            rows.append({"held_out_cycle": str(test["cycle_id"].iloc[0]), "candidate": candidate.name, "feature": feature, "rmse_increase": float(np.mean(increases)), "rmse_increase_sd": float(np.std(increases, ddof=1))})
    details = pd.DataFrame(rows)
    return details.groupby(["candidate", "feature"], as_index=False).agg(mean_rmse_increase=("rmse_increase", "mean"), fold_sd=("rmse_increase", "std"), positive_fold_rate=("rmse_increase", lambda value: float((value > 0).mean()))).sort_values("mean_rmse_increase", ascending=False)


TREE_FAMILIES = {"random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"}
LINEAR_FAMILIES = {"linear", "ridge", "bayesian_ridge", "elastic_net", "huber", "pls", "blend_pls"}


def _transformed_names(pipeline: Pipeline, original: list[str]) -> list[str]:
    try:
        return [str(value) for value in pipeline[:-1].get_feature_names_out(original)]
    except Exception:
        count = int(np.asarray(pipeline[:-1].transform(pd.DataFrame([dict.fromkeys(original, np.nan)]))).shape[1])
        return [f"transformed_feature_{index}" for index in range(count)]


def held_out_shap(frame: pd.DataFrame, candidate: OptimizationCandidate, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate.family not in TREE_FAMILIES:
        return pd.DataFrame(), pd.DataFrame()
    groups = frame["cycle_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index]
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        entries = fitted["entries"] if fitted["checkpoint_specific"] else {None: fitted["entry"]}
        for day, entry in entries.items():
            selected = test if day is None else test.loc[test["review_day"].eq(day)]
            if selected.empty:
                continue
            pipeline = entry["model"]
            transformed = np.asarray(pipeline[:-1].transform(selected[entry["features"]]), dtype=float)
            names = _transformed_names(pipeline, entry["features"])
            explainer = shap.TreeExplainer(pipeline.named_steps["model"])
            values = np.asarray(explainer.shap_values(transformed), dtype=float)
            if values.ndim == 3:
                values = values[..., 0]
            expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
            base_value = float(expected[0]) if len(expected) else 0.0
            model_output = np.asarray(pipeline.named_steps["model"].predict(transformed), dtype=float).reshape(-1)
            for row_index, source in enumerate(selected.to_dict("records")):
                for column_index, name in enumerate(names):
                    rows.append({"held_out_cycle": source["cycle_id"], "building_id": source["building_id"], "review_day": source["review_day"], "feature": name, "feature_value": transformed[row_index, column_index], "shap_value": values[row_index, column_index], "base_value": base_value, "model_output": float(model_output[row_index])})
    local = pd.DataFrame(rows)
    if local.empty:
        return pd.DataFrame(), local
    fold_direction = local.groupby(["held_out_cycle", "feature"])["shap_value"].mean().reset_index()
    global_rows = []
    for feature, group in local.groupby("feature"):
        correlation = group[["feature_value", "shap_value"]].corr().iloc[0, 1] if group["feature_value"].nunique() > 1 else np.nan
        directions = fold_direction.loc[fold_direction["feature"].eq(feature), "shap_value"]
        positive_rate = float((directions > 0).mean()) if len(directions) else np.nan
        global_rows.append({"feature": feature, "mean_abs_shap": float(group["shap_value"].abs().mean()), "mean_shap": float(group["shap_value"].mean()), "value_shap_correlation": correlation, "positive_fold_direction_rate": positive_rate, "direction_stable": bool(positive_rate >= 0.8 or positive_rate <= 0.2) if pd.notna(positive_rate) else False})
    return pd.DataFrame(global_rows).sort_values("mean_abs_shap", ascending=False), local


def standardized_coefficients(frame: pd.DataFrame, candidate: OptimizationCandidate, seed: int = SEED) -> pd.DataFrame:
    if candidate.family not in LINEAR_FAMILIES:
        return pd.DataFrame()
    parameters = tune_candidate(frame, candidate, seed)
    fitted = fit_candidate(frame, candidate, parameters, seed)
    entries = fitted["entries"] if fitted["checkpoint_specific"] else {None: fitted["entry"]}
    rows = []
    for day, entry in entries.items():
        pipeline = entry["model"]
        model = pipeline.named_steps["model"]
        coefficient = np.asarray(getattr(model, "coef_", []), dtype=float).reshape(-1)
        names = _transformed_names(pipeline, entry["features"])
        if len(coefficient) != len(names):
            continue
        rows.extend({"candidate": candidate.name, "review_day": day if day is not None else "pooled", "feature": name, "standardized_coefficient": float(value)} for name, value in zip(names, coefficient))
    return pd.DataFrame(rows).sort_values("standardized_coefficient", key=lambda values: values.abs(), ascending=False) if rows else pd.DataFrame()


def seed_stability(frame: pd.DataFrame, candidate: OptimizationCandidate) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, predictions = [], []
    for seed in FIXED_SEEDS:
        output, _ = evaluate_logo(frame, candidate, "cycle", seed)
        predictions.append(output)
        rows.append({"seed": seed, **summarize(output, candidate.outcome, bootstrap=False)})
    metrics = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "candidate": candidate.name,
        "seeds": len(FIXED_SEEDS),
        "cycle_macro_rmse_mean": float(metrics["cycle_macro_rmse"].mean()),
        "cycle_macro_rmse_sd": float(metrics["cycle_macro_rmse"].std(ddof=1)),
        "r2_mean": float(metrics["r2"].mean()),
        "r2_sd": float(metrics["r2"].std(ddof=1)),
        "bias_range": float(metrics["bias"].max() - metrics["bias"].min()),
    }])
    return pd.concat(predictions, ignore_index=True), metrics, summary


def select_one_se(comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ranked = comparison.sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
    best = ranked.iloc[0]
    limit = float(best["cycle_macro_rmse"] + best["fold_rmse_se"])
    eligible = ranked.loc[ranked["cycle_macro_rmse"].le(limit)]
    selected = eligible.sort_values(["complexity", "cycle_macro_rmse", "candidate"]).iloc[0]
    return str(selected["candidate"]), {
        "lowest_error_candidate": str(best["candidate"]), "lowest_cycle_macro_rmse": float(best["cycle_macro_rmse"]),
        "one_se_limit": limit, "eligible_candidates": eligible["candidate"].astype(str).tolist(),
        "selected_candidate": str(selected["candidate"]), "rule": "simplest candidate within one fold standard error of the lowest cycle-macro RMSE",
    }


def _cycle_wins(predictions: pd.DataFrame, challenger: str, baseline: str) -> int:
    table = predictions.loc[predictions["candidate"].isin([challenger, baseline])].copy()
    table["sq"] = table["error"] ** 2
    rmse = table.groupby(["candidate", "cycle_id"])["sq"].mean().pow(0.5).unstack("candidate")
    return int((rmse[challenger] < rmse[baseline]).sum())


def promotion_gate(comparison: pd.DataFrame, predictions: pd.DataFrame, outcome: str, challenger: str, baseline: str, checkpoint: pd.DataFrame, intervals: pd.DataFrame, audit_metrics: dict[str, Any], frozen_audit_rmse: float | None) -> dict[str, Any]:
    indexed = comparison.set_index("candidate")
    c, b = indexed.loc[challenger], indexed.loc[baseline]
    improvement = (float(b["cycle_macro_rmse"]) - float(c["cycle_macro_rmse"])) / float(b["cycle_macro_rmse"]) * 100
    day14 = checkpoint.set_index(["candidate", "review_day"])
    checkpoint_pass = all(float(day14.loc[(challenger, day), "rmse"]) <= float(day14.loc[(baseline, day), "rmse"]) * 1.05 for day in CHECKPOINTS)
    checks = {
        "at_least_10pct_better_than_baseline": bool(improvement >= 10),
        "positive_held_out_r2": bool(float(c["r2"]) > 0),
        "mae_not_worse_than_5pct": bool(float(c["mae"]) <= float(b["mae"]) * 1.05),
        "bias_within_limit": bool(abs(float(c["bias"])) <= (0.5 if outcome == "recovery" else 50.0)),
        "worst_cycle_not_worse_than_10pct": bool(float(c["worst_cycle_rmse"]) <= float(b["worst_cycle_rmse"]) * 1.10),
        "beats_baseline_in_at_least_four_cycles": bool(_cycle_wins(predictions, challenger, baseline) >= 4),
        "checkpoint_guardrail_passed": checkpoint_pass,
        "interval_80_coverage_credible": bool(0.70 <= float(intervals["covered_80"].mean()) <= 0.90),
        "interval_90_coverage_credible": bool(0.80 <= float(intervals["covered_90"].mean()) <= 1.00),
        "later_cycle_not_materially_worse": bool(frozen_audit_rmse is not None and float(audit_metrics["rmse"]) <= frozen_audit_rmse * 1.10),
    }
    return {"challenger": challenger, "baseline": baseline, "cycle_macro_rmse_improvement_pct": improvement, "checks": checks, "retrospective_gate_passed": bool(all(checks.values())), "operational_promotion_allowed": False, "prospective_cycles_required": 3}


def _plot_outputs(directory: Path, outcome: str, predictions: pd.DataFrame, comparison: pd.DataFrame, checkpoint: pd.DataFrame, intervals: pd.DataFrame, importance: pd.DataFrame, shap_global: pd.DataFrame, ablation: pd.DataFrame, seed_metrics: pd.DataFrame, snapshots: pd.DataFrame, temporal_predictions: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    blue, light, orange, ink = "#2A6F97", "#A9C6D8", "#D17A22", "#263238"
    unit = "percentage points" if outcome == "recovery" else "g"
    top = comparison.head(8).sort_values("cycle_macro_rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(top["candidate"].str.replace("_", " "), top["cycle_macro_rmse"], color=[blue if rank == 0 else light for rank in range(len(top))]); ax.set(xlim=(0, top["cycle_macro_rmse"].max()*1.18), xlabel=f"Cycle-macro RMSE ({unit})", ylabel="", title=f"{outcome.title()} optimization candidates");
    for index, value in enumerate(top["cycle_macro_rmse"]): ax.text(value + top["cycle_macro_rmse"].max()*.015, index, f"{value:.2f}", va="center", color=ink)
    fig.text(.5,.97,"Nested leave-one-harvest-cycle-out validation · 31 building-cycles · lower is better",ha="center",fontsize=9,color="#5F6B73"); fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(directory/"model_comparison.png",dpi=180); plt.close(fig)
    best = str(comparison.iloc[0]["candidate"]); selected = predictions.loc[predictions["candidate"].eq(best)]
    fig, ax = plt.subplots(figsize=(6.5,6)); ax.scatter(selected["actual"],selected["predicted"],c=selected["review_day"],cmap="viridis",alpha=.75,edgecolor="white"); bounds=[min(selected[["actual","predicted"]].min()),max(selected[["actual","predicted"]].max())]; ax.plot(bounds,bounds,"--",color=ink); ax.set(xlabel=f"Actual ({unit})",ylabel=f"Predicted ({unit})",title=f"Held-out actual versus predicted — {best.replace('_',' ')}"); fig.tight_layout(); fig.savefig(directory/"actual_vs_predicted.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5,4.8)); ax.scatter(selected["predicted"],selected["error"],c=selected["review_day"],cmap="viridis",alpha=.75); ax.axhline(0,color=ink,ls="--"); ax.set(xlabel=f"Predicted ({unit})",ylabel=f"Error ({unit})",title="Held-out residuals"); fig.tight_layout(); fig.savefig(directory/"residuals.png",dpi=180); plt.close(fig)
    cp = checkpoint.loc[checkpoint["candidate"].eq(best)]; fig,ax=plt.subplots(figsize=(7.5,4.8)); ax.bar(cp["review_day"].astype(str),cp["rmse"],color=blue); ax.set(xlabel="Review day",ylabel=f"RMSE ({unit})",title="Error by validated checkpoint"); fig.tight_layout(); fig.savefig(directory/"checkpoint_performance.png",dpi=180); plt.close(fig)
    coverage=intervals.groupby("review_day")[["covered_80","covered_90"]].mean()*100; fig,ax=plt.subplots(figsize=(7.5,4.8)); coverage.plot(kind="bar",ax=ax,color=[light,blue]); ax.axhline(80,color=light,ls="--"); ax.axhline(90,color=blue,ls="--"); ax.set(xlabel="Review day",ylabel="Empirical coverage (%)",ylim=(0,110),title="Grouped conformal interval coverage"); fig.tight_layout(); fig.savefig(directory/"interval_coverage.png",dpi=180); plt.close(fig)
    if not importance.empty:
        topi=importance.head(12).sort_values("mean_rmse_increase"); fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.barh(topi["feature"].str.replace("_"," "),topi["mean_rmse_increase"],color=orange); ax.axvline(0,color=ink,lw=.8); ax.set(xlabel=f"Held-out RMSE increase ({unit})",title="Permutation importance"); fig.tight_layout(); fig.savefig(directory/"permutation_importance.png",dpi=180); plt.close(fig)
    if not shap_global.empty:
        tops=shap_global.head(12).sort_values("mean_abs_shap"); fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.barh(tops["feature"].str.replace("_"," "),tops["mean_abs_shap"]*(100 if outcome=="recovery" else 1),color="#A64D79"); ax.set(xlabel=f"Mean absolute SHAP ({unit})",title="Held-out SHAP importance"); fig.tight_layout(); fig.savefig(directory/"shap_importance.png",dpi=180); plt.close(fig)
    if not ablation.empty:
        plot=ablation.sort_values("cycle_macro_rmse",ascending=False); fig,ax=plt.subplots(figsize=(8,4.8)); ax.barh(plot["experiment"],plot["cycle_macro_rmse"],color=light); ax.set(xlabel=f"Cycle-macro RMSE ({unit})",title="Feature recipe ablation"); fig.tight_layout(); fig.savefig(directory/"feature_ablation.png",dpi=180); plt.close(fig)
    if not seed_metrics.empty:
        fig,ax=plt.subplots(figsize=(7.5,4.8)); ax.plot(seed_metrics["seed"].astype(str),seed_metrics["cycle_macro_rmse"],marker="o",color=blue); ax.set(xlabel="Fixed seed",ylabel=f"Cycle-macro RMSE ({unit})",title="Stochastic finalist stability"); fig.tight_layout(); fig.savefig(directory/"seed_stability.png",dpi=180); plt.close(fig)
    factor = 100.0 if outcome == "recovery" else 1.0
    cycle = selected.assign(sq_error=selected["error"] ** 2).groupby("cycle_id")["sq_error"].mean().pow(.5) * factor
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(cycle.index.astype(str),cycle.values,color=light); ax.axhline(cycle.mean(),color=blue,ls="--",label="cycle mean"); ax.set(xlabel="Held-out harvest cycle",ylabel=f"RMSE ({unit})",title="Outer-fold stability"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(directory/"fold_stability.png",dpi=180); plt.close(fig)
    actual_units = selected[["cycle_id","building_id","actual"]].drop_duplicates()
    fig,ax=plt.subplots(figsize=(7.5,4.8)); ax.hist(actual_units["actual"] * factor,bins=min(12,len(actual_units)),color=blue,alpha=.85); ax.set(xlabel=f"Mature outcome ({unit})",ylabel="Building-cycles",title="Outcome distribution"); fig.tight_layout(); fig.savefig(directory/"outcome_distribution.png",dpi=180); plt.close(fig)
    missing = snapshots.isna().mean().mul(100).sort_values(ascending=False).head(15).sort_values()
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.barh(missing.index.str.replace("_"," "),missing.values,color=light); ax.set(xlabel="Missing snapshot rows (%)",title="Highest snapshot missingness"); fig.tight_layout(); fig.savefig(directory/"missingness.png",dpi=180); plt.close(fig)
    if not temporal_predictions.empty:
        temporal = temporal_predictions.assign(sq_error=temporal_predictions["error"] ** 2).groupby(["candidate","cycle_id"])["sq_error"].mean().pow(.5).reset_index()
        temporal["sq_error"] *= factor
        fig,ax=plt.subplots(figsize=(8,4.8))
        for name, group in temporal.groupby("candidate"):
            ax.plot(group["cycle_id"].astype(str),group["sq_error"],marker="o",label=name.replace("_"," "))
        ax.set(xlabel="Later development cycle",ylabel=f"RMSE ({unit})",title="Expanding-window temporal stress"); ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(directory/"temporal_stability.png",dpi=180); plt.close(fig)


def _package_versions() -> dict[str, str]:
    result = {}
    for package in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap", "joblib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _lookup(candidates: tuple[OptimizationCandidate, ...], name: str) -> OptimizationCandidate:
    return next(candidate for candidate in candidates if candidate.name == name)


def _data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "actual": "Mature outcome used only as the supervised label.",
        "review_day": "Validated as-of checkpoint day.",
        "as_of_date": "Latest calendar date allowed into the snapshot.",
        "max_source_day_used": "Maximum source age used; must not exceed review_day.",
        "role": "Development or locked later-cycle audit assignment.",
        "peer_building_count": "Other same-cycle buildings available at the checkpoint.",
    }
    return pd.DataFrame([
        {
            "field": column,
            "dtype": str(frame[column].dtype),
            "missing_rows": int(frame[column].isna().sum()),
            "missing_pct": float(frame[column].isna().mean() * 100),
            "description": descriptions.get(column, "As-of feature reconstructed from the authoritative workbook."),
        }
        for column in frame.columns
    ])


def _quality_extensions(
    dataset: CanaryDataset,
    frames: dict[str, pd.DataFrame],
    audit_cycle: str = AUDIT_CYCLE,
) -> pd.DataFrame:
    measured = dataset.daily.loc[dataset.daily["weight_measured"].fillna(False)].copy()
    measured["weight_change_kg"] = measured.sort_values("age_day").groupby(["cycle_id", "building_id"])["bodyweight_kg"].diff()
    rows = [
        ("Every snapshot is as-of safe", sum(int((frame["max_source_day_used"] > frame["review_day"]).sum()) for frame in frames.values()), "critical"),
        (f"{audit_cycle} is labelled only as locked audit", sum(int(((frame["cycle_id"] == audit_cycle) & (frame["role"] != "later_cycle_audit")).sum()) for frame in frames.values()), "critical"),
        (f"Development snapshots exclude {audit_cycle}", sum(int(((frame["role"] == "development") & (frame["cycle_id"] == audit_cycle)).sum()) for frame in frames.values()), "critical"),
        ("Observed weight changes are non-negative", int((measured["weight_change_kg"] < 0).sum()), "warning"),
        ("Observed interval gain is at most 120 g/day", int((measured["weight_change_kg"] > 0.120 * measured.groupby(["cycle_id", "building_id"])["age_day"].diff()).sum()), "warning"),
        ("Peer context excludes the focal building", sum(int((frame["peer_building_count"] < 1).sum()) for frame in frames.values()), "warning"),
    ]
    return pd.DataFrame([{"check": check, "failed_rows": failed, "severity": severity, "status": "pass" if failed == 0 else "flagged"} for check, failed, severity in rows])


def _benchmark_context(root: Path, outcome: str) -> dict[str, Any]:
    frozen = root / "outputs" / "farmwide_modeling_rebuild"
    if outcome == "recovery":
        previous = pd.read_csv(frozen / "recovery" / "candidate_comparison.csv")
        previous = previous.rename(columns={"rank_by_cycle_macro_rmse": "rank"})
        current_manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
        current = current_manifest["outcomes"]["recovery"]["published_canary_benchmark"]
        return {
            "previous_top": previous.head(5).to_dict("records"),
            "previous_best": previous.iloc[0].to_dict(),
            "current_canary_version": current["version"],
            "current_canary_cv_rmse": float(current["rmse"]),
            "current_canary_audit_rmse": float(current["later_cycle_rmse"]),
        }
    previous = pd.read_csv(frozen / "bodyweight" / "model_comparison.csv").rename(columns={"cycle_macro_rmse_g": "cycle_macro_rmse", "rmse_g": "rmse", "mae_g": "mae", "bias_g": "bias", "worst_cycle_rmse_g": "worst_cycle_rmse"})
    current_manifest = json.loads((root / "models" / "day35_weight_manifest.json").read_text(encoding="utf-8"))
    current = current_manifest["selected_metrics"]
    audit = current_manifest["prospective_latest_cycle_audit"]["metrics"]
    return {
        "previous_top": previous.head(5).to_dict("records"),
        "previous_best": previous.iloc[0].to_dict(),
        "current_canary_version": current_manifest["model_version"],
        "current_canary_cv_rmse": float(current["rmse_kg"] * 1000),
        "current_canary_audit_rmse": float(audit["rmse_kg"] * 1000),
    }


def _comparison_to_previous(comparison: pd.DataFrame, benchmark: dict[str, Any], outcome: str) -> pd.DataFrame:
    prior = pd.DataFrame(benchmark["previous_top"]).copy()
    prior["round"] = "frozen_farmwide_rebuild"
    current = comparison.head(5).copy()
    current["round"] = "optimization_round"
    keep = ["round", "candidate", "cycle_macro_rmse", "rmse", "mae", "r2", "bias", "worst_cycle_rmse"]
    return pd.concat([current.reindex(columns=keep), prior.reindex(columns=keep)], ignore_index=True)


def _score_audit(development: pd.DataFrame, audit: pd.DataFrame, candidate: OptimizationCandidate, output: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters = tune_candidate(development, candidate, seed)
    bundle = fit_candidate(development, candidate, parameters, seed)
    prediction = predict_candidate(bundle, audit, candidate)
    rows = audit[["cycle_id", "building_id", "review_day", "as_of_date", "actual"]].copy()
    rows["predicted"] = prediction
    rows["error"] = rows["predicted"] - rows["actual"]
    rows.to_csv(output / "later_cycle_audit_predictions.csv", index=False)
    metrics = summarize(rows, candidate.outcome, bootstrap=False)
    return metrics, {"parameters": parameters, "bundle": bundle}


def _artifact(output: Path, name: str, candidate: OptimizationCandidate, fitted: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    path = output / name
    payload = {
        "round_version": ROUND_VERSION,
        "deployment_status": "research_shadow",
        "candidate": asdict(candidate),
        "fitted": fitted["bundle"],
        "parameters": fitted["parameters"],
        "feature_schema": {str(day): _features(candidate, int(day) if candidate.checkpoint_specific else None) for day in CHECKPOINTS},
    }
    joblib.dump(payload, path)
    reloaded = joblib.load(path)
    original = predict_candidate(payload["fitted"], frame, candidate)
    restored = predict_candidate(reloaded["fitted"], frame, candidate)
    if not np.allclose(original, restored, rtol=0, atol=1e-10):
        raise AssertionError(f"Reloaded artifact changed predictions: {path}")
    return {"path": str(path), "sha256": _sha(path), "prediction_parity_max_abs": float(np.max(np.abs(original - restored)))}


def _write_outcome_report(output: Path, outcome: str, manifest: dict[str, Any], comparison: pd.DataFrame, shap_global: pd.DataFrame, importance: pd.DataFrame) -> Path:
    unit = "percentage points" if outcome == "recovery" else "grams"
    top = comparison.head(5)
    table_columns = ["candidate", "cycle_macro_rmse", "rmse", "mae", "r2", "bias", "worst_cycle_rmse"]
    table = "| " + " | ".join(table_columns) + " |\n| " + " | ".join(["---"] * len(table_columns)) + " |\n"
    for row in top[table_columns].itertuples(index=False, name=None):
        table += "| " + " | ".join(str(value) if isinstance(value, str) else f"{float(value):.3f}" for value in row) + " |\n"
    drivers = shap_global.head(8) if not shap_global.empty else importance.head(8)
    driver_field = "mean_abs_shap" if not shap_global.empty else "mean_rmse_increase"
    driver_text = ", ".join(f"{row.feature} ({getattr(row, driver_field):.3f})" for row in drivers.itertuples()) or "No compatible learned finalist was selected for SHAP."
    selection = manifest["selection"]
    gate = manifest["promotion_gate"]
    text = f"""# Project Canary {outcome.title()} Optimization Round

## Technical summary

The lowest-error model was **{selection['lowest_error_candidate']}** at **{selection['lowest_cycle_macro_rmse']:.3f} {unit}** cycle-macro RMSE. The one-standard-error research selection was **{selection['selected_candidate']}**. Operational models were not replaced; the retrospective promotion gate was **{'passed' if gate['retrospective_gate_passed'] else 'not passed'}**, and three prospective cycles remain required.

## Top five nested cycle-LOGO results

{table}

These are held-out predictions for complete unseen harvest cycles. R² is descriptive secondary evidence; candidate selection uses cycle-macro RMSE and stability.

![Model comparison](figures/model_comparison.png)

## Predictive drivers, not causes

The leading held-out explanation signals were: {driver_text}. Permutation importance and SHAP measure predictive association. They do not prove that changing a feature will change the outcome.

![Held-out SHAP](figures/shap_importance.png)

## Scope and definitions

The authoritative workbook contains 1,624 building-days and 34 building-cycles. Development evaluation uses 31 building-cycles across six complete cycles at Days 7, 14, 21, and 28. The three 2026-3 buildings were locked until design and tuning were frozen. Recovery is final recorded population divided by beginning population; bodyweight is actually observed average Day 35 weight.

## Validation and model specification

The primary design is nested Leave-One-Group-Out cross-validation with harvest cycle as the group. Each outer fold holds out a complete cycle; inner cycle folds learn imputation, clipping, feature filtering, expected paths, hyperparameters, and blend weights. Building-label LOGO, building-cycle LOGO, expanding-window temporal tests, five-seed stability, feature ablation, and grouped conformal intervals are secondary checks.

## Limitations and uncertainty

Only six development cycles are available, environmental coverage is incomplete, bodyweight target hits are rare, and the 2026-3 recovery endpoint is provisional. Feature engineering cannot manufacture independent cycles. Wide bootstrap intervals, unstable SHAP direction, or negative unseen-cycle R² should be treated as evidence against promotion.

## Recommended next steps

Keep this result in research/shadow status, collect at least three complete prospective cycles, standardize weight sampling and harvest endpoint recording, and improve environmental completeness. Promote only if the prespecified gates remain satisfied prospectively.

## Further questions

The next evidence should test whether peer-relative context remains available reliably in live operation and whether its benefit persists after new cycles add genuine temporal diversity.
"""
    path = output / f"PROJECT_CANARY_{outcome.upper()}_OPTIMIZATION_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def _run_outcome(
    dataset: CanaryDataset,
    root: Path,
    output_root: Path,
    outcome: str,
    seed: int,
    candidates: tuple[OptimizationCandidate, ...],
    development_cycles: tuple[str, ...],
    audit_cycle: str,
) -> dict[str, Any]:
    output = output_root / ("recovery" if outcome == "recovery" else "bodyweight")
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    snapshots = build_optimization_snapshots(dataset, outcome, development_cycles, audit_cycle)
    development = snapshots.loc[snapshots["role"].eq("development")].reset_index(drop=True)
    audit = snapshots.loc[snapshots["role"].eq("later_cycle_audit")].reset_index(drop=True)
    if audit_cycle in set(development["cycle_id"].astype(str)):
        raise AssertionError("Locked audit cycle entered development data.")
    development.to_csv(output / "model_ready_snapshots.csv", index=False)
    audit.to_csv(output / "locked_later_cycle_snapshots.csv", index=False)
    _data_dictionary(snapshots).to_csv(output / "data_dictionary.csv", index=False)
    registry = pd.DataFrame([{**asdict(candidate), "grid": json.dumps(candidate_grid(candidate), default=_json_default), "grid_size": len(candidate_grid(candidate))} for candidate in candidates])
    registry.to_csv(output / "experiment_registry.csv", index=False)
    (output / "predefined_grids.json").write_text(json.dumps({candidate.name: candidate_grid(candidate) for candidate in candidates}, indent=2, default=_json_default), encoding="utf-8")

    primary_prediction_path = output / "all_nested_logo_predictions.csv"
    comparison_path = output / "candidate_comparison.csv"
    hyperparameter_path = output / "nested_hyperparameters.json"
    if primary_prediction_path.exists() and comparison_path.exists() and hyperparameter_path.exists():
        print(f"[{outcome}] reusing completed deterministic primary LOGO search", flush=True)
        oof = pd.read_csv(primary_prediction_path)
        comparison = pd.read_csv(comparison_path)
    else:
        all_predictions, comparison_rows, hyperparameters = [], [], {}
        for position, candidate in enumerate(candidates, start=1):
            print(f"[{outcome}] primary nested cycle LOGO {position}/{len(candidates)}: {candidate.name}", flush=True)
            predictions, parameters = evaluate_logo(development, candidate, "cycle", seed)
            all_predictions.append(predictions)
            comparison_rows.append({"candidate": candidate.name, "family": candidate.family, "target_form": candidate.target_form, "feature_set": candidate.feature_set, "checkpoint_specific": candidate.checkpoint_specific, "complexity": candidate.complexity, "description": candidate.description, **summarize(predictions, outcome)})
            hyperparameters[candidate.name] = parameters
        oof = pd.concat(all_predictions, ignore_index=True)
        comparison = pd.DataFrame(comparison_rows).sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
        comparison["rank"] = np.arange(1, len(comparison) + 1)
        oof.to_csv(primary_prediction_path, index=False)
        comparison.to_csv(comparison_path, index=False)
        hyperparameter_path.write_text(json.dumps(hyperparameters, indent=2, default=_json_default), encoding="utf-8")
    comparison.head(5).to_csv(output / "top_five_models.csv", index=False)

    selected_name, selection = select_one_se(comparison)
    lowest_name = selection["lowest_error_candidate"]
    baseline_pool = ["current_survival", "age_band_remaining_loss"] if outcome == "recovery" else ["historical_remaining_gain", "target_gap_preserving"]
    baseline_name = str(comparison.loc[comparison["candidate"].isin(baseline_pool)].iloc[0]["candidate"])
    selected = _lookup(candidates, selected_name)
    lowest = _lookup(candidates, lowest_name)
    stochastic = comparison.loc[comparison["family"].isin(TREE_FAMILIES)].iloc[0]
    stochastic_candidate = _lookup(candidates, str(stochastic["candidate"]))

    checkpoint = pd.concat([checkpoint_metrics(group, outcome).assign(candidate=name) for name, group in oof.groupby("candidate")], ignore_index=True)
    checkpoint.to_csv(output / "checkpoint_metrics.csv", index=False)
    secondary_predictions, secondary_metrics = [], []
    finalist_names = list(dict.fromkeys([selected_name, lowest_name, baseline_name]))
    for view in ("building_label", "building_cycle"):
        for name in finalist_names:
            candidate = _lookup(candidates, name)
            print(f"[{outcome}] secondary {view}: {name}", flush=True)
            prediction, _ = evaluate_logo(development, candidate, view, seed)
            secondary_predictions.append(prediction)
            secondary_metrics.append({"candidate": name, "validation_view": view, **summarize(prediction, outcome, bootstrap=False)})
    pd.concat(secondary_predictions, ignore_index=True).to_csv(output / "secondary_logo_predictions.csv", index=False)
    pd.DataFrame(secondary_metrics).to_csv(output / "secondary_logo_metrics.csv", index=False)

    temporal_predictions, temporal_rows = [], []
    for name in finalist_names:
        prediction, metrics = temporal_stress(development, _lookup(candidates, name), seed)
        temporal_predictions.append(prediction)
        temporal_rows.append({"candidate": name, **metrics})
    pd.concat(temporal_predictions, ignore_index=True).to_csv(output / "temporal_predictions.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(output / "temporal_metrics.csv", index=False)

    intervals, calibration = conformal_logo(development, lowest, seed)
    intervals.to_csv(output / "conformal_predictions.csv", index=False)
    calibration.to_csv(output / "conformal_calibration.csv", index=False)
    interval_metrics = {"coverage_80": float(intervals["covered_80"].mean()), "coverage_90": float(intervals["covered_90"].mean()), "mean_width_80": float((intervals["upper_80"] - intervals["lower_80"]).mean()), "mean_width_90": float((intervals["upper_90"] - intervals["lower_90"]).mean())}

    importance_frames = []
    for name in list(dict.fromkeys([selected_name, lowest_name])):
        candidate = _lookup(candidates, name)
        if candidate.family not in {"persistence", "baseline", "target_curve", "target_ratio", "historical_ratio", "recent_adg"}:
            importance_frames.append(permutation_importance_logo(development, candidate, seed))
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    importance.to_csv(output / "held_out_permutation_importance.csv", index=False)
    shap_global, shap_local = held_out_shap(development, stochastic_candidate, seed)
    shap_global.insert(0, "candidate", stochastic_candidate.name) if not shap_global.empty else None
    shap_global.to_csv(output / "held_out_shap_global.csv", index=False)
    shap_local.to_csv(output / "held_out_shap_local.csv", index=False)
    transparent_rows = comparison.loc[comparison["family"].isin(LINEAR_FAMILIES)]
    coefficients = standardized_coefficients(development, _lookup(candidates, str(transparent_rows.iloc[0]["candidate"])), seed) if not transparent_rows.empty else pd.DataFrame()
    coefficients.to_csv(output / "standardized_coefficients.csv", index=False)

    seed_predictions, seed_metrics, seed_summary = seed_stability(development, stochastic_candidate)
    seed_predictions.to_csv(output / "five_seed_predictions.csv", index=False)
    seed_metrics.to_csv(output / "five_seed_metrics.csv", index=False)
    seed_summary.to_csv(output / "five_seed_summary.csv", index=False)

    ablation_names = [name for name in ("residual_hist_gradient_peer", "residual_hist_gradient_core", "residual_hist_gradient_no_environment") if name in set(comparison["candidate"])]
    ablation = comparison.loc[comparison["candidate"].isin(ablation_names), ["candidate", "cycle_macro_rmse", "rmse", "mae", "r2", "bias"]].rename(columns={"candidate": "experiment"})
    ablation.to_csv(output / "feature_ablation.csv", index=False)
    peer_context = ablation.loc[ablation["experiment"].isin(["residual_hist_gradient_peer", "residual_hist_gradient_core"])].copy()
    peer_context.to_csv(output / "peer_context_comparison.csv", index=False)

    audit_metrics, lowest_fit = _score_audit(development, audit, lowest, output, seed)
    selected_parameters = tune_candidate(development, selected, seed)
    selected_fit = {"parameters": selected_parameters, "bundle": fit_candidate(development, selected, selected_parameters, seed)}
    artifacts = {
        "one_se_research_selection": _artifact(output, "one_se_research_selection.joblib", selected, selected_fit, development),
        "lowest_error_shadow": _artifact(output, "lowest_error_shadow.joblib", lowest, lowest_fit, development),
    }
    benchmark = _benchmark_context(root, outcome)
    _comparison_to_previous(comparison, benchmark, outcome).to_csv(output / "comparison_to_frozen_round.csv", index=False)
    gate = promotion_gate(comparison, oof, outcome, lowest_name, baseline_name, checkpoint, intervals, audit_metrics, benchmark["current_canary_audit_rmse"])
    gate["current_canary_version"] = benchmark["current_canary_version"]
    gate["current_canary_cv_rmse"] = benchmark["current_canary_cv_rmse"]
    gate["at_least_10pct_better_than_current_canary"] = bool(float(comparison.set_index("candidate").loc[lowest_name, "cycle_macro_rmse"]) <= benchmark["current_canary_cv_rmse"] * 0.90)
    gate["retrospective_gate_passed"] = bool(gate["retrospective_gate_passed"] and gate["at_least_10pct_better_than_current_canary"])

    manifest = {
        "round_version": ROUND_VERSION,
        "outcome": outcome,
        "seed": seed,
        "design_frozen_before_audit": True,
        "primary_validation": "Nested Leave-One-Group-Out by complete harvest cycle; cycle-macro RMSE primary.",
        "development_cycles": development_cycles,
        "locked_audit_cycle": audit_cycle,
        "development_rows": len(development),
        "development_building_cycles": int(development[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "selection": selection,
        "baseline": baseline_name,
        "lowest_error_metrics": comparison.loc[comparison["candidate"].eq(lowest_name)].iloc[0].to_dict(),
        "later_cycle_audit_metrics": audit_metrics,
        "interval_metrics": interval_metrics,
        "explanation_model": stochastic_candidate.name,
        "shap_warning": "SHAP and importance describe predictive association, not causation; unstable directions are flagged.",
        "benchmark": benchmark,
        "promotion_gate": gate,
        "artifacts": artifacts,
        "operational_models_changed": False,
    }
    _plot_outputs(figures, outcome, oof, comparison, checkpoint, intervals, importance, shap_global, ablation, seed_metrics, development, pd.concat(temporal_predictions, ignore_index=True))
    report = _write_outcome_report(output, outcome, manifest, comparison, shap_global, importance)
    manifest["technical_report"] = str(report)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest


def _cycle_key(value: str) -> tuple[int, int, str]:
    try:
        year, cycle = str(value).split("-", 1)
        return int(year), int(cycle), str(value)
    except (TypeError, ValueError):
        return 0, 0, str(value)


def _profile_candidates(profile: str) -> tuple[tuple[OptimizationCandidate, ...], tuple[OptimizationCandidate, ...]]:
    if profile == "full":
        return RECOVERY_CANDIDATES, WEIGHT_CANDIDATES
    if profile == "balanced":
        return (
            tuple(candidate for candidate in RECOVERY_CANDIDATES if candidate.name in BALANCED_RECOVERY_NAMES),
            tuple(candidate for candidate in WEIGHT_CANDIDATES if candidate.name in BALANCED_WEIGHT_NAMES),
        )
    if profile == "smoke":
        return (
            tuple(candidate for candidate in RECOVERY_CANDIDATES if candidate.name in {"current_survival", "age_band_remaining_loss", "remaining_ridge_peer"}),
            tuple(candidate for candidate in WEIGHT_CANDIDATES if candidate.name in {"historical_remaining_gain", "target_curve_ratio", "direct_trajectory_pls"}),
        )
    raise ValueError("profile must be 'balanced', 'full', or 'smoke'")


def run_optimization_round(
    workbook: str | Path,
    output: str | Path | None = None,
    *,
    seed: int = SEED,
    profile: str = "full",
    audit_cycle: str = "latest",
) -> dict[str, Any]:
    """Execute and freeze the research-only optimization round."""
    if seed != SEED:
        raise ValueError(f"The predefined round uses seed {SEED}; received {seed}.")
    root = Path(__file__).resolve().parents[1]
    workbook_path = Path(workbook).resolve()
    output_root = Path(output).resolve() if output else root / "outputs" / "farmwide_modeling_optimization_round"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(workbook_path)
    available_cycles = tuple(sorted(dataset.cycles["cycle_id"].astype(str).unique(), key=_cycle_key))
    resolved_audit_cycle = available_cycles[-1] if audit_cycle == "latest" else str(audit_cycle)
    if resolved_audit_cycle not in available_cycles:
        raise ValueError(f"Audit cycle {resolved_audit_cycle!r} is not present in the workbook")
    development_cycles = tuple(cycle for cycle in available_cycles if cycle != resolved_audit_cycle)
    run_profile = profile
    recovery_candidates, weight_candidates = _profile_candidates(run_profile)
    source_profile, checks = build_source_quality_audit(workbook_path, dataset)
    preview_frames = {
        outcome: build_optimization_snapshots(dataset, outcome, development_cycles, resolved_audit_cycle)
        for outcome in ("recovery", "weight")
    }
    all_checks = pd.concat([checks, _quality_extensions(dataset, preview_frames, resolved_audit_cycle)], ignore_index=True)
    all_checks.to_csv(output_root / "data_quality_checks.csv", index=False)
    (output_root / "source_audit.json").write_text(json.dumps(source_profile, indent=2, default=_json_default), encoding="utf-8")
    if all_checks.loc[all_checks["severity"].eq("critical"), "failed_rows"].sum() > 0:
        raise AssertionError("Critical data-quality checks failed.")
    registry = pd.DataFrame([asdict(candidate) for candidate in (*recovery_candidates, *weight_candidates)])
    registry.to_csv(output_root / "complete_experiment_registry.csv", index=False)
    recovery_manifest_path = output_root / "recovery" / "manifest.json"
    if recovery_manifest_path.exists():
        recovery = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
        print("[recovery] reusing completed frozen outcome manifest", flush=True)
    else:
        recovery = _run_outcome(dataset, root, output_root, "recovery", seed, recovery_candidates, development_cycles, resolved_audit_cycle)
    weight_manifest_path = output_root / "bodyweight" / "manifest.json"
    if weight_manifest_path.exists():
        weight = json.loads(weight_manifest_path.read_text(encoding="utf-8"))
        print("[weight] reusing completed frozen outcome manifest", flush=True)
    else:
        weight = _run_outcome(dataset, root, output_root, "weight", seed, weight_candidates, development_cycles, resolved_audit_cycle)
    combined = pd.concat([
        pd.read_csv(output_root / "recovery" / "top_five_models.csv").assign(outcome="recovery"),
        pd.read_csv(output_root / "bodyweight" / "top_five_models.csv").assign(outcome="weight"),
    ], ignore_index=True, sort=False)
    combined.to_csv(output_root / "top_five_models_by_outcome.csv", index=False)
    manifest = {
        "round_version": ROUND_VERSION,
        "created": pd.Timestamp.now(tz="Asia/Manila").isoformat(),
        "source": source_profile,
        "source_sha256": _sha(workbook_path),
        "run_profile": run_profile,
        "development_cycles": list(development_cycles),
        "locked_audit_cycle": resolved_audit_cycle,
        "package_versions": _package_versions(),
        "validation_views": {
            "primary": "nested harvest-cycle LOGO",
            "building_robustness": "building-label LOGO",
            "optimistic_secondary": "building-cycle LOGO",
            "temporal": "expanding-window by harvest cycle",
        },
        "outcomes": {"recovery": recovery, "weight": weight},
        "operational_models_changed": False,
        "operational_promotion_requires_prospective_cycles": 3,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest
