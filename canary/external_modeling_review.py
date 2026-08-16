"""Independent, leakage-safe modeling review for Project Canary.

This module deliberately lives outside the application model-training path.  It
rebuilds review-date snapshots from the corrected farm workbook, compares a
small set of defensible models with complete-cycle holdouts, and writes all
evidence to ``outputs/external_modeling_review``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    Ridge,
    TweedieRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from .data import CanaryDataset, load_workbook
from .farmwide_features import assert_primary_schema_has_no_identity


SEED = 20260812
DEVELOPMENT_CYCLES = ("2025-2", "2025-3", "2025-4", "2025-5", "2026-1", "2026-2")
AUDIT_CYCLE = "2026-3"
CHECKPOINTS = (7, 14, 21, 28)
RECOVERY_TARGET = 0.95
WEIGHT_TARGET_KG = 1.8

RECOVERY_FEATURES = [
    "review_day", "days_to_day35", "beginning_inventory", "log_beginning_inventory",
    "percentage_alive", "population_loss_pct", "population_loss_rate_pp_day",
    "mortality_daily_per_1000", "mortality_recent_3d_per_1000",
    "mortality_recent_7d_per_1000", "mortality_ewma_per_1000",
    "mortality_trend_per_1000", "mortality_acceleration_per_1000",
    "mortality_volatility_per_1000", "mortality_max_per_1000",
    "mortality_spike_days", "mortality_recent_vs_early_per_1000",
    "population_mortality_reconciliation_gap_per_1000",
    "population_increase_days", "record_completeness_ratio",
    "latest_weight_kg", "weight_target_kg", "weight_ratio_to_target",
    "weight_gap_pct", "weight_measurement_day", "weight_staleness_days",
    "weight_measurement_count", "weight_measurement_interval_days",
    "has_weight_measurement", "temperature_recent_avg_c",
    "temperature_recent_range_c", "temperature_history_avg_c",
    "temperature_history_sd_c", "temperature_heat_excess_degree_days",
    "temperature_cold_excess_degree_days", "temperature_out_of_band_days",
    "temperature_zone_spread_mean_c", "temperature_missing",
    "humidity_recent_avg_pct", "humidity_recent_range_pp",
    "humidity_history_avg_pct", "humidity_history_sd_pct",
    "humidity_high_excess_days", "humidity_low_excess_days",
    "humidity_out_of_band_days", "humidity_zone_spread_mean_pct",
    "humidity_missing", "environment_recorded_days",
    "environment_coverage_ratio", "environment_staleness_days",
]

WEIGHT_FEATURES = [
    "review_day", "days_to_day35", "current_weight_kg", "weight_target_kg",
    "weight_ratio_to_target", "weight_gap_pct", "prior_weight_kg",
    "prior_weight_day", "recent_adg_kg_day", "cumulative_adg_kg_day",
    "percentage_alive", "population_loss_pct", "mortality_recent_3d_per_1000",
    "mortality_recent_7d_per_1000", "temperature_recent_avg_c",
    "temperature_history_avg_c", "temperature_missing",
    "humidity_recent_avg_pct", "humidity_history_avg_pct", "humidity_missing",
    "environment_recorded_days",
]

RECOVERY_COMPACT_FEATURES = [
    "review_day", "percentage_alive", "population_loss_pct",
    "mortality_recent_3d_per_1000", "mortality_recent_7d_per_1000",
    "mortality_ewma_per_1000", "mortality_trend_per_1000",
    "mortality_volatility_per_1000",
    "latest_weight_kg", "weight_ratio_to_target", "weight_staleness_days",
    "temperature_heat_excess_degree_days", "temperature_cold_excess_degree_days",
    "temperature_missing", "humidity_out_of_band_days", "humidity_missing",
    "environment_coverage_ratio", "record_completeness_ratio",
]

WEIGHT_COMPACT_FEATURES = [
    "review_day", "current_weight_kg", "weight_ratio_to_target",
    "recent_adg_kg_day", "cumulative_adg_kg_day", "percentage_alive",
    "mortality_recent_3d_per_1000", "temperature_recent_avg_c",
    "temperature_missing", "humidity_recent_avg_pct", "humidity_missing",
]

FEED_SENSITIVITY_FEATURES = ["feed_daily_per_1000", "feed_recent_7d_per_1000", "feed_missing"]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    target_form: str
    complexity: int
    description: str


RECOVERY_CANDIDATES = (
    Candidate("current_survival", "persistence", "direct", 0, "Current survival persists to endpoint"),
    Candidate("age_band_remaining_loss", "age_baseline", "remaining", 1, "Training-only mean remaining loss by checkpoint"),
    Candidate("remaining_ols", "linear", "remaining", 2, "Ordinary linear remaining-loss model"),
    Candidate("direct_ridge", "ridge", "direct", 3, "Regularized direct final recovery"),
    Candidate("remaining_ridge", "ridge", "remaining", 3, "Regularized remaining-loss regression"),
    Candidate("remaining_elastic_net", "elastic_net", "remaining", 4, "Sparse regularized remaining-loss regression"),
    Candidate("remaining_huber", "huber", "remaining", 4, "Robust remaining-loss regression"),
    Candidate("remaining_tweedie", "tweedie", "remaining", 4, "Positive remaining-loss rate regression with a log link"),
    Candidate("direct_spline_ridge", "spline", "direct", 5, "Regularized nonlinear spline model"),
    Candidate("direct_gradient_boosting", "gradient_boosting", "direct", 6, "Constrained gradient boosting on final recovery"),
    Candidate("remaining_gradient_boosting", "gradient_boosting", "remaining", 6, "Constrained gradient boosting on remaining loss"),
    Candidate("remaining_compact_extra_trees", "compact_extra_trees", "remaining", 6, "Compact constrained Extra Trees on remaining loss"),
    Candidate("direct_random_forest", "random_forest", "direct", 7, "Constrained random forest on final recovery"),
    Candidate("remaining_extra_trees", "extra_trees", "remaining", 7, "Constrained Extra Trees on remaining loss"),
    Candidate("remaining_hist_gradient_boosting", "hist_gradient_boosting", "remaining", 7, "Regularized histogram boosting on remaining loss"),
    Candidate("remaining_xgboost", "xgboost", "remaining", 8, "Regularized XGBoost on remaining loss"),
    Candidate("remaining_lightgbm", "lightgbm", "remaining", 8, "Regularized LightGBM on remaining loss"),
    Candidate("remaining_catboost", "catboost", "remaining", 8, "Regularized CatBoost on remaining loss"),
)

WEIGHT_CANDIDATES = (
    Candidate("historical_remaining_gain", "age_baseline", "remaining", 0, "Training-only mean remaining gain by checkpoint"),
    Candidate("target_curve_pace", "pace", "direct", 1, "Current target-relative pace carried to Day 35"),
    Candidate("remaining_ols", "linear", "remaining", 2, "Ordinary linear remaining-gain model"),
    Candidate("direct_ridge", "ridge", "direct", 3, "Regularized direct Day 35 regression"),
    Candidate("remaining_ridge", "ridge", "remaining", 3, "Regularized remaining-gain regression"),
    Candidate("remaining_elastic_net", "elastic_net", "remaining", 4, "Sparse regularized remaining-gain model"),
    Candidate("remaining_huber", "huber", "remaining", 4, "Robust remaining-gain regression"),
    Candidate("growth_curve_ridge", "growth", "log_direct", 5, "Regularized log-weight growth-curve projection"),
    Candidate("direct_gradient_boosting", "gradient_boosting", "direct", 6, "Constrained gradient boosting"),
    Candidate("remaining_gradient_boosting", "gradient_boosting", "remaining", 6, "Constrained remaining-gain gradient boosting"),
    Candidate("remaining_compact_extra_trees", "compact_extra_trees", "remaining", 6, "Compact constrained Extra Trees on remaining gain"),
    Candidate("direct_random_forest", "random_forest", "direct", 7, "Constrained random forest"),
    Candidate("direct_extra_trees", "extra_trees", "direct", 7, "Constrained Extra Trees"),
)


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


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_at(dataset: CanaryDataset, day: int) -> float:
    match = dataset.targets.loc[dataset.targets["age_day"].eq(day), "target_weight_kg"]
    return float(match.iloc[0]) if not match.empty else np.nan


def _snapshot_features(dataset: CanaryDataset, cycle_id: str, building_id: str, day: int) -> dict[str, Any] | None:
    history = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(cycle_id)
        & dataset.daily["building_id"].astype(str).eq(building_id)
        & dataset.daily["age_day"].le(day)
    ].sort_values("age_day")
    if history.empty or int(history["age_day"].max()) < day:
        return None
    current = history.iloc[-1]
    as_of = pd.Timestamp(current["record_date"])
    beginning = float(current["beginning_inventory"])
    population = float(current["population"])
    alive = population / beginning
    mortality = pd.to_numeric(history["mortality_daily"], errors="coerce")
    mortality_rate = mortality / beginning * 1000
    recent3 = mortality.tail(3).sum(min_count=1) / beginning * 1000
    recent7 = mortality.tail(7).sum(min_count=1) / beginning * 1000
    previous7 = mortality.iloc[-14:-7].sum(min_count=1) / beginning * 1000 if len(mortality) >= 8 else np.nan
    previous3 = mortality.iloc[-6:-3].sum(min_count=1) / beginning * 1000 if len(mortality) >= 4 else np.nan
    ewma = mortality_rate.ewm(span=min(7, max(2, len(mortality_rate))), adjust=False).mean()
    early_rate = mortality_rate.iloc[:-7].mean() if len(mortality_rate) > 7 else np.nan
    positive_rate = mortality_rate.dropna()
    robust_spike_threshold = (
        float(positive_rate.median() + 3.0 * (positive_rate - positive_rate.median()).abs().median())
        if not positive_rate.empty
        else np.nan
    )
    population = pd.to_numeric(history["population"], errors="coerce")
    population_change = -population.diff()
    reconciliation_gap = population_change - mortality
    measured = history.loc[history["weight_measured"].fillna(False)]
    if measured.empty:
        latest_weight = weight_day = weight_target = ratio = gap = stale = np.nan
        prior_weight = prior_day = recent_adg = cumulative_adg = np.nan
        weight_measurement_count = 0
        weight_measurement_interval = np.nan
    else:
        latest = measured.iloc[-1]
        latest_weight = float(latest["bodyweight_kg"])
        weight_day = int(latest["age_day"])
        weight_target = _target_at(dataset, weight_day)
        ratio = latest_weight / weight_target if weight_target > 0 else np.nan
        gap = (weight_target - latest_weight) / weight_target * 100 if weight_target > 0 else np.nan
        stale = day - weight_day
        first = measured.iloc[0]
        prior = measured.iloc[-2] if len(measured) > 1 else None
        prior_weight = float(prior["bodyweight_kg"]) if prior is not None else np.nan
        prior_day = int(prior["age_day"]) if prior is not None else np.nan
        recent_adg = ((latest_weight - prior_weight) / (weight_day - prior_day)) if prior is not None and weight_day > prior_day else np.nan
        cumulative_adg = ((latest_weight - float(first["bodyweight_kg"])) / (weight_day - int(first["age_day"]))) if weight_day > int(first["age_day"]) else np.nan
        weight_measurement_count = int(len(measured))
        weight_measurement_interval = (
            float(np.diff(measured["age_day"].to_numpy(float)).mean())
            if len(measured) > 1
            else np.nan
        )
    env = history.loc[history[["temperature_avg_c", "humidity_avg_pct"]].notna().any(axis=1)]
    recent_env = env.tail(3)
    temp_recent = pd.to_numeric(recent_env["temperature_avg_c"], errors="coerce")
    hum_recent = pd.to_numeric(recent_env["humidity_avg_pct"], errors="coerce")
    temperature_all = pd.to_numeric(history["temperature_avg_c"], errors="coerce")
    humidity_all = pd.to_numeric(history["humidity_avg_pct"], errors="coerce")
    age = pd.to_numeric(history["age_day"], errors="coerce")
    temperature_target = pd.Series(
        np.select([age <= 7, age <= 14, age <= 21, age <= 28], [31.0, 28.5, 25.5, 23.5], default=22.5),
        index=history.index,
    )
    temperature_lower = temperature_target - 1.5
    temperature_upper = temperature_target + 1.5
    humidity_target = pd.Series(
        np.select([age <= 7, age <= 14], [60.0, 55.0], default=50.0),
        index=history.index,
    )
    humidity_lower = humidity_target
    humidity_upper = humidity_target + 10.0
    heat_excess = (temperature_all - temperature_upper).clip(lower=0)
    cold_excess = (temperature_lower - temperature_all).clip(lower=0)
    high_humidity = (humidity_all - humidity_upper).clip(lower=0)
    low_humidity = (humidity_lower - humidity_all).clip(lower=0)
    environment_days = history[["temperature_avg_c", "humidity_avg_pct"]].notna().any(axis=1)
    last_environment_day = int(history.loc[environment_days, "age_day"].max()) if environment_days.any() else None
    feeds = pd.to_numeric(history["feed_daily_bags"], errors="coerce")
    return {
        "cycle_id": cycle_id, "building_id": building_id, "review_day": day,
        "as_of_date": as_of, "max_source_day_used": int(history["age_day"].max()),
        "days_to_day35": max(0, 35 - day), "beginning_inventory": beginning,
        "log_beginning_inventory": float(np.log(beginning)) if beginning > 0 else np.nan,
        "percentage_alive": alive, "population_loss_pct": (1 - alive) * 100,
        "population_loss_rate_pp_day": (1 - alive) * 100 / day,
        "mortality_cumulative_per_1000": float(mortality.sum(min_count=1) / beginning * 1000),
        "mortality_daily_per_1000": float(mortality.iloc[-1] / beginning * 1000) if pd.notna(mortality.iloc[-1]) else np.nan,
        "mortality_recent_3d_per_1000": float(recent3), "mortality_recent_7d_per_1000": float(recent7),
        "mortality_ewma_per_1000": float(ewma.iloc[-1]) if not ewma.empty and pd.notna(ewma.iloc[-1]) else np.nan,
        "mortality_trend_per_1000": float(recent7 - previous7) if pd.notna(previous7) else np.nan,
        "mortality_acceleration_per_1000": float(recent3 - previous3) if pd.notna(previous3) else np.nan,
        "mortality_volatility_per_1000": float(mortality_rate.std(ddof=0)) if mortality_rate.notna().any() else np.nan,
        "mortality_max_per_1000": float(mortality_rate.max()) if mortality_rate.notna().any() else np.nan,
        "mortality_spike_days": int((mortality_rate > robust_spike_threshold).sum()) if pd.notna(robust_spike_threshold) else 0,
        "mortality_recent_vs_early_per_1000": float(mortality_rate.tail(7).mean() - early_rate) if pd.notna(early_rate) else np.nan,
        "population_mortality_reconciliation_gap_per_1000": float(reconciliation_gap.abs().sum(min_count=1) / beginning * 1000),
        "population_increase_days": int((population.diff() > 0).sum()),
        "record_completeness_ratio": float(history["operational_recorded"].fillna(False).mean()),
        "latest_weight_kg": latest_weight, "current_weight_kg": latest_weight,
        "weight_target_kg": weight_target, "weight_ratio_to_target": ratio, "weight_gap_pct": gap,
        "weight_measurement_day": weight_day, "weight_staleness_days": stale,
        "has_weight_measurement": float(not measured.empty), "prior_weight_kg": prior_weight,
        "weight_measurement_count": weight_measurement_count,
        "weight_measurement_interval_days": weight_measurement_interval,
        "prior_weight_day": prior_day, "recent_adg_kg_day": recent_adg,
        "cumulative_adg_kg_day": cumulative_adg,
        "temperature_recent_avg_c": float(temp_recent.mean()) if temp_recent.notna().any() else np.nan,
        "temperature_recent_range_c": float(temp_recent.max() - temp_recent.min()) if temp_recent.notna().sum() > 1 else np.nan,
        "temperature_history_avg_c": float(pd.to_numeric(env["temperature_avg_c"], errors="coerce").mean()) if not env.empty else np.nan,
        "temperature_history_sd_c": float(temperature_all.std(ddof=0)) if temperature_all.notna().any() else np.nan,
        "temperature_heat_excess_degree_days": float(heat_excess.sum(min_count=1)),
        "temperature_cold_excess_degree_days": float(cold_excess.sum(min_count=1)),
        "temperature_out_of_band_days": int(((heat_excess > 0) | (cold_excess > 0)).sum()),
        "temperature_zone_spread_mean_c": float(pd.to_numeric(history.get("temperature_zone_spread_c"), errors="coerce").mean()),
        "temperature_missing": float(temp_recent.notna().sum() == 0),
        "humidity_recent_avg_pct": float(hum_recent.mean()) if hum_recent.notna().any() else np.nan,
        "humidity_recent_range_pp": float(hum_recent.max() - hum_recent.min()) if hum_recent.notna().sum() > 1 else np.nan,
        "humidity_history_avg_pct": float(pd.to_numeric(env["humidity_avg_pct"], errors="coerce").mean()) if not env.empty else np.nan,
        "humidity_history_sd_pct": float(humidity_all.std(ddof=0)) if humidity_all.notna().any() else np.nan,
        "humidity_high_excess_days": float(high_humidity.sum(min_count=1)),
        "humidity_low_excess_days": float(low_humidity.sum(min_count=1)),
        "humidity_out_of_band_days": int(((high_humidity > 0) | (low_humidity > 0)).sum()),
        "humidity_zone_spread_mean_pct": float(pd.to_numeric(history.get("humidity_zone_spread_pct"), errors="coerce").mean()),
        "humidity_missing": float(hum_recent.notna().sum() == 0),
        "environment_recorded_days": int(len(env)), "environment_coverage_ratio": float(len(env) / day),
        "environment_staleness_days": float(day - last_environment_day) if last_environment_day is not None else np.nan,
        "as_of_month_sin": float(np.sin(2 * np.pi * as_of.month / 12)), "as_of_month_cos": float(np.cos(2 * np.pi * as_of.month / 12)),
        "is_lags_building": float(building_id.startswith("Lags")),
        **{f"building_{name.lower().replace(' ', '_')}": float(building_id == name) for name in ("Tags 1", "Tags 2", "Tags 3", "Lags 1", "Lags 2", "Lags 3")},
        "feed_daily_per_1000": float(feeds.iloc[-1] / beginning * 1000) if pd.notna(feeds.iloc[-1]) else np.nan,
        "feed_recent_7d_per_1000": float(feeds.tail(7).sum(min_count=1) / beginning * 1000),
        "feed_missing": float(feeds.tail(7).notna().sum() == 0),
    }


def build_snapshots(
    dataset: CanaryDataset,
    outcome: str,
    development_cycles: tuple[str, ...] = DEVELOPMENT_CYCLES,
    audit_cycle: str = AUDIT_CYCLE,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in dataset.cycles.itertuples(index=False):
        cycle_id, building_id = str(record.cycle_id), str(record.building_id)
        if cycle_id not in {*development_cycles, audit_cycle}:
            continue
        unit_daily = dataset.daily.loc[
            dataset.daily["cycle_id"].astype(str).eq(cycle_id)
            & dataset.daily["building_id"].astype(str).eq(building_id)
        ]
        if outcome == "recovery":
            days = list(CHECKPOINTS)
            y = float(record.final_recovery_rate)
        else:
            observed35 = unit_daily.loc[unit_daily["age_day"].eq(35) & unit_daily["weight_measured"], "bodyweight_kg"]
            if observed35.empty:
                continue
            days, y = list(CHECKPOINTS), float(observed35.iloc[-1])
        for day in sorted(set(days)):
            features = _snapshot_features(dataset, cycle_id, building_id, day)
            if features is None:
                continue
            features["outcome_y"] = y
            features["additional_loss_y"] = max(0.0, features["percentage_alive"] - y) if outcome == "recovery" else np.nan
            features["remaining_gain_y"] = y - features["current_weight_kg"] if outcome == "weight" else np.nan
            features["role"] = "later_cycle_audit" if cycle_id == audit_cycle else "development"
            features["endpoint_warning"] = "Provisional Day 35 last-recorded population proxy" if outcome == "recovery" and cycle_id == audit_cycle else ""
            rows.append(features)
    result = pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True)
    if not result["max_source_day_used"].le(result["review_day"]).all():
        raise AssertionError("A snapshot contains post-review-date data.")
    return result


def unit_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = frame["cycle_id"].astype(str) + "::" + frame["building_id"].astype(str)
    raw = 1.0 / keys.map(keys.value_counts()).to_numpy(float)
    return raw / raw.mean()


def _feature_columns(outcome: str, include_feed: bool = False) -> list[str]:
    base = RECOVERY_FEATURES if outcome == "recovery" else WEIGHT_FEATURES
    return [*base, *FEED_SENSITIVITY_FEATURES] if include_feed else list(base)


def _pipeline(candidate: Candidate, parameters: dict[str, Any]) -> Pipeline:
    if candidate.family == "linear":
        model: Any = LinearRegression()
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("model", model)]
    elif candidate.family == "ridge":
        model: Any = Ridge(alpha=float(parameters.get("alpha", 10.0)))
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("model", model)]
    elif candidate.family == "elastic_net":
        model = ElasticNet(alpha=float(parameters.get("alpha", 0.01)), l1_ratio=float(parameters.get("l1_ratio", 0.25)), max_iter=10000, random_state=SEED)
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("model", model)]
    elif candidate.family == "huber":
        model = HuberRegressor(alpha=float(parameters.get("alpha", 0.01)), epsilon=float(parameters.get("epsilon", 1.35)), max_iter=4000)
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("model", model)]
    elif candidate.family == "tweedie":
        model = TweedieRegressor(
            power=float(parameters.get("power", 1.5)),
            alpha=float(parameters.get("alpha", 10.0)),
            link="log",
            max_iter=4000,
        )
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("model", model)]
    elif candidate.family == "spline":
        model = Ridge(alpha=float(parameters.get("alpha", 10.0)))
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("spline", SplineTransformer(n_knots=int(parameters.get("knots", 3)), degree=2, include_bias=False)), ("scale", StandardScaler()), ("model", model)]
    elif candidate.family in {"extra_trees", "compact_extra_trees"}:
        model = ExtraTreesRegressor(n_estimators=400, max_depth=parameters.get("depth", 4), min_samples_leaf=int(parameters.get("leaf", 4)), max_features=parameters.get("max_features", 0.7), random_state=SEED, n_jobs=1)
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "random_forest":
        model = RandomForestRegressor(n_estimators=400, max_depth=parameters.get("depth", 4), min_samples_leaf=int(parameters.get("leaf", 4)), max_features=parameters.get("max_features", 0.7), random_state=SEED, n_jobs=1)
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=int(parameters.get("trees", 75)), learning_rate=float(parameters.get("rate", 0.04)), max_depth=int(parameters.get("depth", 1)), min_samples_leaf=int(parameters.get("leaf", 4)), loss="huber", random_state=SEED)
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=int(parameters.get("trees", 100)),
            learning_rate=float(parameters.get("rate", 0.03)),
            max_leaf_nodes=int(parameters.get("leaves", 5)),
            min_samples_leaf=int(parameters.get("leaf", 5)),
            l2_regularization=float(parameters.get("l2", 25.0)),
            random_state=SEED,
        )
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "xgboost":
        model = XGBRegressor(
            n_estimators=int(parameters.get("trees", 120)), learning_rate=float(parameters.get("rate", 0.025)),
            max_depth=int(parameters.get("depth", 2)), min_child_weight=float(parameters.get("leaf", 5)),
            reg_lambda=float(parameters.get("l2", 30.0)), reg_alpha=1.0, subsample=0.8,
            colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
            random_state=SEED, n_jobs=1, verbosity=0,
        )
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "lightgbm":
        depth = int(parameters.get("depth", 2))
        model = LGBMRegressor(
            n_estimators=int(parameters.get("trees", 120)), learning_rate=float(parameters.get("rate", 0.025)),
            max_depth=depth, num_leaves=max(2, 2**depth - 1), min_child_samples=int(parameters.get("leaf", 5)),
            reg_lambda=float(parameters.get("l2", 30.0)), reg_alpha=1.0, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=1, deterministic=True,
            force_col_wise=True, verbosity=-1,
        )
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "catboost":
        model = CatBoostRegressor(
            iterations=int(parameters.get("trees", 120)), learning_rate=float(parameters.get("rate", 0.025)),
            depth=int(parameters.get("depth", 2)), l2_leaf_reg=float(parameters.get("l2", 30.0)),
            random_seed=SEED, loss_function="RMSE", random_strength=0.5,
            allow_writing_files=False, thread_count=1, verbose=False,
        )
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)]
    elif candidate.family == "growth":
        model = TransformedTargetRegressor(regressor=Ridge(alpha=float(parameters.get("alpha", 10.0))), func=np.log, inverse_func=np.exp, check_inverse=False)
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler()), ("model", model)]
    else:
        raise ValueError(candidate.family)
    return Pipeline(steps)


def _options(candidate: Candidate) -> list[dict[str, Any]]:
    if candidate.family in {"ridge", "growth"}:
        return [{"alpha": value} for value in (1.0, 10.0, 50.0)]
    if candidate.family == "huber":
        return [{"alpha": a, "epsilon": e} for a, e in ((0.001, 1.2), (0.01, 1.35), (0.1, 1.5))]
    if candidate.family == "tweedie":
        return [{"alpha": alpha, "power": power} for alpha, power in ((1.0, 1.1), (10.0, 1.5), (50.0, 1.9))]
    if candidate.family == "elastic_net":
        return [{"alpha": a, "l1_ratio": ratio} for a, ratio in ((0.001, 0.1), (0.01, 0.25), (0.05, 0.5), (0.1, 0.75))]
    if candidate.family == "spline":
        return [{"alpha": a, "knots": k} for a, k in ((10.0, 3), (50.0, 3), (50.0, 4))]
    if candidate.family in {"extra_trees", "compact_extra_trees", "random_forest"}:
        return [{"depth": d, "leaf": l, "max_features": m} for d, l, m in ((3, 3, 0.7), (4, 4, 0.7), (5, 5, 1.0))]
    if candidate.family == "gradient_boosting":
        return [{"trees": trees, "rate": rate, "depth": depth, "leaf": leaf} for trees, rate, depth, leaf in ((50, 0.03, 1, 4), (75, 0.04, 1, 4), (75, 0.03, 2, 5), (100, 0.03, 1, 5))]
    if candidate.family == "hist_gradient_boosting":
        return [
            {"trees": 75, "rate": 0.04, "leaves": 5, "leaf": 5, "l2": 10.0},
            {"trees": 100, "rate": 0.03, "leaves": 5, "leaf": 8, "l2": 25.0},
            {"trees": 150, "rate": 0.02, "leaves": 8, "leaf": 8, "l2": 50.0},
        ]
    if candidate.family in {"xgboost", "lightgbm", "catboost"}:
        return [
            {"trees": 75, "rate": 0.03, "depth": 1, "leaf": 5, "l2": 20.0},
            {"trees": 120, "rate": 0.025, "depth": 2, "leaf": 5, "l2": 30.0},
            {"trees": 150, "rate": 0.02, "depth": 2, "leaf": 8, "l2": 50.0},
        ]
    return [{}]


def _raw_target(frame: pd.DataFrame, outcome: str, form: str) -> np.ndarray:
    if form == "remaining":
        return frame["additional_loss_y" if outcome == "recovery" else "remaining_gain_y"].to_numpy(float)
    return frame["outcome_y"].to_numpy(float)


def _to_outcome(frame: pd.DataFrame, raw: np.ndarray, outcome: str, form: str) -> np.ndarray:
    pred = np.asarray(raw, dtype=float)
    if form == "remaining":
        pred = frame["percentage_alive"].to_numpy(float) - np.maximum(pred, 0) if outcome == "recovery" else frame["current_weight_kg"].to_numpy(float) + pred
    return np.clip(pred, 0.0, 1.0) if outcome == "recovery" else np.clip(pred, 0.05, 3.5)


def _fit_baseline(train: pd.DataFrame, candidate: Candidate, outcome: str) -> dict[str, Any]:
    if candidate.family == "age_baseline":
        target = _raw_target(train, outcome, "remaining")
        mapping = pd.DataFrame({"day": train["review_day"], "target": target}).groupby("day")["target"].mean().to_dict()
        return {"family": candidate.family, "mapping": mapping, "fallback": float(np.mean(target))}
    return {"family": candidate.family}


def _predict_baseline(fitted: dict[str, Any], frame: pd.DataFrame, candidate: Candidate, outcome: str) -> np.ndarray:
    if candidate.family == "persistence":
        return frame["percentage_alive"].to_numpy(float)
    if candidate.family == "pace":
        return np.clip(frame["weight_ratio_to_target"].to_numpy(float) * WEIGHT_TARGET_KG, 0.05, 3.5)
    mapping, fallback = fitted["mapping"], fitted["fallback"]
    raw = np.array([mapping.get(day, fallback) for day in frame["review_day"]], dtype=float)
    return _to_outcome(frame, raw, outcome, "remaining")


def _fit_model(train: pd.DataFrame, candidate: Candidate, outcome: str, parameters: dict[str, Any], include_feed: bool = False) -> dict[str, Any]:
    if candidate.family in {"persistence", "pace", "age_baseline"}:
        return _fit_baseline(train, candidate, outcome)
    if candidate.family == "compact_extra_trees":
        columns = list(RECOVERY_COMPACT_FEATURES if outcome == "recovery" else WEIGHT_COMPACT_FEATURES)
    else:
        columns = _feature_columns(outcome, include_feed)
    model = _pipeline(candidate, parameters)
    model.fit(train[columns], _raw_target(train, outcome, candidate.target_form), model__sample_weight=unit_weights(train))
    return {"family": candidate.family, "model": model, "features": columns, "parameters": parameters}


def predict_fitted(fitted: dict[str, Any], frame: pd.DataFrame, candidate: Candidate, outcome: str) -> np.ndarray:
    if candidate.family in {"persistence", "pace", "age_baseline"}:
        return _predict_baseline(fitted, frame, candidate, outcome)
    raw = fitted["model"].predict(frame[fitted["features"]])
    return _to_outcome(frame, raw, outcome, candidate.target_form)


def _tune(train: pd.DataFrame, candidate: Candidate, outcome: str) -> dict[str, Any]:
    options = _options(candidate)
    if len(options) == 1:
        return options[0]
    groups = train["cycle_id"].astype(str).to_numpy()
    scored: list[tuple[float, dict[str, Any]]] = []
    for parameters in options:
        fold_errors = []
        for fit_idx, valid_idx in LeaveOneGroupOut().split(train, groups=groups):
            fitted = _fit_model(train.iloc[fit_idx], candidate, outcome, parameters)
            pred = predict_fitted(fitted, train.iloc[valid_idx], candidate, outcome)
            fold_errors.append(mean_squared_error(train.iloc[valid_idx]["outcome_y"], pred, sample_weight=unit_weights(train.iloc[valid_idx])) ** 0.5)
        scored.append((float(np.mean(fold_errors)), parameters))
    return min(scored, key=lambda pair: pair[0])[1]


def target_metrics(actual: np.ndarray, predicted: np.ndarray, outcome: str) -> dict[str, Any]:
    threshold = RECOVERY_TARGET if outcome == "recovery" else WEIGHT_TARGET_KG
    actual_hit, predicted_hit = actual >= threshold, predicted >= threshold
    below_recall = float(np.mean(~predicted_hit[~actual_hit])) if (~actual_hit).any() else np.nan
    above_recall = float(np.mean(predicted_hit[actual_hit])) if actual_hit.any() else np.nan
    return {
        "target_side_accuracy": float(np.mean(actual_hit == predicted_hit)),
        "majority_side_accuracy": float(max(np.mean(actual_hit), np.mean(~actual_hit))),
        "below_target_recall": below_recall, "at_or_above_target_recall": above_recall,
        "balanced_target_accuracy": float(np.nanmean([below_recall, above_recall])),
        "confusion_actual_below_predicted_below": int(np.sum((~actual_hit) & (~predicted_hit))),
        "confusion_actual_below_predicted_above": int(np.sum((~actual_hit) & predicted_hit)),
        "confusion_actual_above_predicted_below": int(np.sum(actual_hit & (~predicted_hit))),
        "confusion_actual_above_predicted_above": int(np.sum(actual_hit & predicted_hit)),
    }


def summarize_predictions(predictions: pd.DataFrame, outcome: str) -> dict[str, Any]:
    actual = predictions["actual"].to_numpy(float)
    predicted = predictions["predicted"].to_numpy(float)
    weights = unit_weights(predictions)
    errors = predicted - actual
    factor = 100.0 if outcome == "recovery" else 1000.0
    cycle_mae = predictions.assign(abs_error=np.abs(errors)).groupby("cycle_id")["abs_error"].mean()
    cycle_rmse = predictions.assign(squared_error=errors**2).groupby("cycle_id")["squared_error"].mean().pow(0.5)
    unit_mae = predictions.assign(abs_error=np.abs(errors)).groupby(["cycle_id", "building_id"])["abs_error"].mean()
    unit_rmse = predictions.assign(squared_error=errors**2).groupby(["cycle_id", "building_id"])["squared_error"].mean().pow(0.5)
    result = {
        "rows": len(predictions), "independent_building_cycles": int(predictions[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "mae": float(mean_absolute_error(actual, predicted) * factor),
        "building_cycle_balanced_mae": float(np.average(np.abs(errors), weights=weights) * factor),
        "unit_macro_mae": float(unit_mae.mean() * factor), "cycle_macro_mae": float(cycle_mae.mean() * factor),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5 * factor),
        "building_cycle_balanced_rmse": float(np.sqrt(np.average(errors**2, weights=weights)) * factor),
        "unit_macro_rmse": float(unit_rmse.mean() * factor), "cycle_macro_rmse": float(cycle_rmse.mean() * factor),
        "r2": float(r2_score(actual, predicted)), "bias": float(np.average(errors, weights=weights) * factor),
        "fold_mae_std": float(cycle_mae.std(ddof=1) * factor), "fold_mae_se": float(cycle_mae.std(ddof=1) / np.sqrt(len(cycle_mae)) * factor),
        "worst_cycle_mae": float(cycle_mae.max() * factor),
        "fold_rmse_std": float(cycle_rmse.std(ddof=1) * factor), "fold_rmse_se": float(cycle_rmse.std(ddof=1) / np.sqrt(len(cycle_rmse)) * factor),
        "worst_cycle_rmse": float(cycle_rmse.max() * factor),
    }
    result.update(target_metrics(actual, predicted, outcome))
    if outcome == "weight":
        result["within_100g_rate"] = float(np.average(np.abs(errors) <= 0.1, weights=weights))
        result["within_200g_rate"] = float(np.average(np.abs(errors) <= 0.2, weights=weights))
    return result


def _cycle_bootstrap(predictions: pd.DataFrame, outcome: str, repeats: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    cycles = predictions["cycle_id"].unique()
    mae_values, rmse_values = [], []
    for _ in range(repeats):
        sampled = rng.choice(cycles, size=len(cycles), replace=True)
        mae_values.append(np.mean([np.abs(predictions.loc[predictions["cycle_id"].eq(c), "error"]).mean() for c in sampled]))
        rmse_values.append(np.mean([np.sqrt(np.mean(predictions.loc[predictions["cycle_id"].eq(c), "error"] ** 2)) for c in sampled]))
    factor = 100.0 if outcome == "recovery" else 1000.0
    mae_low, mae_high = np.quantile(mae_values, [0.025, 0.975]) * factor
    rmse_low, rmse_high = np.quantile(rmse_values, [0.025, 0.975]) * factor
    return {"cycle_bootstrap_mae_95ci_low": float(mae_low), "cycle_bootstrap_mae_95ci_high": float(mae_high), "cycle_bootstrap_rmse_95ci_low": float(rmse_low), "cycle_bootstrap_rmse_95ci_high": float(rmse_high)}


def evaluate_candidate(frame: pd.DataFrame, candidate: Candidate, outcome: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    groups = frame["cycle_id"].astype(str).to_numpy()
    rows, parameters = [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        best = _tune(train, candidate, outcome)
        fitted = _fit_model(train, candidate, outcome, best)
        predicted = predict_fitted(fitted, test, candidate, outcome)
        parameters.append({"held_out_cycle": str(test["cycle_id"].iloc[0]), **best})
        for source, pred in zip(test.to_dict("records"), predicted):
            rows.append({"candidate": candidate.name, "cycle_id": source["cycle_id"], "building_id": source["building_id"], "review_day": int(source["review_day"]), "as_of_date": source["as_of_date"], "actual": float(source["outcome_y"]), "predicted": float(pred), "error": float(pred - source["outcome_y"]), "absolute_error": float(abs(pred - source["outcome_y"]))})
    predictions = pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True)
    summary = summarize_predictions(predictions, outcome)
    summary.update(_cycle_bootstrap(predictions, outcome))
    summary.update({"candidate": candidate.name, "family": candidate.family, "target_form": candidate.target_form, "complexity": candidate.complexity, "description": candidate.description, "outer_fold_parameters": parameters})
    return predictions, summary


def temporal_stress(frame: pd.DataFrame, candidate: Candidate, outcome: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    cycles = [c for c in DEVELOPMENT_CYCLES if c in set(frame["cycle_id"])]
    for position in range(2, len(cycles)):
        train_cycles, test_cycle = cycles[:position], cycles[position]
        train, test = frame[frame["cycle_id"].isin(train_cycles)], frame[frame["cycle_id"].eq(test_cycle)]
        parameters = _tune(train, candidate, outcome)
        predicted = predict_fitted(_fit_model(train, candidate, outcome, parameters), test, candidate, outcome)
        for source, pred in zip(test.to_dict("records"), predicted):
            rows.append({"candidate": candidate.name, "cycle_id": test_cycle, "building_id": source["building_id"], "review_day": int(source["review_day"]), "actual": float(source["outcome_y"]), "predicted": float(pred), "error": float(pred-source["outcome_y"]), "absolute_error": float(abs(pred-source["outcome_y"]))})
    predictions = pd.DataFrame(rows)
    return predictions, summarize_predictions(predictions, outcome)


def select_champion(comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ranked = comparison.sort_values(["cycle_macro_rmse", "rmse", "complexity"])
    best = ranked.iloc[0]
    # Small unit tests and imported comparison tables may predate the fold-SE
    # column. In that case the one-SE rule correctly reduces to minimum RMSE.
    fold_rmse_se = float(best.get("fold_rmse_se", 0.0))
    one_se_limit = float(best["cycle_macro_rmse"] + fold_rmse_se)
    eligible = ranked.loc[ranked["cycle_macro_rmse"].le(one_se_limit)].copy()
    chosen = eligible.sort_values(["complexity", "cycle_macro_rmse", "rmse"]).iloc[0]
    return str(chosen["candidate"]), {
        "selection_metric": "nested leave-one-cycle-out cycle_macro_rmse",
        "selection_rule": "one-standard-error rule; simplest candidate within one SE of the lowest cycle-macro RMSE",
        "lowest_error_candidate": str(best["candidate"]),
        "lowest_cycle_macro_rmse": float(best["cycle_macro_rmse"]),
        "one_se_limit": one_se_limit,
        "eligible_candidates": eligible["candidate"].astype(str).tolist(),
        "selected_candidate": str(chosen["candidate"]),
        "selected_cycle_macro_rmse": float(chosen["cycle_macro_rmse"]),
        "selected_overall_rmse": float(chosen["rmse"]),
        "selected_r2": float(chosen["r2"]),
    }


def _finite_conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal absolute-residual quantile."""

    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    probability = min(1.0, np.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, probability, method="higher"))


def conformal_predictions(frame: pd.DataFrame, candidate: Candidate, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, calibration_rows = [], []
    groups = frame["cycle_id"].astype(str).to_numpy()
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        inner_predictions, _ = evaluate_candidate(train, candidate, outcome)
        residuals = inner_predictions["absolute_error"].to_numpy(float)
        q80 = _finite_conformal_quantile(residuals, 0.80)
        q90 = _finite_conformal_quantile(residuals, 0.90)
        params = _tune(train, candidate, outcome)
        predicted = predict_fitted(_fit_model(train, candidate, outcome, params), test, candidate, outcome)
        low_bound, high_bound = ((0.0, 1.0) if outcome == "recovery" else (0.05, 3.5))
        for source, pred in zip(test.to_dict("records"), predicted):
            actual = float(source["outcome_y"])
            lower80, upper80 = max(low_bound, pred-q80), min(high_bound, pred+q80)
            lower90, upper90 = max(low_bound, pred-q90), min(high_bound, pred+q90)
            target = RECOVERY_TARGET if outcome == "recovery" else WEIGHT_TARGET_KG
            target_status = "Likely below" if upper80 < target else "Likely meets" if lower80 >= target else "Uncertain"
            rows.append({"cycle_id": source["cycle_id"], "building_id": source["building_id"], "review_day": int(source["review_day"]), "actual": actual, "predicted": float(pred), "lower_80": lower80, "upper_80": upper80, "lower_90": lower90, "upper_90": upper90, "covered_80": bool(pred-q80 <= actual <= pred+q80), "covered_90": bool(pred-q90 <= actual <= pred+q90), "target_status_80": target_status})
        calibration_rows.append({"held_out_cycle": str(test["cycle_id"].iloc[0]), "calibration_rows": len(residuals), "q80": q80, "q90": q90, "checkpoint_specific_reliable": bool(len(residuals) >= 20)})
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def checkpoint_table(predictions: pd.DataFrame, outcome: str) -> pd.DataFrame:
    records = []
    for day, group in predictions.groupby("review_day"):
        records.append({"review_day": int(day), **summarize_predictions(group, outcome)})
    return pd.DataFrame(records).sort_values("review_day")


def identity_sensitivity(
    frame: pd.DataFrame, outcome: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate site/building remaining-outcome baselines outside selection."""

    target_column = "additional_loss_y" if outcome == "recovery" else "remaining_gain_y"
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    groups = frame["cycle_id"].astype(str).to_numpy()
    for sensitivity, identity in (
        ("site_group", frame["building_id"].astype(str).str.split().str[0]),
        ("exact_building", frame["building_id"].astype(str)),
    ):
        rows: list[dict[str, Any]] = []
        working = frame.assign(_identity=identity.to_numpy())
        for train_index, test_index in LeaveOneGroupOut().split(working, groups=groups):
            train, test = working.iloc[train_index], working.iloc[test_index]
            means = train.groupby(["review_day", "_identity"])[target_column].mean()
            fallback = train.groupby("review_day")[target_column].mean()
            raw = np.asarray(
                [
                    means.get((int(review_day), str(identity_value)), fallback.get(int(review_day), train[target_column].mean()))
                    for review_day, identity_value in test[["review_day", "_identity"]].itertuples(index=False, name=None)
                ],
                dtype=float,
            )
            predicted = _to_outcome(test, raw, outcome, "remaining")
            for source, prediction in zip(test.to_dict("records"), predicted):
                rows.append(
                    {
                        "sensitivity": sensitivity,
                        "cycle_id": source["cycle_id"],
                        "building_id": source["building_id"],
                        "review_day": int(source["review_day"]),
                        "actual": float(source["outcome_y"]),
                        "predicted": float(prediction),
                        "error": float(prediction - source["outcome_y"]),
                    }
                )
        prediction_frame = pd.DataFrame(rows)
        all_rows.extend(rows)
        summaries.append(
            {
                "sensitivity": sensitivity,
                "eligible_for_primary_selection": False,
                **summarize_predictions(prediction_frame, outcome),
            }
        )
    return pd.DataFrame(all_rows), pd.DataFrame(summaries)


def leave_one_building_label_out(
    frame: pd.DataFrame, candidate: Candidate, outcome: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = frame["building_id"].astype(str).to_numpy()
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=labels):
        train, test = frame.iloc[train_index], frame.iloc[test_index]
        parameters = _tune(train, candidate, outcome)
        predicted = predict_fitted(
            _fit_model(train, candidate, outcome, parameters), test, candidate, outcome
        )
        for source, prediction in zip(test.to_dict("records"), predicted):
            rows.append(
                {
                    "held_out_building_label": source["building_id"],
                    "cycle_id": source["cycle_id"],
                    "building_id": source["building_id"],
                    "review_day": int(source["review_day"]),
                    "actual": float(source["outcome_y"]),
                    "predicted": float(prediction),
                    "error": float(prediction - source["outcome_y"]),
                }
            )
    predictions = pd.DataFrame(rows)
    return predictions, summarize_predictions(predictions, outcome)


def _permutation_importance(frame: pd.DataFrame, candidate: Candidate, outcome: str) -> pd.DataFrame:
    if candidate.family in {"persistence", "pace", "age_baseline"}:
        return pd.DataFrame(columns=["feature", "rmse_increase"])
    records = []
    groups = frame["cycle_id"].astype(str).to_numpy()
    columns = _feature_columns(outcome)
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        parameters = _tune(train, candidate, outcome)
        fitted = _fit_model(train, candidate, outcome, parameters)
        # Permute raw inputs and score predictions in business-outcome units.
        base = mean_squared_error(test["outcome_y"], predict_fitted(fitted, test, candidate, outcome)) ** 0.5
        rng = np.random.default_rng(SEED)
        for column in columns:
            increases = []
            for _ in range(20):
                changed = test.copy()
                changed[column] = rng.permutation(changed[column].to_numpy())
                increases.append(mean_squared_error(test["outcome_y"], predict_fitted(fitted, changed, candidate, outcome)) ** 0.5 - base)
            records.append({"held_out_cycle": str(test["cycle_id"].iloc[0]), "feature": column, "rmse_increase": float(np.mean(increases))})
    result = pd.DataFrame(records).groupby("feature", as_index=False)["rmse_increase"].mean()
    result["rmse_increase"] *= 100 if outcome == "recovery" else 1000
    return result.sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def _tree_shap(frame: pd.DataFrame, candidate: Candidate, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate.family not in {"extra_trees", "compact_extra_trees", "random_forest", "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"}:
        return pd.DataFrame(), pd.DataFrame()
    global_rows, local_rows = [], []
    groups = frame["cycle_id"].astype(str).to_numpy()
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        fitted = _fit_model(train, candidate, outcome, _tune(train, candidate, outcome))
        model = fitted["model"]
        transformed = model.named_steps["impute"].transform(test[fitted["features"]])
        names = model.named_steps["impute"].get_feature_names_out(fitted["features"])
        values = shap.TreeExplainer(model.named_steps["model"]).shap_values(transformed)
        for index, source in enumerate(test.to_dict("records")):
            order = np.argsort(np.abs(values[index]))[::-1][:5]
            for rank, feature_index in enumerate(order, start=1):
                local_rows.append({"cycle_id": source["cycle_id"], "building_id": source["building_id"], "review_day": int(source["review_day"]), "rank": rank, "feature": str(names[feature_index]), "feature_value": float(transformed[index, feature_index]), "shap_value": float(values[index, feature_index])})
        for feature, importance in zip(names, np.mean(np.abs(values), axis=0)):
            global_rows.append({"feature": str(feature), "mean_abs_shap": float(importance)})
    global_frame = pd.DataFrame(global_rows).groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False)
    return global_frame, pd.DataFrame(local_rows)


def _coefficient_table(fitted: dict[str, Any], candidate: Candidate, outcome: str) -> pd.DataFrame:
    """Return standardized learned coefficients when the finalist supports them."""

    if candidate.family not in {"linear", "ridge", "elastic_net", "huber", "growth"}:
        return pd.DataFrame(columns=["feature", "standardized_coefficient", "outcome_direction"])
    pipeline = fitted["model"]
    names = pipeline.named_steps["impute"].get_feature_names_out(fitted["features"])
    estimator = pipeline.named_steps["model"]
    coefficients = estimator.regressor_.coef_ if candidate.family == "growth" else estimator.coef_
    direction_factor = -1 if outcome == "recovery" and candidate.target_form == "remaining" else 1
    result = pd.DataFrame({"feature": names, "standardized_coefficient": coefficients})
    result["outcome_direction"] = np.where(result["standardized_coefficient"] * direction_factor > 0, "higher predicted outcome", "lower predicted outcome")
    return result.reindex(result["standardized_coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _plot_outputs(outcome: str, snapshots: pd.DataFrame, predictions: pd.DataFrame, comparison: pd.DataFrame, intervals: pd.DataFrame, importance: pd.DataFrame, shap_global: pd.DataFrame, shap_local: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    unit = "Recovery (%)" if outcome == "recovery" else "Day 35 weight (g)"
    factor = 100 if outcome == "recovery" else 1000
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5)); values = snapshots.drop_duplicates(["cycle_id", "building_id"])["outcome_y"] * factor
    ax.hist(values, bins=min(10, max(5, len(values)//3)), color="#2A6F97", edgecolor="white"); ax.set(title=f"{outcome.title()} outcome distribution", xlabel=unit, ylabel="Building-cycles"); fig.tight_layout(); fig.savefig(figures/"outcome_distribution.png", dpi=160); plt.close(fig)
    units=snapshots.drop_duplicates(["cycle_id","building_id"]); fig,ax=plt.subplots(figsize=(8,5));
    for position,(cycle,group) in enumerate(units.groupby("cycle_id")):
        ax.scatter(np.full(len(group),position),group["outcome_y"]*factor,label=cycle,alpha=.75)
    ax.set_xticks(range(units["cycle_id"].nunique()),sorted(units["cycle_id"].unique())); ax.set(title="Outcome variation by complete cycle",xlabel="Cycle",ylabel=unit); fig.tight_layout(); fig.savefig(figures/"outcome_by_cycle.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(predictions["actual"]*factor, predictions["predicted"]*factor, alpha=.65, color="#2A6F97"); bounds=[min(predictions[["actual","predicted"]].min())*factor,max(predictions[["actual","predicted"]].max())*factor]; ax.plot(bounds,bounds,"--",color="#555"); ax.set(title="Held-out actual versus predicted",xlabel=f"Actual {unit}",ylabel=f"Predicted {unit}"); fig.tight_layout(); fig.savefig(figures/"actual_vs_predicted.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.scatter(predictions["predicted"]*factor,predictions["error"]*factor,c=predictions["review_day"],cmap="viridis",alpha=.7); ax.axhline(0,color="#555",ls="--"); ax.set(title="Held-out residuals",xlabel=f"Predicted {unit}",ylabel="Prediction error"); fig.tight_layout(); fig.savefig(figures/"residuals.png",dpi=160); plt.close(fig)
    by_age=predictions.assign(squared_error_display=(predictions["error"]*factor)**2).groupby("review_day")["squared_error_display"].mean().pow(0.5); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(by_age.index.astype(str),by_age.values,color="#3A7D44"); ax.set(title="RMSE by review day",xlabel="Review day",ylabel=f"RMSE ({'percentage points' if outcome=='recovery' else 'g'})"); fig.tight_layout(); fig.savefig(figures/"error_by_age.png",dpi=160); plt.close(fig)
    ordered=comparison.sort_values("cycle_macro_rmse").head(5); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(ordered["candidate"],ordered["cycle_macro_rmse"],color="#2A6F97"); ax.invert_yaxis(); ax.set(title="Top five models by whole-cycle RMSE",xlabel="Cycle-macro RMSE"); fig.tight_layout(); fig.savefig(figures/"candidate_stability.png",dpi=160); plt.close(fig)
    miss=snapshots[_feature_columns(outcome)].isna().mean().sort_values(ascending=False).head(12)*100; fig,ax=plt.subplots(figsize=(8,5)); ax.barh(miss.index[::-1],miss.values[::-1],color="#D17A22"); ax.set(title="Highest feature missingness",xlabel="Missing snapshots (%)"); fig.tight_layout(); fig.savefig(figures/"missingness.png",dpi=160); plt.close(fig)
    cov=intervals.groupby("review_day")[["covered_80","covered_90"]].mean()*100; fig,ax=plt.subplots(figsize=(7,4)); cov.plot(kind="bar",ax=ax,color=["#77AADD","#225588"]); ax.axhline(80,color="#77AADD",ls="--"); ax.axhline(90,color="#225588",ls="--"); ax.set(title="Empirical interval coverage by review day",xlabel="Review day",ylabel="Coverage (%)"); fig.tight_layout(); fig.savefig(figures/"interval_coverage.png",dpi=160); plt.close(fig)
    if not importance.empty:
        top=importance.head(12).sort_values("rmse_increase"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top["feature"],top["rmse_increase"],color="#7A5195"); ax.axvline(0,color="#555",lw=.8); ax.set(title="Held-out permutation importance",xlabel="RMSE increase when permuted"); fig.tight_layout(); fig.savefig(figures/"feature_importance.png",dpi=160); plt.close(fig)
    if not shap_global.empty:
        top=shap_global.head(12).sort_values("mean_abs_shap"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top["feature"],top["mean_abs_shap"]*factor,color="#8C6BB1"); ax.set(title="Held-out SHAP importance",xlabel=f"Mean absolute SHAP ({'recovery points' if outcome=='recovery' else 'g'})"); fig.tight_layout(); fig.savefig(figures/"shap_importance.png",dpi=160); plt.close(fig)
    if not shap_local.empty:
        top_names=shap_global.head(10)["feature"].tolist(); plot=shap_local.loc[shap_local["feature"].isin(top_names)].copy(); positions={name:index for index,name in enumerate(reversed(top_names))}; rng=np.random.default_rng(SEED); fig,ax=plt.subplots(figsize=(9,6));
        for name,group in plot.groupby("feature"):
            values=group["feature_value"].to_numpy(float); spread=np.nanmax(values)-np.nanmin(values); normalized=(values-np.nanmin(values))/spread if spread>0 else np.full(len(values),0.5)
            y=np.full(len(group),positions[name])+rng.normal(0,0.08,len(group)); scatter=ax.scatter(group["shap_value"]*factor,y,c=normalized,cmap="coolwarm",vmin=0,vmax=1,s=24,alpha=.75)
        ax.axvline(0,color="#555",lw=.8); ax.set_yticks(range(len(top_names)),list(reversed(top_names))); ax.set(title="Held-out SHAP distribution",xlabel=f"SHAP contribution ({'recovery points' if outcome=='recovery' else 'g'})"); colorbar=fig.colorbar(scatter,ax=ax,label="Feature value (within-feature scale)"); colorbar.set_ticks([0,1],labels=["Low","High"]); fig.tight_layout(); fig.savefig(figures/"shap_beeswarm.png",dpi=160); plt.close(fig)


def _data_dictionary(outcome: str) -> pd.DataFrame:
    rows = []
    for feature in _feature_columns(outcome):
        rows.append({"field": feature, "role": "X", "available_when": "On or before review date", "missingness": "Fold-local median plus indicator in learned models", "leakage_note": "Derived only from source rows with age <= review_day"})
    rows.append({"field": "outcome_y", "role": "Y", "available_when": "After endpoint", "missingness": "Required", "leakage_note": "Never included in X"})
    return pd.DataFrame(rows)


def _benchmark(outcome: str, root: Path) -> dict[str, Any]:
    path = root / "models" / ("recovery_manifest.json" if outcome == "recovery" else "day35_weight_manifest.json")
    manifest = json.loads(path.read_text())
    selected = manifest["selected_metrics"]
    later = manifest.get("prospective_latest_cycle_audit", {}).get("metrics", {})
    return {"version": manifest["model_version"], "method": manifest["selected_model"], "mae": selected["mae"]*100 if outcome=="recovery" else selected["mae_kg"]*1000, "cycle_balanced_mae": selected["cycle_macro_mae"]*100 if outcome=="recovery" else selected["cycle_macro_mae_kg"]*1000, "rmse": selected["rmse"]*100 if outcome=="recovery" else selected["rmse_kg"]*1000, "r2": selected["r2"], "bias": selected["bias"]*100 if outcome=="recovery" else selected["bias_kg"]*1000, "later_cycle_rmse": (later.get("rmse", np.nan)*100 if outcome=="recovery" else later.get("rmse_kg", np.nan)*1000)}


def run_outcome(dataset: CanaryDataset, outcome: str, output: Path, root: Path) -> dict[str, Any]:
    assert_primary_schema_has_no_identity(_feature_columns(outcome))
    if outcome == "recovery":
        assert_primary_schema_has_no_identity(RECOVERY_COMPACT_FEATURES)
    directory, figures = output/outcome, output/outcome/"figures"
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = build_snapshots(dataset, outcome)
    development = snapshots.loc[snapshots["role"].eq("development")].reset_index(drop=True)
    audit = snapshots.loc[snapshots["role"].eq("later_cycle_audit")].reset_index(drop=True)
    candidates = RECOVERY_CANDIDATES if outcome == "recovery" else WEIGHT_CANDIDATES
    all_predictions, summaries = [], []
    for candidate in candidates:
        predictions, summary = evaluate_candidate(development, candidate, outcome)
        all_predictions.append(predictions); summaries.append(summary)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    comparison = pd.DataFrame([{key:value for key,value in summary.items() if key != "outer_fold_parameters"} for summary in summaries]).sort_values("cycle_macro_rmse").reset_index(drop=True)
    comparison["rank_by_cycle_macro_rmse"] = np.arange(1, len(comparison) + 1)
    top_five = comparison.head(5).copy()
    selected_name, selection = select_champion(comparison)
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    shadow_name = str(selection["lowest_error_candidate"])
    shadow_candidate = next(candidate for candidate in candidates if candidate.name == shadow_name)
    selected_predictions = prediction_table.loc[prediction_table["candidate"].eq(selected_name)].copy()
    temporal_predictions, temporal_summary = temporal_stress(development, selected, outcome)
    intervals, calibration = conformal_predictions(development, selected, outcome)
    if shadow_name == selected_name:
        shadow_intervals, shadow_calibration = intervals.copy(), calibration.copy()
    else:
        shadow_intervals, shadow_calibration = conformal_predictions(
            development, shadow_candidate, outcome
        )
    learned = [candidate for candidate in candidates if candidate.family not in {"persistence", "pace", "age_baseline"}]
    best_learned_name = str(comparison.loc[comparison["candidate"].isin([c.name for c in learned])].sort_values("cycle_macro_rmse").iloc[0]["candidate"])
    best_learned = next(candidate for candidate in learned if candidate.name == best_learned_name)
    tree_names = [candidate.name for candidate in candidates if candidate.family in {"extra_trees", "compact_extra_trees", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"}]
    tree_name = str(comparison.loc[comparison["candidate"].isin(tree_names)].sort_values("cycle_macro_rmse").iloc[0]["candidate"])
    tree_candidate = next(candidate for candidate in candidates if candidate.name == tree_name)
    explanation_candidate = selected if selected.family not in {"persistence", "pace", "age_baseline"} else best_learned
    importance = _permutation_importance(development, explanation_candidate, outcome)
    shap_candidate = selected if selected.family in {"extra_trees", "compact_extra_trees", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"} else tree_candidate
    shap_global, shap_local = _tree_shap(development, shap_candidate, outcome)
    best_learned_parameters = _tune(development, best_learned, outcome)
    best_learned_fitted = _fit_model(development, best_learned, outcome, best_learned_parameters)
    coefficients = _coefficient_table(best_learned_fitted, best_learned, outcome)
    final_parameters = _tune(development, selected, outcome)
    fitted = _fit_model(development, selected, outcome, final_parameters)
    audit_predictions = predict_fitted(fitted, audit, selected, outcome)
    audit_export = audit[["cycle_id","building_id","review_day","as_of_date","outcome_y","endpoint_warning"]].copy()
    audit_export["predicted"] = audit_predictions; audit_export["error"] = audit_predictions-audit["outcome_y"].to_numpy(); audit_export["absolute_error"] = np.abs(audit_export["error"])
    shadow_parameters = _tune(development, shadow_candidate, outcome)
    shadow_fitted = _fit_model(development, shadow_candidate, outcome, shadow_parameters)
    shadow_audit_predictions = predict_fitted(shadow_fitted, audit, shadow_candidate, outcome)
    shadow_audit_export = audit[["cycle_id", "building_id", "review_day", "as_of_date", "outcome_y", "endpoint_warning"]].copy()
    shadow_audit_export["candidate"] = shadow_name
    shadow_audit_export["predicted"] = shadow_audit_predictions
    shadow_audit_export["error"] = shadow_audit_predictions - audit["outcome_y"].to_numpy(float)
    shadow_audit_export["absolute_error"] = np.abs(shadow_audit_export["error"])
    benchmark = _benchmark(outcome, root)
    selected_metrics = summarize_predictions(selected_predictions, outcome)
    baseline_name = "age_band_remaining_loss" if outcome=="recovery" else "historical_remaining_gain"
    baseline_metrics = comparison.loc[comparison["candidate"].eq(baseline_name)].iloc[0].to_dict()
    matched_current_name = "remaining_ols" if outcome == "recovery" else "historical_remaining_gain"
    matched_current = comparison.loc[comparison["candidate"].eq(matched_current_name)].iloc[0].to_dict()
    day14_selected = checkpoint_table(selected_predictions, outcome).set_index("review_day").loc[14]
    day14_baseline_predictions = prediction_table.loc[prediction_table["candidate"].eq(baseline_name)]
    day14_baseline = checkpoint_table(day14_baseline_predictions, outcome).set_index("review_day").loc[14]
    audit_metrics = summarize_predictions(audit_export.rename(columns={"outcome_y":"actual"}), outcome)
    replacement_ready = bool(
        selected_name != matched_current_name
        and selected_metrics["cycle_macro_rmse"] < float(matched_current["cycle_macro_rmse"])
        and selected_metrics["r2"] > max(0.0, float(matched_current["r2"]))
        and float(day14_selected["rmse"]) <= float(day14_baseline["rmse"])
        and selected_metrics["r2"] >= 0.05
        and audit_metrics["rmse"] <= benchmark["later_cycle_rmse"]
        and abs(selected_metrics["bias"]) <= max(abs(float(matched_current["bias"]))*1.25, 1.0 if outcome=="recovery" else 10.0)
    )
    recommendation = "Research champion clears the retrospective gates; run it in shadow mode before any application replacement." if replacement_ready else "Retain the current Canary application method; the research champion did not clear every robustness gate."
    bundle = {"outcome": outcome, "candidate": selected, "fitted": fitted, "features": _feature_columns(outcome), "parameters": final_parameters, "source_cycles": list(DEVELOPMENT_CYCLES), "seed": SEED}
    joblib.dump(bundle, directory/"champion.joblib")
    reloaded = joblib.load(directory/"champion.joblib")
    parity = np.allclose(predict_fitted(reloaded["fitted"], audit, reloaded["candidate"], outcome), audit_predictions)
    shadow_bundle = {"outcome": outcome, "candidate": shadow_candidate, "fitted": shadow_fitted, "features": shadow_fitted.get("features", []), "parameters": shadow_parameters, "source_cycles": list(DEVELOPMENT_CYCLES), "seed": SEED, "deployment_status": "shadow"}
    joblib.dump(shadow_bundle, directory/"shadow_challenger.joblib")
    feed_candidate = next(c for c in candidates if c.family in {"ridge"})
    feed_params = _tune(development, feed_candidate, outcome)
    # Sensitivity only: it is intentionally not eligible for champion selection.
    feed_rows=[]
    groups=development["cycle_id"].astype(str).to_numpy()
    for tr,te in LeaveOneGroupOut().split(development,groups=groups):
        fit=_fit_model(development.iloc[tr],feed_candidate,outcome,feed_params,include_feed=True)
        pred=predict_fitted(fit,development.iloc[te],feed_candidate,outcome)
        for source,p in zip(development.iloc[te].to_dict("records"),pred): feed_rows.append({"cycle_id":source["cycle_id"],"building_id":source["building_id"],"review_day":source["review_day"],"actual":source["outcome_y"],"predicted":p,"error":p-source["outcome_y"]})
    feed_sensitivity = summarize_predictions(pd.DataFrame(feed_rows), outcome)
    checkpoint = checkpoint_table(selected_predictions, outcome)
    identity_predictions, identity_comparison = identity_sensitivity(development, outcome)
    building_label_predictions, building_label_metrics = leave_one_building_label_out(
        development, selected, outcome
    )
    component_explanations = audit_export[["cycle_id", "building_id", "review_day", "predicted"]].copy()
    if outcome == "recovery":
        component_explanations["known_current_value"] = audit["percentage_alive"].to_numpy()
        component_explanations["historical_remaining_component"] = component_explanations["known_current_value"] - component_explanations["predicted"]
    else:
        component_explanations["known_current_value"] = audit["current_weight_kg"].to_numpy()
        component_explanations["historical_remaining_component"] = component_explanations["predicted"] - component_explanations["known_current_value"]
    for frame,name in [(snapshots,"model_ready_snapshots.csv"),(prediction_table,"oof_predictions.csv"),(comparison,"candidate_comparison.csv"),(top_five,"top_five_models.csv"),(checkpoint,"checkpoint_metrics.csv"),(temporal_predictions,"temporal_stress_predictions.csv"),(intervals,"conformal_predictions.csv"),(calibration,"conformal_calibration.csv"),(shadow_intervals,"shadow_conformal_predictions.csv"),(shadow_calibration,"shadow_conformal_calibration.csv"),(importance,"permutation_importance.csv"),(coefficients,"standardized_coefficients.csv"),(shap_global,"shap_global.csv"),(shap_local,"shap_local.csv"),(component_explanations,"individual_component_explanations.csv"),(audit_export,"later_cycle_audit_predictions.csv"),(shadow_audit_export,"shadow_later_cycle_audit_predictions.csv"),(identity_predictions,"identity_sensitivity_predictions.csv"),(identity_comparison,"identity_sensitivity_comparison.csv"),(building_label_predictions,"leave_one_building_label_out_predictions.csv"),(_data_dictionary(outcome),"data_dictionary.csv")]: frame.to_csv(directory/name,index=False)
    registry = pd.DataFrame([candidate.__dict__ for candidate in candidates]); registry.to_csv(directory/"candidate_registry.csv",index=False)
    (directory/"nested_hyperparameters.json").write_text(json.dumps({summary["candidate"]: summary["outer_fold_parameters"] for summary in summaries},indent=2,default=_json_default),encoding="utf-8")
    _plot_outputs(outcome, development, selected_predictions, comparison, intervals, importance, shap_global, shap_local, figures)
    manifest = {"outcome": outcome, "primary_objective": "Minimize cycle-macro RMSE across Days 7, 14, 21 and 28", "seed": SEED, "development_cycles": list(DEVELOPMENT_CYCLES), "later_cycle": AUDIT_CYCLE, "snapshot_rows": len(development), "independent_outcomes": int(development[["cycle_id","building_id"]].drop_duplicates().shape[0]), "selected_candidate": selected_name, "selected_final_parameters": final_parameters, "shadow_candidate": shadow_name, "shadow_final_parameters": shadow_parameters, "shadow_later_cycle_audit_metrics": summarize_predictions(shadow_audit_export.rename(columns={"outcome_y": "actual"}), outcome), "shadow_interval_coverage": {"coverage_80": float(shadow_intervals["covered_80"].mean()), "coverage_90": float(shadow_intervals["covered_90"].mean()), "mean_width_80": float((shadow_intervals["upper_80"]-shadow_intervals["lower_80"]).mean()), "mean_width_90": float((shadow_intervals["upper_90"]-shadow_intervals["lower_90"]).mean())}, "best_learned_candidate": best_learned_name, "best_learned_final_parameters": best_learned_parameters, "primary_identity_policy": "Exact building and Tags/Lags are excluded from all primary candidates.", "identity_sensitivity": identity_comparison.to_dict("records"), "leave_one_building_label_out_metrics": building_label_metrics, "permutation_importance_model": explanation_candidate.name, "shap_model": shap_candidate.name, "shap_is_champion": bool(shap_candidate.name == selected_name), "selection": selection, "top_five_models": top_five.to_dict("records"), "selected_metrics": selected_metrics, "checkpoint_metrics": checkpoint.to_dict("records"), "day14_secondary_gate": {"selected_rmse": float(day14_selected["rmse"]), "baseline_rmse": float(day14_baseline["rmse"]), "passed": bool(float(day14_selected["rmse"]) <= float(day14_baseline["rmse"]))}, "temporal_stress_metrics": temporal_summary, "interval_coverage": {"coverage_80": float(intervals["covered_80"].mean()), "coverage_90": float(intervals["covered_90"].mean()), "mean_width_80": float((intervals["upper_80"]-intervals["lower_80"]).mean()), "mean_width_90": float((intervals["upper_90"]-intervals["lower_90"]).mean())}, "published_canary_benchmark": benchmark, "matched_current_method": matched_current, "feed_sensitivity": feed_sensitivity, "later_cycle_audit_metrics": audit_metrics, "replacement_ready": replacement_ready, "recommendation": recommendation, "artifact_reload_parity": bool(parity), "limitations": ["Only 31 independent development building-cycles across six cycles.", "Target-side outcomes are highly imbalanced.", "Environmental coverage is incomplete and absent in the earliest cycle.", "Importance and SHAP are predictive associations, not causal treatment evidence."]}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2,default=_json_default),encoding="utf-8")
    return manifest


def run_review(workbook: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    workbook = Path(workbook).resolve(); root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve() if output else root/"outputs"/"external_modeling_review"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(workbook)
    audit = {"source_workbook": str(workbook), "source_sha256": _source_sha256(workbook), "source_rows": dataset.quality.source_rows, "canonical_rows": dataset.quality.canonical_rows, "unique_building_days": int(dataset.daily[["cycle_id","building_id","age_day"]].drop_duplicates().shape[0]), "total_building_cycles": int(dataset.cycles[["cycle_id","building_id"]].drop_duplicates().shape[0]), "development_building_cycles": int(dataset.cycles[dataset.cycles["cycle_id"].isin(DEVELOPMENT_CYCLES)][["cycle_id","building_id"]].drop_duplicates().shape[0]), "later_cycle_buildings": int(dataset.cycles[dataset.cycles["cycle_id"].eq(AUDIT_CYCLE)]["building_id"].nunique()), "temperature_coverage_pct": dataset.quality.temperature_coverage_pct, "humidity_coverage_pct": dataset.quality.humidity_coverage_pct, "operationally_missing_days": dataset.quality.operationally_missing_days, "weight_measurement_days": dataset.quality.weight_measurement_days, "blocking_errors": list(dataset.quality.blocking_errors), "warnings": list(dataset.quality.warnings), "feed_primary_model_policy": "Excluded because units are unresolved; evaluated only as sensitivity."}
    (output/"source_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    cycle_quality = dataset.daily.groupby("cycle_id", as_index=False).agg(
        building_days=("age_day", "size"), buildings=("building_id", "nunique"),
        temperature_coverage=("temperature_avg_c", lambda values: float(values.notna().mean())),
        humidity_coverage=("humidity_avg_pct", lambda values: float(values.notna().mean())),
        weight_measurements=("weight_measured", "sum"), operational_days=("operational_recorded", "sum"),
    )
    cycle_quality.to_csv(output/"data_quality_by_cycle.csv",index=False)
    prepared_recovery = root/"outputs"/"model_ready"/"recovery_training.csv"
    prepared_weight = root/"outputs"/"model_ready"/"day35_weight_training.csv"
    reconciliation = {
        "authoritative_primary_building_days": dataset.quality.canonical_rows,
        "older_canonical_filename_count": 1666,
        "prepared_recovery_rows": int(len(pd.read_csv(prepared_recovery))) if prepared_recovery.exists() else None,
        "prepared_weight_rows": int(len(pd.read_csv(prepared_weight))) if prepared_weight.exists() else None,
        "prepared_tables_role": "Benchmark and reconciliation only; not authoritative model design.",
        "independent_recovery_rows_note": "Uses the agreed common checkpoints only (Days 7, 14, 21 and 28), yielding 124 development snapshots from 31 independent building-cycles.",
    }
    (output/"source_reconciliation.json").write_text(json.dumps(reconciliation,indent=2),encoding="utf-8")
    results = {outcome: run_outcome(dataset,outcome,output,root) for outcome in ("recovery","weight")}
    master = {"review_version":"codex-independent-rmse-2.0.0","created":"2026-08-12","source_audit":audit,"outcomes":results}
    (output/"manifest.json").write_text(json.dumps(master,indent=2,default=_json_default),encoding="utf-8")
    return master
