"""Independent Day 35 bodyweight modeling review for Project Canary.

The workflow starts from the corrected farm workbook, reconstructs leakage-safe
checkpoint snapshots, compares checkpoint-specific poultry growth models under
nested leave-one-harvest-cycle-out validation, and writes an isolated research
bundle.  It does not modify Canary's application models or inference path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .data import CanaryDataset, load_workbook
from .farmwide_features import assert_primary_schema_has_no_identity


SEED = 20260812
CHECKPOINTS = (7, 14, 21, 28)
DEVELOPMENT_CYCLES = ("2025-2", "2025-3", "2025-4", "2025-5", "2026-1", "2026-2")
AUDIT_CYCLE = "2026-3"
DAY35_TARGET_G = 1800.0


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    target_form: str
    feature_set: str
    complexity: int
    description: str


CANDIDATES = (
    Candidate("historical_remaining_gain", "baseline", "remaining", "current", 0, "Naive projection: current weight plus fold-local expected remaining growth"),
    Candidate("target_gap_preserving", "target_gain", "direct", "current", 0, "Naive projection: add the farm target curve's remaining gain while preserving today's target deficit"),
    Candidate("historical_growth_ratio", "historical_ratio", "direct", "current", 0, "Naive projection: current weight multiplied by the fold-local historical checkpoint-to-Day-35 ratio"),
    Candidate("recent_adg_projection", "recent_adg", "direct", "compact", 1, "Naive projection: blend recent observed ADG with fold-local expected remaining daily gain"),
    Candidate("target_curve_pace", "pace", "direct", "current", 1, "Current target-relative pace projected to Day 35"),
    Candidate("direct_current_weight_ols", "ols", "direct", "current", 1, "Checkpoint-specific direct regression on current observed weight"),
    Candidate("direct_trajectory_pls", "pls", "direct", "trajectory", 2, "One-component partial least squares growth-trajectory model"),
    Candidate("blend_baseline_pls", "blend", "direct", "trajectory", 3, "Fold-tuned blend of remaining-gain baseline and trajectory PLS"),
    Candidate("direct_trajectory_ridge", "ridge", "direct", "trajectory", 3, "Regularized direct regression on the observed weight trajectory"),
    Candidate("remaining_trajectory_ridge", "ridge", "remaining", "trajectory", 3, "Regularized remaining-gain regression on the observed trajectory"),
    Candidate("direct_linear_svr", "svr", "direct", "trajectory", 4, "Linear support-vector regression on the observed trajectory"),
    Candidate("direct_huber_compact", "huber", "direct", "compact", 4, "Robust regression using compact growth and flock-state features"),
    Candidate("direct_elastic_net_compact", "elastic_net", "direct", "compact", 4, "Sparse regularized compact model"),
    Candidate("direct_extra_trees", "extra_trees", "direct", "extended", 5, "Constrained Extra Trees using growth, flock-state, and environment history"),
    Candidate("direct_random_forest", "random_forest", "direct", "extended", 6, "Constrained random forest using extended features"),
    Candidate("direct_gradient_boosting", "gradient_boosting", "direct", "extended", 6, "Constrained Huber gradient boosting using extended features"),
    Candidate("direct_extra_trees_growth", "extra_trees", "direct", "compact", 5, "Extra Trees restricted to growth and flock-state features"),
    Candidate("direct_extra_trees_poultry", "extra_trees", "direct", "poultry", 6, "Extra Trees with poultry-specific target-deficit and environmental exposure features"),
    Candidate("remaining_extra_trees_poultry", "extra_trees", "remaining", "poultry", 6, "Extra Trees for remaining gain with poultry-specific features"),
    Candidate("direct_random_forest_poultry", "random_forest", "direct", "poultry", 7, "Random forest with poultry-specific target-deficit and environmental exposure features"),
    Candidate("direct_gradient_boosting_poultry", "gradient_boosting", "direct", "poultry", 7, "Gradient boosting with poultry-specific target-deficit and environmental exposure features"),
    Candidate("direct_hist_gradient_boosting_poultry", "hist_gradient_boosting", "direct", "poultry", 7, "Histogram gradient boosting with poultry-specific features and native small-sample regularization"),
    Candidate("direct_poultry_core_ridge", "ridge", "direct", "poultry_core", 4, "Regularized direct model using a compact biologically selected environmental feature set"),
    Candidate("remaining_poultry_core_ridge", "ridge", "remaining", "poultry_core", 4, "Regularized remaining-gain model using compact poultry features"),
    Candidate("direct_poultry_core_pls", "pls_multi", "direct", "poultry_core", 4, "Low-rank partial least squares model using compact poultry features"),
    Candidate("direct_xgboost_growth", "xgboost", "direct", "compact", 7, "Regularized XGBoost using observed growth and flock-state features"),
    Candidate("direct_xgboost_poultry", "xgboost", "direct", "poultry_core", 8, "Regularized XGBoost using compact poultry-environment features"),
    Candidate("direct_lightgbm_growth", "lightgbm", "direct", "compact", 7, "Regularized LightGBM using observed growth and flock-state features"),
    Candidate("direct_lightgbm_poultry", "lightgbm", "direct", "poultry_core", 8, "Regularized LightGBM using compact poultry-environment features"),
    Candidate("direct_catboost_growth", "catboost", "direct", "compact", 7, "Regularized CatBoost using observed growth and flock-state features"),
    Candidate("direct_catboost_poultry", "catboost", "direct", "poultry_core", 8, "Regularized CatBoost using compact poultry-environment features"),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_map(dataset: CanaryDataset) -> dict[int, float]:
    return {
        int(row.age_day): float(row.target_weight_kg * 1000)
        for row in dataset.targets.itertuples(index=False)
        if pd.notna(row.age_day) and pd.notna(row.target_weight_kg)
    }


def _safe_float(value: Any) -> float:
    return float(value) if pd.notna(value) else np.nan


def _temperature_reference(age: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Farmer-validation brooding references used as deterministic X features."""
    age = pd.to_numeric(age, errors="coerce")
    target = pd.Series(np.select([age <= 4, age <= 8, age <= 15, age <= 21], [30.0, 28.0, 25.0, 22.0], default=20.0), index=age.index)
    lower = pd.Series(np.select([age <= 4, age <= 8, age <= 15, age <= 21], [28.0, 26.0, 23.0, 19.0], default=18.0), index=age.index)
    upper = pd.Series(np.select([age <= 4, age <= 8, age <= 15, age <= 21], [32.0, 30.0, 28.0, 25.0], default=22.0), index=age.index)
    return target, lower, upper


def _humidity_reference(age: pd.Series, building_id: str | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Farm-wide humidity reference; building identity is sensitivity-only."""
    age = pd.to_numeric(age, errors="coerce")
    target = pd.Series(np.select([age <= 7, age <= 14], [60.0, 55.0], default=50.0), index=age.index)
    lower = target.copy()
    upper = target + 10.0
    return target, lower, upper


def _trend(values: pd.Series, ages: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").notna() & pd.to_numeric(ages, errors="coerce").notna()
    if int(valid.sum()) < 2:
        return np.nan
    return float(np.polyfit(pd.to_numeric(ages[valid], errors="coerce"), pd.to_numeric(values[valid], errors="coerce"), 1)[0])


def _wet_bulb_stull(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Stull approximation used only where both observed inputs are present."""
    temperature = pd.to_numeric(temperature_c, errors="coerce")
    humidity = pd.to_numeric(humidity_pct, errors="coerce").clip(0, 100)
    return (
        temperature * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
        + np.arctan(temperature + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838 * humidity.pow(1.5) * np.arctan(0.023101 * humidity)
        - 4.686035
    )


def _latest_measurements(history: pd.DataFrame, targets: dict[int, float]) -> dict[str, float]:
    measured = history.loc[history["weight_measured"].fillna(False)].sort_values("age_day")
    result: dict[str, float] = {}
    for checkpoint in CHECKPOINTS:
        eligible = measured.loc[measured["age_day"].eq(checkpoint)]
        weight_g = _safe_float(eligible.iloc[-1]["bodyweight_kg"] * 1000) if not eligible.empty else np.nan
        target_g = targets.get(checkpoint, np.nan)
        result[f"weight_day{checkpoint}_g"] = weight_g
        result[f"ratio_day{checkpoint}"] = weight_g / target_g if pd.notna(weight_g) and target_g > 0 else np.nan
        result[f"gap_day{checkpoint}_g"] = weight_g - target_g if pd.notna(weight_g) and pd.notna(target_g) else np.nan
    for start, end in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]):
        start_weight = result[f"weight_day{start}_g"]
        end_weight = result[f"weight_day{end}_g"]
        result[f"gain_day{start}_{end}_g"] = end_weight - start_weight if pd.notna(start_weight) and pd.notna(end_weight) else np.nan
        result[f"adg_day{start}_{end}_g"] = (end_weight - start_weight) / (end - start) if pd.notna(start_weight) and pd.notna(end_weight) else np.nan
    return result


def _snapshot(dataset: CanaryDataset, cycle_id: str, building_id: str, review_day: int, outcome_g: float) -> dict[str, Any] | None:
    unit = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(cycle_id)
        & dataset.daily["building_id"].astype(str).eq(building_id)
    ].sort_values("age_day")
    history = unit.loc[unit["age_day"].le(review_day)].copy()
    if history.empty or int(history["age_day"].max()) != review_day:
        return None
    current = history.iloc[-1]
    measured = history.loc[history["weight_measured"].fillna(False)].sort_values("age_day")
    if measured.empty:
        return None
    latest = measured.iloc[-1]
    targets = _target_map(dataset)
    current_weight_g = float(latest["bodyweight_kg"] * 1000)
    latest_day = int(latest["age_day"])
    current_target_g = targets.get(latest_day, np.nan)
    trajectory = _latest_measurements(history, targets)
    available_weights = [trajectory[f"weight_day{day}_g"] for day in CHECKPOINTS if day <= review_day]
    available_adgs = [trajectory[f"adg_day{start}_{end}_g"] for start, end in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]) if end <= review_day]
    last_adg = available_adgs[-1] if available_adgs else np.nan
    prior_adg = available_adgs[-2] if len(available_adgs) > 1 else np.nan
    mortality = pd.to_numeric(history["mortality_daily"], errors="coerce")
    beginning = float(current["beginning_inventory"])
    population = float(current["population"])
    temperature = pd.to_numeric(history["temperature_avg_c"], errors="coerce")
    temperature_min = pd.to_numeric(history["temperature_min_c"], errors="coerce")
    temperature_max = pd.to_numeric(history["temperature_max_c"], errors="coerce")
    temperature_range = pd.to_numeric(history["temperature_range_c"], errors="coerce")
    humidity = pd.to_numeric(history["humidity_avg_pct"], errors="coerce")
    humidity_min = pd.to_numeric(history["humidity_min_pct"], errors="coerce")
    humidity_max = pd.to_numeric(history["humidity_max_pct"], errors="coerce")
    humidity_range = pd.to_numeric(history["humidity_range_pct"], errors="coerce")
    temperature_target, temperature_lower, temperature_upper = _temperature_reference(history["age_day"])
    humidity_target, humidity_lower, humidity_upper = _humidity_reference(history["age_day"], building_id)
    temperature_gap = temperature - temperature_target
    humidity_gap = humidity - humidity_target
    heat_excess = (temperature - temperature_upper).clip(lower=0)
    cold_excess = (temperature_lower - temperature).clip(lower=0)
    high_humidity_excess = (humidity - humidity_upper).clip(lower=0)
    low_humidity_excess = (humidity_lower - humidity).clip(lower=0)
    wet_bulb = _wet_bulb_stull(temperature, humidity)
    thi = 0.85 * temperature + 0.15 * wet_bulb
    recorded_environment = temperature.notna() & humidity.notna()
    compound_stress = (heat_excess > 0) & (high_humidity_excess > 0) & recorded_environment
    as_of = pd.Timestamp(current["record_date"])

    # Same-checkpoint peer summaries use only records already dated on or before
    # the focal snapshot.  They are operational context, never peer outcomes.
    peer_checkpoint = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(cycle_id)
        & dataset.daily["age_day"].eq(review_day)
        & dataset.daily["record_date"].le(as_of)
        & dataset.daily["weight_measured"].fillna(False)
    ]
    peer_weights_g = pd.to_numeric(peer_checkpoint["bodyweight_kg"], errors="coerce") * 1000
    cohort_mean_g = _safe_float(peer_weights_g.mean())
    cohort_std_g = _safe_float(peer_weights_g.std(ddof=0))

    return {
        "cycle_id": cycle_id,
        "building_id": building_id,
        "review_day": review_day,
        "as_of_date": as_of,
        "max_source_day_used": int(history["age_day"].max()),
        "outcome_day35_weight_g": outcome_g,
        "current_weight_g": current_weight_g,
        "current_target_g": current_target_g,
        "current_ratio_to_target": current_weight_g / current_target_g if current_target_g > 0 else np.nan,
        "current_gap_to_target_g": current_weight_g - current_target_g,
        "current_deficit_to_day35_g": current_weight_g - DAY35_TARGET_G,
        "current_gap_pct": (current_weight_g - current_target_g) / current_target_g * 100 if current_target_g > 0 else np.nan,
        "latest_measurement_day": latest_day,
        "measurement_staleness_days": review_day - latest_day,
        "weight_measurement_count": int(len(measured)),
        "available_checkpoint_count": int(sum(pd.notna(value) for value in available_weights)),
        "last_interval_adg_g_day": last_adg,
        "prior_interval_adg_g_day": prior_adg,
        "adg_acceleration_g_day2": last_adg - prior_adg if pd.notna(last_adg) and pd.notna(prior_adg) else np.nan,
        "placement_to_current_adg_g_day": (current_weight_g - 40.0) / latest_day if latest_day > 0 else np.nan,
        "survival_pct": population / beginning * 100,
        "beginning_inventory": beginning,
        "current_population": population,
        "log_beginning_inventory": float(np.log(beginning)),
        "population_loss_pct": (beginning - population) / beginning * 100,
        "mortality_cumulative_per_1000": float(mortality.sum(min_count=1) / beginning * 1000),
        "mortality_recent_3d_per_1000": float(mortality.tail(3).sum(min_count=1) / beginning * 1000),
        "mortality_recent_7d_per_1000": float(mortality.tail(7).sum(min_count=1) / beginning * 1000),
        "temperature_history_mean_c": _safe_float(temperature.mean()),
        "temperature_history_sd_c": _safe_float(temperature.std(ddof=0)),
        "temperature_recent_7d_mean_c": _safe_float(temperature.tail(7).mean()),
        "temperature_history_min_c": _safe_float(temperature_min.min()),
        "temperature_history_max_c": _safe_float(temperature_max.max()),
        "temperature_range_history_mean_c": _safe_float(temperature_range.mean()),
        "temperature_trend_c_day": _trend(temperature, history["age_day"]),
        "temperature_target_gap_mean_c": _safe_float(temperature_gap.mean()),
        "temperature_target_abs_error_mean_c": _safe_float(temperature_gap.abs().mean()),
        "heat_excess_degree_days": _safe_float(heat_excess.sum(min_count=1)),
        "cold_excess_degree_days": _safe_float(cold_excess.sum(min_count=1)),
        "heat_stress_day_pct": _safe_float((heat_excess.loc[temperature.notna()] > 0).mean() * 100),
        "cold_stress_day_pct": _safe_float((cold_excess.loc[temperature.notna()] > 0).mean() * 100),
        "temperature_coverage": float(temperature.notna().mean()),
        "humidity_history_mean_pct": _safe_float(humidity.mean()),
        "humidity_history_sd_pct": _safe_float(humidity.std(ddof=0)),
        "humidity_recent_7d_mean_pct": _safe_float(humidity.tail(7).mean()),
        "humidity_history_min_pct": _safe_float(humidity_min.min()),
        "humidity_history_max_pct": _safe_float(humidity_max.max()),
        "humidity_range_history_mean_pct": _safe_float(humidity_range.mean()),
        "humidity_trend_pct_day": _trend(humidity, history["age_day"]),
        "humidity_target_gap_mean_pct": _safe_float(humidity_gap.mean()),
        "humidity_target_abs_error_mean_pct": _safe_float(humidity_gap.abs().mean()),
        "high_humidity_excess_days": _safe_float(high_humidity_excess.sum(min_count=1)),
        "low_humidity_excess_days": _safe_float(low_humidity_excess.sum(min_count=1)),
        "high_humidity_day_pct": _safe_float((high_humidity_excess.loc[humidity.notna()] > 0).mean() * 100),
        "low_humidity_day_pct": _safe_float((low_humidity_excess.loc[humidity.notna()] > 0).mean() * 100),
        "humidity_coverage": float(humidity.notna().mean()),
        "thi_history_mean_c": _safe_float(thi.mean()),
        "thi_history_max_c": _safe_float(thi.max()),
        "thi_stress_day_pct": _safe_float((thi.loc[recorded_environment] >= 27.8).mean() * 100),
        "compound_heat_humidity_days": int(compound_stress.sum()),
        "cycle_checkpoint_mean_g": cohort_mean_g,
        "cycle_checkpoint_sd_g": cohort_std_g,
        "cycle_checkpoint_count": int(peer_weights_g.notna().sum()),
        "weight_minus_cycle_checkpoint_mean_g": current_weight_g - cohort_mean_g if pd.notna(cohort_mean_g) else np.nan,
        # Exported for sensitivity analysis only. Primary feature schemas below
        # deliberately exclude exact building and Tags/Lags identity.
        "is_lags_building": float(building_id.startswith("Lags")),
        **trajectory,
    }


def build_snapshots(
    dataset: CanaryDataset,
    development_cycles: tuple[str, ...] = DEVELOPMENT_CYCLES,
    audit_cycle: str = AUDIT_CYCLE,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid_cycles = {*development_cycles, audit_cycle}
    for cycle in dataset.cycles.itertuples(index=False):
        cycle_id = str(cycle.cycle_id)
        building_id = str(cycle.building_id)
        if cycle_id not in valid_cycles:
            continue
        unit = dataset.daily.loc[
            dataset.daily["cycle_id"].astype(str).eq(cycle_id)
            & dataset.daily["building_id"].astype(str).eq(building_id)
        ]
        day35 = unit.loc[unit["age_day"].eq(35) & unit["weight_measured"].fillna(False), "bodyweight_kg"]
        if day35.empty:
            continue
        outcome_g = float(day35.iloc[-1] * 1000)
        for review_day in CHECKPOINTS:
            row = _snapshot(dataset, cycle_id, building_id, review_day, outcome_g)
            if row is not None:
                row["role"] = "later_cycle_audit" if cycle_id == audit_cycle else "development"
                rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True)
    if not frame["max_source_day_used"].le(frame["review_day"]).all():
        raise AssertionError("Post-review-day information entered a snapshot.")
    for checkpoint in CHECKPOINTS:
        future = [f"weight_day{day}_g" for day in CHECKPOINTS if day > checkpoint]
        if future and frame.loc[frame["review_day"].eq(checkpoint), future].notna().any().any():
            raise AssertionError(f"Future checkpoint weight entered Day {checkpoint} snapshots.")
    return frame


def feature_columns(review_day: int, feature_set: str) -> list[str]:
    checkpoint_weights = [f"weight_day{day}_g" for day in CHECKPOINTS if day < review_day]
    checkpoint_ratios = [f"ratio_day{day}" for day in CHECKPOINTS if day <= review_day]
    checkpoint_gaps = [f"gap_day{day}_g" for day in CHECKPOINTS if day <= review_day]
    gains = [f"gain_day{start}_{end}_g" for start, end in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]) if end <= review_day]
    adgs = [f"adg_day{start}_{end}_g" for start, end in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]) if end <= review_day]
    current = ["current_weight_g", "current_ratio_to_target"]
    # For trajectory models, the current checkpoint is represented once as
    # current_weight_g and earlier checkpoints retain their own columns.  Ratio,
    # gain, and ADG versions remain available to compact/extended challengers,
    # but are not duplicated in the low-dimensional PLS/Ridge trajectory.
    trajectory = ["current_weight_g", *checkpoint_weights]
    compact = [
        *trajectory,
        "current_ratio_to_target",
        *checkpoint_ratios,
        *gains,
        *adgs,
        "last_interval_adg_g_day",
        "prior_interval_adg_g_day",
        "adg_acceleration_g_day2",
        "placement_to_current_adg_g_day",
        "survival_pct",
        "mortality_recent_7d_per_1000",
    ]
    extended = [
        *compact,
        "population_loss_pct",
        "mortality_cumulative_per_1000",
        "mortality_recent_3d_per_1000",
        "temperature_history_mean_c",
        "temperature_history_sd_c",
        "temperature_recent_7d_mean_c",
        "temperature_coverage",
        "humidity_history_mean_pct",
        "humidity_history_sd_pct",
        "humidity_recent_7d_mean_pct",
        "humidity_coverage",
        "cycle_checkpoint_mean_g",
        "cycle_checkpoint_sd_g",
        "cycle_checkpoint_count",
        "weight_minus_cycle_checkpoint_mean_g",
    ]
    poultry = [
        *extended,
        "current_gap_to_target_g",
        "current_deficit_to_day35_g",
        "current_gap_pct",
        *checkpoint_gaps,
        "beginning_inventory",
        "current_population",
        "log_beginning_inventory",
        "temperature_history_min_c",
        "temperature_history_max_c",
        "temperature_range_history_mean_c",
        "temperature_trend_c_day",
        "temperature_target_gap_mean_c",
        "temperature_target_abs_error_mean_c",
        "heat_excess_degree_days",
        "cold_excess_degree_days",
        "heat_stress_day_pct",
        "cold_stress_day_pct",
        "humidity_history_min_pct",
        "humidity_history_max_pct",
        "humidity_range_history_mean_pct",
        "humidity_trend_pct_day",
        "humidity_target_gap_mean_pct",
        "humidity_target_abs_error_mean_pct",
        "high_humidity_excess_days",
        "low_humidity_excess_days",
        "high_humidity_day_pct",
        "low_humidity_day_pct",
        "thi_history_mean_c",
        "thi_history_max_c",
        "thi_stress_day_pct",
        "compound_heat_humidity_days",
    ]
    poultry_core = [
        *trajectory,
        "current_gap_to_target_g",
        "last_interval_adg_g_day",
        "survival_pct",
        "log_beginning_inventory",
        "temperature_target_abs_error_mean_c",
        "heat_excess_degree_days",
        "cold_excess_degree_days",
        "humidity_target_abs_error_mean_pct",
        "high_humidity_day_pct",
        "thi_history_max_c",
        "compound_heat_humidity_days",
        "temperature_coverage",
        "humidity_coverage",
    ]
    mapping = {"current": current, "trajectory": trajectory, "compact": compact, "extended": extended, "poultry": poultry, "poultry_core": poultry_core}
    return list(dict.fromkeys(mapping[feature_set]))


def _options(candidate: Candidate) -> list[dict[str, Any]]:
    if candidate.family == "recent_adg":
        return [{"recent_weight": value} for value in (0.0, 0.25, 0.5, 0.75)]
    if candidate.family == "ridge":
        return [{"alpha": value} for value in (1.0, 10.0, 50.0, 100.0)]
    if candidate.family == "pls_multi":
        return [{"components": value} for value in (1, 2, 3)]
    if candidate.family == "svr":
        return [{"C": c, "epsilon": epsilon} for c in (0.03, 0.1, 0.3, 1.0) for epsilon in (25.0, 50.0, 100.0)]
    if candidate.family == "huber":
        return [{"epsilon": value, "alpha": alpha} for value in (1.2, 1.35, 1.5) for alpha in (0.001, 0.01)]
    if candidate.family == "elastic_net":
        return [{"alpha": alpha, "l1_ratio": ratio} for alpha in (1.0, 10.0, 25.0) for ratio in (0.1, 0.5, 0.9)]
    if candidate.family in {"extra_trees", "random_forest"}:
        return [{"depth": depth, "leaf": leaf, "max_features": features} for depth, leaf, features in ((2, 2, 0.7), (3, 2, 1.0), (4, 3, 0.7), (None, 4, 1.0))]
    if candidate.family == "gradient_boosting":
        return [{"trees": trees, "rate": rate, "depth": depth, "leaf": leaf, "loss": loss} for trees, rate, depth, leaf, loss in ((50, 0.03, 1, 3, "squared_error"), (75, 0.03, 1, 4, "huber"), (75, 0.03, 2, 4, "huber"), (100, 0.02, 1, 4, "squared_error"))]
    if candidate.family == "hist_gradient_boosting":
        return [{"iterations": iterations, "rate": rate, "leaf": leaf, "l2": l2} for iterations, rate, leaf, l2 in ((75, 0.04, 5, 10.0), (100, 0.03, 5, 25.0), (100, 0.03, 8, 25.0), (150, 0.02, 8, 50.0))]
    if candidate.family in {"xgboost", "lightgbm", "catboost"}:
        return [
            {"trees": 75, "rate": 0.03, "depth": 1, "leaf": 5, "l2": 20.0},
            {"trees": 120, "rate": 0.025, "depth": 2, "leaf": 5, "l2": 30.0},
            {"trees": 150, "rate": 0.02, "depth": 2, "leaf": 8, "l2": 50.0},
        ]
    if candidate.family == "blend":
        return [{"blend_weight": value} for value in (0.25, 0.5, 0.75)]
    return [{}]


def _pipeline(candidate: Candidate, parameters: dict[str, Any]) -> Pipeline:
    linear_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if candidate.family == "ols":
        model: Any = LinearRegression()
    elif candidate.family == "pls" or candidate.family == "blend":
        model = PLSRegression(n_components=1, scale=False, max_iter=1000)
    elif candidate.family == "pls_multi":
        model = PLSRegression(n_components=int(parameters.get("components", 1)), scale=False, max_iter=1000)
    elif candidate.family == "ridge":
        model = Ridge(alpha=float(parameters.get("alpha", 10.0)))
    elif candidate.family == "svr":
        model = SVR(kernel="linear", C=float(parameters.get("C", 0.1)), epsilon=float(parameters.get("epsilon", 50.0)))
    elif candidate.family == "huber":
        model = HuberRegressor(epsilon=float(parameters.get("epsilon", 1.35)), alpha=float(parameters.get("alpha", 0.01)), max_iter=4000)
    elif candidate.family == "elastic_net":
        model = ElasticNet(alpha=float(parameters.get("alpha", 10.0)), l1_ratio=float(parameters.get("l1_ratio", 0.5)), max_iter=20000, random_state=SEED)
    elif candidate.family == "extra_trees":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", ExtraTreesRegressor(n_estimators=300, max_depth=parameters.get("depth", 3), min_samples_leaf=int(parameters.get("leaf", 3)), max_features=parameters.get("max_features", 0.7), random_state=SEED, n_jobs=1)),
        ])
    elif candidate.family == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=parameters.get("depth", 3), min_samples_leaf=int(parameters.get("leaf", 3)), max_features=parameters.get("max_features", 0.7), random_state=SEED, n_jobs=1)),
        ])
    elif candidate.family == "gradient_boosting":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", GradientBoostingRegressor(n_estimators=int(parameters.get("trees", 75)), learning_rate=float(parameters.get("rate", 0.03)), max_depth=int(parameters.get("depth", 1)), min_samples_leaf=int(parameters.get("leaf", 4)), loss=str(parameters.get("loss", "huber")), random_state=SEED)),
        ])
    elif candidate.family == "hist_gradient_boosting":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", HistGradientBoostingRegressor(max_iter=int(parameters.get("iterations", 100)), learning_rate=float(parameters.get("rate", 0.03)), max_leaf_nodes=int(parameters.get("leaf", 5)), min_samples_leaf=4, l2_regularization=float(parameters.get("l2", 25.0)), loss="squared_error", random_state=SEED)),
        ])
    elif candidate.family == "xgboost":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", XGBRegressor(
                n_estimators=int(parameters.get("trees", 120)),
                learning_rate=float(parameters.get("rate", 0.025)),
                max_depth=int(parameters.get("depth", 2)),
                min_child_weight=float(parameters.get("leaf", 5)),
                reg_lambda=float(parameters.get("l2", 30.0)),
                reg_alpha=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=SEED,
                n_jobs=1,
                verbosity=0,
            )),
        ])
    elif candidate.family == "lightgbm":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", LGBMRegressor(
                n_estimators=int(parameters.get("trees", 120)),
                learning_rate=float(parameters.get("rate", 0.025)),
                max_depth=int(parameters.get("depth", 2)),
                num_leaves=max(2, 2 ** int(parameters.get("depth", 2)) - 1),
                min_child_samples=int(parameters.get("leaf", 5)),
                reg_lambda=float(parameters.get("l2", 30.0)),
                reg_alpha=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )),
        ])
    elif candidate.family == "catboost":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", CatBoostRegressor(
                iterations=int(parameters.get("trees", 120)),
                learning_rate=float(parameters.get("rate", 0.025)),
                depth=int(parameters.get("depth", 2)),
                l2_leaf_reg=float(parameters.get("l2", 30.0)),
                random_seed=SEED,
                loss_function="RMSE",
                random_strength=0.5,
                allow_writing_files=False,
                thread_count=1,
                verbose=False,
            )),
        ])
    else:
        raise ValueError(candidate.family)
    return Pipeline([*linear_steps, ("model", model)])


def _fit_entry(frame: pd.DataFrame, candidate: Candidate, parameters: dict[str, Any], review_day: int) -> dict[str, Any]:
    features = feature_columns(review_day, candidate.feature_set)
    current = frame["current_weight_g"].to_numpy(float)
    if candidate.family == "baseline":
        return {"kind": "baseline", "remaining_gain_g": float(np.mean(frame["outcome_day35_weight_g"].to_numpy(float) - current)), "features": features}
    if candidate.family == "pace":
        return {"kind": "pace", "features": features}
    if candidate.family == "target_gain":
        return {"kind": "target_gain", "features": features}
    if candidate.family == "historical_ratio":
        ratio = frame["outcome_day35_weight_g"].to_numpy(float) / current
        return {"kind": "historical_ratio", "growth_ratio": float(np.mean(ratio)), "features": features}
    if candidate.family == "recent_adg":
        horizon = float(35 - review_day)
        expected_adg = float(np.mean((frame["outcome_day35_weight_g"].to_numpy(float) - current) / horizon))
        return {"kind": "recent_adg", "expected_adg_g_day": expected_adg, "recent_weight": float(parameters.get("recent_weight", 0.25)), "features": features}
    target = frame["outcome_day35_weight_g"].to_numpy(float)
    if candidate.target_form == "remaining":
        target = target - current
    pipeline = _pipeline(candidate, parameters)
    pipeline.fit(frame[features], target)
    if candidate.family == "blend":
        return {
            "kind": "blend",
            "pipeline": pipeline,
            "remaining_gain_g": float(np.mean(frame["outcome_day35_weight_g"].to_numpy(float) - current)),
            "blend_weight": float(parameters.get("blend_weight", 0.5)),
            "features": features,
        }
    return {"kind": "model", "pipeline": pipeline, "target_form": candidate.target_form, "features": features}


def _predict_entry(entry: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    current = frame["current_weight_g"].to_numpy(float)
    if entry["kind"] == "baseline":
        predicted = current + float(entry["remaining_gain_g"])
    elif entry["kind"] == "pace":
        predicted = frame["current_ratio_to_target"].to_numpy(float) * DAY35_TARGET_G
    elif entry["kind"] == "target_gain":
        predicted = current + (DAY35_TARGET_G - frame["current_target_g"].to_numpy(float))
    elif entry["kind"] == "historical_ratio":
        predicted = current * float(entry["growth_ratio"])
    elif entry["kind"] == "recent_adg":
        recent = frame["last_interval_adg_g_day"].fillna(frame["placement_to_current_adg_g_day"]).to_numpy(float)
        weight = float(entry["recent_weight"])
        blended_adg = weight * recent + (1.0 - weight) * float(entry["expected_adg_g_day"])
        predicted = current + (35.0 - frame["review_day"].to_numpy(float)) * blended_adg
    elif entry["kind"] == "blend":
        learned = np.asarray(entry["pipeline"].predict(frame[entry["features"]])).reshape(-1)
        baseline = current + float(entry["remaining_gain_g"])
        weight = float(entry["blend_weight"])
        predicted = weight * learned + (1.0 - weight) * baseline
    else:
        raw = np.asarray(entry["pipeline"].predict(frame[entry["features"]])).reshape(-1)
        predicted = current + raw if entry["target_form"] == "remaining" else raw
    return np.clip(predicted, 500.0, 2500.0)


def _cycle_macro_rmse(actual: np.ndarray, predicted: np.ndarray, cycles: np.ndarray) -> float:
    values = []
    for cycle_id in pd.unique(cycles):
        mask = cycles == cycle_id
        values.append(float(mean_squared_error(actual[mask], predicted[mask]) ** 0.5))
    return float(np.mean(values))


def _tune(train: pd.DataFrame, candidate: Candidate, review_day: int) -> dict[str, Any]:
    options = _options(candidate)
    if len(options) == 1:
        return options[0]
    groups = train["cycle_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < 3:
        return options[0]
    actual = train["outcome_day35_weight_g"].to_numpy(float)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for option in options:
        predicted = np.full(len(train), np.nan)
        for inner_train, inner_test in LeaveOneGroupOut().split(train, groups=groups):
            entry = _fit_entry(train.iloc[inner_train], candidate, option, review_day)
            predicted[inner_test] = _predict_entry(entry, train.iloc[inner_test])
        score = _cycle_macro_rmse(actual, predicted, groups)
        scored.append((score, json.dumps(option, sort_keys=True), option))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def evaluate_candidate(development: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    groups = development["cycle_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    for outer_train, outer_test in LeaveOneGroupOut().split(development, groups=groups):
        held_cycle = str(development.iloc[outer_test]["cycle_id"].iloc[0])
        for review_day in CHECKPOINTS:
            train = development.iloc[outer_train].loc[lambda frame: frame["review_day"].eq(review_day)].reset_index(drop=True)
            test = development.iloc[outer_test].loc[lambda frame: frame["review_day"].eq(review_day)].reset_index(drop=True)
            selected = _tune(train, candidate, review_day)
            entry = _fit_entry(train, candidate, selected, review_day)
            predicted = _predict_entry(entry, test)
            parameters.append({"held_out_cycle": held_cycle, "review_day": review_day, "parameters": selected})
            for source, prediction in zip(test.to_dict("records"), predicted):
                rows.append({
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "cycle_id": source["cycle_id"],
                    "building_id": source["building_id"],
                    "review_day": review_day,
                    "as_of_date": source["as_of_date"],
                    "actual_g": source["outcome_day35_weight_g"],
                    "predicted_g": float(prediction),
                    "error_g": float(prediction - source["outcome_day35_weight_g"]),
                })
    return pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True), parameters


def _bootstrap_intervals(predictions: pd.DataFrame, iterations: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    cycles = predictions["cycle_id"].drop_duplicates().to_numpy()
    rmse_values: list[float] = []
    r2_values: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(cycles, size=len(cycles), replace=True)
        pieces = [predictions.loc[predictions["cycle_id"].eq(cycle)] for cycle in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        rmse_values.append(float(mean_squared_error(boot["actual_g"], boot["predicted_g"]) ** 0.5))
        r2_values.append(float(r2_score(boot["actual_g"], boot["predicted_g"])))
    return {
        "rmse_95ci_low_g": float(np.quantile(rmse_values, 0.025)),
        "rmse_95ci_high_g": float(np.quantile(rmse_values, 0.975)),
        "r2_95ci_low": float(np.quantile(r2_values, 0.025)),
        "r2_95ci_high": float(np.quantile(r2_values, 0.975)),
    }


def summarize(predictions: pd.DataFrame, bootstrap: bool = False) -> dict[str, Any]:
    actual = predictions["actual_g"].to_numpy(float)
    predicted = predictions["predicted_g"].to_numpy(float)
    errors = predicted - actual
    by_cycle = predictions.groupby("cycle_id").apply(
        lambda frame: pd.Series({
            "rmse_g": mean_squared_error(frame["actual_g"], frame["predicted_g"]) ** 0.5,
            "mae_g": mean_absolute_error(frame["actual_g"], frame["predicted_g"]),
        }),
        include_groups=False,
    )
    actual_above = actual >= DAY35_TARGET_G
    predicted_above = predicted >= DAY35_TARGET_G
    unique_outcomes = predictions[["cycle_id", "building_id", "actual_g"]].drop_duplicates()
    unique_target_hits = int((unique_outcomes["actual_g"] >= DAY35_TARGET_G).sum())
    majority_side_accuracy = float(max(unique_target_hits, len(unique_outcomes) - unique_target_hits) / len(unique_outcomes))
    summary: dict[str, Any] = {
        "rows": int(len(predictions)),
        "independent_building_cycles": int(predictions[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "rmse_g": float(mean_squared_error(actual, predicted) ** 0.5),
        "cycle_macro_rmse_g": float(by_cycle["rmse_g"].mean()),
        "mae_g": float(mean_absolute_error(actual, predicted)),
        "cycle_macro_mae_g": float(by_cycle["mae_g"].mean()),
        "r2": float(r2_score(actual, predicted)),
        "bias_g": float(errors.mean()),
        "within_100g_rate": float(np.mean(np.abs(errors) <= 100)),
        "within_200g_rate": float(np.mean(np.abs(errors) <= 200)),
        "worst_cycle_rmse_g": float(by_cycle["rmse_g"].max()),
        "fold_rmse_sd_g": float(by_cycle["rmse_g"].std(ddof=1)),
        "fold_rmse_se_g": float(by_cycle["rmse_g"].std(ddof=1) / np.sqrt(len(by_cycle))),
        "target_side_accuracy": float(np.mean(actual_above == predicted_above)),
        "majority_side_accuracy": majority_side_accuracy,
        "below_target_recall": float(np.mean(~predicted_above[~actual_above])) if (~actual_above).any() else np.nan,
        "at_or_above_target_recall": float(np.mean(predicted_above[actual_above])) if actual_above.any() else np.nan,
        "actual_target_hits": unique_target_hits,
    }
    if bootstrap:
        summary.update(_bootstrap_intervals(predictions))
    return summary


def checkpoint_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"review_day": int(review_day), **summarize(group)}
        for review_day, group in predictions.groupby("review_day", sort=True)
    ])


def identity_sensitivity(development: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate site/building remaining-gain baselines outside selection."""

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    groups = development["cycle_id"].astype(str).to_numpy()
    for sensitivity, identity in (
        ("site_group", development["building_id"].astype(str).str.split().str[0]),
        ("exact_building", development["building_id"].astype(str)),
    ):
        working = development.assign(_identity=identity.to_numpy())
        rows: list[dict[str, Any]] = []
        for train_index, test_index in LeaveOneGroupOut().split(working, groups=groups):
            outer_train, outer_test = working.iloc[train_index], working.iloc[test_index]
            for review_day in CHECKPOINTS:
                train = outer_train.loc[outer_train["review_day"].eq(review_day)]
                test = outer_test.loc[outer_test["review_day"].eq(review_day)]
                gain = train["outcome_day35_weight_g"] - train["current_weight_g"]
                means = pd.DataFrame({"identity": train["_identity"], "gain": gain}).groupby("identity")["gain"].mean()
                fallback = float(gain.mean())
                predicted = test["current_weight_g"].to_numpy(float) + test["_identity"].map(means).fillna(fallback).to_numpy(float)
                for source, prediction in zip(test.to_dict("records"), predicted):
                    rows.append({
                        "sensitivity": sensitivity,
                        "cycle_id": source["cycle_id"],
                        "building_id": source["building_id"],
                        "review_day": review_day,
                        "actual_g": float(source["outcome_day35_weight_g"]),
                        "predicted_g": float(prediction),
                        "error_g": float(prediction - source["outcome_day35_weight_g"]),
                    })
        prediction_frame = pd.DataFrame(rows)
        all_rows.extend(rows)
        summaries.append({"sensitivity": sensitivity, "eligible_for_primary_selection": False, **summarize(prediction_frame)})
    return pd.DataFrame(all_rows), pd.DataFrame(summaries)


def leave_one_building_label_out(
    development: pd.DataFrame, candidate: Candidate
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = development["building_id"].astype(str).to_numpy()
    for train_index, test_index in LeaveOneGroupOut().split(development, groups=labels):
        outer_train, outer_test = development.iloc[train_index], development.iloc[test_index]
        for review_day in CHECKPOINTS:
            train = outer_train.loc[outer_train["review_day"].eq(review_day)].reset_index(drop=True)
            test = outer_test.loc[outer_test["review_day"].eq(review_day)].reset_index(drop=True)
            parameters = _tune(train, candidate, review_day)
            prediction = _predict_entry(_fit_entry(train, candidate, parameters, review_day), test)
            for source, value in zip(test.to_dict("records"), prediction):
                rows.append({
                    "held_out_building_label": source["building_id"],
                    "cycle_id": source["cycle_id"],
                    "building_id": source["building_id"],
                    "review_day": review_day,
                    "actual_g": float(source["outcome_day35_weight_g"]),
                    "predicted_g": float(value),
                    "error_g": float(value - source["outcome_day35_weight_g"]),
                })
    predictions = pd.DataFrame(rows)
    return predictions, summarize(predictions)


def _conformal_for_champion(development: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = development["cycle_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for outer_train, outer_test in LeaveOneGroupOut().split(development, groups=groups):
        held_cycle = str(development.iloc[outer_test]["cycle_id"].iloc[0])
        for review_day in CHECKPOINTS:
            train = development.iloc[outer_train].loc[lambda frame: frame["review_day"].eq(review_day)].reset_index(drop=True)
            test = development.iloc[outer_test].loc[lambda frame: frame["review_day"].eq(review_day)].reset_index(drop=True)
            selected = _tune(train, candidate, review_day)
            # Inner cycle OOF residuals calibrate the outer held-out interval.
            train_groups = train["cycle_id"].astype(str).to_numpy()
            calibration_predictions = np.full(len(train), np.nan)
            for inner_train, inner_test in LeaveOneGroupOut().split(train, groups=train_groups):
                inner_entry = _fit_entry(train.iloc[inner_train], candidate, selected, review_day)
                calibration_predictions[inner_test] = _predict_entry(inner_entry, train.iloc[inner_test])
            residuals = np.abs(calibration_predictions - train["outcome_day35_weight_g"].to_numpy(float))
            q80 = float(np.quantile(residuals, min(1.0, np.ceil((len(residuals) + 1) * 0.80) / len(residuals)), method="higher"))
            q90 = float(np.quantile(residuals, min(1.0, np.ceil((len(residuals) + 1) * 0.90) / len(residuals)), method="higher"))
            entry = _fit_entry(train, candidate, selected, review_day)
            predicted = _predict_entry(entry, test)
            for source, prediction in zip(test.to_dict("records"), predicted):
                actual = float(source["outcome_day35_weight_g"])
                rows.append({
                    "cycle_id": source["cycle_id"], "building_id": source["building_id"], "review_day": review_day,
                    "actual_g": actual, "predicted_g": float(prediction),
                    "lower_80_g": float(prediction - q80), "upper_80_g": float(prediction + q80),
                    "lower_90_g": float(prediction - q90), "upper_90_g": float(prediction + q90),
                    "covered_80": bool(abs(actual - prediction) <= q80), "covered_90": bool(abs(actual - prediction) <= q90),
                    "target_status_80": "Likely below" if prediction + q80 < DAY35_TARGET_G else "Likely meets" if prediction - q80 >= DAY35_TARGET_G else "Uncertain",
                })
            calibration_rows.append({"held_out_cycle": held_cycle, "review_day": review_day, "calibration_rows": int(len(residuals)), "q80_g": q80, "q90_g": q90, "checkpoint_specific_reliable": bool(len(residuals) >= 20)})
    intervals = pd.DataFrame(rows)
    calibration = pd.DataFrame(calibration_rows)
    return intervals, calibration


def _temporal_stress(development: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = list(DEVELOPMENT_CYCLES)
    rows: list[dict[str, Any]] = []
    for index in range(2, len(ordered)):
        training_cycles = ordered[:index]
        test_cycle = ordered[index]
        for review_day in CHECKPOINTS:
            train = development.loc[development["cycle_id"].isin(training_cycles) & development["review_day"].eq(review_day)].reset_index(drop=True)
            test = development.loc[development["cycle_id"].eq(test_cycle) & development["review_day"].eq(review_day)].reset_index(drop=True)
            parameters = _tune(train, candidate, review_day)
            entry = _fit_entry(train, candidate, parameters, review_day)
            predicted = _predict_entry(entry, test)
            for source, prediction in zip(test.to_dict("records"), predicted):
                rows.append({"training_cycles": ",".join(training_cycles), "cycle_id": test_cycle, "building_id": source["building_id"], "review_day": review_day, "actual_g": source["outcome_day35_weight_g"], "predicted_g": float(prediction), "error_g": float(prediction - source["outcome_day35_weight_g"])})
    frame = pd.DataFrame(rows)
    return frame, summarize(frame) if not frame.empty else {}


def _fit_final(development: pd.DataFrame, candidate: Candidate) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    fitted: dict[int, dict[str, Any]] = {}
    parameters: list[dict[str, Any]] = []
    for review_day in CHECKPOINTS:
        train = development.loc[development["review_day"].eq(review_day)].reset_index(drop=True)
        selected = _tune(train, candidate, review_day)
        fitted[review_day] = _fit_entry(train, candidate, selected, review_day)
        parameters.append({"review_day": review_day, "parameters": selected, "features": fitted[review_day]["features"]})
    return fitted, parameters


def _predict_bundle(bundle: dict[int, dict[str, Any]], frame: pd.DataFrame) -> np.ndarray:
    predicted = np.full(len(frame), np.nan)
    for review_day, index in frame.groupby("review_day").groups.items():
        predicted[index] = _predict_entry(bundle[int(review_day)], frame.loc[index])
    return predicted


def _permutation_shap(development: pd.DataFrame, audit: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle, _ = _fit_final(development, candidate)
    global_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for review_day in CHECKPOINTS:
        train = development.loc[development["review_day"].eq(review_day)].reset_index(drop=True)
        explain = pd.concat([
            train.sample(min(12, len(train)), random_state=SEED),
            audit.loc[audit["review_day"].eq(review_day)],
        ], ignore_index=True)
        entry = bundle[review_day]
        features = entry["features"]
        background = train[features].copy()
        background = background.fillna(background.median(numeric_only=True)).fillna(0.0)
        background = background.sample(min(10, len(background)), random_state=SEED)
        x_explain = explain[features].copy()
        x_explain = x_explain.fillna(background.median(numeric_only=True)).fillna(0.0)

        def predict_array(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(values, columns=features)
            return _predict_entry(entry, frame)

        explainer = shap.Explainer(predict_array, background.to_numpy(float), algorithm="permutation", seed=SEED)
        explanation = explainer(x_explain.to_numpy(float), max_evals=max(2 * len(features) + 1, 11), silent=True)
        shap_values = np.asarray(explanation.values)
        for column_index, feature in enumerate(features):
            global_rows.append({"review_day": review_day, "feature": feature, "mean_abs_shap_g": float(np.mean(np.abs(shap_values[:, column_index]))), "mean_shap_g": float(np.mean(shap_values[:, column_index]))})
            for row_index in range(len(explain)):
                local_rows.append({
                    "review_day": review_day,
                    "cycle_id": explain.iloc[row_index]["cycle_id"],
                    "building_id": explain.iloc[row_index]["building_id"],
                    "role": explain.iloc[row_index]["role"],
                    "feature": feature,
                    "feature_value": float(x_explain.iloc[row_index, column_index]),
                    "shap_value_g": float(shap_values[row_index, column_index]),
                    "base_value_g": float(np.asarray(explanation.base_values)[row_index]),
                })
    global_frame = pd.DataFrame(global_rows).groupby("feature", as_index=False).agg(mean_abs_shap_g=("mean_abs_shap_g", "mean"), mean_shap_g=("mean_shap_g", "mean"), checkpoints=("review_day", "nunique")).sort_values("mean_abs_shap_g", ascending=False)
    return global_frame, pd.DataFrame(local_rows)


def _data_quality(dataset: CanaryDataset, snapshots: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    daily = dataset.daily
    outcomes = snapshots[["cycle_id", "building_id", "outcome_day35_weight_g", "role"]].drop_duplicates()
    checks = [
        {"check": "Canonical building-day key uniqueness", "failed_rows": int(daily.duplicated(["cycle_id", "building_id", "age_day"]).sum()), "severity": "critical"},
        {"check": "Positive beginning inventory", "failed_rows": int((daily["beginning_inventory"] <= 0).sum()), "severity": "critical"},
        {"check": "Population does not exceed beginning inventory", "failed_rows": int((daily["population"] > daily["beginning_inventory"]).sum()), "severity": "high"},
        {"check": "Non-negative daily mortality", "failed_rows": int((daily["mortality_daily"] < 0).sum()), "severity": "high"},
        {"check": "Observed bodyweight within 30-5000 g", "failed_rows": int(((daily.loc[daily["weight_measured"], "bodyweight_kg"] < 0.03) | (daily.loc[daily["weight_measured"], "bodyweight_kg"] > 5.0)).sum()), "severity": "high"},
        {"check": "Snapshot uses no future source day", "failed_rows": int((snapshots["max_source_day_used"] > snapshots["review_day"]).sum()), "severity": "critical"},
        {"check": "Development and audit cycle separation", "failed_rows": int(snapshots.loc[snapshots["role"].eq("development"), "cycle_id"].eq(AUDIT_CYCLE).sum()), "severity": "critical"},
    ]
    profile = {
        "source_rows": int(dataset.quality.source_rows),
        "canonical_building_days": int(dataset.quality.canonical_rows),
        "unique_building_days": int(daily[["cycle_id", "building_id", "age_day"]].drop_duplicates().shape[0]),
        "total_building_cycles": int(dataset.cycles[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "development_outcomes": int(outcomes.loc[outcomes["role"].eq("development")].shape[0]),
        "later_cycle_outcomes": int(outcomes.loc[outcomes["role"].eq("later_cycle_audit")].shape[0]),
        "development_snapshots": int(snapshots["role"].eq("development").sum()),
        "temperature_coverage_pct": float(dataset.quality.temperature_coverage_pct),
        "humidity_coverage_pct": float(dataset.quality.humidity_coverage_pct),
        "observed_weight_measurement_days": int(dataset.quality.weight_measurement_days),
        "day35_target_hits_development": int((outcomes.loc[outcomes["role"].eq("development"), "outcome_day35_weight_g"] >= DAY35_TARGET_G).sum()),
        "day35_outcome_min_g": float(outcomes["outcome_day35_weight_g"].min()),
        "day35_outcome_max_g": float(outcomes["outcome_day35_weight_g"].max()),
        "feed_policy": "Excluded from primary candidates because source units remain unresolved.",
    }
    return profile, pd.DataFrame(checks)


def _data_dictionary() -> pd.DataFrame:
    definitions = {
        "current_weight_g": "Most recent actually observed building average weight by the review date",
        "current_ratio_to_target": "Observed weight divided by the farm reference at the same measurement age",
        "current_gap/deficit": "Observed weight deviation from the same-age farm target and from the 1,800 g Day-35 goal",
        "weight_dayN_g": "Actually observed checkpoint weight; future checkpoint columns remain missing",
        "gain/adg": "Observed absolute gain or average daily gain between completed checkpoints",
        "survival/mortality": "Population and mortality history accumulated only through the review date",
        "temperature/humidity": "Observed history, target deviation, extremes, trends, duration, THI, and compound stress through the review date",
        "cycle_checkpoint": "Same-cycle checkpoint measurements already recorded by the snapshot date; no peer outcomes",
        "outcome_day35_weight_g": "Actually observed average building weight on production Day 35",
    }
    return pd.DataFrame([{"field_family": key, "definition": value, "timing_rule": "X: on/before review date" if key != "outcome_day35_weight_g" else "Y: available at Day 35", "leakage_control": "Future observations excluded"} for key, value in definitions.items()])


def _plots(output: Path, development: pd.DataFrame, predictions: pd.DataFrame, champion_predictions: pd.DataFrame, comparison: pd.DataFrame, checkpoint: pd.DataFrame, intervals: pd.DataFrame, shap_global: pd.DataFrame, shap_local: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    blue, dark, orange, pink = "#2A6F97", "#263238", "#D17A22", "#A64D79"

    top = comparison.head(5).sort_values("cycle_macro_rmse_g")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [blue if rank == 1 else "#A9C6D8" for rank in top["rank"]]
    ax.barh(top["candidate"].str.replace("_", " "), top["cycle_macro_rmse_g"], color=colors)
    ax.set(xlabel="Cycle-macro RMSE (g)", ylabel="")
    ax.set_title("Day 35 bodyweight model comparison", pad=24)
    ax.set_xlim(0, max(top["cycle_macro_rmse_g"]) * 1.18)
    for index, value in enumerate(top["cycle_macro_rmse_g"]):
        ax.text(value + 4, index, f"{value:.0f} g", va="center", color=dark)
    fig.text(0.5, 0.92, "Nested leave-one-harvest-cycle-out validation; lower is better", ha="center", fontsize=9, color="#5F6B73")
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(output / "model_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter(champion_predictions["actual_g"], champion_predictions["predicted_g"], c=champion_predictions["review_day"], cmap="viridis", alpha=0.75, edgecolor="white", linewidth=0.4)
    bounds = [min(champion_predictions[["actual_g", "predicted_g"]].min()) - 30, max(champion_predictions[["actual_g", "predicted_g"]].max()) + 30]
    ax.plot(bounds, bounds, "--", color=dark, label="Perfect prediction")
    ax.axhline(DAY35_TARGET_G, color=orange, linestyle=":", label="1,800 g target")
    ax.axvline(DAY35_TARGET_G, color=orange, linestyle=":")
    ax.set(xlim=bounds, ylim=bounds, title="Held-out actual versus predicted Day 35 weight", xlabel="Actual Day 35 weight (g)", ylabel="Predicted Day 35 weight (g)")
    fig.colorbar(scatter, ax=ax, label="Review day"); ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(output / "actual_vs_predicted.png", dpi=180); plt.close(fig)

    external_families = {"xgboost", "lightgbm", "catboost"}
    best_external = str(comparison.loc[comparison["family"].isin(external_families)].iloc[0]["candidate"])
    learned_families = set(comparison["family"]) - {"baseline", "target_gain", "historical_ratio", "recent_adg", "pace"}
    best_learned = str(comparison.loc[comparison["family"].isin(learned_families)].iloc[0]["candidate"])
    option_names = [champion_predictions["candidate"].iloc[0], best_learned, best_external, "target_gap_preserving"]
    option_labels = {
        option_names[0]: "One-SE champion: expected remaining growth",
        best_learned: "Lowest-error trajectory model",
        best_external: "Best new boosting model",
        "target_gap_preserving": "Naive target-curve gain",
    }
    day14_options = predictions.loc[predictions["candidate"].isin(option_names) & predictions["review_day"].eq(14)].copy()
    common_low = float(day14_options[["actual_g", "predicted_g"]].min().min() - 40)
    common_high = float(day14_options[["actual_g", "predicted_g"]].max().max() + 40)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, option_names):
        frame = day14_options.loc[day14_options["candidate"].eq(name)]
        rmse = float(mean_squared_error(frame["actual_g"], frame["predicted_g"]) ** 0.5)
        score = float(r2_score(frame["actual_g"], frame["predicted_g"]))
        ax.scatter(frame["actual_g"], frame["predicted_g"], color=blue, alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.plot([common_low, common_high], [common_low, common_high], "--", color=dark, linewidth=1)
        ax.axhline(DAY35_TARGET_G, color=orange, linestyle=":", linewidth=1)
        ax.axvline(DAY35_TARGET_G, color=orange, linestyle=":", linewidth=1)
        ax.set_title(f"{option_labels[name]}\nRMSE {rmse:.0f} g · R² {score:.2f}", fontsize=11)
        ax.set_xlim(common_low, common_high); ax.set_ylim(common_low, common_high)
    for ax in axes[:, 0]: ax.set_ylabel("Projected Day 35 weight (g)")
    for ax in axes[-1, :]: ax.set_xlabel("Actual Day 35 weight (g)")
    fig.suptitle("Actual versus projected Day 35 bodyweight from Day 14", fontsize=15, y=0.99)
    fig.text(0.5, 0.955, "Complete harvest-cycle holdouts; each point is one building-cycle (n=31)", ha="center", fontsize=9, color="#5F6B73")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(output / "actual_vs_projected_day14_options.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(champion_predictions["predicted_g"], champion_predictions["error_g"], c=champion_predictions["review_day"], cmap="viridis", alpha=0.75)
    ax.axhline(0, color=dark, linestyle="--")
    ax.set(title="Held-out residuals", xlabel="Predicted Day 35 weight (g)", ylabel="Prediction error (g)")
    fig.tight_layout(); fig.savefig(output / "residuals.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(checkpoint["review_day"].astype(str), checkpoint["rmse_g"], color=blue)
    for index, row in checkpoint.reset_index(drop=True).iterrows():
        ax.text(index, row["rmse_g"] + 5, f"R² {row['r2']:.2f}", ha="center", color=dark)
    ax.set(title="Champion performance by forecast checkpoint", xlabel="Review day", ylabel="RMSE (g)", ylim=(0, checkpoint["rmse_g"].max() * 1.25))
    fig.tight_layout(); fig.savefig(output / "performance_by_checkpoint.png", dpi=180); plt.close(fig)

    outcomes = development[["cycle_id", "building_id", "outcome_day35_weight_g"]].drop_duplicates()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(outcomes["outcome_day35_weight_g"], bins=10, color="#A9C6D8", edgecolor=dark)
    ax.axvline(DAY35_TARGET_G, color=orange, linestyle="--", linewidth=2, label="1,800 g target")
    ax.set(title="Observed Day 35 bodyweight distribution", xlabel="Observed average building weight (g)", ylabel="Building-cycles")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output / "outcome_distribution.png", dpi=180); plt.close(fig)

    coverage = intervals.groupby("review_day", as_index=False).agg(coverage_80=("covered_80", "mean"), coverage_90=("covered_90", "mean"))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(coverage)); width = 0.36
    ax.bar(x - width / 2, coverage["coverage_80"] * 100, width, color="#77AADD", label="80% interval")
    ax.bar(x + width / 2, coverage["coverage_90"] * 100, width, color=blue, label="90% interval")
    ax.axhline(80, color="#77AADD", linestyle="--"); ax.axhline(90, color=blue, linestyle="--")
    ax.set(xticks=x, xticklabels=coverage["review_day"].astype(str), ylim=(0, 110), title="Grouped conformal interval coverage", xlabel="Review day", ylabel="Empirical coverage (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output / "interval_coverage.png", dpi=180); plt.close(fig)

    if not shap_global.empty:
        top_shap = shap_global.head(12).sort_values("mean_abs_shap_g")
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.barh(top_shap["feature"].str.replace("_", " "), top_shap["mean_abs_shap_g"], color=pink)
        ax.set(title="Global SHAP importance for the strongest explainable learned model", xlabel="Mean absolute SHAP contribution (g)", ylabel="")
        fig.tight_layout(); fig.savefig(output / "shap_global.png", dpi=180); plt.close(fig)
        top_names = shap_global.head(10)["feature"].tolist()
        plot = shap_local.loc[shap_local["feature"].isin(top_names)].copy()
        positions = {name: index for index, name in enumerate(reversed(top_names))}
        fig, ax = plt.subplots(figsize=(9, 6))
        rng = np.random.default_rng(SEED)
        last_scatter = None
        for feature, group in plot.groupby("feature"):
            values = group["feature_value"].to_numpy(float)
            spread = np.ptp(values)
            normalized = (values - values.min()) / spread if spread > 0 else np.full(len(values), 0.5)
            y = np.full(len(group), positions[feature]) + rng.normal(0, 0.08, len(group))
            last_scatter = ax.scatter(group["shap_value_g"], y, c=normalized, cmap="coolwarm", vmin=0, vmax=1, alpha=0.72, s=22)
        ax.axvline(0, color=dark, linewidth=0.8)
        ax.set_yticks(range(len(top_names)), [name.replace("_", " ") for name in reversed(top_names)])
        ax.set(title="SHAP contribution distribution", xlabel="Contribution to predicted Day 35 weight (g)")
        if last_scatter is not None:
            colorbar = fig.colorbar(last_scatter, ax=ax, label="Feature value")
            colorbar.set_ticks([0, 1], labels=["Low", "High"])
        fig.tight_layout(); fig.savefig(output / "shap_beeswarm.png", dpi=180); plt.close(fig)


def run_review(workbook: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    workbook_path = Path(workbook).resolve()
    root = Path(__file__).resolve().parents[1]
    output_path = Path(output).resolve() if output else root / "outputs" / "bodyweight_modeling_review"
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(workbook_path)
    snapshots = build_snapshots(dataset)
    for review_day in CHECKPOINTS:
        for feature_set in {candidate.feature_set for candidate in CANDIDATES}:
            assert_primary_schema_has_no_identity(feature_columns(review_day, feature_set))
    development = snapshots.loc[snapshots["role"].eq("development")].reset_index(drop=True)
    audit = snapshots.loc[snapshots["role"].eq("later_cycle_audit")].reset_index(drop=True)
    quality_profile, quality_checks = _data_quality(dataset, snapshots)
    if quality_checks.loc[quality_checks["severity"].eq("critical"), "failed_rows"].sum() > 0:
        raise AssertionError("Critical data-quality check failed; see data_quality_checks.csv")

    all_predictions: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    nested_parameters: dict[str, Any] = {}
    for candidate in CANDIDATES:
        predictions, parameters = evaluate_candidate(development, candidate)
        metrics = summarize(predictions, bootstrap=True)
        all_predictions.append(predictions)
        comparison_rows.append({"candidate": candidate.name, "family": candidate.family, "target_form": candidate.target_form, "feature_set": candidate.feature_set, "complexity": candidate.complexity, "description": candidate.description, **metrics})
        nested_parameters[candidate.name] = parameters
    predictions = pd.concat(all_predictions, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows).sort_values(["cycle_macro_rmse_g", "complexity", "candidate"]).reset_index(drop=True)
    comparison["rank"] = np.arange(1, len(comparison) + 1)
    naive_families = {"baseline", "target_gain", "historical_ratio", "recent_adg", "pace"}
    external_families = {"xgboost", "lightgbm", "catboost"}
    naive_comparison = comparison.loc[comparison["family"].isin(naive_families)].copy()
    external_boosting_comparison = comparison.loc[comparison["family"].isin(external_families)].copy()
    lowest_error_name = str(comparison.iloc[0]["candidate"])
    one_se_limit = float(comparison.iloc[0]["cycle_macro_rmse_g"] + comparison.iloc[0]["fold_rmse_se_g"])
    one_se_eligible = comparison.loc[comparison["cycle_macro_rmse_g"].le(one_se_limit)].copy()
    champion_name = str(
        one_se_eligible.sort_values(["complexity", "cycle_macro_rmse_g", "candidate"]).iloc[0]["candidate"]
    )
    champion = next(candidate for candidate in CANDIDATES if candidate.name == champion_name)
    champion_predictions = predictions.loc[predictions["candidate"].eq(champion_name)].reset_index(drop=True)
    champion_checkpoint = checkpoint_metrics(champion_predictions)
    best_naive_name = str(naive_comparison.iloc[0]["candidate"])
    best_external_name = str(external_boosting_comparison.iloc[0]["candidate"])
    plotted_options = list(dict.fromkeys([champion_name, lowest_error_name, best_external_name, "target_gap_preserving"]))
    day14_option_predictions = predictions.loc[predictions["candidate"].isin(plotted_options) & predictions["review_day"].eq(14)].copy()
    intervals, calibration = _conformal_for_champion(development, champion)
    shadow_candidate = next(candidate for candidate in CANDIDATES if candidate.name == lowest_error_name)
    if lowest_error_name == champion_name:
        shadow_intervals, shadow_calibration = intervals.copy(), calibration.copy()
    else:
        shadow_intervals, shadow_calibration = _conformal_for_champion(
            development, shadow_candidate
        )
    temporal_predictions, temporal_metrics = _temporal_stress(development, champion)
    final_bundle, final_parameters = _fit_final(development, champion)
    audit_predictions = _predict_bundle(final_bundle, audit)
    audit_export = audit[["cycle_id", "building_id", "review_day", "as_of_date", "outcome_day35_weight_g"]].copy()
    audit_export["predicted_g"] = audit_predictions
    audit_export["error_g"] = audit_predictions - audit_export["outcome_day35_weight_g"]
    audit_metrics = summarize(audit_export.rename(columns={"outcome_day35_weight_g": "actual_g"}))
    shadow_bundle, shadow_parameters = _fit_final(development, shadow_candidate)
    shadow_audit_predictions = _predict_bundle(shadow_bundle, audit)
    shadow_audit_export = audit[["cycle_id", "building_id", "review_day", "as_of_date", "outcome_day35_weight_g"]].copy()
    shadow_audit_export["candidate"] = lowest_error_name
    shadow_audit_export["predicted_g"] = shadow_audit_predictions
    shadow_audit_export["error_g"] = shadow_audit_predictions - shadow_audit_export["outcome_day35_weight_g"]
    shadow_audit_metrics = summarize(shadow_audit_export.rename(columns={"outcome_day35_weight_g": "actual_g"}))
    identity_predictions, identity_comparison = identity_sensitivity(development)
    building_label_predictions, building_label_metrics = leave_one_building_label_out(
        development, champion
    )

    explainable = comparison.loc[~comparison["family"].isin(["baseline", "pace", "blend"])]
    explanation_name = str(explainable.iloc[0]["candidate"])
    explanation_candidate = next(candidate for candidate in CANDIDATES if candidate.name == explanation_name)
    shap_global, shap_local = _permutation_shap(development, audit, explanation_candidate)

    artifact = {
        "review_version": "bodyweight-specialist-1.2.0",
        "outcome": "Observed average building bodyweight on production Day 35 (g)",
        "candidate": asdict(champion),
        "checkpoint_models": final_bundle,
        "checkpoint_parameters": final_parameters,
        "source_sha256": _sha256(workbook_path),
        "development_cycles": list(DEVELOPMENT_CYCLES),
        "audit_cycle": AUDIT_CYCLE,
        "seed": SEED,
    }
    joblib.dump(artifact, output_path / "champion.joblib")
    reloaded = joblib.load(output_path / "champion.joblib")
    parity = bool(np.allclose(_predict_bundle(reloaded["checkpoint_models"], audit), audit_predictions))
    joblib.dump(
        {
            "review_version": "bodyweight-farmwide-shadow-2.0.0",
            "outcome": artifact["outcome"],
            "candidate": asdict(shadow_candidate),
            "checkpoint_models": shadow_bundle,
            "checkpoint_parameters": shadow_parameters,
            "source_sha256": artifact["source_sha256"],
            "development_cycles": artifact["development_cycles"],
            "audit_cycle": artifact["audit_cycle"],
            "seed": artifact["seed"],
            "deployment_status": "shadow",
        },
        output_path / "shadow_challenger.joblib",
    )

    for frame, name in (
        (snapshots, "model_ready_snapshots.csv"),
        (quality_checks, "data_quality_checks.csv"),
        (_data_dictionary(), "data_dictionary.csv"),
        (pd.DataFrame([asdict(candidate) for candidate in CANDIDATES]), "candidate_registry.csv"),
        (predictions, "all_oof_predictions.csv"),
        (comparison, "model_comparison.csv"),
        (comparison.head(5), "top_five_models.csv"),
        (naive_comparison, "naive_model_comparison.csv"),
        (external_boosting_comparison, "external_boosting_comparison.csv"),
        (day14_option_predictions, "day14_actual_vs_projected_options.csv"),
        (champion_predictions, "champion_oof_predictions.csv"),
        (champion_checkpoint, "checkpoint_metrics.csv"),
        (intervals, "conformal_intervals.csv"),
        (calibration, "conformal_calibration.csv"),
        (shadow_intervals, "shadow_conformal_intervals.csv"),
        (shadow_calibration, "shadow_conformal_calibration.csv"),
        (temporal_predictions, "temporal_stress_predictions.csv"),
        (audit_export, "later_cycle_audit_predictions.csv"),
        (shadow_audit_export, "shadow_later_cycle_audit_predictions.csv"),
        (identity_predictions, "identity_sensitivity_predictions.csv"),
        (identity_comparison, "identity_sensitivity_comparison.csv"),
        (building_label_predictions, "leave_one_building_label_out_predictions.csv"),
        (shap_global, "shap_global.csv"),
        (shap_local, "shap_local.csv"),
    ):
        frame.to_csv(output_path / name, index=False)
    (output_path / "nested_hyperparameters.json").write_text(json.dumps(nested_parameters, indent=2, default=_json_default), encoding="utf-8")

    _plots(output_path / "figures", development, predictions, champion_predictions, comparison, champion_checkpoint, intervals, shap_global, shap_local)

    champion_metrics = summarize(champion_predictions, bootstrap=True)
    manifest = {
        "review_version": "bodyweight-specialist-1.2.0",
        "created": "2026-08-13",
        "source_workbook": str(workbook_path),
        "source_sha256": _sha256(workbook_path),
        "quality_profile": quality_profile,
        "design_improvements": [
            "Checkpoint-specific models instead of forcing one coefficient surface across Days 7, 14, 21, and 28.",
            "Full observed checkpoint trajectory, interval gains, ADG, and acceleration retained without forward-filling measurements.",
            "Compact and extended feature sets tested separately to expose small-sample overfitting.",
            "Poultry-specific challengers add target deficit, age-specific temperature/humidity deviation, exposure duration, extremes, trends, THI, and compound stress.",
            "Extra Trees, Random Forest, Gradient Boosting, and Histogram Gradient Boosting are compared in growth-only and environmental formulations.",
            "XGBoost, LightGBM, and CatBoost are nested-tuned under the same complete-cycle holdouts.",
            "Transparent expected-growth, target-curve, historical-ratio, and recent-ADG projections are evaluated as formal naive candidates.",
            "All tuning and preprocessing nested inside complete harvest-cycle holdouts.",
            "RMSE optimized at the harvest-cycle macro level; 2026-3 locked until selection completed.",
        ],
        "primary_selection_metric": "Nested leave-one-harvest-cycle-out cycle-macro RMSE across Days 7, 14, 21 and 28",
        "champion": champion_name,
        "lowest_error_candidate": lowest_error_name,
        "shadow_candidate": lowest_error_name,
        "shadow_checkpoint_parameters": shadow_parameters,
        "shadow_later_cycle_audit_metrics": shadow_audit_metrics,
        "shadow_interval_metrics": {
            "coverage_80": float(shadow_intervals["covered_80"].mean()),
            "coverage_90": float(shadow_intervals["covered_90"].mean()),
            "mean_width_80_g": float((shadow_intervals["upper_80_g"] - shadow_intervals["lower_80_g"]).mean()),
            "mean_width_90_g": float((shadow_intervals["upper_90_g"] - shadow_intervals["lower_90_g"]).mean()),
        },
        "selection_rule": "One-standard-error rule: simplest model within one SE of the lowest cycle-macro RMSE",
        "one_se_limit_g": one_se_limit,
        "one_se_eligible_candidates": one_se_eligible["candidate"].astype(str).tolist(),
        "primary_identity_policy": "Exact building and Tags/Lags are excluded from all primary candidates.",
        "identity_sensitivity": identity_comparison.to_dict("records"),
        "leave_one_building_label_out_metrics": building_label_metrics,
        "best_naive": best_naive_name,
        "best_external_boosting": best_external_name,
        "champion_metrics": champion_metrics,
        "checkpoint_metrics": champion_checkpoint.to_dict("records"),
        "top_five_models": comparison.head(5).to_dict("records"),
        "naive_model_comparison": naive_comparison.to_dict("records"),
        "external_boosting_comparison": external_boosting_comparison.to_dict("records"),
        "temporal_stress_metrics": temporal_metrics,
        "interval_metrics": {
            "coverage_80": float(intervals["covered_80"].mean()),
            "coverage_90": float(intervals["covered_90"].mean()),
            "mean_width_80_g": float((intervals["upper_80_g"] - intervals["lower_80_g"]).mean()),
            "mean_width_90_g": float((intervals["upper_90_g"] - intervals["lower_90_g"]).mean()),
        },
        "later_cycle_audit_metrics": audit_metrics,
        "shap_model": explanation_name,
        "shap_is_champion": bool(explanation_name == champion_name),
        "artifact_reload_parity": parity,
        "r2_goal_met": bool(champion_metrics["r2"] >= 0.50),
        "operational_recommendation": "Use expected remaining growth as the transparent operational baseline; run trajectory PLS in shadow because it improves cycle-macro RMSE but does not yet reach replacement-grade R2 or uncertainty.",
        "limitations": [
            "Only 31 independent development building-cycles across six harvest cycles.",
            "Only five development outcomes meet or exceed 1,800 g.",
            "Environmental coverage is incomplete and confounded with harvest cycle.",
            "Feed is excluded because its source units and interpretation remain unresolved.",
            "SHAP measures predictive model reliance, not biological causation or treatment effect.",
        ],
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    manifest = run_review(arguments.workbook, arguments.output)
    print(json.dumps({"champion": manifest["champion"], "metrics": manifest["champion_metrics"], "r2_goal_met": manifest["r2_goal_met"]}, indent=2))


if __name__ == "__main__":
    main()
