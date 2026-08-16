"""Biology-aware, research-only modeling round for Project Canary.

The workflow creates daily as-of landmarks, evaluates biologically constrained
forecasting candidates with nested harvest-cycle LOGO validation, and exports
all evidence outside ``models/``.  It intentionally does not change the models
used by the Streamlit application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import math
import os
from typing import Any, Iterable

# Keep PyTensor's compiled artifacts in a task-scoped writable cache rather
# than relying on a user-home directory that may be unavailable in CI.
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/private/tmp/project_canary_pytensor")

import arviz as az
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import shap
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .bodyweight_modeling_review import _snapshot as _weight_snapshot
from .data import CanaryDataset, load_workbook
from .external_modeling_review import (
    AUDIT_CYCLE,
    CHECKPOINTS,
    DEVELOPMENT_CYCLES,
    SEED,
    _snapshot_features as _recovery_snapshot,
)
from .farmwide_modeling import build_source_quality_audit


ROUND_VERSION = "biology-aware-1.0.0"
LANDMARK_DAYS = tuple(range(7, 35))
DAY35_TARGET_G = 1800.0


@dataclass(frozen=True)
class Candidate:
    name: str
    outcome: str
    family: str
    complexity: int
    description: str


RECOVERY_CANDIDATES = (
    Candidate("current_survival", "recovery", "persistence", 0, "Current recorded survival persists to harvest."),
    Candidate("daily_age_remaining_loss", "recovery", "age_baseline", 1, "Fold-local remaining loss by daily landmark age."),
    Candidate("negative_binomial_loss_hazard", "recovery", "negative_binomial", 3, "Population-at-risk negative-binomial remaining-loss model."),
    Candidate("bayesian_monotone_hazard", "recovery", "hazard_ridge", 4, "Regularized positive cumulative-hazard model."),
    Candidate("hazard_residual_ridge", "recovery", "residual_ridge", 5, "Ridge correction to the biological hazard baseline."),
    Candidate("hazard_residual_xgboost", "recovery", "residual_xgboost", 7, "Constrained XGBoost correction to the biological hazard baseline."),
    Candidate("hazard_residual_lightgbm", "recovery", "residual_lightgbm", 7, "Constrained LightGBM correction to the biological hazard baseline."),
    Candidate("hazard_baseline_blend", "recovery", "blend", 6, "Fold-tuned blend of hazard and historical remaining-loss forecasts."),
)

WEIGHT_CANDIDATES = (
    Candidate("historical_remaining_gain", "weight", "age_baseline", 0, "Fold-local remaining gain by daily landmark age."),
    Candidate("target_curve_ratio", "weight", "target_ratio", 1, "Latest observed target-relative pace projected to Day 35."),
    Candidate("target_anchored_kalman", "weight", "kalman", 3, "Target-anchored local-trend state-space update."),
    Candidate("bayesian_gompertz_partial_pooling", "weight", "gompertz", 4, "Empirical-Bayes partial pooling around a fold-local Gompertz curve."),
    Candidate("bayesian_logistic_partial_pooling", "weight", "logistic", 5, "Empirical-Bayes partial pooling around a fold-local logistic curve."),
    Candidate("target_residual_ridge", "weight", "residual_ridge", 5, "Regularized correction to the target-anchored Kalman forecast."),
    Candidate("state_residual_xgboost", "weight", "residual_xgboost", 7, "Constrained XGBoost correction to the state-space forecast."),
    Candidate("kalman_historical_blend", "weight", "blend", 6, "Fold-tuned state-space and historical-gain blend."),
)


RECOVERY_FEATURES = [
    "review_day", "days_to_day35", "beginning_inventory", "log_beginning_inventory",
    "percentage_alive", "population_loss_pct", "population_loss_rate_pp_day",
    "mortality_daily_per_1000", "mortality_recent_3d_per_1000",
    "mortality_recent_7d_per_1000", "mortality_ewma_per_1000",
    "mortality_trend_per_1000", "mortality_acceleration_per_1000",
    "mortality_volatility_per_1000", "mortality_max_per_1000", "mortality_spike_days",
    "mortality_recent_vs_early_per_1000", "mortality_cusum_high",
    "longest_elevated_loss_episode", "early_loss_burden_per_1000",
    "recent_loss_burden_per_1000", "population_mortality_reconciliation_gap_per_1000",
    "population_increase_days", "record_completeness_ratio", "weight_ratio_to_target",
    "weight_gap_pct", "weight_staleness_days", "weight_measurement_count",
    "temperature_heat_excess_degree_days", "temperature_cold_excess_degree_days",
    "temperature_out_of_band_days", "temperature_excursion_longest_run",
    "days_since_temperature_excursion", "humidity_out_of_band_days",
    "humidity_excursion_longest_run", "days_since_humidity_excursion",
    "environment_coverage_ratio", "environment_staleness_days", "temperature_missing",
    "humidity_missing", "peer_survival_mean", "survival_vs_peer",
    "peer_mortality_7d_mean", "mortality_7d_vs_peer", "peer_building_count",
]

WEIGHT_FEATURES = [
    "review_day", "days_to_day35", "current_weight_g", "current_target_g",
    "current_ratio_to_target", "target_ratio_log", "current_gap_to_target_g",
    "target_deficit_auc_g_days", "latest_measurement_day", "measurement_staleness_days",
    "weight_measurement_count", "measurement_spacing_mean_days", "measurement_spacing_sd_days",
    "robust_weight_slope_g_day", "recent_adg_all_g_day", "trajectory_curvature_g_day2",
    "phase_gain_1_7_g", "phase_gain_8_14_g", "phase_gain_15_21_g", "phase_gain_22_28_g",
    "survival_pct", "population_loss_pct", "mortality_recent_3d_per_1000",
    "mortality_recent_7d_per_1000", "temperature_between_weights_mean_c",
    "heat_between_weights_degree_days", "humidity_between_weights_mean_pct",
    "humidity_between_weights_excursion_days", "temperature_coverage", "humidity_coverage",
    "peer_current_weight_mean_g", "current_weight_vs_peer_g", "peer_building_count",
    "weight_day7_g", "weight_day14_g", "weight_day21_g", "weight_day28_g",
]


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


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _longest_run(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _days_since(values: pd.Series, ages: pd.Series) -> float:
    selected = ages.loc[values.fillna(False).astype(bool)]
    return float(ages.max() - selected.max()) if not selected.empty else np.nan


def _recovery_biology_features(dataset: CanaryDataset, cycle: str, building: str, day: int) -> dict[str, float]:
    history = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(cycle)
        & dataset.daily["building_id"].astype(str).eq(building)
        & dataset.daily["age_day"].le(day)
    ].sort_values("age_day")
    beginning = float(history["beginning_inventory"].iloc[-1])
    mortality_rate = pd.to_numeric(history["mortality_daily"], errors="coerce") / beginning * 1000
    median = float(mortality_rate.median())
    mad = float((mortality_rate - median).abs().median())
    threshold = median + max(3.0 * 1.4826 * mad, 0.1)
    elevated = mortality_rate.gt(threshold)
    excess = (mortality_rate - threshold).fillna(0).clip(lower=0)
    cusum = float(excess.cumsum().max()) if len(excess) else 0.0
    ages = pd.to_numeric(history["age_day"], errors="coerce")
    temperature = pd.to_numeric(history["temperature_avg_c"], errors="coerce")
    humidity = pd.to_numeric(history["humidity_avg_pct"], errors="coerce")
    temp_target = pd.Series(np.select([ages <= 7, ages <= 14, ages <= 21, ages <= 28], [31, 28.5, 25.5, 23.5], default=22.5), index=history.index)
    temp_excursion = (temperature - temp_target).abs().gt(1.5) & temperature.notna()
    humidity_target = pd.Series(np.select([ages <= 7, ages <= 14], [60, 55], default=50), index=history.index)
    humidity_excursion = ((humidity < humidity_target) | (humidity > humidity_target + 10)) & humidity.notna()
    return {
        "mortality_cusum_high": cusum,
        "longest_elevated_loss_episode": float(_longest_run(elevated)),
        "early_loss_burden_per_1000": float(mortality_rate.loc[ages.le(14)].sum(min_count=1)),
        "recent_loss_burden_per_1000": float(mortality_rate.tail(7).sum(min_count=1)),
        "temperature_excursion_longest_run": float(_longest_run(temp_excursion)),
        "days_since_temperature_excursion": _days_since(temp_excursion, ages),
        "humidity_excursion_longest_run": float(_longest_run(humidity_excursion)),
        "days_since_humidity_excursion": _days_since(humidity_excursion, ages),
    }


def _weight_biology_features(dataset: CanaryDataset, cycle: str, building: str, day: int) -> dict[str, float]:
    history = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(cycle)
        & dataset.daily["building_id"].astype(str).eq(building)
        & dataset.daily["age_day"].le(day)
    ].sort_values("age_day")
    measured = history.loc[history["weight_measured"].fillna(False)].copy()
    ages = measured["age_day"].to_numpy(float)
    weights = measured["bodyweight_kg"].to_numpy(float) * 1000
    target_map = dataset.targets.set_index("age_day")["target_weight_scaled_g"].to_dict()
    targets = np.asarray([target_map.get(int(age), np.nan) for age in ages], dtype=float)
    residuals = weights - targets
    auc = float(np.trapezoid(residuals, ages)) if len(ages) > 1 else float(residuals[-1] * max(ages[-1], 1))
    slope = float(np.polyfit(ages, weights, 1)[0]) if len(ages) > 1 else np.nan
    recent_adg = float((weights[-1] - weights[-2]) / (ages[-1] - ages[-2])) if len(ages) > 1 and ages[-1] > ages[-2] else np.nan
    curvature = np.nan
    if len(ages) >= 3:
        curvature = float(2 * np.polyfit(ages, weights, 2)[0])
    spacing = np.diff(ages)

    def phase_gain(start: int, end: int) -> float:
        eligible = measured.loc[measured["age_day"].between(start, end)]
        if len(eligible) < 2:
            return np.nan
        return float((eligible["bodyweight_kg"].iloc[-1] - eligible["bodyweight_kg"].iloc[0]) * 1000)

    if len(ages) > 1:
        interval = history.loc[history["age_day"].between(int(ages[-2]) + 1, int(ages[-1]))]
    else:
        interval = history
    interval_temp = pd.to_numeric(interval["temperature_avg_c"], errors="coerce")
    interval_humidity = pd.to_numeric(interval["humidity_avg_pct"], errors="coerce")
    interval_age = pd.to_numeric(interval["age_day"], errors="coerce")
    interval_target = pd.Series(np.select([interval_age <= 7, interval_age <= 14, interval_age <= 21, interval_age <= 28], [31, 28.5, 25.5, 23.5], default=22.5), index=interval.index)
    heat = (interval_temp - (interval_target + 1.5)).clip(lower=0)
    humidity_target = pd.Series(np.select([interval_age <= 7, interval_age <= 14], [60, 55], default=50), index=interval.index)
    hum_excursion = ((interval_humidity < humidity_target) | (interval_humidity > humidity_target + 10)) & interval_humidity.notna()
    return {
        "target_ratio_log": float(np.log(max(weights[-1] / targets[-1], 0.05))),
        "target_deficit_auc_g_days": auc,
        "measurement_spacing_mean_days": float(spacing.mean()) if len(spacing) else np.nan,
        "measurement_spacing_sd_days": float(spacing.std()) if len(spacing) else np.nan,
        "robust_weight_slope_g_day": slope,
        "recent_adg_all_g_day": recent_adg,
        "trajectory_curvature_g_day2": curvature,
        "phase_gain_1_7_g": phase_gain(1, 7),
        "phase_gain_8_14_g": phase_gain(7, 14),
        "phase_gain_15_21_g": phase_gain(14, 21),
        "phase_gain_22_28_g": phase_gain(21, 28),
        "temperature_between_weights_mean_c": float(interval_temp.mean()) if interval_temp.notna().any() else np.nan,
        "heat_between_weights_degree_days": float(heat.sum(min_count=1)),
        "humidity_between_weights_mean_pct": float(interval_humidity.mean()) if interval_humidity.notna().any() else np.nan,
        "humidity_between_weights_excursion_days": float(hum_excursion.sum()),
    }


def _leave_self_out(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    grouped = frame.groupby(["cycle_id", "review_day"])[source]
    count = grouped.transform("count")
    total = grouped.transform("sum")
    frame[target] = np.where(count > 1, (total - frame[source]) / (count - 1), np.nan)
    frame["peer_building_count"] = np.maximum(frame.get("peer_building_count", 0), count - 1)
    return frame


def build_daily_landmarks(
    dataset: CanaryDataset,
    outcome: str,
    development_cycles: tuple[str, ...] = DEVELOPMENT_CYCLES,
    audit_cycle: str = AUDIT_CYCLE,
) -> pd.DataFrame:
    """Build leakage-safe Day 7–34 landmarks with equal building-cycle weights."""
    rows: list[dict[str, Any]] = []
    for cycle_record in dataset.cycles.itertuples(index=False):
        cycle = str(cycle_record.cycle_id)
        building = str(cycle_record.building_id)
        if cycle not in {*development_cycles, audit_cycle}:
            continue
        unit = dataset.daily.loc[
            dataset.daily["cycle_id"].astype(str).eq(cycle)
            & dataset.daily["building_id"].astype(str).eq(building)
        ]
        day35 = unit.loc[unit["age_day"].eq(35) & unit["weight_measured"].fillna(False), "bodyweight_kg"]
        for day in LANDMARK_DAYS:
            if outcome == "recovery":
                row = _recovery_snapshot(dataset, cycle, building, day)
                if row is None:
                    continue
                row.update(_recovery_biology_features(dataset, cycle, building, day))
                row["actual"] = float(cycle_record.final_recovery_rate)
                row["current_value"] = float(row["percentage_alive"])
                row["current_population"] = float(row["current_value"] * row["beginning_inventory"])
                row["remaining_target"] = max(0.0, row["current_value"] - row["actual"])
                row["remaining_loss_count"] = max(0.0, float(row["remaining_target"] * row["beginning_inventory"]))
            else:
                if day35.empty:
                    continue
                outcome_g = float(day35.iloc[-1] * 1000)
                row = _weight_snapshot(dataset, cycle, building, day, outcome_g)
                if row is None:
                    continue
                row.update(_weight_biology_features(dataset, cycle, building, day))
                row["days_to_day35"] = 35 - day
                row["actual"] = outcome_g
                row["current_value"] = float(row["current_weight_g"])
                row["remaining_target"] = outcome_g - row["current_value"]
            row["role"] = "later_cycle_audit" if cycle == audit_cycle else "development"
            row["checkpoint_status"] = "validated_checkpoint" if day in CHECKPOINTS else "between_checkpoint_estimate"
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True)
    frame["peer_building_count"] = 0
    if outcome == "recovery":
        frame = _leave_self_out(frame, "percentage_alive", "peer_survival_mean")
        frame = _leave_self_out(frame, "mortality_recent_7d_per_1000", "peer_mortality_7d_mean")
        frame["survival_vs_peer"] = frame["percentage_alive"] - frame["peer_survival_mean"]
        frame["mortality_7d_vs_peer"] = frame["mortality_recent_7d_per_1000"] - frame["peer_mortality_7d_mean"]
    else:
        frame = _leave_self_out(frame, "current_weight_g", "peer_current_weight_mean_g")
        frame["current_weight_vs_peer_g"] = frame["current_weight_g"] - frame["peer_current_weight_mean_g"]
    counts = frame.groupby(["cycle_id", "building_id"])["review_day"].transform("count")
    frame["sample_weight"] = 1.0 / counts
    frame["sample_weight"] *= len(frame) / frame["sample_weight"].sum()
    if not frame["max_source_day_used"].le(frame["review_day"]).all():
        raise AssertionError("Future evidence entered a daily landmark.")
    if audit_cycle in set(frame.loc[frame["role"].eq("development"), "cycle_id"].astype(str)):
        raise AssertionError("Locked audit cycle entered development landmarks.")
    return frame


def _age_baseline(train: pd.DataFrame) -> dict[str, Any]:
    mapping = train.groupby("review_day").apply(
        lambda group: np.average(group["remaining_target"], weights=group["sample_weight"]),
        include_groups=False,
    ).to_dict()
    fallback = float(np.average(train["remaining_target"], weights=train["sample_weight"]))
    return {"mapping": {int(k): float(v) for k, v in mapping.items()}, "fallback": fallback}


def _baseline_predict(frame: pd.DataFrame, baseline: dict[str, Any], outcome: str) -> np.ndarray:
    remaining = np.asarray([baseline["mapping"].get(int(day), baseline["fallback"]) for day in frame["review_day"]])
    if outcome == "recovery":
        return frame["current_value"].to_numpy(float) - np.maximum(remaining, 0)
    return frame["current_value"].to_numpy(float) + remaining


def _design(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = frame[features].copy().replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    return values.astype(float)


def _nb_features(frame: pd.DataFrame) -> pd.DataFrame:
    age = frame["review_day"].to_numpy(float)
    return pd.DataFrame({
        "const": 1.0,
        "age": age / 35.0,
        "age2": (age / 35.0) ** 2,
        "remaining_horizon": frame["days_to_day35"].to_numpy(float) / 28.0,
        "current_mortality": frame["mortality_recent_7d_per_1000"].fillna(0).to_numpy(float) / 10.0,
        "mortality_trend": frame["mortality_trend_per_1000"].fillna(0).to_numpy(float) / 10.0,
        "cusum": frame["mortality_cusum_high"].fillna(0).to_numpy(float) / 10.0,
        "environment_missing": frame[["temperature_missing", "humidity_missing"]].max(axis=1).to_numpy(float),
    }, index=frame.index)


def _fit_negative_binomial(train: pd.DataFrame) -> dict[str, Any]:
    exog = _nb_features(train)
    endog = train["remaining_loss_count"].to_numpy(float)
    exposure = np.log(train["current_population"].clip(lower=1).to_numpy(float))
    try:
        fitted = sm.GLM(
            endog, exog, family=sm.families.NegativeBinomial(alpha=1.0),
            offset=exposure, freq_weights=train["sample_weight"].to_numpy(float),
        ).fit(maxiter=200, disp=0)
        return {"model": fitted, "fallback_rate": None}
    except Exception:
        rate = float(np.average(endog / train["current_population"].clip(lower=1), weights=train["sample_weight"]))
        return {"model": None, "fallback_rate": rate}


def _predict_negative_binomial(fitted: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if fitted["model"] is None:
        count = frame["current_population"].to_numpy(float) * fitted["fallback_rate"]
    else:
        count = fitted["model"].predict(
            _nb_features(frame), offset=np.log(frame["current_population"].clip(lower=1).to_numpy(float))
        )
    return frame["current_value"].to_numpy(float) - np.maximum(count, 0) / frame["beginning_inventory"].to_numpy(float)


def _curve_gompertz(day: np.ndarray, asymptote: float, rate: float, midpoint: float) -> np.ndarray:
    return asymptote * np.exp(-np.exp(-rate * (day - midpoint)))


def _curve_logistic(day: np.ndarray, asymptote: float, rate: float, midpoint: float) -> np.ndarray:
    return asymptote / (1.0 + np.exp(-rate * (day - midpoint)))


def _fit_curve(train: pd.DataFrame, family: str) -> dict[str, Any]:
    columns = [f"weight_day{day}_g" for day in CHECKPOINTS] + ["actual"]
    curve = _curve_gompertz if family == "gompertz" else _curve_logistic
    observations: list[tuple[float, float]] = []
    for row in train.sort_values("review_day").drop_duplicates(["cycle_id", "building_id"], keep="last").itertuples():
        for day, column in zip((*CHECKPOINTS, 35), columns):
            value = getattr(row, column)
            if pd.notna(value):
                observations.append((float(day), float(value)))
    days = np.asarray([item[0] for item in observations])
    weights = np.asarray([item[1] for item in observations])
    try:
        params, _ = curve_fit(curve, days, weights, p0=(2400, 0.1, 22), bounds=([1500, 0.02, 5], [4000, 0.4, 40]), maxfev=30000)
    except Exception:
        params = np.asarray((2400.0, 0.1, 22.0))
    residuals_by_unit = []
    within = []
    for _, group in train.groupby(["cycle_id", "building_id"]):
        row = group.sort_values("review_day").iloc[-1]
        offsets = []
        for day, column in zip(CHECKPOINTS, columns[:-1]):
            value = row.get(column)
            if pd.notna(value):
                offsets.append(math.log(max(float(value), 1) / max(float(curve(np.asarray([day]), *params)[0]), 1)))
        if offsets:
            residuals_by_unit.append(float(np.mean(offsets)))
            within.extend([value - np.mean(offsets) for value in offsets])
    between_var = max(float(np.var(residuals_by_unit, ddof=1)) if len(residuals_by_unit) > 1 else 0.01, 1e-5)
    noise_var = max(float(np.var(within, ddof=1)) if len(within) > 1 else 0.02, 1e-5)
    return {"params": params, "between_var": between_var, "noise_var": noise_var, "family": family}


def _curve_predict(fitted: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    curve = _curve_gompertz if fitted["family"] == "gompertz" else _curve_logistic
    output = []
    for _, row in frame.iterrows():
        offsets = []
        for day in CHECKPOINTS:
            value = row.get(f"weight_day{day}_g")
            if day <= int(row["review_day"]) and pd.notna(value):
                expected = float(curve(np.asarray([day]), *fitted["params"])[0])
                offsets.append(math.log(max(float(value), 1) / max(expected, 1)))
        n = len(offsets)
        shrinkage = fitted["between_var"] / (fitted["between_var"] + fitted["noise_var"] / max(n, 1))
        offset = shrinkage * float(np.mean(offsets)) if offsets else 0.0
        output.append(float(curve(np.asarray([35]), *fitted["params"])[0]) * math.exp(offset))
    return np.asarray(output)


def _kalman_predict(frame: pd.DataFrame, q: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    output, variance = [], []
    for _, row in frame.iterrows():
        state = np.asarray([0.0, 0.0])
        covariance = np.diag([0.08, 0.002])
        last_day = 0
        for day in CHECKPOINTS:
            value = row.get(f"weight_day{day}_g")
            if day > int(row["review_day"]) or pd.isna(value):
                continue
            target = row.get(f"target_day{day}_g", np.nan)
            if pd.isna(target):
                target = {7: 170, 14: 380, 21: 800, 28: 1200}[day]
            delta = day - last_day
            transition = np.asarray([[1.0, delta], [0.0, 1.0]])
            process = np.diag([q * max(delta, 1), q / 49.0 * max(delta, 1)])
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            observation = math.log(max(float(value) / float(target), 0.05))
            h = np.asarray([[1.0, 0.0]])
            innovation_var = float((h @ covariance @ h.T)[0, 0] + r)
            gain = covariance @ h.T / innovation_var
            state = state + gain[:, 0] * (observation - float(h @ state))
            covariance = (np.eye(2) - gain @ h) @ covariance
            last_day = day
        delta = 35 - last_day
        transition = np.asarray([[1.0, delta], [0.0, 1.0]])
        process = np.diag([q * max(delta, 1), q / 49.0 * max(delta, 1)])
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        output.append(DAY35_TARGET_G * math.exp(float(np.clip(state[0], -1.0, 0.7))))
        variance.append(float(max(covariance[0, 0], 0)))
    return np.asarray(output), np.asarray(variance)


def parameter_grid(candidate: Candidate) -> list[dict[str, Any]]:
    if candidate.family in {"persistence", "age_baseline", "target_ratio", "negative_binomial", "gompertz", "logistic"}:
        return [{}]
    if candidate.family == "kalman":
        return [{"q": q, "r": r} for q, r in ((0.0002, 0.0025), (0.0005, 0.005), (0.001, 0.01))]
    if candidate.family == "hazard_ridge":
        return [{"alpha": value} for value in (1.0, 10.0, 100.0)]
    if candidate.family == "residual_ridge":
        return [{"alpha": value} for value in (10.0, 100.0, 500.0)]
    if candidate.family in {"residual_xgboost", "residual_lightgbm"}:
        return [
            {"trees": 80, "rate": 0.03, "depth": 1, "leaf": 8, "l2": 20.0},
            {"trees": 120, "rate": 0.025, "depth": 2, "leaf": 12, "l2": 40.0},
            {"trees": 160, "rate": 0.02, "depth": 2, "leaf": 16, "l2": 80.0},
        ]
    if candidate.family == "blend":
        return [{"weight": value} for value in (0.25, 0.5, 0.75)]
    raise ValueError(candidate.family)


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _tree_pipeline(candidate: Candidate, parameters: dict[str, Any], seed: int) -> Pipeline:
    if candidate.family == "residual_xgboost":
        model: Any = XGBRegressor(
            n_estimators=parameters["trees"], learning_rate=parameters["rate"], max_depth=parameters["depth"],
            min_child_weight=parameters["leaf"], reg_lambda=parameters["l2"], reg_alpha=1.0,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
            random_state=seed, n_jobs=1, verbosity=0,
        )
    else:
        model = LGBMRegressor(
            n_estimators=parameters["trees"], learning_rate=parameters["rate"], max_depth=parameters["depth"],
            num_leaves=max(2, 2 ** parameters["depth"] - 1), min_child_samples=parameters["leaf"],
            reg_lambda=parameters["l2"], reg_alpha=1.0, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
        )
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("model", model)])


def fit_candidate(train: pd.DataFrame, candidate: Candidate, parameters: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    baseline = _age_baseline(train)
    fitted: dict[str, Any] = {"candidate": asdict(candidate), "parameters": parameters, "baseline": baseline}
    if candidate.family in {"persistence", "age_baseline", "target_ratio", "kalman"}:
        return fitted
    if candidate.family == "blend":
        if candidate.outcome == "recovery":
            fitted["nb"] = _fit_negative_binomial(train)
        return fitted
    if candidate.family == "negative_binomial":
        fitted["nb"] = _fit_negative_binomial(train)
        return fitted
    if candidate.family in {"gompertz", "logistic"}:
        fitted["curve"] = _fit_curve(train, candidate.family)
        return fitted
    features = RECOVERY_FEATURES if candidate.outcome == "recovery" else WEIGHT_FEATURES
    fitted["features"] = features
    if candidate.outcome == "recovery":
        fitted["nb"] = _fit_negative_binomial(train)
        biological = _predict_negative_binomial(fitted["nb"], train)
    else:
        biological, _ = _kalman_predict(train, 0.0005, 0.005)
    if candidate.family == "hazard_ridge":
        target = np.log(np.maximum(train["remaining_target"].to_numpy(float), 1e-6))
        model = _ridge_pipeline(float(parameters["alpha"]))
    else:
        target = train["actual"].to_numpy(float) - biological
        model = _ridge_pipeline(float(parameters["alpha"])) if candidate.family == "residual_ridge" else _tree_pipeline(candidate, parameters, seed)
    fit_params = {"model__sample_weight": train["sample_weight"].to_numpy(float)}
    model.fit(train[features], target, **fit_params)
    fitted["model"] = model
    return fitted


def predict_candidate(fitted: dict[str, Any], frame: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    baseline = _baseline_predict(frame, fitted["baseline"], candidate.outcome)
    if candidate.family == "persistence":
        prediction = frame["current_value"].to_numpy(float)
    elif candidate.family == "age_baseline":
        prediction = baseline
    elif candidate.family == "target_ratio":
        prediction = frame["current_weight_g"].to_numpy(float) / frame["current_target_g"].clip(lower=1).to_numpy(float) * DAY35_TARGET_G
    elif candidate.family == "kalman":
        prediction, _ = _kalman_predict(frame, float(fitted["parameters"]["q"]), float(fitted["parameters"]["r"]))
    elif candidate.family == "negative_binomial":
        prediction = _predict_negative_binomial(fitted["nb"], frame)
    elif candidate.family in {"gompertz", "logistic"}:
        prediction = _curve_predict(fitted["curve"], frame)
    elif candidate.family == "hazard_ridge":
        remaining = np.exp(np.asarray(fitted["model"].predict(frame[fitted["features"]]), dtype=float))
        prediction = frame["current_value"].to_numpy(float) - remaining
    elif candidate.family in {"residual_ridge", "residual_xgboost", "residual_lightgbm"}:
        if candidate.outcome == "recovery":
            biological = _predict_negative_binomial(fitted["nb"], frame)
        else:
            biological, _ = _kalman_predict(frame, 0.0005, 0.005)
        prediction = biological + np.asarray(fitted["model"].predict(frame[fitted["features"]]), dtype=float)
    elif candidate.family == "blend":
        if candidate.outcome == "recovery":
            biological = _predict_negative_binomial(fitted["nb"], frame)
        else:
            biological, _ = _kalman_predict(frame, 0.0005, 0.005)
        prediction = float(fitted["parameters"]["weight"]) * biological + (1 - float(fitted["parameters"]["weight"])) * baseline
    else:
        raise ValueError(candidate.family)
    if candidate.outcome == "recovery":
        return np.minimum(np.clip(prediction, 0.0, 1.0), frame["current_value"].to_numpy(float))
    return np.clip(prediction, 100.0, 3500.0)


def _metric_factor(outcome: str) -> float:
    return 100.0 if outcome == "recovery" else 1.0


def _macro_rmse(actual: np.ndarray, predicted: np.ndarray, groups: np.ndarray, outcome: str) -> float:
    factor = _metric_factor(outcome)
    values = [mean_squared_error(actual[groups == group], predicted[groups == group]) ** 0.5 * factor for group in pd.unique(groups)]
    return float(np.mean(values))


def tune_candidate(train: pd.DataFrame, candidate: Candidate, seed: int = SEED) -> dict[str, Any]:
    grid = parameter_grid(candidate)
    if len(grid) == 1 or train["cycle_id"].nunique() < 2:
        return grid[0]
    groups = train["cycle_id"].astype(str).to_numpy()
    actual = train["actual"].to_numpy(float)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for order, parameters in enumerate(grid):
        predicted = np.full(len(train), np.nan)
        for fit_index, valid_index in LeaveOneGroupOut().split(train, groups=groups):
            fitted = fit_candidate(train.iloc[fit_index], candidate, parameters, seed)
            predicted[valid_index] = predict_candidate(fitted, train.iloc[valid_index], candidate)
        scored.append((_macro_rmse(actual, predicted, groups, candidate.outcome), order, parameters))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def evaluate_nested_logo(frame: pd.DataFrame, candidate: Candidate, *, view: str = "cycle", seed: int = SEED) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if view == "cycle":
        groups = frame["cycle_id"].astype(str).to_numpy()
    elif view == "building_label":
        groups = frame["building_id"].astype(str).to_numpy()
    else:
        groups = (frame["cycle_id"].astype(str) + "::" + frame["building_id"].astype(str)).to_numpy()
    rows, settings = [], []
    for train_index, test_index in LeaveOneGroupOut().split(frame, groups=groups):
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index]
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        held = str(groups[test_index][0])
        settings.append({"held_out_group": held, "validation_view": view, **parameters})
        for source, value in zip(test.to_dict("records"), predicted):
            rows.append({
                "candidate": candidate.name, "validation_view": view, "held_out_group": held,
                "cycle_id": source["cycle_id"], "building_id": source["building_id"],
                "review_day": int(source["review_day"]), "as_of_date": source["as_of_date"],
                "checkpoint_status": source["checkpoint_status"], "actual": float(source["actual"]),
                "predicted": float(value), "error": float(value - source["actual"]), "seed": seed,
            })
    result = pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "review_day"]).reset_index(drop=True)
    return result, settings


def summarize_predictions(predictions: pd.DataFrame, outcome: str) -> dict[str, float]:
    factor = _metric_factor(outcome)
    actual = predictions["actual"].to_numpy(float)
    predicted = predictions["predicted"].to_numpy(float)
    errors = predicted - actual
    cycle_metrics = []
    for cycle, group in predictions.groupby("cycle_id"):
        cycle_metrics.append((str(cycle), mean_squared_error(group["actual"], group["predicted"]) ** 0.5 * factor))
    building_metrics = []
    for _, group in predictions.groupby(["cycle_id", "building_id"]):
        building_metrics.append(mean_squared_error(group["actual"], group["predicted"]) ** 0.5 * factor)
    return {
        "cycle_macro_rmse": float(np.mean([value for _, value in cycle_metrics])),
        "cycle_rmse_std": float(np.std([value for _, value in cycle_metrics], ddof=1)),
        "worst_cycle_rmse": float(max(value for _, value in cycle_metrics)),
        "building_cycle_balanced_rmse": float(np.mean(building_metrics)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5 * factor),
        "mae": float(mean_absolute_error(actual, predicted) * factor),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(errors) * factor),
    }


def checkpoint_metrics(predictions: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for day, group in predictions.groupby("review_day"):
        metric = summarize_predictions(group, outcome)
        rows.append({"review_day": int(day), "validated_checkpoint": bool(day in CHECKPOINTS), "n": len(group), **metric})
    return pd.DataFrame(rows).sort_values("review_day")


def _one_se_select(comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ordered = comparison.sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
    lowest = ordered.iloc[0]
    threshold = float(lowest["cycle_macro_rmse"] + lowest["cycle_rmse_std"] / math.sqrt(len(DEVELOPMENT_CYCLES)))
    eligible = ordered.loc[ordered["cycle_macro_rmse"].le(threshold)].sort_values(["complexity", "cycle_macro_rmse", "candidate"])
    selected = eligible.iloc[0]
    return str(selected["candidate"]), {
        "lowest_error_candidate": str(lowest["candidate"]),
        "lowest_cycle_macro_rmse": float(lowest["cycle_macro_rmse"]),
        "one_se_threshold": threshold,
        "selected_candidate": str(selected["candidate"]),
        "selected_cycle_macro_rmse": float(selected["cycle_macro_rmse"]),
    }


def grouped_cvplus_intervals(frame: pd.DataFrame, candidate: Candidate, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct group-aware CV+ intervals using only each outer fold's training cycles."""
    rows: list[dict[str, Any]] = []
    groups = frame["cycle_id"].astype(str).to_numpy()
    for outer_train_idx, outer_test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        outer_train = frame.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = frame.iloc[outer_test_idx].reset_index(drop=True)
        parameters = tune_candidate(outer_train, candidate, seed)
        inner_groups = outer_train["cycle_id"].astype(str).to_numpy()
        fold_predictions: list[np.ndarray] = []
        residual_groups: list[tuple[np.ndarray, np.ndarray]] = []
        for fit_idx, calibration_idx in LeaveOneGroupOut().split(outer_train, groups=inner_groups):
            fitted = fit_candidate(outer_train.iloc[fit_idx], candidate, parameters, seed)
            calibration = outer_train.iloc[calibration_idx]
            calibration_prediction = predict_candidate(fitted, calibration, candidate)
            residual = np.abs(calibration["actual"].to_numpy(float) - calibration_prediction)
            scale = np.sqrt(calibration["days_to_day35"].to_numpy(float) + 1.0)
            residual_groups.append((residual, residual / scale))
            fold_predictions.append(predict_candidate(fitted, outer_test, candidate))
        center = np.mean(np.vstack(fold_predictions), axis=0)
        pooled = np.concatenate([item[0] for item in residual_groups])
        normalized = np.concatenate([item[1] for item in residual_groups])
        for level in (0.8, 0.9):
            quantile_level = min(1.0, math.ceil((len(pooled) + 1) * level) / len(pooled))
            q_pool = float(np.quantile(pooled, quantile_level, method="higher"))
            q_normalized = float(np.quantile(normalized, quantile_level, method="higher"))
            for idx, source in outer_test.iterrows():
                scale = math.sqrt(float(source["days_to_day35"]) + 1.0)
                for method, width in (("grouped_cvplus_pooled", q_pool), ("age_normalized_cvplus", q_normalized * scale)):
                    lower, upper = float(center[idx] - width), float(center[idx] + width)
                    if candidate.outcome == "recovery":
                        lower = max(0.0, lower)
                        upper = min(float(source["current_value"]), upper)
                    else:
                        lower, upper = max(100.0, lower), min(3500.0, upper)
                    rows.append({
                        "cycle_id": source["cycle_id"], "building_id": source["building_id"],
                        "review_day": int(source["review_day"]), "actual": float(source["actual"]),
                        "predicted": float(center[idx]), "method": method, "level": level,
                        "lower": lower, "upper": upper, "width": upper - lower,
                        "covered": bool(lower <= float(source["actual"]) <= upper),
                    })
    intervals = pd.DataFrame(rows)
    calibration = intervals.groupby(["method", "level", "review_day"], as_index=False).agg(
        coverage=("covered", "mean"), mean_width=("width", "mean"), n=("covered", "size")
    )
    return intervals, calibration


def temporal_stress(frame: pd.DataFrame, candidate: Candidate, seed: int = SEED) -> tuple[pd.DataFrame, dict[str, float]]:
    order = [cycle for cycle in DEVELOPMENT_CYCLES if cycle in set(frame["cycle_id"])]
    rows = []
    for position in range(2, len(order)):
        training_cycles, held_cycle = order[:position], order[position]
        train = frame.loc[frame["cycle_id"].isin(training_cycles)].reset_index(drop=True)
        test = frame.loc[frame["cycle_id"].eq(held_cycle)]
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        predicted = predict_candidate(fitted, test, candidate)
        for source, value in zip(test.to_dict("records"), predicted):
            rows.append({"cycle_id": held_cycle, "building_id": source["building_id"], "review_day": source["review_day"], "actual": source["actual"], "predicted": value})
    predictions = pd.DataFrame(rows)
    return predictions, summarize_predictions(predictions, candidate.outcome)


def _lookup(candidates: Iterable[Candidate], name: str) -> Candidate:
    return next(candidate for candidate in candidates if candidate.name == name)


def _fit_final_and_audit(development: pd.DataFrame, audit: pd.DataFrame, candidate: Candidate, seed: int) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    parameters = tune_candidate(development, candidate, seed)
    fitted = fit_candidate(development, candidate, parameters, seed)
    predicted = predict_candidate(fitted, audit, candidate)
    output = audit[["cycle_id", "building_id", "review_day", "as_of_date", "actual", "checkpoint_status"]].copy()
    output["predicted"] = predicted
    output["error"] = output["predicted"] - output["actual"]
    return fitted, output, summarize_predictions(output, candidate.outcome)


def _tree_shap(frame: pd.DataFrame, candidate: Candidate, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    groups = frame["cycle_id"].astype(str).to_numpy()
    global_rows, local_rows, shap_blocks, value_blocks = [], [], [], []
    names: list[str] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx]
        parameters = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, parameters, seed)
        pipeline = fitted["model"]
        imputer = pipeline.named_steps["impute"]
        model = pipeline.named_steps["model"]
        transformed = imputer.transform(test[fitted["features"]])
        names = list(imputer.get_feature_names_out(fitted["features"]))
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(transformed))
        shap_blocks.append(values)
        value_blocks.append(transformed)
        for row_pos, source in enumerate(test.to_dict("records")):
            for feature, shap_value, feature_value in zip(names, values[row_pos], transformed[row_pos]):
                local_rows.append({
                    "candidate": candidate.name, "cycle_id": source["cycle_id"], "building_id": source["building_id"],
                    "review_day": source["review_day"], "feature": feature,
                    "feature_value": float(feature_value), "shap_value": float(shap_value),
                })
    all_values = np.vstack(shap_blocks)
    all_features = np.vstack(value_blocks)
    for index, feature in enumerate(names):
        direction = np.corrcoef(all_features[:, index], all_values[:, index])[0, 1] if np.std(all_features[:, index]) > 0 else np.nan
        global_rows.append({"candidate": candidate.name, "feature": feature, "mean_abs_shap": float(np.mean(np.abs(all_values[:, index]))), "direction_correlation": direction})
    return pd.DataFrame(global_rows).sort_values("mean_abs_shap", ascending=False), pd.DataFrame(local_rows), all_values, all_features, names


def _permutation_importance(frame: pd.DataFrame, candidate: Candidate, seed: int) -> pd.DataFrame:
    groups = frame["cycle_id"].astype(str).to_numpy()
    rows = []
    rng = np.random.default_rng(seed)
    factor = _metric_factor(candidate.outcome)
    for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
        train, test = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)
        params = tune_candidate(train, candidate, seed)
        fitted = fit_candidate(train, candidate, params, seed)
        base = mean_squared_error(test["actual"], predict_candidate(fitted, test, candidate)) ** 0.5 * factor
        features = fitted.get("features", RECOVERY_FEATURES if candidate.outcome == "recovery" else WEIGHT_FEATURES)
        for feature in features:
            shuffled = test.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            changed = mean_squared_error(test["actual"], predict_candidate(fitted, shuffled, candidate)) ** 0.5 * factor
            rows.append({"held_out_cycle": str(groups[test_idx][0]), "feature": feature, "rmse_increase": changed - base})
    return pd.DataFrame(rows).groupby("feature", as_index=False).agg(mean_rmse_increase=("rmse_increase", "mean"), fold_std=("rmse_increase", "std")).sort_values("mean_rmse_increase", ascending=False)


def _feature_block_ablation(frame: pd.DataFrame, candidate: Candidate, seed: int) -> pd.DataFrame:
    all_features = RECOVERY_FEATURES if candidate.outcome == "recovery" else WEIGHT_FEATURES
    blocks = {
        "all_biology_aware_features": all_features,
        "without_environment": [feature for feature in all_features if not any(token in feature for token in ("temperature", "humidity", "environment"))],
        "without_peer_context": [feature for feature in all_features if "peer" not in feature],
        "without_trajectory": [feature for feature in all_features if not any(token in feature for token in (("mortality", "loss", "cusum") if candidate.outcome == "recovery" else ("weight", "target", "adg", "gain", "trajectory")))],
        "without_data_quality": [feature for feature in all_features if not any(token in feature for token in ("missing", "coverage", "staleness", "reconciliation", "increase"))],
    }
    groups = frame["cycle_id"].astype(str).to_numpy()
    rows = []
    for block, features in blocks.items():
        fold_rows = []
        for train_idx, test_idx in LeaveOneGroupOut().split(frame, groups=groups):
            train, test = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx]
            parameters = tune_candidate(train, candidate, seed)
            if candidate.outcome == "recovery":
                biological_fit = _fit_negative_binomial(train)
                train_base = _predict_negative_binomial(biological_fit, train)
                test_base = _predict_negative_binomial(biological_fit, test)
            else:
                train_base, _ = _kalman_predict(train, 0.0005, 0.005)
                test_base, _ = _kalman_predict(test, 0.0005, 0.005)
            model = _tree_pipeline(candidate, parameters, seed)
            model.fit(train[features], train["actual"].to_numpy(float) - train_base, model__sample_weight=train["sample_weight"].to_numpy(float))
            prediction = test_base + np.asarray(model.predict(test[features]), dtype=float)
            if candidate.outcome == "recovery":
                prediction = np.minimum(np.clip(prediction, 0, 1), test["current_value"].to_numpy(float))
            else:
                prediction = np.clip(prediction, 100, 3500)
            fold = test[["cycle_id", "building_id", "review_day", "actual"]].copy()
            fold["predicted"] = prediction
            fold_rows.append(fold)
        metric = summarize_predictions(pd.concat(fold_rows, ignore_index=True), candidate.outcome)
        rows.append({"feature_block": block, "feature_count": len(features), **metric})
    return pd.DataFrame(rows).sort_values("cycle_macro_rmse")


def _pymc_gompertz_posterior(development: pd.DataFrame, output: Path, seed: int) -> dict[str, Any]:
    """Fit a development-only Bayesian NLME diagnostic; never used for audit selection."""
    latest = development.sort_values("review_day").drop_duplicates(["cycle_id", "building_id"], keep="last")
    unit_lookup = {key: index for index, key in enumerate(latest[["cycle_id", "building_id"]].itertuples(index=False, name=None))}
    days, weights, unit_index = [], [], []
    for row in latest.itertuples(index=False):
        key = (row.cycle_id, row.building_id)
        for day in CHECKPOINTS:
            value = getattr(row, f"weight_day{day}_g")
            if pd.notna(value):
                days.append(float(day)); weights.append(float(value)); unit_index.append(unit_lookup[key])
        days.append(35.0); weights.append(float(row.actual)); unit_index.append(unit_lookup[key])
    coords = {"observation": np.arange(len(days)), "building_cycle": np.arange(len(unit_lookup))}
    with pm.Model(coords=coords) as model:
        day_data = pm.Data("day", np.asarray(days), dims="observation")
        unit_data = pm.Data("unit", np.asarray(unit_index), dims="observation")
        asymptote = pm.LogNormal("asymptote_g", mu=np.log(2400), sigma=0.25)
        rate = pm.HalfNormal("growth_rate", sigma=0.12)
        midpoint = pm.Normal("midpoint_day", mu=22, sigma=7)
        unit_sd = pm.HalfNormal("building_cycle_log_sd", sigma=0.15)
        unit_offset = pm.Normal("building_cycle_offset", mu=0, sigma=unit_sd, dims="building_cycle")
        observation_sd = pm.HalfNormal("observation_sd_g", sigma=150)
        expected = asymptote * pm.math.exp(-pm.math.exp(-rate * (day_data - midpoint))) * pm.math.exp(unit_offset[unit_data])
        pm.Normal("weight_g", mu=expected, sigma=observation_sd, observed=np.asarray(weights), dims="observation")
        trace = pm.sample(draws=500, tune=500, chains=4, cores=1, random_seed=seed, target_accept=0.92, progressbar=False)
    path = output / "bayesian_gompertz_development_posterior.joblib"
    joblib.dump(trace, path)
    summary = az.summary(trace, var_names=["asymptote_g", "growth_rate", "midpoint_day", "building_cycle_log_sd", "observation_sd_g"])
    summary.to_csv(output / "bayesian_gompertz_posterior_summary.csv")
    max_rhat = float(pd.to_numeric(summary["r_hat"], errors="coerce").max())
    min_ess = float(pd.to_numeric(summary["ess_bulk"], errors="coerce").min())
    return {"path": str(path), "sha256": _sha(path), "max_rhat": max_rhat, "min_bulk_ess": min_ess, "diagnostic_reliable": bool(max_rhat <= 1.01 and min_ess >= 400), "diagnostics": summary.reset_index().to_dict("records")}


def _data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        unit = ""
        if column.endswith("_g") or "_g_" in column:
            unit = "grams"
        elif column.endswith("_kg"):
            unit = "kilograms"
        elif "per_1000" in column:
            unit = "per 1,000 beginning birds"
        elif column.endswith("_pct") or column.endswith("_pp"):
            unit = "percent or percentage points"
        elif column.endswith("_c"):
            unit = "degrees Celsius"
        rows.append({
            "field": column, "dtype": str(frame[column].dtype), "unit": unit,
            "missing_count": int(frame[column].isna().sum()),
            "description": "As-of feature or metadata; see feature schema for model use.",
        })
    return pd.DataFrame(rows)


def _plot_outputs(
    output: Path, outcome: str, comparison: pd.DataFrame, oof: pd.DataFrame,
    checkpoints: pd.DataFrame, intervals: pd.DataFrame, shap_global: pd.DataFrame,
    shap_values: np.ndarray | None, feature_values: np.ndarray | None, feature_names: list[str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unit = "Recovery (%)" if outcome == "recovery" else "Day 35 bodyweight (g)"
    factor = _metric_factor(outcome)
    best_names = comparison.head(5)["candidate"].tolist()

    figure, axis = plt.subplots(figsize=(9, 5))
    top = comparison.head(5).sort_values("cycle_macro_rmse")
    axis.barh(top["candidate"].str.replace("_", " "), top["cycle_macro_rmse"], color="#2f6f4e")
    axis.set_xlabel("Cycle-macro RMSE (percentage points)" if outcome == "recovery" else "Cycle-macro RMSE (g)")
    axis.set_title("Top five biology-aware candidates")
    figure.tight_layout(); figure.savefig(output / "top_five_comparison.png", dpi=180); plt.close(figure)

    selected = oof.loc[oof["candidate"].eq(comparison.iloc[0]["candidate"])].copy()
    figure, axis = plt.subplots(figsize=(7, 6))
    colors = {7: "#1f77b4", 14: "#ff7f0e", 21: "#2ca02c", 28: "#d62728"}
    between = selected.loc[~selected["review_day"].isin(CHECKPOINTS)]
    axis.scatter(between["actual"] * factor, between["predicted"] * factor, s=16, alpha=0.25, c="#9aa0a6", label="Between checkpoints")
    for day in CHECKPOINTS:
        group = selected.loc[selected["review_day"].eq(day)]
        axis.scatter(group["actual"] * factor, group["predicted"] * factor, s=34, alpha=0.8, c=colors[day], label=f"Day {day}")
    limits = [min((selected[["actual", "predicted"]] * factor).min()) * 0.98, max((selected[["actual", "predicted"]] * factor).max()) * 1.02]
    axis.plot(limits, limits, "--", color="#333333", linewidth=1)
    axis.set(xlabel=f"Actual {unit}", ylabel=f"Held-out predicted {unit}", title="Actual versus predicted by forecast day")
    axis.legend(frameon=False, fontsize=8); figure.tight_layout(); figure.savefig(output / "actual_vs_predicted_by_day.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(checkpoints["review_day"], checkpoints["rmse"], marker="o", color="#245c43", label="RMSE")
    axes[0].plot(checkpoints["review_day"], checkpoints["mae"], marker="s", color="#e08b35", label="MAE")
    axes[0].set_ylabel("Error (pp)" if outcome == "recovery" else "Error (g)"); axes[0].legend(frameon=False)
    axes[0].set_title("Forecast error as evidence accumulates")
    interval_view = intervals.loc[(intervals["method"].eq("age_normalized_cvplus")) & (intervals["level"].eq(0.8))]
    width = interval_view.groupby("review_day")["width"].mean() * factor
    coverage = interval_view.groupby("review_day")["covered"].mean()
    axes[1].plot(width.index, width.values, marker="o", color="#6a4c93", label="80% interval width")
    axes[1].set_ylabel("Interval width (pp)" if outcome == "recovery" else "Interval width (g)"); axes[1].set_xlabel("Forecast day")
    twin = axes[1].twinx(); twin.plot(coverage.index, coverage.values, color="#2a9d8f", alpha=0.7, label="Coverage"); twin.set_ylabel("Empirical coverage"); twin.set_ylim(0, 1.05)
    for day, color in colors.items():
        for current_axis in axes:
            current_axis.axvline(day, color=color, alpha=0.15, linewidth=4)
    figure.tight_layout(); figure.savefig(output / "accuracy_and_uncertainty_by_day.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(selected["review_day"], selected["error"] * factor, alpha=0.25, s=16, color="#3d6f5a")
    axis.axhline(0, color="#333", linestyle="--", linewidth=1)
    axis.set(xlabel="Forecast day", ylabel="Prediction error (pp)" if outcome == "recovery" else "Prediction error (g)", title="Held-out residuals by day")
    figure.tight_layout(); figure.savefig(output / "residuals_by_day.png", dpi=180); plt.close(figure)

    if not shap_global.empty:
        top_shap = shap_global.head(10).sort_values("mean_abs_shap")
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.barh(top_shap["feature"].str.replace("_", " "), top_shap["mean_abs_shap"], color="#4d7c5b")
        axis.set(xlabel="Mean |SHAP| contribution", title="Top 10 held-out SHAP drivers")
        figure.tight_layout(); figure.savefig(output / "shap_top10.png", dpi=180); plt.close(figure)
    if shap_values is not None and feature_values is not None and len(feature_names):
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, feature_values, feature_names=feature_names, max_display=12, show=False)
        plt.title("Held-out SHAP direction and magnitude")
        plt.tight_layout(); plt.savefig(output / "shap_beeswarm.png", dpi=180, bbox_inches="tight"); plt.close()
        top_two = shap_global.head(2)["feature"].tolist()
        for rank, feature in enumerate(top_two, start=1):
            if feature not in feature_names:
                continue
            index = feature_names.index(feature)
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter(feature_values[:, index], shap_values[:, index], alpha=0.35, s=16, color="#2f6f4e")
            axis.axhline(0, color="#333", linestyle="--", linewidth=1)
            axis.set(xlabel=feature.replace("_", " "), ylabel="SHAP contribution", title=f"SHAP dependence: {feature.replace('_', ' ')}")
            figure.tight_layout(); figure.savefig(output / f"shap_dependence_{rank}.png", dpi=180); plt.close(figure)


def _promotion_gate(comparison: pd.DataFrame, oof: pd.DataFrame, checkpoints: pd.DataFrame, intervals: pd.DataFrame, audit: dict[str, float], outcome: str) -> dict[str, Any]:
    lowest = comparison.iloc[0]
    baseline_name = "daily_age_remaining_loss" if outcome == "recovery" else "historical_remaining_gain"
    baseline = comparison.set_index("candidate").loc[baseline_name]
    cycle_wins = 0
    best_rows = oof.loc[oof["candidate"].eq(lowest["candidate"])]
    baseline_rows = oof.loc[oof["candidate"].eq(baseline_name)]
    for cycle in DEVELOPMENT_CYCLES:
        best_cycle = best_rows.loc[best_rows["cycle_id"].eq(cycle)]
        base_cycle = baseline_rows.loc[baseline_rows["cycle_id"].eq(cycle)]
        if mean_squared_error(best_cycle["actual"], best_cycle["predicted"]) < mean_squared_error(base_cycle["actual"], base_cycle["predicted"]):
            cycle_wins += 1
    checkpoint_table = checkpoints.loc[checkpoints["candidate"].eq(lowest["candidate"])].set_index("review_day")
    baseline_checkpoint = checkpoints.loc[checkpoints["candidate"].eq(baseline_name)].set_index("review_day")
    checkpoint_ok = all(checkpoint_table.loc[day, "rmse"] <= baseline_checkpoint.loc[day, "rmse"] * 1.05 for day in CHECKPOINTS)
    interval_view = intervals.loc[(intervals["method"].eq("age_normalized_cvplus")) & (intervals["level"].eq(0.8))]
    coverage = float(interval_view["covered"].mean())
    late_coverage = float(interval_view.loc[interval_view["review_day"].ge(28), "covered"].mean())
    bias_limit = 0.5 if outcome == "recovery" else 50.0
    gate = {
        "at_least_10pct_better_than_baseline": bool(lowest["cycle_macro_rmse"] <= baseline["cycle_macro_rmse"] * 0.9),
        "positive_held_out_r2": bool(lowest["r2"] > 0),
        "cycle_wins": cycle_wins,
        "wins_at_least_four_cycles": cycle_wins >= 4,
        "acceptable_bias": bool(abs(lowest["bias"]) <= bias_limit),
        "principal_checkpoints_stable": checkpoint_ok,
        "interval_80_coverage": coverage,
        "late_day_interval_coverage": late_coverage,
        "credible_interval_coverage": bool(0.72 <= coverage <= 0.95 and late_coverage >= 0.65),
        "later_cycle_audit_rmse": audit["rmse"],
    }
    gate["retrospective_gate_passed"] = bool(all(value for key, value in gate.items() if key not in {"cycle_wins", "interval_80_coverage", "late_day_interval_coverage", "later_cycle_audit_rmse"}))
    return gate


def _write_report(output: Path, outcome: str, manifest: dict[str, Any], comparison: pd.DataFrame, checkpoints: pd.DataFrame, shap_global: pd.DataFrame) -> Path:
    title = "Harvest Recovery" if outcome == "recovery" else "Day 35 Bodyweight"
    unit = "percentage points" if outcome == "recovery" else "grams"
    top = comparison.head(5)[["candidate", "cycle_macro_rmse", "rmse", "mae", "r2", "bias", "worst_cycle_rmse"]]
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(column) for column in frame.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |")
        return "\n".join(lines)

    table = markdown_table(top)
    principal = checkpoints.loc[checkpoints["review_day"].isin(CHECKPOINTS) & checkpoints["candidate"].eq(manifest["selection"]["lowest_error_candidate"]), ["review_day", "rmse", "mae", "r2"]]
    drivers = ", ".join(shap_global.head(10)["feature"].str.replace("_", " ").tolist()) or "No compatible tree challenger."
    text = f"""# Project Canary Biology-Aware {title} Research Round

## Executive conclusion

The lowest-error candidate was **{manifest['selection']['lowest_error_candidate'].replace('_', ' ')}** with **{manifest['selection']['lowest_cycle_macro_rmse']:.2f} {unit}** cycle-macro RMSE. The one-standard-error selection was **{manifest['selection']['selected_candidate'].replace('_', ' ')}**. The retrospective promotion gate **{'passed' if manifest['promotion_gate']['retrospective_gate_passed'] else 'did not pass'}**. This remains research/shadow evidence and does not replace Canary's operational forecast.

## Top five nested harvest-cycle LOGO results

{table}

![Top-five comparison](figures/top_five_comparison.png)

## Does accuracy improve as more days are observed?

{markdown_table(principal)}

The complete Day 7–34 learning curve is in `daily_metrics.csv`. Intervening days update from available mortality, population, environment and weight-freshness evidence; they do not pretend that a stale weight was newly measured. Later-day interval width must be interpreted together with empirical coverage.

![Accuracy and uncertainty by day](figures/accuracy_and_uncertainty_by_day.png)

## Actual versus predicted and residuals

![Actual versus predicted](figures/actual_vs_predicted_by_day.png)

![Residuals by day](figures/residuals_by_day.png)

## Explainability

Top held-out SHAP signals: {drivers}. SHAP explains the residual-correction tree challenger, not biological causality. A feature pushing a forecast up or down is a predictive association and cannot justify an intervention by itself.

![SHAP drivers](figures/shap_top10.png)

![SHAP direction](figures/shap_beeswarm.png)

## Trust and scope

Primary validation holds out one complete harvest cycle and tunes only within the remaining cycles. Each building-cycle contributes equal total training weight across its 28 landmarks. The three 2026-3 buildings remain a locked later-cycle audit until design and selection are frozen. Exact building identity and feed are excluded. The sample still contains only six independent development cycles, environmental history is incomplete, and recovery audit endpoints remain provisional. These limitations prevent a retrospective result from being treated as production-ready.
"""
    path = output / f"PROJECT_CANARY_BIOLOGY_AWARE_{outcome.upper()}_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def _run_outcome(dataset: CanaryDataset, root: Path, output_root: Path, outcome: str, seed: int) -> dict[str, Any]:
    candidates = RECOVERY_CANDIDATES if outcome == "recovery" else WEIGHT_CANDIDATES
    output = output_root / ("recovery" if outcome == "recovery" else "bodyweight")
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    snapshots = build_daily_landmarks(dataset, outcome)
    development = snapshots.loc[snapshots["role"].eq("development")].reset_index(drop=True)
    audit = snapshots.loc[snapshots["role"].eq("later_cycle_audit")].reset_index(drop=True)
    development.to_csv(output / "daily_model_ready_landmarks.csv", index=False)
    audit.to_csv(output / "locked_later_cycle_landmarks.csv", index=False)
    _data_dictionary(snapshots).to_csv(output / "data_dictionary.csv", index=False)
    registry = pd.DataFrame([{**asdict(candidate), "grid": json.dumps(parameter_grid(candidate)), "grid_size": len(parameter_grid(candidate))} for candidate in candidates])
    registry.to_csv(output / "experiment_registry.csv", index=False)

    prediction_path = output / "all_nested_logo_predictions.csv"
    comparison_path = output / "candidate_comparison.csv"
    if prediction_path.exists() and comparison_path.exists():
        print(f"[{outcome}] reusing completed deterministic nested LOGO predictions", flush=True)
        oof = pd.read_csv(prediction_path)
        comparison = pd.read_csv(comparison_path)
    else:
        predictions, comparison_rows, hyperparameters = [], [], {}
        for position, candidate in enumerate(candidates, start=1):
            print(f"[{outcome}] biology-aware nested LOGO {position}/{len(candidates)}: {candidate.name}", flush=True)
            prediction, settings = evaluate_nested_logo(development, candidate, seed=seed)
            predictions.append(prediction)
            comparison_rows.append({"candidate": candidate.name, "family": candidate.family, "complexity": candidate.complexity, "description": candidate.description, **summarize_predictions(prediction, outcome)})
            hyperparameters[candidate.name] = settings
        oof = pd.concat(predictions, ignore_index=True)
        comparison = pd.DataFrame(comparison_rows).sort_values(["cycle_macro_rmse", "complexity", "candidate"]).reset_index(drop=True)
        comparison["rank"] = np.arange(1, len(comparison) + 1)
        oof.to_csv(prediction_path, index=False)
        comparison.to_csv(comparison_path, index=False)
        (output / "nested_hyperparameters.json").write_text(json.dumps(hyperparameters, indent=2, default=_json_default), encoding="utf-8")
    comparison.head(5).to_csv(output / "top_five_models.csv", index=False)

    selected_name, selection = _one_se_select(comparison)
    lowest_name = selection["lowest_error_candidate"]
    selected = _lookup(candidates, selected_name)
    lowest = _lookup(candidates, lowest_name)
    daily = pd.concat([checkpoint_metrics(group, outcome).assign(candidate=name) for name, group in oof.groupby("candidate")], ignore_index=True)
    daily.to_csv(output / "daily_metrics.csv", index=False)
    daily.loc[daily["review_day"].isin(CHECKPOINTS)].to_csv(output / "checkpoint_metrics.csv", index=False)

    finalists = list(dict.fromkeys([selected_name, lowest_name, "daily_age_remaining_loss" if outcome == "recovery" else "historical_remaining_gain"]))
    secondary_rows, temporal_rows, temporal_predictions = [], [], []
    for name in finalists:
        candidate = _lookup(candidates, name)
        for view in ("building_label", "building_cycle"):
            pred, _ = evaluate_nested_logo(development, candidate, view=view, seed=seed)
            secondary_rows.append({"candidate": name, "validation_view": view, **summarize_predictions(pred, outcome)})
        pred, metric = temporal_stress(development, candidate, seed)
        pred["candidate"] = name; temporal_predictions.append(pred)
        temporal_rows.append({"candidate": name, **metric})
    pd.DataFrame(secondary_rows).to_csv(output / "secondary_logo_metrics.csv", index=False)
    pd.concat(temporal_predictions, ignore_index=True).to_csv(output / "temporal_predictions.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(output / "temporal_metrics.csv", index=False)

    intervals, interval_calibration = grouped_cvplus_intervals(development, lowest, seed)
    intervals.to_csv(output / "grouped_cvplus_intervals.csv", index=False)
    interval_calibration.to_csv(output / "interval_calibration_by_day.csv", index=False)

    explanation_pool = [candidate for candidate in candidates if candidate.family in {"residual_xgboost", "residual_lightgbm"}]
    explanation = min(explanation_pool, key=lambda item: float(comparison.set_index("candidate").loc[item.name, "cycle_macro_rmse"]))
    shap_global, shap_local, shap_values, feature_values, feature_names = _tree_shap(development, explanation, seed)
    shap_global.to_csv(output / "held_out_shap_global.csv", index=False)
    shap_local.to_csv(output / "held_out_shap_local.csv", index=False)
    importance_candidate = lowest if lowest.family not in {"persistence", "age_baseline", "target_ratio", "kalman", "negative_binomial", "gompertz", "logistic", "blend"} else explanation
    importance = _permutation_importance(development, importance_candidate, seed)
    importance.to_csv(output / "held_out_permutation_importance.csv", index=False)
    ablation = _feature_block_ablation(development, explanation, seed)
    ablation.to_csv(output / "feature_block_ablation.csv", index=False)

    final_fit, audit_predictions, audit_metrics = _fit_final_and_audit(development, audit, lowest, seed)
    audit_predictions.to_csv(output / "later_cycle_audit_predictions.csv", index=False)
    artifact_path = output / "lowest_error_research_shadow.joblib"
    payload = {"round_version": ROUND_VERSION, "deployment_status": "research_shadow", "candidate": asdict(lowest), "fitted": final_fit, "feature_schema": RECOVERY_FEATURES if outcome == "recovery" else WEIGHT_FEATURES}
    joblib.dump(payload, artifact_path)
    restored = joblib.load(artifact_path)
    parity = float(np.max(np.abs(predict_candidate(final_fit, development, lowest) - predict_candidate(restored["fitted"], development, lowest))))
    if parity > 1e-10:
        raise AssertionError("Research artifact reload changed predictions.")

    posterior = None
    if outcome == "weight":
        posterior_path = output / "bayesian_gompertz_development_posterior.joblib"
        if posterior_path.exists():
            posterior = {"path": str(posterior_path), "sha256": _sha(posterior_path), "status": "reused_deterministic_development_diagnostic"}
        else:
            try:
                posterior = _pymc_gompertz_posterior(development, output, seed)
            except Exception as exc:
                posterior = {"status": "failed", "reason": str(exc)}
                (output / "bayesian_gompertz_posterior_failure.txt").write_text(str(exc), encoding="utf-8")

    checkpoint_table = daily.loc[daily["review_day"].isin(CHECKPOINTS)]
    gate = _promotion_gate(comparison, oof, checkpoint_table, intervals, audit_metrics, outcome)
    manifest = {
        "round_version": ROUND_VERSION, "outcome": outcome, "seed": seed,
        "design_frozen_before_audit": True,
        "source_rows": int(dataset.quality.source_rows), "development_rows": len(development),
        "development_building_cycles": int(development[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "daily_landmarks": list(LANDMARK_DAYS), "validated_checkpoints": list(CHECKPOINTS),
        "primary_validation": "Nested Leave-One-Group-Out by complete harvest cycle; cycle-macro RMSE primary.",
        "selection": selection, "explanation_model": explanation.name,
        "promotion_gate": gate, "later_cycle_audit_metrics": audit_metrics,
        "artifact": {"path": str(artifact_path), "sha256": _sha(artifact_path), "prediction_parity_max_abs": parity},
        "posterior_diagnostic": posterior,
        "shap_warning": "SHAP and feature importance indicate predictive association, not causation.",
        "operational_models_changed": False,
    }
    _plot_outputs(figures, outcome, comparison, oof, daily.loc[daily["candidate"].eq(lowest_name)], intervals, shap_global, shap_values, feature_values, feature_names)
    report = _write_report(output, outcome, manifest, comparison, checkpoint_table, shap_global)
    manifest["technical_report"] = str(report)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest


def run_biology_aware_round(workbook: str | Path, output: str | Path | None = None, *, seed: int = SEED) -> dict[str, Any]:
    """Execute the isolated biology-aware research round from the raw workbook."""
    if seed != SEED:
        raise ValueError(f"This frozen research design uses seed {SEED}; received {seed}.")
    root = Path(__file__).resolve().parents[1]
    workbook_path = Path(workbook).resolve()
    output_root = Path(output).resolve() if output else root / "outputs" / "biology_aware_modeling_round"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(workbook_path)
    source_profile, source_checks = build_source_quality_audit(workbook_path, dataset)
    recovery_preview = build_daily_landmarks(dataset, "recovery")
    weight_preview = build_daily_landmarks(dataset, "weight")
    extra_checks = pd.DataFrame([
        {"check": "recovery_daily_landmark_count", "severity": "critical", "status": "pass", "failed_rows": int(len(recovery_preview.loc[recovery_preview["role"].eq("development")]) != 31 * len(LANDMARK_DAYS)), "detail": f"Expected {31 * len(LANDMARK_DAYS)} development landmarks."},
        {"check": "bodyweight_daily_landmark_count", "severity": "critical", "status": "pass", "failed_rows": int(len(weight_preview.loc[weight_preview["role"].eq("development")]) != 31 * len(LANDMARK_DAYS)), "detail": f"Expected {31 * len(LANDMARK_DAYS)} development landmarks."},
        {"check": "future_evidence", "severity": "critical", "status": "pass", "failed_rows": int((recovery_preview["max_source_day_used"] > recovery_preview["review_day"]).sum() + (weight_preview["max_source_day_used"] > weight_preview["review_day"]).sum()), "detail": "All as-of evidence must be timestamped on/before review day."},
        {"check": "audit_excluded_from_development", "severity": "critical", "status": "pass", "failed_rows": int((recovery_preview.loc[recovery_preview["role"].eq("development"), "cycle_id"] == AUDIT_CYCLE).sum() + (weight_preview.loc[weight_preview["role"].eq("development"), "cycle_id"] == AUDIT_CYCLE).sum()), "detail": "2026-3 must remain locked."},
        {"check": "equal_building_cycle_weight", "severity": "critical", "status": "pass", "failed_rows": int((recovery_preview.groupby(["cycle_id", "building_id"])["sample_weight"].sum().round(10).nunique() != 1) or (weight_preview.groupby(["cycle_id", "building_id"])["sample_weight"].sum().round(10).nunique() != 1)), "detail": "Each building-cycle must contribute equal total landmark weight."},
        {"check": "environment_history_missing", "severity": "warning", "status": "flagged", "failed_rows": int(((recovery_preview["environment_recorded_days"] == 0)).sum()), "detail": "Environmental features remain secondary when no history is available."},
    ])
    checks = pd.concat([source_checks, extra_checks], ignore_index=True)
    checks.to_csv(output_root / "data_quality_checks.csv", index=False)
    (output_root / "source_audit.json").write_text(json.dumps(source_profile, indent=2, default=_json_default), encoding="utf-8")
    if checks.loc[checks["severity"].eq("critical"), "failed_rows"].sum() > 0:
        raise AssertionError("Critical biology-aware data-quality checks failed.")

    recovery = _run_outcome(dataset, root, output_root, "recovery", seed)
    weight = _run_outcome(dataset, root, output_root, "weight", seed)
    combined = pd.concat([
        pd.read_csv(output_root / "recovery" / "top_five_models.csv").assign(outcome="recovery"),
        pd.read_csv(output_root / "bodyweight" / "top_five_models.csv").assign(outcome="bodyweight"),
    ], ignore_index=True, sort=False)
    combined.to_csv(output_root / "top_five_models_by_outcome.csv", index=False)
    benchmark_root = root / "outputs" / "farmwide_modeling_optimization_round_2026_08_13_refresh"
    if not benchmark_root.exists():
        benchmark_root = root / "outputs" / "farmwide_modeling_optimization_round"
    benchmark_rows = []
    for outcome_name, folder in (("recovery", "recovery"), ("bodyweight", "bodyweight")):
        benchmark_path = benchmark_root / folder / "top_five_models.csv"
        if benchmark_path.exists():
            benchmark_best = pd.read_csv(benchmark_path).sort_values("cycle_macro_rmse").iloc[0]
            biology_best = combined.loc[combined["outcome"].eq(outcome_name)].sort_values("cycle_macro_rmse").iloc[0]
            benchmark_rows.append({
                "outcome": outcome_name, "frozen_benchmark_root": str(benchmark_root),
                "frozen_best_candidate": benchmark_best["candidate"],
                "frozen_cycle_macro_rmse": float(benchmark_best["cycle_macro_rmse"]),
                "biology_best_candidate": biology_best["candidate"],
                "biology_cycle_macro_rmse": float(biology_best["cycle_macro_rmse"]),
                "rmse_change_pct": (float(biology_best["cycle_macro_rmse"]) / float(benchmark_best["cycle_macro_rmse"]) - 1) * 100,
            })
    benchmark_comparison = pd.DataFrame(benchmark_rows)
    benchmark_comparison.to_csv(output_root / "comparison_to_frozen_farmwide_benchmark.csv", index=False)
    manifest = {
        "round_version": ROUND_VERSION, "created": pd.Timestamp.now(tz="Asia/Manila").isoformat(),
        "source_sha256": _sha(workbook_path), "source": source_profile,
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "statsmodels", "pymc", "arviz", "xgboost", "lightgbm", "shap")},
        "validation": {"primary": "nested harvest-cycle LOGO", "robustness": "building-label LOGO", "optimistic": "building-cycle LOGO", "temporal": "expanding-window"},
        "frozen_benchmark": benchmark_comparison.to_dict("records"),
        "outcomes": {"recovery": recovery, "weight": weight},
        "operational_models_changed": False, "research_only": True,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest
