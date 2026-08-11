"""Leakage-safe feature construction and offline model training for Sprint 3."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:  # XGBoost is an optional challenger; some macOS installs lack libomp.
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - depends on the local operating-system runtime.
    XGBRegressor = None  # type: ignore[assignment,misc]

from .data import CanaryDataset


FEATURE_COLUMNS = [
    "cycle_day",
    "forecast_horizon_days",
    "beginning_inventory",
    "percentage_alive",
    "expected_survival_path",
    "survival_gap_pp",
    "cumulative_mortality_rate",
    "population_loss_pct",
    "mortality_daily_per_1000",
    "mortality_recent_3d_per_1000",
    "mortality_trend_delta_per_1000",
    "feed_daily_per_1000_birds",
    "feed_cumulative_per_1000_birds",
    "feed_daily_kg_per_bird",
    "latest_weight_kg",
    "weight_target_kg",
    "weight_gap_pct",
    "weight_measurement_day",
    "weight_staleness_days",
    "temperature_recent_avg_c",
    "temperature_recent_min_c",
    "temperature_recent_max_c",
    "temperature_recent_range_c",
    "temperature_deviation_from_band_c",
    "humidity_recent_avg_pct",
    "humidity_recent_min_pct",
    "humidity_recent_max_pct",
    "humidity_recent_range_pp",
    "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d",
    "environment_staleness_days",
    "is_lags_building",
]

WEIGHT_PROGRESS_FEATURES = {
    "latest_weight_kg",
    "weight_target_kg",
    "weight_gap_pct",
    "weight_measurement_day",
    "weight_staleness_days",
}
RECOVERY_NO_WEIGHT_FEATURE_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in WEIGHT_PROGRESS_FEATURES
    and column != "cumulative_mortality_rate"
]
# Compact on purpose: each item adds a distinct, review-date-safe signal.  We
# exclude algebraic duplicates (forecast horizon, population loss, survival
# gap), raw temperature/humidity summaries that duplicate band deviation, and
# feed until the farm confirms its unit.
RECOVERY_CORE_FEATURE_COLUMNS = [
    "cycle_day",
    "percentage_alive",
    "mortality_recent_3d_per_1000",
    "mortality_trend_delta_per_1000",
    "weight_gap_pct",
    "weight_staleness_days",
    "temperature_deviation_from_band_c",
    "humidity_deviation_from_band_pp",
    "environment_out_of_band_days_7d",
    "environment_staleness_days",
]
RECOVERY_DECISION_DAYS = (7, 14, 21, 28)


@dataclass(frozen=True)
class TrainingResult:
    outcome: str
    selected_model: str
    manifest: dict[str, Any]
    model: object | None


def source_complete_date(dataset: CanaryDataset) -> pd.Timestamp:
    complete = dataset.daily.loc[dataset.daily["daily_complete"], "record_date"]
    if complete.empty:
        raise ValueError("No complete daily observations are available for model training.")
    return pd.Timestamp(complete.max()).normalize()


def complete_cycle_ids(dataset: CanaryDataset) -> list[str]:
    """Return completed historical cycles while always excluding the latest cycle.

    The latest placement cycle is the live decision cycle in the capstone.  It
    remains excluded even if later checkpoint rows have already been entered,
    preventing an earlier as-of forecast from learning its own future outcome.
    """

    cutoff = source_complete_date(dataset)
    cycle_ends = dataset.cycles.groupby("cycle_id")["end_date"].max()
    cycle_starts = dataset.cycles.groupby("cycle_id")["start_date"].min()
    latest_cycle = str(cycle_starts.idxmax())
    completed = cycle_ends.loc[
        (cycle_ends <= cutoff) & (cycle_ends.index.astype(str) != latest_cycle)
    ]
    return completed.index.astype(str).tolist()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


@lru_cache(maxsize=1)
def _risk_environment_rules() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "config" / "risk_rules.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _age_band(ranges: list[dict[str, Any]], age: int) -> dict[str, Any] | None:
    return next(
        (
            band
            for band in ranges
            if int(band["minimum_age"]) <= age <= int(band["maximum_age"])
        ),
        None,
    )


def _outside_band(value: object, band: dict[str, Any] | None) -> float:
    if band is None or pd.isna(value):
        return np.nan
    observed = float(value)
    lower = float(band["minimum"])
    upper = float(band["maximum"])
    return max(lower - observed, 0.0, observed - upper)


def load_final_weight_labels(source: object) -> pd.DataFrame:
    """Read only the building-level final liveweight field from the farm summary."""

    raw = pd.read_excel(source, sheet_name="Performance Summary", header=2, engine="openpyxl")
    raw.columns = [str(column).replace("\n", " ").strip() for column in raw.columns]
    required = {"Farm", "Batch", "Year", "House No.", "Date Delivered", "Ave Live Weight (kg)"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Farm performance summary is missing: {', '.join(missing)}")

    labels = raw[list(required)].copy()
    labels["cycle_id"] = (
        pd.to_numeric(labels["Year"], errors="coerce").astype("Int64").astype("string")
        + "-"
        + pd.to_numeric(labels["Batch"], errors="coerce").astype("Int64").astype("string")
    )
    farm_prefix = labels["Farm"].astype("string").str.strip().map(
        {"Taghangin": "Tags", "Lagundi": "Lags"}
    )
    house = pd.to_numeric(labels["House No."], errors="coerce").astype("Int64").astype("string")
    labels["building_id"] = farm_prefix + " " + house
    labels["final_average_weight_kg"] = pd.to_numeric(
        labels["Ave Live Weight (kg)"], errors="coerce"
    )
    labels["summary_record_date"] = pd.to_datetime(labels["Date Delivered"], errors="coerce").dt.normalize()
    labels["weight_label_source"] = getattr(source, "name", Path(str(source)).name)
    return labels[
        [
            "cycle_id",
            "building_id",
            "final_average_weight_kg",
            "summary_record_date",
            "weight_label_source",
        ]
    ].dropna(subset=["cycle_id", "building_id"])


def extract_feature_row(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
) -> dict[str, object] | None:
    """Create one feature row using only records dated on or before ``as_of``."""

    as_of = pd.Timestamp(as_of).normalize()
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == cycle_id)
        & (dataset.cycles["building_id"] == building_id)
    ]
    if meta.empty:
        return None
    meta_row = meta.iloc[0]
    start = pd.Timestamp(meta_row["start_date"]).normalize()
    if as_of < start:
        return None

    history = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & (dataset.daily["record_date"] <= as_of)
    ].sort_values("age_day")
    operational = history.loc[history["operational_recorded"]]
    if operational.empty:
        return None
    latest = operational.iloc[-1]
    age = int(latest["age_day"])
    beginning = float(meta_row["beginning_inventory"])
    population = float(latest["population"]) if pd.notna(latest["population"]) else np.nan
    percentage_alive = population / beginning if beginning and pd.notna(population) else np.nan

    mortality = operational.loc[operational["mortality_recorded"]].copy()
    mortality_rate = _numeric(mortality["mortality_daily"]) / beginning * 1000
    recent_mortality = float(mortality_rate.tail(3).mean()) if not mortality_rate.empty else np.nan
    baseline_mortality = (
        float(mortality_rate.iloc[-10:-3].mean()) if len(mortality_rate.iloc[-10:-3]) >= 2 else np.nan
    )
    mortality_delta = (
        max(0.0, recent_mortality - baseline_mortality)
        if pd.notna(recent_mortality) and pd.notna(baseline_mortality)
        else np.nan
    )

    feed = operational.loc[operational["feed_recorded"]]
    feed_daily = float(_numeric(feed["feed_daily_bags"]).iloc[-1]) if not feed.empty else np.nan
    feed_cumulative = float(_numeric(feed["feed_daily_bags"]).sum()) if not feed.empty else np.nan
    feed_per_bird = operational.loc[operational["feed_daily_kg_per_bird"].notna()]
    feed_daily_kg_per_bird = (
        float(_numeric(feed_per_bird["feed_daily_kg_per_bird"]).iloc[-1])
        if not feed_per_bird.empty
        else np.nan
    )

    weights = history.loc[history["weight_measured"]]
    if weights.empty:
        latest_weight = weight_day = weight_target = weight_gap = weight_staleness = np.nan
    else:
        weight = weights.iloc[-1]
        latest_weight = float(weight["bodyweight_kg"])
        weight_day = int(weight["age_day"])
        target_match = dataset.targets.loc[dataset.targets["age_day"] == weight_day, "target_weight_kg"]
        weight_target = float(target_match.iloc[0]) if not target_match.empty else np.nan
        weight_gap = (
            (weight_target - latest_weight) / weight_target * 100
            if pd.notna(weight_target) and weight_target > 0
            else np.nan
        )
        weight_staleness = max(0, age - weight_day)

    environment_rows = operational.loc[
        operational[
            [
                "temperature_avg_c",
                "temperature_min_c",
                "temperature_max_c",
                "humidity_avg_pct",
                "humidity_min_pct",
                "humidity_max_pct",
            ]
        ]
        .notna()
        .any(axis=1)
    ]
    recent_environment = environment_rows.tail(3)
    temperature = _numeric(recent_environment["temperature_avg_c"]).mean()
    temperature_min = _numeric(recent_environment["temperature_min_c"]).min()
    temperature_max = _numeric(recent_environment["temperature_max_c"]).max()
    humidity = _numeric(recent_environment["humidity_avg_pct"]).mean()
    humidity_min = _numeric(recent_environment["humidity_min_pct"]).min()
    humidity_max = _numeric(recent_environment["humidity_max_pct"]).max()
    environment_staleness = (
        max(0, age - int(environment_rows.iloc[-1]["age_day"]))
        if not environment_rows.empty
        else np.nan
    )
    rules = _risk_environment_rules()
    temperature_band = _age_band(rules["temperature_ranges_c"], age)
    humidity_band = _age_band(rules["humidity_ranges_pct"], age)
    temperature_deviation = _outside_band(temperature, temperature_band)
    humidity_deviation = _outside_band(humidity, humidity_band)
    recent_seven = environment_rows.tail(7)
    out_of_band_days = 0
    recorded_environment_days = 0
    for _, environment_row in recent_seven.iterrows():
        environment_age = int(environment_row["age_day"])
        temperature_day_band = _age_band(rules["temperature_ranges_c"], environment_age)
        humidity_day_band = _age_band(rules["humidity_ranges_pct"], environment_age)
        temperature_day_deviation = _outside_band(
            environment_row.get("temperature_avg_c"), temperature_day_band
        )
        humidity_day_deviation = _outside_band(
            environment_row.get("humidity_avg_pct"), humidity_day_band
        )
        recorded = pd.notna(temperature_day_deviation) or pd.notna(humidity_day_deviation)
        if recorded:
            recorded_environment_days += 1
            out_of_band_days += int(
                (pd.notna(temperature_day_deviation) and temperature_day_deviation > 0)
                or (pd.notna(humidity_day_deviation) and humidity_day_deviation > 0)
            )

    expected_survival = max(0.95, 1.0 - min(age, 35) / 35 * 0.05)
    survival_gap_pp = (
        max(0.0, expected_survival - percentage_alive) * 100
        if pd.notna(percentage_alive)
        else np.nan
    )

    return {
        "cycle_id": cycle_id,
        "building_id": building_id,
        "as_of_date": as_of,
        "cycle_day": age,
        "forecast_horizon_days": max(0, 35 - age),
        "beginning_inventory": beginning,
        "percentage_alive": percentage_alive,
        "expected_survival_path": expected_survival,
        "survival_gap_pp": survival_gap_pp,
        "cumulative_mortality_rate": 1.0 - percentage_alive if pd.notna(percentage_alive) else np.nan,
        "population_loss_pct": (1.0 - percentage_alive) * 100 if pd.notna(percentage_alive) else np.nan,
        "mortality_daily_per_1000": (
            float(latest["mortality_daily"]) / beginning * 1000
            if pd.notna(latest["mortality_daily"])
            else np.nan
        ),
        "mortality_recent_3d_per_1000": recent_mortality,
        "mortality_trend_delta_per_1000": mortality_delta,
        "feed_daily_per_1000_birds": feed_daily / beginning * 1000 if pd.notna(feed_daily) else np.nan,
        "feed_cumulative_per_1000_birds": feed_cumulative / beginning * 1000 if pd.notna(feed_cumulative) else np.nan,
        "feed_daily_kg_per_bird": feed_daily_kg_per_bird,
        "latest_weight_kg": latest_weight,
        "weight_target_kg": weight_target,
        "weight_gap_pct": weight_gap,
        "weight_measurement_day": weight_day,
        "weight_staleness_days": weight_staleness,
        "temperature_recent_avg_c": temperature,
        "temperature_recent_min_c": temperature_min,
        "temperature_recent_max_c": temperature_max,
        "temperature_recent_range_c": (
            temperature_max - temperature_min
            if pd.notna(temperature_min) and pd.notna(temperature_max)
            else np.nan
        ),
        "temperature_deviation_from_band_c": temperature_deviation,
        "humidity_recent_avg_pct": humidity,
        "humidity_recent_min_pct": humidity_min,
        "humidity_recent_max_pct": humidity_max,
        "humidity_recent_range_pp": (
            humidity_max - humidity_min
            if pd.notna(humidity_min) and pd.notna(humidity_max)
            else np.nan
        ),
        "humidity_deviation_from_band_pp": humidity_deviation,
        "environment_out_of_band_days_7d": (
            float(out_of_band_days) if recorded_environment_days else np.nan
        ),
        "environment_staleness_days": environment_staleness,
        "is_lags_building": 1.0 if building_id.startswith("Lags") else 0.0,
        # A deliberately simple, leakage-safe baseline: assume the latest observed
        # survival rate holds through harvest. It never uses the future recorded end date.
        "naive_recovery_projection": percentage_alive,
        "naive_weight_projection": (
            latest_weight / weight_target * 1.8
            if pd.notna(latest_weight) and pd.notna(weight_target) and weight_target > 0
            else np.nan
        ),
    }


def _eligible_weight_labels(dataset: CanaryDataset) -> pd.DataFrame:
    labels = dataset.cycles[["cycle_id", "building_id", "ending_weight_week5_kg"]].copy()
    measured = (
        dataset.daily.loc[dataset.daily["weight_measured"]]
        .sort_values("age_day")
        .groupby(["cycle_id", "building_id"], as_index=False)
        .tail(1)[["cycle_id", "building_id", "age_day", "record_date", "bodyweight_kg"]]
        .rename(
            columns={
                "age_day": "weight_label_day",
                "record_date": "weight_label_date",
                "bodyweight_kg": "latest_measured_weight_kg",
            }
        )
    )
    labels = labels.merge(measured, on=["cycle_id", "building_id"], how="left")
    labels["weight_label_valid"] = (
        labels["ending_weight_week5_kg"].notna()
        & labels["weight_label_day"].ge(35)
        & (labels["ending_weight_week5_kg"] - labels["latest_measured_weight_kg"]).abs().lt(1e-8)
    )
    return labels


def _eligible_final_weight_labels(
    dataset: CanaryDataset, final_weight_labels: pd.DataFrame
) -> pd.DataFrame:
    """Match final weights to farm cycles and conservatively reject suspect matches."""

    labels = dataset.cycles[["cycle_id", "building_id", "start_date", "end_date"]].merge(
        final_weight_labels,
        on=["cycle_id", "building_id"],
        how="left",
        validate="one_to_one",
    )
    labels["summary_to_start_days"] = (
        labels["summary_record_date"] - labels["start_date"]
    ).dt.days
    labels["weight_label_valid"] = (
        labels["final_average_weight_kg"].between(0.5, 3.5, inclusive="both")
        & labels["summary_to_start_days"].between(-14, 14, inclusive="both")
    )
    labels["weight_label_date"] = labels["end_date"]
    return labels


def build_modeling_snapshots(
    dataset: CanaryDataset,
    outcome: str,
    final_weight_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one daily snapshot per eligible building using only pre-as-of features."""

    if outcome not in {"recovery", "weight"}:
        raise ValueError("Outcome must be 'recovery' or 'weight'.")
    completed = set(complete_cycle_ids(dataset))
    metadata = dataset.cycles.loc[dataset.cycles["cycle_id"].isin(completed)].copy()
    if outcome == "weight":
        weight_labels = (
            _eligible_final_weight_labels(dataset, final_weight_labels)
            if final_weight_labels is not None
            else _eligible_weight_labels(dataset)
        )
        metadata = metadata.merge(
            weight_labels[
                [
                    "cycle_id",
                    "building_id",
                    "weight_label_valid",
                    "weight_label_date",
                    *(["final_average_weight_kg"] if final_weight_labels is not None else []),
                ]
            ],
            on=["cycle_id", "building_id"],
            how="left",
        )
        metadata = metadata.loc[metadata["weight_label_valid"].fillna(False)]

    rows: list[dict[str, object]] = []
    for _, meta in metadata.iterrows():
        cutoff_date = (
            pd.Timestamp(meta["weight_label_date"])
            if outcome == "weight"
            else pd.Timestamp(meta["end_date"])
        )
        daily = dataset.daily.loc[
            (dataset.daily["cycle_id"] == meta["cycle_id"])
            & (dataset.daily["building_id"] == meta["building_id"])
            & dataset.daily["daily_complete"]
            & (dataset.daily["record_date"] < cutoff_date)
        ].sort_values("record_date")
        for current in daily["record_date"].drop_duplicates():
            feature = extract_feature_row(
                dataset, str(meta["cycle_id"]), str(meta["building_id"]), pd.Timestamp(current)
            )
            if feature is None:
                continue
            feature["label_date"] = cutoff_date
            feature["target"] = (
                float(meta["final_recovery_rate"])
                if outcome == "recovery"
                else float(
                    meta["final_average_weight_kg"]
                    if final_weight_labels is not None
                    else meta["ending_weight_week5_kg"]
                )
            )
            rows.append(feature)
    return pd.DataFrame(rows).sort_values(["cycle_id", "building_id", "as_of_date"]).reset_index(drop=True)


def _pipeline(kind: str) -> object:
    if kind == "historical_mean":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", DummyRegressor()),
            ]
        )
    if kind in {"linear_regression", "ridge"}:
        regressor = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearRegression() if kind == "linear_regression" else Ridge(alpha=10.0),
                ),
            ]
        )
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    if kind == "gradient_boosting":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=75,
                        learning_rate=0.04,
                        max_depth=2,
                        min_samples_leaf=4,
                        random_state=42,
                    ),
                ),
            ]
        )
    if kind == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost is unavailable: install the operating-system OpenMP runtime.")
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
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
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def _clip(prediction: np.ndarray, outcome: str) -> np.ndarray:
    return np.clip(prediction, 0.0, 1.0) if outcome == "recovery" else np.clip(prediction, 0.1, 3.5)


def _building_cycle_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every independent building-cycle equal total training influence."""

    keys = frame["cycle_id"].astype(str) + "::" + frame["building_id"].astype(str)
    counts = keys.map(keys.value_counts()).to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.mean()


def _fit_params_for(model: object, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Route sample weights through the fitted sklearn wrapper."""

    if isinstance(model, TransformedTargetRegressor):
        return {"model__sample_weight": weights}
    return {"model__sample_weight": weights}


def _parameter_options(candidate: str) -> list[dict[str, object]]:
    """Small, pre-declared grids appropriate for the limited independent sample."""

    if candidate in {"ridge", "ridge_core"}:
        return [
            {"regressor__model__alpha": alpha}
            for alpha in (10.0, 25.0, 50.0, 100.0)
        ]
    if candidate == "gradient_boosting":
        return [
            {
                "model__n_estimators": n_estimators,
                "model__learning_rate": learning_rate,
                "model__max_depth": max_depth,
                "model__min_samples_leaf": min_leaf,
            }
            for n_estimators, learning_rate, max_depth, min_leaf in (
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
                "model__learning_rate": learning_rate,
                "model__max_depth": max_depth,
                "model__min_child_weight": min_child_weight,
                "model__reg_lambda": reg_lambda,
            }
            for n_estimators, learning_rate, max_depth, min_child_weight, reg_lambda in (
                (50, 0.03, 1, 4, 10.0),
                (75, 0.04, 2, 4, 5.0),
                (100, 0.03, 2, 6, 10.0),
            )
        ]
    return [{}]


def _tune_candidate(
    candidate: str,
    x: pd.DataFrame,
    y: np.ndarray,
    cycle_groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[object, dict[str, object]]:
    """Tune only within the supplied training cycles, then refit on all of them."""

    kind = "ridge" if candidate == "ridge_core" else candidate
    base = _pipeline(kind)
    options = _parameter_options(candidate)
    if len(options) == 1 or len(np.unique(cycle_groups)) < 3:
        best_params = options[0]
    else:
        inner = LeaveOneGroupOut()
        scored: list[tuple[float, dict[str, object]]] = []
        for params in options:
            fold_errors: list[float] = []
            for train_index, valid_index in inner.split(x, y, cycle_groups):
                model = clone(base).set_params(**params)
                model.fit(
                    x.iloc[train_index],
                    y[train_index],
                    **_fit_params_for(model, weights[train_index]),
                )
                prediction = model.predict(x.iloc[valid_index])
                fold_errors.append(float(mean_absolute_error(y[valid_index], prediction)))
            scored.append((float(np.mean(fold_errors)), params))
        best_params = min(scored, key=lambda item: item[0])[1]
    fitted = clone(base).set_params(**best_params)
    fitted.fit(x, y, **_fit_params_for(fitted, weights))
    return fitted, best_params


def _cycle_bootstrap_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    repeats: int = 2000,
) -> dict[str, float]:
    """Bootstrap whole cycles, preserving the true unit of generalization."""

    rng = np.random.default_rng(42)
    cycles = np.unique(groups)
    estimates: list[float] = []
    for _ in range(repeats):
        selected = rng.choice(cycles, size=len(cycles), replace=True)
        cycle_errors = [
            mean_absolute_error(actual[groups == cycle], predicted[groups == cycle])
            for cycle in selected
        ]
        estimates.append(float(np.mean(cycle_errors)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high), "confidence": 0.95}


def _ridge_importance(model: object, feature_columns: list[str]) -> list[dict[str, object]]:
    """Return standardized Ridge coefficients as directional model reliance.

    These values describe the fitted model, not causal effects. The target-scale
    conversion makes each coefficient readable in the outcome's original unit.
    """

    regressor = model.regressor_
    imputer = regressor.named_steps["imputer"]
    ridge = regressor.named_steps["model"]
    names = list(imputer.get_feature_names_out(feature_columns))
    target_scale = float(np.asarray(model.transformer_.scale_).reshape(-1)[0])
    coefficients = np.asarray(ridge.coef_, dtype=float).reshape(-1) * target_scale
    absolute = np.abs(coefficients)
    total = float(absolute.sum())
    records = []
    for name, coefficient, magnitude in zip(names, coefficients, absolute):
        source_name = str(name)
        if source_name.startswith("missingindicator_"):
            source_name = "missing__" + source_name.removeprefix("missingindicator_")
        records.append(
            {
                "feature": source_name,
                "coefficient_per_standard_deviation": float(coefficient),
                "absolute_importance_pct": float(magnitude / total * 100) if total else 0.0,
                "direction": "Raises estimate" if coefficient > 0 else "Lowers estimate",
            }
        )
    return sorted(records, key=lambda item: item["absolute_importance_pct"], reverse=True)


def _horizon_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[0, 7, 14, 21, np.inf],
        labels=["Days 1-7", "Days 8-14", "Days 15-21", "Day 22+"],
    )


def _decision_checkpoint_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Balance recovery training around repeatable management checkpoints.

    Each building-cycle contributes at most Days 7, 14, 21, 28, plus its
    latest eligible pre-outcome snapshot. This prevents longer histories from
    dominating model selection merely because they contain more daily rows.
    """

    checkpoint = snapshots.loc[
        snapshots["cycle_day"].isin(RECOVERY_DECISION_DAYS)
    ].copy()
    latest = (
        snapshots.sort_values("as_of_date")
        .groupby(["cycle_id", "building_id"], as_index=False)
        .tail(1)
    )
    return (
        pd.concat([checkpoint, latest], ignore_index=True)
        .drop_duplicates(["cycle_id", "building_id", "as_of_date"])
        .sort_values(["cycle_id", "building_id", "as_of_date"])
        .reset_index(drop=True)
    )


def build_recovery_training_snapshots(dataset: CanaryDataset) -> pd.DataFrame:
    """Return the exact balanced, leakage-safe rows used to compare recovery models."""

    snapshots = build_modeling_snapshots(dataset, "recovery")
    return _decision_checkpoint_snapshots(snapshots)


def train_outcome_model(
    dataset: CanaryDataset,
    outcome: str,
    final_weight_labels: pd.DataFrame | None = None,
) -> TrainingResult:
    if outcome == "recovery":
        # The strengthened recovery pipeline predicts only the population loss
        # that remains after the review date.  Final recovery is then derived
        # from the accounting identity current survival - additional loss.
        from .strengthened_models import train_recovery_remaining_loss

        source_snapshots = build_modeling_snapshots(dataset, "recovery")
        snapshots = _decision_checkpoint_snapshots(source_snapshots)
        strengthened = train_recovery_remaining_loss(snapshots)
        manifest = strengthened.manifest | {
            "training_source": dataset.source_name,
            "source_complete_date": source_complete_date(dataset).date().isoformat(),
            "source_daily_snapshot_rows": int(len(source_snapshots)),
        }
        return TrainingResult(
            "recovery",
            str(manifest["selected_model"]),
            manifest,
            strengthened.model,
        )

    source_snapshots = build_modeling_snapshots(dataset, outcome, final_weight_labels)
    snapshots = (
        _decision_checkpoint_snapshots(source_snapshots)
        if outcome == "recovery"
        else source_snapshots
    )
    minimum_cycles = 3 if outcome == "recovery" else 4
    if snapshots["cycle_id"].nunique() < minimum_cycles:
        raise ValueError(f"Insufficient complete cycles to train the {outcome} model.")

    x = snapshots[FEATURE_COLUMNS]
    y = snapshots["target"].to_numpy(float)
    groups = snapshots["cycle_id"].astype(str).to_numpy()
    candidates = (
        [
            "historical_mean",
            "linear_regression",
            "ridge_core",
            "gradient_boosting",
            "xgboost",
        ]
        if outcome == "recovery"
        else ["historical_mean", "linear_regression", "ridge", "gradient_boosting", "xgboost"]
    )
    all_candidates = list(candidates)
    if XGBRegressor is None:
        candidates = [candidate for candidate in candidates if candidate != "xgboost"]
    candidate_features = {
        candidate: (
            RECOVERY_CORE_FEATURE_COLUMNS
            if outcome == "recovery" and candidate != "historical_mean"
            else RECOVERY_NO_WEIGHT_FEATURE_COLUMNS
            if candidate == "ridge_no_weight"
            else FEATURE_COLUMNS
        )
        for candidate in candidates
    }
    predictions = {candidate: np.full(len(snapshots), np.nan) for candidate in candidates}
    fold_mae: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    fold_parameters: dict[str, list[dict[str, object]]] = {
        candidate: [] for candidate in candidates
    }
    sample_weights = _building_cycle_weights(snapshots)
    logo = LeaveOneGroupOut()

    for train_index, test_index in logo.split(x, y, groups):
        for candidate in candidates:
            if candidate == "trend_naive":
                column = "naive_recovery_projection" if outcome == "recovery" else "naive_weight_projection"
                fallback = float(np.nanmean(y[train_index]))
                raw = snapshots.iloc[test_index][column].to_numpy(float)
                prediction = np.where(np.isnan(raw), fallback, raw)
                fold_parameters[candidate].append({})
            else:
                columns = candidate_features[candidate]
                model, best_params = _tune_candidate(
                    candidate,
                    x.iloc[train_index][columns],
                    y[train_index],
                    groups[train_index],
                    sample_weights[train_index],
                )
                fold_parameters[candidate].append(best_params)
                prediction = model.predict(x.iloc[test_index][columns])
            prediction = _clip(np.asarray(prediction, dtype=float), outcome)
            predictions[candidate][test_index] = prediction
            fold_mae[candidate].append(float(mean_absolute_error(y[test_index], prediction)))

    target = 0.95 if outcome == "recovery" else 1.8
    metrics: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        prediction = predictions[candidate]
        residual = y - prediction
        cycle_metrics = {
            str(cycle): {
                "rows": int(np.sum(groups == cycle)),
                "mae": float(mean_absolute_error(y[groups == cycle], prediction[groups == cycle])),
                "rmse": float(mean_squared_error(y[groups == cycle], prediction[groups == cycle]) ** 0.5),
                "bias": float(np.mean(prediction[groups == cycle] - y[groups == cycle])),
            }
            for cycle in np.unique(groups)
        }
        cycle_mae = [float(values["mae"]) for values in cycle_metrics.values()]
        horizon_metrics: dict[str, dict[str, float | int]] = {}
        bands = _horizon_band(snapshots["cycle_day"])
        for band in bands.cat.categories:
            mask = bands == band
            if mask.any():
                horizon_metrics[str(band)] = {
                    "rows": int(mask.sum()),
                    "mae": float(mean_absolute_error(y[mask], prediction[mask])),
                    "rmse": float(mean_squared_error(y[mask], prediction[mask]) ** 0.5),
                }
        metrics[candidate] = {
            "mae": float(mean_absolute_error(y, prediction)),
            "cycle_macro_mae": float(np.mean(cycle_mae)),
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "r2": float(r2_score(y, prediction)),
            "bias": float(np.mean(prediction - y)),
            "fold_mae_std": float(np.std(fold_mae[candidate])),
            "target_side_accuracy": float(np.mean((prediction >= target) == (y >= target))),
            "majority_side_accuracy": float(
                max(np.mean(y >= target), np.mean(y < target))
            ),
            "below_target_recall": float(
                np.mean(prediction[y < target] < target) if np.any(y < target) else np.nan
            ),
            "at_or_above_target_recall": float(
                np.mean(prediction[y >= target] >= target)
                if np.any(y >= target)
                else np.nan
            ),
            "confusion_matrix": {
                "actual_below_predicted_below": int(
                    np.sum((y < target) & (prediction < target))
                ),
                "actual_below_predicted_at_or_above": int(
                    np.sum((y < target) & (prediction >= target))
                ),
                "actual_at_or_above_predicted_below": int(
                    np.sum((y >= target) & (prediction < target))
                ),
                "actual_at_or_above_predicted_at_or_above": int(
                    np.sum((y >= target) & (prediction >= target))
                ),
            },
            "uncertainty_half_width_80": float(np.quantile(np.abs(residual), 0.80)),
            "horizon": horizon_metrics,
            "cycle": cycle_metrics,
            "outer_fold_best_parameters": fold_parameters[candidate],
        }

    baseline_name = "historical_mean"
    learned_candidates = [
        candidate for candidate in candidates if candidate != baseline_name
    ]
    best_macro_mae = min(
        float(metrics[candidate]["cycle_macro_mae"])
        for candidate in learned_candidates
    )
    selection_tolerance_pct = 10.0 if outcome == "recovery" else 5.0
    eligible = {
        candidate
        for candidate in learned_candidates
        if float(metrics[candidate]["cycle_macro_mae"])
        <= best_macro_mae * (1 + selection_tolerance_pct / 100)
    }
    simplicity_order = [
        "ridge_core",
        "linear_regression",
        "ridge",
        "gradient_boosting",
        "xgboost",
    ]
    baseline_metrics = metrics[baseline_name]
    gate_eligible = {
        candidate
        for candidate in eligible
        if (
            (
                float(baseline_metrics["cycle_macro_mae"])
                - float(metrics[candidate]["cycle_macro_mae"])
            )
            / float(baseline_metrics["cycle_macro_mae"])
            * 100
            >= 10.0
            and float(metrics[candidate]["r2"]) > 0
        )
    }
    research_champion = next(
        candidate
        for candidate in simplicity_order
        if candidate in (gate_eligible or eligible)
    )
    champion_metrics = metrics[research_champion]
    improvement_pct = (
        (
            float(baseline_metrics["cycle_macro_mae"])
            - float(champion_metrics["cycle_macro_mae"])
        )
        / float(baseline_metrics["cycle_macro_mae"])
        * 100
    )
    regression_gate = bool(improvement_pct >= 10.0 and float(champion_metrics["r2"]) > 0)
    classification_gate = bool(
        float(champion_metrics["target_side_accuracy"])
        > float(champion_metrics["majority_side_accuracy"])
        and float(champion_metrics["below_target_recall"]) > 0
        and float(champion_metrics["at_or_above_target_recall"]) > 0
    )
    # Recovery remains useful as a continuous estimate when its regression gate
    # passes, even if it must not be presented as a validated 95%-side classifier.
    selected = (
        research_champion
        if regression_gate and (outcome == "recovery" or classification_gate)
        else baseline_name
    )
    day14_mask = snapshots["cycle_day"].eq(14).to_numpy()
    day14_prediction = predictions[selected][day14_mask]
    day14_actual = y[day14_mask]
    day14_rows = snapshots.loc[
        day14_mask, ["cycle_id", "building_id", "as_of_date"]
    ].copy()
    day14_rows["predicted"] = day14_prediction
    day14_rows["actual"] = day14_actual
    day14_rows["error"] = day14_prediction - day14_actual
    day14_rows["absolute_error"] = np.abs(day14_rows["error"])
    day14_metrics = (
        {
            "building_cycles": int(len(day14_rows)),
            "mae": float(mean_absolute_error(day14_actual, day14_prediction)),
            "rmse": float(mean_squared_error(day14_actual, day14_prediction) ** 0.5),
            "mean_error": float(np.mean(day14_prediction - day14_actual)),
            "target_side_accuracy": float(
                np.mean((day14_prediction >= target) == (day14_actual >= target))
            ),
            "majority_side_accuracy": float(
                max(np.mean(day14_actual >= target), np.mean(day14_actual < target))
            ),
            "below_target_recall": float(
                np.mean(day14_prediction[day14_actual < target] < target)
                if np.any(day14_actual < target)
                else np.nan
            ),
            "at_or_above_target_recall": float(
                np.mean(day14_prediction[day14_actual >= target] >= target)
                if np.any(day14_actual >= target)
                else np.nan
            ),
            "actual_at_or_above_target": int(np.sum(day14_actual >= target)),
            "actual_below_target": int(np.sum(day14_actual < target)),
            "predicted_at_or_above_target": int(np.sum(day14_prediction >= target)),
            "predicted_below_target": int(np.sum(day14_prediction < target)),
        }
        if len(day14_rows)
        else {}
    )
    final_model: object | None
    if selected == "trend_naive":
        final_model = None
    else:
        final_model, final_parameters = _tune_candidate(
            selected,
            x[candidate_features[selected]],
            y,
            groups,
            sample_weights,
        )
    if selected == "trend_naive":
        final_parameters = {}

    secondary_metrics: dict[str, dict[str, float]] = {}
    within_cycle_groups = (
        snapshots["cycle_id"].astype(str)
        + "::"
        + snapshots["building_id"].astype(str)
    ).to_numpy()
    for candidate in candidates:
        secondary_prediction = np.full(len(snapshots), np.nan)
        for train_index, test_index in LeaveOneGroupOut().split(x, y, within_cycle_groups):
            columns = candidate_features[candidate]
            if candidate == "trend_naive":
                raw = snapshots.iloc[test_index][
                    "naive_recovery_projection"
                    if outcome == "recovery"
                    else "naive_weight_projection"
                ].to_numpy(float)
                fallback = float(np.nanmean(y[train_index]))
                secondary_prediction[test_index] = np.where(np.isnan(raw), fallback, raw)
            else:
                model = _pipeline("ridge" if candidate == "ridge_core" else candidate)
                if candidate == selected and final_parameters:
                    model.set_params(**final_parameters)
                model.fit(
                    x.iloc[train_index][columns],
                    y[train_index],
                    **_fit_params_for(model, sample_weights[train_index]),
                )
                secondary_prediction[test_index] = model.predict(x.iloc[test_index][columns])
        secondary_prediction = _clip(secondary_prediction, outcome)
        secondary_metrics[candidate] = {
            "mae": float(mean_absolute_error(y, secondary_prediction)),
            "rmse": float(mean_squared_error(y, secondary_prediction) ** 0.5),
            "r2": float(r2_score(y, secondary_prediction)),
        }

    permutation_records: list[dict[str, object]] = []
    if selected != "trend_naive":
        columns = candidate_features[selected]
        accumulated: dict[str, list[float]] = {column: [] for column in columns}
        for train_index, test_index in logo.split(x, y, groups):
            model, _ = _tune_candidate(
                selected,
                x.iloc[train_index][columns],
                y[train_index],
                groups[train_index],
                sample_weights[train_index],
            )
            result = permutation_importance(
                model,
                x.iloc[test_index][columns],
                y[test_index],
                scoring="neg_mean_absolute_error",
                n_repeats=20,
                random_state=42,
            )
            for column, importance in zip(columns, result.importances_mean):
                accumulated[column].append(float(max(0.0, importance)))
        averaged = {name: float(np.mean(values)) for name, values in accumulated.items()}
        total = sum(averaged.values())
        permutation_records = sorted(
            [
                {
                    "feature": name,
                    "mean_mae_increase": value,
                    "relative_importance_pct": value / total * 100 if total else 0.0,
                }
                for name, value in averaged.items()
            ],
            key=lambda item: item["mean_mae_increase"],
            reverse=True,
        )

    global_feature_importance = (
        _ridge_importance(final_model, candidate_features[selected])
        if selected in {"ridge", "ridge_no_weight", "ridge_core"}
        else []
    )

    manifest = {
        "outcome": outcome,
        "model_version": (
        "recovery-1.0.0"
            if outcome == "recovery"
            else ("weight-final-0.4.0" if final_weight_labels is not None else "weight-proxy-0.3.0")
        ),
        "selected_model": selected,
        "research_champion": research_champion,
        "operational_model": selected,
        "model_kind": "formula" if selected == "trend_naive" else "fitted",
        "feature_schema_version": "features-0.3.0",
        "feature_columns": candidate_features[selected],
        "all_candidate_features": candidate_features,
        "training_source": (
            dataset.source_name
            if outcome == "recovery" or final_weight_labels is None
            else f"{dataset.source_name} + Farm Performance Summary.xlsx (final average weight only)"
        ),
        "source_complete_date": source_complete_date(dataset).date().isoformat(),
        "training_cycles": sorted(snapshots["cycle_id"].astype(str).unique().tolist()),
        "training_building_cycles": int(snapshots[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "training_snapshot_rows": int(len(snapshots)),
        "source_daily_snapshot_rows": int(len(source_snapshots)),
        "snapshot_sampling": (
            "Days 7, 14, 21, and 28 plus the latest eligible pre-outcome snapshot per building-cycle"
            if outcome == "recovery"
            else "Every eligible leakage-safe daily snapshot"
        ),
        "target": target,
        "label_definition": (
            "Population on the last recorded daily date divided by beginning inventory; used as a capstone proxy for harvest recovery because confirmed harvest status is unavailable"
            if outcome == "recovery"
            else (
                "Building-level Ave Live Weight (kg) from Farm Performance Summary.xlsx, used directly as the final-harvest label"
                if final_weight_labels is not None
                else "Latest observed bodyweight on Day 35 or later; experimental proxy, not validated final-harvest weight"
            )
        ),
        "status": (
            (
                "Validated prototype — continuous forecast passed; 95% target-side classification remains unvalidated"
                if outcome == "recovery" and regression_gate and not classification_gate
                else "Validated prototype — champion gates passed"
                if regression_gate and classification_gate
                else "Experimental — champion gates failed; transparent baseline used operationally"
            )
            if outcome == "recovery" or final_weight_labels is not None
            else "Experimental - proxy label and small sample"
        ),
        "metrics": metrics,
        "selected_metrics": metrics[selected],
        "research_champion_metrics": metrics[research_champion],
        "selection_metric": "nested_leave_one_complete_cycle_out_cycle_macro_mae",
        "selection_tolerance_pct": selection_tolerance_pct,
        "nested_validation": {
            "outer_split": "Leave one complete harvest cycle out",
            "inner_split": "Leave one complete remaining harvest cycle out",
            "optimization_metric": "Cycle-balanced mean absolute error",
            "preprocessing_scope": "Imputation, scaling and tuning are fitted inside each training fold",
            "independent_unit": "Building-cycle; repeated snapshots receive equal total training weight",
        },
        "champion_gates": {
            "baseline": baseline_name,
            "baseline_improvement_pct": improvement_pct,
            "requires_at_least_10pct_mae_improvement": improvement_pct >= 10.0,
            "requires_positive_r2": float(champion_metrics["r2"]) > 0,
            "regression_gate_passed": regression_gate,
            "requires_better_than_majority_target_side_accuracy": (
                float(champion_metrics["target_side_accuracy"])
                > float(champion_metrics["majority_side_accuracy"])
            ),
            "requires_recall_for_both_target_sides": (
                float(champion_metrics["below_target_recall"]) > 0
                and float(champion_metrics["at_or_above_target_recall"]) > 0
            ),
            "target_classification_gate_passed": classification_gate,
            "operational_fallback_applied": selected == baseline_name,
        },
        "primary_whole_cycle_bootstrap_mae_95ci": _cycle_bootstrap_interval(
            y, predictions[selected], groups
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
        "final_fit_parameters": final_parameters,
        "selection_note": (
            "For recovery, candidates are compared on compact leakage-safe inputs. The simplest candidate within 10% of the best learned cycle-balanced MAE is preferred only if it also improves the historical-mean baseline by at least 10% and keeps positive whole-cycle R²."
            if outcome == "recovery"
            else "Select the simplest candidate within 5% of the best cycle-balanced MAE."
        ),
        "unavailable_candidates": (
            {"xgboost": "Not run in this local build because the required macOS OpenMP runtime (libomp) is unavailable."}
            if XGBRegressor is None
            else {}
        ),
        "global_feature_importance": global_feature_importance,
        "held_out_permutation_importance": permutation_records,
        "feature_importance_interpretation": (
            "Standardized Ridge coefficients and held-out permutation importance show model reliance and predictive association. They are not causal effects and cannot justify treatment-size claims."
            if selected == "ridge_regression" and global_feature_importance
            else "Held-out permutation importance and building-specific linear contribution breakdowns show predictive reliance and association. They are not causal effects and cannot justify treatment-size claims."
            if selected == "linear_regression" and permutation_records
            else "Held-out permutation importance shows predictive reliance and association. It is not causal evidence and cannot justify treatment-size claims."
            if permutation_records
            else "Formal held-out feature importance is not available for the selected formula or baseline model."
        ),
        "day14_backtest_metrics": day14_metrics,
        "day14_backtest": [
            {
                "cycle_id": str(record["cycle_id"]),
                "building_id": str(record["building_id"]),
                "as_of_date": pd.Timestamp(record["as_of_date"]).date().isoformat(),
                "predicted": float(record["predicted"]),
                "actual": float(record["actual"]),
                "error": float(record["error"]),
                "absolute_error": float(record["absolute_error"]),
            }
            for record in day14_rows.to_dict(orient="records")
        ],
    }
    if outcome == "weight" and final_weight_labels is not None:
        manifest["verified_outcomes"] = (
            snapshots[["cycle_id", "building_id", "target"]]
            .drop_duplicates()
            .rename(columns={"target": "final_average_weight_kg"})
            .sort_values(["cycle_id", "building_id"])
            .to_dict(orient="records")
        )
    return TrainingResult(outcome, selected, manifest, final_model)


def save_training_result(result: TrainingResult, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / f"{result.outcome}_manifest.json"
    manifest_path.write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    model_path = model_dir / f"{result.outcome}_model.joblib"
    if result.model is not None:
        joblib.dump(result.model, model_path)
    elif model_path.exists():
        model_path.unlink()
