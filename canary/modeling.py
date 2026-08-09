"""Leakage-safe feature construction and offline model training for Sprint 3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CanaryDataset


FEATURE_COLUMNS = [
    "cycle_day",
    "beginning_inventory",
    "percentage_alive",
    "cumulative_mortality_rate",
    "mortality_daily_per_1000",
    "mortality_recent_3d_per_1000",
    "mortality_trend_delta_per_1000",
    "feed_daily_per_1000_birds",
    "feed_cumulative_per_1000_birds",
    "latest_weight_kg",
    "weight_target_kg",
    "weight_gap_pct",
    "weight_measurement_day",
    "weight_staleness_days",
    "temperature_recent_avg_c",
    "humidity_recent_avg_pct",
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
    """Include histories whose maximum recorded date is before the source cutoff."""

    cutoff = source_complete_date(dataset)
    cycle_ends = dataset.cycles.groupby("cycle_id")["end_date"].max()
    return cycle_ends.loc[cycle_ends <= cutoff].index.astype(str).tolist()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


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

    recent_environment = operational.tail(3)
    temperature = _numeric(recent_environment["temperature_avg_c"]).mean()
    humidity = _numeric(recent_environment["humidity_avg_pct"]).mean()

    return {
        "cycle_id": cycle_id,
        "building_id": building_id,
        "as_of_date": as_of,
        "cycle_day": age,
        "beginning_inventory": beginning,
        "percentage_alive": percentage_alive,
        "cumulative_mortality_rate": 1.0 - percentage_alive if pd.notna(percentage_alive) else np.nan,
        "mortality_daily_per_1000": (
            float(latest["mortality_daily"]) / beginning * 1000
            if pd.notna(latest["mortality_daily"])
            else np.nan
        ),
        "mortality_recent_3d_per_1000": recent_mortality,
        "mortality_trend_delta_per_1000": mortality_delta,
        "feed_daily_per_1000_birds": feed_daily / beginning * 1000 if pd.notna(feed_daily) else np.nan,
        "feed_cumulative_per_1000_birds": feed_cumulative / beginning * 1000 if pd.notna(feed_cumulative) else np.nan,
        "latest_weight_kg": latest_weight,
        "weight_target_kg": weight_target,
        "weight_gap_pct": weight_gap,
        "weight_measurement_day": weight_day,
        "weight_staleness_days": weight_staleness,
        "temperature_recent_avg_c": temperature,
        "humidity_recent_avg_pct": humidity,
        "is_lags_building": 1.0 if building_id.startswith("Lags") else 0.0,
        # A deliberately simple, leakage-safe baseline: assume the latest observed
        # survival rate holds through harvest. It never uses the future recorded end date.
        "naive_recovery_projection": percentage_alive,
        "naive_weight_projection": (
            latest_weight / weight_target * 2.0
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
    if kind == "ridge":
        regressor = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    if kind == "random_forest":
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
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def _clip(prediction: np.ndarray, outcome: str) -> np.ndarray:
    return np.clip(prediction, 0.0, 1.0) if outcome == "recovery" else np.clip(prediction, 0.1, 3.5)


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


def train_outcome_model(
    dataset: CanaryDataset,
    outcome: str,
    final_weight_labels: pd.DataFrame | None = None,
) -> TrainingResult:
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
    candidates = ["trend_naive", "historical_mean", "ridge", "random_forest"]
    if outcome == "recovery":
        candidates.append("ridge_no_weight")
    candidate_features = {
        candidate: (
            RECOVERY_NO_WEIGHT_FEATURE_COLUMNS
            if candidate == "ridge_no_weight"
            else FEATURE_COLUMNS
        )
        for candidate in candidates
    }
    predictions = {candidate: np.full(len(snapshots), np.nan) for candidate in candidates}
    fold_mae: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    logo = LeaveOneGroupOut()

    for train_index, test_index in logo.split(x, y, groups):
        for candidate in candidates:
            if candidate == "trend_naive":
                column = "naive_recovery_projection" if outcome == "recovery" else "naive_weight_projection"
                fallback = float(np.nanmean(y[train_index]))
                raw = snapshots.iloc[test_index][column].to_numpy(float)
                prediction = np.where(np.isnan(raw), fallback, raw)
            else:
                model = _pipeline("ridge" if candidate == "ridge_no_weight" else candidate)
                columns = candidate_features[candidate]
                model.fit(x.iloc[train_index][columns], y[train_index])
                prediction = model.predict(x.iloc[test_index][columns])
            prediction = _clip(np.asarray(prediction, dtype=float), outcome)
            predictions[candidate][test_index] = prediction
            fold_mae[candidate].append(float(mean_absolute_error(y[test_index], prediction)))

    target = 0.95 if outcome == "recovery" else 2.0
    metrics: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        prediction = predictions[candidate]
        residual = y - prediction
        cycle_mae = [
            float(mean_absolute_error(y[groups == cycle], prediction[groups == cycle]))
            for cycle in np.unique(groups)
        ]
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
        }

    best_macro_mae = min(
        float(metrics[candidate]["cycle_macro_mae"]) for candidate in candidates
    )
    eligible = {
        candidate
        for candidate in candidates
        if float(metrics[candidate]["cycle_macro_mae"]) <= best_macro_mae * 1.05
    }
    simplicity_order = [
        "trend_naive",
        "historical_mean",
        "ridge_no_weight",
        "ridge",
        "random_forest",
    ]
    selected = next(candidate for candidate in simplicity_order if candidate in eligible)
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
        final_model = _pipeline("ridge" if selected == "ridge_no_weight" else selected)
        final_model.fit(x[candidate_features[selected]], y)

    global_feature_importance = (
        _ridge_importance(final_model, candidate_features[selected])
        if selected in {"ridge", "ridge_no_weight"}
        else []
    )

    manifest = {
        "outcome": outcome,
        "model_version": (
            "recovery-0.5.0"
            if outcome == "recovery"
            else ("weight-final-0.4.0" if final_weight_labels is not None else "weight-proxy-0.3.0")
        ),
        "selected_model": selected,
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
            "Prototype - cycle-held-out validation"
            if outcome == "recovery" or final_weight_labels is not None
            else "Experimental - proxy label and small sample"
        ),
        "metrics": metrics,
        "selected_metrics": metrics[selected],
        "selection_metric": "cycle_macro_mae_within_5pct_then_simplest",
        "selection_tolerance_pct": 5.0,
        "global_feature_importance": global_feature_importance,
        "feature_importance_interpretation": (
            "Standardized Ridge coefficients show which inputs the fitted model relies on and whether higher values push its raw estimate up or down. They are associations, not causal effects."
            if global_feature_importance
            else "Formal coefficient importance is not available for the selected formula or baseline model."
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
