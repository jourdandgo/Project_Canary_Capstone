"""Daily outcome forecasting and plain-language traceability for Sprint 3."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import CanaryDataset
from .day35 import load_day35_manifest, project_day35_weight
from .modeling import FEATURE_COLUMNS, extract_feature_row


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RECOVERY_TARGET = 0.95
WEIGHT_TARGET_KG = 1.8


@lru_cache(maxsize=8)
def load_model_bundle(outcome: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> tuple[dict[str, Any], object | None]:
    model_dir = Path(model_dir)
    manifest_path = model_dir / f"{outcome}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing forecast manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = model_dir / f"{outcome}_model.joblib"
    model = joblib.load(model_path) if manifest["model_kind"] == "fitted" else None
    return manifest, model


def _predict(
    feature: dict[str, object],
    outcome: str,
    manifest: dict[str, Any],
    model: object | None,
) -> float:
    if manifest["model_kind"] == "formula":
        key = "naive_recovery_projection" if outcome == "recovery" else "naive_weight_projection"
        value = feature.get(key)
        if value is None or pd.isna(value):
            raise ValueError("The formula model does not have enough inputs for this forecast.")
        prediction = float(value)
    else:
        columns = manifest.get("feature_columns", FEATURE_COLUMNS)
        values = pd.DataFrame([{column: feature.get(column, np.nan) for column in columns}])
        prediction = float(model.predict(values)[0])
    return float(np.clip(prediction, 0.0, 1.0) if outcome == "recovery" else np.clip(prediction, 0.1, 3.5))


def _confidence_text(outcome: str, cycle_day: int, manifest: dict[str, Any]) -> str:
    if outcome == "weight" and manifest["selected_model"] == "historical_mean":
        return (
            "Baseline estimate · today’s building signals did not improve "
            "cycle-held-out accuracy"
        )
    if cycle_day <= 7:
        timing = "Early estimate"
    elif cycle_day <= 14:
        timing = "Developing estimate"
    elif cycle_day <= 21:
        timing = "Later-cycle estimate"
    else:
        timing = "Later-cycle estimate"
    return f"{timing} · prototype trained on {len(manifest['training_cycles'])} recorded cycles"


def attach_forecasts(
    dataset: CanaryDataset,
    snapshot: pd.DataFrame,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Attach independent forecasts without changing the rules-based risk fields."""

    recovery_manifest, recovery_model = load_model_bundle("recovery", model_dir)
    day35_manifest = load_day35_manifest(Path(model_dir) / "day35_weight_manifest.json")
    rows: list[dict[str, object]] = []

    for _, source in snapshot.iterrows():
        row = source.to_dict()
        row.update(
            {
                "predicted_final_recovery": np.nan,
                "unconstrained_recovery_prediction": np.nan,
                "recovery_constraint_applied": False,
                "recovery_target_gap_pp": np.nan,
                "recovery_interval_low": np.nan,
                "recovery_interval_high": np.nan,
                "recovery_forecast_status": "Not available",
                "recovery_confidence": "Not available",
                "recovery_model_version": recovery_manifest["model_version"],
                "recovery_model_name": recovery_manifest["selected_model"],
                "projected_day35_weight_kg": np.nan,
                "day35_weight_target_gap_kg": np.nan,
                "day35_weight_interval_low_kg": np.nan,
                "day35_weight_interval_high_kg": np.nan,
                "day35_weight_status": "Not available",
                "day35_weight_confidence": "Not available",
                "day35_weight_scope": "Not available",
                "day35_weight_model_version": day35_manifest["model_version"],
                "day35_weight_model_name": day35_manifest["selected_model"],
                "forecast_as_of": pd.Timestamp(source["as_of_date"]),
            }
        )

        state = str(source["state"])
        if state not in {"Active", "Incomplete", "Records ended"}:
            row["recovery_forecast_status"] = "Waiting for placement"
            row["day35_weight_status"] = "Waiting for placement"
            rows.append(row)
            continue

        feature = extract_feature_row(
            dataset,
            str(source["cycle_id"]),
            str(source["building_id"]),
            pd.Timestamp(source["as_of_date"]),
        )
        if feature is None or pd.isna(feature.get("percentage_alive")):
            row["recovery_forecast_status"] = "Not enough current flock data"
        else:
            cycle_day = int(feature["cycle_day"])
            raw_recovery = _predict(
                feature, "recovery", recovery_manifest, recovery_model
            )
            current_recovery = float(feature["percentage_alive"])
            # Under the agreed capstone accounting rule, birds already lost
            # cannot re-enter the flock. A final recovery forecast therefore
            # cannot exceed the currently recorded survival rate.
            recovery = min(raw_recovery, current_recovery)
            constraint_applied = raw_recovery > current_recovery
            recovery_width = float(
                recovery_manifest["selected_metrics"]["uncertainty_half_width_80"]
            )
            recovery_status = {
                "Incomplete": "Forecast available — latest recorded data used",
                "Records ended": (
                    "Last forecast — based on latest recorded data; harvest not confirmed"
                ),
            }.get(state, "Forecast available")
            row.update(
                {
                    "predicted_final_recovery": recovery,
                    "unconstrained_recovery_prediction": raw_recovery,
                    "recovery_constraint_applied": constraint_applied,
                    "recovery_target_gap_pp": (recovery - RECOVERY_TARGET) * 100,
                    "recovery_interval_low": max(0.0, recovery - recovery_width),
                    "recovery_interval_high": min(
                        current_recovery, recovery + recovery_width
                    ),
                    "recovery_forecast_status": recovery_status,
                    "recovery_confidence": _confidence_text(
                        "recovery", cycle_day, recovery_manifest
                    ),
                }
            )

        day35 = project_day35_weight(
            dataset,
            str(source["cycle_id"]),
            str(source["building_id"]),
            pd.Timestamp(source["as_of_date"]),
            day35_manifest,
        )
        day35_prediction = day35["prediction"]
        row.update(
            {
                "projected_day35_weight_kg": day35_prediction,
                "day35_weight_target_gap_kg": (
                    float(day35_prediction) - WEIGHT_TARGET_KG
                    if pd.notna(day35_prediction)
                    else np.nan
                ),
                "day35_weight_interval_low_kg": day35["interval_low"],
                "day35_weight_interval_high_kg": day35["interval_high"],
                "day35_weight_status": day35["status"],
                "day35_weight_confidence": day35["confidence"],
                "day35_weight_scope": day35["scope"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_forecast_history(
    dataset: CanaryDataset,
    risk_history: pd.DataFrame,
    cycle_id: str,
    building_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Recreate the forecast available on each historical as-of date."""

    if risk_history.empty:
        return risk_history.copy()
    daily_frames = []
    for _, row in risk_history.iterrows():
        frame = pd.DataFrame(
            [
                {
                    **row.to_dict(),
                    "cycle_id": cycle_id,
                    "building_id": building_id,
                    "as_of_date": pd.Timestamp(row["record_date"]),
                    "final_recovery_rate": np.nan,
                }
            ]
        )
        daily_frames.append(attach_forecasts(dataset, frame, model_dir))
    return pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()


def forecast_trace(row: pd.Series, model_dir: str | Path = DEFAULT_MODEL_DIR) -> pd.DataFrame:
    """Return owner-readable provenance for both displayed outcomes."""

    recovery, _ = load_model_bundle("recovery", model_dir)
    day35 = load_day35_manifest(Path(model_dir) / "day35_weight_manifest.json")
    observed_day35 = str(row.get("day35_weight_scope")) == "Recorded Day 35 result"
    return pd.DataFrame(
        [
            {
                "Outcome": "Harvest survival",
                "Status": row.get("recovery_forecast_status"),
                "Prediction": (
                    "Not available"
                    if pd.isna(row.get("predicted_final_recovery"))
                    else f"{float(row.get('predicted_final_recovery')):.1%}"
                ),
                "Goal": "95.0%",
                "Likely range": (
                    "Not available"
                    if pd.isna(row.get("recovery_interval_low"))
                    else f"{float(row.get('recovery_interval_low')):.1%}–{float(row.get('recovery_interval_high')):.1%}"
                ),
                "Method": recovery["selected_model"],
                "Version": recovery["model_version"],
                "Training evidence": f"{len(recovery['training_cycles'])} cycles · {recovery['training_building_cycles']} building outcomes",
                "Validation MAE": f"{recovery['selected_metrics']['mae'] * 100:.1f} percentage points",
                "How to interpret it": "Changes as current survival, mortality, feed, and available environment evidence change.",
                "Important limitation": "The training target is last-recorded recovery, used as a capstone proxy because confirmed harvest status is unavailable.",
            },
            {
                "Outcome": "Day 35 average weight",
                "Status": row.get("day35_weight_status"),
                "Prediction": (
                    "Not available"
                    if pd.isna(row.get("projected_day35_weight_kg"))
                    else f"{float(row.get('projected_day35_weight_kg')):.2f} kg"
                ),
                "Goal": "1.80 kg on Day 35",
                "Likely range": (
                    "Not available"
                    if pd.isna(row.get("day35_weight_interval_low_kg"))
                    else f"{float(row.get('day35_weight_interval_low_kg')):.2f}–{float(row.get('day35_weight_interval_high_kg')):.2f} kg"
                ),
                "Method": "Observed Day 35 measurement" if observed_day35 else day35["selected_model"],
                "Version": "Not applicable" if observed_day35 else day35["model_version"],
                "Training evidence": (
                    "Not applicable"
                    if observed_day35
                    else f"{len(day35['training_cycles'])} cycles · {day35['training_building_cycles']} Day 35 outcomes"
                ),
                "Validation MAE": (
                    "Not applicable"
                    if observed_day35
                    else f"{day35['selected_metrics']['mae_kg']:.2f} kg"
                ),
                "How to interpret it": (
                    "This is the recorded average weight on Day 35."
                    if observed_day35
                    else "Uses compact Ridge regression with the latest available checkpoint history, age-target progress, and observed growth signals."
                ),
                "Important limitation": (
                    "No projection limitation; this is an observation."
                    if observed_day35
                    else f"Validated on {day35['training_building_cycles']} historical building outcomes across {len(day35['training_cycles'])} cycles; the small number of 1.8 kg target hits limits classification confidence."
                ),
            },
        ]
    )


def forecast_input_trace(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Show the evidence available at prediction time without claiming causality."""

    feature = extract_feature_row(dataset, cycle_id, building_id, as_of)
    if feature is None:
        return pd.DataFrame()
    recovery, _ = load_model_bundle("recovery", model_dir)
    day35 = load_day35_manifest(Path(model_dir) / "day35_weight_manifest.json")
    formats = {
        "cycle_day": ("Flock age", lambda value: f"Day {int(value)}"),
        "percentage_alive": ("Currently alive", lambda value: f"{value:.1%}"),
        "mortality_recent_3d_per_1000": ("Recent mortality", lambda value: f"{value:.2f} per 1,000"),
        "latest_weight_kg": ("Latest measured weight", lambda value: f"{value:.3f} kg"),
        "weight_target_kg": ("Target on weighing day", lambda value: f"{value:.3f} kg"),
        "weight_gap_pct": ("Weight gap", lambda value: f"{value:.1f}% below target"),
        "weight_measurement_day": ("Last weighing", lambda value: f"Day {int(value)}"),
        "weight_staleness_days": ("Days since weighing", lambda value: f"{int(value)} days"),
        "feed_cumulative_per_1000_birds": ("Recorded feed to date", lambda value: f"{value:.1f} bags per 1,000"),
        "temperature_recent_avg_c": ("Recent temperature", lambda value: f"{value:.1f} °C"),
        "humidity_recent_avg_pct": ("Recent humidity", lambda value: f"{value:.1f}%"),
    }
    rows = []
    for key, (label, formatter) in formats.items():
        value = feature.get(key)
        displayed = "Not recorded" if value is None or pd.isna(value) else formatter(float(value))
        rows.append(
            {
                "Evidence available as of this date": label,
                "Current value": displayed,
                "Harvest survival model": "Used" if key in recovery["feature_columns"] else "Not used",
                "Day 35 weight projection": (
                    "Used"
                    if key
                    in {
                        "latest_weight_kg",
                        "weight_target_kg",
                        "weight_measurement_day",
                        "weight_staleness_days",
                    }
                    else "Not used"
                ),
            }
        )
    return pd.DataFrame(rows)


FEATURE_LABELS = {
    "cycle_day": "Flock age",
    "beginning_inventory": "Beginning population",
    "percentage_alive": "Current survival",
    "mortality_daily_per_1000": "Latest daily mortality",
    "mortality_recent_3d_per_1000": "Recent 3-day mortality",
    "mortality_trend_delta_per_1000": "Mortality trend",
    "feed_daily_per_1000_birds": "Latest feed per 1,000 birds",
    "feed_cumulative_per_1000_birds": "Cumulative feed per 1,000 birds",
    "temperature_recent_avg_c": "Recent temperature",
    "humidity_recent_avg_pct": "Recent humidity",
    "is_lags_building": "Lagundi building indicator",
}


def recovery_feature_contributions(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Explain how each Ridge input moved the raw recovery estimate.

    Contributions are model associations around the fitted intercept. They are
    not causal effects and are calculated before the current-survival cap.
    """

    feature = extract_feature_row(dataset, cycle_id, building_id, as_of)
    manifest, model = load_model_bundle("recovery", model_dir)
    if feature is None or model is None or manifest["selected_model"] not in {
        "ridge",
        "ridge_no_weight",
        "ridge_core",
    }:
        return pd.DataFrame()

    columns = manifest["feature_columns"]
    frame = pd.DataFrame([{column: feature.get(column, np.nan) for column in columns}])
    regressor = model.regressor_
    imputer = regressor.named_steps["imputer"]
    scaler = regressor.named_steps["scale"]
    ridge = regressor.named_steps["model"]
    imputed = imputer.transform(frame)
    standardized = scaler.transform(imputed)
    names = list(imputer.get_feature_names_out(columns))
    target_scale = float(np.asarray(model.transformer_.scale_).reshape(-1)[0])
    contributions = standardized[0] * np.asarray(ridge.coef_).reshape(-1) * target_scale
    rows = []
    for name, contribution in zip(names, contributions):
        raw_name = str(name)
        is_missing_indicator = raw_name.startswith("missingindicator_")
        source_name = raw_name.removeprefix("missingindicator_")
        if is_missing_indicator:
            label = f"Missing-data flag: {FEATURE_LABELS.get(source_name, source_name.replace('_', ' '))}"
            current_value = "Missing" if pd.isna(feature.get(source_name)) else "Recorded"
        else:
            label = FEATURE_LABELS.get(source_name, source_name.replace("_", " ").title())
            value = feature.get(source_name)
            current_value = "Not recorded" if value is None or pd.isna(value) else f"{float(value):.3f}"
        rows.append(
            {
                "Model input": label,
                "Current value": current_value,
                "Effect on raw estimate": float(contribution),
                "Direction": (
                    "Pushes estimate up"
                    if contribution > 0
                    else "Pushes estimate down"
                    if contribution < 0
                    else "No effect at this value"
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(_magnitude=lambda frame: frame["Effect on raw estimate"].abs())
        .sort_values("_magnitude", ascending=False)
        .drop(columns="_magnitude")
        .reset_index(drop=True)
    )
