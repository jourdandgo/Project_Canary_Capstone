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
from .farmwide_features import build_asof_features, checkpoint_status
from .modeling import FEATURE_COLUMNS, extract_feature_row
from .trish_models import predict_v18_outlooks


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RECOVERY_TARGET = 0.95
WEIGHT_TARGET_KG = 1.8


def _interpolated_remaining_loss(
    cycle_day: int, mapping: dict[str, object]
) -> tuple[float, str]:
    """Return a smooth live loss estimate between validated checkpoints.

    Historical validation is performed at Days 7, 14, 21, and 28. Live
    review dates can fall between those checkpoints, so jumping immediately
    to the next checkpoint would understate the loss still remaining. Exact
    checkpoints retain their validated value; intermediate days use linear
    interpolation. Day 28 is held thereafter because the source has no
    verified harvest-date horizon.
    """

    points = sorted((int(day), float(loss)) for day, loss in mapping.items())
    if not points:
        raise ValueError("The recovery manifest has no checkpoint loss values.")
    age = int(cycle_day)
    if age <= points[0][0]:
        day, loss = points[0]
        return loss, f"Day {day} checkpoint held for Day {age}"
    if age >= points[-1][0]:
        day, loss = points[-1]
        return loss, f"Day {day} checkpoint held after Day {day}"
    for (lower_day, lower_loss), (upper_day, upper_loss) in zip(points, points[1:]):
        if lower_day <= age <= upper_day:
            fraction = (age - lower_day) / (upper_day - lower_day)
            loss = lower_loss + fraction * (upper_loss - lower_loss)
            return loss, f"Interpolated between Day {lower_day} and Day {upper_day}"
    raise ValueError(f"Unable to map recovery loss for Day {age}.")


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
    if outcome == "recovery" and manifest.get("prediction_target") == "additional_population_loss_after_review_date":
        current = float(feature["percentage_alive"])
        if manifest["model_kind"] == "formula":
            age = int(feature["cycle_day"])
            loss, _ = _interpolated_remaining_loss(
                age, manifest["additional_loss_by_age_band"]
            )
        else:
            columns = manifest["feature_columns"]
            values = pd.DataFrame(
                [{column: feature.get(column, np.nan) for column in columns}]
            )
            loss = float(model.predict(values)[0])
        return float(current - np.clip(loss, 0.0, current))
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
                "recovery_expected_additional_loss_pp": np.nan,
                "recovery_live_age_policy": "Not available",
                "recovery_target_gap_pp": np.nan,
                "recovery_interval_low": np.nan,
                "recovery_interval_high": np.nan,
                "recovery_interval_90_low": np.nan,
                "recovery_interval_90_high": np.nan,
                "recovery_target_status": "Unavailable",
                "recovery_checkpoint_status": "Unavailable",
                "recovery_deployment_status": "operational",
                "recovery_forecast_status": "Not available",
                "recovery_confidence": "Not available",
                "recovery_model_version": recovery_manifest["model_version"],
                "recovery_model_name": recovery_manifest["selected_model"],
                "recovery_interval_label": "80% empirical interval",
                "projected_day35_weight_kg": np.nan,
                "day35_weight_target_gap_kg": np.nan,
                "day35_weight_interval_low_kg": np.nan,
                "day35_weight_interval_high_kg": np.nan,
                "day35_weight_interval_90_low_kg": np.nan,
                "day35_weight_interval_90_high_kg": np.nan,
                "day35_weight_target_status": "Unavailable",
                "day35_weight_checkpoint_status": "Unavailable",
                "day35_weight_deployment_status": "operational",
                "day35_weight_status": "Not available",
                "day35_weight_confidence": "Not available",
                "day35_weight_scope": "Not available",
                "day35_weight_model_version": day35_manifest["model_version"],
                "day35_weight_model_name": day35_manifest["selected_model"],
                "day35_weight_interval_label": "80% empirical interval",
                "estimated_day_to_1_8kg": np.nan,
                "estimated_day_to_2_0kg": np.nan,
                "projected_sale_window_recovery": np.nan,
                "trish_bundle_version": "Not available",
                "trish_prediction_day": np.nan,
                "forecast_as_of": pd.Timestamp(source["as_of_date"]),
            }
        )

        state = str(source["state"])
        if state not in {"Active", "Incomplete", "Records ended"}:
            row["recovery_forecast_status"] = "Waiting for placement"
            row["day35_weight_status"] = "Waiting for placement"
            rows.append(row)
            continue

        feature = build_asof_features(
            dataset,
            str(source["cycle_id"]),
            str(source["building_id"]),
            pd.Timestamp(source["as_of_date"]),
            "recovery",
        )
        if feature is None or pd.isna(feature.get("percentage_alive")):
            row["recovery_forecast_status"] = "Not enough current flock data"
        else:
            cycle_day = int(feature["cycle_day"])
            raw_recovery = _predict(
                feature, "recovery", recovery_manifest, recovery_model
            )
            current_recovery = float(feature["percentage_alive"])
            expected_loss_pp = (current_recovery - raw_recovery) * 100
            age_policy = "Fitted model"
            if recovery_manifest["model_kind"] == "formula":
                _, age_policy = _interpolated_remaining_loss(
                    cycle_day, recovery_manifest["additional_loss_by_age_band"]
                )
            # Under the agreed capstone accounting rule, birds already lost
            # cannot re-enter the flock. A final recovery forecast therefore
            # cannot exceed the currently recorded survival rate.
            recovery = min(raw_recovery, current_recovery)
            constraint_applied = raw_recovery > current_recovery
            recovery_width = float(
                recovery_manifest["selected_metrics"]["uncertainty_half_width_80"]
            )
            recovery_width_90 = float(
                recovery_manifest["selected_metrics"].get(
                    "uncertainty_half_width_90", recovery_width * 1.25
                )
            )
            recovery_low = max(0.0, recovery - recovery_width)
            recovery_high = min(current_recovery, recovery + recovery_width)
            recovery_low_90 = max(0.0, recovery - recovery_width_90)
            recovery_high_90 = min(current_recovery, recovery + recovery_width_90)
            recovery_target_status = (
                "Likely below"
                if recovery_high < RECOVERY_TARGET
                else "Likely meets"
                if recovery_low >= RECOVERY_TARGET
                else "Uncertain"
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
                    "recovery_expected_additional_loss_pp": expected_loss_pp,
                    "recovery_live_age_policy": age_policy,
                    "recovery_target_gap_pp": (recovery - RECOVERY_TARGET) * 100,
                    "recovery_interval_low": recovery_low,
                    "recovery_interval_high": recovery_high,
                    "recovery_interval_90_low": recovery_low_90,
                    "recovery_interval_90_high": recovery_high_90,
                    "recovery_target_status": recovery_target_status,
                    "recovery_checkpoint_status": checkpoint_status(cycle_day),
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
        day35_low = day35["interval_low"]
        day35_high = day35["interval_high"]
        if pd.notna(day35_prediction) and pd.notna(day35_low) and pd.notna(day35_high):
            day35_width_80 = max(
                float(day35_prediction) - float(day35_low),
                float(day35_high) - float(day35_prediction),
            )
            day35_width_90 = day35_width_80 * 1.25
            day35_low_90 = max(0.1, float(day35_prediction) - day35_width_90)
            day35_high_90 = min(3.5, float(day35_prediction) + day35_width_90)
            day35_target_status = (
                "Likely below"
                if float(day35_high) < WEIGHT_TARGET_KG
                else "Likely meets"
                if float(day35_low) >= WEIGHT_TARGET_KG
                else "Uncertain"
            )
        elif pd.notna(day35_prediction) and str(day35.get("scope")) == "Recorded Day 35 result":
            day35_low_90 = day35_high_90 = np.nan
            day35_target_status = "Likely meets" if float(day35_prediction) >= WEIGHT_TARGET_KG else "Likely below"
        else:
            day35_low_90 = day35_high_90 = np.nan
            day35_target_status = "Unavailable"
        measurement_day = day35.get("measurement_day")
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
                "day35_weight_interval_90_low_kg": day35_low_90,
                "day35_weight_interval_90_high_kg": day35_high_90,
                "day35_weight_target_status": day35_target_status,
                "day35_weight_checkpoint_status": checkpoint_status(
                    int(source.get("cycle_day")) if pd.notna(source.get("cycle_day")) else measurement_day,
                    pd.notna(day35_prediction),
                ),
                "day35_weight_status": day35["status"],
                "day35_weight_confidence": day35["confidence"],
                "day35_weight_scope": day35["scope"],
            }
        )
        if str(day35.get("scope")) == "Recorded Day 35 result":
            row.update(
                {
                    "day35_weight_deployment_status": "recorded outcome",
                    "day35_weight_model_version": "not applicable",
                    "day35_weight_model_name": "Recorded farm measurement",
                    "day35_weight_interval_label": "Not applicable",
                }
            )

        cycle_day = int(source["cycle_day"]) if pd.notna(source.get("cycle_day")) else None
        trish = (
            predict_v18_outlooks(
                str(source["cycle_id"]),
                str(source["building_id"]),
                cycle_day,
                source_sha256=dataset.source_sha256,
            )
            if cycle_day is not None
            else None
        )
        if trish is not None:
            recovery_raw = float(trish["model_1"]["prediction"])
            current_survival = float(feature["percentage_alive"])
            recovery = min(recovery_raw, current_survival)
            recovery_mae = float(trish["model_1"]["reported_logo_mae"])
            recovery_low = max(0.0, recovery - recovery_mae)
            recovery_high = min(current_survival, recovery + recovery_mae)
            recovery_status = (
                "Likely below"
                if recovery_high < RECOVERY_TARGET
                else "Likely meets"
                if recovery_low >= RECOVERY_TARGET
                else "Uncertain"
            )
            recovery_day = int(trish["model_1"]["prediction_day"])
            row.update(
                {
                    "predicted_final_recovery": recovery,
                    "unconstrained_recovery_prediction": recovery_raw,
                    "recovery_constraint_applied": recovery_raw > current_survival,
                    "recovery_live_age_policy": f"Day {recovery_day} Trish v18 outlook held after the early window",
                    "recovery_target_gap_pp": (recovery - RECOVERY_TARGET) * 100,
                    "recovery_interval_low": recovery_low,
                    "recovery_interval_high": recovery_high,
                    "recovery_interval_90_low": np.nan,
                    "recovery_interval_90_high": np.nan,
                    "recovery_target_status": recovery_status,
                    "recovery_checkpoint_status": f"Day {recovery_day} early-warning estimate",
                    "recovery_deployment_status": "Trish v18 prospective deployment",
                    "recovery_forecast_status": "Forecast available",
                    "recovery_confidence": f"{trish['model_1']['algorithm']} · reported LOGO MAE {recovery_mae * 100:.1f} points · 2026-3 excluded from fitting",
                    "recovery_model_version": trish["bundle_version"],
                    "recovery_model_name": "Trish Model 1 · Extra Trees",
                    "recovery_interval_label": "Typical-error reference",
                    "estimated_day_to_1_8kg": float(trish["model_4"]["prediction"]),
                    "estimated_day_to_2_0kg": max(
                        float(trish["model_5"]["prediction"]),
                        float(trish["model_4"]["prediction"]),
                    ),
                    "projected_sale_window_recovery": min(
                        float(trish["model_6"]["prediction"]), current_survival
                    ),
                    "trish_bundle_version": trish["bundle_version"],
                    "trish_prediction_day": recovery_day,
                }
            )

            # Preserve an actual Day 35 measurement once it exists. Before
            # that point, Model 2 supplies Days 1-14 and Model 3 supplies the
            # later Day 15-21 update.
            if (
                str(day35.get("scope")) != "Recorded Day 35 result"
                and pd.notna(source.get("latest_weight_kg"))
            ):
                weight_model_id = str(trish["weight_model_id"])
                weight_info = trish[weight_model_id]
                weight_prediction = float(weight_info["prediction"]) / 1000.0
                weight_mae = float(weight_info["reported_logo_mae"]) / 1000.0
                weight_low = max(0.1, weight_prediction - weight_mae)
                weight_high = min(3.5, weight_prediction + weight_mae)
                weight_status = (
                    "Likely below"
                    if weight_high < WEIGHT_TARGET_KG
                    else "Likely meets"
                    if weight_low >= WEIGHT_TARGET_KG
                    else "Uncertain"
                )
                weight_day = int(weight_info["prediction_day"])
                row.update(
                    {
                        "projected_day35_weight_kg": weight_prediction,
                        "day35_weight_target_gap_kg": weight_prediction - WEIGHT_TARGET_KG,
                        "day35_weight_interval_low_kg": weight_low,
                        "day35_weight_interval_high_kg": weight_high,
                        "day35_weight_interval_90_low_kg": np.nan,
                        "day35_weight_interval_90_high_kg": np.nan,
                        "day35_weight_target_status": weight_status,
                        "day35_weight_checkpoint_status": f"Day {weight_day} early-warning estimate",
                        "day35_weight_deployment_status": "Trish v18 prospective deployment",
                        "day35_weight_status": "Day 35 outlook available — Trish v18",
                        "day35_weight_confidence": f"{weight_info['algorithm']} · reported LOGO MAE {weight_mae * 1000:.0f} g · 2026-3 excluded from fitting",
                        "day35_weight_scope": f"Trish Model {2 if weight_model_id == 'model_2' else 3} outlook",
                        "day35_weight_model_version": trish["bundle_version"],
                        "day35_weight_model_name": f"Trish Model {2 if weight_model_id == 'model_2' else 3} · {weight_info['algorithm']}",
                        "day35_weight_interval_label": "Typical-error reference",
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
                "How to interpret it": "Subtracts the fold-local historical remaining loss for the flock's age from current recorded survival; daily dates between checkpoints are interpolated.",
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
                    else "Adds the fold-local historical remaining gain for the measurement age to the latest actually observed building weight."
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
    """Explain how each linear-model input moved the raw recovery estimate.

    Contributions are model associations around the fitted intercept. They are
    not causal effects and are calculated before the current-survival cap.
    """

    feature = extract_feature_row(dataset, cycle_id, building_id, as_of)
    manifest, model = load_model_bundle("recovery", model_dir)
    if feature is None:
        return pd.DataFrame()

    if manifest["selected_model"] == "age_band_remaining_loss":
        cycle_day = int(feature.get("cycle_day", 0) or 0)
        age_band = (
            "7" if cycle_day <= 7 else "14" if cycle_day <= 14 else "21"
            if cycle_day <= 21 else "28" if cycle_day <= 28 else "35"
        )
        expected_loss = float(
            manifest.get("additional_loss_by_age_band", {}).get(age_band, 0.0)
        )
        return pd.DataFrame(
            [
                {
                    "Model input": "Current survival",
                    "Current value": f"{float(feature['percentage_alive']):.1%}",
                    "Effect on raw estimate": float(feature["percentage_alive"]),
                    "Direction": "Pushes estimate up as the starting survival level",
                },
                {
                    "Model input": f"Historical remaining loss after Day {age_band}",
                    "Current value": f"{expected_loss * 100:.2f} percentage points",
                    "Effect on raw estimate": -expected_loss,
                    "Direction": "Pushes estimate down when expected remaining loss is larger",
                },
            ]
        )

    if model is None or manifest["selected_model"] not in {
        "remaining_loss_linear",
        "remaining_loss_ridge",
        "remaining_loss_huber",
        "remaining_loss_extra_trees",
    }:
        return pd.DataFrame()

    columns = manifest["feature_columns"]
    frame = pd.DataFrame([{column: feature.get(column, np.nan) for column in columns}])
    imputer = model.named_steps["imputer"]
    estimator = model.named_steps["model"]
    imputed = imputer.transform(frame)
    names = list(imputer.get_feature_names_out(columns))
    if manifest["selected_model"] == "remaining_loss_extra_trees":
        # SHAP explains the fitted additional-loss model.  Negating each SHAP
        # value expresses its corresponding effect on final recovery.
        import shap

        contributions = -np.asarray(
            shap.TreeExplainer(estimator).shap_values(imputed)
        ).reshape(1, -1)[0]
    else:
        scaler = model.named_steps["scale"]
        standardized = scaler.transform(imputed)
        contributions = -standardized[0] * np.asarray(estimator.coef_).reshape(-1)
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
    rows.append(
        {
            "Model input": "Current survival starting point",
            "Current value": f"{float(feature['percentage_alive']):.1%}",
            "Effect on raw estimate": float(feature["percentage_alive"]),
            "Direction": "Pushes estimate up as the maximum possible recovery before expected future loss",
        }
    )
    return (
        pd.DataFrame(rows)
        .assign(_magnitude=lambda frame: frame["Effect on raw estimate"].abs())
        .sort_values("_magnitude", ascending=False)
        .drop(columns="_magnitude")
        .reset_index(drop=True)
    )
