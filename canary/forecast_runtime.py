"""Operational runtime for Canary's two explainable forecast outlooks.

The owner-facing product deliberately exposes one method per business outcome:

* an age-adjusted remaining-loss outlook for the end-of-cycle recovery proxy;
* a historical remaining-gain outlook for Day 35 bodyweight.

Research and legacy artifacts remain available for audit, but they are not
routed into live owner-facing predictions.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import CanaryDataset


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "models" / "three_model" / "checkpoint_champion"
CHECKPOINTS = (7, 14, 21, 28)


@lru_cache(maxsize=2)
def load_checkpoint_artifact(
    path: str = str(CHECKPOINT_DIR / "champion.joblib"),
) -> dict[str, Any]:
    return joblib.load(path)


@lru_cache(maxsize=2)
def load_checkpoint_manifest(
    path: str = str(CHECKPOINT_DIR / "manifest.json"),
) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _checkpoint_widths(
    path: str = str(CHECKPOINT_DIR / "champion_oof_predictions.csv"),
) -> dict[int, float]:
    predictions = pd.read_csv(path)
    predictions["absolute_error_g"] = (
        pd.to_numeric(predictions["predicted_g"], errors="coerce")
        - pd.to_numeric(predictions["actual_g"], errors="coerce")
    ).abs()
    return {
        int(day): float(group["absolute_error_g"].quantile(0.80))
        for day, group in predictions.groupby("review_day")
    }


def latest_checkpoint(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
) -> int | None:
    history = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["record_date"].le(pd.Timestamp(as_of))
        & dataset.daily["weight_measured"].fillna(False)
        & dataset.daily["age_day"].isin(CHECKPOINTS)
    ]
    if history.empty:
        return None
    return int(history["age_day"].max())


def predict_checkpoint_bodyweight(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
    bundle_dir: str | Path = CHECKPOINT_DIR,
) -> dict[str, Any] | None:
    """Project Day 35 weight from the latest actual weekly checkpoint.

    The selected method adds the fold-local historical average remaining gain
    for the measurement checkpoint to the building's latest measured weight.
    No value is refreshed between checkpoints unless a new measurement exists.
    """

    bundle = Path(bundle_dir).resolve()
    checkpoint = latest_checkpoint(dataset, cycle_id, building_id, as_of)
    if checkpoint is None:
        return None

    measured_rows = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["age_day"].eq(checkpoint)
        & dataset.daily["weight_measured"].fillna(False)
        & dataset.daily["record_date"].le(pd.Timestamp(as_of))
    ].sort_values("record_date")
    if measured_rows.empty:
        return None

    measured_weight_kg = float(measured_rows.iloc[-1]["bodyweight_kg"])
    measurement_date = pd.Timestamp(measured_rows.iloc[-1]["record_date"])
    artifact = load_checkpoint_artifact(str(bundle / "champion.joblib"))
    entry = artifact["checkpoint_models"].get(checkpoint)
    if entry is None:
        return None

    outcome_rows = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["age_day"].eq(35)
        & dataset.daily["weight_measured"].fillna(False),
        "bodyweight_kg",
    ]
    outcome_g = float(outcome_rows.iloc[-1] * 1000) if not outcome_rows.empty else np.nan

    if entry.get("kind") == "baseline":
        remaining_gain_g = float(entry["remaining_gain_g"])
        prediction_g = measured_weight_kg * 1000.0 + remaining_gain_g
    else:
        # Kept for artifact compatibility. The current operational champion is
        # the transparent baseline; any future learned promotion must keep the
        # same trace contract.
        from .bodyweight_modeling_review import _predict_entry, _snapshot

        row = _snapshot(dataset, str(cycle_id), str(building_id), checkpoint, outcome_g)
        if row is None:
            return None
        prediction_g = float(_predict_entry(entry, pd.DataFrame([row]))[0])
        remaining_gain_g = prediction_g - measured_weight_kg * 1000.0

    width_g = _checkpoint_widths(
        str(bundle / "champion_oof_predictions.csv")
    ).get(checkpoint, 200.0)
    manifest = load_checkpoint_manifest(str(bundle / "manifest.json"))
    metrics = next(
        item
        for item in manifest["checkpoint_metrics"]
        if int(item["review_day"]) == checkpoint
    )
    return {
        "prediction": prediction_g / 1000.0,
        "interval_low": max(0.1, (prediction_g - width_g) / 1000.0),
        "interval_high": min(3.5, (prediction_g + width_g) / 1000.0),
        "prediction_day": checkpoint,
        "latest_measurement_day": checkpoint,
        "latest_measurement_date": measurement_date,
        "measured_weight_kg": measured_weight_kg,
        "remaining_gain_kg": remaining_gain_g / 1000.0,
        "measurement_age_days": max(
            0, (pd.Timestamp(as_of).normalize() - measurement_date.normalize()).days
        ),
        "forecast_as_of": pd.Timestamp(as_of),
        "model_id": "day35_checkpoint_remaining_gain",
        "algorithm": "Historical remaining gain",
        "status": "shadow-pilot baseline",
        "validation_mae": float(metrics["mae_g"]) / 1000.0,
        "cycle_macro_mae": float(manifest["champion_metrics"]["cycle_macro_mae_g"])
        / 1000.0,
        "actual_day35_weight_kg": outcome_g / 1000.0
        if pd.notna(outcome_g)
        else np.nan,
        "calculation": (
            f"{measured_weight_kg:.3f} kg measured on Day {checkpoint} + "
            f"{remaining_gain_g / 1000.0:.3f} kg expected remaining gain = "
            f"{prediction_g / 1000.0:.3f} kg"
        ),
        "manifest": manifest,
    }


def checkpoint_forecast_history(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
) -> pd.DataFrame:
    """Return the auditable sequence of available checkpoint forecasts."""

    unit = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
    ].sort_values("record_date")
    if unit.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        as_of_rows = unit.loc[unit["age_day"].eq(checkpoint), "record_date"]
        if as_of_rows.empty:
            continue
        result = predict_checkpoint_bodyweight(
            dataset, cycle_id, building_id, pd.Timestamp(as_of_rows.max())
        )
        if result is None:
            continue
        rows.append(
            {
                "Checkpoint": f"Day {checkpoint}",
                "Measured weight (kg)": result["measured_weight_kg"],
                "Expected remaining gain (kg)": result["remaining_gain_kg"],
                "Projected Day 35 weight (kg)": result["prediction"],
                "80% interval low (kg)": result["interval_low"],
                "80% interval high (kg)": result["interval_high"],
                "Recorded Day 35 result (kg)": result["actual_day35_weight_kg"],
                "Held-out MAE (g)": result["validation_mae"] * 1000,
            }
        )
    return pd.DataFrame(rows)
