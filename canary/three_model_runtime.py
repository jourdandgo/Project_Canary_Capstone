"""Runtime interface for Canary's two-outcome, three-model shadow trial."""

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
DEFAULT_EXPERIMENT_DIR = ROOT / "models" / "three_model"
LEGACY_DIR = DEFAULT_EXPERIMENT_DIR / "legacy"
CHECKPOINT_DIR = DEFAULT_EXPERIMENT_DIR / "checkpoint_champion"
CHECKPOINTS = (7, 14, 21, 28)


@lru_cache(maxsize=2)
def load_checkpoint_artifact(path: str = str(CHECKPOINT_DIR / "champion.joblib")) -> dict[str, Any]:
    return joblib.load(path)


@lru_cache(maxsize=2)
def load_checkpoint_manifest(path: str = str(CHECKPOINT_DIR / "manifest.json")) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _checkpoint_widths(path: str = str(CHECKPOINT_DIR / "champion_oof_predictions.csv")) -> dict[int, float]:
    predictions = pd.read_csv(path)
    predictions["absolute_error_g"] = (
        pd.to_numeric(predictions["predicted_g"], errors="coerce")
        - pd.to_numeric(predictions["actual_g"], errors="coerce")
    ).abs()
    return {
        int(day): float(group["absolute_error_g"].quantile(0.80))
        for day, group in predictions.groupby("review_day")
    }


def latest_checkpoint(dataset: CanaryDataset, cycle_id: str, building_id: str, as_of: pd.Timestamp) -> int | None:
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
    """Predict Day 35 weight from the latest validated checkpoint at or before as-of."""

    bundle = Path(bundle_dir).resolve()
    checkpoint = latest_checkpoint(dataset, cycle_id, building_id, as_of)
    if checkpoint is None:
        return None
    unit = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["age_day"].eq(35)
        & dataset.daily["weight_measured"].fillna(False),
        "bodyweight_kg",
    ]
    outcome_g = float(unit.iloc[-1] * 1000) if not unit.empty else np.nan
    artifact = load_checkpoint_artifact(str(bundle / "champion.joblib"))
    entry = artifact["checkpoint_models"].get(checkpoint)
    if entry is None:
        return None
    measured = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["age_day"].eq(checkpoint)
        & dataset.daily["weight_measured"].fillna(False),
        "bodyweight_kg",
    ]
    if measured.empty:
        return None
    if entry.get("kind") == "baseline":
        prediction_g = float(measured.iloc[-1] * 1000.0 + entry["remaining_gain_g"])
    else:
        from .bodyweight_modeling_review import _predict_entry, _snapshot

        row = _snapshot(dataset, str(cycle_id), str(building_id), checkpoint, outcome_g)
        if row is None:
            return None
        prediction_g = float(_predict_entry(entry, pd.DataFrame([row]))[0])
    width_g = _checkpoint_widths(str(bundle / "champion_oof_predictions.csv")).get(checkpoint, 200.0)
    manifest = load_checkpoint_manifest(str(bundle / "manifest.json"))
    checkpoint_metrics = next(
        item for item in manifest["checkpoint_metrics"] if int(item["review_day"]) == checkpoint
    )
    return {
        "prediction": prediction_g / 1000.0,
        "interval_low": max(0.1, (prediction_g - width_g) / 1000.0),
        "interval_high": min(3.5, (prediction_g + width_g) / 1000.0),
        "prediction_day": checkpoint,
        "latest_measurement_day": checkpoint,
        "model_id": "checkpoint_champion",
        "algorithm": str(manifest["champion"]).replace("_", " ").title(),
        "status": "shadow" if manifest["champion"] != manifest["best_naive"] else "pilot-ready baseline",
        "validation_mae": float(checkpoint_metrics["mae_g"]) / 1000.0,
        "cycle_macro_mae": float(manifest["champion_metrics"]["cycle_macro_mae_g"]) / 1000.0,
        "actual_day35_weight_kg": outcome_g / 1000.0 if pd.notna(outcome_g) else np.nan,
        "manifest": manifest,
    }


def predict_model_1(cycle_id: str, building_id: str, as_of_day: int) -> dict[str, Any] | None:
    from .three_model_evaluation import predict_reconstructed_legacy

    evidence_day = 7 if int(as_of_day) < 14 else 14
    if int(as_of_day) < 7:
        return None
    result = predict_reconstructed_legacy(
        "model_1", cycle_id, building_id, evidence_day, LEGACY_DIR
    )
    if result is not None:
        result["interval_low"] = max(0.0, float(result["interval_low"]))
        result["interval_high"] = min(1.0, float(result["interval_high"]))
    return result


def predict_model_3(cycle_id: str, building_id: str, as_of_day: int) -> dict[str, Any] | None:
    from .three_model_evaluation import predict_reconstructed_legacy

    if int(as_of_day) < 21:
        return None
    result = predict_reconstructed_legacy(
        "model_3", cycle_id, building_id, 21, LEGACY_DIR
    )
    if result is not None:
        result["prediction"] = float(result["prediction"]) / 1000.0
        result["interval_low"] = max(0.1, float(result["interval_low"]) / 1000.0)
        result["interval_high"] = min(3.5, float(result["interval_high"]) / 1000.0)
        result["validation_mae"] = float(result["validation_mae"]) / 1000.0
        result["cycle_macro_mae"] = float(result["cycle_macro_mae"]) / 1000.0
    return result


def bodyweight_comparison(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    cycle_rows = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["record_date"].le(pd.Timestamp(as_of))
    ]
    if cycle_rows.empty:
        return pd.DataFrame()
    current_day = int(cycle_rows["age_day"].max())
    model3 = predict_model_3(cycle_id, building_id, current_day)
    champion = predict_checkpoint_bodyweight(dataset, cycle_id, building_id, as_of)
    actual = np.nan if champion is None else champion.get("actual_day35_weight_kg", np.nan)
    rows = []
    for checkpoint in CHECKPOINTS:
        checkpoint_as_of = cycle_rows.loc[cycle_rows["age_day"].le(checkpoint), "record_date"]
        checkpoint_result = (
            predict_checkpoint_bodyweight(dataset, cycle_id, building_id, pd.Timestamp(checkpoint_as_of.max()))
            if not checkpoint_as_of.empty and current_day >= checkpoint
            else None
        )
        model3_result = model3 if checkpoint == 21 else None
        rows.append(
            {
                "Checkpoint": f"Day {checkpoint}",
                "Model 3 projection (kg)": np.nan if model3_result is None else model3_result["prediction"],
                "Checkpoint model projection (kg)": np.nan if checkpoint_result is None else checkpoint_result["prediction"],
                "Absolute difference (kg)": (
                    np.nan
                    if model3_result is None or checkpoint_result is None
                    else abs(float(model3_result["prediction"]) - float(checkpoint_result["prediction"]))
                ),
                "Recorded Day 35 result (kg)": actual,
                "Model 3 status": "Not available" if checkpoint != 21 else str(model3_result["status"]) if model3_result else "Awaiting Day 21",
                "Checkpoint model status": "Awaiting checkpoint" if checkpoint_result is None else str(checkpoint_result["status"]),
            }
        )
    return pd.DataFrame(rows)
