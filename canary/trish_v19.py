"""Traceable runtime for Trish's final v19 Model 1 and Model 3 handoff.

The package's final fitted artifacts use all 34 historical building-flocks.
For an honest historical or 2026-3 replay, Canary therefore displays the
saved leave-one-building-flock-out (OOF) prediction rather than rescoring the
same flock with the all-data fitted champion. Arbitrary future-cycle scoring
remains unavailable until the upstream 85-feature transformer is packaged as
an operational input pipeline.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = ROOT / "models" / "trish_v19"
BODYWEIGHT_CHECKPOINTS = (7, 14, 21)


@lru_cache(maxsize=2)
def load_v19_manifest(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> dict[str, Any]:
    return json.loads(
        (Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8")
    )


def validate_v19_bundle(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> list[dict[str, Any]]:
    bundle = Path(bundle_dir)
    manifest = load_v19_manifest(bundle)
    checks: list[dict[str, Any]] = []
    for model_id, metadata in manifest["models"].items():
        artifact = bundle / metadata["artifact_file"]
        dataset = bundle / metadata["feature_dataset"]
        predictions = bundle / metadata["oof_predictions"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None
        checks.append(
            {
                "model_id": model_id,
                "artifact_exists": artifact.exists(),
                "artifact_hash_matches": digest == metadata["artifact_sha256"],
                "feature_dataset_exists": dataset.exists(),
                "oof_predictions_exist": predictions.exists(),
            }
        )
    return checks


@lru_cache(maxsize=4)
def _model_frame(model_id: str, bundle_dir: str = str(DEFAULT_BUNDLE_DIR)) -> pd.DataFrame:
    bundle = Path(bundle_dir)
    metadata = load_v19_manifest(bundle)["models"][model_id]
    features = pd.read_csv(bundle / metadata["feature_dataset"])
    predictions = pd.read_csv(bundle / metadata["oof_predictions"])
    predictions["prediction_day"] = predictions.groupby("flock", sort=False).cumcount() + 1
    features = features.copy()
    features["flock"] = (
        features["harvest_cycle"].astype(str) + "_" + features["bldg"].astype(str)
    )
    merged = features.merge(
        predictions[["flock", "prediction_day", "actual", "predicted"]],
        on=["flock", "prediction_day"],
        how="left",
        validate="one_to_one",
    )
    if merged["predicted"].isna().any():
        raise ValueError(f"Missing OOF prediction rows for {model_id}")
    return merged


def _evidence_day(model_id: str, as_of_day: int) -> int | None:
    day = int(as_of_day)
    if model_id == "model_1":
        return min(max(day, 1), 14)
    if day < 7:
        return None
    if day < 14:
        return 7
    if day < 21:
        return 14
    return 21


def _checkpoint_metrics(frame: pd.DataFrame, evidence_day: int) -> dict[str, float]:
    rows = frame.loc[frame["prediction_day"].eq(int(evidence_day))]
    error = rows["predicted"] - rows["actual"]
    absolute = error.abs()
    return {
        "mae": float(absolute.mean()),
        "error_band_half_width": float(absolute.quantile(0.80)),
        "bias": float(error.mean()),
        "n": int(len(rows)),
    }


def v19_outlook(
    model_id: str,
    cycle_id: str,
    building_id: str,
    as_of_day: int,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
) -> dict[str, Any] | None:
    """Return one OOF replay outlook and its complete evidence lineage."""

    if model_id not in {"model_1", "model_3"}:
        raise KeyError(f"Unsupported v19 model: {model_id}")
    evidence_day = _evidence_day(model_id, as_of_day)
    if evidence_day is None:
        return None
    bundle = Path(bundle_dir)
    manifest = load_v19_manifest(bundle)
    metadata = manifest["models"][model_id]
    frame = _model_frame(model_id, str(bundle))
    match = frame.loc[
        frame["harvest_cycle"].astype(str).eq(str(cycle_id))
        & frame["bldg"].astype(str).eq(str(building_id))
        & frame["prediction_day"].eq(int(evidence_day))
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    metrics = _checkpoint_metrics(frame, evidence_day)
    prediction = float(row["predicted"])
    lower = prediction - metrics["error_band_half_width"]
    upper = prediction + metrics["error_band_half_width"]
    if model_id == "model_1":
        lower, upper = max(0.0, lower), min(1.0, upper)
    else:
        lower, upper = max(100.0, lower), min(3500.0, upper)
    status = (
        f"Recalculated from Day {evidence_day} evidence"
        if int(as_of_day) == evidence_day
        else f"Held from Day {evidence_day} evidence"
    )
    return {
        "model_id": model_id,
        "prediction": prediction,
        "lower_estimate": float(lower),
        "upper_estimate": float(upper),
        "actual": float(row["actual"]),
        "evidence_day": int(evidence_day),
        "as_of_day": int(as_of_day),
        "status": status,
        "checkpoint_mae": metrics["mae"],
        "error_band_half_width": metrics["error_band_half_width"],
        "checkpoint_bias": metrics["bias"],
        "checkpoint_n": int(metrics["n"]),
        "algorithm": metadata["algorithm"],
        "feature_count": int(metadata["feature_count"]),
        "version": manifest["bundle_version"],
        "mlflow_run_id": metadata["mlflow_run_id"],
        "target_definition": metadata["target_definition"],
        "overall_mae": float(
            metadata["validated_mae"]
            if model_id == "model_1"
            else metadata["validated_mae_g"]
        ),
        "overall_r2": float(metadata["validated_r2"]),
        "source_type": "Saved leave-one-building-flock-out prediction",
        "lineage": "Trish v19 final handoff · OOF replay row",
        "boundary": (
            "Planning outlook; not a probability, diagnosis, causal conclusion, or guaranteed result."
        ),
        "feature_row": row,
    }


def v19_input_trace(result: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Return the exact model-ready values present in the replay row."""

    row = result["feature_row"]
    model_id = str(result["model_id"])
    summary_file = (
        DEFAULT_BUNDLE_DIR / "model1_champion_summary.json"
        if model_id == "model_1"
        else DEFAULT_BUNDLE_DIR / "model3_champion_summary.json"
    )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    columns = list(summary["feature_names"])
    records = []
    for column in columns:
        value = row[column]
        records.append(
            {
                "Feature": column,
                "Value supplied": "Missing" if pd.isna(value) else str(value),
                "Evidence cutoff": f"Day {result['evidence_day']}",
            }
        )
    trace = pd.DataFrame(records)
    return trace if limit is None else trace.head(int(limit))


def v19_calculation_trace(result: dict[str, Any]) -> pd.DataFrame:
    """Explain input -> model -> output without pretending a tree is a formula."""

    is_recovery = result["model_id"] == "model_1"
    prediction = (
        f"{result['prediction']:.2%}"
        if is_recovery
        else f"{result['prediction']:.0f} g"
    )
    band = (
        f"{result['lower_estimate']:.2%} to {result['upper_estimate']:.2%}"
        if is_recovery
        else f"{result['lower_estimate']:.0f} to {result['upper_estimate']:.0f} g"
    )
    error = (
        f"{result['checkpoint_mae'] * 100:.2f} percentage points"
        if is_recovery
        else f"{result['checkpoint_mae']:.1f} g"
    )
    return pd.DataFrame(
        [
            {
                "Step": "1 · Select evidence",
                "What Canary used": f"The saved {result['model_id'].replace('_', ' ').title()} row containing information available through Day {result['evidence_day']}.",
                "Result": result["status"],
            },
            {
                "Step": "2 · Align inputs",
                "What Canary used": f"{result['feature_count']} model features in Trish's saved schema and order.",
                "Result": "Exact replay feature row found",
            },
            {
                "Step": "3 · Apply trained model",
                "What Canary used": f"{result['algorithm']} · MLflow run {result['mlflow_run_id']}",
                "Result": prediction,
            },
            {
                "Step": "4 · Add error reference",
                "What Canary used": f"Checkpoint MAE {error}; the band uses the 80th percentile of held-out absolute errors.",
                "Result": band,
            },
            {
                "Step": "5 · Interpret",
                "What Canary used": result["target_definition"],
                "Result": result["boundary"],
            },
        ]
    )


def v19_global_drivers(model_id: str, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> pd.DataFrame:
    """Return validated LOFO associations; positive delta means useful signal."""

    bundle = Path(bundle_dir)
    filename = "model1_lofo_importance.csv" if model_id == "model_1" else "model3_lofo_importance.csv"
    frame = pd.read_csv(bundle / filename).copy()
    frame = frame.rename(
        columns={
            "feature": "Feature",
            "delta_mae": "MAE increase when removed",
            "pct_mae_change": "MAE change when removed (%)",
            "helps_model": "Helped held-out accuracy",
        }
    )
    return frame.sort_values("MAE increase when removed", ascending=False).reset_index(drop=True)
