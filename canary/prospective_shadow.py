"""Prospective, research-only scoring ledger for frozen Canary challengers.

This workflow never writes to ``models/`` and never changes application
inference.  A cycle counts toward promotion only when it starts after the
frozen 2026-3 audit and its outcome has been explicitly confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .bodyweight_modeling_review import _snapshot as _weight_snapshot
from .data import CanaryDataset, load_workbook
from .day35 import load_day35_manifest, project_day35_weight
from .external_modeling_review import AUDIT_CYCLE, CHECKPOINTS, _snapshot_features
from .forecast import _predict, load_model_bundle
from .model_optimization_round import (
    OptimizationCandidate,
    add_optimization_features,
    predict_candidate,
)


SHADOW_VERSION = "prospective-shadow-1.0.0"
DEFAULT_OPTIMIZATION_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "farmwide_modeling_optimization_round"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "prospective_shadow_validation"
)

PREDICTION_COLUMNS = [
    "shadow_version", "outcome", "cycle_id", "building_id", "review_day",
    "as_of_date", "operational_model", "operational_prediction",
    "shadow_model", "shadow_prediction", "actual", "error_operational",
    "error_shadow", "endpoint_confirmed", "counts_toward_promotion",
    "source_workbook_sha256", "captured_at_utc",
]
REGISTRY_COLUMNS = [
    "cycle_id", "outcome", "endpoint_confirmed", "counts_toward_promotion",
    "reason", "building_outcomes", "checkpoint_predictions", "captured_at_utc",
]


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def initialize_shadow_ledger(output: str | Path = DEFAULT_OUTPUT) -> Path:
    """Create the append-safe ledger and frozen protocol if they do not exist."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / "prospective_predictions.csv"
    registry_path = output_path / "cycle_registry.csv"
    if not prediction_path.exists():
        _empty(PREDICTION_COLUMNS).to_csv(prediction_path, index=False)
    if not registry_path.exists():
        _empty(REGISTRY_COLUMNS).to_csv(registry_path, index=False)
    protocol = {
        "shadow_version": SHADOW_VERSION,
        "frozen_audit_cycle": AUDIT_CYCLE,
        "validated_checkpoints": list(CHECKPOINTS),
        "required_qualifying_cycles_per_outcome": 3,
        "qualification": [
            "Cycle must start after the frozen 2026-3 audit cycle.",
            "Recovery endpoint must be explicitly confirmed before recovery counts.",
            "Observed Day 35 average weight must be present and explicitly confirmed before bodyweight counts.",
            "Predictions use the already-frozen challenger artifact; no refitting or retuning is allowed.",
        ],
        "deployment_effect": "None. Research ledger only; operational models remain unchanged.",
    }
    (output_path / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    _write_status(output_path)
    return output_path


def _write_status(output: Path) -> dict[str, Any]:
    registry = pd.read_csv(output / "cycle_registry.csv")
    progress: dict[str, Any] = {}
    for outcome in ("recovery", "bodyweight"):
        rows = registry.loc[
            registry["outcome"].eq(outcome)
            & registry["counts_toward_promotion"].astype(str).str.lower().eq("true")
        ]
        completed = int(rows["cycle_id"].nunique())
        progress[outcome] = {
            "qualifying_cycles": completed,
            "required_cycles": 3,
            "remaining_cycles": max(0, 3 - completed),
            "status": "Evidence complete for review" if completed >= 3 else "Collecting prospective evidence",
        }
    status = {
        "shadow_version": SHADOW_VERSION,
        "progress": progress,
        "operational_models_changed": False,
        "note": "Promotion is not automatic after three cycles; the frozen performance gates must still be evaluated.",
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def shadow_status(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = initialize_shadow_ledger(output)
    return _write_status(output_path)


def _later_than_audit(dataset: CanaryDataset, cycle_id: str) -> bool:
    starts = dataset.cycles.groupby("cycle_id")["start_date"].min()
    if cycle_id not in starts.index or AUDIT_CYCLE not in starts.index:
        return False
    return bool(pd.Timestamp(starts[cycle_id]) > pd.Timestamp(starts[AUDIT_CYCLE]))


def _load_shadow(outcome: str, optimization_root: Path) -> tuple[dict, OptimizationCandidate]:
    folder = "recovery" if outcome == "recovery" else "bodyweight"
    payload = joblib.load(optimization_root / folder / "lowest_error_shadow.joblib")
    candidate = OptimizationCandidate(**payload["candidate"])
    return payload, candidate


def _recovery_rows(dataset: CanaryDataset, cycle_id: str) -> pd.DataFrame:
    records = []
    cycles = dataset.cycles.loc[dataset.cycles["cycle_id"].astype(str).eq(cycle_id)]
    for cycle in cycles.itertuples(index=False):
        for day in CHECKPOINTS:
            row = _snapshot_features(dataset, cycle_id, str(cycle.building_id), int(day))
            if row is None:
                continue
            row.update(
                actual=float(cycle.final_recovery_rate),
                current_value=float(row["percentage_alive"]),
                remaining_target=max(0.0, float(row["percentage_alive"]) - float(cycle.final_recovery_rate)),
            )
            records.append(row)
    return add_optimization_features(pd.DataFrame(records), "recovery") if records else pd.DataFrame()


def _bodyweight_rows(dataset: CanaryDataset, cycle_id: str) -> pd.DataFrame:
    records = []
    cycles = dataset.cycles.loc[dataset.cycles["cycle_id"].astype(str).eq(cycle_id)]
    for cycle in cycles.itertuples(index=False):
        unit = dataset.daily.loc[
            dataset.daily["cycle_id"].astype(str).eq(cycle_id)
            & dataset.daily["building_id"].astype(str).eq(str(cycle.building_id))
        ]
        observed = unit.loc[unit["age_day"].eq(35) & unit["weight_measured"].fillna(False), "bodyweight_kg"]
        if observed.empty:
            continue
        actual = float(observed.iloc[-1] * 1000)
        for day in CHECKPOINTS:
            row = _weight_snapshot(dataset, cycle_id, str(cycle.building_id), int(day), actual)
            if row is None:
                continue
            row.update(actual=actual, current_value=float(row["current_weight_g"]), remaining_target=actual - float(row["current_weight_g"]))
            records.append(row)
    return add_optimization_features(pd.DataFrame(records), "weight") if records else pd.DataFrame()


def _operational_predictions(dataset: CanaryDataset, frame: pd.DataFrame, outcome: str) -> tuple[str, np.ndarray]:
    if outcome == "recovery":
        manifest, model = load_model_bundle("recovery")
        values = []
        for row in frame.itertuples(index=False):
            feature = _snapshot_features(dataset, str(row.cycle_id), str(row.building_id), int(row.review_day))
            values.append(_predict(feature, "recovery", manifest, model))
        return str(manifest["model_version"]), np.asarray(values, dtype=float)
    manifest = load_day35_manifest()
    values = []
    for row in frame.itertuples(index=False):
        result = project_day35_weight(dataset, str(row.cycle_id), str(row.building_id), pd.Timestamp(row.as_of_date), manifest)
        values.append(float(result["prediction"]) * 1000 if pd.notna(result["prediction"]) else np.nan)
    return str(manifest["model_version"]), np.asarray(values, dtype=float)


def capture_cycle(
    workbook: str | Path,
    cycle_id: str,
    *,
    recovery_endpoint_confirmed: bool = False,
    bodyweight_endpoint_confirmed: bool = False,
    output: str | Path = DEFAULT_OUTPUT,
    optimization_root: str | Path = DEFAULT_OPTIMIZATION_ROOT,
) -> dict[str, Any]:
    """Score one genuinely later cycle and append idempotent research records."""

    workbook_path = Path(workbook)
    dataset = load_workbook(workbook_path)
    output_path = initialize_shadow_ledger(output)
    if not _later_than_audit(dataset, cycle_id):
        raise ValueError(f"{cycle_id} is not a cycle later than the frozen {AUDIT_CYCLE} audit.")
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    source_hash = _hash(workbook_path)
    prediction_blocks, registry_rows = [], []
    for outcome, confirmed in (
        ("recovery", recovery_endpoint_confirmed),
        ("bodyweight", bodyweight_endpoint_confirmed),
    ):
        frame = _recovery_rows(dataset, cycle_id) if outcome == "recovery" else _bodyweight_rows(dataset, cycle_id)
        if frame.empty:
            registry_rows.append({
                "cycle_id": cycle_id, "outcome": outcome, "endpoint_confirmed": confirmed,
                "counts_toward_promotion": False, "reason": "No eligible completed outcome/checkpoint rows",
                "building_outcomes": 0, "checkpoint_predictions": 0, "captured_at_utc": timestamp,
            })
            continue
        payload, candidate = _load_shadow(outcome, Path(optimization_root))
        shadow = predict_candidate(payload["fitted"], frame, candidate)
        operational_name, operational = _operational_predictions(dataset, frame, outcome)
        block = frame[["cycle_id", "building_id", "review_day", "as_of_date", "actual"]].copy()
        block.insert(0, "outcome", outcome)
        block.insert(0, "shadow_version", SHADOW_VERSION)
        block["operational_model"] = operational_name
        block["operational_prediction"] = operational
        block["shadow_model"] = candidate.name
        block["shadow_prediction"] = shadow
        block["error_operational"] = block["operational_prediction"] - block["actual"]
        block["error_shadow"] = block["shadow_prediction"] - block["actual"]
        block["endpoint_confirmed"] = bool(confirmed)
        block["counts_toward_promotion"] = bool(confirmed)
        block["source_workbook_sha256"] = source_hash
        block["captured_at_utc"] = timestamp
        prediction_blocks.append(block[PREDICTION_COLUMNS])
        registry_rows.append({
            "cycle_id": cycle_id, "outcome": outcome, "endpoint_confirmed": bool(confirmed),
            "counts_toward_promotion": bool(confirmed),
            "reason": "Confirmed prospective outcome" if confirmed else "Endpoint confirmation still required",
            "building_outcomes": int(frame["building_id"].nunique()),
            "checkpoint_predictions": int(len(frame)), "captured_at_utc": timestamp,
        })

    predictions_path = output_path / "prospective_predictions.csv"
    existing = pd.read_csv(predictions_path)
    additions = pd.concat(prediction_blocks, ignore_index=True) if prediction_blocks else _empty(PREDICTION_COLUMNS)
    combined = pd.concat([existing, additions], ignore_index=True)
    combined = combined.drop_duplicates(["shadow_version", "outcome", "cycle_id", "building_id", "review_day"], keep="last")
    combined.to_csv(predictions_path, index=False)
    registry_path = output_path / "cycle_registry.csv"
    registry = pd.concat([pd.read_csv(registry_path), pd.DataFrame(registry_rows)], ignore_index=True)
    registry = registry.drop_duplicates(["cycle_id", "outcome"], keep="last")
    registry.to_csv(registry_path, index=False)
    return _write_status(output_path)

