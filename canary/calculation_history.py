"""Append-only calculated-risk snapshots for Project Canary.
The ledger stores what Canary calculated at a particular evidence cutoff. A
management override is stored separately and never rewrites these records.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

import pandas as pd

from .data import CanaryDataset


def _audit_root() -> Path:
    configured = os.getenv("CANARY_AUDIT_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "outputs" / "audit_ledger"


DEFAULT_CALCULATION_HISTORY_PATH = _audit_root() / "calculation_snapshots.csv"

SNAPSHOT_COLUMNS = [
    "snapshot_id", "snapshot_fingerprint", "calculated_at", "snapshot_kind",
    "cycle_id", "building_id", "as_of_date", "cycle_day", "state",
    "source_name", "source_sha256", "risk_rule_version", "risk_approval_status",
    "recommendation_rule_version", "recommendation_approval_status",
    "beginning_inventory", "latest_population", "percentage_alive",
    "latest_weight_kg", "weight_measurement_day", "weight_target_at_measurement_kg",
    "weight_gap_pct", "population_loss_pct", "daily_mortality_pct",
    "temperature_avg_c", "temperature_minimum_c", "temperature_maximum_c",
    "humidity_avg_pct", "humidity_minimum_pct", "humidity_maximum_pct",
    "weight_score", "population_loss_score", "daily_mortality_score",
    "environment_score", "risk_score", "available_score_max", "base_risk_rating",
    "risk_rating", "evidence_status", "risk_patterns", "risk_pattern_details",
    "score_equation", "risk_label_rule", "priority_rule_id",
    "recommendation_rule_ids", "recommended_action", "additional_recommended_actions",
    "recommendation_matches_json", "recommendation_urgency",
    "predicted_final_recovery", "recovery_model_id", "trish_prediction_day",
    "projected_day35_weight_kg", "day35_weight_model_id", "trish_weight_prediction_day",
    "data_freshness", "weight_freshness", "environment_status",
]


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _record_from_row(
    row: pd.Series,
    dataset: CanaryDataset,
    snapshot_kind: str,
) -> dict[str, str]:
    record = {column: "" for column in SNAPSHOT_COLUMNS}
    record.update(
        {
            "snapshot_id": str(uuid.uuid4()),
            "calculated_at": datetime.now().astimezone().isoformat(),
            "snapshot_kind": str(snapshot_kind),
            "source_name": dataset.source_name,
            "source_sha256": dataset.source_sha256 or "",
        }
    )
    for column in SNAPSHOT_COLUMNS:
        if column in {"snapshot_id", "snapshot_fingerprint", "calculated_at", "snapshot_kind", "source_name", "source_sha256"}:
            continue
        if column in row.index:
            record[column] = _text(row[column])
    if record["as_of_date"]:
        record["as_of_date"] = pd.Timestamp(record["as_of_date"]).date().isoformat()
    fingerprint_payload = {
        key: value
        for key, value in record.items()
        if key not in {"snapshot_id", "snapshot_fingerprint", "calculated_at"}
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["snapshot_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return record


def load_calculation_snapshots(
    path: str | Path = DEFAULT_CALCULATION_HISTORY_PATH,
) -> pd.DataFrame:
    destination = Path(path)
    if not destination.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    frame = pd.read_csv(destination, dtype=str).fillna("")
    for column in SNAPSHOT_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[SNAPSHOT_COLUMNS]


def record_calculation_snapshots(
    snapshot: pd.DataFrame,
    dataset: CanaryDataset,
    path: str | Path = DEFAULT_CALCULATION_HISTORY_PATH,
    *,
    snapshot_kind: str = "live calculation",
) -> dict[str, object]:
    """Append new calculations and skip exact duplicates idempotently."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_calculation_snapshots(destination)
    known = set(existing["snapshot_fingerprint"])
    candidates = []
    for _, row in snapshot.iterrows():
        if str(row.get("state", "")) == "Inactive" or pd.isna(row.get("risk_score")):
            continue
        record = _record_from_row(row, dataset, snapshot_kind)
        if record["snapshot_fingerprint"] not in known:
            candidates.append(record)
            known.add(record["snapshot_fingerprint"])
    if not candidates:
        return {"inserted": 0, "skipped": int(len(snapshot)), "snapshot_ids": []}
    updated = pd.concat([existing, pd.DataFrame(candidates)], ignore_index=True)[SNAPSHOT_COLUMNS]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    updated.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return {
        "inserted": len(candidates),
        "skipped": max(0, int(len(snapshot)) - len(candidates)),
        "snapshot_ids": [record["snapshot_id"] for record in candidates],
    }


def latest_calculation_snapshot(
    path: str | Path = DEFAULT_CALCULATION_HISTORY_PATH,
    *,
    cycle_id: str,
    building_id: str,
    as_of_date: str,
) -> pd.Series | None:
    ledger = load_calculation_snapshots(path)
    eligible = ledger.loc[
        ledger["cycle_id"].eq(str(cycle_id))
        & ledger["building_id"].eq(str(building_id))
        & ledger["as_of_date"].eq(pd.Timestamp(as_of_date).date().isoformat())
    ].copy()
    if eligible.empty:
        return None
    eligible["_calculated"] = pd.to_datetime(eligible["calculated_at"], errors="coerce")
    return eligible.sort_values(["_calculated", "snapshot_id"]).iloc[-1]
