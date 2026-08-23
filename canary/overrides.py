"""Append-only management overrides that remain separate from system outputs."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import uuid

import pandas as pd

from .calculation_history import _audit_root


DEFAULT_OVERRIDE_HISTORY_PATH = _audit_root() / "management_overrides.csv"
OVERRIDE_FIELDS = {
    "Operational priority": "risk_rating",
    "Primary problem pattern": "risk_pattern",
    "Recommendation": "recommended_action",
}
OVERRIDE_COLUMNS = [
    "override_id", "snapshot_id", "cycle_id", "building_id", "as_of_date",
    "field_label", "field_key", "original_value", "new_value", "rationale",
    "responsible_person", "follow_up_due", "status", "recorded_at",
]
OVERRIDE_STATUSES = {"Active", "Resolved"}


def load_management_overrides(
    path: str | Path = DEFAULT_OVERRIDE_HISTORY_PATH,
) -> pd.DataFrame:
    destination = Path(path)
    if not destination.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    frame = pd.read_csv(destination, dtype=str).fillna("")
    for column in OVERRIDE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[OVERRIDE_COLUMNS]


def record_management_override(
    path: str | Path = DEFAULT_OVERRIDE_HISTORY_PATH,
    *,
    snapshot_id: str,
    cycle_id: str,
    building_id: str,
    as_of_date: str,
    field_label: str,
    original_value: str,
    new_value: str,
    rationale: str,
    responsible_person: str,
    follow_up_due: str = "",
    status: str = "Active",
) -> dict[str, str]:
    if field_label not in OVERRIDE_FIELDS:
        raise ValueError(f"field_label must be one of {sorted(OVERRIDE_FIELDS)}")
    if status not in OVERRIDE_STATUSES:
        raise ValueError(f"status must be one of {sorted(OVERRIDE_STATUSES)}")
    required = {
        "snapshot_id": snapshot_id,
        "cycle_id": cycle_id,
        "building_id": building_id,
        "original_value": original_value,
        "new_value": new_value,
        "rationale": rationale,
        "responsible_person": responsible_person,
    }
    missing = [key for key, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError("Required override fields are missing: " + ", ".join(missing))
    if str(original_value).strip() == str(new_value).strip():
        raise ValueError("The management value must differ from the system value.")
    due = ""
    if str(follow_up_due).strip():
        due = pd.Timestamp(follow_up_due).date().isoformat()
    row = {
        "override_id": str(uuid.uuid4()),
        "snapshot_id": str(snapshot_id),
        "cycle_id": str(cycle_id),
        "building_id": str(building_id),
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "field_label": field_label,
        "field_key": OVERRIDE_FIELDS[field_label],
        "original_value": str(original_value),
        "new_value": str(new_value),
        "rationale": str(rationale),
        "responsible_person": str(responsible_person),
        "follow_up_due": due,
        "status": status,
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_management_overrides(destination)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)[OVERRIDE_COLUMNS]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    updated.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return row


def latest_management_overrides(
    path: str | Path = DEFAULT_OVERRIDE_HISTORY_PATH,
    *,
    cycle_id: str,
    building_id: str | None = None,
    through_date: str,
) -> pd.DataFrame:
    ledger = load_management_overrides(path)
    if ledger.empty:
        return ledger
    eligible = ledger.loc[
        ledger["cycle_id"].eq(str(cycle_id))
        & (pd.to_datetime(ledger["as_of_date"], errors="coerce") <= pd.Timestamp(through_date))
    ].copy()
    if building_id is not None:
        eligible = eligible.loc[eligible["building_id"].eq(str(building_id))]
    if eligible.empty:
        return eligible
    eligible["_recorded"] = pd.to_datetime(eligible["recorded_at"], errors="coerce")
    latest = (
        eligible.sort_values(["building_id", "field_key", "_recorded", "override_id"])
        .groupby(["building_id", "field_key"], as_index=False)
        .tail(1)
        .drop(columns="_recorded")
        .reset_index(drop=True)
    )
    return latest.loc[latest["status"].eq("Active")].reset_index(drop=True)
