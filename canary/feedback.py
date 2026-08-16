"""Validated, append-only feedback records for later alert evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import uuid

import pandas as pd


FEEDBACK_COLUMNS = [
    "feedback_id", "cycle_id", "building_id", "as_of_date", "signal_id",
    "assessment", "action_taken", "responsible_person", "recorded_at", "outcome_notes",
]
VALID_ASSESSMENTS = {"Confirmed", "Dismissed", "Action taken", "Pending review"}


def record_alert_feedback(
    path: str | Path, *, cycle_id: str, building_id: str, as_of_date: str,
    signal_id: str, assessment: str, action_taken: str = "",
    responsible_person: str = "", outcome_notes: str = "",
) -> dict[str, str]:
    """Atomically append a non-causal feedback observation to a CSV ledger."""
    if assessment not in VALID_ASSESSMENTS:
        raise ValueError(f"assessment must be one of {sorted(VALID_ASSESSMENTS)}")
    if not str(cycle_id).strip() or not str(building_id).strip() or not str(signal_id).strip():
        raise ValueError("cycle_id, building_id, and signal_id are required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "feedback_id": str(uuid.uuid4()), "cycle_id": str(cycle_id), "building_id": str(building_id),
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(), "signal_id": str(signal_id),
        "assessment": assessment, "action_taken": str(action_taken),
        "responsible_person": str(responsible_person),
        "recorded_at": datetime.now().astimezone().isoformat(), "outcome_notes": str(outcome_notes),
    }
    existing = pd.read_csv(destination, dtype=str) if destination.exists() else pd.DataFrame(columns=FEEDBACK_COLUMNS)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)[FEEDBACK_COLUMNS]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    updated.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return row


def load_alert_feedback(path: str | Path) -> pd.DataFrame:
    destination = Path(path)
    return pd.read_csv(destination, dtype=str) if destination.exists() else pd.DataFrame(columns=FEEDBACK_COLUMNS)

