"""Append-only management decisions for Canary's human-in-the-loop workflow.

The ledger deliberately records a manager's response *beside* Canary's
recommendation.  It never changes the input data, observed-risk score, risk
label, forecast, or the recommendation rule that was originally shown.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import uuid

import pandas as pd


DEFAULT_MANAGEMENT_DECISIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "management_decisions"
    / "management_decisions.csv"
)

DECISION_TYPES = {
    "Accept recommendation",
    "Modify inspection plan",
    "Defer / monitor",
    "Escalate",
}
FOLLOW_UP_STATUSES = {"Open", "Completed"}
LEDGER_COLUMNS = [
    "decision_id", "cycle_id", "building_id", "as_of_date", "decision_type",
    "system_recommendation", "system_rule_id", "system_urgency",
    "risk_score", "risk_rating", "risk_rule_version", "priority_rule_id",
    "recovery_outlook", "day35_weight_outlook_kg", "final_action", "rationale",
    "responsible_person", "follow_up_due", "follow_up_status", "recorded_at",
]


def record_management_decision(
    path: str | Path = DEFAULT_MANAGEMENT_DECISIONS_PATH,
    *,
    cycle_id: str,
    building_id: str,
    as_of_date: str,
    decision_type: str,
    system_recommendation: str,
    system_rule_id: str,
    system_urgency: str,
    risk_score: object,
    risk_rating: str,
    risk_rule_version: str,
    priority_rule_id: str,
    recovery_outlook: object,
    day35_weight_outlook_kg: object,
    final_action: str,
    rationale: str = "",
    responsible_person: str = "",
    follow_up_due: str = "",
    follow_up_status: str = "Open",
) -> dict[str, str]:
    """Atomically append one manager decision, retaining its system context."""

    if decision_type not in DECISION_TYPES:
        raise ValueError(f"decision_type must be one of {sorted(DECISION_TYPES)}")
    if follow_up_status not in FOLLOW_UP_STATUSES:
        raise ValueError(f"follow_up_status must be one of {sorted(FOLLOW_UP_STATUSES)}")
    required = {
        "cycle_id": cycle_id,
        "building_id": building_id,
        "system_recommendation": system_recommendation,
        "system_rule_id": system_rule_id,
        "final_action": final_action,
    }
    missing = [label for label, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError("Required decision fields are missing: " + ", ".join(missing))
    if decision_type != "Accept recommendation" and not str(rationale).strip():
        raise ValueError("A rationale is required when management changes, defers, or escalates a recommendation.")

    due = ""
    if str(follow_up_due).strip():
        due = pd.Timestamp(follow_up_due).date().isoformat()
    row = {
        "decision_id": str(uuid.uuid4()),
        "cycle_id": str(cycle_id),
        "building_id": str(building_id),
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "decision_type": decision_type,
        "system_recommendation": str(system_recommendation),
        "system_rule_id": str(system_rule_id),
        "system_urgency": str(system_urgency),
        "risk_score": "" if pd.isna(risk_score) else str(risk_score),
        "risk_rating": str(risk_rating),
        "risk_rule_version": str(risk_rule_version),
        "priority_rule_id": str(priority_rule_id),
        "recovery_outlook": "" if pd.isna(recovery_outlook) else str(recovery_outlook),
        "day35_weight_outlook_kg": "" if pd.isna(day35_weight_outlook_kg) else str(day35_weight_outlook_kg),
        "final_action": str(final_action),
        "rationale": str(rationale),
        "responsible_person": str(responsible_person),
        "follow_up_due": due,
        "follow_up_status": follow_up_status,
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_management_decisions(destination)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)[LEDGER_COLUMNS]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    updated.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return row


def load_management_decisions(path: str | Path = DEFAULT_MANAGEMENT_DECISIONS_PATH) -> pd.DataFrame:
    destination = Path(path)
    if not destination.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(destination, dtype=str).fillna("")
    for column in LEDGER_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[LEDGER_COLUMNS]


def latest_management_decisions(
    path: str | Path = DEFAULT_MANAGEMENT_DECISIONS_PATH,
    *,
    cycle_id: str,
    through_date: str,
) -> pd.DataFrame:
    """Return the latest decision available for each building as of a review date."""

    ledger = load_management_decisions(path)
    if ledger.empty:
        return ledger
    eligible = ledger.loc[
        ledger["cycle_id"].eq(str(cycle_id))
        & (pd.to_datetime(ledger["as_of_date"], errors="coerce") <= pd.Timestamp(through_date))
    ].copy()
    if eligible.empty:
        return eligible
    eligible["_recorded"] = pd.to_datetime(eligible["recorded_at"], errors="coerce")
    return (
        eligible.sort_values(["building_id", "_recorded", "decision_id"])
        .groupby("building_id", as_index=False)
        .tail(1)
        .drop(columns="_recorded")
        .reset_index(drop=True)
    )
