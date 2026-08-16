"""Deterministic management priority built after risk and forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRIORITY_ORDER = {
    "Act now": 1,
    "Review today": 2,
    "Monitor closely": 3,
    "On track": 4,
    "Not assessable": 5,
    "Not applicable": 6,
}


def _status(value: object) -> str:
    text = str(value) if pd.notna(value) else "Unavailable"
    return text if text in {"Likely below", "Uncertain", "Likely meets"} else "Unavailable"


def _priority_for(row: pd.Series) -> tuple[str, str]:
    if str(row.get("state")) not in {"Active", "Incomplete", "Records ended"}:
        return "Not applicable", "No active or reviewable flock."
    risk_rating = str(row.get("risk_rating", "Not rated"))
    evidence = str(row.get("evidence_status", "Insufficient"))
    recovery = _status(row.get("recovery_target_status"))
    weight = _status(row.get("day35_weight_target_status"))
    statuses = [recovery, weight]
    unavailable = sum(value == "Unavailable" for value in statuses)
    likely_below = sum(value == "Likely below" for value in statuses)

    if risk_rating in {"High", "Critical"}:
        return "Act now", f"Observed conditions are rated {risk_rating}."
    if risk_rating == "Medium":
        return "Review today", "Observed conditions are rated Medium."
    if likely_below:
        outcomes = []
        if recovery == "Likely below":
            outcomes.append("recovery")
        if weight == "Likely below":
            outcomes.append("Day 35 weight")
        return "Review today", "The 80% forecast interval is below target for " + " and ".join(outcomes) + "."
    if evidence in {"Insufficient", "Reduced evidence"} or unavailable == 2:
        return "Review today", "Critical evidence is missing, stale, or insufficient."
    if "Uncertain" in statuses or unavailable:
        return "Monitor closely", "At least one outcome remains uncertain or unavailable."
    if risk_rating == "Low" and all(value == "Likely meets" for value in statuses):
        return "On track", "Observed risk is Low and both forecast intervals are at or above target."
    return "Not assessable", "Available evidence does not support a complete priority assessment."


def attach_management_priority(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic priority fields without changing risk or forecasts."""

    records: list[dict[str, object]] = []
    for _, source in snapshot.iterrows():
        row = source.to_dict()
        tier, reason = _priority_for(source)
        statuses = [_status(source.get("recovery_target_status")), _status(source.get("day35_weight_target_status"))]
        likely_below_count = sum(value == "Likely below" for value in statuses)
        recovery_downside = max(0.0, -float(source.get("recovery_target_gap_pp", 0.0))) if pd.notna(source.get("recovery_target_gap_pp")) else 0.0
        weight_downside = max(0.0, -float(source.get("day35_weight_target_gap_kg", 0.0)) * 1000) if pd.notna(source.get("day35_weight_target_gap_kg")) else 0.0
        staleness = [source.get("data_staleness_days"), source.get("weight_staleness_days"), source.get("environment_staleness_days")]
        row.update(
            {
                "management_priority": tier,
                "management_priority_rank": PRIORITY_ORDER[tier],
                "management_priority_reason": reason,
                "forecast_likely_below_count": likely_below_count,
                "forecast_downside_index": recovery_downside + weight_downside / 100.0,
                "priority_evidence_staleness_days": float(np.nanmax(pd.to_numeric(pd.Series(staleness), errors="coerce"))) if pd.to_numeric(pd.Series(staleness), errors="coerce").notna().any() else np.nan,
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def rank_management_priorities(snapshot: pd.DataFrame) -> pd.DataFrame:
    ranked = attach_management_priority(snapshot) if "management_priority" not in snapshot else snapshot.copy()
    ranked["_risk_sort"] = pd.to_numeric(ranked.get("risk_score"), errors="coerce").fillna(-1)
    ranked["_stale_sort"] = pd.to_numeric(ranked.get("priority_evidence_staleness_days"), errors="coerce").fillna(-1)
    return (
        ranked.sort_values(
            ["management_priority_rank", "_risk_sort", "forecast_likely_below_count", "forecast_downside_index", "_stale_sort", "building_order"],
            ascending=[True, False, False, False, False, True],
        )
        .drop(columns=["_risk_sort", "_stale_sort"])
        .reset_index(drop=True)
    )
