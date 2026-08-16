import pandas as pd

from canary import attach_management_priority, rank_management_priorities


def _row(building: str, rating: str, score: int, recovery: str, weight: str, evidence: str = "Complete") -> dict:
    return {
        "building_id": building,
        "building_order": int(building[-1]) - 1,
        "state": "Active",
        "risk_rating": rating,
        "risk_score": score,
        "evidence_status": evidence,
        "recovery_target_status": recovery,
        "day35_weight_target_status": weight,
        "recovery_target_gap_pp": -2.0 if recovery == "Likely below" else 1.0,
        "day35_weight_target_gap_kg": -0.2 if weight == "Likely below" else 0.1,
        "data_staleness_days": 0,
        "weight_staleness_days": 0,
        "environment_staleness_days": 0,
    }


def test_priority_keeps_observed_risk_and_forecast_outlook_separate():
    source = pd.DataFrame(
        [
            _row("Tags 1", "Critical", 7, "Likely meets", "Likely meets"),
            _row("Tags 2", "Low", 1, "Likely below", "Uncertain"),
            _row("Tags 3", "Low", 0, "Likely meets", "Likely meets"),
        ]
    )
    result = attach_management_priority(source)
    assert result["risk_score"].tolist() == source["risk_score"].tolist()
    assert result["management_priority"].tolist() == ["Act now", "Review today", "On track"]


def test_priority_ranking_uses_tier_before_forecast_downside():
    source = pd.DataFrame(
        [
            _row("Tags 1", "Low", 1, "Likely below", "Likely below"),
            _row("Tags 2", "High", 4, "Likely meets", "Likely meets"),
            _row("Tags 3", "Low", 0, "Uncertain", "Likely meets"),
        ]
    )
    ranked = rank_management_priorities(source)
    assert ranked["building_id"].tolist() == ["Tags 2", "Tags 1", "Tags 3"]
