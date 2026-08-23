from pathlib import Path

from canary.management_decisions import (
    latest_management_decisions,
    load_management_decisions,
    record_management_decision,
)


def _record(path: Path, *, building: str, action: str = "Accept recommendation", as_of: str = "2026-07-10"):
    return record_management_decision(
        path,
        cycle_id="2026-3",
        building_id=building,
        as_of_date=as_of,
        decision_type=action,
        system_recommendation="Inspect the flock.",
        system_rule_id="DOC-002",
        system_urgency="Within 24 hours",
        risk_score=4,
        risk_rating="Medium",
        risk_rule_version="risk-rules-test",
        priority_rule_id="PRIORITY-BASE-BAND",
        recovery_outlook=0.93,
        day35_weight_outlook_kg=1.55,
        final_action="Inspect the flock.",
        rationale="Different shift timing." if action != "Accept recommendation" else "",
        responsible_person="Doc Raymond",
        follow_up_due="2026-07-11",
    )


def test_management_decisions_are_append_only_and_keep_system_context(tmp_path: Path) -> None:
    ledger = tmp_path / "management_decisions.csv"
    first = _record(ledger, building="Tags 1")
    _record(ledger, building="Tags 2", action="Escalate")
    stored = load_management_decisions(ledger)
    assert len(stored) == 2
    assert stored.iloc[0]["decision_id"] == first["decision_id"]
    assert stored.iloc[0]["system_rule_id"] == "DOC-002"
    assert stored.iloc[0]["risk_rule_version"] == "risk-rules-test"


def test_non_acceptance_requires_reason_and_latest_is_as_of_aware(tmp_path: Path) -> None:
    ledger = tmp_path / "management_decisions.csv"
    _record(ledger, building="Tags 1", as_of="2026-07-07")
    _record(ledger, building="Tags 1", action="Modify inspection plan", as_of="2026-07-10")
    early = latest_management_decisions(ledger, cycle_id="2026-3", through_date="2026-07-08")
    later = latest_management_decisions(ledger, cycle_id="2026-3", through_date="2026-07-10")
    assert early.iloc[0]["decision_type"] == "Accept recommendation"
    assert later.iloc[0]["decision_type"] == "Modify inspection plan"
