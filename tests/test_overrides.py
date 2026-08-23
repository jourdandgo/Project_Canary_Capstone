from pathlib import Path

import pytest

from canary import (
    latest_management_overrides,
    load_management_overrides,
    record_management_override,
)


def _record(path: Path, *, new_value: str = "High") -> dict[str, str]:
    return record_management_override(
        path,
        snapshot_id="snapshot-1",
        cycle_id="2026-3",
        building_id="Tags 1",
        as_of_date="2026-08-07",
        field_label="Operational priority",
        original_value="Critical",
        new_value=new_value,
        rationale="Sensor was checked and the original reading was invalid.",
        responsible_person="Doc Raymond",
        follow_up_due="2026-08-08",
    )


def test_override_ledger_preserves_before_after_reason_and_person(tmp_path: Path) -> None:
    ledger = tmp_path / "management_overrides.csv"
    first = _record(ledger)
    stored = load_management_overrides(ledger)

    assert len(stored) == 1
    assert stored.iloc[0]["override_id"] == first["override_id"]
    assert stored.iloc[0]["original_value"] == "Critical"
    assert stored.iloc[0]["new_value"] == "High"
    assert stored.iloc[0]["rationale"]
    assert stored.iloc[0]["responsible_person"] == "Doc Raymond"


def test_latest_override_is_field_specific_and_original_value_cannot_be_overwritten(tmp_path: Path) -> None:
    ledger = tmp_path / "management_overrides.csv"
    _record(ledger, new_value="High")
    _record(ledger, new_value="Medium")
    latest = latest_management_overrides(
        ledger, cycle_id="2026-3", building_id="Tags 1", through_date="2026-08-07"
    )

    assert len(load_management_overrides(ledger)) == 2
    assert len(latest) == 1
    assert latest.iloc[0]["new_value"] == "Medium"
    assert latest.iloc[0]["original_value"] == "Critical"


def test_override_requires_a_real_change_and_reason(tmp_path: Path) -> None:
    ledger = tmp_path / "management_overrides.csv"
    with pytest.raises(ValueError, match="must differ"):
        record_management_override(
            ledger,
            snapshot_id="snapshot-1",
            cycle_id="2026-3",
            building_id="Tags 1",
            as_of_date="2026-08-07",
            field_label="Operational priority",
            original_value="High",
            new_value="High",
            rationale="No change.",
            responsible_person="Doc Raymond",
        )


def test_resolved_event_removes_the_active_overlay_without_deleting_history(tmp_path: Path) -> None:
    ledger = tmp_path / "management_overrides.csv"
    active = _record(ledger, new_value="High")
    record_management_override(
        ledger,
        snapshot_id=active["snapshot_id"],
        cycle_id=active["cycle_id"],
        building_id=active["building_id"],
        as_of_date="2026-08-08",
        field_label=active["field_label"],
        original_value="High",
        new_value="Critical",
        rationale="The sensor was replaced and the system value is again accepted.",
        responsible_person="Doc Raymond",
        status="Resolved",
    )

    assert len(load_management_overrides(ledger)) == 2
    latest = latest_management_overrides(
        ledger, cycle_id="2026-3", building_id="Tags 1", through_date="2026-08-08"
    )
    assert latest.empty
