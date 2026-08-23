from pathlib import Path
import os
from canary import (
    apply_recommendations,
    default_as_of_date,
    latest_calculation_snapshot,
    load_calculation_snapshots,
    load_workbook,
    record_calculation_snapshots,
    score_cycle_snapshot,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_calculation_snapshots_are_append_only_and_idempotent(tmp_path: Path) -> None:
    dataset = load_workbook(SOURCE)
    as_of = default_as_of_date(dataset, "2026-3")
    snapshot = apply_recommendations(score_cycle_snapshot(dataset, "2026-3", as_of))
    ledger = tmp_path / "calculation_snapshots.csv"

    first = record_calculation_snapshots(snapshot, dataset, ledger)
    second = record_calculation_snapshots(snapshot, dataset, ledger)
    stored = load_calculation_snapshots(ledger)

    assert first["inserted"] == 3
    assert second["inserted"] == 0
    assert len(stored) == 3
    assert stored["snapshot_fingerprint"].nunique() == 3
    assert stored["risk_rule_version"].str.len().gt(0).all()
    assert stored["recommendation_rule_ids"].str.len().gt(0).all()


def test_latest_snapshot_resolves_exact_building_and_evidence_date(tmp_path: Path) -> None:
    dataset = load_workbook(SOURCE)
    as_of = default_as_of_date(dataset, "2026-3")
    snapshot = apply_recommendations(score_cycle_snapshot(dataset, "2026-3", as_of))
    ledger = tmp_path / "calculation_snapshots.csv"
    record_calculation_snapshots(snapshot, dataset, ledger)

    found = latest_calculation_snapshot(
        ledger,
        cycle_id="2026-3",
        building_id="Tags 1",
        as_of_date=str(as_of),
    )

    assert found is not None
    assert found["building_id"] == "Tags 1"
    assert found["as_of_date"] == str(as_of)
