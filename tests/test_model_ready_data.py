from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_model_ready_payload import build_payload


ROOT = Path(__file__).resolve().parents[1]


def _frame(payload: dict[str, object], name: str) -> pd.DataFrame:
    return pd.DataFrame(payload["tables"][name])


def test_model_ready_counts_and_grouped_validation() -> None:
    payload = build_payload(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    summary = payload["summary"]
    assert summary == {
        **summary,
        "canonical_building_day_rows": 1624,
        "recovery_building_outcomes": 31,
        "day35_weight_building_outcomes": 31,
        "recovery_training_rows": 155,
        "weight_training_rows": 124,
        "latest_cycle_day35_audit_rows": 12,
        "latest_cycle_day35_audit_outcomes": 3,
        "total_recorded_building_outcomes": 34,
        "latest_cycle_recovery_audit_candidates": 3,
        "latest_cycle_recovery_audit_rows": 12,
        "recovery_daily_audit_rows": 1479,
    }
    outcomes = _frame(payload, "Building Outcomes")
    assert len(outcomes) == 34
    latest_outcomes = outcomes.loc[outcomes["cycle_id"].astype(str).eq("2026-3")]
    assert len(latest_outcomes) == 3
    assert not latest_outcomes["eligible_recovery_model"].any()
    assert not latest_outcomes["eligible_day35_weight_model"].any()
    assert latest_outcomes["day_35_weight_g"].notna().all()
    recovery_audit = _frame(payload, "Latest Recovery Audit")
    assert len(recovery_audit) == 12
    assert set(recovery_audit["cycle_id"].astype(str)) == {"2026-3"}
    assert recovery_audit["absolute_error_percentage_points"].notna().all()
    # The corrected 2026-3 source ends at Day 35; no forward-filled Day 36-49
    # rows are allowed back into the latest-cycle audit.
    assert recovery_audit["as_of_date"].max() <= "2026-07-31"
    recovery = _frame(payload, "Recovery Training")
    assert recovery["validation_cycle"].astype(str).equals(recovery["cycle_id"].astype(str))
    assert (pd.to_datetime(recovery["as_of_date"]) < pd.to_datetime(recovery["label_date"])).all()
    latest = _frame(payload, "Latest Cycle Weight Audit")
    assert set(latest["cycle_id"].astype(str)) == {"2026-3"}
    assert latest[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 3
    assert set(latest["training_role"]) == {
        "Prospective audit only - excluded from model fitting and champion selection"
    }


def test_weight_snapshots_never_include_future_checkpoints() -> None:
    payload = build_payload(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    weight = _frame(payload, "Weight Training")
    for checkpoint in (7, 14, 21, 28):
        future = weight["measurement_day"] < checkpoint
        assert weight.loc[future, f"weight_day_{checkpoint}_kg"].isna().all()
    assert weight["validation_cycle"].astype(str).equals(weight["cycle_id"].astype(str))
