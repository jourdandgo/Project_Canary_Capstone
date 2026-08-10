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
        "canonical_building_day_rows": 1666,
        "recovery_building_outcomes": 25,
        "day35_weight_building_outcomes": 31,
        "recovery_training_rows": 122,
        "weight_training_rows": 124,
        "recovery_daily_audit_rows": 1122,
    }
    recovery = _frame(payload, "Recovery Training")
    assert recovery["validation_cycle"].astype(str).equals(recovery["cycle_id"].astype(str))
    assert (pd.to_datetime(recovery["as_of_date"]) < pd.to_datetime(recovery["label_date"])).all()


def test_weight_snapshots_never_include_future_checkpoints() -> None:
    payload = build_payload(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    weight = _frame(payload, "Weight Training")
    for checkpoint in (7, 14, 21, 28):
        future = weight["measurement_day"] < checkpoint
        assert weight.loc[future, f"weight_day_{checkpoint}_kg"].isna().all()
    assert weight["validation_cycle"].astype(str).equals(weight["cycle_id"].astype(str))
