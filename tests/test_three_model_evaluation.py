import json
from pathlib import Path

import pandas as pd

from canary.data import load_workbook
from canary.three_model_runtime import (
    CHECKPOINT_DIR,
    LEGACY_DIR,
    bodyweight_comparison,
    predict_model_1,
    predict_model_3,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifest(model_id: str) -> dict:
    return json.loads((LEGACY_DIR / f"{model_id}_manifest.json").read_text(encoding="utf-8"))


def test_three_model_cohort_and_thi_contract() -> None:
    model_1 = _manifest("model_1")
    model_3 = _manifest("model_3")
    checkpoint = json.loads((CHECKPOINT_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert model_1["development_building_cycles"] == 31
    assert model_3["development_building_cycles"] == 31
    assert model_1["audit_buildings"] == 3
    assert model_3["audit_buildings"] == 3
    assert model_1["audit_cycle"] == model_3["audit_cycle"] == "2026-3"
    assert {"max_thi_day", "thi_stress_days"}.issubset(model_1["thi_features"])
    assert {"max_thi_day", "thi_stress_days"}.issubset(model_3["thi_features"])
    assert checkpoint["quality_profile"]["development_outcomes"] == 31
    assert checkpoint["quality_profile"]["later_cycle_outcomes"] == 3


def test_reconstructed_model_1_uses_canonical_recovery_proxy() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    prediction = predict_model_1("2026-3", "Tags 1", 14)
    expected = dataset.cycles.loc[
        dataset.cycles["cycle_id"].astype(str).eq("2026-3")
        & dataset.cycles["building_id"].astype(str).eq("Tags 1"),
        "final_recovery_rate",
    ].iloc[0]
    assert prediction is not None
    assert prediction["audit_actual"] == expected
    assert prediction["manifest"]["target_policy"].startswith("Canonical last-recorded")


def test_bodyweight_comparison_exposes_model_3_only_at_day_21() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    rows = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq("2026-3")
        & dataset.daily["building_id"].astype(str).eq("Tags 1")
    ]
    comparison = bodyweight_comparison(
        dataset, "2026-3", "Tags 1", pd.Timestamp(rows["record_date"].max())
    )
    assert comparison["Checkpoint"].tolist() == ["Day 7", "Day 14", "Day 21", "Day 28"]
    assert comparison.loc[comparison["Checkpoint"].ne("Day 21"), "Model 3 projection (kg)"].isna().all()
    assert comparison.loc[comparison["Checkpoint"].eq("Day 21"), "Model 3 projection (kg)"].notna().all()
    assert comparison["Checkpoint model projection (kg)"].notna().all()


def test_model_3_holds_the_day_21_prediction_after_day_21() -> None:
    day_21 = predict_model_3("2026-3", "Tags 2", 21)
    day_28 = predict_model_3("2026-3", "Tags 2", 28)
    assert day_21 is not None and day_28 is not None
    assert day_21["prediction_day"] == day_28["prediction_day"] == 21
    assert day_21["prediction"] == day_28["prediction"]
