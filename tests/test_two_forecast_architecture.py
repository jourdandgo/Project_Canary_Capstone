import json
from pathlib import Path

import pandas as pd

from canary import attach_forecasts, load_workbook, score_cycle_snapshot
from canary.forecast_runtime import checkpoint_forecast_history


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "FARM HARVEST DATA.xlsx"


def test_registry_exposes_two_outcomes_and_two_live_models() -> None:
    registry = json.loads(
        (ROOT / "config" / "trish_model_registry.json").read_text(encoding="utf-8")
    )
    assert registry["registry_version"] == "trish-v19-final-2026-08-22"
    assert [item["model_id"] for item in registry["models"]] == [
        "trish_model_1", "trish_model_3"
    ]


def test_bodyweight_forecast_is_held_between_actual_weigh_ins() -> None:
    dataset = load_workbook(SOURCE)
    day22_date = dataset.daily.loc[
        dataset.daily["cycle_id"].eq("2026-3")
        & dataset.daily["building_id"].eq("Tags 1")
        & dataset.daily["age_day"].eq(22),
        "record_date",
    ].max()
    day23_date = dataset.daily.loc[
        dataset.daily["cycle_id"].eq("2026-3")
        & dataset.daily["building_id"].eq("Tags 1")
        & dataset.daily["age_day"].eq(23),
        "record_date",
    ].max()
    day22 = attach_forecasts(
        dataset, score_cycle_snapshot(dataset, "2026-3", pd.Timestamp(day22_date))
    ).set_index("building_id").loc["Tags 1"]
    day23 = attach_forecasts(
        dataset, score_cycle_snapshot(dataset, "2026-3", pd.Timestamp(day23_date))
    ).set_index("building_id").loc["Tags 1"]
    assert day22["projected_day35_weight_kg"] == day23["projected_day35_weight_kg"]
    assert int(day22["trish_weight_prediction_day"]) == int(day23["trish_weight_prediction_day"]) == 21
    assert day23["day35_weight_checkpoint_status"] == "Held from Day 21 evidence"


def test_checkpoint_history_reconciles_the_visible_formula() -> None:
    dataset = load_workbook(SOURCE)
    history = checkpoint_forecast_history(dataset, "2026-3", "Tags 1")
    assert history["Checkpoint"].tolist() == ["Day 7", "Day 14", "Day 21", "Day 28"]
    expected = history["Measured weight (kg)"] + history["Expected remaining gain (kg)"]
    assert expected.round(12).equals(history["Projected Day 35 weight (kg)"].round(12))


def test_forecasts_remain_independent_of_risk_points() -> None:
    dataset = load_workbook(SOURCE)
    as_of = pd.Timestamp("2026-07-24")
    scored = score_cycle_snapshot(dataset, "2026-3", as_of)
    forecasted = attach_forecasts(dataset, scored)
    pd.testing.assert_series_equal(
        scored["risk_score"], forecasted["risk_score"], check_dtype=False
    )


def test_model_3_is_available_at_the_validated_day14_checkpoint() -> None:
    dataset = load_workbook(SOURCE)
    as_of = dataset.daily.loc[
        dataset.daily["cycle_id"].eq("2026-3")
        & dataset.daily["age_day"].eq(14),
        "record_date",
    ].max()
    forecasted = attach_forecasts(
        dataset, score_cycle_snapshot(dataset, "2026-3", pd.Timestamp(as_of))
    )
    active = forecasted.loc[forecasted["state"].eq("Active")]
    assert active["projected_day35_weight_kg"].notna().all()
    assert active["day35_weight_model_id"].eq("M3").all()
    assert active["trish_weight_prediction_day"].eq(14).all()


def test_model_3_is_the_only_named_day21_bodyweight_outlook() -> None:
    dataset = load_workbook(SOURCE)
    as_of = dataset.daily.loc[
        dataset.daily["cycle_id"].eq("2026-3")
        & dataset.daily["age_day"].eq(21),
        "record_date",
    ].max()
    scored = score_cycle_snapshot(dataset, "2026-3", pd.Timestamp(as_of))
    model_3 = attach_forecasts(dataset, scored).set_index("building_id")
    assert model_3.loc["Tags 1", "day35_weight_model_id"] == "M3"
    assert pd.notna(model_3.loc["Tags 1", "projected_day35_weight_kg"])
    pd.testing.assert_series_equal(
        scored.set_index("building_id")["risk_score"], model_3["risk_score"], check_dtype=False
    )
