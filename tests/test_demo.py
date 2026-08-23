from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from canary import (
    baseline_without_cycle,
    build_cycle_snapshot,
    load_workbook,
    merge_replay_csv,
    validate_replay_prefix,
)
from canary.forecast import attach_forecasts
from canary.risk import load_risk_rules, score_cycle_snapshot
from canary.trish_models import (
    predict_v18_feature_scenario,
    v18_feature_row,
    v18_scenario_contributions,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "FARM HARVEST DATA.xlsx"
DEMO = ROOT / "demo_data" / "2026-3"


@pytest.mark.parametrize("day,expected_rows", [(7, 21), (14, 42), (15, 45), (21, 63), (28, 84), (35, 105)])
def test_demo_csv_is_continuous_source_backed_and_routes_models(day, expected_rows):
    reference = load_workbook(SOURCE)
    path = DEMO / f"Project_Canary_2026-3_Day_{day:02d}.csv"
    raw = pd.read_csv(path)
    validation = validate_replay_prefix(raw, reference)
    assert validation.valid
    assert validation.row_count == expected_rows
    assert set(validation.buildings) == {"Tags 1", "Tags 2", "Tags 3"}
    for _, group in raw.groupby("building_id"):
        assert group["age_day"].tolist() == list(range(1, day + 1))

    dataset, _ = merge_replay_csv(path.read_bytes(), path.name, reference)
    snapshot = score_cycle_snapshot(dataset, "2026-3", validation.cutoff_date, load_risk_rules())
    forecasts = attach_forecasts(dataset, snapshot).set_index("building_id")
    assert forecasts.loc["Tags 1", "recovery_model_name"] == "Trish Model 1 · Extra Trees"
    if day < 35:
        assert forecasts.loc["Tags 1", "day35_weight_model_name"] == "Trish Model 3 · CatBoost"
        expected_checkpoint = 7 if day < 14 else 14 if day < 21 else 21
        assert int(forecasts.loc["Tags 1", "trish_weight_prediction_day"]) == expected_checkpoint
    else:
        assert forecasts.loc["Tags 1", "day35_weight_model_name"] == "Recorded farm measurement"
    assert forecasts.loc["Lags 1", "state"] == "Inactive"


def test_invalid_replay_is_rejected_without_v18_lineage():
    reference = load_workbook(SOURCE)
    path = DEMO / "Project_Canary_2026-3_Day_07.csv"
    raw = pd.read_csv(path)
    raw.loc[0, "population"] -= 1
    validation = validate_replay_prefix(raw, reference)
    assert not validation.valid
    assert "population" in validation.reason


def test_reset_baseline_removes_2026_3():
    baseline = baseline_without_cycle(load_workbook(SOURCE))
    assert "2026-3" not in set(baseline.daily["cycle_id"].astype(str))
    assert "2026-3" not in set(baseline.cycles["cycle_id"].astype(str))
    assert baseline.cycles.groupby("cycle_id")["start_date"].min().idxmax() == "2026-2"


def test_prediction_lab_scenario_changes_prediction_and_returns_shap():
    located = v18_feature_row("model_2", "2026-3", "Tags 1", 14)
    assert located is not None
    baseline, _, _ = located
    before = predict_v18_feature_scenario("model_2", baseline)
    scenario = baseline.copy()
    scenario["average_humidity"] = float(scenario["average_humidity"]) * 1.2
    after = predict_v18_feature_scenario("model_2", scenario)
    assert after != before
    contributions = v18_scenario_contributions("model_2", scenario)
    assert not contributions.empty
    assert {"feature", "value", "contribution"}.issubset(contributions.columns)


def test_streamlit_upload_and_reset_round_trip(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45)
    app.run()
    path = DEMO / "Project_Canary_2026-3_Day_14.csv"
    app.file_uploader[0].set_value((path.name, path.read_bytes(), "text/csv"))
    app.run()
    assert not app.exception
    assert any("2026-3 loaded through Day 14" in item.value for item in app.success)
    uploaded_context = " ".join(
        item.value for item in app.markdown if isinstance(item.value, str)
    )
    assert "Day 14 review" in uploaded_context
    assert "Records through <strong>17 Jul 2026</strong>" in uploaded_context
    review = next(widget for widget in app.date_input if widget.label == "Review date")
    assert str(review.value) == "2026-07-17"
    reset = next(button for button in app.button if button.label == "Reset demo")
    reset.click().run()
    assert not app.exception
    cycle = next(widget for widget in app.selectbox if widget.label == "Harvest cycle")
    assert cycle.value == "2026-2"
    rendered = " ".join(
        item.value
        for item in [*app.markdown, *app.caption, *app.info]
        if isinstance(item.value, str)
    )
    assert "All recorded buildings completed by <strong>03 Aug 2026</strong>" in rendered
    assert "Individual building completion dates are shown below" in rendered
    assert "Harvest completed" in rendered
    assert "Critical risk" not in rendered
    assert "Projected harvest recovery" not in rendered


def test_source_cycle_boundaries_are_staggered_and_demo_cutoff_is_exact():
    dataset = load_workbook(SOURCE)
    cycle_2 = dataset.cycles.loc[dataset.cycles["cycle_id"].eq("2026-2")].set_index(
        "building_id"
    )
    assert cycle_2.loc["Tags 1", "end_date"] == pd.Timestamp("2026-06-06")
    assert cycle_2.loc["Tags 2", "end_date"] == pd.Timestamp("2026-06-06")
    assert cycle_2.loc["Tags 3", "end_date"] == pd.Timestamp("2026-06-06")
    assert cycle_2.loc["Lags 1", "end_date"] == pd.Timestamp("2026-07-20")
    assert cycle_2.loc["Lags 2", "end_date"] == pd.Timestamp("2026-07-28")
    assert cycle_2.loc["Lags 3", "end_date"] == pd.Timestamp("2026-08-03")

    day_7 = pd.read_csv(DEMO / "Project_Canary_2026-3_Day_07.csv")
    assert pd.to_datetime(day_7["record_date"]).min() == pd.Timestamp("2026-07-04")
    assert pd.to_datetime(day_7["record_date"]).max() == pd.Timestamp("2026-07-10")
    assert day_7["age_day"].min() == 1
    assert day_7["age_day"].max() == 7
    assert len(day_7) == 21


@pytest.mark.parametrize("view", ["Prediction Lab", "How Canary Works", "About Canary"])
def test_defense_pages_render(monkeypatch, view):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", view)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=60)
    app.run()
    assert not app.exception
