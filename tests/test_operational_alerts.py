from dataclasses import replace
from pathlib import Path
import os

from canary import (
    build_operational_driver_trace,
    evaluate_operational_alerts,
    evaluate_persistent_signals,
    load_operational_alert_rules,
    load_workbook,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_operational_alerts_use_the_supplied_tropical_age_bands():
    rules = load_operational_alert_rules()
    assert "pending Doc Raymond" in rules["approval_status"]
    assert rules["temperature_ranges_c"][0] == {
        "day_min": 1, "day_max": 6, "minimum": 29, "target": 31, "maximum": 33
    }
    assert rules["temperature_ranges_c"][-1]["day_max"] == 999
    assert rules["humidity_ranges_pct"][0]["minimum"] == 60
    assert rules["humidity_ranges_pct"][-1]["maximum"] == 65


def test_operational_alerts_are_explainable_when_present():
    dataset = load_workbook(SOURCE)
    alerts = evaluate_operational_alerts(dataset, "2025-5", "Tags 1", "2025-11-25")
    for alert in alerts:
        assert {"check", "severity", "evidence", "next_check"}.issubset(alert)


def test_operational_driver_trace_keeps_unavailable_inputs_explicit():
    dataset = load_workbook(SOURCE)
    trace = build_operational_driver_trace(dataset, "2025-5", "Tags 1", "2025-11-25")
    assert {
        "Temperature",
        "Humidity",
        "Feed intake",
        "Water intake",
        "Combined heat-stress index",
    }.issubset(set(trace["Possible operational driver"]))
    environment = trace.loc[trace["Possible operational driver"].isin(["Temperature", "Humidity"])]
    assert environment["Effect on risk score"].str.startswith("Formal environmental dimension").all()
    supporting = trace.loc[~trace["Possible operational driver"].isin(["Temperature", "Humidity"])]
    assert set(supporting["Effect on risk score"]) == {"None — supporting diagnostic only"}


def test_feed_alerts_stay_disabled_until_units_are_confirmed():
    dataset = load_workbook(SOURCE)
    alerts = evaluate_operational_alerts(dataset, "2026-3", "Tags 2", "2026-07-25")
    assert load_operational_alert_rules()["feed_alert_enabled"] is False
    assert not any(alert["check"] == "Feed intake" for alert in alerts)


def test_temperature_gap_is_measured_to_range_boundary_not_midpoint():
    dataset = load_workbook(SOURCE)
    alerts = evaluate_operational_alerts(dataset, "2026-3", "Tags 2", "2026-07-09")
    temperature = next(alert for alert in alerts if alert["check"] == "Temperature")

    assert "provisional reference range" in temperature["target"]
    assert "above the provisional range" in temperature["gap"]


def test_persistent_environment_signal_requires_three_consecutive_recorded_days():
    dataset = load_workbook(SOURCE)
    daily = dataset.daily.copy()
    mask = (
        daily["cycle_id"].eq("2026-3")
        & daily["building_id"].eq("Tags 2")
        & daily["age_day"].isin([7, 8, 9])
    )
    assert mask.sum() == 3
    daily.loc[mask, "temperature_avg_c"] = 40.0
    modified = replace(dataset, daily=daily)
    as_of = daily.loc[mask, "record_date"].max()

    signals = evaluate_persistent_signals(modified, "2026-3", "Tags 2", as_of)
    temperature = next(signal for signal in signals if signal["check"] == "Temperature")

    assert "3 consecutive recorded days" in temperature["title"]
    assert len(temperature["trace"]) == 3
    assert temperature["risk_score_effect"].startswith("None")


def test_unresolved_weight_signal_does_not_claim_three_weight_measurements():
    dataset = load_workbook(SOURCE)
    daily = dataset.daily.copy()
    building = daily.loc[
        daily["cycle_id"].eq("2026-3") & daily["building_id"].eq("Tags 2")
    ]
    as_of = building.loc[building["age_day"].eq(9), "record_date"].max()

    signals = evaluate_persistent_signals(dataset, "2026-3", "Tags 2", as_of)
    weight = next((signal for signal in signals if signal["check"] == "Bodyweight"), None)
    if weight is not None:
        assert "unresolved" in weight["title"]
        assert "one checkpoint" in weight["basis"]
        assert len(weight["trace"]) == 1
