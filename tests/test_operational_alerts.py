from pathlib import Path
import os

from canary import (
    build_operational_driver_trace,
    evaluate_operational_alerts,
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


def test_current_cycle_surfaces_specific_feed_gap_and_action():
    dataset = load_workbook(SOURCE)
    alerts = evaluate_operational_alerts(dataset, "2026-3", "Tags 2", "2026-07-25")
    feed = next(alert for alert in alerts if alert["check"] == "Feed intake")

    assert "below the Day 22 target" in feed["title"]
    assert "74 g/bird" in feed["evidence"]
    assert "provisional target 130 g/bird" in feed["evidence"]
    assert "feed availability" in feed["next_check"]


def test_temperature_gap_is_measured_to_range_boundary_not_midpoint():
    dataset = load_workbook(SOURCE)
    alerts = evaluate_operational_alerts(dataset, "2026-3", "Tags 2", "2026-07-09")
    temperature = next(alert for alert in alerts if alert["check"] == "Temperature")

    assert "provisional reference range" in temperature["target"]
    assert "above the provisional range" in temperature["gap"]
