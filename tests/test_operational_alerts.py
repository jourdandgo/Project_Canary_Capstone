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


def test_operational_alerts_are_provisional_and_do_not_affect_risk():
    rules = load_operational_alert_rules()
    assert rules["approval_status"].startswith("Pending")
    assert rules["risk_score_effect"] == "None"
    assert rules["temperature_ranges_c"] == [
        {"day_min": 1, "day_max": 7, "minimum": 29, "target": 31, "maximum": 33},
        {"day_min": 8, "day_max": 14, "minimum": 25.5, "target": 27.5, "maximum": 29.5},
        {"day_min": 15, "day_max": 21, "minimum": 26, "target": 27.25, "maximum": 28.5},
    ]


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
    assert set(trace["Effect on risk score"]) == {
        "None — supporting diagnostic only"
    }


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
