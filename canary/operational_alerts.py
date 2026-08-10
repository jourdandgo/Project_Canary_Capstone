"""Provisional operational checks kept separate from Canary's core risk score."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import CanaryDataset


DEFAULT_OPERATIONAL_ALERTS_PATH = Path(__file__).resolve().parent.parent / "config" / "operational_alerts.json"


def load_operational_alert_rules(path: str | Path = DEFAULT_OPERATIONAL_ALERTS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _age_range(ranges: list[dict], age: int) -> dict | None:
    return next((item for item in ranges if item["day_min"] <= age <= item["day_max"]), None)


def _alert(
    check: str,
    severity: str,
    evidence: str,
    next_check: str,
    *,
    title: str,
    target: str,
    gap: str,
) -> dict[str, object]:
    return {
        "check": check,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "target": target,
        "gap": gap,
        "next_check": next_check,
    }


def evaluate_operational_alerts(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: object,
    rules: dict | None = None,
) -> list[dict[str, object]]:
    """Return explainable secondary alerts using only evidence available as of the review date."""

    rules = rules or load_operational_alert_rules()
    history = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & (dataset.daily["record_date"] <= pd.Timestamp(as_of))
        & dataset.daily["operational_recorded"]
    ].sort_values("record_date")
    if history.empty:
        return []
    alerts: list[dict[str, object]] = []
    latest = history.iloc[-1]
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == cycle_id)
        & (dataset.cycles["building_id"] == building_id)
    ]
    beginning = float(meta.iloc[0]["beginning_inventory"]) if not meta.empty else float("nan")

    if pd.notna(latest["mortality_daily"]) and pd.notna(beginning) and beginning > 0:
        mortality_pct = float(latest["mortality_daily"]) / beginning * 100
        thresholds = rules["mortality_daily_pct"]
        if mortality_pct >= thresholds["watch"]:
            severity = (
                "Critical" if mortality_pct >= thresholds["critical"]
                else "Warning" if mortality_pct >= thresholds["warning"]
                else "Watch"
            )
            alerts.append(_alert(
                "Daily mortality",
                severity,
                f"{mortality_pct:.2f}% of beginning birds were recorded as mortality on Day {int(latest['age_day'])}.",
                "Confirm the count now. Inspect bird condition and verify water, feed, ventilation, temperature, and humidity; escalate clinical or continuing mortality concerns.",
                title="Daily mortality is above the provisional limit",
                target=f"Below {thresholds['watch']:.2f}% of beginning birds per day",
                gap=f"{max(mortality_pct - thresholds['watch'], 0):.2f} percentage points above the watch limit",
            ))

    for column, check, ranges_key, unit in (
        ("temperature_avg_c", "Temperature", "temperature_ranges_c", "°C"),
        ("humidity_avg_pct", "Humidity", "humidity_ranges_pct", "%"),
    ):
        observed = history.loc[history[column].notna()]
        if observed.empty:
            continue
        record = observed.iloc[-1]
        age = int(record["age_day"])
        stale_days = int(latest["age_day"] - age)
        if stale_days > int(rules.get("maximum_current_reading_age_days", 2)):
            continue
        expected = _age_range(rules[ranges_key], age)
        if expected is None:
            continue
        value = float(record[column])
        if value < expected["minimum"] or value > expected["maximum"]:
            direction = "below" if value < expected["minimum"] else "above"
            target_gap = value - float(expected["target"])
            range_gap = (
                float(expected["minimum"]) - value
                if value < expected["minimum"]
                else value - float(expected["maximum"])
            )
            if check == "Temperature":
                deviations = rules.get("temperature_deviation_from_range_c", {"warning": 1, "critical": 2})
                absolute_gap = range_gap
                severity = "Critical" if absolute_gap >= deviations["critical"] else "Warning" if absolute_gap >= deviations["warning"] else "Watch"
                title = f"Temperature is too {'low' if target_gap < 0 else 'high'} for Day {age}"
                action = (
                    f"Verify the reading at bird height. Check heaters, drafts, curtains, and air leaks; bring conditions within the provisional {expected['minimum']:g}–{expected['maximum']:g}°C reference range."
                    if target_gap < 0
                    else f"Verify the reading at bird height. Check fans, inlets or curtains, cooling pads, airflow, and water availability; bring conditions within the provisional {expected['minimum']:g}–{expected['maximum']:g}°C reference range."
                )
            else:
                severity = "Critical" if value > 70 else "Warning" if value > expected["maximum"] else "Watch"
                title = f"Humidity is too {'low' if target_gap < 0 else 'high'} for Day {age}"
                action = (
                    f"Verify the sensor. Check ventilation schedule, litter dust, and air movement; bring humidity toward {expected['target']:.0f}% within the approved {expected['minimum']:.0f}–{expected['maximum']:.0f}% range."
                    if target_gap < 0
                    else f"Verify the sensor. Check ventilation, litter moisture, leaks, drinkers, and cooling-pad or water-pump timing; bring humidity toward {expected['target']:.0f}% within the approved {expected['minimum']:.0f}–{expected['maximum']:.0f}% range."
                )
            freshness = "current" if stale_days == 0 else f"{stale_days} day(s) old"
            alerts.append(_alert(
                check,
                severity,
                f"{value:.1f}{unit} on Day {age} ({freshness}); provisional range {expected['minimum']:.0f}–{expected['maximum']:.0f}{unit}, target {expected['target']:.0f}{unit}.",
                action,
                title=title,
                target=f"{expected['minimum']:g}–{expected['maximum']:g}{unit} provisional reference range; midpoint {expected['target']:g}{unit}",
                gap=f"{range_gap:.1f}{unit} {direction} the provisional range",
            ))

    feed = history.loc[history["feed_daily_kg_per_bird"].notna()]
    if not feed.empty:
        record = feed.iloc[-1]
        age = int(record["age_day"])
        stale_days = int(latest["age_day"] - age)
        targets = rules.get("feed_targets_g_per_bird", [])
        if age <= len(targets) and stale_days <= int(rules.get("maximum_current_reading_age_days", 2)):
            observed_g = float(record["feed_daily_kg_per_bird"]) * 1000
            target_g = float(targets[age - 1])
            gap_pct = (target_g - observed_g) / target_g * 100 if target_g > 0 else 0
            if gap_pct >= rules.get("feed_gap_pct", {}).get("watch", 5):
                limits = rules["feed_gap_pct"]
                severity = "Critical" if gap_pct >= limits["critical"] else "Warning" if gap_pct >= limits["warning"] else "Watch"
                alerts.append(_alert(
                    "Feed intake",
                    severity,
                    f"{observed_g:.0f} g/bird was recorded on Day {age}; provisional target {target_g:.0f} g/bird.",
                    "Confirm the unit and reading, check feed availability and quality, feeder allocation and line operation, bird access, water access, house temperature, and flock condition.",
                    title=f"Feed intake is {gap_pct:.0f}% below the Day {age} target",
                    target=f"{target_g:.0f} g/bird for Day {age}",
                    gap=f"{target_g - observed_g:.0f} g/bird below target",
                ))

    severity_order = {"Critical": 0, "Warning": 1, "Watch": 2}
    alerts.sort(key=lambda item: severity_order.get(str(item["severity"]), 9))

    return alerts


def build_operational_driver_trace(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: object,
    rules: dict | None = None,
) -> pd.DataFrame:
    """Show recorded operational conditions and the validation status of each check."""

    rules = rules or load_operational_alert_rules()
    history = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & (dataset.daily["record_date"] <= pd.Timestamp(as_of))
        & dataset.daily["operational_recorded"]
    ].sort_values("record_date")
    if history.empty:
        return pd.DataFrame()
    latest = history.iloc[-1]
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == cycle_id)
        & (dataset.cycles["building_id"] == building_id)
    ]
    beginning = float(meta.iloc[0]["beginning_inventory"]) if not meta.empty else float("nan")
    rows: list[dict[str, str]] = []

    def add_row(
        check: str,
        recorded: str,
        reference: str,
        status: str,
        next_check: str,
        effect: str = "None — supporting diagnostic only",
    ) -> None:
        rows.append(
            {
                "Possible operational driver": check,
                "Latest recorded evidence": recorded,
                "Reference": reference,
                "Status": status,
                "What management should check": next_check,
                "Effect on risk score": effect,
            }
        )

    if pd.notna(latest["mortality_daily"]) and pd.notna(beginning) and beginning > 0:
        mortality_pct = float(latest["mortality_daily"]) / beginning * 100
        limits = rules["mortality_daily_pct"]
        if mortality_pct >= limits["critical"]:
            mortality_status = "Critical alert"
        elif mortality_pct >= limits["warning"]:
            mortality_status = "Warning"
        elif mortality_pct >= limits["watch"]:
            mortality_status = "Watch"
        else:
            mortality_status = "Within provisional limit"
        add_row(
            "Daily mortality",
            f"{mortality_pct:.2f}% on Day {int(latest['age_day'])}",
            f"Watch ≥{limits['watch']:.2f}% · warning ≥{limits['warning']:.2f}% · critical ≥{limits['critical']:.2f}%",
            mortality_status,
            "Confirm the count, then inspect flock condition, water, feed, ventilation, temperature, and humidity.",
        )
    else:
        add_row("Daily mortality", "Not recorded", "Provisional thresholds available", "No evidence", "Record or verify today's mortality count.")

    environment_specs = (
        ("temperature_avg_c", "Temperature", "temperature_ranges_c", "°C"),
        ("humidity_avg_pct", "Humidity", "humidity_ranges_pct", "%"),
    )
    for column, check, range_key, unit in environment_specs:
        observed = history.loc[history[column].notna()]
        if observed.empty:
            add_row(
                check,
                "Not recorded",
                "Age-specific provisional range",
                "No evidence",
                f"Verify or record the latest {check.lower()} reading.",
                "Formal environmental dimension — not scored without current evidence",
            )
            continue
        record = observed.iloc[-1]
        age = int(record["age_day"])
        expected = _age_range(rules[range_key], age)
        value = float(record[column])
        if expected is None:
            add_row(
                check,
                f"{value:.1f}{unit} on Day {age}",
                "No proposed range for this age",
                "Needs farm threshold",
                f"Confirm the acceptable {check.lower()} range for Day {age}.",
                "Formal environmental dimension — not scored without an age range",
            )
            continue
        reference = f"Proposed Day {expected['day_min']}–{expected['day_max']} range: {expected['minimum']:.0f}–{expected['maximum']:.0f}{unit}"
        if value < expected["minimum"]:
            status = "Below proposed range"
        elif value > expected["maximum"]:
            status = "Above proposed range"
        else:
            status = "Within proposed range"
        next_check = (
            "Verify the sensor, ventilation, airflow, and cooling or heating conditions."
            if check == "Temperature"
            else "Verify the sensor, ventilation, litter condition, and possible water leakage."
        )
        add_row(
            check,
            f"{value:.1f}{unit} on Day {age}",
            reference,
            status,
            next_check,
            "Formal environmental dimension — worse of temperature or humidity deviation",
        )

    feed = history.loc[history["feed_daily_kg_per_bird"].notna()]
    if feed.empty:
        feed_value = "Not recorded"
        feed_status = "No evidence"
        feed_reference = "Age-based provisional target; unit needs farm confirmation"
    else:
        feed_row = feed.iloc[-1]
        feed_age = int(feed_row["age_day"])
        feed_g = float(feed_row["feed_daily_kg_per_bird"]) * 1000
        feed_targets = rules.get("feed_targets_g_per_bird", [])
        feed_target = float(feed_targets[feed_age - 1]) if 0 < feed_age <= len(feed_targets) else float("nan")
        feed_value = f"{feed_g:.0f} g/bird on Day {feed_age}"
        feed_reference = f"Provisional target: {feed_target:.0f} g/bird" if pd.notna(feed_target) else "No proposed target for this age"
        feed_status = "Below provisional target" if pd.notna(feed_target) and feed_g < feed_target else "At or above provisional target"
    add_row(
        "Feed intake",
        feed_value,
        feed_reference,
        feed_status,
        "Confirm the unit, then verify feed availability, quality, feeder allocation, bird access, and feed-line operation.",
    )
    add_row(
        "Water intake",
        "Not available in the current standardized workbook",
        "No threshold can be applied",
        "Data unavailable",
        "Add a reliable building-day water-intake field before using water as an automated driver.",
    )
    add_row(
        "Combined heat-stress index",
        "Not calculated",
        "THI formula and age-specific alert bands need farm approval",
        "Deferred",
        "Use the separate temperature and humidity checks until the farm approves one THI convention.",
    )
    return pd.DataFrame(rows)
