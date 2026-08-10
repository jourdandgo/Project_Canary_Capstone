"""Transparent, rules-based operational risk scoring for Project Canary."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import CanaryDataset
from .state import build_cycle_snapshot, cycle_date_bounds


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "risk_rules.json"
DIMENSION_ORDER = ("weight", "population_loss", "daily_mortality", "environment")
DIMENSION_LABELS = {
    "weight": "Weight gap",
    "population_loss": "Population loss",
    "daily_mortality": "Daily mortality",
    "environment": "Environmental conditions",
}


class RiskConfigurationError(ValueError):
    """Raised when a risk-rules file is structurally unsafe to use."""


def load_risk_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_risk_rules(rules)
    return rules


def _check_cutoffs(values: object, label: str, *, allow_zero_first: bool = True) -> list[float]:
    if not isinstance(values, list) or len(values) != 3:
        raise RiskConfigurationError(f"{label} requires exactly three point cutoffs.")
    numeric = [float(value) for value in values]
    minimum = 0 if allow_zero_first else 0.0000001
    if numeric[0] < minimum or not numeric[0] < numeric[1] < numeric[2]:
        raise RiskConfigurationError(f"{label} cutoffs must be non-negative and strictly increasing.")
    return numeric


def validate_risk_rules(rules: dict[str, Any]) -> None:
    required = {
        "version",
        "approval_status",
        "rating_bands",
        "dimension_cutoffs",
        "temperature_ranges_c",
        "humidity_ranges_pct",
        "maximum_environment_reading_age_days",
        "notes",
    }
    missing = sorted(required - set(rules))
    if missing:
        raise RiskConfigurationError("Risk rules are missing: " + ", ".join(missing))
    dimensions = rules["dimension_cutoffs"]
    required_dimensions = {
        "weight_gap_pct",
        "population_loss_pct",
        "daily_mortality_pct",
        "temperature_deviation_c",
        "humidity_deviation_pp",
    }
    missing_dimensions = sorted(required_dimensions - set(dimensions))
    if missing_dimensions:
        raise RiskConfigurationError("Risk cutoffs are missing: " + ", ".join(missing_dimensions))
    for key in required_dimensions:
        _check_cutoffs(dimensions[key], key)
    for key, upper_limit, label in (
        ("temperature_ranges_c", 60, "Temperature"),
        ("humidity_ranges_pct", 100, "Humidity"),
    ):
        ranges = sorted(rules[key], key=lambda item: int(item["minimum_age"]))
        if not ranges:
            raise RiskConfigurationError(f"At least one {label.lower()} range is required.")
        expected_minimum = 1
        for band in ranges:
            minimum = int(band["minimum_age"])
            maximum = int(band["maximum_age"])
            lower = float(band["minimum"])
            upper = float(band["maximum"])
            if minimum != expected_minimum or maximum < minimum or not 0 <= lower < upper <= upper_limit:
                raise RiskConfigurationError(
                    f"{label} ranges must cover all ages without gaps and use valid values."
                )
            expected_minimum = maximum + 1
        if ranges[-1]["maximum_age"] < 999:
            raise RiskConfigurationError(f"The last {label.lower()} range must cover days beyond Day 35.")
    if int(rules["maximum_environment_reading_age_days"]) < 0:
        raise RiskConfigurationError("Environmental reading age cannot be negative.")

    max_score = len(DIMENSION_ORDER) * 3
    covered_scores: list[int] = []
    for band in rules["rating_bands"]:
        minimum = int(band["minimum"])
        maximum = int(band["maximum"])
        if not str(band["label"]).strip() or maximum < minimum:
            raise RiskConfigurationError("Each rating band requires a label and valid score range.")
        covered_scores.extend(range(minimum, maximum + 1))
    if sorted(covered_scores) != list(range(max_score + 1)):
        raise RiskConfigurationError(f"Rating bands must cover every score from 0 through {max_score} exactly once.")


def save_risk_rules(rules: dict[str, Any], path: str | Path = DEFAULT_RULES_PATH) -> None:
    validate_risk_rules(rules)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _score_threshold(value: object, cutoffs: list[float]) -> object:
    if pd.isna(value):
        return pd.NA
    numeric = float(value)
    if numeric <= cutoffs[0]:
        return 0
    if numeric <= cutoffs[1]:
        return 1
    if numeric <= cutoffs[2]:
        return 2
    return 3


def _rating(total: int, rules: dict[str, Any]) -> str:
    for band in rules["rating_bands"]:
        if int(band["minimum"]) <= total <= int(band["maximum"]):
            return str(band["label"])
    raise RiskConfigurationError(f"No rating band covers score {total}.")


def _rating_rule(total: int, rules: dict[str, Any]) -> str:
    for band in rules["rating_bands"]:
        if int(band["minimum"]) <= total <= int(band["maximum"]):
            return f"Score {int(band['minimum'])}-{int(band['maximum'])} => {band['label']}"
    raise RiskConfigurationError(f"No rating band covers score {total}.")


def _humidity_range(rules: dict[str, Any], age: int) -> dict[str, Any]:
    for band in rules["humidity_ranges_pct"]:
        if int(band["minimum_age"]) <= age <= int(band["maximum_age"]):
            return band
    raise RiskConfigurationError(f"No humidity range covers Day {age}.")


def _temperature_range(rules: dict[str, Any], age: int) -> dict[str, Any]:
    for band in rules["temperature_ranges_c"]:
        if int(band["minimum_age"]) <= age <= int(band["maximum_age"]):
            return band
    raise RiskConfigurationError(f"No temperature range covers Day {age}.")


def _latest_operational_signals(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
    cycle_day: int,
    beginning_inventory: object,
    rules: dict[str, Any],
) -> dict[str, object]:
    history = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & (dataset.daily["record_date"] <= as_of)
        & dataset.daily["operational_recorded"]
    ].sort_values(["record_date", "age_day"])
    result: dict[str, object] = {
        "population_loss_pct": pd.NA,
        "population_loss_score": pd.NA,
        "daily_mortality_pct": pd.NA,
        "daily_mortality_score": pd.NA,
        "temperature_range_c": pd.NA,
        "temperature_avg_c": pd.NA,
        "temperature_minimum_c": pd.NA,
        "temperature_maximum_c": pd.NA,
        "temperature_deviation_c": pd.NA,
        "temperature_direction": "Not scored",
        "temperature_score": pd.NA,
        "humidity_avg_pct": pd.NA,
        "humidity_minimum_pct": pd.NA,
        "humidity_maximum_pct": pd.NA,
        "humidity_deviation_pp": pd.NA,
        "humidity_direction": "Not scored",
        "humidity_score": pd.NA,
        "environment_score": pd.NA,
        "environment_driver": "Not scored",
        "environment_measurement_day": pd.NA,
        "environment_staleness_days": pd.NA,
        "environment_status": "No environmental reading recorded on or before the review date.",
        "environment_last_temperature_range_c": pd.NA,
        "environment_last_temperature_avg_c": pd.NA,
        "environment_last_temperature_minimum_c": pd.NA,
        "environment_last_temperature_maximum_c": pd.NA,
        "environment_last_humidity_avg_pct": pd.NA,
        "environment_last_humidity_minimum_pct": pd.NA,
        "environment_last_humidity_maximum_pct": pd.NA,
    }
    if history.empty:
        return result
    latest = history.iloc[-1]
    beginning = float(beginning_inventory) if pd.notna(beginning_inventory) else float("nan")
    if pd.notna(latest.get("population")) and pd.notna(beginning) and beginning > 0:
        loss = max(0.0, (beginning - float(latest["population"])) / beginning * 100)
        result["population_loss_pct"] = loss
        result["population_loss_score"] = _score_threshold(
            loss, rules["dimension_cutoffs"]["population_loss_pct"]
        )
    if pd.notna(latest.get("mortality_daily")) and pd.notna(beginning) and beginning > 0:
        mortality = max(0.0, float(latest["mortality_daily"]) / beginning * 100)
        result["daily_mortality_pct"] = mortality
        result["daily_mortality_score"] = _score_threshold(
            mortality, rules["dimension_cutoffs"]["daily_mortality_pct"]
        )

    environment_history = history.loc[
        history[["temperature_avg_c", "temperature_min_c", "temperature_max_c", "humidity_avg_pct"]]
        .notna()
        .any(axis=1)
    ]
    if environment_history.empty:
        return result
    environment = environment_history.iloc[-1]
    environment_age = int(environment["age_day"])
    staleness = max(0, int(cycle_day) - environment_age)
    result["environment_measurement_day"] = environment_age
    result["environment_staleness_days"] = staleness
    maximum_age = int(rules["maximum_environment_reading_age_days"])
    if pd.notna(environment.get("temperature_min_c")) and pd.notna(environment.get("temperature_max_c")):
        result["environment_last_temperature_range_c"] = max(
            0.0, float(environment["temperature_max_c"]) - float(environment["temperature_min_c"])
        )
    if pd.notna(environment.get("temperature_avg_c")):
        last_temperature = float(environment["temperature_avg_c"])
        accepted_temperature = _temperature_range(rules, environment_age)
        result.update(
            {
                "environment_last_temperature_avg_c": last_temperature,
                "environment_last_temperature_minimum_c": float(accepted_temperature["minimum"]),
                "environment_last_temperature_maximum_c": float(accepted_temperature["maximum"]),
            }
        )
    if pd.notna(environment.get("humidity_avg_pct")):
        last_humidity = float(environment["humidity_avg_pct"])
        accepted = _humidity_range(rules, environment_age)
        result.update(
            {
                "environment_last_humidity_avg_pct": last_humidity,
                "environment_last_humidity_minimum_pct": float(accepted["minimum"]),
                "environment_last_humidity_maximum_pct": float(accepted["maximum"]),
            }
        )
    if staleness > maximum_age:
        result["environment_status"] = (
            f"Stale — last environmental reading was Day {environment_age}, {staleness} day(s) old; "
            f"maximum allowed is {maximum_age} day(s)."
        )
        return result
    result["environment_status"] = (
        f"Current — recorded Day {environment_age}, {staleness} day(s) old."
    )

    component_scores: list[tuple[int, str]] = []
    if pd.notna(environment.get("temperature_avg_c")):
        temperature = float(environment["temperature_avg_c"])
        accepted_temperature = _temperature_range(rules, environment_age)
        lower_temperature = float(accepted_temperature["minimum"])
        upper_temperature = float(accepted_temperature["maximum"])
        if temperature < lower_temperature:
            temperature_deviation = lower_temperature - temperature
            temperature_direction = "Low Temperature"
        elif temperature > upper_temperature:
            temperature_deviation = temperature - upper_temperature
            temperature_direction = "High Temperature"
        else:
            temperature_deviation = 0.0
            temperature_direction = "Within Range"
        temperature_score = int(_score_threshold(
            temperature_deviation, rules["dimension_cutoffs"]["temperature_deviation_c"]
        ))
        result.update({
            "temperature_avg_c": temperature,
            "temperature_minimum_c": lower_temperature,
            "temperature_maximum_c": upper_temperature,
            "temperature_deviation_c": temperature_deviation,
            "temperature_direction": temperature_direction,
            "temperature_score": temperature_score,
        })
        component_scores.append((temperature_score, temperature_direction))

    if pd.notna(environment.get("humidity_avg_pct")):
        humidity = float(environment["humidity_avg_pct"])
        accepted = _humidity_range(rules, environment_age)
        lower = float(accepted["minimum"])
        upper = float(accepted["maximum"])
        if humidity < lower:
            deviation = lower - humidity
            direction = "Low Humidity"
        elif humidity > upper:
            deviation = humidity - upper
            direction = "High Humidity"
        else:
            deviation = 0.0
            direction = "Within Range"
        humidity_score = int(
            _score_threshold(deviation, rules["dimension_cutoffs"]["humidity_deviation_pp"])
        )
        result.update(
            {
                "humidity_avg_pct": humidity,
                "humidity_minimum_pct": lower,
                "humidity_maximum_pct": upper,
                "humidity_deviation_pp": deviation,
                "humidity_direction": direction,
                "humidity_score": humidity_score,
            }
        )
        component_scores.append((humidity_score, direction))

    if component_scores:
        component_scores.sort(key=lambda item: (-item[0], item[1]))
        result["environment_score"] = component_scores[0][0]
        result["environment_driver"] = component_scores[0][1]
    return result


def _base_signals(
    dataset: CanaryDataset,
    snapshot: pd.DataFrame,
    cycle_id: str,
    as_of: pd.Timestamp,
    rules: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in snapshot.iterrows():
        record = row.to_dict()
        record.update({"weight_gap_pct": pd.NA, "weight_score": pd.NA})
        if row["state"] == "Inactive" or pd.isna(row["cycle_day"]):
            records.append(record)
            continue
        observed_weight = row["latest_weight_kg"]
        weight_target = row["weight_target_at_measurement_kg"]
        if pd.notna(observed_weight) and pd.notna(weight_target) and float(weight_target) > 0:
            weight_gap = max(0.0, (float(weight_target) - float(observed_weight)) / float(weight_target) * 100)
            record["weight_gap_pct"] = weight_gap
            record["weight_score"] = _score_threshold(
                weight_gap, rules["dimension_cutoffs"]["weight_gap_pct"]
            )
        record.update(
            _latest_operational_signals(
                dataset,
                cycle_id,
                str(row["building_id"]),
                as_of,
                int(row["cycle_day"]),
                row.get("beginning_inventory"),
                rules,
            )
        )
        records.append(record)
    return pd.DataFrame(records)


def _score_evidence(row: pd.Series) -> dict[str, str]:
    evidence: dict[str, str] = {}
    if pd.notna(row.get("weight_score")):
        evidence["weight"] = (
            f"Weight is {float(row['weight_gap_pct']):.1f}% below the Day {int(row['weight_measurement_day'])} target "
            f"({float(row['latest_weight_kg']):.3f} vs {float(row['weight_target_at_measurement_kg']):.3f} kg; "
            f"score {int(row['weight_score'])}; {row['weight_freshness']})."
        )
    if pd.notna(row.get("population_loss_score")):
        evidence["population_loss"] = (
            f"Population loss is {float(row['population_loss_pct']):.2f}% of beginning birds "
            f"(score {int(row['population_loss_score'])}; {row['data_freshness']})."
        )
    if pd.notna(row.get("daily_mortality_score")):
        evidence["daily_mortality"] = (
            f"Latest daily mortality is {float(row['daily_mortality_pct']):.2f}% of beginning birds "
            f"(score {int(row['daily_mortality_score'])}; {row['data_freshness']})."
        )
    if pd.notna(row.get("environment_score")):
        driver = str(row.get("environment_driver"))
        if driver in {"High Temperature", "Low Temperature"}:
            detail = (
                f"average temperature is {float(row['temperature_avg_c']):.1f}°C versus the "
                f"{float(row['temperature_minimum_c']):.0f}–{float(row['temperature_maximum_c']):.0f}°C age range"
            )
        elif driver in {"High Humidity", "Low Humidity"}:
            detail = (
                f"humidity is {float(row['humidity_avg_pct']):.1f}% versus the "
                f"{float(row['humidity_minimum_pct']):.0f}–{float(row['humidity_maximum_pct']):.0f}% age range"
            )
        else:
            detail = "recorded conditions are within the provisional bands"
        evidence["environment"] = (
            f"Environmental condition: {driver}; {detail} "
            f"(score {int(row['environment_score'])}; recorded Day {int(row['environment_measurement_day'])})."
        )
    return evidence


def _threshold_description(cutoffs: list[float], unit: str) -> str:
    first, second, third = cutoffs
    return (
        f"0: <= {first:g}{unit}; 1: > {first:g} to {second:g}{unit}; "
        f"2: > {second:g} to {third:g}{unit}; 3: > {third:g}{unit}"
    )


def build_dimension_trace(scored_row: pd.Series | dict[str, object], rules: dict[str, Any] | None = None) -> pd.DataFrame:
    rules = rules or load_risk_rules()
    row = pd.Series(scored_row)
    if pd.isna(row.get("cycle_day")):
        return pd.DataFrame(columns=["Dimension", "Raw observations", "Calculation", "Applied thresholds", "Score", "Data status"])
    cutoffs = rules["dimension_cutoffs"]
    if pd.isna(row.get("environment_score")):
        last_parts: list[str] = []
        if pd.notna(row.get("environment_last_temperature_avg_c")):
            last_parts.append(
                f"last average temperature {float(row['environment_last_temperature_avg_c']):.1f}°C"
            )
        if pd.notna(row.get("environment_last_humidity_avg_pct")):
            last_parts.append(
                f"last humidity {float(row['environment_last_humidity_avg_pct']):.1f}%"
            )
        environment_raw = "; ".join(last_parts) if last_parts else "No recorded temperature-range or humidity evidence"
    else:
        current_parts: list[str] = []
        if pd.notna(row.get("temperature_avg_c")):
            current_parts.append(
                f"average temperature {float(row['temperature_avg_c']):.1f}°C "
                f"(accepted {float(row['temperature_minimum_c']):.0f}–{float(row['temperature_maximum_c']):.0f}°C)"
            )
        if pd.notna(row.get("humidity_avg_pct")):
            current_parts.append(
                f"humidity {float(row['humidity_avg_pct']):.1f}% "
                f"(accepted {float(row['humidity_minimum_pct']):.0f}–{float(row['humidity_maximum_pct']):.0f}%)"
            )
        environment_raw = "; ".join(current_parts)
    trace = [
        {
            "Dimension": "Weight gap",
            "Raw observations": "Unavailable" if pd.isna(row.get("latest_weight_kg")) else f"{float(row['latest_weight_kg']):.3f} kg on Day {int(row['weight_measurement_day'])}; target {float(row['weight_target_at_measurement_kg']):.3f} kg",
            "Calculation": "Not scored" if pd.isna(row.get("weight_gap_pct")) else f"max((target - actual) / target, 0) = {float(row['weight_gap_pct']):.2f}%",
            "Applied thresholds": _threshold_description(cutoffs["weight_gap_pct"], "%"),
            "Score": row.get("weight_score", pd.NA),
            "Data status": row.get("weight_freshness", "Unavailable"),
        },
        {
            "Dimension": "Population loss",
            "Raw observations": "Unavailable" if pd.isna(row.get("percentage_alive")) else f"{float(row['percentage_alive']):.2%} alive from {int(row['beginning_inventory']):,} beginning birds",
            "Calculation": "Not scored" if pd.isna(row.get("population_loss_pct")) else f"(beginning - current) / beginning = {float(row['population_loss_pct']):.2f}%",
            "Applied thresholds": _threshold_description(cutoffs["population_loss_pct"], "%"),
            "Score": row.get("population_loss_score", pd.NA),
            "Data status": row.get("data_freshness", "Unavailable"),
        },
        {
            "Dimension": "Daily mortality",
            "Raw observations": "Unavailable" if pd.isna(row.get("daily_mortality_pct")) else f"{float(row['daily_mortality_pct']):.2f}% of beginning birds on recorded Day {int(row['latest_operational_day'])}",
            "Calculation": "latest daily mortality / beginning population",
            "Applied thresholds": _threshold_description(cutoffs["daily_mortality_pct"], "%"),
            "Score": row.get("daily_mortality_score", pd.NA),
            "Data status": row.get("data_freshness", "Unavailable"),
        },
        {
            "Dimension": "Environmental conditions",
            "Raw observations": environment_raw,
            "Calculation": f"Higher of temperature-deviation score and humidity-deviation score; driver: {row.get('environment_driver', 'Not scored')}",
            "Applied thresholds": f"Temperature outside age range: {_threshold_description(cutoffs['temperature_deviation_c'], '°C')}; humidity outside age range: {_threshold_description(cutoffs['humidity_deviation_pp'], ' pp')}",
            "Score": row.get("environment_score", pd.NA),
            "Data status": row.get(
                "environment_status",
                "Not scored" if pd.isna(row.get("environment_score")) else f"Day {int(row['environment_measurement_day'])}; {int(row['environment_staleness_days'])} day(s) old",
            ),
        },
    ]
    result = pd.DataFrame(trace)
    result["Score"] = result["Score"].astype("Int64")
    return result


def _pattern_for(row: pd.Series, available: dict[str, int]) -> str:
    if not available:
        return "Missing or Stale Evidence"
    highest = max(available.values())
    if highest == 0:
        return "No Material Concern"
    priority = ("daily_mortality", "population_loss", "environment", "weight")
    driver = next(dimension for dimension in priority if available.get(dimension) == highest)
    if driver == "daily_mortality":
        return "High Mortality"
    if driver == "population_loss":
        return "Rapid Population Loss"
    if driver == "environment":
        environment_driver = str(row.get("environment_driver", "Abnormal Temperature Fluctuation"))
        return environment_driver if environment_driver != "Within Range" else "Abnormal Temperature Fluctuation"
    return "Low Body Weight"


def score_cycle_snapshot(
    dataset: CanaryDataset,
    cycle_id: str,
    as_of: date | pd.Timestamp,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rules = rules or load_risk_rules()
    as_of_ts = pd.Timestamp(as_of).normalize()
    frame = _base_signals(dataset, build_cycle_snapshot(dataset, cycle_id, as_of_ts), cycle_id, as_of_ts, rules)
    scored_columns = [f"{dimension}_score" for dimension in DIMENSION_ORDER]
    output_records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        record.update(
            {
                "risk_score": pd.NA,
                "risk_rating": "Not rated",
                "risk_pattern": "Not applicable",
                "scored_dimensions": 0,
                "evidence_status": "Not eligible",
                "why_primary": row["status_note"],
                "why_supporting": "",
                "risk_rule_version": rules["version"],
                "risk_approval_status": rules["approval_status"],
                "score_equation": "Not applicable",
                "risk_label_rule": "Not applicable",
                "identified_problem": "Not applicable",
                "recommended_action": "Not applicable for this building state.",
                "recommendation_rule_id": "Not applicable",
            }
        )
        if row["state"] not in {"Active", "Incomplete", "Records ended"}:
            output_records.append(record)
            continue
        scores = {dimension: row.get(f"{dimension}_score") for dimension in DIMENSION_ORDER}
        available = {dimension: int(value) for dimension, value in scores.items() if pd.notna(value)}
        if not available:
            record["why_primary"] = "Insufficient observations to calculate operational risk."
            record["evidence_status"] = "Insufficient"
            output_records.append(record)
            continue
        total = sum(available.values())
        evidence = _score_evidence(row)
        highest = max(available.values())
        primary_dimensions = [dimension for dimension in DIMENSION_ORDER if available.get(dimension) == highest]
        pattern = _pattern_for(row, available)
        record.update(
            {
                "risk_score": total,
                "risk_rating": _rating(total, rules),
                "risk_pattern": pattern,
                "scored_dimensions": len(available),
                "evidence_status": "Complete" if len(available) == 4 else "Reduced evidence",
                "why_primary": " ".join(evidence[d] for d in primary_dimensions if d in evidence) or "No material concern in scored dimensions.",
                "why_supporting": " ".join(evidence[d] for d in DIMENSION_ORDER if d in evidence and d not in primary_dimensions),
                "score_equation": " + ".join(f"{DIMENSION_LABELS[d]} {available[d]}" for d in DIMENSION_ORDER if d in available) + f" = {total}",
                "risk_label_rule": _rating_rule(total, rules),
                "identified_problem": pattern,
            }
        )
        output_records.append(record)
    output = pd.DataFrame(output_records)
    for column in ["risk_score", *scored_columns]:
        if column in output:
            output[column] = output[column].astype("Int64")
    return output


def build_risk_history(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    through: date | pd.Timestamp,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rules = rules or load_risk_rules()
    minimum, maximum = cycle_date_bounds(dataset, cycle_id)
    end = min(pd.Timestamp(through).date(), maximum)
    records: list[dict[str, object]] = []
    for current in pd.date_range(minimum, end, freq="D"):
        match = score_cycle_snapshot(dataset, cycle_id, current, rules)
        match = match.loc[match["building_id"] == building_id]
        if match.empty:
            continue
        row = match.iloc[0]
        records.append(
            {
                "record_date": current,
                "cycle_day": row["cycle_day"],
                "state": row["state"],
                "risk_score": row["risk_score"],
                "risk_rating": row["risk_rating"],
                "risk_pattern": row["risk_pattern"],
                "weight_score": row.get("weight_score", pd.NA),
                "population_loss_score": row.get("population_loss_score", pd.NA),
                "daily_mortality_score": row.get("daily_mortality_score", pd.NA),
                "environment_score": row.get("environment_score", pd.NA),
                "evidence_status": row["evidence_status"],
            }
        )
    return pd.DataFrame(records)
