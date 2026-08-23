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
PATTERN_PRIORITY = (
    "High Mortality",
    "Rapid Population Loss",
    "High Temperature",
    "Low Temperature",
    "High Humidity",
    "Low Humidity",
    "Low Body Weight",
    "Missing or Stale Evidence",
    "No Material Concern",
)


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
        "priority_policy",
        "threshold_provenance",
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

    priority_policy = rules["priority_policy"]
    required_policy = {
        "minimum_scored_dimensions",
        "domains",
    }
    missing_policy = sorted(required_policy - set(priority_policy))
    if missing_policy:
        raise RiskConfigurationError(
            "Priority policy is missing: " + ", ".join(missing_policy)
        )
    if not 1 <= int(priority_policy["minimum_scored_dimensions"]) <= len(DIMENSION_ORDER):
        raise RiskConfigurationError("Minimum scored dimensions must be between 1 and 4.")
    domains = priority_policy["domains"]
    if set(domains) != set(DIMENSION_ORDER) or not all(str(value).strip() for value in domains.values()):
        raise RiskConfigurationError("Priority-policy domains must map every risk dimension.")
    if set(rules["threshold_provenance"]) != required_dimensions:
        raise RiskConfigurationError("Threshold provenance must cover every configured cutoff.")

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


def _priority_decision(
    available: dict[str, int], rules: dict[str, Any]
) -> tuple[str, str, str, bool]:
    """Return the published score-band label without automatic overrides."""
    total = sum(available.values())
    base_label = _rating(total, rules)
    base_rule = _rating_rule(total, rules)
    return base_label, f"Base band: {base_rule}.", f"PRIORITY-BASE-{base_label.upper()}", False


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
        return pd.DataFrame(columns=["Dimension", "Domain", "Raw observations", "Calculation", "Applied thresholds", "Threshold source", "Score", "Data status"])
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
            "Domain": str(rules["priority_policy"]["domains"]["weight"]).title(),
            "Raw observations": "Unavailable" if pd.isna(row.get("latest_weight_kg")) else f"{float(row['latest_weight_kg']):.3f} kg on Day {int(row['weight_measurement_day'])}; target {float(row['weight_target_at_measurement_kg']):.3f} kg",
            "Calculation": "Not scored" if pd.isna(row.get("weight_gap_pct")) else f"max((target - actual) / target, 0) = {float(row['weight_gap_pct']):.2f}%",
            "Applied thresholds": _threshold_description(cutoffs["weight_gap_pct"], "%"),
            "Threshold source": rules["threshold_provenance"]["weight_gap_pct"],
            "Score": row.get("weight_score", pd.NA),
            "Data status": row.get("weight_freshness", "Unavailable"),
        },
        {
            "Dimension": "Population loss",
            "Domain": str(rules["priority_policy"]["domains"]["population_loss"]).title(),
            "Raw observations": "Unavailable" if pd.isna(row.get("percentage_alive")) else f"{float(row['percentage_alive']):.2%} alive from {int(row['beginning_inventory']):,} beginning birds",
            "Calculation": "Not scored" if pd.isna(row.get("population_loss_pct")) else f"(beginning - current) / beginning = {float(row['population_loss_pct']):.2f}%",
            "Applied thresholds": _threshold_description(cutoffs["population_loss_pct"], "%"),
            "Threshold source": rules["threshold_provenance"]["population_loss_pct"],
            "Score": row.get("population_loss_score", pd.NA),
            "Data status": row.get("data_freshness", "Unavailable"),
        },
        {
            "Dimension": "Daily mortality",
            "Domain": str(rules["priority_policy"]["domains"]["daily_mortality"]).title(),
            "Raw observations": "Unavailable" if pd.isna(row.get("daily_mortality_pct")) else f"{float(row['daily_mortality_pct']):.2f}% of beginning birds on recorded Day {int(row['latest_operational_day'])}",
            "Calculation": "latest daily mortality / beginning population",
            "Applied thresholds": _threshold_description(cutoffs["daily_mortality_pct"], "%"),
            "Threshold source": rules["threshold_provenance"]["daily_mortality_pct"],
            "Score": row.get("daily_mortality_score", pd.NA),
            "Data status": row.get("data_freshness", "Unavailable"),
        },
        {
            "Dimension": "Environmental conditions",
            "Domain": str(rules["priority_policy"]["domains"]["environment"]).title(),
            "Raw observations": environment_raw,
            "Calculation": f"Higher of temperature-deviation score and humidity-deviation score; driver: {row.get('environment_driver', 'Not scored')}",
            "Applied thresholds": f"Temperature outside age range: {_threshold_description(cutoffs['temperature_deviation_c'], '°C')}; humidity outside age range: {_threshold_description(cutoffs['humidity_deviation_pp'], ' pp')}",
            "Threshold source": (
                f"Temperature: {rules['threshold_provenance']['temperature_deviation_c']} "
                f"Humidity: {rules['threshold_provenance']['humidity_deviation_pp']}"
            ),
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


def _detected_patterns(
    row: pd.Series,
    available: dict[str, int],
    *,
    include_missing: bool,
) -> list[tuple[str, object, str]]:
    """Return every supported problem pattern, ordered for management review.

    The first item remains the primary headline for backward compatibility.
    Additional items are retained for multi-pattern recommendation matching.
    """

    candidates: list[tuple[str, int, str]] = []

    def add(pattern: str, value: object, dimension: str) -> None:
        if pd.notna(value) and int(value) > 0:
            candidates.append((pattern, int(value), dimension))

    add("High Mortality", row.get("daily_mortality_score"), "Daily mortality")
    add("Rapid Population Loss", row.get("population_loss_score"), "Population loss")
    if str(row.get("temperature_direction")) == "High Temperature":
        add("High Temperature", row.get("temperature_score"), "Environmental conditions")
    elif str(row.get("temperature_direction")) == "Low Temperature":
        add("Low Temperature", row.get("temperature_score"), "Environmental conditions")
    if str(row.get("humidity_direction")) == "High Humidity":
        add("High Humidity", row.get("humidity_score"), "Environmental conditions")
    elif str(row.get("humidity_direction")) == "Low Humidity":
        add("Low Humidity", row.get("humidity_score"), "Environmental conditions")
    add("Low Body Weight", row.get("weight_score"), "Weight gap")

    priority = {pattern: index for index, pattern in enumerate(PATTERN_PRIORITY)}
    candidates.sort(key=lambda item: (-item[1], priority[item[0]]))
    detected: list[tuple[str, object, str]] = list(candidates)
    if include_missing:
        detected.append(("Missing or Stale Evidence", pd.NA, "Data availability"))
    if not detected:
        detected.append(("No Material Concern", 0, "All scored dimensions"))
    return detected


def build_pattern_trace(
    scored_row: pd.Series | dict[str, object],
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Explain every problem-pattern criterion and whether it matched."""

    rules = rules or load_risk_rules()
    row = pd.Series(scored_row)
    cutoffs = rules["dimension_cutoffs"]
    score_columns = [f"{dimension}_score" for dimension in DIMENSION_ORDER]
    scored_count = sum(pd.notna(row.get(column)) for column in score_columns)
    complete = scored_count == len(DIMENSION_ORDER)
    all_zero = complete and all(int(row.get(column)) == 0 for column in score_columns)

    def value_or_missing(value: object, formatter) -> str:
        return "Unavailable or stale" if pd.isna(value) else formatter(float(value))

    rows = [
        {
            "Problem pattern": "Low Body Weight",
            "Criterion": f"Measured weight gap > {cutoffs['weight_gap_pct'][0]:g}% of the age-specific target",
            "Recorded evidence": value_or_missing(row.get("weight_gap_pct"), lambda value: f"{value:.1f}% below target"),
            "Detected": pd.notna(row.get("weight_score")) and int(row.get("weight_score")) > 0,
        },
        {
            "Problem pattern": "High Mortality",
            "Criterion": f"Latest daily mortality > {cutoffs['daily_mortality_pct'][0]:g}% of beginning population",
            "Recorded evidence": value_or_missing(row.get("daily_mortality_pct"), lambda value: f"{value:.2f}% of beginning population"),
            "Detected": pd.notna(row.get("daily_mortality_score")) and int(row.get("daily_mortality_score")) > 0,
        },
        {
            "Problem pattern": "Rapid Population Loss",
            "Criterion": f"Cumulative population loss > {cutoffs['population_loss_pct'][0]:g}% of beginning population",
            "Recorded evidence": value_or_missing(row.get("population_loss_pct"), lambda value: f"{value:.2f}% of beginning population"),
            "Detected": pd.notna(row.get("population_loss_score")) and int(row.get("population_loss_score")) > 0,
        },
        {
            "Problem pattern": "High Temperature",
            "Criterion": "Latest current daily average is above the age-specific upper temperature limit",
            "Recorded evidence": "Unavailable or stale" if pd.isna(row.get("temperature_avg_c")) else f"{float(row['temperature_avg_c']):.1f}°C; upper limit {float(row['temperature_maximum_c']):.1f}°C",
            "Detected": str(row.get("temperature_direction")) == "High Temperature" and pd.notna(row.get("temperature_score")) and int(row.get("temperature_score")) > 0,
        },
        {
            "Problem pattern": "Low Temperature",
            "Criterion": "Latest current daily average is below the age-specific lower temperature limit",
            "Recorded evidence": "Unavailable or stale" if pd.isna(row.get("temperature_avg_c")) else f"{float(row['temperature_avg_c']):.1f}°C; lower limit {float(row['temperature_minimum_c']):.1f}°C",
            "Detected": str(row.get("temperature_direction")) == "Low Temperature" and pd.notna(row.get("temperature_score")) and int(row.get("temperature_score")) > 0,
        },
        {
            "Problem pattern": "High Humidity",
            "Criterion": "Latest current daily average is above the age-specific upper humidity limit",
            "Recorded evidence": "Unavailable or stale" if pd.isna(row.get("humidity_avg_pct")) else f"{float(row['humidity_avg_pct']):.1f}%; upper limit {float(row['humidity_maximum_pct']):.1f}%",
            "Detected": str(row.get("humidity_direction")) == "High Humidity" and pd.notna(row.get("humidity_score")) and int(row.get("humidity_score")) > 0,
        },
        {
            "Problem pattern": "Low Humidity",
            "Criterion": "Latest current daily average is below the age-specific lower humidity limit",
            "Recorded evidence": "Unavailable or stale" if pd.isna(row.get("humidity_avg_pct")) else f"{float(row['humidity_avg_pct']):.1f}%; lower limit {float(row['humidity_minimum_pct']):.1f}%",
            "Detected": str(row.get("humidity_direction")) == "Low Humidity" and pd.notna(row.get("humidity_score")) and int(row.get("humidity_score")) > 0,
        },
        {
            "Problem pattern": "Missing or Stale Evidence",
            "Criterion": "At least one of the four risk dimensions cannot be scored from current evidence",
            "Recorded evidence": f"{scored_count}/4 dimensions scored; {row.get('evidence_status', 'status unavailable')}",
            "Detected": not complete,
        },
        {
            "Problem pattern": "No Material Concern",
            "Criterion": "All four dimensions are available and each scores 0/3",
            "Recorded evidence": f"{scored_count}/4 dimensions scored",
            "Detected": all_zero,
        },
    ]
    return pd.DataFrame(rows)


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
                "base_risk_rating": "Not rated",
                "risk_pattern": "Not applicable",
                "risk_patterns": "Not applicable",
                "risk_pattern_details": "Not applicable",
                "risk_pattern_count": 0,
                "scored_dimensions": 0,
                "available_score_max": 0,
                "evidence_status": "Not eligible",
                "why_primary": row["status_note"],
                "why_supporting": "",
                "risk_rule_version": rules["version"],
                "risk_approval_status": rules["approval_status"],
                "score_equation": "Not applicable",
                "risk_label_rule": "Not applicable",
                "priority_rule_id": "Not applicable",
                "priority_override_applied": False,
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
            record["risk_pattern"] = "Missing or Stale Evidence"
            record["risk_patterns"] = "Missing or Stale Evidence"
            record["risk_pattern_details"] = "Missing or Stale Evidence"
            record["risk_pattern_count"] = 1
            record["why_primary"] = "Insufficient observations to calculate operational risk."
            record["evidence_status"] = "Insufficient"
            output_records.append(record)
            continue
        total = sum(available.values())
        evidence = _score_evidence(row)
        highest = max(available.values())
        primary_dimensions = [dimension for dimension in DIMENSION_ORDER if available.get(dimension) == highest]
        scored_dimensions = len(available)
        minimum_scored = int(rules["priority_policy"]["minimum_scored_dimensions"])
        base_rating = _rating(total, rules)
        priority_rating, priority_reason, priority_rule_id, override_applied = _priority_decision(available, rules)
        if scored_dimensions < minimum_scored:
            evidence_status = "Insufficient evidence"
        else:
            evidence_status = "Complete" if scored_dimensions == len(DIMENSION_ORDER) else "Reduced evidence"
        detected_patterns = _detected_patterns(
            row,
            available,
            include_missing=scored_dimensions < len(DIMENSION_ORDER),
        )
        pattern = detected_patterns[0][0]
        pattern_names = [item[0] for item in detected_patterns]
        pattern_details = [
            item[0] if pd.isna(item[1]) else f"{item[0]} ({int(item[1])}/3)"
            for item in detected_patterns
        ]
        record.update(
            {
                "risk_score": total,
                "risk_rating": priority_rating,
                "base_risk_rating": base_rating,
                "risk_pattern": pattern,
                "risk_patterns": " | ".join(pattern_names),
                "risk_pattern_details": " | ".join(pattern_details),
                "risk_pattern_count": len(pattern_names),
                "scored_dimensions": scored_dimensions,
                "available_score_max": scored_dimensions * 3,
                "evidence_status": evidence_status,
                "why_primary": " ".join(evidence[d] for d in primary_dimensions if d in evidence) or "No material concern in scored dimensions.",
                "why_supporting": " ".join(evidence[d] for d in DIMENSION_ORDER if d in evidence and d not in primary_dimensions),
                "score_equation": " + ".join(f"{DIMENSION_LABELS[d]} {available[d]}" for d in DIMENSION_ORDER if d in available) + f" = {total}",
                "risk_label_rule": priority_reason,
                "priority_rule_id": priority_rule_id,
                "priority_override_applied": override_applied,
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
