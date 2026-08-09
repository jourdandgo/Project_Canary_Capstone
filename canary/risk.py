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
DIMENSION_ORDER = ("weight", "survival", "mortality", "peer")
DIMENSION_LABELS = {
    "weight": "Weight gap",
    "survival": "Survival path",
    "mortality": "Mortality trend",
    "peer": "Peer comparison",
}


class RiskConfigurationError(ValueError):
    """Raised when a risk-rules file is structurally unsafe to use."""


def load_risk_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        rules = json.load(stream)
    validate_risk_rules(rules)
    return rules


def validate_risk_rules(rules: dict[str, Any]) -> None:
    """Reject incomplete or internally inconsistent scoring configurations."""

    required = {
        "version",
        "approval_status",
        "survival_target",
        "rating_bands",
        "age_bands",
        "mortality_trend",
        "peer_comparison",
        "farm_wide",
    }
    missing = sorted(required - set(rules))
    if missing:
        raise RiskConfigurationError("Risk rules are missing: " + ", ".join(missing))
    if not rules["age_bands"] or not rules["rating_bands"]:
        raise RiskConfigurationError("Risk rules must include age and rating bands.")

    target_rate = float(rules["survival_target"]["final_target_rate"])
    target_day = int(rules["survival_target"]["target_day"])
    if not 0 < target_rate <= 1 or target_day < 1:
        raise RiskConfigurationError("The survival target must be a rate above 0 through 1 and a positive day.")

    def check_cutoffs(values: object, label: str) -> list[float]:
        if not isinstance(values, list) or len(values) != 3:
            raise RiskConfigurationError(f"{label} requires exactly three point cutoffs.")
        numeric = [float(value) for value in values]
        if any(value < 0 for value in numeric) or not numeric[0] < numeric[1] < numeric[2]:
            raise RiskConfigurationError(f"{label} cutoffs must be non-negative and strictly increasing.")
        return numeric

    age_bands = sorted(rules["age_bands"], key=lambda item: int(item["minimum_age"]))
    expected_minimum = 1
    for band in age_bands:
        minimum = int(band["minimum_age"])
        maximum = int(band["maximum_age"])
        if minimum != expected_minimum or maximum < minimum:
            raise RiskConfigurationError("Age bands must cover every production day without gaps or overlaps.")
        for key, label in (
            ("weight_gap_pct", "Weight-gap"),
            ("survival_gap_pp", "Survival-gap"),
            ("mortality_trend_delta_per_1000", "Mortality-trend"),
        ):
            check_cutoffs(band[key], f"{band['label']} {label}")
        expected_minimum = maximum + 1
    if age_bands[-1]["maximum_age"] < 999:
        raise RiskConfigurationError("The final age band must cover production days beyond Day 35.")

    peer = rules["peer_comparison"]
    if int(peer["minimum_comparable_buildings"]) < 3:
        raise RiskConfigurationError("Peer scoring requires at least three comparable buildings including the building reviewed.")
    if int(peer["maximum_age_difference_days"]) < 0:
        raise RiskConfigurationError("Peer age tolerance cannot be negative.")
    for key, label in (
        ("weight_gap_excess_pct", "Peer weight-gap"),
        ("survival_gap_excess_pp", "Peer survival-gap"),
        ("mortality_rate_excess_per_1000", "Peer mortality"),
    ):
        check_cutoffs(peer[key], label)

    max_score = len(DIMENSION_ORDER) * 3
    covered_scores: list[int] = []
    for band in rules["rating_bands"]:
        minimum = int(band["minimum"])
        maximum = int(band["maximum"])
        if not str(band["label"]).strip() or maximum < minimum:
            raise RiskConfigurationError("Each rating band requires a label and valid score range.")
        covered_scores.extend(range(minimum, maximum + 1))
    if sorted(covered_scores) != list(range(0, max_score + 1)):
        raise RiskConfigurationError(f"Rating bands must cover every score from 0 through {max_score} exactly once.")


def save_risk_rules(
    rules: dict[str, Any], path: str | Path = DEFAULT_RULES_PATH
) -> None:
    """Validate and atomically save owner-reviewed scoring thresholds."""

    validate_risk_rules(rules)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _age_band(rules: dict[str, Any], age: int) -> dict[str, Any]:
    for band in rules["age_bands"]:
        if int(band["minimum_age"]) <= age <= int(band["maximum_age"]):
            return band
    raise RiskConfigurationError(f"No risk threshold band covers production Day {age}.")


def _score_threshold(value: object, cutoffs: list[float]) -> object:
    if pd.isna(value):
        return pd.NA
    numeric = float(value)
    if len(cutoffs) != 3 or sorted(cutoffs) != cutoffs:
        raise RiskConfigurationError("Each dimension requires three ascending cutoffs.")
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


def _expected_alive(age: int, rules: dict[str, Any]) -> float:
    final_target = float(rules["survival_target"]["final_target_rate"])
    target_day = int(rules["survival_target"]["target_day"])
    elapsed = min(max(age, 0), target_day) / target_day
    return 1.0 - ((1.0 - final_target) * elapsed)


def _mortality_signal(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of: pd.Timestamp,
    rules: dict[str, Any],
) -> dict[str, object]:
    observations = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & (dataset.daily["record_date"] <= as_of)
        & dataset.daily["mortality_recorded"]
    ].sort_values("age_day")
    if observations.empty:
        return {"recent": pd.NA, "baseline": pd.NA, "delta": pd.NA, "label": "Not scored"}

    recent_days = int(rules["mortality_trend"]["recent_days"])
    baseline_days = int(rules["mortality_trend"]["baseline_days"])
    min_baseline = int(rules["mortality_trend"]["minimum_baseline_observations"])
    latest_age = int(observations["age_day"].max())
    recent = observations.loc[observations["age_day"].between(latest_age - recent_days + 1, latest_age)]
    baseline = observations.loc[
        observations["age_day"].between(latest_age - recent_days - baseline_days + 1, latest_age - recent_days)
    ]
    expected_recent_ages = set(range(latest_age - recent_days + 1, latest_age + 1))
    if set(recent["age_day"].astype(int)) != expected_recent_ages or len(baseline) < min_baseline:
        return {"recent": pd.NA, "baseline": pd.NA, "delta": pd.NA, "label": "Not scored"}

    recent_rates = pd.to_numeric(recent["mortality_daily"], errors="coerce") / pd.to_numeric(
        recent["beginning_inventory"], errors="coerce"
    ) * 1000
    baseline_rates = pd.to_numeric(baseline["mortality_daily"], errors="coerce") / pd.to_numeric(
        baseline["beginning_inventory"], errors="coerce"
    ) * 1000
    recent_rate = float(recent_rates.mean())
    baseline_rate = float(baseline_rates.mean())
    delta = recent_rate - baseline_rate
    if delta <= 0:
        label = "Stable or improving"
    elif recent_rate >= max(baseline_rate * 2, baseline_rate + 0.5):
        label = "Spiking"
    elif delta > 0.2:
        label = "Worsening"
    else:
        label = "Increasing"
    return {"recent": recent_rate, "baseline": baseline_rate, "delta": max(0.0, delta), "label": label}


def _base_signals(
    dataset: CanaryDataset,
    snapshot: pd.DataFrame,
    cycle_id: str,
    as_of: pd.Timestamp,
    rules: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in snapshot.iterrows():
        record: dict[str, object] = row.to_dict()
        record.update(
            {
                "risk_age_band": pd.NA,
                "weight_gap_pct": pd.NA,
                "weight_score": pd.NA,
                "expected_alive_rate": pd.NA,
                "survival_gap_pp": pd.NA,
                "survival_score": pd.NA,
                "mortality_recent_per_1000": pd.NA,
                "mortality_baseline_per_1000": pd.NA,
                "mortality_trend_delta_per_1000": pd.NA,
                "mortality_trend_label": "Not scored",
                "mortality_score": pd.NA,
            }
        )
        if row["state"] == "Inactive" or pd.isna(row["cycle_day"]):
            records.append(record)
            continue

        age = int(row["cycle_day"])
        band = _age_band(rules, age)
        observed_weight = row["latest_weight_kg"]
        weight_target = row["weight_target_at_measurement_kg"]
        if pd.notna(observed_weight) and pd.notna(weight_target) and float(weight_target) > 0:
            weight_gap_pct = max(0.0, (float(weight_target) - float(observed_weight)) / float(weight_target) * 100)
        else:
            weight_gap_pct = pd.NA

        observation_age = row["latest_operational_day"]
        if pd.notna(row["percentage_alive"]) and pd.notna(observation_age):
            expected_alive = _expected_alive(int(observation_age), rules)
            survival_gap_pp = max(0.0, (expected_alive - float(row["percentage_alive"])) * 100)
        else:
            expected_alive = pd.NA
            survival_gap_pp = pd.NA

        mortality = _mortality_signal(dataset, cycle_id, str(row["building_id"]), as_of, rules)
        record.update(
            {
                "risk_age_band": band["label"],
                "weight_gap_pct": weight_gap_pct,
                "weight_score": _score_threshold(weight_gap_pct, band["weight_gap_pct"]),
                "expected_alive_rate": expected_alive,
                "survival_gap_pp": survival_gap_pp,
                "survival_score": _score_threshold(survival_gap_pp, band["survival_gap_pp"]),
                "mortality_recent_per_1000": mortality["recent"],
                "mortality_baseline_per_1000": mortality["baseline"],
                "mortality_trend_delta_per_1000": mortality["delta"],
                "mortality_trend_label": mortality["label"],
                "mortality_score": _score_threshold(
                    mortality["delta"], band["mortality_trend_delta_per_1000"]
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _apply_peer_scores(frame: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    peer_rules = rules["peer_comparison"]
    min_group = int(peer_rules["minimum_comparable_buildings"])
    age_tolerance = int(peer_rules["maximum_age_difference_days"])
    frame["peer_score"] = pd.NA
    frame["peer_count"] = 0
    frame["peer_driver"] = "Not scored"
    frame["peer_evidence"] = "Not scored: fewer than three comparable active buildings."

    eligible = frame.loc[
        frame["state"].isin(["Active", "Incomplete", "Records ended"])
    ].copy()
    for index, row in eligible.iterrows():
        if pd.isna(row["latest_operational_day"]):
            continue
        comparable = eligible.loc[
            (eligible["latest_operational_day"] - float(row["latest_operational_day"])).abs()
            <= age_tolerance
        ]
        if len(comparable) < min_group:
            frame.at[index, "peer_evidence"] = (
                f"Not scored: {len(comparable)} comparable building(s); {min_group} required."
            )
            continue
        peers = comparable.drop(index=index)
        component_scores: list[tuple[int, str, str]] = []
        component_specs = [
            ("weight_gap_pct", "weight_gap_excess_pct", "weight gap", "%"),
            ("survival_gap_pp", "survival_gap_excess_pp", "survival gap", " pp"),
            ("mortality_recent_per_1000", "mortality_rate_excess_per_1000", "recent mortality", "/1,000"),
        ]
        for metric, cutoff_key, label, unit in component_specs:
            peer_values = pd.to_numeric(peers[metric], errors="coerce").dropna()
            if pd.isna(row.get(metric)) or len(peer_values) < min_group - 1:
                continue
            peer_median = float(peer_values.median())
            excess = max(0.0, float(row[metric]) - peer_median)
            score = int(_score_threshold(excess, peer_rules[cutoff_key]))
            component_scores.append(
                (score, label, f"{label} is {excess:.2f}{unit} worse than the peer median")
            )
        frame.at[index, "peer_count"] = len(peers)
        if not component_scores:
            frame.at[index, "peer_evidence"] = "Not scored: comparable peer measures are unavailable."
            continue
        component_scores.sort(key=lambda item: (-item[0], item[1]))
        score, driver, evidence = component_scores[0]
        frame.at[index, "peer_score"] = score
        frame.at[index, "peer_driver"] = driver.title()
        frame.at[index, "peer_evidence"] = f"{evidence} across {len(peers)} peer(s)."
    return frame


def _score_evidence(row: pd.Series) -> dict[str, str]:
    evidence: dict[str, str] = {}
    if pd.notna(row.get("weight_score")):
        evidence["weight"] = (
            f"Weight gap: {float(row['weight_gap_pct']):.1f}% below the Day {int(row['weight_measurement_day'])} "
            f"target ({float(row['latest_weight_kg']):.3f} vs {float(row['weight_target_at_measurement_kg']):.3f} kg; "
            f"score {int(row['weight_score'])}; {row['weight_freshness']})."
        )
    if pd.notna(row.get("survival_score")):
        evidence["survival"] = (
            f"Survival path: {float(row['percentage_alive']):.1%} alive versus "
            f"{float(row['expected_alive_rate']):.1%} expected on recorded Day {int(row['latest_operational_day'])} "
            f"({float(row['survival_gap_pp']):.2f} pp gap; score {int(row['survival_score'])}; "
            f"{row['data_freshness']})."
        )
    if pd.notna(row.get("mortality_score")):
        evidence["mortality"] = (
            f"Mortality trend: {row['mortality_trend_label']}; recent rate "
            f"{float(row['mortality_recent_per_1000']):.2f}/1,000 versus "
            f"{float(row['mortality_baseline_per_1000']):.2f}/1,000 baseline "
            f"(score {int(row['mortality_score'])})."
        )
    if pd.notna(row.get("peer_score")):
        evidence["peer"] = f"Peer comparison: {row['peer_evidence']} (score {int(row['peer_score'])})."
    return evidence


def _threshold_description(cutoffs: list[float], unit: str) -> str:
    first, second, third = cutoffs
    return (
        f"0: <= {first:g}{unit}; 1: > {first:g} to {second:g}{unit}; "
        f"2: > {second:g} to {third:g}{unit}; 3: > {third:g}{unit}"
    )


def build_dimension_trace(
    scored_row: pd.Series | dict[str, object],
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return an audit-friendly calculation trace for one scored building."""

    rules = rules or load_risk_rules()
    row = pd.Series(scored_row)
    if pd.isna(row.get("cycle_day")):
        return pd.DataFrame(
            columns=["Dimension", "Raw observations", "Calculation", "Applied thresholds", "Score", "Data status"]
        )
    band = _age_band(rules, int(row["cycle_day"]))
    peer_driver = str(row.get("peer_driver", ""))
    if peer_driver == "Weight Gap":
        peer_cutoffs = rules["peer_comparison"]["weight_gap_excess_pct"]
        peer_unit = "%"
    elif peer_driver == "Survival Gap":
        peer_cutoffs = rules["peer_comparison"]["survival_gap_excess_pp"]
        peer_unit = " pp"
    else:
        peer_cutoffs = rules["peer_comparison"]["mortality_rate_excess_per_1000"]
        peer_unit = "/1,000"

    trace = [
        {
            "Dimension": "Weight gap",
            "Raw observations": (
                "Unavailable"
                if pd.isna(row.get("latest_weight_kg"))
                else f"Observed {float(row['latest_weight_kg']):.3f} kg on Day {int(row['weight_measurement_day'])}; target {float(row['weight_target_at_measurement_kg']):.3f} kg"
            ),
            "Calculation": (
                "Not scored"
                if pd.isna(row.get("weight_gap_pct"))
                else f"(target - observed) / target = {float(row['weight_gap_pct']):.2f}% shortfall"
            ),
            "Applied thresholds": _threshold_description(band["weight_gap_pct"], "%"),
            "Score": row.get("weight_score", pd.NA),
            "Data status": row.get("weight_freshness", "Unavailable"),
        },
        {
            "Dimension": "Survival path",
            "Raw observations": (
                "Unavailable"
                if pd.isna(row.get("percentage_alive"))
                else f"{float(row['percentage_alive']):.2%} alive; {float(row['expected_alive_rate']):.2%} expected on recorded Day {int(row['latest_operational_day'])}"
            ),
            "Calculation": (
                "Not scored"
                if pd.isna(row.get("survival_gap_pp"))
                else f"max(expected - actual, 0) = {float(row['survival_gap_pp']):.2f} pp"
            ),
            "Applied thresholds": _threshold_description(band["survival_gap_pp"], " pp"),
            "Score": row.get("survival_score", pd.NA),
            "Data status": row.get("data_freshness", "Unavailable"),
        },
        {
            "Dimension": "Mortality trend",
            "Raw observations": (
                "Unavailable"
                if pd.isna(row.get("mortality_recent_per_1000"))
                else f"Recent {float(row['mortality_recent_per_1000']):.2f}/1,000; baseline {float(row['mortality_baseline_per_1000']):.2f}/1,000"
            ),
            "Calculation": (
                "Not scored"
                if pd.isna(row.get("mortality_trend_delta_per_1000"))
                else f"max(recent - baseline, 0) = {float(row['mortality_trend_delta_per_1000']):.2f}/1,000 ({row['mortality_trend_label']})"
            ),
            "Applied thresholds": _threshold_description(band["mortality_trend_delta_per_1000"], "/1,000"),
            "Score": row.get("mortality_score", pd.NA),
            "Data status": row.get("data_freshness", "Unavailable"),
        },
        {
            "Dimension": "Peer comparison",
            "Raw observations": str(row.get("peer_evidence", "Unavailable")),
            "Calculation": "Worst available age-normalized excess versus peer median",
            "Applied thresholds": _threshold_description(peer_cutoffs, peer_unit),
            "Score": row.get("peer_score", pd.NA),
            "Data status": f"{int(row.get('peer_count', 0))} peer(s)",
        },
    ]
    result = pd.DataFrame(trace)
    result["Score"] = result["Score"].astype("Int64")
    return result


def score_cycle_snapshot(
    dataset: CanaryDataset,
    cycle_id: str,
    as_of: date | pd.Timestamp,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach reproducible operational-risk outputs to a six-building snapshot."""

    rules = rules or load_risk_rules()
    as_of_ts = pd.Timestamp(as_of).normalize()
    frame = _base_signals(
        dataset, build_cycle_snapshot(dataset, cycle_id, as_of_ts), cycle_id, as_of_ts, rules
    )
    frame = _apply_peer_scores(frame, rules)

    scored_columns = [f"{dimension}_score" for dimension in DIMENSION_ORDER]
    eligible_mask = frame["state"].isin(["Active", "Incomplete", "Records ended"])
    core_scores = frame.loc[
        eligible_mask, ["weight_score", "survival_score", "mortality_score"]
    ].astype("Float64")
    elevated_core = core_scores.fillna(0.0).sum(axis=1) >= int(
        rules["farm_wide"]["minimum_core_score"]
    )
    farm_rules = rules["farm_wide"]
    farm_wide = (
        len(elevated_core) >= int(farm_rules["minimum_scorable_buildings"])
        and float(elevated_core.mean()) >= float(farm_rules["minimum_elevated_share"])
    )

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
        primary_dimensions = [d for d in DIMENSION_ORDER if available.get(d) == highest]
        primary_text = " ".join(evidence[d] for d in primary_dimensions if d in evidence)
        supporting_text = " ".join(
            evidence[d] for d in DIMENSION_ORDER if d in evidence and d not in primary_dimensions
        )

        weight_elevated = available.get("weight", 0) > 0
        survival_elevated = available.get("survival", 0) > 0 or available.get("mortality", 0) > 0
        peer_elevated = available.get("peer", 0) > 0
        if farm_wide and (weight_elevated or survival_elevated):
            pattern = "Farm-Wide Drift"
        elif peer_elevated:
            pattern = "Localized Building Drift"
        elif weight_elevated and survival_elevated:
            pattern = "Growth + Survival Drift"
        elif survival_elevated:
            pattern = "Survival Concern Only"
        elif weight_elevated:
            pattern = "Weight Lag Only"
        else:
            pattern = "No Material Drift"

        record.update(
            {
                "risk_score": total,
                "risk_rating": _rating(total, rules),
                "risk_pattern": pattern,
                "scored_dimensions": len(available),
                "evidence_status": "Complete" if len(available) == 4 else "Reduced evidence",
                "why_primary": primary_text or "No material drift detected in scored dimensions.",
                "why_supporting": supporting_text,
                "score_equation": " + ".join(
                    f"{DIMENSION_LABELS[dimension]} {available[dimension]}"
                    for dimension in DIMENSION_ORDER
                    if dimension in available
                )
                + f" = {total}",
                "risk_label_rule": _rating_rule(total, rules),
                "identified_problem": pattern,
                "recommended_action": "Pending Doc Raymond-approved action mapping; no action has been inferred.",
                "recommendation_rule_id": "Pending Sprint 4",
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
    """Build as-of-safe daily risk history for one building."""

    rules = rules or load_risk_rules()
    minimum, maximum = cycle_date_bounds(dataset, cycle_id)
    end = min(pd.Timestamp(through).date(), maximum)
    records: list[dict[str, object]] = []
    for current in pd.date_range(minimum, end, freq="D"):
        snapshot = score_cycle_snapshot(dataset, cycle_id, current, rules)
        match = snapshot.loc[snapshot["building_id"] == building_id]
        if not match.empty:
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
                    "survival_score": row.get("survival_score", pd.NA),
                    "mortality_score": row.get("mortality_score", pd.NA),
                    "peer_score": row.get("peer_score", pd.NA),
                    "evidence_status": row["evidence_status"],
                }
            )
    return pd.DataFrame(records)
