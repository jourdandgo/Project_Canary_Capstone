from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import os

import pandas as pd
import pytest

from canary import (
    RiskConfigurationError,
    build_dimension_trace,
    build_risk_history,
    default_as_of_date,
    load_risk_rules,
    load_workbook,
    score_cycle_snapshot,
    save_risk_rules,
    validate_risk_rules,
)
from canary.risk import _rating


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(SOURCE)


def test_rating_mapping_matches_the_approved_prd_ranges():
    rules = load_risk_rules()
    expected = {
        0: "Low",
        1: "Low",
        2: "Medium",
        3: "Medium",
        4: "High",
        5: "High",
        6: "Critical",
        12: "Critical",
    }
    assert {score: _rating(score, rules) for score in expected} == expected


def test_day_14_score_reconciles_to_the_four_dimensions(dataset):
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2025-5")
        & (dataset.cycles["building_id"] == "Tags 1")
    ].iloc[0]
    as_of = pd.Timestamp(meta["start_date"]) + pd.DateOffset(days=13)
    row = score_cycle_snapshot(dataset, "2025-5", as_of).query("building_id == 'Tags 1'").iloc[0]

    assert row["weight_gap_pct"] == pytest.approx((0.380 - 0.235) / 0.380 * 100)
    assert [row["weight_score"], row["population_loss_score"], row["daily_mortality_score"], row["environment_score"]] == [3, 0, 0, 3]
    assert row["risk_score"] == 6
    assert row["risk_rating"] == "Critical"
    assert row["risk_score"] == sum(
        row[column] for column in ["weight_score", "population_loss_score", "daily_mortality_score", "environment_score"]
    )
    assert "0.235 vs 0.380 kg" in row["why_primary"]
    assert row["score_equation"] == (
        "Weight gap 3 + Population loss 0 + Daily mortality 0 + Environmental conditions 3 = 6"
    )
    assert row["risk_label_rule"] == "Score 6-12 => Critical"
    assert row["identified_problem"] == row["risk_pattern"]
    assert row["recommendation_rule_id"] == "Not applicable"

    trace = build_dimension_trace(row)
    assert trace["Dimension"].tolist() == [
        "Weight gap",
        "Population loss",
        "Daily mortality",
        "Environmental conditions",
    ]
    assert trace["Score"].sum() == row["risk_score"]
    assert trace["Applied thresholds"].str.len().gt(0).all()
    assert trace["Calculation"].str.len().gt(0).all()


def test_future_rows_do_not_change_an_earlier_as_of_score(dataset):
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2025-5")
        & (dataset.cycles["building_id"] == "Tags 1")
    ].iloc[0]
    as_of = pd.Timestamp(meta["start_date"]) + pd.DateOffset(days=13)
    baseline = score_cycle_snapshot(dataset, "2025-5", as_of)

    changed_daily = dataset.daily.copy()
    future = (
        (changed_daily["cycle_id"] == "2025-5")
        & (changed_daily["building_id"] == "Tags 1")
        & (changed_daily["record_date"] > as_of)
    )
    changed_daily.loc[future, "mortality_daily"] = 9999
    changed = replace(dataset, daily=changed_daily)
    rescored = score_cycle_snapshot(changed, "2025-5", as_of)

    columns = ["risk_score", "weight_score", "population_loss_score", "daily_mortality_score", "environment_score"]
    pd.testing.assert_series_equal(
        baseline.query("building_id == 'Tags 1'").iloc[0][columns],
        rescored.query("building_id == 'Tags 1'").iloc[0][columns],
    )


def test_missing_weight_is_not_treated_as_zero_risk(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    daily = dataset.daily.copy()
    mask = (daily["cycle_id"] == "2026-3") & (daily["building_id"] == "Tags 1")
    daily.loc[mask, "bodyweight_kg"] = pd.NA
    daily.loc[mask, "weight_measured"] = False
    row = score_cycle_snapshot(
        replace(dataset, daily=daily), "2026-3", as_of
    ).query("building_id == 'Tags 1'").iloc[0]

    assert pd.isna(row["weight_score"])
    assert row["evidence_status"] == "Reduced evidence"
    available = [row[column] for column in ["population_loss_score", "daily_mortality_score", "environment_score"]]
    assert row["risk_score"] == sum(value for value in available if pd.notna(value))
    assert row["cycle_day"] == 35
    assert row["risk_rating"] != "Not rated"


def test_stale_environment_is_explained_and_not_scored_as_safe(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    row = score_cycle_snapshot(dataset, "2026-3", as_of).query("building_id == 'Tags 1'").iloc[0]

    assert row["cycle_day"] == 35
    assert row["environment_measurement_day"] == 17
    assert row["environment_staleness_days"] == 18
    assert pd.isna(row["environment_score"])
    assert row["environment_status"].startswith("Stale")
    assert pd.notna(row["environment_last_temperature_range_c"])

    trace = build_dimension_trace(row)
    environment = trace.loc[trace["Dimension"] == "Environmental conditions"].iloc[0]
    assert "last average temperature" in environment["Raw observations"]
    assert "maximum allowed is 2 day(s)" in environment["Data status"]


def test_current_environment_is_scored_with_freshness(dataset):
    cycle_start = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2026-3")
        & (dataset.cycles["building_id"] == "Tags 1"),
        "start_date",
    ].iloc[0]
    as_of = pd.Timestamp(cycle_start) + pd.DateOffset(days=16)
    row = score_cycle_snapshot(dataset, "2026-3", as_of).query("building_id == 'Tags 1'").iloc[0]

    assert row["cycle_day"] == 17
    assert row["environment_score"] == 3
    assert row["environment_status"].startswith("Current")


def test_risk_history_stops_at_selected_date_and_stays_as_of_safe(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    history = build_risk_history(dataset, "2026-3", "Tags 1", as_of)

    assert history["record_date"].max().date() == as_of
    assert history["cycle_day"].max() == 35
    assert history.loc[history["risk_score"].notna(), "risk_score"].between(0, 12).all()


def test_snapshot_before_later_building_placements_does_not_fail(dataset):
    cycle = "2026-2"
    first_date = dataset.cycles.loc[dataset.cycles["cycle_id"] == cycle, "start_date"].min()
    snapshot = score_cycle_snapshot(dataset, cycle, first_date)

    assert len(snapshot) == 6
    assert snapshot.loc[snapshot["state"] == "Inactive", "risk_score"].isna().all()


def test_risk_rule_validation_rejects_non_ascending_cutoffs():
    rules = deepcopy(load_risk_rules())
    rules["dimension_cutoffs"]["weight_gap_pct"] = [5.0, 30.0, 15.0]

    with pytest.raises(RiskConfigurationError, match="strictly increasing"):
        validate_risk_rules(rules)


def test_risk_rules_save_round_trip(tmp_path):
    rules = deepcopy(load_risk_rules())
    rules["version"] = "risk-rules-test"
    destination = tmp_path / "risk_rules.json"

    save_risk_rules(rules, destination)

    assert load_risk_rules(destination)["version"] == "risk-rules-test"
