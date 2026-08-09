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
        str(Path(__file__).resolve().parents[2] / "FARM HARVEST DATA.xlsx"),
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

    assert row["weight_gap_pct"] == pytest.approx((0.400 - 0.235) / 0.400 * 100)
    assert [row["weight_score"], row["survival_score"], row["mortality_score"], row["peer_score"]] == [3, 0, 0, 2]
    assert row["risk_score"] == 5
    assert row["risk_rating"] == "High"
    assert row["risk_score"] == sum(
        row[column] for column in ["weight_score", "survival_score", "mortality_score", "peer_score"]
    )
    assert "0.235 vs 0.400 kg" in row["why_primary"]
    assert row["score_equation"] == (
        "Weight gap 3 + Survival path 0 + Mortality trend 0 + Peer comparison 2 = 5"
    )
    assert row["risk_label_rule"] == "Score 4-5 => High"
    assert row["identified_problem"] == row["risk_pattern"]
    assert row["recommendation_rule_id"] == "Pending Sprint 4"

    trace = build_dimension_trace(row)
    assert trace["Dimension"].tolist() == [
        "Weight gap",
        "Survival path",
        "Mortality trend",
        "Peer comparison",
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

    columns = ["risk_score", "weight_score", "survival_score", "mortality_score", "peer_score"]
    pd.testing.assert_series_equal(
        baseline.query("building_id == 'Tags 1'").iloc[0][columns],
        rescored.query("building_id == 'Tags 1'").iloc[0][columns],
    )


def test_missing_weight_is_not_treated_as_zero_risk(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    row = score_cycle_snapshot(dataset, "2026-3", as_of).query("building_id == 'Tags 1'").iloc[0]

    assert pd.isna(row["weight_score"])
    assert row["evidence_status"] == "Reduced evidence"
    available = [row[column] for column in ["survival_score", "mortality_score", "peer_score"]]
    assert row["risk_score"] == sum(value for value in available if pd.notna(value))
    assert row["cycle_day"] == 22
    assert row["risk_rating"] != "Not rated"


def test_risk_history_stops_at_selected_date_and_stays_as_of_safe(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    history = build_risk_history(dataset, "2026-3", "Tags 1", as_of)

    assert history["record_date"].max().date() == as_of
    assert history["cycle_day"].max() == 22
    assert history.loc[history["risk_score"].notna(), "risk_score"].between(0, 12).all()


def test_snapshot_before_later_building_placements_does_not_fail(dataset):
    cycle = "2026-2"
    first_date = dataset.cycles.loc[dataset.cycles["cycle_id"] == cycle, "start_date"].min()
    snapshot = score_cycle_snapshot(dataset, cycle, first_date)

    assert len(snapshot) == 6
    assert snapshot.loc[snapshot["state"] == "Inactive", "risk_score"].isna().all()


def test_risk_rule_validation_rejects_non_ascending_cutoffs():
    rules = deepcopy(load_risk_rules())
    rules["age_bands"][0]["weight_gap_pct"] = [5.0, 30.0, 15.0]

    with pytest.raises(RiskConfigurationError, match="strictly increasing"):
        validate_risk_rules(rules)


def test_risk_rules_save_round_trip(tmp_path):
    rules = deepcopy(load_risk_rules())
    rules["version"] = "risk-rules-test"
    destination = tmp_path / "risk_rules.json"

    save_risk_rules(rules, destination)

    assert load_risk_rules(destination)["version"] == "risk-rules-test"
