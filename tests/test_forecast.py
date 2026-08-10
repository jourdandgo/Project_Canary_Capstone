from pathlib import Path
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from canary import (
    attach_forecasts,
    default_as_of_date,
    forecast_trace,
    load_workbook,
    recovery_feature_contributions,
    score_cycle_snapshot,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(SOURCE)


def test_active_buildings_receive_recovery_forecast_without_altering_risk(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    risk = score_cycle_snapshot(dataset, "2026-3", as_of)
    forecast = attach_forecasts(dataset, risk)

    pd.testing.assert_series_equal(risk["risk_score"], forecast["risk_score"], check_dtype=False)
    active = forecast.loc[forecast["state"] == "Active"]
    assert active["predicted_final_recovery"].between(0, 1).all()
    assert active["recovery_forecast_status"].eq("Forecast available").all()
    assert np.allclose(
        active["recovery_target_gap_pp"],
        (active["predicted_final_recovery"] - 0.95) * 100,
    )
    assert (active["recovery_interval_low"] <= active["predicted_final_recovery"]).all()
    assert (active["recovery_interval_high"] >= active["predicted_final_recovery"]).all()
    assert (active["predicted_final_recovery"] <= active["percentage_alive"]).all()
    assert (active["recovery_interval_high"] <= active["percentage_alive"]).all()


def test_missing_current_weight_does_not_receive_a_fake_building_projection(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    daily = dataset.daily.copy()
    mask = (daily["cycle_id"] == "2026-3") & (daily["building_id"] == "Tags 1")
    daily.loc[mask, "bodyweight_kg"] = np.nan
    daily.loc[mask, "weight_measured"] = False
    changed = replace(dataset, daily=daily)
    forecast = attach_forecasts(changed, score_cycle_snapshot(changed, "2026-3", as_of))
    active = forecast.loc[forecast["building_id"] == "Tags 1"]

    assert active["projected_day35_weight_kg"].isna().all()
    assert active["day35_weight_scope"].eq("Unavailable").all()
    assert active["day35_weight_status"].str.contains("measured weight", case=False).all()


def test_inactive_buildings_do_not_receive_forecasts(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    forecast = attach_forecasts(dataset, score_cycle_snapshot(dataset, "2026-3", as_of))
    inactive = forecast.loc[forecast["state"] == "Inactive"]

    assert inactive["predicted_final_recovery"].isna().all()
    assert inactive["recovery_forecast_status"].eq("Waiting for placement").all()


def test_incomplete_day_keeps_updating_from_latest_known_observations(dataset):
    forecast = attach_forecasts(
        dataset,
        score_cycle_snapshot(dataset, "2026-3", pd.Timestamp("2026-07-26")),
    )
    incomplete = forecast.loc[forecast["state"] == "Incomplete"]

    assert len(incomplete) == 3
    assert incomplete["predicted_final_recovery"].between(0, 1).all()
    assert incomplete["recovery_forecast_status"].eq(
        "Forecast available — latest recorded data used"
    ).all()
    assert incomplete["projected_day35_weight_kg"].notna().all()
    assert incomplete["day35_weight_scope"].eq("Building projection").all()


def test_records_ended_trace_keeps_forecasts_and_discloses_proxy(dataset):
    as_of = default_as_of_date(dataset, "2026-1")
    ended = attach_forecasts(
        dataset, score_cycle_snapshot(dataset, "2026-1", as_of)
    ).loc[lambda frame: frame["state"] == "Records ended"].iloc[0]

    trace = forecast_trace(ended)
    recovery = trace.loc[trace["Outcome"] == "Harvest survival"].iloc[0]
    assert "harvest not confirmed" in recovery["Status"]
    assert "last-recorded recovery" in recovery["Important limitation"]


def test_day14_projection_is_building_specific_when_weights_exist(dataset):
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2025-5")
        & (dataset.cycles["building_id"] == "Tags 1")
    ].iloc[0]
    as_of = pd.Timestamp(meta["start_date"]) + pd.DateOffset(days=13)
    forecast = attach_forecasts(
        dataset, score_cycle_snapshot(dataset, "2025-5", as_of)
    )
    available = forecast.loc[forecast["projected_day35_weight_kg"].notna()]

    assert len(available) >= 3
    assert available["projected_day35_weight_kg"].nunique() > 1
    assert available["day35_weight_scope"].eq("Building projection").all()
    assert available["day35_weight_status"].str.contains("Day 35 projection").all()


def test_recovery_contributions_explain_direction_without_claiming_cause(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    contributions = recovery_feature_contributions(
        dataset, "2026-3", "Tags 1", as_of
    )
    assert not contributions.empty
    assert {
        "Model input",
        "Current value",
        "Effect on raw estimate",
        "Direction",
    }.issubset(contributions.columns)
    assert contributions["Direction"].str.startswith(("Pushes", "No effect")).all()
