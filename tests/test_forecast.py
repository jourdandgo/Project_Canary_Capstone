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
    load_model_bundle,
    load_workbook,
    recovery_feature_contributions,
    score_cycle_snapshot,
)
from canary.business_value import ValueAssumptions, attach_business_value
from canary.forecast import _interpolated_remaining_loss


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
    assert active["recovery_interval_label"].eq("Typical-error reference").all()
    assert active["recovery_model_name"].eq("Trish Model 1 · Extra Trees").all()
    assert active["recovery_target_status"].isin(["Likely below", "Uncertain", "Likely meets"]).all()
    assert active["recovery_checkpoint_status"].ne("Unavailable").all()
    assert (active["predicted_final_recovery"] <= active["percentage_alive"]).all()
    assert (active["recovery_interval_high"] <= active["percentage_alive"]).all()


def test_day22_recovery_interpolates_between_day21_and_day28(dataset):
    as_of = dataset.daily.loc[
        dataset.daily["cycle_id"].eq("2026-3")
        & dataset.daily["age_day"].eq(22),
        "record_date",
    ].max()
    forecast = attach_business_value(
        attach_forecasts(dataset, score_cycle_snapshot(dataset, "2026-3", as_of)),
        ValueAssumptions(),
    )
    active = forecast.loc[forecast["state"] == "Active"].set_index("building_id")
    manifest, _ = load_model_bundle("recovery")
    if active["recovery_model_name"].eq("Trish Model 1 · Extra Trees").all():
        assert active["recovery_live_age_policy"].str.contains(
            "Day 14 Trish v18 outlook"
        ).all()
    elif manifest["model_kind"] == "formula":
        losses = manifest["additional_loss_by_age_band"]
        expected_loss = float(losses["21"]) + (float(losses["28"]) - float(losses["21"])) / 7
        assert np.allclose(
            active["recovery_expected_additional_loss_pp"].astype(float),
            expected_loss * 100,
        )
        assert active["recovery_live_age_policy"].str.contains("checkpoint").all()
    else:
        assert active["recovery_expected_additional_loss_pp"].notna().all()
        assert active["recovery_live_age_policy"].eq("Fitted model").all()
        # A fitted model should vary across multiple simultaneously active
        # buildings when their inputs differ.  At the latest source date only
        # one building can remain active, so variability is not testable.
        if len(active) > 1:
            assert active["recovery_expected_additional_loss_pp"].nunique() > 1
    assert (
        active["predicted_final_recovery"].astype(float)
        <= active["percentage_alive"].astype(float)
    ).all()
    assert (active["gross_revenue_at_risk_php"].astype(float) >= 0).all()


def test_recovery_checkpoint_values_are_preserved_and_intermediate_days_are_smooth():
    manifest, _ = load_model_bundle("recovery")
    losses = manifest["additional_loss_by_age_band"]

    for day in (7, 14, 21, 28):
        observed, _ = _interpolated_remaining_loss(day, losses)
        assert observed == pytest.approx(float(losses[str(day)]))

    daily_losses = [
        _interpolated_remaining_loss(day, losses)[0] for day in range(7, 29)
    ]
    assert np.all(np.diff(daily_losses) <= 1e-12)


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


def test_refreshed_complete_day_keeps_building_forecasts_current(dataset):
    forecast = attach_forecasts(
        dataset,
        score_cycle_snapshot(dataset, "2026-3", pd.Timestamp("2026-07-26")),
    )
    active = forecast.loc[forecast["state"] == "Active"]

    # The refreshed workbook now has complete Day 23 rows for all three Tags
    # buildings on this replay date.
    assert set(active["building_id"]) == {"Tags 1", "Tags 2", "Tags 3"}
    assert active["predicted_final_recovery"].between(0, 1).all()
    assert active["recovery_forecast_status"].eq("Forecast available").all()
    assert active["projected_day35_weight_kg"].notna().all()
    assert active["day35_weight_scope"].eq("Trish Model 3 outlook").all()
    assert active["day35_weight_model_name"].str.contains("Trish Model 3").all()


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
