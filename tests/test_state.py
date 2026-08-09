from pathlib import Path
import os

import pandas as pd
import pytest

from canary.data import load_workbook
from canary.state import (
    CANONICAL_BUILDINGS,
    build_cycle_snapshot,
    cycle_date_bounds,
    default_as_of_date,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[2] / "FARM HARVEST DATA.xlsx"),
    )
)


def test_snapshot_always_contains_the_six_physical_buildings():
    dataset = load_workbook(SOURCE)
    minimum, _ = cycle_date_bounds(dataset, "2025-2")
    snapshot = build_cycle_snapshot(dataset, "2025-2", minimum)

    assert snapshot["building_id"].tolist() == list(CANONICAL_BUILDINGS)
    assert len(snapshot) == 6
    assert snapshot["building_id"].is_unique


def test_default_date_uses_latest_complete_operation_not_planned_end():
    dataset = load_workbook(SOURCE)
    _, maximum = cycle_date_bounds(dataset, "2026-3")
    suggested = default_as_of_date(dataset, "2026-3")
    expected = dataset.daily.loc[
        (dataset.daily["cycle_id"] == "2026-3") & dataset.daily["daily_complete"],
        "record_date",
    ].max().date()

    assert suggested == expected
    assert suggested < maximum
    assert (build_cycle_snapshot(dataset, "2026-3", suggested)["state"] == "Active").any()


def test_day_14_weight_is_observed_and_fresh():
    dataset = load_workbook(SOURCE)
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2025-5")
        & (dataset.cycles["building_id"] == "Tags 1")
    ].iloc[0]
    as_of = pd.Timestamp(meta["start_date"]) + pd.DateOffset(days=13)
    snapshot = build_cycle_snapshot(dataset, "2025-5", as_of)
    tags1 = snapshot.loc[snapshot["building_id"] == "Tags 1"].iloc[0]

    assert tags1["state"] == "Active"
    assert tags1["cycle_day"] == 14
    assert tags1["weight_measurement_day"] == 14
    assert tags1["weight_staleness_days"] == 0
    assert tags1["weight_freshness"] == "Current"
    assert tags1["latest_weight_kg"] == 0.235
    assert tags1["weight_target_at_measurement_kg"] == 0.4


def test_padded_day_is_incomplete_and_uses_last_observed_day():
    dataset = load_workbook(SOURCE)
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2025-2")
        & (dataset.cycles["building_id"] == "Tags 1")
    ].iloc[0]
    as_of = pd.Timestamp(meta["start_date"]) + pd.DateOffset(days=29)
    snapshot = build_cycle_snapshot(dataset, "2025-2", as_of)
    tags1 = snapshot.loc[snapshot["building_id"] == "Tags 1"].iloc[0]

    assert tags1["state"] == "Incomplete"
    assert tags1["cycle_day"] == 30
    assert tags1["latest_operational_day"] == 25
    assert tags1["data_staleness_days"] == 5
    assert tags1["latest_population"] == 6411
    assert tags1["percentage_alive"] == pytest.approx(6411 / 6800)


def test_end_date_is_last_recorded_state_not_assumed_harvest():
    dataset = load_workbook(SOURCE)
    meta = dataset.cycles.loc[
        (dataset.cycles["cycle_id"] == "2025-5")
        & (dataset.cycles["building_id"] == "Tags 1")
    ].iloc[0]
    snapshot = build_cycle_snapshot(dataset, "2025-5", meta["end_date"])
    tags1 = snapshot.loc[snapshot["building_id"] == "Tags 1"].iloc[0]

    assert tags1["state"] == "Records ended"
    assert tags1["last_recorded_population"] == 6768
    assert tags1["latest_population"] == 6768
    assert tags1["last_recorded_recovery_rate"] == pytest.approx(6768 / 7104)
    assert "harvest completion is not confirmed" in tags1["status_note"]
