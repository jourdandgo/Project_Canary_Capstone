from pathlib import Path
import os

import pandas as pd
import pytest

from canary.data import load_workbook


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[2] / "FARM HARVEST DATA.xlsx"),
    )
)


def test_real_workbook_resolves_to_unique_building_days():
    dataset = load_workbook(SOURCE)
    quality = dataset.quality

    assert quality.source_rows == 1785
    assert quality.canonical_rows == 1666
    assert quality.duplicate_keys == 119
    assert quality.duplicate_rows_consolidated == 119
    assert quality.production_conflict_keys == 0
    assert quality.passed
    assert not dataset.daily.duplicated(["cycle_id", "building_id", "age_day"]).any()


def test_environment_duplicates_are_aggregated_without_multiplying_production():
    dataset = load_workbook(SOURCE)
    row = dataset.daily.loc[
        (dataset.daily["cycle_id"] == "2026-2")
        & (dataset.daily["building_id"] == "Lags 1")
        & (dataset.daily["age_day"] == 1)
    ].iloc[0]

    assert row["source_row_count"] == 2
    assert bool(row["had_source_duplicates"])
    assert row["population"] == 11174
    assert row["mortality_daily"] == 16
    assert row["temperature_avg_c"] == pytest.approx((31.853277 + 33.122017) / 2, rel=1e-6)


def test_blank_operational_days_remain_missing_not_zero():
    dataset = load_workbook(SOURCE)
    row = dataset.daily.loc[
        (dataset.daily["cycle_id"] == "2025-2")
        & (dataset.daily["building_id"] == "Tags 1")
        & (dataset.daily["age_day"] == 26)
    ].iloc[0]

    assert pd.isna(row["mortality_daily"])
    assert pd.isna(row["feed_daily_bags"])
    assert not bool(row["mortality_recorded"])
    assert not bool(row["feed_recorded"])
    assert not bool(row["operational_recorded"])


def test_target_is_two_kilograms_after_day_35():
    dataset = load_workbook(SOURCE)
    targets = dataset.targets.set_index("age_day")

    assert targets.loc[35, "target_weight_kg"] == 2.0
    assert targets.loc[49, "target_weight_kg"] == 2.0
