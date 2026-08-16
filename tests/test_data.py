from pathlib import Path
import os

import pandas as pd
import pytest

from canary.data import load_workbook


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_real_workbook_resolves_to_unique_building_days():
    dataset = load_workbook(SOURCE)
    quality = dataset.quality

    # The refreshed farm workbook is already consolidated to one row per
    # building-day.  The earlier workbook had 1,785 rows and 119 Zone A/B
    # duplicates; those source duplicates are no longer present in this file.
    assert quality.source_rows == 1624
    assert quality.canonical_rows == 1624
    assert quality.duplicate_keys == 0
    assert quality.duplicate_rows_consolidated == 0
    assert quality.zone_aggregated_days == 0
    assert quality.maximum_environment_sections == 1
    assert quality.production_conflict_keys == 0
    assert quality.passed
    assert not dataset.daily.duplicated(["cycle_id", "building_id", "age_day"]).any()


def test_environment_is_already_one_row_per_building_day():
    dataset = load_workbook(SOURCE)
    row = dataset.daily.loc[
        (dataset.daily["cycle_id"] == "2026-2")
        & (dataset.daily["building_id"] == "Lags 1")
        & (dataset.daily["age_day"] == 1)
    ].iloc[0]

    assert row["source_row_count"] == 1
    assert not bool(row["had_source_duplicates"])
    assert row["population"] == 11174
    assert row["mortality_daily"] == 16
    assert row["environment_section_count"] == 1
    assert not bool(row["zone_aggregated"])


def test_blank_mortality_remains_missing_not_zero():
    dataset = load_workbook(SOURCE)
    row = dataset.daily.loc[
        (dataset.daily["cycle_id"] == "2026-1")
        & (dataset.daily["building_id"] == "Lags 3")
        & (dataset.daily["age_day"] == 44)
    ].iloc[0]

    assert pd.isna(row["mortality_daily"])
    assert not bool(row["mortality_recorded"])
    assert row["feed_daily_bags"] == 19
    assert bool(row["feed_recorded"])
    assert bool(row["operational_recorded"])
    assert not bool(row["daily_complete"])


def test_revised_targets_are_smoothed_and_hold_at_1_8kg_after_day_35():
    dataset = load_workbook(SOURCE)
    targets = dataset.targets.set_index("age_day")

    assert targets.loc[7, "target_weight_kg"] == 0.170
    assert targets.loc[14, "target_weight_kg"] == 0.380
    assert targets.loc[21, "target_weight_kg"] == 0.800
    assert targets.loc[28, "target_weight_kg"] == 1.200
    assert targets.loc[35, "target_weight_kg"] == 1.800
    assert targets.loc[49, "target_weight_kg"] == 1.800
    assert targets.loc[13, "target_weight_scaled_g"] < targets.loc[14, "target_weight_scaled_g"]
    assert targets.loc[29, "target_weight_scaled_g"] == 1284
