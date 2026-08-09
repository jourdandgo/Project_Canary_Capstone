from pathlib import Path
import os

import pandas as pd
import pytest

from canary import (
    build_historical_outcomes,
    latest_cycle_id,
    load_final_weight_labels,
    load_workbook,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[2] / "FARM HARVEST DATA.xlsx"),
    )
)
PERFORMANCE = Path(
    os.getenv(
        "CANARY_TEST_PERFORMANCE_WORKBOOK",
        SOURCE.with_name("Farm Performance Summary.xlsx"),
    )
)


def test_latest_cycle_is_inferred_from_placement_dates():
    dataset = load_workbook(SOURCE)
    assert latest_cycle_id(dataset) == "2026-3"


def test_historical_outcomes_use_last_population_and_valid_final_weights():
    dataset = load_workbook(SOURCE)
    labels = load_final_weight_labels(PERFORMANCE)
    outcomes = build_historical_outcomes(dataset, labels)

    tags1 = outcomes.loc[
        (outcomes["cycle_id"] == "2025-2")
        & (outcomes["building_id"] == "Tags 1")
    ].iloc[0]
    assert tags1["completion_date"] == pd.Timestamp("2025-05-16")
    assert tags1["actual_harvest_recovery"] == pytest.approx(
        tags1["actual_ending_population"] / tags1["beginning_inventory"]
    )
    assert tags1["actual_final_average_weight_kg"] == 1.853
    assert tags1["actual_final_weight_status"] == "Recorded final average weight"


def test_suspicious_and_missing_final_weights_are_not_presented_as_actuals():
    dataset = load_workbook(SOURCE)
    labels = load_final_weight_labels(PERFORMANCE)
    outcomes = build_historical_outcomes(dataset, labels)

    suspect = outcomes.loc[
        (outcomes["cycle_id"] == "2025-4")
        & (outcomes["building_id"] == "Lags 1")
    ].iloc[0]
    missing = outcomes.loc[
        (outcomes["cycle_id"] == "2026-2")
        & (outcomes["building_id"] == "Tags 1")
    ].iloc[0]
    assert pd.isna(suspect["actual_final_average_weight_kg"])
    assert "excluded" in suspect["actual_final_weight_status"].lower()
    assert pd.isna(missing["actual_final_average_weight_kg"])
    assert missing["actual_final_weight_status"].startswith("Not available")
