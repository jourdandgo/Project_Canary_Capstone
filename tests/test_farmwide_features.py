from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from canary import (
    assert_primary_schema_has_no_identity,
    build_asof_features,
    checkpoint_status,
    load_workbook,
)
from canary.external_modeling_review import RECOVERY_COMPACT_FEATURES, RECOVERY_FEATURES


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")


def test_checkpoint_status_distinguishes_validation_scope():
    assert checkpoint_status(7) == "Validated checkpoint"
    assert checkpoint_status(10) == "Between-checkpoint estimate"
    assert checkpoint_status(4) == "Early fallback before Day 7"
    assert checkpoint_status(31) == "Late off-checkpoint estimate"
    assert checkpoint_status(None, False) == "Unavailable"


def test_primary_recovery_schemas_exclude_identity():
    assert_primary_schema_has_no_identity(RECOVERY_FEATURES)
    assert_primary_schema_has_no_identity(RECOVERY_COMPACT_FEATURES)
    with pytest.raises(AssertionError, match="identity"):
        assert_primary_schema_has_no_identity(["review_day", "is_lags_building"])


def test_asof_service_records_evidence_timestamps_and_no_future_data(dataset):
    meta = dataset.cycles.query("cycle_id == '2025-5' and building_id == 'Tags 1'").iloc[0]
    as_of = pd.Timestamp(meta["start_date"]) + pd.DateOffset(days=13)
    baseline = build_asof_features(dataset, "2025-5", "Tags 1", as_of, "recovery")
    assert baseline is not None
    assert baseline["max_source_date_used"] <= as_of
    assert baseline["max_source_day_used"] <= 14
    assert baseline["checkpoint_status"] == "Validated checkpoint"
    assert "mortality_ewma_per_1000" in baseline
    assert "population_mortality_reconciliation_gap_per_1000" in baseline

    changed_daily = dataset.daily.copy()
    future = (
        changed_daily["cycle_id"].eq("2025-5")
        & changed_daily["building_id"].eq("Tags 1")
        & changed_daily["record_date"].gt(as_of)
    )
    changed_daily.loc[future, ["mortality_daily", "population", "bodyweight_kg"]] = [9999, 1, 3.4]
    changed = build_asof_features(replace(dataset, daily=changed_daily), "2025-5", "Tags 1", as_of, "recovery")
    for field in (
        "percentage_alive",
        "mortality_recent_7d_per_1000",
        "latest_weight_kg",
        "temperature_history_mean_c",
    ):
        assert changed[field] == pytest.approx(baseline[field], nan_ok=True)
