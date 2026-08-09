from dataclasses import replace
from io import BytesIO
from pathlib import Path
import os

import pandas as pd
import pytest

from canary import (
    FEATURE_COLUMNS,
    build_modeling_snapshots,
    complete_cycle_ids,
    extract_feature_row,
    load_final_weight_labels,
    load_workbook,
    train_outcome_model,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[2] / "FARM HARVEST DATA.xlsx"),
    )
)
PERFORMANCE_SUMMARY = Path(
    os.getenv(
        "CANARY_TEST_PERFORMANCE_SUMMARY",
        str(Path(__file__).resolve().parents[2] / "Farm Performance Summary.xlsx"),
    )
)


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(SOURCE)


def test_only_whole_completed_cycles_are_used(dataset):
    assert complete_cycle_ids(dataset) == ["2025-2", "2025-3", "2025-4", "2025-5", "2026-1"]

    recovery = build_modeling_snapshots(dataset, "recovery")
    assert recovery["cycle_id"].nunique() == 5
    assert recovery[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 25
    assert (recovery["as_of_date"] < recovery["label_date"]).all()


def test_weight_proxy_excludes_label_day_and_later_rows(dataset):
    weight = build_modeling_snapshots(dataset, "weight")
    assert weight["cycle_id"].nunique() == 4
    assert weight[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 19
    assert (weight["as_of_date"] < weight["label_date"]).all()


def test_final_weight_summary_is_used_directly_and_suspect_rows_are_excluded(dataset):
    labels = load_final_weight_labels(PERFORMANCE_SUMMARY)
    assert not labels["cycle_id"].isin(["2026-2", "2026-3"]).any()

    weight = build_modeling_snapshots(dataset, "weight", labels)
    accepted = weight[["cycle_id", "building_id", "target"]].drop_duplicates()
    assert accepted["cycle_id"].nunique() == 5
    assert len(accepted) == 17
    assert not (
        (accepted["cycle_id"] == "2026-1")
        & accepted["building_id"].isin(["Lags 1", "Lags 2"])
    ).any()
    tags_1 = accepted.loc[
        (accepted["cycle_id"] == "2026-1") & (accepted["building_id"] == "Tags 1"),
        "target",
    ].iloc[0]
    assert tags_1 == pytest.approx(2.0476)


def test_final_weight_summary_can_be_loaded_from_cloud_upload():
    source = BytesIO(PERFORMANCE_SUMMARY.read_bytes())
    source.name = "Farm Performance Summary.xlsx"

    labels = load_final_weight_labels(source)

    assert not labels.empty
    assert set(labels["weight_label_source"]) == {"Farm Performance Summary.xlsx"}


def test_final_weight_training_uses_verified_label_version(dataset):
    labels = load_final_weight_labels(PERFORMANCE_SUMMARY)
    weight = train_outcome_model(dataset, "weight", labels)
    assert weight.manifest["model_version"] == "weight-final-0.4.0"
    assert weight.manifest["training_building_cycles"] == 17
    assert "used directly" in weight.manifest["label_definition"]
    assert weight.manifest["selected_metrics"]["mae"] < 0.10


def test_future_changes_cannot_change_an_earlier_feature_row(dataset):
    as_of = pd.Timestamp("2025-11-25")
    baseline = extract_feature_row(dataset, "2025-5", "Tags 1", as_of)

    changed_daily = dataset.daily.copy()
    future = (
        (changed_daily["cycle_id"] == "2025-5")
        & (changed_daily["building_id"] == "Tags 1")
        & (changed_daily["record_date"] > as_of)
    )
    changed_daily.loc[future, ["mortality_daily", "bodyweight_kg"]] = [9999, 9.9]
    changed = replace(dataset, daily=changed_daily)
    rescored = extract_feature_row(changed, "2025-5", "Tags 1", as_of)

    for column in FEATURE_COLUMNS:
        left, right = baseline[column], rescored[column]
        if pd.isna(left):
            assert pd.isna(right)
        else:
            assert right == pytest.approx(left)


def test_training_selects_best_validated_candidate_and_versions_artifact(dataset):
    recovery = train_outcome_model(dataset, "recovery")
    weight = train_outcome_model(dataset, "weight")

    best_macro = min(
        metrics["cycle_macro_mae"]
        for metrics in recovery.manifest["metrics"].values()
    )
    assert recovery.manifest["selected_metrics"]["cycle_macro_mae"] <= best_macro * 1.05
    assert recovery.selected_model == "ridge_no_weight"
    assert recovery.manifest["model_version"] == "recovery-0.5.0"
    assert recovery.manifest["training_snapshot_rows"] <= 25 * 5
    assert recovery.manifest["source_daily_snapshot_rows"] > recovery.manifest["training_snapshot_rows"]
    assert "ridge_no_weight" in recovery.manifest["metrics"]
    assert set(recovery.manifest["feature_columns"]).issubset(FEATURE_COLUMNS)
    assert recovery.manifest["selected_metrics"]["mae"] < 0.02
    assert recovery.manifest["day14_backtest_metrics"]["building_cycles"] == 25
    assert recovery.manifest["selection_metric"] == "cycle_macro_mae_within_5pct_then_simplest"
    assert "confusion_matrix" in recovery.manifest["selected_metrics"]
    assert len(recovery.manifest["day14_backtest"]) == 25
    assert recovery.manifest["global_feature_importance"]
    assert recovery.manifest["global_feature_importance"][0]["feature"] == "percentage_alive"
    assert sum(
        item["absolute_importance_pct"]
        for item in recovery.manifest["global_feature_importance"]
    ) == pytest.approx(100.0)
    assert {
        "cycle_id",
        "building_id",
        "as_of_date",
        "predicted",
        "actual",
        "error",
        "absolute_error",
    }.issubset(recovery.manifest["day14_backtest"][0])
    assert weight.selected_model == "historical_mean"
    assert weight.manifest["model_version"] == "weight-proxy-0.3.0"
    assert "proxy" in weight.manifest["label_definition"]
