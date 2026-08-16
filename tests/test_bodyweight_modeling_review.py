from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from canary.bodyweight_modeling_review import (
    AUDIT_CYCLE,
    CANDIDATES,
    CHECKPOINTS,
    _predict_bundle,
    _fit_entry,
    _predict_entry,
    build_snapshots,
    evaluate_candidate,
    feature_columns,
)
from canary.data import load_workbook


ROOT = Path(__file__).resolve().parents[1]


def test_bodyweight_snapshots_preserve_outcome_grain_and_timing() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    snapshots = build_snapshots(dataset)
    development = snapshots.query("role == 'development'")
    audit = snapshots.query("role == 'later_cycle_audit'")
    assert dataset.quality.canonical_rows == 1624
    assert len(development) == 124
    assert development[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 31
    assert audit[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 3
    assert set(audit["cycle_id"]) == {AUDIT_CYCLE}
    assert set(development["review_day"]) == set(CHECKPOINTS)
    assert snapshots["max_source_day_used"].le(snapshots["review_day"]).all()
    for checkpoint in CHECKPOINTS:
        future_columns = [f"weight_day{day}_g" for day in CHECKPOINTS if day > checkpoint]
        assert not snapshots.loc[snapshots["review_day"].eq(checkpoint), future_columns].notna().any().any()


def test_trajectory_pls_oof_predictions_are_deterministic() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    development = build_snapshots(dataset).query("role == 'development'").reset_index(drop=True)
    candidate = next(item for item in CANDIDATES if item.name == "direct_trajectory_pls")
    first, _ = evaluate_candidate(development, candidate)
    second, _ = evaluate_candidate(development, candidate)
    assert np.allclose(first["predicted_g"], second["predicted_g"])


def test_poultry_features_are_timing_safe_and_include_target_environment_signals() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    snapshots = build_snapshots(dataset)
    expected = {
        "current_gap_to_target_g",
        "temperature_target_abs_error_mean_c",
        "humidity_target_abs_error_mean_pct",
        "heat_excess_degree_days",
        "thi_history_max_c",
        "compound_heat_humidity_days",
    }
    assert expected.issubset(feature_columns(28, "poultry"))
    assert snapshots["max_source_day_used"].le(snapshots["review_day"]).all()
    day7 = snapshots.loc[snapshots["review_day"].eq(7)]
    assert day7["heat_excess_degree_days"].notna().sum() > 0
    assert day7["high_humidity_day_pct"].notna().sum() > 0


def test_expected_remaining_growth_projection_is_fold_local() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    development = build_snapshots(dataset).query("role == 'development' and review_day == 14").reset_index(drop=True)
    candidate = next(item for item in CANDIDATES if item.name == "historical_remaining_gain")
    train = development.query("cycle_id != '2026-2'").reset_index(drop=True)
    test = development.query("cycle_id == '2026-2'").reset_index(drop=True)
    entry = _fit_entry(train, candidate, {}, 14)
    expected_gain = np.mean(train["outcome_day35_weight_g"] - train["current_weight_g"])
    assert np.isclose(entry["remaining_gain_g"], expected_gain)
    assert np.allclose(_predict_entry(entry, test), test["current_weight_g"] + expected_gain)


def test_external_boosting_candidates_are_registered() -> None:
    families = {candidate.family for candidate in CANDIDATES}
    assert {"xgboost", "lightgbm", "catboost"}.issubset(families)


def test_exported_champion_reloads_with_prediction_parity() -> None:
    artifact_path = ROOT / "outputs" / "bodyweight_modeling_review" / "champion.joblib"
    artifact = joblib.load(artifact_path)
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    audit = build_snapshots(dataset).query("role == 'later_cycle_audit'").reset_index(drop=True)
    expected = np.loadtxt(
        ROOT / "outputs" / "bodyweight_modeling_review" / "later_cycle_audit_predictions.csv",
        delimiter=",",
        skiprows=1,
        usecols=5,
    )
    actual = _predict_bundle(artifact["checkpoint_models"], audit)
    assert np.allclose(actual, expected)
