from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from canary.data import load_workbook
from canary.external_modeling_review import (
    AUDIT_CYCLE,
    RECOVERY_CANDIDATES,
    build_snapshots,
    evaluate_candidate,
    select_champion,
    target_metrics,
    unit_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_independent_snapshots_preserve_grain_and_temporal_integrity() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    recovery = build_snapshots(dataset, "recovery")
    weight = build_snapshots(dataset, "weight")
    assert dataset.quality.canonical_rows == 1624
    assert dataset.cycles[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 34
    assert recovery.query("role == 'development'")[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 31
    assert weight.query("role == 'development'")[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 31
    assert len(recovery.query("role == 'development'")) == 124
    assert len(weight.query("role == 'development'")) == 124
    assert set(recovery.query("role == 'development'")["review_day"]) == {7, 14, 21, 28}
    assert set(recovery.query("role == 'later_cycle_audit'")["cycle_id"]) == {AUDIT_CYCLE}
    assert recovery["max_source_day_used"].le(recovery["review_day"]).all()
    assert weight["max_source_day_used"].le(weight["review_day"]).all()
    assert weight.loc[weight["review_day"].eq(14), "weight_measurement_day"].le(14).all()


def test_unit_weights_give_each_building_cycle_equal_total_weight() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    rows = build_snapshots(dataset, "recovery").query("role == 'development'").copy()
    rows["weight"] = unit_weights(rows)
    totals = rows.groupby(["cycle_id", "building_id"])["weight"].sum()
    assert np.allclose(totals, totals.iloc[0])


def test_target_metrics_and_rmse_champion_selection() -> None:
    metrics = target_metrics(np.array([0.94, 0.96]), np.array([0.93, 0.94]), "recovery")
    assert metrics["confusion_actual_below_predicted_below"] == 1
    assert metrics["confusion_actual_above_predicted_below"] == 1
    comparison = pd.DataFrame([
        {"candidate": "complex", "cycle_macro_rmse": 1.0, "rmse": 1.1, "r2": 0.2, "complexity": 5},
        {"candidate": "simple", "cycle_macro_rmse": 1.15, "rmse": 1.2, "r2": 0.1, "complexity": 1},
    ])
    selected, _ = select_champion(comparison)
    assert selected == "complex"


def test_baseline_evaluation_is_deterministic_and_artifacts_reload() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    rows = build_snapshots(dataset, "recovery").query("role == 'development'").reset_index(drop=True)
    candidate = RECOVERY_CANDIDATES[1]
    first, _ = evaluate_candidate(rows, candidate, "recovery")
    second, _ = evaluate_candidate(rows, candidate, "recovery")
    assert np.allclose(first["predicted"], second["predicted"])
    artifact = ROOT / "outputs" / "external_modeling_review" / "recovery" / "champion.joblib"
    assert joblib.load(artifact)["outcome"] == "recovery"
