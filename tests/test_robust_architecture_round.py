from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from canary.data import load_workbook
from canary.robust_architecture_round import (
    CHECKPOINTS,
    RECOVERY_CANDIDATES,
    WEIGHT_CANDIDATES,
    _profile_candidates,
    _promotion_gate,
    candidate_grid,
    evaluate_nested_logo,
    feature_columns,
    fit_candidate,
    predict_candidate,
    summarize,
)
from canary.biology_aware_modeling import LANDMARK_DAYS, build_daily_landmarks


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")


@pytest.mark.parametrize("outcome", ["recovery", "weight"])
def test_daily_landmarks_are_asof_safe_balanced_and_locked(dataset, outcome: str) -> None:
    frame = build_daily_landmarks(dataset, outcome)
    development = frame.query("role == 'development'")
    assert set(development["review_day"]) == set(LANDMARK_DAYS)
    assert len(development) == 31 * len(LANDMARK_DAYS)
    assert development["max_source_day_used"].le(development["review_day"]).all()
    assert "2026-3" not in set(development["cycle_id"].astype(str))
    total_weight = development.groupby(["cycle_id", "building_id"])["sample_weight"].sum()
    assert total_weight.nunique() == 1


def test_all_three_architectures_and_required_boosters_are_registered() -> None:
    for registry in (RECOVERY_CANDIDATES, WEIGHT_CANDIDATES):
        assert {"pooled", "checkpoint", "hybrid"}.issubset({candidate.architecture for candidate in registry})
        assert {"random_forest", "extra_trees", "hist_gradient", "xgboost", "lightgbm", "catboost"}.issubset({candidate.family for candidate in registry})
    assert "negative_binomial" in {candidate.family for candidate in RECOVERY_CANDIDATES}
    assert {"kalman", "gompertz", "logistic"}.issubset({candidate.family for candidate in WEIGHT_CANDIDATES})


def test_primary_schemas_exclude_identity_and_feed() -> None:
    for candidate in (*RECOVERY_CANDIDATES, *WEIGHT_CANDIDATES):
        if candidate.family in {"persistence", "age_baseline", "target_ratio"}:
            continue
        lowered = " ".join(feature_columns(candidate)).lower()
        assert "building_id" not in lowered
        assert "tags" not in lowered
        assert "lags" not in lowered
        assert "feed" not in lowered


def test_checkpoint_model_refuses_intervening_days(dataset) -> None:
    frame = build_daily_landmarks(dataset, "weight").query("role == 'development'").reset_index(drop=True)
    candidate = next(item for item in WEIGHT_CANDIDATES if item.name == "checkpoint_remaining_ridge")
    fitted = fit_candidate(frame, candidate, candidate_grid(candidate)[0])
    with pytest.raises(ValueError, match="intervening"):
        predict_candidate(fitted, frame.query("review_day == 10"), candidate)
    checkpoint = frame.loc[frame["review_day"].isin(CHECKPOINTS)]
    assert len(predict_candidate(fitted, checkpoint, candidate)) == len(checkpoint)


def test_pooled_and_hybrid_models_score_day10(dataset) -> None:
    frame = build_daily_landmarks(dataset, "weight").query("role == 'development'").reset_index(drop=True)
    day10 = frame.query("review_day == 10")
    for name in ("pooled_remaining_ridge", "hybrid_remaining_ridge"):
        candidate = next(item for item in WEIGHT_CANDIDATES if item.name == name)
        fitted = fit_candidate(frame, candidate, candidate_grid(candidate)[0])
        prediction = predict_candidate(fitted, day10, candidate)
        assert len(prediction) == 31
        assert np.isfinite(prediction).all()


def test_primary_logo_holds_out_complete_cycles(dataset) -> None:
    frame = build_daily_landmarks(dataset, "recovery").query("role == 'development'").reset_index(drop=True)
    groups = frame["cycle_id"].astype(str).to_numpy()
    for train, test in LeaveOneGroupOut().split(frame, groups=groups):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(set(groups[test])) == 1


def test_screening_profile_contains_each_architecture_and_is_deterministic(dataset) -> None:
    candidates = _profile_candidates("weight", "all", "screening")
    assert {"pooled", "checkpoint", "hybrid"}.issubset({candidate.architecture for candidate in candidates})
    candidate = next(item for item in candidates if item.name == "historical_remaining_gain")
    frame = build_daily_landmarks(dataset, "weight").query("role == 'development'").reset_index(drop=True)
    first, _ = evaluate_nested_logo(frame, candidate)
    second, _ = evaluate_nested_logo(frame, candidate)
    assert np.allclose(first["predicted"], second["predicted"])


def test_full_profile_compares_direct_and_remaining_targets() -> None:
    for outcome in ("recovery", "weight"):
        candidates = _profile_candidates(outcome, "all", "full")
        forms = {(candidate.architecture, candidate.family, candidate.target_form) for candidate in candidates}
        for architecture in ("pooled", "checkpoint", "hybrid"):
            assert (architecture, "ridge", "direct") in forms
            assert (architecture, "ridge", "remaining") in forms


def test_target_side_confusion_counts_match_prediction_rows(dataset) -> None:
    frame = build_daily_landmarks(dataset, "weight").query("role == 'development' and review_day in [7, 14, 21, 28]").reset_index(drop=True)
    candidate = next(item for item in WEIGHT_CANDIDATES if item.name == "historical_remaining_gain")
    predictions, _ = evaluate_nested_logo(frame, candidate)
    metrics = summarize(predictions, "weight", bootstrap=False)
    assert metrics["target_tn"] + metrics["target_fp"] + metrics["target_fn"] + metrics["target_tp"] == len(predictions)


def test_promotion_gate_reports_matched_baseline_improvement() -> None:
    comparison = pd.DataFrame([
        {"candidate": "baseline", "cycle_macro_rmse": 100.0, "mae": 80.0, "r2": 0.1, "bias": 0.0, "worst_cycle_rmse": 150.0},
        {"candidate": "challenger", "cycle_macro_rmse": 89.0, "mae": 75.0, "r2": 0.2, "bias": 10.0, "worst_cycle_rmse": 155.0},
    ])
    rows = []
    for cycle in ("c1", "c2", "c3", "c4", "c5", "c6"):
        for day in CHECKPOINTS:
            rows.extend([
                {"candidate": "baseline", "cycle_id": cycle, "review_day": day, "actual": 1000.0, "predicted": 1100.0},
                {"candidate": "challenger", "cycle_id": cycle, "review_day": day, "actual": 1000.0, "predicted": 1089.0},
            ])
    gate = _promotion_gate(
        comparison,
        pd.DataFrame(rows),
        "challenger",
        "baseline",
        "weight",
        pd.DataFrame({"covered_80": [True] * 8 + [False] * 2}),
        {"cycle_macro_rmse": 90.0},
        {"cycle_macro_rmse": 100.0},
    )
    assert gate["cycle_macro_rmse_improvement_pct"] == pytest.approx(11.0)
    assert gate["retrospective_gate_passed"]
