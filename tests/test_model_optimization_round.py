from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from canary.data import load_workbook
from canary.model_optimization_round import (
    AUDIT_CYCLE,
    CHECKPOINTS,
    RECOVERY_CANDIDATES,
    WEIGHT_CANDIDATES,
    QuantileWinsorizer,
    _features,
    _group_values,
    build_optimization_snapshots,
    candidate_grid,
    evaluate_logo,
    fit_candidate,
    predict_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")


@pytest.mark.parametrize("outcome", ["recovery", "weight"])
def test_optimization_snapshots_are_locked_and_asof_safe(dataset, outcome: str) -> None:
    frame = build_optimization_snapshots(dataset, outcome)
    development = frame.query("role == 'development'")
    audit = frame.query("role == 'later_cycle_audit'")
    assert len(development) == 124
    assert development[["cycle_id", "building_id"]].drop_duplicates().shape[0] == 31
    assert set(development["review_day"]) == set(CHECKPOINTS)
    assert AUDIT_CYCLE not in set(development["cycle_id"])
    assert set(audit["cycle_id"]) == {AUDIT_CYCLE}
    assert frame["max_source_day_used"].le(frame["review_day"]).all()


def test_peer_context_is_leave_self_out(dataset) -> None:
    frame = build_optimization_snapshots(dataset, "weight")
    group = frame.query("cycle_id == '2025-5' and review_day == 14").copy()
    assert len(group) > 1
    for row in group.itertuples():
        expected = group.loc[group["building_id"].ne(row.building_id), "current_weight_g"].mean()
        assert row.peer_current_weight_mean_g == pytest.approx(expected)
        assert row.peer_building_count == len(group) - 1


def test_primary_feature_schemas_exclude_identity_and_feed() -> None:
    for candidate in (*RECOVERY_CANDIDATES, *WEIGHT_CANDIDATES):
        days = CHECKPOINTS if candidate.checkpoint_specific else (None,)
        for day in days:
            columns = _features(candidate, day)
            lowered = " ".join(columns).lower()
            assert "building_id" not in lowered
            assert "tags" not in lowered
            assert "lags" not in lowered
            assert "feed" not in lowered


def test_logo_cycle_folds_have_complete_group_separation(dataset) -> None:
    frame = build_optimization_snapshots(dataset, "recovery").query("role == 'development'").reset_index(drop=True)
    groups = _group_values(frame, "cycle")
    for train, test in LeaveOneGroupOut().split(frame, groups=groups):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(set(groups[test])) == 1


def test_fold_local_winsorizer_does_not_learn_from_test_extreme() -> None:
    transformer = QuantileWinsorizer(0.0, 0.95).fit(np.array([[0.0], [1.0], [2.0]]))
    assert transformer.upper_bounds_[0] < 1000
    transformed = transformer.transform(np.array([[1000.0]]))
    assert transformed[0, 0] == pytest.approx(transformer.upper_bounds_[0])


def test_predefined_grids_and_simple_logo_are_deterministic(dataset) -> None:
    candidate = next(item for item in RECOVERY_CANDIDATES if item.name == "age_band_remaining_loss")
    assert candidate_grid(candidate) == candidate_grid(candidate)
    frame = build_optimization_snapshots(dataset, "recovery").query("role == 'development'").reset_index(drop=True)
    first, _ = evaluate_logo(frame, candidate)
    second, _ = evaluate_logo(frame, candidate)
    assert np.allclose(first["predicted"], second["predicted"])


def test_external_boosting_families_are_in_both_registries() -> None:
    for registry in (RECOVERY_CANDIDATES, WEIGHT_CANDIDATES):
        assert {"xgboost", "lightgbm", "catboost"}.issubset({candidate.family for candidate in registry})


def test_final_recovery_search_includes_direct_remaining_and_residual_formulations() -> None:
    forms = {(candidate.family, candidate.target_form) for candidate in RECOVERY_CANDIDATES}
    for family in ("gradient_boosting", "hist_gradient_boosting"):
        assert {(family, "direct"), (family, "remaining"), (family, "baseline_residual")} <= forms
    for family in ("xgboost", "lightgbm", "catboost"):
        assert {(family, "remaining"), (family, "baseline_residual")} <= forms


def test_final_weight_search_includes_naive_and_pooled_challengers() -> None:
    families = {candidate.family for candidate in WEIGHT_CANDIDATES}
    assert {"baseline", "target_curve", "target_ratio", "historical_ratio", "recent_adg"} <= families
    pooled = [candidate for candidate in WEIGHT_CANDIDATES if candidate.name.startswith("pooled_")]
    assert pooled and all(not candidate.checkpoint_specific for candidate in pooled)


def test_target_ratio_naive_projection_matches_target_curve_math(dataset) -> None:
    frame = build_optimization_snapshots(dataset, "weight").query("role == 'development' and review_day == 7").reset_index(drop=True)
    candidate = next(item for item in WEIGHT_CANDIDATES if item.name == "target_curve_ratio")
    fitted = fit_candidate(frame, candidate, candidate_grid(candidate)[0])
    prediction = predict_candidate(fitted, frame, candidate)
    expected = frame["current_weight_g"].to_numpy(float) * 1800.0 / frame["current_target_g"].to_numpy(float)
    assert np.allclose(prediction, expected)
