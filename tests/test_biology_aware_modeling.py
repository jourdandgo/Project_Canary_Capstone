from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from canary.biology_aware_modeling import (
    AUDIT_CYCLE,
    LANDMARK_DAYS,
    RECOVERY_CANDIDATES,
    build_daily_landmarks,
    fit_candidate,
    predict_candidate,
)
from canary.data import load_workbook


ROOT = Path(__file__).resolve().parents[1]


def test_daily_landmarks_are_complete_equal_weight_and_leakage_safe() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    for outcome in ("recovery", "weight"):
        frame = build_daily_landmarks(dataset, outcome)
        development = frame.loc[frame["role"].eq("development")]
        assert len(development) == 31 * len(LANDMARK_DAYS)
        assert development["max_source_day_used"].le(development["review_day"]).all()
        assert AUDIT_CYCLE not in set(development["cycle_id"].astype(str))
        influence = development.groupby(["cycle_id", "building_id"])["sample_weight"].sum()
        assert np.allclose(influence, influence.iloc[0])
        off_checkpoint = development.loc[~development["review_day"].isin([7, 14, 21, 28])]
        assert off_checkpoint["checkpoint_status"].eq("between_checkpoint_estimate").all()


def test_recovery_predictions_obey_current_survival_constraint() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    frame = build_daily_landmarks(dataset, "recovery")
    train = frame.loc[frame["role"].eq("development") & ~frame["cycle_id"].eq("2025-2")].reset_index(drop=True)
    test = frame.loc[frame["cycle_id"].eq("2025-2")].reset_index(drop=True)
    candidate = next(item for item in RECOVERY_CANDIDATES if item.name == "negative_binomial_loss_hazard")
    fitted = fit_candidate(train, candidate, {})
    predicted = predict_candidate(fitted, test, candidate)
    assert np.all(predicted <= test["current_value"].to_numpy(float) + 1e-12)
    assert np.all((predicted >= 0) & (predicted <= 1))


def test_no_identity_or_feed_in_primary_feature_schemas() -> None:
    from canary.biology_aware_modeling import RECOVERY_FEATURES, WEIGHT_FEATURES

    for feature in [*RECOVERY_FEATURES, *WEIGHT_FEATURES]:
        lowered = feature.lower()
        assert "building_id" not in lowered
        assert "is_lags" not in lowered
        assert "feed" not in lowered

