from pathlib import Path

import pytest

from canary.research_evidence import (
    DEFAULT_RESEARCH_ROOT,
    display_name,
    load_outcome_research_evidence,
)


@pytest.mark.parametrize(
    ("outcome", "challenger", "selected", "explanation_model"),
    [
        (
            "recovery",
            "residual_lightgbm_peer",
            "age_band_remaining_loss",
            "residual_lightgbm_peer",
        ),
        (
            "bodyweight",
            "historical_remaining_gain",
            "historical_remaining_gain",
            "residual_xgboost_peer",
        ),
    ],
)
def test_frozen_research_evidence_is_complete(
    outcome, challenger, selected, explanation_model
):
    evidence = load_outcome_research_evidence(outcome)
    assert evidence.challenger == challenger
    assert evidence.one_se_selection == selected
    assert evidence.explanation_model == explanation_model
    assert len(evidence.top_five) == 5
    assert len(evidence.challenger_predictions) == 124
    assert len(evidence.selected_predictions) == 124
    assert evidence.challenger_checkpoints["review_day"].tolist() == [7, 14, 21, 28]
    assert evidence.selected_checkpoints["review_day"].tolist() == [7, 14, 21, 28]
    assert len(evidence.top_shap) == 10
    assert evidence.manifest["operational_models_changed"] is False


def test_missing_research_artifacts_fail_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_outcome_research_evidence("recovery", tmp_path)


def test_invalid_outcome_is_rejected():
    with pytest.raises(ValueError, match="outcome must"):
        load_outcome_research_evidence("feed", DEFAULT_RESEARCH_ROOT)


def test_model_names_are_reader_friendly():
    assert display_name("direct_trajectory_pls") == "Direct Trajectory PLS"
    assert display_name("residual_xgboost_peer") == "Residual XGBoost Peer"
