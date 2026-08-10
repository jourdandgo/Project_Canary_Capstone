from copy import deepcopy
from pathlib import Path
import os

import pandas as pd
import pytest

from canary import (
    RecommendationConfigurationError,
    apply_recommendations,
    build_recommendation_trace,
    default_as_of_date,
    load_recommendation_playbook,
    load_workbook,
    save_recommendation_playbook,
    score_cycle_snapshot,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


@pytest.fixture(scope="module")
def dataset():
    return load_workbook(SOURCE)


def test_every_operational_pattern_has_a_deterministic_rule():
    playbook = load_recommendation_playbook()
    patterns = {rule["pattern"] for rule in playbook["rules"]}
    assert {"Low Body Weight", "High Mortality", "Rapid Population Loss", "Abnormal Temperature Fluctuation", "High Humidity", "Low Humidity"}.issubset(patterns)
    assert len({rule["rule_id"] for rule in playbook["rules"]}) == 11


def test_pattern_and_risk_rating_map_to_action_and_urgency(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    risk = score_cycle_snapshot(dataset, "2026-3", as_of)
    result = apply_recommendations(risk)

    tags2 = result.query("building_id == 'Tags 2'").iloc[0]
    assert tags2["risk_pattern"] == "Low Body Weight"
    assert tags2["recommendation_rule_id"] == "DOC-002"
    assert tags2["recommendation_urgency"] == "Current shift"
    assert "weight" in tags2["recommended_action"].lower()
    assert tags2["recommendation_guidance_status"].startswith("Preliminary")

    trace = build_recommendation_trace(tags2)
    assert trace.loc[trace["Decision element"] == "Action rule", "Applied value"].iloc[0] == "DOC-002"
    assert trace.loc[trace["Decision element"] == "Risk-level urgency", "Applied value"].iloc[0] == "High → Current shift"


def test_recommendations_do_not_change_risk_values(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    risk = score_cycle_snapshot(dataset, "2026-3", as_of)
    result = apply_recommendations(risk)
    columns = ["risk_score", "risk_rating", "weight_score", "population_loss_score", "daily_mortality_score", "environment_score"]
    pd.testing.assert_frame_equal(risk[columns], result[columns], check_dtype=False)


def test_missing_evidence_rule_is_used_when_no_material_drift_is_under_supported(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    risk = score_cycle_snapshot(dataset, "2026-3", as_of)
    row = risk.query("building_id == 'Tags 1'").copy()
    row.loc[:, "risk_pattern"] = "No Material Concern"
    row.loc[:, "evidence_status"] = "Reduced evidence"

    result = apply_recommendations(row).iloc[0]
    assert result["recommendation_rule_id"] == "DOC-011"
    assert result["recommendation_pattern"] == "Missing or Stale Evidence"


def test_inactive_waits_while_records_ended_retains_last_record_guidance(dataset):
    as_of = default_as_of_date(dataset, "2026-3")
    current = apply_recommendations(score_cycle_snapshot(dataset, "2026-3", as_of))
    inactive = current.loc[current["state"] == "Inactive"]
    assert inactive["recommendation_rule_id"].eq("Not applicable").all()

    historical_end = dataset.cycles.loc[dataset.cycles["cycle_id"] == "2025-5", "end_date"].max()
    completed = apply_recommendations(score_cycle_snapshot(dataset, "2025-5", historical_end))
    ended = completed.loc[completed["state"] == "Records ended"]
    assert not ended.empty
    assert ended["risk_score"].notna().all()
    assert ended["recommendation_rule_id"].ne("Not applicable").all()
    assert ended["recommendation_rule_id"].ne("Completed cycle").all()


def test_admin_save_is_validated_and_updates_overall_approval(tmp_path):
    playbook = deepcopy(load_recommendation_playbook())
    for rule in playbook["rules"]:
        rule["approval_status"] = "Approved"
        rule["approval_date"] = "2026-08-07"
    destination = tmp_path / "recommendations.json"
    save_recommendation_playbook(playbook, destination)
    saved = load_recommendation_playbook(destination)

    assert saved["approval_status"] == "Approved by Doc Raymond"
    assert saved["version"].endswith("-approved")

    broken = deepcopy(saved)
    broken["rules"][0]["approval_status"] = "Maybe"
    with pytest.raises(RecommendationConfigurationError):
        save_recommendation_playbook(broken, destination)
