from pathlib import Path
import os

from canary import load_workbook, train_day35_weight_baseline


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_day35_model_compares_five_models_on_complete_cycle_holdouts():
    manifest = train_day35_weight_baseline(load_workbook(SOURCE))
    metrics = manifest["candidate_metrics"]

    assert len(metrics) == 5
    assert "historical_remaining_gain" in metrics
    assert "checkpoint_linear_remaining_gain" in metrics
    assert "ridge_remaining_gain" in metrics
    assert "huber_remaining_gain" in metrics
    assert "gradient_boosting_remaining_gain" in metrics
    assert len(manifest["candidate_registry"]) == 5
    assert manifest["research_champion"] in metrics
    assert manifest["selected_model"] == "historical_remaining_gain"
    assert manifest["champion_gates"]["operational_fallback_applied"] is True
    assert manifest["champion_gates"]["regression_gate_passed"] is False
    assert all("r2" in value for value in metrics.values())
    assert manifest["model_version"] == "day35-weight-2.2.0"
    assert manifest["selection_metric"] == "nested_leave_one_complete_cycle_out_cycle_macro_mae_kg"
    assert manifest["training_building_cycles"] == 31
    assert manifest["day14_backtest_metrics"]["building_cycles"] == 31
    assert len(manifest["day14_backtest"]) == 31
    assert manifest["prospective_latest_cycle_audit"]["cycle_id"] == "2026-3"
    assert manifest["prospective_latest_cycle_audit"]["independent_outcomes"] == 3
    assert manifest["target_weight_kg"] == 1.8
    assert manifest["actual_target_hits"] == 1
    assert manifest["day14_backtest_metrics"]["target_side_accuracy"] > 0.8
    assert {
        "selected_method_drivers",
        "feature_importance_interpretation",
        "held_out_permutation_importance",
        "secondary_within_cycle_metrics",
        "primary_whole_cycle_bootstrap_mae_95ci",
    }.issubset(manifest)
