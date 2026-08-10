from pathlib import Path
import os

from canary import load_workbook, train_day35_weight_baseline


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_day35_model_compares_adg_ridge_and_transparent_baselines():
    manifest = train_day35_weight_baseline(load_workbook(SOURCE))
    metrics = manifest["candidate_metrics"]

    assert "recent_linear_adg" in metrics
    assert "ridge_regression" in metrics
    assert "random_forest" in metrics
    assert "gradient_boosting" in metrics
    best_macro = min(value["cycle_macro_mae_kg"] for value in metrics.values())
    assert manifest["selected_metrics"]["cycle_macro_mae_kg"] <= best_macro * 1.05
    assert manifest["selected_model"] == "ridge_regression"
    assert (
        metrics[manifest["selected_model"]]["mae_kg"]
        < metrics["recent_linear_adg"]["mae_kg"]
    )
    assert manifest["model_version"] == "day35-weight-0.4.0"
    assert manifest["selection_metric"] == "cycle_macro_mae_kg_within_5pct_then_simplest"
    assert manifest["training_building_cycles"] == 31
    assert manifest["day14_backtest_metrics"]["building_cycles"] == 31
    assert len(manifest["day14_backtest"]) == 31
    assert manifest["target_weight_kg"] == 1.8
    assert manifest["actual_target_hits"] == 5
    assert manifest["day14_backtest_metrics"]["target_side_accuracy"] > 0.8
    assert {"selected_method_drivers", "feature_importance_interpretation"}.issubset(manifest)
