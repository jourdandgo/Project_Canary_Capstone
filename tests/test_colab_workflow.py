from pathlib import Path

import nbformat

from canary.colab_workflow import RunConfig
from canary.model_optimization_round import _profile_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_profile_contains_required_model_families() -> None:
    recovery, weight = _profile_candidates("balanced")
    recovery_families = {candidate.family for candidate in recovery}
    weight_families = {candidate.family for candidate in weight}
    required = {"baseline", "ridge", "huber", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"}
    assert required.issubset(recovery_families)
    assert {"baseline", "pls", "ridge", "huber", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"}.issubset(weight_families)


def test_full_profile_remains_complete() -> None:
    recovery, weight = _profile_candidates("full")
    assert len(recovery) == 25
    assert len(weight) == 30


def test_colab_notebook_is_valid_and_reader_facing() -> None:
    path = ROOT / "notebooks" / "Project_Canary_End_to_End_Model_Refresh_Colab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    for required in (
        "Capstone-focused EDA",
        "Model comparison and champion selection",
        "Does accuracy improve",
        "Feature importance and SHAP",
        "Gemini optimization prompt",
        "Download the complete evidence bundle",
    ):
        assert required in source


def test_run_config_defaults_are_safe() -> None:
    config = RunConfig("workbook.xlsx", "output")
    assert config.run_profile == "balanced"
    assert config.audit_cycle == "latest"
    assert config.seed == 20260812
    assert config.daily_start == 7
    assert config.daily_end == 34
