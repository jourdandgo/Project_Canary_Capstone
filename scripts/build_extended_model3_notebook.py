"""Build the reader-facing notebook for the extended Model 3 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "analysis" / "extended_model3_experiment" / "results.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "Extended_Model3_Day28_Evaluation.ipynb"


def build() -> Path:
    results = json.loads(RESULTS_PATH.read_text())
    dev = results["selected_development_metrics"]
    checkpoints = {
        int(row["prediction_day"]): row
        for row in results["selected_checkpoint_metrics"]
    }
    official = results["original_model3_official"]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            f"""# Extended Model 3: Day 7–28 Evaluation

## tl;dr

The extension is technically feasible, but the evidence does **not** support replacing Trish's final Model 3 with a larger all-feature CatBoost.

- Trish's separately trained terminal Day-21 model reports **{official['terminal_day21_model_mae_g']:.1f} g MAE** under leave-one-building-flock-out validation across all 34 flocks.
- Under the stricter leave-one-production-cycle-out design on 31 development flocks, the reduced extended CatBoost reaches **{dev['cycle_macro_mae_g']:.1f} g cycle-macro MAE** across Days 7, 14, 21, and 28.
- Its checkpoint MAE is **{checkpoints[7]['pooled_mae_g']:.1f} g at Day 7**, **{checkpoints[14]['pooled_mae_g']:.1f} g at Day 14**, **{checkpoints[21]['pooled_mae_g']:.1f} g at Day 21**, and **{checkpoints[28]['pooled_mae_g']:.1f} g at Day 28**.
- A transparent historical remaining-gain method is slightly weaker in development but materially stronger on the three-building 2026-3 audit. It is the safer provisional checkpoint outlook.

**Recommendation:** retain Trish's Model 3 as the established Day-21 benchmark. If Canary needs Days 7, 14, 21, and 28 now, place the historical remaining-gain checkpoint method in shadow use and keep the learned extension experimental until more complete cycles show that it generalizes.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

### Decision

Can Trish's Day-21 bodyweight model be extended to Days 7, 14, 21, and 28 without sacrificing defensibility?

### Key assumptions

- Day-35 bodyweight is the recorded Day-35 measurement.
- Official checkpoint weights at Days 7, 14, 21, and 28 are observed measurements.
- Forecast rows contain only information available through the checkpoint.
- All buildings from the same production cycle are held out together during development validation.
- The three buildings in cycle 2026-3 are excluded from fitting and used as a later-cycle audit.
- The 2026-3 audit is small and is not a substitute for several future unseen cycles. Upstream Trish feature work used all 34 flocks, so it should be described as a later-cycle audit rather than a perfectly untouched external validation set.

The experiment compares a historical remaining-gain baseline; linear, regularized, robust, tree, boosting, CatBoost, and XGBoost candidates; a full 94-feature CatBoost extension; and a reduced CatBoost using growth features plus Trish's confirmed contextual fields. Fold-local medians are used for missing values. No application files are changed.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if not (ROOT / "scripts" / "extended_model3_experiment.py").exists():
    raise RuntimeError("Run this notebook from Project_Canary_GitHub_Ready or its notebooks directory.")
sys.path.insert(0, str(ROOT / "scripts"))

from extended_model3_experiment import run, OUTPUT_DIR

results = run()
results["data_quality"]"""
        ),
        nbf.v4.new_markdown_cell(
            """## Data

Each modeling row represents one building-flock at one official weighing checkpoint. The 31 development building-flocks cover cycles 2025-2 through 2026-2. Cycle 2026-3 contributes three later-cycle audit buildings. Repeated checkpoint rows from the same cycle are never split between training and validation.

The feature construction retains Trish's temperature, humidity, THI, feed, mortality, housing, and growth-history definitions. The extension adds the latest observed checkpoint weight, prior checkpoint weight, seven-day gain, gain rate, a simple trajectory projection, checkpoint age, and days remaining to Day 35.
"""
        ),
        nbf.v4.new_code_cell(
            """modeling_frame = pd.read_csv(OUTPUT_DIR / "checkpoint_modeling_frame.csv")
pd.DataFrame({
    "check": ["Rows", "Building-flocks", "Development flocks", "Audit flocks", "Duplicate keys", "Missing targets"],
    "result": [
        len(modeling_frame),
        modeling_frame[["harvest_cycle", "bldg"]].drop_duplicates().shape[0],
        modeling_frame.loc[modeling_frame.harvest_cycle != "2026-3", ["harvest_cycle", "bldg"]].drop_duplicates().shape[0],
        modeling_frame.loc[modeling_frame.harvest_cycle == "2026-3", ["harvest_cycle", "bldg"]].drop_duplicates().shape[0],
        modeling_frame.duplicated(["harvest_cycle", "bldg", "prediction_day"]).sum(),
        modeling_frame.bodyweight_at_day_35.isna().sum(),
    ],
})"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 1. Candidate comparison"),
        nbf.v4.new_code_cell(
            """comparison = pd.read_csv(OUTPUT_DIR / "candidate_comparison.csv")
comparison.round({
    "pooled_mae_g": 1,
    "cycle_macro_mae_g": 1,
    "rmse_g": 1,
    "r2": 3,
    "bias_g": 1,
    "worst_cycle_mae_g": 1,
    "within_100g_pct": 1,
    "within_200g_pct": 1,
})"""
        ),
        nbf.v4.new_markdown_cell(
            """The full-feature CatBoost extension is weaker than the simpler checkpoint baseline. Reducing the learned model to observed growth plus a small amount of validated context improves development performance, which is consistent with the small effective sample: 31 independent outcomes cannot reliably support a very wide feature set.

### 2. Apples-to-apples Day-21 comparison

The published and strict results answer different questions and must not be placed in one table without their validation labels.
"""
        ),
        nbf.v4.new_code_cell(
            """official = results["original_model3_official"]
strict = results["original_model3_strict_day21_reconstruction"]
extended_day21 = results["selected_day21_metrics"]
pd.DataFrame([
    {
        "method": "Trish terminal Day-21 Model 3",
        "validation": "Leave one building-flock out, 34 flocks",
        "scope": "Day 21 only",
        "MAE_g": official["terminal_day21_model_mae_g"],
    },
    {
        "method": "Trish Model 3 strict reconstruction",
        "validation": "Leave one production cycle out, 31 flocks",
        "scope": "Day 21 only",
        "MAE_g": strict["cycle_macro_mae_g"],
    },
    {
        "method": "Reduced extended CatBoost",
        "validation": "Leave one production cycle out, 31 flocks",
        "scope": "Day-21 slice",
        "MAE_g": extended_day21["cycle_macro_mae_g"],
    },
]).round(1)"""
        ),
        nbf.v4.new_markdown_cell("### 3. Accuracy by checkpoint"),
        nbf.v4.new_code_cell(
            """checkpoint_metrics = pd.read_csv(OUTPUT_DIR / "selected_checkpoint_metrics.csv")
checkpoint_metrics.round({
    "pooled_mae_g": 1,
    "cycle_macro_mae_g": 1,
    "rmse_g": 1,
    "r2": 3,
    "bias_g": 1,
    "worst_cycle_mae_g": 1,
    "within_100g_pct": 1,
    "within_200g_pct": 1,
})"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(OUTPUT_DIR / "extended_model3_performance.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Later-cycle audit

The audit contains only three Tags buildings from one cycle. It is useful as a stability warning, not as a definitive ranking. R² is not interpretable here because the three outcomes occupy a very narrow range; MAE and bias are more informative.
"""
        ),
        nbf.v4.new_code_cell(
            """audit_summary = pd.read_csv(OUTPUT_DIR / "audit_2026_3_summary.csv")
audit_summary.round({
    "pooled_mae_g": 1,
    "cycle_macro_mae_g": 1,
    "rmse_g": 1,
    "r2": 3,
    "bias_g": 1,
    "worst_cycle_mae_g": 1,
    "within_100g_pct": 1,
    "within_200g_pct": 1,
})"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **The extension is possible.** Canary can issue evidence-based Day-35 bodyweight outlooks after the actual Day 7, 14, 21, and 28 weigh-ins.
2. **It is a weekly checkpoint model, not a validated daily bodyweight model.** Between weigh-ins, the forecast should stay unchanged and display the age of its latest weight evidence.
3. **More features did not mean better forecasts.** The full 94-feature CatBoost underperformed the transparent remaining-gain benchmark. The reduced CatBoost improved development MAE but did not generalize as well to 2026-3.
4. **Do not claim that the extension beats Trish's Model 3 based on the published numbers alone.** Trish's 103.66 g result uses leave-one-building-flock-out validation across all 34 flocks; the extension's principal comparison holds out complete production cycles and excludes 2026-3 from fitting.
5. **Recommended decision:** keep Trish's Model 3 as the Day-21 benchmark. Put the historical remaining-gain checkpoint method in shadow mode for Days 7, 14, 21, and 28. Retain the reduced CatBoost as an experimental challenger and reconsider promotion after several new completed cycles.

The forecasts remain planning outlooks. THI, temperature, humidity, mortality, feed, and housing fields are contextual predictors, not proof that changing one factor will cause a specific bodyweight change.
"""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
