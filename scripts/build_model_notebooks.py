"""Build and execute Project Canary's two reader-facing model notebooks."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks"
OUTPUT.mkdir(exist_ok=True)


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


SETUP = r"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "canary").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "FARM HARVEST DATA.xlsx"
MODEL_READY_DIR = ROOT / "outputs" / "model_ready"

from canary import load_workbook

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
dataset = load_workbook(DATA_PATH)
print(f"Source: {dataset.source_name}")
print(f"Canonical building-day rows: {len(dataset.daily):,}")
print(f"Recorded building-cycles: {len(dataset.cycles):,}")
print(f"Blocking data-quality checks passed: {dataset.quality.passed}")
print(f"Non-blocking warnings: {len(dataset.quality.warnings)}")
"""


BOOTSTRAP = r"""
def cycle_bootstrap_mae(frame, error_column, repeats=5000, seed=42):
    # Bootstrap whole cycles, never individual rows, to preserve grouped evidence.
    rng = np.random.default_rng(seed)
    grouped = {cycle: group for cycle, group in frame.groupby("cycle_id")}
    cycles = np.array(list(grouped))
    estimates = []
    for _ in range(repeats):
        selected = rng.choice(cycles, size=len(cycles), replace=True)
        errors = np.concatenate([grouped[cycle][error_column].to_numpy(float) for cycle in selected])
        estimates.append(np.mean(np.abs(errors)))
    return np.quantile(estimates, [0.025, 0.975])
"""


def recovery_notebook():
    cells = [
        md("""
# Project Canary — Harvest-Recovery Forecast

**Purpose:** reproduce the complete forecasting workflow in a form the capstone team can run and defend.

**Business question:** using only records known on a review date, how much additional population loss should we expect, and what final recovery does that imply against the 95% goal?

This notebook does not calculate the independent rules-based risk score and does not prove that any input causes recovery to change.
"""),
        md("""
## 1. Define Y, X, and the unit of analysis

- **Y target:** additional population loss after the review date = current percentage alive − completed-cycle recovery proxy.
- **Final output:** current percentage alive − predicted additional loss.
- **Important:** this is the agreed capstone recovery proxy, not a verified harvest-event label.
- **One outcome:** one building in one completed cycle.
- **One training snapshot:** that building's facts known at a selected age. To avoid overweighting long cycles, training retains Days 7, 14, 21, 28, plus the last eligible pre-outcome snapshot.
- **Candidate X inputs:** production age; current survival; recent mortality; weight gap and measurement freshness; and temperature/humidity deviations from approved age bands. Feed is withheld until its recorded unit is confirmed. The compact set excludes building identity, raw inventory size, and algebraic duplicates.
"""),
        code(SETUP),
        code(r"""
from canary import build_modeling_snapshots, build_recovery_training_snapshots, train_outcome_model

daily_snapshots = build_modeling_snapshots(dataset, "recovery")
training_snapshots = build_recovery_training_snapshots(dataset)
coverage = pd.DataFrame({
    "Measure": ["Complete cycles", "Distinct building outcomes", "All leakage-safe daily snapshots", "Balanced decision snapshots"],
    "Count": [training_snapshots["cycle_id"].nunique(), training_snapshots[["cycle_id", "building_id"]].drop_duplicates().shape[0], len(daily_snapshots), len(training_snapshots)],
})
coverage
"""),
        code(r"""
exported = pd.read_csv(MODEL_READY_DIR / "recovery_training.csv")
assert len(exported) == len(training_snapshots)
expected_keys = set(zip(training_snapshots.cycle_id.astype(str), training_snapshots.building_id, training_snapshots.as_of_date.astype(str)))
exported_keys = set(zip(exported.cycle_id.astype(str), exported.building_id, exported.as_of_date.astype(str)))
assert exported_keys == expected_keys
print(f"Export reconciliation passed: the CSV contains the exact {len(exported)} balanced recovery snapshots.")
"""),
        md("""
## 2. Preprocessing and validation

1. Convert the workbook to one canonical building-day row; zone rows are aggregated before modeling.
2. Construct every snapshot with records dated on or before its review date; later records are excluded.
3. Give each building-cycle equal total weight despite repeated checkpoints.
4. Use **nested leave-one-complete-cycle-out cross-validation**: the outer loop tests a completely unseen cycle; the inner loop tunes only within the remaining cycles. Imputation and scaling stay inside those folds.
5. Compare exactly five candidates: age-band remaining-loss baseline, linear regression, Ridge regression, constrained Gradient Boosting, and constrained Extra Trees.
6. A learned recovery model must beat the baseline by at least 10% in cycle-macro MAE, keep positive whole-cycle R², and remain stable. A separate balanced target-side gate controls whether Canary may describe it as a 95% hit/miss classifier.

No random row split is used because rows from the same flock history are related and would leak information across train and test sets.
"""),
        code(r"""
result = train_outcome_model(dataset, "recovery")
manifest = result.manifest
print("Operational recovery method:", manifest["selected_model"])
print("Best learned challenger:", manifest["research_champion"])
print("Champion gates:", manifest["champion_gates"])
print("Model version:", manifest["model_version"])
print("Selected X inputs:")
for feature in manifest["feature_columns"]:
    print(" -", feature)
"""),
        md("## 3. Candidate comparison"),
        code(r"""
comparison = pd.DataFrame([
    {
        "Candidate": entry["model"],
        "Available": entry["available"],
        "Role": "Operational" if entry["model"] == manifest["selected_model"] else "Best learned challenger" if entry["model"] == manifest["research_champion"] else "Compared",
        "MAE (points)": manifest["metrics"].get(entry["model"], {}).get("mae", np.nan) * 100,
        "Cycle-macro MAE (points)": manifest["metrics"].get(entry["model"], {}).get("cycle_macro_mae", np.nan) * 100,
        "RMSE (points)": manifest["metrics"].get(entry["model"], {}).get("rmse", np.nan) * 100,
        "R²": manifest["metrics"].get(entry["model"], {}).get("r2", np.nan),
        "Target-side accuracy": manifest["metrics"].get(entry["model"], {}).get("target_side_accuracy", np.nan),
    }
    for entry in manifest["candidate_registry"]
]).sort_values("Cycle-macro MAE (points)")
comparison.round({"MAE (points)": 2, "Cycle-macro MAE (points)": 2, "RMSE (points)": 2, "R²": 3, "Bias (points)": 2, "Target-side accuracy": 3})
"""),
        code(r"""
cycle_performance = pd.DataFrame.from_dict(manifest["selected_metrics"]["cycle"], orient="index")
cycle_performance.index.name = "Held-out cycle"
cycle_performance.assign(
    mae_points=cycle_performance.mae * 100,
    rmse_points=cycle_performance.rmse * 100,
    bias_points=cycle_performance.bias * 100,
)[["rows", "mae_points", "rmse_points", "bias_points"]].round(2)
"""),
        code(r"""
selected = manifest["selected_metrics"]
print(f"Selected held-out MAE: {selected['mae']*100:.2f} percentage points")
print(f"Selected held-out RMSE: {selected['rmse']*100:.2f} percentage points")
print(f"80% empirical error half-width: ±{selected['uncertainty_half_width_80']*100:.2f} points")
print(f"Target-side accuracy: {selected['target_side_accuracy']:.1%}")
print(f"Majority baseline accuracy: {selected['majority_side_accuracy']:.1%}")
"""),
        md("""
**Interpretation:** the operational method is read directly from the versioned manifest. Selection prioritizes cycle-balanced MAE, positive R², stability across held-out cycles, and simplicity. The current release uses ordinary linear remaining-loss regression because it improves cycle-balanced MAE over the age-band baseline and is as accurate as Ridge while remaining easier to explain. Its R² is still low and its at/above-95% recall is weak, so present it as an experimental continuous estimate with uncertainty—not a probability or guarantee of target attainment.
"""),
        md("## 4. What the selected model relies on"),
        code(r"""
importance = pd.DataFrame(manifest["held_out_permutation_importance"])
importance.head(10).rename(columns={
    "feature": "Input",
    "mean_mae_increase": "Held-out MAE increase",
    "relative_importance_pct": "Relative held-out reliance (%)",
}).round(4)
"""),
        md("""
These are out-of-fold permutation importances from complete unseen cycles. They show predictive reliance and are **associations, not causal effects**.
"""),
        md("## 4B. Held-out SHAP — direction and magnitude"),
        code(r"""
shap_summary = pd.DataFrame(manifest["held_out_shap_importance"])
shap_summary.head(10).rename(columns={
    "feature": "Input",
    "mean_abs_shap_recovery": "Mean absolute SHAP effect",
    "relative_mean_abs_shap_pct": "Relative SHAP reliance (%)",
    "direction_when_value_increases": "General direction when higher",
})[["Input", "Mean absolute SHAP effect", "Relative SHAP reliance (%)", "General direction when higher"]].round(4)
"""),
        md("""
SHAP was calculated on each complete outer held-out cycle for the strongest tree challenger—not on its training fit. It is shown as a non-linear sensitivity analysis and does **not** explain the operational linear model. Because the tree predicts **additional loss**, SHAP signs are negated so positive values mean the feature raised final recovery and negative values mean it lowered final recovery. This explains model behavior; it does not prove that intervening on the feature will cause the predicted change.
"""),
        md("## 5. Day 14 held-out proof and one complete example"),
        code(BOOTSTRAP),
        code(r"""
day14 = pd.DataFrame(manifest["day14_backtest"])
day14["error_points"] = day14["error"] * 100
day14["absolute_error_points"] = day14["absolute_error"] * 100
ci = cycle_bootstrap_mae(day14, "error_points")
metrics = manifest["day14_backtest_metrics"]
print(f"Day 14 building outcomes: {metrics['building_cycles']}")
print(f"Day 14 MAE: {metrics['mae']*100:.2f} points")
print(f"Cycle-bootstrap 95% interval for Day 14 MAE: {ci[0]:.2f} to {ci[1]:.2f} points")
example = day14.iloc[0]
print("\nExample")
print(f"Cycle/building: {example.cycle_id} / {example.building_id}")
print(f"Day 14 held-out projection: {example.predicted:.1%}")
print(f"Last-recorded actual proxy: {example.actual:.1%}")
print(f"Error = projected - actual: {example.error_points:+.2f} percentage points")
day14.head(8)[["cycle_id", "building_id", "predicted", "actual", "error_points"]]
"""),
        md("""
## 6. Why SMOTE or oversampling is not used

- The outcome is continuous regression, while standard SMOTE is designed for classification.
- The scarce item is the number of independent building-cycle outcomes—not the number of spreadsheet rows. Synthetic rows do not create new farms or cycles.
- Interpolating flock records could create biologically implausible combinations and falsely narrow validation error.
- Oversampling before grouped validation could leak the held-out cycle.

**Safer small-data strategy used here:** simple regularized candidates, complete-cycle holdouts, balanced checkpoints, empirical uncertainty, cycle-level bootstrap intervals, and transparent limitations. The strongest improvement is collecting more standardized completed cycles with verified harvest events.
"""),
        md("""
## 7. Defense takeaway

Canary's recovery output is a **nested whole-cycle-validated estimate of the agreed last-recorded recovery proxy**. The refreshed model uses 31 independent outcomes across six completed cycles. Its held-out error is roughly 1–2 percentage points, but its low R² and weak recall of the small number of outcomes at or above 95% require cautious use. Use it to size likely gaps and guide attention, not to claim certainty.
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def weight_notebook():
    cells = [
        md("""
# Project Canary — Day 35 Average-Weight Forecast

**Purpose:** reproduce the complete weight-model workflow in a form the capstone team can run and defend.

**Business question:** given the weights recorded so far, what average building weight should we expect on Day 35, compared with the 1,800 g milestone?
"""),
        md("""
## 1. Define Y, X, and the unit of analysis

- **Y target:** remaining growth = observed Day 35 bodyweight − current checkpoint bodyweight.
- **Final output:** current measured weight + predicted remaining gain.
- **One independent outcome:** one building in one cycle with a Day 35 measurement.
- **Training rows:** up to four checkpoint views of that outcome—Day 7, 14, 21, and 28. These are repeated views, not 124 independent flocks.
- **Candidate X inputs:** measurement day; latest/checkpoint weights; weight ÷ the farm target for that day; recent and cumulative average daily gain; current survival; and environmental-band exposure known by that review date.
- The interpolated 1,800 g target curve is an input/reference. It is **not** the Y label and does not manufacture an actual Day 35 result.
"""),
        code(SETUP),
        code(r"""
from canary import build_day35_feature_rows, build_day35_training_rows, train_day35_weight_baseline

rows = build_day35_training_rows(dataset)
engineered_rows = build_day35_feature_rows(dataset)
outcomes = rows[["cycle_id", "building_id", "actual_day35_weight_kg"]].drop_duplicates()
coverage = pd.DataFrame({
    "Measure": ["Historical cycles", "Distinct Day 35 building outcomes", "Checkpoint training rows", "At/above 1,800 g", "Below 1,800 g"],
    "Count": [rows["cycle_id"].nunique(), len(outcomes), len(rows), outcomes["actual_day35_weight_kg"].ge(1.8).sum(), outcomes["actual_day35_weight_kg"].lt(1.8).sum()],
})
coverage
"""),
        code(r"""
exported = pd.read_csv(MODEL_READY_DIR / "day35_weight_training.csv")
assert len(exported) == len(engineered_rows) == 124
shared = [column for column in engineered_rows.columns if column in exported.columns]
left = engineered_rows[shared].copy().sort_values(["cycle_id", "building_id", "measurement_day"]).reset_index(drop=True)
right = exported[shared].copy().sort_values(["cycle_id", "building_id", "measurement_day"]).reset_index(drop=True)
for column in left.columns:
    if pd.api.types.is_numeric_dtype(left[column]):
        assert np.allclose(left[column], pd.to_numeric(right[column]), equal_nan=True)
    else:
        assert left[column].astype(str).equals(right[column].astype(str))
print("Export reconciliation passed: the CSV contains the exact 124 engineered weight rows.")
"""),
        md("""
## 2. Preprocessing and validation

1. Correct weight rows during workbook standardization and aggregate zone records to one building-day.
2. Keep only observed checkpoint weights and observed Day 35 labels; never fill a missing Day 35 Y from the target curve.
3. At each checkpoint, hide future checkpoint weights.
4. Give each building-cycle equal total weight across its repeated checkpoints.
5. Compare exactly five candidates: historical remaining-gain baseline, checkpoint linear regression, Ridge regression, robust Huber regression, and constrained Gradient Boosting.
6. Use **nested leave-one-complete-cycle-out cross-validation** so tuning, imputation and scaling never see the outer test cycle.
7. Optimize **cycle-macro MAE in kilograms**. A learned model replaces historical remaining gain only if it improves MAE by 10%, keeps positive R², places at least 70% within 200 g, remains stable, and improves target-side classification.
"""),
        code(r"""
manifest = train_day35_weight_baseline(dataset)
print("Operational method:", manifest["selected_model"])
print("Best learned challenger:", manifest["research_champion"])
print("Champion gates:", manifest["champion_gates"])
print("Model version:", manifest["model_version"])
print("Selected X inputs:")
for feature in manifest["feature_columns"]:
    print(" -", feature)
"""),
        md("## 3. Candidate comparison"),
        code(r"""
comparison = pd.DataFrame([
    {
        "Candidate": entry["model"],
        "Available": entry["available"],
        "Role": "Operational fallback" if entry["model"] == manifest["selected_model"] else "Best learned challenger" if entry["model"] == manifest["research_champion"] else "Compared",
        "MAE (g)": manifest["candidate_metrics"].get(entry["model"], {}).get("mae_kg", np.nan) * 1000,
        "Cycle-macro MAE (g)": manifest["candidate_metrics"].get(entry["model"], {}).get("cycle_macro_mae_kg", np.nan) * 1000,
        "RMSE (g)": manifest["candidate_metrics"].get(entry["model"], {}).get("rmse_kg", np.nan) * 1000,
        "R²": manifest["candidate_metrics"].get(entry["model"], {}).get("r2", np.nan),
        "Within 200 g": manifest["candidate_metrics"].get(entry["model"], {}).get("within_200g_rate", np.nan),
        "Target-side accuracy": manifest["candidate_metrics"].get(entry["model"], {}).get("target_side_accuracy", np.nan),
    }
    for entry in manifest["candidate_registry"]
]).sort_values("Cycle-macro MAE (g)")
comparison.round({"MAE (g)": 0, "Cycle-macro MAE (g)": 0, "RMSE (g)": 0, "R²": 3, "Bias (g)": 0, "Within 200 g": 3, "Target-side accuracy": 3})
"""),
        code(r"""
cycle_performance = pd.DataFrame.from_dict(manifest["selected_metrics"]["cycle"], orient="index")
cycle_performance.index.name = "Held-out cycle"
cycle_performance.assign(
    mae_g=cycle_performance.mae_kg * 1000,
    rmse_g=cycle_performance.rmse_kg * 1000,
    bias_g=cycle_performance.bias_kg * 1000,
)[["rows", "mae_g", "rmse_g", "bias_g"]].round(2)
"""),
        code(r"""
selected = manifest["selected_metrics"]
print(f"Selected held-out MAE: {selected['mae_kg']*1000:.0f} g")
print(f"Selected held-out RMSE: {selected['rmse_kg']*1000:.0f} g")
print(f"Held-out bias: {selected['bias_kg']*1000:+.0f} g")
print(f"Within 200 g: {selected['within_200g_rate']:.1%}")
print(f"Correct side of 1,800 g: {selected['target_side_accuracy']:.1%}")
print(f"Historical target hits: {manifest['actual_target_hits']} of {manifest['training_building_cycles']}")
"""),
        md("""
**Interpretation:** no learned challenger cleared all approved gates, so historical remaining gain stays operational. The high target-side percentage partly reflects that 26 of 31 historical outcomes are below 1,800 g; it is not proof of balanced target classification.
"""),
        md("## 3B. Prospective audit on the newly completed 2026-3 cycle"),
        code(r"""
audit = manifest.get("prospective_latest_cycle_audit", {})
audit_metrics = audit.get("metrics", {})
print("Cycle:", audit.get("cycle_id"))
print("This cycle was excluded from training and champion selection.")
print(f"Independent outcomes: {audit.get('independent_outcomes', 0)}")
print(f"Checkpoint forecasts: {audit_metrics.get('rows', 0)}")
print(f"MAE: {audit_metrics.get('mae_kg', np.nan)*1000:.0f} g")
print(f"RMSE: {audit_metrics.get('rmse_kg', np.nan)*1000:.0f} g")
print(f"Within 200 g: {audit_metrics.get('within_200g_rate', np.nan):.1%}")
pd.DataFrame(audit.get("predictions", [])).head(12)
"""),
        md("""
The 2026-3 prospective audit is encouraging but contains only three building outcomes. Its negative R² is caused by the very narrow spread of their actual Day 35 weights, so MAE and the actual-versus-predicted plot are more informative here. This small audit does not replace the six-cycle historical validation.
"""),
        md("## 4. What the selected model relies on"),
        code(r"""
importance = pd.DataFrame(manifest["research_champion_permutation_importance"])
importance.head(10).rename(columns={
    "feature": "Input",
    "mean_mae_increase_kg": "Held-out MAE increase (kg)",
    "relative_importance_pct": "Relative held-out reliance (%)",
}).round(4)
"""),
        md("""
These are held-out permutation importances for the best learned challenger, not causal effects. Weight features are correlated; use the operational forecast and recorded evidence rather than treating importance as an intervention instruction.
"""),
        md("## 5. Day 14 held-out proof and one complete example"),
        code(BOOTSTRAP),
        code(r"""
day14 = pd.DataFrame(manifest["day14_backtest"])
day14["error_g"] = day14["error_kg"] * 1000
ci = cycle_bootstrap_mae(day14, "error_g")
metrics = manifest["day14_backtest_metrics"]
print(f"Day 14 building outcomes: {metrics['building_cycles']}")
print(f"Day 14 MAE: {metrics['mae_kg']*1000:.0f} g")
print(f"Cycle-bootstrap 95% interval for Day 14 MAE: {ci[0]:.0f} to {ci[1]:.0f} g")
example = day14.iloc[0]
print("\nExample")
print(f"Cycle/building: {example.cycle_id} / {example.building_id}")
print(f"Day 14 measured weight: {example.current_weight_kg*1000:.0f} g")
print(f"Projected Day 35 weight: {example.predicted_day35_weight_kg*1000:.0f} g")
print(f"Recorded Day 35 weight: {example.actual_day35_weight_kg*1000:.0f} g")
print(f"Error = projected - recorded: {example.error_g:+.0f} g")
day14.head(8)[["cycle_id", "building_id", "current_weight_kg", "predicted_day35_weight_kg", "actual_day35_weight_kg", "error_g"]]
"""),
        md("""
## 6. Historical remaining gain—why it remains in the comparison

For every eligible training building at a checkpoint age:

`remaining gain = observed Day 35 weight − checkpoint weight`

The baseline averages those gains in the training cycles and adds the average to the current weight. During validation, the held-out cycle is excluded. It remains the live operational method because the learned challengers did not clear every gate.
"""),
        code(r"""
remaining = pd.DataFrame({
    "Checkpoint day": [int(day) for day in manifest["remaining_gain_by_measurement_day_kg"] if int(day) < 35],
    "Average historical remaining gain (g)": [gain * 1000 for day, gain in manifest["remaining_gain_by_measurement_day_kg"].items() if int(day) < 35],
}).sort_values("Checkpoint day")
remaining.round(0)
"""),
        md("""
## 7. Why SMOTE or oversampling is not used

- This is regression, and standard SMOTE is a classification method.
- The 124 checkpoint rows come from only 31 independent building outcomes. Duplicating or synthesizing rows would not create new flocks.
- Synthetic weight paths may violate biological growth and can make validation look falsely precise.
- The class-like target imbalance is reported explicitly instead of hidden.

**Safer strategy:** regularized linear models, simple baselines, nested complete-cycle holdouts, checkpoint/horizon metrics, cycle-level bootstrap intervals, and more standardized Day 35 outcomes over time. A hierarchical model can be considered later, after more cycles—not as a capstone requirement.
"""),
        md("""
## 8. Defense takeaway

Canary's weight output uses **historical remaining gain as the operational fallback**, validated on 31 historical Day 35 building outcomes and their earlier checkpoints. Learned linear and boosted challengers were tested under nested whole-cycle validation but did not clear all replacement gates. This is useful directional decision support, not a guarantee that a building will hit 1,800 g.
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def execute_and_write(notebook, filename: str) -> None:
    namespace = {"__name__": "__main__"}
    execution_count = 0
    previous_cwd = Path.cwd()
    import os
    os.chdir(ROOT)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            source = "".join(cell["source"])
            stream = io.StringIO()
            try:
                with contextlib.redirect_stdout(stream):
                    exec(compile(source, f"{filename}:cell-{execution_count}", "exec"), namespace)
            except Exception as exc:
                cell["outputs"] = [{"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc), "traceback": []}]
                raise
            cell["execution_count"] = execution_count
            output = stream.getvalue()
            if output:
                cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}]
    finally:
        os.chdir(previous_cwd)
    path = OUTPUT / filename
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    execute_and_write(recovery_notebook(), "Project_Canary_Harvest_Recovery_Model.ipynb")
    execute_and_write(weight_notebook(), "Project_Canary_Day35_Weight_Model.ipynb")
