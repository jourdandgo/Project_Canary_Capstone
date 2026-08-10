"""Build and execute Project Canary's two reader-facing model notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks"
OUTPUT.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SETUP = r"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "canary").exists():
    ROOT = ROOT.parent
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

**Business question:** using only records known on a review date, what last-recorded harvest-recovery proxy should we expect for this building, compared with the 95% goal?

This notebook does not calculate the independent rules-based risk score and does not prove that any input causes recovery to change.
"""),
        md("""
## 1. Define Y, X, and the unit of analysis

- **Y target:** population on the building's last recorded daily date ÷ beginning population.
- **Important:** this is the agreed capstone recovery proxy, not a verified harvest-event label.
- **One outcome:** one building in one completed cycle.
- **One training snapshot:** that building's facts known at a selected age. To avoid overweighting long cycles, training retains Days 7, 14, 21, 28, plus the last eligible pre-outcome snapshot.
- **Candidate X inputs:** production age; current survival; mortality level and recent trend; feed; latest weight evidence; and available recent temperature/humidity summaries. The selected compact model deliberately excludes building identity and raw inventory size.
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
assert len(exported) == len(training_snapshots) == 122
expected_keys = set(zip(training_snapshots.cycle_id.astype(str), training_snapshots.building_id, training_snapshots.as_of_date.astype(str)))
exported_keys = set(zip(exported.cycle_id.astype(str), exported.building_id, exported.as_of_date.astype(str)))
assert exported_keys == expected_keys
print("Export reconciliation passed: the CSV contains the exact 122 balanced recovery snapshots.")
"""),
        md("""
## 2. Preprocessing and validation

1. Convert the workbook to one canonical building-day row; zone rows are aggregated before modeling.
2. Construct every snapshot with records dated on or before its review date; later records are excluded.
3. Median-impute missing numeric inputs inside each training fold. Ridge also adds missingness indicators and standardizes inputs.
4. Use **leave-one-complete-cycle-out cross-validation**: train on all but one cycle and test on the unseen cycle. This is the appropriate grouped equivalent of K-fold CV here.
5. Compare candidates primarily on **cycle-macro MAE** so each cycle has equal influence. If methods are within 10% of the best, choose the simpler explainable method.

No random row split is used because rows from the same flock history are related and would leak information across train and test sets.
"""),
        code(r"""
result = train_outcome_model(dataset, "recovery")
manifest = result.manifest
print("Champion:", manifest["selected_model"])
print("Model version:", manifest["model_version"])
print("Selected X inputs:")
for feature in manifest["feature_columns"]:
    print(" -", feature)
"""),
        md("## 3. Candidate comparison"),
        code(r"""
comparison = pd.DataFrame([
    {
        "Candidate": name,
        "MAE (points)": metrics["mae"] * 100,
        "Cycle-macro MAE (points)": metrics["cycle_macro_mae"] * 100,
        "RMSE (points)": metrics["rmse"] * 100,
        "Bias (points)": metrics["bias"] * 100,
        "Target-side accuracy": metrics["target_side_accuracy"],
    }
    for name, metrics in manifest["metrics"].items()
]).sort_values("Cycle-macro MAE (points)")
comparison.round({"MAE (points)": 2, "Cycle-macro MAE (points)": 2, "RMSE (points)": 2, "Bias (points)": 2, "Target-side accuracy": 3})
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
**Interpretation:** the compact Ridge is useful as a continuous estimate, but its target-side accuracy does not beat the majority baseline. It should be presented as a prototype projection with uncertainty—not as a proven classifier of 95% target attainment.
"""),
        md("## 4. What the selected model relies on"),
        code(r"""
importance = pd.DataFrame(manifest["global_feature_importance"])
importance.head(10).rename(columns={
    "feature": "Input",
    "coefficient_per_standard_deviation": "Recovery change for +1 SD",
    "absolute_importance_pct": "Share of absolute reliance (%)",
    "direction": "Direction",
}).round(4)
"""),
        md("""
These are standardized Ridge coefficients. They show model reliance after accounting for other inputs; they are **associations, not causal effects**. Missing-value indicators can rank highly because environmental coverage is sparse.
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

Canary's recovery output is a **cycle-held-out Ridge estimate of the agreed last-recorded recovery proxy**. Its held-out MAE is roughly 1–2 percentage points, but it is not yet strong at recognizing the small number of cycles that finish at or above 95%. Use it to rank likely outcome gaps and guide attention, not to claim certainty.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}})


def weight_notebook():
    cells = [
        md("""
# Project Canary — Day 35 Average-Weight Forecast

**Purpose:** reproduce the complete weight-model workflow in a form the capstone team can run and defend.

**Business question:** given the weights recorded so far, what average building weight should we expect on Day 35, compared with the 1,800 g milestone?
"""),
        md("""
## 1. Define Y, X, and the unit of analysis

- **Y target:** observed building average bodyweight on production Day 35.
- **One independent outcome:** one building in one cycle with a Day 35 measurement.
- **Training rows:** up to four checkpoint views of that outcome—Day 7, 14, 21, and 28. These are repeated views, not 124 independent flocks.
- **X inputs for Ridge:** measurement day; latest weight; weight ÷ the interpolated farm target for that day; recent and cumulative average daily gain; and the Day 7/14/21/28 checkpoint weights known by that review date.
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
4. Median-impute missing X values inside the training fold; Ridge inputs are standardized.
5. Use **leave-one-complete-cycle-out cross-validation** so every test prediction comes from a model that never saw that cycle.
6. Optimize **cycle-macro MAE in kilograms**. Choose the simplest model within 5% of the best to avoid rewarding tiny unstable gains.
"""),
        code(r"""
manifest = train_day35_weight_baseline(dataset)
print("Champion:", manifest["selected_model"])
print("Model version:", manifest["model_version"])
print("Selected X inputs:")
for feature in manifest["ridge_parameters"]["features"]:
    print(" -", feature)
"""),
        md("## 3. Candidate comparison"),
        code(r"""
comparison = pd.DataFrame([
    {
        "Candidate": name,
        "MAE (g)": metrics["mae_kg"] * 1000,
        "Cycle-macro MAE (g)": metrics["cycle_macro_mae_kg"] * 1000,
        "RMSE (g)": metrics["rmse_kg"] * 1000,
        "Bias (g)": metrics["bias_kg"] * 1000,
        "Within 200 g": metrics["within_200g_rate"],
        "Target-side accuracy": metrics["target_side_accuracy"],
    }
    for name, metrics in manifest["candidate_metrics"].items()
]).sort_values("Cycle-macro MAE (g)")
comparison.round({"MAE (g)": 0, "Cycle-macro MAE (g)": 0, "RMSE (g)": 0, "Bias (g)": 0, "Within 200 g": 3, "Target-side accuracy": 3})
"""),
        code(r"""
cycle_performance = pd.DataFrame.from_dict(manifest["selected_metrics"]["cycle"], orient="index")
cycle_performance.index.name = "Held-out cycle"
cycle_performance.assign(
    mae_g=cycle_performance.mae_kg * 1000,
    rmse_g=cycle_performance.rmse_kg * 1000,
    bias_g=cycle_performance.bias_kg * 1000,
)[["rows", "mae_g", "rmse_g", "bias_g", "within_200g_rate", "target_side_accuracy"]].round(2)
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
**Interpretation:** Ridge is selected because it has the best cycle-macro MAE and remains simple and explainable. The target-side percentage looks high partly because 26 of 31 historical outcomes are below 1,800 g; the model recognizes below-target outcomes much better than the five hits.
"""),
        md("## 4. What the selected model relies on"),
        code(r"""
importance = pd.DataFrame(manifest["ridge_feature_importance"])
importance.head(10).rename(columns={
    "feature": "Input",
    "coefficient_kg_per_standard_deviation": "Weight change for +1 SD (kg)",
    "absolute_importance_pct": "Share of absolute reliance (%)",
    "direction": "Direction",
}).round(4)
"""),
        md("""
These are standardized Ridge coefficients, not causal effects. Weight features are correlated, so a counterintuitive sign for one variable does not mean management should reverse it; use the overall forecast and recorded operational evidence.
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

The baseline averages those gains in the training cycles and adds the average to the current weight. During validation, the held-out cycle is excluded. It is a transparent benchmark and fallback—not the live champion while Ridge remains better.
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

**Safer strategy:** regularized Ridge, simple baselines, complete-cycle holdouts, checkpoint/horizon metrics, cycle-level bootstrap intervals, and more standardized Day 35 outcomes over time. A hierarchical model can be considered later, after more cycles—not as a capstone requirement.
"""),
        md("""
## 8. Defense takeaway

Canary's weight output is a **cycle-held-out Ridge regression for observed Day 35 average weight**, trained on 31 historical building outcomes and their earlier checkpoints. Overall held-out MAE is about 172 g; at Day 14 it is about 167 g. This is useful directional decision support, not a guarantee that a building will hit 1,800 g.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}})


def execute_and_write(notebook, filename: str) -> None:
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    path = OUTPUT / filename
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    execute_and_write(recovery_notebook(), "Project_Canary_Harvest_Recovery_Model.ipynb")
    execute_and_write(weight_notebook(), "Project_Canary_Day35_Weight_Model.ipynb")
