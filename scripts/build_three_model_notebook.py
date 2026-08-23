"""Create the executed reader-facing notebook for Canary's three-model trial."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "notebooks" / "Project_Canary_Three_Model_Evaluation.ipynb"
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Project Canary: Three-Model Evaluation

## tl;dr

Canary retains **two business outcomes** while evaluating **three engines**. Reconstructed Model 1 remains experimental because it did not beat its transparent recovery baseline. Reconstructed Model 3 is a useful Day 21 shadow benchmark. For the primary checkpoint bodyweight outlook, the transparent historical remaining-gain method remains the champion: more complex learned and THI-inclusive candidates did not produce a stable enough improvement to justify replacing it.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

### Key assumptions

- Thirty-one building-cycles from 2025-2 through 2026-2 form the development cohort.
- The three buildings from 2026-3 stay together as one later-cycle audit.
- Validation holds out complete production cycles; preprocessing and candidate tuning stay inside training folds.
- Model 1 uses Canary's canonical last-recorded-population recovery proxy.
- THI features are predictive candidates, not causal evidence or approved action thresholds.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
EXPERIMENT = ROOT / "outputs" / "three_model_evaluation"
LEGACY = EXPERIMENT / "legacy"
CHECKPOINT = EXPERIMENT / "checkpoint_champion"

model_1 = json.loads((LEGACY / "model_1_manifest.json").read_text())
model_3 = json.loads((LEGACY / "model_3_manifest.json").read_text())
checkpoint = json.loads((CHECKPOINT / "manifest.json").read_text())
comparison = pd.read_csv(CHECKPOINT / "model_comparison.csv")
checkpoint_metrics = pd.DataFrame(checkpoint["checkpoint_metrics"])
"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """cohort = pd.DataFrame([
    {"Evidence": "Development", "Production cycles": len(model_1["development_cycles"]), "Building outcomes": model_1["development_building_cycles"]},
    {"Evidence": "Later-cycle audit", "Production cycles": 1, "Building outcomes": model_1["audit_buildings"]},
])
cohort
"""
        ),
        nbf.v4.new_code_cell(
            """assert model_1["development_building_cycles"] == 31
assert model_3["development_building_cycles"] == 31
assert model_1["audit_cycle"] == model_3["audit_cycle"] == "2026-3"
assert checkpoint["quality_profile"]["development_outcomes"] == 31
assert checkpoint["quality_profile"]["later_cycle_outcomes"] == 3
print("Cohort and audit boundaries verified.")
"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """summary = pd.DataFrame([
    {
        "Engine": "Reconstructed Model 1",
        "Outcome": "Recovery proxy",
        "Review points": "Days 7 and 14",
        "Development MAE": f'{model_1["selected_metrics"]["mae"] * 100:.2f} pp',
        "Later-cycle audit MAE": f'{model_1["audit_metrics"]["mae"] * 100:.2f} pp',
        "Status": model_1["status"],
    },
    {
        "Engine": "Reconstructed Model 3",
        "Outcome": "Day 35 bodyweight",
        "Review points": "Day 21",
        "Development MAE": f'{model_3["selected_metrics"]["mae"]:.0f} g',
        "Later-cycle audit MAE": f'{model_3["audit_metrics"]["mae"]:.0f} g',
        "Status": model_3["status"],
    },
    {
        "Engine": "Checkpoint champion",
        "Outcome": "Day 35 bodyweight",
        "Review points": "Days 7, 14, 21 and 28",
        "Development MAE": f'{checkpoint["champion_metrics"]["mae_g"]:.0f} g',
        "Later-cycle audit MAE": f'{checkpoint["later_cycle_audit_metrics"]["mae_g"]:.0f} g',
        "Status": "pilot-ready baseline",
    },
])
summary
"""
        ),
        nbf.v4.new_code_cell(
            """plot = checkpoint_metrics[["review_day", "mae_g"]].copy()
model3_day21 = model_3["selected_metrics"]["mae"]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(plot["review_day"], plot["mae_g"], marker="o", linewidth=2.5, color="#286245", label="Checkpoint champion")
ax.scatter([21], [model3_day21], s=90, color="#c7a600", label="Model 3 (Day 21)", zorder=3)
for x, y in zip(plot["review_day"], plot["mae_g"]):
    ax.annotate(f"{y:.0f} g", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
ax.set(title="Day 35 bodyweight error falls as new measured checkpoints arrive", xlabel="Review day", ylabel="Complete-cycle held-out MAE (g)", xticks=[7, 14, 21, 28])
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_code_cell(
            """comparison[["candidate", "family", "feature_set", "cycle_macro_rmse_g", "cycle_macro_mae_g", "r2"]].head(10)
"""
        ),
        nbf.v4.new_markdown_cell(
            """### THI finding

The original Model 1 and Model 3 schemas retain maximum daily/cycle THI and THI stress-day features. The checkpoint tournament also tested wet-bulb-based THI history, maximum THI, stress-day share, and compound heat-humidity exposure. THI-inclusive candidate families did not outperform the selected transparent checkpoint baseline. This does not show that THI is biologically unimportant; it shows that the current sample and environmental coverage do not support giving THI additional predictive or operating authority yet.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Keep Model 1 visible only as an **experimental** recovery-proxy outlook.
2. Keep Model 3 as a **Day 21 shadow benchmark**, not the main bodyweight forecast.
3. Use historical remaining gain as the provisional checkpoint bodyweight method at Days 7, 14, 21 and 28.
4. Continue collecting standardized weight and environment records. Re-evaluate the learned Extra Trees challenger and THI-inclusive candidates after additional complete cycles.
5. Keep all forecasts independent from the observed-condition risk score.
"""
        ),
    ]
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
