"""Build and execute the reader-facing Gompertz comparison notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "Gompertz_Fourth_Model_Evaluation.ipynb"


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# tl;dr

The checkpoint-anchored Gompertz candidate reached **124.8 g held-out MAE**, compared with **127.1 g** for historical remaining gain. The 2.3 g pooled difference is too small to be operationally meaningful, and a paired cycle bootstrap includes no improvement. On the untouched 2026-3 audit, Gompertz was slightly weaker (**84.6 g versus 77.6 g**). The checkpoint model should therefore remain the provisional default; Gompertz is a research comparison only.

When the current weight is blank, the Gompertz curve returns the same population-level Day 35 estimate for every building. That is a generic prior, not a building-specific forecast.
"""
    ),
    markdown(
        """
## Context & Methods

**Question.** Can a Gompertz growth curve improve the Day 35 bodyweight outlook, including when a checkpoint weight is missing?

**Design.** The experiment uses 31 development building-cycles from 2025-2 through 2026-2. Each outer fold holds out one complete production cycle. The three 2026-3 buildings remain untouched until the method is frozen. The candidate fits a robust population Gompertz curve on training weights from Days 7, 14, 21, 28, and 35.

With a measured checkpoint weight, the candidate preserves the building's current deviation and adds the curve-implied remaining gain. Without a measured weight, only the population curve endpoint is available.
"""
    ),
    code(
        """
from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "canary").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_gompertz_candidate import evaluate

OUTPUT = ROOT / "analysis" / "gompertz_evaluation"
results = evaluate(OUTPUT)
print(f"Source: {results['source']}")
print(f"Development building-cycles: {results['development_building_cycles']}")
print(f"Untouched audit: {results['audit_buildings']} buildings from {results['audit_cycle']}")
"""
    ),
    markdown(
        """
## Data

The source is the canonical farm workbook. The target is recorded average Day 35 bodyweight. The candidate uses only bodyweights available within each training fold; 2026-3 does not participate in fitting or model choice.
"""
    ),
    code(
        """
assert results["development_building_cycles"] == 31
assert results["audit_cycle"] == "2026-3"
assert results["audit_buildings"] == 3
assert results["development_metrics"]["rows"] == 124
assert all(item["rows"] == 31 for item in results["checkpoint_metrics"])
print("Cohort and checkpoint assertions passed.")
"""
    ),
    markdown("## Results"),
    code(
        """
checkpoint = results["existing_models"]["checkpoint_bodyweight"]
checkpoint_audit = results["existing_models"]["checkpoint_bodyweight_audit"]
model3 = results["existing_models"]["model_3_day21_bodyweight"]
model3_audit = results["existing_models"]["model_3_day21_bodyweight_audit"]
gompertz = results["development_metrics"]
gompertz_audit = results["audit_metrics"]

comparison = pd.DataFrame([
    {"Engine": "Model 3 — XGBoost", "Forecast availability": "Day 21 only", "Development MAE": f"{model3['mae']:.1f} g", "2026-3 audit MAE": f"{model3_audit['mae']:.1f} g"},
    {"Engine": "Checkpoint — historical remaining gain", "Forecast availability": "Days 7, 14, 21, 28", "Development MAE": f"{checkpoint['mae_g']:.1f} g", "2026-3 audit MAE": f"{checkpoint_audit['mae_g']:.1f} g"},
    {"Engine": "Gompertz — anchored remaining gain", "Forecast availability": "Days 7, 14, 21, 28", "Development MAE": f"{gompertz['mae_g']:.1f} g", "2026-3 audit MAE": f"{gompertz_audit['mae_g']:.1f} g"},
])
comparison
"""
    ),
    code(
        """
gompertz_by_day = pd.DataFrame(results["checkpoint_metrics"])[["review_day", "mae_g"]].rename(columns={"mae_g": "Gompertz MAE (g)"})
checkpoint_by_day = pd.DataFrame(results["existing_models"]["checkpoint_bodyweight"].get("checkpoint_metrics", []))
stored_checkpoint = pd.read_csv(ROOT / "models" / "three_model" / "checkpoint_champion" / "checkpoint_metrics.csv")
stored_checkpoint = stored_checkpoint[["review_day", "mae_g"]].rename(columns={"mae_g": "Checkpoint MAE (g)"})
gompertz_by_day.merge(stored_checkpoint, on="review_day")
"""
    ),
    code(
        """
paired = results["paired_comparison_with_checkpoint"]
pd.Series({
    "Cycle-macro MAE difference, Gompertz minus checkpoint (g)": paired["point_difference_g"],
    "Paired bootstrap 95% low (g)": paired["ci95_low_g"],
    "Paired bootstrap 95% high (g)": paired["ci95_high_g"],
    "Bootstrap probability Gompertz is lower-error": paired["probability_gompertz_lower_mae"],
})
"""
    ),
    code(
        """
pd.Series(results["blank_weight_fallback"])
"""
    ),
    markdown(
        """
## Takeaways

1. The anchored Gompertz candidate is numerically close to the checkpoint method, not decisively better.
2. Its small development advantage is within paired uncertainty, and it is weaker on the later 2026-3 audit.
3. The fitted mature-weight asymptote is weakly identified because the records end at Day 35; one outer fold reaches the allowed upper bound. This makes biological interpretation of the parameters unsafe.
4. A blank checkpoint weight removes the building-specific anchor. The output then becomes a generic cohort estimate and should be labelled as such, not shown as a normal building forecast.
5. Do not change the Canary app based on this experiment. Retain historical remaining gain as the provisional checkpoint model and keep Gompertz available only for research comparison.
"""
    ),
]

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3"}
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, DESTINATION)

client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, DESTINATION)
print(DESTINATION)
