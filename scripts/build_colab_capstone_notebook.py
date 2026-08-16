#!/usr/bin/env python3
"""Build the reader-facing Project Canary Colab modeling notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Project_Canary_End_to_End_Model_Refresh_Colab.ipynb"
PROMPT = (ROOT / "docs" / "PROJECT_CANARY_GEMINI_COLAB_OPTIMIZATION_PROMPT.md").read_text(encoding="utf-8")


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": OUT.name, "provenance": []},
    })
    notebook.cells = [
        md("""
        # Project Canary — End-to-End Model Refresh and Capstone Evidence

        **Purpose:** upload an updated `FARM HARVEST DATA.xlsx`, rebuild the farm-wide harvest-recovery and Day 35 bodyweight evidence, and export manuscript-ready and presentation-ready artifacts.

        This notebook is deliberately conservative. It treats harvest cycle as the primary independence group, keeps the newest cycle locked until model selection is frozen, and never treats target/interpolated bodyweights as observations.
        """),
        md("""
        ## TL;DR

        Run the notebook from top to bottom. The default `balanced` profile compares transparent baselines, regularized regression, tree ensembles, XGBoost, LightGBM and CatBoost under nested whole-cycle LOGO-CV. It then generates EDA, top-five tables, daily and checkpoint performance, uncertainty, SHAP, capstone notes and a downloadable ZIP.

        **Status boundary:** outputs are research/shadow evidence. The notebook never automatically deploys a model into the owner dashboard.
        """),
        md("## 1. Setup and reproducibility"),
        code("""
        from pathlib import Path
        import json, os, subprocess, sys
        IN_COLAB = "google.colab" in sys.modules
        REPO_URL = "https://github.com/jourdandgo/Project_Canary_Capstone.git"
        REPO_REF = os.getenv("CANARY_REPO_REF", "main")  # Pin to a reviewed commit for formal submission.
        if IN_COLAB:
            REPO_ROOT = Path("/content/Project_Canary_Capstone")
            if not REPO_ROOT.exists():
                subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_ROOT)])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_ROOT / "requirements-colab.txt")])
            os.chdir(REPO_ROOT)
        else:
            REPO_ROOT = Path.cwd().resolve()
            if not (REPO_ROOT / "canary").exists():
                raise RuntimeError("Run the notebook from the canary_app/Project Canary repository root.")
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        print("Repository:", REPO_ROOT)
        print("Profile dependency setup complete.")
        """),
        md("### Configuration\nThe newest recorded cycle is locked automatically. Set `AUDIT_CYCLE` explicitly only when the team has frozen a different audit cycle."),
        code("""
        RUN_PROFILE = os.getenv("CANARY_RUN_PROFILE", "balanced")  # balanced | full; smoke is reserved for automated tests
        AUDIT_CYCLE = os.getenv("CANARY_AUDIT_CYCLE", "latest")
        FIXED_SEED = 20260812
        OUTPUT_DIR = REPO_ROOT / "outputs" / "colab_capstone_refresh_latest"
        if IN_COLAB:
            WORKBOOK_PATH = Path("/content/FARM HARVEST DATA.xlsx")
        else:
            WORKBOOK_PATH = (REPO_ROOT.parent / "FARM HARVEST DATA.xlsx").resolve()
        print({"run_profile": RUN_PROFILE, "audit_cycle": AUDIT_CYCLE, "seed": FIXED_SEED, "output": str(OUTPUT_DIR)})
        """),
        md("### Upload the authoritative workbook"),
        code("""
        if IN_COLAB and not WORKBOOK_PATH.exists():
            from google.colab import files
            uploaded = files.upload()
            selected = next((name for name in uploaded if name.lower().endswith(".xlsx")), None)
            if selected is None:
                raise ValueError("Upload FARM HARVEST DATA.xlsx")
            Path(selected).replace(WORKBOOK_PATH)
        if not WORKBOOK_PATH.exists():
            raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")
        print("Authoritative workbook:", WORKBOOK_PATH)
        """),
        md("## 2. Source and data-quality audit"),
        code("""
        from canary.data import load_workbook
        dataset = load_workbook(WORKBOOK_PATH)
        source_preview = {
            "source_rows": dataset.quality.source_rows,
            "canonical_rows": dataset.quality.canonical_rows,
            "building_cycles": int(len(dataset.cycles)),
            "harvest_cycles": int(dataset.cycles.cycle_id.nunique()),
            "weight_measurement_days": dataset.quality.weight_measurement_days,
            "temperature_coverage_pct": dataset.quality.temperature_coverage_pct,
            "humidity_coverage_pct": dataset.quality.humidity_coverage_pct,
            "blocking_errors": list(dataset.quality.blocking_errors),
            "warnings": list(dataset.quality.warnings),
        }
        source_preview
        """),
        md("""
        **What this means:** critical contract failures stop the run. Warnings remain visible because unusual biology, incomplete environment, stale weights and unresolved feed units affect how strongly the results can be interpreted.
        """),
        md("## 3. Run the end-to-end audit, EDA, training and export"),
        code("""
        from canary.colab_workflow import RunConfig, run_capstone_workflow
        config = RunConfig(
            workbook_path=WORKBOOK_PATH,
            output_dir=OUTPUT_DIR,
            run_profile=RUN_PROFILE,
            seed=FIXED_SEED,
            audit_cycle=AUDIT_CYCLE,
        )
        result = run_capstone_workflow(config)
        print("Completed:", result["output_dir"])
        print("Export ZIP:", result["zip_path"])
        """),
        md("## 4. Capstone-focused EDA"),
        code("""
        import pandas as pd
        from IPython.display import Image, display
        eda_summary = pd.read_csv(OUTPUT_DIR / "eda" / "tables" / "eda_summary.csv")
        display(eda_summary)
        """),
        code("""
        for name in ["01_dataset_hierarchy", "02_outcome_distributions", "03_weight_trajectories", "04_survival_trajectories", "07_measurement_coverage", "08_missingness_heatmap"]:
            display(Image(filename=str(OUTPUT_DIR / "eda" / "figures" / f"{name}.png"), width=900))
        display(pd.read_csv(OUTPUT_DIR / "eda" / "figure_catalog.csv"))
        """),
        md("""
        ### Key EDA interpretation

        The many daily rows provide useful trajectory evidence, but they do not create hundreds of independent production events. Building observations share feed, weather, management and timing within harvest cycles. Consequently, whole-cycle holdout validation is the defensible test of performance on a future cycle.
        """),
        md("## 5. Model comparison and champion selection"),
        code("""
        model_tables = {}
        for outcome in ["recovery", "bodyweight"]:
            table = pd.read_csv(OUTPUT_DIR / outcome / "top_five_models.csv").sort_values("rank")
            model_tables[outcome] = table
            print("\\n", outcome.upper())
            display(table[["rank", "candidate", "family", "cycle_macro_rmse", "rmse", "mae", "r2", "bias", "worst_cycle_rmse"]].round(3))
            display(Image(filename=str(OUTPUT_DIR / "capstone_assets" / outcome / "model_comparison.png"), width=850))
        """),
        md("""
        **Selection rule:** cycle-macro RMSE is primary. The one-standard-error rule selects the simplest statistically competitive candidate. A slightly lower-error learned model can remain a shadow model when its gain is too small, unstable, biased or poorly calibrated.
        """),
        md("## 6. Actual versus predicted and residual diagnostics"),
        code("""
        for outcome in ["recovery", "bodyweight"]:
            display(Image(filename=str(OUTPUT_DIR / "capstone_assets" / outcome / "actual_vs_predicted_checkpoints.png"), width=760))
            display(Image(filename=str(OUTPUT_DIR / "capstone_assets" / outcome / "residual_diagnostics.png"), width=950))
            display(Image(filename=str(OUTPUT_DIR / "capstone_assets" / outcome / "fold_stability.png"), width=850))
        """),
        md("## 7. Does accuracy improve as more days become available?"),
        code("""
        display(Image(filename=str(OUTPUT_DIR / "daily_accuracy" / "daily_forecast_accuracy_learning_curve.png"), width=1000))
        for outcome in ["recovery", "bodyweight"]:
            daily = pd.read_csv(OUTPUT_DIR / "daily_accuracy" / f"{outcome}_daily_metrics.csv")
            display(daily.loc[daily.review_day.isin([7, 10, 14, 20, 21, 28, 34])].round(3))
            display(Image(filename=str(OUTPUT_DIR / "daily_accuracy" / f"{outcome}_actual_vs_predicted_by_day.png"), width=1000))
        """),
        md("""
        **Interpretation:** forecasts are available on Day 10, Day 20 and every day from Day 7–34. Days 7/14/21/28 are validation anchors. Recovery updates with new population and mortality evidence. Bodyweight forecasts improve mainly when a genuinely new weight measurement is recorded; a stale measurement is never relabelled as a new daily measurement.
        """),
        md("## 8. Uncertainty and target-side performance"),
        code("""
        for outcome in ["recovery", "bodyweight"]:
            intervals = pd.read_csv(OUTPUT_DIR / outcome / "conformal_predictions.csv")
            coverage = intervals.groupby("review_day")[["covered_80", "covered_90"]].mean()
            print(outcome.upper(), "interval coverage")
            display(coverage.round(3))
            display(Image(filename=str(OUTPUT_DIR / "capstone_assets" / outcome / "target_confusion_matrix.png"), width=620))
        """),
        md("## 9. Feature importance and SHAP"),
        code("""
        for outcome in ["recovery", "bodyweight"]:
            print("\\n", outcome.upper())
            global_shap = pd.read_csv(OUTPUT_DIR / outcome / "held_out_shap_global.csv").sort_values("mean_abs_shap", ascending=False)
            display(global_shap.head(10).round(4))
            for name in ["shap_top10", "shap_beeswarm", "shap_dependence", "shap_local_waterfall"]:
                path = OUTPUT_DIR / "capstone_assets" / outcome / f"{name}.png"
                if path.exists(): display(Image(filename=str(path), width=900))
        """),
        md("""
        **SHAP warning:** these explanations apply to the compatible learned shadow model selected for explanation. A transparent baseline is explained by its explicit remaining-loss or remaining-gain calculation. SHAP and feature importance show predictive association, not causation, and must not be used to prescribe treatment.
        """),
        md("## 10. Capstone defense and manuscript handoff"),
        code("""
        from IPython.display import Markdown
        display(Markdown((OUTPUT_DIR / "CAPSTONE_DEFENSE_NOTES.md").read_text()))
        print("Manuscript tables:", OUTPUT_DIR / "manuscript_tables")
        print("Presentation figures (PNG + SVG):", OUTPUT_DIR / "capstone_assets")
        """),
        md("## 11. Gemini optimization prompt"),
        md(PROMPT),
        md("## 12. Download the complete evidence bundle"),
        code("""
        print(result["zip_path"])
        if IN_COLAB:
            from google.colab import files
            files.download(result["zip_path"])
        """),
        md("""
        ## Takeaways

        Use the exported `CAPSTONE_DEFENSE_NOTES.md`, top-five tables, actual-versus-predicted charts, daily-learning curves and SHAP figures to supplement the manuscript and final presentation. Report limitations directly: few independent harvest cycles, structured missingness, target imbalance and absent management predictors. Failure of complex ML to beat a transparent baseline is a valid and defensible result.
        """),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUT)
    print(OUT)


if __name__ == "__main__":
    build()
