"""Reusable Colab-facing Project Canary model refresh and evidence export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_workbook
from .model_optimization_round import SEED, run_optimization_round


CHECKPOINTS = (7, 14, 21, 28)
CHECKPOINT_COLORS = {7: "#7564E8", 14: "#00A6A6", 21: "#F2A900", 28: "#E45756"}
GREEN, BLUE, GOLD, RED, INK, PALE = "#174C3C", "#377EB8", "#F2A900", "#E45756", "#263238", "#DCEAE4"


@dataclass(frozen=True)
class RunConfig:
    workbook_path: str | Path
    output_dir: str | Path
    run_profile: str = "balanced"
    seed: int = SEED
    audit_cycle: str = "latest"
    checkpoints: tuple[int, ...] = CHECKPOINTS
    daily_start: int = 7
    daily_end: int = 34


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _observed_day35(daily: pd.DataFrame) -> pd.DataFrame:
    return (
        daily.loc[daily["age_day"].eq(35) & daily["weight_measured"].fillna(False), ["cycle_id", "building_id", "bodyweight_kg"]]
        .drop_duplicates(["cycle_id", "building_id"], keep="last")
        .assign(day35_weight_g=lambda frame: frame["bodyweight_kg"] * 1000)
    )


def build_eda(workbook: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_workbook(workbook)
    daily, cycles, targets = dataset.daily.copy(), dataset.cycles.copy(), dataset.targets.copy()
    figures = output / "eda" / "figures"
    tables = output / "eda" / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    day35 = _observed_day35(daily)
    outcomes = cycles.merge(day35[["cycle_id", "building_id", "day35_weight_g"]], on=["cycle_id", "building_id"], how="left")
    outcomes.to_csv(tables / "building_cycle_outcomes.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "Raw building-day rows", "value": int(dataset.quality.source_rows), "interpretation": "Repeated daily records are correlated observations, not independent flocks."},
        {"metric": "Canonical building-day rows", "value": int(dataset.quality.canonical_rows), "interpretation": "One canonical row per building, cycle and day after reconciliation."},
        {"metric": "Building-cycles", "value": int(cycles[["cycle_id", "building_id"]].drop_duplicates().shape[0]), "interpretation": "Each building-cycle supplies one mature outcome."},
        {"metric": "Harvest cycles", "value": int(cycles["cycle_id"].nunique()), "interpretation": "Harvest cycle is the primary independence group used in validation."},
        {"metric": "Observed Day 35 weights", "value": int(len(day35)), "interpretation": "Only actually observed Day 35 weights are supervised labels."},
        {"metric": "Temperature coverage (%)", "value": float(dataset.quality.temperature_coverage_pct), "interpretation": "Incomplete coverage limits stable environmental attribution."},
        {"metric": "Humidity coverage (%)", "value": float(dataset.quality.humidity_coverage_pct), "interpretation": "Missingness is retained as evidence rather than silently filled."},
    ])
    summary.to_csv(tables / "eda_summary.csv", index=False)

    catalog: list[dict[str, str]] = []
    def record(name: str, title: str, meaning: str) -> None:
        catalog.append({"figure": name, "title": title, "what_this_means": meaning})

    hierarchy = pd.DataFrame({"level": ["Daily records", "Building-cycles", "Harvest cycles"], "count": [len(daily), len(cycles), cycles["cycle_id"].nunique()]})
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(hierarchy["level"], hierarchy["count"], color=[BLUE, GREEN, GOLD]);
    for i, value in enumerate(hierarchy["count"]): ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontweight="bold")
    ax.set(title="Canary has many rows but few independent production events", ylabel="Count"); ax.spines[["top", "right"]].set_visible(False); _save(fig, figures / "01_dataset_hierarchy")
    record("01_dataset_hierarchy", "Dataset hierarchy", "The effective validation sample is governed by harvest cycles, not by the number of repeated daily rows.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].hist(outcomes["final_recovery_rate"].dropna() * 100, bins=10, color=GREEN, alpha=.88); axes[0].axvline(95, color=RED, ls="--", label="95% target"); axes[0].set(title="Final harvest recovery", xlabel="Recovery (%)", ylabel="Building-cycles"); axes[0].legend(frameon=False)
    axes[1].hist(outcomes["day35_weight_g"].dropna(), bins=10, color=BLUE, alpha=.88); axes[1].axvline(1800, color=RED, ls="--", label="1,800 g target"); axes[1].set(title="Observed Day 35 bodyweight", xlabel="Weight (g)", ylabel="Building-cycles"); axes[1].legend(frameon=False)
    _save(fig, figures / "02_outcome_distributions"); record("02_outcome_distributions", "Outcome distributions", "Target-side outcomes are imbalanced, so continuous regression and target-side recalls are reported together.")

    observed = daily.loc[daily["weight_measured"].fillna(False)].copy(); observed["weight_g"] = observed["bodyweight_kg"] * 1000
    fig, ax = plt.subplots(figsize=(10, 6))
    for (_, _), group in observed.groupby(["cycle_id", "building_id"]): ax.plot(group["age_day"], group["weight_g"], color=BLUE, alpha=.24, marker="o", ms=2)
    ax.plot(targets["age_day"], targets["target_weight_scaled_g"], color=RED, lw=2.5, label="Approved target curve"); ax.set(title="Observed building growth trajectories", xlabel="Age (days)", ylabel="Observed bodyweight (g)"); ax.legend(frameon=False); _save(fig, figures / "03_weight_trajectories")
    record("03_weight_trajectories", "Observed growth trajectories", "Measurements are sparse and heterogeneous; the target curve is a reference, never an observed flock weight.")

    survival = daily.loc[daily["population"].notna() & daily["beginning_inventory"].gt(0)].copy(); survival["survival_pct"] = survival["population"] / survival["beginning_inventory"] * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (_, _), group in survival.groupby(["cycle_id", "building_id"]): ax.plot(group["age_day"], group["survival_pct"], color=GREEN, alpha=.23)
    ax.axhline(95, color=RED, ls="--", label="95% recovery reference"); ax.set(title="Recorded survival trajectories", xlabel="Age (days)", ylabel="Birds alive (%)"); ax.legend(frameon=False); _save(fig, figures / "04_survival_trajectories")
    record("04_survival_trajectories", "Survival trajectories", "Later survival contains more endpoint information, but remaining losses and cycle drift prevent perfect persistence.")

    target_map = targets.set_index("age_day")["target_weight_scaled_g"]
    observed["target_g"] = observed["age_day"].map(target_map); observed["deficit_g"] = observed["weight_g"] - observed["target_g"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (_, _), group in observed.groupby(["cycle_id", "building_id"]): ax.plot(group["age_day"], group["deficit_g"], color=GOLD, alpha=.30, marker="o", ms=2)
    ax.axhline(0, color=INK, ls="--"); ax.set(title="Observed deficit from the approved target curve", xlabel="Age (days)", ylabel="Observed minus target (g)"); _save(fig, figures / "05_target_deficit")
    record("05_target_deficit", "Target deficit trajectories", "Deficit is a useful biological reference, but predicting Day 35 deficit is mathematically equivalent to predicting weight when the target is fixed.")

    mortality = daily.loc[daily["mortality_daily"].notna()].copy(); mortality["mortality_per_1000"] = mortality["mortality_daily"] / mortality["beginning_inventory"] * 1000
    cycle_mortality = mortality.groupby(["cycle_id", "age_day"], as_index=False)["mortality_per_1000"].mean()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cycle, group in cycle_mortality.groupby("cycle_id"): ax.plot(group["age_day"], group["mortality_per_1000"], label=cycle, alpha=.78)
    ax.set(title="Average daily mortality signal by harvest cycle", xlabel="Age (days)", ylabel="Deaths per 1,000 beginning birds"); ax.legend(frameon=False, ncol=4, fontsize=8); _save(fig, figures / "06_mortality_by_cycle")
    record("06_mortality_by_cycle", "Mortality signal by cycle", "Mortality patterns cluster within production cycles, which is why whole-cycle holdout validation is required.")

    coverage = daily.groupby("cycle_id").agg(temperature_coverage=("temperature_avg_c", lambda s: s.notna().mean()), humidity_coverage=("humidity_avg_pct", lambda s: s.notna().mean()), measured_weight_days=("weight_measured", "sum")).reset_index()
    coverage.to_csv(tables / "measurement_coverage_by_cycle.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].bar(coverage["cycle_id"], coverage["temperature_coverage"] * 100, color=GREEN, label="Temperature"); axes[0].bar(coverage["cycle_id"], coverage["humidity_coverage"] * 100, color=BLUE, alpha=.65, label="Humidity"); axes[0].set(title="Environmental coverage", ylabel="Recorded daily rows (%)"); axes[0].tick_params(axis="x", rotation=35); axes[0].legend(frameon=False)
    axes[1].bar(coverage["cycle_id"], coverage["measured_weight_days"], color=GOLD); axes[1].set(title="Observed weight records", ylabel="Measured building-days"); axes[1].tick_params(axis="x", rotation=35)
    _save(fig, figures / "07_measurement_coverage"); record("07_measurement_coverage", "Measurement coverage", "Coverage differs materially by cycle, limiting the model's ability to separate environment from cycle effects.")

    feature_families = pd.DataFrame({
        "Population": daily["population"].notna(), "Mortality": daily["mortality_daily"].notna(),
        "Observed weight": daily["weight_measured"].fillna(False), "Temperature": daily["temperature_avg_c"].notna(),
        "Humidity": daily["humidity_avg_pct"].notna(),
    }).assign(cycle_id=daily["cycle_id"]).groupby("cycle_id").mean() * 100
    fig, ax = plt.subplots(figsize=(9, 5)); image = ax.imshow(feature_families.T, cmap="YlGn", vmin=0, vmax=100, aspect="auto"); ax.set_xticks(range(len(feature_families)), feature_families.index, rotation=35); ax.set_yticks(range(len(feature_families.columns)), feature_families.columns); ax.set_title("Evidence availability by cycle (%)"); fig.colorbar(image, ax=ax, label="Available rows (%)"); _save(fig, figures / "08_missingness_heatmap")
    record("08_missingness_heatmap", "Missingness by cycle", "Missing evidence is structured by cycle rather than random, so missingness indicators and temporal tests are essential.")

    checkpoint_availability = []
    for day in CHECKPOINTS:
        available = observed.loc[observed["age_day"].le(day)].groupby(["cycle_id", "building_id"])["age_day"].max()
        checkpoint_availability.append({"day": day, "buildings_with_measurement": int(available.notna().sum()), "median_latest_measurement_day": float(available.median())})
    checkpoint_availability = pd.DataFrame(checkpoint_availability); checkpoint_availability.to_csv(tables / "checkpoint_measurement_availability.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(checkpoint_availability["day"].astype(str), checkpoint_availability["buildings_with_measurement"], color=[CHECKPOINT_COLORS[d] for d in CHECKPOINTS]); ax.set(title="Observed weight availability at forecast checkpoints", xlabel="Forecast day", ylabel="Building-cycles with a prior measurement"); _save(fig, figures / "09_checkpoint_availability")
    record("09_checkpoint_availability", "Checkpoint availability", "A Day 10 forecast is possible, but bodyweight accuracy only improves when genuinely new measurements become available.")

    correlation_frame = pd.DataFrame({
        "Recovery": outcomes["final_recovery_rate"] * 100,
        "Day35 weight": outcomes["day35_weight_g"],
        "Beginning inventory": outcomes["beginning_inventory"],
        "Mortality rate": pd.to_numeric(outcomes["Mortality Rate"], errors="coerce"),
    }).corr(numeric_only=True)
    correlation_frame.to_csv(tables / "descriptive_correlations.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.3)); image = ax.imshow(correlation_frame, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(len(correlation_frame)), correlation_frame.columns, rotation=30, ha="right"); ax.set_yticks(range(len(correlation_frame)), correlation_frame.index)
    for i in range(len(correlation_frame)):
        for j in range(len(correlation_frame)): ax.text(j, i, f"{correlation_frame.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Descriptive correlations — not causal effects"); fig.colorbar(image, ax=ax); _save(fig, figures / "10_descriptive_correlations")
    record("10_descriptive_correlations", "Descriptive correlations", "Small-sample associations can be cycle-confounded and must not be interpreted as intervention effects.")

    catalog_frame = pd.DataFrame(catalog); catalog_frame.to_csv(output / "eda" / "figure_catalog.csv", index=False)
    return summary, catalog_frame


def _model_visuals(round_root: Path, outcome: str) -> pd.DataFrame:
    root = round_root / outcome; figures = round_root / "capstone_assets" / outcome
    manifest = json.loads((root / "manifest.json").read_text())
    comparison = pd.read_csv(root / "candidate_comparison.csv").sort_values("rank")
    predictions = pd.read_csv(root / "all_nested_logo_predictions.csv")
    selected_name = manifest["selection"]["selected_candidate"]
    selected = predictions.loc[predictions["candidate"].eq(selected_name)].copy()
    factor = 100.0 if outcome == "recovery" else 1.0; unit = "percentage points" if outcome == "recovery" else "grams"
    selected[["actual_plot", "predicted_plot", "error_plot"]] = selected[["actual", "predicted", "error"]] * factor
    catalog: list[dict[str, str]] = []

    top = comparison.head(10).sort_values("cycle_macro_rmse")
    fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(top["candidate"].str.replace("_", " "), top["cycle_macro_rmse"], color=[GREEN if name == selected_name else PALE for name in top["candidate"]]); ax.set(title=f"{outcome.title()} model comparison on unseen harvest cycles", xlabel=f"Cycle-macro RMSE ({unit})"); _save(fig, figures / "model_comparison"); catalog.append({"figure":"model_comparison","what_this_means":"Lower error is better; champion selection also considers uncertainty, simplicity and stability."})

    fig, ax = plt.subplots(figsize=(7, 6.5));
    for day, group in selected.groupby("review_day"): ax.scatter(group["actual_plot"], group["predicted_plot"], color=CHECKPOINT_COLORS.get(int(day), GREEN), label=f"Day {int(day)}", alpha=.78, edgecolor="white", s=52)
    lo=min(selected["actual_plot"].min(),selected["predicted_plot"].min()); hi=max(selected["actual_plot"].max(),selected["predicted_plot"].max()); ax.plot([lo,hi],[lo,hi],"--",color=INK); ax.set(title="Held-out actual versus predicted",xlabel=f"Actual ({unit})",ylabel=f"Predicted ({unit})"); ax.legend(frameon=False); _save(fig, figures / "actual_vs_predicted_checkpoints"); catalog.append({"figure":"actual_vs_predicted_checkpoints","what_this_means":"Each point was predicted while its entire harvest cycle was excluded from training."})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True); axes[0].scatter(selected["predicted_plot"], selected["error_plot"], c=selected["review_day"], cmap="viridis", alpha=.72); axes[0].axhline(0,color=INK,ls="--"); axes[0].set(title="Residual versus prediction",xlabel=f"Prediction ({unit})",ylabel=f"Actual − predicted ({unit})"); axes[1].hist(selected["error_plot"],bins=12,color=BLUE); axes[1].axvline(0,color=INK,ls="--"); axes[1].set(title="Residual distribution",xlabel=f"Error ({unit})"); _save(fig, figures / "residual_diagnostics"); catalog.append({"figure":"residual_diagnostics","what_this_means":"Residual shape reveals systematic bias, extreme errors and compression toward the mean."})

    cycle_rmse = selected.assign(sq=selected["error_plot"]**2).groupby("cycle_id")["sq"].mean().pow(.5)
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); ax.bar(cycle_rmse.index,cycle_rmse.values,color=BLUE); ax.axhline(cycle_rmse.mean(),color=RED,ls="--",label="Cycle mean"); ax.set(title="Error stability by held-out harvest cycle",ylabel=f"RMSE ({unit})"); ax.tick_params(axis="x",rotation=30); ax.legend(frameon=False); _save(fig, figures / "fold_stability"); catalog.append({"figure":"fold_stability","what_this_means":"Large fold differences indicate cycle drift and explain why row-level validation is optimistic."})

    actual_side = selected["actual_plot"] >= (95 if outcome == "recovery" else 1800); predicted_side = selected["predicted_plot"] >= (95 if outcome == "recovery" else 1800)
    matrix = pd.crosstab(actual_side.map({False:"Below",True:"Meets"}),predicted_side.map({False:"Below",True:"Meets"})).reindex(index=["Below","Meets"],columns=["Below","Meets"],fill_value=0)
    matrix.to_csv(figures / "target_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.5,4.8)); image=ax.imshow(matrix,cmap="YlGn"); ax.set_xticks([0,1],["Predicted below","Predicted meets"]); ax.set_yticks([0,1],["Actual below","Actual meets"])
    for i in range(2):
        for j in range(2): ax.text(j,i,int(matrix.iloc[i,j]),ha="center",va="center",fontsize=15,fontweight="bold")
    ax.set_title("Target-side confusion matrix"); fig.colorbar(image,ax=ax); _save(fig, figures / "target_confusion_matrix"); catalog.append({"figure":"target_confusion_matrix","what_this_means":"Target-side performance is supplementary to continuous RMSE because target classes are imbalanced."})

    shap_global_path=root/"held_out_shap_global.csv"; shap_local_path=root/"held_out_shap_local.csv"
    if shap_global_path.exists() and shap_local_path.exists():
        global_frame=pd.read_csv(shap_global_path).sort_values("mean_abs_shap",ascending=False); local=pd.read_csv(shap_local_path)
        if global_frame.empty or float(global_frame["mean_abs_shap"].max()) <= 1e-12:
            for name, title in (("shap_top10", "No incremental SHAP signal"), ("shap_beeswarm", "No directional SHAP pattern")):
                fig, ax = plt.subplots(figsize=(9, 5.4)); ax.axis("off"); ax.text(.5,.60,title,ha="center",va="center",fontsize=20,fontweight="bold",color=GREEN); ax.text(.5,.42,"The learned residual challenger reproduced the transparent baseline.\nIts fitted correction was zero, so there are no additional feature effects to interpret.",ha="center",va="center",fontsize=12,color=INK); _save(fig,figures/name)
            catalog.append({"figure":"shap_top10","what_this_means":"The learned residual model found no reliable incremental signal beyond the transparent baseline."})
            catalog.append({"figure":"shap_beeswarm","what_this_means":"Zero residual corrections imply no directional learned effect to explain."})
            return pd.DataFrame(catalog)
        top_features=global_frame.head(10)["feature"].tolist(); top_plot=global_frame.head(10).sort_values("mean_abs_shap")
        fig,ax=plt.subplots(figsize=(9,5.8)); ax.barh(top_plot["feature"].str.replace("_"," "),top_plot["mean_abs_shap"]*factor,color=GREEN); ax.set(title="Top 10 held-out SHAP drivers",xlabel=f"Mean |SHAP| ({unit})"); _save(fig,figures/"shap_top10"); catalog.append({"figure":"shap_top10","what_this_means":"Magnitude shows model reliance, not causal importance."})
        swarm=local.loc[local["feature"].isin(top_features)].copy(); order={name:index for index,name in enumerate(reversed(top_features))}
        fig,ax=plt.subplots(figsize=(10,6.5));
        for feature,group in swarm.groupby("feature"):
            y=np.full(len(group),order[feature])+np.linspace(-.18,.18,len(group)); color=group["feature_value"]
            ax.scatter(group["shap_value"]*factor,y,c=color,cmap="coolwarm",alpha=.48,s=18)
        ax.axvline(0,color=INK,ls="--"); ax.set_yticks(range(len(order)),[name.replace("_"," ") for name in reversed(top_features)]); ax.set(title="Held-out SHAP direction and distribution",xlabel=f"SHAP contribution ({unit})"); _save(fig,figures/"shap_beeswarm"); catalog.append({"figure":"shap_beeswarm","what_this_means":"Points right of zero push the compatible learned model upward; directions may differ across cycles."})
        leading=top_features[0]; dependence=local.loc[local["feature"].eq(leading)]
        fig,ax=plt.subplots(figsize=(7.5,5)); ax.scatter(dependence["feature_value"],dependence["shap_value"]*factor,c=dependence["review_day"],cmap="viridis",alpha=.65); ax.axhline(0,color=INK,ls="--"); ax.set(title=f"SHAP dependence — {leading.replace('_',' ')}",xlabel="Transformed feature value",ylabel=f"SHAP contribution ({unit})"); _save(fig,figures/"shap_dependence"); catalog.append({"figure":"shap_dependence","what_this_means":"The shape is a predictive response pattern, not an intervention effect."})
        example=local.groupby(["held_out_cycle","building_id","review_day"],as_index=False)["shap_value"].apply(lambda g:g.abs().sum()).sort_values("shap_value",ascending=False).iloc[0]
        local_example=local.loc[(local["held_out_cycle"].eq(example["held_out_cycle"]))&(local["building_id"].eq(example["building_id"]))&(local["review_day"].eq(example["review_day"]))].copy().sort_values("shap_value",key=lambda s:s.abs(),ascending=False).head(10).sort_values("shap_value")
        fig,ax=plt.subplots(figsize=(9,5.5)); ax.barh(local_example["feature"].str.replace("_"," "),local_example["shap_value"]*factor,color=np.where(local_example["shap_value"]>=0,GREEN,RED)); ax.axvline(0,color=INK,lw=.8); ax.set(title=f"Individual held-out explanation — {example['held_out_cycle']} {example['building_id']} Day {int(example['review_day'])}",xlabel=f"Local SHAP contribution ({unit})"); _save(fig,figures/"shap_local_waterfall"); catalog.append({"figure":"shap_local_waterfall","what_this_means":"This decomposes one held-out learned-model response; it does not prescribe an action."})
    return pd.DataFrame(catalog)


def _write_capstone_notes(round_root: Path, audit: dict[str, Any]) -> Path:
    recovery=json.loads((round_root/"recovery"/"manifest.json").read_text()); weight=json.loads((round_root/"bodyweight"/"manifest.json").read_text())
    r=pd.read_csv(round_root/"recovery"/"candidate_comparison.csv"); w=pd.read_csv(round_root/"bodyweight"/"candidate_comparison.csv")
    def row(frame:pd.DataFrame,name:str)->pd.Series: return frame.loc[frame["candidate"].eq(name)].iloc[0]
    rn=recovery["selection"]["selected_candidate"]; wn=weight["selection"]["selected_candidate"]; rm=row(r,rn); wm=row(w,wn)
    text=f"""# Project Canary capstone modeling evidence

## Executive conclusion

- Recovery champion: **{rn.replace('_',' ')}** — cycle-macro RMSE {rm.cycle_macro_rmse:.2f} percentage points, MAE {rm.mae:.2f}, held-out R² {rm.r2:.3f}.
- Day 35 bodyweight champion: **{wn.replace('_',' ')}** — cycle-macro RMSE {wm.cycle_macro_rmse:.1f} g, MAE {wm.mae:.1f} g, held-out R² {wm.r2:.3f}.
- Validation holds out a complete harvest cycle. Results describe performance on a future production event, not interpolation within a known cycle.
- Daily forecasts are available from Day 7 through Day 34. Days 7, 14, 21 and 28 remain the principal validation anchors.
- Models remain capstone/shadow evidence and are not production-approved.

## How to defend the workflow

The pipeline begins from the authoritative workbook, preserves actually observed weights, aggregates environmental zones to one building-day, creates as-of features, fits all preprocessing inside training folds, and compares transparent baselines with regularized and tree-based models under nested cycle LOGO-CV. The one-standard-error rule prefers the simplest statistically competitive candidate.

## Trust and limitations

The analysis is defensible for a capstone because leakage controls, grouped validation, uncertainty and negative results are explicit. It is not production-ready because the data contain only {len(audit['development_cycles'])} independent development harvest cycles, environmental measurement is incomplete, target-side outcomes are imbalanced and management predictors are missing. SHAP and feature importance show predictive association, not causation.

## Highest-value next data

Verified harvest endpoints; separate deaths/culls/removals; water and unit-confirmed feed; placement weight; strain and sex mix; stocking density; vaccination, medication and illness events; ventilation, CO₂ and ammonia; weight sample size, variability, scale and sampling time.
"""
    path=round_root/"CAPSTONE_DEFENSE_NOTES.md"; path.write_text(text,encoding="utf-8"); return path


def run_capstone_workflow(config: RunConfig) -> dict[str, Any]:
    workbook=Path(config.workbook_path).resolve(); output=Path(config.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    if not workbook.exists(): raise FileNotFoundError(workbook)
    summary,catalog=build_eda(workbook,output)
    manifest=run_optimization_round(workbook,output,seed=config.seed,profile=config.run_profile,audit_cycle=config.audit_cycle)
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"scripts"/"build_daily_capstone_accuracy.py"),"--workbook",str(workbook),"--output",str(output/"daily_accuracy"),"--audit-cycle",str(manifest["locked_audit_cycle"])],check=True)
    visual_catalog=[]
    for outcome in ("recovery","bodyweight"):
        visual_catalog.append(_model_visuals(output,outcome).assign(outcome=outcome))
    pd.concat(visual_catalog,ignore_index=True).to_csv(output/"capstone_assets"/"figure_catalog.csv",index=False)
    notes=_write_capstone_notes(output,manifest)
    manuscript=output/"manuscript_tables"; manuscript.mkdir(exist_ok=True)
    for outcome in ("recovery","bodyweight"):
        shutil.copy2(output/outcome/"top_five_models.csv",manuscript/f"{outcome}_top_five_models.csv")
        shutil.copy2(output/outcome/"checkpoint_metrics.csv",manuscript/f"{outcome}_checkpoint_metrics.csv")
        shutil.copy2(output/"daily_accuracy"/f"{outcome}_daily_metrics.csv",manuscript/f"{outcome}_daily_metrics.csv")
    run_manifest={**manifest,"config":asdict(config),"authoritative_workbook":str(workbook),"source_sha256":_hash(workbook),"eda_figures":int(len(catalog)),"capstone_notes":str(notes),"export_status":"research_shadow_not_production_approved"}
    (output/"COLAB_RUN_MANIFEST.json").write_text(json.dumps(run_manifest,indent=2,default=str),encoding="utf-8")
    archive=shutil.make_archive(str(output.parent/f"{output.name}_EXPORT"),"zip",root_dir=output)
    return {"output_dir":str(output),"zip_path":archive,"manifest":run_manifest,"eda_summary":summary.to_dict("records")}
