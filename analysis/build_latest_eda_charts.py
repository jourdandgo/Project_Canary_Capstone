"""Build presentation-ready EDA charts from the refreshed canonical workbook."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from canary import complete_cycle_ids, load_workbook


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "latest_eda"
DATA = ROOT / "data" / "FARM HARVEST DATA.xlsx"
GREEN = "#174f3b"
LIME = "#91c529"
RED = "#c94f3d"
GOLD = "#d9a928"
MUTED = "#708078"


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    ds = load_workbook(DATA)
    cycles = complete_cycle_ids(ds)
    outcomes = ds.cycles.loc[ds.cycles["cycle_id"].isin(cycles)].copy()
    outcomes["recovery_pct"] = outcomes["ending_inventory"] / outcomes["beginning_inventory"] * 100
    weights = ds.daily.loc[ds.daily["weight_measured"], ["cycle_id", "building_id", "age_day", "bodyweight_kg"]].copy()
    wide = weights.pivot_table(index=["cycle_id", "building_id"], columns="age_day", values="bodyweight_kg", aggfunc="last").reset_index()
    wide = wide.rename(columns={7:"day7",14:"day14",21:"day21",28:"day28",35:"day35"})
    evidence = outcomes.merge(wide, on=["cycle_id", "building_id"], how="left")

    cycle_recovery = outcomes.groupby("cycle_id").apply(
        lambda g: g["ending_inventory"].sum() / g["beginning_inventory"].sum() * 100,
        include_groups=False,
    ).reset_index(name="recovery_pct")
    plt.figure(figsize=(10, 5.4))
    ax = sns.lineplot(data=cycle_recovery, x="cycle_id", y="recovery_pct", marker="o", linewidth=3, color=GREEN)
    ax.axhline(95, linestyle="--", color=RED, label="95% goal")
    ax.set(xlabel="Harvest cycle", ylabel="Inventory-weighted recovery proxy (%)", title="Recovery varies across completed cycles")
    ax.legend(frameon=False)
    save("01_recovery_by_cycle.png")

    weight_cycle = evidence.groupby("cycle_id", as_index=False)["day35"].mean()
    plt.figure(figsize=(10, 5.4))
    ax = sns.lineplot(data=weight_cycle, x="cycle_id", y="day35", marker="o", linewidth=3, color=GREEN)
    ax.axhline(1.8, linestyle="--", color=RED, label="1,800 g milestone")
    ax.set(xlabel="Harvest cycle", ylabel="Recorded Day 35 average weight (kg)", title="Day 35 weight attainment is also inconsistent")
    ax.legend(frameon=False)
    save("02_day35_weight_by_cycle.png")

    building = outcomes.groupby("building_id", as_index=False).agg(mean_recovery=("recovery_pct", "mean"), n=("cycle_id", "size")).sort_values("mean_recovery")
    plt.figure(figsize=(9.5, 5.8))
    ax = sns.barplot(data=building, x="mean_recovery", y="building_id", color=GREEN)
    ax.axvline(95, linestyle="--", color=RED)
    ax.set(xlabel="Mean historical recovery proxy (%)", ylabel="Building", title="Building results differ, but every building has limited history")
    for patch, n in zip(ax.patches, building["n"]):
        ax.text(patch.get_width()+0.08, patch.get_y()+patch.get_height()/2, f"n={n}", va="center", fontsize=9)
    save("03_recovery_by_building.png")

    paired = evidence.dropna(subset=["day14", "day35"]).copy()
    paired[["day14", "day35"]] = paired[["day14", "day35"]].astype(float)
    plt.figure(figsize=(8.5, 6.2))
    ax = sns.regplot(data=paired, x="day14", y="day35", scatter_kws={"s":65,"alpha":0.78,"color":GREEN}, line_kws={"color":GOLD,"linewidth":2.5})
    ax.axvline(0.38, linestyle="--", color=MUTED)
    ax.axhline(1.8, linestyle="--", color=RED)
    ax.set(xlabel="Recorded Day 14 weight (kg)", ylabel="Recorded Day 35 weight (kg)", title="Higher Day 14 weight is associated with higher Day 35 weight")
    save("04_day14_vs_day35.png")

    paired_r = evidence.dropna(subset=["day14", "recovery_pct"]).copy()
    paired_r[["day14", "recovery_pct"]] = paired_r[["day14", "recovery_pct"]].astype(float)
    plt.figure(figsize=(8.5, 6.2))
    ax = sns.regplot(data=paired_r, x="day14", y="recovery_pct", scatter_kws={"s":65,"alpha":0.78,"color":GREEN}, line_kws={"color":GOLD,"linewidth":2.5})
    ax.axvline(0.38, linestyle="--", color=MUTED)
    ax.axhline(95, linestyle="--", color=RED)
    ax.set(xlabel="Recorded Day 14 weight (kg)", ylabel="Final recovery proxy (%)", title="Day 14 growth also has a moderate recovery association")
    save("05_day14_vs_recovery.png")

    daily = ds.daily.loc[ds.daily["cycle_id"].isin(cycles)].copy()
    daily["mortality_per_1000"] = daily["mortality_daily"] / daily["beginning_inventory"] * 1000
    env = daily.groupby(["cycle_id","building_id"], as_index=False).agg(
        mean_temp=("temperature_avg_c","mean"),
        mean_humidity=("humidity_avg_pct","mean"),
        temp_range=("temperature_range_c","mean"),
        mortality_per_1000=("mortality_per_1000","mean"),
    ).merge(outcomes[["cycle_id","building_id","recovery_pct"]], on=["cycle_id","building_id"])
    corr = env[["mean_temp","mean_humidity","temp_range","mortality_per_1000","recovery_pct"]].corr()[["recovery_pct"]].drop("recovery_pct").reset_index()
    corr.columns = ["signal","correlation"]
    corr["signal"] = corr["signal"].map({"mean_temp":"Mean temperature","mean_humidity":"Mean humidity","temp_range":"Daily temperature range","mortality_per_1000":"Mean daily mortality/1,000"})
    corr = corr.sort_values("correlation")
    plt.figure(figsize=(9.5, 5.5))
    ax = sns.barplot(data=corr, x="correlation", y="signal", palette=[RED if v < 0 else LIME for v in corr["correlation"]], hue="signal", legend=False)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set(xlabel="Pearson correlation with recovery proxy", ylabel="", title="Operational signals are associated with recovery—but do not prove cause")
    save("06_operational_associations.png")

    targets = ds.targets.set_index("age_day")["target_weight_kg"]
    observed = weights.copy()
    observed["target"] = observed["age_day"].map(targets)
    observed["attainment_pct"] = observed["bodyweight_kg"] / observed["target"] * 100
    checkpoint = observed.groupby("age_day", as_index=False).agg(n=("bodyweight_kg","size"), mean_attainment=("attainment_pct","mean"), median_attainment=("attainment_pct","median"))
    checkpoint = checkpoint.loc[checkpoint["age_day"].isin([7,14,21,28,35])]
    plt.figure(figsize=(9.5, 5.5))
    ax = sns.lineplot(data=checkpoint, x="age_day", y="mean_attainment", marker="o", linewidth=3, color=GREEN)
    ax.axhline(100, linestyle="--", color=RED)
    ax.set(xlabel="Production day", ylabel="Mean weight attainment vs age target (%)", title="Average growth remains below the revised in-house curve")
    ax.set_xticks([7,14,21,28,35])
    save("07_target_attainment_by_day.png")

    summary = {
        "completed_cycles": len(cycles),
        "building_outcomes": int(len(outcomes)),
        "recovery_by_cycle": cycle_recovery.to_dict(orient="records"),
        "weight_by_cycle": weight_cycle.to_dict(orient="records"),
        "day14_day35_pearson": float(paired["day14"].corr(paired["day35"])),
        "day14_recovery_pearson": float(paired_r["day14"].corr(paired_r["recovery_pct"])),
        "day14_target_hits": int((paired["day14"] >= 0.38).sum()),
        "day35_target_hits": int((paired["day35"] >= 1.8).sum()),
        "checkpoint_attainment": checkpoint.to_dict(orient="records"),
        "operational_correlations": corr.to_dict(orient="records"),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    evidence.to_csv(OUT / "eda_building_outcomes.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
