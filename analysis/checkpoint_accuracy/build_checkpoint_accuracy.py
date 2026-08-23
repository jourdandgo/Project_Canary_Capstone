"""Build source-backed checkpoint-accuracy evidence for Project Canary.

This analysis follows the application's deployed routing:

* Harvest recovery: Model 1 refreshes through Day 14, then holds.
* Day 35 bodyweight: Model 2 is used through Day 14; Model 3 refreshes
  through Day 21, then holds.

Historical values are cycle-grouped LOGO-CV errors. Prospective values are
recomputed from the exact deployment bundle used by the Canary dashboard and
the untouched 2026-3 outcome labels. Held forecasts are described as holds;
they are never plotted as if a new prediction had been made.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent
TRISH = PROJECT_ROOT / "capstone_FINAL_v18" / "artifacts"
TRISH_MASTER = PROJECT_ROOT / "capstone_FINAL_v18" / "data" / "intermediate" / "master_dataset.csv"
DEPLOYED_SNAPSHOT = APP_ROOT / "models" / "trish_v18" / "prediction_snapshot.csv"
OUTPUT = APP_ROOT / "analysis" / "checkpoint_accuracy"


def _mae(frame: pd.DataFrame, day: int, column: str) -> float:
    values = frame.loc[frame["prediction_day"].eq(day), column]
    if values.empty:
        raise ValueError(f"No values found for Day {day} in {column}")
    return float(values.mean())


def build_summary() -> pd.DataFrame:
    recovery_cv = pd.read_csv(
        TRISH
        / "model_1_harvest_recovery_1_to_14"
        / "extra_trees_logocv_predictions.csv"
    )
    weight_m2_cv = pd.read_csv(
        TRISH
        / "model_2_bodyweight_1_to_14"
        / "catboost_logocv_predictions.csv"
    )
    weight_m3_cv = pd.read_csv(
        TRISH
        / "model_3_bodyweight_1_to_21"
        / "model_b_champion_logocv_predictions.csv"
    )
    deployment = pd.read_csv(DEPLOYED_SNAPSHOT)
    master = pd.read_csv(TRISH_MASTER)
    actual_recovery = master.loc[
        master["harvest_cycle"].astype(str).eq("2026-3"),
        ["bldg", "final_harvest_recovery"],
    ].drop_duplicates()
    actual_weight = master.loc[
        master["harvest_cycle"].astype(str).eq("2026-3")
        & master["age"].eq(35),
        ["bldg", "bodyweight_g"],
    ].drop_duplicates()
    recovery_holdout = deployment.loc[deployment["model_id"].eq("model_1")].merge(
        actual_recovery, on="bldg", validate="many_to_one"
    )
    weight_m2_holdout = deployment.loc[deployment["model_id"].eq("model_2")].merge(
        actual_weight, on="bldg", validate="many_to_one"
    )
    weight_m3_holdout = deployment.loc[deployment["model_id"].eq("model_3")].merge(
        actual_weight, on="bldg", validate="many_to_one"
    )

    recovery_cv["absolute_error"] = (
        recovery_cv["actual"] - recovery_cv["predicted"]
    ).abs() * 100
    recovery_holdout["absolute_error"] = (
        recovery_holdout["prediction"]
        - recovery_holdout["final_harvest_recovery"]
    ).abs() * 100
    weight_m2_cv["absolute_error"] = (
        weight_m2_cv["actual_g"] - weight_m2_cv["predicted_g"]
    ).abs()
    weight_m2_holdout["absolute_error"] = (
        weight_m2_holdout["prediction"] - weight_m2_holdout["bodyweight_g"]
    ).abs()
    weight_m3_cv["absolute_error"] = (
        weight_m3_cv["actual"] - weight_m3_cv["predicted"]
    ).abs()
    weight_m3_holdout["absolute_error"] = (
        weight_m3_holdout["prediction"] - weight_m3_holdout["bodyweight_g"]
    ).abs()

    recovery_cv_day7 = _mae(recovery_cv, 7, "absolute_error")
    recovery_cv_day14 = _mae(recovery_cv, 14, "absolute_error")
    recovery_holdout_day7 = _mae(recovery_holdout, 7, "absolute_error")
    recovery_holdout_day14 = _mae(recovery_holdout, 14, "absolute_error")
    weight_cv_day7 = _mae(weight_m2_cv, 7, "absolute_error")
    weight_cv_day14 = _mae(weight_m2_cv, 14, "absolute_error")
    weight_cv_day21 = _mae(weight_m3_cv, 21, "absolute_error")
    weight_holdout_day7 = _mae(weight_m2_holdout, 7, "absolute_error")
    weight_holdout_day14 = _mae(weight_m2_holdout, 14, "absolute_error")
    weight_holdout_day21 = _mae(weight_m3_holdout, 21, "absolute_error")

    rows: list[dict[str, object]] = []
    for evidence, sample, outcome, unit, values, models, statuses in (
        (
            "Historical cycle-grouped LOGO-CV",
            "34 building-cycles",
            "Harvest recovery",
            "percentage points",
            [recovery_cv_day7, recovery_cv_day14, None, None],
            ["M1", "M1", "M1", "M1"],
            ["Recalculated", "Recalculated", "Held from Day 14", "Held from Day 14"],
        ),
        (
            "Prospective 2026-3 holdout",
            "3 buildings",
            "Harvest recovery",
            "percentage points",
            [recovery_holdout_day7, recovery_holdout_day14, None, None],
            ["M1", "M1", "M1", "M1"],
            ["Recalculated", "Recalculated", "Held from Day 14", "Held from Day 14"],
        ),
        (
            "Historical cycle-grouped LOGO-CV",
            "34 building-cycles",
            "Day 35 bodyweight",
            "grams",
            [weight_cv_day7, weight_cv_day14, weight_cv_day21, None],
            ["M2", "M2", "M3", "M3"],
            ["Recalculated", "Recalculated", "Recalculated", "Held from Day 21"],
        ),
        (
            "Prospective 2026-3 holdout",
            "3 buildings",
            "Day 35 bodyweight",
            "grams",
            [weight_holdout_day7, weight_holdout_day14, weight_holdout_day21, None],
            ["M2", "M2", "M3", "M3"],
            ["Recalculated", "Recalculated", "Recalculated", "Held from Day 21"],
        ),
    ):
        for day, value, model, status in zip(
            [7, 14, 21, 28], values, models, statuses, strict=True
        ):
            rows.append(
                {
                    "evidence_set": evidence,
                    "sample": sample,
                    "outcome": outcome,
                    "checkpoint_day": day,
                    "model": model,
                    "forecast_status": status,
                    "mean_absolute_error": value,
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def build_holdout_detail() -> pd.DataFrame:
    deployment = pd.read_csv(DEPLOYED_SNAPSHOT)
    master = pd.read_csv(TRISH_MASTER)
    actual_recovery = master.loc[
        master["harvest_cycle"].astype(str).eq("2026-3"),
        ["bldg", "final_harvest_recovery"],
    ].drop_duplicates()
    actual_weight = master.loc[
        master["harvest_cycle"].astype(str).eq("2026-3")
        & master["age"].eq(35),
        ["bldg", "bodyweight_g"],
    ].drop_duplicates()
    recovery = deployment.loc[deployment["model_id"].eq("model_1")].merge(
        actual_recovery, on="bldg", validate="many_to_one"
    )
    weight_m2 = deployment.loc[deployment["model_id"].eq("model_2")].merge(
        actual_weight, on="bldg", validate="many_to_one"
    )
    weight_m3 = deployment.loc[deployment["model_id"].eq("model_3")].merge(
        actual_weight, on="bldg", validate="many_to_one"
    )

    rows: list[dict[str, object]] = []
    for building in sorted(recovery["bldg"].unique()):
        for display_day in (7, 14, 21, 28):
            recovery_day = 7 if display_day == 7 else 14
            recovery_row = recovery.loc[
                recovery["bldg"].eq(building)
                & recovery["prediction_day"].eq(recovery_day)
            ].iloc[0]
            if display_day in (7, 14):
                weight_day = display_day
                weight_model = "M2"
                weight_row = weight_m2.loc[
                    weight_m2["bldg"].eq(building)
                    & weight_m2["prediction_day"].eq(weight_day)
                ].iloc[0]
            else:
                weight_day = 21
                weight_model = "M3"
                weight_row = weight_m3.loc[
                    weight_m3["bldg"].eq(building)
                    & weight_m3["prediction_day"].eq(weight_day)
                ].iloc[0]
            rows.append(
                {
                    "building": building,
                    "display_day": display_day,
                    "recovery_evidence_day": recovery_day,
                    "recovery_model": "M1",
                    "recovery_forecast_status": (
                        "Recalculated" if display_day in (7, 14) else "Held from Day 14"
                    ),
                    "actual_recovery_pct": float(recovery_row["final_harvest_recovery"]) * 100,
                    "predicted_recovery_pct": float(recovery_row["prediction"]) * 100,
                    "recovery_absolute_error_pp": abs(
                        float(recovery_row["prediction"])
                        - float(recovery_row["final_harvest_recovery"])
                    ) * 100,
                    "weight_evidence_day": weight_day,
                    "weight_model": weight_model,
                    "weight_forecast_status": (
                        "Held from Day 21" if display_day == 28 else "Recalculated"
                    ),
                    "actual_day35_weight_g": float(weight_row["bodyweight_g"]),
                    "predicted_day35_weight_g": float(weight_row["prediction"]),
                    "weight_absolute_error_g": abs(
                        float(weight_row["prediction"])
                        - float(weight_row["bodyweight_g"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
    background = "#073226"
    panel = "#0d4031"
    canary = "#b8ef32"
    cyan = "#4dd6c4"
    white = "#f7fbf8"
    muted = "#b8c9c2"
    grid = "#376354"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), facecolor=background)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.77, bottom=0.22, wspace=0.22)
    fig.suptitle(
        "Later evidence improved bodyweight accuracy on average—but not recovery accuracy",
        x=0.07,
        y=0.94,
        ha="left",
        color=white,
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.87,
        "Mean absolute error at Canary's operational checkpoints · lower is better",
        color=muted,
        fontsize=12.5,
    )

    panels = [
        (axes[0], "Harvest recovery", "Mean absolute error (percentage points)", "percentage points"),
        (axes[1], "Day 35 bodyweight", "Mean absolute error (grams)", "grams"),
    ]
    for axis, outcome, ylabel, unit in panels:
        axis.set_facecolor(panel)
        subset = summary.loc[summary["outcome"].eq(outcome)]
        for evidence, color, label in (
            ("Historical cycle-grouped LOGO-CV", canary, "Historical LOGO-CV"),
            ("Prospective 2026-3 holdout", cyan, "2026-3 prospective holdout"),
        ):
            series = subset.loc[subset["evidence_set"].eq(evidence)].sort_values(
                "checkpoint_day"
            )
            fresh = series.loc[series["forecast_status"].eq("Recalculated")]
            axis.plot(
                fresh["checkpoint_day"],
                fresh["mean_absolute_error"],
                color=color,
                marker="o",
                linewidth=2.7,
                markersize=8,
                label=label,
            )
            for _, row in fresh.iterrows():
                value = float(row["mean_absolute_error"])
                text = f"{value:.2f}" if unit == "percentage points" else f"{value:.0f} g"
                axis.annotate(
                    text,
                    (row["checkpoint_day"], value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                )
        axis.set_title(outcome, color=white, fontsize=15, fontweight="bold", loc="left", pad=14)
        axis.set_xlabel("Review checkpoint", color=muted, fontsize=10.5, labelpad=10)
        axis.set_ylabel(ylabel, color=muted, fontsize=10.5, labelpad=10)
        axis.set_xticks([7, 14, 21, 28], ["Day 7", "Day 14", "Day 21", "Day 28"])
        axis.tick_params(colors=muted, labelsize=10)
        axis.grid(axis="y", color=grid, alpha=0.45, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.legend(frameon=False, labelcolor=white, fontsize=9.5, loc="best")

    axes[0].text(
        0.98,
        0.95,
        "Day 21–28: M1 held from Day 14",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=muted,
        fontsize=9.5,
    )
    axes[1].text(
        0.98,
        0.79,
        "Day 28: M3 held from Day 21",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color=muted,
        fontsize=9.5,
    )
    fig.text(
        0.07,
        0.065,
        "Historical: 34 building-cycles, cycle-grouped LOGO-CV. Prospective: 3 untouched 2026-3 Tags buildings.\n"
        "Blank checkpoints mean no new forecast: recovery holds after Day 14; bodyweight holds after Day 21. "
        "Directional evidence—not a guarantee for every building.",
        color=muted,
        fontsize=9.0,
    )
    fig.savefig(OUTPUT / "checkpoint_accuracy.png", dpi=220, facecolor=background)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    detail = build_holdout_detail()
    summary.to_csv(OUTPUT / "checkpoint_accuracy_summary.csv", index=False)
    detail.to_csv(OUTPUT / "checkpoint_accuracy_2026_3_building_detail.csv", index=False)
    plot_summary(summary)


if __name__ == "__main__":
    main()
