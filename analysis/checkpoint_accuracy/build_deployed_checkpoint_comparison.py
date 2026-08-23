"""Build the exact 2026-3 checkpoint comparison shown by deployed Canary.

The output deliberately distinguishes a newly calculated forecast from a
carried-forward forecast. Recovery refreshes through Day 14; Day 35
bodyweight refreshes through Day 21.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = APP_ROOT / "demo_data" / "2026-3"
DETAIL_FILE = APP_ROOT / "analysis" / "checkpoint_accuracy" / "checkpoint_accuracy_2026_3_building_detail.csv"
OUTPUT = APP_ROOT / "analysis" / "checkpoint_accuracy"
CHECKPOINTS = (7, 14, 21, 28)


def build_comparison() -> pd.DataFrame:
    detail = pd.read_csv(DETAIL_FILE)
    observed_rows: list[pd.DataFrame] = []
    for day in CHECKPOINTS:
        source = pd.read_csv(
            DEMO_ROOT / f"Project_Canary_2026-3_Day_{day:02d}.csv"
        )
        current = (
            source.sort_values("age_day")
            .groupby("building_id", as_index=False)
            .tail(1)
            .copy()
        )
        current["display_day"] = day
        current["current_recovery_pct"] = (
            current["population"] / current["beginning_inventory"] * 100
        )
        current["current_bodyweight_g"] = current["bodyweight_kg"] * 1000
        observed_rows.append(
            current[
                [
                    "building_id",
                    "display_day",
                    "record_date",
                    "current_recovery_pct",
                    "current_bodyweight_g",
                ]
            ]
        )
    observed = pd.concat(observed_rows, ignore_index=True).rename(
        columns={"building_id": "building"}
    )
    comparison = detail.merge(
        observed,
        on=["building", "display_day"],
        how="left",
        validate="one_to_one",
    )
    ordered = [
        "building",
        "display_day",
        "record_date",
        "current_recovery_pct",
        "current_bodyweight_g",
        "recovery_model",
        "recovery_evidence_day",
        "recovery_forecast_status",
        "predicted_recovery_pct",
        "actual_recovery_pct",
        "recovery_absolute_error_pp",
        "weight_model",
        "weight_evidence_day",
        "weight_forecast_status",
        "predicted_day35_weight_g",
        "actual_day35_weight_g",
        "weight_absolute_error_g",
    ]
    return comparison[ordered].sort_values(["building", "display_day"])


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby("display_day", as_index=False)
        .agg(
            mean_recovery_absolute_error_pp=("recovery_absolute_error_pp", "mean"),
            mean_weight_absolute_error_g=("weight_absolute_error_g", "mean"),
            recovery_forecast_status=("recovery_forecast_status", "first"),
            weight_forecast_status=("weight_forecast_status", "first"),
        )
    )


def plot_comparison(comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    background = "#062f24"
    panel = "#0c4031"
    white = "#f6fbf7"
    muted = "#b8cbc3"
    grid = "#3a6657"
    canary = "#b8ef32"
    colors = {"Tags 1": "#51d7c5", "Tags 2": "#ff9a45", "Tags 3": "#d6ec70"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), facecolor=background)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.22, wspace=0.22)
    fig.suptitle(
        "2026-3 forecast error by checkpoint",
        x=0.07,
        y=0.95,
        ha="left",
        color=white,
        fontsize=23,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.885,
        "Absolute gap between Canary's deployed forecast and the recorded Day 35 / ending outcome",
        color=muted,
        fontsize=12.5,
    )

    panels = (
        (axes[0], "recovery_absolute_error_pp", "Ending recovery", "Absolute error (percentage points)", 14),
        (axes[1], "weight_absolute_error_g", "Day 35 bodyweight", "Absolute error (grams)", 21),
    )
    for axis, metric, title, ylabel, last_refresh in panels:
        axis.set_facecolor(panel)
        for building, color in colors.items():
            series = comparison.loc[comparison["building"].eq(building)].sort_values("display_day")
            fresh = series.loc[series["display_day"].le(last_refresh)]
            held = series.loc[series["display_day"].ge(last_refresh)]
            axis.plot(
                fresh["display_day"],
                fresh[metric],
                color=color,
                marker="o",
                linewidth=2.6,
                markersize=7.5,
                label=building,
            )
            if len(held) > 1:
                axis.plot(
                    held["display_day"],
                    held[metric],
                    color=color,
                    marker="o",
                    markerfacecolor=panel,
                    markeredgewidth=1.8,
                    linestyle=(0, (3, 3)),
                    linewidth=1.8,
                    alpha=0.8,
                )
            for _, row in fresh.iterrows():
                value = float(row[metric])
                label = f"{value:.1f}" if metric.startswith("recovery") else f"{value:.0f}"
                axis.annotate(
                    label,
                    (row["display_day"], value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )
        axis.axvline(last_refresh, color=canary, linewidth=1.0, alpha=0.55)
        axis.text(
            last_refresh + 0.35,
            0.86,
            f"Forecast held after Day {last_refresh}",
            transform=axis.get_xaxis_transform(),
            color=muted,
            fontsize=9.3,
            va="top",
        )
        axis.set_title(title, color=white, fontsize=15.5, fontweight="bold", loc="left", pad=14)
        axis.set_xlabel("Uploaded checkpoint", color=muted, fontsize=10.5, labelpad=10)
        axis.set_ylabel(ylabel, color=muted, fontsize=10.5, labelpad=10)
        axis.set_xticks(CHECKPOINTS, [f"Day {day}" for day in CHECKPOINTS])
        axis.tick_params(colors=muted, labelsize=10)
        axis.grid(axis="y", color=grid, alpha=0.45, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.legend(frameon=False, labelcolor=white, fontsize=9.5, loc="upper right")

    d7 = summary.loc[summary["display_day"].eq(7)].iloc[0]
    d14 = summary.loc[summary["display_day"].eq(14)].iloc[0]
    d21 = summary.loc[summary["display_day"].eq(21)].iloc[0]
    fig.text(
        0.07,
        0.085,
        "AVERAGE ACROSS TAGS 1–3  "
        f"Recovery: {d7['mean_recovery_absolute_error_pp']:.2f} pp (D7) → "
        f"{d14['mean_recovery_absolute_error_pp']:.2f} pp (D14)  |  "
        f"Bodyweight: {d7['mean_weight_absolute_error_g']:.0f} g (D7) → "
        f"{d14['mean_weight_absolute_error_g']:.0f} g (D14) → "
        f"{d21['mean_weight_absolute_error_g']:.0f} g (D21)",
        color=white,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.041,
        "Solid markers are newly calculated forecasts. Open markers and dashed segments are carried-forward values, not new predictions.\n"
        "2026-3 is a three-building prospective replay; results are directional, not proof of monotonic improvement.",
        color=muted,
        fontsize=9.0,
    )
    fig.savefig(OUTPUT / "deployed_checkpoint_error_trends.png", dpi=220, facecolor=background)
    plt.close(fig)


def main() -> None:
    comparison = build_comparison()
    summary = build_summary(comparison)
    comparison.to_csv(OUTPUT / "deployed_checkpoint_building_comparison.csv", index=False)
    summary.to_csv(OUTPUT / "deployed_checkpoint_summary.csv", index=False)
    plot_comparison(comparison, summary)


if __name__ == "__main__":
    main()
