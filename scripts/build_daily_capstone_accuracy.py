#!/usr/bin/env python3
"""Validate Canary's selected transparent forecasts at every Day 7-34 landmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canary.biology_aware_modeling import build_daily_landmarks
from canary.data import load_workbook


CHECKPOINTS = (7, 14, 21, 28)
COLORS = {7: "#7B61FF", 14: "#00A6A6", 21: "#F59E0B", 28: "#E45756"}


def _interpolate(mapping: dict[int, float], age: float) -> float:
    points = sorted(mapping)
    if age <= points[0]:
        return float(mapping[points[0]])
    if age >= points[-1]:
        return float(mapping[points[-1]])
    for lo, hi in zip(points[:-1], points[1:]):
        if lo <= age <= hi:
            share = (age - lo) / (hi - lo)
            return float(mapping[lo] + share * (mapping[hi] - mapping[lo]))
    raise AssertionError(f"Unable to interpolate age {age}")


def _fold_mapping(train: pd.DataFrame, outcome: str) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for day in CHECKPOINTS:
        rows = train.loc[train["review_day"].eq(day)]
        if rows.empty:
            raise AssertionError(f"No training rows for Day {day}")
        if outcome == "recovery":
            remaining = rows["current_value"] - rows["actual"]
        else:
            remaining = rows["actual"] - rows["current_value"]
        mapping[day] = float(np.average(remaining, weights=rows["sample_weight"]))
    return mapping


def _predict_logo(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    development = frame.loc[frame["role"].eq("development")].copy()
    results: list[pd.DataFrame] = []
    for held_cycle in sorted(development["cycle_id"].astype(str).unique()):
        train = development.loc[~development["cycle_id"].astype(str).eq(held_cycle)]
        test = development.loc[development["cycle_id"].astype(str).eq(held_cycle)].copy()
        mapping = _fold_mapping(train, outcome)
        if outcome == "recovery":
            remaining = test["review_day"].map(lambda day: _interpolate(mapping, float(day))).to_numpy()
            prediction = test["current_value"].to_numpy(float) - np.maximum(remaining, 0.0)
            prediction = np.minimum(prediction, test["current_value"].to_numpy(float))
            unit = "percentage points"
            scale = 100.0
            prediction *= scale
            actual = test["actual"].to_numpy(float) * scale
        else:
            # Operational bodyweight logic is anchored to the age of the latest
            # actual measurement, not the review day. Stale weights are never
            # represented as newly measured daily observations.
            measurement_age = test["latest_measurement_day"].fillna(7).clip(7, 28)
            remaining = measurement_age.map(lambda day: _interpolate(mapping, float(day))).to_numpy()
            prediction = test["current_value"].to_numpy(float) + remaining
            actual = test["actual"].to_numpy(float)
            unit = "grams"
        test["actual_value"] = actual
        test["prediction"] = prediction
        test["residual"] = actual - prediction
        test["absolute_error"] = np.abs(test["residual"])
        test["squared_error"] = test["residual"] ** 2
        test["held_out_cycle"] = held_cycle
        test["metric_unit"] = unit
        results.append(test)
    return pd.concat(results, ignore_index=True)


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for day, group in predictions.groupby("review_day", sort=True):
        cycle_rmse = group.groupby("cycle_id")["squared_error"].mean().pow(0.5)
        cycle_mae = group.groupby("cycle_id")["absolute_error"].mean()
        y = group["actual_value"].to_numpy(float)
        p = group["prediction"].to_numpy(float)
        rows.append({
            "review_day": int(day),
            "n_snapshots": int(len(group)),
            "n_building_cycles": int(group[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
            "n_harvest_cycles": int(group["cycle_id"].nunique()),
            "cycle_macro_rmse": float(cycle_rmse.mean()),
            "cycle_macro_mae": float(cycle_mae.mean()),
            "pooled_rmse": float(mean_squared_error(y, p) ** 0.5),
            "pooled_mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
            "bias": float(np.mean(p - y)),
            "is_validated_checkpoint": bool(day in CHECKPOINTS),
        })
    return pd.DataFrame(rows)


def _checkpoint_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    checkpoint = predictions.loc[predictions["review_day"].isin(CHECKPOINTS)]
    rows = []
    for label, group in (("All Day 7-34 forecasts", predictions), ("Four validated checkpoints", checkpoint)):
        y = group["actual_value"].to_numpy(float)
        p = group["prediction"].to_numpy(float)
        cycle_rmse = group.groupby("cycle_id")["squared_error"].mean().pow(0.5)
        rows.append({
            "scope": label,
            "n_snapshots": int(len(group)),
            "cycle_macro_rmse": float(cycle_rmse.mean()),
            "pooled_rmse": float(mean_squared_error(y, p) ** 0.5),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
            "bias": float(np.mean(p - y)),
        })
    return pd.DataFrame(rows)


def _learning_curve_plot(recovery: pd.DataFrame, weight: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.5), constrained_layout=False)
    panels = [
        (axes[0], recovery, "Harvest recovery forecast error by as-of day", "Error (percentage points)"),
        (axes[1], weight, "Day 35 bodyweight forecast error by as-of day", "Error (grams)"),
    ]
    for ax, data, title, ylabel in panels:
        ax.plot(data["review_day"], data["cycle_macro_rmse"], color="#174C3C", marker="o", ms=3.5, lw=2.2, label="Cycle-macro RMSE")
        ax.plot(data["review_day"], data["cycle_macro_mae"], color="#377EB8", marker="o", ms=3.0, lw=1.8, label="Cycle-macro MAE")
        for day in CHECKPOINTS:
            row = data.loc[data["review_day"].eq(day)].iloc[0]
            ax.axvline(day, color=COLORS[day], lw=1.2, alpha=0.55, ls="--")
            ax.scatter(day, row["cycle_macro_rmse"], color=COLORS[day], edgecolor="white", s=75, zorder=5)
            ax.annotate(
                f"D{day}\nRMSE {row['cycle_macro_rmse']:.1f}",
                (day, row["cycle_macro_rmse"]),
                xytext=(0, -28 if day == 7 else 10), textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color=COLORS[day], fontweight="bold",
            )
        for day in (10, 20):
            row = data.loc[data["review_day"].eq(day)].iloc[0]
            ax.annotate(
                f"Day {day}: {row['cycle_macro_rmse']:.1f}",
                (day, row["cycle_macro_rmse"]), xytext=(7, -18), textcoords="offset points",
                fontsize=8, color="#333333", arrowprops={"arrowstyle": "-", "color": "#777777"},
            )
        first = data.iloc[0]; last = data.iloc[-1]
        change = (last["cycle_macro_rmse"] / first["cycle_macro_rmse"] - 1.0) * 100
        direction = "lower" if change < 0 else "higher"
        ax.text(
            0.99, 0.08,
            f"Day 34 RMSE is {abs(change):.0f}% {direction} than Day 7",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F3F7F5", "edgecolor": "#B8CCC5"},
        )
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(6.5, 34.5)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Forecast made using evidence available through this flock age")
    fig.suptitle(
        "Forecasts are available daily; accuracy is evaluated without mixing harvest cycles",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.28)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _actual_predicted_plot(predictions: pd.DataFrame, outcome: str, destination: Path) -> None:
    days = (7, 10, 14, 20, 21, 28)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    selected = predictions.loc[predictions["review_day"].isin(days)]
    lo = min(selected["actual_value"].min(), selected["prediction"].min())
    hi = max(selected["actual_value"].max(), selected["prediction"].max())
    padding = (hi - lo) * 0.06
    for ax, day in zip(axes.flat, days):
        group = selected.loc[selected["review_day"].eq(day)]
        color = COLORS.get(day, "#174C3C")
        ax.scatter(group["actual_value"], group["prediction"], color=color, alpha=0.78, s=42, edgecolor="white", linewidth=0.5)
        ax.plot([lo - padding, hi + padding], [lo - padding, hi + padding], color="#555555", ls="--", lw=1.1)
        rmse = mean_squared_error(group["actual_value"], group["prediction"]) ** 0.5
        mae = mean_absolute_error(group["actual_value"], group["prediction"])
        ax.set_title(f"Day {day} | RMSE {rmse:.1f}, MAE {mae:.1f}", color=color, fontsize=10, fontweight="bold")
        ax.set_xlim(lo - padding, hi + padding); ax.set_ylim(lo - padding, hi + padding)
        ax.grid(alpha=0.18); ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Actual"); axes[1, 1].set_xlabel("Actual"); axes[1, 2].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted"); axes[1, 0].set_ylabel("Predicted")
    unit = "percentage points" if outcome == "recovery" else "grams"
    fig.suptitle(f"Actual versus predicted — {outcome} ({unit})\nColor identifies the forecast day; dashed line is perfect prediction", fontsize=14, fontweight="bold")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=ROOT / "data" / "FARM HARVEST DATA.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "farmwide_modeling_latest_data_round" / "daily_accuracy")
    parser.add_argument("--audit-cycle", default="latest")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(args.workbook)
    cycles = sorted(
        dataset.cycles["cycle_id"].astype(str).unique(),
        key=lambda value: tuple(int(part) for part in value.split("-", 1)),
    )
    audit_cycle = cycles[-1] if args.audit_cycle == "latest" else str(args.audit_cycle)
    development_cycles = tuple(cycle for cycle in cycles if cycle != audit_cycle)
    all_metrics: dict[str, pd.DataFrame] = {}
    all_predictions: dict[str, pd.DataFrame] = {}
    for outcome in ("recovery", "bodyweight"):
        landmarks = build_daily_landmarks(
            dataset,
            "recovery" if outcome == "recovery" else "weight",
            development_cycles,
            audit_cycle,
        )
        predictions = _predict_logo(landmarks, outcome)
        metrics = _metrics(predictions)
        all_predictions[outcome] = predictions
        all_metrics[outcome] = metrics
        predictions.to_csv(args.output / f"{outcome}_daily_logo_predictions.csv", index=False)
        metrics.to_csv(args.output / f"{outcome}_daily_metrics.csv", index=False)
        _checkpoint_overall(predictions).to_csv(args.output / f"{outcome}_overall_vs_checkpoint_metrics.csv", index=False)
        _actual_predicted_plot(predictions, outcome, args.output / f"{outcome}_actual_vs_predicted_by_day.png")
    _learning_curve_plot(all_metrics["recovery"], all_metrics["bodyweight"], args.output / "daily_forecast_accuracy_learning_curve.png")
    manifest = {
        "workbook": str(args.workbook.resolve()),
        "validated_days": list(CHECKPOINTS),
        "daily_landmarks": [7, 34],
        "validation": "Outer leave-one-harvest-cycle-out; fold-local baseline mappings",
        "recovery_model": "Fold-local checkpoint remaining-loss baseline interpolated by current review day",
        "bodyweight_model": "Fold-local checkpoint remaining-gain baseline anchored to latest actual measurement day",
        "warning": "Between-checkpoint results are research estimates; Days 7, 14, 21 and 28 remain the principal validated anchors.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote daily capstone accuracy evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
