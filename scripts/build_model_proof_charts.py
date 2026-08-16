"""Build presentation-ready model-proof charts from the canonical manifests.

The manifests are the single source of truth used by the app and notebooks.
This script intentionally does not recompute an alternative experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "final_model_iteration"
RECOVERY = ROOT / "models" / "recovery_manifest.json"
WEIGHT = ROOT / "models" / "day35_weight_manifest.json"

GREEN = "#174f3b"
LIME = "#91c529"
RED = "#c94f3d"

RECOVERY_LABELS = {
    "age_band_remaining_loss": "Age-band baseline",
    "remaining_loss_linear": "Ordinary linear",
    "remaining_loss_ridge": "Ridge",
    "remaining_loss_gradient_boosting": "Gradient Boosting",
    "remaining_loss_extra_trees": "Constrained Extra Trees",
}

WEIGHT_LABELS = {
    "historical_remaining_gain": "Historical remaining gain",
    "checkpoint_linear_remaining_gain": "Checkpoint linear",
    "ridge_remaining_gain": "Ridge",
    "huber_remaining_gain": "Robust Huber",
    "gradient_boosting_remaining_gain": "Gradient Boosting",
}


def _comparison_frame(manifest: dict, outcome: str) -> pd.DataFrame:
    rows = []
    candidate_metrics = manifest.get("candidate_metrics", manifest.get("metrics", {}))
    for key, metrics in candidate_metrics.items():
        if outcome == "recovery":
            rows.append(
                {
                    "model": RECOVERY_LABELS[key],
                    "cycle_mae": metrics["cycle_macro_mae"] * 100,
                    "r2": metrics["r2"],
                }
            )
        else:
            rows.append(
                {
                    "model": WEIGHT_LABELS[key],
                    "cycle_mae": metrics["cycle_macro_mae_kg"] * 1000,
                    "r2": metrics["r2"],
                }
            )
    return pd.DataFrame(rows).sort_values("cycle_mae")


def _comparison_chart(frame: pd.DataFrame, outcome: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=frame, x="cycle_mae", y="model", color=GREEN, ax=axes[0])
    axes[0].set_title("Cycle-balanced MAE")
    axes[0].set_xlabel("percentage points" if outcome == "recovery" else "grams")
    axes[0].set_ylabel("")
    sns.barplot(data=frame, x="r2", y="model", color=LIME, ax=axes[1])
    axes[1].axvline(0, color="#7b8794", linewidth=1)
    axes[1].set_title("Held-out R²")
    axes[1].set_xlabel("variance explained")
    axes[1].set_ylabel("")
    fig.suptitle(
        f"{outcome.title()} model comparison — complete-cycle holdouts",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{outcome}_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _shap_chart(manifest: dict) -> None:
    frame = pd.DataFrame(manifest["held_out_shap_importance"]).head(10).copy()
    frame["effect_points"] = frame["mean_abs_shap_recovery"] * 100
    frame["direction"] = frame["direction_when_value_increases"].map(
        lambda value: "Raises estimate" if "raises" in value.lower() else "Lowers estimate"
    )
    palette = {"Raises estimate": LIME, "Lowers estimate": RED}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.barplot(
        data=frame,
        x="effect_points",
        y="feature",
        hue="direction",
        dodge=False,
        palette=palette,
        ax=ax,
    )
    ax.set_title("Recovery model — held-out SHAP importance")
    ax.set_xlabel("mean absolute movement in projected recovery (percentage points)")
    ax.set_ylabel("")
    ax.legend(title="When the input is higher", loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "recovery_oof_shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _weight_checkpoint_chart(manifest: dict) -> None:
    rows = []
    for key in (
        "historical_remaining_gain",
        "ridge_remaining_gain",
        "gradient_boosting_remaining_gain",
    ):
        label = WEIGHT_LABELS[key]
        for day_label, values in manifest["candidate_metrics"][key]["horizon"].items():
            rows.append(
                {
                    "day": int(day_label.split()[-1]),
                    "model": label,
                    "mae_g": values["mae_kg"] * 1000,
                    "r2": values["r2"],
                }
            )
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=frame, x="day", y="mae_g", hue="model", marker="o", ax=axes[0])
    axes[0].set_title("Weight MAE by checkpoint")
    axes[0].set_ylabel("grams")
    sns.lineplot(data=frame, x="day", y="r2", hue="model", marker="o", ax=axes[1], legend=False)
    axes[1].axhline(0, color="#7b8794", linewidth=1)
    axes[1].set_title("Weight R² by checkpoint")
    axes[1].set_ylabel("variance explained")
    fig.suptitle(
        "Day 35 weight reliability by review checkpoint",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "weight_checkpoint_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _actual_vs_predicted_chart(manifest: dict, outcome: str) -> None:
    records = manifest.get("backtest_predictions", [])
    if not records:
        return
    frame = pd.DataFrame(records)
    if outcome == "recovery":
        actual = "actual_final_recovery_proxy"
        predicted = "predicted_final_recovery"
        scale = 100.0
        unit = "%"
        color = "cycle_day"
        title = "Harvest recovery: actual proxy vs held-out prediction"
    else:
        actual = "actual_day35_weight_kg"
        predicted = "predicted_day35_weight_kg"
        scale = 1000.0
        unit = "g"
        color = "measurement_day"
        title = "Day 35 weight: actual vs held-out prediction"
    frame["actual_display"] = frame[actual] * scale
    frame["predicted_display"] = frame[predicted] * scale
    low = min(frame["actual_display"].min(), frame["predicted_display"].min())
    high = max(frame["actual_display"].max(), frame["predicted_display"].max())
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.scatterplot(
        data=frame,
        x="actual_display",
        y="predicted_display",
        hue=color,
        palette="viridis",
        s=62,
        alpha=0.78,
        ax=ax,
    )
    ax.plot([low, high], [low, high], linestyle="--", color="#7b8794", label="Perfect prediction")
    ax.set_xlabel(f"Actual ({unit})")
    ax.set_ylabel(f"Held-out prediction ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Review day", loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"{outcome}_actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _recovery_checkpoint_chart(manifest: dict) -> None:
    rows = [
        {
            "day": int(day),
            "mae_points": values["mae"] * 100,
            "rmse_points": values["rmse"] * 100,
            "r2": values["r2"],
        }
        for day, values in manifest.get("checkpoint_performance", {}).items()
    ]
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("day")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    melted = frame.melt(
        id_vars="day",
        value_vars=["mae_points", "rmse_points"],
        var_name="metric",
        value_name="error_points",
    )
    melted["metric"] = melted["metric"].map(
        {"mae_points": "MAE", "rmse_points": "RMSE"}
    )
    sns.lineplot(data=melted, x="day", y="error_points", hue="metric", marker="o", ax=axes[0])
    axes[0].set_title("Error by review checkpoint")
    axes[0].set_ylabel("recovery percentage points")
    sns.lineplot(data=frame, x="day", y="r2", marker="o", color=GREEN, ax=axes[1])
    axes[1].axhline(0, color="#7b8794", linewidth=1)
    axes[1].set_title("Held-out R² by checkpoint")
    axes[1].set_ylabel("variance explained")
    fig.suptitle("Does recovery accuracy improve as more days are observed?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "recovery_accuracy_by_checkpoint.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))
    weight = json.loads(WEIGHT.read_text(encoding="utf-8"))
    recovery_frame = _comparison_frame(recovery, "recovery")
    weight_frame = _comparison_frame(weight, "weight")
    recovery_frame.to_csv(OUT / "recovery_manifest_comparison.csv", index=False)
    weight_frame.to_csv(OUT / "weight_manifest_comparison.csv", index=False)
    _comparison_chart(recovery_frame, "recovery")
    _comparison_chart(weight_frame, "weight")
    if recovery.get("held_out_shap_importance"):
        _shap_chart(recovery)
    _actual_vs_predicted_chart(recovery, "recovery")
    _actual_vs_predicted_chart(weight, "weight")
    _recovery_checkpoint_chart(recovery)
    _weight_checkpoint_chart(weight)
    print(OUT)


if __name__ == "__main__":
    main()
