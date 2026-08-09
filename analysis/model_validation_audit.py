"""Recompute Project Canary's key model and data-quality claims."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from canary import load_workbook


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = Path(
    os.getenv(
        "CANARY_DEFAULT_WORKBOOK",
        str(ROOT.parent / "FARM HARVEST DATA.xlsx"),
    )
)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def build_audit() -> dict:
    dataset = load_workbook(WORKBOOK)
    recovery = json.loads((ROOT / "models" / "recovery_manifest.json").read_text())
    weight = json.loads((ROOT / "models" / "day35_weight_manifest.json").read_text())

    recovery_rows = recovery["day14_backtest"]
    recovery_actual = np.asarray([row["actual"] for row in recovery_rows], dtype=float)
    recovery_predicted = np.asarray([row["predicted"] for row in recovery_rows], dtype=float)
    recovery_recomputed = _metrics(recovery_actual, recovery_predicted)

    weight_rows = weight["day14_backtest"]
    weight_actual = np.asarray(
        [row["actual_day35_weight_kg"] for row in weight_rows], dtype=float
    )
    weight_predicted = np.asarray(
        [row["predicted_day35_weight_kg"] for row in weight_rows], dtype=float
    )
    weight_recomputed = _metrics(weight_actual, weight_predicted)

    recovery_d14 = recovery["day14_backtest_metrics"]
    weight_d14 = weight["day14_backtest_metrics"]
    recovery_candidates = recovery["metrics"]
    weight_candidates = weight["candidate_metrics"]
    recovery_best_macro = min(
        float(metrics["cycle_macro_mae"])
        for metrics in recovery_candidates.values()
    )
    weight_best_macro = min(
        float(metrics["cycle_macro_mae_kg"])
        for metrics in weight_candidates.values()
    )
    importance = recovery["global_feature_importance"]

    checks = {
        "source_to_canonical_reconciliation": (
            dataset.quality.source_rows - dataset.quality.duplicate_rows_consolidated
            == dataset.quality.canonical_rows
        ),
        "no_blocking_source_errors": not dataset.quality.blocking_errors,
        "recovery_day14_mae_recomputes": _close(
            recovery_recomputed["mae"], recovery_d14["mae"]
        ),
        "recovery_day14_rmse_recomputes": _close(
            recovery_recomputed["rmse"], recovery_d14["rmse"]
        ),
        "recovery_day14_bias_recomputes": _close(
            recovery_recomputed["bias"], recovery_d14["mean_error"]
        ),
        "weight_day14_mae_recomputes": _close(
            weight_recomputed["mae"], weight_d14["mae_kg"]
        ),
        "weight_day14_rmse_recomputes": _close(
            weight_recomputed["rmse"], weight_d14["rmse_kg"]
        ),
        "weight_day14_bias_recomputes": _close(
            weight_recomputed["bias"], weight_d14["mean_error_kg"]
        ),
        "recovery_winner_within_selection_tolerance": (
            float(recovery["selected_metrics"]["cycle_macro_mae"])
            <= recovery_best_macro * 1.05
        ),
        "weight_winner_within_selection_tolerance": (
            float(weight["selected_metrics"]["cycle_macro_mae_kg"])
            <= weight_best_macro * 1.05
        ),
        "recovery_importance_sums_to_100": _close(
            sum(float(item["absolute_importance_pct"]) for item in importance),
            100.0,
            tolerance=1e-9,
        ),
    }

    return {
        "report": "Project Canary model validation audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "workbook": WORKBOOK.name,
            "source_rows": dataset.quality.source_rows,
            "canonical_building_days": dataset.quality.canonical_rows,
            "repeated_rows_consolidated": dataset.quality.duplicate_rows_consolidated,
            "blocking_errors": list(dataset.quality.blocking_errors),
        },
        "recovery_model": {
            "selected_model": recovery["selected_model"],
            "cycles": len(recovery["training_cycles"]),
            "distinct_building_outcomes": recovery["training_building_cycles"],
            "balanced_checkpoint_rows": recovery["training_snapshot_rows"],
            "overall_mae_percentage_points": recovery["selected_metrics"]["mae"] * 100,
            "day14_mae_percentage_points": recovery_recomputed["mae"] * 100,
            "day14_rmse_percentage_points": recovery_recomputed["rmse"] * 100,
            "day14_bias_percentage_points": recovery_recomputed["bias"] * 100,
            "day14_target_side_accuracy": recovery_d14["target_side_accuracy"],
            "day14_majority_side_accuracy": recovery_d14["majority_side_accuracy"],
            "day14_at_or_above_target_recall": recovery_d14["at_or_above_target_recall"],
            "top_model_reliance": importance[:8],
            "verdict": "Directional point estimate only; target-side discrimination is not established.",
        },
        "day35_weight_model": {
            "selected_model": weight["selected_model"],
            "cycles": len(weight["training_cycles"]),
            "distinct_day35_outcomes": weight["training_building_cycles"],
            "checkpoint_rows": weight["training_checkpoint_rows"],
            "overall_mae_grams": weight["selected_metrics"]["mae_kg"] * 1000,
            "day14_mae_grams": weight_recomputed["mae"] * 1000,
            "day14_rmse_grams": weight_recomputed["rmse"] * 1000,
            "day14_bias_grams": weight_recomputed["bias"] * 1000,
            "day14_within_200g_rate": weight_d14["within_200g_rate"],
            "historical_day35_target_hits": weight["actual_target_hits"],
            "verdict": "Useful age-aware baseline; uncertainty is material and target-hit discrimination cannot be tested.",
        },
        "checks": checks,
        "overall_status": "PASS" if all(checks.values()) else "FAIL",
    }


def _write_review(payload: dict) -> None:
    recovery = payload["recovery_model"]
    weight = payload["day35_weight_model"]
    source = payload["source"]
    lines = [
        "# Project Canary Model Validation Review",
        "",
        f"Status: **{payload['overall_status']}**",
        "",
        "## Executive verdict",
        "",
        f"- Recovery: {recovery['verdict']} Held-out MAE is {recovery['overall_mae_percentage_points']:.2f} points overall and {recovery['day14_mae_percentage_points']:.2f} points at Day 14.",
        f"- Day 35 weight: {weight['verdict']} Held-out MAE is {weight['overall_mae_grams']:.0f} g overall and {weight['day14_mae_grams']:.0f} g from Day 14.",
        "- Risk: retain as a transparent operational-priority score, not as a probability model. Thresholds still require farm validation.",
        "",
        "## Data foundation",
        "",
        f"The source contained {source['source_rows']:,} rows. Canary consolidated {source['repeated_rows_consolidated']:,} repeated rows into {source['canonical_building_days']:,} unique building-day records with {len(source['blocking_errors'])} blocking conflicts.",
        "",
        "## Recovery model",
        "",
        f"The selected method is `{recovery['selected_model']}`, trained/evaluated across {recovery['cycles']} cycles and {recovery['distinct_building_outcomes']} distinct building outcomes. The {recovery['balanced_checkpoint_rows']} checkpoint rows are repeated time snapshots, not independent flock outcomes.",
        "",
        f"At Day 14, target-side accuracy is {recovery['day14_target_side_accuracy']:.1%}, equal to the {recovery['day14_majority_side_accuracy']:.1%} majority baseline; at/above-target recall is {recovery['day14_at_or_above_target_recall']:.1%}.",
        "",
        "## Day 35 weight model",
        "",
        f"The selected method is `{weight['selected_model']}`, validated across {weight['cycles']} cycles and {weight['distinct_day35_outcomes']} Day 35 building outcomes. At Day 14, {weight['day14_within_200g_rate']:.1%} of projections were within 200 g. Historical Day 35 target hits: {weight['historical_day35_target_hits']}.",
        "",
        "## Required interpretation",
        "",
        "Feature reliance and per-building contributions are statistical associations, not proof of cause. Temperature, humidity, mortality, feed, water, and heat-stress checks belong in a separate diagnostic layer until farm-approved thresholds and units are available.",
        "",
    ]
    (ROOT / "docs" / "MODEL_VALIDATION_REVIEW.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    audit = build_audit()
    (ROOT / "analysis" / "model_validation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    _write_review(audit)
    print(
        f"{audit['overall_status']}: {sum(audit['checks'].values())}/{len(audit['checks'])} checks passed"
    )
