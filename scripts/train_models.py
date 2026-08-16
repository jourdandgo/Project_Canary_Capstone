"""Train and version Project Canary Sprint 3 forecast models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from canary import (
    extract_feature_row,
    load_workbook,
    save_day35_manifest,
    train_day35_weight_baseline,
)
from canary.forecast import _predict
from canary.modeling import load_final_weight_labels, save_training_result, train_outcome_model


def _add_latest_recovery_audit(dataset, result) -> None:
    """Attach a genuinely later-cycle checkpoint audit without refitting."""

    latest_start = pd.to_datetime(dataset.cycles["start_date"]).max()
    latest = dataset.cycles.loc[
        pd.to_datetime(dataset.cycles["start_date"]).eq(latest_start)
    ]
    records: list[dict[str, object]] = []
    for outcome in latest.itertuples(index=False):
        for day in (7, 14, 21, 28):
            as_of = pd.Timestamp(outcome.start_date) + pd.Timedelta(days=day - 1)
            feature = extract_feature_row(
                dataset, str(outcome.cycle_id), str(outcome.building_id), as_of
            )
            if feature is None or pd.isna(feature.get("percentage_alive")):
                continue
            prediction = _predict(
                feature, "recovery", result.manifest, result.model
            )
            actual = float(outcome.final_recovery_rate)
            records.append(
                {
                    "cycle_id": str(outcome.cycle_id),
                    "building_id": str(outcome.building_id),
                    "review_day": day,
                    "as_of_date": as_of.date().isoformat(),
                    "current_percentage_alive": float(feature["percentage_alive"]),
                    "predicted_final_recovery": float(prediction),
                    "last_recorded_recovery_proxy": actual,
                    "error": float(prediction - actual),
                    "absolute_error": float(abs(prediction - actual)),
                }
            )
    if not records:
        return
    frame = pd.DataFrame(records)
    actual = frame["last_recorded_recovery_proxy"].to_numpy(float)
    predicted = frame["predicted_final_recovery"].to_numpy(float)
    checkpoint_metrics = {}
    for day, group in frame.groupby("review_day"):
        group_actual = group["last_recorded_recovery_proxy"].to_numpy(float)
        group_predicted = group["predicted_final_recovery"].to_numpy(float)
        checkpoint_metrics[str(int(day))] = {
            "rows": int(len(group)),
            "mae": float(mean_absolute_error(group_actual, group_predicted)),
            "rmse": float(mean_squared_error(group_actual, group_predicted) ** 0.5),
            "bias": float(np.mean(group_predicted - group_actual)),
        }
    result.manifest["prospective_latest_cycle_audit"] = {
        "cycle_id": str(frame["cycle_id"].iloc[0]),
        "training_exclusion": "Excluded from fitting, preprocessing, tuning and champion selection.",
        "endpoint_definition": "Day 35 last-recorded population divided by beginning population; provisional proxy, not verified harvest recovery.",
        "independent_outcomes": int(
            frame[["cycle_id", "building_id"]].drop_duplicates().shape[0]
        ),
        "checkpoint_rows": int(len(frame)),
        "metrics": {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r2": float(r2_score(actual, predicted)),
            "bias": float(np.mean(predicted - actual)),
        },
        "checkpoint_metrics": checkpoint_metrics,
        "predictions": records,
    }


def _write_model_card(summaries: dict, day35: dict, output: Path) -> None:
    recovery = summaries["recovery"]
    weight = summaries["weight"]
    day14 = recovery["day14_backtest_metrics"]
    day14_weight = day35["day14_backtest_metrics"]
    lines = [
        "# Project Canary Model Card",
        "",
        "## Decision use",
        "",
        "Canary forecasts harvest recovery and projects average liveweight on Day 35. These outputs do not set or change the independent rules-based risk rating, diagnose disease, or prescribe treatment.",
        "",
        "Day 35 is the farm-approved 1.8 kg weight milestone. The primary weight output is a Day 35 projection, not final liveweight at an unknown harvest date.",
        "",
        "## Selected methods",
        "",
        "| Outcome | Version | Selected method | Cycles | Distinct building outcomes | Validation MAE | Status |",
        "|---|---|---:|---:|---:|---:|---|",
        f"| Predicted harvest recovery | {recovery['model_version']} | {recovery['selected_model']} | {len(recovery['training_cycles'])} | {recovery['training_building_cycles']} | {recovery['selected_metrics']['mae'] * 100:.2f} points | Prototype; trained on last-recorded recovery proxy |",
        f"| Projected Day 35 weight | {day35['model_version']} | {day35['selected_model']} | {len(day35['training_cycles'])} | {day35['training_building_cycles']} | {day35['selected_metrics']['mae_kg']:.3f} kg | Prototype; {day35['actual_target_hits']} historical 1.8 kg hits |",
        "",
        "Validation is nested: the outer loop holds out one complete recorded cycle, while the inner loop tunes only within the remaining cycles. Repeated snapshots receive equal building-cycle weight. Recovery uses Days 7, 14, 21, 28, and the latest eligible checkpoint; Day 35 weight uses checkpoints at Days 7, 14, 21, and 28.",
        "",
        "Recovery whole-cycle holdouts: " + ", ".join(recovery["training_cycles"]) + ". Day 35 weight whole-cycle holdouts: " + ", ".join(day35["training_cycles"]) + ". The latest cycle with newly recorded Day 35 weights is reserved as a prospective audit and is not used in model fitting or champion selection.",
        "",
        "For live dates between recovery checkpoints, expected remaining loss is linearly interpolated between the surrounding Days 7, 14, 21, and 28 values. The Day 28 value is held after Day 28 because a verified harvest-date horizon is unavailable.",
        "",
        f"Recovery learned challenger: {recovery['research_champion']}; operational method: {recovery['selected_model']}. Continuous-estimate gate passed: {recovery['champion_gates']['regression_gate_passed']}; 95% classification gate passed: {recovery['champion_gates']['target_classification_gate_passed']}.",
        f"Weight learned challenger: {day35['research_champion']}; operational method: {day35['selected_model']}. Learned-model regression gate passed: {day35['champion_gates']['regression_gate_passed']}; target-classification gate passed: {day35['champion_gates']['target_classification_gate_passed']}.",
        "",
        "## Day 14 recovery backtest",
        "",
        "For every eligible building-cycle, Canary recreated the forecast using only information available on Day 14, then compared it with last-recorded recovery.",
        "",
        f"- Building-cycles evaluated: {day14['building_cycles']}",
        f"- Day 14 MAE: {day14['mae'] * 100:.2f} percentage points",
        f"- Day 14 RMSE: {day14['rmse'] * 100:.2f} percentage points",
        f"- Actual at/above 95%: {day14['actual_at_or_above_target']}; correctly recognized: {day14['at_or_above_target_recall']:.1%}",
        f"- Actual below 95%: {day14['actual_below_target']}; warned below target: {day14['below_target_recall']:.1%}",
        f"- Target-side accuracy: {day14['target_side_accuracy']:.1%}; always-below majority baseline: {day14['majority_side_accuracy']:.1%}",
        "- Interpretation: target-side accuracy does not beat the majority baseline and must not be presented as discrimination proof.",
        "",
        "## Recovery candidate comparison",
        "",
        "| Candidate | MAE | Cycle-balanced MAE | RMSE | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate, metrics in recovery["metrics"].items():
        lines.append(
            f"| {candidate} | {metrics['mae'] * 100:.2f} pts | {metrics['cycle_macro_mae'] * 100:.2f} pts | {metrics['rmse'] * 100:.2f} pts | {metrics['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recovery model reliance",
            "",
            "Held-out permutation importance describes predictive reliance on unseen cycles; it does not prove causality.",
            "",
            "| Model input | Relative reliance | Held-out MAE increase |",
            "|---|---:|---:|",
        ]
    )
    recovery_importance = recovery.get("held_out_permutation_importance", []) or recovery.get(
        "research_champion_permutation_importance", []
    )
    for item in recovery_importance:
        lines.append(
            f"| {item['feature']} | {item['relative_importance_pct']:.1f}% | {item['mean_mae_increase'] * 100:.3f} recovery points |"
        )
    lines.extend(
        [
            "",
            "## Day 35 candidate comparison",
            "",
            "| Candidate | MAE | Cycle-balanced MAE | RMSE | R² | Within 200 g |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate, metrics in day35["candidate_metrics"].items():
        lines.append(
            f"| {candidate} | {metrics['mae_kg']:.3f} kg | {metrics['cycle_macro_mae_kg']:.3f} kg | {metrics['rmse_kg']:.3f} kg | {metrics['r2']:.3f} | {metrics['within_200g_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Day 14 to Day 35 weight backtest",
            "",
            f"- Building-cycles evaluated: {day14_weight['building_cycles']}",
            f"- MAE: {day14_weight['mae_kg'] * 1000:.0f} g",
            f"- RMSE: {day14_weight['rmse_kg'] * 1000:.0f} g",
            f"- Bias: {day14_weight['mean_error_kg'] * 1000:+.0f} g",
            f"- Within 200 g: {day14_weight['within_200g_rate']:.1%}",
            f"- Correct side of the 1.8 kg target: {day14_weight['target_side_accuracy']:.1%}",
            f"- Historical Day 35 results at/above 1.8 kg: {day35['actual_target_hits']}; below: {day35['actual_target_misses']}",
            "",
            "## Important limitations",
            "",
            f"- Recovery is trained on {len(recovery['training_cycles'])} recorded cycle histories and {recovery['training_building_cycles']} building outcomes. The label is last-recorded population divided by beginning population, not confirmed actual-harvest recovery.",
            f"- Day 35 weight uses {day35['training_building_cycles']} building outcomes across {len(day35['training_cycles'])} cycles. The current cycle is excluded from training.",
            "- Both comparisons use nested whole-cycle validation and cycle-balanced MAE as the primary metric. RMSE and R² are secondary checks; target-side metrics describe decision usefulness.",
            "- Uncertainty ranges use the 80th percentile of held-out absolute errors. They are empirical prototype ranges, not formal clinical or statistical guarantees.",
            "- Risk thresholds remain provisional until farm experts approve them. Recommendations remain pending Doc Raymond's action table.",
            "",
            "## Day 35 weight improvement plan",
            "",
            f"The best learned challenger is {day35['research_champion']}. The operational method is {day35['selected_model']} because no learned challenger cleared every approved gate.",
            "",
            "1. Standardize weights near Days 7, 14, 21, 28, and 35, including sample size and zone.",
            "2. Continue comparing historical remaining gain, checkpoint-calibrated linear regression, Ridge, robust Huber regression, and constrained Gradient Boosting.",
            "3. Keep one building record per checkpoint and hold out complete unseen cycles.",
            "4. Report MAE in grams, bias, within-100 g / within-200 g rates, and target-hit usefulness once target hits exist.",
            "",
            "## Retraining",
            "",
            "Run `uv run python -m scripts.train_models <workbook.xlsx> --performance-summary <summary.xlsx> --output models`. Daily dashboard use performs inference only and never retrains a model.",
            "",
        ]
    )
    (output / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--performance-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()

    dataset = load_workbook(args.workbook)
    final_weight_labels = load_final_weight_labels(args.performance_summary)
    summaries = {}
    for outcome in ("recovery", "weight"):
        result = train_outcome_model(
            dataset,
            outcome,
            final_weight_labels if outcome == "weight" else None,
        )
        if outcome == "recovery":
            _add_latest_recovery_audit(dataset, result)
        save_training_result(result, args.output)
        summaries[outcome] = result.manifest

    day35_manifest = train_day35_weight_baseline(dataset)
    save_day35_manifest(day35_manifest, args.output / "day35_weight_manifest.json")

    # Portable, versioned supplementary artifacts for the capstone team.  The
    # dashboard uses the manifest for transparent inference; these packages
    # preserve the exact selected-model parameters and feature schema.
    recovery_model = summaries["recovery"]
    recovery_artifact = args.output / "harvest_recovery_champion.pkl"
    recovery_model_path = args.output / "recovery_model.joblib"
    joblib.dump(
        {
            "outcome": "harvest_recovery",
            "selected_model": recovery_model["selected_model"],
            "model_version": recovery_model["model_version"],
            "feature_columns": recovery_model["feature_columns"],
            "validation": recovery_model["selected_metrics"],
            "model": joblib.load(recovery_model_path) if recovery_model_path.exists() else None,
            "prediction_target": recovery_model.get("prediction_target"),
            "formula_parameters": recovery_model.get("additional_loss_by_age_band"),
        },
        recovery_artifact,
    )
    day35_model_path = args.output / "day35_weight_model.joblib"
    joblib.dump(
        {
            "outcome": "day35_average_weight",
            "selected_model": day35_manifest["selected_model"],
            "research_champion": day35_manifest["research_champion"],
            "model_version": day35_manifest["model_version"],
            "feature_columns": day35_manifest.get("feature_columns", []),
            "model": joblib.load(day35_model_path) if day35_model_path.exists() else None,
            "prediction_target": day35_manifest.get("prediction_target"),
            "remaining_gain_by_measurement_day_kg": day35_manifest[
                "remaining_gain_by_measurement_day_kg"
            ],
            "validation": day35_manifest["selected_metrics"],
            "note": "Operational package. Historical remaining gain is retained whenever no learned remaining-gain challenger clears every approved gate.",
        },
        args.output / "day35_weight_champion.pkl",
    )

    training_summary = {**summaries, "day35_weight": day35_manifest}
    (args.output / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )
    _write_model_card(summaries, day35_manifest, args.output)
    for outcome, manifest in summaries.items():
        metrics = manifest["selected_metrics"]
        print(
            f"{outcome}: {manifest['selected_model']} | "
            f"MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
            f"cycles={len(manifest['training_cycles'])}"
        )
    day35_metrics = day35_manifest["selected_metrics"]
    print(
        f"day35 weight: {day35_manifest['selected_model']} | "
        f"MAE={day35_metrics['mae_kg']:.4f} | "
        f"RMSE={day35_metrics['rmse_kg']:.4f} | "
        f"cycles={len(day35_manifest['training_cycles'])}"
    )


if __name__ == "__main__":
    main()
