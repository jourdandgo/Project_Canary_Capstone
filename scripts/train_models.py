"""Train and version Project Canary Sprint 3 forecast models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canary import load_workbook, save_day35_manifest, train_day35_weight_baseline
from canary.modeling import load_final_weight_labels, save_training_result, train_outcome_model


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
        "Day 35 is the 2.0 kg weight milestone. The primary weight output is a Day 35 projection, not final liveweight at an unknown harvest date.",
        "",
        "## Selected methods",
        "",
        "| Outcome | Version | Selected method | Cycles | Distinct building outcomes | Validation MAE | Status |",
        "|---|---|---:|---:|---:|---:|---|",
        f"| Predicted harvest recovery | {recovery['model_version']} | {recovery['selected_model']} | {len(recovery['training_cycles'])} | {recovery['training_building_cycles']} | {recovery['selected_metrics']['mae'] * 100:.2f} points | Prototype; trained on last-recorded recovery proxy |",
        f"| Projected Day 35 weight | {day35['model_version']} | {day35['selected_model']} | {len(day35['training_cycles'])} | {day35['training_building_cycles']} | {day35['selected_metrics']['mae_kg']:.3f} kg | Prototype; no historical 2.0 kg hits |",
        "",
        "Validation holds out one complete recorded cycle at a time. Recovery training is balanced to Days 7, 14, 21, 28, and the latest eligible checkpoint for each building-cycle. The Day 35 comparison uses one building checkpoint at Days 7, 14, 21, and 28.",
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
        "## Recovery model reliance",
        "",
        "Standardized Ridge coefficients describe association and direction in the fitted model; they do not prove causality.",
        "",
        "| Model input | Relative reliance | Direction | Standardized effect |",
        "|---|---:|---|---:|",
    ]
    for item in recovery.get("global_feature_importance", []):
        lines.append(
            f"| {item['feature']} | {item['absolute_importance_pct']:.1f}% | {item['direction']} | {item['coefficient_per_standard_deviation'] * 100:+.2f} recovery points |"
        )
    lines.extend(
        [
            "",
            "## Day 35 candidate comparison",
            "",
            "| Candidate | MAE | RMSE | Within 200 g |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate, metrics in day35["candidate_metrics"].items():
        lines.append(
            f"| {candidate} | {metrics['mae_kg']:.3f} kg | {metrics['rmse_kg']:.3f} kg | {metrics['within_200g_rate']:.1%} |"
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
            "- All evaluated Day 35 outcomes were below 2.0 kg, so target-hit discrimination cannot be tested.",
            "",
            "## Important limitations",
            "",
            "- Recovery is trained on five recorded cycle histories and 25 building outcomes. The label is last-recorded population divided by beginning population, not confirmed actual-harvest recovery.",
            "- Day 35 weight uses 19 building outcomes across four cycles. All 19 are below 2.0 kg, so target-hit classification cannot yet be evaluated.",
            "- Selection uses cycle-balanced MAE, then chooses the simplest candidate within 5% of the best result to avoid promoting complexity for a trivial gain.",
            "- Uncertainty ranges use the 80th percentile of held-out absolute errors. They are empirical prototype ranges, not formal clinical or statistical guarantees.",
            "- Risk thresholds remain provisional until farm experts approve them. Recommendations remain pending Doc Raymond's action table.",
            "",
            "## Day 35 weight improvement plan",
            "",
            "The current age-aware baseline adds the historically observed remaining gain from the measurement age to the latest building weight. It is building-responsive, but still limited-data.",
            "",
            "1. Standardize weights near Days 7, 14, 21, 28, and 35, including sample size and zone.",
            "2. Continue comparing the age-aware remaining-gain baseline with target-curve, recent-ADG, and compact Ridge candidates as new data arrives.",
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
        save_training_result(result, args.output)
        summaries[outcome] = result.manifest

    day35_manifest = train_day35_weight_baseline(dataset)
    save_day35_manifest(day35_manifest, args.output / "day35_weight_manifest.json")

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
