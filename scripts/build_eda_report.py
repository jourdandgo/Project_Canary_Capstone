"""Build the canonical report artifact for the Project Canary evidence audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "analysis" / "eda_results.json").read_text())
OUTPUT = ROOT / "analysis" / "report_artifact.json"


sources = [
    {
        "id": "farm-data",
        "label": "FARM HARVEST DATA.xlsx",
        "path": "FARM HARVEST DATA.xlsx",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT cycle, building, day14_weight_g, day14_target_g, day14_gap_pct, day35_weight_g, final_recovery_pct, final_average_weight_kg, met_day14_target FROM day14_points WHERE day14_weight_g IS NOT NULL",
            "description": "Select the reviewed exact-Day-14 building-cycle evidence extracted from the farm workbook.",
            "tables_used": ["day14_points"],
            "filters": ["Exact Day 14 weight is present", "Completed harvest cycles only"],
            "metric_definitions": ["Final recovery = ending inventory / beginning inventory"],
        },
    },
    {
        "id": "performance-summary",
        "label": "Farm Performance Summary.xlsx",
        "path": "Farm Performance Summary.xlsx",
    },
    {
        "id": "model-proof",
        "label": "Project Canary model manifests",
        "path": "models/MODEL_CARD.md",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT output, method, building_cycles, mae, target_side_accuracy, decision_use FROM model_rows ORDER BY output",
            "description": "Select the reviewed model-validation summary derived from the versioned manifests.",
            "tables_used": ["model_rows"],
            "filters": ["Leave-one-complete-cycle-out results only"],
        },
    },
    {
        "id": "farmer-validation",
        "label": "Farmer Validation Workbook.xlsx",
        "path": "Farmer Validation Workbook.xlsx",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT item, farmer_workbook, current_canary, recommended_use FROM threshold_rows ORDER BY item",
            "description": "Select the reviewed threshold comparison extracted from the farmer workbook.",
            "tables_used": ["threshold_rows"],
            "filters": ["RISK SCORING MATRIX excluded"],
        },
    },
    {
        "id": "farmer-playbook",
        "label": "Farmer Validation Workbook.xlsx — interventions and root causes",
        "path": "Farmer Validation Workbook.xlsx",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": "SELECT pattern, farmer_guidance, canary_use FROM playbook_rows ORDER BY pattern",
            "description": "Select the reviewed intervention and root-cause mappings extracted from the farmer workbook.",
            "tables_used": ["playbook_rows"],
            "filters": ["RISK SCORING MATRIX excluded"],
        },
    },
]


points = []
for row in RESULTS["evidence_rows"]:
    if row.get("day14_weight_kg") is None:
        continue
    points.append(
        {
            "cycle": row["cycle_id"],
            "building": row["building_id"],
            "day14_weight_g": round(row["day14_weight_kg"] * 1000, 1),
            "day14_target_g": round(row["day14_target_kg"] * 1000, 1),
            "day14_gap_pct": round(
                (row["day14_target_kg"] - row["day14_weight_kg"])
                / row["day14_target_kg"]
                * 100,
                1,
            ),
            "day35_weight_g": (
                round(row["day35_weight_kg"] * 1000, 1)
                if row.get("day35_weight_kg") is not None
                else None
            ),
            "final_recovery_pct": round(row["recomputed_recovery"] * 100, 2),
            "final_average_weight_kg": (
                round(row["final_average_weight_kg"], 3)
                if row.get("final_average_weight_kg") is not None
                else None
            ),
            "met_day14_target": row["day14_weight_kg"] >= row["day14_target_kg"],
        }
    )


associations = [
    {
        "outcome": "Day 35 measured weight",
        "pairs": 19,
        "raw_correlation": 0.49,
        "raw_p_value": 0.035,
        "within_cycle_correlation": 0.13,
        "within_cycle_p_value": 0.622,
        "verdict": "Raw relationship, but weak after comparing buildings within the same cycle",
    },
    {
        "outcome": "Final average liveweight",
        "pairs": 14,
        "raw_correlation": -0.28,
        "raw_p_value": 0.330,
        "within_cycle_correlation": 0.35,
        "within_cycle_p_value": 0.298,
        "verdict": "No reliable relationship demonstrated",
    },
    {
        "outcome": "Final harvest recovery",
        "pairs": 19,
        "raw_correlation": 0.63,
        "raw_p_value": 0.004,
        "within_cycle_correlation": 0.75,
        "within_cycle_p_value": 0.001,
        "verdict": "Promising positive association; still observational and small-sample",
    },
]


model_rows = [
    {
        "output": "Recovery — all daily snapshots",
        "method": RESULTS["model_summary"]["recovery_selected_model"].replace("_", " ").title(),
        "building_cycles": RESULTS["model_summary"]["recovery_building_cycles"],
        "mae": f"{RESULTS['model_summary']['recovery_overall_mae_pp']:.2f} percentage points",
        "target_side_accuracy": f"{RESULTS['model_summary']['recovery_target_side_accuracy_pct']:.1f}%",
        "decision_use": "Useful prototype estimate with visible uncertainty",
    },
    {
        "output": "Recovery — exact Day 14",
        "method": "Cycle-held-out backtest",
        "building_cycles": RESULTS["model_summary"]["recovery_building_cycles"],
        "mae": f"{RESULTS['model_summary']['recovery_day14_mae_pp']:.2f} percentage points",
        "target_side_accuracy": f"{RESULTS['model_summary']['recovery_day14_target_side_accuracy_pct']:.1f}%",
        "decision_use": "Useful early-warning evidence, not a guarantee",
    },
    {
        "output": "Final average weight",
        "method": "Historical farm mean",
        "building_cycles": RESULTS["model_summary"]["weight_building_cycles"],
        "mae": f"{RESULTS['model_summary']['weight_mae_kg']:.3f} kg",
        "target_side_accuracy": f"{RESULTS['model_summary']['weight_target_side_accuracy_pct']:.1f}%",
        "decision_use": "Baseline only; not a building-personalized forecast",
    },
]


threshold_rows = [
    {
        "item": "Bodyweight shortfall",
        "farmer_workbook": "Warning 10%; critical 30%",
        "current_canary": "Point cutoffs 5% / 15% / 30%",
        "recommended_use": "Reconcile before changing points; the workbook does not define the middle cutoff",
    },
    {
        "item": "Daily mortality",
        "farmer_workbook": "0.1% normal; 0.2% warning; 0.3% critical",
        "current_canary": "Age-aware increase versus recent baseline, per 1,000 birds",
        "recommended_use": "Add as a separate absolute alert; do not replace the trend score",
    },
    {
        "item": "Cumulative population loss",
        "farmer_workbook": "3% / 5% / 7%",
        "current_canary": "Gap from an age-based path toward 95% alive on Day 35",
        "recommended_use": "Strong candidate to simplify and expert-calibrate the survival dimension",
    },
    {
        "item": "Predicted recovery",
        "farmer_workbook": "95% normal; 93% warning; critical shown as >90%",
        "current_canary": "Forecast displayed separately; does not alter risk points",
        "recommended_use": "Keep separate and correct likely typo to <90% after confirmation",
    },
    {
        "item": "Temperature",
        "farmer_workbook": "Age targets plus 2°C / 3°C / 5°C deviation alerts",
        "current_canary": "Model feature only; not a core risk point",
        "recommended_use": "Add a secondary operational alert after validating targets and sensor coverage",
    },
    {
        "item": "Humidity",
        "farmer_workbook": "Age targets plus 60% warning and 70% critical",
        "current_canary": "Model feature only; not a core risk point",
        "recommended_use": "Do not automate yet; normal/target/warning values overlap and need clarification",
    },
    {
        "item": "Feed intake",
        "farmer_workbook": "Age schedule; 10% warning and 50% critical shortfall",
        "current_canary": "Model feature; recorded as bags rather than grams per bird",
        "recommended_use": "Useful after bag weight and feed-unit conversion are confirmed",
    },
]


playbook_rows = [
    {
        "pattern": "Weight or feed lag",
        "farmer_guidance": "Check feed quality, feeder allocation, temperatures and flock health",
        "canary_use": "Merge into Weight Lag Only checklist; retain representative reweighing first",
    },
    {
        "pattern": "High mortality or rapid population loss",
        "farmer_guidance": "Check flock health and ventilation within 24 hours",
        "canary_use": "Strengthens Survival Concern and Growth + Survival actions",
    },
    {
        "pattern": "Temperature or humidity problem",
        "farmer_guidance": "Check ventilation, heaters, fans, pads and water-pump timing within 6 hours",
        "canary_use": "Add as a secondary environment-specific checklist and urgency",
    },
    {
        "pattern": "Poor recovery forecast",
        "farmer_guidance": "Check flock health and condemn rate within 24 hours",
        "canary_use": "Use health inspection; keep condemn rate separate because current recovery excludes it",
    },
    {
        "pattern": "Root-cause hypotheses",
        "farmer_guidance": "Illness, heat, ventilation/equipment, rain, air leaks and weather",
        "canary_use": "Show as checks to investigate—not diagnoses or asserted causes",
    },
]


artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "Project Canary Evidence and Practicality Audit",
        "description": "Day 14 evidence, current model usefulness, EDA findings and Farmer Validation Workbook implications.",
        "generatedAt": "2026-08-08T12:00:00+08:00",
        "sources": sources,
        "charts": [
            {
                "id": "day14-recovery",
                "title": "Day 14 weight and final harvest recovery",
                "subtitle": "19 building-cycles across four completed cycles; final recovery in percentage points",
                "type": "scatter",
                "dataset": "day14_points",
                "sourceId": "farm-data",
                "encodings": {
                    "x": {"field": "day14_weight_g", "type": "quantitative", "label": "Day 14 weight", "unit": "g"},
                    "y": {"field": "final_recovery_pct", "type": "quantitative", "label": "Final recovery", "unit": "%"},
                    "color": {"field": "cycle", "type": "nominal", "label": "Cycle"},
                    "label": {"field": "building", "type": "text", "label": "Building"},
                    "tooltip": [
                        {"field": "cycle", "label": "Cycle"},
                        {"field": "building", "label": "Building"},
                        {"field": "day14_target_g", "label": "Day 14 target", "unit": "g"},
                        {"field": "day35_weight_g", "label": "Day 35 weight", "unit": "g"},
                    ],
                },
                "layout": "full",
                "maxRows": 25,
            },
            {
                "id": "day14-day35",
                "title": "Day 14 weight and Day 35 measured weight",
                "subtitle": "19 exact-age pairs; cycle-level differences explain much of the raw relationship",
                "type": "scatter",
                "dataset": "day14_points",
                "sourceId": "farm-data",
                "encodings": {
                    "x": {"field": "day14_weight_g", "type": "quantitative", "label": "Day 14 weight", "unit": "g"},
                    "y": {"field": "day35_weight_g", "type": "quantitative", "label": "Day 35 weight", "unit": "g"},
                    "color": {"field": "cycle", "type": "nominal", "label": "Cycle"},
                    "label": {"field": "building", "type": "text", "label": "Building"},
                    "tooltip": [
                        {"field": "final_recovery_pct", "label": "Final recovery", "unit": "%"},
                        {"field": "final_average_weight_kg", "label": "Final average weight", "unit": "kg"},
                    ],
                },
                "layout": "full",
                "maxRows": 25,
            },
        ],
        "tables": [
            {
                "id": "association-table",
                "title": "Day 14 association checks",
                "subtitle": "Building-cycle grain; exact Day 14 measurements only",
                "dataset": "associations",
                "sourceId": "farm-data",
                "defaultSort": {"field": "outcome", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "outcome", "label": "Outcome", "type": "text"},
                    {"field": "pairs", "label": "Pairs", "format": "number"},
                    {"field": "raw_correlation", "label": "Raw correlation", "format": "number"},
                    {"field": "within_cycle_correlation", "label": "Within-cycle correlation", "format": "number"},
                    {"field": "within_cycle_p_value", "label": "Within-cycle p-value", "format": "number"},
                    {"field": "verdict", "label": "Interpretation", "type": "text"},
                ],
            },
            {
                "id": "model-table",
                "title": "Current predictive evidence",
                "subtitle": "Leave-one-complete-cycle-out validation",
                "dataset": "model_rows",
                "sourceId": "model-proof",
                "defaultSort": {"field": "output", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "output", "label": "Output", "type": "text"},
                    {"field": "method", "label": "Method", "type": "text"},
                    {"field": "building_cycles", "label": "Building-cycles", "format": "number"},
                    {"field": "mae", "label": "Average error", "type": "text"},
                    {"field": "target_side_accuracy", "label": "Correct target side", "type": "text"},
                    {"field": "decision_use", "label": "Practical interpretation", "type": "text"},
                ],
            },
            {
                "id": "threshold-table",
                "title": "Farmer workbook and current Canary rules",
                "subtitle": "Proposed thresholds compared with the implemented prototype",
                "dataset": "threshold_rows",
                "sourceId": "farmer-validation",
                "defaultSort": {"field": "item", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "item", "label": "Item", "type": "text"},
                    {"field": "farmer_workbook", "label": "Farmer workbook", "type": "text"},
                    {"field": "current_canary", "label": "Current Canary", "type": "text"},
                    {"field": "recommended_use", "label": "Recommended use", "type": "text"},
                ],
            },
            {
                "id": "playbook-table",
                "title": "Farmer playbook content",
                "subtitle": "Simple mappings that can strengthen deterministic recommendations",
                "dataset": "playbook_rows",
                "sourceId": "farmer-playbook",
                "defaultSort": {"field": "pattern", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "pattern", "label": "Pattern", "type": "text"},
                    {"field": "farmer_guidance", "label": "Farmer guidance", "type": "text"},
                    {"field": "canary_use", "label": "Recommended Canary use", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Project Canary Evidence and Practicality Audit"},
            {
                "id": "executive-summary",
                "type": "markdown",
                "body": "## Executive Summary\n\n- **Day 14 weight is associated with final recovery in the available data.** Across 19 building-cycles, heavier Day 14 birds had higher final recovery (raw correlation 0.63; within-cycle correlation 0.75). This is promising but observational, not proof that increasing weight causes recovery to improve.\n\n- **The Day 14-to-Day 35 weight story is weaker than it first appears.** The raw correlation is moderate (0.49), but falls to 0.13 when buildings are compared within the same cycle. Cycle-wide conditions appear to explain much of the pattern.\n\n- **There is no reliable evidence yet that Day 14 weight predicts the farm's final average liveweight.** Only 14 paired records are available and the relationship is statistically inconclusive.\n\n- **Recovery prediction is usable as a limited-data early-warning estimate; final-weight prediction is not yet personalized.** The risk system is useful for transparent triage, but its thresholds still need farm calibration.",
            },
            {
                "id": "day14-finding",
                "type": "markdown",
                "body": "## Day 14 is a promising recovery signal, not yet a proven weight predictor\n\nThe strongest result is the relationship between Day 14 weight and final recovery. In the raw data, an additional 100 g at Day 14 is associated with roughly 1.9 percentage points higher final recovery. The within-cycle sensitivity is also positive. Because this is observational data from only four cycles with exact Day 14 weights, use the result as evidence for monitoring—not a causal guarantee or intervention effect.\n\nThe target-hit framing cannot yet be tested properly: 18 of 19 measured building-cycles were below the 400 g Day 14 target and only one met it. That single building finished at 95.6% recovery, but one observation is not a comparison group.",
                "sourceId": "farm-data",
            },
            {"id": "recovery-chart", "type": "chart", "chartId": "day14-recovery"},
            {
                "id": "day35-finding",
                "type": "markdown",
                "body": "## Cycle conditions explain much of the Day 35 weight relationship\n\nAcross all 19 paired observations in this EDA view, Day 14 and Day 35 measured weights move together. However, the relationship largely disappears after removing each cycle's average. This supports a broad cycle-level statement—better-performing cycles tend to be heavier at both ages—but not a strong building-level rule that a 100 g Day 14 difference will reliably persist to Day 35.\n\nThe model-ready data contain only five Day 35 outcomes at or above the revised 1.8 kg milestone. Day 35 should therefore remain a management checkpoint, not be confused with actual final-harvest weight or treated as a well-balanced target-classification problem.",
                "sourceId": "farm-data",
            },
            {"id": "day35-chart", "type": "chart", "chartId": "day14-day35"},
            {"id": "association-detail", "type": "table", "tableId": "association-table"},
            {
                "id": "practicality",
                "type": "markdown",
                "body": "## Canary is practical for triage and recovery estimation—with clear limits\n\n**Rules-based risk:** practical because it is deterministic, inspectable and actionable. It answers which building deserves attention and why. It should not be called the probability of missing a target, and current thresholds should remain provisional.\n\n**Recovery forecast:** credible as a continuous-estimate prototype. Ordinary linear regression improves cycle-balanced MAE by 14.5% over the historical-mean baseline, but it does not beat the majority baseline for classifying 95% target hits and misses. Its drivers are associations, not causes.\n\n**Day 35 weight forecast:** remains experimental. No learned candidate cleared the predeclared improvement, within-200 g, and target-side gates, so the transparent historical remaining-gain method remains operational. More completed cycles and standardized bodyweight measurements are needed before claiming a dependable learned predictor.",
            },
            {"id": "model-evidence", "type": "table", "tableId": "model-table"},
            {
                "id": "farmer-rules",
                "type": "markdown",
                "body": "## The Farmer Validation Workbook is useful—but as a calibration layer\n\nThe workbook provides valuable owner-facing material: age-specific temperature, humidity and feed targets; absolute alert thresholds; severity rankings; response times; intervention checks; and likely causes. The safest design is to keep Canary's four core dimensions and add these items as secondary operational alerts and explanation fields. Do not automatically convert every severity number into risk points.\n\nSeveral entries need clarification before automation: humidity target and warning bands overlap; the predicted-recovery critical sign appears reversed; the mortality sheet is blank; and feed targets are grams per bird while source observations are bags. The feed sheet also contains unexplained unlabeled columns and formulas outside the visible five-column target table.",
                "sourceId": "farmer-validation",
            },
            {"id": "threshold-evidence", "type": "table", "tableId": "threshold-table"},
            {
                "id": "playbook-finding",
                "type": "markdown",
                "body": "## The intervention and root-cause mappings can strengthen recommendations now\n\nThe intervention mappings are directly usable as deterministic inspection guidance after wording cleanup and owner approval. They add useful response times—six hours for environmental problems and 24 hours for feed, weight, mortality and recovery concerns. Root-cause entries should appear as “things to check,” never as diagnoses. Responsible-person fields are mostly blank and should be completed before operational use.",
                "sourceId": "farmer-playbook",
            },
            {"id": "playbook-evidence", "type": "table", "tableId": "playbook-table"},
            {
                "id": "recommendations",
                "type": "markdown",
                "body": "## Recommended next steps\n\n1. **Recalibrate the weight dimension with Doc Raymond.** At Day 14, 14 of 19 historical observations would receive the maximum 3 weight-risk points under current cutoffs. Confirm whether this reflects genuinely severe underperformance or a target/sampling mismatch.\n\n2. **Keep the recovery model, but test a simpler feature set.** A model without bodyweight features slightly improves overall held-out MAE; choose the simplest model that remains stable as new cycles arrive.\n\n3. **Label final-weight output as a farm baseline until it becomes personalized.** Collect standardized Day 7, 14, 21, 28 and 35 samples plus final building weights.\n\n4. **Add secondary operational alerts rather than expanding the 0–12 score immediately.** Temperature, humidity, feed and absolute mortality can guide checks without obscuring the agreed four-dimensional risk logic.\n\n5. **Import the farmer playbook after a short sign-off pass.** Preserve problem pattern, action, responsible person, response time, escalation trigger and approval status.\n\n6. **Add an EDA and model-proof page to the dashboard.** Show sample counts, target-attainment distribution, exact Day 14 backtest, model-versus-baseline performance, and limitations in plain language.",
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": "## Further Questions\n\n- Is 400 g the correct Day 14 target for the same weighing method used in the historical records?\n- Are the low Day 35 measurements comparable with the final average-liveweight labels?\n- Should cumulative population-loss thresholds supplement or replace the current linear survival path?\n- Is the recovery critical threshold meant to be below 90%, rather than above 90%?\n- What is the weight of one feed bag, and are feed records issued, consumed or delivered quantities?\n- Who owns each intervention and when must veterinary escalation occur?",
            },
            {
                "id": "caveats",
                "type": "markdown",
                "body": "## Caveats and Assumptions\n\n- Exact Day 14 evidence covers 19 building-cycles in four completed cycles; final-weight pairing covers only 14.\n- Buildings repeat across cycles, and operational conditions are not randomized. Correlation does not establish causation.\n- Only one building-cycle met the 400 g Day 14 target, so target-hit versus target-miss comparisons are not reliable.\n- Recovery uses ending inventory divided by beginning inventory and still requires confirmation for transfers, culls and partial harvests.\n- The analysis notebook could not be executed because the supplied runtime lacks `nbformat`; the reproducible Python audit executed successfully and generated the reviewed results used here.",
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-08-08T12:00:00+08:00",
        "status": "ready",
        "datasets": {
            "day14_points": points,
            "associations": associations,
            "model_rows": model_rows,
            "threshold_rows": threshold_rows,
            "playbook_rows": playbook_rows,
        },
    },
    "sources": sources,
}


OUTPUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
print(OUTPUT)
