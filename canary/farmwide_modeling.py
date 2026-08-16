"""Unified farm-wide modeling, audit, promotion, and reporting workflow."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .bodyweight_modeling_review import (
    AUDIT_CYCLE,
    CANDIDATES as BODYWEIGHT_CANDIDATES,
    CHECKPOINTS,
    DEVELOPMENT_CYCLES,
    SEED,
    feature_columns as bodyweight_feature_columns,
    run_review as run_bodyweight_review,
)
from .data import CanaryDataset, load_workbook
from .external_modeling_review import (
    RECOVERY_CANDIDATES,
    RECOVERY_COMPACT_FEATURES,
    RECOVERY_FEATURES,
    run_outcome,
)
from .farmwide_features import assert_primary_schema_has_no_identity


REBUILD_VERSION = "farmwide-rebuild-1.0.0"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def build_source_quality_audit(
    workbook: Path, dataset: CanaryDataset
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reconcile authoritative counts and high-impact modeling risks."""

    raw_daily = pd.read_excel(workbook, sheet_name="Farm Harvest Data (Daily)")
    raw_cycles = pd.read_excel(workbook, sheet_name="Farm Harvest Data (By Cycle)")
    raw_daily["Date"] = pd.to_datetime(raw_daily["Date"], errors="coerce").dt.normalize()
    raw_cycles["Start Date"] = pd.to_datetime(raw_cycles["Start Date"], errors="coerce").dt.normalize()
    raw_daily["_building"] = raw_daily["Bldg."].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    raw_cycles["_building"] = raw_cycles["Bldg."].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Use a merge rather than tuple attribute names because source headers contain spaces.
    dated = raw_daily.merge(
        raw_cycles[["Harvest Cycle", "_building", "Start Date"]],
        on=["Harvest Cycle", "_building"],
        how="left",
        validate="many_to_one",
    )
    calculated_age = (dated["Date"] - dated["Start Date"]).dt.days + 1
    reported_age = pd.to_numeric(dated["Age"], errors="coerce")
    age_mismatch = int((calculated_age.notna() & reported_age.notna() & calculated_age.ne(reported_age)).sum())

    weight_kg = pd.to_numeric(raw_daily.get("Bodyweight (kgs)"), errors="coerce")
    weight_g = pd.to_numeric(raw_daily.get("Bodyweight (g)"), errors="coerce")
    both_weight_units = weight_kg.notna() & weight_g.notna()
    unit_mismatch = int((both_weight_units & ((weight_kg * 1000 - weight_g).abs() > 1.0)).sum())

    daily = dataset.daily.sort_values(["cycle_id", "building_id", "age_day"]).copy()
    daily["population_change"] = daily.groupby(["cycle_id", "building_id"])["population"].diff()
    daily["population_mortality_gap"] = -daily["population_change"] - daily["mortality_daily"]
    beginning_inconsistent = int(
        daily.groupby(["cycle_id", "building_id"])["beginning_inventory"].nunique(dropna=True).gt(1).sum()
    )
    checks = [
        ("Authoritative canonical building-days equal 1,624", int(dataset.quality.canonical_rows != 1624), "critical"),
        ("Canonical building-day key uniqueness", int(daily.duplicated(["cycle_id", "building_id", "age_day"]).sum()), "critical"),
        ("Recorded building-cycles equal 34", int(dataset.cycles[["cycle_id", "building_id"]].drop_duplicates().shape[0] != 34), "critical"),
        ("Development building-cycles equal 31", int(dataset.cycles.loc[dataset.cycles["cycle_id"].isin(DEVELOPMENT_CYCLES), ["cycle_id", "building_id"]].drop_duplicates().shape[0] != 31), "critical"),
        ("Beginning inventory is consistent within building-cycle", beginning_inconsistent, "critical"),
        ("Date and production age agree with placement date", age_mismatch, "high"),
        ("Observed kg and g bodyweight fields agree", unit_mismatch, "high"),
        ("Population does not exceed beginning inventory", int((daily["population"] > daily["beginning_inventory"]).sum()), "high"),
        ("Daily mortality is non-negative", int((daily["mortality_daily"] < 0).sum()), "high"),
        ("Environmental aggregation remains one row per building-day", int(daily.duplicated(["cycle_id", "building_id", "age_day"]).sum()), "critical"),
        ("Observed bodyweights remain explicitly measured", int((daily["weight_measured"] != daily["bodyweight_kg"].notna()).sum()), "critical"),
        ("Population changes reconcile exactly to daily mortality", int((daily["population_mortality_gap"].abs() > 0.5).fillna(False).sum()), "warning"),
    ]
    check_frame = pd.DataFrame(
        [
            {
                "check": name,
                "failed_rows": failed,
                "severity": severity,
                "status": "pass" if failed == 0 else "flagged",
            }
            for name, failed, severity in checks
        ]
    )
    profile = {
        "source_workbook": str(workbook),
        "source_sha256": _sha256(workbook),
        "raw_daily_rows": int(len(raw_daily)),
        "canonical_building_days": int(dataset.quality.canonical_rows),
        "total_building_cycles": int(dataset.cycles[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "development_building_cycles": int(dataset.cycles.loc[dataset.cycles["cycle_id"].isin(DEVELOPMENT_CYCLES), ["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "later_cycle_buildings": int(dataset.cycles.loc[dataset.cycles["cycle_id"].eq(AUDIT_CYCLE), "building_id"].nunique()),
        "temperature_coverage_pct": float(dataset.quality.temperature_coverage_pct),
        "humidity_coverage_pct": float(dataset.quality.humidity_coverage_pct),
        "zone_aggregated_days": int(dataset.quality.zone_aggregated_days),
        "maximum_environment_sections": int(dataset.quality.maximum_environment_sections),
        "observed_weight_measurement_days": int(dataset.quality.weight_measurement_days),
        "population_reconciliation_flag_days": int((daily["population_mortality_gap"].abs() > 0.5).fillna(False).sum()),
        "feed_policy": "Excluded from primary models and operational alerts until units are confirmed.",
        "recovery_endpoint_policy": "Last recorded population / beginning population; 2026-3 remains provisional at Day 35.",
    }
    return profile, check_frame


def _cycle_rmse(predictions: pd.DataFrame, candidate: str, actual: str, predicted: str) -> pd.Series:
    selected = predictions.loc[predictions["candidate"].eq(candidate)]
    return selected.groupby("cycle_id").apply(
        lambda frame: float(mean_squared_error(frame[actual], frame[predicted]) ** 0.5),
        include_groups=False,
    )


def _promotion_gate(
    outcome: str,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    checkpoints: pd.DataFrame,
    shadow: str,
    baseline: str,
    current: str,
    interval_metrics: dict[str, Any],
    audit_shadow: dict[str, Any],
    audit_current_rmse: float | None,
) -> dict[str, Any]:
    suffix = "_g" if outcome == "weight" else ""
    cycle_rmse_field = f"cycle_macro_rmse{suffix}"
    mae_field = f"mae{suffix}"
    bias_field = f"bias{suffix}"
    worst_field = f"worst_cycle_rmse{suffix}"
    shadow_row = comparison.loc[comparison["candidate"].eq(shadow)].iloc[0]
    baseline_row = comparison.loc[comparison["candidate"].eq(baseline)].iloc[0]
    current_row = comparison.loc[comparison["candidate"].eq(current)].iloc[0]
    improvement_baseline = (float(baseline_row[cycle_rmse_field]) - float(shadow_row[cycle_rmse_field])) / float(baseline_row[cycle_rmse_field]) * 100
    improvement_current = (float(current_row[cycle_rmse_field]) - float(shadow_row[cycle_rmse_field])) / float(current_row[cycle_rmse_field]) * 100
    actual_col, predicted_col = ("actual_g", "predicted_g") if outcome == "weight" else ("actual", "predicted")
    shadow_cycle = _cycle_rmse(predictions, shadow, actual_col, predicted_col)
    baseline_cycle = _cycle_rmse(predictions, baseline, actual_col, predicted_col)
    cycle_wins = int((shadow_cycle < baseline_cycle).sum())
    checkpoint_frame = checkpoints.loc[checkpoints["candidate"].isin([shadow, baseline])] if "candidate" in checkpoints else pd.DataFrame()
    if checkpoint_frame.empty:
        checkpoint_pass = False
    else:
        metric = "rmse_g" if outcome == "weight" else "rmse"
        pivot = checkpoint_frame.pivot(index="review_day", columns="candidate", values=metric)
        checkpoint_pass = bool((pivot[shadow] <= pivot[baseline] * 1.05).all())
    audit_rmse_field = "rmse_g" if outcome == "weight" else "rmse"
    audit_not_worse = bool(
        audit_current_rmse is not None
        and float(audit_shadow[audit_rmse_field]) <= float(audit_current_rmse) * 1.10
    )
    retrospective = {
        "rmse_improvement_vs_baseline_pct": improvement_baseline,
        "rmse_improvement_vs_current_pct": improvement_current,
        "at_least_10pct_better_than_both": bool(improvement_baseline >= 10 and improvement_current >= 10),
        "positive_held_out_r2": bool(float(shadow_row["r2"]) > 0),
        "mae_not_worse_than_5pct": bool(float(shadow_row[mae_field]) <= float(baseline_row[mae_field]) * 1.05),
        "bias_within_limit": bool(abs(float(shadow_row[bias_field])) <= (50.0 if outcome == "weight" else 0.5)),
        "worst_cycle_not_worse_than_10pct": bool(float(shadow_row[worst_field]) <= float(baseline_row[worst_field]) * 1.10),
        "beats_baseline_in_at_least_four_cycles": bool(cycle_wins >= 4),
        "cycles_beating_baseline": cycle_wins,
        "checkpoint_guardrail_passed": checkpoint_pass,
        "interval_80_coverage_credible": bool(0.70 <= float(interval_metrics["coverage_80"]) <= 0.90),
        "interval_90_coverage_credible": bool(0.80 <= float(interval_metrics["coverage_90"]) <= 1.00),
        "later_cycle_not_materially_worse": audit_not_worse,
    }
    retrospective_passed = bool(all(value for key, value in retrospective.items() if isinstance(value, bool)))
    return {
        "outcome": outcome,
        "shadow_candidate": shadow,
        "baseline": baseline,
        "current_canary_method": current,
        "retrospective_checks": retrospective,
        "retrospective_gate_passed": retrospective_passed,
        "prospective_cycles_required": 3,
        "prospective_cycles_completed": 0,
        "operational_promotion_allowed": False,
        "decision": "Shadow evaluation" if retrospective_passed else "Retain operational baseline; challenger remains research-only",
    }


def _checkpoint_candidate_metrics(predictions: pd.DataFrame, outcome: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    actual_col, prediction_col = ("actual_g", "predicted_g") if outcome == "weight" else ("actual", "predicted")
    factor = 1.0 if outcome == "weight" else 100.0
    for (candidate, day), frame in predictions.groupby(["candidate", "review_day"]):
        records.append(
            {
                "candidate": candidate,
                "review_day": int(day),
                "rmse_g" if outcome == "weight" else "rmse": float(mean_squared_error(frame[actual_col], frame[prediction_col]) ** 0.5 * factor),
            }
        )
    return pd.DataFrame(records)


def _write_report(output: Path, manifest: dict[str, Any]) -> Path:
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render compact Markdown without pandas' optional tabulate dependency."""
        display = frame.copy()
        for column in display.select_dtypes(include=["float"]).columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        headers = [str(column).replace("_", " ") for column in display.columns]
        rows = [[str(value).replace("|", "\\|") for value in row] for row in display.itertuples(index=False, name=None)]
        return "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            + ["| " + " | ".join(row) + " |" for row in rows]
        )

    recovery = manifest["outcomes"]["recovery"]
    weight = manifest["outcomes"]["weight"]
    recovery_top = pd.read_csv(output / "recovery" / "top_five_models.csv")
    weight_top = pd.read_csv(output / "bodyweight" / "top_five_models.csv")
    recovery_display = recovery_top[[
        "candidate", "cycle_macro_rmse", "rmse", "mae", "r2", "bias",
        "worst_cycle_rmse", "balanced_target_accuracy",
    ]].rename(columns={"cycle_macro_rmse": "cycle_macro_rmse_pp", "rmse": "rmse_pp", "mae": "mae_pp", "bias": "bias_pp"})
    weight_display = weight_top[[
        "candidate", "cycle_macro_rmse_g", "rmse_g", "mae_g", "r2", "bias_g",
        "worst_cycle_rmse_g", "within_200g_rate",
    ]]
    lines = [
        "# Project Canary Farm-Wide Modeling Rebuild",
        "",
        "## Technical summary",
        "",
        f"Canary now has one reproducible farm-wide research workflow for both outcomes. The recovery one-standard-error champion is **{recovery['selected_candidate']}** and its lowest-error shadow is **{recovery['shadow_candidate']}**. The bodyweight one-standard-error champion is **{weight['champion']}** and its lowest-error shadow is **{weight['shadow_candidate']}**.",
        "",
        "No application model was automatically replaced. Promotion remains blocked until every retrospective check passes and the challenger completes three new prospective cycles in shadow mode.",
        "",
        "## Recovery top five",
        "",
        markdown_table(recovery_display.head(5)),
        "",
        "## Day 35 bodyweight top five",
        "",
        markdown_table(weight_display.head(5)),
        "",
        "## Scope, data, and definitions",
        "",
        "The corrected workbook is authoritative: 1,624 building-days, 34 building-cycles, 31 development outcomes across six cycles, and three locked 2026-3 audit buildings. Recovery is last recorded population divided by beginning population. Weight is an actually observed Day 35 building average. Days 7, 14, 21, and 28 are the validated checkpoints.",
        "",
        "## Method and robustness",
        "",
        "Models use nested leave-one-complete-cycle-out validation, fold-local preprocessing and tuning, equal building-cycle influence, cycle-macro RMSE selection, a one-standard-error simplicity rule, expanding-window stress tests, grouped conformal intervals, identity sensitivity checks, and a leave-one-building-label-out stress test. Exact building and Tags/Lags are excluded from all primary models.",
        "",
        "## Limitations and decision",
        "",
        "Only six development cycles are available, environmental coverage is incomplete, target-side outcomes are imbalanced, feed units remain unresolved, and the 2026-3 recovery endpoint is provisional. SHAP and permutation importance describe predictive association, not causation.",
        "",
        f"- Recovery: **{manifest['promotion']['recovery']['decision']}**.",
        f"- Bodyweight: **{manifest['promotion']['weight']['decision']}**.",
        "",
        "## Recommended next steps",
        "",
        "1. Continue standardized Days 7, 14, 21, 28, and 35 weighing with sample size and scale metadata.",
        "2. Verify the true harvest endpoint, transfers, culls, and ending population.",
        "3. Complete building-day temperature and humidity capture and sensor calibration.",
        "4. Resolve feed units before enabling feed features or alerts.",
        "5. Re-evaluate promotion after at least three new complete cycles.",
        "",
        "## Further questions",
        "",
        "What recovery and weight error is acceptable for a management decision, and will the owner approve the environmental thresholds and standardized sampling protocol?",
    ]
    path = output / "PROJECT_CANARY_FARMWIDE_MODELING_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def finalize_existing_rebuild(output: str | Path) -> dict[str, Any]:
    """Rebuild only the cross-outcome manifest/report from completed outcome outputs."""
    output_path = Path(output).resolve()
    recovery_manifest = json.loads((output_path / "recovery" / "manifest.json").read_text(encoding="utf-8"))
    bodyweight_manifest = json.loads((output_path / "bodyweight" / "manifest.json").read_text(encoding="utf-8"))
    promotion = json.loads((output_path / "promotion_decisions.json").read_text(encoding="utf-8"))
    source = json.loads((output_path / "source_audit.json").read_text(encoding="utf-8"))
    manifest = {
        "rebuild_version": REBUILD_VERSION,
        "seed": SEED,
        "source": source,
        "package_versions": _package_versions(),
        "architecture": {
            "model_scope": "One shared farm-wide model per outcome, scored separately for each building snapshot.",
            "observed_risk": "Transparent 0-12 deterministic score remains separate.",
            "forecast_status": "80% interval drives Likely below / Uncertain / Likely meets; 90% interval is supplemental.",
            "management_priority": "Deterministic reconciliation of observed risk, forecast downside, and evidence quality.",
            "application_models_replaced": False,
        },
        "outcomes": {"recovery": recovery_manifest, "weight": bodyweight_manifest},
        "promotion": promotion,
        "artifact_files": {
            "recovery_champion": "recovery/champion.joblib",
            "recovery_shadow": "recovery/shadow_challenger.joblib",
            "weight_champion": "bodyweight/champion.joblib",
            "weight_shadow": "bodyweight/shadow_challenger.joblib",
        },
    }
    report_path = _write_report(output_path, manifest)
    manifest["technical_report"] = str(report_path)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    return manifest


def run_farmwide_rebuild(
    workbook: str | Path,
    output: str | Path | None = None,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"This frozen review uses seed {SEED}; received {seed}.")
    workbook_path = Path(workbook).resolve()
    root = Path(__file__).resolve().parents[1]
    output_path = Path(output).resolve() if output else root / "outputs" / "farmwide_modeling_rebuild"
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(workbook_path)
    quality_profile, quality_checks = build_source_quality_audit(workbook_path, dataset)
    quality_checks.to_csv(output_path / "data_quality_checks.csv", index=False)
    (output_path / "source_audit.json").write_text(json.dumps(quality_profile, indent=2, default=_json_default), encoding="utf-8")
    if quality_checks.loc[quality_checks["severity"].eq("critical"), "failed_rows"].sum() > 0:
        raise AssertionError("Critical source-quality checks failed; see data_quality_checks.csv")

    assert_primary_schema_has_no_identity(RECOVERY_FEATURES)
    assert_primary_schema_has_no_identity(RECOVERY_COMPACT_FEATURES)
    bodyweight_schemas = {
        str(day): {
            feature_set: bodyweight_feature_columns(day, feature_set)
            for feature_set in sorted({candidate.feature_set for candidate in BODYWEIGHT_CANDIDATES})
        }
        for day in CHECKPOINTS
    }
    for schemas in bodyweight_schemas.values():
        for columns in schemas.values():
            assert_primary_schema_has_no_identity(columns)
    feature_schema = {
        "identity_policy": "Exact building and Tags/Lags excluded from primary models; evaluated only in sensitivity outputs.",
        "feed_policy": "Excluded from primary models because units are unresolved.",
        "recovery_primary_features": RECOVERY_FEATURES,
        "recovery_compact_features": RECOVERY_COMPACT_FEATURES,
        "bodyweight_checkpoint_feature_schemas": bodyweight_schemas,
    }
    (output_path / "feature_schemas.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    pd.DataFrame([candidate.__dict__ | {"outcome": "recovery"} for candidate in RECOVERY_CANDIDATES] + [asdict(candidate) | {"outcome": "weight"} for candidate in BODYWEIGHT_CANDIDATES]).to_csv(output_path / "candidate_registry.csv", index=False)

    recovery_manifest = run_outcome(dataset, "recovery", output_path, root)
    bodyweight_manifest = run_bodyweight_review(workbook_path, output_path / "bodyweight")

    recovery_comparison = pd.read_csv(output_path / "recovery" / "candidate_comparison.csv")
    recovery_predictions = pd.read_csv(output_path / "recovery" / "oof_predictions.csv")
    recovery_checkpoint_all = _checkpoint_candidate_metrics(recovery_predictions, "recovery")
    recovery_checkpoint_all.to_csv(output_path / "recovery" / "all_candidate_checkpoint_metrics.csv", index=False)
    recovery_audit_current = recovery_manifest.get("published_canary_benchmark", {}).get("later_cycle_rmse")
    recovery_gate = _promotion_gate(
        "recovery", recovery_comparison, recovery_predictions, recovery_checkpoint_all,
        recovery_manifest["shadow_candidate"], "age_band_remaining_loss", "remaining_ols",
        recovery_manifest["shadow_interval_coverage"], recovery_manifest["shadow_later_cycle_audit_metrics"],
        float(recovery_audit_current) if recovery_audit_current is not None else None,
    )

    weight_comparison = pd.read_csv(output_path / "bodyweight" / "model_comparison.csv")
    weight_predictions = pd.read_csv(output_path / "bodyweight" / "all_oof_predictions.csv")
    weight_checkpoint_all = _checkpoint_candidate_metrics(weight_predictions, "weight")
    weight_checkpoint_all.to_csv(output_path / "bodyweight" / "all_candidate_checkpoint_metrics.csv", index=False)
    current_weight_manifest = json.loads((root / "models" / "day35_weight_manifest.json").read_text(encoding="utf-8"))
    current_weight_audit = current_weight_manifest.get("prospective_latest_cycle_audit", {}).get("metrics", {}).get("rmse_kg")
    weight_gate = _promotion_gate(
        "weight", weight_comparison, weight_predictions, weight_checkpoint_all,
        bodyweight_manifest["shadow_candidate"], "historical_remaining_gain", "historical_remaining_gain",
        bodyweight_manifest["shadow_interval_metrics"], bodyweight_manifest["shadow_later_cycle_audit_metrics"],
        float(current_weight_audit) * 1000 if current_weight_audit is not None else None,
    )

    combined_top_five = pd.concat(
        [
            pd.read_csv(output_path / "recovery" / "top_five_models.csv").assign(outcome="recovery"),
            pd.read_csv(output_path / "bodyweight" / "top_five_models.csv").assign(outcome="weight"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_top_five.to_csv(output_path / "top_five_models_by_outcome.csv", index=False)
    promotion = {"recovery": recovery_gate, "weight": weight_gate}
    (output_path / "promotion_decisions.json").write_text(json.dumps(promotion, indent=2, default=_json_default), encoding="utf-8")
    manifest = {
        "rebuild_version": REBUILD_VERSION,
        "seed": seed,
        "source": quality_profile,
        "package_versions": _package_versions(),
        "architecture": {
            "model_scope": "One shared farm-wide model per outcome, scored separately for each building snapshot.",
            "observed_risk": "Transparent 0-12 deterministic score remains separate.",
            "forecast_status": "80% interval drives Likely below / Uncertain / Likely meets; 90% interval is supplemental.",
            "management_priority": "Deterministic reconciliation of observed risk, forecast downside, and evidence quality.",
            "application_models_replaced": False,
        },
        "outcomes": {"recovery": recovery_manifest, "weight": bodyweight_manifest},
        "promotion": promotion,
        "artifact_files": {
            "recovery_champion": "recovery/champion.joblib",
            "recovery_shadow": "recovery/shadow_challenger.joblib",
            "weight_champion": "bodyweight/champion.joblib",
            "weight_shadow": "bodyweight/shadow_challenger.joblib",
        },
    }
    report_path = _write_report(output_path, manifest)
    manifest["technical_report"] = str(report_path)
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest
