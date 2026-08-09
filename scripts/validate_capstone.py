"""Reproducible capstone acceptance checks for Project Canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canary.data import CanaryDataset, load_workbook
from canary.forecast import attach_forecasts
from canary.recommendations import apply_recommendations, load_recommendation_playbook
from canary.risk import load_risk_rules, score_cycle_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = Path(
    os.getenv(
        "CANARY_DEFAULT_WORKBOOK",
        str(PROJECT_ROOT.parent / "FARM HARVEST DATA.xlsx"),
    )
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "capstone_validation.json"

SCENARIOS = (
    {
        "id": "day_14_early_warning",
        "label": "Day 14 early-warning view",
        "cycle_id": "2025-5",
        "as_of": "2025-11-25",
        "purpose": "Confirm the agreed early-control window remains fully supported.",
    },
    {
        "id": "day_22_continuous_cycle",
        "label": "Day 22 continuous-cycle view",
        "cycle_id": "2026-3",
        "as_of": "2026-07-25",
        "purpose": "Confirm risk and forecasts continue after Day 14.",
    },
    {
        "id": "mixed_building_states",
        "label": "Staggered placements and record coverage",
        "cycle_id": "2026-2",
        "as_of": "2026-06-12",
        "purpose": "Confirm records-ended, active, and inactive buildings coexist safely.",
    },
    {
        "id": "day_48_stale_weight",
        "label": "Day 48 with a stale weight",
        "cycle_id": "2025-5",
        "as_of": "2026-02-12",
        "purpose": "Confirm forecasts and risk remain available after Day 35 without presenting an old weight as current.",
    },
    {
        "id": "missing_daily_entry",
        "label": "Missing current-day observations",
        "cycle_id": "2026-3",
        "as_of": "2026-07-26",
        "purpose": "Confirm delayed data is explicit while calculations use only the latest known observations.",
    },
)

FIVE_OUTPUT_COLUMNS = (
    "risk_rating",
    "why_primary",
    "predicted_final_recovery",
    "projected_day35_weight_kg",
    "recommended_action",
)


def _plain(value: Any) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _check(checks: list[dict[str, Any]], name: str, condition: bool, evidence: str) -> None:
    checks.append({"check": name, "passed": bool(condition), "evidence": evidence})


def _scenario_checks(
    dataset: CanaryDataset,
    scenario: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    risk = score_cycle_snapshot(dataset, scenario["cycle_id"], pd.Timestamp(scenario["as_of"]))
    forecast = attach_forecasts(dataset, risk)
    view = apply_recommendations(forecast)
    checks: list[dict[str, Any]] = []

    _check(checks, "Six-building coverage", len(view) == 6, f"Rendered {len(view)} building rows.")
    _check(
        checks,
        "Required outputs present",
        set(FIVE_OUTPUT_COLUMNS).issubset(view.columns),
        "Risk, Why, two predicted outcomes, and recommended action fields are present.",
    )
    _check(
        checks,
        "Risk remains rules-based",
        risk["risk_score"].astype("Float64").equals(view["risk_score"].astype("Float64")),
        "Forecasting and recommendation steps did not change any risk score.",
    )

    eligible = view[view["state"].isin(["Active", "Incomplete", "Records ended"])]
    _check(
        checks,
        "Eligible buildings are explained",
        eligible["risk_score"].notna().all()
        and eligible["why_primary"].astype(str).str.len().gt(0).all()
        and eligible["recommended_action"].astype(str).str.len().gt(0).all(),
        f"{len(eligible)} reviewable building(s) have traceable risk and action outputs.",
    )
    _check(
        checks,
        "Recovery forecast behavior",
        eligible["predicted_final_recovery"].notna().all(),
        f"{eligible['predicted_final_recovery'].notna().sum()} of {len(eligible)} eligible building(s) have a recovery estimate.",
    )

    _check(
        checks,
        "Day 35 weight projection behavior",
        eligible.loc[
            eligible["latest_weight_kg"].notna(), "projected_day35_weight_kg"
        ].notna().all()
        and eligible.loc[
            eligible["latest_weight_kg"].isna(), "projected_day35_weight_kg"
        ].isna().all(),
        f"{eligible['projected_day35_weight_kg'].notna().sum()} of {len(eligible)} reviewable building(s) have a measured-weight-based Day 35 projection; no value is invented without a weight.",
    )

    inactive = view[view["state"] == "Inactive"]
    _check(
        checks,
        "Inactive buildings do not receive forecasts",
        inactive["predicted_final_recovery"].isna().all()
        and inactive["projected_day35_weight_kg"].isna().all(),
        f"{len(inactive)} inactive building(s) correctly withheld.",
    )
    records_ended = view[view["state"] == "Records ended"]
    _check(
        checks,
        "Records-ended buildings do not claim confirmed harvest",
        records_ended["recovery_forecast_status"].astype(str).str.contains(
            "harvest not confirmed", case=False
        ).all(),
        f"{len(records_ended)} records-ended building(s) retain a last forecast with an explicit harvest-status caveat.",
    )

    scenario_id = scenario["id"]
    if scenario_id == "day_14_early_warning":
        _check(
            checks,
            "Day 14 boundary",
            (eligible["cycle_day"].astype(int) == 14).all(),
            "All placed buildings are evaluated on Day 14.",
        )
    elif scenario_id == "day_22_continuous_cycle":
        _check(
            checks,
            "After-Day-14 continuity",
            (eligible["cycle_day"].astype(int) > 14).all(),
            "All placed buildings continue receiving outputs on Day 22.",
        )
    elif scenario_id == "mixed_building_states":
        states = set(view["state"])
        _check(
            checks,
            "Mixed operational states",
            {"Records ended", "Active", "Inactive"}.issubset(states),
            "One view safely contains records-ended, current, and not-yet-placed buildings.",
        )
    elif scenario_id == "day_48_stale_weight":
        later = eligible[eligible["cycle_day"].astype(int) > 35]
        _check(
            checks,
            "After-Day-35 continuity and freshness",
            not later.empty
            and later["weight_staleness_days"].gt(0).all()
            and later["weight_freshness"].astype(str).str.startswith("Stale").all(),
            "The Day 48 building remains evaluated and its Day 35 weight is visibly stale.",
        )
    elif scenario_id == "missing_daily_entry":
        _check(
            checks,
            "Missing-day continuity",
            not eligible.empty
            and eligible["state"].eq("Incomplete").all()
            and eligible["recovery_forecast_status"].eq(
                "Forecast available — latest recorded data used"
            ).all(),
            "Incomplete buildings use the latest known data and are not mistaken for unplaced flocks.",
        )

    return view, checks


def build_validation_payload(workbook: Path) -> dict[str, Any]:
    dataset = load_workbook(workbook)
    risk_rules = load_risk_rules()
    recommendations = load_recommendation_playbook()
    scenario_results = []
    total_checks = 0
    passed_checks = 0

    for scenario in SCENARIOS:
        view, checks = _scenario_checks(dataset, scenario)
        total_checks += len(checks)
        passed_checks += sum(item["passed"] for item in checks)
        rows = []
        for _, row in view.iterrows():
            rows.append(
                {
                    key: _plain(row.get(key))
                    for key in (
                        "building_id",
                        "state",
                        "cycle_day",
                        "latest_operational_day",
                        "data_freshness",
                        "latest_weight_kg",
                        "weight_measurement_day",
                        "weight_freshness",
                        "risk_score",
                        "risk_rating",
                        "risk_pattern",
                        "why_primary",
                        "score_equation",
                        "predicted_final_recovery",
                        "recovery_target_gap_pp",
                        "recovery_forecast_status",
                        "projected_day35_weight_kg",
                        "day35_weight_target_gap_kg",
                        "day35_weight_status",
                        "recommended_action",
                        "recommendation_rule_id",
                        "recommendation_guidance_status",
                    )
                }
            )
        scenario_results.append({**scenario, "checks": checks, "building_outputs": rows})

    return {
        "report": "Project Canary capstone validation — Day 35 storyline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_workbook": workbook.name,
        "source_workbook_path": str(workbook),
        "source_rows": dataset.quality.source_rows,
        "canonical_building_days": dataset.quality.canonical_rows,
        "blocking_data_errors": list(dataset.quality.blocking_errors),
        "data_warnings": list(dataset.quality.warnings),
        "risk_rule_version": risk_rules["version"],
        "risk_rule_approval_status": risk_rules["approval_status"],
        "recommendation_rule_version": recommendations["version"],
        "recommendation_approval_status": recommendations["approval_status"],
        "overall_status": "PASS" if passed_checks == total_checks else "FAIL",
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "scenarios": scenario_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_validation_payload(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{payload['overall_status']}: {payload['passed_checks']}/{payload['total_checks']} "
        f"checks passed. Evidence: {args.output}"
    )
    return 0 if payload["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
