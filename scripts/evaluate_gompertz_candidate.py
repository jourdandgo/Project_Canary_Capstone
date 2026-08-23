"""Evaluate a Gompertz Day 35 bodyweight candidate without changing the app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canary import build_day35_training_rows, load_workbook


CHECKPOINTS = (7, 14, 21, 28)
CURVE_DAYS = np.asarray([7.0, 14.0, 21.0, 28.0, 35.0])


def gompertz(age: np.ndarray | float, asymptote: float, rate: float, inflection: float) -> np.ndarray:
    """Standard three-parameter Gompertz growth curve."""

    age_array = np.asarray(age, dtype=float)
    return asymptote * np.exp(-np.exp(-rate * (age_array - inflection)))


def _building_weight_history(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cycle_id",
        "building_id",
        "actual_day35_weight_kg",
        *[f"weight_day_{day}_kg" for day in CHECKPOINTS],
    ]
    return rows[columns].groupby(["cycle_id", "building_id"], as_index=False).max()


def fit_gompertz(rows: pd.DataFrame) -> dict[str, float]:
    """Fit one robust population curve using training buildings only.

    The curve is used only from Day 7 through Day 35. The mature-weight
    asymptote is weakly identified over this short window, so it is bounded and
    reported as a limitation rather than interpreted biologically.
    """

    history = _building_weight_history(rows)
    ages: list[float] = []
    weights: list[float] = []
    for _, record in history.iterrows():
        for day in CURVE_DAYS:
            column = (
                "actual_day35_weight_kg"
                if int(day) == 35
                else f"weight_day_{int(day)}_kg"
            )
            value = record[column]
            if pd.notna(value):
                ages.append(float(day))
                weights.append(float(value))

    x = np.asarray(ages, dtype=float)
    y = np.asarray(weights, dtype=float)
    result = least_squares(
        lambda parameters: gompertz(x, *parameters) - y,
        x0=np.asarray([4.0, 0.05, 34.0]),
        bounds=(np.asarray([1.8, 0.015, 5.0]), np.asarray([8.0, 0.30, 60.0])),
        loss="huber",
        f_scale=0.10,
        max_nfev=20_000,
    )
    return {
        "asymptote_kg": float(result.x[0]),
        "growth_rate": float(result.x[1]),
        "inflection_day": float(result.x[2]),
        "fit_cost": float(result.cost),
        "converged": bool(result.success),
    }


def _curve_values(parameters: dict[str, float], ages: np.ndarray | float) -> np.ndarray:
    return gompertz(
        ages,
        parameters["asymptote_kg"],
        parameters["growth_rate"],
        parameters["inflection_day"],
    )


def predict_anchored(rows: pd.DataFrame, parameters: dict[str, float]) -> np.ndarray:
    """Preserve today's measured deviation and add curve-implied remaining gain."""

    current = rows["current_weight_kg"].to_numpy(float)
    review_day = rows["measurement_day"].to_numpy(float)
    remaining_gain = _curve_values(parameters, 35.0) - _curve_values(parameters, review_day)
    return current + remaining_gain


def predict_unanchored(rows: pd.DataFrame, parameters: dict[str, float]) -> np.ndarray:
    """Return the generic population endpoint when no current weight exists."""

    return np.full(len(rows), float(_curve_values(parameters, 35.0)))


def _metrics(frame: pd.DataFrame, actual: str, predicted: str) -> dict[str, Any]:
    y = frame[actual].to_numpy(float)
    p = frame[predicted].to_numpy(float)
    error = p - y
    cycle_mae = frame.assign(_absolute_error=np.abs(error)).groupby("cycle_id")["_absolute_error"].mean()
    return {
        "rows": int(len(frame)),
        "independent_building_cycles": int(frame[["cycle_id", "building_id"]].drop_duplicates().shape[0]),
        "mae_g": float(mean_absolute_error(y, p) * 1000),
        "cycle_macro_mae_g": float(cycle_mae.mean() * 1000),
        "rmse_g": float(mean_squared_error(y, p) ** 0.5 * 1000),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "bias_g": float(error.mean() * 1000),
        "within_100g_rate": float(np.mean(np.abs(error) <= 0.1)),
        "within_200g_rate": float(np.mean(np.abs(error) <= 0.2)),
        "worst_cycle_mae_g": float(cycle_mae.max() * 1000),
    }


def _checkpoint_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for day in CHECKPOINTS:
        subset = frame.loc[frame["measurement_day"].eq(day)]
        values = _metrics(subset, "actual_day35_weight_kg", "predicted_day35_weight_kg")
        output.append({"review_day": day, **values})
    return output


def _paired_cycle_bootstrap(
    checkpoint: pd.DataFrame,
    gompertz_frame: pd.DataFrame,
    repeats: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    keys = ["cycle_id", "building_id", "review_day"]
    left = checkpoint[keys + ["actual_g", "predicted_g"]].rename(
        columns={"predicted_g": "checkpoint_predicted_g"}
    )
    right = gompertz_frame.assign(
        actual_g=gompertz_frame["actual_day35_weight_kg"] * 1000,
        gompertz_predicted_g=gompertz_frame["predicted_day35_weight_kg"] * 1000,
        review_day=gompertz_frame["measurement_day"],
    )[keys + ["actual_g", "gompertz_predicted_g"]]
    paired = left.merge(right, on=keys, suffixes=("_checkpoint", "_gompertz"), validate="one_to_one")
    if not np.allclose(paired["actual_g_checkpoint"], paired["actual_g_gompertz"]):
        raise AssertionError("Checkpoint and Gompertz outcomes do not reconcile.")
    paired["mae_difference_g"] = (
        np.abs(paired["gompertz_predicted_g"] - paired["actual_g_gompertz"])
        - np.abs(paired["checkpoint_predicted_g"] - paired["actual_g_gompertz"])
    )
    cycle_difference = paired.groupby("cycle_id")["mae_difference_g"].mean()
    rng = np.random.default_rng(seed)
    cycles = cycle_difference.index.to_numpy()
    estimates = np.asarray(
        [cycle_difference.loc[rng.choice(cycles, len(cycles), replace=True)].mean() for _ in range(repeats)]
    )
    return {
        "point_difference_g": float(cycle_difference.mean()),
        "ci95_low_g": float(np.quantile(estimates, 0.025)),
        "ci95_high_g": float(np.quantile(estimates, 0.975)),
        "probability_gompertz_lower_mae": float(np.mean(estimates < 0)),
    }


def evaluate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    all_rows = build_day35_training_rows(dataset, include_latest_cycle=True)
    development = all_rows.loc[~all_rows["cycle_id"].astype(str).eq("2026-3")].reset_index(drop=True)
    audit = all_rows.loc[all_rows["cycle_id"].astype(str).eq("2026-3")].reset_index(drop=True)
    if development[["cycle_id", "building_id"]].drop_duplicates().shape[0] != 31:
        raise AssertionError("Expected 31 development building-cycles.")
    if audit[["cycle_id", "building_id"]].drop_duplicates().shape[0] != 3:
        raise AssertionError("Expected three untouched 2026-3 audit buildings.")

    oof_parts: list[pd.DataFrame] = []
    fold_parameters: list[dict[str, Any]] = []
    for held_cycle in sorted(development["cycle_id"].astype(str).unique()):
        train = development.loc[~development["cycle_id"].astype(str).eq(held_cycle)]
        test = development.loc[development["cycle_id"].astype(str).eq(held_cycle)].copy()
        parameters = fit_gompertz(train)
        test["predicted_day35_weight_kg"] = predict_anchored(test, parameters)
        test["error_kg"] = test["predicted_day35_weight_kg"] - test["actual_day35_weight_kg"]
        oof_parts.append(test)
        fold_parameters.append({"held_cycle": held_cycle, **parameters})
    oof = pd.concat(oof_parts).sort_values(["cycle_id", "building_id", "measurement_day"]).reset_index(drop=True)

    frozen_parameters = fit_gompertz(development)
    audit = audit.copy()
    audit["predicted_day35_weight_kg"] = predict_anchored(audit, frozen_parameters)
    audit["error_kg"] = audit["predicted_day35_weight_kg"] - audit["actual_day35_weight_kg"]

    # Missing-weight fallback is evaluated once per independent building outcome.
    development_outcomes = _building_weight_history(development)[["cycle_id", "building_id", "actual_day35_weight_kg"]]
    blank_parts: list[pd.DataFrame] = []
    for held_cycle in sorted(development_outcomes["cycle_id"].astype(str).unique()):
        train = development.loc[~development["cycle_id"].astype(str).eq(held_cycle)]
        test = development_outcomes.loc[development_outcomes["cycle_id"].astype(str).eq(held_cycle)].copy()
        parameters = fit_gompertz(train)
        test["gompertz_blank_prediction_kg"] = predict_unanchored(test, parameters)
        historical_mean = _building_weight_history(train)["actual_day35_weight_kg"].mean()
        test["historical_mean_blank_prediction_kg"] = historical_mean
        blank_parts.append(test)
    blank_oof = pd.concat(blank_parts).sort_values(["cycle_id", "building_id"]).reset_index(drop=True)
    audit_outcomes = _building_weight_history(audit)[["cycle_id", "building_id", "actual_day35_weight_kg"]]
    audit_outcomes["gompertz_blank_prediction_kg"] = predict_unanchored(audit_outcomes, frozen_parameters)
    audit_outcomes["historical_mean_blank_prediction_kg"] = _building_weight_history(development)["actual_day35_weight_kg"].mean()

    checkpoint_manifest = json.loads((ROOT / "models" / "three_model" / "checkpoint_champion" / "manifest.json").read_text())
    legacy_manifest = json.loads((ROOT / "models" / "three_model" / "legacy" / "legacy_manifest.json").read_text())
    checkpoint_oof = pd.read_csv(ROOT / "models" / "three_model" / "checkpoint_champion" / "champion_oof_predictions.csv")
    paired = _paired_cycle_bootstrap(checkpoint_oof, oof)

    blank_gompertz = blank_oof.rename(columns={"gompertz_blank_prediction_kg": "predicted"})
    blank_mean = blank_oof.rename(columns={"historical_mean_blank_prediction_kg": "predicted"})
    blank_audit_gompertz = audit_outcomes.rename(columns={"gompertz_blank_prediction_kg": "predicted"})
    blank_audit_mean = audit_outcomes.rename(columns={"historical_mean_blank_prediction_kg": "predicted"})

    results = {
        "experiment_scope": "Training and comparison only; no Canary app inference or interface changes.",
        "source": str(ROOT / "data" / "FARM HARVEST DATA.xlsx"),
        "source_sha256": dataset.source_sha256,
        "development_cycles": sorted(development["cycle_id"].astype(str).unique().tolist()),
        "development_building_cycles": 31,
        "audit_cycle": "2026-3",
        "audit_buildings": 3,
        "gompertz_definition": "W(t) = A * exp(-exp(-k * (t - M)))",
        "operational_candidate": "Current measured weight + Gompertz-implied remaining gain from checkpoint to Day 35",
        "frozen_parameters": frozen_parameters,
        "fold_parameters": fold_parameters,
        "development_metrics": _metrics(oof, "actual_day35_weight_kg", "predicted_day35_weight_kg"),
        "checkpoint_metrics": _checkpoint_metrics(oof),
        "audit_metrics": _metrics(audit, "actual_day35_weight_kg", "predicted_day35_weight_kg"),
        "audit_checkpoint_metrics": _checkpoint_metrics(audit),
        "paired_comparison_with_checkpoint": paired,
        "blank_weight_fallback": {
            "interpretation": "Generic cohort prior only; it is identical for every building in a fold and is not an individualized forecast.",
            "gompertz_development_mae_g": float(mean_absolute_error(blank_oof["actual_day35_weight_kg"], blank_oof["gompertz_blank_prediction_kg"]) * 1000),
            "historical_mean_development_mae_g": float(mean_absolute_error(blank_oof["actual_day35_weight_kg"], blank_oof["historical_mean_blank_prediction_kg"]) * 1000),
            "gompertz_audit_mae_g": float(mean_absolute_error(audit_outcomes["actual_day35_weight_kg"], audit_outcomes["gompertz_blank_prediction_kg"]) * 1000),
            "historical_mean_audit_mae_g": float(mean_absolute_error(audit_outcomes["actual_day35_weight_kg"], audit_outcomes["historical_mean_blank_prediction_kg"]) * 1000),
        },
        "existing_models": {
            "model_1_recovery": legacy_manifest["models"]["model_1"]["selected_metrics"],
            "model_1_recovery_audit": legacy_manifest["models"]["model_1"]["audit_metrics"],
            "model_3_day21_bodyweight": legacy_manifest["models"]["model_3"]["selected_metrics"],
            "model_3_day21_bodyweight_audit": legacy_manifest["models"]["model_3"]["audit_metrics"],
            "checkpoint_bodyweight": checkpoint_manifest["champion_metrics"],
            "checkpoint_bodyweight_audit": checkpoint_manifest["later_cycle_audit_metrics"],
        },
        "recommendation": "Do not replace the checkpoint model. The Gompertz candidate's small historical MAE advantage is not material, its paired uncertainty includes no improvement, and it is weaker on the prospective audit. Keep it as a research comparison only.",
    }

    oof.to_csv(output_dir / "gompertz_oof_predictions.csv", index=False)
    audit.to_csv(output_dir / "gompertz_2026_3_audit_predictions.csv", index=False)
    blank_oof.to_csv(output_dir / "blank_weight_fallback_oof.csv", index=False)
    audit_outcomes.to_csv(output_dir / "blank_weight_fallback_2026_3.csv", index=False)
    pd.DataFrame(results["checkpoint_metrics"]).to_csv(output_dir / "gompertz_checkpoint_metrics.csv", index=False)
    pd.DataFrame(fold_parameters).to_csv(output_dir / "gompertz_fold_parameters.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "engine": "Model 1 - Extra Trees",
                "outcome": "End-of-cycle recovery proxy",
                "forecast_points": "Days 7 and 14",
                "development_mae": legacy_manifest["models"]["model_1"]["selected_metrics"]["mae"] * 100,
                "development_mae_unit": "percentage points",
                "audit_mae": legacy_manifest["models"]["model_1"]["audit_metrics"]["mae"] * 100,
                "audit_mae_unit": "percentage points",
                "interpretation": "Different outcome; not directly comparable with bodyweight models.",
            },
            {
                "engine": "Model 3 - XGBoost",
                "outcome": "Day 35 bodyweight",
                "forecast_points": "Day 21",
                "development_mae": legacy_manifest["models"]["model_3"]["selected_metrics"]["mae"],
                "development_mae_unit": "grams",
                "audit_mae": legacy_manifest["models"]["model_3"]["audit_metrics"]["mae"],
                "audit_mae_unit": "grams",
                "interpretation": "Shadow Day 21 benchmark.",
            },
            {
                "engine": "Checkpoint - historical remaining gain",
                "outcome": "Day 35 bodyweight",
                "forecast_points": "Days 7, 14, 21, and 28",
                "development_mae": checkpoint_manifest["champion_metrics"]["mae_g"],
                "development_mae_unit": "grams",
                "audit_mae": checkpoint_manifest["later_cycle_audit_metrics"]["mae_g"],
                "audit_mae_unit": "grams",
                "interpretation": "Provisional default.",
            },
            {
                "engine": "Gompertz - anchored remaining gain",
                "outcome": "Day 35 bodyweight",
                "forecast_points": "Days 7, 14, 21, and 28",
                "development_mae": results["development_metrics"]["mae_g"],
                "development_mae_unit": "grams",
                "audit_mae": results["audit_metrics"]["mae_g"],
                "audit_mae_unit": "grams",
                "interpretation": "Research comparison; no app integration recommended.",
            },
        ]
    )
    comparison.to_csv(output_dir / "four_engine_comparison.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "gompertz_evaluation")
    args = parser.parse_args()
    results = evaluate(args.output)
    summary = {
        "gompertz_development_mae_g": results["development_metrics"]["mae_g"],
        "gompertz_audit_mae_g": results["audit_metrics"]["mae_g"],
        "paired_difference_vs_checkpoint_g": results["paired_comparison_with_checkpoint"],
        "blank_weight_fallback": results["blank_weight_fallback"],
        "recommendation": results["recommendation"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
