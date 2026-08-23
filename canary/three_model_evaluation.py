"""Leakage-safe evaluation and inference helpers for Canary's three-model trial.

The trial keeps two business outcomes:

* reconstructed Model 1 estimates the end-of-cycle recovery proxy;
* reconstructed Model 3 benchmarks Day 35 bodyweight from Day 21; and
* a checkpoint bundle estimates Day 35 bodyweight at Days 7, 14, 21 and 28.

Model 1 and Model 3 deliberately retain Trish's locked feature schemas and
algorithms.  Their validation is rebuilt with complete production cycles held
out, while 2026-3 remains a separate later-cycle audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor


DEVELOPMENT_CYCLES = ("2025-2", "2025-3", "2025-4", "2025-5", "2026-1", "2026-2")
AUDIT_CYCLE = "2026-3"
SEED = 20260821


@dataclass(frozen=True)
class LegacySpec:
    model_id: str
    outcome: str
    algorithm: str
    dataset: str
    features: str
    target: str
    evaluation_days: tuple[int, ...]
    unit: str


LEGACY_SPECS = (
    LegacySpec(
        model_id="model_1",
        outcome="end_of_cycle_recovery_proxy",
        algorithm="Extra Trees",
        dataset="data/gold/selected_dataset.csv",
        features="artifacts/_champion_models_final/Model1_HarvestRecovery_extra_trees_features.pkl",
        target="final_harvest_recovery",
        evaluation_days=(7, 14),
        unit="proportion",
    ),
    LegacySpec(
        model_id="model_3",
        outcome="day35_bodyweight_g",
        algorithm="XGBoost",
        dataset="artifacts/model_3_bodyweight_1_to_21/bodyweight_day21_training_dataset.csv",
        features="artifacts/_champion_models_final/Model3_BodyweightDay21_xgboost_features.pkl",
        target="bodyweight_at_day_35",
        evaluation_days=(21,),
        unit="g",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(spec: LegacySpec) -> object:
    if spec.model_id == "model_1":
        return ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=SEED,
            n_jobs=1,
        )
    return XGBRegressor(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.035,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=5.0,
        objective="reg:absoluteerror",
        random_state=SEED,
        verbosity=0,
        n_jobs=1,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    cycle_values: dict[str, dict[str, float]] = {}
    for cycle in sorted(pd.Series(cycles).astype(str).unique()):
        mask = np.asarray(pd.Series(cycles).astype(str).eq(cycle))
        error = predicted[mask] - actual[mask]
        cycle_values[cycle] = {
            "rows": int(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
        }
    error = predicted - actual
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(error)),
        "cycle_macro_mae": float(np.mean([item["mae"] for item in cycle_values.values()])),
        "worst_cycle_mae": float(max(item["mae"] for item in cycle_values.values())),
        "uncertainty_half_width_80": float(np.quantile(np.abs(error), 0.80)),
        "cycle": cycle_values,
    }


def _baseline_prediction(spec: LegacySpec, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if spec.model_id == "model_1":
        train_current = 1.0 - pd.to_numeric(train["mortality_percent_cum"], errors="coerce") / 100.0
        train_loss = train_current - pd.to_numeric(train[spec.target], errors="coerce")
        mean_loss = pd.DataFrame({"day": train["prediction_day"], "loss": train_loss}).groupby("day")["loss"].mean()
        test_current = 1.0 - pd.to_numeric(test["mortality_percent_cum"], errors="coerce") / 100.0
        return np.clip(test_current.to_numpy(float) - test["prediction_day"].map(mean_loss).fillna(train_loss.mean()).to_numpy(float), 0.0, 1.0)
    train_gain = pd.to_numeric(train[spec.target], errors="coerce") - pd.to_numeric(train["bodyweight_g"], errors="coerce")
    mean_gain = float(train_gain.mean())
    return pd.to_numeric(test["bodyweight_g"], errors="coerce").to_numpy(float) + mean_gain


def _prepare(
    source: Path,
    spec: LegacySpec,
    recovery_labels: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(source / spec.dataset)
    features = list(joblib.load(source / spec.features))
    required = {"harvest_cycle", "bldg", "prediction_day", spec.target, *features}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{spec.model_id} is missing required columns: {missing}")
    data = data.loc[data[spec.target].notna()].copy()
    data["harvest_cycle"] = data["harvest_cycle"].astype(str)
    data["bldg"] = data["bldg"].astype(str)
    data["prediction_day"] = pd.to_numeric(data["prediction_day"], errors="raise").astype(int)
    if spec.model_id == "model_1" and recovery_labels is not None:
        canonical = pd.Series(
            [recovery_labels.get((cycle, building), np.nan) for cycle, building in zip(data["harvest_cycle"], data["bldg"])],
            index=data.index,
            dtype=float,
        )
        if canonical.isna().any():
            missing_keys = data.loc[canonical.isna(), ["harvest_cycle", "bldg"]].drop_duplicates().to_dict("records")
            raise ValueError(f"Canonical recovery labels are missing for: {missing_keys}")
        data[spec.target] = canonical
    return data, features


def evaluate_legacy_model(
    source: Path,
    output: Path,
    spec: LegacySpec,
    recovery_labels: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    data, features = _prepare(source, spec, recovery_labels)
    development = data.loc[data["harvest_cycle"].isin(DEVELOPMENT_CYCLES)].copy()
    audit = data.loc[data["harvest_cycle"].eq(AUDIT_CYCLE)].copy()
    evaluated = development.loc[development["prediction_day"].isin(spec.evaluation_days)].copy()
    predictions = np.full(len(evaluated), np.nan)
    baselines = np.full(len(evaluated), np.nan)
    groups = evaluated["harvest_cycle"].to_numpy(str)

    for train_index, test_index in LeaveOneGroupOut().split(evaluated, evaluated[spec.target], groups):
        held_cycle = str(evaluated.iloc[test_index]["harvest_cycle"].iloc[0])
        training_rows = development.loc[~development["harvest_cycle"].eq(held_cycle)].copy()
        model = _model(spec)
        model.fit(training_rows[features], training_rows[spec.target])
        predictions[test_index] = np.asarray(model.predict(evaluated.iloc[test_index][features]), dtype=float)
        baselines[test_index] = _baseline_prediction(spec, training_rows, evaluated.iloc[test_index])

    actual = evaluated[spec.target].to_numpy(float)
    selected_metrics = _metrics(actual, predictions, groups)
    baseline_metrics = _metrics(actual, baselines, groups)
    improvement = (baseline_metrics["cycle_macro_mae"] - selected_metrics["cycle_macro_mae"]) / baseline_metrics["cycle_macro_mae"] * 100.0
    gate_passed = bool(improvement >= 10.0 and selected_metrics["r2"] > 0 and selected_metrics["worst_cycle_mae"] <= baseline_metrics["worst_cycle_mae"] * 1.25)

    fitted = _model(spec)
    fitted.fit(development[features], development[spec.target])
    audit_eval = audit.loc[audit["prediction_day"].isin(spec.evaluation_days)].copy()
    audit_prediction = np.asarray(fitted.predict(audit_eval[features]), dtype=float)
    audit_metrics = _metrics(
        audit_eval[spec.target].to_numpy(float),
        audit_prediction,
        audit_eval["harvest_cycle"].to_numpy(str),
    )

    model_file = output / f"{spec.model_id}.joblib"
    feature_file = output / f"{spec.model_id}_features.json"
    audit_file = output / f"{spec.model_id}_audit_features.csv.gz"
    joblib.dump(fitted, model_file)
    feature_file.write_text(json.dumps(features, indent=2), encoding="utf-8")
    audit_eval[["harvest_cycle", "bldg", "prediction_day", spec.target, *features]].to_csv(audit_file, index=False, compression="gzip")

    oof = evaluated[["harvest_cycle", "bldg", "prediction_day", spec.target]].copy()
    oof["prediction"] = predictions
    oof["baseline_prediction"] = baselines
    oof.to_csv(output / f"{spec.model_id}_oof_predictions.csv", index=False)

    audit_export = audit_eval[["harvest_cycle", "bldg", "prediction_day", spec.target]].copy()
    audit_export["prediction"] = audit_prediction
    audit_export["error"] = audit_prediction - audit_export[spec.target]
    audit_export.to_csv(output / f"{spec.model_id}_audit_predictions.csv", index=False)

    thi_features = [name for name in features if "thi" in name.lower()]
    manifest = {
        "model_id": spec.model_id,
        "outcome": spec.outcome,
        "target_column": spec.target,
        "algorithm": spec.algorithm,
        "unit": spec.unit,
        "evaluation_days": list(spec.evaluation_days),
        "development_cycles": list(DEVELOPMENT_CYCLES),
        "development_building_cycles": int(development[["harvest_cycle", "bldg"]].drop_duplicates().shape[0]),
        "audit_cycle": AUDIT_CYCLE,
        "audit_buildings": int(audit[["harvest_cycle", "bldg"]].drop_duplicates().shape[0]),
        "selected_metrics": selected_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_improvement_pct": float(improvement),
        "deployment_gate_passed": gate_passed,
        "status": "shadow" if gate_passed else "experimental",
        "feature_count": len(features),
        "thi_features": thi_features,
        "feature_policy": "Locked original feature schema; 2026-3 excluded from fitting and evaluation-based selection.",
        "target_policy": (
            "Canonical last-recorded population divided by beginning population from Canary's farm workbook."
            if spec.model_id == "model_1"
            else "Observed average building bodyweight recorded on production Day 35."
        ),
        "audit_metrics": audit_metrics,
        "model_file": model_file.name,
        "model_sha256": _sha256(model_file),
        "features_file": feature_file.name,
        "audit_features_file": audit_file.name,
        "limitations": [
            "The original feature list is retained as a legacy benchmark and was not reselected inside the corrected cohort.",
            "Repeated daily rows are not independent outcomes; all rows from a complete production cycle remain together during validation.",
            "Environmental and THI variables are predictive candidates, not causal evidence or approved intervention thresholds.",
        ],
    }
    (output / f"{spec.model_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_legacy_reconstructions(
    source: str | Path,
    output: str | Path,
    canonical_workbook: str | Path,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    from .data import load_workbook

    dataset = load_workbook(Path(canonical_workbook).resolve())
    recovery_labels = {
        (str(row.cycle_id), str(row.building_id)): float(row.final_recovery_rate)
        for row in dataset.cycles.itertuples(index=False)
        if pd.notna(row.final_recovery_rate)
    }
    models = {
        spec.model_id: evaluate_legacy_model(
            source_path,
            output_path,
            spec,
            recovery_labels if spec.model_id == "model_1" else None,
        )
        for spec in LEGACY_SPECS
    }
    manifest = {
        "experiment_version": "three-model-evaluation-1.0.0",
        "business_outcomes": ["end_of_cycle_recovery_proxy", "day35_bodyweight"],
        "model_engines": ["model_1", "model_3", "checkpoint_champion"],
        "development_cycles": list(DEVELOPMENT_CYCLES),
        "audit_cycle": AUDIT_CYCLE,
        "canonical_workbook_sha256": _sha256(Path(canonical_workbook).resolve()),
        "models": models,
        "risk_score_relationship": "Forecasts are independent planning outlooks and never add, remove, or change observed-condition risk points.",
    }
    (output_path / "legacy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@lru_cache(maxsize=4)
def load_legacy_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_model(path: str) -> object:
    return joblib.load(path)


@lru_cache(maxsize=8)
def _load_feature_rows(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def predict_reconstructed_legacy(
    model_id: str,
    cycle_id: str,
    building_id: str,
    prediction_day: int,
    bundle_dir: str | Path,
) -> dict[str, Any] | None:
    """Score a saved audit feature row for the prototype comparison view."""

    bundle = Path(bundle_dir).resolve()
    manifest = load_legacy_manifest(bundle / f"{model_id}_manifest.json")
    valid_days = [int(day) for day in manifest["evaluation_days"]]
    if int(prediction_day) not in valid_days or str(cycle_id) != str(manifest["audit_cycle"]):
        return None
    feature_rows = _load_feature_rows(str(bundle / manifest["audit_features_file"]))
    row = feature_rows.loc[
        feature_rows["harvest_cycle"].astype(str).eq(str(cycle_id))
        & feature_rows["bldg"].astype(str).eq(str(building_id))
        & feature_rows["prediction_day"].astype(int).eq(int(prediction_day))
    ]
    if row.empty:
        return None
    features = json.loads((bundle / manifest["features_file"]).read_text(encoding="utf-8"))
    fitted_model = _load_model(str(bundle / manifest["model_file"]))
    prediction = float(
        np.asarray(fitted_model.predict(row[features]), dtype=float).reshape(-1)[0]
    )
    importance = np.asarray(
        getattr(fitted_model, "feature_importances_", np.zeros(len(features))),
        dtype=float,
    )
    ranked_inputs: list[dict[str, object]] = []
    if len(importance) == len(features):
        for position in np.argsort(importance)[::-1][:10]:
            feature_name = str(features[int(position)])
            value = row.iloc[0][feature_name]
            ranked_inputs.append(
                {
                    "feature": feature_name,
                    "importance": float(importance[int(position)]),
                    "value": None if pd.isna(value) else value,
                }
            )
    width = float(manifest["selected_metrics"]["uncertainty_half_width_80"])
    return {
        "prediction": prediction,
        "interval_low": prediction - width,
        "interval_high": prediction + width,
        "prediction_day": int(prediction_day),
        "model_id": model_id,
        "algorithm": manifest["algorithm"],
        "status": manifest["status"],
        "validation_mae": float(manifest["selected_metrics"]["mae"]),
        "cycle_macro_mae": float(manifest["selected_metrics"]["cycle_macro_mae"]),
        "audit_actual": float(row.iloc[0][manifest["target_column"]]),
        "feature_count": len(features),
        "top_inputs": ranked_inputs,
        "manifest": manifest,
    }
