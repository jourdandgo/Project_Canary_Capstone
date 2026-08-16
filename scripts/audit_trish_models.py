"""Compare Trish's Extra Trees approach with Project Canary's canonical models.

The audit deliberately uses complete-harvest-cycle holdouts.  It reports both
row-level and cycle-balanced metrics because the training tables contain
multiple as-of snapshots for each building outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("/Users/jourdan.go/Downloads/PROJECT CANARY/canary_app")
TRISH = Path("/Users/jourdan.go/Downloads/PROJECT CANARY/trish_capstone_work")
OUT = ROOT / "analysis" / "trish_model_audit"
OUT.mkdir(parents=True, exist_ok=True)


def linear_pipeline(kind: str) -> TransformedTargetRegressor:
    model = LinearRegression() if kind == "linear" else Ridge(alpha=10.0)
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def tree_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", model),
        ]
    )


def metric_row(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    outcome: str,
    fold_mae: list[float],
) -> dict[str, float | str]:
    factor = 100.0 if outcome == "recovery" else 1000.0
    cycle_mae = [mean_absolute_error(y[groups == g], pred[groups == g]) for g in np.unique(groups)]
    target = 0.95 if outcome == "recovery" else 1.8
    tolerance = 0.02 if outcome == "recovery" else 0.2
    return {
        "model": name,
        "cycle_macro_mae": float(np.mean(cycle_mae) * factor),
        "pooled_mae": float(mean_absolute_error(y, pred) * factor),
        "pooled_rmse": float(mean_squared_error(y, pred) ** 0.5 * factor),
        "pooled_r2": float(r2_score(y, pred)),
        "bias": float(np.mean(pred - y) * factor),
        "fold_mae_std": float(np.std(fold_mae) * factor),
        "within_tolerance_pct": float(np.mean(np.abs(pred - y) <= tolerance) * 100.0),
        "target_side_accuracy_pct": float(np.mean((pred >= target) == (y >= target)) * 100.0),
    }


def canonical_recovery() -> pd.DataFrame:
    rows = pd.read_csv(CANONICAL / "outputs/model_ready/recovery_training.csv")
    manifest = json.loads((CANONICAL / "models/recovery_manifest.json").read_text())
    features = list(manifest["feature_columns"])
    x = rows[features]
    y = rows["final_recovery_proxy_y"].to_numpy(float)
    groups = rows["validation_cycle"].astype(str).to_numpy()
    models = {
        "Historical mean": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor())]),
        "Linear regression": linear_pipeline("linear"),
        "Ridge (compact)": linear_pipeline("ridge"),
        "Random forest": tree_pipeline(
            RandomForestRegressor(
                n_estimators=250, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1
            )
        ),
        "Extra Trees (Trish algorithm)": tree_pipeline(
            ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)
        ),
    }
    splitter = LeaveOneGroupOut()
    results = []
    for name, template in models.items():
        pred = np.full(len(rows), np.nan)
        fold_mae: list[float] = []
        for train, test in splitter.split(x, y, groups):
            model = clone(template)
            model.fit(x.iloc[train], y[train])
            fold_pred = np.clip(model.predict(x.iloc[test]), 0.0, 1.0)
            pred[test] = fold_pred
            fold_mae.append(mean_absolute_error(y[test], fold_pred))
        results.append(metric_row(name, y, pred, groups, "recovery", fold_mae))
    return pd.DataFrame(results).sort_values("cycle_macro_mae").reset_index(drop=True)


def historical_remaining_gain(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame
) -> np.ndarray:
    fallback = float(train_rows["actual_day35_weight_kg_y"].mean())
    gains = (
        train_rows.assign(
            remaining_gain=train_rows["actual_day35_weight_kg_y"]
            - train_rows["current_weight_kg"]
        )
        .groupby("measurement_day")["remaining_gain"]
        .mean()
    )
    return np.asarray(
        [
            float(row.current_weight_kg + gains.get(row.measurement_day, fallback - row.current_weight_kg))
            for row in test_rows.itertuples()
        ]
    )


def canonical_weight() -> pd.DataFrame:
    rows = pd.read_csv(CANONICAL / "outputs/model_ready/day35_weight_training.csv")
    identifiers = {
        "cycle_id",
        "building_id",
        "validation_cycle",
        "actual_day35_weight_kg_y",
        # This is derived directly from the label and therefore cannot be an X.
        "remaining_gain_to_day35_kg_y",
    }
    features = [column for column in rows.columns if column not in identifiers]
    x = rows[features]
    y = rows["actual_day35_weight_kg_y"].to_numpy(float)
    groups = rows["validation_cycle"].astype(str).to_numpy()
    models = {
        "Linear regression": linear_pipeline("linear"),
        "Ridge": linear_pipeline("ridge"),
        "Random forest": tree_pipeline(
            RandomForestRegressor(
                n_estimators=500, max_depth=3, min_samples_leaf=3, random_state=42, n_jobs=1
            )
        ),
        "Extra Trees (Trish algorithm)": tree_pipeline(
            ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)
        ),
    }
    splitter = LeaveOneGroupOut()
    predictions = {"Historical remaining gain": np.full(len(rows), np.nan)}
    fold_errors = {"Historical remaining gain": []}
    for name in models:
        predictions[name] = np.full(len(rows), np.nan)
        fold_errors[name] = []
    for train, test in splitter.split(x, y, groups):
        baseline_pred = historical_remaining_gain(rows.iloc[train], rows.iloc[test])
        predictions["Historical remaining gain"][test] = baseline_pred
        fold_errors["Historical remaining gain"].append(mean_absolute_error(y[test], baseline_pred))
        for name, template in models.items():
            model = clone(template)
            model.fit(x.iloc[train], y[train])
            fold_pred = np.clip(model.predict(x.iloc[test]), 0.1, 3.5)
            predictions[name][test] = fold_pred
            fold_errors[name].append(mean_absolute_error(y[test], fold_pred))
    results = [
        metric_row(name, y, pred, groups, "weight", fold_errors[name])
        for name, pred in predictions.items()
    ]
    return pd.DataFrame(results).sort_values("cycle_macro_mae").reset_index(drop=True)


def trish_whole_cycle() -> pd.DataFrame:
    rows = pd.read_csv(TRISH / "data/gold/selected_dataset.csv")
    submitted = joblib.load(TRISH / "extra_trees.pkl")
    encoded = pd.get_dummies(
        rows.drop(columns=["harvest_recovery", "harvest_cycle", "bldg", "prediction_day"]),
        drop_first=True,
    )
    x = encoded.reindex(columns=list(submitted.feature_names_in_), fill_value=0)
    y = rows["harvest_recovery"].to_numpy(float)
    groups = rows["harvest_cycle"].astype(str).to_numpy()
    models = {
        "Historical mean": DummyRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees (submitted settings)": ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=1
        ),
    }
    splitter = LeaveOneGroupOut()
    results = []
    for name, template in models.items():
        pred = np.full(len(rows), np.nan)
        fold_mae: list[float] = []
        for train, test in splitter.split(x, y, groups):
            model = clone(template)
            model.fit(x.iloc[train], y[train])
            fold_pred = np.clip(model.predict(x.iloc[test]), 0.0, 1.0)
            pred[test] = fold_pred
            fold_mae.append(mean_absolute_error(y[test], fold_pred))
        results.append(metric_row(name, y, pred, groups, "recovery", fold_mae))
    return pd.DataFrame(results).sort_values("cycle_macro_mae").reset_index(drop=True)


def submitted_artifact_check() -> dict[str, object]:
    model = joblib.load(TRISH / "extra_trees.pkl")
    x_test = joblib.load(TRISH / "artifacts/X_test.pkl")
    y_test = np.asarray(joblib.load(TRISH / "artifacts/y_test.pkl"), dtype=float)
    encoded = pd.get_dummies(x_test, drop_first=True).reindex(
        columns=list(model.feature_names_in_), fill_value=0
    )
    pred = model.predict(encoded)
    importance = pd.DataFrame(
        {"feature": model.feature_names_in_, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "trish_extra_trees_feature_importance.csv", index=False)
    return {
        "test_rows": int(len(y_test)),
        "mae_percentage_points": float(mean_absolute_error(y_test, pred) * 100.0),
        "rmse_percentage_points": float(mean_squared_error(y_test, pred) ** 0.5 * 100.0),
        "r2": float(r2_score(y_test, pred)),
        "feature_count": int(len(model.feature_names_in_)),
        "top_features": importance.head(10).to_dict(orient="records"),
    }


def data_audit() -> dict[str, object]:
    rows = pd.read_csv(TRISH / "data/gold/selected_dataset.csv")
    outcomes = rows[["harvest_cycle", "bldg", "harvest_recovery"]].drop_duplicates()
    return {
        "snapshot_rows": int(len(rows)),
        "building_cycle_outcomes": int(len(outcomes)),
        "cycles": sorted(outcomes["harvest_cycle"].astype(str).unique().tolist()),
        "features_after_ids_and_target": int(rows.shape[1] - 4),
        "interpolated_weight_share_pct": float(rows["bodyweight_g_is_interpolated"].mean() * 100.0),
        "target_curve_fallback_weight_share_pct": float(
            rows["bodyweight_g_is_extrapolated_fallback"].mean() * 100.0
        ),
        "current_cycle_2026_3_included_as_labeled_outcome": bool(
            (outcomes["harvest_cycle"].astype(str) == "2026-3").any()
        ),
    }


def main() -> None:
    canonical_recovery_result = canonical_recovery()
    canonical_weight_result = canonical_weight()
    trish_cycle_result = trish_whole_cycle()
    canonical_recovery_result.to_csv(OUT / "canonical_recovery_five_model_comparison.csv", index=False)
    canonical_weight_result.to_csv(OUT / "canonical_weight_five_model_comparison.csv", index=False)
    trish_cycle_result.to_csv(OUT / "trish_recovery_whole_cycle_comparison.csv", index=False)
    summary = {
        "data_audit": data_audit(),
        "submitted_artifact": submitted_artifact_check(),
        "canonical_recovery": canonical_recovery_result.to_dict(orient="records"),
        "canonical_weight": canonical_weight_result.to_dict(orient="records"),
        "trish_whole_cycle_recovery": trish_cycle_result.to_dict(orient="records"),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
