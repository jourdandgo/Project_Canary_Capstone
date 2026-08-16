"""Build the deployable Project Canary bundle from Trish's v18 handoff.

The supplied final pickles are fitted on every available flock.  Canary's
current demo cycle (2026-3) is therefore in their training data.  This build
step preserves Trish's champion algorithms, feature lists and engineered
rows, but refits deployment copies on earlier cycles only.  The current
cycle stays untouched until inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = APP_ROOT.parent / "capstone_FINAL_v18"
DEFAULT_OUTPUT = APP_ROOT / "models" / "trish_v18"
HOLDOUT_CYCLE = "2026-3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catboost() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=150,
        depth=2,
        learning_rate=0.05,
        random_state=42,
        verbose=False,
    )


def _xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=1,
    )


MODEL_SPECS = {
    "model_1": {
        "label": "Harvest recovery outlook",
        "outcome": "recovery",
        "algorithm": "Extra Trees",
        "window_end": 14,
        "dataset": "data/gold/selected_dataset.csv",
        "features": "artifacts/_champion_models_final/Model1_HarvestRecovery_extra_trees_features.pkl",
        "target": "final_harvest_recovery",
        "factory": lambda: ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "reported_mae": 0.015788,
        "reported_r2": 0.369881,
    },
    "model_2": {
        "label": "Day 35 bodyweight outlook (Day 1-14)",
        "outcome": "day35_weight",
        "algorithm": "CatBoost",
        "window_end": 14,
        "dataset": "artifacts/model_2_bodyweight_1_to_14/bodyweight_training_dataset.csv",
        "features": "artifacts/_champion_models_final/Model2_BodyweightDay14_catboost_features.pkl",
        "target": "bodyweight_at_day_35",
        "factory": _catboost,
        "reported_mae": 134.194026,
        "reported_r2": 0.143771,
    },
    "model_3": {
        "label": "Day 35 bodyweight update (Day 1-21)",
        "outcome": "day35_weight",
        "algorithm": "XGBoost",
        "window_end": 21,
        "dataset": "artifacts/model_3_bodyweight_1_to_21/bodyweight_day21_training_dataset.csv",
        "features": "artifacts/_champion_models_final/Model3_BodyweightDay21_xgboost_features.pkl",
        "target": "bodyweight_at_day_35",
        "factory": _xgboost,
        "reported_mae": 122.549051,
        "reported_r2": 0.212013,
    },
    "model_4": {
        "label": "Estimated day to 1.8 kg",
        "outcome": "age_to_1_8kg",
        "algorithm": "Gradient Boosting",
        "window_end": 14,
        "dataset": "artifacts/model_4_5_age_to_threshold_1_to_14/age_to_threshold_training_dataset.csv",
        "features": "artifacts/_champion_models_final/Model4_AgeTo1_8kg_gradient_boosting_features.pkl",
        "target": "predicted_1_8kg_age_day",
        "factory": lambda: GradientBoostingRegressor(random_state=42),
        "reported_mae": 1.415453,
        "reported_r2": 0.243696,
    },
    "model_5": {
        "label": "Estimated day to 2.0 kg",
        "outcome": "age_to_2_0kg",
        "algorithm": "CatBoost",
        "window_end": 14,
        "dataset": "artifacts/model_4_5_age_to_threshold_1_to_14/age_to_threshold_training_dataset.csv",
        "features": "artifacts/_champion_models_final/Model5_AgeTo2_0kg_catboost_features.pkl",
        "target": "predicted_2_0kg_age_day",
        "factory": _catboost,
        "reported_mae": 1.522509,
        "reported_r2": 0.173370,
    },
    "model_6": {
        "label": "Sale-window recovery outlook",
        "outcome": "sale_window_recovery",
        "algorithm": "CatBoost",
        "window_end": 14,
        "dataset": "artifacts/model_6_recovery_sale_ready_1_to_14/training_dataset.csv",
        "score_dataset": "data/gold/early_prediction_dataset.csv",
        "features": "artifacts/_champion_models_final/Model6_RecoveryAtSaleReady_catboost_features.pkl",
        "target": "recoverable_ratio_at_target_day",
        "factory": _catboost,
        "reported_mae": 0.009322,
        "reported_r2": 0.505342,
    },
}


def build_bundle(source: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_models: dict[str, object] = {}
    prediction_frames: list[pd.DataFrame] = []

    for model_id, spec in MODEL_SPECS.items():
        dataset_path = source / str(spec["dataset"])
        score_path = source / str(spec.get("score_dataset", spec["dataset"]))
        features_path = source / str(spec["features"])
        features = list(joblib.load(features_path))
        training = pd.read_csv(dataset_path)
        scoring = pd.read_csv(score_path)

        missing_train = sorted(set(features).difference(training.columns))
        missing_score = sorted(set(features).difference(scoring.columns))
        if missing_train or missing_score:
            raise ValueError(
                f"{model_id} feature mismatch: train={missing_train}, score={missing_score}"
            )

        target = str(spec["target"])
        train = training.loc[
            training[target].notna() & ~training["harvest_cycle"].astype(str).eq(HOLDOUT_CYCLE)
        ].copy()
        holdout = scoring.loc[
            scoring["harvest_cycle"].astype(str).eq(HOLDOUT_CYCLE)
            & scoring["prediction_day"].le(int(spec["window_end"]))
        ].copy()
        if train.empty or holdout.empty:
            raise ValueError(f"{model_id} lacks train or holdout rows")

        model = spec["factory"]()
        model.fit(train[features], train[target])
        prediction = np.asarray(model.predict(holdout[features]), dtype=float).reshape(-1)

        model_file = output / f"{model_id}.joblib"
        feature_file = output / f"{model_id}_features.json"
        joblib.dump(model, model_file)
        feature_file.write_text(json.dumps(features, indent=2), encoding="utf-8")

        frame = holdout[["harvest_cycle", "bldg", "prediction_day"]].copy()
        frame["model_id"] = model_id
        frame["prediction"] = prediction
        frame["feature_row"] = np.arange(len(frame), dtype=int)
        prediction_frames.append(frame)

        holdout_feature_file = output / f"{model_id}_holdout_features.csv.gz"
        holdout[["harvest_cycle", "bldg", "prediction_day", *features]].to_csv(
            holdout_feature_file, index=False, compression="gzip"
        )
        manifest_models[model_id] = {
            "label": spec["label"],
            "outcome": spec["outcome"],
            "algorithm": spec["algorithm"],
            "window_end": spec["window_end"],
            "feature_count": len(features),
            "target": target,
            "training_rows": len(train),
            "training_cycles": sorted(train["harvest_cycle"].astype(str).unique().tolist()),
            "holdout_cycle": HOLDOUT_CYCLE,
            "holdout_rows": len(holdout),
            "reported_logo_mae": spec["reported_mae"],
            "reported_logo_r2": spec["reported_r2"],
            "model_file": model_file.name,
            "model_sha256": _sha256(model_file),
            "features_file": feature_file.name,
            "holdout_features_file": holdout_feature_file.name,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "prediction_snapshot.csv", index=False)
    raw_workbook = source / "data/raw/FARM HARVEST DATA (with connected temp).xlsx"
    manifest = {
        "bundle_version": "trish-v18-prospective-2026-3",
        "source_handoff": source.name,
        "source_workbook_sha256": _sha256(raw_workbook),
        "holdout_cycle": HOLDOUT_CYCLE,
        "operating_principle": "Risk is rules-based; model outputs are supporting outlooks.",
        "deployment_note": (
            "Champion algorithms were refitted on cycles before 2026-3. "
            "The 2026-3 feature rows remain prospective inference inputs."
        ),
        "models": manifest_models,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_bundle(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
