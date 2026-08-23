"""Leakage-safe evaluation of an extended Day-35 bodyweight model.

This experiment does not alter the Streamlit application or its model registry.
It builds one row per official weighing checkpoint (Days 7, 14, 21, and 28),
uses only information available through that checkpoint, validates by holding
out complete production cycles, and keeps cycle 2026-3 for a final audit.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
V18_ROOT = WORKSPACE / "capstone_FINAL_v18"
V19_ROOT = WORKSPACE / "canary v19"
OUTPUT_DIR = ROOT / "analysis" / "extended_model3_experiment"
CHECKPOINTS = (7, 14, 21, 28)
AUDIT_CYCLE = "2026-3"
RANDOM_STATE = 42

# Reduced extension derived from Trish's confirmed useful Model 3 context
# (Day-7 weight, humidity range, and mortality severity), plus only the new
# forward-known checkpoint growth fields. The full 94-feature CatBoost remains
# in the comparison so the effect of parsimony is visible rather than assumed.
REDUCED_EXTENSION_FEATURES = [
    "bodyweight_at_day7",
    "humidity_range",
    "high_mortality_severity",
    "latest_actual_bodyweight_g",
    "previous_checkpoint_bodyweight_g",
    "latest_checkpoint_gain_g",
    "latest_checkpoint_gain_g_per_day",
    "latest_checkpoint_gain_ratio",
    "projected_day35_from_latest_gain",
    "days_to_day35",
    "checkpoint_index",
]


@dataclass
class Candidate:
    name: str
    factory: Callable[[], object]


def _load_feature_builder():
    sys.path.insert(0, str(V18_ROOT))
    from src.features.early_prediction_dataset import EarlyPredictionDataset

    return EarlyPredictionDataset


def load_sources() -> tuple[pd.DataFrame, list[str]]:
    modeling = pd.read_csv(V18_ROOT / "data" / "gold" / "modeling_dataset.csv")
    bundle = joblib.load(
        V19_ROOT / "artifacts" / "model3_bodyweight" / "champion.joblib"
    )
    return modeling, list(bundle["feature_names"])


def build_checkpoint_frame(modeling: pd.DataFrame, trish_features: list[str]) -> pd.DataFrame:
    """Create checkpoint rows using history ending at the scoring checkpoint.

    The V18 feature builder is reused for Trish's environmental, feed,
    mortality, THI, housing, and cumulative-history definitions. Bodyweight
    values at the official checkpoint are actual recorded measurements. New
    interval-growth features are calculated only from the current and prior
    observed checkpoints.
    """
    EarlyPredictionDataset = _load_feature_builder()
    feature_builder = EarlyPredictionDataset()
    records: list[dict] = []

    for (cycle, building), group in modeling.groupby(["harvest_cycle", "bldg"]):
        group = group.sort_values("age").copy()
        day35 = group.loc[group["age"] == 35, "bodyweight_g"]
        if day35.empty:
            continue
        target = float(day35.iloc[0])
        final_recovery = float(group["final_harvest_recovery"].iloc[0])

        for checkpoint in CHECKPOINTS:
            current = group.loc[group["age"] == checkpoint]
            if current.empty or pd.isna(current["bodyweight_kgs"].iloc[0]):
                continue
            history = group.loc[group["age"] <= checkpoint]
            row = feature_builder.create_feature_vector(
                history, cycle, building, checkpoint, final_recovery
            )
            row["bodyweight_at_day_35"] = target

            # Fold-specific later; no full-dataset median is used here.
            row["day_7_bodyweight_lower_than_mean"] = np.nan

            previous_day = checkpoint - 7
            previous = group.loc[group["age"] == previous_day, "bodyweight_kgs"]
            previous_weight = (
                float(previous.iloc[0] * 1000)
                if not previous.empty and pd.notna(previous.iloc[0])
                else np.nan
            )
            current_weight = float(current["bodyweight_kgs"].iloc[0] * 1000)
            gain = current_weight - previous_weight if pd.notna(previous_weight) else np.nan
            gain_per_day = gain / 7 if pd.notna(gain) else np.nan

            row.update(
                {
                    "latest_actual_bodyweight_g": current_weight,
                    "previous_checkpoint_bodyweight_g": previous_weight,
                    "latest_checkpoint_gain_g": gain,
                    "latest_checkpoint_gain_g_per_day": gain_per_day,
                    "latest_checkpoint_gain_ratio": (
                        current_weight / previous_weight
                        if pd.notna(previous_weight) and previous_weight > 0
                        else np.nan
                    ),
                    "projected_day35_from_latest_gain": (
                        current_weight + gain_per_day * (35 - checkpoint)
                        if pd.notna(gain_per_day)
                        else np.nan
                    ),
                    "days_to_day35": 35 - checkpoint,
                    "checkpoint_index": checkpoint // 7,
                    "bodyweight_measurement_freshness_days": 0,
                }
            )
            records.append(row)

    frame = pd.DataFrame(records)

    # Start from Trish's audited 85-feature schema, then add only forward-known
    # checkpoint fields. Every missing value remains explicit until fold-local
    # imputation during validation.
    extended = [
        "latest_actual_bodyweight_g",
        "previous_checkpoint_bodyweight_g",
        "latest_checkpoint_gain_g",
        "latest_checkpoint_gain_g_per_day",
        "latest_checkpoint_gain_ratio",
        "projected_day35_from_latest_gain",
        "days_to_day35",
        "checkpoint_index",
        "bodyweight_measurement_freshness_days",
    ]
    keep = [
        "harvest_cycle",
        "bldg",
        "prediction_day",
        "bodyweight_at_day_35",
        *trish_features,
        *extended,
    ]
    missing = [column for column in keep if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Checkpoint builder is missing required fields: {missing}")
    return frame[keep].copy()


def _prepare_fold(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    day7_median = float(train["bodyweight_at_day7"].dropna().median())
    for data in (train, test):
        data["day_7_bodyweight_lower_than_mean"] = (
            data["bodyweight_at_day7"] < day7_median
        ).astype(float)
    return train[features], test[features]


def _scaled(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _trees(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", model),
        ]
    )


def candidates() -> list[Candidate]:
    return [
        Candidate("linear_regression", lambda: _scaled(LinearRegression())),
        Candidate("ridge", lambda: _scaled(Ridge(alpha=100.0))),
        Candidate(
            "huber",
            lambda: _scaled(HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=2000)),
        ),
        Candidate(
            "extra_trees",
            lambda: _trees(
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_features=0.7,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
        ),
        Candidate(
            "gradient_boosting",
            lambda: _trees(
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    loss="huber",
                )
            ),
        ),
        Candidate(
            "catboost_extended",
            lambda: _trees(
                CatBoostRegressor(
                    iterations=300,
                    depth=2,
                    learning_rate=0.03,
                    l2_leaf_reg=5.0,
                    loss_function="MAE",
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=RANDOM_STATE,
                    thread_count=1,
                )
            ),
        ),
        Candidate(
            "catboost_reduced_growth_context",
            lambda: _trees(
                CatBoostRegressor(
                    iterations=300,
                    depth=2,
                    learning_rate=0.03,
                    l2_leaf_reg=5.0,
                    loss_function="MAE",
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=RANDOM_STATE,
                    thread_count=1,
                )
            ),
        ),
        Candidate(
            "xgboost",
            lambda: _trees(
                XGBRegressor(
                    n_estimators=300,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    objective="reg:absoluteerror",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                )
            ),
        ),
    ]


def remaining_gain_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    gains = train.assign(
        remaining_gain=train["bodyweight_at_day_35"] - train["latest_actual_bodyweight_g"]
    )
    lookup = gains.groupby("prediction_day")["remaining_gain"].mean()
    return (
        test["latest_actual_bodyweight_g"]
        + test["prediction_day"].map(lookup).to_numpy()
    ).to_numpy()


def cycle_cv(
    frame: pd.DataFrame,
    features: list[str],
    model_factory: Callable[[], object] | None,
    label: str,
) -> pd.DataFrame:
    rows = []
    for holdout_cycle in sorted(frame["harvest_cycle"].unique()):
        train = frame.loc[frame["harvest_cycle"] != holdout_cycle].copy()
        test = frame.loc[frame["harvest_cycle"] == holdout_cycle].copy()
        if model_factory is None:
            prediction = remaining_gain_predictions(train, test)
        else:
            x_train, x_test = _prepare_fold(train, test, features)
            model = model_factory()
            model.fit(x_train, train["bodyweight_at_day_35"])
            prediction = model.predict(x_test)
        fold = test[
            ["harvest_cycle", "bldg", "prediction_day", "bodyweight_at_day_35"]
        ].copy()
        fold["predicted"] = np.asarray(prediction, dtype=float)
        fold["model"] = label
        rows.append(fold)
    result = pd.concat(rows, ignore_index=True)
    result["error_g"] = result["predicted"] - result["bodyweight_at_day_35"]
    result["absolute_error_g"] = result["error_g"].abs()
    return result


def summarize_predictions(predictions: pd.DataFrame) -> dict:
    actual = predictions["bodyweight_at_day_35"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    cycle_mae = predictions.groupby("harvest_cycle")["absolute_error_g"].mean()
    return {
        "pooled_mae_g": float(mean_absolute_error(actual, predicted)),
        "cycle_macro_mae_g": float(cycle_mae.mean()),
        "rmse_g": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias_g": float(np.mean(predicted - actual)),
        "worst_cycle_mae_g": float(cycle_mae.max()),
        "within_100g_pct": float((predictions["absolute_error_g"] <= 100).mean() * 100),
        "within_200g_pct": float((predictions["absolute_error_g"] <= 200).mean() * 100),
    }


def checkpoint_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for day, group in predictions.groupby("prediction_day"):
        summary = summarize_predictions(group)
        records.append({"prediction_day": int(day), **summary})
    return pd.DataFrame(records).sort_values("prediction_day")


def evaluate_original_model3_official() -> tuple[dict, pd.DataFrame]:
    """Recover Trish's official LOGO outputs and attach their day index."""
    official = pd.read_csv(
        V19_ROOT / "artifacts" / "model3_bodyweight" / "logocv_predictions.csv"
    )
    official["prediction_day"] = official.groupby("flock").cumcount() + 1
    official["error_g"] = official["predicted"] - official["actual"]
    official["absolute_error_g"] = official["error_g"].abs()
    day21 = official.loc[official["prediction_day"] == 21]
    checkpoint_analysis = pd.read_csv(
        V19_ROOT / "artifacts" / "model3_bodyweight" / "checkpoint_analysis.csv"
    )
    terminal_day21 = checkpoint_analysis.loc[
        checkpoint_analysis["scheme"] == "Terminal only (Day 21)"
    ].iloc[0]
    metrics = {
        "published_pooled_days1_21_mae_g": float(official["absolute_error_g"].mean()),
        "full_window_model_scored_at_day21_mae_g": float(
            day21["absolute_error_g"].mean()
        ),
        "terminal_day21_model_mae_g": float(terminal_day21["mae_pooled"]),
        "full_window_day21_bias_g": float(day21["error_g"].mean()),
        "full_window_day21_within_100g_pct": float(
            (day21["absolute_error_g"] <= 100).mean() * 100
        ),
        "full_window_day21_within_200g_pct": float(
            (day21["absolute_error_g"] <= 200).mean() * 100
        ),
        "validation_design": "leave one building-flock out across all 34 flocks",
        "note": (
            "103.66 g is Trish's separately retrained terminal-Day-21 model. "
            "105.69 g is the Day-21 slice of the full Day-1-to-21 champion."
        ),
    }
    return metrics, official


def train_and_audit(
    train: pd.DataFrame,
    audit: pd.DataFrame,
    features: list[str],
    model_factory: Callable[[], object] | None,
    label: str,
) -> pd.DataFrame:
    if model_factory is None:
        prediction = remaining_gain_predictions(train, audit)
    else:
        x_train, x_audit = _prepare_fold(train, audit, features)
        model = model_factory()
        model.fit(x_train, train["bodyweight_at_day_35"])
        prediction = model.predict(x_audit)
    result = audit[
        ["harvest_cycle", "bldg", "prediction_day", "bodyweight_at_day_35"]
    ].copy()
    result["predicted"] = np.asarray(prediction, dtype=float)
    result["model"] = label
    result["error_g"] = result["predicted"] - result["bodyweight_at_day_35"]
    result["absolute_error_g"] = result["error_g"].abs()
    return result


def save_chart(comparison: pd.DataFrame, checkpoint: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # Keep the extreme linear-regression result in the exact table but omit it
    # from the chart so the useful 100-200 g comparison remains readable.
    ordered = (
        comparison.loc[comparison["model"] != "linear_regression"]
        .sort_values("cycle_macro_mae_g")
        .reset_index(drop=True)
    )
    colors = [
        "#1F4E79" if name == "historical_remaining_gain" else "#5C8A32"
        for name in ordered["model"]
    ]
    axes[0].barh(ordered["model"], ordered["cycle_macro_mae_g"], color=colors)
    axes[0].invert_yaxis()
    for index, value in enumerate(ordered["cycle_macro_mae_g"]):
        axes[0].text(value + 2, index, f"{value:.0f}", va="center", fontsize=8)
    axes[0].set_xlabel("Cycle-macro MAE (g)")
    axes[0].set_title("Extended-model candidates (31 development flocks)")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].plot(
        checkpoint["prediction_day"],
        checkpoint["cycle_macro_mae_g"],
        marker="o",
        linewidth=2.2,
        color="#1F4E79",
    )
    axes[1].set_xticks(CHECKPOINTS)
    axes[1].set_xlabel("Forecast checkpoint")
    axes[1].set_ylabel("Cycle-macro MAE (g)")
    axes[1].set_title("Reduced CatBoost MAE by checkpoint")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "extended_model3_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling, trish_features = load_sources()
    frame = build_checkpoint_frame(modeling, trish_features)
    frame.to_csv(OUTPUT_DIR / "checkpoint_modeling_frame.csv", index=False)

    audit = frame.loc[frame["harvest_cycle"] == AUDIT_CYCLE].copy()
    development = frame.loc[frame["harvest_cycle"] != AUDIT_CYCLE].copy()
    features = [
        column
        for column in frame.columns
        if column
        not in {"harvest_cycle", "bldg", "prediction_day", "bodyweight_at_day_35"}
    ]

    quality = {
        "rows": int(len(frame)),
        "flocks": int(frame[["harvest_cycle", "bldg"]].drop_duplicates().shape[0]),
        "development_flocks": int(
            development[["harvest_cycle", "bldg"]].drop_duplicates().shape[0]
        ),
        "audit_flocks": int(audit[["harvest_cycle", "bldg"]].drop_duplicates().shape[0]),
        "checkpoints": sorted(frame["prediction_day"].unique().astype(int).tolist()),
        "duplicate_flock_checkpoint_rows": int(
            frame.duplicated(["harvest_cycle", "bldg", "prediction_day"]).sum()
        ),
        "missing_day35_targets": int(frame["bodyweight_at_day_35"].isna().sum()),
        "development_cycles": sorted(development["harvest_cycle"].unique().tolist()),
        "audit_cycle": AUDIT_CYCLE,
        "n_features": len(features),
        "actual_checkpoint_weight_source": "bodyweight_kgs at Days 7, 14, 21, and 28",
    }

    prediction_sets = []
    baseline = cycle_cv(development, features, None, "historical_remaining_gain")
    prediction_sets.append(baseline)
    comparison_rows = [
        {"model": "historical_remaining_gain", **summarize_predictions(baseline)}
    ]

    candidate_map: dict[str, Callable[[], object] | None] = {
        "historical_remaining_gain": None
    }
    candidate_feature_map: dict[str, list[str]] = {
        "historical_remaining_gain": features
    }
    for candidate in candidates():
        candidate_features = (
            REDUCED_EXTENSION_FEATURES
            if candidate.name == "catboost_reduced_growth_context"
            else features
        )
        predictions = cycle_cv(
            development, candidate_features, candidate.factory, candidate.name
        )
        prediction_sets.append(predictions)
        comparison_rows.append(
            {"model": candidate.name, **summarize_predictions(predictions)}
        )
        candidate_map[candidate.name] = candidate.factory
        candidate_feature_map[candidate.name] = candidate_features

    all_predictions = pd.concat(prediction_sets, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows).sort_values("cycle_macro_mae_g")
    selected_name = str(comparison.iloc[0]["model"])
    selected_predictions = all_predictions.loc[
        all_predictions["model"] == selected_name
    ].copy()
    selected_checkpoint = checkpoint_metrics(selected_predictions)

    # Day-21-only CatBoost using the same audited Trish feature family and the
    # same complete-cycle folds as the extension. This is the apples-to-apples
    # benchmark; the official v19 number uses a less strict flock-level split.
    day21 = development.loc[development["prediction_day"] == 21].copy()
    trish_day21_features = [feature for feature in trish_features if feature in features]
    original_factory = lambda: _trees(
        CatBoostRegressor(
            iterations=150,
            depth=2,
            learning_rate=0.05,
            verbose=False,
            allow_writing_files=False,
            random_seed=RANDOM_STATE,
            thread_count=1,
        )
    )
    original_strict = cycle_cv(
        day21,
        trish_day21_features,
        original_factory,
        "trish_model3_day21_strict_reconstruction",
    )
    original_strict_metrics = summarize_predictions(original_strict)

    selected_day21 = selected_predictions.loc[
        selected_predictions["prediction_day"] == 21
    ]
    selected_day21_metrics = summarize_predictions(selected_day21)
    official_metrics, official_predictions = evaluate_original_model3_official()

    # Persist research-only artifacts outside the app model registry. They are
    # intentionally marked experimental and must not be routed into Streamlit
    # without a separate promotion decision.
    selected_features = candidate_feature_map[selected_name]
    final_train = development.copy()
    final_day7_median = float(final_train["bodyweight_at_day7"].dropna().median())
    final_train["day_7_bodyweight_lower_than_mean"] = (
        final_train["bodyweight_at_day7"] < final_day7_median
    ).astype(float)
    final_model = candidate_map[selected_name]()
    if final_model is not None:
        final_model.fit(
            final_train[selected_features], final_train["bodyweight_at_day_35"]
        )
        joblib.dump(
            {
                "model": final_model,
                "model_name": selected_name,
                "status": "experimental_not_for_app",
                "feature_names": selected_features,
                "day7_weight_median_g": final_day7_median,
                "training_cycles": sorted(development["harvest_cycle"].unique().tolist()),
                "excluded_audit_cycle": AUDIT_CYCLE,
                "valid_checkpoints": list(CHECKPOINTS),
                "target": "bodyweight_at_day_35",
                "development_metrics": summarize_predictions(selected_predictions),
            },
            OUTPUT_DIR / "experimental_extended_model3.joblib",
        )

    baseline_lookup = (
        development.assign(
            remaining_gain_g=(
                development["bodyweight_at_day_35"]
                - development["latest_actual_bodyweight_g"]
            )
        )
        .groupby("prediction_day")["remaining_gain_g"]
        .mean()
        .to_dict()
    )
    (OUTPUT_DIR / "checkpoint_remaining_gain_baseline.json").write_text(
        json.dumps(
            {
                "status": "shadow_candidate_not_for_app",
                "training_cycles": sorted(development["harvest_cycle"].unique().tolist()),
                "excluded_audit_cycle": AUDIT_CYCLE,
                "mean_remaining_gain_g_by_checkpoint": {
                    str(int(day)): float(value) for day, value in baseline_lookup.items()
                },
            },
            indent=2,
        )
    )

    audit_names = list(
        dict.fromkeys(
            [
                "historical_remaining_gain",
                "catboost_extended",
                "catboost_reduced_growth_context",
                selected_name,
            ]
        )
    )
    audit_sets = []
    for name in audit_names:
        audit_sets.append(
            train_and_audit(
                development,
                audit,
                candidate_feature_map[name],
                candidate_map[name],
                name,
            )
        )
    original_audit = train_and_audit(
        day21,
        audit.loc[audit["prediction_day"] == 21],
        trish_day21_features,
        original_factory,
        "trish_model3_day21_strict_reconstruction",
    )
    audit_predictions = pd.concat([*audit_sets, original_audit], ignore_index=True)
    audit_summary = (
        audit_predictions.groupby("model", sort=False)
        .apply(lambda group: pd.Series(summarize_predictions(group)), include_groups=False)
        .reset_index()
    )

    all_predictions.to_csv(OUTPUT_DIR / "development_cycle_oof_predictions.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "candidate_comparison.csv", index=False)
    selected_checkpoint.to_csv(OUTPUT_DIR / "selected_checkpoint_metrics.csv", index=False)
    original_strict.to_csv(OUTPUT_DIR / "original_model3_strict_day21_predictions.csv", index=False)
    official_predictions.to_csv(OUTPUT_DIR / "original_model3_official_oof_with_day.csv", index=False)
    audit_predictions.to_csv(OUTPUT_DIR / "audit_2026_3_predictions.csv", index=False)
    audit_summary.to_csv(OUTPUT_DIR / "audit_2026_3_summary.csv", index=False)
    save_chart(comparison, selected_checkpoint)

    result = {
        "data_quality": quality,
        "selection_rule": (
            "Lowest cycle-macro MAE under leave-one-production-cycle-out validation; "
            "bias, worst-cycle error, checkpoint stability, audit performance, and "
            "simplicity are secondary checks."
        ),
        "development_champion": selected_name,
        "recommended_for_shadow_use": "historical_remaining_gain",
        "recommendation_reason": (
            "The reduced CatBoost has the best development cycle-macro MAE, "
            "but its advantage over the transparent baseline is modest and it "
            "is materially worse on the later-cycle audit. Keep it as the "
            "experimental extended Model 3; use the checkpoint remaining-gain "
            "method as the provisional operational comparator until more cycles "
            "confirm that the learned model generalizes."
        ),
        "selected_development_metrics": summarize_predictions(selected_predictions),
        "selected_checkpoint_metrics": selected_checkpoint.to_dict(orient="records"),
        "selected_day21_metrics": selected_day21_metrics,
        "original_model3_official": official_metrics,
        "original_model3_strict_day21_reconstruction": original_strict_metrics,
        "audit_summary": audit_summary.to_dict(orient="records"),
        "limitations": [
            "Only 31 independent building-flocks are available for development.",
            "Cycle 2026-3 contains only three buildings from one production cycle.",
            "Checkpoint bodyweights are observed weekly; daily bodyweight forecasts are not validated.",
            "Historical between-checkpoint bodyweights in V18 are curve-shaped interpolations, not measurements; official scoring rows use actual checkpoint weights.",
            "Environmental, feed, mortality, THI, and housing features are predictive context, not causal effect estimates.",
        ],
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))
