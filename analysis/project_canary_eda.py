"""Reproducible EDA for Day 14 evidence and model practicality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

from canary.data import load_workbook
from canary.modeling import (
    FEATURE_COLUMNS,
    _clip,
    _pipeline,
    build_modeling_snapshots,
    complete_cycle_ids,
    load_final_weight_labels,
)


PROJECT = Path(__file__).resolve().parents[2]
FARM_DATA = PROJECT / "FARM HARVEST DATA.xlsx"
PERFORMANCE = PROJECT / "Farm Performance Summary.xlsx"
FARMER_VALIDATION = PROJECT / "Farmer Validation Workbook.xlsx"
OUTPUT = Path(__file__).with_name("eda_results.json")


def association(frame: pd.DataFrame, x: str, y: str) -> dict[str, object]:
    pair = frame[[x, y, "cycle_id"]].dropna().copy()
    pair[x] = pd.to_numeric(pair[x], errors="coerce").astype(float)
    pair[y] = pd.to_numeric(pair[y], errors="coerce").astype(float)
    result: dict[str, object] = {"n": len(pair)}
    if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return result
    pearson = stats.pearsonr(pair[x], pair[y])
    spearman = stats.spearmanr(pair[x], pair[y])
    slope, _, _, _, _ = stats.linregress(pair[x], pair[y])
    pair["x_within_cycle"] = pair[x] - pair.groupby("cycle_id")[x].transform("mean")
    pair["y_within_cycle"] = pair[y] - pair.groupby("cycle_id")[y].transform("mean")
    valid = pair.loc[
        (pair["x_within_cycle"].abs() > 1e-12)
        | (pair["y_within_cycle"].abs() > 1e-12)
    ]
    within_r = (
        valid["x_within_cycle"].corr(valid["y_within_cycle"])
        if len(valid) >= 3
        and valid["x_within_cycle"].nunique() > 1
        and valid["y_within_cycle"].nunique() > 1
        else np.nan
    )
    cycle_count = int(pair["cycle_id"].nunique())
    within_df = len(pair) - cycle_count - 1
    within_p = None
    within_slope = None
    if pd.notna(within_r) and within_df > 0 and abs(within_r) < 1:
        within_t = float(within_r) * np.sqrt(within_df / (1 - float(within_r) ** 2))
        within_p = float(2 * stats.t.sf(abs(within_t), df=within_df))
        denominator = float((valid["x_within_cycle"] ** 2).sum())
        if denominator > 0:
            within_slope = float(
                (valid["x_within_cycle"] * valid["y_within_cycle"]).sum()
                / denominator
                * 0.1
            )
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "slope_per_100g": float(slope * 0.1),
            "within_cycle_r": float(within_r) if pd.notna(within_r) else None,
            "within_cycle_p": within_p,
            "within_cycle_slope_per_100g": within_slope,
            "cycles": cycle_count,
        }
    )
    return result


def build_results() -> dict[str, object]:
    dataset = load_workbook(FARM_DATA)
    completed_cycles = set(complete_cycle_ids(dataset))
    weights = dataset.daily.loc[
        dataset.daily["weight_measured"],
        ["cycle_id", "building_id", "age_day", "record_date", "bodyweight_kg"],
    ].copy()
    target_lookup = dataset.targets.set_index("age_day")["target_weight_kg"]
    weights["target_weight_kg"] = weights["age_day"].map(target_lookup)
    weights["target_attainment_pct"] = (
        weights["bodyweight_kg"] / weights["target_weight_kg"] * 100
    )

    day14 = weights.loc[weights["age_day"].eq(14)].rename(
        columns={
            "bodyweight_kg": "day14_weight_kg",
            "target_weight_kg": "day14_target_kg",
            "target_attainment_pct": "day14_target_attainment_pct",
            "record_date": "day14_date",
        }
    )[
        [
            "cycle_id",
            "building_id",
            "day14_date",
            "day14_weight_kg",
            "day14_target_kg",
            "day14_target_attainment_pct",
        ]
    ]
    day35 = weights.loc[weights["age_day"].eq(35)].rename(
        columns={
            "bodyweight_kg": "day35_weight_kg",
            "target_weight_kg": "day35_target_kg",
            "target_attainment_pct": "day35_target_attainment_pct",
            "record_date": "day35_date",
        }
    )[
        [
            "cycle_id",
            "building_id",
            "day35_date",
            "day35_weight_kg",
            "day35_target_kg",
            "day35_target_attainment_pct",
        ]
    ]
    outcomes = dataset.cycles.loc[
        dataset.cycles["cycle_id"].isin(completed_cycles),
        [
            "cycle_id",
            "building_id",
            "beginning_inventory",
            "ending_inventory",
            "final_recovery_rate",
            "end_date",
        ],
    ].copy()
    outcomes["recomputed_recovery"] = (
        outcomes["ending_inventory"] / outcomes["beginning_inventory"]
    )

    summary = load_final_weight_labels(PERFORMANCE)
    final_weight = outcomes[["cycle_id", "building_id", "end_date"]].merge(
        summary, on=["cycle_id", "building_id"], how="left", validate="one_to_one"
    )
    final_weight = final_weight.merge(
        dataset.cycles[["cycle_id", "building_id", "start_date"]],
        on=["cycle_id", "building_id"],
        how="left",
    )
    final_weight["summary_to_start_days"] = (
        final_weight["summary_record_date"] - final_weight["start_date"]
    ).dt.days
    final_weight["label_valid"] = (
        final_weight["final_average_weight_kg"].between(0.5, 3.5, inclusive="both")
        & final_weight["summary_to_start_days"].between(-14, 14, inclusive="both")
    )
    final_weight = final_weight.loc[
        final_weight["label_valid"],
        ["cycle_id", "building_id", "final_average_weight_kg"],
    ]

    evidence = outcomes.merge(day14, on=["cycle_id", "building_id"], how="left")
    evidence = evidence.merge(day35, on=["cycle_id", "building_id"], how="left")
    evidence = evidence.merge(
        final_weight, on=["cycle_id", "building_id"], how="left"
    )
    coverage = {
        "completed_cycles": len(completed_cycles),
        "completed_building_cycles": len(outcomes),
        "exact_day14_weights": int(evidence["day14_weight_kg"].notna().sum()),
        "exact_day35_weights": int(evidence["day35_weight_kg"].notna().sum()),
        "paired_day14_day35": int(
            evidence[["day14_weight_kg", "day35_weight_kg"]]
            .notna()
            .all(axis=1)
            .sum()
        ),
        "paired_day14_recovery": int(
            evidence[["day14_weight_kg", "recomputed_recovery"]]
            .notna()
            .all(axis=1)
            .sum()
        ),
        "paired_day14_final_weight": int(
            evidence[["day14_weight_kg", "final_average_weight_kg"]]
            .notna()
            .all(axis=1)
            .sum()
        ),
        "trusted_final_weight_labels": int(
            evidence["final_average_weight_kg"].notna().sum()
        ),
    }
    associations = {
        "day14_to_day35_weight": association(
            evidence, "day14_weight_kg", "day35_weight_kg"
        ),
        "day14_to_final_average_weight": association(
            evidence, "day14_weight_kg", "final_average_weight_kg"
        ),
        "day14_to_final_recovery": association(
            evidence, "day14_weight_kg", "recomputed_recovery"
        ),
        "day14_target_attainment_to_recovery": association(
            evidence, "day14_target_attainment_pct", "recomputed_recovery"
        ),
    }

    snapshots = build_modeling_snapshots(dataset, "recovery")
    weight_features = {
        "latest_weight_kg",
        "weight_target_kg",
        "weight_gap_pct",
        "weight_measurement_day",
        "weight_staleness_days",
    }
    feature_sets = {
        "full_model": FEATURE_COLUMNS,
        "without_weight_features": [
            column for column in FEATURE_COLUMNS if column not in weight_features
        ],
    }
    y = snapshots["target"].to_numpy(float)
    groups = snapshots["cycle_id"].astype(str).to_numpy()
    day14_mask = snapshots["cycle_day"].eq(14).to_numpy()
    ablation: dict[str, object] = {}
    for name, columns in feature_sets.items():
        prediction = np.full(len(snapshots), np.nan)
        for train_index, test_index in LeaveOneGroupOut().split(
            snapshots[columns], y, groups
        ):
            model = _pipeline("ridge")
            model.fit(snapshots.iloc[train_index][columns], y[train_index])
            prediction[test_index] = _clip(
                model.predict(snapshots.iloc[test_index][columns]), "recovery"
            )
        ablation[name] = {
            "day14_mae_pp": float(
                mean_absolute_error(y[day14_mask], prediction[day14_mask]) * 100
            ),
            "day14_target_side_accuracy_pct": float(
                np.mean(
                    (prediction[day14_mask] >= 0.95) == (y[day14_mask] >= 0.95)
                )
                * 100
            ),
            "overall_mae_pp": float(mean_absolute_error(y, prediction) * 100),
        }
    target_groups = evidence.loc[evidence["day14_weight_kg"].notna()].copy()
    target_groups["day14_target_status"] = np.where(
        target_groups["day14_weight_kg"] >= target_groups["day14_target_kg"],
        "Met/exceeded",
        "Below",
    )
    group_summary = (
        target_groups.groupby("day14_target_status")
        .agg(
            building_cycles=("building_id", "size"),
            mean_day14_weight_kg=("day14_weight_kg", "mean"),
            mean_day35_weight_kg=("day35_weight_kg", "mean"),
            day35_weight_n=("day35_weight_kg", "count"),
            mean_final_weight_kg=("final_average_weight_kg", "mean"),
            final_weight_n=("final_average_weight_kg", "count"),
            mean_final_recovery=("recomputed_recovery", "mean"),
        )
        .reset_index()
    )

    recovery_manifest = json.loads(Path("models/recovery_manifest.json").read_text())
    weight_manifest = json.loads(Path("models/weight_manifest.json").read_text())
    model_summary = {
        "recovery_selected_model": recovery_manifest["selected_model"],
        "recovery_building_cycles": recovery_manifest["training_building_cycles"],
        "recovery_overall_mae_pp": recovery_manifest["selected_metrics"]["mae"]
        * 100,
        "recovery_overall_rmse_pp": recovery_manifest["selected_metrics"]["rmse"]
        * 100,
        "recovery_target_side_accuracy_pct": recovery_manifest[
            "selected_metrics"
        ]["target_side_accuracy"]
        * 100,
        "recovery_day14_mae_pp": recovery_manifest["day14_backtest_metrics"]["mae"]
        * 100,
        "recovery_day14_target_side_accuracy_pct": recovery_manifest[
            "day14_backtest_metrics"
        ]["target_side_accuracy"]
        * 100,
        "weight_selected_model": weight_manifest["selected_model"],
        "weight_building_cycles": weight_manifest["training_building_cycles"],
        "weight_mae_kg": weight_manifest["selected_metrics"]["mae"],
        "weight_target_side_accuracy_pct": weight_manifest["selected_metrics"][
            "target_side_accuracy"
        ]
        * 100,
    }
    farmer_book = pd.read_excel(
        FARMER_VALIDATION, sheet_name=None, header=None, engine="openpyxl"
    )
    farmer_profile = {
        name: {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "non_empty_cells": int(frame.notna().sum().sum()),
        }
        for name, frame in farmer_book.items()
        if name != "RISK SCORING MATRIX"
    }
    return {
        "coverage": coverage,
        "associations": associations,
        "day14_target_groups": group_summary.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "model_summary": model_summary,
        "recovery_weight_feature_ablation": ablation,
        "evidence_rows": evidence.replace({np.nan: None})
        .assign(
            day14_date=lambda x: x["day14_date"].astype("string"),
            day35_date=lambda x: x["day35_date"].astype("string"),
            end_date=lambda x: x["end_date"].astype("string"),
        )
        .to_dict(orient="records"),
        "farmer_workbook_profile": farmer_profile,
    }


if __name__ == "__main__":
    results = build_results()
    OUTPUT.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: results[k] for k in ["coverage", "associations", "day14_target_groups", "model_summary"]}, indent=2, default=str))
