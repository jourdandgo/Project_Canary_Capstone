"""Final-sprint audit for proposed risk inputs, Day 14 evidence, and target curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from canary.data import APPROVED_WEIGHT_TARGETS_G, load_workbook


APP_ROOT = Path(__file__).resolve().parents[1]
FARM_DATA = APP_ROOT / "data" / "FARM HARVEST DATA.xlsx"
OUTPUT = Path(__file__).with_name("capstone_final_sprint_audit.json")


def gompertz(age: np.ndarray, asymptote: float, displacement: float, rate: float) -> np.ndarray:
    return asymptote * np.exp(-displacement * np.exp(-rate * age))


def _quantiles(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"n": 0}
    quantiles = numeric.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    return {
        "n": int(len(numeric)),
        "minimum": float(quantiles.loc[0]),
        "p10": float(quantiles.loc[0.1]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "maximum": float(quantiles.loc[1]),
    }


def build_audit() -> dict[str, object]:
    dataset = load_workbook(FARM_DATA)
    daily = dataset.daily.copy()
    operational = daily.loc[daily["operational_recorded"]].copy()
    operational["temperature_range_c"] = (
        pd.to_numeric(operational["temperature_max_c"], errors="coerce")
        - pd.to_numeric(operational["temperature_min_c"], errors="coerce")
    )
    operational["humidity_range_pct"] = (
        pd.to_numeric(operational["humidity_max_pct"], errors="coerce")
        - pd.to_numeric(operational["humidity_min_pct"], errors="coerce")
    )

    coverage_by_cycle = []
    for cycle, frame in operational.groupby("cycle_id", sort=True):
        coverage_by_cycle.append(
            {
                "cycle_id": str(cycle),
                "rows": int(len(frame)),
                "buildings": int(frame["building_id"].nunique()),
                "temperature_avg_pct": float(frame["temperature_avg_c"].notna().mean() * 100),
                "temperature_range_pct": float(frame["temperature_range_c"].notna().mean() * 100),
                "humidity_avg_pct": float(frame["humidity_avg_pct"].notna().mean() * 100),
                "feed_per_bird_pct": float(frame["feed_daily_kg_per_bird"].notna().mean() * 100),
            }
        )

    range_values = operational["temperature_range_c"].dropna()
    range_bands = {
        "at_or_below_2c": int((range_values <= 2).sum()),
        "above_2_to_3c": int(((range_values > 2) & (range_values <= 3)).sum()),
        "above_3_to_5c": int(((range_values > 3) & (range_values <= 5)).sum()),
        "above_5c": int((range_values > 5).sum()),
    }

    ages = np.array(sorted(APPROVED_WEIGHT_TARGETS_G), dtype=float)
    weights = np.array([APPROVED_WEIGHT_TARGETS_G[int(age)] for age in ages], dtype=float)
    parameters, _ = curve_fit(
        gompertz,
        ages,
        weights,
        p0=(3500.0, 5.0, 0.08),
        bounds=([1800.0, 0.01, 0.001], [10000.0, 20.0, 1.0]),
        maxfev=100000,
    )
    fitted = gompertz(ages, *parameters)
    gompertz_checkpoints = [
        {
            "age_day": int(age),
            "approved_g": float(actual),
            "gompertz_g": float(predicted),
            "error_g": float(predicted - actual),
        }
        for age, actual, predicted in zip(ages, weights, fitted)
    ]

    return {
        "source": str(FARM_DATA),
        "quality": {
            "source_rows": dataset.quality.source_rows,
            "canonical_rows": dataset.quality.canonical_rows,
            "temperature_coverage_pct": dataset.quality.temperature_coverage_pct,
            "humidity_coverage_pct": dataset.quality.humidity_coverage_pct,
        },
        "coverage_by_cycle": coverage_by_cycle,
        "candidate_risk_inputs": {
            "temperature_range_c": _quantiles(operational["temperature_range_c"]),
            "temperature_range_threshold_counts": range_bands,
            "temperature_avg_c": _quantiles(operational["temperature_avg_c"]),
            "humidity_avg_pct": _quantiles(operational["humidity_avg_pct"]),
            "feed_daily_g_per_bird": _quantiles(operational["feed_daily_kg_per_bird"] * 1000),
        },
        "gompertz_candidate": {
            "formula": "A * exp(-B * exp(-k * age))",
            "parameters": {
                "asymptote_g": float(parameters[0]),
                "displacement": float(parameters[1]),
                "rate_per_day": float(parameters[2]),
            },
            "checkpoint_rmse_g": float(np.sqrt(np.mean((fitted - weights) ** 2))),
            "checkpoint_mae_g": float(np.mean(np.abs(fitted - weights))),
            "maximum_absolute_checkpoint_error_g": float(np.max(np.abs(fitted - weights))),
            "checkpoints": gompertz_checkpoints,
        },
    }


if __name__ == "__main__":
    result = build_audit()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
