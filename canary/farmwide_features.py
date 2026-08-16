"""Shared, leakage-safe as-of feature service for Canary's farm-wide models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import CanaryDataset
from .modeling import extract_feature_row


CHECKPOINT_DAYS = (7, 14, 21, 28)
PRIMARY_IDENTITY_FEATURES = frozenset(
    {
        "building_id",
        "is_lags_building",
        "is_tags_building",
        "site_group",
        "building_history_mean",
    }
)


def checkpoint_status(cycle_day: int | float | None, evidence_available: bool = True) -> str:
    """Return the owner-facing validation status for a live forecast date."""

    if not evidence_available or cycle_day is None or pd.isna(cycle_day):
        return "Unavailable"
    day = int(cycle_day)
    if day in CHECKPOINT_DAYS:
        return "Validated checkpoint"
    if day < CHECKPOINT_DAYS[0]:
        return "Early fallback before Day 7"
    if day < CHECKPOINT_DAYS[-1]:
        return "Between-checkpoint estimate"
    return "Late off-checkpoint estimate"


def _trend(values: pd.Series, days: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    days = pd.to_numeric(days, errors="coerce")
    valid = values.notna() & days.notna()
    if int(valid.sum()) < 2:
        return np.nan
    return float(np.polyfit(days[valid], values[valid], 1)[0])


def build_asof_features(
    dataset: CanaryDataset,
    cycle_id: str,
    building_id: str,
    as_of_date: object,
    outcome: str,
) -> dict[str, Any] | None:
    """Build one farm-wide feature row using only evidence known by ``as_of``.

    This service is the shared application/research entry point. It retains the
    established application fields and adds timestamp, freshness, trajectory,
    reconciliation, and environmental-exposure evidence. Building/site identity
    is returned only as metadata and is never part of a primary feature schema.
    """

    if outcome not in {"recovery", "weight"}:
        raise ValueError("outcome must be 'recovery' or 'weight'")
    as_of = pd.Timestamp(as_of_date).normalize()
    base = extract_feature_row(dataset, cycle_id, building_id, as_of)
    if base is None:
        return None
    history = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["record_date"].le(as_of)
    ].sort_values(["record_date", "age_day"])
    if history.empty:
        return None

    latest = history.iloc[-1]
    age = int(latest["age_day"])
    beginning = float(latest["beginning_inventory"])
    mortality = pd.to_numeric(history["mortality_daily"], errors="coerce")
    mortality_rate = mortality / beginning * 1000 if beginning > 0 else mortality * np.nan
    population = pd.to_numeric(history["population"], errors="coerce")
    reconciliation_gap = -population.diff() - mortality
    weights = history.loc[history["weight_measured"].fillna(False)].copy()
    environment = history.loc[
        history[["temperature_avg_c", "humidity_avg_pct"]].notna().any(axis=1)
    ]
    operational = history["operational_recorded"].fillna(False)

    base.update(
        {
            "outcome": outcome,
            "feature_as_of_date": as_of,
            "max_source_date_used": pd.Timestamp(history["record_date"].max()),
            "max_source_day_used": int(history["age_day"].max()),
            "checkpoint_status": checkpoint_status(age),
            "latest_population_date": pd.Timestamp(
                history.loc[history["population"].notna(), "record_date"].max()
            )
            if history["population"].notna().any()
            else pd.NaT,
            "latest_weight_date": pd.Timestamp(weights["record_date"].max())
            if not weights.empty
            else pd.NaT,
            "latest_environment_date": pd.Timestamp(environment["record_date"].max())
            if not environment.empty
            else pd.NaT,
            "record_completeness_ratio": float(operational.mean()),
            "log_beginning_inventory": float(np.log(beginning)) if beginning > 0 else np.nan,
            "mortality_recent_1d_per_1000": float(mortality_rate.iloc[-1])
            if len(mortality_rate) and pd.notna(mortality_rate.iloc[-1])
            else np.nan,
            "mortality_recent_7d_per_1000": float(mortality_rate.tail(7).sum(min_count=1)),
            "mortality_ewma_per_1000": float(
                mortality_rate.ewm(span=min(7, max(2, len(mortality_rate))), adjust=False)
                .mean()
                .iloc[-1]
            )
            if mortality_rate.notna().any()
            else np.nan,
            "mortality_volatility_per_1000": float(mortality_rate.std(ddof=0)),
            "mortality_max_per_1000": float(mortality_rate.max()),
            "mortality_trend_slope_per_1000_day": _trend(
                mortality_rate, history["age_day"]
            ),
            "population_mortality_reconciliation_gap_per_1000": float(
                reconciliation_gap.abs().sum(min_count=1) / beginning * 1000
            )
            if beginning > 0
            else np.nan,
            "population_increase_days": int((population.diff() > 0).sum()),
            "weight_measurement_count": int(len(weights)),
            "weight_measurement_interval_days": float(
                np.diff(weights["age_day"].to_numpy(float)).mean()
            )
            if len(weights) > 1
            else np.nan,
            "observed_weight_trend_kg_day": _trend(
                weights["bodyweight_kg"], weights["age_day"]
            )
            if not weights.empty
            else np.nan,
            "temperature_history_mean_c": float(
                pd.to_numeric(history["temperature_avg_c"], errors="coerce").mean()
            ),
            "temperature_history_sd_c": float(
                pd.to_numeric(history["temperature_avg_c"], errors="coerce").std(ddof=0)
            ),
            "humidity_history_mean_pct": float(
                pd.to_numeric(history["humidity_avg_pct"], errors="coerce").mean()
            ),
            "humidity_history_sd_pct": float(
                pd.to_numeric(history["humidity_avg_pct"], errors="coerce").std(ddof=0)
            ),
            "environment_coverage_ratio": float(len(environment) / max(age, 1)),
            "environment_zone_days": int(history["zone_aggregated"].fillna(False).sum()),
            "temperature_zone_spread_mean_c": float(
                pd.to_numeric(history["temperature_zone_spread_c"], errors="coerce").mean()
            ),
            "humidity_zone_spread_mean_pct": float(
                pd.to_numeric(history["humidity_zone_spread_pct"], errors="coerce").mean()
            ),
            "building_id_metadata": str(building_id),
            "site_group_metadata": "Lags" if str(building_id).startswith("Lags") else "Tags",
        }
    )
    if base["max_source_date_used"] > as_of or base["max_source_day_used"] > age:
        raise AssertionError("An as-of feature row contains future evidence.")
    return base


def assert_primary_schema_has_no_identity(feature_columns: list[str] | tuple[str, ...]) -> None:
    forbidden = PRIMARY_IDENTITY_FEATURES.intersection(feature_columns)
    building_dummies = [name for name in feature_columns if str(name).startswith("building_")]
    if forbidden or building_dummies:
        raise AssertionError(
            "Primary farm-wide feature schema contains identity features: "
            + ", ".join(sorted({*forbidden, *building_dummies}))
        )
