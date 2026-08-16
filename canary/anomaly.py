"""Age-adjusted anomaly signals kept separate from Canary's 0–12 risk score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import CanaryDataset


@dataclass(frozen=True)
class AnomalySignal:
    metric: str
    status: str
    latest_value: float | None
    expected_value: float | None
    ewma_z: float | None
    cusum: float | None
    latest_evidence_date: str | None
    explanation: str
    changes_risk_score: bool = False


def _safe(value: Any) -> float | None:
    return None if pd.isna(value) or not np.isfinite(float(value)) else float(value)


def _status(ewma_z: float | None, cusum: float | None) -> str:
    if ewma_z is None or cusum is None:
        return "Unavailable"
    if ewma_z >= 3.0 or cusum >= 6.0:
        return "Warning"
    if ewma_z >= 2.0 or cusum >= 3.0:
        return "Watch"
    return "No signal"


def _signal(metric: str, values: pd.Series, expected: pd.Series, dates: pd.Series, *, adverse: str = "high") -> AnomalySignal:
    observed = pd.to_numeric(values, errors="coerce")
    reference = pd.to_numeric(expected, errors="coerce")
    valid = observed.notna() & reference.notna()
    if not valid.any():
        return AnomalySignal(metric, "Unavailable", None, None, None, None, None, "Required evidence is missing.")
    residual = observed.loc[valid] - reference.loc[valid]
    if adverse == "low":
        residual = -residual
    scale = max(float(residual.abs().median()) * 1.4826, 1e-6)
    z = residual / scale
    ewma = z.ewm(alpha=0.35, adjust=False).mean()
    cusum_series = (z - 0.5).clip(lower=0).cumsum()
    latest_index = observed.loc[valid].index[-1]
    ewma_value = _safe(ewma.iloc[-1])
    cusum_value = _safe(cusum_series.iloc[-1])
    status = _status(ewma_value, cusum_value)
    explanation = (
        f"{metric} is {status.lower()} relative to the age-adjusted historical reference. "
        "This is an investigation signal, not a probability or causal diagnosis."
    )
    return AnomalySignal(
        metric=metric, status=status, latest_value=_safe(observed.loc[latest_index]),
        expected_value=_safe(reference.loc[latest_index]), ewma_z=ewma_value, cusum=cusum_value,
        latest_evidence_date=pd.Timestamp(dates.loc[latest_index]).date().isoformat(), explanation=explanation,
    )


def build_age_adjusted_anomalies(
    dataset: CanaryDataset, cycle_id: str, building_id: str, as_of_date: pd.Timestamp | str
) -> list[dict[str, Any]]:
    """Return as-of EWMA/CUSUM signals using only cycles that started earlier."""
    as_of = pd.Timestamp(as_of_date).normalize()
    focal = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq(str(cycle_id))
        & dataset.daily["building_id"].astype(str).eq(str(building_id))
        & dataset.daily["record_date"].le(as_of)
    ].sort_values("age_day").copy()
    if focal.empty:
        return []
    focal_start = pd.Timestamp(focal["record_date"].min())
    eligible_cycles = dataset.cycles.loc[pd.to_datetime(dataset.cycles["start_date"]) < focal_start, "cycle_id"].astype(str).unique()
    history = dataset.daily.loc[dataset.daily["cycle_id"].astype(str).isin(eligible_cycles)].copy()
    history["mortality_per_1000"] = pd.to_numeric(history["mortality_daily"], errors="coerce") / history["beginning_inventory"].clip(lower=1) * 1000
    mortality_reference = history.groupby("age_day")["mortality_per_1000"].median()
    focal["mortality_per_1000"] = pd.to_numeric(focal["mortality_daily"], errors="coerce") / focal["beginning_inventory"].clip(lower=1) * 1000
    focal["mortality_expected"] = focal["age_day"].map(mortality_reference)

    ages = pd.to_numeric(focal["age_day"], errors="coerce")
    focal["temperature_expected"] = np.select([ages <= 7, ages <= 14, ages <= 21, ages <= 28], [31.0, 28.5, 25.5, 23.5], default=22.5)
    focal["humidity_expected"] = np.select([ages <= 7, ages <= 14], [60.0, 55.0], default=50.0)
    signals = [
        _signal("Daily mortality per 1,000", focal["mortality_per_1000"], focal["mortality_expected"], focal["record_date"]),
        _signal("Temperature deviation", (pd.to_numeric(focal["temperature_avg_c"], errors="coerce") - focal["temperature_expected"]).abs(), pd.Series(0.0, index=focal.index), focal["record_date"]),
        _signal("Humidity deviation", (pd.to_numeric(focal["humidity_avg_pct"], errors="coerce") - focal["humidity_expected"]).abs(), pd.Series(0.0, index=focal.index), focal["record_date"]),
    ]
    return [asdict(signal) for signal in signals]

