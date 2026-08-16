"""Versioned prediction contracts shared by training, inference, and UI code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    outcome: str
    cycle_id: str
    building_id: str
    as_of_date: pd.Timestamp
    estimate: float | None
    interval_80: tuple[float | None, float | None]
    interval_90: tuple[float | None, float | None]
    target: float
    target_status: str
    latest_observation_dates: dict[str, str | None] = field(default_factory=dict)
    evidence_status: str = "Available"
    checkpoint_status: str = "Unavailable"
    model_version: str = "unknown"
    deployment_status: str = "operational"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
