"""Transparent gross-revenue scenarios for Project Canary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RECOVERY_GOAL = 0.95
DEFAULT_PRICE_PHP_PER_KG = 120.0
DEFAULT_SALE_WEIGHT_KG = 2.0
DEFAULT_RECOVERY_IMPROVEMENT_PP = 1.0
DEFAULT_CYCLES_PER_YEAR = 5


@dataclass(frozen=True)
class ValueAssumptions:
    """Editable planning assumptions; none are claimed as current market facts."""

    price_php_per_kg: float = DEFAULT_PRICE_PHP_PER_KG
    sale_weight_kg: float = DEFAULT_SALE_WEIGHT_KG
    recovery_improvement_pp: float = DEFAULT_RECOVERY_IMPROVEMENT_PP
    cycles_per_year: int = DEFAULT_CYCLES_PER_YEAR


def estimate_recovery_value(
    beginning_population: object,
    predicted_recovery: object,
    assumptions: ValueAssumptions,
) -> dict[str, float]:
    """Estimate gross revenue tied to the predicted recovery gap.

    The selected improvement is capped at the gap to the 95% goal. This keeps
    the scenario focused on redeeming modeled value at risk rather than
    claiming value above the capstone target.
    """

    if pd.isna(beginning_population) or pd.isna(predicted_recovery):
        return {
            "recovery_gap_pp": np.nan,
            "birds_per_point": np.nan,
            "birds_at_risk": np.nan,
            "gross_revenue_per_bird_php": np.nan,
            "gross_revenue_at_risk_php": np.nan,
            "scenario_improvement_pp": np.nan,
            "scenario_recovered_birds": np.nan,
            "scenario_gross_revenue_php": np.nan,
            "scenario_annual_gross_revenue_php": np.nan,
        }

    population = max(0.0, float(beginning_population))
    predicted = float(np.clip(float(predicted_recovery), 0.0, 1.0))
    gap_pp = max(0.0, (RECOVERY_GOAL - predicted) * 100)
    applied_improvement_pp = min(
        max(0.0, float(assumptions.recovery_improvement_pp)), gap_pp
    )
    birds_per_point = population * 0.01
    birds_at_risk = population * gap_pp / 100
    scenario_birds = population * applied_improvement_pp / 100
    revenue_per_bird = max(0.0, float(assumptions.sale_weight_kg)) * max(
        0.0, float(assumptions.price_php_per_kg)
    )
    scenario_value = scenario_birds * revenue_per_bird
    return {
        "recovery_gap_pp": gap_pp,
        "birds_per_point": birds_per_point,
        "birds_at_risk": birds_at_risk,
        "gross_revenue_per_bird_php": revenue_per_bird,
        "gross_revenue_at_risk_php": birds_at_risk * revenue_per_bird,
        "scenario_improvement_pp": applied_improvement_pp,
        "scenario_recovered_birds": scenario_birds,
        "scenario_gross_revenue_php": scenario_value,
        "scenario_annual_gross_revenue_php": scenario_value
        * max(0, int(assumptions.cycles_per_year)),
    }


def attach_business_value(
    snapshot: pd.DataFrame,
    assumptions: ValueAssumptions,
) -> pd.DataFrame:
    """Attach one auditable value scenario to every building row."""

    rows: list[dict[str, object]] = []
    for _, source in snapshot.iterrows():
        row = source.to_dict()
        row.update(
            estimate_recovery_value(
                source.get("beginning_inventory"),
                source.get("predicted_final_recovery"),
                assumptions,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)
