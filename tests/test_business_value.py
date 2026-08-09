import pandas as pd
import pytest

from canary import ValueAssumptions, attach_business_value, estimate_recovery_value


def test_value_at_risk_and_improvement_scenario_are_auditable():
    assumptions = ValueAssumptions(
        price_php_per_kg=120,
        sale_weight_kg=2.0,
        recovery_improvement_pp=1.0,
        cycles_per_year=5,
    )
    result = estimate_recovery_value(10_000, 0.92, assumptions)

    assert result["recovery_gap_pp"] == pytest.approx(3.0)
    assert result["birds_per_point"] == pytest.approx(100)
    assert result["birds_at_risk"] == pytest.approx(300)
    assert result["gross_revenue_per_bird_php"] == pytest.approx(240)
    assert result["gross_revenue_at_risk_php"] == pytest.approx(72_000)
    assert result["scenario_gross_revenue_php"] == pytest.approx(24_000)
    assert result["scenario_annual_gross_revenue_php"] == pytest.approx(120_000)


def test_scenario_is_capped_at_gap_to_goal():
    assumptions = ValueAssumptions(recovery_improvement_pp=5.0)
    result = estimate_recovery_value(10_000, 0.94, assumptions)

    assert result["scenario_improvement_pp"] == pytest.approx(1.0)
    assert result["scenario_recovered_birds"] == pytest.approx(100)


def test_value_is_zero_when_prediction_is_already_at_goal():
    result = estimate_recovery_value(10_000, 0.96, ValueAssumptions())
    assert result["recovery_gap_pp"] == 0
    assert result["gross_revenue_at_risk_php"] == 0
    assert result["scenario_gross_revenue_php"] == 0


def test_missing_inputs_remain_missing():
    frame = pd.DataFrame(
        [{"building_id": "Tags 1", "beginning_inventory": 7000, "predicted_final_recovery": pd.NA}]
    )
    result = attach_business_value(frame, ValueAssumptions()).iloc[0]
    assert pd.isna(result["gross_revenue_at_risk_php"])
