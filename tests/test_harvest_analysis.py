from pathlib import Path
import os

import pandas as pd
import pytest

from canary import (
    attach_forecasts,
    build_harvest_analysis_rows,
    load_day35_manifest,
    load_final_weight_labels,
    load_model_bundle,
    load_risk_rules,
    load_workbook,
    score_cycle_snapshot,
    summarize_harvest_analysis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.getenv("CANARY_TEST_WORKBOOK", ROOT / "data" / "FARM HARVEST DATA.xlsx"))
PERFORMANCE = ROOT / "data" / "Farm Performance Summary.xlsx"


@pytest.fixture(scope="module")
def analysis_rows():
    dataset = load_workbook(SOURCE)
    recovery, _ = load_model_bundle("recovery")
    weight = load_day35_manifest()
    as_of = pd.Timestamp("2026-07-25")
    current = attach_forecasts(
        dataset,
        score_cycle_snapshot(dataset, "2026-3", as_of, load_risk_rules()),
    )
    labels = load_final_weight_labels(PERFORMANCE)
    return build_harvest_analysis_rows(dataset, labels, current, recovery, weight)


def test_all_cycle_grid_and_target_specific_outcome_counts(analysis_rows):
    recorded = analysis_rows.loc[analysis_rows["start_date"].notna()]
    assert len(analysis_rows) == 7 * 6
    assert len(recorded) == 34
    assert recorded["cycle_id"].nunique() == 7
    assert analysis_rows["recovery_training_eligible"].sum() == 25
    assert analysis_rows["weight_training_eligible"].sum() == 31

    by_cycle = analysis_rows.groupby("cycle_id")[[
        "recovery_training_eligible",
        "weight_training_eligible",
    ]].sum()
    assert by_cycle.loc["2026-2"].tolist() == [0, 6]
    assert by_cycle.loc["2026-3"].tolist() == [0, 0]


def test_historical_actuals_and_current_projections_do_not_mix(analysis_rows):
    historical = analysis_rows.loc[
        analysis_rows["reporting_status"].eq("Historical records ended")
    ]
    current = analysis_rows.loc[analysis_rows["reporting_status"].eq("Current flock")]

    assert historical["projected_recovery"].isna().all()
    assert historical["projected_day35_weight_kg"].isna().all()
    assert current["historical_recovery_proxy"].isna().all()
    assert current["historical_day35_weight_kg"].isna().all()
    assert current["projected_recovery"].notna().all()


def test_2026_2_recovery_proxy_is_flagged_and_excluded(analysis_rows):
    cycle = analysis_rows.loc[
        analysis_rows["cycle_id"].eq("2026-2")
        & analysis_rows["start_date"].notna()
    ]
    assert len(cycle) == 6
    assert cycle["historical_recovery_proxy"].notna().all()
    assert not cycle["recovery_training_eligible"].any()
    assert cycle["weight_training_eligible"].all()
    assert cycle["data_quality_note"].str.contains("excluded from recovery training").all()


def test_historical_recovery_kpi_reconciles_to_population_totals(analysis_rows):
    summary = summarize_harvest_analysis(analysis_rows)
    historical = analysis_rows.loc[
        analysis_rows["reporting_status"].eq("Historical records ended")
        & analysis_rows["beginning_inventory"].notna()
        & analysis_rows["recorded_ending_population"].notna()
    ]
    expected = historical["recorded_ending_population"].sum() / historical[
        "beginning_inventory"
    ].sum()
    assert summary["historical_recovery"] == pytest.approx(expected)
    assert summary["building_records"] == 34


def test_current_review_date_controls_only_current_cycle_projection():
    dataset = load_workbook(SOURCE)
    recovery, _ = load_model_bundle("recovery")
    weight = load_day35_manifest()
    labels = load_final_weight_labels(PERFORMANCE)

    def rows_at(as_of: str):
        current = attach_forecasts(
            dataset,
            score_cycle_snapshot(
                dataset, "2026-3", pd.Timestamp(as_of), load_risk_rules()
            ),
        )
        return build_harvest_analysis_rows(
            dataset, labels, current, recovery, weight
        )

    early = rows_at("2026-07-05")
    later = rows_at("2026-07-25")
    early_historical = early.loc[
        early["reporting_status"].eq("Historical records ended"),
        ["cycle_id", "building_id", "historical_recovery_proxy"],
    ].reset_index(drop=True)
    later_historical = later.loc[
        later["reporting_status"].eq("Historical records ended"),
        ["cycle_id", "building_id", "historical_recovery_proxy"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_historical, later_historical)

    early_current = early.loc[
        early["cycle_id"].eq("2026-3"), "projected_recovery"
    ]
    later_current = later.loc[
        later["cycle_id"].eq("2026-3"), "projected_recovery"
    ]
    assert not early_current.equals(later_current)
