"""Project Canary Streamlit application — Sprint 5 capstone prototype."""

from __future__ import annotations

import html
import json
import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from canary import (
    ALLOWED_APPROVAL_STATUSES,
    DEFAULT_RECOMMENDATIONS_PATH,
    DEFAULT_RULES_PATH,
    RecommendationConfigurationError,
    RiskConfigurationError,
    WorkbookValidationError,
    apply_recommendations,
    attach_forecasts,
    attach_historical_day14_backtests,
    build_operational_driver_trace,
    build_cycle_snapshot,
    build_dimension_trace,
    build_forecast_history,
    build_recommendation_trace,
    build_risk_history,
    cycle_date_bounds,
    default_as_of_date,
    forecast_input_trace,
    forecast_trace,
    load_day35_manifest,
    load_model_bundle,
    load_final_weight_labels,
    load_recommendation_playbook,
    load_risk_rules,
    load_workbook,
    score_cycle_snapshot,
    save_recommendation_playbook,
    save_risk_rules,
    recovery_feature_contributions,
)
from canary.outcomes import build_historical_outcomes, latest_cycle_id
from canary.operational_alerts import evaluate_operational_alerts
from canary.business_value import (
    DEFAULT_CYCLES_PER_YEAR,
    DEFAULT_PRICE_PHP_PER_KG,
    DEFAULT_RECOVERY_IMPROVEMENT_PP,
    DEFAULT_SALE_WEIGHT_KG,
    ValueAssumptions,
    attach_business_value,
)


st.set_page_config(page_title="Project Canary", page_icon="🐤", layout="wide")
st.markdown(
    """
    <style>
      :root { --green:#173f31; --green-2:#286245; --canary:#c7f24b; --muted:#607069; --line:#dae5df; --soft:#f1f6f3; }
      .stApp { background:#f7faf8; }
      [data-testid="stSidebar"] { background:#edf5f0; border-right:1px solid #d7e4dc; }
      .block-container { padding-top:1.65rem; padding-bottom:3rem; max-width:1500px; }
      .hero { padding:1.05rem 1.35rem; border-radius:18px; background:linear-gradient(130deg,#133c2c,#286245); color:white; margin-bottom:.75rem; box-shadow:0 12px 26px rgba(19,60,44,.11); }
      .hero small { color:var(--canary); font-weight:850; letter-spacing:.08em; }
      .hero h1 { margin:.12rem 0 .2rem; font-size:1.78rem; }
      .hero p { color:#deebe4; margin:0; max-width:900px; }
      .context { display:flex; flex-wrap:wrap; gap:.45rem .8rem; align-items:center; background:white; border:1px solid var(--line); border-radius:13px; padding:.65rem .85rem; margin:0 0 .85rem; color:#465b51; font-size:.82rem; }
      .context strong { color:var(--green); }
      .context .dot { color:#9aac9f; }
      .start { background:#f1f8d9; border:1px solid #d7e99c; border-left:6px solid #8ab625; padding:.9rem 1rem; border-radius:12px; color:#29431f; min-height:72px; }
      .card-body { min-height:0; display:flex; flex-direction:column; gap:.55rem; }
      .head { display:flex; justify-content:space-between; gap:.5rem; align-items:center; }
      .name { color:var(--green); font-size:1.08rem; font-weight:850; }
      .pill { border-radius:999px; padding:.2rem .58rem; font-size:.72rem; font-weight:850; }
      .low { background:#e5f6df; color:#256a39; } .medium { background:#fff1bf; color:#785800; }
      .high { background:#ffe0c2; color:#934600; } .critical { background:#f7d3d7; color:#961e2a; }
      .not-rated { background:#e9efec; color:#64726c; }
      .scheduled { background:#e7f0fb; color:#355d86; }
      .completed { background:#e8efeb; color:#355347; }
      .meta { color:var(--muted); font-size:.76rem; margin:.05rem 0; }
      .issue { color:#294f3c; font-weight:800; margin:.05rem 0; }
      .line { padding:.42rem 0; border-top:1px solid #edf2ef; }
      .label { color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; }
      .value { color:var(--green); font-size:1rem; font-weight:800; }
      .sub { color:#5e6d66; font-size:.75rem; }
      .action { background:#f5f8f6; border-radius:9px; padding:.52rem .62rem; color:#3f5149; font-size:.75rem; }
      .signal-grid { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; margin:.15rem 0; }
      .signal { background:#f3f7f4; border:1px solid #e3ece6; border-radius:10px; padding:.58rem .62rem; min-height:76px; }
      .signal .value { display:block; font-size:1.05rem; margin:.1rem 0; }
      .card-summary { display:flex; align-items:center; justify-content:space-between; gap:.4rem; }
      .micro-badge { display:inline-flex; align-items:center; border-radius:999px; background:#edf3ef; color:#52665c; padding:.16rem .46rem; font-size:.66rem; font-weight:750; }
      .outcome-stack { border:1px solid #e1ebe5; border-radius:12px; overflow:hidden; background:#fbfdfc; }
      .outcome-row { display:grid; grid-template-columns:5.1rem 1fr; align-items:start; gap:.45rem; padding:.62rem .7rem; }
      .outcome-row + .outcome-row { border-top:1px solid #e6eee9; }
      .outcome-name { color:#40584d; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.035em; }
      .outcome-detail { min-width:0; }
      .outcome-flow { display:flex; flex-wrap:wrap; align-items:baseline; gap:.18rem .3rem; color:#65766e; font-size:.72rem; line-height:1.35; }
      .outcome-flow strong { color:var(--green); font-size:.9rem; }
      .outcome-arrow { color:#90a198; font-weight:850; }
      .gap-tag { display:inline-block; margin-top:.12rem; font-size:.68rem; font-weight:750; }
      .value-strip { padding:.5rem .62rem; border-radius:10px; background:#eef6e5; border:1px solid #d9e8c7; display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
      .value-strip strong { color:#214b34; font-size:.98rem; white-space:nowrap; }
      .driver { border-left:4px solid #94bc2b; padding:.42rem .55rem; background:#f7faeb; border-radius:7px; color:#3f5149; font-size:.76rem; }
      .backtest { border-top:1px solid #e4ece7; padding-top:.52rem; color:#52645b; font-size:.72rem; line-height:1.45; }
      .backtest strong { color:var(--green); }
      .state-note { background:#f5f8f6; border:1px dashed #cbd9d1; border-radius:10px; padding:.68rem; color:#53665d; font-size:.78rem; margin:.5rem 0; }
      .title { color:var(--green); font-size:1.25rem; font-weight:850; margin:.55rem 0 .15rem; }
      .subtitle { color:var(--muted); font-size:.88rem; margin-bottom:.75rem; }
      .good { color:#20713a; } .bad { color:#a13b28; } .neutral { color:#607069; }
      div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:.7rem .82rem; border-radius:14px; }
      .notice { background:#fff8dc; border:1px solid #ead994; padding:.75rem .9rem; border-radius:11px; }
      .score-tile { background:white; border:1px solid var(--line); border-radius:13px; padding:.72rem .78rem; min-height:112px; }
      .score-tile .score { color:var(--green); font-size:1.18rem; font-weight:850; margin:.12rem 0 .25rem; }
      .score-tile .evidence { color:var(--muted); font-size:.72rem; line-height:1.35; }
      .forecast-note { background:#f4f8f5; border-left:4px solid #80a98f; border-radius:8px; padding:.62rem .72rem; color:#40544a; font-size:.79rem; min-height:62px; }
      .model-badge { display:inline-block; background:#eaf1ed; color:#3e5b4c; border-radius:999px; padding:.18rem .5rem; font-size:.68rem; font-weight:800; margin-bottom:.45rem; }
      .empty-state { background:#f4f7f5; border:1px dashed #cbd9d1; border-radius:11px; padding:.9rem; color:#53665d; font-size:.82rem; margin-top:.8rem; }
      .evidence-note { color:#8a6500; font-size:.74rem; font-weight:700; margin:.2rem 0 .55rem; }
      .intro-grid { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin:.2rem 0 .75rem; }
      .intro-panel { position:relative; overflow:hidden; min-height:142px; border:1px solid #315e4b; border-radius:16px; padding:1rem 1.08rem; background:linear-gradient(145deg,#173f31,#285e47); color:white; box-shadow:0 9px 22px rgba(19,60,44,.09); }
      .intro-panel.solution { background:linear-gradient(145deg,#244d37,#3f704e); }
      .intro-panel::after { content:""; position:absolute; width:110px; height:110px; right:-28px; bottom:-46px; border-radius:50%; background:rgba(199,242,75,.12); }
      .intro-kicker { display:block; color:var(--canary); font-size:.68rem; font-weight:850; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.28rem; }
      .intro-panel strong { display:block; color:white; font-size:1.1rem; margin-bottom:.35rem; max-width:85%; }
      .intro-panel span { display:block; color:#dfebe5; font-size:.8rem; line-height:1.48; max-width:88%; }
      .decision-question { display:grid; grid-template-columns:auto 1fr; gap:.9rem; align-items:center; background:#f3f8da; border:1px solid #d6e99c; border-left:7px solid #8bb627; border-radius:16px; padding:.92rem 1.05rem; margin:0 0 .85rem; box-shadow:0 7px 18px rgba(72,100,32,.06); }
      .decision-icon { display:flex; align-items:center; justify-content:center; width:46px; height:46px; border-radius:13px; background:#173f31; color:var(--canary); font-size:1.2rem; font-weight:900; }
      .decision-kicker { color:#60732a; font-size:.67rem; font-weight:850; letter-spacing:.08em; text-transform:uppercase; }
      .decision-question strong { display:block; color:#23412d; font-size:1.02rem; line-height:1.35; margin:.1rem 0 .25rem; }
      .decision-goals { display:flex; flex-wrap:wrap; gap:.35rem; }
      .goal-chip { display:inline-flex; padding:.16rem .48rem; border-radius:999px; background:white; border:1px solid #d4dfc4; color:#526348; font-size:.66rem; font-weight:750; }
      .executive-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.68rem; margin:.15rem 0 .8rem; }
      .executive-card { background:white; border:1px solid var(--line); border-radius:15px; padding:.78rem .85rem; min-height:112px; box-shadow:0 6px 16px rgba(17,59,43,.045); }
      .executive-card.priority { background:linear-gradient(145deg,#173f31,#295f47); border-color:#295f47; }
      .executive-card .eyebrow { color:#6b7a73; font-size:.66rem; font-weight:800; letter-spacing:.055em; text-transform:uppercase; }
      .executive-card .metric-value { color:var(--green); font-size:1.48rem; font-weight:900; line-height:1.12; margin:.24rem 0 .18rem; }
      .executive-card .metric-sub { color:#687870; font-size:.7rem; line-height:1.35; }
      .executive-card.priority .eyebrow { color:#c7f24b; }
      .executive-card.priority .metric-value { color:white; font-size:1.05rem; }
      .executive-card.priority .metric-sub { color:#deebe4; }
      .priority-cell { min-height:82px; display:flex; flex-direction:column; justify-content:center; }
      .priority-kicker { color:#75857d; font-size:.66rem; font-weight:850; letter-spacing:.06em; text-transform:uppercase; }
      .priority-name { color:var(--green); font-size:1.18rem; font-weight:900; margin:.18rem 0; }
      .priority-copy { color:#4d6057; font-size:.78rem; line-height:1.42; }
      .pattern-title { display:block; color:#294f3c; font-weight:850; }
      .pattern-subtitle { display:block; color:#607069; font-size:.71rem; line-height:1.38; margin-top:.13rem; }
      [data-testid="stSidebarNav"] { padding-top:.25rem; }
      [data-testid="stSidebarNav"] a { border-radius:10px; margin:.08rem .35rem; }
      [data-testid="stSidebarNav"] a[aria-current="page"] { background:#dcece2; color:var(--green); font-weight:800; }
      [data-testid="stVerticalBlockBorderWrapper"] { background:white; border-color:var(--line); border-radius:17px; box-shadow:0 6px 18px rgba(17,59,43,.055); transition:transform .16s ease, box-shadow .16s ease; }
      [data-testid="stVerticalBlockBorderWrapper"]:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(17,59,43,.09); }
      div[data-testid="stButton"] > button { border-radius:10px; font-weight:750; }
      div[data-testid="stButton"] > button[kind="primary"] { background:var(--green); border-color:var(--green); }
      @media (max-width: 900px) {
        .card-body { min-height:0; }
        .hero h1 { font-size:1.55rem; }
        .signal-grid { grid-template-columns:1fr 1fr; }
        .intro-grid { grid-template-columns:1fr; }
        .executive-grid { grid-template-columns:1fr 1fr; }
        .decision-question { grid-template-columns:1fr; }
        [data-testid="stHorizontalBlock"]:has(.card-body) { flex-direction:column !important; gap:.75rem !important; }
        [data-testid="stHorizontalBlock"]:has(.card-body) > div { width:100% !important; flex:1 1 100% !important; }
      }
      @media (max-width: 620px) { .executive-grid { grid-template-columns:1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _default_workbook() -> Path:
    configured = os.getenv("CANARY_DEFAULT_WORKBOOK")
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parent.parent / "FARM HARVEST DATA.xlsx"


def _default_performance_workbook() -> Path:
    configured = os.getenv("CANARY_PERFORMANCE_WORKBOOK")
    if configured:
        return Path(configured).expanduser()
    return _default_workbook().with_name("Farm Performance Summary.xlsx")


@st.cache_data(show_spinner="Checking the farm workbook…")
def _load_path(path: str, modified_ns: int):
    del modified_ns
    return load_workbook(path)


@st.cache_data(show_spinner="Checking the uploaded workbook…")
def _load_upload(content: bytes, name: str):
    return load_workbook(content, source_name=name)


@st.cache_data(show_spinner="Checking historical final weights…")
def _load_performance_path(path: str, modified_ns: int):
    del modified_ns
    return load_final_weight_labels(path)


@st.cache_data(show_spinner="Checking the uploaded final-weight workbook…")
def _load_performance_upload(content: bytes, name: str):
    source = BytesIO(content)
    source.name = name
    return load_final_weight_labels(source)


def _percent(value: object) -> str:
    return "Not available" if pd.isna(value) else f"{float(value):.1%}"


def _weight(value: object) -> str:
    return "Not available" if pd.isna(value) else f"{float(value):.2f} kg"


def _grams(value: object) -> str:
    return "Not available" if pd.isna(value) else f"{float(value) * 1000:,.0f} g"


def _php(value: object) -> str:
    return "Not available" if pd.isna(value) else f"₱{float(value):,.0f}"


def _gap(value: object, unit: str) -> tuple[str, str]:
    if pd.isna(value):
        return "No comparison yet", "neutral"
    number = float(value)
    if number >= 0:
        return f"{abs(number):.1f}{unit} above goal", "good"
    return f"{abs(number):.1f}{unit} below goal", "bad"


def _signed(value: object, decimals: int = 1, suffix: str = "") -> str:
    return "Not available" if pd.isna(value) else f"{float(value):+.{decimals}f}{suffix}"


def _relative_change(current: object, future: object) -> object:
    if pd.isna(current) or pd.isna(future) or float(current) == 0:
        return pd.NA
    return (float(future) - float(current)) / float(current) * 100


def _current_vs_outlook_table(row: pd.Series) -> pd.DataFrame:
    recovery_change_pp = (
        (float(row["predicted_final_recovery"]) - float(row["percentage_alive"])) * 100
        if pd.notna(row["predicted_final_recovery"]) and pd.notna(row["percentage_alive"])
        else pd.NA
    )
    recovery_goal_pct = (
        float(row["recovery_target_gap_pp"]) / 95 * 100
        if pd.notna(row["recovery_target_gap_pp"])
        else pd.NA
    )
    weight_change = (
        float(row["projected_day35_weight_kg"]) - float(row["latest_weight_kg"])
        if pd.notna(row["projected_day35_weight_kg"]) and pd.notna(row["latest_weight_kg"])
        else pd.NA
    )
    weight_goal_pct = (
        float(row["day35_weight_target_gap_kg"]) / 2.0 * 100
        if pd.notna(row["day35_weight_target_gap_kg"])
        else pd.NA
    )
    weight_source = str(row.get("day35_weight_scope", "Not available"))
    weight_day = (
        f"Day {int(row['weight_measurement_day'])}"
        if pd.notna(row.get("weight_measurement_day"))
        else "No weighing"
    )
    return pd.DataFrame(
        [
            {
                "Outcome": "Harvest recovery",
                "Current recorded status": _percent(row["percentage_alive"]),
                "Predicted final outcome": _percent(row["predicted_final_recovery"]),
                "Expected change": _signed(recovery_change_pp, 1, " pts"),
                "Change (%)": _signed(_relative_change(row["percentage_alive"], row["predicted_final_recovery"]), 1, "%"),
                "Gap to goal": _signed(row["recovery_target_gap_pp"], 1, " pts"),
                "Gap to goal (%)": _signed(recovery_goal_pct, 1, "%"),
                "Basis": "Building-specific recovery model",
            },
            {
                "Outcome": "Day 35 average liveweight",
                "Current recorded status": f"{_weight(row['latest_weight_kg'])} ({weight_day})",
                "Predicted final outcome": _weight(row["projected_day35_weight_kg"]),
                "Expected change": _signed(weight_change, 2, " kg"),
                "Change (%)": _signed(_relative_change(row["latest_weight_kg"], row["projected_day35_weight_kg"]), 1, "%"),
                "Gap to goal": _signed(row["day35_weight_target_gap_kg"], 2, " kg"),
                "Gap to goal (%)": _signed(weight_goal_pct, 1, "%"),
                "Basis": weight_source,
            },
        ]
    )


def _candidate_metrics_table(manifest: dict[str, object], outcome: str) -> pd.DataFrame:
    """Create a defense-friendly comparison of every tested model."""

    rows = []
    factor = 100 if outcome == "recovery" else 1
    unit = "percentage points" if outcome == "recovery" else "kg"
    for candidate, metrics in manifest["metrics"].items():
        rows.append(
            {
                "Candidate": candidate.replace("_", " ").title(),
                "Selected": "Yes" if candidate == manifest["selected_model"] else "",
                f"MAE ({unit})": round(float(metrics["mae"]) * factor, 3),
                f"Cycle-balanced MAE ({unit})": round(
                    float(metrics.get("cycle_macro_mae", metrics["mae"])) * factor, 3
                ),
                f"RMSE ({unit})": round(float(metrics["rmse"]) * factor, 3),
                f"Bias ({unit})": round(
                    float(metrics.get("bias", metrics.get("mean_error", 0))) * factor,
                    3,
                ),
                "Cycle-to-cycle MAE variability": round(
                    float(metrics["fold_mae_std"]) * factor, 3
                ),
                "Below-goal recall": f"{float(metrics['below_target_recall']):.1%}",
                "At/above-goal recall": f"{float(metrics['at_or_above_target_recall']):.1%}",
            }
        )
    return pd.DataFrame(rows)


def _horizon_metrics_table(manifest: dict[str, object], outcome: str) -> pd.DataFrame:
    factor = 100 if outcome == "recovery" else 1
    unit = "percentage points" if outcome == "recovery" else "kg"
    return pd.DataFrame(
        [
            {
                "Forecast timing": horizon,
                "Daily snapshots": values["rows"],
                f"MAE ({unit})": round(float(values["mae"]) * factor, 3),
                f"RMSE ({unit})": round(float(values["rmse"]) * factor, 3),
            }
            for horizon, values in manifest["selected_metrics"]["horizon"].items()
        ]
    )


def _day35_candidate_metrics_table(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Method": candidate.replace("_", " ").title(),
                "Selected": "Yes" if candidate == manifest["selected_model"] else "",
                "MAE": f"{float(metrics['mae_kg']) * 1000:.0f} g",
                "Cycle-balanced MAE": f"{float(metrics.get('cycle_macro_mae_kg', metrics['mae_kg'])) * 1000:.0f} g",
                "RMSE": f"{float(metrics['rmse_kg']) * 1000:.0f} g",
                "Bias": f"{float(metrics['bias_kg']) * 1000:+.0f} g",
                "Within 200 g": f"{float(metrics['within_200g_rate']):.1%}",
            }
            for candidate, metrics in manifest["candidate_metrics"].items()
        ]
    )


def _day35_horizon_metrics_table(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Forecast made on": horizon,
                "Buildings": int(metrics["rows"]),
                "MAE": f"{float(metrics['mae_kg']) * 1000:.0f} g",
                "RMSE": f"{float(metrics['rmse_kg']) * 1000:.0f} g",
                "Within 200 g": f"{float(metrics['within_200g_rate']):.1%}",
            }
            for horizon, metrics in manifest["selected_metrics"]["horizon"].items()
        ]
    )


FEATURE_DISPLAY = {
    "cycle_day": "Flock age",
    "beginning_inventory": "Beginning population",
    "percentage_alive": "Current survival",
    "mortality_daily_per_1000": "Latest daily mortality",
    "mortality_recent_3d_per_1000": "Recent 3-day mortality",
    "mortality_trend_delta_per_1000": "Mortality trend",
    "feed_daily_per_1000_birds": "Latest feed per 1,000 birds",
    "feed_cumulative_per_1000_birds": "Cumulative feed per 1,000 birds",
    "temperature_recent_avg_c": "Recent temperature",
    "humidity_recent_avg_pct": "Recent humidity",
    "is_lags_building": "Lagundi building indicator",
}


def _global_recovery_importance_table(manifest: dict[str, object]) -> pd.DataFrame:
    rows = []
    for item in manifest.get("global_feature_importance", []):
        feature = str(item["feature"])
        missing = feature.startswith("missing__")
        source = feature.removeprefix("missing__")
        rows.append(
            {
                "Model input": (
                    f"Missing-data flag: {FEATURE_DISPLAY.get(source, source.replace('_', ' ').title())}"
                    if missing
                    else FEATURE_DISPLAY.get(source, source.replace("_", " ").title())
                ),
                "Relative reliance": f"{float(item['absolute_importance_pct']):.1f}%",
                "When this input is higher": item["direction"],
                "Standardized effect": f"{float(item['coefficient_per_standard_deviation']) * 100:+.2f} recovery pts",
            }
        )
    return pd.DataFrame(rows)


def _owner_recovery_driver_table(contributions: pd.DataFrame) -> pd.DataFrame:
    """Return the five clearest recorded inputs moving one recovery estimate."""

    if contributions.empty:
        return pd.DataFrame()
    recorded = contributions.loc[
        ~contributions["Model input"].str.startswith("Missing-data flag:")
    ].head(5).copy()
    if recorded.empty:
        return pd.DataFrame()
    recorded["Effect on estimate"] = recorded["Effect on raw estimate"].map(
        lambda value: (
            f"Raises by {abs(float(value)) * 100:.2f} pts"
            if float(value) > 0
            else f"Lowers by {abs(float(value)) * 100:.2f} pts"
            if float(value) < 0
            else "No material effect"
        )
    )
    return recorded.rename(columns={"Model input": "Factor"})[
        ["Factor", "Current value", "Effect on estimate"]
    ]


def _eda_coverage_table(dataset) -> pd.DataFrame:
    cycle_order = (
        dataset.cycles.groupby("cycle_id")["start_date"].min().sort_values().index
    )
    rows = []
    for cycle_id in cycle_order:
        daily = dataset.daily.loc[dataset.daily["cycle_id"] == cycle_id]
        metadata = dataset.cycles.loc[dataset.cycles["cycle_id"] == cycle_id]
        rows.append(
            {
                "Cycle": cycle_id,
                "Buildings with data": int(metadata["building_id"].nunique()),
                "Building-day records": int(len(daily)),
                "Buildings with weight": int(
                    daily.loc[daily["weight_measured"], "building_id"].nunique()
                ),
                "Weight measurements": int(daily["weight_measured"].sum()),
                "Environment coverage": float(
                    daily[["temperature_avg_c", "humidity_avg_pct"]]
                    .notna()
                    .any(axis=1)
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _eda_survival_paths(dataset, latest_cycle: str) -> pd.DataFrame:
    metadata = dataset.cycles.loc[
        dataset.cycles["cycle_id"] != latest_cycle,
        [
            "cycle_id",
            "building_id",
            "beginning_inventory",
            "final_recovery_rate",
        ],
    ].copy()
    if metadata.empty:
        return pd.DataFrame()
    median_recovery = float(metadata["final_recovery_rate"].median())
    metadata["Outcome group"] = np.where(
        metadata["final_recovery_rate"] >= median_recovery,
        "Higher-recovery half",
        "Lower-recovery half",
    )
    daily = dataset.daily.loc[
        dataset.daily["operational_recorded"] & dataset.daily["age_day"].le(35),
        ["cycle_id", "building_id", "age_day", "population"],
    ].merge(metadata, on=["cycle_id", "building_id"], how="inner")
    daily["Recorded survival (%)"] = (
        pd.to_numeric(daily["population"], errors="coerce")
        / daily["beginning_inventory"]
        * 100
    )
    return (
        daily.groupby(["age_day", "Outcome group"], as_index=False)[
            "Recorded survival (%)"
        ]
        .mean()
        .pivot(index="age_day", columns="Outcome group", values="Recorded survival (%)")
        .sort_index()
    )


def _next_step(row: pd.Series) -> str:
    return str(row["recommended_action"])


def _day35_milestone(dataset, cycle_id: str, building_id: str, as_of: object, row: pd.Series) -> tuple[str, str]:
    """Return a plain-language Day 35 milestone status without inferring an unobserved weight."""

    if row["state"] == "Inactive" or pd.isna(row["cycle_day"]):
        return "Not started", "The flock has not yet been placed in this building."
    cycle_day = int(row["cycle_day"])
    if cycle_day < 35:
        remaining = 35 - cycle_day
        if pd.isna(row["latest_weight_kg"]):
            evidence = "No measured weight is available yet to judge progress."
        else:
            relationship = "on or above" if float(row["weight_gap_pct"]) <= 0 else "below"
            evidence = (
                f"Latest measurement: {float(row['latest_weight_kg']):.3f} kg on Day "
                f"{int(row['weight_measurement_day'])}, {relationship} that day’s target."
            )
        return f"Upcoming — {remaining} day{'s' if remaining != 1 else ''} remaining", evidence

    day35 = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id)
        & (dataset.daily["building_id"] == building_id)
        & (dataset.daily["age_day"] == 35)
        & (dataset.daily["record_date"] <= pd.Timestamp(as_of))
        & dataset.daily["weight_measured"]
    ]
    if day35.empty:
        return "Unknown", "No measured Day 35 weight was recorded; Canary will not infer one from a later harvest weight."
    observed = float(day35.sort_values("record_date").iloc[-1]["bodyweight_kg"])
    if observed >= 2.0:
        return "Achieved", f"Recorded Day 35 weight was {observed:.3f} kg, meeting the 2.0 kg milestone."
    return "Missed", f"Recorded Day 35 weight was {observed:.3f} kg, below the 2.0 kg milestone."


VIEW_PRIORITIES = "Home"
VIEW_DETAILS = "Building View"
VIEW_VALUE = "Business Value"
VIEW_ACTIONS = "Action Playbook"
VIEW_CHECKS = "Data & Settings"
VIEW_EVIDENCE = "EDA"
VIEW_METHODS = "Canary Methodology"


PAGE_HOME = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
PAGE_BUILDING = st.Page(
    "pages/building.py", title="Building View", icon=":material/domain:"
)
PAGE_VALUE = st.Page(
    "pages/business_value.py", title="Business Value", icon=":material/payments:"
)
PAGE_EDA = st.Page("pages/eda.py", title="EDA & Insights", icon=":material/insights:")
PAGE_METHODS = st.Page(
    "pages/methodology.py", title="Canary Methodology", icon=":material/schema:"
)
PAGE_ACTIONS = st.Page(
    "pages/action_playbook.py", title="Action Playbook", icon=":material/checklist:"
)
PAGE_DATA = st.Page(
    "pages/data_settings.py", title="Data & Settings", icon=":material/database:"
)

navigation = st.navigation(
    {
        "Farm owner": [PAGE_HOME, PAGE_BUILDING, PAGE_VALUE],
        "Capstone evidence": [PAGE_EDA, PAGE_METHODS],
        "Administration": [PAGE_ACTIONS, PAGE_DATA],
    }
)
navigation.run()
selected_view = os.getenv(
    "CANARY_TEST_VIEW", st.session_state.get("_canary_view", VIEW_PRIORITIES)
)


VALUE_STATE_DEFAULTS = {
    "value_price_php_per_kg": DEFAULT_PRICE_PHP_PER_KG,
    "value_sale_weight_kg": DEFAULT_SALE_WEIGHT_KG,
    "value_recovery_improvement_pp": DEFAULT_RECOVERY_IMPROVEMENT_PP,
    "value_cycles_per_year": DEFAULT_CYCLES_PER_YEAR,
}


def _value_assumptions() -> ValueAssumptions:
    return ValueAssumptions(
        price_php_per_kg=float(
            st.session_state.get(
                "value_price_php_per_kg", DEFAULT_PRICE_PHP_PER_KG
            )
        ),
        sale_weight_kg=float(
            st.session_state.get("value_sale_weight_kg", DEFAULT_SALE_WEIGHT_KG)
        ),
        recovery_improvement_pp=float(
            st.session_state.get(
                "value_recovery_improvement_pp",
                DEFAULT_RECOVERY_IMPROVEMENT_PP,
            )
        ),
        cycles_per_year=int(
            st.session_state.get("value_cycles_per_year", DEFAULT_CYCLES_PER_YEAR)
        ),
    )


def _reset_value_assumptions() -> None:
    for key, value in VALUE_STATE_DEFAULTS.items():
        st.session_state[key] = value


def _remember_selected_cycle(widget_key: str) -> None:
    """Keep the chosen cycle stable when moving between Streamlit pages."""

    st.session_state["canary_cycle_choice"] = st.session_state[widget_key]


def _open_building_details(building_id: str) -> None:
    st.session_state["detail_building"] = building_id
    st.switch_page(PAGE_BUILDING)


def _show_priorities() -> None:
    st.switch_page(PAGE_HOME)


def _show_evidence() -> None:
    st.switch_page(PAGE_EDA)


def _card_driver(row: pd.Series) -> str:
    """Return the clearest single reason for a building's rating."""
    scores = {
        "weight": row.get("weight_score", pd.NA),
        "survival": row.get("survival_score", pd.NA),
        "mortality": row.get("mortality_score", pd.NA),
        "peer": row.get("peer_score", pd.NA),
    }
    available = {name: float(value) for name, value in scores.items() if pd.notna(value)}
    if not available or max(available.values()) <= 0:
        return "No material warning signal is above the current rule thresholds."
    leading = max(available, key=available.get)
    if leading == "weight" and pd.notna(row.get("weight_gap_pct")):
        return f"Weight is {abs(float(row['weight_gap_pct'])):.1f}% below its age-specific target."
    if leading == "survival" and pd.notna(row.get("survival_gap_pp")):
        return f"Survival is {abs(float(row['survival_gap_pp'])):.1f} points below the expected path."
    if leading == "mortality":
        return "Recent mortality level or trend is above the current rule threshold."
    if leading == "peer":
        return "Performance is weaker than comparable buildings in the same cycle."
    return str(row.get("risk_pattern", "Recorded warning signal"))


PATTERN_DISPLAY = {
    "Farm-Wide Drift": (
        "Farm-wide concern",
        "Several buildings show warning signals at the same time.",
    ),
    "Localized Building Drift": (
        "Building-specific concern",
        "This building is weaker than comparable buildings in this cycle.",
    ),
    "Growth + Survival Drift": (
        "Weight and survival concern",
        "Both growth progress and survival signals need attention.",
    ),
    "Survival Concern Only": (
        "Survival concern",
        "Survival or mortality is off track; weight is not the leading issue.",
    ),
    "Weight Lag Only": (
        "Weight behind target",
        "Measured weight is behind the farm target for this age.",
    ),
    "No Material Drift": (
        "No material concern",
        "No scored warning sign is above the current thresholds.",
    ),
}


def _pattern_display(pattern: object) -> tuple[str, str]:
    internal = str(pattern)
    return PATTERN_DISPLAY.get(
        internal,
        (internal, "Open the building details to review the recorded warning evidence."),
    )


def _attach_owner_action_context(
    snapshot: pd.DataFrame,
    dataset: object,
    cycle_id: str,
    as_of: object,
) -> pd.DataFrame:
    """Attach the most specific supported operating alert to each current building."""

    output = snapshot.copy()
    for column in (
        "owner_reason_title",
        "owner_reason_detail",
        "owner_action",
        "owner_action_basis",
    ):
        output[column] = pd.NA
    for index, row in output.iterrows():
        if str(row.get("state")) not in {"Active", "Incomplete"}:
            continue
        alerts = evaluate_operational_alerts(
            dataset, cycle_id, str(row["building_id"]), pd.Timestamp(as_of)
        )
        if alerts:
            top = alerts[0]
            output.at[index, "owner_reason_title"] = str(top["title"])
            output.at[index, "owner_reason_detail"] = str(top["evidence"])
            output.at[index, "owner_action"] = str(top["next_check"])
            output.at[index, "owner_action_basis"] = (
                f"{top['severity']} operational alert · provisional threshold pending Doc Raymond validation"
            )
            continue
        risk_score = row.get("risk_score", pd.NA)
        if pd.notna(risk_score) and float(risk_score) > 0:
            output.at[index, "owner_reason_title"] = "Performance warning; operating cause not yet confirmed"
            output.at[index, "owner_reason_detail"] = _card_driver(row)
            output.at[index, "owner_action"] = (
                "Record or verify current temperature, humidity, feed intake, and water availability; then inspect the specific system that is outside its age-based target."
            )
            output.at[index, "owner_action_basis"] = "Performance rule triggered; no current operating alert identified"
        else:
            output.at[index, "owner_reason_title"] = "No material warning signal"
            output.at[index, "owner_reason_detail"] = _card_driver(row)
            output.at[index, "owner_action"] = str(row.get("recommended_action", "Continue normal monitoring."))
            output.at[index, "owner_action_basis"] = "Routine monitoring"
    return output


def _building_card(row: pd.Series) -> str:
    building_id = html.escape(str(row["building_id"]))
    state = str(row["state"])

    if state == "Harvest completed":
        completion = pd.Timestamp(row["completion_date"]).strftime("%d %b %Y")
        actual_weight = _grams(row.get("actual_final_average_weight_kg", pd.NA))
        weight_status = html.escape(str(row.get("actual_final_weight_status", "Not available")))
        backtest_parts = []
        if pd.notna(row.get("day14_projected_recovery", pd.NA)):
            recovery_error = float(row["day14_recovery_error"]) * 100
            backtest_parts.append(
                f"Recovery: projected <strong>{_percent(row['day14_projected_recovery'])}</strong> → "
                f"actual <strong>{_percent(row['actual_harvest_recovery'])}</strong> "
                f"({recovery_error:+.1f} pts prediction error)"
            )
        if pd.notna(row.get("day14_projected_day35_weight_kg", pd.NA)):
            weight_error = float(row["day14_weight_error_kg"]) * 1000
            backtest_parts.append(
                f"Day 35 weight: projected <strong>{_grams(row['day14_projected_day35_weight_kg'])}</strong> → "
                f"recorded <strong>{_grams(row['day14_actual_day35_weight_kg'])}</strong> "
                f"({weight_error:+.0f} g prediction error)"
            )
        backtest = (
            '<div class="backtest"><strong>Historical Day 14 model check</strong><br>'
            + "<br>".join(backtest_parts)
            + "</div>"
            if backtest_parts
            else '<div class="backtest"><strong>Historical Day 14 model check</strong><br>Not available for this building-cycle.</div>'
        )
        return f"""
        <div class="card-body">
          <div class="head"><div class="name">{building_id}</div><span class="pill completed">Harvest completed</span></div>
          <div class="meta">Completed on {completion}</div>
          <div class="outcome-stack">
            <div class="outcome-row"><div class="outcome-name">Actual harvest recovery</div><div class="outcome-detail"><div class="outcome-flow"><span>Recorded result</span><strong>{_percent(row.get('actual_harvest_recovery', pd.NA))}</strong></div></div></div>
            <div class="outcome-row"><div class="outcome-name">Actual final avg weight (g)</div><div class="outcome-detail"><div class="outcome-flow"><span>Recorded result</span><strong>{actual_weight}</strong></div></div></div>
          </div>
          <div class="sub">Recorded outcomes only · {weight_status}</div>
          {backtest}
        </div>
        """

    if state == "Inactive":
        has_cycle_record = pd.notna(row["placement_date"])
        status = "Not started" if has_cycle_record else "No cycle data"
        status_class = "scheduled" if has_cycle_record else "not-rated"
        message = (
            f"This flock starts on {pd.Timestamp(row['placement_date']).strftime('%d %b %Y')}. "
            "Choose a later review date to see its risk and outlook."
            if has_cycle_record
            else "No building data for this building for the selected cycle."
        )
        return f"""
        <div class="card-body">
          <div class="head"><div class="name">{building_id}</div><span class="pill {status_class}">{status}</span></div>
          <div class="empty-state"><strong>{html.escape(message)}</strong><br><br>No risk score, outlook, or action is shown until flock data is available.</div>
        </div>
        """

    rating = str(row["risk_rating"])
    rating_class = rating.lower().replace(" ", "-")
    rating_text = "Not rated" if rating == "Not rated" else f"{rating} risk"
    day = f"Day {int(row['cycle_day'])}"
    recovery_gap, recovery_class = _gap(row["recovery_target_gap_pp"], " pts")
    if pd.isna(row["projected_day35_weight_kg"]):
        weight_gap, weight_class = "Needs a measured weight", "neutral"
    else:
        weight_gap, weight_class = _gap(row["day35_weight_target_gap_kg"] * 1000, " g")
    weight_value = _grams(row["projected_day35_weight_kg"])
    freshness_badge = ""
    if row["state"] == "Incomplete":
        freshness_badge = '<span class="micro-badge">Daily update incomplete</span>'
    elif row["state"] == "Records ended":
        freshness_badge = '<span class="micro-badge">Latest available record</span>'
    evidence_note = f'<span class="micro-badge">{int(row["scored_dimensions"])}/4 risk checks</span>'
    predicted_recovery = row.get("predicted_final_recovery", pd.NA)
    revenue_at_risk = row.get("gross_revenue_at_risk_php", pd.NA)
    current_recovery = row.get("percentage_alive", pd.NA)
    latest_weight = row.get("latest_weight_kg", pd.NA)
    weight_day = row.get("weight_measurement_day", pd.NA)
    current_weight_text = (
        "No weight yet"
        if pd.isna(latest_weight)
        else f"{_grams(latest_weight)} · measured Day {int(weight_day)}"
    )
    driver = str(row.get("owner_reason_detail", _card_driver(row)))
    pattern_title = str(row.get("owner_reason_title", _pattern_display(row["risk_pattern"])[0]))
    pattern_subtitle = str(row.get("owner_action_basis", _pattern_display(row["risk_pattern"])[1]))
    owner_action = str(row.get("owner_action", row["recommended_action"]))
    return f"""
    <div class="card-body">
      <div class="head"><div class="name">{building_id}</div><span class="pill {rating_class}">{html.escape(rating_text)}</span></div>
      <div class="card-summary"><div class="meta">{day} · Score {'—' if pd.isna(row['risk_score']) else str(int(row['risk_score'])) + '/12'}</div><div>{evidence_note} {freshness_badge}</div></div>
      <div class="driver"><span class="pattern-title">{html.escape(pattern_title)}</span><span class="pattern-subtitle">{html.escape(pattern_subtitle)}<br><strong>Why now:</strong> {html.escape(driver)}</span></div>
      <div class="outcome-stack">
        <div class="outcome-row"><div class="outcome-name">Harvest recovery</div><div class="outcome-detail"><div class="outcome-flow"><span>Current recorded: {_percent(current_recovery)}</span><span class="outcome-arrow">→</span><strong>Projected: {_percent(predicted_recovery)}</strong></div><span class="gap-tag {recovery_class}">{html.escape(recovery_gap)} · harvest goal 95%</span></div></div>
        <div class="outcome-row"><div class="outcome-name">Average weight (g)</div><div class="outcome-detail"><div class="outcome-flow"><span>Latest: {html.escape(current_weight_text)}</span><span class="outcome-arrow">→</span><strong>Projected Day 35: {weight_value}</strong></div><span class="gap-tag {weight_class}">{html.escape(weight_gap)} · Day 35 goal 2,000 g</span></div></div>
      </div>
      <div class="value-strip"><div class="label">Gross revenue at risk to 95%</div><strong>{_php(revenue_at_risk)}</strong></div>
      <div class="action"><div class="label">Next action · {html.escape(str(row['recommendation_urgency']))}</div>{html.escape(owner_action)}</div>
    </div>
    """


with st.sidebar:
    st.header("Choose what to review")
    uploaded = st.file_uploader("Farm workbook", type=["xlsx"], help="Canary reads the file but never changes it.")
    uploaded_performance = st.file_uploader(
        "Final-weight workbook (optional)",
        type=["xlsx"],
        help="Upload Farm Performance Summary.xlsx to show actual final average weights for completed cycles. Canary reads only its final average-weight field.",
    )

try:
    if uploaded is not None:
        dataset = _load_upload(uploaded.getvalue(), uploaded.name)
        source_description = f"Uploaded: {uploaded.name}"
    else:
        default_path = _default_workbook()
        if not default_path.exists():
            st.info("Upload FARM HARVEST DATA.xlsx to begin.")
            st.stop()
        dataset = _load_path(str(default_path), default_path.stat().st_mtime_ns)
        source_description = default_path.name
except WorkbookValidationError as exc:
    st.error(str(exc))
    st.stop()

if dataset.quality.blocking_errors:
    st.error("Canary found a data problem that must be corrected before results can be shown.")
    for error in dataset.quality.blocking_errors:
        st.error(error)
    st.stop()

with st.sidebar:
    st.success("Workbook ready")
    st.caption(source_description)
    cycle_options = dataset.cycles.groupby("cycle_id")["start_date"].min().sort_values().index.tolist()
    st.caption("Choose the production batch you want to review.")
    # Streamlit treats a widget on each page as a separate widget, even when it
    # has the same visible label. Keep the business selection in a page-neutral
    # shadow value so Overview -> Building View -> Overview never changes cycle.
    remembered_cycle = st.session_state.get("canary_cycle_choice", cycle_options[-1])
    if remembered_cycle not in cycle_options:
        remembered_cycle = cycle_options[-1]
    cycle_widget_key = f"canary_cycle_widget_{selected_view.lower().replace(' ', '_')}"
    # A page may retain an older widget value. Remove it only when it conflicts
    # with the page-neutral choice, then let ``index`` initialize the widget.
    # This avoids Streamlit's "default plus Session State value" warning.
    if st.session_state.get(cycle_widget_key) != remembered_cycle:
        st.session_state.pop(cycle_widget_key, None)
    selected_cycle = st.selectbox(
        "Harvest cycle",
        cycle_options,
        index=cycle_options.index(remembered_cycle),
        key=cycle_widget_key,
        on_change=_remember_selected_cycle,
        args=(cycle_widget_key,),
        help="Choose the production batch or growing round you want to review.",
    )
    st.session_state["canary_cycle_choice"] = selected_cycle
    current_cycle = latest_cycle_id(dataset)
    historical_cycle = selected_cycle != current_cycle
    minimum_date, maximum_date = cycle_date_bounds(dataset, selected_cycle)
    if historical_cycle:
        selected_date = maximum_date
        st.info(
            "Historical cycle: Canary shows completed outcomes at each building’s last recorded date."
        )
    else:
        selected_date = st.date_input(
            "Review date",
            value=default_as_of_date(dataset, selected_cycle),
            min_value=minimum_date,
            max_value=maximum_date,
            format="DD/MM/YYYY",
            help="Choose the day you want Canary to stand on. Only information recorded on or before this day is used.",
        )
        st.caption("Review date is an ‘as-of’ date—not the predicted harvest date. Future records are never used.")

rules = load_risk_rules()
recommendation_playbook = load_recommendation_playbook()
value_assumptions = _value_assumptions()
# Model artifacts can be replaced while a local Streamlit session is open. Clear the
# lightweight loaders so an older cached manifest never breaks a refreshed page.
if hasattr(load_model_bundle, "cache_clear"):
    load_model_bundle.cache_clear()
if hasattr(load_day35_manifest, "cache_clear"):
    load_day35_manifest.cache_clear()
recovery_manifest, _recovery_model = load_model_bundle("recovery")
day35_manifest = load_day35_manifest()
performance_path = _default_performance_workbook()
final_weight_labels = None
performance_source_status = "Farm Performance Summary is not available."
if uploaded_performance is not None:
    try:
        final_weight_labels = _load_performance_upload(
            uploaded_performance.getvalue(), uploaded_performance.name
        )
        performance_source_status = f"Uploaded: {uploaded_performance.name}"
    except (OSError, ValueError) as exc:
        performance_source_status = f"Uploaded final-weight source could not be read: {exc}"
elif performance_path.exists():
    try:
        final_weight_labels = _load_performance_path(
            str(performance_path), performance_path.stat().st_mtime_ns
        )
        performance_source_status = performance_path.name
    except (OSError, ValueError) as exc:
        performance_source_status = f"Final-weight source could not be read: {exc}"

if historical_cycle:
    snapshot = build_cycle_snapshot(dataset, selected_cycle, selected_date)
    outcomes = build_historical_outcomes(dataset, final_weight_labels)
    outcomes = attach_historical_day14_backtests(
        outcomes, recovery_manifest, day35_manifest
    )
    outcomes = outcomes.loc[outcomes["cycle_id"] == selected_cycle]
    snapshot = snapshot.merge(
        outcomes,
        on=["cycle_id", "building_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )
    recorded = snapshot["placement_date"].notna()
    snapshot.loc[recorded, "state"] = "Harvest completed"
    snapshot.loc[recorded, "status_note"] = (
        "Historical cycle completed under the capstone last-recorded-date convention."
    )
    ranked = snapshot.sort_values("building_order").reset_index(drop=True)
else:
    risk_snapshot = score_cycle_snapshot(dataset, selected_cycle, selected_date, rules)
    snapshot = apply_recommendations(
        attach_forecasts(dataset, risk_snapshot), recommendation_playbook
    )
    snapshot = attach_business_value(snapshot, value_assumptions)
    snapshot = _attach_owner_action_context(
        snapshot, dataset, selected_cycle, selected_date
    )
    ranked = (
        snapshot.assign(_rated=snapshot["risk_score"].notna())
        .sort_values(["_rated", "risk_score", "building_order"], ascending=[False, False, True])
        .drop(columns="_rated")
        .reset_index(drop=True)
    )
all_buildings = snapshot.sort_values("building_order").reset_index(drop=True)
if historical_cycle:
    placed = ranked.iloc[0:0]
    needs_attention = 0
else:
    placed = ranked.loc[ranked["state"].isin(["Active", "Incomplete"])]
    needs_attention = placed["risk_rating"].isin(["Medium", "High", "Critical"]).sum()
cycle_meta = dataset.cycles.loc[dataset.cycles["cycle_id"] == selected_cycle]
recorded_buildings = int(cycle_meta["building_id"].nunique())
unrecorded_names = all_buildings.loc[
    (all_buildings["state"] == "Inactive") & all_buildings["placement_date"].isna(),
    "building_id",
].tolist()

if selected_view == VIEW_PRIORITIES:
    st.markdown(
        """
        <div class="hero"><small>PROJECT CANARY · EARLY WARNING AND DECISION SUPPORT</small>
          <h1>Identify at-risk buildings early and act before targets are missed.</h1>
          <p>Project Canary is an early-warning and decision-support system for broiler farms. It identifies buildings at risk of missing the 2,000 g Day 35 and 95% harvest-recovery targets, projects both outcomes from the latest available data, explains why a building was flagged, and recommends what management should check next.</p>
        </div>
        <div class="intro-grid">
          <div class="intro-panel"><span class="intro-kicker">The management problem</span><strong>Daily records do not clearly show which building needs attention first.</strong><span>Weight, survival, mortality, feed, and environmental readings are spread across rows and dates, so the six buildings must be compared manually.</span></div>
          <div class="intro-panel solution"><span class="intro-kicker">What Canary does</span><strong>Canary turns the latest records into one daily decision view.</strong><span>It scores operational risk, projects Day 35 weight and harvest recovery, shows each gap to target, and recommends the next inspection.</span></div>
        </div>
        <div class="decision-question">
          <div class="decision-icon">?</div>
          <div><span class="decision-kicker">The business question</span><strong>Which buildings are at risk of missing the 2,000 g Day 35 or 95% harvest-recovery goals, what results are currently projected, why are they at risk, and what should management do next?</strong><div class="decision-goals"><span class="goal-chip">2,000 g average weight by Day 35</span><span class="goal-chip">95% recovery at harvest</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="context">
      <strong>Cycle {html.escape(str(selected_cycle))}</strong>
      <span class="dot">•</span>
      <span>{'Completion summary' if historical_cycle else 'Showing what was known by'} <strong>{pd.Timestamp(selected_date).strftime('%d %b %Y')}</strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_view == VIEW_PRIORITIES:
    if historical_cycle:
        summary = st.columns(3)
        historical_recorded = ranked.loc[ranked["state"] == "Harvest completed"]
        recovery_actuals = int(historical_recorded["actual_harvest_recovery"].notna().sum())
        weight_actuals = int(
            historical_recorded["actual_final_average_weight_kg"].notna().sum()
        )
        summary[0].metric(
            "Completed buildings",
            f"{len(historical_recorded)} of 6",
            help="Buildings with recorded data in this completed historical cycle.",
        )
        summary[1].metric(
            "Actual recovery results",
            f"{recovery_actuals} of {len(historical_recorded)}",
            help="Completed buildings with ending and beginning population available.",
        )
        summary[2].metric(
            "Actual weight results",
            f"{weight_actuals} of {len(historical_recorded)}",
            help="Completed buildings with a defensible final average weight match in Farm Performance Summary.",
        )
    else:
        total_revenue_at_risk = placed["gross_revenue_at_risk_php"].sum(
            min_count=1
        )
        priority = ranked.loc[ranked["state"].isin(["Active", "Incomplete"])].head(1)
        attention_names = placed.loc[
            placed["risk_rating"].isin(["Medium", "High", "Critical"]), "building_id"
        ].tolist()
        forecastable_recovery = placed.loc[
            placed["predicted_final_recovery"].notna()
            & placed["beginning_inventory"].notna()
        ]
        if forecastable_recovery.empty or forecastable_recovery["beginning_inventory"].sum() <= 0:
            portfolio_recovery = pd.NA
            portfolio_gap_text = "No recovery outlook is available"
        else:
            portfolio_recovery = float(
                (
                    forecastable_recovery["predicted_final_recovery"]
                    * forecastable_recovery["beginning_inventory"]
                ).sum()
                / forecastable_recovery["beginning_inventory"].sum()
            )
            portfolio_gap = (portfolio_recovery - 0.95) * 100
            portfolio_gap_text = (
                f"{abs(portfolio_gap):.1f} pts {'above' if portfolio_gap >= 0 else 'below'} the 95% goal"
            )
        if priority.empty:
            priority_name = "No current flock"
            priority_sub = "Nothing to review on this date"
        else:
            first = priority.iloc[0]
            priority_name = html.escape(str(first["building_id"]))
            priority_sub = html.escape(
                f"{first['risk_rating']} risk · {first['recommendation_urgency']}"
            )
        attention_value = f"{int(needs_attention)} of {len(placed)}"
        attention_sub = (
            ", ".join(map(str, attention_names))
            if attention_names
            else "No Medium, High, or Critical ratings"
        )
        st.markdown(
            f"""
            <div class="executive-grid">
              <div class="executive-card"><div class="eyebrow">Buildings needing attention</div><div class="metric-value">{html.escape(attention_value)}</div><div class="metric-sub">{html.escape(attention_sub)}</div></div>
              <div class="executive-card"><div class="eyebrow">Projected harvest recovery</div><div class="metric-value">{_percent(portfolio_recovery)}</div><div class="metric-sub">{html.escape(portfolio_gap_text)} · inventory-weighted</div></div>
              <div class="executive-card"><div class="eyebrow">Estimated gross revenue at risk</div><div class="metric-value">{_php(total_revenue_at_risk)}</div><div class="metric-sub">Recovery gap to 95% using editable planning assumptions</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not priority.empty:
            with st.container(border=True):
                priority_columns = st.columns(
                    [1.1, 1.7, 2.5, 1.0], vertical_alignment="center"
                )
                owner_reason_title = str(first.get("owner_reason_title", _pattern_display(first["risk_pattern"])[0]))
                owner_reason_detail = str(first.get("owner_reason_detail", _card_driver(first)))
                owner_action = str(first.get("owner_action", first["recommended_action"]))
                with priority_columns[0]:
                    st.markdown(
                        f'<div class="priority-cell"><span class="priority-kicker">Review first</span><span class="priority-name">{html.escape(str(first["building_id"]))}</span><span class="priority-copy">{html.escape(str(first["risk_rating"]))} risk · {int(first["risk_score"])}/12</span></div>',
                        unsafe_allow_html=True,
                    )
                with priority_columns[1]:
                    st.markdown(
                        f'<div class="priority-cell"><span class="priority-kicker">Most actionable recorded signal</span><span class="priority-name" style="font-size:.92rem">{html.escape(owner_reason_title)}</span><span class="priority-copy">{html.escape(owner_reason_detail)}</span></div>',
                        unsafe_allow_html=True,
                    )
                with priority_columns[2]:
                    st.markdown(
                        f'<div class="priority-cell"><span class="priority-kicker">What management should do</span><span class="priority-copy">{html.escape(owner_action)}</span></div>',
                        unsafe_allow_html=True,
                    )
                with priority_columns[3]:
                    if st.button(
                        f"Open {first['building_id']}",
                        key="review_top_priority",
                        type="primary",
                        use_container_width=True,
                    ):
                        _open_building_details(str(first["building_id"]))
        else:
            st.info("No flock needs a current review on this date. See the six building cards below for their status.")

    st.markdown(
        f'<div class="title">{"Completed building results" if historical_cycle else "Canary Command Center"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Tags 1–3, then Lags 1–3. Earlier cycles show recorded actuals plus a clearly labeled held-out Day 14 model check—never a current risk rating or recommendation.</div>'
        if historical_cycle
        else '<div class="subtitle">See which buildings are at risk of missing the 2,000 g Day 35 or 95% harvest-recovery goals, why they were flagged, and what management should check next.</div>',
        unsafe_allow_html=True,
    )
    for start in (0, 3):
        columns = st.columns(3)
        for column, (_, row) in zip(columns, all_buildings.iloc[start : start + 3].iterrows()):
            with column:
                with st.container(border=True):
                    st.markdown(_building_card(row), unsafe_allow_html=True)
                    no_cycle_data = row["state"] == "Inactive" and pd.isna(row["placement_date"])
                    if st.button(
                        "No details available" if no_cycle_data else f"View {row['building_id']} details",
                        key=f"view_details_{row['building_id']}",
                        use_container_width=True,
                        disabled=no_cycle_data,
                    ):
                        _open_building_details(str(row["building_id"]))
    if not historical_cycle and not recommendation_playbook["approval_status"].startswith("Approved"):
        st.caption("Recommended actions are preliminary guidance pending Doc Raymond’s review.")
    if not historical_cycle:
        st.caption(
            f"Business-value estimate uses an editable planning assumption of ₱{value_assumptions.price_php_per_kg:,.0f}/kg and "
            f"{value_assumptions.sale_weight_kg:.2f} kg per recovered bird. Adjust these in Business Value."
        )

if selected_view == VIEW_VALUE:
    st.markdown('<div class="title">Business Value</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Translate the predicted recovery gap into birds and estimated gross revenue, using assumptions the farm owner can change.</div>',
        unsafe_allow_html=True,
    )
    if historical_cycle:
        st.info(
            f"Business-value scenarios are intended for the latest active cycle ({current_cycle}). "
            "This historical cycle already shows its recorded results."
        )
    else:
        with st.container(border=True):
            st.markdown("**Adjust the planning assumptions**")
            st.caption(
                "The starting values are placeholders—not current market-price claims or farm-approved economics."
            )
            assumption_columns = st.columns(4)
            with assumption_columns[0]:
                st.slider(
                    "Live chicken price (₱/kg)",
                    min_value=50.0,
                    max_value=300.0,
                    value=DEFAULT_PRICE_PHP_PER_KG,
                    step=5.0,
                    key="value_price_php_per_kg",
                    help="Expected selling price per kilogram. Enter the farm's current planning assumption.",
                )
            with assumption_columns[1]:
                st.slider(
                    "Sale weight per bird (kg)",
                    min_value=1.0,
                    max_value=3.5,
                    value=DEFAULT_SALE_WEIGHT_KG,
                    step=0.05,
                    key="value_sale_weight_kg",
                    help="Expected kilograms sold for each additional recovered bird.",
                )
            with assumption_columns[2]:
                st.slider(
                    "Recovery improvement (points)",
                    min_value=0.5,
                    max_value=5.0,
                    value=DEFAULT_RECOVERY_IMPROVEMENT_PP,
                    step=0.5,
                    key="value_recovery_improvement_pp",
                    help="Scenario only: the percentage-point improvement to examine, capped at each building's gap to 95%.",
                )
            with assumption_columns[3]:
                st.slider(
                    "Cycles per year",
                    min_value=1,
                    max_value=10,
                    value=DEFAULT_CYCLES_PER_YEAR,
                    step=1,
                    key="value_cycles_per_year",
                    help="Used only to annualize the selected per-cycle scenario.",
                )
            st.button(
                "Reset assumptions",
                on_click=_reset_value_assumptions,
                use_container_width=False,
            )

        active_value = placed.loc[placed["predicted_final_recovery"].notna()].copy()
        if active_value.empty:
            st.info("No current recovery forecasts are available for a value scenario.")
        else:
            total_at_risk = active_value["gross_revenue_at_risk_php"].sum()
            total_scenario = active_value["scenario_gross_revenue_php"].sum()
            total_birds = active_value["scenario_recovered_birds"].sum()
            value_metrics = st.columns(4)
            value_metrics[0].metric(
                "Estimated gross revenue at risk",
                _php(total_at_risk),
                help="Predicted recovery gap to 95% × beginning birds × assumed revenue per recovered bird.",
            )
            value_metrics[1].metric(
                "Selected scenario opportunity",
                _php(total_scenario),
                help="Gross revenue associated with the selected recovery improvement, capped at the gap to 95%.",
            )
            value_metrics[2].metric(
                "Additional birds represented",
                f"{total_birds:,.0f}",
                help="Bird-equivalent represented by the selected improvement scenario across current buildings.",
            )
            value_metrics[3].metric(
                "Annualized scenario",
                _php(total_scenario * value_assumptions.cycles_per_year),
                help="Selected per-cycle scenario multiplied by the editable number of cycles per year.",
            )

            value_table = active_value[
                [
                    "building_id",
                    "beginning_inventory",
                    "predicted_final_recovery",
                    "recovery_gap_pp",
                    "birds_at_risk",
                    "gross_revenue_at_risk_php",
                    "scenario_improvement_pp",
                    "scenario_recovered_birds",
                    "scenario_gross_revenue_php",
                ]
            ].copy()
            value_table["Predicted recovery"] = value_table[
                "predicted_final_recovery"
            ].map(lambda value: f"{value:.1%}")
            value_table["Gap to 95%"] = value_table["recovery_gap_pp"].map(
                lambda value: f"{value:.1f} pts"
            )
            value_table["Birds at risk"] = value_table["birds_at_risk"].round()
            value_table["Gross revenue at risk"] = value_table[
                "gross_revenue_at_risk_php"
            ].map(_php)
            value_table["Scenario improvement"] = value_table[
                "scenario_improvement_pp"
            ].map(lambda value: f"{value:.1f} pts")
            value_table["Birds represented"] = value_table[
                "scenario_recovered_birds"
            ].round()
            value_table["Scenario gross revenue"] = value_table[
                "scenario_gross_revenue_php"
            ].map(_php)
            value_table = value_table.rename(
                columns={
                    "building_id": "Building",
                    "beginning_inventory": "Beginning birds",
                }
            )[
                [
                    "Building",
                    "Beginning birds",
                    "Predicted recovery",
                    "Gap to 95%",
                    "Birds at risk",
                    "Gross revenue at risk",
                    "Scenario improvement",
                    "Birds represented",
                    "Scenario gross revenue",
                ]
            ]
            st.subheader("Building-level estimate")
            st.dataframe(value_table, hide_index=True, width="stretch")

        with st.expander("How the estimate is calculated"):
            st.markdown(
                """
                **Birds represented by one recovery point** = beginning population × 1%

                **Gross revenue per recovered bird** = assumed sale weight × assumed price per kg

                **Gross revenue at risk** = beginning population × predicted gap to 95% × gross revenue per bird

                **Selected scenario opportunity** = beginning population × selected recovery improvement × gross revenue per bird

                The selected improvement is capped at the building's predicted gap to 95%.
                """
            )
        st.warning(
            "This is an estimated gross-revenue scenario—not profit, guaranteed savings, or proof that a recommended action will produce the selected improvement. "
            "It excludes feed, labor, electricity, treatment, intervention cost, mortality timing, and price changes."
        )

if selected_view == VIEW_DETAILS:
    detail_heading, detail_back = st.columns([5, 1.15], vertical_alignment="center")
    with detail_heading:
        st.markdown(
            f'<div class="title">{"Completed building result" if historical_cycle else "Building decision view"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="subtitle">Review the recorded recovery and final average weight for this completed building.</div>'
            if historical_cycle
            else '<div class="subtitle">See what drove the risk score, what outcome is expected, and what management should check next.</div>',
            unsafe_allow_html=True,
        )
    with detail_back:
        if st.button("← Overview", use_container_width=True):
            _show_priorities()
    detail_options = snapshot.loc[snapshot["placement_date"].notna(), "building_id"].tolist()
    if st.session_state.get("detail_building") not in detail_options:
        st.session_state["detail_building"] = detail_options[0]
    chosen = st.selectbox("Building", detail_options, key="detail_building")
    building = snapshot.loc[snapshot["building_id"] == chosen].iloc[0]
    if historical_cycle:
        completed_on = pd.Timestamp(building["completion_date"]).strftime("%d %b %Y")
        actual_recovery = building.get("actual_harvest_recovery", pd.NA)
        actual_weight = building.get("actual_final_average_weight_kg", pd.NA)
        hcols = st.columns(3)
        hcols[0].metric("Harvest completed on", completed_on)
        hcols[1].metric("Actual harvest recovery", _percent(actual_recovery))
        hcols[2].metric("Actual final average weight", _weight(actual_weight))

        st.subheader("How the actual results were determined")
        ending_population = building.get("actual_ending_population", pd.NA)
        beginning_population = building.get("beginning_inventory", pd.NA)
        if pd.notna(ending_population) and pd.notna(beginning_population):
            st.markdown(
                f"**Actual harvest recovery:** {int(ending_population):,} ending birds ÷ "
                f"{int(beginning_population):,} beginning birds = **{_percent(actual_recovery)}**"
            )
        else:
            st.info("The beginning or ending population needed to calculate actual recovery is missing.")

        if pd.notna(actual_weight):
            st.markdown(
                f"**Actual final average weight:** **{_weight(actual_weight)}**, matched to this "
                f"building and cycle from {performance_source_status}."
            )
        else:
            st.warning(
                f"**Actual final average weight: Not available.** "
                f"{building.get('actual_final_weight_status', 'No defensible source value was found.')}"
            )

        st.subheader("Historical Day 14 model check")
        st.caption(
            "This recreates what Canary would have projected using only Day 14 information while excluding this entire cycle from training. It is a held-out backtest, not a forecast that was issued at the time."
        )
        backtest_rows = []
        if pd.notna(building.get("day14_projected_recovery", pd.NA)):
            recovery_error = float(building["day14_recovery_error"]) * 100
            backtest_rows.append(
                {
                    "Outcome": "Harvest recovery",
                    "Day 14 projection": _percent(building["day14_projected_recovery"]),
                    "Observed result": _percent(building["actual_harvest_recovery"]),
                    "Prediction error": f"{recovery_error:+.2f} pts",
                    "Interpretation": f"Prediction was {abs(recovery_error):.2f} points {'high' if recovery_error > 0 else 'low' if recovery_error < 0 else 'exact'}.",
                }
            )
        if pd.notna(building.get("day14_projected_day35_weight_kg", pd.NA)):
            weight_error_g = float(building["day14_weight_error_kg"]) * 1000
            backtest_rows.append(
                {
                    "Outcome": "Day 35 average weight",
                    "Day 14 projection": _grams(building["day14_projected_day35_weight_kg"]),
                    "Observed result": _grams(building["day14_actual_day35_weight_kg"]),
                    "Prediction error": f"{weight_error_g:+.0f} g",
                    "Interpretation": f"Prediction was {abs(weight_error_g):.0f} g {'high' if weight_error_g > 0 else 'low' if weight_error_g < 0 else 'exact'}.",
                }
            )
        if backtest_rows:
            st.dataframe(pd.DataFrame(backtest_rows), hide_index=True, width="stretch")
        else:
            st.info("No leakage-safe Day 14 backtest is available for this building-cycle.")

        st.info(
            "Historical display rule for this capstone: every cycle before the latest cycle is treated as completed. "
            "The completion date shown is this building’s last recorded daily date. Historical screens do not show current "
            "risk ratings or recommendations; the Day 14 comparison above is explicitly labeled as a historical backtest."
        )
        with st.expander("Source and calculation details"):
            st.markdown(
                f"- Farm daily source: **{dataset.source_name}**\n"
                f"- Cycle and building: **{selected_cycle} · {chosen}**\n"
                f"- Completion-date convention: **last recorded daily date ({completed_on})**\n"
                f"- Recovery formula: **ending recorded population ÷ beginning population**\n"
                f"- Final-weight source: **{performance_source_status}**"
            )
        st.stop()

    dcols = st.columns(5)
    dcols[0].metric("Risk level", building["risk_rating"])
    dcols[1].metric("Risk score", "—" if pd.isna(building["risk_score"]) else f"{int(building['risk_score'])}/12")
    dcols[2].metric(
        "Predicted harvest recovery",
        _percent(building["predicted_final_recovery"]),
    )
    detail_weight = "No projection" if pd.isna(building["projected_day35_weight_kg"]) else _weight(building["projected_day35_weight_kg"])
    dcols[3].metric(
        "Projected Day 35 weight",
        detail_weight,
        help="Estimated average liveweight on production Day 35, compared with the 2.0 kg milestone.",
    )
    dcols[4].metric(
        "Gross revenue at risk",
        _php(building.get("gross_revenue_at_risk_php", pd.NA)),
        help="Editable estimate based on the predicted recovery gap to 95%. Adjust assumptions in Business Value.",
    )

    st.subheader("1 · Decision summary and next check")
    dimension_trace = build_dimension_trace(building, rules)
    operational_alerts = evaluate_operational_alerts(
        dataset, selected_cycle, chosen, pd.Timestamp(selected_date)
    )
    action_style = "success" if str(building["recommendation_guidance_status"]).startswith("Farm-approved") else "warning"
    getattr(st, action_style)(
        f"{building['recommendation_urgency']} — {building.get('owner_action', _next_step(building))}"
    )
    owner_columns = st.columns(3)
    with owner_columns[0]:
        with st.container(border=True):
            st.markdown("**Why this building needs attention**")
            scored_reasons = dimension_trace.loc[
                dimension_trace["Score"].notna() & (dimension_trace["Score"] > 0)
            ].sort_values("Score", ascending=False)
            if scored_reasons.empty:
                st.write("No recorded warning sign is above the current thresholds.")
            else:
                for _, reason in scored_reasons.head(3).iterrows():
                    st.markdown(
                        f"- **{reason['Dimension']} · {int(reason['Score'])}/3:** {reason['Raw observations']}"
                    )
            owner_total = "—" if pd.isna(building["risk_score"]) else f"{int(building['risk_score'])}/12"
            st.caption(
                f"Total: {owner_total} → {building['risk_rating']} risk. This is an attention rating, not a probability."
            )
    with owner_columns[1]:
        with st.container(border=True):
            st.markdown("**Possible contributing conditions**")
            if operational_alerts:
                for alert in operational_alerts[:3]:
                    st.markdown(f"- **{alert['check']}:** {alert['evidence']}")
                st.caption(
                    "These conditions were recorded alongside the warning. They are leads to investigate—not proof of cause."
                )
            else:
                st.write("No provisional mortality, temperature, or humidity alert was triggered.")
                st.caption("Feed, water, and heat-stress checks remain limited by unapproved thresholds or missing data.")
    with owner_columns[2]:
        with st.container(border=True):
            st.markdown("**What management should check next**")
            st.write(building.get("owner_action", _next_step(building)))
            if operational_alerts:
                st.markdown("**Also verify:**")
                for alert in operational_alerts[:2]:
                    st.markdown(f"- {alert['next_check']}")
            st.caption(
                f"{building.get('owner_action_basis', building['recommendation_guidance_status'])} · Recommendation rule {building['recommendation_rule_id']}"
            )
    with st.expander("See why this action was selected"):
        st.dataframe(build_recommendation_trace(building), hide_index=True, width="stretch")
        st.markdown(f"**What to check:** {building['recommendation_inspection_checklist']}")
        st.markdown(f"**Escalate when:** {building['recommendation_escalation_trigger']}")
        st.caption(
            f"{building['recommendation_guidance_status']} · Rule {building['recommendation_rule_id']}"
        )

    st.subheader("2 · Risk score breakdown")
    risk_table = dimension_trace.rename(
        columns={
            "Raw observations": "What Canary observed",
            "Score": "Points",
            "Applied thresholds": "Rule applied",
        }
    )[["Dimension", "What Canary observed", "Points", "Rule applied"]].copy()
    risk_table["Points"] = risk_table["Points"].map(
        lambda value: "Not scored" if pd.isna(value) else f"{int(value)}/3"
    )
    risk_total = pd.DataFrame(
        [
            {
                "Dimension": "TOTAL / FINAL RATING",
                "What Canary observed": building["score_equation"],
                "Points": "—" if pd.isna(building["risk_score"]) else f"{int(building['risk_score'])}/12",
                "Rule applied": f"{building['risk_rating']} · {building['risk_label_rule']}",
            }
        ]
    )
    st.dataframe(pd.concat([risk_table, risk_total], ignore_index=True), hide_index=True, width="stretch")
    st.caption(
        "The risk rating is an operational priority score, not the probability of missing the 95% or 2.0 kg goals."
    )

    st.subheader("3 · Forecast deep dive")
    st.caption(
        f"As-of {pd.Timestamp(selected_date).strftime('%d %b %Y')}: later records are excluded. Forecasts do not change the separate rules-based risk score."
    )
    fcols = st.columns(2)
    with fcols[0]:
        if pd.notna(building["recovery_interval_low"]):
            st.caption(
                f"Expected survival range: {_percent(building['recovery_interval_low'])}–{_percent(building['recovery_interval_high'])}. {building['recovery_confidence']}"
            )
        else:
            st.caption(building["recovery_forecast_status"])
    with fcols[1]:
        if pd.notna(building["day35_weight_interval_low_kg"]):
            st.caption(f"Expected Day 35 range: {_weight(building['day35_weight_interval_low_kg'])}–{_weight(building['day35_weight_interval_high_kg'])}. {building['day35_weight_confidence']}")
        else:
            st.caption(building["day35_weight_status"])

    milestone_status, milestone_evidence = _day35_milestone(
        dataset, selected_cycle, chosen, selected_date, building
    )
    st.info(f"**Day 35 milestone: {milestone_status}.** {milestone_evidence}")

    history = build_risk_history(dataset, selected_cycle, chosen, selected_date, rules)
    forecast_history = build_forecast_history(dataset, history, selected_cycle, chosen)

    forecast_columns = st.columns(2)
    with forecast_columns[0]:
        with st.container(border=True):
            st.markdown('<span class="model-badge">OUTCOME 1 · HARVEST SURVIVAL</span>', unsafe_allow_html=True)
            recovery_delta = None if pd.isna(building["recovery_target_gap_pp"]) else f"{float(building['recovery_target_gap_pp']):+.1f} pts vs 95% goal"
            st.metric("Current prediction", _percent(building["predicted_final_recovery"]), recovery_delta)
            if pd.notna(building["recovery_interval_low"]):
                st.write(f"Likely range: **{_percent(building['recovery_interval_low'])}–{_percent(building['recovery_interval_high'])}**")
            recovery_note = "This estimate updates as survival, mortality, feed, and available environment evidence are recorded. Its historical training target is last-recorded recovery, used as the capstone proxy until true harvest status is available."
            st.markdown(f'<div class="forecast-note">{recovery_note}</div>', unsafe_allow_html=True)
    with forecast_columns[1]:
        with st.container(border=True):
            st.markdown('<span class="model-badge">OUTCOME 2 · DAY 35 AVERAGE WEIGHT</span>', unsafe_allow_html=True)
            weight_delta = None if pd.isna(building["day35_weight_target_gap_kg"]) else f"{float(building['day35_weight_target_gap_kg']) * 1000:+.0f} g vs 2.0 kg milestone"
            weight_metric_label = "Observed result" if building["day35_weight_scope"] == "Recorded Day 35 result" else "Current projection"
            st.metric(weight_metric_label, detail_weight, weight_delta)
            if pd.notna(building["day35_weight_interval_low_kg"]):
                st.write(f"Likely range: **{_weight(building['day35_weight_interval_low_kg'])}–{_weight(building['day35_weight_interval_high_kg'])}**")
            weight_note = (
                "This is the building's recorded Day 35 measurement."
                if building["day35_weight_scope"] == "Recorded Day 35 result"
                else "This building-responsive baseline adds historically observed remaining growth from the weighing age to the latest measured weight."
            )
            st.markdown(f'<div class="forecast-note">{weight_note}</div>', unsafe_allow_html=True)

    st.markdown("**Current status → projected outcome → gap to goal**")
    st.dataframe(
        _current_vs_outlook_table(building),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Recovery is compared with 95%. Weight is projected specifically to Day 35 and compared with 2.0 kg."
    )

    recovery_contributions = recovery_feature_contributions(
        dataset, selected_cycle, chosen, pd.Timestamp(selected_date)
    )
    st.markdown("**What is driving today’s outlook?**")
    driver_columns = st.columns(2)
    with driver_columns[0]:
        with st.container(border=True):
            st.markdown("**Harvest-recovery forecast · strongest building-specific factors**")
            owner_drivers = _owner_recovery_driver_table(recovery_contributions)
            if owner_drivers.empty:
                st.info("Model-driver details are not available for this estimate.")
            else:
                st.dataframe(owner_drivers, hide_index=True, width="stretch")
            st.caption(
                "These factors move the model estimate up or down. They do not prove what caused the outcome."
            )
    with driver_columns[1]:
        with st.container(border=True):
            st.markdown("**Day 35 weight projection · direct drivers**")
            weight_driver_rows = [
                {
                    "Factor": "Latest measured building weight",
                    "Current evidence": (
                        "Not recorded"
                        if pd.isna(building["latest_weight_kg"])
                        else f"{_grams(building['latest_weight_kg'])} on Day {int(building['weight_measurement_day'])}"
                    ),
                    "How it is used": "Starting point for the projection",
                },
                {
                    "Factor": "Measurement day",
                    "Current evidence": (
                        "Not available"
                        if pd.isna(building["weight_measurement_day"])
                        else f"Day {int(building['weight_measurement_day'])}"
                    ),
                    "How it is used": "Selects historical remaining growth to Day 35",
                },
            ]
            st.dataframe(pd.DataFrame(weight_driver_rows), hide_index=True, width="stretch")
            st.caption(
                "The selected weight method has two direct drivers. Canary does not invent five importance values for a two-input formula."
            )

    if (
        pd.notna(building.get("weight_score"))
        and float(building["weight_score"]) >= 2
        and pd.notna(building["projected_day35_weight_kg"])
    ):
        st.warning(
            "The Day 35 projection does not cancel the rules-based warning. Today’s measured-weight gap remains a reason to inspect this building."
        )

    with st.expander("See the forecast evidence and model proof"):
        evidence_tab, driver_tab, proof_tab = st.tabs(
            ["Evidence available today", "What moved the estimates", "Validation and limitations"]
        )
        with evidence_tab:
            st.caption("This is an input audit—not a claim that every input caused the prediction.")
            evidence = forecast_input_trace(dataset, selected_cycle, chosen, pd.Timestamp(selected_date))
            if evidence.empty:
                st.info("No prediction-time evidence is available for this building and date.")
            else:
                st.dataframe(evidence, hide_index=True, width="stretch")
        with driver_tab:
            st.markdown("**Harvest-recovery model: building-specific drivers**")
            if recovery_contributions.empty:
                st.info("A building-specific model-contribution view is not available for this method.")
            else:
                contribution_view = recovery_contributions.head(8).copy()
                contribution_view["Effect on raw estimate"] = contribution_view[
                    "Effect on raw estimate"
                ].map(lambda value: f"{float(value) * 100:+.2f} pts")
                st.dataframe(contribution_view, hide_index=True, width="stretch")
                st.caption(
                    "These are model associations before the prediction is capped at current survival. They show what moved this estimate relative to the fitted baseline; they do not prove cause or prescribe an adjustment."
                )
            st.markdown("**Day 35 weight method: direct drivers**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Driver": "Latest measured building weight",
                            "Current evidence": (
                                "Not recorded"
                                if pd.isna(building["latest_weight_kg"])
                                else f"{_grams(building['latest_weight_kg'])} on Day {int(building['weight_measurement_day'])}"
                            ),
                            "How it affects the result": "Adds directly to the Day 35 projection",
                        },
                        {
                            "Driver": "Measurement day",
                            "Current evidence": (
                                "Not available"
                                if pd.isna(building["weight_measurement_day"])
                                else f"Day {int(building['weight_measurement_day'])}"
                            ),
                            "How it affects the result": "Selects the historically observed remaining gain to Day 35",
                        },
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The selected Day 35 method is a transparent formula, so coefficient-based feature importance does not apply. Target progress and recent ADG were tested in competing models but were not used by the winner."
            )
        with proof_tab:
            st.dataframe(forecast_trace(building), hide_index=True, width="stretch")
            st.info("The Day 35 weight method is trained only on bodyweights recorded on Day 35 in Farm Harvest Data. Farm Performance Summary is not the target for this primary projection.")

    st.subheader("4 · Additional operational checks")
    st.caption(
        "These checks help investigate possible causes behind a warning. They are separate from both the risk score and the prediction-model drivers."
    )
    if operational_alerts:
        for alert in operational_alerts:
            message = (
                f"**{alert['severity']} · {alert['title']}**  \n"
                f"Recorded evidence: {alert['evidence']}  \n"
                f"Gap: {alert['gap']}  \n"
                f"**Next action:** {alert['next_check']}"
            )
            st.warning(message)
    else:
        st.info("No current operating alert could be confirmed from the recorded mortality, temperature, humidity, or feed checks. Review missing inputs before concluding that conditions are normal.")
    with st.expander("See all possible operational drivers, including missing inputs"):
        driver_trace = build_operational_driver_trace(
            dataset, selected_cycle, chosen, pd.Timestamp(selected_date)
        )
        if driver_trace.empty:
            st.info("No operational evidence is available for this building and date.")
        else:
            st.dataframe(driver_trace, hide_index=True, width="stretch")
    st.caption(
        "Temperature, humidity, feed, and daily-mortality thresholds are provisional. Water is not in the current standardized data, the Daily FI/bird unit needs confirmation, and THI is deferred until one formula and its age-specific limits are approved. None of these checks changes the formal risk score."
    )

    st.subheader("5 · How the outlook changed")
    if forecast_history.empty:
        st.info("No daily history is available for this selection.")
    else:
        chart = forecast_history.set_index("as_of_date")
        hcols = st.columns(3)
        with hcols[0]:
            st.caption("Rules-based risk score (out of 12)")
            st.line_chart(chart[["risk_score"]], height=240)
        with hcols[1]:
            st.caption("Expected harvest survival (%)")
            survival_chart = chart[["predicted_final_recovery"]].mul(100)
            survival_chart["95% goal"] = 95.0
            st.line_chart(survival_chart, height=240)
        with hcols[2]:
            st.caption("Projected Day 35 average weight (kg)")
            weight_chart = chart[["projected_day35_weight_kg"]].copy()
            weight_chart["2.0 kg milestone"] = 2.0
            st.line_chart(weight_chart, height=240)

    with st.expander("Technical audit details"):
        st.markdown(
            f"""
            - Source: **{dataset.source_name}**
            - Cycle / building / review date: **{selected_cycle} · {chosen} · {pd.Timestamp(selected_date).strftime('%d %b %Y')}**
            - Risk rules: **{building['risk_rule_version']}** ({building['risk_approval_status']})
            - Recovery model: **{building['recovery_model_version']}**
            - Day 35 weight method: **{building['day35_weight_model_version']}**
            - Recommendation rule: **{building['recommendation_rule_id']}**
            - Recommendation version: **{building['recommendation_rule_version']}**
            - Recommendation status: **{building['recommendation_guidance_status']}**
            - Weight freshness: **{building['weight_freshness']}**
            - Daily-data freshness: **{building['data_freshness']}**
            """
        )

if selected_view == VIEW_ACTIONS:
    st.markdown('<div class="title">Action playbook</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Each problem pattern maps to one deterministic action. The risk level controls how quickly to respond.</div>',
        unsafe_allow_html=True,
    )
    saved_message = st.session_state.pop("recommendation_saved_message", None)
    if saved_message:
        st.success(saved_message)
    if recommendation_playbook["approval_status"].startswith("Approved"):
        st.success(recommendation_playbook["approval_status"])
    else:
        st.warning(recommendation_playbook["approval_status"])
    action_summary = pd.DataFrame(recommendation_playbook["rules"])[
        ["rule_id", "pattern", "dashboard_action", "approval_status"]
    ].rename(
        columns={
            "rule_id": "Rule",
            "pattern": "Problem pattern",
            "dashboard_action": "Dashboard recommendation",
            "approval_status": "Approval",
        }
    )
    st.dataframe(action_summary, hide_index=True, width="stretch")

    severity_summary = pd.DataFrame(recommendation_playbook["severity_guide"])[
        ["risk_rating", "urgency", "owner_instruction"]
    ].rename(
        columns={
            "risk_rating": "Risk level",
            "urgency": "Response timing",
            "owner_instruction": "Owner instruction",
        }
    )
    with st.expander("Risk-level response timing"):
        st.dataframe(severity_summary, hide_index=True, width="stretch")

    st.subheader("Review or edit one rule")
    st.caption(
        "Saving changes updates the local recommendation configuration. The confirmation box prevents accidental edits."
    )
    rule_options = [rule["rule_id"] for rule in recommendation_playbook["rules"]]
    selected_rule_id = st.selectbox("Rule to review", rule_options, key="recommendation_rule_admin")
    selected_rule = next(
        rule for rule in recommendation_playbook["rules"] if rule["rule_id"] == selected_rule_id
    )
    with st.form("recommendation_rule_form"):
        admin_action = st.text_area(
            "Proposed dashboard recommendation",
            value=selected_rule["dashboard_action"],
            height=90,
        )
        admin_checklist = st.text_area(
            "Inspection checklist",
            value=selected_rule["inspection_checklist"],
            height=130,
        )
        admin_escalation = st.text_area(
            "Escalate when",
            value=selected_rule["escalation_trigger"],
            height=110,
        )
        approval_choices = sorted(ALLOWED_APPROVAL_STATUSES)
        admin_status = st.selectbox(
            "Approval status",
            approval_choices,
            index=approval_choices.index(selected_rule["approval_status"]),
        )
        admin_comments = st.text_area(
            "Doc Raymond comments",
            value=selected_rule["owner_comments"],
            height=80,
        )
        admin_approved_wording = st.text_area(
            "Approved dashboard wording (optional)",
            value=selected_rule["approved_wording"],
            height=90,
        )
        admin_approval_date = st.text_input(
            "Approval date (YYYY-MM-DD)", value=selected_rule["approval_date"]
        )
        confirm_save = st.checkbox(
            "I confirm that I want to change this recommendation rule."
        )
        save_rule = st.form_submit_button("Save rule")
    if save_rule:
        if not confirm_save:
            st.error("Confirm the change before saving.")
        elif not admin_action.strip() or not admin_checklist.strip() or not admin_escalation.strip():
            st.error("Recommendation, inspection checklist, and escalation guidance cannot be blank.")
        else:
            updated_playbook = deepcopy(recommendation_playbook)
            rule_to_update = next(
                rule for rule in updated_playbook["rules"] if rule["rule_id"] == selected_rule_id
            )
            rule_to_update.update(
                {
                    "dashboard_action": admin_action.strip(),
                    "inspection_checklist": admin_checklist.strip(),
                    "escalation_trigger": admin_escalation.strip(),
                    "approval_status": admin_status,
                    "owner_comments": admin_comments.strip(),
                    "approved_wording": admin_approved_wording.strip(),
                    "approval_date": admin_approval_date.strip(),
                }
            )
            try:
                save_recommendation_playbook(updated_playbook, DEFAULT_RECOMMENDATIONS_PATH)
            except (RecommendationConfigurationError, OSError) as exc:
                st.error(str(exc))
            else:
                st.session_state["recommendation_saved_message"] = (
                    f"Saved {selected_rule_id}. The displayed guidance is now refreshed."
                )
                st.rerun()

if selected_view == VIEW_CHECKS:
    quality = dataset.quality
    st.markdown('<div class="title">Data & Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Review workbook readiness, source coverage, and the documented capstone data conventions.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Workbook health")
    st.success("Data checks passed. Canary created one reliable record per building and production day.")
    qcols = st.columns(4)
    qcols[0].metric("Source rows", f"{quality.source_rows:,}")
    qcols[1].metric("Building-day records", f"{quality.canonical_rows:,}")
    qcols[2].metric("Repeated rows combined", f"{quality.duplicate_rows_consolidated:,}")
    qcols[3].metric("Conflicting records", f"{quality.production_conflict_keys:,}")
    st.subheader(f"Coverage for cycle {selected_cycle}")
    st.markdown(f"The workbook has data for **{recorded_buildings} of 6 buildings** in this cycle.")
    if unrecorded_names:
        st.markdown(f"**No building data for this cycle:** {', '.join(unrecorded_names)}.")
    if historical_cycle:
        completed = ranked.loc[ranked["state"] == "Harvest completed"]
        st.markdown(
            f"Actual recovery is available for **{completed['actual_harvest_recovery'].notna().sum()} of {len(completed)} completed buildings**. "
            f"Actual final average weight is available for **{completed['actual_final_average_weight_kg'].notna().sum()} of {len(completed)}**. "
            "Missing final weights are not imputed."
        )
    else:
        measured_operating = int(placed["latest_weight_kg"].notna().sum())
        st.markdown(
            f"Measured weight is available for **{measured_operating} of {len(placed)} current flocks**. "
            "Where it is absent, Canary asks for a measured weight instead of fabricating a building projection."
        )
    st.subheader("Items to know")
    for warning in quality.warnings:
        st.warning(warning)
    with st.expander("What Canary does when preparing the workbook"):
        st.markdown(
            f"""
            - Keeps one production record for each cycle, building, and production day.
            - Combines repeated environmental readings into daily minimum, maximum, and average values.
            - Keeps blank mortality and feed entries as missing—not zero.
            - Treats weight as measured only on the day it was recorded.
            - Stops if repeated rows disagree on core production values.
            """
        )

    st.divider()
    st.subheader("Risk score rules")
    st.caption(
        f"Current version: {rules['version']} · Status: {rules['approval_status']}. "
        "Every building detail page shows the exact measurements, cutoffs, points, total, and label rule that were applied."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Risk check": "1 · Growth progress",
                    "What is measured": "Measured weight gap versus the farm target for that weighing day",
                    "Why it is kept": "Directly tests whether the flock is following the age-specific growth curve",
                },
                {
                    "Risk check": "2 · Survival position",
                    "What is measured": "Current survival gap versus the assumed path to 95% on Day 35",
                    "Why it is kept": "Shows cumulative loss already experienced",
                },
                {
                    "Risk check": "3 · Mortality momentum",
                    "What is measured": "Recent 3-day mortality rate versus the preceding baseline",
                    "Why it is kept": "Shows whether losses are accelerating now",
                },
                {
                    "Risk check": "4 · Peer context",
                    "What is measured": "Worst gap versus comparable buildings at a similar age",
                    "Why it is kept": "Helps separate a building-specific concern from farm-wide conditions",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Audit verdict: keep the four checks for the capstone because they answer different operational questions and match the agreed scope. "
        "However, survival and mortality are related, and peer context can repeat the same underlying signal. Their cutoffs and point weight remain provisional until farm review and historical calibration."
    )
    st.caption(
        "The Farmer Validation Workbook supports separate alerts for body-weight deviation, daily mortality, population loss, temperature, and humidity. "
        "It does not fully specify Canary's four-level 0–3 scoring cutoffs, survival path, peer points, or final Low/Medium/High/Critical bands."
    )

    if st.session_state.pop("risk_rules_saved_message", None):
        st.success("Risk score rules saved. New scores now show the updated version and thresholds.")

    with st.expander("Review or edit scoring thresholds"):
        st.warning(
            "Changing these values changes every current risk score. Use Doc Raymond-approved values, update the rule version, and test the result before operational use."
        )
        st.markdown(
            "**How the point cutoffs work:** 0 points at or below the first cutoff; 1 point above the first through the second; "
            "2 points above the second through the third; and 3 points above the third."
        )
        age_rows = []
        for band in rules["age_bands"]:
            age_rows.append(
                {
                    "Age band": band["label"],
                    "Weight 0 max (%)": band["weight_gap_pct"][0],
                    "Weight 1 max (%)": band["weight_gap_pct"][1],
                    "Weight 2 max (%)": band["weight_gap_pct"][2],
                    "Survival 0 max (pts)": band["survival_gap_pp"][0],
                    "Survival 1 max (pts)": band["survival_gap_pp"][1],
                    "Survival 2 max (pts)": band["survival_gap_pp"][2],
                    "Mortality 0 max (/1,000)": band["mortality_trend_delta_per_1000"][0],
                    "Mortality 1 max (/1,000)": band["mortality_trend_delta_per_1000"][1],
                    "Mortality 2 max (/1,000)": band["mortality_trend_delta_per_1000"][2],
                }
            )
        age_editor = st.data_editor(
            pd.DataFrame(age_rows),
            disabled=["Age band"],
            hide_index=True,
            width="stretch",
            key="risk_age_threshold_editor",
        )

        st.markdown("**Peer-context cutoffs**")
        peer_specs = [
            ("Weight gap versus peers (%)", "weight_gap_excess_pct"),
            ("Survival gap versus peers (pts)", "survival_gap_excess_pp"),
            ("Recent mortality versus peers (/1,000)", "mortality_rate_excess_per_1000"),
        ]
        peer_editor = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "Peer measure": label,
                        "0-point maximum": rules["peer_comparison"][key][0],
                        "1-point maximum": rules["peer_comparison"][key][1],
                        "2-point maximum": rules["peer_comparison"][key][2],
                    }
                    for label, key in peer_specs
                ]
            ),
            disabled=["Peer measure"],
            hide_index=True,
            width="stretch",
            key="risk_peer_threshold_editor",
        )

        st.markdown("**Final score-to-label bands**")
        rating_editor = st.data_editor(
            pd.DataFrame(rules["rating_bands"]).rename(
                columns={"label": "Label", "minimum": "Minimum score", "maximum": "Maximum score"}
            ),
            disabled=["Label"],
            hide_index=True,
            width="stretch",
            key="risk_rating_band_editor",
        )

        target_cols = st.columns(2)
        with target_cols[0]:
            risk_survival_target = st.number_input(
                "Final survival goal (%)",
                min_value=1.0,
                max_value=100.0,
                value=float(rules["survival_target"]["final_target_rate"]) * 100,
                step=0.1,
                key="risk_survival_target",
            )
        with target_cols[1]:
            risk_survival_day = st.number_input(
                "Target day for survival path",
                min_value=1,
                max_value=999,
                value=int(rules["survival_target"]["target_day"]),
                step=1,
                key="risk_survival_target_day",
            )
        version_cols = st.columns(2)
        with version_cols[0]:
            risk_rule_version = st.text_input(
                "Rule version",
                value=str(rules["version"]),
                help="Use a new version whenever approved thresholds change so results remain traceable.",
            )
        with version_cols[1]:
            risk_approval_status = st.selectbox(
                "Validation status",
                ["Provisional - farm validation required", "Farm-approved by Doc Raymond"],
                index=(
                    1
                    if rules["approval_status"] == "Farm-approved by Doc Raymond"
                    else 0
                ),
            )
        confirm_risk_change = st.checkbox(
            "I confirm these scoring changes were reviewed and the version is correct.",
            key="confirm_risk_rule_change",
        )
        if st.button("Save risk score rules", type="primary"):
            if not confirm_risk_change:
                st.error("Confirm the scoring change before saving.")
            elif not risk_rule_version.strip():
                st.error("Rule version cannot be blank.")
            else:
                updated_rules = deepcopy(rules)
                updated_rules["version"] = risk_rule_version.strip()
                updated_rules["approval_status"] = risk_approval_status
                updated_rules["survival_target"] = {
                    "final_target_rate": float(risk_survival_target) / 100,
                    "target_day": int(risk_survival_day),
                }
                for band, (_, edited) in zip(updated_rules["age_bands"], age_editor.iterrows()):
                    band["weight_gap_pct"] = [
                        float(edited["Weight 0 max (%)"]),
                        float(edited["Weight 1 max (%)"]),
                        float(edited["Weight 2 max (%)"]),
                    ]
                    band["survival_gap_pp"] = [
                        float(edited["Survival 0 max (pts)"]),
                        float(edited["Survival 1 max (pts)"]),
                        float(edited["Survival 2 max (pts)"]),
                    ]
                    band["mortality_trend_delta_per_1000"] = [
                        float(edited["Mortality 0 max (/1,000)"]),
                        float(edited["Mortality 1 max (/1,000)"]),
                        float(edited["Mortality 2 max (/1,000)"]),
                    ]
                for (_, key), (_, edited) in zip(peer_specs, peer_editor.iterrows()):
                    updated_rules["peer_comparison"][key] = [
                        float(edited["0-point maximum"]),
                        float(edited["1-point maximum"]),
                        float(edited["2-point maximum"]),
                    ]
                updated_rules["rating_bands"] = [
                    {
                        "label": str(edited["Label"]),
                        "minimum": int(edited["Minimum score"]),
                        "maximum": int(edited["Maximum score"]),
                    }
                    for _, edited in rating_editor.iterrows()
                ]
                try:
                    save_risk_rules(updated_rules, DEFAULT_RULES_PATH)
                except (RiskConfigurationError, OSError, TypeError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["risk_rules_saved_message"] = True
                    st.rerun()

if selected_view == VIEW_EVIDENCE:
    evidence_path = Path(__file__).resolve().parent / "analysis" / "eda_results.json"
    st.markdown('<div class="title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Five questions that test whether the data, early-warning storyline, and forecast methods are useful—and where the evidence remains limited.</div>',
        unsafe_allow_html=True,
    )
    if not evidence_path.exists():
        st.warning("The evidence summary has not been generated yet.")
    else:
        eda = json.loads(evidence_path.read_text(encoding="utf-8"))
        coverage = eda["coverage"]
        associations = eda["associations"]
        day14_groups = eda["day14_target_groups"]
        day14_met = next(item for item in day14_groups if item["day14_target_status"] == "Met/exceeded")
        evidence_rows = pd.DataFrame(eda["evidence_rows"])
        recovery_manifest, _ = load_model_bundle("recovery")
        day35_manifest = load_day35_manifest()
        ecols = st.columns(4)
        ecols[0].metric("Historical building-cycles", coverage["completed_building_cycles"])
        ecols[1].metric("Day 14 → Day 35 pairs", coverage["paired_day14_day35"])
        ecols[2].metric("Day 14 ↔ recovery", f"r = {associations['day14_to_final_recovery']['pearson_r']:.2f}")
        ecols[3].metric(
            "Day 35 projection MAE",
            f"{day35_manifest['selected_metrics']['mae_kg'] * 1000:.0f} g",
        )

        st.success(
            "Strongest useful signal: higher Day 14 weight was associated with higher final harvest recovery in this history. "
            "This supports Day 14 as an early management checkpoint."
        )
        st.warning(
            "Important limit: association is not proof that improving weight alone causes better recovery. "
            "There are only five recorded historical cycles, and birds, weather, feed, disease, and management can move together."
        )

        question_tabs = st.tabs(
            [
                "1 · Data coverage",
                "2 · Day 14 → Day 35",
                "3 · Day 14 → Recovery",
                "4 · Survival paths",
                "5 · Model accuracy",
            ]
        )
        with question_tabs[0]:
            st.subheader("How complete and fresh is the farm data?")
            st.write(
                "Coverage varies materially by cycle and building. Weight is the main limiting input, so Canary always shows when a building-specific Day 35 projection cannot be made."
            )
            coverage_table = _eda_coverage_table(dataset)
            coverage_chart = coverage_table.set_index("Cycle")[[
                "Buildings with data",
                "Buildings with weight",
            ]]
            st.bar_chart(coverage_chart, height=280)
            display_coverage = coverage_table.copy()
            display_coverage["Environment coverage"] = display_coverage[
                "Environment coverage"
            ].map(lambda value: f"{value:.0%}")
            st.dataframe(display_coverage, hide_index=True, width="stretch")
            st.caption(
                "Limitation: a building missing from the workbook cannot yet be distinguished from a building that was not used. This remains a Doc Raymond validation item."
            )

        with question_tabs[1]:
            relationship = associations["day14_to_day35_weight"]
            st.subheader("Is Day 14 weight associated with Day 35 weight?")
            st.write(
                f"Across {relationship['n']} paired building-cycles, higher Day 14 weight had a moderate raw association with higher Day 35 weight (r = {relationship['pearson_r']:.2f})."
            )
            day35_pairs = evidence_rows.dropna(
                subset=["day14_weight_kg", "day35_weight_kg"]
            )
            st.scatter_chart(
                day35_pairs,
                x="day14_weight_kg",
                y="day35_weight_kg",
                color="cycle_id",
                height=320,
            )
            st.caption(
                f"Within-cycle association is much weaker (r = {relationship['within_cycle_r']:.2f}). Only {day14_met['building_cycles']} of {relationship['n']} records met the 400 g target, and none of the observed Day 35 outcomes reached 2.0 kg. Association is not causal proof."
            )

        with question_tabs[2]:
            relationship = associations["day14_to_final_recovery"]
            st.subheader("Is Day 14 weight associated with harvest recovery?")
            st.write(
                f"Yes, in this limited history: higher Day 14 weight was associated with higher last-recorded recovery across {relationship['n']} paired building-cycles (r = {relationship['pearson_r']:.2f})."
            )
            recovery_pairs = evidence_rows.dropna(
                subset=["day14_weight_kg", "recomputed_recovery"]
            ).copy()
            recovery_pairs["Recovery (%)"] = recovery_pairs[
                "recomputed_recovery"
            ] * 100
            st.scatter_chart(
                recovery_pairs,
                x="day14_weight_kg",
                y="Recovery (%)",
                color="cycle_id",
                height=320,
            )
            st.caption(
                f"The within-cycle relationship is r = {relationship['within_cycle_r']:.2f}, but the sample covers only {relationship['cycles']} cycles with paired weights. Other conditions can influence both weight and recovery; Canary therefore says associated with, not caused by."
            )

        with question_tabs[3]:
            st.subheader("When do better and worse recovery paths begin to separate?")
            survival_paths = _eda_survival_paths(dataset, current_cycle)
            if survival_paths.empty:
                st.info("There is not enough historical population data for this comparison.")
            else:
                st.write(
                    "Historical buildings in the lower-recovery half generally show a weaker survival path over the growing cycle. This supports monitoring trajectory—not only the latest mortality number."
                )
                st.line_chart(survival_paths, height=330)
                st.caption(
                    f"Sample: {coverage['completed_building_cycles']} completed historical building-cycles, split at the historical median recovery. This is a descriptive group average; it does not identify the cause of mortality."
                )

        with question_tabs[4]:
            st.subheader("How accurate are the two predictive methods?")
            accuracy_columns = st.columns(2)
            with accuracy_columns[0]:
                st.markdown("**Harvest-recovery forecast**")
                st.metric(
                    "Held-out MAE",
                    f"{recovery_manifest['selected_metrics']['mae'] * 100:.2f} points",
                )
                st.dataframe(
                    _horizon_metrics_table(recovery_manifest, "recovery"),
                    hide_index=True,
                    width="stretch",
                )
            with accuracy_columns[1]:
                st.markdown("**Day 35 weight projection**")
                st.metric(
                    "Held-out MAE",
                    f"{day35_manifest['selected_metrics']['mae_kg'] * 1000:.0f} g",
                )
                st.dataframe(
                    _day35_horizon_metrics_table(day35_manifest),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "Both methods hold out complete cycles. Recovery has only five training cycles; Day 35 weight has 19 outcomes across four cycles and no 2.0 kg target hits. These metrics support planning estimates, not guaranteed target classification."
            )

        with st.expander("Additional findings worth investigating"):
            st.markdown(
                """
                - How much performance variation comes from the harvest cycle versus the individual building?
                - Do temperature or humidity deviations align with mortality where environmental data is complete?
                - How often is weight missing or stale, and how does freshness affect projection error?
                - Which historical patterns represent the largest estimated gross-revenue opportunity under owner-approved assumptions?
                """
            )
        st.caption(
            "Historical evidence snapshot. Source files: FARM HARVEST DATA.xlsx and final average weight only from Farm Performance Summary.xlsx."
        )

if selected_view == VIEW_METHODS:
    recovery_manifest, _ = load_model_bundle("recovery")
    day35_manifest = load_day35_manifest()
    source_daily_snapshots = int(
        recovery_manifest.get(
            "source_daily_snapshot_rows", recovery_manifest["training_snapshot_rows"]
        )
    )
    day35_candidates = day35_manifest.get("candidate_metrics", {})
    st.markdown('<div class="title">Canary Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">See what Canary predicts, which historical outcome was used, how the methods were tested, and where the evidence remains limited.</div>',
        unsafe_allow_html=True,
    )

    rmetrics = recovery_manifest["selected_metrics"]
    wmetrics = day35_manifest["selected_metrics"]
    st.subheader("Executive summary")
    st.markdown(
        """
        **The short version:** Canary keeps three different decision layers separate. The risk score tells us
        where operational concern is highest. The recovery model estimates harvest recovery using a clearly
        disclosed last-recorded recovery proxy. The weight method projects average weight specifically on Day 35.
        """
    )
    executive = st.columns(3)
    with executive[0]:
        with st.container(border=True):
            st.markdown("**1 · Risk score**")
            st.write("Transparent rules, not machine learning")
            st.caption("Four checks total 0–12 points. Use it to decide where to inspect first and open the component scores to see why.")
    with executive[1]:
        with st.container(border=True):
            st.markdown("**2 · Recovery forecast**")
            st.write(f"Average held-out error: **{float(rmetrics['mae']) * 100:.2f} points**")
            st.caption("Typical point error is promising, but target-side accuracy does not beat the always-below-95% baseline. Use the estimate and range—not a classification claim.")
    with executive[2]:
        with st.container(border=True):
            st.markdown("**3 · Day 35 weight outlook**")
            st.write(f"Average held-out error: **{float(wmetrics['mae_kg']) * 1000:.0f} g**")
            st.caption("The age-aware baseline uses the building’s latest measured weight plus historically observed remaining growth to Day 35.")

    st.info(
        "How to present this in one minute: ‘Canary keeps three layers separate: transparent risk rules, two predictive outlooks, and a deterministic action playbook. "
        "The weight output targets 2.0 kg on Day 35. The recovery output targets 95% at harvest, while clearly disclosing that its historical training label is the last recorded population ratio.’"
    )

    st.subheader("Data foundation: how the workbook becomes model-ready")
    foundation_steps = [
        ("1 · Read", "Load Farm Data, Target Weights, and the approved final-weight source without changing them."),
        ("2 · Standardize", "Normalize cycle, building, date, age, population, mortality, feed, weight, and environment fields."),
        ("3 · Consolidate", f"Combine repeated environment rows into one reliable building-day record ({dataset.quality.canonical_rows:,} rows)."),
        ("4 · Validate", "Preserve missing values, reject conflicting duplicates, and attach the correct age-specific target weight."),
        ("5 · Snapshot", "Create leakage-safe as-of records and keep entire cycles together during validation."),
    ]
    for start, width in ((0, 3), (3, 2)):
        foundation = st.columns(width)
        for column, (title, description) in zip(
            foundation, foundation_steps[start : start + width]
        ):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(description)
    st.caption(
        "Key rule: future rows are never allowed into an earlier prediction. Blank mortality, feed, or weight values remain missing—not zero."
    )

    st.subheader("Three decision layers")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Component": "1 · Rules-based risk",
                    "Input": "Current weight, survival, mortality trend, and peer evidence",
                    "Process": "Four transparent 0–3 checks; total 0–12",
                    "Output": "Low / Medium / High / Critical, with the exact why",
                    "Business use": "Choose where to inspect first",
                },
                {
                    "Component": "2 · Predictive outlooks",
                    "Input": "Leakage-safe current flock snapshot plus historical labeled cycles",
                    "Process": "Compare baselines and compact models on completely unseen cycles",
                    "Output": "Predicted recovery and projected Day 35 weight, with ranges",
                    "Business use": "Estimate the likely size of the outcome gap",
                },
                {
                    "Component": "3 · Recommendation playbook",
                    "Input": "Detected problem pattern and rules-based severity",
                    "Process": "Deterministic lookup in the editable farm playbook",
                    "Output": "Inspection focus, urgency, and escalation condition",
                    "Business use": "Turn a warning into a concrete next check",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander("How a daily prediction is produced—in plain language"):
        st.markdown(
            """
            1. Choose a harvest cycle and review date.
            2. Canary freezes the data at that date; later records are excluded.
            3. It creates one current snapshot per building from the available flock records.
            4. Canary estimates harvest recovery and projects average weight on Day 35.
            5. The results are compared with the 95% recovery goal and 2.0 kg Day 35 milestone.
            6. Only the latest cycle receives predictions. Earlier cycles show completed actuals under the capstone's documented last-recorded-date convention.
            """
        )

    st.subheader("What each result can—and cannot—tell us")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Output": "Risk score",
                    "Good for": "Prioritizing inspection and explaining warning signs",
                    "Do not interpret as": "A probability of missing either final goal",
                },
                {
                    "Output": "Predicted recovery",
                    "Good for": "Estimating the likely final survival result",
                    "Do not interpret as": "A guaranteed result or disease diagnosis",
                },
                {
                    "Output": "Projected Day 35 weight",
                    "Good for": "Estimating the Day 35 milestone from the latest building weight",
                    "Do not interpret as": "Final liveweight at an unknown harvest date",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander("When is a result actual versus predicted?"):
        result_logic = pd.DataFrame(
            [
                {
                    "Status": "Current flock",
                    "Canary shows": "Predicted recovery + projected Day 35 weight",
                    "Basis": "Only records available by the review date",
                },
                {
                    "Status": "Earlier cycle",
                    "Canary shows": "Completion date + actual outcomes when available",
                    "Basis": "Ending birds ÷ beginning birds; matched final-weight source",
                },
                {
                    "Status": "No flock data",
                    "Canary shows": "No result",
                    "Basis": "Not calculated",
                },
            ]
        )
        st.dataframe(result_logic, hide_index=True, width="stretch")
        st.caption(
            "For this capstone, cycles before the latest cycle are displayed as completed using each building’s last recorded date. The source does not contain a verified harvest-event flag."
        )

    risk_tab, recovery_tab, weight_tab, action_tab = st.tabs(
        ["1 · Risk scoring", "2A · Recovery model", "2B · Day 35 weight", "3 · Recommendations"]
    )

    with recovery_tab:
        recovery_name = (
            "Ridge regression (without weight inputs)"
            if recovery_manifest["selected_model"] == "ridge_no_weight"
            else recovery_manifest["selected_model"].replace("_", " ").title()
        )
        st.subheader(f"Recovery model: {recovery_name}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Workflow": "Business question", "Plain-language explanation": "Given what is known on the review date, what last-recorded recovery should we expect for this building?"},
                    {"Workflow": "Goal / Y", "Plain-language explanation": "Estimate each building’s recovery at harvest. Historical proxy: population on the last recorded date ÷ beginning population."},
                    {"Workflow": "Inputs / X", "Plain-language explanation": "Age, building group, survival, mortality, feed, and available environment signals known on the review date."},
                    {"Workflow": "Methods tried", "Plain-language explanation": "Historical mean, Ridge regression with weight signals, and Ridge regression without weight signals."},
                    {"Workflow": "Fair comparison", "Plain-language explanation": "Leave one complete cycle out, predict it, and repeat. No daily row from that cycle remains in training."},
                    {"Workflow": "Winner", "Plain-language explanation": f"{recovery_name}; simplest method within 5% of the best cycle-balanced error."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Historical recovery target (Y): population on the last recorded daily date ÷ beginning population. "
            "Canary compared a historical-mean baseline with Ridge regression variants using leave-one-cycle-out validation. "
            f"Training evidence: {recovery_manifest['training_snapshot_rows']:,} balanced decision snapshots from "
            f"{recovery_manifest['training_building_cycles']} building outcomes across {len(recovery_manifest['training_cycles'])} cycles. "
            f"Canary started with {source_daily_snapshots:,} eligible daily snapshots and retained Days 7, 14, 21, 28, and the latest eligible checkpoint."
        )
        with st.expander("See recovery-model preprocessing step by step"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Step": "1 · Standardize", "What happened": "Repeated source rows were consolidated to one building-day; dates, ages, counts, feed, weight, temperature, and humidity were normalized."},
                        {"Step": "2 · Create Y", "What happened": "For each completed building history, last recorded population ÷ beginning population became the recovery proxy label."},
                        {"Step": "3 · Create X snapshots", "What happened": "Each row used only observations available on or before that review date. Future data was excluded."},
                        {"Step": "4 · Balance repeated rows", "What happened": "Each building-cycle contributed Days 7, 14, 21, 28, plus its latest eligible checkpoint so long histories did not dominate."},
                        {"Step": "5 · Handle missing inputs", "What happened": "Training medians filled missing numeric values and missing-value indicators preserved the fact that data was absent."},
                        {"Step": "6 · Validate", "What happened": "One entire harvest cycle was held out at a time. Candidate errors were calculated only on unseen cycles."},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        rcols = st.columns(5)
        rcols[0].metric("Recorded cycles", len(recovery_manifest["training_cycles"]))
        rcols[1].metric("Building outcomes", recovery_manifest["training_building_cycles"])
        rcols[2].metric("MAE", f"{float(rmetrics['mae']) * 100:.2f} pts")
        rcols[3].metric("RMSE", f"{float(rmetrics['rmse']) * 100:.2f} pts")
        rcols[4].metric("Majority baseline", f"{float(rmetrics['majority_side_accuracy']):.1%}")

        st.markdown("**Input features used**")
        selected_recovery_features = set(recovery_manifest["feature_columns"])
        recovery_features = pd.DataFrame(
            [
                {"Feature group": "Flock timing and identity", "Inputs": "Cycle day, beginning inventory, Tags/Lags indicator"},
                {"Feature group": "Survival and mortality", "Inputs": "% alive, daily mortality, recent 3-day mortality, mortality trend"},
                {"Feature group": "Feed", "Inputs": "Daily and cumulative feed per 1,000 birds"},
                {"Feature group": "Weight progress", "Inputs": "Tested, but excluded from the winner because held-out accuracy did not improve"},
                {"Feature group": "Environment", "Inputs": "Recent average temperature and humidity when available"},
            ]
        )
        st.dataframe(recovery_features, hide_index=True, width="stretch")
        st.caption(
            f"Selected model uses {len(selected_recovery_features)} inputs. It intentionally excludes weight-progress inputs because they did not improve held-out MAE, and removes the mathematically redundant cumulative-mortality-rate copy. Missing numeric inputs are filled from the training-data median and marked with missing-value indicators."
        )
        st.markdown("**Which inputs the selected recovery model relies on**")
        global_importance = _global_recovery_importance_table(recovery_manifest)
        if global_importance.empty:
            st.info("Formal feature importance is not available for the selected recovery method.")
        else:
            recorded_top_five = global_importance.loc[
                ~global_importance["Model input"].str.startswith("Missing-data flag:")
            ].head(5)
            st.markdown("**Top five recorded inputs in the fitted recovery model**")
            st.dataframe(recorded_top_five, hide_index=True, width="stretch")
            st.caption(
                "This is the model-wide ranking across historical training data. Open Building View for the factors moving one selected building’s estimate."
            )
            with st.expander("See every recovery-model input, including missing-data flags"):
                st.dataframe(global_importance, hide_index=True, width="stretch")
            importance_records = recovery_manifest["global_feature_importance"]
            missing_reliance = sum(
                float(item["absolute_importance_pct"])
                for item in importance_records
                if str(item["feature"]).startswith("missing__")
            )
            environment_reliance = sum(
                float(item["absolute_importance_pct"])
                for item in importance_records
                if item["feature"] in {"temperature_recent_avg_c", "humidity_recent_avg_pct"}
            )
            top_recorded = next(
                item
                for item in importance_records
                if not str(item["feature"]).startswith("missing__")
            )
            st.info(
                f"Current fitted-model reading: **{FEATURE_DISPLAY.get(top_recorded['feature'], top_recorded['feature'])}** is the strongest recorded input. "
                f"Temperature and humidity values together account for about **{environment_reliance:.1f}%** of coefficient magnitude, while missing-data flags account for **{missing_reliance:.1f}%**. "
                "That is a warning that data completeness itself is entangled with the fitted pattern; Canary should not claim that environment caused the outcome."
            )
            st.caption(
                "Relative reliance is based on the absolute standardized Ridge coefficients. Direction shows whether a higher value pushes the raw estimate up or down after accounting for the other model inputs. Correlated inputs can share or swap importance, and none of these values proves causation."
            )

        st.markdown("**How it was validated**")
        st.write(
            "Canary leaves one complete recorded cycle out, trains on the other cycles, predicts the unseen cycle, and repeats this for every cycle. "
            "Canary first finds the best cycle-balanced MAE, then chooses the simplest method within 5% of that result. "
            "Daily rows from the same cycle never appear in both training and validation."
        )
        st.dataframe(
            _candidate_metrics_table(recovery_manifest, "recovery"),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Selected-model performance by forecast timing**")
        st.dataframe(
            _horizon_metrics_table(recovery_manifest, "recovery"),
            hide_index=True,
            width="stretch",
        )
        st.info(
            f"The selected {recovery_name} model misses last-recorded recovery by about {float(rmetrics['mae']) * 100:.2f} percentage points on average. "
            f"Its likely range uses ±{float(rmetrics['uncertainty_half_width_80']) * 100:.2f} percentage points, "
            "based on the 80th percentile of held-out errors."
        )
        st.success(
            "Business interpretation: use the estimate to size the likely recovery gap and prioritize follow-up. Do not present it as a guaranteed result or as proof that a building will hit 95%."
        )
        st.warning(
            "Straight verdict: this model is useful as a directional point estimate, but it is not yet a trustworthy hit-versus-miss classifier. It recognized historical below-95% cases, but did not recognize the small number of at/above-95% cases better than the majority baseline."
        )
        st.caption(
            "At inference, the final-recovery estimate is capped at current recorded survival because, under the agreed capstone formula, birds already lost cannot re-enter the flock."
        )

        st.subheader("Day 14 prediction versus last-recorded recovery")
        day14_metrics = recovery_manifest["day14_backtest_metrics"]
        d14cols = st.columns(5)
        d14cols[0].metric("Buildings tested", day14_metrics["building_cycles"])
        d14cols[1].metric("Day 14 MAE", f"{float(day14_metrics['mae']) * 100:.2f} pts")
        d14cols[2].metric("Day 14 RMSE", f"{float(day14_metrics['rmse']) * 100:.2f} pts")
        d14cols[3].metric("Average bias", f"{float(day14_metrics['mean_error']) * 100:+.2f} pts")
        d14cols[4].metric("Majority baseline", f"{float(day14_metrics['majority_side_accuracy']):.1%}")
        st.write(
            "For each building history, Canary recreated the forecast using only information available on Day 14. "
            "That entire cycle was excluded from training, and the prediction was compared with the last-recorded recovery proxy."
        )
        st.warning(
            f"Target-side caution: {day14_metrics['actual_below_target']} of {day14_metrics['building_cycles']} outcomes were below 95%, and only "
            f"{day14_metrics['actual_at_or_above_target']} were at or above it. The model predicted all {day14_metrics['predicted_below_target']} below target. "
            f"Its {float(day14_metrics['target_side_accuracy']):.1%} accuracy therefore equals the always-below majority baseline and does not prove it can recognize target hitters."
        )
        day14_table = pd.DataFrame(recovery_manifest["day14_backtest"])
        day14_table = day14_table.assign(
            **{
                "Predicted recovery": day14_table["predicted"].map(lambda value: f"{value:.1%}"),
                "Last-recorded recovery": day14_table["actual"].map(lambda value: f"{value:.1%}"),
                "Error": day14_table["error"].map(lambda value: f"{value * 100:+.2f} pts"),
                "Absolute error": day14_table["absolute_error"].map(
                    lambda value: f"{value * 100:.2f} pts"
                ),
            }
        ).rename(
            columns={
                "cycle_id": "Cycle",
                "building_id": "Building",
                "as_of_date": "Day 14 date",
            }
        )
        st.dataframe(
            day14_table[
                [
                    "Cycle",
                    "Building",
                    "Day 14 date",
                    "Predicted recovery",
                    "Last-recorded recovery",
                    "Error",
                    "Absolute error",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Positive bias means the model predicted recovery slightly higher than the last-recorded result. "
            "This is a historical backtest, not evidence of future guaranteed performance."
        )

    with weight_tab:
        st.subheader("Day 35 weight method: age-aware remaining-growth baseline")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Workflow": "Business question", "Plain-language explanation": "Given the latest measured weight, what average building weight should we expect on Day 35?"},
                    {"Workflow": "Goal / Y", "Plain-language explanation": "Estimate the building’s recorded average liveweight specifically on production Day 35."},
                    {"Workflow": "Inputs / X", "Plain-language explanation": "Latest measured weight, weighing day, target progress, and recent gain when available."},
                    {"Workflow": "Methods tried", "Plain-language explanation": "Historical remaining gain, recent straight-line ADG, and compact Ridge regression."},
                    {"Workflow": "Fair comparison", "Plain-language explanation": "Hold out one complete cycle at a time and compare error in grams on unseen buildings."},
                    {"Workflow": "Winner", "Plain-language explanation": "Age-aware historical remaining gain; transparent and within 5% of the best cycle-balanced result."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Training evidence: {day35_manifest['training_checkpoint_rows']} checkpoint rows from "
            f"{day35_manifest['training_building_cycles']} Day 35 building outcomes across {len(day35_manifest['training_cycles'])} cycles."
        )
        with st.expander("See Day 35 weight preprocessing step by step"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Step": "1 · Create Y", "What happened": "The target was the building's observed average bodyweight on production Day 35 in Farm Harvest Data."},
                        {"Step": "2 · Create checkpoints", "What happened": "Only buildings with a Day 35 measurement were eligible. Their Day 7, 14, 21, and 28 measurements became prediction checkpoints."},
                        {"Step": "3 · Build candidate inputs", "What happened": "Candidates used measurement day, current weight, progress versus the age target, and recent gain when a prior weighing existed."},
                        {"Step": "4 · Keep cycles separate", "What happened": "Every record from one complete cycle was held out together; no building-day from that cycle remained in training."},
                        {"Step": "5 · Compare in grams", "What happened": "MAE, RMSE, bias, within-100 g, within-200 g, and performance by checkpoint were compared."},
                        {"Step": "6 · Select simply", "What happened": "Canary chose the simplest method within 5% of the best cycle-balanced MAE."},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        st.markdown("**Inputs (X) available at the weighing checkpoint**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Input": "Measurement day", "Why it matters": "The amount of growth remaining changes sharply with age"},
                    {"Input": "Latest measured building weight", "Why it matters": "Makes the projection building-specific"},
                    {"Input": "Age-specific target weight", "Why it matters": "Shows whether the flock is ahead of or behind the farm curve"},
                    {"Input": "Previous measured weight / recent ADG", "Why it matters": "Used by the ADG and compact Ridge comparison candidates"},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**What directly drives the selected weight projection**")
        st.dataframe(
            pd.DataFrame(day35_manifest["selected_method_drivers"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(day35_manifest["feature_importance_interpretation"])
        wcols = st.columns(5)
        wcols[0].metric("Cycles", len(day35_manifest["training_cycles"]))
        wcols[1].metric("Day 35 outcomes", day35_manifest["training_building_cycles"])
        wcols[2].metric("MAE", f"{float(wmetrics['mae_kg']) * 1000:.0f} g")
        wcols[3].metric("RMSE", f"{float(wmetrics['rmse_kg']) * 1000:.0f} g")
        wcols[4].metric("Within 200 g", f"{float(wmetrics['within_200g_rate']):.1%}")

        st.dataframe(
            _day35_candidate_metrics_table(day35_manifest),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Performance by decision checkpoint**")
        st.dataframe(
            _day35_horizon_metrics_table(day35_manifest),
            hide_index=True,
            width="stretch",
        )
        st.warning(
            f"All {day35_manifest['actual_target_misses']} historical Day 35 outcomes were below 2.0 kg; there were no target hits. "
            "Canary can evaluate error in grams, but it cannot yet prove that this method distinguishes Day 35 target hitters from misses."
        )
        st.success(
            "Business interpretation: use this as an age-aware estimate of the Day 35 gap. The selected method responds to each building’s latest measured weight instead of giving every building the same average."
        )
        st.subheader("Why this is stronger than straight-line ADG")
        st.markdown(
            f"""
            A straight-line ADG projection assumes that the growth rate observed early in life continues unchanged.
            In the historical holdout test it had an MAE of **{day35_candidates.get('recent_linear_adg', wmetrics)['mae_kg'] * 1000:.0f} g**.
            The selected historical remaining-gain method reduced that to **{wmetrics['mae_kg'] * 1000:.0f} g**.

            Canary also trained a compact Ridge regression using age, current weight, progress against the target curve,
            and recent ADG. Ridge reached **{day35_candidates.get('ridge_regression', wmetrics)['mae_kg'] * 1000:.0f} g MAE**—close, but not better.
            Ridge had the slightly lowest cycle-balanced MAE, but the difference was only about 1%.
            Under Canary's simple-winner rule, the transparent remaining-gain method remains the winner because it is within 5% of the best cycle-balanced result, has lower overall row-level MAE, and is easier to explain.
            """
        )
        st.subheader("Day 14 projection versus recorded Day 35 weight")
        weight_day14_metrics = day35_manifest["day14_backtest_metrics"]
        weight_d14_cols = st.columns(4)
        weight_d14_cols[0].metric("Buildings tested", weight_day14_metrics["building_cycles"])
        weight_d14_cols[1].metric("Day 14 MAE", f"{float(weight_day14_metrics['mae_kg']) * 1000:.0f} g")
        weight_d14_cols[2].metric("Day 14 RMSE", f"{float(weight_day14_metrics['rmse_kg']) * 1000:.0f} g")
        weight_d14_cols[3].metric("Average bias", f"{float(weight_day14_metrics['mean_error_kg']) * 1000:+.0f} g")
        weight_day14 = pd.DataFrame(day35_manifest["day14_backtest"])
        weight_day14_view = weight_day14.assign(
            **{
                "Day 14 measured weight": weight_day14["current_weight_kg"].map(lambda value: f"{value * 1000:.0f} g"),
                "Projected Day 35": weight_day14["predicted_day35_weight_kg"].map(lambda value: f"{value * 1000:.0f} g"),
                "Recorded Day 35": weight_day14["actual_day35_weight_kg"].map(lambda value: f"{value * 1000:.0f} g"),
                "Prediction error": weight_day14["error_kg"].map(lambda value: f"{value * 1000:+.0f} g"),
            }
        ).rename(columns={"cycle_id": "Cycle", "building_id": "Building"})
        st.dataframe(
            weight_day14_view[
                ["Cycle", "Building", "Day 14 measured weight", "Projected Day 35", "Recorded Day 35", "Prediction error"]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Positive error means the Day 14 projection was too high; negative error means it was too low. Every comparison held out the complete cycle."
        )

    with risk_tab:
        st.warning(
            f"Risk thresholds: {rules['approval_status']}. Farm approval is still required before operational use."
        )
        st.subheader("Day 35 and harvest are different checkpoints")
        st.markdown(
            """
            - **Day 35 milestone:** the flock should weigh at least **2.0 kg** on average.
            - **Weight outlook:** projects average weight specifically on Day 35; it is not a final-harvest weight forecast.
            - **Completed-cycle convention:** cycles before the latest are shown as completed on each building's last recorded date. This is a documented capstone convention—not a verified harvest-event flag.
            """
        )
        st.subheader("What the risk score means")
        st.markdown(
            """
            Canary scores four warning signs from 0 to 3: weight versus the age target,
            survival versus the age path, recent mortality trend, and performance versus peer buildings.

            - **Low:** 0–1
            - **Medium:** 2–3
            - **High:** 4–5
            - **Critical:** 6–12

            Missing evidence is shown as missing and is never silently scored as zero.
            """
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Check": "Growth progress",
                        "Calculation": "Shortfall from the target for the actual weighing day",
                        "Operational meaning": "Is measured weight following the farm growth curve?",
                    },
                    {
                        "Check": "Survival position",
                        "Calculation": "Shortfall from the assumed path to 95% by Day 35",
                        "Operational meaning": "How much cumulative survival loss is already visible?",
                    },
                    {
                        "Check": "Mortality momentum",
                        "Calculation": "Recent 3-day rate minus the preceding baseline",
                        "Operational meaning": "Are losses accelerating now?",
                    },
                    {
                        "Check": "Peer context",
                        "Calculation": "Worst gap versus the median of similar-age buildings",
                        "Operational meaning": "Is the concern building-specific rather than shared?",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.warning(
            "Validation limit: this is an expert-rule operational priority score, not a trained outcome model. "
            "In a preliminary audit of 31 historical Day 14 building snapshots, the score bands did not show a consistent step-by-step ordering in either last-recorded recovery or recorded Day 35 weight. "
            "Canary therefore does not claim that the risk label predicts the final target result; the two separate forecast layers estimate outcomes."
        )
        st.info(
            "Design review: keep all four checks for now because they answer different operational questions and match the agreed scope. "
            "Survival and mortality are related, while peer context may repeat the same underlying signal. Their thresholds and additive points remain provisional until farm approval and calibration."
        )
        st.markdown("**Straight recommendation for the next scoring version**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Decision": "Keep now", "Recommendation": "Retain weight, survival, mortality trend, and peer context for the capstone because each is traceable and matches the agreed scope."},
                    {"Decision": "Validate next", "Recommendation": "Add an absolute daily-mortality guardrail so a persistently high level cannot look safe merely because it is no longer rising."},
                    {"Decision": "Validate next", "Recommendation": "Test whether peer context should add all 0–3 points or act as a smaller modifier, because it can repeat the same weight, survival, or mortality signal."},
                    {"Decision": "Do not add yet", "Recommendation": "Do not put temperature, humidity, THI, feed, or water directly into the score until their age-based thresholds and units are approved."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Canary already shows environment and intake evidence as a separate possible-cause layer. That separation keeps the risk label stable while still giving management a practical place to investigate."
        )
        threshold_table = pd.DataFrame(rules["age_bands"]).rename(
            columns={
                "label": "Production age",
                "weight_gap_pct": "Weight-gap cutoffs (%)",
                "survival_gap_pp": "Survival-gap cutoffs (points)",
                "mortality_trend_delta_per_1000": "Mortality-trend cutoffs (/1,000)",
            }
        )
        st.dataframe(
            threshold_table[
                [
                    "Production age",
                    "Weight-gap cutoffs (%)",
                    "Survival-gap cutoffs (points)",
                    "Mortality-trend cutoffs (/1,000)",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Peer-context thresholds**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Peer measure": "Weight-gap excess (%)",
                        "Cutoffs for 0 / 1 / 2 points": rules["peer_comparison"]["weight_gap_excess_pct"],
                    },
                    {
                        "Peer measure": "Survival-gap excess (points)",
                        "Cutoffs for 0 / 1 / 2 points": rules["peer_comparison"]["survival_gap_excess_pp"],
                    },
                    {
                        "Peer measure": "Recent mortality excess (/1,000)",
                        "Cutoffs for 0 / 1 / 2 points": rules["peer_comparison"]["mortality_rate_excess_per_1000"],
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "For each cutoff list: at or below the first value = 0 points; above the first through second = 1; above second through third = 2; and above third = 3. "
            "The Farmer Validation Workbook supports several separate alert values, but it does not yet approve this complete point matrix."
        )
        if st.button("Review or edit risk thresholds", key="open_risk_rule_admin"):
            st.switch_page(PAGE_CHECKS)

    with action_tab:
        st.subheader("From recorded signal to next action")
        st.markdown(
            "Canary first looks for a specific, current operating alert supported by recorded evidence. If one exists, that alert leads the owner-facing action. If none exists, Canary labels the operating cause as unconfirmed and falls back to the broader problem-pattern playbook. It never asks a model to invent a treatment."
        )
        action_method = pd.DataFrame(recommendation_playbook["rules"])[
            ["pattern", "dashboard_action", "approval_status"]
        ].rename(
            columns={
                "pattern": "Problem pattern",
                "dashboard_action": "Recommended management focus",
                "approval_status": "Validation status",
            }
        )
        st.dataframe(action_method, hide_index=True, width="stretch")
        st.info(
            "Use the recommendation as inspection and management guidance—not disease diagnosis, automatic treatment, or a guarantee that either target will be reached."
        )
        st.subheader("How Canary handles temperature, humidity, feed, water, and heat stress")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Step": "1 · Detect",
                        "What Canary does": "Compares recorded temperature, humidity, daily feed per bird, and mortality with age-aware provisional limits.",
                        "Owner decision": "See the actual value, target or range, gap, reading day, and alert severity.",
                    },
                    {
                        "Step": "2 · Connect",
                        "What Canary does": "Uses the strongest current operating alert as the card's most actionable recorded signal.",
                        "Owner decision": "Prioritize a practical check such as sensors, fans, inlets, curtains, cooling pads, heaters, litter, feeder access, or water availability.",
                    },
                    {
                        "Step": "3 · Protect against overclaiming",
                        "What Canary does": "Labels thresholds as provisional and conditions as actionable leads—not proven causes of mortality or growth loss.",
                        "Owner decision": "Verify the sensor, bird behavior, and equipment response; move toward the approved range under the farm SOP.",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.warning(
            "Capstone boundary: these operating alerts do not add formal risk points yet. Water is not available in the standardized workbook, the Daily FI/bird unit and targets need confirmation, and THI awaits one approved poultry-specific formula and age-specific bands. When the current measurement is absent, Canary must say the cause is not confirmed."
        )

    with st.expander("What the performance metrics mean"):
        st.markdown(
            """
            - **MAE:** average size of the prediction error. Lower is better.
            - **RMSE:** penalizes large mistakes more strongly than MAE. Lower is better.
            - **Cycle-to-cycle MAE variability:** how much accuracy changes across held-out cycles. Lower is more stable.
            - **Below-goal recall:** among outcomes that finished below the goal, how many Canary warned about.
            - **At/above-goal recall:** among outcomes that reached the goal, how many Canary correctly recognized.
            - **Majority baseline:** the accuracy obtained by always choosing the historically more common side of the goal.
            - **Likely range:** point prediction plus or minus the 80th percentile of held-out absolute errors. It is a prototype range, not a guarantee.
            """
        )
    st.caption(
        f"Recovery model: {recovery_manifest['model_version']} · Day 35 weight method: {day35_manifest['model_version']} · "
        f"Risk configuration: {DEFAULT_RULES_PATH.name} · Source: {dataset.source_name}"
    )
