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
    DAY35_TARGET_KG,
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
    if configured:
        return Path(configured).expanduser()
    app_root = Path(__file__).resolve().parent
    bundled = app_root / "data" / "FARM HARVEST DATA.xlsx"
    return bundled if bundled.exists() else app_root.parent / "FARM HARVEST DATA.xlsx"


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
        float(row["day35_weight_target_gap_kg"]) / DAY35_TARGET_KG * 100
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
    candidate_names = {
        "trend_naive": "Current-survival projection",
        "historical_mean": "Historical mean",
        "linear_regression": "Ordinary linear regression",
        "ridge_core": "Compact Ridge",
        "ridge_no_weight": "Ridge without weight",
        "ridge": "Ridge with all tested inputs",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
    }
    registry = manifest.get("candidate_registry") or [
        {"model": candidate, "available": True, "reason": "Evaluated"}
        for candidate in manifest["metrics"]
    ]
    for entry in registry:
        candidate = entry["model"]
        metrics = manifest["metrics"].get(candidate)
        if metrics is None:
            rows.append(
                {
                    "Candidate": candidate_names.get(candidate, candidate.replace("_", " ").title()),
                    "Role": "Unavailable locally",
                    f"MAE ({unit})": np.nan,
                    f"Cycle-balanced MAE ({unit})": np.nan,
                    f"RMSE ({unit})": np.nan,
                    "R²": np.nan,
                    f"Bias ({unit})": np.nan,
                    "Cycle-to-cycle MAE variability": np.nan,
                    "Below-goal recall": "—",
                    "At/above-goal recall": "—",
                }
            )
            continue
        rows.append(
            {
                "Candidate": candidate_names.get(candidate, candidate.replace("_", " ").title()),
                "Role": (
                    "Operational"
                    if candidate == manifest["selected_model"]
                    else "Best learned challenger"
                    if candidate == manifest.get("research_champion")
                    else "Compared"
                ),
                f"MAE ({unit})": round(float(metrics["mae"]) * factor, 3),
                f"Cycle-balanced MAE ({unit})": round(
                    float(metrics.get("cycle_macro_mae", metrics["mae"])) * factor, 3
                ),
                f"RMSE ({unit})": round(float(metrics["rmse"]) * factor, 3),
                "R²": round(float(metrics.get("r2", float("nan"))), 3),
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
    rows = []
    registry = manifest.get("candidate_registry") or [
        {"model": candidate, "available": True}
        for candidate in manifest["candidate_metrics"]
    ]
    for entry in registry:
        candidate = entry["model"]
        metrics = manifest["candidate_metrics"].get(candidate)
        if metrics is None:
            rows.append(
                {
                    "Method": candidate.replace("_", " ").title(),
                    "Role": "Unavailable locally",
                    "MAE": "—",
                    "Cycle-balanced MAE": "—",
                    "RMSE": "—",
                    "R²": "—",
                    "Bias": "—",
                    "Within 200 g": "—",
                    "Correct side of 1.8 kg": "—",
                }
            )
            continue
        rows.append(
            {
                "Method": candidate.replace("_", " ").title(),
                "Role": (
                    "Operational fallback"
                    if candidate == manifest["selected_model"]
                    else "Best learned challenger"
                    if candidate == manifest.get("research_champion")
                    else "Compared"
                ),
                "MAE": f"{float(metrics['mae_kg']) * 1000:.0f} g",
                "Cycle-balanced MAE": f"{float(metrics.get('cycle_macro_mae_kg', metrics['mae_kg'])) * 1000:.0f} g",
                "RMSE": f"{float(metrics['rmse_kg']) * 1000:.0f} g",
                "R²": f"{float(metrics.get('r2', float('nan'))):.3f}",
                "Bias": f"{float(metrics['bias_kg']) * 1000:+.0f} g",
                "Within 200 g": f"{float(metrics['within_200g_rate']):.1%}",
                "Correct side of 1.8 kg": f"{float(metrics['target_side_accuracy']):.1%}",
            }
        )
    return pd.DataFrame(rows)


def _recovery_comparison_summary(manifest: dict[str, object]) -> pd.DataFrame:
    """Return standard validation metrics for at most five recovery candidates."""

    detailed = _candidate_metrics_table(manifest, "recovery")
    return detailed.rename(
        columns={
            "Candidate": "Model",
            "MAE (percentage points)": "MAE",
            "Cycle-balanced MAE (percentage points)": "Cycle MAE",
            "RMSE (percentage points)": "RMSE",
        }
    )[["Model", "Role", "MAE", "Cycle MAE", "RMSE", "R²"]]


def _day35_comparison_summary(manifest: dict[str, object]) -> pd.DataFrame:
    """Return standard validation metrics for at most five weight candidates."""

    detailed = _day35_candidate_metrics_table(manifest)
    return detailed.rename(
        columns={
            "Method": "Model",
            "Cycle-balanced MAE": "Cycle MAE",
        }
    )[["Model", "Role", "MAE", "Cycle MAE", "RMSE", "R²", "Within 200 g"]]


def _day35_horizon_metrics_table(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Forecast made on": horizon,
                "Buildings": int(metrics["rows"]),
                "MAE": f"{float(metrics['mae_kg']) * 1000:.0f} g",
                "RMSE": f"{float(metrics['rmse_kg']) * 1000:.0f} g",
                "Within 200 g": f"{float(metrics['within_200g_rate']):.1%}",
                "Correct side of 1.8 kg": f"{float(metrics['target_side_accuracy']):.1%}",
            }
            for horizon, metrics in manifest["selected_metrics"]["horizon"].items()
        ]
    )


def _recovery_cycle_metrics_table(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Held-out cycle": cycle,
                "Snapshots": int(metrics["rows"]),
                "MAE": f"{float(metrics['mae']) * 100:.2f} pts",
                "RMSE": f"{float(metrics['rmse']) * 100:.2f} pts",
                "Bias": f"{float(metrics['bias']) * 100:+.2f} pts",
            }
            for cycle, metrics in manifest["selected_metrics"].get("cycle", {}).items()
        ]
    )


def _day35_cycle_metrics_table(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Held-out cycle": cycle,
                "Checkpoint rows": int(metrics["rows"]),
                "MAE": f"{float(metrics['mae_kg']) * 1000:.0f} g",
                "RMSE": f"{float(metrics['rmse_kg']) * 1000:.0f} g",
                "Bias": f"{float(metrics['bias_kg']) * 1000:+.0f} g",
            }
            for cycle, metrics in manifest["selected_metrics"].get("cycle", {}).items()
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
    "weight_gap_pct": "Weight gap versus age target",
    "weight_staleness_days": "Days since last weighing",
    "temperature_deviation_from_band_c": "Temperature outside approved band",
    "humidity_deviation_from_band_pp": "Humidity outside approved band",
    "environment_out_of_band_days_7d": "Recent environment days outside band",
    "environment_staleness_days": "Days since environment reading",
    "is_lags_building": "Lagundi building indicator",
}


def _global_recovery_importance_table(manifest: dict[str, object]) -> pd.DataFrame:
    rows = []
    coefficient_records = manifest.get("global_feature_importance", [])
    if not coefficient_records:
        for item in manifest.get("held_out_permutation_importance", []):
            feature = str(item["feature"])
            rows.append(
                {
                    "Model input": FEATURE_DISPLAY.get(
                        feature, feature.replace("_", " ").title()
                    ),
                    "Relative reliance": f"{float(item['relative_importance_pct']):.1f}%",
                    "When this input is higher": "Direction depends on the fitted relationship",
                    "Held-out MAE increase": f"{float(item['mean_mae_increase']) * 100:.3f} recovery pts",
                }
            )
        return pd.DataFrame(rows)
    for item in coefficient_records:
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


def _age_band(age: int, ranges: list[dict[str, object]]) -> tuple[float, float]:
    """Return the configured lower and upper band for one production age."""

    for band in ranges:
        if int(band["minimum_age"]) <= int(age) <= int(band["maximum_age"]):
            return float(band["minimum"]), float(band["maximum"])
    return float("nan"), float("nan")


def _eda_environment_profile(
    dataset, latest_cycle: str, rules: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Profile historical environment coverage against the provisional bands."""

    daily = dataset.daily.loc[
        (dataset.daily["cycle_id"] != latest_cycle)
        & dataset.daily["age_day"].between(1, 35),
        [
            "cycle_id",
            "building_id",
            "age_day",
            "temperature_avg_c",
            "humidity_avg_pct",
            "mortality_daily",
            "beginning_inventory",
        ],
    ].copy()
    temperature_bands = daily["age_day"].map(
        lambda age: _age_band(int(age), rules["temperature_ranges_c"])
    )
    humidity_bands = daily["age_day"].map(
        lambda age: _age_band(int(age), rules["humidity_ranges_pct"])
    )
    daily["Temperature minimum"] = [band[0] for band in temperature_bands]
    daily["Temperature maximum"] = [band[1] for band in temperature_bands]
    daily["Humidity minimum"] = [band[0] for band in humidity_bands]
    daily["Humidity maximum"] = [band[1] for band in humidity_bands]
    daily["Temperature outside band"] = daily["temperature_avg_c"].notna() & ~daily[
        "temperature_avg_c"
    ].between(daily["Temperature minimum"], daily["Temperature maximum"])
    daily["Humidity outside band"] = daily["humidity_avg_pct"].notna() & ~daily[
        "humidity_avg_pct"
    ].between(daily["Humidity minimum"], daily["Humidity maximum"])
    daily["Environment outside band"] = (
        daily["Temperature outside band"] | daily["Humidity outside band"]
    )
    environment_available = daily[["temperature_avg_c", "humidity_avg_pct"]].notna().any(axis=1)
    comparable = daily.loc[environment_available].copy()
    profile = (
        comparable.groupby("age_day", as_index=False)
        .agg(
            **{
                "Recorded temperature": ("temperature_avg_c", "mean"),
                "Temperature minimum": ("Temperature minimum", "first"),
                "Temperature maximum": ("Temperature maximum", "first"),
                "Recorded humidity": ("humidity_avg_pct", "mean"),
                "Humidity minimum": ("Humidity minimum", "first"),
                "Humidity maximum": ("Humidity maximum", "first"),
            }
        )
        .set_index("age_day")
    )
    mortality_comparable = comparable.loc[
        comparable["mortality_daily"].notna()
        & comparable["beginning_inventory"].gt(0)
    ].copy()
    mortality_comparable["Mortality per 1,000"] = (
        mortality_comparable["mortality_daily"]
        / mortality_comparable["beginning_inventory"]
        * 1000
    )
    outside_rate = (
        float(comparable["Environment outside band"].mean()) if not comparable.empty else float("nan")
    )
    return profile, {
        "eligible_rows": float(len(daily)),
        "environment_rows": float(len(comparable)),
        "coverage": float(environment_available.mean()) if len(daily) else float("nan"),
        "outside_rate": outside_rate,
        "within_rows": float((~comparable["Environment outside band"]).sum()),
        "outside_rows": float(comparable["Environment outside band"].sum()),
        "mortality_rows": float(len(mortality_comparable)),
        "building_cycles": float(
            comparable[["cycle_id", "building_id"]].drop_duplicates().shape[0]
        ),
    }


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
    if observed >= DAY35_TARGET_KG:
        return "Achieved", f"Recorded Day 35 weight was {observed:.3f} kg, meeting the 1.8 kg milestone."
    return "Missed", f"Recorded Day 35 weight was {observed:.3f} kg, below the 1.8 kg milestone."


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
        "population_loss": row.get("population_loss_score", pd.NA),
        "daily_mortality": row.get("daily_mortality_score", pd.NA),
        "environment": row.get("environment_score", pd.NA),
    }
    available = {name: float(value) for name, value in scores.items() if pd.notna(value)}
    if not available or max(available.values()) <= 0:
        return "No material warning signal is above the current rule thresholds."
    leading = max(available, key=available.get)
    if leading == "weight" and pd.notna(row.get("weight_gap_pct")):
        return f"Weight is {abs(float(row['weight_gap_pct'])):.1f}% below its age-specific target."
    if leading == "population_loss" and pd.notna(row.get("population_loss_pct")):
        return f"Population loss is {float(row['population_loss_pct']):.1f}% of beginning birds."
    if leading == "daily_mortality" and pd.notna(row.get("daily_mortality_pct")):
        return f"Latest daily mortality is {float(row['daily_mortality_pct']):.2f}% of beginning birds."
    if leading == "environment":
        driver = str(row.get("environment_driver", "Environmental condition"))
        return f"{driver} is outside the current provisional rule."
    return str(row.get("risk_pattern", "Recorded warning signal"))


PATTERN_DISPLAY = {
    "Low Body Weight": (
        "Weight behind target",
        "Measured weight is behind the farm target for this age.",
    ),
    "High Mortality": ("High daily mortality", "The latest mortality rate exceeds the current limit."),
    "Rapid Population Loss": ("Population loss", "The surviving population has fallen beyond the current limit."),
    "Abnormal Temperature Fluctuation": ("Large temperature swing", "The daily maximum-to-minimum temperature range is above the current limit."),
    "High Temperature": ("Temperature above the age range", "The latest average temperature is above the tropical operating range for this flock age."),
    "Low Temperature": ("Temperature below the age range", "The latest average temperature is below the tropical operating range for this flock age."),
    "High Humidity": ("Humidity above range", "Recorded humidity is above the current age-specific range."),
    "Low Humidity": ("Humidity below range", "Recorded humidity is below the current age-specific range."),
    "No Material Concern": (
        "No material concern",
        "No scored warning sign is above the current thresholds.",
    ),
    "Missing or Stale Evidence": ("Evidence needs updating", "One or more required measurements is missing or stale."),
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
        risk_score = row.get("risk_score", pd.NA)
        if pd.notna(risk_score) and float(risk_score) > 0:
            display_title, _ = _pattern_display(row.get("risk_pattern"))
            output.at[index, "owner_reason_title"] = display_title
            output.at[index, "owner_reason_detail"] = _card_driver(row)
            output.at[index, "owner_action"] = str(row.get("recommended_action", "Inspect the leading warning signal."))
            output.at[index, "owner_action_basis"] = (
                f"Risk rule {row.get('risk_rule_version', 'unknown')} · action rule {row.get('recommendation_rule_id', 'unknown')}"
            )
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
    evidence_details: list[str] = []
    if pd.isna(row.get("weight_score")):
        evidence_details.append("Weight not scored: no usable measured weight")
    if pd.isna(row.get("environment_score")):
        evidence_details.append(f"Environment not scored: {row.get('environment_status', 'no current reading')}")
    evidence_detail = (
        f'<div class="evidence-note">{html.escape(" · ".join(evidence_details))}</div>'
        if evidence_details
        else ""
    )
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
      {evidence_detail}
      <div class="driver"><span class="pattern-title">{html.escape(pattern_title)}</span><span class="pattern-subtitle">{html.escape(pattern_subtitle)}<br><strong>Why now:</strong> {html.escape(driver)}</span></div>
      <div class="outcome-stack">
        <div class="outcome-row"><div class="outcome-name">Harvest recovery</div><div class="outcome-detail"><div class="outcome-flow"><span>Current recorded: {_percent(current_recovery)}</span><span class="outcome-arrow">→</span><strong>Projected: {_percent(predicted_recovery)}</strong></div><span class="gap-tag {recovery_class}">{html.escape(recovery_gap)} · harvest goal 95%</span></div></div>
        <div class="outcome-row"><div class="outcome-name">Average weight (g)</div><div class="outcome-detail"><div class="outcome-flow"><span>Latest: {html.escape(current_weight_text)}</span><span class="outcome-arrow">→</span><strong>Projected Day 35: {weight_value}</strong></div><span class="gap-tag {weight_class}">{html.escape(weight_gap)} · Day 35 goal 1,800 g</span></div></div>
      </div>
      <div class="value-strip"><div class="label">Gross revenue at risk to 95%</div><strong>{_php(revenue_at_risk)}</strong></div>
      <div class="action"><div class="label">Next action · {html.escape(str(row['recommendation_urgency']))}</div>{html.escape(owner_action)}</div>
    </div>
    """


with st.sidebar:
    st.header("Choose what to review")
    uploaded = st.file_uploader(
        "Update daily farm data (optional)",
        type=["xlsx"],
        help="The app starts with its bundled capstone data. Upload a newer standardized FARM HARVEST DATA.xlsx to replace it for this session and recalculate the dashboard.",
    )
    uploaded_performance = st.file_uploader(
        "Update final-weight data (optional)",
        type=["xlsx"],
        help="The bundled final-weight summary is used by default. Upload a newer Farm Performance Summary.xlsx to replace it for this session. Canary reads only its final average-weight field.",
    )

try:
    if uploaded is not None:
        dataset = _load_upload(uploaded.getvalue(), uploaded.name)
        source_description = f"Uploaded: {uploaded.name}"
    else:
        default_path = _default_workbook()
        if not default_path.exists():
            st.info("No bundled farm data was found. Upload FARM HARVEST DATA.xlsx to begin.")
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
        st.caption(
            "Review date means: ‘What would Canary have shown using only records available by this date?’ "
            "Moving it backward replays an earlier decision point; risk scores, forecasts, and actions are recomputed, and later records are excluded."
        )

rules = load_risk_rules()
recommendation_playbook = load_recommendation_playbook()
environment_columns = ["temperature_min_c", "temperature_max_c", "humidity_avg_pct"]
environment_denominator = int(dataset.daily["operational_recorded"].sum())
environment_direct_rows = int(
    dataset.daily.loc[dataset.daily["operational_recorded"], environment_columns]
    .notna()
    .any(axis=1)
    .sum()
)
environment_direct_coverage_pct = (
    environment_direct_rows / environment_denominator * 100
    if environment_denominator
    else 0.0
)
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
          <p>Project Canary is an early-warning and decision-support system for broiler farms. It identifies buildings at risk of missing the 1,800 g Day 35 and 95% harvest-recovery targets, projects both outcomes from the latest available data, explains why a building was flagged, and recommends what management should check next.</p>
        </div>
        <div class="intro-grid">
          <div class="intro-panel"><span class="intro-kicker">The management problem</span><strong>Daily records do not clearly show which building needs attention first.</strong><span>Weight, survival, mortality, feed, and environmental readings are spread across rows and dates, so the six buildings must be compared manually.</span></div>
          <div class="intro-panel solution"><span class="intro-kicker">What Canary does</span><strong>Canary turns the latest records into one daily decision view.</strong><span>It scores operational risk, projects Day 35 weight and harvest recovery, shows each gap to target, and recommends the next inspection.</span></div>
        </div>
        <div class="decision-question">
          <div class="decision-icon">?</div>
          <div><span class="decision-kicker">The business question</span><strong>Which buildings are at risk of missing the 1,800 g Day 35 or 95% harvest-recovery goals, what results are currently projected, why are they at risk, and what should management do next?</strong><div class="decision-goals"><span class="goal-chip">1,800 g average weight by Day 35</span><span class="goal-chip">95% recovery at harvest</span></div></div>
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
        recovery_portfolio = historical_recorded.loc[
            historical_recorded["actual_harvest_recovery"].notna()
            & historical_recorded["beginning_inventory"].notna()
            & (historical_recorded["beginning_inventory"] > 0)
        ]
        if recovery_portfolio.empty:
            portfolio_actual_recovery = pd.NA
            portfolio_actual_note = "No beginning/ending population pair is available"
        else:
            portfolio_actual_recovery = float(
                (recovery_portfolio["actual_harvest_recovery"] * recovery_portfolio["beginning_inventory"]).sum()
                / recovery_portfolio["beginning_inventory"].sum()
            )
            portfolio_actual_gap = (portfolio_actual_recovery - 0.95) * 100
            portfolio_actual_note = (
                f"{abs(portfolio_actual_gap):.1f} pts {'above' if portfolio_actual_gap >= 0 else 'below'} 95% · "
                f"{len(recovery_portfolio)} building(s), inventory-weighted"
            )
        summary[1].metric(
            "Final harvest recovery",
            _percent(portfolio_actual_recovery),
            portfolio_actual_note,
            help="Across the cycle: total estimated ending birds divided by total beginning birds for completed buildings with both values available.",
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
        else '<div class="subtitle">See which buildings are at risk of missing the 1,800 g Day 35 or 95% harvest-recovery goals, why they were flagged, and what management should check next.</div>',
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
        help="Estimated average liveweight on production Day 35, compared with the 1.8 kg milestone.",
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
    )[["Dimension", "What Canary observed", "Calculation", "Points", "Rule applied", "Data status"]].copy()
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
        "The risk rating is an operational priority score, not the probability of missing the 95% or 1.8 kg goals."
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
            weight_delta = None if pd.isna(building["day35_weight_target_gap_kg"]) else f"{float(building['day35_weight_target_gap_kg']) * 1000:+.0f} g vs 1.8 kg milestone"
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
        "Recovery is compared with 95%. Weight is projected specifically to Day 35 and compared with 1.8 kg."
    )

    recovery_contributions = recovery_feature_contributions(
        dataset, selected_cycle, chosen, pd.Timestamp(selected_date)
    )
    st.markdown("### Understand how each forecast was built")
    st.caption(
        "Choose an outcome below. Each model follows the same six-part explanation so the owner and panel can trace the question, data, method, result, and limitation."
    )
    recovery_model_tab, weight_model_tab = st.tabs(
        ["Harvest Recovery Model", "Day 35 Average Weight Model"]
    )

    with recovery_model_tab:
        st.markdown("### A. Harvest Recovery Model")
        st.caption("Business question: using only records available today, what final recovery level should we expect against the 95% goal?")

        st.markdown("#### B. Executive Summary")
        with st.container(border=True):
            summary_cols = st.columns(4)
            summary_cols[0].metric("This building", _percent(building["predicted_final_recovery"]))
            summary_cols[1].metric("Goal", "95.0%")
            summary_cols[2].metric(
                "Typical error",
                f"{float(recovery_manifest['selected_metrics']['mae']) * 100:.2f} pts",
                help="Mean absolute error on complete harvest cycles that were excluded from training.",
            )
            summary_cols[3].metric("95% hit/miss test", "Not validated")
            st.write(
                "Canary uses **ordinary linear regression** for the continuous recovery estimate because it reduced cycle-balanced error by "
                f"**{float(recovery_manifest['champion_gates']['baseline_improvement_pct']):.1f}%** versus the historical-mean baseline on completely unseen cycles. "
                "It does **not** reliably classify whether a flock will finish above or below 95%."
            )

        st.markdown("#### C. Input and Output Variables")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Role": "Output (Y)", "Variable": "Final recovery proxy", "Plain-language definition": "Last-recorded population ÷ beginning population for a completed historical building-cycle."},
                    {"Role": "Inputs (X)", "Variable": "Age and current survival", "Plain-language definition": "Production day and percentage of beginning birds still recorded alive as of the review date."},
                    {"Role": "Inputs (X)", "Variable": "Mortality signals", "Plain-language definition": "Recent three-day mortality and whether mortality is accelerating or improving."},
                    {"Role": "Inputs (X)", "Variable": "Growth evidence", "Plain-language definition": "Latest weight gap versus the age target and days since the last weighing."},
                    {"Role": "Inputs (X)", "Variable": "Environment evidence", "Plain-language definition": "Temperature/humidity deviation, recent days outside the approved bands, and reading freshness."},
                    {"Role": "Excluded", "Variable": "Future outcomes and unconfirmed feed", "Plain-language definition": "Future ending population, future weights, building identity, and feed are not used; feed waits for unit confirmation."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("See the exact ten recovery-model inputs"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Technical input": recovery_manifest["feature_columns"],
                        "Business meaning": [
                            FEATURE_DISPLAY.get(item, item.replace("_", " ").title())
                            for item in recovery_manifest["feature_columns"]
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.markdown("#### D. Pre-processing Steps")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Stage": "1 · Data cleaning", "What happened": "Standardized cycle, building, date and units; combined Zone A/B environment rows; reduced 1,785 source rows to 1,666 unique building-days; preserved missing values instead of treating them as zero."},
                    {"Stage": "2 · Label and snapshot creation", "What happened": "Created 25 completed building-cycle Y labels and 122 Day 7/14/21/28/latest decision snapshots. Every X value was frozen as of that snapshot date."},
                    {"Stage": "3 · Feature engineering", "What happened": "Created current survival, recent mortality, mortality change, age-target weight gap, days since weighing, environmental deviation, recent out-of-band days and reading freshness."},
                    {"Stage": "4 · Fold-only preparation", "What happened": "Within each training fold, numeric gaps were median-imputed, missingness flags were added, and linear-model inputs were standardized. The held-out cycle never set these values."},
                    {"Stage": "5 · Train/test split", "What happened": "No random 80/20 row split was used. The outer test fold removed one complete harvest cycle, trained on all remaining cycles, predicted the unseen cycle, and repeated for every cycle."},
                    {"Stage": "6 · Cross-validation and tuning", "What happened": "An inner whole-cycle cross-validation loop tuned candidate settings using training cycles only. Snapshots were weighted so every building-cycle had equal total influence."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown("#### E. Model Selection and Comparison")
        st.caption("MAE is the typical absolute miss. Cycle MAE gives every harvest cycle equal weight. RMSE penalizes larger misses. R² measures held-out variance explained. Lower MAE/RMSE and higher R² are better.")
        st.dataframe(
            _recovery_comparison_summary(recovery_manifest),
            hide_index=True,
            width="stretch",
        )
        recovery_gates = recovery_manifest["champion_gates"]
        st.info(
            f"**Selected:** Ordinary linear regression. It passed the continuous-estimate gates with positive R² and a {float(recovery_gates['baseline_improvement_pct']):.1f}% cycle-MAE improvement. "
            "It failed the separate 95% classification gate, so the dashboard shows a continuous estimate and range—not a probability of success."
        )
        if recovery_manifest.get("unavailable_candidates"):
            st.caption("XGBoost is the fifth declared challenger. No comparable result is shown because the required OpenMP runtime is unavailable on this Mac; a future Linux/CI run must use the same whole-cycle protocol.")
        with st.expander("Panel question: R² is low—can we trust this forecast?"):
            st.markdown(
                f"""
                **The honest answer:** held-out R² is **{float(recovery_manifest['selected_metrics']['r2']):.3f}**, so the model explains only about
                **{max(float(recovery_manifest['selected_metrics']['r2']), 0) * 100:.0f}%** of recovery variation in unseen cycles. Most variation remains unexplained.

                This is why Canary is described as a **validated prototype**, not a production guarantee. The model is retained because it improves
                cycle-balanced MAE by **{float(recovery_gates['baseline_improvement_pct']):.1f}%** over the historical-mean baseline and produces an interpretable
                continuous estimate. However, it does not beat the majority rule for classifying 95% target attainment, so management should use the point estimate,
                range, and operational evidence—not treat it as a probability of success.

                Likely reasons include only **25 independent building outcomes across five eligible training cycles**, the proxy recovery label,
                missing or uneven measurements, and unrecorded factors such as health events and management interventions.
                """
            )
        with st.expander("What Canary adopted from the teammate model—and what it rejected"):
            st.markdown(
                "**Adopted:** structured cleaning layers, Zone A/B aggregation, environmental and growth-feature ideas, missingness flags, group-aware validation, and boosted-model challengers.\n\n"
                "**Not adopted:** treating repeated daily snapshots as independent flocks, leaving out only one building while the same cycle remains in training, future-informed bodyweight interpolation, feature selection before validation, or deploying a submitted pickle that cannot be reproduced on the corrected workbook."
            )

        st.markdown("#### F. Interpretation")
        interpretation_cols = st.columns(2)
        with interpretation_cols[0]:
            st.markdown("**What this means for this building**")
            if pd.notna(building["recovery_interval_low"]):
                st.write(
                    f"Expected recovery is **{_percent(building['predicted_final_recovery'])}**, with a likely range of "
                    f"**{_percent(building['recovery_interval_low'])}–{_percent(building['recovery_interval_high'])}**. "
                    f"The estimated gap to 95% is **{float(building['recovery_target_gap_pp']):+.1f} points**."
                )
            else:
                st.write(building["recovery_forecast_status"])
            owner_drivers = _owner_recovery_driver_table(recovery_contributions)
            if owner_drivers.empty:
                st.info("Building-specific model drivers are unavailable for this date.")
            else:
                st.dataframe(owner_drivers, hide_index=True, width="stretch")
        with interpretation_cols[1]:
            st.markdown("**How management should use it**")
            st.write(
                "Use the estimate to size the likely recovery gap and decide how urgently to investigate. "
                "Use the recorded risk dimensions and operational alerts—not model importance alone—to choose the inspection or intervention."
            )
            st.warning(
                "Drivers show predictive association, not cause. Canary cannot claim that changing temperature or humidity by a specific amount will create a specific recovery improvement."
            )

    with weight_model_tab:
        st.markdown("### A. Day 35 Average Weight Model")
        st.caption("Business question: given the latest measured weight and its age, what average building weight should we expect on Day 35 against 1,800 g?")

        st.markdown("#### B. Executive Summary")
        with st.container(border=True):
            weight_summary_cols = st.columns(4)
            weight_summary_cols[0].metric("This building", detail_weight)
            weight_summary_cols[1].metric("Day 35 goal", "1,800 g")
            weight_summary_cols[2].metric(
                "Typical error",
                f"{float(day35_manifest['selected_metrics']['mae_kg']) * 1000:.0f} g",
                help="Mean absolute error on complete harvest cycles that were excluded from training.",
            )
            weight_summary_cols[3].metric(
                "Within 200 g",
                f"{float(day35_manifest['selected_metrics']['within_200g_rate']):.1%}",
            )
            st.write(
                "Canary uses **historical remaining gain** as the operational method. It is not a trained regression model: it adds the average growth historically remaining from the current weighing age to the building’s latest measured weight. "
                "No learned challenger passed the predeclared replacement gates."
            )

        st.markdown("#### C. Input and Output Variables")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Role": "Output (Y)", "Variable": "Observed Day 35 average weight", "Plain-language definition": "The building’s recorded average bodyweight on production Day 35."},
                    {"Role": "Operational inputs", "Variable": "Latest measured weight and measurement day", "Plain-language definition": "The building-specific starting weight and the age used to select expected remaining growth."},
                    {"Role": "Operational reference", "Variable": "Historical remaining gain", "Plain-language definition": "Average observed Day 35 weight minus checkpoint weight, calculated using training cycles only."},
                    {"Role": "Learned-model candidates", "Variable": "Growth, target progress, survival, environment and freshness", "Plain-language definition": "These were tested for Ridge, linear and boosted challengers but do not drive the operational fallback."},
                    {"Role": "Excluded", "Variable": "Future checkpoint weights", "Plain-language definition": "At Day 14, Day 21, Day 28 and Day 35 measurements remain hidden."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown("#### D. Pre-processing Steps")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Stage": "1 · Data cleaning", "What happened": "Kept corrected observed bodyweights separate from the interpolated target curve; combined Zone A/B rows and retained missing weights as missing."},
                    {"Stage": "2 · Label and checkpoint creation", "What happened": "Retained 31 observed Day 35 Y outcomes and created 124 Day 7/14/21/28 checkpoint rows. Later checkpoint weights remained blank at earlier review dates."},
                    {"Stage": "3 · Feature engineering", "What happened": "Created measurement age, target ratio/deficit, recent and cumulative gain, available prior checkpoints, current survival, environmental exposure, missingness and staleness."},
                    {"Stage": "4 · Fold-only preparation", "What happened": "Imputation, missingness handling, scaling and feature filtering were fitted only on training cycles. The target curve was a reference—not a substitute outcome."},
                    {"Stage": "5 · Train/test split", "What happened": "No random row split was used. The outer loop removed one complete cycle, trained on the remaining cycles and predicted the unseen cycle, repeating across all six cycles."},
                    {"Stage": "6 · Cross-validation and selection", "What happened": "Inner whole-cycle cross-validation tuned learned candidates. A learned model then had to beat the remaining-gain baseline by 10%, keep positive R², reach 70% within 200 g and improve target-side usefulness."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown("#### E. Model Selection and Comparison")
        st.caption("MAE is the typical absolute miss. Cycle MAE gives every harvest cycle equal weight. RMSE penalizes larger misses. R² measures held-out variance explained. Lower MAE/RMSE and higher R² are better.")
        st.dataframe(
            _day35_comparison_summary(day35_manifest),
            hide_index=True,
            width="stretch",
        )
        weight_gates = day35_manifest["champion_gates"]
        st.warning(
            f"**Selected:** Historical remaining gain. The best learned challenger was {day35_manifest['research_champion'].replace('_', ' ').title()}, "
            f"but it was {abs(float(weight_gates['baseline_improvement_pct'])):.1f}% worse than the baseline on cycle-balanced MAE and reached only "
            f"{float(day35_manifest['research_champion_metrics']['within_200g_rate']):.1%} within 200 g. It did not earn deployment."
        )
        if day35_manifest.get("unavailable_candidates"):
            st.caption("XGBoost remains a declared fifth challenger with no local metric. It must be run under the identical nested whole-cycle protocol in Linux/CI before it can compete.")
        with st.expander("Panel question: R² is low—can we trust this projection?"):
            st.markdown(
                f"""
                **The honest answer:** the operational remaining-gain method has held-out R² of **{float(day35_manifest['selected_metrics']['r2']):.3f}**,
                so it explains only about **{max(float(day35_manifest['selected_metrics']['r2']), 0) * 100:.0f}%** of weight variation in unseen cycles.
                It is an **experimental planning estimate**, not a precise guarantee.

                Canary still shows it because errors are expressed in practical units—typical MAE is about
                **{float(day35_manifest['selected_metrics']['mae_kg']) * 1000:.0f} g**—and because every learned challenger performed worse under the same prospective test.
                The system therefore defaults to the simplest auditable method rather than claiming that a weak learned model is better.

                The low R² reflects only **31 independent Day 35 outcomes**, inconsistent historical weighing, strong cycle-to-cycle differences,
                few buildings above the 1,800 g goal, and unrecorded feed, health and intervention effects. More standardized completed cycles are the real remedy.
                """
            )

        st.markdown("#### F. Interpretation")
        weight_interpretation_cols = st.columns(2)
        with weight_interpretation_cols[0]:
            st.markdown("**How this building’s projection was calculated**")
            if pd.notna(building["projected_day35_weight_kg"]) and pd.notna(building["latest_weight_kg"]):
                remaining_gain_g = (
                    float(building["projected_day35_weight_kg"])
                    - float(building["latest_weight_kg"])
                ) * 1000
                st.write(
                    f"Latest measured weight **{_grams(building['latest_weight_kg'])}** on Day **{int(building['weight_measurement_day'])}** "
                    f"+ expected remaining gain **{remaining_gain_g:.0f} g** = projected Day 35 weight **{_grams(building['projected_day35_weight_kg'])}**."
                )
            else:
                st.info("A measured building weight is required before Canary can produce a Day 35 projection.")
            st.dataframe(
                pd.DataFrame(day35_manifest["selected_method_drivers"]),
                hide_index=True,
                width="stretch",
            )
        with weight_interpretation_cols[1]:
            st.markdown("**How management should use it**")
            st.write(
                "Use the projected gram gap to decide whether to reweigh and inspect feed access, water access, feeder allocation, bird condition, and house conditions. "
                "The forecast does not override a rules-based weight warning."
            )
            st.warning(
                "The current method is an experimental planning estimate: typical held-out error is about "
                f"{float(day35_manifest['selected_metrics']['mae_kg']) * 1000:.0f} g, and only "
                f"{float(day35_manifest['selected_metrics']['within_200g_rate']):.1%} of historical checkpoint predictions were within 200 g."
            )
        with st.expander("See what the best learned weight challenger relied on"):
            st.dataframe(
                pd.DataFrame(day35_manifest.get("research_champion_drivers", [])),
                hide_index=True,
                width="stretch",
            )
            st.caption("These Ridge associations are research evidence only. They do not power the operational remaining-gain forecast and do not prove causality.")

    if (
        pd.notna(building.get("weight_score"))
        and float(building["weight_score"]) >= 2
        and pd.notna(building["projected_day35_weight_kg"])
    ):
        st.warning(
            "The Day 35 projection does not cancel the rules-based warning. Today’s measured-weight gap remains a reason to inspect this building."
        )

    with st.expander("See raw forecast inputs and calculation trace"):
        st.caption("This is an audit of what was available on the review date—not a claim that every input caused the prediction.")
        evidence = forecast_input_trace(dataset, selected_cycle, chosen, pd.Timestamp(selected_date))
        if evidence.empty:
            st.info("No prediction-time evidence is available for this building and date.")
        else:
            st.dataframe(evidence, hide_index=True, width="stretch")
        st.dataframe(forecast_trace(building), hide_index=True, width="stretch")
        st.info("MAE is the typical absolute miss in business units. RMSE gives extra weight to large misses. R² describes variance explained on held-out cycles but is not the sole decision rule.")

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
            weight_chart["1.8 kg milestone"] = DAY35_TARGET_KG
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
    action_summary = pd.DataFrame(recommendation_playbook["rules"])
    action_summary["source"] = action_summary.apply(
        lambda rule: "Canary team safeguard"
        if rule["rule_id"] in {"DOC-001", "DOC-011"}
        else "Farmer Validation Workbook (Doc Raymond)",
        axis=1,
    )
    action_summary = action_summary[
        ["rule_id", "pattern", "dashboard_action", "source", "approval_status"]
    ].rename(
        columns={
            "rule_id": "Rule",
            "pattern": "Problem pattern",
            "dashboard_action": "Dashboard recommendation",
            "source": "Source",
            "approval_status": "Approval",
        }
    )
    st.dataframe(action_summary, hide_index=True, width="stretch")
    st.caption(recommendation_playbook.get("provenance_note", ""))

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
                    "Risk check": "2 · Population loss",
                    "What is measured": "Cumulative population loss from beginning inventory",
                    "Why it is kept": "Directly shows how much of the flock has already been lost",
                },
                {
                    "Risk check": "3 · Daily mortality",
                    "What is measured": "Latest daily mortality as a share of beginning birds",
                    "Why it is kept": "Catches an urgent current loss even before cumulative loss becomes large",
                },
                {
                    "Risk check": "4 · Environmental conditions",
                    "What is measured": "Higher of daily temperature swing and humidity deviation from the age range",
                    "Why it is kept": "Connects the warning to an operating condition management can inspect",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Design revision: mortality trend and peer comparison no longer add points. Peer results remain useful context, but the formal score now uses simpler building-level evidence that management can verify directly."
    )
    st.caption(
        "The Farmer Validation Workbook supplies the starting weight, population-loss, and daily-mortality references. "
        "Temperature and humidity now use the supplied tropical age bands. The distance outside each band remains provisional until Doc Raymond approves the severity cutoffs."
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
        dimension_specs = [
            ("Weight gap (%)", "weight_gap_pct"),
            ("Population loss (%)", "population_loss_pct"),
            ("Daily mortality (%)", "daily_mortality_pct"),
            ("Temperature outside age range (°C)", "temperature_deviation_c"),
            ("Humidity outside age range (points)", "humidity_deviation_pp"),
        ]
        dimension_editor = st.data_editor(
            pd.DataFrame([
                {
                    "Measure": label,
                    "0-point maximum": rules["dimension_cutoffs"][key][0],
                    "1-point maximum": rules["dimension_cutoffs"][key][1],
                    "2-point maximum": rules["dimension_cutoffs"][key][2],
                }
                for label, key in dimension_specs
            ]),
            disabled=["Measure"],
            hide_index=True,
            width="stretch",
            key="risk_dimension_threshold_editor",
        )

        st.markdown("**Accepted temperature range by age**")
        temperature_editor = st.data_editor(
            pd.DataFrame(rules["temperature_ranges_c"]).rename(columns={
                "label": "Age band", "minimum_age": "First day", "maximum_age": "Last day",
                "minimum": "Minimum temperature (°C)", "maximum": "Maximum temperature (°C)",
            }),
            disabled=["Age band", "First day", "Last day"],
            hide_index=True,
            width="stretch",
            key="risk_temperature_range_editor",
        )

        st.markdown("**Accepted humidity range by age**")
        humidity_editor = st.data_editor(
            pd.DataFrame(rules["humidity_ranges_pct"]).rename(columns={
                "label": "Age band", "minimum_age": "First day", "maximum_age": "Last day",
                "minimum": "Minimum humidity (%)", "maximum": "Maximum humidity (%)",
            }),
            disabled=["Age band", "First day", "Last day"],
            hide_index=True,
            width="stretch",
            key="risk_humidity_range_editor",
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

        environment_reading_age = st.number_input(
            "Maximum age of environmental reading (days)", min_value=0, max_value=14,
            value=int(rules["maximum_environment_reading_age_days"]), step=1,
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
                [
                    "Provisional - tropical bands supplied; severity distances require farm sign-off",
                    "Farm-approved by Doc Raymond",
                ],
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
                for (_, key), (_, edited) in zip(dimension_specs, dimension_editor.iterrows()):
                    updated_rules["dimension_cutoffs"][key] = [
                        float(edited["0-point maximum"]),
                        float(edited["1-point maximum"]),
                        float(edited["2-point maximum"]),
                    ]
                for band, (_, edited) in zip(updated_rules["temperature_ranges_c"], temperature_editor.iterrows()):
                    band["minimum"] = float(edited["Minimum temperature (°C)"])
                    band["maximum"] = float(edited["Maximum temperature (°C)"])
                for band, (_, edited) in zip(updated_rules["humidity_ranges_pct"], humidity_editor.iterrows()):
                    band["minimum"] = float(edited["Minimum humidity (%)"])
                    band["maximum"] = float(edited["Maximum humidity (%)"])
                updated_rules["maximum_environment_reading_age_days"] = int(environment_reading_age)
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
        '<div class="subtitle">Seven business questions covering data readiness, early warning, environment, outcomes, and forecast usefulness—with limitations shown beside the evidence.</div>',
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

        st.info(
            "Most defensible early-weight finding: higher Day 14 weight was moderately associated with higher Day 35 weight in this history. "
            "The recovery relationship points in the same direction, but is weaker and is not conclusive."
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
                "4 · Environment",
                "5 · Survival paths",
                "6 · Model accuracy",
                "7 · Target attainment",
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
                f"Within-cycle association is much weaker (r = {relationship['within_cycle_r']:.2f}). Target-attainment evidence must be interpreted against the revised 380 g Day 14 and 1.8 kg Day 35 goals. Association is not causal proof."
            )
            st.warning(
                f"Only {int(day14_met['building_cycles'])} of {coverage['paired_day14_day35']} historical building-cycles met the revised 380 g Day 14 target. That one flock recorded {float(day14_met['mean_day35_weight_kg']):.2f} kg on Day 35. This is directionally encouraging, but far too small for a reliable met-versus-missed comparison."
            )

        with question_tabs[2]:
            relationship = associations["day14_to_final_recovery"]
            st.subheader("Is Day 14 weight associated with harvest recovery?")
            st.write(
                f"The relationship points upward but is weak in this limited history: across {relationship['n']} paired building-cycles, "
                f"the raw correlation is r = {relationship['pearson_r']:.2f} (p = {relationship['pearson_p']:.2f}). "
                "That is not strong enough to claim that higher Day 14 weight reliably produces higher recovery."
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
            st.subheader("Are environmental readings and thresholds ready to guide action?")
            environment_profile, environment_stats = _eda_environment_profile(
                dataset, current_cycle, rules
            )
            environment_kpis = st.columns(4)
            environment_kpis[0].metric(
                "Environment coverage",
                f"{environment_stats['coverage']:.0%}",
            )
            environment_kpis[1].metric(
                "Historical readings",
                f"{int(environment_stats['environment_rows']):,}",
            )
            environment_kpis[2].metric(
                "Building-cycles covered",
                int(environment_stats["building_cycles"]),
            )
            environment_kpis[3].metric(
                "Outside provisional bands",
                f"{environment_stats['outside_rate']:.0%}",
            )
            if environment_profile.empty:
                st.info("There are not enough recorded temperature or humidity readings for this comparison.")
            else:
                environment_columns = st.columns(2)
                with environment_columns[0]:
                    st.markdown("**Recorded temperature versus age-specific band**")
                    st.line_chart(
                        environment_profile[
                            ["Recorded temperature", "Temperature minimum", "Temperature maximum"]
                        ],
                        height=300,
                    )
                with environment_columns[1]:
                    st.markdown("**Recorded humidity versus age-specific band**")
                    st.line_chart(
                        environment_profile[
                            ["Recorded humidity", "Humidity minimum", "Humidity maximum"]
                        ],
                        height=300,
                    )
            st.warning(
                f"The current provisional bands mark {environment_stats['outside_rate']:.0%} of historical recorded environment-days as outside range, "
                f"leaving only {int(environment_stats['within_rows'])} within-band records. That imbalance is too extreme for a fair within-versus-outside mortality comparison. "
                "The readings are useful for inspection context, but the bands and sensors must be validated with Doc Raymond before making strong causal or intervention claims."
            )
            st.caption(
                "Business interpretation: Canary can identify recorded deviations, but this EDA does not prove that a specific temperature or humidity change caused mortality."
            )

        with question_tabs[4]:
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

        with question_tabs[5]:
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
                "Both methods hold out complete cycles. Recovery has five eligible training cycles; Day 35 weight uses 31 historical building outcomes across six cycles. These metrics support planning estimates, not guaranteed target classification."
            )

        with question_tabs[6]:
            st.subheader("How often have historical buildings achieved the two farm goals?")
            completed = evidence_rows.dropna(
                subset=["day35_weight_kg", "recomputed_recovery"]
            ).copy()
            completed["Day 35 goal met"] = completed["day35_weight_kg"] >= DAY35_TARGET_KG
            completed["Recovery goal met"] = completed["recomputed_recovery"] >= 0.95
            target_kpis = st.columns(4)
            target_kpis[0].metric("Completed building-cycles", len(completed))
            target_kpis[1].metric(
                "Met 1,800 g on Day 35",
                f"{int(completed['Day 35 goal met'].sum())} of {len(completed)}",
            )
            target_kpis[2].metric(
                "Met 95% recovery",
                f"{int(completed['Recovery goal met'].sum())} of {len(completed)}",
            )
            target_kpis[3].metric(
                "Met both goals",
                f"{int((completed['Day 35 goal met'] & completed['Recovery goal met']).sum())} of {len(completed)}",
            )
            cycle_outcomes = (
                completed.groupby("cycle_id", as_index=False)
                .agg(
                    **{
                        "Buildings": ("building_id", "nunique"),
                        "Average Day 35 weight (g)": ("day35_weight_kg", lambda values: values.mean() * 1000),
                        "Average recovery (%)": ("recomputed_recovery", lambda values: values.mean() * 100),
                        "Day 35 goal hits": ("Day 35 goal met", "sum"),
                        "Recovery goal hits": ("Recovery goal met", "sum"),
                    }
                )
                .rename(columns={"cycle_id": "Cycle"})
            )
            cycle_outcomes["Average Day 35 weight (g)"] = cycle_outcomes[
                "Average Day 35 weight (g)"
            ].round(0)
            cycle_outcomes["Average recovery (%)"] = cycle_outcomes[
                "Average recovery (%)"
            ].round(1)
            st.dataframe(cycle_outcomes, hide_index=True, width="stretch")
            st.info(
                "Target attainment is historically imbalanced: most completed buildings are below one or both goals. This explains why target-side accuracy can look high even when the model rarely recognizes the smaller at-target group. Always compare target classification with the majority baseline."
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
            st.caption("The learned challengers did not clear the approved accuracy gates. Canary therefore uses historical remaining gain as the transparent operational fallback.")

    st.info(
        "How to present this in one minute: ‘Canary keeps three layers separate: transparent risk rules, two predictive outlooks, and a deterministic action playbook. "
        "The weight output targets 1.8 kg on Day 35. The recovery output targets 95% at harvest, while clearly disclosing that its historical training label is the last recorded population ratio."
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
                    "Input": "Current weight gap, population loss, daily mortality, and age-specific temperature/humidity evidence",
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
            5. The results are compared with the 95% recovery goal and 1.8 kg Day 35 milestone.
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

    with st.expander("Why Canary does not use SMOTE or oversampling"):
        st.markdown(
            """
            - Both forecasts are **regression** problems; standard SMOTE is a classification technique.
            - The scarce evidence is independent building-cycle outcomes—not spreadsheet rows. Synthetic rows do not create new flocks.
            - Artificial flock histories could be biologically implausible and could make validation error look too small.
            - Canary instead uses simple regularized models, complete-cycle holdouts, balanced checkpoints, uncertainty ranges, and explicit limitations.

            The most valuable improvement is more standardized completed cycles with verified harvest events—not synthetic observations.
            """
        )
    st.info(
        "Supplementary reproducible evidence is included in the repository: "
        "`notebooks/Project_Canary_Harvest_Recovery_Model.ipynb` and "
        "`notebooks/Project_Canary_Day35_Weight_Model.ipynb`."
    )

    st.subheader("Download the model-ready evidence")
    st.caption(
        "These are the exact auditable rows behind the model comparisons—not a manually prepared substitute. "
        "The outcome sheet has one row per building-cycle; the training sheets contain leakage-safe as-of snapshots."
    )
    evidence_root = Path(__file__).resolve().parent
    evidence_files = [
        ("Model-ready workbook", evidence_root / "outputs" / "model_ready" / "Project_Canary_Model_Ready_Data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("Recovery training CSV", evidence_root / "outputs" / "model_ready" / "recovery_training.csv", "text/csv"),
        ("Day 35 weight CSV", evidence_root / "outputs" / "model_ready" / "day35_weight_training.csv", "text/csv"),
        ("Recovery notebook", evidence_root / "notebooks" / "Project_Canary_Harvest_Recovery_Model.ipynb", "application/x-ipynb+json"),
        ("Day 35 weight notebook", evidence_root / "notebooks" / "Project_Canary_Day35_Weight_Model.ipynb", "application/x-ipynb+json"),
    ]
    download_columns = st.columns(len(evidence_files))
    for column, (label, file_path, mime) in zip(download_columns, evidence_files):
        with column:
            if file_path.exists():
                st.download_button(label, file_path.read_bytes(), file_name=file_path.name, mime=mime, width="stretch")
            else:
                st.caption(f"{label}: regenerate evidence exports")

    st.subheader("Data lineage")
    lineage = pd.DataFrame(
        [
            {"Stage": "1 · Source workbook", "Rows / outcomes": f"{dataset.quality.source_rows:,} source rows", "Control": "Read-only input"},
            {"Stage": "2 · Canonical building-days", "Rows / outcomes": f"{dataset.quality.canonical_rows:,} rows", "Control": "Zone rows consolidated; conflicts checked"},
            {"Stage": "3 · Labeled outcomes", "Rows / outcomes": f"{recovery_manifest['training_building_cycles']} recovery / {day35_manifest['training_building_cycles']} weight", "Control": "Y retained only for completed historical evidence"},
            {"Stage": "4 · Training snapshots", "Rows / outcomes": f"{recovery_manifest['training_snapshot_rows']} recovery / {day35_manifest['training_checkpoint_rows']} weight", "Control": "Only facts known by the review date"},
            {"Stage": "5 · Validation", "Rows / outcomes": "One complete cycle held out at a time", "Control": "No same-cycle train/test mixing"},
            {"Stage": "6 · Live forecast", "Rows / outcomes": "One as-of snapshot per active building", "Control": "Inference only; no daily retraining"},
        ]
    )
    st.dataframe(lineage, hide_index=True, width="stretch")

    risk_tab, recovery_tab, weight_tab, action_tab = st.tabs(
        ["1 · Risk scoring", "2A · Recovery model", "2B · Day 35 weight", "3 · Recommendations"]
    )

    with recovery_tab:
        recovery_name = {
            "ridge_core": "Compact Ridge regression",
            "ridge_no_weight": "Ridge regression (without weight inputs)",
        }.get(
            recovery_manifest["selected_model"],
            recovery_manifest["selected_model"].replace("_", " ").title(),
        )
        st.subheader(f"Recovery model: {recovery_name}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Workflow": "Business question", "Plain-language explanation": "Given what is known on the review date, what last-recorded recovery should we expect for this building?"},
                    {"Workflow": "Goal / Y", "Plain-language explanation": "Estimate each building’s recovery at harvest. Historical proxy: population on the last recorded date ÷ beginning population."},
                    {"Workflow": "Inputs / X", "Plain-language explanation": "Age, current survival, recent mortality, weight gap/freshness, and temperature/humidity band deviations known on the review date. Feed is withheld until its unit is confirmed."},
                    {"Workflow": "Methods tried", "Plain-language explanation": "Historical mean, ordinary linear regression, Compact Ridge, and constrained Gradient Boosting. XGBoost is documented as unavailable on this Mac rather than presented as a comparable result."},
                    {"Workflow": "Fair comparison", "Plain-language explanation": "Nested validation: hold out one complete cycle; tune only within the remaining cycles; then predict the unseen cycle."},
                    {"Workflow": "Winner", "Plain-language explanation": f"{recovery_name} for the continuous estimate. It cleared the MAE/R² gate but not the separate 95% classification gate."},
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
                        {"Step": "6 · Validate", "What happened": "Nested validation held out one entire cycle. Imputation, scaling, feature filtering and tuning used only the remaining cycles."},
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
                {"Feature group": "Flock timing", "Inputs": "Cycle day"},
                {"Feature group": "Survival and mortality", "Inputs": "% alive, daily mortality, recent 3-day mortality, mortality trend"},
                {"Feature group": "Weight progress", "Inputs": "Gap versus the age target and days since last weighing"},
                {"Feature group": "Environment", "Inputs": "Temperature/humidity deviation from approved age bands, recent out-of-band days, and reading freshness"},
                {"Feature group": "Feed", "Inputs": "Withheld from the final compact set until the farm confirms the recorded unit"},
            ]
        )
        st.dataframe(recovery_features, hide_index=True, width="stretch")
        st.caption(
            f"Selected model uses {len(selected_recovery_features)} compact inputs. It excludes raw beginning inventory, building identity, algebraic duplicates, and unconfirmed feed units. Current survival remains because it is known on the review date and logically constrains final recovery. Missing numeric inputs are filled using training-fold medians and marked with missing-value indicators."
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
            importance_records = recovery_manifest.get(
                "held_out_permutation_importance", []
            )
            top_recorded = importance_records[0] if importance_records else None
            st.info(
                (
                    f"Held-out model reading: **{FEATURE_DISPLAY.get(top_recorded['feature'], top_recorded['feature'])}** caused the largest average loss of accuracy when shuffled in unseen cycles. "
                    if top_recorded
                    else "Held-out driver ranking is unavailable. "
                )
                + "This is predictive association—not proof that changing the factor will change recovery."
            )
            st.caption(
                "The ranking uses out-of-fold permutation importance from complete held-out cycles. Correlated inputs can share or swap importance, and none of these values proves causation."
            )

        st.markdown("**How it was validated**")
        st.write(
            "Canary uses nested whole-cycle validation. The outer loop leaves one complete cycle unseen. The inner loop tunes the model using only the remaining cycles. "
            "Imputation, scaling and tuning never see the held-out cycle, and repeated checkpoints are weighted so each building-cycle has equal total influence."
        )
        st.dataframe(
            _candidate_metrics_table(recovery_manifest, "recovery"),
            hide_index=True,
            width="stretch",
        )
        with st.expander("See recovery performance for every held-out cycle"):
            cycle_table = _recovery_cycle_metrics_table(recovery_manifest)
            if cycle_table.empty:
                st.caption("Retrain the versioned model bundle to generate cycle-level evidence.")
            else:
                st.dataframe(cycle_table, hide_index=True, width="stretch")
        recovery_gates = recovery_manifest["champion_gates"]
        st.caption(
            f"Selection detail: {recovery_name} improved cycle-balanced MAE by "
            f"{float(recovery_gates['baseline_improvement_pct']):.1f}% versus the historical-mean baseline and kept positive R². "
            "Its continuous forecast gate passed; its 95% target-side classification gate did not."
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
        st.subheader("Day 35 weight method: gated learned models with a transparent fallback")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Workflow": "Business question", "Plain-language explanation": "Given the latest measured weight, what average building weight should we expect on Day 35?"},
                    {"Workflow": "Goal / Y", "Plain-language explanation": "Estimate the building’s recorded average liveweight specifically on production Day 35."},
                    {"Workflow": "Inputs / X", "Plain-language explanation": "Latest/checkpoint weights, weighing day, target progress, recent gain, current survival, and environmental-band exposure known at that checkpoint."},
                    {"Workflow": "Methods tried", "Plain-language explanation": "Historical remaining gain, ordinary linear regression, compact Ridge, and constrained Gradient Boosting. XGBoost is documented as unavailable on this Mac rather than presented as a comparable result."},
                    {"Workflow": "Fair comparison", "Plain-language explanation": "Nested whole-cycle validation: tune only on training cycles, then compare errors in grams on the completely unseen cycle."},
                    {"Workflow": "Operational result", "Plain-language explanation": "Historical remaining gain remains the live method because no learned challenger beat it by 10%, kept positive R², reached 70% within 200 g, and improved target-side classification."},
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
                        {"Step": "6 · Apply gates", "What happened": "A learned model replaces the baseline only if it clears all approved MAE, R², within-200 g, and target-side gates."},
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
        with st.expander("See the revised daily target-weight curve"):
            target_view = dataset.targets.loc[
                dataset.targets["age_day"].le(35),
                [
                    "age_day",
                    "target_weight_linear_g",
                    "daily_gain_linear_g",
                    "target_weight_scaled_g",
                    "daily_gain_scaled_g",
                ],
            ].rename(
                columns={
                    "age_day": "Day",
                    "target_weight_linear_g": "Linear target (g)",
                    "daily_gain_linear_g": "Linear daily gain (g)",
                    "target_weight_scaled_g": "Smoothed target used by Canary (g)",
                    "daily_gain_scaled_g": "Smoothed daily gain (g)",
                }
            )
            st.dataframe(target_view, hide_index=True, width="stretch")
            st.caption(
                "Doc Raymond’s approved checkpoints are fixed at 170, 380, 800, 1,200, and 1,800 g on Days 7, 14, 21, 28, and 35. The smoothed values preserve the former farm curve’s within-week shape and are used for daily target comparisons."
            )
        st.markdown("**What the operational method uses**")
        st.dataframe(
            pd.DataFrame(day35_manifest["selected_method_drivers"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "These are the direct inputs to the transparent historical remaining-gain formula."
        )
        if day35_manifest.get("research_champion_drivers"):
            with st.expander("See what the best learned challenger relied on"):
                st.dataframe(
                    pd.DataFrame(day35_manifest["research_champion_drivers"]),
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
        with st.expander("See weight performance for every held-out cycle"):
            cycle_table = _day35_cycle_metrics_table(day35_manifest)
            if cycle_table.empty:
                st.caption("Retrain the versioned Day 35 manifest to generate cycle-level evidence.")
            else:
                st.dataframe(cycle_table, hide_index=True, width="stretch")
        st.warning(
            f"The historical Day 35 set contains {day35_manifest['actual_target_hits']} results at/above 1.8 kg and "
            f"{day35_manifest['actual_target_misses']} below it. Target-side accuracy is now measurable, but the small hit group still limits confidence."
        )
        st.info(
            "Business interpretation: the operational remaining-gain method still responds to each building’s latest measured weight and measurement day. The learned models remain experimental until they clear every gate."
        )
        st.subheader("Why the transparent baseline remains operational")
        st.markdown(
            f"""
            Historical remaining gain reached **{day35_candidates.get('historical_remaining_gain', wmetrics)['mae_kg'] * 1000:.0f} g MAE** and remains the transparent benchmark.
            Ordinary linear regression reached **{day35_candidates.get('linear_regression', wmetrics)['mae_kg'] * 1000:.0f} g MAE**.
            Constrained Gradient Boosting reached **{day35_candidates.get('gradient_boosting', wmetrics)['mae_kg'] * 1000:.0f} g MAE**.
            XGBoost is documented as an unavailable challenger on this Mac because its OpenMP runtime is absent; it is not included in the fair numerical comparison.

            The best learned challenger was **{day35_manifest['research_champion'].replace('_', ' ').title()}**, but it did not improve cycle-balanced MAE by the required 10% and did not reach 70% within 200 g. Therefore the **historical remaining-gain method stays operational**. This is the more defensible small-data decision.
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
        st.subheader("What the risk score means")
        st.markdown(
            """
            Canary gives 0–3 points to four directly observed building-level checks. The total sets the operational-priority label; it is not a probability of missing either target.

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
                        "Check": "Population loss",
                        "Calculation": "(Beginning birds − current birds) ÷ beginning birds",
                        "Operational meaning": "How much of the flock has already been lost?",
                    },
                    {
                        "Check": "Daily mortality",
                        "Calculation": "Latest daily mortality ÷ beginning birds",
                        "Operational meaning": "Is there an urgent current loss?",
                    },
                    {
                        "Check": "Environmental conditions",
                        "Calculation": "Higher of average-temperature deviation and humidity deviation from their age-specific ranges",
                        "Operational meaning": "Is a recorded operating condition outside the provisional limit?",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.warning(
            f"Validation limit: this is an expert-rule priority score, not a trained outcome model. Direct environmental readings cover {environment_direct_coverage_pct:.1f}% of operational building-days in the loaded workbook. Canary can carry a reading forward for at most {int(rules['maximum_environment_reading_age_days'])} days; after that it is shown as stale and not scored. The tropical age bands are supplied, but the distances used to assign 1, 2, or 3 points still require farm sign-off."
        )
        st.info(
            "Why this version is clearer: the old mortality-trend and peer points were removed. Peer comparisons remain diagnostic context; daily mortality and environmental conditions now connect the score to checks management can perform."
        )
        threshold_table = pd.DataFrame([
            {"Check": label, "0 / 1 / 2-point maximums": rules["dimension_cutoffs"][key]}
            for label, key in [
                ("Weight gap (%)", "weight_gap_pct"),
                ("Population loss (%)", "population_loss_pct"),
                ("Daily mortality (%)", "daily_mortality_pct"),
                ("Temperature outside age range (°C)", "temperature_deviation_c"),
                ("Humidity outside age range (points)", "humidity_deviation_pp"),
            ]
        ])
        st.dataframe(
            threshold_table,
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "At or below the first value = 0 points; above first through second = 1; above second through third = 2; above third = 3. The environmental score uses the worse of temperature or humidity deviation, so the two related signals are not double-counted."
        )
        st.markdown("**Age-specific tropical temperature reference**")
        st.dataframe(pd.DataFrame(rules["temperature_ranges_c"]), hide_index=True, width="stretch")
        st.markdown("**Age-specific humidity reference**")
        st.dataframe(pd.DataFrame(rules["humidity_ranges_pct"]), hide_index=True, width="stretch")
        if st.button("Review or edit risk thresholds", key="open_risk_rule_admin"):
            st.switch_page(PAGE_CHECKS)

    with action_tab:
        st.subheader("From recorded signal to next action")
        st.markdown(
            "Canary first looks for a specific, current operating alert supported by recorded evidence. If one exists, that alert leads the owner-facing action. If none exists, Canary labels the operating cause as unconfirmed and falls back to the broader problem-pattern playbook. It never asks a model to invent a treatment."
        )
        action_method = pd.DataFrame(recommendation_playbook["rules"])
        action_method["source"] = action_method.apply(
            lambda rule: "Canary team safeguard"
            if rule["rule_id"] in {"DOC-001", "DOC-011"}
            else "Farmer Validation Workbook (Doc Raymond)",
            axis=1,
        )
        action_method = action_method[
            ["pattern", "dashboard_action", "source", "approval_status"]
        ].rename(
            columns={
                "pattern": "Problem pattern",
                "dashboard_action": "Recommended management focus",
                "source": "Source",
                "approval_status": "Validation status",
            }
        )
        st.dataframe(action_method, hide_index=True, width="stretch")
        st.caption(recommendation_playbook.get("provenance_note", ""))
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
