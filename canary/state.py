"""As-of building-state logic for Sprint 1."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .data import CanaryDataset


CANONICAL_BUILDINGS = (
    "Tags 1",
    "Tags 2",
    "Tags 3",
    "Lags 1",
    "Lags 2",
    "Lags 3",
)


def cycle_date_bounds(dataset: CanaryDataset, cycle_id: str) -> tuple[date, date]:
    cycle = dataset.cycles.loc[dataset.cycles["cycle_id"] == cycle_id]
    if cycle.empty:
        raise KeyError(f"Unknown harvest cycle: {cycle_id}")
    return cycle["start_date"].min().date(), cycle["end_date"].max().date()


def default_as_of_date(dataset: CanaryDataset, cycle_id: str) -> date:
    """Choose the latest date with complete daily operations for the cycle.

    The workbook ``End Date`` is only the maximum date in the daily records;
    it is not proof of harvest. Prefer the latest complete operational date so
    the dashboard opens on the freshest decision-ready evidence.
    """

    minimum, maximum = cycle_date_bounds(dataset, cycle_id)
    cycle_daily = dataset.daily.loc[
        (dataset.daily["cycle_id"] == cycle_id) & dataset.daily["daily_complete"]
    ]
    if cycle_daily.empty:
        return minimum
    latest = cycle_daily["record_date"].max().date()
    return max(minimum, min(latest, maximum))


def _freshness_label(days: object) -> str:
    if pd.isna(days):
        return "No observation"
    value = int(days)
    if value == 0:
        return "Current"
    if value <= 2:
        return f"Delayed {value}d"
    return f"Stale {value}d"


def _safe_int(value: object) -> object:
    return int(round(float(value))) if pd.notna(value) else pd.NA


def build_cycle_snapshot(
    dataset: CanaryDataset,
    cycle_id: str,
    as_of: date | pd.Timestamp,
) -> pd.DataFrame:
    """Return exactly six building records using information known by ``as_of``."""

    as_of_ts = pd.Timestamp(as_of).normalize()
    cycle_meta = dataset.cycles.loc[dataset.cycles["cycle_id"] == cycle_id]
    cycle_daily = dataset.daily.loc[dataset.daily["cycle_id"] == cycle_id]
    latest_cycle = str(
        dataset.cycles.groupby("cycle_id")["start_date"].min().idxmax()
    )
    is_latest_cycle = str(cycle_id) == latest_cycle
    target_by_age = dataset.targets.set_index("age_day")["target_weight_kg"]
    records: list[dict[str, object]] = []

    for order, building in enumerate(CANONICAL_BUILDINGS):
        meta_match = cycle_meta.loc[cycle_meta["building_id"] == building]
        base: dict[str, object] = {
            "building_order": order,
            "building_id": building,
            "cycle_id": cycle_id,
            "as_of_date": as_of_ts,
            "state": "Inactive",
            "cycle_day": pd.NA,
            "beginning_inventory": pd.NA,
            "latest_population": pd.NA,
            "percentage_alive": pd.NA,
            "latest_operational_day": pd.NA,
            "data_staleness_days": pd.NA,
            "data_freshness": "No observation",
            "latest_weight_kg": pd.NA,
            "weight_measurement_day": pd.NA,
            "weight_target_at_measurement_kg": pd.NA,
            "weight_staleness_days": pd.NA,
            "weight_freshness": "No measurement",
            "last_recorded_population": pd.NA,
            "last_recorded_recovery_rate": pd.NA,
            "placement_date": pd.NaT,
            "latest_recorded_date": pd.NaT,
            "status_note": "No flock recorded for this building in the selected cycle.",
        }

        if meta_match.empty:
            records.append(base)
            continue

        meta = meta_match.iloc[0]
        start = pd.Timestamp(meta["start_date"]).normalize()
        end = pd.Timestamp(meta["end_date"]).normalize()
        base.update(
            {
                "placement_date": start,
                "latest_recorded_date": end,
                "beginning_inventory": _safe_int(meta["beginning_inventory"]),
            }
        )

        if as_of_ts < start:
            base["status_note"] = f"Placement begins {start.strftime('%d %b %Y')}."
            records.append(base)
            continue

        selected_cycle_day = max(1, (as_of_ts - start).days + 1)
        final_cycle_day = max(1, (end - start).days + 1)
        base["cycle_day"] = min(selected_cycle_day, final_cycle_day)

        building_daily = cycle_daily.loc[
            (cycle_daily["building_id"] == building)
            & (cycle_daily["record_date"] <= as_of_ts)
        ].sort_values(["record_date", "age_day"])
        operational = building_daily.loc[building_daily["operational_recorded"]]
        latest_operational = operational.iloc[-1] if not operational.empty else None

        if latest_operational is not None:
            latest_age = int(latest_operational["age_day"])
            base["latest_operational_day"] = latest_age
            base["data_staleness_days"] = max(0, selected_cycle_day - latest_age)
            base["data_freshness"] = _freshness_label(base["data_staleness_days"])
            base["latest_population"] = _safe_int(latest_operational["population"])
            beginning = meta["beginning_inventory"]
            if pd.notna(latest_operational["population"]) and pd.notna(beginning) and beginning:
                base["percentage_alive"] = float(latest_operational["population"] / beginning)

        weights = building_daily.loc[building_daily["weight_measured"]]
        if not weights.empty:
            latest_weight = weights.iloc[-1]
            weight_day = int(latest_weight["age_day"])
            base["latest_weight_kg"] = float(latest_weight["bodyweight_kg"])
            base["weight_measurement_day"] = weight_day
            base["weight_target_at_measurement_kg"] = (
                float(target_by_age.loc[weight_day]) if weight_day in target_by_age.index else pd.NA
            )
            base["weight_staleness_days"] = max(0, selected_cycle_day - weight_day)
            base["weight_freshness"] = _freshness_label(base["weight_staleness_days"])

        if as_of_ts >= end and not is_latest_cycle:
            # ``end`` is the maximum daily-record date, not a confirmed harvest
            # date. Preserve the last recorded evidence and keep model outputs
            # as forecasts rather than silently replacing them with "actuals".
            base["state"] = "Records ended"
            base["cycle_day"] = final_cycle_day
            base["last_recorded_population"] = _safe_int(meta["ending_inventory"])
            base["last_recorded_recovery_rate"] = (
                float(meta["final_recovery_rate"])
                if pd.notna(meta["final_recovery_rate"])
                else pd.NA
            )
            base["latest_population"] = base["last_recorded_population"]
            base["percentage_alive"] = base["last_recorded_recovery_rate"]
            base["status_note"] = (
                "Latest recorded day reached; harvest completion is not confirmed."
            )
        else:
            exact_day = building_daily.loc[building_daily["record_date"] == as_of_ts]
            exact_complete = bool(exact_day["daily_complete"].any()) if not exact_day.empty else False
            if exact_complete:
                base["state"] = "Active"
                if is_latest_cycle and as_of_ts >= end:
                    base["status_note"] = (
                        "Latest current-cycle record; harvest completion is not confirmed."
                    )
                else:
                    base["status_note"] = "Daily mortality and feed observations are available."
            else:
                base["state"] = "Incomplete"
                if exact_day.empty:
                    base["status_note"] = "No building-day record exists for the selected date."
                else:
                    missing: list[str] = []
                    if not exact_day["mortality_recorded"].any():
                        missing.append("mortality")
                    if not exact_day["feed_recorded"].any():
                        missing.append("feed")
                    base["status_note"] = "Missing recorded " + " and ".join(missing) + "."

        records.append(base)

    return pd.DataFrame(records).sort_values("building_order").reset_index(drop=True)
