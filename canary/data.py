"""Workbook ingestion and canonical building-day preparation for Project Canary."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd


DAILY_SHEET = "Farm Harvest Data (Daily)"
CYCLE_SHEET = "Farm Harvest Data (By Cycle)"
TARGET_SHEET = "Target Weights"
APPROVED_WEIGHT_TARGETS_G = {
    # Day 0 is a working placement anchor retained from the existing farm curve;
    # Doc Raymond's revised approved checkpoints begin on Day 7.
    0: 40.0,
    7: 170.0,
    14: 380.0,
    21: 800.0,
    28: 1200.0,
    35: 1800.0,
}
TEMPERATURE_SHEET = "Temperature"

DAILY_REQUIRED = {
    "Harvest Cycle",
    "Bldg.",
    "Age",
    "Date",
    "Beginning Inventory",
    "Population",
    "mortality_daily",
    "mortality_cum(hds)",
    "feedconsumption_daily",
    "Bodyweight (kgs)",
}

CYCLE_REQUIRED = {
    "Harvest Cycle",
    "Bldg.",
    "Start Date",
    "End Date",
    "Beginning Inventory",
    "Ending Inventory",
    "Harvest Recovery",
}

TARGET_REQUIRED = {
    "Age",
    "Target Weight (Scaled Interpolation)",
}

PRODUCTION_FIELDS = [
    "record_date",
    "beginning_inventory",
    "population",
    "mortality_daily",
    "mortality_cumulative",
    "feed_daily_bags",
    "feed_cumulative_bags",
    "feed_daily_kg_per_bird",
    "bodyweight_kg",
]

ENVIRONMENT_FIELDS = [
    "temperature_min_c",
    "temperature_max_c",
    "temperature_avg_c",
    "humidity_min_pct",
    "humidity_max_pct",
    "humidity_avg_pct",
]


class WorkbookValidationError(ValueError):
    """Raised when an uploaded workbook cannot be processed safely."""


@dataclass(frozen=True)
class DataQualityReport:
    source_rows: int
    canonical_rows: int
    unique_source_keys: int
    duplicate_keys: int
    duplicate_rows_consolidated: int
    multi_environment_days: int
    zone_aggregated_days: int
    maximum_environment_sections: int
    production_conflict_keys: int
    operationally_missing_days: int
    incomplete_daily_records: int
    weight_measurement_days: int
    temperature_coverage_pct: float
    humidity_coverage_pct: float
    blocking_errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.blocking_errors


@dataclass(frozen=True)
class CanaryDataset:
    daily: pd.DataFrame
    cycles: pd.DataFrame
    targets: pd.DataFrame
    quality: DataQualityReport
    source_name: str


def _normalize_building(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = " ".join(str(value).strip().split())
    compact = text.replace(" ", "").lower()
    mapping = {
        "tags1": "Tags 1",
        "tags2": "Tags 2",
        "tags3": "Tags 3",
        "lags1": "Lags 1",
        "lags2": "Lags 2",
        "lags3": "Lags 3",
    }
    return mapping.get(compact, text)


def _normalize_blank(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return pd.NA
    return value


def _first_valid(series: pd.Series) -> object:
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else pd.NA


def _as_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _validate_sheets(book: dict[str, pd.DataFrame]) -> None:
    required = {DAILY_SHEET, CYCLE_SHEET, TARGET_SHEET}
    missing = sorted(required - set(book))
    if missing:
        raise WorkbookValidationError(
            "Missing required sheet(s): " + ", ".join(missing)
        )

    checks = [
        (DAILY_SHEET, DAILY_REQUIRED),
        (CYCLE_SHEET, CYCLE_REQUIRED),
        (TARGET_SHEET, TARGET_REQUIRED),
    ]
    for sheet, required_columns in checks:
        missing_columns = sorted(required_columns - set(book[sheet].columns))
        if missing_columns:
            raise WorkbookValidationError(
                f"{sheet} is missing column(s): {', '.join(missing_columns)}"
            )


def _prepare_environment(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate building-level or Zone A/Zone B readings to building-day grain.

    Section averages are combined using an unweighted mean because the source
    workbook does not contain section-size weights.  Minimum and maximum values
    retain the full observed building envelope.  Section-to-section spreads are
    preserved as diagnostics but are not automatically used as model features.
    """

    environment = raw.copy().map(_normalize_blank).rename(
        columns={
            "Harvest Cycle": "cycle_id",
            "Bldg.": "building_id",
            "Age": "age_day",
            "Date": "environment_record_date",
            "Building Section": "environment_section",
            "Min Temperature": "temperature_min_c",
            "Max Temperature": "temperature_max_c",
            "Average Temperature": "temperature_avg_c",
            "Min Humidity ": "humidity_min_pct",
            "Max Humidity": "humidity_max_pct",
            "Average Humidity": "humidity_avg_pct",
        }
    )
    for column in ["environment_section", *ENVIRONMENT_FIELDS]:
        if column not in environment:
            environment[column] = pd.NA
    environment["cycle_id"] = environment["cycle_id"].astype("string").str.strip()
    environment["building_id"] = environment["building_id"].map(_normalize_building)
    environment["environment_record_date"] = pd.to_datetime(
        environment["environment_record_date"], errors="coerce"
    ).dt.normalize()
    _as_numeric(environment, ["age_day", *ENVIRONMENT_FIELDS])
    key = ["cycle_id", "building_id", "age_day"]
    environment = environment.dropna(subset=key).copy()
    environment["age_day"] = environment["age_day"].astype(int)
    environment["has_environment_reading"] = environment[ENVIRONMENT_FIELDS].notna().any(axis=1)

    rows: list[dict[str, object]] = []
    for group_key, group in environment.groupby(key, sort=False, dropna=False):
        observed = group.loc[group["has_environment_reading"]].copy()
        labels = sorted(
            {
                str(value).strip()
                for value in observed["environment_section"].dropna()
                if str(value).strip()
            }
        )
        section_count = int(len(observed))
        temperature_averages = observed["temperature_avg_c"].dropna()
        humidity_averages = observed["humidity_avg_pct"].dropna()
        rows.append(
            {
                **dict(zip(key, group_key)),
                "environment_record_date": _first_valid(group["environment_record_date"]),
                "temperature_min_c": observed["temperature_min_c"].min(),
                "temperature_max_c": observed["temperature_max_c"].max(),
                "temperature_avg_c": temperature_averages.mean(),
                "humidity_min_pct": observed["humidity_min_pct"].min(),
                "humidity_max_pct": observed["humidity_max_pct"].max(),
                "humidity_avg_pct": humidity_averages.mean(),
                "environment_section_count": section_count,
                "environment_sections": ", ".join(labels) if labels else "Building-level",
                "zone_aggregated": section_count > 1,
                "temperature_zone_spread_c": (
                    float(temperature_averages.max() - temperature_averages.min())
                    if len(temperature_averages) > 1
                    else pd.NA
                ),
                "humidity_zone_spread_pct": (
                    float(humidity_averages.max() - humidity_averages.min())
                    if len(humidity_averages) > 1
                    else pd.NA
                ),
                "environment_aggregation_method": (
                    "Unweighted mean of section averages; envelope min/max"
                    if section_count > 1
                    else "Single building reading"
                ),
            }
        )
    return pd.DataFrame(rows)


def _prepare_daily(
    raw: pd.DataFrame, raw_environment: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, DataQualityReport]:
    daily = raw.copy()
    daily = daily.map(_normalize_blank)
    rename = {
        "Harvest Cycle": "cycle_id",
        "Bldg.": "building_id",
        "Age": "age_day",
        "Date": "record_date",
        "Beginning Inventory": "beginning_inventory",
        "Population": "population",
        "mortality_daily": "mortality_daily",
        "mortality_cum(hds)": "mortality_cumulative",
        "feedconsumption_daily": "feed_daily_bags",
        "feedconsumption_cummulative": "feed_cumulative_bags",
        "Daily FI/bird": "feed_daily_kg_per_bird",
        "Bodyweight (kgs)": "bodyweight_kg",
        "Min Temperature": "temperature_min_c",
        "Max Temperature": "temperature_max_c",
        "Average Temperature": "temperature_avg_c",
        "Min Humidity ": "humidity_min_pct",
        "Max Humidity": "humidity_max_pct",
        "Average Humidity": "humidity_avg_pct",
    }
    daily = daily.rename(columns=rename)

    for optional in ["feed_cumulative_bags", "feed_daily_kg_per_bird", *ENVIRONMENT_FIELDS]:
        if optional not in daily:
            daily[optional] = pd.NA

    daily["cycle_id"] = daily["cycle_id"].astype("string").str.strip()
    daily["building_id"] = daily["building_id"].map(_normalize_building)
    daily["record_date"] = pd.to_datetime(daily["record_date"], errors="coerce").dt.normalize()
    _as_numeric(
        daily,
        [
            "age_day",
            "beginning_inventory",
            "population",
            "mortality_daily",
            "mortality_cumulative",
            "feed_daily_bags",
            "feed_cumulative_bags",
            "feed_daily_kg_per_bird",
            "bodyweight_kg",
            *ENVIRONMENT_FIELDS,
        ],
    )

    key = ["cycle_id", "building_id", "age_day"]
    missing_key_mask = daily[key + ["record_date"]].isna().any(axis=1)
    missing_key_rows = int(missing_key_mask.sum())
    if missing_key_rows:
        daily = daily.loc[~missing_key_mask].copy()

    daily["age_day"] = daily["age_day"].astype(int)
    daily["mortality_recorded"] = daily["mortality_daily"].notna()
    daily["feed_recorded"] = daily["feed_daily_bags"].notna()
    daily["weight_measured"] = daily["bodyweight_kg"].notna()

    source_rows = len(daily)
    group_sizes = daily.groupby(key, dropna=False).size()
    duplicate_keys = int((group_sizes > 1).sum())
    unique_source_keys = int(len(group_sizes))
    duplicate_rows_consolidated = int(source_rows - unique_source_keys)

    conflict_keys: list[tuple[object, ...]] = []
    for group_key, group in daily.groupby(key, dropna=False, sort=False):
        conflicting = False
        for column in PRODUCTION_FIELDS:
            values = group[column].dropna().unique()
            if len(values) > 1:
                conflicting = True
                break
        if conflicting:
            conflict_keys.append(group_key)

    aggregations: dict[str, object] = {
        column: _first_valid for column in PRODUCTION_FIELDS
    }
    aggregations.update(
        {
            "mortality_recorded": "max",
            "feed_recorded": "max",
            "weight_measured": "max",
            "temperature_min_c": "min",
            "temperature_max_c": "max",
            "temperature_avg_c": "mean",
            "humidity_min_pct": "min",
            "humidity_max_pct": "max",
            "humidity_avg_pct": "mean",
        }
    )
    canonical = daily.groupby(key, as_index=False, sort=False).agg(aggregations)
    if raw_environment is not None:
        environment = _prepare_environment(raw_environment)
        if not environment.empty:
            environment_columns = [
                *ENVIRONMENT_FIELDS,
                "environment_record_date",
                "environment_section_count",
                "environment_sections",
                "zone_aggregated",
                "temperature_zone_spread_c",
                "humidity_zone_spread_pct",
                "environment_aggregation_method",
            ]
            canonical = canonical.drop(columns=ENVIRONMENT_FIELDS).merge(
                environment[key + environment_columns],
                on=key,
                how="left",
                validate="one_to_one",
            )
    if "environment_section_count" not in canonical:
        canonical["environment_section_count"] = canonical[ENVIRONMENT_FIELDS].notna().any(axis=1).astype(int)
        canonical["environment_sections"] = "Building-level"
        canonical["zone_aggregated"] = canonical["source_row_count"] > 1 if "source_row_count" in canonical else False
        canonical["temperature_zone_spread_c"] = pd.NA
        canonical["humidity_zone_spread_pct"] = pd.NA
        canonical["environment_aggregation_method"] = "Daily-sheet fallback aggregation"
    canonical["temperature_range_c"] = (
        canonical["temperature_max_c"] - canonical["temperature_min_c"]
    )
    canonical["humidity_range_pct"] = (
        canonical["humidity_max_pct"] - canonical["humidity_min_pct"]
    )
    canonical["source_row_count"] = canonical[key].merge(
        group_sizes.rename("source_row_count").reset_index(), on=key, how="left"
    )["source_row_count"]
    canonical["had_source_duplicates"] = canonical["source_row_count"] > 1
    canonical["operational_recorded"] = (
        canonical["mortality_recorded"] | canonical["feed_recorded"]
    )
    canonical["daily_complete"] = (
        canonical["mortality_recorded"] & canonical["feed_recorded"]
    )

    canonical = canonical.sort_values(key).reset_index(drop=True)
    canonical_duplicate_keys = int(canonical.duplicated(key).sum())

    blocking_errors: list[str] = []
    if missing_key_rows:
        blocking_errors.append(
            f"{missing_key_rows} daily row(s) have a missing cycle, building, age, or date."
        )
    if conflict_keys:
        blocking_errors.append(
            f"{len(conflict_keys)} duplicated building-day key(s) contain conflicting production values."
        )
    if canonical_duplicate_keys:
        blocking_errors.append(
            f"Canonical processing left {canonical_duplicate_keys} duplicate building-day key(s)."
        )

    warnings: list[str] = []
    if duplicate_rows_consolidated:
        warnings.append(
            f"Consolidated {duplicate_rows_consolidated} repeated source row(s) created by multiple environmental sections. Section averages use an unweighted mean because section-size weights are unavailable."
        )
    operationally_missing = int((~canonical["operational_recorded"]).sum())
    if operationally_missing:
        warnings.append(
            f"{operationally_missing} building-day row(s) contain no recorded mortality or feed observation."
        )
    incomplete_daily = int((~canonical["daily_complete"]).sum())
    warnings.append(
        "Bodyweight is sampled rather than measured daily; the latest actual measurement is retained with its age."
    )

    report = DataQualityReport(
        source_rows=source_rows,
        canonical_rows=len(canonical),
        unique_source_keys=unique_source_keys,
        duplicate_keys=duplicate_keys,
        duplicate_rows_consolidated=duplicate_rows_consolidated,
        multi_environment_days=int(canonical["zone_aggregated"].fillna(False).sum()),
        zone_aggregated_days=int(canonical["zone_aggregated"].fillna(False).sum()),
        maximum_environment_sections=int(canonical["environment_section_count"].fillna(0).max()),
        production_conflict_keys=len(conflict_keys),
        operationally_missing_days=operationally_missing,
        incomplete_daily_records=incomplete_daily,
        weight_measurement_days=int(canonical["weight_measured"].sum()),
        temperature_coverage_pct=float(canonical["temperature_avg_c"].notna().mean() * 100),
        humidity_coverage_pct=float(canonical["humidity_avg_pct"].notna().mean() * 100),
        blocking_errors=tuple(blocking_errors),
        warnings=tuple(warnings),
    )
    return canonical, report


def _prepare_cycles(raw: pd.DataFrame) -> pd.DataFrame:
    cycles = raw.copy().map(_normalize_blank).rename(
        columns={
            "Harvest Cycle": "cycle_id",
            "Bldg.": "building_id",
            "Start Date": "start_date",
            "End Date": "end_date",
            "Beginning Inventory": "beginning_inventory",
            "Ending Inventory": "ending_inventory",
            "Harvest Recovery": "final_recovery_rate",
            "Ending Weight (As of Week 5)": "ending_weight_week5_kg",
        }
    )
    cycles["cycle_id"] = cycles["cycle_id"].astype("string").str.strip()
    cycles["building_id"] = cycles["building_id"].map(_normalize_building)
    for column in ["start_date", "end_date"]:
        cycles[column] = pd.to_datetime(cycles[column], errors="coerce").dt.normalize()
    _as_numeric(
        cycles,
        [
            "beginning_inventory",
            "ending_inventory",
            "final_recovery_rate",
            "ending_weight_week5_kg",
        ],
    )
    key = ["cycle_id", "building_id"]
    duplicate_count = int(cycles.duplicated(key).sum())
    if duplicate_count:
        raise WorkbookValidationError(
            f"{CYCLE_SHEET} contains {duplicate_count} duplicate cycle-building record(s)."
        )
    required_nulls = cycles[key + ["start_date", "end_date", "beginning_inventory"]].isna().any(axis=1)
    if required_nulls.any():
        raise WorkbookValidationError(
            f"{CYCLE_SHEET} contains {int(required_nulls.sum())} incomplete required record(s)."
        )
    return cycles.sort_values(key).reset_index(drop=True)


def _prepare_targets(raw: pd.DataFrame, maximum_age: int) -> pd.DataFrame:
    targets = raw.copy().map(_normalize_blank).rename(
        columns={
            "Age": "age_day",
            "Target Weight (Scaled Interpolation)": "target_weight_scaled_g",
            "Target Weight (Linear Interpolation)": "target_weight_linear_g",
        }
    )
    _as_numeric(targets, ["age_day", "target_weight_scaled_g", "target_weight_linear_g"])
    targets = targets.dropna(subset=["age_day", "target_weight_scaled_g"]).copy()
    targets["age_day"] = targets["age_day"].astype(int)
    targets = targets.drop_duplicates("age_day", keep="last").sort_values("age_day")

    legacy_scaled = targets.set_index("age_day")["target_weight_scaled_g"].copy()

    # Farm-approved checkpoint targets supersede the legacy checkpoint values.
    # We retain two daily views: a straight-line interpolation and a smoothed
    # curve that reuses the former farm curve's within-week proportional shape.
    # Both hit every approved checkpoint exactly and remain flat after Day 35.
    checkpoint_days = np.asarray(list(APPROVED_WEIGHT_TARGETS_G), dtype=float)
    checkpoint_weights = np.asarray(list(APPROVED_WEIGHT_TARGETS_G.values()), dtype=float)
    target_days = np.arange(0, max(maximum_age, 35) + 1, dtype=int)
    linear = np.interp(target_days, checkpoint_days, checkpoint_weights)
    smoothed = linear.copy()
    checkpoint_items = list(APPROVED_WEIGHT_TARGETS_G.items())
    for (start_day, start_weight), (end_day, end_weight) in zip(
        checkpoint_items[:-1], checkpoint_items[1:]
    ):
        segment_days = np.arange(start_day, end_day + 1, dtype=int)
        if start_day in legacy_scaled and end_day in legacy_scaled:
            old_start = float(legacy_scaled.loc[start_day])
            old_end = float(legacy_scaled.loc[end_day])
            if old_end > old_start:
                progress = (
                    legacy_scaled.reindex(segment_days).interpolate().to_numpy(float)
                    - old_start
                ) / (old_end - old_start)
            else:
                progress = (segment_days - start_day) / (end_day - start_day)
        else:
            progress = (segment_days - start_day) / (end_day - start_day)
        smoothed[segment_days] = start_weight + progress * (end_weight - start_weight)
    targets = pd.DataFrame(
        {
            "age_day": target_days,
            "target_weight_scaled_g": np.rint(smoothed),
            "target_weight_linear_g": np.rint(linear),
        }
    )
    targets["target_source"] = np.select(
        [
            targets["age_day"].eq(0),
            targets["age_day"].isin([7, 14, 21, 28, 35]),
            targets["age_day"].between(1, 34),
        ],
        [
            "Working placement anchor retained from former farm curve",
            "Farm-approved revised checkpoint",
            "Estimated daily target; former curve shape rescaled between approved checkpoints",
        ],
        default="Day 35 target carried forward for post-milestone monitoring",
    )
    targets["daily_gain_scaled_g"] = targets["target_weight_scaled_g"].diff()
    targets["daily_gain_linear_g"] = targets["target_weight_linear_g"].diff()
    targets["target_weight_kg"] = targets["target_weight_scaled_g"] / 1000.0
    return targets.reset_index(drop=True)


def load_workbook(
    source: str | Path | bytes | bytearray | BinaryIO,
    source_name: str | None = None,
) -> CanaryDataset:
    """Read and validate the farm workbook without changing the source file."""

    read_source: object
    if isinstance(source, (bytes, bytearray)):
        read_source = BytesIO(source)
    else:
        read_source = source

    try:
        book = pd.read_excel(read_source, sheet_name=None, engine="openpyxl")
    except Exception as exc:  # pragma: no cover - exact engine errors vary
        raise WorkbookValidationError(f"Could not read the workbook: {exc}") from exc

    _validate_sheets(book)
    daily, quality = _prepare_daily(book[DAILY_SHEET], book.get(TEMPERATURE_SHEET))
    cycles = _prepare_cycles(book[CYCLE_SHEET])
    targets = _prepare_targets(book[TARGET_SHEET], int(daily["age_day"].max()))

    if source_name is None:
        source_name = Path(source).name if isinstance(source, (str, Path)) else "Uploaded workbook.xlsx"

    return CanaryDataset(
        daily=daily,
        cycles=cycles,
        targets=targets,
        quality=quality,
        source_name=source_name,
    )
