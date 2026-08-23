"""Defense-demo CSV generation, validation, and session-safe overlays."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .data import CanaryDataset, WorkbookValidationError


DEMO_CYCLE = "2026-3"
DEMO_BUILDINGS = ("Tags 1", "Tags 2", "Tags 3")
DEMO_CUTOFFS = (7, 14, 15, 21, 28, 35)
DEMO_COLUMNS = [
    "cycle_id", "building_id", "age_day", "record_date",
    "beginning_inventory", "population", "mortality_daily",
    "mortality_cumulative", "feed_daily_bags", "feed_cumulative_bags",
    "feed_daily_kg_per_bird", "bodyweight_kg", "temperature_min_c",
    "temperature_max_c", "temperature_avg_c", "humidity_min_pct",
    "humidity_max_pct", "humidity_avg_pct",
]
COMPARE_COLUMNS = [column for column in DEMO_COLUMNS if column != "record_date"]
KEY_COLUMNS = ["cycle_id", "building_id", "age_day"]


@dataclass(frozen=True)
class ReplayValidation:
    valid: bool
    reason: str
    cycle_id: str | None = None
    cutoff_day: int | None = None
    cutoff_date: pd.Timestamp | None = None
    row_count: int = 0
    buildings: tuple[str, ...] = ()
    fingerprint: str | None = None


def _normalised(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame[DEMO_COLUMNS].copy()
    view["cycle_id"] = view["cycle_id"].astype("string").str.strip()
    view["building_id"] = view["building_id"].astype("string").str.strip()
    view["age_day"] = pd.to_numeric(view["age_day"], errors="coerce").astype("Int64")
    view["record_date"] = pd.to_datetime(view["record_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in DEMO_COLUMNS[4:]:
        view[column] = pd.to_numeric(view[column], errors="coerce").round(8)
    return view.sort_values(KEY_COLUMNS).reset_index(drop=True)


def cycle_prefix_fingerprint(frame: pd.DataFrame) -> str:
    payload = _normalised(frame).to_csv(index=False, na_rep="").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def demo_prefix(reference: CanaryDataset, cutoff_day: int) -> pd.DataFrame:
    cutoff = int(cutoff_day)
    if cutoff not in DEMO_CUTOFFS:
        raise ValueError(f"Unsupported demo cutoff: Day {cutoff}")
    frame = reference.daily.loc[
        reference.daily["cycle_id"].astype(str).eq(DEMO_CYCLE)
        & reference.daily["building_id"].isin(DEMO_BUILDINGS)
        & reference.daily["age_day"].le(cutoff),
        DEMO_COLUMNS,
    ].copy()
    return _normalised(frame)


def validate_replay_prefix(frame: pd.DataFrame, reference: CanaryDataset) -> ReplayValidation:
    missing = [column for column in DEMO_COLUMNS if column not in frame.columns]
    if missing:
        return ReplayValidation(False, "Missing required column(s): " + ", ".join(missing))
    try:
        view = _normalised(frame)
    except Exception as exc:
        return ReplayValidation(False, f"The CSV could not be normalized: {exc}")
    if view.empty:
        return ReplayValidation(False, "The CSV contains no building-day records.")
    if view[KEY_COLUMNS].isna().any(axis=None):
        return ReplayValidation(False, "Cycle, building, and age must be present on every row.")
    cycles = tuple(view["cycle_id"].dropna().astype(str).unique())
    if cycles != (DEMO_CYCLE,):
        return ReplayValidation(False, "The defense replay accepts one cycle only: 2026-3.")
    buildings = tuple(sorted(view["building_id"].dropna().astype(str).unique()))
    if not set(buildings).issubset(DEMO_BUILDINGS) or set(buildings) != set(DEMO_BUILDINGS):
        return ReplayValidation(False, "2026-3 must contain exactly Tags 1, Tags 2, and Tags 3.")
    if view.duplicated(KEY_COLUMNS).any():
        return ReplayValidation(False, "Duplicate cycle-building-day rows were found.")
    cutoff = int(view["age_day"].max())
    if cutoff not in DEMO_CUTOFFS:
        return ReplayValidation(False, "Use one of the prepared cutoffs: Day 7, 14, 15, 21, 28, or 35.")
    for building, group in view.groupby("building_id"):
        ages = group["age_day"].astype(int).tolist()
        if ages != list(range(1, cutoff + 1)):
            return ReplayValidation(False, f"{building} must contain continuous Days 1 through {cutoff}.")
    expected = demo_prefix(reference, cutoff)
    if len(view) != len(expected):
        return ReplayValidation(False, f"Day {cutoff} should contain {len(expected)} rows; received {len(view)}.")
    if not view[KEY_COLUMNS + ["record_date"]].equals(expected[KEY_COLUMNS + ["record_date"]]):
        return ReplayValidation(False, "Cycle, building, age, or date does not match the source-backed 2026-3 replay.")
    for column in COMPARE_COLUMNS[3:]:
        left = pd.to_numeric(view[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=1e-7, atol=1e-7, equal_nan=True):
            return ReplayValidation(False, f"{column} does not match the source-backed 2026-3 prefix.")
    cutoff_date = pd.to_datetime(view["record_date"]).max().normalize()
    return ReplayValidation(
        True,
        f"Validated source-backed 2026-3 replay through Day {cutoff}",
        DEMO_CYCLE,
        cutoff,
        cutoff_date,
        len(view),
        buildings,
        cycle_prefix_fingerprint(view),
    )


def _canonicalise_csv(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index, columns=list(columns))
    for column in DEMO_COLUMNS:
        output[column] = frame[column]
    output["record_date"] = pd.to_datetime(output["record_date"]).dt.normalize()
    output["age_day"] = output["age_day"].astype(int)
    for column in DEMO_COLUMNS[4:]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["mortality_recorded"] = output["mortality_daily"].notna()
    output["feed_recorded"] = output["feed_daily_bags"].notna()
    output["weight_measured"] = output["bodyweight_kg"].notna()
    env = output[["temperature_min_c", "temperature_max_c", "temperature_avg_c", "humidity_min_pct", "humidity_max_pct", "humidity_avg_pct"]].notna().any(axis=1)
    output["environment_record_date"] = output["record_date"].where(env)
    output["environment_section_count"] = env.astype(int)
    output["environment_sections"] = np.where(env, "Building-level", "")
    output["zone_aggregated"] = False
    output["environment_aggregation_method"] = np.where(env, "Single building reading", "No reading")
    output["temperature_range_c"] = output["temperature_max_c"] - output["temperature_min_c"]
    output["humidity_range_pct"] = output["humidity_max_pct"] - output["humidity_min_pct"]
    output["source_row_count"] = 1
    output["had_source_duplicates"] = False
    output["operational_recorded"] = output[["population", "mortality_daily", "feed_daily_bags"]].notna().any(axis=1)
    output["daily_complete"] = output["mortality_recorded"] & output["feed_recorded"]
    return output


def baseline_without_cycle(reference: CanaryDataset, cycle_id: str = DEMO_CYCLE) -> CanaryDataset:
    return replace(
        reference,
        daily=reference.daily.loc[~reference.daily["cycle_id"].astype(str).eq(str(cycle_id))].copy(),
        cycles=reference.cycles.loc[~reference.cycles["cycle_id"].astype(str).eq(str(cycle_id))].copy(),
        source_name=f"Historical baseline through 2026-2",
        replay_validated=False,
        replay_status="No current-cycle replay loaded",
        replay_cutoff_day=None,
        replay_fingerprint=None,
    )


def merge_replay_csv(content: bytes, name: str, reference: CanaryDataset) -> tuple[CanaryDataset, ReplayValidation]:
    try:
        frame = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise WorkbookValidationError(f"Could not read the CSV: {exc}") from exc
    validation = validate_replay_prefix(frame, reference)
    if not validation.valid:
        raise WorkbookValidationError(validation.reason)
    baseline = baseline_without_cycle(reference)
    canonical = _canonicalise_csv(_normalised(frame), reference.daily.columns)
    daily = pd.concat([baseline.daily, canonical], ignore_index=True).sort_values(KEY_COLUMNS).reset_index(drop=True)
    cycle_rows = reference.cycles.loc[
        reference.cycles["cycle_id"].astype(str).eq(DEMO_CYCLE)
        & reference.cycles["building_id"].isin(DEMO_BUILDINGS)
    ].copy()
    cycle_rows["end_date"] = validation.cutoff_date
    cycle_rows["ending_inventory"] = np.nan
    cycle_rows["final_recovery_rate"] = np.nan
    cycle_rows["ending_weight_week5_kg"] = np.nan
    cycles = pd.concat([baseline.cycles, cycle_rows], ignore_index=True).sort_values(["start_date", "building_id"]).reset_index(drop=True)
    return replace(
        baseline,
        daily=daily,
        cycles=cycles,
        source_name=name,
        source_sha256=hashlib.sha256(content).hexdigest(),
        replay_validated=True,
        replay_status=validation.reason,
        replay_cutoff_day=validation.cutoff_day,
        replay_fingerprint=validation.fingerprint,
    ), validation


def write_demo_bundle(reference: CanaryDataset, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for cutoff in DEMO_CUTOFFS:
        frame = demo_prefix(reference, cutoff)
        name = f"Project_Canary_2026-3_Day_{cutoff:02d}.csv"
        path = output / name
        frame.to_csv(path, index=False, na_rep="")
        entries.append({
            "file": name,
            "cycle": DEMO_CYCLE,
            "cutoff_day": cutoff,
            "cutoff_date": str(pd.to_datetime(frame["record_date"]).max().date()),
            "row_count": len(frame),
            "buildings": list(DEMO_BUILDINGS),
            "source": "FARM HARVEST DATA.xlsx · canonical 2026-3 building-day records",
            "normalized_sha256": cycle_prefix_fingerprint(frame),
        })
    manifest = {"schema_version": "canary-demo-csv-v1", "entries": entries, "columns": DEMO_COLUMNS}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = output / "Project_Canary_2026-3_Demo_Checkpoints.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for item in entries:
            archive.write(output / item["file"], item["file"])
        archive.write(output / "manifest.json", "manifest.json")
    return manifest
