"""Registry and deployment gates for the teammate model handoff.

The registry deliberately does not unpickle or activate a model.  A serialized
estimator is only one part of an operational prediction pipeline: Canary also
needs the exact as-of feature builder, preprocessing values, and target
definition before a teammate artifact can replace a current forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "trish_model_registry.json"
DEFAULT_BUNDLE_DIR = ROOT / "models" / "trish_v18"


@dataclass(frozen=True)
class TrishModelEntry:
    model_id: str
    outcome: str
    owner_label: str
    role: str
    algorithm: str
    observation_window: str
    model_path: Path | None
    features_path: Path | None
    feature_count: int
    reported_mae: float
    reported_r2: float
    expected_sha256: str | None
    owner_visibility: str
    integration_status: str
    boundary: str

    @property
    def activation_ready(self) -> bool:
        return self.integration_status == "ready"


def _resolve_optional(base: Path, filename: object) -> Path | None:
    if filename is None:
        return None
    return (base / str(filename)).resolve()


def load_trish_registry(path: Path = DEFAULT_REGISTRY) -> tuple[dict[str, Any], list[TrishModelEntry]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = (path.parent / payload["source_folder"]).resolve()
    entries = [
        TrishModelEntry(
            model_id=item["model_id"],
            outcome=item["outcome"],
            owner_label=item["owner_label"],
            role=item["role"],
            algorithm=item["algorithm"],
            observation_window=item["observation_window"],
            model_path=_resolve_optional(source, item["model_file"]),
            features_path=_resolve_optional(source, item["features_file"]),
            feature_count=int(item["feature_count"]),
            reported_mae=float(item["reported_mae"]),
            reported_r2=float(item["reported_r2"]),
            expected_sha256=item["artifact_sha256"],
            owner_visibility=item["owner_visibility"],
            integration_status=item["integration_status"],
            boundary=item["boundary"],
        )
        for item in payload["models"]
    ]
    return payload, entries


def validate_trish_registry(path: Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    """Return non-executing artifact checks for every registered model."""

    _, entries = load_trish_registry(path)
    checks: list[dict[str, Any]] = []
    for entry in entries:
        model_exists = entry.model_path is not None and entry.model_path.is_file()
        features_exist = entry.features_path is not None and entry.features_path.is_file()
        digest = None
        if model_exists and entry.model_path is not None:
            digest = hashlib.sha256(entry.model_path.read_bytes()).hexdigest()
        checks.append(
            {
                "model_id": entry.model_id,
                "model_exists": model_exists,
                "features_exist": features_exist,
                "hash_matches": bool(
                    digest
                    and entry.expected_sha256
                    and digest == entry.expected_sha256
                ),
                "activation_ready": entry.activation_ready,
                "integration_status": entry.integration_status,
            }
        )
    return checks


def owner_model_plan(path: Path = DEFAULT_REGISTRY) -> list[TrishModelEntry]:
    """Return models intended for the owner experience, in operating order."""

    _, entries = load_trish_registry(path)
    visible_roles = {"primary", "automatic_refresh"}
    return [entry for entry in entries if entry.owner_visibility in visible_roles]


@lru_cache(maxsize=2)
def load_v18_manifest(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=12)
def _load_v18_model(model_id: str, bundle_dir: str = str(DEFAULT_BUNDLE_DIR)) -> object:
    bundle = Path(bundle_dir)
    manifest = load_v18_manifest(bundle)
    metadata = manifest["models"][model_id]
    path = bundle / metadata["model_file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise ValueError(f"Trish v18 artifact hash mismatch for {model_id}")
    return joblib.load(path)


@lru_cache(maxsize=12)
def _load_v18_feature_rows(
    model_id: str, bundle_dir: str = str(DEFAULT_BUNDLE_DIR)
) -> pd.DataFrame:
    bundle = Path(bundle_dir)
    manifest = load_v18_manifest(bundle)
    path = bundle / manifest["models"][model_id]["holdout_features_file"]
    return pd.read_csv(path)


def _v18_row(
    model_id: str,
    cycle_id: str,
    building_id: str,
    cycle_day: int,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
) -> tuple[pd.Series, dict[str, Any]] | None:
    bundle = Path(bundle_dir)
    manifest = load_v18_manifest(bundle)
    metadata = manifest["models"][model_id]
    if str(cycle_id) != str(manifest["holdout_cycle"]):
        return None
    day = min(max(int(cycle_day), 1), int(metadata["window_end"]))
    rows = _load_v18_feature_rows(model_id, str(bundle)).loc[
        lambda frame: frame["harvest_cycle"].astype(str).eq(str(cycle_id))
        & frame["bldg"].astype(str).eq(str(building_id))
        & frame["prediction_day"].astype(int).eq(day)
    ]
    if rows.empty:
        return None
    return rows.iloc[0], metadata


def predict_v18_outlooks(
    cycle_id: str,
    building_id: str,
    cycle_day: int,
    source_sha256: str | None = None,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
) -> dict[str, Any] | None:
    """Score one building-date only when the source workbook matches v18."""

    bundle = Path(bundle_dir)
    manifest = load_v18_manifest(bundle)
    if source_sha256 != manifest.get("source_workbook_sha256"):
        return None
    weight_model_id = "model_2" if int(cycle_day) <= 14 else "model_3"
    selected = ["model_1", weight_model_id, "model_4", "model_5", "model_6"]
    output: dict[str, Any] = {
        "bundle_version": manifest["bundle_version"],
        "holdout_cycle": manifest["holdout_cycle"],
    }
    for model_id in selected:
        located = _v18_row(model_id, cycle_id, building_id, cycle_day, bundle)
        if located is None:
            return None
        row, metadata = located
        features = json.loads(
            (bundle / metadata["features_file"]).read_text(encoding="utf-8")
        )
        frame = pd.DataFrame([{feature: row[feature] for feature in features}])
        prediction = float(
            np.asarray(
                _load_v18_model(model_id, str(bundle)).predict(frame), dtype=float
            ).reshape(-1)[0]
        )
        output[model_id] = {
            "prediction": prediction,
            "prediction_day": int(row["prediction_day"]),
            "algorithm": metadata["algorithm"],
            "label": metadata["label"],
            "reported_logo_mae": float(metadata["reported_logo_mae"]),
            "reported_logo_r2": float(metadata["reported_logo_r2"]),
        }
    output["weight_model_id"] = weight_model_id
    return output


def v18_local_contributions(
    model_id: str,
    cycle_id: str,
    building_id: str,
    cycle_day: int,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
) -> pd.DataFrame:
    """Return local SHAP associations for the selected owner-facing model."""

    bundle = Path(bundle_dir)
    located = _v18_row(model_id, cycle_id, building_id, cycle_day, bundle)
    if located is None:
        return pd.DataFrame()
    row, metadata = located
    features = json.loads(
        (bundle / metadata["features_file"]).read_text(encoding="utf-8")
    )
    frame = pd.DataFrame([{feature: row[feature] for feature in features}])
    model = _load_v18_model(model_id, str(bundle))
    if metadata["algorithm"] == "CatBoost":
        from catboost import Pool

        values = np.asarray(
            model.get_feature_importance(Pool(frame), type="ShapValues"), dtype=float
        )[0, :-1]
    else:
        import shap

        values = np.asarray(shap.TreeExplainer(model).shap_values(frame), dtype=float).reshape(-1)
    result = pd.DataFrame(
        {
            "feature": features,
            "value": [row[feature] for feature in features],
            "contribution": values,
        }
    )
    result["absolute_contribution"] = result["contribution"].abs()
    return result.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)
