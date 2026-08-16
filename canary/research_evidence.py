"""Read-only access to the finalized farm-wide modeling evidence.

The application deliberately keeps these research artifacts separate from the
operational model bundles.  This module only prepares evidence for display; it
does not change application inference or promote a challenger.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESEARCH_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "colab_capstone_refresh_latest"
)
DEFAULT_PROSPECTIVE_STATUS = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "prospective_shadow_validation"
    / "status.json"
)


@dataclass(frozen=True)
class OutcomeResearchEvidence:
    """Frozen held-out evidence for one modeled outcome."""

    outcome: str
    root: Path
    manifest: dict
    top_five: pd.DataFrame
    checkpoint_metrics: pd.DataFrame
    predictions: pd.DataFrame
    shap_global: pd.DataFrame
    shap_local: pd.DataFrame

    @property
    def challenger(self) -> str:
        return str(self.manifest["selection"]["lowest_error_candidate"])

    @property
    def one_se_selection(self) -> str:
        return str(self.manifest["selection"]["selected_candidate"])

    @property
    def selected_metrics(self) -> pd.Series:
        rows = self.top_five.loc[self.top_five["candidate"] == self.one_se_selection]
        if rows.empty:
            raise ValueError(f"Missing top-five metrics for {self.outcome} selection")
        return rows.iloc[0]

    @property
    def explanation_model(self) -> str:
        return str(self.manifest["explanation_model"])

    @property
    def challenger_metrics(self) -> pd.Series:
        rows = self.top_five.loc[self.top_five["candidate"] == self.challenger]
        if rows.empty:
            raise ValueError(f"Missing top-five metrics for {self.outcome} challenger")
        return rows.iloc[0]

    @property
    def challenger_predictions(self) -> pd.DataFrame:
        rows = self.predictions.loc[
            (self.predictions["candidate"] == self.challenger)
            & (self.predictions["validation_view"] == "cycle")
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing held-out predictions for {self.outcome} challenger")
        return rows

    @property
    def challenger_checkpoints(self) -> pd.DataFrame:
        rows = self.checkpoint_metrics.loc[
            self.checkpoint_metrics["candidate"] == self.challenger
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing checkpoint metrics for {self.outcome} challenger")
        return rows.sort_values("review_day")

    @property
    def selected_predictions(self) -> pd.DataFrame:
        rows = self.predictions.loc[
            (self.predictions["candidate"] == self.one_se_selection)
            & (self.predictions["validation_view"] == "cycle")
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing held-out predictions for {self.outcome} selection")
        return rows

    @property
    def selected_checkpoints(self) -> pd.DataFrame:
        rows = self.checkpoint_metrics.loc[
            self.checkpoint_metrics["candidate"] == self.one_se_selection
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing checkpoint metrics for {self.outcome} selection")
        return rows.sort_values("review_day")

    @property
    def top_shap(self) -> pd.DataFrame:
        rows = self.shap_global.loc[
            self.shap_global["candidate"] == self.explanation_model
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing SHAP evidence for {self.outcome} explanation model")
        return rows.sort_values("mean_abs_shap", ascending=False).head(10)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required research artifact is missing: {path}")
    return path


@lru_cache(maxsize=4)
def load_outcome_research_evidence(
    outcome: str, root: str | Path = DEFAULT_RESEARCH_ROOT
) -> OutcomeResearchEvidence:
    """Load and validate one frozen outcome evidence bundle."""

    if outcome not in {"recovery", "bodyweight"}:
        raise ValueError("outcome must be 'recovery' or 'bodyweight'")
    root_path = Path(root).resolve()
    outcome_root = root_path / outcome
    manifest = json.loads(_require(outcome_root / "manifest.json").read_text(encoding="utf-8"))
    required_manifest_keys = {
        "round_version",
        "selection",
        "promotion_gate",
        "explanation_model",
        "primary_validation",
        "operational_models_changed",
    }
    missing = required_manifest_keys.difference(manifest)
    if missing:
        raise ValueError(f"Research manifest is missing keys: {sorted(missing)}")
    manifest_outcome = "bodyweight" if manifest["outcome"] == "weight" else manifest["outcome"]
    if manifest_outcome != outcome:
        raise ValueError(f"Research manifest outcome mismatch: {manifest['outcome']}")

    evidence = OutcomeResearchEvidence(
        outcome=outcome,
        root=outcome_root,
        manifest=manifest,
        top_five=pd.read_csv(_require(outcome_root / "top_five_models.csv")),
        checkpoint_metrics=pd.read_csv(_require(outcome_root / "checkpoint_metrics.csv")),
        predictions=pd.read_csv(_require(outcome_root / "all_nested_logo_predictions.csv")),
        shap_global=pd.read_csv(_require(outcome_root / "held_out_shap_global.csv")),
        shap_local=pd.read_csv(_require(outcome_root / "held_out_shap_local.csv")),
    )
    # Exercise the important joins during loading so the UI fails clearly if a
    # future export becomes incomplete or inconsistent.
    _ = evidence.challenger_metrics
    _ = evidence.challenger_predictions
    _ = evidence.challenger_checkpoints
    _ = evidence.selected_metrics
    _ = evidence.selected_predictions
    _ = evidence.selected_checkpoints
    _ = evidence.top_shap
    return evidence


def display_name(candidate: str) -> str:
    """Convert registry identifiers into concise reader-facing model names."""

    acronyms = {"pls": "PLS", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    words = str(candidate).split("_")
    return " ".join(acronyms.get(word, word.capitalize()) for word in words)


def feature_display_name(feature: str) -> str:
    """Convert feature identifiers into readable labels without changing meaning."""

    replacements = {
        "pct": "%",
        "g": "(g)",
        "kg": "(kg)",
        "ewm": "EWMA",
        "adg": "ADG",
    }
    return " ".join(replacements.get(word, word.capitalize()) for word in str(feature).split("_"))


def load_prospective_shadow_status(path: str | Path = DEFAULT_PROSPECTIVE_STATUS) -> dict:
    """Read the research ledger progress without creating or mutating it."""

    status_path = Path(path)
    if not status_path.exists():
        return {
            "progress": {
                outcome: {"qualifying_cycles": 0, "required_cycles": 3, "remaining_cycles": 3}
                for outcome in ("recovery", "bodyweight")
            },
            "operational_models_changed": False,
        }
    return json.loads(status_path.read_text(encoding="utf-8"))
