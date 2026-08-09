"""Deterministic, versioned recommendation matching for Project Canary."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RECOMMENDATIONS_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "recommendation_playbook_draft.json"
)
REQUIRED_RULE_FIELDS = {
    "rule_id",
    "pattern",
    "dashboard_action",
    "inspection_checklist",
    "escalation_trigger",
    "approval_status",
    "owner_comments",
    "approved_wording",
    "approval_date",
}
ALLOWED_APPROVAL_STATUSES = {"Pending Review", "Approved", "Revise", "Rejected"}


class RecommendationConfigurationError(ValueError):
    """Raised when recommendation configuration is incomplete or unsafe."""


def validate_recommendation_playbook(playbook: dict[str, Any]) -> None:
    for key in ("version", "approval_status", "intended_use", "severity_guide", "rules"):
        if key not in playbook:
            raise RecommendationConfigurationError(f"Recommendation playbook is missing '{key}'.")
    if not playbook["rules"]:
        raise RecommendationConfigurationError("Recommendation playbook has no rules.")

    rule_ids: set[str] = set()
    patterns: set[str] = set()
    for rule in playbook["rules"]:
        missing = REQUIRED_RULE_FIELDS - set(rule)
        if missing:
            raise RecommendationConfigurationError(
                f"Rule {rule.get('rule_id', 'unknown')} is missing: {', '.join(sorted(missing))}."
            )
        if rule["rule_id"] in rule_ids or rule["pattern"] in patterns:
            raise RecommendationConfigurationError("Rule IDs and problem patterns must be unique.")
        if rule["approval_status"] not in ALLOWED_APPROVAL_STATUSES:
            raise RecommendationConfigurationError(
                f"Unsupported approval status: {rule['approval_status']}."
            )
        approval_date = str(rule["approval_date"]).strip()
        if approval_date:
            try:
                date.fromisoformat(approval_date)
            except ValueError as exc:
                raise RecommendationConfigurationError(
                    f"Rule {rule['rule_id']} approval date must use YYYY-MM-DD."
                ) from exc
        if rule["approval_status"] == "Approved" and not approval_date:
            raise RecommendationConfigurationError(
                f"Rule {rule['rule_id']} requires an approval date before it can be approved."
            )
        rule_ids.add(str(rule["rule_id"]))
        patterns.add(str(rule["pattern"]))

    severity = {str(item.get("risk_rating")): item for item in playbook["severity_guide"]}
    missing_ratings = {"Low", "Medium", "High", "Critical"} - set(severity)
    if missing_ratings:
        raise RecommendationConfigurationError(
            "Severity guide is missing: " + ", ".join(sorted(missing_ratings))
        )


def load_recommendation_playbook(
    path: str | Path = DEFAULT_RECOMMENDATIONS_PATH,
) -> dict[str, Any]:
    playbook = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_recommendation_playbook(playbook)
    return playbook


def _overall_approval_status(playbook: dict[str, Any]) -> str:
    statuses = [rule["approval_status"] for rule in playbook["rules"]]
    return (
        "Approved by Doc Raymond"
        if statuses and all(status == "Approved" for status in statuses)
        else "Pending Doc Raymond review"
    )


def save_recommendation_playbook(
    playbook: dict[str, Any],
    path: str | Path = DEFAULT_RECOMMENDATIONS_PATH,
) -> None:
    """Validate and atomically save an administrator-reviewed playbook."""

    updated = deepcopy(playbook)
    updated["approval_status"] = _overall_approval_status(updated)
    if updated["approval_status"].startswith("Approved"):
        updated["version"] = str(updated["version"]).replace("-draft", "-approved")
    else:
        updated["version"] = str(updated["version"]).replace("-approved", "-draft")
    validate_recommendation_playbook(updated)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _selected_pattern(row: pd.Series) -> str:
    pattern = str(row.get("risk_pattern", "No Material Drift"))
    evidence = str(row.get("evidence_status", ""))
    if evidence == "Insufficient" or (
        pattern == "No Material Drift" and evidence not in {"Complete", "Not eligible"}
    ):
        return "Missing or Stale Evidence"
    return pattern


def _guidance_status(rule: dict[str, Any], playbook: dict[str, Any]) -> str:
    if rule["approval_status"] == "Approved" and playbook["approval_status"].startswith("Approved"):
        return "Farm-approved guidance"
    if rule["approval_status"] == "Rejected":
        return "Rejected — no recommendation in use"
    return "Preliminary guidance — pending Doc Raymond review"


def apply_recommendations(
    snapshot: pd.DataFrame,
    playbook: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach an explainable action rule without changing risk or forecast values."""

    playbook = playbook or load_recommendation_playbook()
    validate_recommendation_playbook(playbook)
    rule_by_pattern = {rule["pattern"]: rule for rule in playbook["rules"]}
    severity_by_rating = {
        item["risk_rating"]: item for item in playbook["severity_guide"]
    }
    records: list[dict[str, object]] = []

    for _, source in snapshot.iterrows():
        row = source.to_dict()
        state = str(source.get("state", ""))
        if state == "Inactive":
            row.update(
                {
                    "recommended_action": "No flock is active in this building.",
                    "recommendation_rule_id": "Not applicable",
                    "recommendation_pattern": "Not applicable",
                    "recommendation_urgency": "Not applicable",
                    "recommendation_urgency_instruction": "Not applicable",
                    "recommendation_inspection_checklist": "Not applicable",
                    "recommendation_escalation_trigger": "Not applicable",
                    "recommendation_rule_version": playbook["version"],
                    "recommendation_approval_status": playbook["approval_status"],
                    "recommendation_rule_approval": "Not applicable",
                    "recommendation_guidance_status": "Not applicable",
                }
            )
            records.append(row)
            continue
        pattern = _selected_pattern(source)
        rule = rule_by_pattern.get(pattern)
        if rule is None:
            raise RecommendationConfigurationError(
                f"No recommendation rule is configured for pattern '{pattern}'."
            )
        rating = str(source.get("risk_rating", "Not rated"))
        severity = severity_by_rating.get(
            rating,
            {
                "urgency": "Before a major decision",
                "owner_instruction": "Update the missing evidence as soon as practical.",
            },
        )
        guidance_status = _guidance_status(rule, playbook)
        if rule["approval_status"] == "Rejected":
            action = "No recommendation is active for this pattern. Ask the farm owner to review the rule."
        elif rule["approval_status"] == "Approved" and rule["approved_wording"].strip():
            action = rule["approved_wording"].strip()
        else:
            action = rule["dashboard_action"].strip()

        row.update(
            {
                "recommended_action": action,
                "recommendation_rule_id": rule["rule_id"],
                "recommendation_pattern": pattern,
                "recommendation_urgency": severity["urgency"],
                "recommendation_urgency_instruction": severity["owner_instruction"],
                "recommendation_inspection_checklist": rule["inspection_checklist"],
                "recommendation_escalation_trigger": rule["escalation_trigger"],
                "recommendation_rule_version": playbook["version"],
                "recommendation_approval_status": playbook["approval_status"],
                "recommendation_rule_approval": rule["approval_status"],
                "recommendation_guidance_status": guidance_status,
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def build_recommendation_trace(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Decision element": "Identified pattern", "Applied value": row["recommendation_pattern"]},
            {"Decision element": "Risk-level urgency", "Applied value": f"{row['risk_rating']} → {row['recommendation_urgency']}"},
            {"Decision element": "Action rule", "Applied value": row["recommendation_rule_id"]},
            {"Decision element": "Rule version", "Applied value": row["recommendation_rule_version"]},
            {"Decision element": "Rule approval", "Applied value": row["recommendation_rule_approval"]},
            {"Decision element": "Overall status", "Applied value": row["recommendation_approval_status"]},
        ]
    )
