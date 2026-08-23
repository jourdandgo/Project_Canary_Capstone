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
        version = str(updated["version"]).replace("-draft", "-approved")
        updated["version"] = version.replace("-doc-validation", "-approved")
    else:
        updated["version"] = str(updated["version"]).replace("-approved", "-draft")
    validate_recommendation_playbook(updated)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _selected_patterns(row: pd.Series) -> list[str]:
    raw = str(row.get("risk_patterns", row.get("risk_pattern", "No Material Concern")))
    patterns = [pattern.strip() for pattern in raw.split("|") if pattern.strip()]
    if not patterns:
        patterns = [str(row.get("risk_pattern", "No Material Concern"))]
    evidence = str(row.get("evidence_status", ""))
    evidence_missing = evidence.startswith("Insufficient") or evidence == "Reduced evidence"
    if evidence_missing and "Missing or Stale Evidence" not in patterns:
        patterns.append("Missing or Stale Evidence")
    if evidence_missing and patterns == ["No Material Concern", "Missing or Stale Evidence"]:
        patterns = ["Missing or Stale Evidence"]
    return list(dict.fromkeys(patterns))


def _guidance_status(rule: dict[str, Any], playbook: dict[str, Any]) -> str:
    if rule["approval_status"] == "Approved" and playbook["approval_status"].startswith("Approved"):
        return "Farm-approved guidance"
    if rule["approval_status"] == "Rejected":
        return "Rejected — no recommendation in use"
    return "Preliminary guidance — pending Doc Raymond review"


def _rule_source(rule: dict[str, Any]) -> str:
    """State whether the operational basis came from the farm or is a safety fallback."""

    if rule["rule_id"] in {"DOC-001", "DOC-011"}:
        return "Canary team safeguard"
    return "Farmer Validation Workbook (Doc Raymond)"


def _action_for_rule(rule: dict[str, Any], playbook: dict[str, Any]) -> str:
    if rule["approval_status"] == "Rejected":
        return "No recommendation is active for this pattern. Ask the farm owner to review the rule."
    if rule["approval_status"] == "Approved" and rule["approved_wording"].strip():
        return rule["approved_wording"].strip()
    return rule["dashboard_action"].strip()


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
                    "recommendation_patterns": "Not applicable",
                    "recommendation_rule_ids": "Not applicable",
                    "recommendation_match_count": 0,
                    "recommendation_matches_json": "[]",
                    "additional_recommended_actions": "",
                    "recommendation_urgency": "Not applicable",
                    "recommendation_urgency_instruction": "Not applicable",
                    "recommendation_inspection_checklist": "Not applicable",
                    "recommendation_escalation_trigger": "Not applicable",
                    "recommendation_rule_version": playbook["version"],
                    "recommendation_approval_status": playbook["approval_status"],
                    "recommendation_rule_approval": "Not applicable",
                    "recommendation_guidance_status": "Not applicable",
                    "recommendation_source": "Not applicable",
                    "recommendation_wording_provenance": "Not applicable",
                }
            )
            records.append(row)
            continue
        patterns = _selected_patterns(source)
        missing_patterns = [pattern for pattern in patterns if pattern not in rule_by_pattern]
        if missing_patterns:
            raise RecommendationConfigurationError(
                "No recommendation rule is configured for pattern(s): " + ", ".join(missing_patterns)
            )
        matched_rules = [rule_by_pattern[pattern] for pattern in patterns]
        rule = matched_rules[0]
        pattern = patterns[0]
        rating = str(source.get("risk_rating", "Not rated"))
        severity = severity_by_rating.get(
            rating,
            {
                "urgency": "Before a major decision",
                "owner_instruction": "Update the missing evidence as soon as practical.",
            },
        )
        guidance_status = _guidance_status(rule, playbook)
        action = _action_for_rule(rule, playbook)
        matches = [
            {
                "pattern": matched["pattern"],
                "rule_id": matched["rule_id"],
                "action": _action_for_rule(matched, playbook),
                "inspection_checklist": matched["inspection_checklist"],
                "escalation_trigger": matched["escalation_trigger"],
                "possible_causes": matched.get("possible_causes", "Not specified"),
                "response_time": matched.get("response_time", severity["urgency"]),
                "responsible_person": matched.get("responsible_person", "Farm manager"),
                "approval_status": matched["approval_status"],
                "source": matched.get("source", _rule_source(matched)),
            }
            for matched in matched_rules
        ]

        row.update(
            {
                "recommended_action": action,
                "recommendation_rule_id": rule["rule_id"],
                "recommendation_pattern": pattern,
                "recommendation_patterns": " | ".join(patterns),
                "recommendation_rule_ids": " | ".join(match["rule_id"] for match in matches),
                "recommendation_match_count": len(matches),
                "recommendation_matches_json": json.dumps(matches),
                "additional_recommended_actions": " | ".join(
                    f"{match['pattern']}: {match['action']}" for match in matches[1:]
                ),
                "recommendation_urgency": severity["urgency"],
                "recommendation_urgency_instruction": severity["owner_instruction"],
                "recommendation_inspection_checklist": rule["inspection_checklist"],
                "recommendation_escalation_trigger": rule["escalation_trigger"],
                "recommendation_rule_version": playbook["version"],
                "recommendation_approval_status": playbook["approval_status"],
                "recommendation_rule_approval": rule["approval_status"],
                "recommendation_guidance_status": guidance_status,
                "recommendation_possible_causes": rule.get("possible_causes", "Not specified"),
                "recommendation_response_time": rule.get("response_time", severity["urgency"]),
                "recommendation_responsible_person": rule.get("responsible_person", "Farm manager"),
                "recommendation_source": rule.get("source", _rule_source(rule)),
                "recommendation_wording_provenance": rule.get(
                    "wording_provenance",
                    "Trigger and action basis retained; Canary team expanded the wording into checks and escalation guidance.",
                ),
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def build_recommendation_trace(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Decision element": "Primary identified pattern", "Applied value": row["recommendation_pattern"]},
            {"Decision element": "All detected patterns", "Applied value": row.get("recommendation_patterns", row["recommendation_pattern"])},
            {"Decision element": "Risk-level urgency", "Applied value": f"{row['risk_rating']} → {row['recommendation_urgency']}"},
            {"Decision element": "Action rule", "Applied value": row["recommendation_rule_id"]},
            {"Decision element": "All matched rules", "Applied value": row.get("recommendation_rule_ids", row["recommendation_rule_id"])},
            {"Decision element": "Rule version", "Applied value": row["recommendation_rule_version"]},
            {"Decision element": "Rule approval", "Applied value": row["recommendation_rule_approval"]},
            {"Decision element": "Overall status", "Applied value": row["recommendation_approval_status"]},
            {"Decision element": "Source", "Applied value": row.get("recommendation_source", "Not specified")},
            {"Decision element": "Wording provenance", "Applied value": row.get("recommendation_wording_provenance", "Not specified")},
            {"Decision element": "Possible causes to verify", "Applied value": row.get("recommendation_possible_causes", "Not specified")},
            {"Decision element": "Response time", "Applied value": row.get("recommendation_response_time", row["recommendation_urgency"])},
        ]
    )


def build_matched_recommendation_table(row: pd.Series | dict[str, object]) -> pd.DataFrame:
    """Return one auditable guidance row for every detected problem pattern."""

    source = pd.Series(row)
    try:
        matches = json.loads(str(source.get("recommendation_matches_json", "[]")))
    except json.JSONDecodeError:
        matches = []
    rows = []
    for index, match in enumerate(matches):
        rows.append(
            {
                "Role": "Primary" if index == 0 else "Additional",
                "Problem pattern": match["pattern"],
                "Matched rule": match["rule_id"],
                "Recommended inspection": match["action"],
                "Detailed checklist": match["inspection_checklist"],
                "Escalate when": match["escalation_trigger"],
                "Responsible function": match["responsible_person"],
                "Guidance status": match["approval_status"],
            }
        )
    return pd.DataFrame(rows)
