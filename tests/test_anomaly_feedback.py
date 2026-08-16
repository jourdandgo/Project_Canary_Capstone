from __future__ import annotations

from pathlib import Path

from canary.anomaly import build_age_adjusted_anomalies
from canary.data import load_workbook
from canary.feedback import load_alert_feedback, record_alert_feedback


ROOT = Path(__file__).resolve().parents[1]


def test_anomalies_are_separate_non_probability_signals() -> None:
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    as_of = dataset.daily.loc[
        dataset.daily["cycle_id"].astype(str).eq("2026-3")
        & dataset.daily["building_id"].astype(str).eq("Tags 1"), "record_date"
    ].max()
    signals = build_age_adjusted_anomalies(dataset, "2026-3", "Tags 1", as_of)
    assert signals
    assert all(signal["changes_risk_score"] is False for signal in signals)
    assert all(signal["status"] in {"Unavailable", "No signal", "Watch", "Warning"} for signal in signals)


def test_feedback_ledger_appends_atomically(tmp_path: Path) -> None:
    path = tmp_path / "feedback.csv"
    first = record_alert_feedback(
        path, cycle_id="2026-3", building_id="Tags 1", as_of_date="2026-04-10",
        signal_id="mortality-ewma", assessment="Confirmed", responsible_person="Tester",
    )
    record_alert_feedback(
        path, cycle_id="2026-3", building_id="Tags 2", as_of_date="2026-04-10",
        signal_id="temperature-cusum", assessment="Dismissed",
    )
    ledger = load_alert_feedback(path)
    assert len(ledger) == 2
    assert ledger.iloc[0]["feedback_id"] == first["feedback_id"]
    assert set(ledger["assessment"]) == {"Confirmed", "Dismissed"}
