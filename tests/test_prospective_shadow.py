import json
from pathlib import Path

import pandas as pd
import pytest

from canary.prospective_shadow import (
    PREDICTION_COLUMNS,
    capture_cycle,
    initialize_shadow_ledger,
    shadow_status,
)


SOURCE = Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"


def test_shadow_ledger_initializes_at_zero_without_changing_models(tmp_path):
    output = initialize_shadow_ledger(tmp_path)
    status = shadow_status(output)
    assert status["operational_models_changed"] is False
    assert status["progress"]["recovery"]["qualifying_cycles"] == 0
    assert status["progress"]["bodyweight"]["qualifying_cycles"] == 0
    assert pd.read_csv(output / "prospective_predictions.csv").columns.tolist() == PREDICTION_COLUMNS
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["frozen_audit_cycle"] == "2026-3"
    assert protocol["required_qualifying_cycles_per_outcome"] == 3


def test_frozen_audit_cycle_cannot_be_counted(tmp_path):
    with pytest.raises(ValueError, match="not a cycle later"):
        capture_cycle(
            SOURCE,
            "2026-3",
            recovery_endpoint_confirmed=True,
            bodyweight_endpoint_confirmed=True,
            output=tmp_path,
        )
