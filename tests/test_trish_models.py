import pytest

from canary.trish_v19 import (
    load_v19_manifest,
    validate_v19_bundle,
    v19_calculation_trace,
    v19_global_drivers,
    v19_input_trace,
    v19_outlook,
)


def test_v19_registry_contains_only_the_two_final_models():
    manifest = load_v19_manifest()

    assert manifest["bundle_version"] == "trish-v19-final-2026-08-22"
    assert set(manifest["models"]) == {"model_1", "model_3"}
    assert manifest["models"]["model_1"]["feature_count"] == 85
    assert manifest["models"]["model_3"]["feature_count"] == 85


def test_v19_artifacts_and_source_evidence_pass_integrity_checks():
    checks = validate_v19_bundle()

    assert {row["model_id"] for row in checks} == {"model_1", "model_3"}
    assert all(row["artifact_exists"] for row in checks)
    assert all(row["artifact_hash_matches"] for row in checks)
    assert all(row["feature_dataset_exists"] for row in checks)
    assert all(row["oof_predictions_exist"] for row in checks)


@pytest.mark.parametrize(
    "model_id,day,expected_evidence_day",
    [
        ("model_1", 7, 7),
        ("model_1", 21, 14),
        ("model_3", 7, 7),
        ("model_3", 15, 14),
        ("model_3", 28, 21),
    ],
)
def test_v19_replay_routes_to_the_correct_evidence_day(model_id, day, expected_evidence_day):
    result = v19_outlook(model_id, "2026-3", "Tags 1", day)

    assert result is not None
    assert result["evidence_day"] == expected_evidence_day
    assert result["source_type"] == "Saved leave-one-building-flock-out prediction"


def test_v19_trace_is_exact_and_model_ready():
    result = v19_outlook("model_3", "2026-3", "Tags 1", 21)
    assert result is not None

    inputs = v19_input_trace(result)
    calculation = v19_calculation_trace(result)
    drivers = v19_global_drivers("model_3")

    assert len(inputs) == 85
    assert len(calculation) == 5
    assert not drivers.empty
    assert {"Feature", "Value supplied", "Evidence cutoff"}.issubset(inputs.columns)
    assert calculation["Step"].iloc[0].startswith("1 · Select evidence")
