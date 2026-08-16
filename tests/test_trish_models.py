from canary.trish_models import (
    load_trish_registry,
    load_v18_manifest,
    owner_model_plan,
    predict_v18_outlooks,
    validate_trish_registry,
)


def test_registry_records_all_six_ready_models():
    payload, entries = load_trish_registry()

    assert payload["registry_version"] == "trish-v18-prospective-2026-3"
    assert [entry.model_id for entry in entries] == [
        "trish_model_1",
        "trish_model_2",
        "trish_model_3",
        "trish_model_4",
        "trish_model_5",
        "trish_model_6",
    ]
    assert all(entry.activation_ready for entry in entries)


def test_owner_plan_is_recovery_plus_day14_and_day21_weight():
    entries = owner_model_plan()

    assert [entry.model_id for entry in entries] == [
        "trish_model_1",
        "trish_model_2",
        "trish_model_3",
    ]


def test_registered_artifacts_are_present_and_hash_checked_where_expected():
    checks = validate_trish_registry()
    by_id = {row["model_id"]: row for row in checks}

    for model_id in ("trish_model_1", "trish_model_2", "trish_model_3", "trish_model_4", "trish_model_5", "trish_model_6"):
        assert by_id[model_id]["model_exists"]
        assert by_id[model_id]["features_exist"]
        assert by_id[model_id]["hash_matches"]
        assert by_id[model_id]["activation_ready"]


def test_v18_bundle_contains_six_prospective_deployment_models():
    manifest = load_v18_manifest()

    assert manifest["holdout_cycle"] == "2026-3"
    assert set(manifest["models"]) == {
        "model_1", "model_2", "model_3", "model_4", "model_5", "model_6"
    }
    assert all("2026-3" not in item["training_cycles"] for item in manifest["models"].values())


def test_v18_owner_outlook_switches_from_model_2_to_model_3():
    source_sha256 = load_v18_manifest()["source_workbook_sha256"]
    day14 = predict_v18_outlooks("2026-3", "Tags 1", 14, source_sha256)
    day21 = predict_v18_outlooks("2026-3", "Tags 1", 21, source_sha256)

    assert day14 is not None and day14["weight_model_id"] == "model_2"
    assert day21 is not None and day21["weight_model_id"] == "model_3"
    assert 0 < day14["model_1"]["prediction"] <= 1
    assert day21["model_3"]["prediction"] > 500


def test_v18_outlook_rejects_a_different_workbook():
    assert predict_v18_outlooks("2026-3", "Tags 1", 14, "not-the-v18-workbook") is None
