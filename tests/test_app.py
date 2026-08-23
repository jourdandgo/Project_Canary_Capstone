from pathlib import Path
from datetime import date
import os

from streamlit.testing.v1 import AppTest

from canary import (
    DEFAULT_MANAGEMENT_DECISIONS_PATH,
    FOLLOW_UP_STATUSES,
    latest_management_decisions,
    load_management_decisions,
    record_management_decision,
)


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_package_root_exports_used_by_app_are_importable():
    """Catch package-root export mismatches before Streamlit serves the app."""

    assert DEFAULT_MANAGEMENT_DECISIONS_PATH.name == "management_decisions.csv"
    assert FOLLOW_UP_STATUSES
    assert callable(latest_management_decisions)
    assert callable(load_management_decisions)
    assert callable(record_management_decision)


def test_dashboard_renders_without_streamlit_errors(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert [uploader.label for uploader in app.file_uploader] == [
        "Upload current-cycle data",
        "Upload final-weight workbook",
    ]
    cycle = next(widget for widget in app.selectbox if widget.label == "Harvest cycle")
    as_of = next(widget for widget in app.date_input if widget.label == "Review date")
    assert cycle.value == "2026-3"
    # The corrected latest workbook now ends cleanly on Day 35.  When that
    # final recorded day is complete, the default review date may equal max.
    assert as_of.value <= as_of.max
    pages_dir = Path(__file__).parents[1] / "pages"
    assert {page.name for page in pages_dir.glob("*.py") if not page.name.startswith("_")} == {
        "about_canary.py",
        "home.py",
        "building.py",
        "harvest_analysis.py",
        "business_value.py",
        "eda.py",
        "model_evidence.py",
        "methodology.py",
        "action_playbook.py",
        "action_history.py",
        "data_settings.py",
        "prediction_lab.py",
        "how_canary_works.py",
    }
    detail_buttons = [
        button
        for button in app.button
        if button.label.startswith("See how ") and "predictions were made" in button.label
    ]
    assert len(detail_buttons) == 3
    unavailable_buttons = [button for button in app.button if button.label == "No details available"]
    assert len(unavailable_buttons) == 3
    assert all(button.disabled for button in unavailable_buttons)

    monkeypatch.setenv("CANARY_TEST_VIEW", "Building View")
    detail = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    detail.session_state["detail_building"] = "Tags 3"
    detail.run()
    assert not detail.exception
    chosen = next(widget for widget in detail.selectbox if widget.label == "Building")
    assert chosen.value == "Tags 3"
    # The refreshed 2026-3 workbook now contains observed Day 35 weights.  The
    # milestone can therefore be Achieved or Missed at the default as-of date;
    # it must no longer be forced to the pre-refresh "Upcoming" state.
    assert any("Day 35 milestone:" in info.value for info in detail.info)
    assert [subheader.value for subheader in detail.subheader][:5] == [
        "1 · Decision summary and next check",
        "2 · Traceable observed-risk breakdown",
        "3 · Forecast provenance",
        "4 · Forecast outlooks",
        "5 · View each forecast calculation",
    ]
    expander_labels = [expander.label for expander in detail.expander]
    assert "See why this action was selected" in expander_labels
    assert "View problem-pattern criteria · evidence → match" in expander_labels
    assert "Record a management override" in expander_labels
    assert "View 3-day signal calculation · recorded evidence → rule → flag" in expander_labels
    assert "Data availability for this review" in expander_labels
    assert "See raw source inputs and technical audit trace" in expander_labels
    detail_visible = " ".join(
        item.value
        for item in [*detail.markdown, *detail.caption]
        if isinstance(item.value, str)
    )
    assert "Why this building needs attention" in detail_visible
    assert "Possible contributing conditions" in detail_visible
    assert "What management should check next" in detail_visible
    assert "View risk calculation · input → rule → score → label" in expander_labels
    assert "View calculation · Model 1 · End-of-cycle recovery proxy" in expander_labels
    assert "View calculation · Model 3 · Day 35 bodyweight" in expander_labels
    assert "Canary supports prioritization and investigation" in detail_visible
    assert "Model Selection and Comparison" not in detail_visible

    cycle = next(widget for widget in detail.selectbox if widget.label == "Harvest cycle")
    cycle.set_value("2025-5").run()
    assert not detail.exception
    assert not [widget for widget in detail.date_input if widget.label == "Review date"]
    metric_labels = [metric.label for metric in detail.metric]
    assert "Harvest completed on" in metric_labels
    assert "Recorded ending recovery" in metric_labels
    assert "Actual final average weight" in metric_labels
    assert "Risk level" not in metric_labels
    assert "Predicted harvest recovery" not in metric_labels


def test_historical_home_shows_actuals_without_risk_or_predictions(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    cycle = next(widget for widget in app.selectbox if widget.label == "Harvest cycle")
    cycle.set_value("2025-5").run()

    assert not app.exception
    metric_labels = [metric.label for metric in app.metric]
    assert metric_labels[:3] == [
        "Completed buildings",
        "Recorded ending recovery",
        "Recorded weight results",
    ]
    assert not [widget for widget in app.date_input if widget.label == "Review date"]
    rendered = " ".join(
        item.value for item in app.markdown if isinstance(item.value, str)
    )
    assert rendered.count("Harvest completed") >= 6
    assert "Recorded ending recovery" in rendered
    assert (
        "Actual final average weight (g)" in rendered
        or "Recorded Day 35 weight (g)" in rendered
    )
    assert "Risk score" not in rendered
    assert "Predicted recovery" not in rendered


def test_incomplete_day_counts_placed_buildings_and_keeps_a_priority(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    as_of = next(widget for widget in app.date_input if widget.label == "Review date")
    as_of.set_value(date(2026, 7, 26)).run()

    assert not app.exception
    rendered = " ".join(
        item.value for item in app.markdown if isinstance(item.value, str)
    )
    assert "Buildings needing attention" in rendered
    assert "Projected recovery proxy" in rendered
    assert not any("No placed building" in info.value for info in app.info)


def test_action_history_exposes_calculations_overrides_and_actions(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Action History")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Calculated risk history",
        "Management overrides",
        "Management actions",
    ]
    button_labels = [button.label for button in app.button]
    assert "Save selected review date" in button_labels
    assert "Save / backfill window" in button_labels


def test_rule_admin_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Action Playbook")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()
    save = next(button for button in app.button if button.label == "Save rule")
    save.click().run()

    assert not app.exception
    assert any("Confirm the change before saving" in error.value for error in app.error)


def test_settings_opens_with_data_checks_by_default(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Data & Settings")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()
    assert not app.exception
    assert any("Data checks passed" in item.value for item in app.success)
    assert any(
        button.label == "Save risk score rules" for button in app.button
    )


def test_risk_rule_admin_requires_confirmation(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Data & Settings")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()
    save = next(button for button in app.button if button.label == "Save risk score rules")
    save.click().run()

    assert not app.exception
    assert any("Confirm the scoring change before saving" in error.value for error in app.error)


def test_model_proof_exposes_targets_features_and_validation(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Canary Methodology")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    assert not app.exception
    visible_text = " ".join(
        element.value
        for collection in (app.markdown, app.caption, app.subheader, app.info, app.warning)
        for element in collection
        if isinstance(element.value, str)
    )
    assert "last recorded daily date ÷ beginning population" in visible_text
    assert "Ridge regression" in visible_text
    assert "leave" in visible_text.lower()
    assert "Extra Trees" in visible_text
    assert "last-recorded recovery" in visible_text
    assert "Executive summary" in visible_text
    assert "1.8 kg on Day 35" in visible_text
    assert "Day 14 prediction versus last-recorded recovery" in visible_text
    assert "Data foundation" in visible_text
    assert "balanced decision snapshots" in visible_text
    assert "gated learned models with a transparent fallback" in visible_text
    assert "nested whole-cycle validation" in visible_text.lower()
    assert "historical remaining gain" in visible_text.lower()
    assert "ordinary linear regression" in visible_text.lower()
    assert "gradient boosting" in visible_text.lower()
    assert "Which inputs the selected recovery model relies on" in visible_text
    assert "Top five recorded inputs in the fitted recovery model" in visible_text
    assert "How Canary handles temperature, humidity, feed, water, and heat stress" in visible_text
    assert "Day 14 projection versus recorded Day 35 weight" in visible_text
    assert "Goal / Y" in " ".join(str(frame.value) for frame in app.dataframe)


def test_home_prioritizes_today_decision_without_repeating_product_brief(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()
    assert not app.exception
    visible = " ".join(
        item.value for item in [*app.markdown, *app.caption] if isinstance(item.value, str)
    )
    assert "DAILY MANAGEMENT VIEW" in visible
    assert "What needs attention today?" in visible
    assert "Observed risk is separate from forecasts" in visible
    assert "Management remains in control" in visible
    assert "The management problem" not in visible
    assert "The management gap" not in visible
    assert "Canary Command Center" in visible
    assert "Buildings needing attention" in visible
    assert "Projected recovery proxy" in visible
    assert "Review first" in visible
    assert "Estimated gross revenue at risk" not in visible
    assert "Day 35 weight outlooks" not in visible
    assert not app.segmented_control


def test_evidence_page_states_findings_and_limits(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "EDA")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()
    assert not app.exception
    visible = " ".join(item.value for item in [*app.markdown, *app.info, *app.warning, *app.success] if isinstance(item.value, str))
    assert "association is not proof" in visible
    assert "Farm Insights" in visible
    assert len(app.tabs) == 7
    assert [tab.label for tab in app.tabs] == [
        "1 · Data coverage",
        "2 · Day 14 → Day 35",
        "3 · Day 14 → Recovery",
        "4 · Environment",
        "5 · Survival paths",
        "6 · Forecast limits",
        "7 · Target attainment",
    ]


def test_model_evidence_uses_frozen_logo_predictions_and_shadow_status(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Model Evidence")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=45)
    app.run()

    assert not app.exception
    visible = " ".join(
        item.value
        for item in [*app.markdown, *app.caption, *app.info, *app.warning, *app.success]
        if isinstance(item.value, str)
    )
    assert "Two outcomes, two validated model artifacts" in visible
    assert "Model 1 for the end-of-cycle recovery proxy" in visible
    assert "leave-one-building-flock-out" in visible
    assert "does not manufacture a Day 28 forecast" in visible
    assert "predictive association, not causal effect" in visible
    assert len(app.dataframe) >= 5


def test_business_value_page_exposes_editable_assumptions_and_estimates(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Business Value")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    assert not app.exception
    slider_labels = [slider.label for slider in app.slider]
    assert "Live chicken price (₱/kg)" in slider_labels
    assert "Sale weight per bird (kg)" in slider_labels
    assert "Recovery improvement (points)" in slider_labels
    price_slider = next(
        slider for slider in app.slider if slider.label == "Live chicken price (₱/kg)"
    )
    sale_weight_slider = next(
        slider for slider in app.slider if slider.label == "Sale weight per bird (kg)"
    )
    cycles_slider = next(
        slider for slider in app.slider if slider.label == "Cycles per year"
    )
    assert price_slider.value == 120.0
    assert sale_weight_slider.value == 2.0
    assert cycles_slider.value == 5
    metric_labels = [metric.label for metric in app.metric]
    assert "Estimated gross revenue at risk" in metric_labels
    assert "Selected scenario opportunity" in metric_labels
    visible = " ".join(
        item.value
        for item in [*app.markdown, *app.warning, *app.caption]
        if isinstance(item.value, str)
    )
    assert "not profit" in visible

    original_value = next(
        metric for metric in app.metric if metric.label == "Estimated gross revenue at risk"
    ).value
    price_slider.set_value(100.0).run()
    revised_value = next(
        metric for metric in app.metric if metric.label == "Estimated gross revenue at risk"
    ).value
    assert revised_value != original_value


def test_harvest_analysis_is_all_cycle_and_target_specific(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    monkeypatch.setenv("CANARY_TEST_VIEW", "Harvest Analysis")
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert not [widget for widget in app.selectbox if widget.label == "Harvest cycle"]
    assert any(widget.label == "Review date" for widget in app.date_input)
    assert [tab.label for tab in app.tabs] == [
            "Recovery proxy",
        "Day 35 weight",
        "Detailed history",
    ]
    metric_labels = [metric.label for metric in app.metric]
    assert metric_labels[:5] == [
        "History in view",
        "Historical recovery proxy",
        "Recorded Day 35 weight",
        "Current projected recovery",
        "Current projected Day 35 weight",
    ]
    visible = " ".join(
        item.value
        for item in [*app.markdown, *app.caption, *app.info]
        if isinstance(item.value, str)
    )
    assert "Harvest Analysis" in visible
    assert "31 historical building outcomes" in visible
    assert "31 observed Day 35 outcomes" in visible
    assert "Model release" in visible
    assert any(
        button.label == "Download filtered harvest history (CSV)"
        for button in app.download_button
    )
