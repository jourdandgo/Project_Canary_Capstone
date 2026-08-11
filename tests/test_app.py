from pathlib import Path
from datetime import date
import os

from streamlit.testing.v1 import AppTest


SOURCE = Path(
    os.getenv(
        "CANARY_TEST_WORKBOOK",
        str(Path(__file__).resolve().parents[1] / "data" / "FARM HARVEST DATA.xlsx"),
    )
)


def test_dashboard_renders_without_streamlit_errors(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert [uploader.label for uploader in app.file_uploader] == [
        "Update daily farm data (optional)",
        "Update final-weight data (optional)",
    ]
    cycle = next(widget for widget in app.selectbox if widget.label == "Harvest cycle")
    as_of = next(widget for widget in app.date_input if widget.label == "Review date")
    assert cycle.value == "2026-3"
    assert as_of.value < as_of.max
    pages_dir = Path(__file__).parents[1] / "pages"
    assert {page.name for page in pages_dir.glob("*.py") if not page.name.startswith("_")} == {
        "home.py",
        "building.py",
        "harvest_analysis.py",
        "business_value.py",
        "eda.py",
        "methodology.py",
        "action_playbook.py",
        "data_settings.py",
    }
    detail_buttons = [button for button in app.button if button.label.startswith("View ")]
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
    assert any("Day 35 milestone: Upcoming" in info.value for info in detail.info)
    assert [subheader.value for subheader in detail.subheader][:5] == [
        "1 · Decision summary and next check",
        "2 · Risk score breakdown",
        "3 · Forecast deep dive",
        "4 · Additional operational checks",
        "5 · How the outlook changed",
    ]
    expander_labels = [expander.label for expander in detail.expander]
    assert "See raw forecast inputs and calculation trace" in expander_labels
    assert "What Canary adopted from the teammate model—and what it rejected" in expander_labels
    assert "See why this action was selected" in expander_labels
    assert "Technical audit details" in expander_labels
    detail_visible = " ".join(
        item.value
        for item in [*detail.markdown, *detail.caption]
        if isinstance(item.value, str)
    )
    assert "Why this building needs attention" in detail_visible
    assert "Possible contributing conditions" in detail_visible
    assert "What management should check next" in detail_visible
    assert "A. Harvest Recovery Model" in detail_visible
    assert "A. Day 35 Average Weight Model" in detail_visible
    assert "B. Executive Summary" in detail_visible
    assert "C. Input and Output Variables" in detail_visible
    assert "D. Pre-processing Steps" in detail_visible
    assert "E. Model Selection and Comparison" in detail_visible
    assert "F. Interpretation" in detail_visible
    assert "How this building’s projection was calculated" in detail_visible

    cycle = next(widget for widget in detail.selectbox if widget.label == "Harvest cycle")
    cycle.set_value("2025-5").run()
    assert not detail.exception
    assert not [widget for widget in detail.date_input if widget.label == "Review date"]
    metric_labels = [metric.label for metric in detail.metric]
    assert "Harvest completed on" in metric_labels
    assert "Actual harvest recovery" in metric_labels
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
        "Final harvest recovery",
        "Actual weight results",
    ]
    assert not [widget for widget in app.date_input if widget.label == "Review date"]
    rendered = " ".join(
        item.value for item in app.markdown if isinstance(item.value, str)
    )
    assert rendered.count("Harvest completed") >= 6
    assert "Actual harvest recovery" in rendered
    assert "Actual final avg weight (g)" in rendered
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
    assert "Projected harvest recovery" in rendered
    assert not any("No placed building" in info.value for info in app.info)


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
    assert "age-band remaining-loss baseline" in visible_text
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


def test_home_explains_scope_denominators_and_business_question(monkeypatch):
    monkeypatch.setenv("CANARY_DEFAULT_WORKBOOK", str(SOURCE))
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30)
    app.run()
    assert not app.exception
    visible = " ".join(
        item.value for item in [*app.markdown, *app.caption] if isinstance(item.value, str)
    )
    assert "The management problem" in visible
    assert "What Canary does" in visible
    assert "Which buildings are at risk" in visible
    assert "Canary Command Center" in visible
    assert "Buildings needing attention" in visible
    assert "Projected harvest recovery" in visible
    assert "Estimated gross revenue at risk" in visible
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
    assert "Exploratory Data Analysis" in visible
    assert len(app.tabs) == 7
    assert [tab.label for tab in app.tabs] == [
        "1 · Data coverage",
        "2 · Day 14 → Day 35",
        "3 · Day 14 → Recovery",
        "4 · Environment",
        "5 · Survival paths",
        "6 · Model accuracy",
        "7 · Target attainment",
    ]


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
        "Harvest recovery",
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
    assert "25 independent outcomes" in visible
    assert "31 observed Day 35 outcomes" in visible
    assert "Model release" in visible
    assert any(
        button.label == "Download filtered harvest history (CSV)"
        for button in app.download_button
    )
