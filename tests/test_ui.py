"""Focused UI smoke tests for ByteFlow branding and the main upload experience.

Run with: python -m pytest tests/test_ui.py -q
These tests use Streamlit's headless AppTest and do not need a real API key.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.conftest import FakeLLM
from utils.config import Settings

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 240


def make_app(monkeypatch, tmp_path, key=None):
    settings = Settings(
        llm_api_key=key,
        outputs_dir=tmp_path / "out",
        max_retries=1,
    )
    monkeypatch.setattr("utils.config.get_settings", lambda: settings)
    monkeypatch.setattr("agents.llm.OpenAICompatibleLLM", lambda _settings: FakeLLM())
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


def visible_text(at):
    elements = (
        list(at.title) + list(at.header) + list(at.subheader) + list(at.caption)
        + list(at.markdown) + list(at.info) + list(at.error)
    )
    return " ".join(str(element.value) for element in elements)


def test_byteflow_branding_and_welcome_screen(monkeypatch, tmp_path):
    at = make_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert any("ByteFlow" in str(element.value) for element in at.title)
    text = visible_text(at)
    assert "Turn raw data into clear decisions." in text
    assert "Upload a CSV or Excel file" in text


def test_upload_controls_are_available(monkeypatch, tmp_path):
    at = make_app(monkeypatch, tmp_path)
    at.run()

    assert not at.exception
    assert len(at.sidebar.file_uploader) == 1
    assert any(button.label == "Clear my data" for button in at.sidebar.button)
    assert any("Your workspace" in str(element.value) for element in at.sidebar.header)


def test_run_button_is_disabled_without_provider_key(monkeypatch, tmp_path):
    at = make_app(monkeypatch, tmp_path, key=None)
    at.run()

    assert not at.exception
    run_buttons = [button for button in at.button if "Run analysis" in button.label]
    assert run_buttons and run_buttons[0].disabled
