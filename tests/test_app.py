"""End-to-end tests of the Streamlit interface, driven headlessly with Streamlit's AppTest and a FakeLLM."""
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.conftest import FakeLLM
from utils.config import Settings

TIMEOUT = 240
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def make_app(monkeypatch, tmp_path, key="gsk_testkey1234567890"):
    settings = Settings(llm_api_key=key, outputs_dir=tmp_path / "out", max_retries=1)
    monkeypatch.setattr("utils.config.get_settings", lambda: settings)
    monkeypatch.setattr("agents.llm.OpenAICompatibleLLM", lambda s: FakeLLM())
    return AppTest.from_file(APP, default_timeout=TIMEOUT), settings


def button(at, label):
    return next(b for b in list(at.button) + list(at.sidebar.button) if b.label == label)


def csv(df):
    return df.to_csv(index=False).encode()


def upload(at, name, data):
    at.sidebar.file_uploader[0].upload(name, data, "text/csv")
    return at.run(timeout=TIMEOUT)


def all_text(at):
    return " ".join(str(x.value) for x in list(at.markdown) + list(at.error) + list(at.warning) + list(at.info))


def test_app_loads_and_asks_for_an_upload(monkeypatch, tmp_path):
    at, _ = make_app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception and "Upload a CSV or Excel file" in all_text(at)


def test_without_an_api_key_the_run_button_is_disabled(monkeypatch, tmp_path, sample_df):
    at, _ = make_app(monkeypatch, tmp_path, key=None)
    at.run()
    assert any("No API key" in e.value for e in at.sidebar.error)
    upload(at, "d.csv", csv(sample_df))
    assert button(at, "Run analysis").disabled


def test_upload_shows_preview_and_column_info(monkeypatch, tmp_path, sample_df):
    at, _ = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "d.csv", csv(sample_df))
    assert not at.exception and at.session_state["df"].shape == sample_df.shape
    assert len(at.dataframe) >= 2 and any(m.label == "Rows" and m.value == "300" for m in at.metric)


def test_bad_upload_shows_a_safe_error_and_keeps_no_data(monkeypatch, tmp_path):
    at, _ = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "pic.csv", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    assert any("does not match" in e.value for e in at.error) and at.session_state["df"] is None
    assert not at.exception


def test_full_analysis_through_the_ui(monkeypatch, tmp_path, sample_df):
    at, settings = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "d.csv", csv(sample_df))
    button(at, "Run analysis").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception
    result = at.session_state["result"]
    assert result["status"] == "done" and result["ml_results"]["task"] == "classification"
    assert "Automated Data Analysis Report" in all_text(at)
    assert [t.label for t in at.tabs] == [
    "Preview (first 50 rows)",
    "Column information",
    "Report",
    "Data quality",
    "Statistics",
    "Machine learning",
    "Charts",
    "Workflow log",
    "Export Analysis",
]
    assert {d.label for d in at.download_button} >= {"Download report (.md)", "Download all charts (.zip)"}
    assert (settings.outputs_dir / at.session_state["sid"]).is_dir()


def test_clear_my_data_deletes_everything(monkeypatch, tmp_path, sample_df):
    at, settings = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "d.csv", csv(sample_df))
    button(at, "Run analysis").click()
    at.run(timeout=TIMEOUT)
    sid = at.session_state["sid"]
    assert (settings.outputs_dir / sid).is_dir()
    button(at, "Clear my data").click()
    at.run(timeout=TIMEOUT)
    assert at.session_state["df"] is None and at.session_state["result"] is None
    assert not (settings.outputs_dir / sid).exists() and at.session_state["sid"] != sid
    at.run(timeout=TIMEOUT)  # the file is still in the uploader widget: it must NOT silently reload
    assert at.session_state["df"] is None
    upload(at, "other.csv", csv(sample_df.head(120)))  # a different file is accepted normally
    assert at.session_state["df"].shape[0] == 120


def test_removing_the_file_drops_the_data(monkeypatch, tmp_path, sample_df):
    at, _ = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "d.csv", csv(sample_df))
    assert at.session_state["df"] is not None
    at.sidebar.file_uploader[0].set_value(None)
    at.run(timeout=TIMEOUT)
    assert at.session_state["df"] is None


def test_second_run_is_blocked_by_the_cooldown(monkeypatch, tmp_path, sample_df):
    at, _ = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "d.csv", csv(sample_df))
    button(at, "Run analysis").click()
    at.run(timeout=TIMEOUT)
    at.session_state["limiter"].last_run = time.time()  # "the run just finished": independent of how slow this machine is
    button(at, "Run analysis").click()
    at.run(timeout=TIMEOUT)
    assert any("Please wait" in w.value for w in at.warning)
    assert at.session_state["limiter"].runs == 1


def test_hostile_column_names_cannot_inject_html_into_the_report(monkeypatch, tmp_path, sample_df):
    evil = sample_df.rename(columns={"city": "<script>alert(1)</script>"})
    at, _ = make_app(monkeypatch, tmp_path)
    at.run()
    upload(at, "d.csv", csv(evil))
    button(at, "Run analysis").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["result"]["status"] == "done"
    assert "<script>" not in at.session_state["result"]["report_markdown"]
    assert all("<script>" not in str(m.value) for m in at.markdown)


def test_invalid_env_settings_show_a_friendly_error_not_a_traceback(monkeypatch, tmp_path):
    def boom():
        raise ValueError("bad value: sk-should-never-be-shown-123456")
    monkeypatch.setattr("utils.config.get_settings", boom)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert any("settings are invalid" in e.value for e in at.error)
    assert all("sk-should-never" not in e.value for e in at.error)