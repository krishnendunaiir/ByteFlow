"""Tests for the app helper functions (no Streamlit needed)."""
import io
import os
import zipfile

import pytest

from utils.app_support import RunLimiter, build_charts_zip, purge_old_sessions, read_chart_bytes
from utils.security import SecurityError, new_session_id

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


def test_purge_removes_only_old_session_folders(tmp_path):
    old, fresh = new_session_id(), new_session_id()
    for sid in (old, fresh):
        (tmp_path / sid).mkdir()
        (tmp_path / sid / "c.png").write_bytes(PNG)
    (tmp_path / "not-a-session").mkdir()  # must never be touched
    (tmp_path / ".gitkeep").write_text("")
    os.utime(tmp_path / old, (1000, 1000))
    assert purge_old_sessions(tmp_path, max_age_seconds=3600) == 1
    assert not (tmp_path / old).exists() and (tmp_path / fresh).exists()
    assert (tmp_path / "not-a-session").exists() and (tmp_path / ".gitkeep").exists()
    assert purge_old_sessions(tmp_path / "missing") == 0


def test_read_chart_bytes_only_serves_pngs_inside_the_session(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    (session / "ok.png").write_bytes(PNG)
    (session / "data.csv").write_text("secret")
    (tmp_path / "outside.png").write_bytes(PNG)
    assert read_chart_bytes(session / "ok.png", session) == PNG
    with pytest.raises(SecurityError):
        read_chart_bytes(session / "data.csv", session)          # wrong type
    with pytest.raises(SecurityError):
        read_chart_bytes(session / "missing.png", session)       # does not exist
    with pytest.raises(SecurityError):
        read_chart_bytes("../outside.png", session)  # traversal is reduced to a bare filename that does not exist here


def test_charts_zip_uses_neutral_names_and_skips_missing_files(tmp_path):
    (tmp_path / "a.png").write_bytes(PNG)
    charts = [{"path": str(tmp_path / "a.png"), "title": "Good chart\x07"},
              {"path": str(tmp_path / "gone.png"), "title": "Missing"}]
    with zipfile.ZipFile(io.BytesIO(build_charts_zip(charts, tmp_path))) as zf:
        assert sorted(zf.namelist()) == ["captions.txt", "chart_01.png"]
        assert zf.read("chart_01.png") == PNG
        assert "Good chart" in zf.read("captions.txt").decode() and "\x07" not in zf.read("captions.txt").decode()


def test_run_limiter_cooldown_and_cap():
    lim = RunLimiter(min_interval_seconds=10, max_runs=2)
    assert lim.check(now=100) is None
    lim.record(now=100)
    assert "wait" in lim.check(now=105)
    assert lim.check(now=111) is None
    lim.record(now=111)
    assert "limit of 2" in lim.check(now=500)