"""Helpers for the Streamlit app that can be unit-tested without Streamlit:
safe chart access, chart zip export, stale-session cleanup and a per-session run limiter."""
from __future__ import annotations

import io
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from utils.security import SecurityError, ensure_within_directory, sanitize_text_for_llm

_SESSION_DIR_RE = re.compile(r"^[0-9a-f]{32}$")


def purge_old_sessions(outputs_dir: Path, max_age_seconds: int = 24 * 3600, now: float | None = None) -> int:
    """Delete session folders older than `max_age_seconds` (data is not kept longer than needed).
    Only folders named like a session id are touched. Returns how many were removed."""
    base = Path(outputs_dir)
    if not base.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for child in base.iterdir():
        if child.is_dir() and not child.is_symlink() and _SESSION_DIR_RE.match(child.name):
            if now - child.stat().st_mtime > max_age_seconds:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
    return removed


def read_chart_bytes(path: str | Path, session_dir: Path) -> bytes:
    """Read a chart PNG, but only if it really lives directly inside this session's folder."""
    safe = ensure_within_directory(session_dir, Path(path).name)
    if safe.suffix != ".png" or not safe.is_file():
        raise SecurityError("The chart file is not available.")
    return safe.read_bytes()


def build_charts_zip(charts: list[dict[str, str]], session_dir: Path) -> bytes:
    """All charts as one in-memory zip with neutral filenames and a captions file."""
    buf = io.BytesIO()
    captions: list[str] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, chart in enumerate(charts, start=1):
            try:
                data = read_chart_bytes(chart["path"], session_dir)
            except (SecurityError, OSError):
                continue
            name = f"chart_{i:02d}.png"
            zf.writestr(name, data)
            captions.append(f"{name}: {sanitize_text_for_llm(chart.get('title', ''), 80)}")
        zf.writestr("captions.txt", "\n".join(captions) + "\n")
    return buf.getvalue()


@dataclass
class RunLimiter:
    """Protects the free API quota: a cooldown between runs and a cap per browser session."""
    min_interval_seconds: float = 10.0
    max_runs: int = 20
    runs: int = 0
    last_run: float = 0.0

    def check(self, now: float | None = None) -> str | None:
        """Return a message if a new run is not allowed right now, else None."""
        now = time.time() if now is None else now
        if self.runs >= self.max_runs:
            return f"This session reached its limit of {self.max_runs} analyses. Reload the page to start a new session."
        wait = self.min_interval_seconds - (now - self.last_run)
        if self.runs and wait > 0:
            return f"Please wait {int(wait) + 1} more second(s) before starting another analysis."
        return None

    def record(self, now: float | None = None) -> None:
        self.runs += 1
        self.last_run = time.time() if now is None else now