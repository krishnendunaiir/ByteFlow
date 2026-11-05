"""Security helpers: formula-injection defence, safe paths, secret redaction."""
from __future__ import annotations

import html
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)\S+"),
]
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class SecurityError(Exception):
    """Raised when an operation would violate a security rule."""


class UserFacingError(Exception):
    """An error whose message is safe to show to the user."""


# ---------- spreadsheet / CSV formula injection ----------
def neutralize_formula(value: Any) -> Any:
    """Prefix risky text with an apostrophe so Excel treats it as plain text."""
    if isinstance(value, str) and value.lstrip(" ").startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy that is safe to write to CSV/XLSX (headers and text cells)."""
    out = df.copy()
    out.columns = [neutralize_formula(str(c)) for c in out.columns]
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[col] = s.map(neutralize_formula)
    return out


# ---------- filenames, paths, sessions ----------
def safe_internal_filename(extension: str) -> str:
    """Random filename; the user-supplied name is never used on disk."""
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in ALLOWED_EXTENSIONS | {".png", ".md", ".html", ".json"}:
        raise SecurityError("Unsupported file extension.")
    return f"{uuid.uuid4().hex}{ext}"


def sanitize_display_name(name: str, max_len: int = 80) -> str:
    """Make a user filename safe to *display* (never used as a path)."""
    base = name.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip(".") or "upload"
    return base[:max_len]


def ensure_within_directory(base: Path, candidate: str | Path) -> Path:
    """Resolve `candidate` under `base`; raise if it escapes (path traversal)."""
    base_resolved = Path(base).resolve()
    resolved = (base_resolved / candidate).resolve()
    if not resolved.is_relative_to(base_resolved):
        raise SecurityError("Path escapes the allowed directory.")
    return resolved


def new_session_id() -> str:
    return uuid.uuid4().hex


def session_output_dir(outputs_dir: Path, session_id: str, create: bool = True) -> Path:
    """Per-session output folder, validated against traversal."""
    if not _SESSION_ID_RE.match(session_id or ""):
        raise SecurityError("Invalid session id.")
    path = ensure_within_directory(outputs_dir, session_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_session_dir(outputs_dir: Path, session_id: str) -> None:
    """Delete a session's output folder (only if it is inside outputs_dir)."""
    path = session_output_dir(outputs_dir, session_id, create=False)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


# ---------- secrets and error messages ----------
def redact_secrets(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    """Remove API-key-like strings from text before logging or display."""
    for secret in extra_secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
    return text


def safe_error_message(exc: BaseException) -> str:
    """Message for users: full text only for UserFacingError, else generic."""
    if isinstance(exc, UserFacingError):
        return redact_secrets(str(exc))
    return f"An internal error occurred ({type(exc).__name__}). Details were written to the log."


# ---------- untrusted text handling ----------
def sanitize_text_for_llm(text: Any, max_len: int = 100) -> str:
    """Shorten and clean dataset-derived text before it goes into a prompt."""
    s = _CONTROL_CHARS_RE.sub("", str(text))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] + ("..." if len(s) > max_len else "")


def escape_html(text: Any) -> str:
    """Escape text before it is ever placed into HTML."""
    return html.escape(str(text))