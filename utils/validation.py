"""Upload and question validation. Uploaded files are untrusted input."""
from __future__ import annotations

import io
import zipfile
from typing import Literal

import filetype
import pandas as pd
from pydantic import BaseModel

from utils.config import Settings
from utils.logging_config import get_logger
from utils.security import (
    ALLOWED_EXTENSIONS,
    UserFacingError,
    sanitize_display_name,
)

logger = get_logger(__name__)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class UploadInfo(BaseModel):
    display_name: str
    file_type: Literal["csv", "xlsx"]
    size_bytes: int
    n_rows: int
    n_columns: int


def _extension(filename: str) -> str:
    name = filename.replace("\\", "/").split("/")[-1].lower()
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def _check_zip_safety(data: bytes, settings: Settings) -> None:
    """Reject zip bombs and malformed archives before parsing an .xlsx."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > settings.max_zip_entries:
                raise UserFacingError("The Excel file contains too many internal parts.")
            if sum(i.file_size for i in infos) > settings.max_uncompressed_bytes:
                raise UserFacingError("The Excel file is too large when decompressed.")
    except zipfile.BadZipFile as exc:
        raise UserFacingError("The file is not a valid .xlsx workbook.") from exc


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise header names only (strip, fill blanks, make unique)."""
    seen: dict[str, int] = {}
    names = []
    for i, col in enumerate(df.columns):
        name = str(col).strip() or f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df.columns = names
    return df


def _read_csv(data: bytes, settings: Settings) -> pd.DataFrame:
    if b"\x00" in data:
        raise UserFacingError("The file contains binary data and is not a valid CSV.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(data), encoding=encoding, nrows=settings.max_rows + 1,
                on_bad_lines="error", low_memory=False,
            )
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # parser errors, empty data, etc.
            logger.warning("CSV parse failed: %s", type(exc).__name__)
            raise UserFacingError("The CSV file is malformed or empty.") from exc
    raise UserFacingError("The CSV file could not be decoded.")


def _read_xlsx(data: bytes) -> pd.DataFrame:
    # pandas opens workbooks with data_only=True: formulas are NEVER evaluated.
    try:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl", sheet_name=0)
    except Exception as exc:
        logger.warning("XLSX parse failed: %s", type(exc).__name__)
        raise UserFacingError("The Excel file is corrupted or unreadable.") from exc


def validate_and_load_upload(filename: str, data: bytes, settings: Settings) -> tuple[pd.DataFrame, UploadInfo]:
    """Validate an uploaded file and return (DataFrame, UploadInfo).

    Raises UserFacingError with a safe message on any problem.
    """
    ext = _extension(filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise UserFacingError("Only .csv and .xlsx files are accepted.")
    if not data:
        raise UserFacingError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise UserFacingError(f"File exceeds the {settings.max_upload_mb} MB upload limit.")

    if ext == ".xlsx":
        if not data.startswith(b"PK\x03\x04"):
            raise UserFacingError("File content does not match the .xlsx extension.")
        _check_zip_safety(data, settings)
        kind = filetype.guess(data)
        if kind is None or kind.mime != XLSX_MIME:
            raise UserFacingError("File content does not match the .xlsx extension.")
        df = _read_xlsx(data)
    else:
        if filetype.guess(data) is not None:  # real CSVs are plain text: no signature
            raise UserFacingError("File content does not match the .csv extension.")
        df = _read_csv(data, settings)

    if df.empty or df.shape[1] == 0:
        raise UserFacingError("The dataset contains no data.")
    if len(df) > settings.max_rows:
        raise UserFacingError(f"Dataset has more than {settings.max_rows:,} rows (limit).")
    if df.shape[1] > settings.max_columns:
        raise UserFacingError(f"Dataset has more than {settings.max_columns} columns (limit).")

    df = _clean_columns(df)
    info = UploadInfo(
        display_name=sanitize_display_name(filename), file_type=ext[1:],  # type: ignore[arg-type]
        size_bytes=len(data), n_rows=len(df), n_columns=df.shape[1],
    )
    return df, info


def validate_question(question: str, settings: Settings) -> str:
    """Validate the user's natural-language question."""
    if not isinstance(question, str) or not question.strip():
        raise UserFacingError("Please enter a question.")
    if len(question) > settings.max_question_chars:
        raise UserFacingError(f"Question is too long (max {settings.max_question_chars} characters).")
    from utils.security import sanitize_text_for_llm
    return sanitize_text_for_llm(question, max_len=settings.max_question_chars)