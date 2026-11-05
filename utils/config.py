"""Central configuration. All settings come from environment variables / .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    """Validated application settings. Tests can build their own instance."""

    # Any OpenAI-compatible provider works (Groq, Google Gemini, OpenRouter, OpenAI...)
    llm_api_key: SecretStr | None = None  # SecretStr hides the value in repr/logs
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = Field(0.0, ge=0.0, le=1.0)
    llm_timeout_seconds: int = Field(60, ge=5, le=300)
    llm_max_api_retries: int = Field(2, ge=0, le=5)

    max_upload_mb: int = Field(20, ge=1, le=200)
    max_uncompressed_mb: int = Field(200, ge=1, le=2000)  # zip-bomb guard for .xlsx
    max_zip_entries: int = Field(1000, ge=10, le=10000)
    max_rows: int = Field(200_000, ge=1)
    max_columns: int = Field(200, ge=1)
    max_question_chars: int = Field(1000, ge=50, le=5000)

    max_retries: int = Field(2, ge=0, le=5)  # critic -> earlier node retries
    random_seed: int = 42
    ml_sample_rows: int = Field(50_000, ge=1000)
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    @field_validator("llm_base_url")
    @classmethod
    def _require_https(cls, v: str) -> str:
        """Never send the API key over plain HTTP."""
        if not v.startswith("https://"):
            raise ValueError("LLM_BASE_URL must start with https://")
        return v

    @property
    def llm_configured(self) -> bool:
        return self.llm_api_key is not None and bool(self.llm_api_key.get_secret_value().strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_uncompressed_bytes(self) -> int:
        return self.max_uncompressed_mb * 1024 * 1024


_ENV_MAP = {
    "LLM_API_KEY": "llm_api_key",
    "LLM_BASE_URL": "llm_base_url",
    "LLM_MODEL": "llm_model",
    "MAX_UPLOAD_MB": "max_upload_mb",
    "MAX_RETRIES": "max_retries",
    "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables, .env, or Streamlit secrets."""

    load_dotenv(PROJECT_ROOT / ".env")

    values = {
        field: os.environ[env]
        for env, field in _ENV_MAP.items()
        if os.environ.get(env)
    }

    # Use Streamlit Cloud secrets when environment variables are absent.
    if "llm_api_key" not in values:
        try:
            import streamlit as st

            secret_key = st.secrets.get("LLM_API_KEY")

            if secret_key:
                values["llm_api_key"] = secret_key

        except Exception:
            pass

    if values.get("llm_api_key") == "your-key-here":
        values.pop("llm_api_key")

    return Settings(**values)