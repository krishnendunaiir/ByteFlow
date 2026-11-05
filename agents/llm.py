"""LLM access through any OpenAI-compatible API (Groq, Google Gemini, OpenRouter, OpenAI...).

The LLM never receives raw datasets or runs code. It receives small, sanitized summaries and must
answer with ONE JSON object that is validated against a Pydantic schema before anyone uses it."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from utils.config import Settings
from utils.logging_config import get_logger
from utils.security import UserFacingError, redact_secrets

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

JSON_RULES = ("Respond with ONE JSON object only: no markdown, no code fences, no commentary. "
              "It must validate against this JSON Schema:\n")


class LLMError(UserFacingError):
    """Any LLM problem. The message is always safe to show (it never contains the API key)."""


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, schema: type[T], max_tokens: int = 1500) -> T: ...


def extract_json(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a model answer (tolerates code fences and <think> blocks)."""
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(s[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON answer is not an object")
    return obj


class OpenAICompatibleLLM:
    """Calls the provider with a timeout and automatic retries (rate limits, 5xx, connection errors)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_configured:
            raise LLMError("No API key is configured. Add LLM_API_KEY to your .env file.")
        self._settings = settings
        self._secret = settings.llm_api_key.get_secret_value()
        self._client = OpenAI(
            api_key=self._secret, base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_api_retries,
        )
        self._cache: dict[str, str] = {}  # identical prompts are answered once per session
        self.n_calls = 0

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        key = hashlib.sha256(f"{system}\x00{user}\x00{max_tokens}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        try:
            self.n_calls += 1
            resp = self._client.chat.completions.create(
                model=self._settings.llm_model, temperature=self._settings.llm_temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except openai.AuthenticationError as exc:
            raise LLMError("The AI provider rejected the API key. Check LLM_API_KEY in .env.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("The AI provider's rate limit was reached. Wait a minute and try again.") from exc
        except openai.NotFoundError as exc:
            raise LLMError("The AI model was not found. Check LLM_MODEL in .env.") from exc
        except openai.APITimeoutError as exc:
            raise LLMError("The AI provider took too long to answer (timeout).") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the AI provider. Check your internet connection and LLM_BASE_URL.") from exc
        except openai.APIStatusError as exc:
            # Keep the HTTP status and the provider's (redacted, truncated) reason: without them a
            # 400 cannot be diagnosed (bad model id, max_tokens too high, unsupported parameter...).
            detail = redact_secrets(str(getattr(exc, "message", "")), extra_secrets=(self._secret,)).strip()[:300]
            logger.warning("LLM API error: HTTP %s %s", exc.status_code, detail)
            raise LLMError(f"The AI provider rejected the request (HTTP {exc.status_code}): {detail or 'no details'}") from exc
        except openai.APIError as exc:
            logger.warning("LLM API error: %s", type(exc).__name__)
            raise LLMError("The AI provider returned an error. Please try again later.") from exc
        content = resp.choices[0].message.content if resp.choices else None
        if not content or not content.strip():
            raise LLMError("The AI provider returned an empty answer.")
        if len(self._cache) < 64:
            self._cache[key] = content
        return content

    def complete_json(self, system: str, user: str, schema: type[T], max_tokens: int = 1500) -> T:
        """Ask for JSON and validate it. One automatic repair attempt, then a clear error."""
        full_system = f"{system}\n\n{JSON_RULES}{json.dumps(schema.model_json_schema())}"
        feedback = ""
        for _ in range(2):
            raw = self._chat(full_system, user + feedback, max_tokens)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                logger.warning("LLM answer failed validation: %s", type(exc).__name__)
                feedback = ("\n\nYour previous answer was not valid for the schema. "
                            "Reply again with ONLY the corrected JSON object.")
        raise LLMError("The AI model did not return a valid structured answer.")