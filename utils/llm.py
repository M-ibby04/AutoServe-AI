"""LLM helper utilities for LaunchMind agents."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

import requests


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.2
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised when an LLM request cannot be completed successfully."""


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an OpenAI-compatible chat completion API and return assistant text."""
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string.")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string.")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise LLMError("Missing LLM API key. Set OPENAI_API_KEY or LLM_API_KEY.")

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    timeout = _read_int_env("LLM_TIMEOUT", DEFAULT_TIMEOUT)
    max_retries = _read_int_env("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    temperature = _read_float_env("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    debug_llm = os.getenv("DEBUG_LLM", "").strip().lower() == "true"

    if debug_llm:
        print("[LLM DEBUG] system_prompt:")
        print(system_prompt)
        print("[LLM DEBUG] user_prompt:")
        print(user_prompt)

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                raise LLMError(
                    f"Temporary LLM API failure: {response.status_code} {response.text}"
                )

            response.raise_for_status()
            return _extract_text(response.json())
        except requests.exceptions.Timeout as exc:
            last_error = exc
            logger.warning(
                "LLM request timed out on attempt %s/%s.", attempt, max_retries
            )
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            logger.warning(
                "LLM connection error on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response else "unknown"
            logger.error(
                "LLM HTTP error on attempt %s/%s: %s",
                attempt,
                max_retries,
                status_code,
            )
            break
        except LLMError as exc:
            last_error = exc
            logger.warning(
                "Retryable LLM error on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            logger.error("Unexpected LLM response format: %s", exc)
            raise LLMError("LLM returned an unexpected response format.") from exc

        if attempt < max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))

    logger.error("LLM request failed after %s attempts.", max_retries)
    raise LLMError("Unable to get a successful LLM response.") from last_error


def sanitize_llm_json(raw_response: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from an LLM response string."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise LLMError("LLM response was empty.")

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fences(cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(_extract_json_object(cleaned))

    if not isinstance(parsed, dict):
        raise LLMError("Expected a JSON object from the LLM.")
    return parsed


def _extract_text(response_data: Dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible response body."""
    choices = response_data.get("choices")
    if not choices:
        raise ValueError("Response does not contain choices.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        combined = "".join(text_parts).strip()
        if combined:
            return combined

    raise ValueError("Response does not contain assistant text.")


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences from a response string."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str:
    """Extract the first top-level JSON object from a string."""
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise LLMError("Could not locate a JSON object in the LLM response.")
    return text[start_index : end_index + 1]


def _read_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise LLMError(f"Environment variable {name} must be an integer.") from exc


def _read_float_env(name: str, default: float) -> float:
    """Read a float environment variable with a fallback default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise LLMError(f"Environment variable {name} must be a float.") from exc


__all__ = ["LLMError", "call_llm", "sanitize_llm_json"]
