"""Slack API helper utilities for LaunchMind."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class SlackAPIError(Exception):
    """Raised when Slack API operations fail."""


class SlackAPI:
    """Minimal Slack client for posting Block Kit launch messages."""

    def __init__(self) -> None:
        """Load Slack configuration from environment variables."""
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.channel = _normalize_slack_channel(os.getenv("SLACK_CHANNEL"))
        self.api_base_url = os.getenv("SLACK_API_BASE_URL", "https://slack.com/api").rstrip("/")
        self.timeout = _read_int_env("SLACK_TIMEOUT", DEFAULT_TIMEOUT)
        self.max_retries = _read_int_env("SLACK_MAX_RETRIES", DEFAULT_MAX_RETRIES)

        if not self.bot_token:
            raise SlackAPIError("Missing SLACK_BOT_TOKEN environment variable.")
        if not self.channel:
            raise SlackAPIError(
                "Missing SLACK_CHANNEL environment variable. Use a channel ID like "
                "'C12345678' or a channel name like 'launches' or '#launches'."
            )

        self.headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def post_message(
        self,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post a message to Slack."""
        payload: Dict[str, Any] = {
            "channel": channel or self.channel,
            "text": text,
        }
        if blocks:
            payload["blocks"] = blocks

        response = self._request(
            method="POST",
            endpoint="/chat.postMessage",
            json=payload,
        )
        if not response.get("ok"):
            raise SlackAPIError(f"Slack chat.postMessage failed: {response}")
        return response

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a Slack API request with retry handling."""
        url = f"{self.api_base_url}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=json,
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise SlackAPIError(
                        f"Temporary Slack API failure: {response.status_code} {response.text}"
                    )
                if response.status_code >= 400:
                    raise SlackAPIError(
                        f"Slack API request failed: {response.status_code} {response.text}"
                    )
                return response.json()
            except (requests.RequestException, SlackAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))

        raise SlackAPIError(f"Slack request failed after retries: {last_error}") from last_error


def _read_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SlackAPIError(f"Environment variable {name} must be an integer.") from exc


def _normalize_slack_channel(raw_channel: Optional[str]) -> Optional[str]:
    """Normalize Slack channel input from env or caller input."""
    if raw_channel is None:
        return None

    channel = raw_channel.strip()
    if not channel:
        return None
    if channel.startswith("#"):
        channel = channel[1:].strip()
    return channel or None


__all__ = ["SlackAPI", "SlackAPIError"]
