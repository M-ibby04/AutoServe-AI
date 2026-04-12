"""Email helper utilities for LaunchMind."""

from __future__ import annotations

import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from typing import Any, Dict, Optional

import requests


DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class EmailAPIError(Exception):
    """Raised when sending email fails."""


class EmailAPI:
    """Send real email using SendGrid or SMTP credentials from the environment."""

    def __init__(self) -> None:
        """Load email configuration from environment variables."""
        self.sendgrid_api_key = _read_optional_env(
            "SENDGRID_API_KEY",
            placeholder_values={"your_sendgrid_api_key"},
        )
        verified_sender_email = _read_optional_env(
            "VERIFIED_SENDER_EMAIL",
            placeholder_values={"verified-sender@example.com"},
        )
        from_email = _read_optional_env("FROM_EMAIL")
        self.from_email = verified_sender_email or from_email
        self.test_recipient_email = _read_optional_env(
            "TEST_RECIPIENT_EMAIL",
            placeholder_values={"test-inbox@example.com"},
        )
        self.timeout = _read_int_env("EMAIL_TIMEOUT", DEFAULT_TIMEOUT)
        self.max_retries = _read_int_env("EMAIL_MAX_RETRIES", DEFAULT_MAX_RETRIES)

        self.smtp_host = _read_optional_env("SMTP_HOST")
        self.smtp_port = _read_int_env("SMTP_PORT", 587)
        self.smtp_username = _read_optional_env("SMTP_USERNAME")
        self.smtp_password = _read_optional_env("SMTP_PASSWORD")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() != "false"

        if not self.from_email:
            raise EmailAPIError(
                "Missing VERIFIED_SENDER_EMAIL or FROM_EMAIL environment variable."
            )
        if not self.test_recipient_email:
            raise EmailAPIError("Missing TEST_RECIPIENT_EMAIL environment variable.")
        if not self.sendgrid_api_key and not (
            self.smtp_host and self.smtp_username and self.smtp_password
        ):
            raise EmailAPIError(
                "Configure SENDGRID_API_KEY or SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD."
            )

    def send_email(
        self,
        subject: str,
        text_body: str,
        html_body: str,
        to_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email through SendGrid or SMTP."""
        recipient = to_email or self.test_recipient_email
        if not isinstance(subject, str) or not subject.strip():
            raise EmailAPIError("Email subject must be a non-empty string.")
        if not isinstance(text_body, str) or not text_body.strip():
            raise EmailAPIError("Email text body must be a non-empty string.")
        if not isinstance(html_body, str) or not html_body.strip():
            raise EmailAPIError("Email HTML body must be a non-empty string.")

        if self.sendgrid_api_key:
            return self._send_with_sendgrid(
                subject=subject.strip(),
                text_body=text_body.strip(),
                html_body=html_body.strip(),
                to_email=recipient,
            )
        return self._send_with_smtp(
            subject=subject.strip(),
            text_body=text_body.strip(),
            html_body=html_body.strip(),
            to_email=recipient,
        )

    def _send_with_sendgrid(
        self,
        subject: str,
        text_body: str,
        html_body: str,
        to_email: str,
    ) -> Dict[str, Any]:
        """Send email using SendGrid's mail send API."""
        url = "https://api.sendgrid.com/v3/mail/send"
        headers = {
            "Authorization": f"Bearer {self.sendgrid_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": self.from_email},
            "subject": subject,
            "content": [
                {"type": "text/plain", "value": text_body},
                {"type": "text/html", "value": html_body},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise EmailAPIError(
                        f"Temporary SendGrid failure: {response.status_code} {response.text}"
                    )
                if response.status_code >= 400:
                    raise EmailAPIError(
                        f"SendGrid mail send failed: {response.status_code} {response.text}"
                    )
                return {
                    "provider": "sendgrid",
                    "status_code": response.status_code,
                    "to_email": to_email,
                    "from_email": self.from_email,
                    "message": "Email sent successfully via SendGrid.",
                }
            except (requests.RequestException, EmailAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))

        raise EmailAPIError(f"Email send failed after retries: {last_error}") from last_error

    def _send_with_smtp(
        self,
        subject: str,
        text_body: str,
        html_body: str,
        to_email: str,
    ) -> Dict[str, Any]:
        """Send email using SMTP credentials."""
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.from_email
        message["To"] = to_email
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.smtp_use_tls:
                    context = ssl.create_default_context()
                    with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
                        server.starttls(context=context)
                        server.login(self.smtp_username, self.smtp_password)
                        server.send_message(message)
                else:
                    with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
                        server.login(self.smtp_username, self.smtp_password)
                        server.send_message(message)

                return {
                    "provider": "smtp",
                    "status_code": 250,
                    "to_email": to_email,
                    "from_email": self.from_email,
                    "message": "Email sent successfully via SMTP.",
                }
            except (smtplib.SMTPException, OSError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))

        raise EmailAPIError(f"SMTP email send failed after retries: {last_error}") from last_error


def _read_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise EmailAPIError(f"Environment variable {name} must be an integer.") from exc


def _read_optional_env(
    name: str,
    placeholder_values: Optional[set[str]] = None,
) -> Optional[str]:
    """Read an optional env var while ignoring empty and placeholder values."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    cleaned = raw_value.strip()
    if not cleaned:
        return None
    if placeholder_values and cleaned in placeholder_values:
        return None
    return cleaned


__all__ = ["EmailAPI", "EmailAPIError"]
