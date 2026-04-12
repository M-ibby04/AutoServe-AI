"""Marketing agent for AutoServe AI launch messaging."""

from __future__ import annotations

from html import escape
import json
from typing import Any, Dict, Optional

from message_bus import MessageBus
from utils.email_api import EmailAPI, EmailAPIError
from utils.llm import LLMError, call_llm, sanitize_llm_json
from utils.slack_api import SlackAPI, SlackAPIError


class MarketingAgent:
    """Generate launch copy, send outreach email, and post the Slack announcement."""

    agent_name = "marketing"

    def __init__(self, message_bus: MessageBus) -> None:
        """Store the shared message bus and preview cache."""
        self.message_bus = message_bus
        self.preview_product_spec: Optional[Dict[str, Any]] = None

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle preview, task, and revision messages."""
        if message.get("message_type") == "result":
            payload = message.get("payload", {})
            if payload.get("status") == "awaiting_ceo_approval":
                self.preview_product_spec = payload.get("product_spec")
                print("[Marketing Agent] Received product spec preview. Waiting for CEO approval...")
                return None

        self._validate_incoming_message(message)
        payload = message["payload"]
        feedback = payload.get("feedback")
        previous_result = payload.get("previous_result")
        engineer_result = payload.get("engineer_result")

        if message["message_type"] == "revision_request":
            print("[Marketing Agent] Revision request received from CEO.")
            print(f"[Marketing Agent] Feedback: {feedback}")
        else:
            print("[Marketing Agent] Approved marketing task received from CEO.")
        print("[Marketing Agent] Generating AutoServe AI launch messaging...")

        try:
            deliverable = self.generate_marketing_package(
                startup_name=payload["startup_name"],
                startup_idea=payload["startup_idea"],
                product_spec=payload["product_spec"],
                engineer_result=engineer_result,
                feedback=feedback,
                previous_result=previous_result,
            )

            print("[Marketing Agent] Sending launch email...")
            email_api = EmailAPI()
            email_result = email_api.send_email(
                subject=deliverable["email_subject"],
                text_body=deliverable["email_body_text"],
                html_body=deliverable["email_body_html"],
            )

            print("[Marketing Agent] Posting Slack launch message...")
            slack_api = SlackAPI()
            slack_response = slack_api.post_message(
                text=deliverable["slack_fallback_text"],
                blocks=self._build_slack_blocks(
                    tagline=deliverable["tagline"],
                    short_description=deliverable["short_description"],
                    pr_url=deliverable["pr_url"],
                    pr_url_available=deliverable["pr_url_available"],
                ),
            )

            result_payload = {
                "startup_name": payload["startup_name"],
                "startup_idea": payload["startup_idea"],
                "product_spec": payload["product_spec"],
                "pr_url": deliverable["pr_url"],
                "tagline": deliverable["tagline"],
                "short_description": deliverable["short_description"],
                "email_subject": deliverable["email_subject"],
                "email_body_text": deliverable["email_body_text"],
                "email_body_html": deliverable["email_body_html"],
                "social_posts": deliverable["social_posts"],
                "slack_fallback_text": deliverable["slack_fallback_text"],
                "pr_url_available": deliverable["pr_url_available"],
                "pr_reference_text": deliverable["pr_reference_text"],
                "email_result": email_result,
                "slack_result": {
                    "channel": slack_response.get("channel"),
                    "ts": slack_response.get("ts"),
                    "message_text": deliverable["slack_fallback_text"],
                },
            }

            print("[Marketing Agent] Email and Slack announcement completed.")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="ceo",
                message_type="result",
                payload=result_payload,
                parent_message_id=message["message_id"],
            )
        except (LLMError, EmailAPIError, SlackAPIError, ValueError) as exc:
            print(f"[Marketing Agent] Failure: {exc}")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="ceo",
                message_type="failure",
                payload={
                    "error": str(exc),
                    "stage": "marketing",
                },
                parent_message_id=message["message_id"],
            )

    def generate_marketing_package(
        self,
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
        engineer_result: Dict[str, Any],
        feedback: Optional[str],
        previous_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate AutoServe AI marketing assets with the LLM."""
        raw_pr_url = engineer_result.get("pr_url")
        pr_url_available = isinstance(raw_pr_url, str) and bool(raw_pr_url.strip())
        pr_url = raw_pr_url.strip() if pr_url_available else ""
        pr_reference_text = (
            f"GitHub PR: {pr_url}"
            if pr_url_available
            else "GitHub PR is not available yet. Landing page artifact exists but PR creation is still pending."
        )

        system_prompt = (
            "You are the Marketing Agent for AutoServe AI. Return only valid JSON "
            "with no markdown fences."
        )
        engineer_context = self._summarize_engineer_result(engineer_result)
        user_prompt = self._build_prompt(
            startup_name=startup_name,
            startup_idea=startup_idea,
            product_spec=product_spec,
            engineer_result=engineer_context,
            pr_reference_text=pr_reference_text,
            feedback=feedback,
            previous_result=previous_result,
        )
        parsed = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        parsed["pr_url"] = pr_url
        parsed["pr_url_available"] = pr_url_available
        parsed["pr_reference_text"] = pr_reference_text
        return self._validate_deliverable(parsed)

    def _build_prompt(
        self,
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
        engineer_result: Dict[str, Any],
        pr_reference_text: str,
        feedback: Optional[str],
        previous_result: Optional[Dict[str, Any]],
    ) -> str:
        """Build the marketing prompt."""
        revision_section = ""
        if feedback:
            previous_result_summary = self._summarize_previous_result(previous_result)
            revision_section = (
                "Revision feedback from CEO or QA:\n"
                f"{feedback}\n\n"
                "Previous marketing output:\n"
                f"{json.dumps(previous_result_summary, indent=2)}\n\n"
                "Revise the copy while preserving the best parts of the earlier version.\n\n"
            )

        return f"""
Startup name: {startup_name}

Startup idea:
{startup_idea}

Approved product specification:
{json.dumps(product_spec, indent=2)}

Engineering output:
{json.dumps(engineer_result, indent=2)}

GitHub launch reference:
{pr_reference_text}

{revision_section}Create launch-ready marketing copy for AutoServe AI.

AutoServe AI positioning to preserve:
- AI-powered WhatsApp and phone automation for small businesses
- Helps clinics, bakeries, grocery stores, and similar teams handle inquiries, bookings, and orders
- Focus on fewer missed calls/messages, faster replies, and lower manual workload

Return only valid JSON with exactly this structure:
{{
  "tagline": "string under 10 words",
  "short_description": "2-3 sentence description",
  "email_subject": "string",
  "email_preheader": "string",
  "email_opening": "string",
  "email_value_points": ["string", "string", "string"],
  "email_call_to_action": "string",
  "social_posts": {{
    "x": "string",
    "linkedin": "string",
    "instagram": "string"
  }},
  "slack_fallback_text": "string"
}}

Rules:
- The tagline must be under 10 words.
- short_description must feel specific to AutoServe AI.
- The email content should sound like outreach to a small-business lead.
- Include exactly 3 email_value_points.
- social posts must feel platform-appropriate but still practical.
- slack_fallback_text must mention the tagline and a one-line launch summary.
- If a PR link is available, reference it naturally.
- If a PR link is not available, use a fallback phrase like 'GitHub PR pending' or 'landing page PR coming soon'.
- Do not include any text outside the JSON object.
""".strip()

    @staticmethod
    def _summarize_engineer_result(engineer_result: Dict[str, Any]) -> Dict[str, Any]:
        """Trim the engineering payload before sending it to the marketing prompt."""
        html_preview = ""
        html_value = engineer_result.get("html")
        if isinstance(html_value, str) and html_value.strip():
            cleaned_html = html_value.strip()
            html_preview = (
                cleaned_html
                if len(cleaned_html) <= 1800
                else cleaned_html[:1780].rstrip() + "\n...[truncated]"
            )

        return {
            "headline": engineer_result.get("headline"),
            "subheadline": engineer_result.get("subheadline"),
            "call_to_action": engineer_result.get("call_to_action"),
            "summary": engineer_result.get("summary"),
            "landing_page_path": engineer_result.get("landing_page_path"),
            "branch": engineer_result.get("branch"),
            "issue_url": engineer_result.get("issue_url"),
            "pr_url": engineer_result.get("pr_url"),
            "commit_sha": engineer_result.get("commit_sha"),
            "html_preview": html_preview,
        }

    @staticmethod
    def _summarize_previous_result(previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Trim bulky prior marketing output before sending it back to the LLM."""
        if not isinstance(previous_result, dict):
            return {}

        html_value = previous_result.get("email_body_html")
        html_preview = ""
        if isinstance(html_value, str) and html_value.strip():
            cleaned_html = html_value.strip()
            html_preview = (
                cleaned_html
                if len(cleaned_html) <= 1600
                else cleaned_html[:1580].rstrip() + "\n...[truncated]"
            )

        return {
            "tagline": previous_result.get("tagline"),
            "short_description": previous_result.get("short_description"),
            "email_subject": previous_result.get("email_subject"),
            "email_body_text": previous_result.get("email_body_text"),
            "email_body_html_preview": html_preview,
            "social_posts": previous_result.get("social_posts"),
            "slack_fallback_text": previous_result.get("slack_fallback_text"),
            "pr_url": previous_result.get("pr_url"),
        }

    def _validate_deliverable(self, deliverable: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the marketing package."""
        required_fields = [
            "tagline",
            "short_description",
            "email_subject",
            "email_preheader",
            "email_opening",
            "email_value_points",
            "email_call_to_action",
            "social_posts",
            "slack_fallback_text",
            "pr_url",
            "pr_url_available",
            "pr_reference_text",
        ]
        for field in required_fields:
            if field not in deliverable:
                raise LLMError(f"Marketing deliverable is missing '{field}'.")

        for field in [
            "tagline",
            "short_description",
            "email_subject",
            "email_preheader",
            "email_opening",
            "email_call_to_action",
            "slack_fallback_text",
            "pr_reference_text",
        ]:
            value = deliverable[field]
            if not isinstance(value, str) or not value.strip():
                raise LLMError(f"Marketing field '{field}' must be a non-empty string.")
        if not isinstance(deliverable["pr_url"], str):
            raise LLMError("Marketing field 'pr_url' must be a string.")
        if not isinstance(deliverable["pr_url_available"], bool):
            raise LLMError("Marketing field 'pr_url_available' must be a boolean.")

        if len(deliverable["tagline"].split()) >= 10:
            raise LLMError("Marketing tagline must be under 10 words.")

        social_posts = deliverable["social_posts"]
        if not isinstance(social_posts, dict):
            raise LLMError("social_posts must be a JSON object.")
        for platform in ["x", "linkedin", "instagram"]:
            value = social_posts.get(platform)
            if not isinstance(value, str) or not value.strip():
                raise LLMError(f"social_posts.{platform} must be a non-empty string.")

        email_value_points = deliverable["email_value_points"]
        if not isinstance(email_value_points, list) or len(email_value_points) != 3:
            raise LLMError("email_value_points must be a list with exactly 3 items.")
        normalized_value_points = []
        for point in email_value_points:
            if not isinstance(point, str) or not point.strip():
                raise LLMError("email_value_points must contain only non-empty strings.")
            normalized_value_points.append(point.strip())

        normalized_copy = self._strengthen_positioning_copy(
            short_description=deliverable["short_description"].strip(),
            email_opening=deliverable["email_opening"].strip(),
            email_value_points=normalized_value_points,
            slack_fallback_text=deliverable["slack_fallback_text"].strip(),
        )

        email_body_text = self._render_email_body_text(
            short_description=normalized_copy["short_description"],
            email_opening=normalized_copy["email_opening"],
            email_value_points=normalized_copy["email_value_points"],
            email_call_to_action=deliverable["email_call_to_action"].strip(),
            pr_reference_text=deliverable["pr_reference_text"].strip(),
        )
        email_body_html = self._render_email_body_html(
            tagline=deliverable["tagline"].strip(),
            short_description=normalized_copy["short_description"],
            email_preheader=deliverable["email_preheader"].strip(),
            email_opening=normalized_copy["email_opening"],
            email_value_points=normalized_copy["email_value_points"],
            email_call_to_action=deliverable["email_call_to_action"].strip(),
            pr_reference_text=deliverable["pr_reference_text"].strip(),
            pr_url=deliverable["pr_url"].strip(),
            pr_url_available=deliverable["pr_url_available"],
        )

        return {
            "tagline": deliverable["tagline"].strip(),
            "short_description": normalized_copy["short_description"],
            "email_subject": deliverable["email_subject"].strip(),
            "email_body_text": email_body_text,
            "email_body_html": email_body_html,
            "social_posts": {
                "x": social_posts["x"].strip(),
                "linkedin": social_posts["linkedin"].strip(),
                "instagram": social_posts["instagram"].strip(),
            },
            "slack_fallback_text": normalized_copy["slack_fallback_text"],
            "pr_url": deliverable["pr_url"].strip(),
            "pr_url_available": deliverable["pr_url_available"],
            "pr_reference_text": deliverable["pr_reference_text"].strip(),
        }

    @staticmethod
    def _strengthen_positioning_copy(
        short_description: str,
        email_opening: str,
        email_value_points: list[str],
        slack_fallback_text: str,
    ) -> Dict[str, Any]:
        """Ensure the final marketing copy always includes required channel and industry language."""
        industry_phrase = "clinics, bakeries, and grocery stores"
        channel_phrase = "WhatsApp and phone-call automation"

        normalized_short_description = short_description.strip()
        if "whatsapp" not in normalized_short_description.lower():
            normalized_short_description = normalized_short_description.rstrip(".")
            normalized_short_description += ". Built around WhatsApp and phone-call automation"
        if "bakery" not in normalized_short_description.lower():
            normalized_short_description = normalized_short_description.rstrip(".")
            normalized_short_description += f" for {industry_phrase}"
        if not normalized_short_description.endswith("."):
            normalized_short_description += "."

        normalized_email_opening = email_opening.strip()
        if "whatsapp" not in normalized_email_opening.lower() or "bakery" not in normalized_email_opening.lower():
            normalized_email_opening = (
                "AutoServe AI helps clinics, bakeries, and grocery stores handle "
                "WhatsApp and phone-call inquiries with less manual back-and-forth."
            )

        normalized_points = [point.strip() for point in email_value_points]
        combined_points = " ".join(normalized_points).lower()
        if "bakery" not in combined_points:
            normalized_points[-1] = (
                "Support clinics, bakeries, and grocery stores with clearer booking, "
                "order, and inquiry handling."
            )
        if "whatsapp" not in " ".join(normalized_points).lower():
            normalized_points[0] = (
                "Handle WhatsApp and phone-call inquiries faster without relying on "
                "manual follow-up for every message."
            )

        normalized_slack_text = slack_fallback_text.strip()
        if "whatsapp" not in normalized_slack_text.lower() or "bakery" not in normalized_slack_text.lower():
            normalized_slack_text = (
                f"{normalized_slack_text.rstrip('.')} "
                f"Built for {industry_phrase} using {channel_phrase}."
            ).strip()

        return {
            "short_description": normalized_short_description,
            "email_opening": normalized_email_opening,
            "email_value_points": normalized_points,
            "slack_fallback_text": normalized_slack_text,
        }

    @staticmethod
    def _render_email_body_text(
        short_description: str,
        email_opening: str,
        email_value_points: list[str],
        email_call_to_action: str,
        pr_reference_text: str,
    ) -> str:
        """Render a reliable plain-text launch email from structured copy."""
        bullet_block = "\n".join(f"- {point}" for point in email_value_points)
        return (
            f"{email_opening}\n\n"
            f"{short_description}\n\n"
            f"{bullet_block}\n\n"
            f"{email_call_to_action}\n"
            f"{pr_reference_text}\n\n"
            "Best,\n"
            "AutoServe AI"
        ).strip()

    @staticmethod
    def _render_email_body_html(
        tagline: str,
        short_description: str,
        email_preheader: str,
        email_opening: str,
        email_value_points: list[str],
        email_call_to_action: str,
        pr_reference_text: str,
        pr_url: str,
        pr_url_available: bool,
    ) -> str:
        """Render a reliable HTML launch email from structured copy."""
        bullet_items = "".join(f"<li>{escape(point)}</li>" for point in email_value_points)
        pr_html = (
            f'<p style="margin:16px 0 0;"><a href="{escape(pr_url)}" '
            'style="color:#176b73;font-weight:700;text-decoration:none;">View the GitHub PR</a></p>'
            if pr_url_available and pr_url
            else f"<p style=\"margin:16px 0 0;\">{escape(pr_reference_text)}</p>"
        )
        return (
            "<!DOCTYPE html>"
            "<html lang=\"en\"><body style=\"margin:0;background:#f5efe5;font-family:Arial,sans-serif;color:#15252c;\">"
            "<div style=\"max-width:640px;margin:0 auto;padding:32px 20px;\">"
            f"<p style=\"display:none;max-height:0;overflow:hidden;opacity:0;\">{escape(email_preheader)}</p>"
            "<div style=\"background:#ffffff;border:1px solid #e6ddd1;border-radius:18px;padding:32px;\">"
            f"<p style=\"margin:0 0 10px;color:#176b73;font-weight:700;\">{escape(tagline)}</p>"
            "<h1 style=\"margin:0 0 14px;font-size:28px;line-height:1.2;\">AutoServe AI Launch Update</h1>"
            f"<p style=\"margin:0 0 16px;font-size:16px;line-height:1.7;\">{escape(email_opening)}</p>"
            f"<p style=\"margin:0 0 16px;font-size:16px;line-height:1.7;\">{escape(short_description)}</p>"
            f"<ul style=\"margin:0 0 20px 20px;padding:0;line-height:1.8;\">{bullet_items}</ul>"
            f"<p style=\"margin:0 0 12px;font-size:16px;line-height:1.7;font-weight:700;\">{escape(email_call_to_action)}</p>"
            f"{pr_html}"
            "</div></div></body></html>"
        )

    def _build_slack_blocks(
        self,
        tagline: str,
        short_description: str,
        pr_url: str,
        pr_url_available: bool,
    ) -> list[Dict[str, Any]]:
        """Build the required Slack Block Kit payload."""
        pr_block_text = (
            f"*GitHub PR:* <{pr_url}|View the landing page pull request>"
            if pr_url_available and pr_url
            else "*GitHub PR:* Pending. The landing page artifact is ready and the PR link will be added after GitHub publishing succeeds."
        )
        return [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "AutoServe AI Launch Update"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Tagline:* {tagline}\n*Launch summary:* {short_description}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": pr_block_text,
                },
            },
        ]

    def _validate_incoming_message(self, message: Dict[str, Any]) -> None:
        """Validate actionable marketing messages."""
        if message.get("to_agent") != self.agent_name:
            raise ValueError("MarketingAgent can only process messages addressed to 'marketing'.")
        if message.get("message_type") not in {"task", "revision_request"}:
            raise ValueError("MarketingAgent only accepts 'task' or 'revision_request' messages.")

        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Marketing task payload must be a dictionary.")
        required_fields = ["startup_name", "startup_idea", "product_spec", "engineer_result"]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Marketing task payload is missing '{field}'.")


__all__ = ["MarketingAgent"]
