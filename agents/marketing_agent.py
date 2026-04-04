"""Marketing agent for AutoServe AI launch messaging."""

from __future__ import annotations

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
        user_prompt = self._build_prompt(
            startup_name=startup_name,
            startup_idea=startup_idea,
            product_spec=product_spec,
            engineer_result=engineer_result,
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
            revision_section = (
                "Revision feedback from CEO or QA:\n"
                f"{feedback}\n\n"
                "Previous marketing output:\n"
                f"{json.dumps(previous_result or {}, indent=2)}\n\n"
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
  "email_body_text": "string",
  "email_body_html": "string",
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
- The email should sound like outreach to a small-business lead.
- social posts must feel platform-appropriate but still practical.
- slack_fallback_text must mention the tagline and a one-line launch summary.
- If a PR link is available, reference it naturally.
- If a PR link is not available, use a fallback phrase like 'GitHub PR pending' or 'landing page PR coming soon'.
- Do not include any text outside the JSON object.
""".strip()

    def _validate_deliverable(self, deliverable: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the marketing package."""
        required_fields = [
            "tagline",
            "short_description",
            "email_subject",
            "email_body_text",
            "email_body_html",
            "social_posts",
            "slack_fallback_text",
            "pr_url",
            "pr_url_available",
            "pr_reference_text",
        ]
        for field in required_fields:
            if field not in deliverable:
                raise LLMError(f"Marketing deliverable is missing '{field}'.")

        for field in ["tagline", "short_description", "email_subject", "email_body_text", "email_body_html", "slack_fallback_text", "pr_reference_text"]:
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

        return {
            "tagline": deliverable["tagline"].strip(),
            "short_description": deliverable["short_description"].strip(),
            "email_subject": deliverable["email_subject"].strip(),
            "email_body_text": deliverable["email_body_text"].strip(),
            "email_body_html": deliverable["email_body_html"].strip(),
            "social_posts": {
                "x": social_posts["x"].strip(),
                "linkedin": social_posts["linkedin"].strip(),
                "instagram": social_posts["instagram"].strip(),
            },
            "slack_fallback_text": deliverable["slack_fallback_text"].strip(),
            "pr_url": deliverable["pr_url"].strip(),
            "pr_url_available": deliverable["pr_url_available"],
            "pr_reference_text": deliverable["pr_reference_text"].strip(),
        }

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
