"""Engineer agent for generating and publishing the AutoServe AI landing page."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from message_bus import MessageBus
from utils.github_api import GitHubAPI, GitHubAPIError
from utils.llm import LLMError, call_llm, sanitize_llm_json


class EngineerAgent:
    """Generate AutoServe AI landing page assets and publish them to GitHub."""

    agent_name = "engineer"

    def __init__(self, message_bus: MessageBus, workspace_root: str) -> None:
        """Initialize the engineer agent and local workspace paths."""
        self.message_bus = message_bus
        self.workspace_root = Path(workspace_root)
        self.landing_page_path = self.workspace_root / "landing_page.html"
        self.preview_product_spec: Optional[Dict[str, Any]] = None

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle preview, task, and revision messages."""
        if message.get("message_type") == "result":
            payload = message.get("payload", {})
            if payload.get("status") == "awaiting_ceo_approval":
                self.preview_product_spec = payload.get("product_spec")
                print("[Engineer Agent] Received product spec preview. Waiting for CEO approval...")
                return None

        self._validate_incoming_message(message)
        payload = message["payload"]
        feedback = payload.get("feedback")
        previous_result = payload.get("previous_result")

        if message["message_type"] == "revision_request":
            print("[Engineer Agent] Revision request received from CEO.")
            print(f"[Engineer Agent] Feedback: {feedback}")
        else:
            print("[Engineer Agent] Approved engineering task received from CEO.")
        print("[Engineer Agent] Generating AutoServe AI landing page package...")

        try:
            deliverable = self.generate_deliverable(
                startup_name=payload["startup_name"],
                startup_idea=payload["startup_idea"],
                product_spec=payload["product_spec"],
                launch_goal=payload.get("launch_goal", ""),
                feedback=feedback,
                previous_result=previous_result,
            )

            self._write_landing_page(deliverable["landing_page_path"], deliverable["html"])
            github_result = self._publish_to_github(deliverable, previous_result)

            result_payload = {
                "startup_name": payload["startup_name"],
                "startup_idea": payload["startup_idea"],
                "product_spec": payload["product_spec"],
                "landing_page_path": deliverable["landing_page_path"],
                "landing_page_local_path": str(self.landing_page_path),
                "headline": deliverable["headline"],
                "subheadline": deliverable["subheadline"],
                "call_to_action": deliverable["call_to_action"],
                "branch": github_result["branch"],
                "issue_url": github_result["issue_url"],
                "issue_number": github_result["issue_number"],
                "pr_url": github_result["pr_url"],
                "pr_number": github_result["pr_number"],
                "commit_sha": github_result["commit_sha"],
                "html": deliverable["html"],
                "summary": deliverable["summary"],
            }

            print("[Engineer Agent] GitHub issue, branch, commit, and PR are ready.")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="ceo",
                message_type="result",
                payload=result_payload,
                parent_message_id=message["message_id"],
            )
        except (LLMError, GitHubAPIError, OSError, ValueError) as exc:
            print(f"[Engineer Agent] Failure: {exc}")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="ceo",
                message_type="failure",
                payload={
                    "error": str(exc),
                    "stage": "engineering",
                },
                parent_message_id=message["message_id"],
            )

    def generate_deliverable(
        self,
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
        launch_goal: str,
        feedback: Optional[str],
        previous_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a landing page package using the LLM."""
        system_prompt = (
            "You are the Engineer Agent for AutoServe AI. Return only valid JSON "
            "with no markdown fences."
        )
        user_prompt = self._build_prompt(
            startup_name=startup_name,
            startup_idea=startup_idea,
            product_spec=product_spec,
            launch_goal=launch_goal,
            feedback=feedback,
            previous_result=previous_result,
        )
        parsed = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        return self._validate_deliverable(parsed)

    def _build_prompt(
        self,
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
        launch_goal: str,
        feedback: Optional[str],
        previous_result: Optional[Dict[str, Any]],
    ) -> str:
        """Build the AutoServe AI engineering prompt."""
        revision_section = ""
        if feedback:
            revision_section = (
                "Revision feedback from CEO or QA:\n"
                f"{feedback}\n\n"
                "Previous engineering output:\n"
                f"{json.dumps(previous_result or {}, indent=2)}\n\n"
                "Improve the existing landing page instead of starting from scratch.\n\n"
            )

        return f"""
Startup name: {startup_name}

Startup idea:
{startup_idea}

Launch goal:
{launch_goal}

Approved product specification:
{json.dumps(product_spec, indent=2)}

{revision_section}Create the landing page and GitHub execution package for AutoServe AI.

Requirements for the landing page:
- Build a complete HTML page with embedded CSS.
- The page must feel specific to AutoServe AI.
- Mention WhatsApp and phone-call automation.
- Cover inquiries, bookings, and orders.
- Speak to clinics, bakeries, grocery stores, and similar service businesses.
- Include: hero section, problem/solution framing, industry use cases, key features, why AutoServe AI, CTA section.
- Keep tone practical and MVP-credible.
- Include one primary CTA button.

Return only valid JSON with exactly this structure:
{{
  "landing_page_path": "landing_page.html",
  "headline": "string",
  "subheadline": "string",
  "call_to_action": "string",
  "summary": "string",
  "issue_title": "Initial landing page",
  "issue_body": "string",
  "branch_name": "string",
  "commit_message": "string",
  "pr_title": "string",
  "pr_body": "string",
  "html": "full html string"
}}

Rules:
- branch_name should be short, lowercase, and git-safe.
- issue_title can stay 'Initial landing page' unless a revision title is clearly better.
- html must start with <!DOCTYPE html>.
- Do not include markdown fences or commentary.
""".strip()

    def _validate_deliverable(self, deliverable: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated landing page package."""
        required_string_fields = [
            "landing_page_path",
            "headline",
            "subheadline",
            "call_to_action",
            "summary",
            "issue_title",
            "issue_body",
            "branch_name",
            "commit_message",
            "pr_title",
            "pr_body",
            "html",
        ]
        for field in required_string_fields:
            value = deliverable.get(field)
            if not isinstance(value, str) or not value.strip():
                raise LLMError(f"Engineer deliverable field '{field}' must be a non-empty string.")

        html = deliverable["html"].strip()
        html_lower = html.lower()
        if not html_lower.startswith("<!doctype html>"):
            raise LLMError("Landing page HTML must start with <!DOCTYPE html>.")
        if "<html" not in html_lower:
            raise LLMError("Landing page HTML must include an <html> element.")
        if not any(marker in html_lower for marker in ["<h1", "<header", "hero"]):
            raise LLMError("Landing page HTML must include a hero/headline structure.")
        if not any(marker in html_lower for marker in ["whatsapp", "message automation", "messaging automation"]):
            raise LLMError(
                "Landing page HTML must mention WhatsApp or messaging automation."
            )
        if not any(marker in html_lower for marker in ["phone", "call automation", "voice automation", "phone-call automation", "missed calls"]):
            raise LLMError("Landing page HTML must mention phone or call automation.")
        if not any(marker in html_lower for marker in ["<button", "call-to-action", "get started", "book a demo", "start automating"]):
            raise LLMError("Landing page HTML must include a clear CTA/button.")

        branch_name = self._sanitize_branch_name(deliverable["branch_name"])
        deliverable["branch_name"] = branch_name
        deliverable["landing_page_path"] = "landing_page.html"
        deliverable["html"] = html
        return deliverable

    def _publish_to_github(
        self,
        deliverable: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create or update GitHub artifacts for the landing page."""
        github = GitHubAPI()

        issue_number = None
        issue_url = None
        branch_name = deliverable["branch_name"]
        pr_number = None
        pr_url = None

        if previous_result:
            issue_number = previous_result.get("issue_number")
            issue_url = previous_result.get("issue_url")
            branch_name = previous_result.get("branch", branch_name)
            pr_number = previous_result.get("pr_number")
            pr_url = previous_result.get("pr_url")

        if not issue_number:
            print("[Engineer Agent] Creating GitHub issue...")
            issue = github.create_issue(
                title=deliverable["issue_title"],
                body=deliverable["issue_body"],
            )
            issue_number = issue["number"]
            issue_url = issue["html_url"]

        print("[Engineer Agent] Creating or reusing GitHub branch...")
        github.create_branch(branch_name)

        print("[Engineer Agent] Committing landing page to GitHub...")
        commit_response = github.upsert_file(
            path=deliverable["landing_page_path"],
            content=deliverable["html"],
            commit_message=deliverable["commit_message"],
            branch=branch_name,
        )
        commit_sha = commit_response.get("commit", {}).get("sha")
        if not isinstance(commit_sha, str) or not commit_sha.strip():
            raise GitHubAPIError("GitHub commit response did not include a commit SHA.")

        if not pr_number:
            print("[Engineer Agent] Opening GitHub pull request...")
            pull_request = github.create_pull_request(
                title=deliverable["pr_title"],
                body=deliverable["pr_body"],
                head_branch=branch_name,
            )
            pr_number = pull_request["number"]
            pr_url = pull_request["html_url"]

        return {
            "branch": branch_name,
            "issue_number": issue_number,
            "issue_url": issue_url,
            "pr_number": pr_number,
            "pr_url": pr_url,
            "commit_sha": commit_sha.strip(),
        }

    def _write_landing_page(self, relative_path: str, html: str) -> None:
        """Write the generated landing page into the local workspace for demo use."""
        if relative_path != "landing_page.html":
            raise ValueError("EngineerAgent only writes landing_page.html in this project.")
        self.landing_page_path.write_text(html, encoding="utf-8")
        print(f"[Engineer Agent] Local landing page written to {self.landing_page_path}.")

    def _validate_incoming_message(self, message: Dict[str, Any]) -> None:
        """Validate task and revision payloads."""
        if message.get("to_agent") != self.agent_name:
            raise ValueError("EngineerAgent can only process messages addressed to 'engineer'.")
        if message.get("message_type") not in {"task", "revision_request"}:
            raise ValueError("EngineerAgent only accepts 'task' or 'revision_request' messages.")

        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Engineer task payload must be a dictionary.")
        required_fields = ["startup_name", "startup_idea", "product_spec"]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Engineer task payload is missing '{field}'.")

    @staticmethod
    def _sanitize_branch_name(branch_name: str) -> str:
        """Normalize a git-safe branch name."""
        cleaned = branch_name.strip().lower()
        cleaned = re.sub(r"[^a-z0-9/_-]+", "-", cleaned)
        cleaned = cleaned.strip("-/")
        if not cleaned:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            cleaned = f"launchmind/autoserve-ai-{timestamp}"
        return cleaned


__all__ = ["EngineerAgent"]
