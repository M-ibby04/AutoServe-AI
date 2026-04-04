"""QA agent for reviewing AutoServe AI engineering and marketing outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from message_bus import MessageBus
from utils.github_api import GitHubAPI, GitHubAPIError
from utils.llm import LLMError, call_llm, sanitize_llm_json


class QAAgent:
    """Review launch outputs, raise issues, and post GitHub review comments when needed."""

    agent_name = "qa"

    def __init__(self, message_bus: MessageBus) -> None:
        """Store the shared message bus."""
        self.message_bus = message_bus

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Review the launch package and send a structured QA report to the CEO."""
        self._validate_incoming_message(message)
        payload = message["payload"]

        print("[QA Agent] Reviewing AutoServe AI landing page and marketing output...")

        try:
            report = self.review_outputs(
                startup_name=payload["startup_name"],
                startup_idea=payload["startup_idea"],
                product_spec=payload["product_spec"],
                engineer_result=payload["engineer_result"],
                marketing_result=payload["marketing_result"],
            )

            review_comments: List[Dict[str, Any]] = []
            if report["verdict"] == "fail" and "engineer" in report["target_agents"]:
                review_comments = self._post_review_comments(
                    engineer_result=payload["engineer_result"],
                    report=report,
                )

            result_payload = {
                "verdict": report["verdict"],
                "issues": report["issues"],
                "recommendations": report["recommendations"],
                "target_agents": report["target_agents"],
                "agent_feedback": report["agent_feedback"],
                "review_comments": review_comments,
            }

            print(f"[QA Agent] QA verdict: {report['verdict']}.")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="ceo",
                message_type="result",
                payload=result_payload,
                parent_message_id=message["message_id"],
            )
        except (LLMError, GitHubAPIError, ValueError) as exc:
            print(f"[QA Agent] Failure: {exc}")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="ceo",
                message_type="failure",
                payload={"error": str(exc), "stage": "qa"},
                parent_message_id=message["message_id"],
            )

    def review_outputs(
        self,
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
        engineer_result: Dict[str, Any],
        marketing_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use the LLM to evaluate engineering and marketing alignment."""
        system_prompt = (
            "You are the QA Agent for AutoServe AI. Return only valid JSON with no "
            "markdown fences."
        )
        user_prompt = f"""
Startup name: {startup_name}

Startup idea:
{startup_idea}

Approved product specification:
{json.dumps(product_spec, indent=2)}

Engineer output:
{json.dumps(engineer_result, indent=2)}

Marketing output:
{json.dumps(marketing_result, indent=2)}

Review both outputs for:
- alignment with the product specification
- realism of claims
- clarity of the landing page and CTA
- specificity to clinics, bakeries, grocery stores, WhatsApp, and phone-call automation
- quality of the marketing tone and launch messaging

Return only valid JSON with exactly this structure:
{{
  "verdict": "pass" or "fail",
  "issues": ["string"],
  "recommendations": ["string"],
  "target_agents": ["engineer", "marketing"],
  "agent_feedback": {{
    "engineer": "string",
    "marketing": "string"
  }}
}}

Rules:
- Include at least one issue and one recommendation if verdict is fail.
- target_agents should name only the agents that need changes.
- agent_feedback should be specific and actionable.
""".strip()

        parsed = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        return self._validate_report(parsed)

    def _post_review_comments(
        self,
        engineer_result: Dict[str, Any],
        report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Post at least two safe PR comments when engineering issues are found."""
        pr_number = engineer_result.get("pr_number")
        if not isinstance(pr_number, int):
            return []

        github = GitHubAPI()
        comments_to_post = [
            {
                "body": f"QA note: {report['issues'][0]}",
            },
            {
                "body": (
                    "QA recommendation: "
                    f"{report['recommendations'][0] if report['recommendations'] else 'Tighten the page copy and CTA clarity.'}"
                ),
            },
        ]

        posted_comments: List[Dict[str, Any]] = []
        for comment in comments_to_post:
            posted_comments.append(
                github.create_issue_comment(
                    issue_number=pr_number,
                    body=f"QA review: {comment['body']}",
                )
            )

        return posted_comments

    def _validate_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the QA report shape."""
        verdict = report.get("verdict")
        issues = report.get("issues")
        recommendations = report.get("recommendations")
        target_agents = report.get("target_agents")
        agent_feedback = report.get("agent_feedback")

        if verdict not in {"pass", "fail"}:
            raise LLMError("QA verdict must be either 'pass' or 'fail'.")
        if not isinstance(issues, list):
            raise LLMError("QA issues must be a list.")
        if not isinstance(recommendations, list):
            raise LLMError("QA recommendations must be a list.")
        if not isinstance(target_agents, list):
            raise LLMError("QA target_agents must be a list.")
        if not isinstance(agent_feedback, dict):
            raise LLMError("QA agent_feedback must be an object.")

        normalized_issues = self._normalize_string_list(issues, "issues")
        normalized_recommendations = self._normalize_string_list(
            recommendations, "recommendations"
        )
        normalized_targets = [
            target for target in target_agents if target in {"engineer", "marketing"}
        ]

        normalized_feedback: Dict[str, str] = {}
        for agent_name in ["engineer", "marketing"]:
            feedback = agent_feedback.get(agent_name, "")
            if isinstance(feedback, str) and feedback.strip():
                normalized_feedback[agent_name] = feedback.strip()

        if verdict == "fail":
            if not normalized_issues:
                raise LLMError("QA fail verdict requires at least one issue.")
            if not normalized_recommendations:
                raise LLMError("QA fail verdict requires at least one recommendation.")
            if not normalized_targets:
                raise LLMError("QA fail verdict requires at least one target agent.")

        return {
            "verdict": verdict,
            "issues": normalized_issues,
            "recommendations": normalized_recommendations,
            "target_agents": normalized_targets,
            "agent_feedback": normalized_feedback,
        }

    def _normalize_string_list(self, values: Any, field_name: str) -> List[str]:
        """Validate a list of non-empty strings."""
        if not isinstance(values, list):
            raise LLMError(f"Field '{field_name}' must be a list.")
        normalized: List[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            normalized.append(value.strip())
        return normalized

    def _validate_incoming_message(self, message: Dict[str, Any]) -> None:
        """Validate QA task payloads."""
        if message.get("to_agent") != self.agent_name:
            raise ValueError("QAAgent can only process messages addressed to 'qa'.")
        if message.get("message_type") not in {"task", "revision_request"}:
            raise ValueError("QAAgent only accepts 'task' or 'revision_request' messages.")

        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("QA payload must be a dictionary.")
        required_fields = [
            "startup_name",
            "startup_idea",
            "product_spec",
            "engineer_result",
            "marketing_result",
        ]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"QA payload is missing '{field}'.")


__all__ = ["QAAgent"]
