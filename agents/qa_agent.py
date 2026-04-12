"""QA agent for reviewing AutoServe AI engineering and marketing outputs."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from message_bus import MessageBus
from utils.github_api import GitHubAPI, GitHubAPIError
from utils.llm import LLMError


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
            if isinstance(payload.get("engineer_result"), dict):
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
        """Review engineering and marketing outputs using deterministic launch checks."""
        engineer_issues = self._collect_engineering_issues(
            product_spec=product_spec,
            engineer_result=engineer_result,
        )
        marketing_issues = self._collect_marketing_issues(
            marketing_result=marketing_result,
        )

        if not engineer_issues and not marketing_issues:
            return {
                "verdict": "pass",
                "issues": [],
                "recommendations": [],
                "target_agents": [],
                "agent_feedback": {},
            }

        issues: List[str] = []
        recommendations: List[str] = []
        target_agents: List[str] = []
        agent_feedback: Dict[str, str] = {}

        if engineer_issues:
            target_agents.append("engineer")
            issues.extend(f"Engineer: {issue}" for issue in engineer_issues[:3])
            recommendations.append(
                "Revise the landing page so the headline, CTA, feature framing, and GitHub artifact metadata clearly match the approved product specification."
            )
            agent_feedback["engineer"] = (
                "Please revise the engineering output to address these issues: "
                + "; ".join(engineer_issues[:4])
            )

        if marketing_issues:
            target_agents.append("marketing")
            issues.extend(f"Marketing: {issue}" for issue in marketing_issues[:3])
            recommendations.append(
                "Revise the launch messaging so it stays specific to WhatsApp and phone-call automation for clinics, bakeries, and grocery stores without overclaiming."
            )
            agent_feedback["marketing"] = (
                "Please revise the marketing output to address these issues: "
                + "; ".join(marketing_issues[:4])
            )

        return self._validate_report(
            {
                "verdict": "fail",
                "issues": issues,
                "recommendations": recommendations,
                "target_agents": target_agents,
                "agent_feedback": agent_feedback,
            }
        )

    @staticmethod
    def _summarize_engineering_output(engineer_result: Dict[str, Any]) -> Dict[str, Any]:
        """Trim the engineering payload before sending it to the QA LLM prompt."""
        html_preview = ""
        html_value = engineer_result.get("html")
        if isinstance(html_value, str) and html_value.strip():
            cleaned_html = html_value.strip()
            html_preview = (
                cleaned_html
                if len(cleaned_html) <= 2200
                else cleaned_html[:2180].rstrip() + "\n...[truncated]"
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
    def _summarize_marketing_output(marketing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Trim the marketing payload before sending it to the QA LLM prompt."""
        email_html_preview = ""
        html_value = marketing_result.get("email_body_html")
        if isinstance(html_value, str) and html_value.strip():
            cleaned_html = html_value.strip()
            email_html_preview = (
                cleaned_html
                if len(cleaned_html) <= 1800
                else cleaned_html[:1780].rstrip() + "\n...[truncated]"
            )

        return {
            "tagline": marketing_result.get("tagline"),
            "short_description": marketing_result.get("short_description"),
            "email_subject": marketing_result.get("email_subject"),
            "email_body_text": marketing_result.get("email_body_text"),
            "email_body_html_preview": email_html_preview,
            "social_posts": marketing_result.get("social_posts"),
            "slack_fallback_text": marketing_result.get("slack_fallback_text"),
            "pr_url": marketing_result.get("pr_url"),
            "email_result": marketing_result.get("email_result"),
            "slack_result": marketing_result.get("slack_result"),
        }

    def _collect_engineering_issues(
        self,
        product_spec: Dict[str, Any],
        engineer_result: Dict[str, Any],
    ) -> List[str]:
        """Apply deterministic launch-readiness checks to the engineering output."""
        issues: List[str] = []
        if not isinstance(engineer_result, dict):
            return ["engineering output is not a JSON object"]

        required_fields = {
            "headline": "add a landing-page headline",
            "subheadline": "add a landing-page subheadline",
            "call_to_action": "add a primary CTA",
            "summary": "add an engineering summary",
            "landing_page_path": "include the landing page path",
            "branch": "include the branch name",
            "issue_url": "include the GitHub issue URL",
            "pr_url": "include the GitHub pull request URL",
            "commit_sha": "include the GitHub commit SHA",
        }
        for field_name, issue in required_fields.items():
            value = engineer_result.get(field_name)
            if not isinstance(value, str) or not value.strip():
                issues.append(issue)

        feature_text = " ".join(
            " ".join(str(feature.get(field, "")) for field in ("name", "description"))
            for feature in product_spec.get("features", [])
            if isinstance(feature, dict)
        ).lower()
        combined_text = " ".join(
            str(engineer_result.get(field, ""))
            for field in ("headline", "subheadline", "summary", "html")
        ).lower()

        if "whatsapp" not in combined_text:
            issues.append("mention WhatsApp automation in the landing page")
        if "phone" not in combined_text and "call" not in combined_text:
            issues.append("mention phone or call automation in the landing page")
        if not any(term in str(engineer_result.get("call_to_action", "")).lower() for term in ("book", "demo", "get started", "start", "launch")):
            issues.append("make the primary CTA more explicit")

        if "booking" in feature_text and "booking" not in combined_text:
            issues.append("reflect booking capture in the landing-page messaging")
        if "order" in feature_text and "order" not in combined_text:
            issues.append("reflect order capture in the landing-page messaging")

        repo_slug = os.getenv("GITHUB_REPO", "").strip()
        if repo_slug:
            repo_prefix = f"https://github.com/{repo_slug}/"
            for field_name, label in (("issue_url", "issue"), ("pr_url", "pull request")):
                value = engineer_result.get(field_name)
                if isinstance(value, str) and value.strip() and not value.startswith(repo_prefix):
                    issues.append(f"make sure the GitHub {label} URL points to the configured repository")

        html_value = engineer_result.get("html")
        if not isinstance(html_value, str) or not html_value.strip():
            issues.append("include the generated HTML in the engineering payload")
        else:
            html_lower = html_value.lower()
            if "<!doctype html>" not in html_lower:
                issues.append("start the generated HTML with a proper doctype")
            if "<h1" not in html_lower:
                issues.append("include a visible hero headline in the HTML")
            if self._contains_unsupported_quantified_claims(html_lower):
                issues.append("remove unsupported quantified claims from the landing page copy")

        return issues

    def _collect_marketing_issues(self, marketing_result: Dict[str, Any]) -> List[str]:
        """Apply deterministic launch-readiness checks to the marketing output."""
        issues: List[str] = []
        if not isinstance(marketing_result, dict):
            return ["marketing output is not a JSON object"]

        required_fields = {
            "tagline": "add a concise tagline",
            "short_description": "add a short launch description",
            "email_subject": "add an email subject line",
            "email_body_text": "add a plain-text email body",
            "email_body_html": "add an HTML email body",
            "slack_fallback_text": "add Slack fallback text",
        }
        for field_name, issue in required_fields.items():
            value = marketing_result.get(field_name)
            if not isinstance(value, str) or not value.strip():
                issues.append(issue)

        social_posts = marketing_result.get("social_posts")
        if not isinstance(social_posts, dict):
            issues.append("include platform-specific social posts")
        else:
            for platform in ("x", "linkedin", "instagram"):
                value = social_posts.get(platform)
                if not isinstance(value, str) or not value.strip():
                    issues.append(f"include a non-empty {platform} post")

        combined_text = " ".join(
            str(marketing_result.get(field, ""))
            for field in (
                "tagline",
                "short_description",
                "email_subject",
                "email_body_text",
                "email_body_html",
                "slack_fallback_text",
            )
        ).lower()
        if "whatsapp" not in combined_text:
            issues.append("mention WhatsApp in the marketing copy")
        if "phone" not in combined_text and "call" not in combined_text:
            issues.append("mention phone or call automation in the marketing copy")

        industry_variants = {
            "clinic": ("clinic", "clinics"),
            "bakery": ("bakery", "bakeries"),
            "grocery": ("grocery", "groceries", "grocery store", "grocery stores"),
        }
        for label, variants in industry_variants.items():
            if not any(term in combined_text for term in variants):
                issues.append(f"mention {label}-style businesses in the marketing copy")

        if self._contains_unsupported_quantified_claims(combined_text):
            issues.append("remove unsupported quantified claims from the marketing copy")

        email_result = marketing_result.get("email_result")
        if not isinstance(email_result, dict):
            issues.append("include the email delivery result details")
        else:
            if not email_result.get("to_email"):
                issues.append("include the email recipient in the delivery result")
            if not email_result.get("message"):
                issues.append("include the email delivery confirmation message")

        slack_result = marketing_result.get("slack_result")
        if not isinstance(slack_result, dict):
            issues.append("include the Slack delivery result details")
        else:
            if not slack_result.get("channel"):
                issues.append("include the Slack channel in the delivery result")
            if not slack_result.get("ts"):
                issues.append("include the Slack message timestamp in the delivery result")

        repo_slug = os.getenv("GITHUB_REPO", "").strip()
        pr_url = marketing_result.get("pr_url")
        if repo_slug and isinstance(pr_url, str) and pr_url.strip():
            expected_prefix = f"https://github.com/{repo_slug}/"
            if not pr_url.startswith(expected_prefix):
                issues.append("make sure the PR URL points to the configured repository")

        return issues

    @staticmethod
    def _contains_unsupported_quantified_claims(text: str) -> bool:
        """Detect unsupported quantified marketing claims that should not pass QA."""
        if not isinstance(text, str):
            return False
        patterns = (
            r"\b\d{1,3}%\b",
            r"\b\d+(?:\.\d+)?x\b",
            r"\bdouble(?:d)?\b",
            r"\btrip(?:le|led)\b",
            r"\bsave \d+",
            r"\b\d+\s+hours?\b",
            r"\b\d+\s+minutes?\b",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _post_review_comments(
        self,
        engineer_result: Dict[str, Any],
        report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Post at least two inline PR review comments on the landing page HTML."""
        pr_number = engineer_result.get("pr_number")
        if not isinstance(pr_number, int):
            return []

        github = GitHubAPI()
        posted_comments: List[Dict[str, Any]] = []
        path = engineer_result.get("landing_page_path")
        if not isinstance(path, str) or not path.strip():
            path = "landing_page.html"

        commit_id = github.get_pull_request_head_sha(pr_number)
        html_content = engineer_result.get("html", "")
        candidate_comments = self._build_inline_review_comments(
            engineer_result=engineer_result,
            report=report,
            html_content=html_content if isinstance(html_content, str) else "",
        )

        for comment in candidate_comments:
            try:
                posted_comments.append(
                    github.create_pr_review_comment(
                        pull_number=pr_number,
                        body=comment["body"],
                        commit_id=commit_id,
                        path=path,
                        line=comment["line"],
                    )
                )
            except GitHubAPIError:
                posted_comments.append(
                    github.create_issue_comment(
                        issue_number=pr_number,
                        body=f"QA fallback review: {comment['body']} (intended for {path}:{comment['line']})",
                    )
                )

        return posted_comments

    def _build_inline_review_comments(
        self,
        engineer_result: Dict[str, Any],
        report: Dict[str, Any],
        html_content: str,
    ) -> List[Dict[str, Any]]:
        """Build two anchored review comments on the landing page HTML."""
        engineer_issues = [
            issue.split("Engineer: ", 1)[1].strip()
            for issue in report.get("issues", [])
            if isinstance(issue, str) and issue.startswith("Engineer: ")
        ]
        recommendations = [
            item.strip()
            for item in report.get("recommendations", [])
            if isinstance(item, str) and item.strip()
        ]

        headline_line = self._find_line_number(
            html_content,
            candidates=("<h1", "class=\"hero-copy\"", "<section class=\"hero\""),
            fallback=1,
        )
        cta_line = self._find_line_number(
            html_content,
            candidates=("btn btn-primary", "href=\"#cta\"", "id=\"cta\""),
            fallback=max(1, headline_line),
        )

        if report.get("verdict") == "pass":
            headline_issue = (
                "QA review note: the hero headline and subheadline clearly communicate "
                "WhatsApp and phone-call automation for clinics, bakeries, and grocery stores."
            )
            cta_recommendation = (
                "QA review note: the primary CTA is visible and makes the next step "
                "clear for a launch reviewer."
            )
        else:
            headline_issue = (
                engineer_issues[0]
                if engineer_issues
                else "Ensure the hero section communicates the approved WhatsApp and phone-call value proposition clearly."
            )
            cta_recommendation = (
                recommendations[0]
                if recommendations
                else "Tighten the CTA so a reviewer can immediately see the next action and why it matters."
            )

        comments = [
            {
                "line": headline_line,
                "body": headline_issue if headline_issue.startswith("QA review") else f"QA review: {headline_issue}",
            },
            {
                "line": cta_line,
                "body": cta_recommendation if cta_recommendation.startswith("QA review") else f"QA recommendation: {cta_recommendation}",
            },
        ]

        # Keep the assignment requirement explicit: at least two comments on the HTML file.
        return comments[:2]

    @staticmethod
    def _find_line_number(
        text: str,
        candidates: tuple[str, ...],
        fallback: int,
    ) -> int:
        """Find the first matching line number for a list of HTML markers."""
        if not isinstance(text, str) or not text:
            return fallback

        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            lowered = line.lower()
            for candidate in candidates:
                if candidate.lower() in lowered:
                    return index

        return fallback

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
