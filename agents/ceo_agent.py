"""CEO agent for orchestrating the LaunchMind workflow."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from message_bus import MessageBus
from utils.llm import LLMError, call_llm, sanitize_llm_json


class CEOAgent:
    """Coordinate the full workflow and make review-based decisions."""

    agent_name = "ceo"

    def __init__(self, message_bus: MessageBus) -> None:
        """Store shared state needed for orchestration."""
        self.message_bus = message_bus
        self.startup_name = "AutoServe AI"
        self.state: Dict[str, Any] = {
            "startup_idea": "",
            "product_spec": None,
            "engineer_result": None,
            "marketing_result": None,
            "qa_report": None,
            "final_summary": None,
            "completed": False,
        }
        self.decision_log: List[Dict[str, Any]] = []
        self.revision_counts = {"product": 0, "engineer": 0, "marketing": 0}
        self.max_revisions = _read_positive_int_env("MAX_REVISIONS", 3)
        self.pending_revision_targets: set[str] = set()
        self.qa_dispatch_signature: Optional[str] = None

    def start_workflow(self, startup_idea: str) -> Dict[str, Any]:
        """Begin the workflow by creating the first Product Agent task."""
        if not isinstance(startup_idea, str) or not startup_idea.strip():
            raise ValueError("startup_idea must be a non-empty string.")

        self.state["startup_idea"] = startup_idea.strip()
        print("[CEO Agent] AutoServe AI workflow started.")
        print("[CEO Agent] Decomposing startup idea into the first structured product task...")

        task_payload = self.plan_product_task(self.state["startup_idea"])
        self._log_decision(
            stage="task_planning",
            verdict="dispatch",
            summary="Initial product task created and sent to Product Agent.",
        )

        return self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="product",
            message_type="task",
            payload=task_payload,
        )

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process inbound messages addressed to the CEO and orchestrate next steps."""
        self._validate_incoming_message(message)

        print(
            f"[CEO Agent] Received {message['message_type']} from "
            f"{message['from_agent']}."
        )

        from_agent = message["from_agent"]
        if from_agent == "product":
            return self._handle_product_message(message)
        if from_agent == "engineer":
            return self._handle_engineer_message(message)
        if from_agent == "marketing":
            return self._handle_marketing_message(message)
        if from_agent == "qa":
            return self._handle_qa_message(message)

        raise ValueError(f"Unsupported sender for CEOAgent: {from_agent}")

    def is_workflow_complete(self) -> bool:
        """Return whether the workflow reached a final successful state."""
        return bool(self.state["completed"])

    def build_final_summary(self) -> Dict[str, Any]:
        """Generate a final structured launch summary using the LLM."""
        if not self.state["product_spec"]:
            raise ValueError("Cannot build final summary before the product spec exists.")
        if not self.state["engineer_result"]:
            raise ValueError("Cannot build final summary before engineer output exists.")
        if not self.state["marketing_result"]:
            raise ValueError("Cannot build final summary before marketing output exists.")
        if not self.state["qa_report"]:
            raise ValueError("Cannot build final summary before QA output exists.")

        system_prompt = (
            "You are the CEO Agent for AutoServe AI. Return only valid JSON with "
            "no markdown fences and no extra text."
        )
        user_prompt = self._build_final_summary_prompt()
        try:
            summary = sanitize_llm_json(
                call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            )
        except LLMError:
            summary = self._build_fallback_final_summary()
        self.state["final_summary"] = summary
        self.state["completed"] = True
        self._log_decision(
            stage="final_summary",
            verdict="pass",
            summary="Workflow completed and final summary prepared.",
        )
        return summary

    def plan_product_task(self, startup_idea: str) -> Dict[str, Any]:
        """Use the LLM to turn the startup idea into a product task payload."""
        system_prompt = (
            "You are the CEO Agent of AutoServe AI. Create a clear structured task "
            "for the Product Agent. Return only valid JSON."
        )
        user_prompt = f"""
Startup name: {self.startup_name}

Startup idea:
{startup_idea}

Business direction to preserve:
- Target users: small business owners/managers in clinics, bakeries, grocery stores, and similar service businesses
- Channels: WhatsApp + phone calls
- Core value: automate repetitive customer interactions
- Outcomes: faster response time, fewer missed inquiries, easier bookings/orders, reduced manual burden
- Product feel: practical MVP for a student demo, but still commercially credible

Return only valid JSON with exactly this structure:
{{
  "startup_idea": "string",
  "focus_area": "string",
  "objectives": ["string"],
  "constraints": ["string"]
}}

Rules:
- focus_area must be specific to product definition for AutoServe AI.
- Include 3 to 5 concrete objectives.
- Include 2 to 4 practical constraints.
- Keep everything tailored to WhatsApp and phone automation for small businesses.
""".strip()

        parsed = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        focus_area = parsed.get("focus_area")
        objectives = parsed.get("objectives")
        constraints = parsed.get("constraints")

        if not isinstance(focus_area, str) or not focus_area.strip():
            raise LLMError("CEO product task must include a non-empty focus_area.")
        if not isinstance(objectives, list) or not objectives:
            raise LLMError("CEO product task must include a non-empty objectives list.")
        if not isinstance(constraints, list) or not constraints:
            raise LLMError("CEO product task must include a non-empty constraints list.")

        return {
            "startup_idea": startup_idea,
            "focus_area": focus_area.strip(),
            "objectives": self._normalize_string_list(objectives, "objectives"),
            "constraints": self._normalize_string_list(constraints, "constraints"),
        }

    def review_product_spec(self, product_spec: Dict[str, Any]) -> Dict[str, str]:
        """Review the Product Agent output using concrete launch-readiness checks."""
        issues = self._collect_product_spec_issues(product_spec)
        if not issues:
            return {
                "verdict": "pass",
                "feedback": (
                    "The product specification is structurally complete, specific to "
                    "AutoServe AI, and actionable for Engineering and Marketing."
                ),
            }

        return {
            "verdict": "revise",
            "feedback": "Please revise the product spec to address these gaps: " + "; ".join(issues),
        }

    def review_engineering_output(self, engineering_result: Dict[str, Any]) -> Dict[str, str]:
        """Review the Engineer Agent output using concrete launch-readiness checks."""
        issues = self._collect_engineering_issues(engineering_result)
        if not issues:
            return {
                "verdict": "pass",
                "feedback": (
                    "The engineering output is structurally complete, aligned with the "
                    "product specification, and backed by believable GitHub artifacts."
                ),
            }

        return {
            "verdict": "revise",
            "feedback": "Please revise the engineering output to address these gaps: " + "; ".join(issues),
        }

    def review_marketing_output(self, marketing_result: Dict[str, Any]) -> Dict[str, str]:
        """Review the Marketing Agent output using concrete launch-readiness checks."""
        issues = self._collect_marketing_issues(marketing_result)
        if not issues:
            return {
                "verdict": "pass",
                "feedback": (
                    "The marketing output is specific to AutoServe AI, commercially "
                    "credible, and ready for launch delivery across email and Slack."
                ),
            }

        return {
            "verdict": "revise",
            "feedback": "Please revise the marketing output to address these gaps: " + "; ".join(issues),
        }

    def review_qa_report(self, qa_report: Dict[str, Any]) -> Dict[str, str]:
        """Review the QA verdict and decide the next orchestration step."""
        if not isinstance(qa_report, dict):
            return {
                "verdict": "act",
                "feedback": "QA returned an invalid report structure. Re-run the required revisions.",
            }

        verdict = qa_report.get("verdict")
        if verdict == "pass":
            return {
                "verdict": "accept",
                "feedback": "QA approved the launch package and no further revisions are required.",
            }

        issues = qa_report.get("issues")
        recommendations = qa_report.get("recommendations")
        issue_text = "; ".join(
            issue.strip() for issue in issues if isinstance(issue, str) and issue.strip()
        ) if isinstance(issues, list) else ""
        recommendation_text = "; ".join(
            item.strip()
            for item in recommendations
            if isinstance(item, str) and item.strip()
        ) if isinstance(recommendations, list) else ""

        parts = ["QA identified launch issues that still need revisions."]
        if issue_text:
            parts.append(f"Issues: {issue_text}.")
        if recommendation_text:
            parts.append(f"Recommendations: {recommendation_text}.")

        return {
            "verdict": "act",
            "feedback": " ".join(parts).strip(),
        }

    def _handle_product_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Review Product Agent output and either revise or move forward."""
        payload = message["payload"]
        product_spec = payload["product_spec"]
        review = self.review_product_spec(product_spec)
        verdict = review["verdict"]
        feedback = review["feedback"]

        self._log_decision(stage="product_review", verdict=verdict, summary=feedback)

        if verdict == "revise":
            return self._request_product_revision(message, payload, feedback)

        print("[CEO Agent] Product specification approved.")
        self.state["product_spec"] = product_spec
        engineer_task = self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="engineer",
            message_type="task",
            payload={
                "startup_name": self.startup_name,
                "startup_idea": payload["startup_idea"],
                "product_spec": product_spec,
                "launch_goal": "Create the first public landing page and GitHub PR for AutoServe AI.",
            },
            parent_message_id=message["message_id"],
        )
        return {
            "engineer_task": engineer_task,
        }

    def _handle_engineer_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Review engineering output and decide whether to revise or move on."""
        if message["message_type"] == "failure":
            self._log_decision(
                stage="engineering_failure",
                verdict="failure",
                summary=message["payload"].get("error", "Engineer Agent reported a failure."),
            )
            return None

        engineering_result = message["payload"]
        review = self.review_engineering_output(engineering_result)
        verdict = review["verdict"]
        feedback = review["feedback"]
        self._log_decision(stage="engineering_review", verdict=verdict, summary=feedback)

        if verdict == "revise":
            return self._request_engineer_revision(message, engineering_result, feedback)

        print("[CEO Agent] Engineer output approved.")
        self.state["engineer_result"] = engineering_result
        self.state["qa_report"] = None
        self.pending_revision_targets.discard("engineer")
        if not self.state["marketing_result"]:
            print("[CEO Agent] Dispatching approved marketing task with PR context.")
            return self.message_bus.send_message(
                from_agent=self.agent_name,
                to_agent="marketing",
                message_type="task",
                payload={
                    "startup_name": self.startup_name,
                    "startup_idea": self.state["startup_idea"],
                    "product_spec": self.state["product_spec"],
                    "engineer_result": engineering_result,
                    "launch_goal": "Prepare launch messaging, outreach email, and Slack announcement for AutoServe AI.",
                },
                parent_message_id=message["message_id"],
            )
        return self._dispatch_qa_if_ready(message)

    def _handle_marketing_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Review marketing output and decide whether to revise or move on."""
        if message["message_type"] == "failure":
            self._log_decision(
                stage="marketing_failure",
                verdict="failure",
                summary=message["payload"].get("error", "Marketing Agent reported a failure."),
            )
            return None

        marketing_result = message["payload"]
        review = self.review_marketing_output(marketing_result)
        verdict = review["verdict"]
        feedback = review["feedback"]
        self._log_decision(stage="marketing_review", verdict=verdict, summary=feedback)

        if verdict == "revise":
            return self._request_marketing_revision(message, marketing_result, feedback)

        print("[CEO Agent] Marketing output approved.")
        self.state["marketing_result"] = marketing_result
        self.state["qa_report"] = None
        self.pending_revision_targets.discard("marketing")
        return self._dispatch_qa_if_ready(message)

    def _handle_qa_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Review the QA output and decide on revisions or final summary generation."""
        if message["message_type"] == "failure":
            self._log_decision(
                stage="qa_failure",
                verdict="failure",
                summary=message["payload"].get("error", "QA Agent reported a failure."),
            )
            return None

        qa_report = message["payload"]
        self.state["qa_report"] = qa_report
        review = self.review_qa_report(qa_report)
        verdict = review["verdict"]
        feedback = review["feedback"]
        self._log_decision(stage="qa_review", verdict=verdict, summary=feedback)

        if qa_report.get("verdict") == "fail" or verdict == "act":
            print("[CEO Agent] QA flagged issues. Dispatching revisions.")
            messages: Dict[str, Any] = {}
            targets = qa_report.get("target_agents", [])
            if "engineer" in targets:
                engineer_feedback = self._select_agent_feedback(qa_report, "engineer", feedback)
                self.revision_counts["engineer"] += 1
                if self.revision_counts["engineer"] > self.max_revisions:
                    raise RuntimeError("Engineer Agent exceeded the maximum number of revisions.")
                messages["engineer_revision"] = self._send_revision_to_engineer(
                    parent_message=message,
                    feedback=engineer_feedback,
                )
            if "marketing" in targets:
                marketing_feedback = self._select_agent_feedback(qa_report, "marketing", feedback)
                self.revision_counts["marketing"] += 1
                if self.revision_counts["marketing"] > self.max_revisions:
                    raise RuntimeError("Marketing Agent exceeded the maximum number of revisions.")
                messages["marketing_revision"] = self._send_revision_to_marketing(
                    parent_message=message,
                    feedback=marketing_feedback,
                )
            return messages

        print("[CEO Agent] QA approved the launch package. Building final summary...")
        summary = self.build_final_summary()
        return self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent=self.agent_name,
            message_type="confirmation",
            payload={"status": "workflow_complete", "final_summary": summary},
            parent_message_id=message["message_id"],
        )

    def _request_product_revision(
        self,
        message: Dict[str, Any],
        payload: Dict[str, Any],
        feedback: str,
    ) -> Dict[str, Any]:
        """Send a revision request back to the Product Agent."""
        self.revision_counts["product"] += 1
        if self.revision_counts["product"] > self.max_revisions:
            raise RuntimeError("Product Agent exceeded the maximum number of revisions.")

        print("[CEO Agent] Sending revision request to Product Agent.")
        return self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="product",
            message_type="revision_request",
            payload={
                "startup_idea": payload["startup_idea"],
                "focus_area": payload["focus_area"],
                "objectives": payload.get("objectives", []),
                "constraints": payload.get("constraints", []),
                "feedback": feedback,
                "previous_product_spec": payload["product_spec"],
            },
            parent_message_id=message["message_id"],
        )

    def _request_engineer_revision(
        self,
        message: Dict[str, Any],
        engineering_result: Dict[str, Any],
        feedback: str,
    ) -> Dict[str, Any]:
        """Send a revision request to the Engineer Agent."""
        self.revision_counts["engineer"] += 1
        if self.revision_counts["engineer"] > self.max_revisions:
            raise RuntimeError("Engineer Agent exceeded the maximum number of revisions.")

        print("[CEO Agent] Sending revision request to Engineer Agent.")
        return self._send_revision_to_engineer(
            parent_message=message,
            feedback=feedback,
            previous_result=engineering_result,
        )

    def _request_marketing_revision(
        self,
        message: Dict[str, Any],
        marketing_result: Dict[str, Any],
        feedback: str,
    ) -> Dict[str, Any]:
        """Send a revision request to the Marketing Agent."""
        self.revision_counts["marketing"] += 1
        if self.revision_counts["marketing"] > self.max_revisions:
            raise RuntimeError("Marketing Agent exceeded the maximum number of revisions.")

        print("[CEO Agent] Sending revision request to Marketing Agent.")
        return self._send_revision_to_marketing(
            parent_message=message,
            feedback=feedback,
            previous_result=marketing_result,
        )

    def _send_revision_to_engineer(
        self,
        parent_message: Dict[str, Any],
        feedback: str,
        previous_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal helper for Engineer Agent revision dispatch."""
        self.pending_revision_targets.add("engineer")
        return self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="engineer",
            message_type="revision_request",
            payload={
                "startup_name": self.startup_name,
                "startup_idea": self.state["startup_idea"],
                "product_spec": self.state["product_spec"],
                "feedback": feedback,
                "previous_result": previous_result or self.state["engineer_result"],
            },
            parent_message_id=parent_message["message_id"],
        )

    def _send_revision_to_marketing(
        self,
        parent_message: Dict[str, Any],
        feedback: str,
        previous_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal helper for Marketing Agent revision dispatch."""
        self.pending_revision_targets.add("marketing")
        return self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="marketing",
            message_type="revision_request",
            payload={
                "startup_name": self.startup_name,
                "startup_idea": self.state["startup_idea"],
                "product_spec": self.state["product_spec"],
                "feedback": feedback,
                "previous_result": previous_result or self.state["marketing_result"],
                "engineer_result": self.state["engineer_result"],
            },
            parent_message_id=parent_message["message_id"],
        )

    def _dispatch_qa_if_ready(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch the QA task once engineer and marketing outputs are both approved."""
        if not self.state["engineer_result"] or not self.state["marketing_result"]:
            return None
        if self.pending_revision_targets:
            return None

        qa_signature = self._build_qa_signature(
            self.state["engineer_result"],
            self.state["marketing_result"],
        )
        if qa_signature == self.qa_dispatch_signature:
            return None

        self.qa_dispatch_signature = qa_signature

        print("[CEO Agent] Engineering and marketing outputs approved. Dispatching QA task.")
        return self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="qa",
            message_type="task",
            payload={
                "startup_name": self.startup_name,
                "startup_idea": self.state["startup_idea"],
                "product_spec": self.state["product_spec"],
                "engineer_result": self.state["engineer_result"],
                "marketing_result": self.state["marketing_result"],
                "qa_signature": qa_signature,
            },
            parent_message_id=message["message_id"],
        )

    def _build_final_summary_prompt(self) -> str:
        """Build the final CEO summary prompt."""
        engineer_summary = self._summarize_engineering_for_final_summary(
            self.state["engineer_result"]
        )
        marketing_summary = self._summarize_marketing_for_final_summary(
            self.state["marketing_result"]
        )
        return f"""
Startup name: {self.startup_name}

Startup idea:
{self.state['startup_idea']}

Product specification:
{json.dumps(self.state['product_spec'], indent=2)}

Engineering result:
{json.dumps(engineer_summary, indent=2)}

Marketing result:
{json.dumps(marketing_summary, indent=2)}

QA report:
{json.dumps(self.state['qa_report'], indent=2)}

Decision log:
{json.dumps(self.decision_log, indent=2)}

Return only valid JSON:
{{
  "startup_name": "AutoServe AI",
  "launch_status": "ready",
  "summary": "string",
  "artifacts": {{
    "landing_page_path": "string",
    "github_issue_url": "string",
    "github_pr_url": "string",
    "slack_channel": "string",
    "email_recipient": "string"
  }},
  "key_takeaways": ["string"]
}}
""".strip()

    def _validate_incoming_message(self, message: Dict[str, Any]) -> None:
        """Validate messages handled by the CEO."""
        if message.get("to_agent") != self.agent_name:
            raise ValueError("CEOAgent can only process messages addressed to 'ceo'.")
        if message.get("from_agent") not in {"product", "engineer", "marketing", "qa"}:
            raise ValueError("CEOAgent received a message from an unsupported sender.")
        if message.get("message_type") not in {"confirmation", "result", "failure"}:
            raise ValueError(
                "CEOAgent only accepts confirmation, result, or failure messages."
            )
        if not isinstance(message.get("payload"), dict):
            raise ValueError("Incoming CEO message payload must be a dictionary.")

    def _normalize_review(
        self,
        review: Dict[str, Any],
        allowed_verdicts: set[str],
    ) -> Dict[str, str]:
        """Normalize and validate LLM review objects."""
        verdict = review.get("verdict")
        feedback = review.get("feedback")
        if verdict not in allowed_verdicts:
            raise LLMError(f"Unexpected review verdict. Expected one of {sorted(allowed_verdicts)}.")
        if not isinstance(feedback, str) or not feedback.strip():
            raise LLMError("Review feedback must be a non-empty string.")
        return {"verdict": verdict, "feedback": feedback.strip()}

    def _normalize_string_list(self, values: List[Any], field_name: str) -> List[str]:
        """Validate a list of non-empty strings."""
        normalized: List[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise LLMError(f"Field '{field_name}' must contain only non-empty strings.")
            normalized.append(value.strip())
        return normalized

    def _select_agent_feedback(
        self,
        qa_report: Dict[str, Any],
        agent_name: str,
        fallback_feedback: str,
    ) -> str:
        """Pick the most relevant QA feedback for a specific agent."""
        agent_feedback = qa_report.get("agent_feedback", {})
        if isinstance(agent_feedback, dict):
            feedback = agent_feedback.get(agent_name)
            if isinstance(feedback, str) and feedback.strip():
                return feedback.strip()
        return fallback_feedback

    @staticmethod
    def _truncate_text(value: Any, max_length: int) -> str:
        """Trim large text fields before sending them to review prompts."""
        if not isinstance(value, str):
            return ""
        cleaned = value.strip()
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[: max_length - 20].rstrip() + "\n...[truncated]"

    def _collect_product_spec_issues(self, product_spec: Dict[str, Any]) -> List[str]:
        """Check whether the product spec is concrete enough to unblock downstream agents."""
        issues: List[str] = []
        if not isinstance(product_spec, dict):
            return ["the product specification payload is not a JSON object"]

        value_proposition = product_spec.get("value_proposition")
        if not isinstance(value_proposition, str) or not value_proposition.strip():
            issues.append("add a concrete value proposition")
        else:
            value_prop_lower = value_proposition.lower()
            if "whatsapp" not in value_prop_lower:
                issues.append("mention WhatsApp explicitly in the value proposition")
            if "phone" not in value_prop_lower and "call" not in value_prop_lower:
                issues.append("mention phone or call automation explicitly in the value proposition")

        personas = product_spec.get("personas")
        if not isinstance(personas, list) or len(personas) < 3:
            issues.append("include at least three concrete personas")
        else:
            persona_text = " ".join(
                " ".join(
                    str(persona.get(field, ""))
                    for field in ("name", "role", "pain_point")
                )
                for persona in personas
                if isinstance(persona, dict)
            ).lower()
            if "clinic" not in persona_text:
                issues.append("include a clinic-specific persona")
            if "bakery" not in persona_text:
                issues.append("include a bakery-specific persona")
            if "grocery" not in persona_text:
                issues.append("include a grocery-specific persona")

        features = product_spec.get("features")
        if not isinstance(features, list) or len(features) < 5:
            issues.append("include at least five launch-ready features")
        else:
            feature_text = " ".join(
                " ".join(
                    str(feature.get(field, ""))
                    for field in ("name", "description")
                )
                for feature in features
                if isinstance(feature, dict)
            ).lower()
            required_feature_checks = {
                "WhatsApp inquiry handling": "whatsapp",
                "phone-call handling": ("phone", "call"),
                "booking capture": "booking",
                "order capture": "order",
                "staff handoff or dashboard workflow": ("handoff", "dashboard", "summary"),
            }
            for label, terms in required_feature_checks.items():
                if isinstance(terms, tuple):
                    if not any(term in feature_text for term in terms):
                        issues.append(f"cover {label} in the feature set")
                elif terms not in feature_text:
                    issues.append(f"cover {label} in the feature set")

        user_stories = product_spec.get("user_stories")
        if not isinstance(user_stories, list) or len(user_stories) != 3:
            issues.append("include exactly three user stories")
        else:
            story_text = " ".join(
                story for story in user_stories if isinstance(story, str)
            ).lower()
            if "clinic" not in story_text:
                issues.append("include a clinic user story")
            if "bakery" not in story_text:
                issues.append("include a bakery user story")
            if "grocery" not in story_text:
                issues.append("include a grocery user story")
            for story in user_stories:
                if not isinstance(story, str):
                    issues.append("keep every user story as a string")
                    continue
                story_lower = story.lower()
                if not (
                    story_lower.startswith("as a")
                    and " i want " in story_lower
                    and " so that " in story_lower
                ):
                    issues.append("keep every user story in 'As a / I want / so that' format")
                    break

        return issues

    def _collect_engineering_issues(self, engineering_result: Dict[str, Any]) -> List[str]:
        """Check whether the engineering output is concrete and internally consistent."""
        issues: List[str] = []
        if not isinstance(engineering_result, dict):
            return ["the engineering payload is not a JSON object"]

        required_text_fields = {
            "headline": "add a clear landing-page headline",
            "subheadline": "add a clear landing-page subheadline",
            "call_to_action": "add a clear primary CTA",
            "summary": "add a concise engineering summary",
            "landing_page_path": "include the landing page path",
            "branch": "include the GitHub branch name",
            "issue_url": "include the GitHub issue URL",
            "pr_url": "include the GitHub pull request URL",
            "commit_sha": "include the GitHub commit SHA",
        }
        for field_name, message in required_text_fields.items():
            value = engineering_result.get(field_name)
            if not isinstance(value, str) or not value.strip():
                issues.append(message)

        product_spec = self.state.get("product_spec") or {}
        feature_text = " ".join(
            feature.get("name", "")
            for feature in product_spec.get("features", [])
            if isinstance(feature, dict)
        ).lower()
        summary_text = " ".join(
            str(engineering_result.get(field, ""))
            for field in ("headline", "subheadline", "summary")
        ).lower()
        html_text = str(engineering_result.get("html", "")).lower()
        combined_text = " ".join([summary_text, html_text])

        if "whatsapp" not in combined_text:
            issues.append("mention WhatsApp automation in the landing page output")
        if "phone" not in combined_text and "call" not in combined_text:
            issues.append("mention phone or call automation in the landing page output")
        if not any(term in combined_text for term in ("book", "demo", "get started", "start")):
            issues.append("make the landing page CTA more explicit")

        if "booking" in feature_text and "booking" not in combined_text:
            issues.append("reflect booking capture in the landing page messaging")
        if "order" in feature_text and "order" not in combined_text:
            issues.append("reflect order capture in the landing page messaging")

        repo_slug = os.getenv("GITHUB_REPO", "").strip()
        if repo_slug:
            repo_url_root = f"https://github.com/{repo_slug}/"
            issue_url = engineering_result.get("issue_url", "")
            pr_url = engineering_result.get("pr_url", "")
            if isinstance(issue_url, str) and issue_url and not issue_url.startswith(repo_url_root):
                issues.append("make sure the GitHub issue URL points to the configured repository")
            if isinstance(pr_url, str) and pr_url and not pr_url.startswith(repo_url_root):
                issues.append("make sure the GitHub PR URL points to the configured repository")

        commit_sha = engineering_result.get("commit_sha", "")
        if isinstance(commit_sha, str) and commit_sha and len(commit_sha.strip()) < 7:
            issues.append("return a full commit SHA instead of a very short identifier")

        html_value = engineering_result.get("html", "")
        if not isinstance(html_value, str) or not html_value.strip():
            issues.append("include the generated HTML in the engineering payload")
        else:
            html_lower = html_value.lower()
            if "<!doctype html>" not in html_lower:
                issues.append("make sure the generated HTML starts with a proper doctype")
            if "<h1" not in html_lower:
                issues.append("include a visible hero headline in the landing page HTML")

        return issues

    def _collect_marketing_issues(self, marketing_result: Dict[str, Any]) -> List[str]:
        """Check whether the marketing output is concrete and ready for launch use."""
        issues: List[str] = []
        if not isinstance(marketing_result, dict):
            return ["the marketing payload is not a JSON object"]

        required_fields = {
            "tagline": "add a concise tagline",
            "short_description": "add a short launch description",
            "email_subject": "add an email subject line",
            "email_body_text": "add a plain-text email body",
            "email_body_html": "add an HTML email body",
            "slack_fallback_text": "add Slack fallback text",
        }
        for field_name, message in required_fields.items():
            value = marketing_result.get(field_name)
            if not isinstance(value, str) or not value.strip():
                issues.append(message)

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
        industry_checks = {
            "clinic": ("clinic", "clinics"),
            "bakery": ("bakery", "bakeries"),
            "grocery": ("grocery", "groceries", "grocery store", "grocery stores"),
        }
        for label, variants in industry_checks.items():
            if not any(variant in combined_text for variant in variants):
                issues.append(f"mention {label}-style businesses in the marketing copy")
                break

        for risky_phrase in ("guarantee", "instant", "24/7"):
            if risky_phrase in combined_text:
                issues.append("remove exaggerated marketing promises")
                break

        pr_url = marketing_result.get("pr_url", "")
        repo_slug = os.getenv("GITHUB_REPO", "").strip()
        if repo_slug and isinstance(pr_url, str) and pr_url.strip():
            expected_prefix = f"https://github.com/{repo_slug}/"
            if not pr_url.startswith(expected_prefix):
                issues.append("make sure the PR URL points to the configured repository")

        return issues

    def _summarize_engineering_for_final_summary(
        self,
        engineering_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Trim engineering output before using it in the final CEO summary prompt."""
        if not isinstance(engineering_result, dict):
            return {}

        return {
            "headline": engineering_result.get("headline"),
            "subheadline": engineering_result.get("subheadline"),
            "call_to_action": engineering_result.get("call_to_action"),
            "summary": engineering_result.get("summary"),
            "landing_page_path": engineering_result.get("landing_page_path"),
            "branch": engineering_result.get("branch"),
            "issue_url": engineering_result.get("issue_url"),
            "pr_url": engineering_result.get("pr_url"),
            "commit_sha": engineering_result.get("commit_sha"),
            "html_preview": self._truncate_text(engineering_result.get("html", ""), 1800),
        }

    def _summarize_marketing_for_final_summary(
        self,
        marketing_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Trim marketing output before using it in the final CEO summary prompt."""
        if not isinstance(marketing_result, dict):
            return {}

        return {
            "tagline": marketing_result.get("tagline"),
            "short_description": marketing_result.get("short_description"),
            "email_subject": marketing_result.get("email_subject"),
            "email_body_text_preview": self._truncate_text(
                marketing_result.get("email_body_text", ""),
                1200,
            ),
            "slack_fallback_text": marketing_result.get("slack_fallback_text"),
            "social_posts": marketing_result.get("social_posts"),
            "pr_url": marketing_result.get("pr_url"),
            "email_result": marketing_result.get("email_result"),
            "slack_result": marketing_result.get("slack_result"),
        }

    def _build_qa_signature(
        self,
        engineering_result: Dict[str, Any],
        marketing_result: Dict[str, Any],
    ) -> str:
        """Create a compact signature for the currently approved launch artifacts."""
        signature_payload = {
            "engineer": {
                "headline": engineering_result.get("headline"),
                "call_to_action": engineering_result.get("call_to_action"),
                "issue_url": engineering_result.get("issue_url"),
                "pr_url": engineering_result.get("pr_url"),
                "commit_sha": engineering_result.get("commit_sha"),
            },
            "marketing": {
                "tagline": marketing_result.get("tagline"),
                "email_subject": marketing_result.get("email_subject"),
                "pr_url": marketing_result.get("pr_url"),
                "email_to": (
                    marketing_result.get("email_result", {}).get("to_email")
                    if isinstance(marketing_result.get("email_result"), dict)
                    else None
                ),
                "slack_ts": (
                    marketing_result.get("slack_result", {}).get("ts")
                    if isinstance(marketing_result.get("slack_result"), dict)
                    else None
                ),
            },
        }
        return json.dumps(signature_payload, sort_keys=True)

    def _build_fallback_final_summary(self) -> Dict[str, Any]:
        """Create a deterministic summary when the final LLM summary is unavailable."""
        engineer_result = self.state["engineer_result"] or {}
        marketing_result = self.state["marketing_result"] or {}
        email_result = marketing_result.get("email_result", {})
        slack_result = marketing_result.get("slack_result", {})

        return {
            "startup_name": self.startup_name,
            "launch_status": "ready",
            "summary": (
                "AutoServe AI completed the product, engineering, marketing, and QA workflow. "
                "The landing page artifact is published to GitHub, launch messaging was delivered "
                "through email and Slack, and the final launch package is ready."
            ),
            "artifacts": {
                "landing_page_path": engineer_result.get("landing_page_path", "landing_page.html"),
                "github_issue_url": engineer_result.get("issue_url", ""),
                "github_pr_url": engineer_result.get("pr_url", ""),
                "slack_channel": slack_result.get("channel", ""),
                "email_recipient": email_result.get("to_email", ""),
            },
            "key_takeaways": [
                "The product specification stayed focused on WhatsApp and phone-call automation for small business teams.",
                "Engineering produced a publishable landing page and updated the linked GitHub branch and pull request.",
                "Marketing delivered launch-ready email and Slack messaging aligned with clinics, bakeries, and grocery stores.",
            ],
        }

    def _log_decision(self, stage: str, verdict: str, summary: str) -> None:
        """Append a CEO decision to the decision log."""
        self.decision_log.append(
            {
                "stage": stage,
                "verdict": verdict,
                "summary": summary,
            }
        )


__all__ = ["CEOAgent"]


def _read_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable with a fallback default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default
