"""CEO agent for orchestrating the LaunchMind workflow."""

from __future__ import annotations

import json
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
        self.max_revisions = 2

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
        summary = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
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
        """Review the Product Agent output and decide pass or revise."""
        system_prompt = (
            "You are the CEO Agent reviewing an AutoServe AI product specification. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Startup name: {self.startup_name}

Review this product specification for AutoServe AI:
{json.dumps(product_spec, indent=2)}

AutoServe AI context:
- AI-powered WhatsApp and phone automation for small businesses
- Use cases include clinics, bakeries, grocery stores, and similar service teams
- The MVP should help with inquiries, bookings, and orders

Return only valid JSON:
{{
  "verdict": "pass" or "revise",
  "feedback": "string"
}}

Choose "revise" if the output is vague, generic, weakly prioritized, or not specific enough to AutoServe AI.
""".strip()
        review = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        return self._normalize_review(review, {"pass", "revise"})

    def review_engineering_output(self, engineering_result: Dict[str, Any]) -> Dict[str, str]:
        """Review the Engineer Agent output."""
        system_prompt = (
            "You are the CEO Agent reviewing AutoServe AI engineering output. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Startup name: {self.startup_name}

Approved product specification:
{json.dumps(self.state['product_spec'], indent=2)}

Engineer output:
{json.dumps(engineering_result, indent=2)}

Return only valid JSON:
{{
  "verdict": "pass" or "revise",
  "feedback": "string"
}}

Review for:
- strong AutoServe AI positioning
- clear landing page structure and CTA
- believable GitHub execution details
- consistency with the product spec
""".strip()
        review = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        return self._normalize_review(review, {"pass", "revise"})

    def review_marketing_output(self, marketing_result: Dict[str, Any]) -> Dict[str, str]:
        """Review the Marketing Agent output."""
        system_prompt = (
            "You are the CEO Agent reviewing AutoServe AI marketing output. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Startup name: {self.startup_name}

Approved product specification:
{json.dumps(self.state['product_spec'], indent=2)}

Marketing output:
{json.dumps(marketing_result, indent=2)}

Return only valid JSON:
{{
  "verdict": "pass" or "revise",
  "feedback": "string"
}}

Review for:
- specificity to WhatsApp and phone automation for small businesses
- useful, believable positioning for clinics, bakeries, and grocery stores
- strong but not overhyped messaging
- email and Slack content that feels launch-ready
""".strip()
        review = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        return self._normalize_review(review, {"pass", "revise"})

    def review_qa_report(self, qa_report: Dict[str, Any]) -> Dict[str, str]:
        """Review the QA verdict and decide the next orchestration step."""
        system_prompt = (
            "You are the CEO Agent reviewing a QA report for AutoServe AI. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Startup name: {self.startup_name}

QA report:
{json.dumps(qa_report, indent=2)}

Return only valid JSON:
{{
  "verdict": "accept" or "act",
  "feedback": "string"
}}

Choose "act" if QA found issues that require revisions from Engineer or Marketing.
Choose "accept" only if the workflow can be finalized.
""".strip()
        review = sanitize_llm_json(call_llm(system_prompt=system_prompt, user_prompt=user_prompt))
        return self._normalize_review(review, {"accept", "act"})

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
            },
            parent_message_id=message["message_id"],
        )

    def _build_final_summary_prompt(self) -> str:
        """Build the final CEO summary prompt."""
        return f"""
Startup name: {self.startup_name}

Startup idea:
{self.state['startup_idea']}

Product specification:
{json.dumps(self.state['product_spec'], indent=2)}

Engineering result:
{json.dumps(self.state['engineer_result'], indent=2)}

Marketing result:
{json.dumps(self.state['marketing_result'], indent=2)}

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
