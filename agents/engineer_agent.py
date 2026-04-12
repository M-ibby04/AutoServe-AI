"""Engineer agent for generating and publishing the AutoServe AI landing page."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

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
        """Generate landing-page content with the LLM, then render the final HTML."""
        system_prompt = (
            "You are the Engineer Agent for AutoServe AI. You are also a strong B2B "
            "landing-page designer with excellent conversion-focused writing skills. "
            "Return only valid JSON with no markdown fences."
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
        return self._validate_deliverable(
            deliverable=parsed,
            startup_name=startup_name,
            startup_idea=startup_idea,
            product_spec=product_spec,
        )

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
            previous_result_summary = self._summarize_previous_result(previous_result)
            revision_section = (
                "Revision feedback from CEO or QA:\n"
                f"{feedback}\n\n"
                "Previous engineering output:\n"
                f"{json.dumps(previous_result_summary, indent=2)}\n\n"
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

{revision_section}Create the content strategy and GitHub execution package for a polished,
professional AutoServe AI landing page.

Landing page direction:
- The page is for a real startup, not a class exercise mockup.
- The tone should feel commercially credible, operationally grounded, and conversion-focused.
- Mention WhatsApp and phone-call automation clearly.
- Cover inquiries, bookings, and orders clearly.
- Speak directly to clinics, bakeries, grocery stores, and similar service businesses.
- Avoid generic AI buzzwords, fake testimonials, fake company logos, fake metrics, or exaggerated promises.
- The resulting page will be rendered into a premium, responsive SaaS layout by the application, so your job is to provide strong copy and structure.

Return only valid JSON with exactly this structure:
{{
  "headline": "string",
  "subheadline": "string",
  "call_to_action": "string",
  "summary": "string",
  "hero_supporting_points": ["string", "string", "string"],
  "problem_statement": "string",
  "solution_statement": "string",
  "industry_cards": [
    {{
      "industry": "string",
      "challenge": "string",
      "outcome": "string"
    }}
  ],
  "proof_points": ["string", "string", "string"],
  "workflow_steps": [
    {{
      "title": "string",
      "description": "string"
    }}
  ],
  "faq": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],
  "final_cta_title": "string",
  "final_cta_text": "string",
  "issue_title": "Initial landing page",
  "issue_body": "string",
  "branch_name": "string",
  "commit_message": "string",
  "pr_title": "string",
  "pr_body": "string"
}}

Rules:
- headline must feel premium and specific, not generic.
- subheadline must explain the business outcome in plain language.
- call_to_action should be short, direct, and demo-ready.
- Do not use unsupported percentages, ROI claims, revenue promises, time-saved claims, multipliers, or numeric performance improvements anywhere in the landing-page copy.
- Do not write phrases like "up to 70%", "90% faster", "3x", "double conversions", "save 10 hours", or similar quantified promises unless that exact evidence exists in the provided product specification, which it does not here.
- Use qualitative but credible value language instead, such as "reduce manual workload", "improve response speed", or "capture more bookings with less manual follow-up".
- Include exactly 3 hero_supporting_points.
- Include exactly 3 industry_cards for clinics, bakeries, and grocery stores.
- Include exactly 3 proof_points grounded in realistic operational value.
- Include exactly 3 workflow_steps.
- Include exactly 3 faq entries.
- final_cta_title and final_cta_text should push toward demo booking or launch interest.
- branch_name should be short, lowercase, and git-safe.
- issue_title can stay 'Initial landing page' unless a revision title is clearly better.
- Do not include markdown fences or commentary.
""".strip()

    def _validate_deliverable(
        self,
        deliverable: Dict[str, Any],
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate generated landing-page content and render professional HTML."""
        required_string_fields = [
            "headline",
            "subheadline",
            "call_to_action",
            "summary",
            "problem_statement",
            "solution_statement",
            "final_cta_title",
            "final_cta_text",
            "issue_title",
            "issue_body",
            "branch_name",
            "commit_message",
            "pr_title",
            "pr_body",
        ]
        for field in required_string_fields:
            value = deliverable.get(field)
            if not isinstance(value, str) or not value.strip():
                raise LLMError(f"Engineer deliverable field '{field}' must be a non-empty string.")

        for field in [
            "headline",
            "subheadline",
            "summary",
            "problem_statement",
            "solution_statement",
            "final_cta_title",
            "final_cta_text",
        ]:
            self._ensure_no_unsupported_claims(field, deliverable[field].strip())

        hero_supporting_points = self._normalize_string_list(
            deliverable.get("hero_supporting_points"),
            field_name="hero_supporting_points",
            expected_length=3,
        )
        proof_points = self._normalize_string_list(
            deliverable.get("proof_points"),
            field_name="proof_points",
            expected_length=3,
        )
        industry_cards = self._normalize_named_cards(
            deliverable.get("industry_cards"),
            field_name="industry_cards",
            required_keys=("industry", "challenge", "outcome"),
            expected_length=3,
        )
        workflow_steps = self._normalize_named_cards(
            deliverable.get("workflow_steps"),
            field_name="workflow_steps",
            required_keys=("title", "description"),
            expected_length=3,
        )
        faq_entries = self._normalize_named_cards(
            deliverable.get("faq"),
            field_name="faq",
            required_keys=("question", "answer"),
            expected_length=3,
        )

        for index, point in enumerate(hero_supporting_points, start=1):
            self._ensure_no_unsupported_claims(
                f"hero_supporting_points[{index}]",
                point,
            )
        for index, point in enumerate(proof_points, start=1):
            self._ensure_no_unsupported_claims(
                f"proof_points[{index}]",
                point,
            )
        for index, card in enumerate(industry_cards, start=1):
            self._ensure_no_unsupported_claims(
                f"industry_cards[{index}].challenge",
                card["challenge"],
            )
            self._ensure_no_unsupported_claims(
                f"industry_cards[{index}].outcome",
                card["outcome"],
            )

        html = self._render_landing_page(
            startup_name=startup_name,
            startup_idea=startup_idea,
            product_spec=product_spec,
            content={
                "headline": deliverable["headline"].strip(),
                "subheadline": deliverable["subheadline"].strip(),
                "call_to_action": deliverable["call_to_action"].strip(),
                "summary": deliverable["summary"].strip(),
                "hero_supporting_points": hero_supporting_points,
                "problem_statement": deliverable["problem_statement"].strip(),
                "solution_statement": deliverable["solution_statement"].strip(),
                "industry_cards": industry_cards,
                "proof_points": proof_points,
                "workflow_steps": workflow_steps,
                "faq": faq_entries,
                "final_cta_title": deliverable["final_cta_title"].strip(),
                "final_cta_text": deliverable["final_cta_text"].strip(),
            },
        )

        html_lower = html.lower()
        if not html_lower.startswith("<!doctype html>"):
            raise LLMError("Landing page HTML must start with <!DOCTYPE html>.")
        if "<html" not in html_lower:
            raise LLMError("Landing page HTML must include an <html> element.")
        if not any(marker in html_lower for marker in ["<h1", "<header", "hero"]):
            raise LLMError("Landing page HTML must include a hero/headline structure.")
        if not any(
            marker in html_lower
            for marker in ["whatsapp", "message automation", "messaging automation"]
        ):
            raise LLMError("Landing page HTML must mention WhatsApp or messaging automation.")
        if not any(
            marker in html_lower
            for marker in ["phone", "call automation", "voice automation", "phone-call automation", "missed calls"]
        ):
            raise LLMError("Landing page HTML must mention phone or call automation.")
        if not any(
            marker in html_lower
            for marker in ["<button", "call-to-action", "get started", "book a demo", "start automating"]
        ):
            raise LLMError("Landing page HTML must include a clear CTA/button.")

        deliverable["branch_name"] = self._sanitize_branch_name(deliverable["branch_name"])
        deliverable["landing_page_path"] = "landing_page.html"
        deliverable["html"] = html
        return deliverable

    @staticmethod
    def _summarize_previous_result(previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Trim bulky revision context before sending it back to the LLM."""
        if not isinstance(previous_result, dict):
            return {}

        html_value = previous_result.get("html", "")
        html_preview = ""
        if isinstance(html_value, str) and html_value.strip():
            cleaned_html = html_value.strip()
            html_preview = (
                cleaned_html
                if len(cleaned_html) <= 2400
                else cleaned_html[:2380].rstrip() + "\n...[truncated]"
            )

        return {
            "headline": previous_result.get("headline"),
            "subheadline": previous_result.get("subheadline"),
            "call_to_action": previous_result.get("call_to_action"),
            "summary": previous_result.get("summary"),
            "branch": previous_result.get("branch"),
            "issue_url": previous_result.get("issue_url"),
            "issue_number": previous_result.get("issue_number"),
            "pr_url": previous_result.get("pr_url"),
            "pr_number": previous_result.get("pr_number"),
            "commit_sha": previous_result.get("commit_sha"),
            "html_preview": html_preview,
        }

    @staticmethod
    def _normalize_string_list(
        values: Any,
        field_name: str,
        expected_length: int,
    ) -> List[str]:
        """Validate a fixed-length list of non-empty strings."""
        if not isinstance(values, list) or len(values) != expected_length:
            raise LLMError(
                f"Engineer field '{field_name}' must be a list with exactly {expected_length} items."
            )

        normalized: List[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise LLMError(f"Engineer field '{field_name}' must contain only non-empty strings.")
            normalized.append(value.strip())
        return normalized

    @staticmethod
    def _normalize_named_cards(
        values: Any,
        field_name: str,
        required_keys: tuple[str, ...],
        expected_length: int,
    ) -> List[Dict[str, str]]:
        """Validate fixed-length lists of named content objects."""
        if not isinstance(values, list) or len(values) != expected_length:
            raise LLMError(
                f"Engineer field '{field_name}' must be a list with exactly {expected_length} items."
            )

        normalized: List[Dict[str, str]] = []
        for value in values:
            if not isinstance(value, dict):
                raise LLMError(f"Engineer field '{field_name}' must contain only objects.")

            normalized_item: Dict[str, str] = {}
            for key in required_keys:
                field_value = value.get(key)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise LLMError(
                        f"Engineer field '{field_name}' requires a non-empty '{key}' string."
                    )
                normalized_item[key] = field_value.strip()
            normalized.append(normalized_item)
        return normalized

    @staticmethod
    def _ensure_no_unsupported_claims(field_name: str, text: str) -> None:
        """Reject risky quantified marketing claims that are not backed by evidence."""
        normalized = text.lower()
        risky_patterns = [
            r"\bup to\s+\d",
            r"\d+\s*%",
            r"\b\d+(\.\d+)?x\b",
            r"\bdouble\b",
            r"\btriple\b",
            r"\b\d+\s*(hours|hour|days|day|minutes|minute)\b",
            r"\b(save|saves|saved|cut|cuts|reduce|reduces|reduced|boost|boosts|increase|increases|improve|improves|improved)\b[^.]{0,25}\b\d",
            r"\b(roi|return on investment)\b",
            r"\bguarantee(d)?\b",
            r"\binstant(ly)?\b",
            r"\b24/7\b",
        ]
        for pattern in risky_patterns:
            if re.search(pattern, normalized):
                raise LLMError(
                    f"Engineer field '{field_name}' includes an unsupported quantified or exaggerated claim: {text}"
                )

    def _render_landing_page(
        self,
        startup_name: str,
        startup_idea: str,
        product_spec: Dict[str, Any],
        content: Dict[str, Any],
    ) -> str:
        """Render a polished, responsive landing page from structured content."""
        hero_points_html = "".join(
            f"<li>{escape(point)}</li>" for point in content["hero_supporting_points"]
        )
        proof_points_html = "".join(
            f"<article class=\"proof-card\"><p>{escape(point)}</p></article>"
            for point in content["proof_points"]
        )
        industry_cards_html = "".join(
            (
                "<article class=\"industry-card\">"
                f"<span class=\"pill\">{escape(card['industry'])}</span>"
                f"<h3>{escape(card['challenge'])}</h3>"
                f"<p>{escape(card['outcome'])}</p>"
                "</article>"
            )
            for card in content["industry_cards"]
        )
        persona_cards_html = "".join(
            (
                "<article class=\"persona-card\">"
                f"<h3>{escape(persona['name'])}</h3>"
                f"<strong>{escape(persona['role'])}</strong>"
                f"<p>{escape(persona['pain_point'])}</p>"
                "</article>"
            )
            for persona in product_spec.get("personas", [])
            if isinstance(persona, dict)
        )
        feature_cards_html = "".join(
            (
                "<article class=\"feature-card\">"
                f"<span class=\"priority\">Priority {escape(str(feature['priority']))}</span>"
                f"<h3>{escape(feature['name'])}</h3>"
                f"<p>{escape(feature['description'])}</p>"
                "</article>"
            )
            for feature in product_spec.get("features", [])
            if isinstance(feature, dict)
        )
        workflow_steps_html = "".join(
            (
                "<article class=\"workflow-step\">"
                f"<span class=\"step-number\">0{index}</span>"
                f"<h3>{escape(step['title'])}</h3>"
                f"<p>{escape(step['description'])}</p>"
                "</article>"
            )
            for index, step in enumerate(content["workflow_steps"], start=1)
        )
        faq_html = "".join(
            (
                "<details class=\"faq-item\">"
                f"<summary>{escape(item['question'])}</summary>"
                f"<p>{escape(item['answer'])}</p>"
                "</details>"
            )
            for item in content["faq"]
        )

        feature_count = len(product_spec.get("features", []))
        template = Template(
            """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>$title</title>
  <meta name="description" content="$summary" />
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
      --bg: #f5efe5;
      --paper: rgba(255, 252, 246, 0.88);
      --card: rgba(255, 255, 255, 0.92);
      --line: rgba(21, 37, 44, 0.1);
      --ink: #15252c;
      --muted: #5f6f79;
      --primary: #176b73;
      --primary-deep: #0f4f55;
      --accent: #eb8d41;
      --accent-soft: rgba(235, 141, 65, 0.14);
      --shadow: 0 24px 50px rgba(21, 37, 44, 0.12);
      --radius-xl: 30px;
      --radius-lg: 22px;
      --radius-md: 16px;
      --max-width: 1180px;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Manrope", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(23, 107, 115, 0.18), transparent 28%),
        radial-gradient(circle at left center, rgba(235, 141, 65, 0.14), transparent 25%),
        linear-gradient(180deg, #f8f3ea 0%, #f5efe5 50%, #efe7dc 100%);
      line-height: 1.6;
    }

    a { color: inherit; text-decoration: none; }
    .container { width: min(calc(100% - 32px), var(--max-width)); margin: 0 auto; }
    .topbar { position: sticky; top: 0; z-index: 20; backdrop-filter: blur(14px); background: rgba(245, 239, 229, 0.78); border-bottom: 1px solid rgba(21, 37, 44, 0.06); }
    .topbar-inner { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 18px 0; }
    .brand { display: flex; align-items: center; gap: 12px; font-family: "Space Grotesk", sans-serif; font-weight: 700; }
    .brand-mark { width: 42px; height: 42px; border-radius: 14px; display: grid; place-items: center; color: #fff; background: linear-gradient(135deg, var(--primary), #2a919b); box-shadow: 0 14px 24px rgba(23, 107, 115, 0.24); }
    .nav { display: flex; align-items: center; gap: 18px; color: var(--muted); font-size: 0.95rem; }
    .hero { display: grid; grid-template-columns: 1.2fr 0.85fr; gap: 24px; padding: 64px 0 22px; }
    .panel, .hero-side-card, .cta-panel { background: var(--paper); border: 1px solid var(--line); box-shadow: var(--shadow); border-radius: var(--radius-xl); }
    .panel { padding: 40px; }
    .eyebrow, .pill, .priority, .step-number { display: inline-flex; align-items: center; padding: 8px 13px; border-radius: 999px; font-size: 0.84rem; font-weight: 700; }
    .eyebrow, .pill { background: rgba(23, 107, 115, 0.09); color: var(--primary-deep); }
    .priority { background: var(--accent-soft); color: #8f4d1e; }
    .step-number { background: rgba(21, 37, 44, 0.06); color: var(--ink); }
    h1, h2, h3 { margin: 0; font-family: "Space Grotesk", sans-serif; letter-spacing: -0.03em; line-height: 1.08; }
    h1 { margin-top: 20px; font-size: clamp(2.8rem, 5vw, 4.9rem); max-width: 12ch; }
    h2 { font-size: clamp(1.9rem, 3vw, 2.8rem); }
    h3 { font-size: 1.25rem; }
    .hero-copy, .section-copy, .muted, .feature-card p, .industry-card p, .persona-card p, .workflow-step p, .faq-item p, .proof-card p { color: var(--muted); }
    .hero-copy { margin-top: 16px; max-width: 60ch; font-size: 1.08rem; }
    .actions { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 28px; }
    .btn { display: inline-flex; align-items: center; justify-content: center; min-height: 50px; padding: 0 22px; border-radius: 999px; font-weight: 700; transition: transform 160ms ease, box-shadow 160ms ease; }
    .btn:hover { transform: translateY(-1px); }
    .btn-primary { color: #fff; background: linear-gradient(135deg, var(--primary), #25919b); box-shadow: 0 16px 28px rgba(23, 107, 115, 0.24); }
    .btn-secondary { background: rgba(255,255,255,0.7); border: 1px solid var(--line); }
    .hero-points { list-style: none; padding: 0; margin: 28px 0 0; display: grid; gap: 12px; }
    .hero-points li { display: flex; gap: 12px; color: var(--muted); }
    .hero-points li::before { content: ""; width: 10px; height: 10px; flex: 0 0 auto; margin-top: 8px; border-radius: 50%; background: linear-gradient(135deg, var(--accent), #ffb978); box-shadow: 0 0 0 6px rgba(235, 141, 65, 0.15); }
    .hero-side { display: grid; gap: 18px; }
    .hero-side-card { padding: 28px; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }
    .stat { padding: 18px; border-radius: var(--radius-md); background: var(--card); border: 1px solid var(--line); }
    .stat strong { display: block; font-family: "Space Grotesk", sans-serif; font-size: 1.55rem; }
    section { padding: 18px 0; }
    .section-head { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 22px; }
    .section-head p { margin: 0; max-width: 58ch; color: var(--muted); }
    .proof-grid, .industry-grid, .persona-grid, .feature-grid, .workflow-grid { display: grid; gap: 18px; }
    .proof-grid { grid-template-columns: repeat(3, 1fr); }
    .industry-grid, .persona-grid, .feature-grid, .workflow-grid { grid-template-columns: repeat(3, 1fr); }
    .proof-card, .industry-card, .persona-card, .feature-card, .workflow-step, .split-card { padding: 26px; background: var(--card); border: 1px solid var(--line); border-radius: var(--radius-lg); box-shadow: 0 18px 32px rgba(21, 37, 44, 0.08); }
    .split { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }
    .split-card h2 { margin-bottom: 12px; }
    .persona-card strong { display: inline-block; margin: 10px 0 8px; color: var(--primary-deep); }
    .faq-grid { display: grid; gap: 14px; }
    .faq-item { padding: 18px 20px; border-radius: var(--radius-md); background: var(--card); border: 1px solid var(--line); }
    .faq-item summary { cursor: pointer; font-family: "Space Grotesk", sans-serif; font-weight: 700; }
    .faq-item summary::-webkit-details-marker { display: none; }
    .cta-panel { margin: 20px 0 72px; padding: 34px; display: grid; grid-template-columns: 1.1fr auto; gap: 22px; align-items: center; background: linear-gradient(135deg, rgba(23,107,115,0.98), rgba(15,79,85,0.95)); color: #fff; border-color: rgba(255,255,255,0.08); }
    .cta-panel p { color: rgba(255,255,255,0.82); margin: 10px 0 0; max-width: 56ch; }
    .cta-panel .btn-primary { background: #fff; color: var(--primary-deep); box-shadow: none; }
    footer { padding: 0 0 34px; color: var(--muted); font-size: 0.94rem; }
    @media (max-width: 1000px) {
      .hero, .split, .cta-panel { grid-template-columns: 1fr; }
      .proof-grid, .industry-grid, .persona-grid, .feature-grid, .workflow-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 720px) {
      .topbar-inner, .nav, .section-head { flex-direction: column; align-items: flex-start; }
      .panel, .hero-side-card, .proof-card, .industry-card, .persona-card, .feature-card, .workflow-step, .split-card, .cta-panel { padding: 22px; }
      .proof-grid, .industry-grid, .persona-grid, .feature-grid, .workflow-grid, .stats { grid-template-columns: 1fr; }
      .hero { padding-top: 42px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <div class="brand"><div class="brand-mark">AI</div><div>$startup_name</div></div>
      <nav class="nav">
        <a href="#industries">Industries</a>
        <a href="#features">Features</a>
        <a href="#workflow">How It Works</a>
        <a class="btn btn-secondary" href="#cta">Book a Demo</a>
      </nav>
    </div>
  </header>

  <main class="container">
    <section class="hero" id="top">
      <div class="panel">
        <span class="eyebrow">AI operations for busy service teams</span>
        <h1>$headline</h1>
        <p class="hero-copy">$subheadline</p>
        <div class="actions">
          <a class="btn btn-primary" href="#cta">$call_to_action</a>
          <a class="btn btn-secondary" href="#features">See the workflow</a>
        </div>
        <ul class="hero-points">$hero_points_html</ul>
      </div>
      <div class="hero-side">
        <article class="hero-side-card">
          <span class="pill">Launch summary</span>
          <h3 style="margin-top: 14px;">Built to reduce missed demand across voice and messaging.</h3>
          <p class="muted">$summary</p>
          <div class="stats">
            <div class="stat"><strong>2</strong><span>customer channels unified</span></div>
            <div class="stat"><strong>3</strong><span>core service workflows covered</span></div>
            <div class="stat"><strong>$feature_count</strong><span>launch-ready MVP features</span></div>
          </div>
        </article>
        <article class="hero-side-card">
          <span class="pill">Business context</span>
          <p class="muted" style="margin-top: 14px;">$startup_idea</p>
        </article>
      </div>
    </section>

    <section>
      <div class="proof-grid">$proof_points_html</div>
    </section>

    <section class="split">
      <article class="split-card">
        <span class="pill">The problem</span>
        <h2>Customer demand does not pause when your staff gets busy.</h2>
        <p>$problem_statement</p>
      </article>
      <article class="split-card">
        <span class="pill">The solution</span>
        <h2>One assistant layer across WhatsApp and phone-call workflows.</h2>
        <p>$solution_statement</p>
      </article>
    </section>

    <section id="industries">
      <div class="section-head">
        <div><span class="pill">Target industries</span><h2 style="margin-top: 14px;">Operational support for businesses that lose revenue when replies slip.</h2></div>
        <p>AutoServe AI is positioned for small service businesses that need faster replies, cleaner handoffs, and fewer missed opportunities across inbound demand.</p>
      </div>
      <div class="industry-grid">$industry_cards_html</div>
    </section>

    <section>
      <div class="section-head">
        <div><span class="pill">Who it serves</span><h2 style="margin-top: 14px;">Grounded in real frontline workflows, not abstract AI use cases.</h2></div>
        <p>The product spec centers on the people who actually manage bookings, orders, call overflow, and repeated customer questions every day.</p>
      </div>
      <div class="persona-grid">$persona_cards_html</div>
    </section>

    <section id="features">
      <div class="section-head">
        <div><span class="pill">Core features</span><h2 style="margin-top: 14px;">The minimum viable system a small business would seriously consider paying for.</h2></div>
        <p>These features are prioritized for practical launch value, fast onboarding, and clear operational payoff across messaging and voice.</p>
      </div>
      <div class="feature-grid">$feature_cards_html</div>
    </section>

    <section id="workflow">
      <div class="section-head">
        <div><span class="pill">How it works</span><h2 style="margin-top: 14px;">From first inquiry to clear staff handoff in three steps.</h2></div>
        <p>AutoServe AI keeps the customer experience responsive while still preserving team oversight for edge cases and service quality.</p>
      </div>
      <div class="workflow-grid">$workflow_steps_html</div>
    </section>

    <section>
      <div class="section-head">
        <div><span class="pill">FAQ</span><h2 style="margin-top: 14px;">Questions a serious buyer will ask before scheduling a demo.</h2></div>
        <p>These answers keep the landing page commercially grounded and ready for early customer conversations.</p>
      </div>
      <div class="faq-grid">$faq_html</div>
    </section>

    <section id="cta">
      <div class="cta-panel">
        <div>
          <span class="pill" style="background: rgba(255,255,255,0.14); color: #fff;">Ready for launch</span>
          <h2 style="margin-top: 14px;">$final_cta_title</h2>
          <p>$final_cta_text</p>
        </div>
        <div><a class="btn btn-primary" href="#top">$call_to_action</a></div>
      </div>
    </section>
  </main>

  <footer class="container">
    <p>$startup_name helps small service businesses manage inquiries, bookings, and orders with AI across WhatsApp and phone calls.</p>
  </footer>
</body>
</html>
"""
        )

        return template.substitute(
            title=escape(f"{startup_name} | AI WhatsApp and Phone Automation"),
            startup_name=escape(startup_name),
            headline=escape(content["headline"]),
            subheadline=escape(content["subheadline"]),
            call_to_action=escape(content["call_to_action"]),
            summary=escape(content["summary"]),
            startup_idea=escape(startup_idea),
            hero_points_html=hero_points_html,
            proof_points_html=proof_points_html,
            industry_cards_html=industry_cards_html,
            persona_cards_html=persona_cards_html,
            feature_cards_html=feature_cards_html,
            workflow_steps_html=workflow_steps_html,
            faq_html=faq_html,
            problem_statement=escape(content["problem_statement"]),
            solution_statement=escape(content["solution_statement"]),
            final_cta_title=escape(content["final_cta_title"]),
            final_cta_text=escape(content["final_cta_text"]),
            feature_count=str(feature_count),
        )

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
            existing_pull_request = github.find_open_pull_request(head_branch=branch_name)
            if existing_pull_request:
                print("[Engineer Agent] Reusing existing open GitHub pull request...")
                pr_number = existing_pull_request["number"]
                pr_url = existing_pull_request["html_url"]
            else:
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
