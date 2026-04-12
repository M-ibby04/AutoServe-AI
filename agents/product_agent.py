"""Product agent for generating AutoServe AI product specifications."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from message_bus import MessageBus
from utils.llm import LLMError, call_llm, sanitize_llm_json


class ProductAgent:
    """Generate and revise the product specification for AutoServe AI."""

    agent_name = "product"

    def __init__(self, message_bus: MessageBus) -> None:
        """Store the shared message bus used for agent communication."""
        self.message_bus = message_bus

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Process a CEO task or revision request for product strategy."""
        self._validate_incoming_message(message)

        payload = message["payload"]
        startup_idea = payload["startup_idea"].strip()
        focus_area = payload.get("focus_area", "Define a practical MVP clearly.")
        objectives = payload.get("objectives", [])
        constraints = payload.get("constraints", [])
        feedback = payload.get("feedback")
        previous_product_spec = payload.get("previous_product_spec")

        if message["message_type"] == "revision_request":
            print("[Product Agent] Revision request received from CEO.")
            print(f"[Product Agent] Feedback: {feedback}")
        else:
            print("[Product Agent] New product task received from CEO.")
        print(f"[Product Agent] Focus area: {focus_area}")
        print("[Product Agent] Generating structured product specification for AutoServe AI...")

        product_spec = self.generate_product_spec(
            startup_idea=startup_idea,
            focus_area=focus_area,
            objectives=objectives,
            constraints=constraints,
            feedback=feedback,
            previous_product_spec=previous_product_spec,
        )

        preview_payload = {
            "startup_idea": startup_idea,
            "focus_area": focus_area,
            "objectives": objectives,
            "constraints": constraints,
            "status": "awaiting_ceo_approval",
            "product_spec": product_spec,
        }

        print("[Product Agent] Sharing product spec preview with Engineer Agent.")
        engineer_message = self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="engineer",
            message_type="result",
            payload=preview_payload,
            parent_message_id=message["message_id"],
        )

        print("[Product Agent] Sharing product spec preview with Marketing Agent.")
        marketing_message = self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="marketing",
            message_type="result",
            payload=preview_payload,
            parent_message_id=message["message_id"],
        )

        print("[Product Agent] Sending structured product output to CEO for review.")
        confirmation_message = self.message_bus.send_message(
            from_agent=self.agent_name,
            to_agent="ceo",
            message_type="confirmation",
            payload={
                "startup_idea": startup_idea,
                "focus_area": focus_area,
                "objectives": objectives,
                "constraints": constraints,
                "feedback": feedback,
                "status": (
                    "product_spec_revised"
                    if message["message_type"] == "revision_request"
                    else "product_spec_created"
                ),
                "product_spec": product_spec,
            },
            parent_message_id=message["message_id"],
        )

        return {
            "engineer_message": engineer_message,
            "marketing_message": marketing_message,
            "confirmation_message": confirmation_message,
        }

    def generate_product_spec(
        self,
        startup_idea: str,
        focus_area: str,
        objectives: List[str],
        constraints: List[str],
        feedback: Optional[str] = None,
        previous_product_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate and normalize a structured AutoServe AI product spec."""
        system_prompt = (
            "You are the Product Agent for AutoServe AI, an AI-powered WhatsApp and "
            "phone automation platform for small businesses such as clinics, bakeries, "
            "and grocery stores. Return only valid JSON with no markdown fences."
        )
        user_prompt = self.build_prompt(
            startup_idea=startup_idea,
            focus_area=focus_area,
            objectives=objectives,
            constraints=constraints,
            feedback=feedback,
            previous_product_spec=previous_product_spec,
        )

        raw_response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        return self._parse_product_spec(raw_response)

    def build_prompt(
        self,
        startup_idea: str,
        focus_area: str,
        objectives: List[str],
        constraints: List[str],
        feedback: Optional[str] = None,
        previous_product_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the product-generation prompt for the LLM."""
        revision_section = ""
        if feedback:
            revision_section = (
                "Revision feedback from CEO:\n"
                f"{feedback}\n\n"
                "Previous product specification:\n"
                f"{json.dumps(previous_product_spec or {}, indent=2)}\n\n"
                "Use the previous product spec as a baseline. Improve weak areas, keep "
                "good decisions, and make the result more specific and execution-ready.\n\n"
            )

        objectives_text = "\n".join(f"- {objective}" for objective in objectives) or "- None provided"
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) or "- None provided"

        return f"""
Startup name: AutoServe AI

Startup idea:
{startup_idea}

Focus area:
{focus_area}

Objectives:
{objectives_text}

Constraints:
{constraints_text}

{revision_section}Create a product specification tailored to AutoServe AI.

AutoServe AI specifics to preserve:
- Target users are small business owners and managers in clinics, bakeries, grocery stores, and similar service businesses.
- Primary channels are WhatsApp and phone calls.
- Core value is automating repetitive customer interactions such as inquiries, bookings, and orders.
- Main outcomes are faster response time, fewer missed inquiries, easier bookings/orders, and reduced manual workload.
- The product should feel like a realistic MVP for a student demo, not an overbuilt enterprise platform.

Return only valid JSON with exactly this structure:
{{
  "value_proposition": "string",
  "personas": [
    {{
      "name": "string",
      "role": "string",
      "pain_point": "string"
    }}
  ],
  "features": [
    {{
      "name": "string",
      "description": "string",
      "priority": 1
    }}
  ],
  "user_stories": [
    "As a ..., I want ..., so that ...",
    "As a ..., I want ..., so that ...",
    "As a ..., I want ..., so that ..."
  ]
}}

Rules:
- value_proposition must be concrete and specific to AutoServe AI.
- Include exactly 3 personas.
- Use one persona for a clinic, one for a bakery, and one for a grocery store.
- Each pain point should clearly connect to missed calls, slow WhatsApp replies, lost bookings, or lost orders.
- Include at least 5 features.
- Make the feature set operationally specific. Cover WhatsApp inquiry handling, phone-call handling, booking capture, order capture, and a staff handoff or dashboard workflow.
- Feature priority must be an integer where 1 is highest priority, 2 is medium priority, and 3 is lowest priority.
- Include exactly 3 user stories.
- Use one user story for each of the three core business types: clinic, bakery, and grocery store.
- Every user story must strictly follow the 'As a / I want / so that' format.
- The result must be detailed enough that an engineer can build a landing page and a marketer can write launch messaging without guessing.
- Keep the product practical for small business onboarding and demo execution.
- Do not include any text outside the JSON object.
""".strip()

    def _validate_incoming_message(self, message: Dict[str, Any]) -> None:
        """Validate the incoming CEO message."""
        if message.get("to_agent") != self.agent_name:
            raise ValueError("ProductAgent can only process messages addressed to 'product'.")
        if message.get("message_type") not in {"task", "revision_request"}:
            raise ValueError("ProductAgent only accepts 'task' or 'revision_request' messages.")

        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Incoming message payload must be a dictionary.")
        if not isinstance(payload.get("startup_idea"), str) or not payload["startup_idea"].strip():
            raise ValueError("Incoming payload must include a non-empty 'startup_idea' string.")
        if "focus_area" in payload and not isinstance(payload["focus_area"], str):
            raise ValueError("Incoming 'focus_area' must be a string when provided.")
        if "feedback" in payload and payload["feedback"] is not None and not isinstance(
            payload["feedback"], str
        ):
            raise ValueError("Incoming 'feedback' must be a string when provided.")
        if "previous_product_spec" in payload and payload["previous_product_spec"] is not None:
            if not isinstance(payload["previous_product_spec"], dict):
                raise ValueError("Incoming 'previous_product_spec' must be a dictionary.")

    def _parse_product_spec(self, raw_response: str) -> Dict[str, Any]:
        """Parse, validate, and normalize the product specification JSON."""
        parsed = sanitize_llm_json(raw_response)

        value_proposition = parsed.get("value_proposition")
        personas = parsed.get("personas")
        features = parsed.get("features")
        user_stories = parsed.get("user_stories")

        if not isinstance(value_proposition, str) or not value_proposition.strip():
            raise LLMError("Product spec must include a non-empty 'value_proposition'.")

        normalized_personas = self._validate_personas(personas)
        normalized_features = self._validate_features(features)
        normalized_user_stories = self._validate_user_stories(user_stories)

        return {
            "value_proposition": value_proposition.strip(),
            "personas": normalized_personas,
            "features": normalized_features,
            "user_stories": normalized_user_stories,
        }

    def _validate_personas(self, personas: Any) -> List[Dict[str, str]]:
        """Validate persona entries."""
        if not isinstance(personas, list) or len(personas) < 3:
            raise LLMError("Product spec must include at least 3 personas.")

        normalized_personas: List[Dict[str, str]] = []
        for persona in personas:
            if not isinstance(persona, dict):
                raise LLMError("Each persona must be a JSON object.")

            name = persona.get("name")
            role = persona.get("role")
            pain_point = persona.get("pain_point")
            if not all(
                isinstance(field, str) and field.strip()
                for field in (name, role, pain_point)
            ):
                raise LLMError(
                    "Each persona must include non-empty name, role, and pain_point fields."
                )

            normalized_personas.append(
                {
                    "name": name.strip(),
                    "role": role.strip(),
                    "pain_point": pain_point.strip(),
                }
            )

        return normalized_personas

    def _validate_features(self, features: Any) -> List[Dict[str, Any]]:
        """Validate feature entries and return them sorted by priority."""
        if not isinstance(features, list) or len(features) < 5:
            raise LLMError("Product spec must include at least 5 features.")

        normalized_features: List[Dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                raise LLMError("Each feature must be a JSON object.")

            name = feature.get("name")
            description = feature.get("description")
            priority = feature.get("priority")
            if not isinstance(name, str) or not name.strip():
                raise LLMError("Each feature must include a non-empty name.")
            if not isinstance(description, str) or not description.strip():
                raise LLMError("Each feature must include a non-empty description.")
            if not isinstance(priority, int) or priority not in {1, 2, 3}:
                raise LLMError("Each feature priority must be an integer: 1, 2, or 3.")

            normalized_features.append(
                {
                    "name": name.strip(),
                    "description": description.strip(),
                    "priority": priority,
                }
            )

        return sorted(normalized_features, key=lambda item: (item["priority"], item["name"]))

    def _validate_user_stories(self, user_stories: Any) -> List[str]:
        """Validate user stories."""
        if not isinstance(user_stories, list) or len(user_stories) != 3:
            raise LLMError("Product spec must include exactly 3 user stories.")

        normalized_user_stories: List[str] = []
        for story in user_stories:
            if not isinstance(story, str) or not story.strip():
                raise LLMError("Each user story must be a non-empty string.")

            cleaned_story = story.strip()
            story_lower = cleaned_story.lower()
            if (
                not story_lower.startswith("as a")
                or " i want " not in story_lower
                or " so that " not in story_lower
            ):
                raise LLMError(
                    "Each user story must follow the 'As a / I want / so that' format."
                )
            normalized_user_stories.append(cleaned_story)

        return normalized_user_stories


__all__ = ["ProductAgent"]
