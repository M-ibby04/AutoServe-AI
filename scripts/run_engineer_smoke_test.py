"""Engineer Agent smoke test helper.

Default behavior generates and validates the landing page package locally without
touching GitHub. Use --publish to run the full Engineer Agent flow, which will
create real GitHub artifacts for the configured repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from dotenv import load_dotenv


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agents.engineer_agent import EngineerAgent
from message_bus import MessageBus


SAMPLE_STARTUP_IDEA = (
    "An AI-powered voice and chat automation platform for small businesses such "
    "as clinics, bakeries, and grocery stores. It should help them handle customer "
    "inquiries, bookings, and orders automatically through WhatsApp and phone calls, "
    "reducing manual workload and improving response time."
)

SAMPLE_PRODUCT_SPEC: Dict[str, Any] = {
    "value_proposition": (
        "AutoServe AI helps small businesses automate WhatsApp chats and phone-call "
        "inquiries so teams can respond faster, capture more bookings, and miss fewer orders."
    ),
    "personas": [
        {
            "name": "Ayesha",
            "role": "Clinic receptionist",
            "pain_point": "She loses time answering the same appointment questions all day.",
        },
        {
            "name": "Usman",
            "role": "Bakery owner",
            "pain_point": "He misses cake-order calls during busy hours and loses sales.",
        },
        {
            "name": "Hina",
            "role": "Grocery store manager",
            "pain_point": "Her team struggles to manage delivery inquiries across calls and WhatsApp.",
        },
    ],
    "features": [
        {
            "name": "WhatsApp inquiry automation",
            "description": "Answers common customer questions instantly through WhatsApp.",
            "priority": 1,
        },
        {
            "name": "Call handling assistant",
            "description": "Handles routine phone-call inquiries and reduces missed calls.",
            "priority": 1,
        },
        {
            "name": "Booking workflow capture",
            "description": "Collects booking details for clinics and service businesses.",
            "priority": 1,
        },
        {
            "name": "Order intake automation",
            "description": "Captures bakery and grocery orders with less manual effort.",
            "priority": 2,
        },
        {
            "name": "Team dashboard summary",
            "description": "Shows pending conversations, bookings, and unresolved customer intents.",
            "priority": 2,
        },
    ],
    "user_stories": [
        "As a clinic manager, I want appointment questions handled automatically so that my staff can focus on patients.",
        "As a bakery owner, I want orders captured from calls and WhatsApp so that I do not lose customers during rush hours.",
        "As a grocery store manager, I want routine inquiries answered quickly so that my team can reduce response delays.",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test."""
    parser = argparse.ArgumentParser(
        description=(
            "Preview or publish the Engineer Agent flow using a realistic sample payload."
        )
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Run the full Engineer Agent flow, including real GitHub issue/branch/commit/PR creation.",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        help="In preview mode, also write landing_page.html locally for manual inspection.",
    )
    parser.add_argument(
        "--reuse-issue-number",
        type=int,
        help="Reuse an existing GitHub issue number when publishing an updated landing page.",
    )
    parser.add_argument(
        "--reuse-issue-url",
        help="Reuse an existing GitHub issue URL when publishing an updated landing page.",
    )
    parser.add_argument(
        "--reuse-pr-number",
        type=int,
        help="Reuse an existing pull request number when publishing an updated landing page.",
    )
    parser.add_argument(
        "--reuse-pr-url",
        help="Reuse an existing pull request URL when publishing an updated landing page.",
    )
    parser.add_argument(
        "--reuse-branch",
        help="Reuse an existing branch name when publishing an updated landing page.",
    )
    return parser.parse_args()


def ensure_llm_env() -> None:
    """Validate the minimum LLM config needed for generation."""
    import os

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    alt_key = os.getenv("LLM_API_KEY", "").strip()
    if not openai_key and not alt_key:
        raise RuntimeError(
            "Missing LLM credentials. Set OPENAI_API_KEY or LLM_API_KEY in .env."
        )


def ensure_github_env() -> None:
    """Validate the minimum GitHub config needed for publish mode."""
    import os

    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN in .env.")
    if not repo or "/" not in repo or repo.startswith("http"):
        raise RuntimeError(
            "GITHUB_REPO must be in owner/repo format, for example M-ibby04/AutoServe-AI."
        )


def build_engineer_payload() -> Dict[str, Any]:
    """Build a realistic sample Engineer task payload."""
    return {
        "startup_name": "AutoServe AI",
        "startup_idea": SAMPLE_STARTUP_IDEA,
        "product_spec": SAMPLE_PRODUCT_SPEC,
        "launch_goal": "Create the first public landing page and GitHub PR for AutoServe AI.",
    }


def build_previous_result(args: argparse.Namespace) -> Dict[str, Any] | None:
    """Build an optional previous_result payload for update publishes."""
    previous_result: Dict[str, Any] = {}
    if args.reuse_issue_number is not None:
        previous_result["issue_number"] = args.reuse_issue_number
    if args.reuse_issue_url:
        previous_result["issue_url"] = args.reuse_issue_url
    if args.reuse_pr_number is not None:
        previous_result["pr_number"] = args.reuse_pr_number
    if args.reuse_pr_url:
        previous_result["pr_url"] = args.reuse_pr_url
    if args.reuse_branch:
        previous_result["branch"] = args.reuse_branch

    return previous_result or None


def run_preview(agent: EngineerAgent) -> int:
    """Generate and validate the engineer deliverable without GitHub side effects."""
    payload = build_engineer_payload()
    deliverable = agent.generate_deliverable(
        startup_name=payload["startup_name"],
        startup_idea=payload["startup_idea"],
        product_spec=payload["product_spec"],
        launch_goal=payload["launch_goal"],
        feedback=None,
        previous_result=None,
    )

    print("[Preview] Landing page package generated successfully.")
    print(json.dumps(
        {
            "headline": deliverable["headline"],
            "subheadline": deliverable["subheadline"],
            "call_to_action": deliverable["call_to_action"],
            "issue_title": deliverable["issue_title"],
            "branch_name": deliverable["branch_name"],
            "pr_title": deliverable["pr_title"],
        },
        indent=2,
    ))
    return 0


def write_preview_html(agent: EngineerAgent) -> None:
    """Generate and write the preview HTML locally."""
    payload = build_engineer_payload()
    deliverable = agent.generate_deliverable(
        startup_name=payload["startup_name"],
        startup_idea=payload["startup_idea"],
        product_spec=payload["product_spec"],
        launch_goal=payload["launch_goal"],
        feedback=None,
        previous_result=None,
    )
    agent._write_landing_page(deliverable["landing_page_path"], deliverable["html"])
    print(f"[Preview] Local HTML written to {agent.landing_page_path}")


def run_publish(
    agent: EngineerAgent,
    bus: MessageBus,
    previous_result: Dict[str, Any] | None = None,
) -> int:
    """Run the real Engineer Agent task flow, including GitHub publishing."""
    payload = build_engineer_payload()
    if previous_result:
        payload["previous_result"] = previous_result

    task_message = bus.send_message(
        from_agent="ceo",
        to_agent="engineer",
        message_type="task",
        payload=payload,
    )
    result = agent.process_message(task_message)
    if result is None:
        raise RuntimeError("EngineerAgent returned no message in publish mode.")

    print(f"[Publish] Message type: {result['message_type']}")
    print(json.dumps(result["payload"], indent=2))

    if result["message_type"] == "failure":
        return 1
    return 0


def main() -> int:
    """Run the selected smoke test mode."""
    load_dotenv(dotenv_path=WORKSPACE_ROOT / ".env", override=True)

    args = parse_args()
    bus = MessageBus()
    agent = EngineerAgent(bus, str(WORKSPACE_ROOT))

    ensure_llm_env()
    if args.publish:
        ensure_github_env()
        return run_publish(agent, bus, previous_result=build_previous_result(args))

    exit_code = run_preview(agent)
    if args.write_html:
        write_preview_html(agent)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
