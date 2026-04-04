"""Main runner for the LaunchMind AutoServe AI multi-agent workflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict

from dotenv import load_dotenv

from agents.ceo_agent import CEOAgent
from agents.engineer_agent import EngineerAgent
from agents.marketing_agent import MarketingAgent
from agents.product_agent import ProductAgent
from agents.qa_agent import QAAgent
from message_bus import MessageBus


DEFAULT_STARTUP_IDEA = (
    "An AI-powered voice and chat automation platform for small businesses such "
    "as clinics, bakeries, and grocery stores. It should help them handle customer "
    "inquiries, bookings, and orders automatically through WhatsApp and phone calls, "
    "reducing manual workload and improving response time."
)
MAX_WORKFLOW_ROUNDS = 30


def run_workflow(startup_idea: str) -> Dict[str, Any]:
    """Run the full AutoServe AI LaunchMind workflow."""
    workspace_root = Path(__file__).resolve().parent
    load_dotenv(workspace_root / ".env")

    bus = MessageBus()
    ceo_agent = CEOAgent(bus)
    agents = {
        "product": ProductAgent(bus),
        "engineer": EngineerAgent(bus, str(workspace_root)),
        "marketing": MarketingAgent(bus),
        "qa": QAAgent(bus),
        "ceo": ceo_agent,
    }

    print("=" * 100)
    print("[Main] Starting LaunchMind workflow for AutoServe AI")
    print("=" * 100)
    ceo_agent.start_workflow(startup_idea)

    for round_number in range(1, MAX_WORKFLOW_ROUNDS + 1):
        print(f"[Main] Workflow round {round_number}")
        progress_made = False

        for agent_name in ["product", "engineer", "marketing", "qa", "ceo"]:
            messages = bus.get_new_messages(agent_name)
            if not messages:
                continue

            progress_made = True
            for message in messages:
                if agent_name == "ceo" and message.get("from_agent") == "ceo":
                    print("[Main] CEO internal confirmation received.")
                    continue

                print(
                    f"[Main] Dispatching {message['message_type']} from "
                    f"{message['from_agent']} to {message['to_agent']}."
                )
                agents[agent_name].process_message(message)

        if ceo_agent.is_workflow_complete():
            break

        if not progress_made:
            raise RuntimeError(
                "Workflow stalled with no new messages. Check configuration and agent outputs."
            )

    if not ceo_agent.is_workflow_complete():
        raise RuntimeError("Workflow ended before reaching a successful completion state.")

    print("=" * 100)
    print("[Main] Final Summary")
    print("=" * 100)
    print(ceo_agent.state["final_summary"])
    print("=" * 100)
    bus.print_message_log()
    return ceo_agent.state["final_summary"]


def main() -> int:
    """CLI entrypoint for the LaunchMind workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    startup_idea = os.getenv("STARTUP_IDEA") or DEFAULT_STARTUP_IDEA

    if len(sys.argv) > 1:
        startup_idea = " ".join(sys.argv[1:]).strip()

    try:
        run_workflow(startup_idea)
        return 0
    except Exception as exc:  # noqa: BLE001 - keep CLI failure reporting simple
        print(f"[Main] Workflow failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
