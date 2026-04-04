"""Shared in-memory message bus for the LaunchMind multi-agent system."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4


AGENT_NAMES = {"ceo", "product", "engineer", "marketing", "qa"}
ALLOWED_MESSAGE_TYPES = {
    "task",
    "result",
    "revision_request",
    "confirmation",
    "failure",
}


class MessageValidationError(ValueError):
    """Raised when a message does not match the required schema."""


class MessageBus:
    """Store and retrieve structured inter-agent messages in memory."""

    def __init__(self) -> None:
        """Initialize the message log and per-agent offsets."""
        self._messages: List[Dict[str, Any]] = []
        self._agent_offsets: Dict[str, int] = {agent_name: 0 for agent_name in AGENT_NAMES}

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        parent_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create, validate, store, and return a message."""
        message: Dict[str, Any] = {
            "message_id": str(uuid4()),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if parent_message_id is not None:
            message["parent_message_id"] = parent_message_id

        self._validate_message(message)
        self._messages.append(deepcopy(message))
        return deepcopy(message)

    def get_messages(self, agent_name: str) -> List[Dict[str, Any]]:
        """Return all messages addressed to the given agent."""
        self._validate_agent_name(agent_name)
        return [
            deepcopy(message)
            for message in self._messages
            if message["to_agent"] == agent_name
        ]

    def get_new_messages(
        self,
        agent_name: str,
        last_seen_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return new messages for an agent since the last tracked offset."""
        self._validate_agent_name(agent_name)

        start_index = (
            last_seen_index
            if last_seen_index is not None
            else self._agent_offsets.get(agent_name, 0)
        )
        if not isinstance(start_index, int) or start_index < 0:
            raise ValueError("last_seen_index must be a non-negative integer.")

        new_messages = [
            deepcopy(message)
            for index, message in enumerate(self._messages)
            if index >= start_index and message["to_agent"] == agent_name
        ]
        self._agent_offsets[agent_name] = len(self._messages)
        return new_messages

    def reset_agent_offset(self, agent_name: str, index: int = 0) -> None:
        """Reset the tracked unread pointer for a specific agent."""
        self._validate_agent_name(agent_name)
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer.")
        self._agent_offsets[agent_name] = index

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Return the full message history."""
        return deepcopy(self._messages)

    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Return a single message by ID, if present."""
        for message in self._messages:
            if message["message_id"] == message_id:
                return deepcopy(message)
        return None

    def get_workflow_trace(self, message_id: str) -> List[Dict[str, Any]]:
        """Follow a message chain from a child message back to the root."""
        trace: List[Dict[str, Any]] = []
        current_id: Optional[str] = message_id

        while current_id:
            message = self.get_message_by_id(current_id)
            if message is None:
                break
            trace.append(message)
            current_id = message.get("parent_message_id")

        trace.reverse()
        return trace

    def print_message_log(self) -> None:
        """Print the full message history in a demo-friendly terminal format."""
        if not self._messages:
            print("[MessageBus] No messages have been sent yet.")
            return

        print("[MessageBus] Full Message Log")
        print("=" * 100)
        for index, message in enumerate(self._messages, start=1):
            print(
                f"{index:02d}. {message['timestamp']} | "
                f"{message['from_agent']} -> {message['to_agent']} | "
                f"{message['message_type']}"
            )
            print(f"    message_id: {message['message_id']}")
            if "parent_message_id" in message:
                print(f"    parent_message_id: {message['parent_message_id']}")
            print(
                "    payload: "
                f"{json.dumps(message['payload'], indent=2, sort_keys=True)}"
            )
            print("-" * 100)

    def _validate_message(self, message: Dict[str, Any]) -> None:
        """Validate message structure, types, and JSON serializability."""
        required_fields = {
            "message_id": str,
            "from_agent": str,
            "to_agent": str,
            "message_type": str,
            "payload": dict,
            "timestamp": str,
        }
        missing_fields = [field for field in required_fields if field not in message]
        if missing_fields:
            raise MessageValidationError(
                f"Message is missing required fields: {missing_fields}"
            )

        for field, expected_type in required_fields.items():
            if not isinstance(message[field], expected_type):
                raise MessageValidationError(
                    f"Field '{field}' must be of type {expected_type.__name__}."
                )

        self._validate_agent_name(message["from_agent"])
        self._validate_agent_name(message["to_agent"])

        if message["message_type"] not in ALLOWED_MESSAGE_TYPES:
            raise MessageValidationError(
                "Field 'message_type' must be one of: "
                f"{sorted(ALLOWED_MESSAGE_TYPES)}"
            )

        if "parent_message_id" in message and not isinstance(
            message["parent_message_id"], str
        ):
            raise MessageValidationError(
                "Field 'parent_message_id' must be a string when provided."
            )

        self._validate_timestamp(message["timestamp"])
        self._ensure_json_serializable(message)

    @staticmethod
    def _validate_timestamp(timestamp: str) -> None:
        """Ensure the timestamp is a valid ISO 8601 string."""
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise MessageValidationError(
                "Field 'timestamp' must be a valid ISO 8601 string."
            ) from exc

    @staticmethod
    def _ensure_json_serializable(message: Dict[str, Any]) -> None:
        """Ensure the full message can be JSON serialized."""
        try:
            json.dumps(message)
        except (TypeError, ValueError) as exc:
            raise MessageValidationError("Message must be JSON-serializable.") from exc

    @staticmethod
    def _validate_agent_name(agent_name: str) -> None:
        """Ensure agent names stay consistent across the system."""
        if not isinstance(agent_name, str) or agent_name not in AGENT_NAMES:
            raise MessageValidationError(
                f"Agent name must be one of: {sorted(AGENT_NAMES)}"
            )


__all__ = [
    "AGENT_NAMES",
    "ALLOWED_MESSAGE_TYPES",
    "MessageBus",
    "MessageValidationError",
]
