from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class CommandEmitterProfile(BehaviorProfile):
    """Translates decisions to OpenKore commands."""

    agent_id = "command_emitter"
    role = "Command Emitter"
    goal = "Translate agent decisions into precise OpenKore macro commands"
    backstory = (
        "The final step in the decision pipeline. This agent takes "
        "abstract action plans and emits concrete OpenKore command strings "
        "that the bot understands — movement, combat, storage, everything."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        # Always ready — acts when an action is dispatched to it
        return 0.3

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        pending_action = signals.get("pending_action")
        if not pending_action:
            return None

        kind = pending_action.get("kind", "")
        command = pending_action.get("command", "")
        if not command:
            return None

        return {
            "kind": "emit",
            "command": command,
            "confidence": 0.95,
            "reason": f"Emitting {kind} command: {command}",
        }
