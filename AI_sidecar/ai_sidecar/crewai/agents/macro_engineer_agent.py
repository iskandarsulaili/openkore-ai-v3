from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class MacroEngineerProfile(BehaviorProfile):
    """Macro recording, playback."""

    agent_id = "macro_engineer"
    role = "Macro Engineer"
    goal = "Record, manage, and play back macros for repetitive tasks"
    backstory = (
        "The automation specialist. This agent records repetitive action "
        "sequences as macros, manages the macro library, and replays them "
        "on demand — turning multi-step routines into single commands."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        macro_request = signals.get("macro_request")
        if macro_request:
            return 0.9
        recording = signals.get("macro_recording")
        if recording:
            return 0.7
        return 0.0

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        macro = signals.get("macro_request")
        if macro:
            action = macro.get("action", "play")
            name = macro.get("name", "default")
            if action == "record":
                return {"kind": "macro_record", "command": f"macro record {name}", "confidence": 0.9, "reason": f"Recording macro '{name}'"}
            return {"kind": "macro_play", "command": f"macro play {name}", "confidence": 0.85, "reason": f"Playing macro '{name}'"}

        if signals.get("macro_recording"):
            return {"kind": "macro_stop", "command": "macro stop", "confidence": 0.7, "reason": "Stopping macro recording"}

        return None
