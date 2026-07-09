from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NPCDialogOption:
    """A single dialog option presented by an NPC."""
    index: int
    text: str


@dataclass
class NPCDialogState:
    """Current state of an NPC conversation."""
    npc_id: str
    npc_name: str
    npc_type: str  # "warp", "kafra", "shop", "quest", "job_change", "refine", "storage", "skill", "heal", "generic"
    dialog_history: list[str] = field(default_factory=list)
    available_options: list[NPCDialogOption] = field(default_factory=list)
    current_text: str = ""
    is_complete: bool = False
    context: dict[str, Any] = field(default_factory=dict)


class NPCDialogEngine:
    """Handles NPC conversations with both heuristic and LLM-powered modes.
    
    Common NPCs (warp, Kafra, shops) use predefined response sequences.
    Complex NPCs (quest, job change) can use LLM to decide responses.
    """

    # Common NPC response sequences (heuristic mode)
    NPC_ROUTINES: dict[str, list[str]] = {
        "warp": ["1", "1"],  # Talk -> Warp menu -> Select destination
        "kafra": ["1"],  # Talk -> Open storage
        "heal": ["1"],  # Talk -> Get healed
        "skill": ["1"],  # Talk -> Skill menu
        "storage": ["1", "1"],  # Talk -> Storage -> Open
        "buy": ["1", "1"],  # Talk -> Buy menu -> Browse
        "sell": ["2"],  # Talk -> Sell menu
    }

    # Common warp destinations
    WARP_DESTINATIONS: dict[str, dict[str, Any]] = {
        "prontera": {"npc": "Warp Portal", "sequence": ["1", "2", "1"]},  # Talk -> Warps -> Prontera
        "morocc": {"npc": "Warp Portal", "sequence": ["1", "2", "3"]},
        "payon": {"npc": "Warp Portal", "sequence": ["1", "2", "5"]},
        "geffen": {"npc": "Warp Portal", "sequence": ["1", "2", "4"]},
        "aldebaran": {"npc": "Warp Portal", "sequence": ["1", "2", "6"]},
    }

    def __init__(self, experience_db=None, llm_adapter=None):
        self._exp_db = experience_db
        self._llm = llm_adapter
        self._active_dialogs: dict[str, NPCDialogState] = {}

    def start_dialog(self, bot_id: str, npc_id: str, npc_name: str, npc_type: str = "generic") -> str:
        """Start or continue an NPC conversation. Returns the command to execute."""
        state = NPCDialogState(
            npc_id=npc_id,
            npc_name=npc_name,
            npc_type=npc_type,
        )
        self._active_dialogs[bot_id] = state
        return f"talknpc {npc_id}"

    def process_response(self, bot_id: str, npc_text: str, options: list[dict[str, Any]]) -> str | None:
        """Process NPC response text and available options. Returns next command or None if complete."""
        state = self._active_dialogs.get(bot_id)
        if state is None:
            return None

        state.dialog_history.append(npc_text)
        state.current_text = npc_text
        state.available_options = [
            NPCDialogOption(index=o.get("index", i), text=o.get("text", ""))
            for i, o in enumerate(options)
        ] if options else []

        if not options:
            # No more options — conversation complete
            state.is_complete = True
            return None

        # Try heuristic first
        choice = self._heuristic_choice(state)
        if choice is not None:
            return f"response {choice}"

        # Fall back to LLM if available
        if self._llm is not None:
            return self._llm_choice(state)

        # Last resort: pick first option
        return "response 1"

    def _heuristic_choice(self, state: NPCDialogState) -> str | None:
        """Heuristic NPC response selection based on NPC type."""
        npc_type = state.npc_type
        
        # Check if we have a routine for this NPC type
        routine = self.NPC_ROUTINES.get(npc_type)
        if routine:
            step = len(state.dialog_history)
            if step < len(routine):
                return routine[step]
            # Routine complete
            state.is_complete = True
            return None
        
        # Check warp destinations
        for dest, info in self.WARP_DESTINATIONS.items():
            if state.npc_name == info["npc"]:
                seq = info["sequence"]
                step = len(state.dialog_history)
                if step < len(seq):
                    return seq[step]
                state.is_complete = True
                return None

        return None

    def _llm_choice(self, state: NPCDialogState) -> str:
        """Use LLM to decide which NPC response option to choose."""
        try:
            prompt = f"""You are a Ragnarok Online bot talking to an NPC.
NPC: {state.npc_name} ({state.npc_type})
NPC says: {state.current_text[:200]}
Available responses:
"""
            for opt in state.available_options:
                prompt += f"{opt.index}. {opt.text[:100]}\n"
            prompt += f"\nConversation history: {len(state.dialog_history)} turns\n"
            prompt += "\nRespond with ONLY the option number (a single digit) that best progresses the objective."

            # Use the LLM adapter synchronously
            if self._llm is not None:
                import asyncio
                try:
                    result = asyncio.run(self._llm(prompt))
                    if result and result.strip().isdigit():
                        return f"response {result.strip()}"
                except Exception:
                    pass
        except Exception:
            logger.exception("npc_llm_choice_failed")

        return "response 1"

    def get_state(self, bot_id: str) -> NPCDialogState | None:
        return self._active_dialogs.get(bot_id)

    def end_dialog(self, bot_id: str) -> None:
        self._active_dialogs.pop(bot_id, None)
