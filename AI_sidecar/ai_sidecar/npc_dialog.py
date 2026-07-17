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
    npc_type: str  # "warp", "kafra", "vendor", "quest", "job_change", "refine", "storage", "skill", "heal", "generic"
    dialog_history: list[str] = field(default_factory=list)
    available_options: list[NPCDialogOption] = field(default_factory=list)
    current_text: str = ""
    is_complete: bool = False
    context: dict[str, Any] = field(default_factory=dict)
    sequence_attempted: str = ""  # The npc_steps sequence being used (e.g. "c r1 c r1")
    failure_count: int = 0  # How many times this dialog has failed


# ── NPC type detection from name patterns ──────────────────────────────────

NPC_TYPE_PATTERNS: dict[str, list[str]] = {
    "kafra": ["kafra", "storage", "keeper", "warehouse"],
    "vendor": ["tool", "dealer", "shop", "item", "mart", "store", "merchant", "trade", "pawn", "general", "goods", "seller", "buyer"],
    "warp": ["warp", "portal", "gate", "kafra travel"],
    "healer": ["heal", "nun", "nurse", "priest", "monk", "sister", "recovery", "doctor"],
    "quest": ["quest", "mission", "notice", "board", "guide", "eden"],
    "job_change": ["job", "class", "master", "guild", "association", "change", "advance"],
    "refiner": ["refine", "smith", "forge", "upgrade", "enchant", "blacksmith"],
    "skill": ["skill", "trainer", "master"],
    "identify": ["identify", "appraise", "kara", "judgement", "appraiser"],
}


class NPCDialogEngine:
    """Handles NPC conversations with both heuristic and LLM-powered modes.

    Common NPCs (warp, Kafra, shops) use predefined response sequences.
    Vendor/shop NPCs get special handling for buy/sell/cancel dialog flows.
    Complex NPCs (quest, job change) can use LLM to decide responses.
    """

    # Common NPC response sequences (heuristic mode) - OpenKore response number format
    NPC_ROUTINES: dict[str, list[str]] = {
        "warp": ["1", "1"],        # Talk -> Warp menu -> Select destination
        "kafra": ["1"],             # Talk -> Open storage
        "heal": ["1"],              # Talk -> Get healed
        "healer": ["1"],            # Talk -> Get healed
        "skill": ["1"],             # Talk -> Skill menu
        "storage": ["1", "1"],      # Talk -> Storage -> Open
        "buy": ["1", "1"],          # Talk -> Buy menu -> Browse
        "vendor": ["1", "1"],       # Talk -> Buy menu -> Browse (default for vendors)
        "sell": ["2"],              # Talk -> Sell menu
        "identify": ["1"],          # Talk -> Identify all
        "refiner": ["1", "1"],      # Talk -> Refine menu -> Confirm
        "quest": ["1"],             # Talk -> Quest info (use LLM for complex quest dialogs)
        "job_change": ["1", "1"],   # Talk -> Job menu -> Select
        "generic": ["1"],           # Talk -> First option (generic fallback)
    }

    # Common warp destinations
    WARP_DESTINATIONS: dict[str, dict[str, Any]] = {
        "prontera": {"npc": "Warp Portal", "sequence": ["1", "2", "1"]},
        "morocc": {"npc": "Warp Portal", "sequence": ["1", "2", "3"]},
        "payon": {"npc": "Warp Portal", "sequence": ["1", "2", "5"]},
        "geffen": {"npc": "Warp Portal", "sequence": ["1", "2", "4"]},
        "aldebaran": {"npc": "Warp Portal", "sequence": ["1", "2", "6"]},
    }

    # Vendor names that should trigger shop mode (buy/sell/cancel)
    VENDOR_NAME_PATTERNS: list[str] = [
        "tool", "dealer", "item", "shop", "mart", "general", "goods",
        "weapon", "armor", "potion", "accessory", "consumable",
    ]

    # NPC names that are known to use specific sequences
    NPC_SPECIFIC_SEQUENCES: dict[str, dict[str, Any]] = {
        "tool dealer": {
            "sequences": {
                "buy": "c r1 c r1",
                "sell": "c r2 c r1",
            },
            "notes": "Standard tool dealer - buy menu then browse",
            "type": "vendor",
        },
        "kafra": {
            "sequences": {},
            "notes": "OpenKore handles Kafra via built-in AI",
            "type": "kafra",
        },
        "warp portal": {
            "sequences": {},
            "notes": "Use warp sequence with destination selection",
            "type": "warp",
        },
    }

    def __init__(self, experience_db=None, llm_adapter=None):
        self._exp_db = experience_db
        self._llm = llm_adapter
        self._active_dialogs: dict[str, NPCDialogState] = {}

    def start_dialog(self, bot_id: str, npc_id: str, npc_name: str, npc_type: str = "generic") -> str:
        """Start or continue an NPC conversation. Returns the command to execute."""
        # Auto-detect NPC type from name if not specified
        if npc_type == "generic" or npc_type == "":
            detected = self._detect_npc_type(npc_name)
            if detected:
                npc_type = detected
                logger.info("npc_type_auto_detected: name=%s type=%s", npc_name, npc_type)

        state = NPCDialogState(
            npc_id=npc_id,
            npc_name=npc_name,
            npc_type=npc_type,
        )

        # Check for known vendor NPC sequences
        if npc_type == "vendor":
            seq_info = self.NPC_SPECIFIC_SEQUENCES.get(npc_name.lower().strip())
            if seq_info:
                state.sequence_attempted = seq_info["sequences"].get("buy", "c r1 c r1")
                logger.info("npc_vendor_known_sequence: name=%s seq=%s", npc_name, state.sequence_attempted)
            else:
                state.sequence_attempted = "c r1 c r1"
                logger.info("npc_vendor_default_sequence: name=%s seq=%s", npc_name, state.sequence_attempted)

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

        # Detect failure patterns
        if self._is_dialog_failure(npc_text):
            state.failure_count += 1
            logger.warning("npc_dialog_possible_failure: bot=%s npc=%s text=%s",
                           bot_id, state.npc_name, npc_text[:100])
            if state.failure_count >= 3:
                logger.error("npc_dialog_repeated_failure: bot=%s npc=%s count=%d",
                             bot_id, state.npc_name, state.failure_count)
                state.is_complete = True
                return None

        if not options:
            # No more options — conversation complete
            state.is_complete = True
            return None

        # Detect buy/sell/cancel shop interface
        if self._is_shop_interface(npc_text, options):
            logger.info("npc_shop_interface_detected: bot=%s npc=%s", bot_id, state.npc_name)
            return self._handle_shop_interface(state)

        # Try heuristic first
        choice = self._heuristic_choice(state)
        if choice is not None:
            return f"response {choice}"

        # Fall back to LLM if available
        if self._llm is not None:
            return self._llm_choice(state)

        # Last resort: pick first option
        return "response 1"

    def _is_dialog_failure(self, npc_text: str) -> bool:
        """Check if the NPC response indicates a dialog failure."""
        lower = npc_text.lower()
        failure_patterns = [
            "did not respond", "no response", "talking to wrong npc",
            "wrong npc", "that npc is not here", "npc not found",
            "conversation ended unexpectedly", "no such npc",
        ]
        for pattern in failure_patterns:
            if pattern in lower:
                return True
        return False

    def _is_shop_interface(self, npc_text: str, options: list[dict[str, Any]]) -> bool:
        """Detect if the NPC is showing a buy/sell/cancel shop interface."""
        lower = npc_text.lower()
        # Shop interfaces usually mention buy/sell/cancel
        shop_keywords = ["buy", "sell", "cancel", "purchase", "trade", "browse"]
        if any(k in lower for k in shop_keywords):
            return True
        # Check options for shop-like choices
        option_texts = [str(o.get("text", "")).lower() for o in options]
        # If options include buy AND sell, it's a shop
        has_buy = any("buy" in t for t in option_texts)
        has_sell = any("sell" in t for t in option_texts)
        has_cancel = any("cancel" in t for t in option_texts)
        if (has_buy or has_sell) and has_cancel:
            return True
        return False

    def _handle_shop_interface(self, state: NPCDialogState) -> str:
        """Handle the buy/sell/cancel shop interface."""
        options = state.available_options
        npc_name = state.npc_name.lower()

        # Check if we need to buy (default for vendors with 'tool' or 'dealer' in name)
        is_tool_dealer = any(p in npc_name for p in ["tool", "dealer", "item", "general", "goods"])
        objective = state.context.get("objective", "buy" if is_tool_dealer else "check")

        for opt in options:
            opt_text = opt.text.lower()
            if objective == "buy" and any(k in opt_text for k in ["buy", "purchase", "browse", "shop"]):
                return f"response {opt.index}"
            if objective == "sell" and "sell" in opt_text:
                return f"response {opt.index}"
            # Default: close/cancel if we can't determine
            if "cancel" in opt_text or "close" in opt_text or "exit" in opt_text:
                logger.info("npc_shop_cancel: npc=%s", state.npc_name)
                state.is_complete = True
                return f"response {opt.index}"

        # Fallback: pick buy option if available
        for opt in options:
            opt_text = opt.text.lower()
            if any(k in opt_text for k in ["buy", "1", "purchase"]):
                return f"response {opt.index}"

        return "response 1"

    def _heuristic_choice(self, state: NPCDialogState) -> str | None:
        """Heuristic NPC response selection based on NPC type."""
        npc_type = state.npc_type

        # Check if we have a routine for this NPC type
        routine = self.NPC_ROUTINES.get(npc_type)
        if not routine:
            # Try the type's base name (e.g., "vendor" is already in routines)
            routine = self.NPC_ROUTINES.get(npc_type)

        if routine:
            step = len(state.dialog_history)
            if step < len(routine):
                return routine[step]
            # Routine complete
            state.is_complete = True
            return None

        # Check warp destinations
        dest_result = self._check_warp_destination(state)
        if dest_result is not None:
            return dest_result

        return None

    def _check_warp_destination(self, state: NPCDialogState) -> str | None:
        """Check if this is a warp NPC and return appropriate sequence."""
        for dest, info in self.WARP_DESTINATIONS.items():
            if state.npc_name.lower() == info["npc"].lower():
                seq = info["sequence"]
                step = len(state.dialog_history)
                if step < len(seq):
                    return seq[step]
                state.is_complete = True
                return None
        return None

    def _llm_choice(self, state: NPCDialogState) -> str:
        """Use LLM to decide which NPC response option to choose, with RO context."""
        try:
            npc_type_desc = {
                "warp": "Warp/Skill destination selection NPC",
                "kafra": "Kafra storage/service NPC",
                "vendor": "Item shop/vendor NPC - for buying potions, equipment, etc.",
                "healer": "Healer NPC - provides free healing",
                "quest": "Quest NPC - handles quest progression",
                "job_change": "Job change NPC - handles class advancement",
                "refiner": "Refiner NPC - upgrades/enchants equipment",
                "skill": "Skill NPC - teaches or resets skills",
                "identify": "Identifier NPC - identifies unknown items",
                "generic": "Generic NPC - unknown purpose",
            }.get(state.npc_type, "Unknown NPC type")

            prompt = f"""You are a Ragnarok Online bot talking to an NPC.
NPC: {state.npc_name}
NPC Type: {state.npc_type} ({npc_type_desc})
NPC says: {state.current_text[:300]}
Available responses:"""
            for opt in state.available_options:
                prompt += f"\n{opt.index}. {opt.text[:150]}"
            prompt += f"""
Conversation history: {len(state.dialog_history)} turns
Context: {json.dumps(state.context, indent=2) if state.context else "No additional context"}

Select the BEST response option that progresses the bot's objective.
- If this is a vendor shop: choose "Buy" to purchase items
- If this is a warp NPC: choose the correct destination
- If this is a quest NPC: choose dialogue that progresses the quest
- If in doubt, choose the first option (option 1)

Respond with ONLY the option number (a single digit)."""

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

    @staticmethod
    def _detect_npc_type(npc_name: str) -> str | None:
        """Detect NPC type from name using pattern matching."""
        if not npc_name:
            return None
        lower_name = npc_name.lower().strip()
        for npc_type, patterns in NPC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in lower_name:
                    return npc_type
        return None

    def get_state(self, bot_id: str) -> NPCDialogState | None:
        return self._active_dialogs.get(bot_id)

    def end_dialog(self, bot_id: str) -> None:
        self._active_dialogs.pop(bot_id, None)

    def record_dialog_failure(self, bot_id: str, npc_name: str) -> None:
        """Record a dialog failure from external signal (e.g., bridge plugin)."""
        state = self._active_dialogs.get(bot_id)
        if state is None:
            state = NPCDialogState(
                npc_id="unknown",
                npc_name=npc_name,
                npc_type=self._detect_npc_type(npc_name) or "generic",
            )
            self._active_dialogs[bot_id] = state
        state.failure_count += 1
        logger.warning("npc_dialog_failure_recorded: bot=%s npc=%s total=%d",
                       bot_id, npc_name, state.failure_count)
