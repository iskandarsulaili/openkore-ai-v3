"""NPC dialogue parsing, conversation state tracking, and response selection."""
from __future__ import annotations

import logging
import re
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Tracks ongoing NPC conversation state."""
    npc_name: str = ""
    npc_id: int = 0
    npc_type: str = ""  # merchant, quest, storage, repair, job, healer
    stage: str = "init"  # init -> talking -> selecting -> confirming -> done
    dialogue_history: list[str] = field(default_factory=list)
    options_seen: list[dict] = field(default_factory=list)
    selected_option: int = -1
    turn_count: int = 0
    last_npc_talk: str = ""
    completed: bool = False


# Common RO NPC dialogue patterns
_NPC_RESPONSE_MAP: dict[str, dict[str, list[str]]] = {
    "merchant": {
        "greeting": [
            "what can i do for you",
            "welcome", "hello", "how may i help you",
            "what do you want to buy",
            "looking for something",
            "come in",
        ],
        "buy_menu": [
            "buy", "purchase", "take a look",
            "what are you selling",
        ],
        "sell_menu": [
            "sell", "i have something to sell",
        ],
    },
    "quest": {
        "available": [
            "i have a request", "could you help me",
            "i need your help", "a little favor",
            "quest", "mission", "task",
        ],
        "in_progress": [
            "did you get it", "have you found",
            "come back when", "bring me",
        ],
        "complete": [
            "thank you", "well done", "here is your reward",
            "you have completed",
        ],
    },
    "storage": {
        "greeting": [
            "kafra", "storage", "keep your items",
            "deposit", "withdraw",
        ],
        "menu": [
            "deposit", "withdraw", "open storage",
        ],
    },
    "repair": {
        "greeting": [
            "broken", "repair", "fix",
            "sharpen", "mend",
        ],
    },
    "healer": {
        "greeting": [
            "heal", "cure", "blessing",
            "recover", "restore",
        ],
    },
    "job": {
        "greeting": [
            "job change", "class change", "advancement",
            "become a", "are you ready to change",
        ],
    },
}


class NPCDialogueEngine:
    """Handles NPC dialogue parsing, conversation state, and response selection."""

    def __init__(self, db: Any = None) -> None:
        self._conversations: dict[str, ConversationState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def get_or_create_conversation(self, bot_id: str, npc_name: str = "") -> ConversationState:
        """Get existing conversation state or create a new one."""
        if bot_id not in self._conversations:
            self._conversations[bot_id] = ConversationState(npc_name=npc_name)
        conv = self._conversations[bot_id]
        if npc_name:
            conv.npc_name = npc_name
        return conv

    def reset_conversation(self, bot_id: str) -> None:
        """Reset conversation state for a bot."""
        if bot_id in self._conversations:
            self._conversations[bot_id] = ConversationState()

    def parse_npc_talk(self, npc_talk: str, bot_id: str) -> dict[str, Any]:
        """Parse what the NPC is saying and classify the conversation stage.

        Returns a dict with:
            - stage: current conversation stage
            - npc_type: guessed NPC type
            - options: detected response options (if any)
            - confidence: how confident the match is
        """
        conv = self.get_or_create_conversation(bot_id)
        conv.last_npc_talk = npc_talk
        conv.dialogue_history.append(npc_talk)
        conv.turn_count += 1

        talk_lower = npc_talk.lower()

        # Detect NPC type from dialogue
        npc_type = self._classify_npc(talk_lower)
        if npc_type:
            conv.npc_type = npc_type

        # Detect response options (e.g. "1. Buy  2. Sell  3. Cancel")
        options = self._extract_options(npc_talk)
        conv.options_seen = options

        # Determine stage
        stage = self._determine_stage(talk_lower, conv)
        conv.stage = stage

        return {
            "stage": stage,
            "npc_type": conv.npc_type,
            "options": options,
            "turn_count": conv.turn_count,
            "npc_talk": npc_talk,
        }

    def select_response(self, bot_id: str, preference: str = "auto") -> int:
        """Select the best response option index based on NPC type and intent.

        Args:
            bot_id: Bot identifier
            preference: 'auto', 'buy', 'sell', 'deposit', 'withdraw', etc.

        Returns:
            Option index (0-based), or -1 if no option matches
        """
        conv = self.get_or_create_conversation(bot_id)
        if not conv.options_seen:
            return -1

        # Map preferences to option text keywords
        pref_keywords: dict[str, list[str]] = {
            "buy": ["buy", "purchase", "shop", "trade"],
            "sell": ["sell", "dispose"],
            "deposit": ["deposit", "store"],
            "withdraw": ["withdraw", "take out"],
            "storage": ["storage", "kafra"],
            "repair": ["repair", "fix", "sharpen"],
            "heal": ["heal", "cure", "recovery", "blessing"],
            "quest": ["quest", "accept", "receive"],
            "cancel": ["cancel", "no", "exit", "leave"],
            "continue": ["continue", "next", "yes", "ok"],
        }

        # Get keywords for the preference
        keywords = pref_keywords.get(preference, ["continue", "next", "ok", "yes"])
        if preference == "auto":
            # Auto-pick based on NPC type
            type_to_pref = {
                "merchant": "buy",
                "storage": "deposit",
                "repair": "repair",
                "healer": "heal",
                "quest": "continue",
                "job": "continue",
            }
            keywords = pref_keywords.get(type_to_pref.get(conv.npc_type, "continue"),
                                         ["continue", "next", "yes", "ok"])

        # Score options by keyword match
        best_idx = -1
        best_score = 0
        for i, opt in enumerate(conv.options_seen):
            opt_lower = (opt.get("text", "") or "").lower()
            score = sum(1 for kw in keywords if kw in opt_lower)
            # Prefer numeric shorter options (likely the actual action)
            if score > best_score or (score == best_score and opt.get("index", 0) < 10):
                best_score = score
                best_idx = i

        if best_idx >= 0:
            conv.selected_option = best_idx
            conv.stage = "selecting"

        return best_idx

    def get_response_command(self, bot_id: str) -> str | None:
        """Get the OpenKore 'talk resp' command for the selected option.

        Returns None if no option was selected.
        """
        conv = self.get_or_create_conversation(bot_id)
        if conv.selected_option < 0:
            return None
        # OpenKore uses 1-based response indexing
        option_num = conv.selected_option + 1
        return f"talk resp {option_num}"

    def is_conversation_active(self, bot_id: str) -> bool:
        """Check if an NPC conversation is in progress."""
        if bot_id not in self._conversations:
            return False
        conv = self._conversations[bot_id]
        return conv.stage not in ("done", "init") and not conv.completed

    def mark_complete(self, bot_id: str) -> None:
        """Mark conversation as complete."""
        conv = self.get_or_create_conversation(bot_id)
        conv.completed = True
        conv.stage = "done"

    def _classify_npc(self, talk_lower: str) -> str:
        """Classify NPC type from dialogue text."""
        scores: dict[str, int] = {}
        for npc_type, patterns in _NPC_RESPONSE_MAP.items():
            for stage_name, phrases in patterns.items():
                for phrase in phrases:
                    if phrase in talk_lower:
                        scores[npc_type] = scores.get(npc_type, 0) + 1
        if scores:
            return max(scores, key=scores.get)
        return ""

    def _extract_options(self, npc_talk: str) -> list[dict]:
        """Extract numbered response options from NPC text.

        Handles formats like:
          "1. Buy  2. Sell  3. Cancel"
          "#1 Buy #2 Sell"
          "-- Buy -- Sell -- Cancel"
        """
        options = []

        # Try: number) text or number. text
        opt_pattern = re.findall(
            r'(?:^|\s+)(\d+)[\.\)]\s*([^\d]+?)(?=\s+\d+[\.\)]|\s*$)',
            npc_talk,
        )
        if opt_pattern:
            for idx, text in opt_pattern:
                options.append({"index": int(idx), "text": text.strip()})
            return options

        # Try: #number text
        opt_pattern2 = re.findall(r'#(\d+)\s*([^#]+?)(?=\s*#\d+|$)', npc_talk)
        if opt_pattern2:
            for idx, text in opt_pattern2:
                options.append({"index": int(idx), "text": text.strip()})
            return options

        # Try: -- text -- text format (visual menu)
        parts = re.split(r'[—–\-]+', npc_talk)
        if len(parts) > 2:  # At least separator + 2 options
            menu_texts = [p.strip() for p in parts if p.strip()]
            if len(menu_texts) >= 2:
                options = [{"index": i + 1, "text": t} for i, t in enumerate(menu_texts)]

        return options

    def _determine_stage(self, talk_lower: str, conv: ConversationState) -> str:
        """Determine conversation stage from dialogue."""
        if any(kw in talk_lower for kw in ["thank you", "reward", "completed", "here is"]):
            if conv.stage in ("selecting", "talking"):
                return "confirming"
            return "done"

        if any(kw in talk_lower for kw in ["what can i do", "welcome", "hello", "how may i help"]):
            if conv.turn_count <= 2:
                return "init"
            return "talking"

        if conv.options_seen:
            return "selecting"

        return "talking"

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove a bot's conversation state."""
        self._conversations.pop(bot_id, None)
