"""
LLM Conversation Engine — holds full conversations, not just one-liners.

A real player can hold a 10-minute conversation. They build rapport,
extract info, make friends. This module routes incoming chat to the LLM
for intelligent responses, maintaining conversation history per player.

Features:
- LLM-powered response generation via model_router
- Conversation memory (what was said, by whom, when)
- Personality system (different bots have different chat styles)
- Context awareness (reply to what was said, reference previous conversation)
- Template fallback when LLM is unavailable
- Thread-safe
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Personality Profiles ──────────────────────────────────────────────

PERSONALITIES: dict[str, dict[str, Any]] = {
    "friendly": {
        "description": "Warm, helpful, and chatty",
        "greeting_style": "Hey! How's it going?",
        "farewell_style": "Catch you later!",
        "response_style": "friendly and casual",
        "emoji_use": "moderate",
        "small_talk": True,
    },
    "stoic": {
        "description": "Quiet, efficient, minimal chatter",
        "greeting_style": "o/",
        "farewell_style": "o/",
        "response_style": "short and direct",
        "emoji_use": "rare",
        "small_talk": False,
    },
    "merchant": {
        "description": "Focused on trade, always looking for deals",
        "greeting_style": "Check out my wares!",
        "farewell_style": "Thanks for your business!",
        "response_style": "business-like, mentions prices and items",
        "emoji_use": "low",
        "small_talk": False,
    },
    "pvper": {
        "description": "Competitive, trash-talking, aggressive",
        "greeting_style": "You looking for a fight?",
        "farewell_style": "Run while you can.",
        "response_style": "aggressive and competitive",
        "emoji_use": "low",
        "small_talk": False,
    },
    "noob": {
        "description": "Pretends to be new, asks questions, seems innocent",
        "greeting_style": "Hi! I'm new here...",
        "farewell_style": "Bye! Thanks for the help!",
        "response_style": "naive and curious, asks questions back",
        "emoji_use": "high",
        "small_talk": True,
    },
}

DEFAULT_PERSONALITY = "friendly"

# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class Conversation:
    """A conversation thread with another player."""
    player_name: str
    messages: list[dict] = field(default_factory=list)
    context: str = ""
    started_at: float = 0.0
    last_message_at: float = 0.0
    active: bool = True
    topic: str = ""
    rapport: float = 0.5  # 0.0 (hostile) - 1.0 (best friends)
    interaction_count: int = 0


@dataclass(slots=True)
class ConversationEngine:
    """Holds natural conversations using LLM for response generation."""

    _lock: RLock = field(default_factory=RLock)
    _conversations: dict[str, Conversation] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "conversations": 0, "llm_responses": 0, "template_fallback": 0,
        "personality_shifts": 0, "context_references": 0,
    })
    _llm_call: Callable | None = None
    _personality: str = DEFAULT_PERSONALITY
    _bot_name: str = ""

    # ── Configuration ──

    def set_llm_call(self, fn: Callable) -> None:
        """Set the LLM call function (from model_router)."""
        self._llm_call = fn

    def set_personality(self, personality: str) -> None:
        """Set the bot's conversation personality."""
        if personality in PERSONALITIES:
            with self._lock:
                self._personality = personality
                self._stats["personality_shifts"] += 1
            logger.info("conversation_personality: set to '%s'", personality)
        else:
            logger.warning("conversation_personality: unknown '%s', using default", personality)

    def set_bot_name(self, name: str) -> None:
        """Set the bot's display name."""
        self._bot_name = name

    def get_personality(self) -> str:
        return self._personality

    # ── Conversation Management ──

    def start_conversation(self, player_name: str, initial_message: str = "") -> Conversation:
        """Start a new conversation thread."""
        conv = Conversation(
            player_name=player_name,
            started_at=time.time(),
            last_message_at=time.time(),
        )
        if initial_message:
            conv.messages.append({
                "speaker": player_name,
                "text": initial_message,
                "time": time.time(),
            })
        with self._lock:
            self._conversations[player_name] = conv
            self._stats["conversations"] += 1
        logger.info("conversation_started: with %s", player_name)
        return conv

    def receive_message(self, player_name: str, message: str) -> str | None:
        """Receive a chat message and generate a response.

        Uses LLM if available, falls back to template responses.
        Returns None if no response should be sent (bot is silent).
        """
        now = time.time()

        with self._lock:
            conv = self._conversations.get(player_name)
            if conv is None:
                conv = self.start_conversation(player_name, message)
            else:
                conv.messages.append({
                    "speaker": player_name,
                    "text": message,
                    "time": now,
                })
                conv.last_message_at = now
                conv.interaction_count += 1
                # Keep last 30 messages for context window
                if len(conv.messages) > 30:
                    conv.messages = conv.messages[-30:]

        # Try LLM first
        if self._llm_call is not None:
            try:
                response = self._generate_llm_response(player_name)
                if response:
                    with self._lock:
                        conv.messages.append({
                            "speaker": "me",
                            "text": response,
                            "time": time.time(),
                        })
                        self._stats["llm_responses"] += 1
                    return response
            except Exception as e:
                logger.warning("conversation_llm_failed: %s", e)

        # LLM unavailable — use template fallback
        self._stats["template_fallback"] += 1
        return self._template_response(message, player_name)

    def _generate_llm_response(self, player_name: str) -> str | None:
        """Generate a natural response using the LLM with full context."""
        if self._llm_call is None:
            return None

        with self._lock:
            conv = self._conversations.get(player_name)
            if conv is None:
                return None

            personality = PERSONALITIES.get(self._personality, PERSONALITIES[DEFAULT_PERSONALITY])
            bot_name = self._bot_name or "Player"

            # Build conversation history for LLM
            history_lines = []
            for m in conv.messages[-15:]:  # Last 15 messages
                speaker = "You" if m["speaker"] == "me" else m["speaker"]
                history_lines.append(f"{speaker}: {m['text']}")

            history = "\n".join(history_lines)

            # Detect topic from recent messages
            recent_text = " ".join(m["text"] for m in conv.messages[-5:])
            detected_topic = self._detect_topic(recent_text)

            # Build rapport-aware prompt
            rapport_level = conv.rapport
            if rapport_level < 0.3:
                rapport_desc = "stranger — be cautious and polite"
            elif rapport_level < 0.6:
                rapport_desc = "acquaintance — friendly but not too personal"
            else:
                rapport_desc = "friend — warm and familiar"

        prompt = (
            f"You are {bot_name}, a player in Ragnarok Online. "
            f"Your personality: {personality['description']}. "
            f"Response style: {personality['response_style']}. "
            f"Emoji use: {personality['emoji_use']}. "
            f"Relationship with {player_name}: {rapport_desc}. "
            f"Detected topic: {detected_topic}. "
            f"Keep responses short (1-3 lines), natural, and in-character. "
            f"Reference previous conversation when relevant. "
            f"Conversation so far:\n{history}\n"
            f"Your response:"
        )

        try:
            result = self._llm_call(prompt)
            if result and isinstance(result, str):
                response = result.strip()[:300]  # Cap at 300 chars
                # Update rapport based on response
                self._update_rapport(player_name, response)
                return response
        except Exception:
            pass
        return None

    def _detect_topic(self, text: str) -> str:
        """Detect conversation topic from text."""
        text_lower = text.lower()
        topics = {
            "trading/buying": ["buy", "sell", "price", "zeny", "cost", "trade", "shop", "item"],
            "farming/leveling": ["farm", "level", "exp", "grind", "hunt", "map", "monster"],
            "pvp/woe": ["pvp", "woe", "fight", "kill", "guild", "castle", "war"],
            "guild": ["guild", "alliance", "member", "recruit"],
            "gear/equipment": ["gear", "weapon", "armor", "card", "upgrade", "refine"],
            "quests": ["quest", "npc", "mission", "task"],
            "social": ["hello", "hi", "hey", "how", "what's up", "friend"],
            "help/advice": ["help", "advice", "how to", "where", "guide", "tip"],
        }
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        return "general chat"

    def _update_rapport(self, player_name: str, response: str) -> None:
        """Update rapport based on response tone."""
        with self._lock:
            conv = self._conversations.get(player_name)
            if conv is None:
                return
            # Positive signals
            positive = ["thanks", "ty", "np", "lol", "haha", "sure", "ok", "friend", "help"]
            negative = ["no", "stop", "leave", "go away", "shut", "idiot", "noob"]
            response_lower = response.lower()
            delta = 0.0
            for word in positive:
                if word in response_lower:
                    delta += 0.02
            for word in negative:
                if word in response_lower:
                    delta -= 0.05
            conv.rapport = max(0.0, min(1.0, conv.rapport + delta))

    def _template_response(self, message: str, player_name: str = "") -> str | None:
        """Fallback template responses when LLM is unavailable."""
        msg_lower = message.lower()

        greetings = ["hey", "hi", "hello", "sup", "yo", "whats up", "howdy", "greetings"]
        farewells = ["bye", "cya", "g2g", "gtg", "later", "see you", "goodbye"]
        questions = ["?", "who", "what", "where", "when", "why", "how", "can you", "do you"]
        thanks = ["thanks", "ty", "thx", "thank you", "appreciate"]
        trade = ["buy", "sell", "price", "how much", "trade", "shop", "item", "zeny"]
        pvp = ["pvp", "fight", "duel", "1v1", "pk"]

        if any(g in msg_lower for g in greetings):
            return random.choice(["hey o/", "sup", "hello!", "hi there", "how's it going?"])
        elif any(f in msg_lower for f in farewells):
            return random.choice(["cya later", "see you around", "later!", "take care"])
        elif any(t in msg_lower for t in trade):
            return random.choice(["what are you selling?", "got any good deals?", "check my shop maybe", "prices are crazy lately"])
        elif any(p in msg_lower for p in pvp):
            return random.choice(["not looking for a fight rn", "maybe later", "you wouldn't stand a chance", "lol ok"])
        elif any(q in msg_lower for q in questions):
            return random.choice(["idk man, just farming", "not sure tbh", "let me think...", "good question"])
        elif any(t in msg_lower for t in thanks):
            return random.choice(["np", "anytime", "no problem", "happy to help"])
        else:
            return random.choice(["yeah", "for real", "lol", "true", "same"])

    # ── Context & Stats ──

    def get_conversation_context(self) -> str:
        """Get formatted conversation context for LLM prompts."""
        with self._lock:
            active = [c for c in self._conversations.values() if c.active]
            if not active:
                return ""

            lines = ["── Active Conversations ──"]
            for conv in active[-5:]:  # Last 5
                last = conv.messages[-1] if conv.messages else {}
                lines.append(
                    f"  With {conv.player_name}: "
                    f"'{last.get('text', '')[:60]}' "
                    f"({len(conv.messages)} msgs, "
                    f"rapport={conv.rapport:.1f})"
                )
            return "\n".join(lines)

    def expire_old(self, max_age_seconds: int = 600) -> None:
        """Mark conversations older than max_age as inactive."""
        now = time.time()
        with self._lock:
            for conv in list(self._conversations.values()):
                if now - conv.last_message_at > max_age_seconds:
                    conv.active = False

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def get_conversation_summary(self) -> str:
        """Get a summary of all conversations for persistence."""
        with self._lock:
            summary = []
            for conv in self._conversations.values():
                summary.append({
                    "player_name": conv.player_name,
                    "message_count": len(conv.messages),
                    "rapport": conv.rapport,
                    "active": conv.active,
                    "last_message_at": conv.last_message_at,
                    "topic": conv.topic,
                })
            return json.dumps(summary)


# Global instance
_conversation: ConversationEngine | None = None
_conversation_lock = RLock()


def get_conversation_engine() -> ConversationEngine:
    global _conversation
    with _conversation_lock:
        if _conversation is None:
            _conversation = ConversationEngine()
        return _conversation
