"""
LLM Conversation Engine — holds full conversations, not just one-liners.

A real player can hold a 10-minute conversation. They build rapport,
extract info, make friends. This module routes incoming chat to the LLM
for intelligent responses, maintaining conversation history per player.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """A conversation thread with another player."""
    player_name: str
    messages: list[dict] = field(default_factory=list)
    context: str = ""
    started_at: float = 0.0
    last_message_at: float = 0.0
    active: bool = True


@dataclass(slots=True)
class ConversationEngine:
    """Holds natural conversations using LLM for response generation."""
    
    _lock: RLock = field(default_factory=RLock)
    _conversations: dict[str, Conversation] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "conversations": 0, "llm_responses": 0, "template_fallback": 0,
    })
    _llm_call: Callable | None = None
    
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
                # Keep last 20 messages for context window
                if len(conv.messages) > 20:
                    conv.messages = conv.messages[-20:]
        
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
        return self._template_response(message)
    
    def _generate_llm_response(self, player_name: str) -> str | None:
        """Generate a natural response using the LLM."""
        if self._llm_call is None:
            return None
        
        with self._lock:
            conv = self._conversations.get(player_name)
            if conv is None:
                return None
            
            # Build conversation history for LLM
            history = "\n".join(
                f"{m['speaker']}: {m['text']}"
                for m in conv.messages[-10:]  # Last 10 messages
            )
        
        prompt = (
            f"You are {player_name}'s conversation partner in Ragnarok Online. "
            f"Respond naturally as a player. Keep responses short (1-2 lines). "
            f"Be friendly but not overly helpful. This is a game. "
            f"Conversation so far:\n{history}\n"
            f"Your response:"
        )
        
        try:
            result = self._llm_call(prompt)
            if result and isinstance(result, str):
                return result.strip()[:200]  # Cap at 200 chars
        except Exception:
            pass
        return None
    
    def _template_response(self, message: str) -> str | None:
        """Fallback template responses when LLM is unavailable."""
        msg_lower = message.lower()
        
        greetings = ["hey", "hi", "hello", "sup", "yo", "whats up", "howdy"]
        farewells = ["bye", "cya", "g2g", "gtg", "later", "see you"]
        questions = ["?", "who", "what", "where", "when", "why", "how", "can you", "do you"]
        thanks = ["thanks", "ty", "thx", "thank you", "appreciate"]
        
        if any(g in msg_lower for g in greetings):
            return "hey o/" if msg_lower.count(greetings[0]) > 0 else "sup"
        elif any(f in msg_lower for f in farewells):
            return "cya later"
        elif any(q in msg_lower for q in questions):
            return "idk man, just farming"
        elif any(t in msg_lower for t in thanks):
            return "np"
        else:
            return "yeah"  # Generic acknowledgment
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for LLM prompts."""
        with self._lock:
            active = [c for c in self._conversations.values() if c.active]
            if not active:
                return ""
            
            lines = ["── Active Conversations ──"]
            for conv in active[-3:]:  # Last 3
                last = conv.messages[-1] if conv.messages else {}
                lines.append(f"  With {conv.player_name}: "
                           f"'{last.get('text', '')[:50]}' "
                           f"({len(conv.messages)} msgs)")
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


# Global instance
_conversation: ConversationEngine | None = None
_conversation_lock = RLock()


def get_conversation_engine() -> ConversationEngine:
    global _conversation
    with _conversation_lock:
        if _conversation is None:
            _conversation = ConversationEngine()
        return _conversation
