"""
Social intelligence v2 — stealth-first player interaction handler.

Default mode: ABSOLUTE SILENCE. Never speak unless explicitly decided.
Handles: buff requests, trade, party invites, whispers, public chat.
AI decides to ignore/accept/reject based on context and risk assessment.
Responses include realistic human imperfections: typos, delayed typing, casual language.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Realistic human-like response templates
RESPONSES = {
    "greeting": ["hey", "hi", "yo", "sup", "hello"],
    "farewell": ["cya", "later", "gtg", "bye"],
    "buff_accept": ["sure", "ok", "np", "k", "one sec"],
    "buff_decline": ["sry no sp", "cant rn", "maybe later", "afk"],
    "trade_accept": ["sure wat u got", "ok show me", "k"],
    "trade_decline": ["no thx", "not selling", "keeping it", "sry"],
    "party_accept": ["sure", "ok inv", "k"],
    "party_decline": ["busy rn", "sry in party", "maybe later"],
    "unknown": ["?", "huh", "what", "??", "not sure wat u mean"],
    "afk": ["brb", "afk sec", "one min"],
}

# Typing delay simulation (seconds)
MIN_TYPING_DELAY = 1.0
MAX_TYPING_DELAY = 4.0

# Typo probability
TYPO_PROBABILITY = 0.15


def _add_typo(text: str) -> str:
    """Add a realistic typo to text."""
    if random.random() > TYPO_PROBABILITY:
        return text
    if len(text) < 3:
        return text
    idx = random.randint(0, len(text) - 1)
    # Swap adjacent characters
    if idx < len(text) - 1:
        chars = list(text)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    return text


def _humanize(text: str) -> str:
    """Add human-like imperfections to text."""
    text = _add_typo(text)
    # Random capitalization
    if random.random() < 0.1:
        text = text.lower()
    # Random punctuation
    if random.random() < 0.2:
        text += random.choice(["", ".", "!", "..", "..."])
    return text


@dataclass(slots=True)
class SocialIntelligenceV2:
    """Stealth-first social interaction handler."""
    
    _lock: RLock = field(default_factory=RLock)
    _whisper_history: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    _reputation: dict[str, float] = field(default_factory=dict)  # player -> reputation score
    _blocked_players: set[str] = field(default_factory=set)
    _last_response_time: dict[str, float] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {"greetings": 0, "replies": 0, "ignored": 0, "blocked": 0})
    
    def process_public_chat(self, sender: str, message: str, context: dict[str, Any]) -> str | None:
        """Process a public chat message. Returns a response or None (stay silent)."""
        msg_lower = message.lower()
        
        # Check if we're being addressed directly
        addressed = any(name in msg_lower for name in ["bot", "kicap", sender[:4]])
        if not addressed:
            return None  # Not talking to us — stay silent
        
        return self._handle_interaction(sender, message, context, "public")
    
    def process_whisper(self, sender: str, message: str, context: dict[str, Any]) -> str | None:
        """Process a whisper. Returns a response or None."""
        return self._handle_interaction(sender, message, context, "whisper")
    
    def _handle_interaction(self, sender: str, message: str, context: dict[str, Any], channel: str) -> str | None:
        """Handle any player interaction. Returns response or None."""
        msg_lower = message.lower()
        
        with self._lock:
            # Track history
            self._whisper_history[sender].append({
                "message": message,
                "channel": channel,
                "timestamp": time.time(),
            })
            self._whisper_history[sender] = self._whisper_history[sender][-20:]
        
        # Check if blocked
        if sender in self._blocked_players:
            with self._lock:
                self._stats["ignored"] += 1
            return None
        
        # Check cooldown (don't spam)
        last_time = self._last_response_time.get(sender, 0)
        if time.time() - last_time < 10:
            return None
        
        # Check reputation
        rep = self._reputation.get(sender, 0.5)
        if rep < 0.2:
            with self._lock:
                self._stats["ignored"] += 1
            return None
        
        # Determine interaction type
        response = None
        
        # Buff request
        if any(kw in msg_lower for kw in ["buff", "heal", "bless", "agi", "boost"]):
            if rep > 0.3 and random.random() < 0.7:
                response = random.choice(RESPONSES["buff_accept"])
            else:
                response = random.choice(RESPONSES["buff_decline"])
        
        # Trade request
        elif any(kw in msg_lower for kw in ["buy", "sell", "trade", "price", "how much"]):
            if rep > 0.5 and random.random() < 0.5:
                response = random.choice(RESPONSES["trade_accept"])
            else:
                response = random.choice(RESPONSES["trade_decline"])
        
        # Party invite
        elif any(kw in msg_lower for kw in ["party", "pt", "invite", "join"]):
            if rep > 0.4 and random.random() < 0.6:
                response = random.choice(RESPONSES["party_accept"])
            else:
                response = random.choice(RESPONSES["party_decline"])
        
        # Greeting
        elif any(g in msg_lower for g in ["hello", "hi", "hey", "sup", "yo"]):
            response = random.choice(RESPONSES["greeting"])
        
        # Farewell
        elif any(f in msg_lower for f in ["bye", "cya", "later", "gtg"]):
            response = random.choice(RESPONSES["farewell"])
        
        # Unknown
        else:
            if random.random() < 0.3:  # 30% chance to respond to unknown
                response = random.choice(RESPONSES["unknown"])
        
        if response is not None:
            response = _humanize(response)
            with self._lock:
                self._last_response_time[sender] = time.time()
                self._stats["replies"] += 1
                # Simulate typing delay
                delay = random.uniform(MIN_TYPING_DELAY, MAX_TYPING_DELAY)
                logger.info("social_response: to=%s msg=%s delay=%.1fs", sender, response, delay)
        
        return response
    
    def update_reputation(self, player: str, delta: float) -> None:
        """Update a player's reputation score."""
        with self._lock:
            current = self._reputation.get(player, 0.5)
            self._reputation[player] = max(0.0, min(1.0, current + delta))
    
    def block_player(self, player: str) -> None:
        """Block a player (stop responding to them)."""
        with self._lock:
            self._blocked_players.add(player)
            self._stats["blocked"] += 1
    
    def get_greeting(self) -> str:
        """Get a greeting for login."""
        with self._lock:
            self._stats["greetings"] += 1
        return _humanize(random.choice(RESPONSES["greeting"]))
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
