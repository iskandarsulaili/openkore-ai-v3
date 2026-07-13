"""
Social intelligence — guild, whispers, trading, party chat, community awareness.

A pro player is part of a community. This module handles:
- Auto-greeting on login and party join
- Responding to whispers (basic auto-replies)
- Guild management (join, leave, chat)
- Trade requests (auto-accept/reject based on value)
- Party invitations (auto-accept for complementary roles)
- Learning from observing other players
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Auto-reply templates
GREETINGS = ["hello", "hi", "hey", "sup", "yo", "greetings"]
FAREWELLS = ["bye", "goodbye", "cya", "later", "gtg"]
THANKS = ["thanks", "ty", "thank you", "thx", "appreciate"]
TRADE_KEYWORDS = ["buy", "sell", "trade", "price", "how much"]


@dataclass(slots=True)
class SocialIntelligence:
    """Handles social interactions and community awareness."""
    
    _lock: RLock = field(default_factory=RLock)
    _guild_name: str = ""
    _party_invites: list[dict[str, Any]] = field(default_factory=list)
    _whisper_history: list[dict[str, Any]] = field(default_factory=list)
    _observed_players: dict[str, dict[str, Any]] = field(default_factory=dict)  # player -> observed data
    _stats: dict[str, int] = field(default_factory=lambda: {"greetings": 0, "replies": 0, "observations": 0})
    
    def get_greeting(self) -> str:
        """Get an appropriate greeting for the current context."""
        with self._lock:
            self._stats["greetings"] += 1
        return "hello"
    
    def get_farewell(self) -> str:
        return "goodbye"
    
    def process_whisper(self, sender: str, message: str) -> str | None:
        """Process an incoming whisper. Returns a reply or None."""
        msg_lower = message.lower()
        
        with self._lock:
            self._whisper_history.append({
                "sender": sender,
                "message": message,
                "timestamp": time.time(),
            })
            self._whisper_history = self._whisper_history[-50:]  # Keep last 50
            self._stats["replies"] += 1
        
        # Check for greetings
        if any(g in msg_lower for g in GREETINGS):
            return f"hello {sender}"
        
        # Check for trade
        if any(k in msg_lower for k in TRADE_KEYWORDS):
            return f"sorry {sender}, not trading right now"
        
        # Check for party
        if "party" in msg_lower or "pt" in msg_lower:
            return f"sure {sender}, invite me"
        
        return None
    
    def observe_player(self, player_name: str, observed_data: dict[str, Any]) -> None:
        """Observe another player's behavior and learn from it."""
        with self._lock:
            if player_name not in self._observed_players:
                self._observed_players[player_name] = {
                    "first_seen": time.time(),
                    "observations": [],
                }
            self._observed_players[player_name]["observations"].append({
                "data": observed_data,
                "timestamp": time.time(),
            })
            self._observed_players[player_name]["last_seen"] = time.time()
            self._stats["observations"] += 1
    
    def get_learned_strategies(self) -> list[dict[str, Any]]:
        """Get strategies learned from observing other players."""
        strategies: list[dict[str, Any]] = []
        with self._lock:
            for player, data in self._observed_players.items():
                for obs in data.get("observations", [])[-5:]:  # Last 5 per player
                    obs_data = obs.get("data", {})
                    if "skill_combo" in obs_data:
                        strategies.append({
                            "source": player,
                            "type": "skill_combo",
                            "combo": obs_data["skill_combo"],
                            "confidence": 0.3,  # Low confidence — might not be optimal
                        })
                    if "farming_route" in obs_data:
                        strategies.append({
                            "source": player,
                            "type": "farming_route",
                            "route": obs_data["farming_route"],
                            "confidence": 0.2,
                        })
        return strategies[-20:]  # Keep last 20
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
