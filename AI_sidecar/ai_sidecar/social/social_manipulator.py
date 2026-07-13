"""
Social manipulation system — proactive social intelligence for deception and influence.

A top player doesn't just avoid social interaction. They manipulate.
They know when to talk, what to say, who to befriend, who to avoid,
when to lie, and when to tell the truth. They spread rumors, make
alliances, and gather intelligence through conversation.

This module replaces the old "stealth-first, silence by default"
approach with a proactive social strategy engine.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlayerProfile:
    """Profile of another player we've interacted with."""
    name: str
    guild: str = ""
    level: int = 0
    class_name: str = ""
    trust_score: float = 0.0  # -1.0 (enemy) to 1.0 (ally)
    first_seen: float = 0.0
    last_seen: float = 0.0
    interaction_count: int = 0
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # "gm", "pker", "trader", "ally", "rival"


@dataclass
class SocialScript:
    """A pre-written social script for common situations."""
    trigger: str  # greeting, farewell, trade_offer, warning, rumor
    templates: list[str] = field(default_factory=list)
    cooldown_seconds: int = 60
    risk_level: str = "low"  # low, medium, high


@dataclass(slots=True)
class SocialManipulator:
    """Proactive social intelligence for deception and influence."""
    
    _lock: RLock = field(default_factory=RLock)
    _profiles: dict[str, PlayerProfile] = field(default_factory=dict)
    _scripts: dict[str, SocialScript] = field(default_factory=dict)
    _reputation: float = 0.0  # Overall server reputation
    _stats: dict[str, int] = field(default_factory=lambda: {
        "interactions": 0, "deceptions": 0, "alliances": 0, "intel_gathered": 0,
    })
    _last_chat_time: float = 0.0
    _chat_cooldown: float = 5.0  # Don't spam chat
    
    def __post_init__(self):
        self._init_scripts()
    
    def _init_scripts(self) -> None:
        """Initialize social scripts for common situations."""
        self._scripts = {
            "greeting": SocialScript(
                trigger="greeting",
                templates=[
                    "hey {name}",
                    "sup {name}",
                    "hello {name}",
                    "yo {name}",
                    "hi {name}",
                ],
                cooldown_seconds=300,
                risk_level="low",
            ),
            "farewell": SocialScript(
                trigger="farewell",
                templates=[
                    "gotta go, cya",
                    "brb",
                    "later {name}",
                    "gtg",
                ],
                cooldown_seconds=60,
                risk_level="low",
            ),
            "trade_offer": SocialScript(
                trigger="trade_offer",
                templates=[
                    "selling {item} cheap, {price}z",
                    "buying {item} {price}z each",
                    "WTS {item} {price}z",
                    "WTB {item} paying {price}z",
                ],
                cooldown_seconds=120,
                risk_level="medium",
            ),
            "warning": SocialScript(
                trigger="warning",
                templates=[
                    "watch out, {danger} at {map}",
                    "saw {danger} near {map}, be careful",
                    "heard {danger} is hunting at {map}",
                ],
                cooldown_seconds=600,
                risk_level="medium",
            ),
            "rumor": SocialScript(
                trigger="rumor",
                templates=[
                    "heard {rumor}",
                    "someone said {rumor}",
                    "idk if true but {rumor}",
                ],
                cooldown_seconds=900,
                risk_level="high",
            ),
            "compliment": SocialScript(
                trigger="compliment",
                templates=[
                    "nice {item}",
                    "gz on {achievement}",
                    "sick {item}",
                ],
                cooldown_seconds=600,
                risk_level="low",
            ),
            "help_request": SocialScript(
                trigger="help_request",
                templates=[
                    "can anyone help with {quest}?",
                    "need {item}, anyone have?",
                    "looking for party at {map}",
                ],
                cooldown_seconds=300,
                risk_level="low",
            ),
        }
    
    def record_interaction(self, player_name: str, interaction_type: str, detail: str = "") -> None:
        """Record an interaction with another player."""
        with self._lock:
            now = time.time()
            profile = self._profiles.setdefault(player_name, PlayerProfile(name=player_name))
            profile.last_seen = now
            if profile.first_seen == 0:
                profile.first_seen = now
            profile.interaction_count += 1
            if detail:
                profile.notes.append(f"[{interaction_type}] {detail}")
            self._stats["interactions"] += 1
    
    def update_trust(self, player_name: str, delta: float) -> None:
        """Update trust score for a player."""
        with self._lock:
            profile = self._profiles.get(player_name)
            if profile:
                profile.trust_score = max(-1.0, min(1.0, profile.trust_score + delta))
    
    def tag_player(self, player_name: str, tag: str) -> None:
        """Tag a player (gm, pker, trader, ally, rival)."""
        with self._lock:
            profile = self._profiles.get(player_name)
            if profile and tag not in profile.tags:
                profile.tags.append(tag)
    
    def get_player_context(self, player_name: str) -> str:
        """Get formatted context about a player."""
        with self._lock:
            profile = self._profiles.get(player_name)
            if not profile:
                return f"Unknown player: {player_name}"
            
            trust_label = "ally" if profile.trust_score > 0.3 else \
                         "neutral" if profile.trust_score > -0.3 else "enemy"
            
            return (
                f"Player: {profile.name} | Guild: {profile.guild} | "
                f"Trust: {trust_label} ({profile.trust_score:.1f}) | "
                f"Tags: {', '.join(profile.tags) or 'none'} | "
                f"Interactions: {profile.interaction_count}"
            )
    
    def generate_chat(self, trigger: str, **kwargs) -> str | None:
        """Generate a chat message for a social trigger."""
        now = time.time()
        if now - self._last_chat_time < self._chat_cooldown:
            return None
        
        script = self._scripts.get(trigger)
        if not script or not script.templates:
            return None
        
        template = random.choice(script.templates)
        message = template.format(**kwargs)
        
        self._last_chat_time = now
        with self._lock:
            self._stats["interactions"] += 1
            if script.risk_level == "high":
                self._stats["deceptions"] += 1
        
        return message
    
    def should_speak(self, context: str = "") -> bool:
        """Decide whether to speak based on context and safety."""
        from ai_sidecar.timing.timing_awareness import get_timing
        timing = get_timing()
        safety = timing.get_safety_rating()
        
        # More likely to speak during safe hours
        if safety > 0.7:
            return random.random() < 0.3  # 30% chance
        elif safety > 0.4:
            return random.random() < 0.1  # 10% chance
        else:
            return False  # Stay silent during dangerous hours
    
    def get_social_context(self) -> str:
        """Get formatted social context for LLM prompts."""
        with self._lock:
            lines = ["── Social Context ──"]
            lines.append(f"  Reputation: {self._reputation:.1f}")
            lines.append(f"  Known players: {len(self._profiles)}")
            
            # Show recent interactions
            recent = sorted(
                self._profiles.values(),
                key=lambda p: -p.last_seen
            )[:5]
            for p in recent:
                lines.append(f"  {self.get_player_context(p.name)}")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_social: SocialManipulator | None = None
_social_lock = RLock()


def get_social() -> SocialManipulator:
    """Get or create the global social manipulator."""
    global _social
    with _social_lock:
        if _social is None:
            _social = SocialManipulator()
        return _social
