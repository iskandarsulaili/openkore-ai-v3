"""
Server personality profiles — adapts bot behavior to each server's culture.

Every server has different rules, culture, and politics. Some allow multi-client,
some ban it. Some have corrupt GMs, some are strict. Some have strong player
economies, some are dominated by a single guild.

This module stores server-specific profiles and adjusts behavior accordingly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServerProfile:
    """Profile for a specific server's personality."""
    
    # Server identity
    name: str
    ip: str = ""
    port: int = 6900
    
    # Rules
    allows_multi_client: bool = True
    allows_botting: bool = False  # How tolerated are bots
    gm_strictness: str = "moderate"  # lenient, moderate, strict, brutal
    has_corrupt_gms: bool = False
    
    # Economy
    economy_type: str = "free"  # free, controlled, inflated, deflated
    player_count: str = "medium"  # low, medium, high, massive
    
    # Culture
    pvp_focus: str = "low"  # low, medium, high
    woe_importance: str = "medium"  # low, medium, high
    community_friendliness: str = "friendly"  # toxic, neutral, friendly, very_friendly
    language: str = "english"
    
    # Behavior adjustments
    farm_aggressiveness: float = 0.5  # 0.0 (cautious) to 1.0 (aggressive)
    social_frequency: float = 0.3  # 0.0 (silent) to 1.0 (chatty)
    risk_tolerance: float = 0.3  # 0.0 (safety first) to 1.0 (high risk)
    anti_detection_level: str = "moderate"  # minimal, moderate, paranoid
    
    # Observations
    observations: list[str] = field(default_factory=list)
    last_updated: float = 0.0


@dataclass(slots=True)
class ServerPersonalityEngine:
    """Adapts bot behavior to each server's unique culture."""
    
    _lock: RLock = field(default_factory=RLock)
    _profiles: dict[str, ServerProfile] = field(default_factory=dict)
    _current_server: str = ""
    _stats: dict[str, int] = field(default_factory=lambda: {"switches": 0, "observations": 0})
    
    def get_profile(self, server_name: str = "") -> ServerProfile:
        """Get or create a profile for a server."""
        name = server_name or self._current_server or "default"
        with self._lock:
            if name not in self._profiles:
                self._profiles[name] = ServerProfile(name=name)
                logger.info("server_personality_created: %s", name)
            return self._profiles[name]
    
    def set_current_server(self, name: str, ip: str = "", port: int = 6900) -> None:
        """Switch to a different server profile."""
        with self._lock:
            profile = self.get_profile(name)
            if ip:
                profile.ip = ip
            if port:
                profile.port = port
            self._current_server = name
            self._stats["switches"] += 1
            logger.info("server_personality_switched: %s", name)
    
    def observe(self, observation: str) -> None:
        """Record an observation about the current server."""
        with self._lock:
            profile = self.get_profile()
            profile.observations.append(f"[{time.strftime('%Y-%m-%d %H:%M')}] {observation}")
            if len(profile.observations) > 100:
                profile.observations = profile.observations[-100:]
            self._stats["observations"] += 1
    
    def adjust_behavior(self, trait: str, value: Any) -> None:
        """Adjust a specific behavioral trait for the current server."""
        with self._lock:
            profile = self.get_profile()
            if hasattr(profile, trait):
                setattr(profile, trait, value)
                logger.info("server_personality_adjusted: %s=%s", trait, value)
    
    def get_behavior_context(self) -> str:
        """Get formatted behavior settings for LLM prompts."""
        with self._lock:
            profile = self.get_profile()
            lines = [f"── Server Personality: {profile.name} ──"]
            lines.append(f"  Rules: multi_client={profile.allows_multi_client} botting_tolerance={'low' if not profile.allows_botting else 'medium'}")
            lines.append(f"  GMs: {profile.gm_strictness} | Corruption: {'yes' if profile.has_corrupt_gms else 'no'}")
            lines.append(f"  Economy: {profile.economy_type} | Players: {profile.player_count}")
            lines.append(f"  PvP: {profile.pvp_focus} | WoE: {profile.woe_importance}")
            lines.append(f"  Community: {profile.community_friendliness}")
            lines.append(f"  Behavior: farm={profile.farm_aggressiveness:.1f} social={profile.social_frequency:.1f} risk={profile.risk_tolerance:.1f}")
            lines.append(f"  Anti-detection: {profile.anti_detection_level}")
            if profile.observations:
                lines.append("  Recent observations:")
                for obs in profile.observations[-3:]:
                    lines.append(f"    {obs}")
            return "\n".join(lines)
    
    def get_farm_aggressiveness(self) -> float:
        with self._lock:
            return self.get_profile().farm_aggressiveness
    
    def get_social_frequency(self) -> float:
        with self._lock:
            return self.get_profile().social_frequency
    
    def get_risk_tolerance(self) -> float:
        with self._lock:
            return self.get_profile().risk_tolerance
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_server: ServerPersonalityEngine | None = None
_server_lock = RLock()


def get_server_personality() -> ServerPersonalityEngine:
    global _server
    with _server_lock:
        if _server is None:
            _server = ServerPersonalityEngine()
        return _server
