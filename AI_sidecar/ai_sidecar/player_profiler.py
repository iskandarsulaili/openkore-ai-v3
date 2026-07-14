"""
Player Profiling System — categorizes every player seen, tracks behavior
patterns, and adjusts bot behavior accordingly.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlayerProfile:
    """A profile for a player observed in the game."""
    name: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    sighting_count: int = 0
    category: str = "unknown"  # farmer, pker, gm, trader, party_member, competitor
    threat_level: int = 0  # 0-10, 10=most threatening
    trust_score: int = 50  # 0-100, 100=most trusted
    is_gm: bool = False
    is_pker: bool = False
    is_bot: bool = False
    is_friend: bool = False
    job_class: str = ""
    base_level: int = 0
    guild_name: str = ""
    last_map: str = ""
    notes: str = ""


class PlayerProfiler:
    """Profiles every player seen and adjusts behavior accordingly."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._profiles: dict[str, PlayerProfile] = {}
        self._max_profiles: int = 1000
        self._suspicious_names: list[str] = [
            "GM", "GameMaster", "Admin", "Support", "Moderator",
            "Helper", "Staff", "Dev", "Developer",
        ]

    # ── Public API ──

    def observe_player(self, name: str, job_class: str = "", base_level: int = 0,
                       guild: str = "", map_name: str = "", is_attacking_us: bool = False,
                       is_following_us: bool = False, distance: float = 0.0) -> None:
        """Observe a player and update their profile."""
        with self._lock:
            now = time.time()
            if name not in self._profiles:
                self._profiles[name] = PlayerProfile(
                    name=name, first_seen=now, last_seen=now,
                    sighting_count=1, job_class=job_class,
                    base_level=base_level, guild_name=guild,
                    last_map=map_name,
                )
                self._classify_player(name)
            else:
                profile = self._profiles[name]
                profile.last_seen = now
                profile.sighting_count += 1
                if job_class:
                    profile.job_class = job_class
                if base_level:
                    profile.base_level = base_level
                if guild:
                    profile.guild_name = guild
                profile.last_map = map_name

                # Update threat level
                if is_attacking_us:
                    profile.threat_level = min(10, profile.threat_level + 2)
                    profile.is_pker = True
                    profile.trust_score = max(0, profile.trust_score - 10)
                if is_following_us and distance < 10:
                    profile.threat_level = min(10, profile.threat_level + 1)
                    profile.trust_score = max(0, profile.trust_score - 5)

            # Enforce max profiles
            if len(self._profiles) > self._max_profiles:
                oldest = min(self._profiles.items(), key=lambda x: x[1].last_seen)
                del self._profiles[oldest[0]]

    def _classify_player(self, name: str) -> None:
        """Classify a player into a category."""
        profile = self._profiles.get(name)
        if not profile:
            return

        # Check for GM names
        for suspicious in self._suspicious_names:
            if suspicious.lower() in name.lower():
                profile.category = "gm"
                profile.is_gm = True
                profile.threat_level = 10
                profile.trust_score = 0
                return

        # Check for bot-like names (random letters/numbers)
        import re
        if re.search(r'[a-z]{5,}\d{3,}', name, re.IGNORECASE):
            profile.is_bot = True
            profile.category = "bot"

    def get_player(self, name: str) -> PlayerProfile | None:
        with self._lock:
            return self._profiles.get(name)

    def get_threats(self, min_threat: int = 5) -> list[PlayerProfile]:
        """Get players that pose a threat."""
        with self._lock:
            return [p for p in self._profiles.values() if p.threat_level >= min_threat]

    def get_friends(self) -> list[PlayerProfile]:
        with self._lock:
            return [p for p in self._profiles.values() if p.is_friend]

    def get_gms(self) -> list[PlayerProfile]:
        with self._lock:
            return [p for p in self._profiles.values() if p.is_gm]

    def get_competitors(self, map_name: str) -> list[PlayerProfile]:
        """Get players competing for kills on the same map."""
        with self._lock:
            return [p for p in self._profiles.values() if p.last_map == map_name and p.category == "farmer"]

    def mark_friend(self, name: str) -> None:
        with self._lock:
            profile = self._profiles.get(name)
            if profile:
                profile.is_friend = True
                profile.trust_score = 100
                profile.threat_level = 0

    def get_player_summary(self) -> str:
        with self._lock:
            lines = [f"── Player Profiler ──"]
            lines.append(f"Players tracked: {len(self._profiles)}")
            threats = self.get_threats()
            if threats:
                lines.append(f"Threats: {', '.join(f'{p.name}({p.threat_level})' for p in threats[:5])}")
            gms = self.get_gms()
            if gms:
                lines.append(f"GMs detected: {', '.join(p.name for p in gms)}")
            friends = self.get_friends()
            if friends:
                lines.append(f"Friends: {', '.join(p.name for p in friends)}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._profiles.clear()


# ── Global Singleton ──

_player_profiler: PlayerProfiler | None = None
_player_profiler_lock = RLock()


def get_player_profiler() -> PlayerProfiler:
    global _player_profiler
    with _player_profiler_lock:
        if _player_profiler is None:
            _player_profiler = PlayerProfiler()
        return _player_profiler
