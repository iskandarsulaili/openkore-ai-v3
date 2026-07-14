"""
Anti-Killsteal Module — handles competing players and secures kills.

A pro player adjusts when others are nearby: they secure kills with finishers,
switch targets to avoid KS wars, or leave crowded maps entirely.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NearbyPlayer:
    """A nearby player character."""
    name: str
    distance: float
    job_class: str = "unknown"
    is_party_member: bool = False
    is_guild_member: bool = False
    is_attacking_same_target: bool = False
    is_aggressive: bool = False
    last_seen: float = 0.0
    encounter_count: int = 0


@dataclass
class KillstealSituation:
    """Current killsteal situation assessment."""
    has_competition: bool = False
    competitor_count: int = 0
    nearest_competitor_distance: float = 999.0
    target_hp_pct: float = 1.0
    target_distance: float = 0.0
    my_dps_estimate: float = 0.0
    competitor_dps_estimate: float = 0.0
    can_secure_kill: bool = False
    should_switch_target: bool = False
    should_leave_map: bool = False
    recommended_action: str = "continue_farming"
    confidence: float = 0.0


class AntiKillsteal:
    """Manages competition with other players for kills and resources."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._nearby_players: dict[str, NearbyPlayer] = {}
        self._ks_events: list[dict] = []
        self._crowded_maps: dict[str, float] = {}  # map_name -> last_crowded_time
        self._blacklisted_players: set[str] = set()
        self._max_ks_events: int = 100
        self._crowded_timeout_s: float = 300.0  # 5 min before re-checking a crowded map

    # ── Public API ──

    def update_nearby_players(self, players: list[dict]) -> None:
        """Update the list of nearby players."""
        with self._lock:
            now = time.time()
            seen: set[str] = set()
            for p in players:
                name = str(p.get("name", ""))
                if not name:
                    continue
                seen.add(name)
                if name in self._blacklisted_players:
                    continue
                if name in self._nearby_players:
                    np = self._nearby_players[name]
                    np.distance = float(p.get("distance", 999))
                    np.job_class = str(p.get("job_class", "unknown"))
                    np.is_attacking_same_target = bool(p.get("attacking_same_target", False))
                    np.last_seen = now
                    np.encounter_count += 1
                else:
                    self._nearby_players[name] = NearbyPlayer(
                        name=name,
                        distance=float(p.get("distance", 999)),
                        job_class=str(p.get("job_class", "unknown")),
                        is_party_member=bool(p.get("is_party_member", False)),
                        is_guild_member=bool(p.get("is_guild_member", False)),
                        last_seen=now,
                        encounter_count=1,
                    )

            # Remove stale players (not seen in 30s)
            stale = [n for n, p in self._nearby_players.items() if n not in seen and now - p.last_seen > 30]
            for n in stale:
                del self._nearby_players[n]

    def assess_situation(
        self,
        target_hp_pct: float = 1.0,
        target_distance: float = 0.0,
        my_dps: float = 100.0,
        my_hp_pct: float = 1.0,
        map_name: str = "",
    ) -> KillstealSituation:
        """Assess the current killsteal situation."""
        with self._lock:
            now = time.time()
            competitors = [p for p in self._nearby_players.values() if not p.is_party_member and now - p.last_seen < 10]

            situation = KillstealSituation(
                has_competition=len(competitors) > 0,
                competitor_count=len(competitors),
                target_hp_pct=target_hp_pct,
                target_distance=target_distance,
                my_dps_estimate=my_dps,
            )

            if competitors:
                situation.nearest_competitor_distance = min(p.distance for p in competitors)

                # Check if map is known to be crowded
                if map_name and map_name in self._crowded_maps:
                    if now - self._crowded_maps[map_name] < self._crowded_timeout_s:
                        situation.should_leave_map = True
                        situation.recommended_action = "leave_map"
                        situation.confidence = 0.7
                        return situation

                # Can we secure the kill?
                if target_hp_pct < 0.3 and target_distance < 10:
                    situation.can_secure_kill = True
                    situation.recommended_action = "use_finisher"
                    situation.confidence = 0.8
                    return situation

                # Should we switch target?
                if target_hp_pct > 0.5 and situation.nearest_competitor_distance < 5:
                    situation.should_switch_target = True
                    situation.recommended_action = "switch_target"
                    situation.confidence = 0.6
                    return situation

                # Many competitors — consider leaving
                if len(competitors) >= 3:
                    situation.should_leave_map = True
                    situation.recommended_action = "leave_map"
                    situation.confidence = 0.5
                    return situation

            situation.confidence = 0.9
            return situation

    def record_ks_event(self, monster_name: str, competitor_name: str, map_name: str) -> None:
        """Record a killsteal event."""
        with self._lock:
            self._ks_events.append({
                "monster": monster_name,
                "competitor": competitor_name,
                "map": map_name,
                "timestamp": time.time(),
            })
            if len(self._ks_events) > self._max_ks_events:
                self._ks_events.pop(0)

            # If same competitor KS us 3+ times, blacklist
            ks_count = sum(1 for e in self._ks_events if e["competitor"] == competitor_name)
            if ks_count >= 3:
                self._blacklisted_players.add(competitor_name)
                logger.info("blacklisted_player: %s (3+ KS events)", competitor_name)

    def mark_map_crowded(self, map_name: str) -> None:
        """Mark a map as crowded."""
        with self._lock:
            self._crowded_maps[map_name] = time.time()

    def is_map_crowded(self, map_name: str) -> bool:
        """Check if a map is currently considered crowded."""
        with self._lock:
            if map_name not in self._crowded_maps:
                return False
            return time.time() - self._crowded_maps[map_name] < self._crowded_timeout_s

    def get_nearby_players(self) -> list[NearbyPlayer]:
        with self._lock:
            now = time.time()
            return [p for p in self._nearby_players.values() if now - p.last_seen < 30]

    def get_competitors(self) -> list[NearbyPlayer]:
        with self._lock:
            now = time.time()
            return [p for p in self._nearby_players.values() if not p.is_party_member and now - p.last_seen < 10]

    def get_ks_events(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._ks_events[-limit:])

    def get_ks_rate_per_hour(self) -> float:
        with self._lock:
            if not self._ks_events:
                return 0.0
            now = time.time()
            window = min(now - self._ks_events[0]["timestamp"], 3600)
            if window <= 0:
                return 0.0
            return len(self._ks_events) / window * 3600

    def get_most_common_ks_competitor(self) -> str | None:
        with self._lock:
            if not self._ks_events:
                return None
            from collections import Counter
            counts = Counter(e["competitor"] for e in self._ks_events)
            return counts.most_common(1)[0][0]

    def get_most_ks_map(self) -> str | None:
        with self._lock:
            if not self._ks_events:
                return None
            from collections import Counter
            counts = Counter(e["map"] for e in self._ks_events)
            return counts.most_common(1)[0][0]

    def is_blacklisted(self, player_name: str) -> bool:
        with self._lock:
            return player_name in self._blacklisted_players

    def get_ks_summary(self) -> str:
        with self._lock:
            lines = [f"── Killsteal Summary ──"]
            lines.append(f"Nearby players: {len(self._nearby_players)}")
            lines.append(f"Competitors: {len(self.get_competitors())}")
            lines.append(f"KS events: {len(self._ks_events)}")
            lines.append(f"KS rate: {self.get_ks_rate_per_hour():.1f}/hr")
            lines.append(f"Blacklisted: {len(self._blacklisted_players)}")
            lines.append(f"Crowded maps: {len(self._crowded_maps)}")
            top_comp = self.get_most_common_ks_competitor()
            if top_comp:
                lines.append(f"Worst competitor: {top_comp}")
            top_map = self.get_most_ks_map()
            if top_map:
                lines.append(f"Worst map: {top_map}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._nearby_players.clear()
            self._ks_events.clear()
            self._crowded_maps.clear()
            self._blacklisted_players.clear()


# ── Global Singleton ──

_anti_ks: AntiKillsteal | None = None
_anti_ks_lock = RLock()


def get_anti_killsteal() -> AntiKillsteal:
    global _anti_ks
    with _anti_ks_lock:
        if _anti_ks is None:
            _anti_ks = AntiKillsteal()
        return _anti_ks
