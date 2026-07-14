"""
PK Avoidance System — tracks known PKers, switches maps when detected,
uses safe routes, emergency teleports on PvP flag, and prioritizes safe zones.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PKerRecord:
    """A record of a player who has PK'd us."""
    name: str
    kill_count: int = 1
    last_seen: float = 0.0
    last_map: str = ""
    job_class: str = ""
    base_level: int = 0
    threat_level: int = 5  # 1-10
    is_active: bool = True


@dataclass
class SafeZone:
    """A safe zone where PK is not possible."""
    map_name: str
    x: int = 0
    y: int = 0
    radius: int = 20
    is_town: bool = True


class PKAvoidance:
    """Tracks PKers and avoids them actively."""

    # Known safe zones (towns)
    SAFE_ZONES: list[SafeZone] = [
        SafeZone("prontera", 150, 120, 50, True),
        SafeZone("geffen", 120, 80, 40, True),
        SafeZone("payon", 100, 100, 30, True),
        SafeZone("alberta", 80, 80, 30, True),
        SafeZone("izlude", 100, 80, 30, True),
        SafeZone("morocc", 100, 100, 30, True),
        SafeZone("aldebaran", 120, 100, 30, True),
        SafeZone("xmas", 100, 100, 30, True),
        SafeZone("comodo", 100, 100, 30, True),
        SafeZone("yuno", 120, 100, 30, True),
        SafeZone("amatsu", 80, 80, 30, True),
        SafeZone("kunlun", 80, 80, 30, True),
        SafeZone("lighthalzen", 100, 100, 30, True),
        SafeZone("rachel", 100, 100, 30, True),
        SafeZone("veins", 80, 80, 30, True),
    ]

    def __init__(self) -> None:
        self._lock = RLock()
        self._pkers: dict[str, PKerRecord] = {}
        self._current_map: str = ""
        self._pvp_flag: bool = False
        self._last_pk_time: float = 0.0
        self._evasion_mode: bool = False
        self._enqueue_fn: Callable | None = None

    # ── Public API ──

    def record_pk(self, attacker_name: str, map_name: str, job_class: str = "",
                  base_level: int = 0) -> None:
        """Record a PK incident."""
        with self._lock:
            now = time.time()
            if attacker_name in self._pkers:
                record = self._pkers[attacker_name]
                record.kill_count += 1
                record.last_seen = now
                record.last_map = map_name
                record.threat_level = min(10, record.threat_level + 1)
            else:
                self._pkers[attacker_name] = PKerRecord(
                    name=attacker_name,
                    kill_count=1,
                    last_seen=now,
                    last_map=map_name,
                    job_class=job_class,
                    base_level=base_level,
                    threat_level=5,
                )
            self._last_pk_time = now
            self._pvp_flag = True
            self._evasion_mode = True
            logger.warning("pk_recorded: %s on %s (total kills=%d)", attacker_name, map_name,
                          self._pkers[attacker_name].kill_count)

    def is_pker_nearby(self, players: list[dict]) -> bool:
        """Check if any known PKer is nearby."""
        with self._lock:
            for player in players:
                name = str(player.get("name", ""))
                if name in self._pkers:
                    record = self._pkers[name]
                    if record.is_active and time.time() - record.last_seen < 300:
                        return True
            return False

    def get_safest_map(self, current_map: str = "") -> str:
        """Get the safest map to go to."""
        with self._lock:
            # If we were just PK'd, go to a town
            if time.time() - self._last_pk_time < 60:
                return "prontera"

            # Check if current map has PKers
            for record in self._pkers.values():
                if record.is_active and record.last_map == current_map and time.time() - record.last_seen < 300:
                    # Switch to a different farming map
                    alternatives = ["payon_dun01", "gef_dun01", "moc_fild01", "prt_fild01"]
                    for alt in alternatives:
                        if alt != current_map:
                            return alt
                    return "prontera"

            return current_map

    def should_evacuate(self, players: list[dict]) -> bool:
        """Check if we should evacuate the current map."""
        with self._lock:
            if not self._evasion_mode:
                return False
            return self.is_pker_nearby(players)

    def get_evacuation_target(self) -> str:
        """Get the target map for evacuation."""
        with self._lock:
            return self.get_safest_map(self._current_map)

    def clear_evasion(self) -> None:
        """Clear evasion mode after a cooldown."""
        with self._lock:
            if time.time() - self._last_pk_time > 600:  # 10 min cooldown
                self._evasion_mode = False
                self._pvp_flag = False

    def update_map(self, map_name: str) -> None:
        with self._lock:
            self._current_map = map_name

    def get_pk_summary(self) -> str:
        with self._lock:
            lines = [f"── PK Avoidance ──"]
            lines.append(f"Evasion mode: {self._evasion_mode}")
            lines.append(f"PKers tracked: {len(self._pkers)}")
            active = [p for p in self._pkers.values() if p.is_active]
            if active:
                lines.append(f"Active threats: {', '.join(f'{p.name}({p.threat_level})' for p in sorted(active, key=lambda x: -x.threat_level)[:5])}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._pkers.clear()
            self._pvp_flag = False
            self._last_pk_time = 0.0
            self._evasion_mode = False


# ── Global Singleton ──

_pk_avoid: PKAvoidance | None = None
_pk_avoid_lock = RLock()


def get_pk_avoidance() -> PKAvoidance:
    global _pk_avoid
    with _pk_avoid_lock:
        if _pk_avoid is None:
            _pk_avoid = PKAvoidance()
        return _pk_avoid
