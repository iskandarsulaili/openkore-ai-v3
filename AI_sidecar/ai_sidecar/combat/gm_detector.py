"""
GM Detection and Evasion — watches for suspicious characters and acts natural.

A pro player knows: "That character with the weird name and no equipment
is probably a GM. If a GM appears, act natural — stop botting, move
erratically, maybe even log out."
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SuspiciousPlayer:
    """A player that might be a GM."""
    name: str
    job_class: str = "unknown"
    level: int = 0
    distance: float = 999.0
    is_equipped: bool = True
    watch_duration_s: float = 0.0
    first_spotted: float = 0.0
    last_seen: float = 0.0
    sightings: int = 0
    suspicion_score: float = 0.0  # 0-100
    is_gm: bool = False
    reason: str = ""


@dataclass
class GMAlert:
    """A GM detection alert."""
    player_name: str
    suspicion_score: float
    reason: str
    timestamp: float
    action_taken: str  # "logged_out", "ai_manual", "moved_away", "acted_natural"


class GMDetector:
    """Detects and evades Game Masters."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._suspicious_players: dict[str, SuspiciousPlayer] = {}
        self._gm_alerts: list[GMAlert] = []
        self._max_alerts: int = 50
        self._evasion_mode: bool = False
        self._evasion_until: float = 0.0
        self._natural_moves: int = 0
        self._enqueue_fn: Callable | None = None
        self._gm_name_patterns = ["gm", "gamemaster", "admin", "support", "moderator", "staff", "helper", "guide"]
        self._suspicious_behaviors = [
            "watching_without_moving",
            "following_for_extended_period",
            "no_equipment_visible",
            "unusual_name_pattern",
            "appearing_and_disappearing",
            "ignoring_monsters",
            "same_position_for_minutes",
        ]

    # ── Public API ──

    def update_players(self, players: list[dict], my_x: int = 0, my_y: int = 0) -> list[GMAlert]:
        """Update player list and check for GMs. Returns new alerts."""
        with self._lock:
            now = time.time()
            alerts: list[GMAlert] = []

            for p in players:
                name = str(p.get("name", ""))
                if not name:
                    continue

                dist = float(p.get("distance", 999))
                job = str(p.get("job_class", "unknown"))
                level = int(p.get("level", 0))
                has_equipment = bool(p.get("has_equipment", True))

                if name in self._suspicious_players:
                    sp = self._suspicious_players[name]
                    sp.distance = dist
                    sp.last_seen = now
                    sp.sightings += 1
                    sp.watch_duration_s = now - sp.first_spotted
                else:
                    sp = SuspiciousPlayer(
                        name=name,
                        job_class=job,
                        level=level,
                        distance=dist,
                        is_equipped=has_equipment,
                        first_spotted=now,
                        last_seen=now,
                        sightings=1,
                    )
                    self._suspicious_players[name] = sp

                # Compute suspicion score
                score = 0.0
                reasons = []

                # Check name patterns
                name_lower = name.lower()
                for pattern in self._gm_name_patterns:
                    if pattern in name_lower:
                        score += 40
                        reasons.append(f"name_contains_{pattern}")

                # Check for no equipment
                if not has_equipment:
                    score += 30
                    reasons.append("no_equipment")

                # Check for watching behavior (same position, not moving)
                if sp.sightings > 5 and sp.watch_duration_s > 30:
                    score += 20
                    reasons.append(f"watching_{sp.watch_duration_s:.0f}s")

                # Check for unusual level (very high or very low for the map)
                if level > 99 or level < 10:
                    score += 10
                    reasons.append(f"unusual_level_{level}")

                # Check for following behavior
                if sp.sightings > 10 and sp.watch_duration_s > 120:
                    score += 20
                    reasons.append("following_2min+")

                sp.suspicion_score = min(100, score)
                sp.reason = ", ".join(reasons)
                sp.is_gm = score >= 50

                # Alert if suspicion is high enough
                if score >= 50 and not any(a.player_name == name for a in self._gm_alerts[-10:]):
                    action = self._determine_evasion_action(score)
                    alert = GMAlert(
                        player_name=name,
                        suspicion_score=score,
                        reason=sp.reason,
                        timestamp=now,
                        action_taken=action,
                    )
                    self._gm_alerts.append(alert)
                    if len(self._gm_alerts) > self._max_alerts:
                        self._gm_alerts.pop(0)
                    alerts.append(alert)
                    logger.warning("gm_detected: %s (score=%.0f, reason=%s, action=%s)",
                                   name, score, sp.reason, action)

            # Clean up stale players (not seen in 60s)
            stale = [n for n, sp in self._suspicious_players.items() if now - sp.last_seen > 60]
            for n in stale:
                del self._suspicious_players[n]

            return alerts

    def _determine_evasion_action(self, suspicion_score: float) -> str:
        """Determine what action to take based on suspicion level."""
        if suspicion_score >= 80:
            return "log_out"
        elif suspicion_score >= 60:
            return "ai_manual_and_move"
        elif suspicion_score >= 50:
            return "act_natural"
        return "monitor"

    def act_natural(self) -> list[str]:
        """Generate a sequence of natural-looking actions."""
        with self._lock:
            actions: list[str] = []
            # Pause briefly (simulating alt-tab)
            actions.append("pause_500ms")
            # Move in a non-optimal direction
            actions.append("move_random_direction")
            # Stand still for a moment (simulating checking inventory)
            actions.append("pause_1000ms")
            # Resume farming
            actions.append("resume_farming")
            self._natural_moves += 1
            return actions

    def get_random_direction(self) -> tuple[int, int]:
        """Get a random direction to move (non-optimal, human-like)."""
        with self._lock:
            dx = random.choice([-5, -3, -1, 1, 3, 5])
            dy = random.choice([-5, -3, -1, 1, 3, 5])
            return (dx, dy)

    def is_evasion_active(self) -> bool:
        with self._lock:
            if not self._evasion_mode:
                return False
            if time.time() > self._evasion_until:
                self._evasion_mode = False
                return False
            return True

    def set_evasion_mode(self, duration_s: float = 60.0) -> None:
        with self._lock:
            self._evasion_mode = True
            self._evasion_until = time.time() + duration_s
            logger.info("evasion_mode_activated: duration=%.0fs", duration_s)

    def get_suspicious_players(self) -> list[SuspiciousPlayer]:
        with self._lock:
            return [sp for sp in self._suspicious_players.values() if sp.suspicion_score > 0]

    def get_gm_alerts(self, limit: int = 10) -> list[GMAlert]:
        with self._lock:
            return list(self._gm_alerts[-limit:])

    def get_highest_threat(self) -> SuspiciousPlayer | None:
        with self._lock:
            threats = [sp for sp in self._suspicious_players.values() if sp.is_gm]
            if not threats:
                return None
            return max(threats, key=lambda sp: sp.suspicion_score)

    def get_gm_summary(self) -> str:
        with self._lock:
            lines = [f"── GM Detection Summary ──"]
            lines.append(f"Evasion mode: {self._evasion_mode}")
            lines.append(f"Suspicious players: {len(self._suspicious_players)}")
            lines.append(f"GM alerts: {len(self._gm_alerts)}")
            lines.append(f"Natural moves: {self._natural_moves}")
            top = self.get_highest_threat()
            if top:
                lines.append(f"Highest threat: {top.name} (score={top.suspicion_score:.0f})")
                lines.append(f"  Reason: {top.reason}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._suspicious_players.clear()
            self._gm_alerts.clear()
            self._evasion_mode = False
            self._evasion_until = 0.0
            self._natural_moves = 0


# ── Global Singleton ──

_gm_detector: GMDetector | None = None
_gm_detector_lock = RLock()


def get_gm_detector() -> GMDetector:
    global _gm_detector
    with _gm_detector_lock:
        if _gm_detector is None:
            _gm_detector = GMDetector()
        return _gm_detector
