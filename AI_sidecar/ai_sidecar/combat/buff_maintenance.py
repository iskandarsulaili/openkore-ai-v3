"""
Buff Maintenance Module

Tracks and maintains character buffs. Provides thread-safe access to a registry
of known buffs and methods for determining which buffs need recasting based on
active buff state, available SP, and available skills.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class Buff:
    """A buff that can be cast on self or party members."""

    name: str
    skill_name: str
    duration_seconds: int
    recast_threshold_seconds: int
    sp_cost: int
    target: str = "self"
    required_job: str = "novice"
    priority: int = 50
    tags: list[str] = field(default_factory=list)


class BuffMaintenance:
    """Thread-safe registry and evaluator for character buffs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._buffs: dict[str, Buff] = {}
        self._load_default_buffs()

    # ── Default buff population ──────────────────────────────────────────

    def _load_default_buffs(self) -> None:
        """Pre-populate the registry with common Ragnarok Online buffs."""
        defaults: list[Buff] = [
            Buff("Blessing", "Blessing", 300, 30, 15, "self", "novice", 90, ["offensive"]),
            Buff("Increase Agility", "Increase Agility", 300, 30, 15, "self", "novice", 85, ["offensive", "movement"]),
            Buff("Endure", "Endure", 30, 5, 10, "self", "novice", 80, ["defensive"]),
            Buff("Improve Concentration", "Improve Concentration", 180, 20, 20, "self", "novice", 75, ["offensive"]),
            Buff("Energy Coat", "Energy Coat", 300, 30, 20, "self", "novice", 70, ["defensive"]),
            Buff("Assumptio", "Assumptio", 20, 5, 25, "self", "novice", 95, ["defensive"]),
            Buff("Kyrie Eleison", "Kyrie Eleison", 300, 30, 20, "self", "novice", 85, ["defensive"]),
            Buff("Gloria", "Gloria", 180, 20, 20, "self", "novice", 70, ["offensive"]),
            Buff("Magnificat", "Magnificat", 180, 20, 20, "self", "novice", 65, ["regen"]),
            Buff("Impositio Manus", "Impositio Manus", 180, 20, 20, "self", "novice", 75, ["offensive"]),
            Buff("Aspersio", "Aspersio", 180, 20, 20, "self", "novice", 70, ["offensive", "elemental"]),
            Buff("Elemental Converter", "Elemental Converter", 180, 20, 10, "self", "novice", 80, ["offensive", "elemental"]),
            Buff("Concentration", "Concentration", 120, 15, 15, "self", "novice", 80, ["offensive"]),
            Buff("True Sight", "True Sight", 180, 20, 20, "self", "novice", 75, ["offensive"]),
            Buff("Wind Walker", "Wind Walker", 180, 20, 15, "self", "novice", 70, ["movement"]),
            Buff("Ode to Solitude", "Ode to Solitude", 180, 20, 20, "self", "novice", 70, ["offensive"]),
            Buff("Poem of Bragi", "Poem of Bragi", 180, 20, 20, "self", "novice", 65, ["support"]),
            Buff("Assassin Cross of Sunset", "Assassin Cross of Sunset", 180, 20, 20, "self", "novice", 80, ["offensive"]),
            Buff("Cloaking", "Cloaking", 60, 10, 15, "self", "novice", 60, ["defensive", "stealth"]),
            Buff("Enchant Poison", "Enchant Poison", 180, 20, 15, "self", "novice", 75, ["offensive", "elemental"]),
            Buff("Enchant Deadly Poison", "Enchant Deadly Poison", 180, 20, 20, "self", "novice", 85, ["offensive", "elemental"]),
        ]
        for buff in defaults:
            self._buffs[buff.name] = buff

    # ── Registration ───────────────────────────────────────────────────

    def register_buff(self, buff: Buff) -> None:
        """Register or update a buff in the registry."""
        with self._lock:
            self._buffs[buff.name] = buff

    # ── Lookup ───────────────────────────────────────────────────────────

    def get_buff(self, name: str) -> Buff | None:
        """Look up a buff by name. Returns None if not found."""
        with self._lock:
            return self._buffs.get(name)

    def get_buffs_by_tag(self, tag: str) -> list[Buff]:
        """Return all buffs that have the given tag."""
        with self._lock:
            return [b for b in self._buffs.values() if tag in b.tags]

    def get_buffs_for_job(self, job_class: str) -> list[Buff]:
        """Return all buffs whose required_job matches the given job class.

        A buff with required_job == "novice" is available to every class.
        """
        with self._lock:
            return [
                b
                for b in self._buffs.values()
                if b.required_job == "novice" or b.required_job == job_class
            ]

    def get_all_buffs(self) -> list[Buff]:
        """Return a copy of every registered buff."""
        with self._lock:
            return list(self._buffs.values())

    # ── Duration helpers ─────────────────────────────────────────────────

    def get_buff_duration(self, name: str) -> int:
        """Return the full duration in seconds for a named buff.

        Returns 0 if the buff is not registered.
        """
        buff = self.get_buff(name)
        return buff.duration_seconds if buff else 0

    def get_buff_remaining(self, name: str, active_buffs: dict[str, Any]) -> int:
        """Return the remaining time in seconds for a named buff.

        ``active_buffs`` is a dict mapping buff name -> dict with at least
        an ``"expires_at"`` key (a Unix timestamp).  Returns 0 if the buff
        is not active or not registered.
        """
        if name not in active_buffs:
            return 0
        expires_at = active_buffs[name].get("expires_at", 0)
        remaining = int(expires_at - time.time())
        return max(remaining, 0)

    def is_buff_active(self, name: str, active_buffs: dict[str, Any]) -> bool:
        """Return True if the named buff is currently active."""
        return self.get_buff_remaining(name, active_buffs) > 0

    def should_recast(self, name: str, active_buffs: dict[str, Any]) -> bool:
        """Return True if the named buff should be recast.

        A buff needs recasting when it is not active OR when its remaining
        duration is below the recast threshold.
        """
        buff = self.get_buff(name)
        if buff is None:
            return False
        remaining = self.get_buff_remaining(name, active_buffs)
        if remaining <= 0:
            return True
        return remaining < buff.recast_threshold_seconds

    # ── Decision helpers ─────────────────────────────────────────────────

    def get_expiring_buffs(self, active_buffs: dict[str, Any]) -> list[Buff]:
        """Return all registered buffs that need recasting.

        A buff needs recasting when it is not active or its remaining
        duration is below its recast threshold.
        """
        with self._lock:
            result: list[Buff] = []
            for buff in self._buffs.values():
                if self.should_recast(buff.name, active_buffs):
                    result.append(buff)
            return result

    def get_buffs_to_cast(
        self,
        active_buffs: dict[str, Any],
        current_sp: int,
        available_skills: set[str],
    ) -> list[Buff]:
        """Return buffs that should be cast right now.

        Filters expiring buffs by:
        1. The skill is in ``available_skills``.
        2. The character has enough SP.
        3. Sorted by priority descending (highest first).
        """
        expiring = self.get_expiring_buffs(active_buffs)
        affordable = [
            b
            for b in expiring
            if b.skill_name in available_skills and current_sp >= b.sp_cost
        ]
        affordable.sort(key=lambda b: b.priority, reverse=True)
        return affordable

    def get_buff_priority_queue(
        self,
        active_buffs: dict[str, Any],
        current_sp: int,
    ) -> list[Buff]:
        """Return all registered buffs sorted by priority descending.

        Filters out buffs that are already active and not near expiry.
        Includes buffs the character may not have the skill for — the
        caller is expected to filter further.
        """
        with self._lock:
            candidates: list[Buff] = []
            for buff in self._buffs.values():
                remaining = self.get_buff_remaining(buff.name, active_buffs)
                if remaining <= 0 or remaining < buff.recast_threshold_seconds:
                    candidates.append(buff)
            candidates.sort(key=lambda b: b.priority, reverse=True)
            return candidates

    # ── Status display ───────────────────────────────────────────────────

    def get_buff_status_text(self, active_buffs: dict[str, Any]) -> str:
        """Return a human-readable status string of active and expiring buffs.

        Example::

            Blessing (45s)  |  Increase Agility (12s, recast!)  |  Endure (expired)
        """
        with self._lock:
            parts: list[str] = []
            for buff in self._buffs.values():
                remaining = self.get_buff_remaining(buff.name, active_buffs)
                if remaining > 0:
                    label = f"{buff.name} ({remaining}s)"
                    if remaining < buff.recast_threshold_seconds:
                        label += " (recast!)"
                    parts.append(label)
                else:
                    parts.append(f"{buff.name} (expired)")
            return "  |  ".join(parts) if parts else "No buffs registered"


# ── Global singleton ─────────────────────────────────────────────────────

_BUFF_MAINTENANCE: BuffMaintenance | None = None
_BUFF_MAINTENANCE_LOCK = RLock()


def get_buff_maintenance() -> BuffMaintenance:
    """Return the global BuffMaintenance singleton."""
    global _BUFF_MAINTENANCE
    if _BUFF_MAINTENANCE is None:
        with _BUFF_MAINTENANCE_LOCK:
            if _BUFF_MAINTENANCE is None:
                _BUFF_MAINTENANCE = BuffMaintenance()
    return _BUFF_MAINTENANCE
