"""
WoE Combat AI — real-time navigation to tactical positions, guardian kill
sequencing, emperium break automation, and retreat logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class WoEPhase:
    """Current phase of WoE combat."""
    phase: str = "idle"  # idle, approach, clear_guardians, break_emperium, defend, retreat
    target_position: tuple[int, int] = (0, 0)
    target_monster_id: int = 0
    started_at: float = 0.0
    enemies_nearby: int = 0
    allies_nearby: int = 0
    emperium_hp_pct: float = 1.0
    guardians_alive: int = 0


class WoECombatAI:
    """Executes WoE combat tactics in real-time."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._phase: WoEPhase = WoEPhase()
        self._castle_map: str = ""
        self._enqueue_fn: Callable | None = None
        self._last_action: float = 0.0
        self._action_cooldown: float = 1.0

    # ── Public API ──

    def set_castle(self, map_name: str) -> None:
        with self._lock:
            self._castle_map = map_name
            self._phase = WoEPhase(phase="approach", started_at=time.time())

    def update_battlefield(self, enemies: int = 0, allies: int = 0,
                           guardians: int = 0, emperium_hp: float = 1.0) -> None:
        """Update battlefield state and adjust phase."""
        with self._lock:
            self._phase.enemies_nearby = enemies
            self._phase.allies_nearby = allies
            self._phase.guardians_alive = guardians
            self._phase.emperium_hp_pct = emperium_hp

            # Phase transitions
            if self._phase.phase == "approach" and guardians > 0:
                self._phase.phase = "clear_guardians"
                logger.info("woe_phase: clear_guardians (%d alive)", guardians)
            elif self._phase.phase == "clear_guardians" and guardians == 0:
                self._phase.phase = "break_emperium"
                logger.info("woe_phase: break_emperium")
            elif self._phase.phase == "break_emperium" and emperium_hp <= 0:
                self._phase.phase = "idle"
                logger.info("woe_phase: victory")
            elif enemies > allies * 2 and self._phase.phase not in ("retreat",):
                self._phase.phase = "retreat"
                logger.info("woe_phase: retreat (outnumbered %d vs %d)", enemies, allies)

    def get_action(self) -> str | None:
        """Get the next action based on current phase."""
        with self._lock:
            now = time.time()
            if now - self._last_action < self._action_cooldown:
                return None
            self._last_action = now

            if self._phase.phase == "approach":
                return "move_to_castle_entrance"
            elif self._phase.phase == "clear_guardians":
                return "attack_nearest_guardian"
            elif self._phase.phase == "break_emperium":
                return "attack_emperium"
            elif self._phase.phase == "defend":
                return "hold_chokepoint"
            elif self._phase.phase == "retreat":
                return "retreat_to_safe_zone"
            return None

    def get_phase(self) -> WoEPhase:
        with self._lock:
            return self._phase

    def get_woe_summary(self) -> str:
        with self._lock:
            return (
                f"── WoE Combat ──\n"
                f"Phase: {self._phase.phase}\n"
                f"Castle: {self._castle_map}\n"
                f"Enemies: {self._phase.enemies_nearby} | Allies: {self._phase.allies_nearby}\n"
                f"Guardians: {self._phase.guardians_alive} | Emperium: {self._phase.emperium_hp_pct:.0%}"
            )

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._phase = WoEPhase()
            self._castle_map = ""


# ── Global Singleton ──

_woe_combat: WoECombatAI | None = None
_woe_combat_lock = RLock()


def get_woe_combat_ai() -> WoECombatAI:
    global _woe_combat
    with _woe_combat_lock:
        if _woe_combat is None:
            _woe_combat = WoECombatAI()
        return _woe_combat
