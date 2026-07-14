"""
Real Combat Loop — continuous combat execution independent of the LLM.

Target → approach → skill rotation → cooldown management → reposition → next target.
The LLM only gets involved for strategic decisions. Combat is a reflex loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

COMBAT_TICK_INTERVAL = 0.2  # 200ms per combat tick


@dataclass
class CombatState:
    """Current state of the combat loop."""
    is_active: bool = False
    current_target_id: int = 0
    current_target_name: str = ""
    current_target_hp_pct: float = 1.0
    current_target_distance: float = 0.0
    current_skill: str = ""
    current_skill_cooldown_ms: int = 0
    last_skill_time: float = 0.0
    aggro_count: int = 0
    my_hp_pct: float = 1.0
    my_sp_pct: float = 1.0
    is_in_combat: bool = False
    combat_started_at: float = 0.0
    kills_this_session: int = 0
    ticks_this_session: int = 0


class CombatLoop:
    """Continuous combat loop that runs independently of the LLM."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._state: CombatState = CombatState()
        self._enqueue_fn: Callable | None = None
        self._get_snapshot_fn: Callable | None = None
        self._get_reflex_fn: Callable | None = None
        self._get_chain_fn: Callable | None = None
        self._get_executor_fn: Callable | None = None
        self._last_tick: float = 0.0

    # ── Public API ──

    def start(self) -> None:
        """Start the combat loop."""
        with self._lock:
            self._state.is_active = True
            logger.info("combat_loop_started")

    def stop(self) -> None:
        """Stop the combat loop."""
        with self._lock:
            self._state.is_active = False
            logger.info("combat_loop_stopped")

    def tick(self) -> str | None:
        """Execute one combat tick. Returns the action to perform, or None."""
        with self._lock:
            if not self._state.is_active:
                return None

            now = time.time()
            if now - self._last_tick < COMBAT_TICK_INTERVAL:
                return None
            self._last_tick = now
            self._state.ticks_this_session += 1

            # 1. Emergency check — always first
            if self._state.my_hp_pct < 0.35:
                return "use_potion_or_heal"
            if self._state.my_hp_pct < 0.15 and self._state.aggro_count > 3:
                return "flee_to_safe_spot"
            if self._state.my_hp_pct < 0.08:
                return "teleport_away"

            # 2. No target — find one
            if self._state.current_target_id == 0 or self._state.current_target_hp_pct <= 0:
                self._state.current_target_id = 0
                self._state.is_in_combat = False
                return "find_next_target"

            # 3. Target is too far — approach
            if self._state.current_target_distance > 3:
                return "approach_target"

            # 4. Target is casting — interrupt
            if self._state.current_target_distance < 10:
                return "interrupt_caster"

            # 5. Execute skill rotation
            if self._state.current_skill:
                if now - self._state.last_skill_time >= self._state.current_skill_cooldown_ms / 1000.0:
                    skill = self._state.current_skill
                    self._state.last_skill_time = now
                    return f"use_skill_{skill}"

            # 6. Check SP — if low, use basic attack
            if self._state.my_sp_pct < 0.2:
                return "use_basic_attack_only"

            return "continue_combat"

    def update_state(self, snapshot: dict) -> None:
        """Update combat state from a snapshot."""
        with self._lock:
            vitals = snapshot.get("vitals", {})
            combat = snapshot.get("combat", {})
            actors = snapshot.get("actors", [])

            self._state.my_hp_pct = float(vitals.get("hp_ratio", 1.0))
            self._state.my_sp_pct = float(vitals.get("sp_ratio", 1.0))
            self._state.aggro_count = int(combat.get("aggro_count", 0))

            # Find current target
            target_id = int(combat.get("target_id", 0))
            if target_id > 0:
                for actor in actors:
                    if int(actor.get("actor_id", 0)) == target_id:
                        self._state.current_target_id = target_id
                        self._state.current_target_name = str(actor.get("name", ""))
                        self._state.current_target_hp_pct = float(actor.get("hp_pct", 1.0))
                        self._state.current_target_distance = float(actor.get("distance", 0))
                        self._state.is_in_combat = True
                        if self._state.combat_started_at == 0:
                            self._state.combat_started_at = time.time()
                        break

            # Check if target is dead
            if self._state.current_target_hp_pct <= 0 and self._state.is_in_combat:
                self._state.kills_this_session += 1
                self._state.current_target_id = 0
                self._state.is_in_combat = False
                self._state.combat_started_at = 0

    def set_target(self, target_id: int, target_name: str = "") -> None:
        with self._lock:
            self._state.current_target_id = target_id
            self._state.current_target_name = target_name
            self._state.is_in_combat = True
            self._state.combat_started_at = time.time()

    def set_skill(self, skill_name: str, cooldown_ms: int = 0) -> None:
        with self._lock:
            self._state.current_skill = skill_name
            self._state.current_skill_cooldown_ms = cooldown_ms

    def get_state(self) -> CombatState:
        with self._lock:
            return self._state

    def get_combat_summary(self) -> str:
        with self._lock:
            return (
                f"── Combat Loop ──\n"
                f"Active: {self._state.is_active}\n"
                f"Target: {self._state.current_target_name} (ID={self._state.current_target_id}, HP={self._state.current_target_hp_pct:.0%})\n"
                f"Distance: {self._state.current_target_distance:.1f}\n"
                f"Aggro: {self._state.aggro_count} | HP: {self._state.my_hp_pct:.0%} | SP: {self._state.my_sp_pct:.0%}\n"
                f"Kills: {self._state.kills_this_session} | Ticks: {self._state.ticks_this_session}"
            )

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def set_get_snapshot_fn(self, fn: Callable) -> None:
        with self._lock:
            self._get_snapshot_fn = fn

    def set_get_reflex_fn(self, fn: Callable) -> None:
        with self._lock:
            self._get_reflex_fn = fn

    def set_get_chain_fn(self, fn: Callable) -> None:
        with self._lock:
            self._get_chain_fn = fn

    def set_get_executor_fn(self, fn: Callable) -> None:
        with self._lock:
            self._get_executor_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._state = CombatState()


# ── Global Singleton ──

_combat_loop: CombatLoop | None = None
_combat_loop_lock = RLock()


def get_combat_loop() -> CombatLoop:
    global _combat_loop
    with _combat_loop_lock:
        if _combat_loop is None:
            _combat_loop = CombatLoop()
        return _combat_loop
