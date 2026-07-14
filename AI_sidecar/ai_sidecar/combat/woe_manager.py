"""
WoE/GvG Mode — specialized behavior for War of Emperium and Guild vs Guild combat.

During WoE, priorities change: capture the emperium, defend the castle,
kill enemy players, support allies. This module manages that transition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class WoEMode(Enum):
    OFF = "off"
    ATTACK = "attack"
    DEFEND = "defend"
    SUPPORT = "support"
    SCOUT = "scout"


class WoERole(Enum):
    EMPERIUM_BREAKER = "emperium_breaker"
    GUARDIAN_KILLER = "guardian_killer"
    PLAYER_KILLER = "player_killer"
    HEALER = "healer"
    SUPPORT = "support"
    SCOUT = "scout"


@dataclass
class WoEConfig:
    """Configuration for WoE behavior."""
    enabled: bool = True
    mode: WoEMode = WoEMode.OFF
    role: WoERole = WoERole.PLAYER_KILLER
    castle_name: str = ""
    guild_name: str = ""
    ally_guilds: list[str] = field(default_factory=list)
    enemy_guilds: list[str] = field(default_factory=list)
    emperium_hp_threshold: int = 100000
    guardian_priority: bool = True
    min_party_size: int = 3
    retreat_hp_pct: float = 0.3
    use_consumables: bool = True
    alert_on_enemy_sight: bool = True


@dataclass
class WoEState:
    """Current WoE state."""
    is_woe_hours: bool = False
    current_castle: str = ""
    current_mode: WoEMode = WoEMode.OFF
    allies_nearby: int = 0
    enemies_nearby: int = 0
    emperium_hp: int = 0
    emperium_max_hp: int = 0
    guardians_alive: int = 0
    last_updated: float = 0.0


class WoEManager:
    """Manages WoE/GvG behavior."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._config = WoEConfig()
        self._state = WoEState()
        self._woe_maps: set[str] = {
            "prt_gld", "prt_gld01", "prt_gld02", "prt_gld03", "prt_gld04",
            "gef_gld", "gef_gld01", "gef_gld02", "gef_gld03", "gef_gld04",
            "pay_gld", "pay_gld01", "pay_gld02", "pay_gld03", "pay_gld04",
            "alde_gld", "alde_gld01", "alde_gld02", "alde_gld03", "alde_gld04",
            "sch_gld", "sch_gld01", "sch_gld02", "sch_gld03", "sch_gld04",
        }
        self._woe_schedule = {
            "wednesday": (20, 22),  # 8-10 PM
            "saturday": (20, 23),   # 8-11 PM
            "sunday": (20, 23),     # 8-11 PM
        }

    # ── Public API ──

    def is_woe_map(self, map_name: str) -> bool:
        """Check if a map is a WoE map."""
        with self._lock:
            base = map_name.lower().strip()
            return base in self._woe_maps or any(base.startswith(m) for m in self._woe_maps)

    def is_woe_time(self) -> bool:
        """Check if it's currently WoE hours."""
        import datetime
        with self._lock:
            now = datetime.datetime.now()
            day = now.strftime("%A").lower()
            if day not in self._woe_schedule:
                return False
            start, end = self._woe_schedule[day]
            return start <= now.hour < end

    def get_mode(self) -> WoEMode:
        with self._lock:
            if not self._config.enabled or not self._state.is_woe_hours:
                return WoEMode.OFF
            return self._state.current_mode

    def set_mode(self, mode: WoEMode) -> None:
        with self._lock:
            self._state.current_mode = mode

    def set_role(self, role: WoERole) -> None:
        with self._lock:
            self._config.role = role

    def get_role(self) -> WoERole:
        with self._lock:
            return self._config.role

    def update_state(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._state.last_updated = __import__("time").time()

    def get_state(self) -> WoEState:
        with self._lock:
            return self._state

    def get_config(self) -> WoEConfig:
        with self._lock:
            return self._config

    def set_config(self, config: WoEConfig) -> None:
        with self._lock:
            self._config = config

    def get_priority_targets(self) -> list[str]:
        """Get priority target types for current mode."""
        with self._lock:
            if self._state.current_mode == WoEMode.ATTACK:
                return ["emperium", "guardian", "enemy_player"]
            elif self._state.current_mode == WoEMode.DEFEND:
                return ["enemy_player", "guardian", "emperium"]
            elif self._state.current_mode == WoEMode.SUPPORT:
                return ["ally_player", "heal", "buff"]
            elif self._state.current_mode == WoEMode.SCOUT:
                return ["enemy_player", "intel"]
            return []

    def get_behavior_instructions(self) -> str:
        """Get behavior instructions for the current WoE state."""
        with self._lock:
            if not self._state.is_woe_hours:
                return "Normal farming mode"
            
            mode = self._state.current_mode
            role = self._config.role
            
            lines = [f"WOE MODE ACTIVE — {mode.value.upper()}"]
            lines.append(f"Role: {role.value}")
            lines.append(f"Castle: {self._state.current_castle or 'unknown'}")
            lines.append(f"Allies nearby: {self._state.allies_nearby}")
            lines.append(f"Enemies nearby: {self._state.enemies_nearby}")
            
            if mode == WoEMode.ATTACK:
                lines.append("Priority: Break emperium > Kill guardians > Kill players")
                lines.append("Stay with party. Use AoE on grouped enemies.")
                lines.append("Retreat if HP < 30%.")
            elif mode == WoEMode.DEFEND:
                lines.append("Priority: Kill players > Protect guardians > Protect emperium")
                lines.append("Hold chokepoints. Do not chase kills outside castle.")
                lines.append("Alert guild when enemies spotted.")
            elif mode == WoEMode.SUPPORT:
                lines.append("Priority: Heal allies > Buff allies > Debuff enemies")
                lines.append("Stay behind frontline. Keep buffs up on party.")
            elif mode == WoEMode.SCOUT:
                lines.append("Priority: Scout enemy positions > Report intel > Avoid combat")
                lines.append("Do not engage. Report enemy count and positions.")
            
            return "\n".join(lines)

    def get_woe_maps(self) -> list[str]:
        with self._lock:
            return sorted(self._woe_maps)

    def add_woe_map(self, map_name: str) -> None:
        with self._lock:
            self._woe_maps.add(map_name.lower().strip())

    def set_woe_schedule(self, day: str, start_hour: int, end_hour: int) -> None:
        with self._lock:
            self._woe_schedule[day.lower()] = (start_hour, end_hour)


# ── Global Singleton ──

_woe_manager: WoEManager | None = None
_woe_manager_lock = RLock()


def get_woe_manager() -> WoEManager:
    global _woe_manager
    with _woe_manager_lock:
        if _woe_manager is None:
            _woe_manager = WoEManager()
        return _woe_manager
