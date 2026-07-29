"""Game time tracking, day/night awareness, and time-dependent behavior."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# In Ragnarok Online, 1 real hour = 1 game day (some servers vary)
_GAME_HOURS_PER_REAL_SECOND = 1.0 / 3600.0
_DAY_START_HOUR = 6
_NIGHT_START_HOUR = 18


@dataclass
class TimeState:
    """Track game time state."""
    game_hour: float = 12.0
    is_day: bool = True
    is_night: bool = False
    real_time_base: float = 0.0
    game_time_base: float = 12.0
    day_cycle_duration_minutes: int = 60
    map_weather: str = "clear"
    map_name: str = ""


_NIGHT_SPAWN_MONSTERS: dict[str, list[str]] = {
    "prontera": ["Ghostring", "Angeling"],
    "morocc": ["Mummy", "Evil Druid"],
    "aldebaran": ["Nightmare"],
}

_NIGHT_BONUSES: dict[str, dict[str, Any]] = {
    "ninja": {"atk_bonus": 1.5, "description": "Ninja ATK +50% at night"},
    "rogue": {"atk_bonus": 1.2, "description": "Rogue ATK +20% at night"},
}


class GameTimeTracker:
    """Track game time, day/night cycle, and time-dependent behavior."""

    def __init__(self, db: Any = None) -> None:
        self._time_states: dict[str, TimeState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_time(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Assess game time and emit time-dependent actions.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        job_name = str(signals.get("job_name", "novice") or "novice").lower()

        ts = self._time_states.setdefault(bot_id, TimeState())
        ts.map_name = map_name

        game_time = signals.get("game_time", None)
        if game_time is not None:
            ts.game_hour = float(game_time)

        self._update_time_state(ts)

        if ts.is_night:
            night_mobs = _NIGHT_SPAWN_MONSTERS.get(map_name, [])
            if night_mobs:
                actions.append({
                    "type": "night_spawn_hunt",
                    "priority": 4,
                    "reason": f"Night time — hunt night-specific monsters: {', '.join(night_mobs)}",
                    "monsters": night_mobs,
                })

        if ts.is_night:
            bonus = _NIGHT_BONUSES.get(job_name)
            if bonus:
                actions.append({
                    "type": "night_bonus_active",
                    "priority": 3,
                    "reason": f"Night bonus active! {bonus['description']}",
                    "bonus": bonus,
                })

        if ts.game_hour >= 1 and ts.game_hour <= 4:
            actions.append({
                "type": "night_rest_suggestion",
                "priority": 1,
                "reason": f"Late night ({ts.game_hour:.0f}:00) — consider resting",
            })

        return actions

    def get_time_of_day(self, bot_id: str) -> str:
        """Get current time of day string."""
        ts = self._time_states.get(bot_id)
        if not ts:
            return "unknown"
        hour = int(ts.game_hour)
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def is_currently_day(self, bot_id: str) -> bool:
        ts = self._time_states.get(bot_id)
        if not ts:
            return True
        return ts.is_day

    def is_currently_night(self, bot_id: str) -> bool:
        ts = self._time_states.get(bot_id)
        if not ts:
            return False
        return ts.is_night

    def get_game_time_string(self, bot_id: str) -> str:
        ts = self._time_states.get(bot_id)
        if not ts:
            return "Unknown time"
        hour = int(ts.game_hour)
        minute = int((ts.game_hour - hour) * 60)
        period = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minute:02d} {period} ({'Day' if ts.is_day else 'Night'})"

    def get_night_monsters(self, map_name: str) -> list[str]:
        return _NIGHT_SPAWN_MONSTERS.get(map_name, [])

    def should_use_trap(self, bot_id: str, monster_element: str = "") -> bool:
        ts = self._time_states.get(bot_id)
        if not ts:
            return False
        return ts.is_night

    def _update_time_state(self, ts: TimeState) -> None:
        if ts.real_time_base == 0:
            ts.real_time_base = time.time()
            return

        elapsed = time.time() - ts.real_time_base
        game_minutes_passed = elapsed * (60.0 / ts.day_cycle_duration_minutes)
        game_hours_passed = game_minutes_passed / 60.0

        ts.game_hour = (ts.game_time_base + game_hours_passed) % 24.0

        hour = ts.game_hour
        if _DAY_START_HOUR <= hour < _NIGHT_START_HOUR:
            ts.is_day = True
            ts.is_night = False
        else:
            ts.is_day = False
            ts.is_night = True

    def reset_time(self, bot_id: str, game_hour: float = 12.0) -> None:
        ts = self._time_states.get(bot_id)
        if ts:
            ts.game_hour = game_hour
            ts.real_time_base = time.time()
            ts.game_time_base = game_hour

    def cleanup_bot(self, bot_id: str) -> None:
        self._time_states.pop(bot_id, None)
