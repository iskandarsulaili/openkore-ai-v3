"""Mercenary hiring and management."""
from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


_MERCENARY_TYPES: dict[str, dict[str, Any]] = {
    "archer": {
        "name": "Archer Mercenary", "class": "archer",
        "min_level": 40, "cost_per_hour": 5000,
        "skill": "AC_DOUBLE", "description": "Ranged DPS mercenary",
    },
    "swordman": {
        "name": "Swordman Mercenary", "class": "swordman",
        "min_level": 30, "cost_per_hour": 3000,
        "skill": "SM_BASH", "description": "Melee tank mercenary",
    },
    "mage": {
        "name": "Mage Mercenary", "class": "mage",
        "min_level": 50, "cost_per_hour": 8000,
        "skill": "MG_FIREBOLT", "description": "Magic DPS mercenary",
    },
    "healer": {
        "name": "Healer Mercenary", "class": "acolyte",
        "min_level": 45, "cost_per_hour": 6000,
        "skill": "AL_HEAL", "description": "Healing support mercenary",
    },
}


@dataclass
class MercenaryState:
    """Track mercenary state."""
    hired: bool = False
    merc_type: str = ""
    name: str = ""
    level: int = 1
    hired_at: float = 0.0
    cost_per_hour: int = 0
    total_cost: int = 0
    remaining_time: float = 0.0


class MercenaryManager:
    """Handle mercenary hiring, management, and renewal."""

    def __init__(self, db: Any = None) -> None:
        self._mercenary_states: dict[str, MercenaryState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_mercenary(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check mercenary state and recommend actions.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        base_level = int(signals.get("base_level", 1) or 1)
        zeny = int(signals.get("zeny", 0) or 0)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        now = __import__("time").time()

        mstate = self._mercenary_states.get(bot_id)

        if mstate and mstate.hired:
            elapsed = now - mstate.hired_at
            remaining = mstate.remaining_time - elapsed

            if remaining <= 300:
                if zeny >= mstate.cost_per_hour:
                    actions.append({
                        "type": "renew_mercenary",
                        "priority": 6,
                        "reason": f"Mercenary {mstate.name} expiring soon ({remaining:.0f}s) — renew",
                        "cost": mstate.cost_per_hour,
                    })
            return actions

        if base_level < 30:
            return actions

        if zeny < 10000:
            return actions

        is_hunting_map = any(
            keyword in map_name for keyword in
            ["dun", "field", "cave", "tower", "dungeon", "map"]
        )

        if is_hunting_map:
            recommended = self._recommend_mercenary(bot_id, signals)
            if recommended:
                cost = recommended["cost_per_hour"]
                if zeny >= cost * 3:
                    actions.append({
                        "type": "hire_mercenary",
                        "priority": 5,
                        "reason": f"Hire {recommended['name']} for hunting ({cost}z/hour)",
                        "mercenary_type": recommended["class"],
                        "cost_per_hour": cost,
                    })

        return actions

    def hire_mercenary(self, bot_id: str, merc_type: str) -> str | None:
        """Hire a mercenary of the given type.

        Returns the hire command, or None if type unknown.
        """
        merc_info = _MERCENARY_TYPES.get(merc_type)
        if not merc_info:
            logger.warning("[mercenary] %s: unknown mercenary type '%s'", bot_id, merc_type)
            return None

        self._mercenary_states[bot_id] = MercenaryState(
            hired=True,
            merc_type=merc_type,
            name=merc_info["name"],
            level=merc_info["min_level"],
            hired_at=__import__("time").time(),
            cost_per_hour=merc_info["cost_per_hour"],
            remaining_time=3600,
        )

        logger.info("[mercenary] %s: hired %s", bot_id, merc_info["name"])
        return f"merc {merc_type}"

    def renew_mercenary(self, bot_id: str) -> str | None:
        """Renew a mercenary contract.

        Returns the renew command, or None if no mercenary to renew.
        """
        mstate = self._mercenary_states.get(bot_id)
        if not mstate or not mstate.hired:
            return None

        mstate.hired_at = __import__("time").time()
        mstate.remaining_time = 3600
        mstate.total_cost += mstate.cost_per_hour

        logger.info("[mercenary] %s: renewed %s", bot_id, mstate.name)
        return "merc renew"

    def dismiss_mercenary(self, bot_id: str) -> str:
        """Dismiss current mercenary."""
        mstate = self._mercenary_states.get(bot_id)
        if mstate:
            mstate.hired = False
        return "merc dismiss"

    def _recommend_mercenary(
        self,
        bot_id: str,
        signals: dict[str, Any],
    ) -> dict | None:
        """Recommend the best mercenary based on situation."""
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        zeny = int(signals.get("zeny", 0) or 0)

        if base_level >= _MERCENARY_TYPES["healer"]["min_level"]:
            return _MERCENARY_TYPES["healer"]

        if job_name in ("swordman", "knight", "thief", "assassin"):
            if base_level >= _MERCENARY_TYPES["archer"]["min_level"]:
                return _MERCENARY_TYPES["archer"]

        affordable = [
            m for m in _MERCENARY_TYPES.values()
            if base_level >= m["min_level"] and zeny >= m["cost_per_hour"] * 2
        ]
        if affordable:
            return min(affordable, key=lambda m: m["cost_per_hour"])

        return None

    def get_mercenary_status(self, bot_id: str) -> dict | None:
        """Get mercenary status summary."""
        mstate = self._mercenary_states.get(bot_id)
        if not mstate or not mstate.hired:
            return None

        elapsed = __import__("time").time() - mstate.hired_at
        remaining = max(0, mstate.remaining_time - elapsed)

        return {
            "name": mstate.name,
            "type": mstate.merc_type,
            "remaining_time": remaining,
            "total_cost": mstate.total_cost,
        }

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove mercenary state for a bot."""
        self._mercenary_states.pop(bot_id, None)
