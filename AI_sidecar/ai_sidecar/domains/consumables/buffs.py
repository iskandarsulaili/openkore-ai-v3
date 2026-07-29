"""Auto-buff management — Blessing, AGI Up, and class-specific buffs."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Skill-based buff definitions per class
_CLASS_BUFFS: dict[str, list[dict[str, Any]]] = {
    "acolyte": [
        {"skill": "AL_BLESSING", "name": "Blessing", "min_level": 5, "stat_boost": "str/dex/int",
         "priority": 9, "sp_cost": 28},
        {"skill": "AL_INCAGI", "name": "AGI Up", "min_level": 5, "stat_boost": "agi",
         "priority": 8, "sp_cost": 20},
        {"skill": "AL_CRUIS", "name": "Ruwach", "min_level": 1, "stat_boost": "hide_detect",
         "priority": 3, "sp_cost": 10},
    ],
    "priest": [
        {"skill": "AL_BLESSING", "name": "Blessing", "min_level": 10, "stat_boost": "str/dex/int",
         "priority": 9, "sp_cost": 28},
        {"skill": "AL_INCAGI", "name": "AGI Up", "min_level": 10, "stat_boost": "agi",
         "priority": 8, "sp_cost": 20},
        {"skill": "PR_IMPOSITIO", "name": "Impositio Manus", "min_level": 5, "stat_boost": "atk",
         "priority": 7, "sp_cost": 20},
        {"skill": "PR_BENEDICTIO", "name": "Benedictio", "min_level": 3, "stat_boost": "blessing",
         "priority": 6, "sp_cost": 30},
    ],
    "mage": [
        {"skill": "MG_SRECOVERY", "name": "Soul Recovery", "min_level": 1, "stat_boost": "sp_regen",
         "priority": 5, "sp_cost": 0},
    ],
    "wizard": [
        {"skill": "MG_SRECOVERY", "name": "Soul Recovery", "min_level": 1, "stat_boost": "sp_regen",
         "priority": 5, "sp_cost": 0},
        {"skill": "WZ_ESTIMATION", "name": "Sight", "min_level": 1, "stat_boost": "vision",
         "priority": 4, "sp_cost": 10},
    ],
    "thief": [
        {"skill": "TF_HIDING", "name": "Hiding", "min_level": 1, "stat_boost": "stealth",
         "priority": 6, "sp_cost": 10},
    ],
    "archer": [
        {"skill": "AC_OWL", "name": "Owl's Eye", "min_level": 1, "stat_boost": "dex",
         "priority": 7, "sp_cost": 0},
    ],
    "hunter": [
        {"skill": "AC_OWL", "name": "Owl's Eye", "min_level": 5, "stat_boost": "dex",
         "priority": 7, "sp_cost": 0},
        {"skill": "HT_BEAST", "name": "Beast Stance", "min_level": 1, "stat_boost": "falcon",
         "priority": 4, "sp_cost": 15},
    ],
    "swordman": [
        {"skill": "SM_PROVOKE", "name": "Provoke", "min_level": 5, "stat_boost": "aggro",
         "priority": 5, "sp_cost": 5},
    ],
}

# Item-based buffs (scrolls, potions)
_ITEM_BUFFS: dict[str, dict[str, Any]] = {
    "awakening_potion": {
        "item_id": "5020", "name": "Awakening Potion",
        "stat_boost": "aspd", "bonus": 10, "duration": 300, "min_level": 30,
    },
    "concentration_potion": {
        "item_id": "5021", "name": "Concentration Potion",
        "stat_boost": "atk", "bonus": 20, "duration": 300, "min_level": 40,
    },
}


class AutoBuffManager:
    """Manage auto-buff application based on class and situation."""

    def __init__(self, db: Any = None) -> None:
        self._active_buffs: dict[str, dict[str, float]] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_buffs(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if buffs need refreshing.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        current_sp = int(signals.get("sp", 0) or 0)
        known_skills = signals.get("skills", []) or []
        active_buffs_list = signals.get("buffs", []) or []

        active_buff_names = {
            b.get("name", "") if isinstance(b, dict) else str(b)
            for b in active_buffs_list
        }

        now = __import__("time").time()
        bot_buffs = self._active_buffs.setdefault(bot_id, {})

        class_buffs = _CLASS_BUFFS.get(job_name, [])
        has_any_class_buff = False

        for buff_def in class_buffs:
            skill_id = buff_def["skill"]
            if skill_id not in known_skills:
                continue
            if current_sp < buff_def["sp_cost"] + 10:
                continue

            buff_name = buff_def["name"]
            is_active = (
                buff_name.lower() in {n.lower() for n in active_buff_names}
                or bot_buffs.get(skill_id, 0) > now
            )

            if not is_active:
                actions.append({
                    "type": "cast_buff",
                    "priority": buff_def["priority"],
                    "reason": f"Cast {buff_name} ({buff_def['stat_boost']}) for {job_name}",
                    "skill": skill_id,
                    "skill_name": buff_name,
                })
                has_any_class_buff = True

        base_level = int(signals.get("base_level", 1) or 1)
        for buff_id, buff_def in _ITEM_BUFFS.items():
            if base_level < buff_def["min_level"]:
                continue
            buff_name = buff_def["name"]
            is_active = buff_name.lower() in {n.lower() for n in active_buff_names}
            if not is_active:
                actions.append({
                    "type": "use_item_buff",
                    "priority": 4,
                    "reason": f"Use {buff_name} (+{buff_def['bonus']} {buff_def['stat_boost']})",
                    "item_id": buff_def["item_id"],
                    "item_name": buff_name,
                })

        return actions

    def record_buff_cast(self, bot_id: str, skill_id: str, duration: float = 180.0) -> None:
        """Record that a buff was cast (avoid recasting)."""
        now = __import__("time").time()
        self._active_buffs.setdefault(bot_id, {})[skill_id] = now + duration

    def get_buff_command(self, skill_id: str, target: str = "self") -> str:
        """Generate a buff casting command."""
        if target and target != "self":
            return f"use_skill {skill_id} on {target}"
        return f"use_skill {skill_id}"

    def get_class_buffs(self, job_name: str) -> list[dict]:
        """Get all buff definitions for a job class."""
        return _CLASS_BUFFS.get(job_name.lower(), [])

    def is_buff_active(self, bot_id: str, skill_id: str) -> bool:
        """Check if a buff is still active (by recorded cast time)."""
        bot_buffs = self._active_buffs.get(bot_id, {})
        expiry = bot_buffs.get(skill_id, 0)
        return expiry > __import__("time").time()

    def reset_buffs(self, bot_id: str) -> None:
        """Reset all buff tracking for a bot."""
        if bot_id in self._active_buffs:
            self._active_buffs[bot_id] = {}

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove buff state for a bot."""
        self._active_buffs.pop(bot_id, None)

# Alias for compatibility
BuffManager = AutoBuffManager
