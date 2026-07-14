"""
Guild Integration — join a guild, participate in guild activities,
benefit from guild skills and storage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class GuildSkill:
    """A guild skill that provides stat bonuses."""
    skill_name: str
    level: int = 0
    max_level: int = 5
    stat_bonus: str = ""
    bonus_value: int = 0
    description: str = ""


@dataclass
class GuildInfo:
    """Information about the bot's guild."""
    guild_name: str = ""
    guild_id: int = 0
    guild_level: int = 0
    member_count: int = 0
    online_count: int = 0
    skills: list[GuildSkill] = field(default_factory=list)
    has_storage: bool = False
    storage_items: int = 0
    alliance_count: int = 0
    enemy_count: int = 0
    last_activity: str = ""


class GuildManager:
    """Manages guild integration."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._guild: GuildInfo | None = None
        self._is_member: bool = False
        self._guild_skills: dict[str, GuildSkill] = {}
        self._last_guild_check: float = 0.0
        self._guild_check_interval: float = 60.0
        self._enqueue_fn: Callable | None = None
        self._load_known_guild_skills()

    def _load_known_guild_skills(self) -> None:
        """Load guild skills from the knowledge database."""
        try:
            from ai_sidecar.knowledge_loader import get_guild_skills
            db_skills = get_guild_skills()
            for skill in db_skills:
                name = skill.get("Name", "")
                max_lv = skill.get("MaxLevel", 5)
                desc = skill.get("Description", "")
                self._guild_skills[name] = GuildSkill(
                    skill_name=name, max_level=max_lv, description=desc,
                )
            logger.info("guild_skills_loaded_from_db: %d skills", len(self._guild_skills))
        except Exception as e:
            logger.warning("guild_skills_db_load_failed: %s (DB is the source of truth)", e)

    # ── Public API ──

    def update_guild_info(self, guild_data: dict) -> None:
        """Update guild information from snapshot data."""
        with self._lock:
            self._guild = GuildInfo(
                guild_name=str(guild_data.get("name", "")),
                guild_id=int(guild_data.get("id", 0)),
                guild_level=int(guild_data.get("level", 0)),
                member_count=int(guild_data.get("members", 0)),
                online_count=int(guild_data.get("online", 0)),
                has_storage=bool(guild_data.get("has_storage", False)),
                storage_items=int(guild_data.get("storage_items", 0)),
                alliance_count=int(guild_data.get("alliances", 0)),
                enemy_count=int(guild_data.get("enemies", 0)),
            )
            self._is_member = bool(guild_data.get("is_member", False))
            self._last_guild_check = time.time()

    def get_guild_info(self) -> GuildInfo | None:
        with self._lock:
            return self._guild

    def is_in_guild(self) -> bool:
        with self._lock:
            return self._is_member

    def get_guild_skill_bonus(self, stat: str) -> int:
        """Get total bonus for a stat from all guild skills."""
        with self._lock:
            total = 0
            for skill in self._guild_skills.values():
                if skill.stat_bonus == stat and skill.level > 0:
                    total += skill.bonus_value * skill.level
            return total

    def get_all_bonuses(self) -> dict[str, int]:
        """Get all stat bonuses from guild skills."""
        with self._lock:
            bonuses: dict[str, int] = {}
            for skill in self._guild_skills.values():
                if skill.level > 0:
                    key = skill.stat_bonus
                    bonuses[key] = bonuses.get(key, 0) + skill.bonus_value * skill.level
            return bonuses

    def get_guild_summary(self) -> str:
        with self._lock:
            if not self._guild:
                return "Not in a guild"
            g = self._guild
            lines = [f"── Guild: {g.guild_name} ──"]
            lines.append(f"Level {g.guild_level} | {g.member_count} members ({g.online_count} online)")
            lines.append(f"Storage: {'Yes' if g.has_storage else 'No'} ({g.storage_items} items)")
            lines.append(f"Alliances: {g.alliance_count} | Enemies: {g.enemy_count}")
            bonuses = self.get_all_bonuses()
            if bonuses:
                lines.append("Bonuses: " + ", ".join(f"{k}+{v}" for k, v in bonuses.items()))
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._guild = None
            self._is_member = False
            self._last_guild_check = 0.0


# ── Global Singleton ──

_guild_mgr: GuildManager | None = None
_guild_mgr_lock = RLock()


def get_guild_manager() -> GuildManager:
    global _guild_mgr
    with _guild_mgr_lock:
        if _guild_mgr is None:
            _guild_mgr = GuildManager()
        return _guild_mgr
