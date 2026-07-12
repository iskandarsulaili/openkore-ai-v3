"""JobPlanner — Job change planning and progression tracking for OpenKore bots.

Maps each class to its advancement path through Ragnarok Online's job
system: Novice -> First Class -> Second Class -> Transcendent -> 3rd Class
-> 4th Class. Tracks job change requirements per class and provides NPC
location data for executing job change quests.
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Job Advancement Paths ──────────────────────────────────────────────
# Maps each class to its next advancement(s).
# Format: current_class -> { next_class_name: (map_name, npc_x, npc_y) }

_JOB_ADVANCEMENT: dict[str, dict[str, tuple[str, int, int]]] = {
    # Novice -> First classes (job level 10+)
    "novice": {
        "swordman": ("prontera", 163, 42),
        "mage": ("geffen", 60, 140),
        "archer": ("payon", 186, 230),
        "acolyte": ("prontera", 195, 230),
        "merchant": ("alberta", 48, 152),
        "thief": ("morocc", 144, 306),
    },
    # First classes -> Second classes (base 40+, job level 50)
    "swordman": {
        "knight": ("prontera", 170, 30),
        "crusader": ("payon", 100, 200),
    },
    "mage": {
        "wizard": ("geffen", 50, 130),
        "sage": ("geffen", 70, 150),
    },
    "archer": {
        "hunter": ("payon", 150, 240),
        "bard": ("payon", 165, 255),
        "dancer": ("payon", 165, 255),
    },
    "acolyte": {
        "priest": ("prontera", 195, 230),
        "monk": ("morocc", 120, 280),
    },
    "merchant": {
        "blacksmith": ("geffen", 80, 160),
        "alchemist": ("aldebaran", 100, 100),
    },
    "thief": {
        "assassin": ("morocc", 120, 280),
        "rogue": ("alberta", 60, 160),
    },
    # Second classes -> Transcendent / High classes (base 50+)
    "knight": {
        "lord_knight": ("prontera", 170, 30),
    },
    "crusader": {
        "paladin": ("payon", 100, 200),
    },
    "wizard": {
        "high_wizard": ("geffen", 50, 130),
    },
    "sage": {
        "professor": ("geffen", 70, 150),
    },
    "hunter": {
        "sniper": ("payon", 150, 240),
    },
    "bard": {
        "clown": ("payon", 165, 255),
        "minstrel": ("payon", 165, 255),
    },
    "dancer": {
        "gypsy": ("payon", 165, 255),
        "wanderer": ("payon", 165, 255),
    },
    "priest": {
        "high_priest": ("prontera", 195, 230),
    },
    "monk": {
        "champion": ("morocc", 120, 280),
    },
    "blacksmith": {
        "whitesmith": ("geffen", 80, 160),
    },
    "alchemist": {
        "creator": ("aldebaran", 100, 100),
    },
    "assassin": {
        "assassin_cross": ("morocc", 120, 280),
    },
    "rogue": {
        "stalker": ("alberta", 60, 160),
    },
}

# ── Transcendent / High first class -> 3rd class ──
_HIGH_FIRST_TO_THIRD: dict[str, dict[str, tuple[str, int, int]]] = {
    "swordman_high": {"rune_knight": ("prontera", 170, 30)},
    "mage_high": {"warlock": ("geffen", 50, 130)},
    "archer_high": {"ranger": ("payon", 150, 240)},
    "acolyte_high": {"arch_bishop": ("prontera", 195, 230)},
    "merchant_high": {"mechanic": ("geffen", 80, 160)},
    "thief_high": {"guillotine_cross": ("morocc", 120, 280)},
}

# ── 3rd class -> 4th class ──
_THIRD_TO_FOURTH: dict[str, dict[str, tuple[str, int, int]]] = {
    "rune_knight": {"dragon_knight": ("prontera", 170, 30)},
    "warlock": {"arch_mage": ("geffen", 50, 130)},
    "ranger": {"windhawk": ("payon", 150, 240)},
    "arch_bishop": {"cardinal": ("prontera", 195, 230)},
    "mechanic": {"meister": ("geffen", 80, 160)},
    "guillotine_cross": {"shadow_cross": ("morocc", 120, 280)},
    "paladin": {"imperial_guard": ("payon", 100, 200)},
    "professor": {"elemental_master": ("geffen", 70, 150)},
    "minstrel": {"troubadour": ("payon", 165, 255)},
    "wanderer": {"trouvere": ("payon", 165, 255)},
    "champion": {"inquisitor": ("morocc", 120, 280)},
    "whitesmith": {"meister": ("geffen", 80, 160)},
    "creator": {"biolo": ("aldebaran", 100, 100)},
    "assassin_cross": {"guillotine_cross": ("morocc", 120, 280)},
    "stalker": {"shadow_chaser": ("alberta", 60, 160)},
    "clown": {"minstrel": ("payon", 165, 255)},
    "gypsy": {"wanderer": ("payon", 165, 255)},
    "sniper": {"ranger": ("payon", 150, 240)},
    "high_priest": {"arch_bishop": ("prontera", 195, 230)},
    "high_wizard": {"warlock": ("geffen", 50, 130)},
    "lord_knight": {"rune_knight": ("prontera", 170, 30)},
}

# ── Job Tier Requirements ──────────────────────────────────────────────
# For each class, defines the level requirements to advance.
# The advancement function maps the class to its expected job tier.

_JOB_TIERS: dict[str, dict[str, int | str]] = {
    # Tier 0: Novice -> First class
    "novice": {"tier": 0, "min_base_level": 1, "min_job_level": 10},
    # Tier 1: First class -> Second class
    "swordman": {"tier": 1, "min_base_level": 40, "min_job_level": 50},
    "mage": {"tier": 1, "min_base_level": 40, "min_job_level": 50},
    "archer": {"tier": 1, "min_base_level": 40, "min_job_level": 50},
    "acolyte": {"tier": 1, "min_base_level": 40, "min_job_level": 50},
    "merchant": {"tier": 1, "min_base_level": 40, "min_job_level": 50},
    "thief": {"tier": 1, "min_base_level": 40, "min_job_level": 50},
    # Tier 2: Second class -> Transcendent / High
    "knight": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "crusader": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "wizard": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "sage": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "hunter": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "bard": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "dancer": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "priest": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "monk": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "blacksmith": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "alchemist": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "assassin": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "rogue": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    "paladin": {"tier": 2, "min_base_level": 99, "min_job_level": 50},
    # Tier 3: 3rd classes
    "rune_knight": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "warlock": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "ranger": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "arch_bishop": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "mechanic": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "guillotine_cross": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "shadow_chaser": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "minstrel": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "wanderer": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "sura": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "genetic": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "shadow_cross": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "royal_guard": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    "sorcerer": {"tier": 3, "min_base_level": 99, "min_job_level": 70},
    # Tier 4: 4th classes
    "dragon_knight": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "arch_mage": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "windhawk": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "cardinal": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "meister": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "imperial_guard": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "elemental_master": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "troubadour": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "trouvere": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "inquisitor": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
    "biolo": {"tier": 4, "min_base_level": 200, "min_job_level": 100},
}

# ── NPC Name Patterns ──────────────────────────────────────────────────
_JOB_CHANGE_NPC_NAMES: dict[str, str] = {
    "swordman": "Captain of the Guard",
    "mage": "Wizard Guild Master",
    "archer": "Archer Guild Master",
    "acolyte": "High Priest",
    "merchant": "Merchant Guild Master",
    "thief": "Thief Guild Master",
    "knight": "Knight Guild Master",
    "crusader": "Crusader Guild Master",
    "wizard": "Sage of Geffen",
    "sage": "Sage Guild Master",
    "hunter": "Hunter Guild Master",
    "bard": "Bard Guild Master",
    "dancer": "Dancer Guild Master",
    "priest": "Pope",
    "monk": "Monk Guild Master",
    "blacksmith": "Blacksmith Guild Master",
    "alchemist": "Alchemist Guild Master",
    "assassin": "Assassin Guild Master",
    "rogue": "Rogue Guild Master",
    "paladin": "Paladin Guild Master",
    "lord_knight": "Rune Knight Guild Master",
    "high_wizard": "Arch Mage Guild Master",
    "high_priest": "Cardinal Guild Master",
    "sniper": "Ranger Guild Master",
    "clown": "Minstrel Guild Master",
    "gypsy": "Wanderer Guild Master",
    "champion": "Sura Guild Master",
    "whitesmith": "Meister Guild Master",
    "creator": "Genetic Guild Master",
    "assassin_cross": "Guillotine Cross Guild Master",
    "stalker": "Shadow Chaser Guild Master",
    "rune_knight": "Rune Knight Master",
    "warlock": "Arch Mage Master",
    "ranger": "Windhawk Master",
    "arch_bishop": "Cardinal Master",
    "mechanic": "Meister Master",
    "guillotine_cross": "Shadow Cross Master",
}


class JobPlanner:
    """Thread-safe job change planning and progression tracking.

    Maps Ragnarok Online's job advancement tree:
        Novice -> First Class -> Second Class
        -> Transcendent/High -> 3rd Class -> 4th Class

    Provides readiness checks, next-job lookups, and NPC location data.
    """

    def __init__(self) -> None:
        self._lock = RLock()

    # ── Public API ──────────────────────────────────────────────────────

    def check_readiness(
        self,
        bot_id: str,
        base_level: int,
        job_level: int,
        class_name: str,
    ) -> dict[str, Any]:
        """Check if a bot is ready for job advancement.

        Returns a dict with:
        - ``can_change``: ``True`` if requirements are met
        - ``next_class``: The next class name, or ``None`` at max tier
        - ``current_class``: Normalised class name
        - ``requirements``: Dict of unmet requirements
        - ``npc_info``: NPC location data if available
        """
        class_key = class_name.lower().strip()
        tier_info = _JOB_TIERS.get(class_key, {})
        current_tier = tier_info.get("tier", 0)

        # Determine next class
        next_class = self.get_next_job(class_key)

        # Build requirements status
        requirements: dict[str, Any] = {
            "base_level": {"current": base_level, "required": tier_info.get("min_base_level", 0)},
            "job_level": {"current": job_level, "required": tier_info.get("min_job_level", 0)},
            "has_quest_items": True,  # Placeholder — bridge v2 will supply quest items
            "quest_items": [],
        }

        base_ok = base_level >= (tier_info.get("min_base_level", 0) or 0)
        job_ok = job_level >= (tier_info.get("min_job_level", 0) or 0)
        can_change = bool(base_ok and job_ok and next_class is not None)

        # Get NPC info for the next job
        npc_info = self._lookup_npc(class_key, next_class) if next_class else None

        result: dict[str, Any] = {
            "can_change": can_change,
            "next_class": next_class,
            "current_class": class_key,
            "current_tier": current_tier,
            "requirements": requirements,
            "npc_info": npc_info,
            "blockers": [],
        }

        # Enrich with specific blockers
        if not base_ok:
            result["blockers"].append(
                f"need base level {tier_info.get('min_base_level', 0)} "
                f"(have {base_level})"
            )
        if not job_ok:
            result["blockers"].append(
                f"need job level {tier_info.get('min_job_level', 0)} "
                f"(have {job_level})"
            )
        if next_class is None:
            result["blockers"].append(f"'{class_key}' has no known advancement")

        logger.info(
            "job_planner[%s]: %s -> %s (tier %d) can_change=%s",
            bot_id, class_key, next_class or "(max)", current_tier, can_change,
        )

        return result

    def get_next_job(self, current_class: str) -> str | None:
        """Return the next job class name for advancement.

        Returns ``None`` if the class is at the maximum known tier.
        """
        class_key = current_class.lower().strip()

        # Check in high-first -> third
        if class_key in _HIGH_FIRST_TO_THIRD:
            targets = _HIGH_FIRST_TO_THIRD[class_key]
            if targets:
                return next(iter(targets.keys()))

        # Check in third -> fourth
        if class_key in _THIRD_TO_FOURTH:
            targets = _THIRD_TO_FOURTH[class_key]
            if targets:
                return next(iter(targets.keys()))

        # Check in standard advancement
        if class_key in _JOB_ADVANCEMENT:
            targets = _JOB_ADVANCEMENT[class_key]
            if targets:
                return next(iter(targets.keys()))

        return None

    def get_all_advancements(self, current_class: str) -> dict[str, tuple[str, int, int]]:
        """Return all possible advancement targets for a class.

        Returns a dict of ``{next_class: (map, x, y)}`` or empty dict
        if no advancements are known.
        """
        class_key = current_class.lower().strip()

        combined: dict[str, tuple[str, int, int]] = {}
        combined.update(_JOB_ADVANCEMENT.get(class_key, {}))
        combined.update(_HIGH_FIRST_TO_THIRD.get(class_key, {}))
        combined.update(_THIRD_TO_FOURTH.get(class_key, {}))

        return combined

    def get_job_change_npc(self, current_class: str) -> dict[str, Any]:
        """Get NPC location and name for job change.

        Returns a dict with:
        - ``npc_location``: ``(map_name, x, y)`` tuple
        - ``npc_name``: NPC name string pattern
        - ``available_jobs``: List of next jobs with their NPC locations
        """
        class_key = current_class.lower().strip()
        advancements = self.get_all_advancements(class_key)

        if not advancements:
            return {
                "npc_location": None,
                "npc_name": None,
                "available_jobs": [],
                "note": f"No known advancement from '{class_key}'",
            }

        available_jobs_list = []
        for job_name, (map_name, x, y) in advancements.items():
            npc_name = _JOB_CHANGE_NPC_NAMES.get(
                job_name,
                _JOB_CHANGE_NPC_NAMES.get(class_key, f"{job_name} Guild Master"),
            )
            available_jobs_list.append({
                "job_name": job_name,
                "npc_map": map_name,
                "npc_x": x,
                "npc_y": y,
                "npc_name": npc_name,
            })

        first = advancements[next(iter(advancements))]
        first_npc_name = _JOB_CHANGE_NPC_NAMES.get(
            next(iter(advancements)),
            f"{next(iter(advancements)).replace('_', ' ').title()} Guild Master",
        )

        return {
            "npc_location": first,
            "npc_name": first_npc_name,
            "available_jobs": available_jobs_list,
        }

    def get_job_tier(self, class_name: str) -> int:
        """Return the numerical job tier for a class.

        Tier 0 = Novice
        Tier 1 = First class (Swordman, Mage, etc.)
        Tier 2 = Second class (Knight, Wizard, etc.) / Transcendent equivalents
        Tier 3 = 3rd class (Rune Knight, Warlock, etc.)
        Tier 4 = 4th class (Dragon Knight, Arch Mage, etc.)
        """
        info = _JOB_TIERS.get(class_name.lower().strip())
        if info:
            return int(info.get("tier", 0))
        return 0

    # ── Internal helpers ────────────────────────────────────────────────

    def _lookup_npc(
        self,
        current_class: str,
        next_class: str,
    ) -> dict[str, Any] | None:
        """Look up NPC information for a specific class transition."""
        advancements = self.get_all_advancements(current_class)
        npc_data = advancements.get(next_class)
        if not npc_data:
            return None

        map_name, x, y = npc_data
        npc_name = _JOB_CHANGE_NPC_NAMES.get(
            next_class,
            f"{next_class.replace('_', ' ').title()} Guild Master",
        )

        return {
            "map": map_name,
            "x": x,
            "y": y,
            "npc_name": npc_name,
        }
