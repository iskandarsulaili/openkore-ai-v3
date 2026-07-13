"""
Build philosophy manager — coherent character builds with stat/skill/gear optimization.

The LLM selects a build at job change and optimizes every decision toward it.
Builds are not hardcoded — the LLM can create novel builds and track performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Pre-defined build templates
BUILD_TEMPLATES: dict[str, dict[str, Any]] = {
    "agi_knight": {
        "name": "AGI Knight",
        "classes": ["swordman", "knight", "lord_knight", "rune_knight"],
        "stat_priority": ["AGI", "STR", "VIT", "DEX", "INT", "LUK"],
        "stat_targets": {"AGI": 99, "STR": 80, "VIT": 50, "DEX": 40},
        "skill_priority": ["bash", "magnum_break", "two_hand_quicken", "aura_blade", "bowling_bash"],
        "weapon_priority": ["sword", "two_hand_sword"],
        "armor_priority": ["body", "shoes", "robe"],
        "card_priority": ["drainliar", "hydra", "skel_worker"],
        "description": "High attack speed, good DPS, self-buffing",
    },
    "int_mage": {
        "name": "INT Mage",
        "classes": ["mage", "wizard", "high_wizard", "warlock"],
        "stat_priority": ["INT", "DEX", "VIT", "AGI", "STR", "LUK"],
        "stat_targets": {"INT": 99, "DEX": 60, "VIT": 40},
        "skill_priority": ["cold_bolt", "fire_bolt", "lightning_bolt", "frost_diver", "fire_ball", "storm_gust"],
        "weapon_priority": ["staff", "two_hand_staff"],
        "armor_priority": ["body", "shoes", "head"],
        "card_priority": ["drainliar", "marse", "sohee"],
        "description": "High magic damage, AoE farming, elemental advantage",
    },
    "dex_hunter": {
        "name": "DEX Hunter",
        "classes": ["archer", "hunter", "sniper", "ranger"],
        "stat_priority": ["DEX", "AGI", "INT", "VIT", "STR", "LUK"],
        "stat_targets": {"DEX": 99, "AGI": 70, "INT": 30},
        "skill_priority": ["double_strafing", "arrow_shower", "blitz_beat", "falcon_assault"],
        "weapon_priority": ["bow"],
        "armor_priority": ["body", "shoes", "garment"],
        "card_priority": ["drainliar", "hydra", "skel_worker"],
        "description": "Ranged DPS, falcon support, versatile farming",
    },
    "full_support_priest": {
        "name": "Full Support Priest",
        "classes": ["acolyte", "priest", "high_priest", "arch_bishop"],
        "stat_priority": ["INT", "DEX", "VIT", "AGI", "STR", "LUK"],
        "stat_targets": {"INT": 99, "DEX": 70, "VIT": 50},
        "skill_priority": ["heal", "magnificat", "kyrie_eleison", "gloria", "resurrection", "turn_undead"],
        "weapon_priority": ["mace", "staff"],
        "armor_priority": ["body", "shoes", "head"],
        "card_priority": ["drainliar", "sohee"],
        "description": "Party healer, buffer, undead hunter",
    },
    "str_merchant": {
        "name": "STR Merchant",
        "classes": ["merchant", "blacksmith", "whitesmith", "mechanic"],
        "stat_priority": ["STR", "DEX", "VIT", "AGI", "INT", "LUK"],
        "stat_targets": {"STR": 99, "DEX": 50, "VIT": 50},
        "skill_priority": ["bash", "magnum_break", "weapon_perfection", "over_thrust", "cart_revolution"],
        "weapon_priority": ["axe", "two_hand_axe", "mace"],
        "armor_priority": ["body", "shoes", "shoulder"],
        "card_priority": ["drainliar", "hydra"],
        "description": "High melee damage, crafting, cart combat",
    },
    "dex_assassin": {
        "name": "DEX Assassin",
        "classes": ["thief", "assassin", "assassin_cross", "guillotine_cross"],
        "stat_priority": ["STR", "AGI", "DEX", "VIT", "INT", "LUK"],
        "stat_targets": {"STR": 80, "AGI": 99, "DEX": 40},
        "skill_priority": ["double_attack", "sonic_blow", "hiding", "venom_dust", "grimtooth"],
        "weapon_priority": ["dagger", "katar"],
        "armor_priority": ["body", "shoes", "garment"],
        "card_priority": ["drainliar", "hydra", "skel_worker"],
        "description": "High burst damage, poison, stealth",
    },
}


@dataclass(slots=True)
class BuildManager:
    """Manages character builds — stat/skill/gear optimization toward a coherent build."""
    
    _lock: RLock = field(default_factory=RLock)
    _active_builds: dict[str, dict[str, Any]] = field(default_factory=dict)  # bot_id -> build
    _build_performance: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # build_name -> [metrics]
    
    def get_available_builds(self, player_class: str) -> list[dict[str, Any]]:
        """Get builds available for a given class."""
        available = []
        for name, template in BUILD_TEMPLATES.items():
            if player_class.lower() in [c.lower() for c in template["classes"]]:
                available.append({"name": name, **template})
        return available
    
    def select_build(self, bot_id: str, build_name: str) -> bool:
        """Select a build for a bot."""
        template = BUILD_TEMPLATES.get(build_name)
        if template is None:
            return False
        with self._lock:
            self._active_builds[bot_id] = {
                "name": build_name,
                "template": template,
                "selected_at": __import__("time").time(),
                "stats_allocated": {},
                "skills_learned": [],
                "gear_acquired": [],
            }
        logger.info("build_selected: bot=%s build=%s", bot_id, build_name)
        return True
    
    def get_next_stat(self, bot_id: str, current_stats: dict[str, int]) -> str | None:
        """Get the next stat to allocate based on build priority."""
        build = self._active_builds.get(bot_id)
        if build is None:
            return None
        
        template = build.get("template", {})
        priority = template.get("stat_priority", [])
        targets = template.get("stat_targets", {})
        
        for stat in priority:
            current = current_stats.get(stat, 0)
            target = targets.get(stat, 99)
            if current < target:
                return stat
        
        return None
    
    def get_next_skill(self, bot_id: str, known_skills: list[str]) -> str | None:
        """Get the next skill to learn based on build priority."""
        build = self._active_builds.get(bot_id)
        if build is None:
            return None
        
        priority = build.get("template", {}).get("skill_priority", [])
        for skill in priority:
            if skill not in known_skills:
                return skill
        
        return None
    
    def get_weapon_type(self, bot_id: str) -> str | None:
        """Get the preferred weapon type for this build."""
        build = self._active_builds.get(bot_id)
        if build is None:
            return None
        priority = build.get("template", {}).get("weapon_priority", [])
        return priority[0] if priority else None
    
    def record_performance(self, build_name: str, metrics: dict[str, Any]) -> None:
        """Record performance metrics for a build."""
        with self._lock:
            if build_name not in self._build_performance:
                self._build_performance[build_name] = []
            self._build_performance[build_name].append(metrics)
            self._build_performance[build_name] = self._build_performance[build_name][-100:]
    
    def get_build_performance(self, build_name: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._build_performance.get(build_name, []))
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return {"active_builds": len(self._active_builds), "tracked_builds": len(self._build_performance)}
