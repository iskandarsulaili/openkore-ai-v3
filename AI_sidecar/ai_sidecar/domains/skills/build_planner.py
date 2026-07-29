"""Skill Build Planner — class-specific skill builds from level 1-99.

Data-driven from build_plans.yaml and ro_mechanics.yaml.
Provides build priorities for every class progression path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_DATA = Path(__file__).parent.parent.parent / "data"

# Pre-renewal class progression: what each first job becomes
CLASS_PROGRESSION = {
    "novice": "novice",
    "swordman": "knight", "knight": "lord_knight",
    "mage": "wizard", "wizard": "high_wizard",
    "archer": "hunter", "hunter": "sniper",
    "thief": "assassin", "assassin": "assassin_cross",
    "acolyte": "priest", "priest": "high_priest",
    "merchant": "blacksmith", "blacksmith": "mastersmith",
    "taekwon": "soul_linker",
    "gunslinger": "gunslinger",
    "ninja": "ninja",
}

# Default skill builds for all classes (used when YAML data missing)
DEFAULT_BUILDS: dict[str, list[dict[str, Any]]] = {
    "archer": [
        {"skill_id": "OWLS_EYE", "max_level": 10, "priority": 1, "reason": "DEX+30, accuracy and damage"},
        {"skill_id": "VULTURES_EYE", "max_level": 10, "priority": 2, "reason": "+10 attack range"},
        {"skill_id": "IMPROVE_CONCENTRATION", "max_level": 10, "priority": 3, "reason": "DEX+12, AGI+12, falcon"},
        {"skill_id": "ARROW_SHOWER", "max_level": 10, "priority": 4, "reason": "AoE farming skill"},
        {"skill_id": "DOUBLE_STRAFING", "max_level": 10, "priority": 5, "reason": "Burst single-target"},
    ],
    "swordman": [
        {"skill_id": "SM_SWORD", "max_level": 10, "priority": 1, "reason": "Sword mastery +20 ATK"},
        {"skill_id": "SM_PROVOKE", "max_level": 5, "priority": 2, "reason": "Reduce enemy DEF"},
        {"skill_id": "SM_BASH", "max_level": 10, "priority": 3, "reason": "Main attack skill"},
        {"skill_id": "SM_MAGNUM", "max_level": 10, "priority": 4, "reason": "AoE attack"},
        {"skill_id": "HP_RECOVERY", "max_level": 10, "priority": 5, "reason": "HP regen while sitting"},
    ],
    "mage": [
        {"skill_id": "MG_SRECOVERY", "max_level": 10, "priority": 1, "reason": "SP regen"},
        {"skill_id": "MG_FIREBOLT", "max_level": 10, "priority": 2, "reason": "Fire damage, good vs undead"},
        {"skill_id": "MG_COLDBOLT", "max_level": 10, "priority": 3, "reason": "Water damage, good vs fire"},
        {"skill_id": "MG_LIGHTNINGBOLT", "max_level": 10, "priority": 4, "reason": "Wind damage, good vs water"},
        {"skill_id": "MG_FIREWALL", "max_level": 10, "priority": 5, "reason": "Defensive wall"},
    ],
    "thief": [
        {"skill_id": "TF_DOUBLE", "max_level": 10, "priority": 1, "reason": "Double attack proc"},
        {"skill_id": "TF_MISS", "max_level": 10, "priority": 2, "reason": "Improve flee +20"},
        {"skill_id": "TF_STEAL", "max_level": 5, "priority": 3, "reason": "Steal from monsters"},
        {"skill_id": "TF_HIDING", "max_level": 10, "priority": 4, "reason": "Stealth/invisibility"},
        {"skill_id": "TF_POISON", "max_level": 10, "priority": 5, "reason": "Poison attack"},
    ],
    "acolyte": [
        {"skill_id": "AL_HEAL", "max_level": 10, "priority": 1, "reason": "Healing skill"},
        {"skill_id": "AL_TELEPORT", "max_level": 4, "priority": 2, "reason": "Teleport skill (4 = directional)"},
        {"skill_id": "AL_WARP", "max_level": 4, "priority": 3, "reason": "Warp portal creation"},
        {"skill_id": "AL_BLESSING", "max_level": 10, "priority": 4, "reason": "STAT +10 buff"},
        {"skill_id": "AL_INC_AGI", "max_level": 10, "priority": 5, "reason": "AGI +12 buff"},
    ],
    "merchant": [
        {"skill_id": "MC_VENDING", "max_level": 10, "priority": 1, "reason": "Set up shop"},
        {"skill_id": "MC_DISCOUNT", "max_level": 10, "priority": 2, "reason": "Buy items cheaper"},
        {"skill_id": "MC_OVERCHARGE", "max_level": 10, "priority": 3, "reason": "Sell items for more"},
        {"skill_id": "MC_PUSHCART", "max_level": 10, "priority": 4, "reason": "Extra inventory"},
        {"skill_id": "MC_IDENTIFY", "max_level": 1, "priority": 5, "reason": "Identify items"},
    ],
    "novice": [
        {"skill_id": "NV_BASIC", "max_level": 1, "priority": 1, "reason": "Basic Skill for sitting"},
        {"skill_id": "NV_FIRST_AID", "max_level": 1, "priority": 2, "reason": "First Aid"},
        {"skill_id": "NV_TRICKDEAD", "max_level": 1, "priority": 3, "reason": "Trick Dead"},
    ],
}


class SkillBuildPlanner:
    """Data-driven skill build planner for all RO classes.
    
    Provides class-specific skill build priorities from level 1 to 99.
    Falls back to DEFAULT_BUILDS when YAML data is unavailable.
    """
    
    def __init__(self, data_path: str | Path | None = None) -> None:
        self._lock = RLock()
        self._data_path = Path(data_path or _DEFAULT_DATA)
        self._builds: dict[str, Any] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load build plans from YAML."""
        # Try build_plans.yaml first
        build_path = self._data_path / "build_plans.yaml"
        if build_path.exists():
            try:
                with open(build_path) as f:
                    data = yaml.safe_load(f) or {}
                    self._builds = data.get("skill_builds", data.get("builds", data))
                logger.info("build_planner_loaded: path=%s", build_path)
            except Exception as e:
                logger.warning("build_planner_yaml_failed: %s", e)
        
        # Also try ro_mechanics.yaml for skill data
        mech_path = self._data_path / "ro_mechanics.yaml"
        if mech_path.exists():
            try:
                with open(mech_path) as f:
                    data = yaml.safe_load(f) or {}
                    # Merge any skill builds from mechanics data
                    mech_builds = data.get("skill_builds", {})
                    if mech_builds:
                        self._builds.update(mech_builds)
            except Exception:
                pass
    
    # ── Public API ──
    
    def next_skill_to_learn(
        self, job: str, current_skills: dict[str, int]
    ) -> dict[str, Any] | None:
        """Determine the next skill this character should learn.
        
        Args:
            job: Character's class (lowercase, e.g., 'archer', 'swordman')
            current_skills: dict of {skill_id: current_level}
            
        Returns:
            dict with skill_id, target_level, reason, priority or None if all maxed
        """
        with self._lock:
            build = self._get_build(job)
            if not build:
                return None
            
            for entry in build:
                sid = entry["skill_id"]
                max_lv = entry["max_level"]
                current = current_skills.get(sid, 0)
                
                if current < max_lv:
                    return {
                        "skill_id": sid,
                        "current_level": current,
                        "target_level": max_lv,
                        "next_level": current + 1,
                        "reason": entry.get("reason", f"Priority {entry.get('priority', 99)}"),
                        "priority": entry.get("priority", 99),
                    }
            
            return None  # All skills maxed
    
    def get_build(self, job: str) -> list[dict[str, Any]]:
        """Get the full skill build plan for a class."""
        with self._lock:
            return list(self._get_build(job))
    
    def get_progression_path(self, first_job: str) -> list[str]:
        """Get the class progression path (e.g., archer -> hunter -> sniper)."""
        path = [first_job]
        next_job = CLASS_PROGRESSION.get(first_job.lower(), None)
        if next_job and next_job != first_job.lower():
            path.append(next_job)
            next_next = CLASS_PROGRESSION.get(next_job, None)
            if next_next and next_next != next_job:
                path.append(next_next)
        return path
    
    def get_stat_priority(self, job: str) -> list[str]:
        """Get stat allocation priority for this class.
        Returns ordered list: highest priority first.
        """
        default_priorities = {
            "archer": ["dex", "agi", "str", "vit", "int", "luk"],
            "hunter": ["dex", "agi", "str", "vit", "int", "luk"],
            "sniper": ["dex", "agi", "int", "str", "vit", "luk"],
            "swordman": ["str", "vit", "dex", "agi", "int", "luk"],
            "knight": ["str", "vit", "dex", "int", "agi", "luk"],
            "lord_knight": ["str", "vit", "dex", "agi", "int", "luk"],
            "mage": ["int", "dex", "vit", "str", "agi", "luk"],
            "wizard": ["int", "dex", "vit", "str", "agi", "luk"],
            "high_wizard": ["int", "dex", "vit", "str", "agi", "luk"],
            "thief": ["agi", "dex", "str", "vit", "int", "luk"],
            "assassin": ["agi", "str", "dex", "vit", "int", "luk"],
            "assassin_cross": ["str", "agi", "dex", "vit", "int", "luk"],
            "acolyte": ["int", "dex", "vit", "str", "agi", "luk"],
            "priest": ["int", "dex", "vit", "str", "agi", "luk"],
            "high_priest": ["int", "dex", "vit", "str", "agi", "luk"],
            "merchant": ["str", "dex", "vit", "int", "agi", "luk"],
            "blacksmith": ["str", "dex", "vit", "int", "agi", "luk"],
            "mastersmith": ["str", "dex", "vit", "int", "agi", "luk"],
        }
        return list(default_priorities.get(job.lower(), ["str", "agi", "dex", "vit", "int", "luk"]))
    
    def auto_assign_command(self, job: str, skill_points: int = 0) -> str | None:
        """Generate an auto-assign command for skill points.
        
        Returns a command string like 'setAutoSkill 1' or None if no points.
        """
        if skill_points <= 0:
            return None
        return "setAutoSkill 1"
    
    def auto_stat_command(self, job: str, stat_points: int = 0) -> str | None:
        """Generate stat point assignment command based on class priority.
        
        Returns command like 'str 1' or None if no points.
        """
        if stat_points <= 0:
            return None
        priority = self.get_stat_priority(job)
        if priority:
            return f"stat_add {priority[0]} 1"
        return None
    
    def recommendation_for_level(
        self, job: str, level: int, current_skills: dict[str, int]
    ) -> dict[str, Any]:
        """Get a full recommendation for what to do at this level.
        
        Returns dict with next_skill, stat_priority, progression_path, etc.
        """
        with self._lock:
            next_skill = self.next_skill_to_learn(job, current_skills)
            path = self.get_progression_path(job)
            stat_prio = self.get_stat_priority(job)
            build = self._get_build(job)
            total_points = sum(s["max_level"] - current_skills.get(s["skill_id"], 0) for s in build) if build else 0
            
            return {
                "job": job,
                "level": level,
                "next_skill": next_skill,
                "stat_priority": stat_prio,
                "progression_path": path,
                "skill_points": total_points,
            }
    
    def _get_build(self, job: str) -> list[dict[str, Any]]:
        """Get build plan for a class, falling back to defaults."""
        if not job:
            return DEFAULT_BUILDS.get("novice", [])
        
        job_lower = job.lower().replace(" ", "_")
        
        # Try YAML data first
        for key in [job_lower, job_lower.upper(), job.upper()]:
            if key in self._builds:
                build = self._builds[key]
                if isinstance(build, list):
                    return build
                if isinstance(build, dict):
                    return build.get("priority_skills", build.get("skills", []))
        
        # Fall back to defaults
        return list(DEFAULT_BUILDS.get(job_lower, DEFAULT_BUILDS.get("novice", [])))
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "builds_loaded": len(self._builds),
            "default_builds": len(DEFAULT_BUILDS),
        }


def create_build_planner(data_path: str | None = None) -> SkillBuildPlanner:
    """Factory function."""
    return SkillBuildPlanner(data_path=data_path)
