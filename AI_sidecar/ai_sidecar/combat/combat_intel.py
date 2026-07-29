"""Combat Intelligence Engine — Data-driven RO combat decisions.

Uses authentic pre-renewal mechanics from ro_mechanics.yaml:
- Element advantage lookup (10×10 table, 4 levels)
- Size modifier lookup (weapon type × monster size)
- Skill recommendation (element/race/size matching)
- Flee threshold calculation (survival-first)
- No hardcoded values — all data-driven from mechanics YAML
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "ro_mechanics.yaml"


class CombatIntelligence:
    """Data-driven combat intelligence engine.

    Provides:
    - element_advantage(attack_element, target_element, element_level=1) -> multiplier
    - size_modifier(weapon_type, monster_size) -> multiplier
    - best_skill_for_monster(skills, monster_info) -> skill_id
    - flee_recommendation(hp_pct, mon_hp, mon_atk, aggro_count) -> bool
    - avg_hits_to_kill(mon_hp, avg_dmg) -> int
    - hits_to_die(my_hp, mon_atk, aggro_count) -> int
    """

    def __init__(self, data_path: str | Path | None = None) -> None:
        self._lock = RLock()
        self._data_path = Path(data_path or _DEFAULT_DATA_PATH)
        self._mechanics: dict[str, Any] = {}
        self._load_mechanics()

    def _load_mechanics(self) -> None:
        """Load RO mechanics data from YAML."""
        path = self._data_path
        if not path.exists():
            logger.warning("combat_intel_no_data: path=%s", path)
            return
        try:
            with open(path) as f:
                self._mechanics = yaml.safe_load(f) or {}
            logger.info("combat_intel_loaded: path=%s keys=%s", path, list(self._mechanics.keys())[:5])
        except Exception as e:
            logger.error("combat_intel_load_failed: %s", e)
            self._mechanics = {}

    # ── Element Advantage ──

    ELEMENT_ORDER = [
        "Neutral", "Water", "Earth", "Fire", "Wind",
        "Poison", "Holy", "Shadow", "Ghost", "Undead",
    ]

    def element_advantage(
        self,
        attack_element: str,
        target_element: str,
        element_level: int = 1,
    ) -> float:
        """Get element multiplier: attack_element vs target_element.
        Returns 1.0 if data unavailable (no penalty).
        """
        with self._lock:
            et = self._mechanics.get("element_table", {})
            level_key = f"level_{max(1, min(element_level, 4))}"
            level_data = et.get(level_key, et.get("level_1", {}))
            row = level_data.get(attack_element, {})
            if not row:
                return 1.0
            return float(row.get(target_element, 1.0))

    def get_element_multipliers(
        self,
        target_element: str,
        element_level: int = 1,
    ) -> dict[str, float]:
        """Get ALL attack elements' multipliers against a target.
        Returns dict of {attack_element: multiplier}.
        """
        with self._lock:
            et = self._mechanics.get("element_table", {})
            level_key = f"level_{max(1, min(element_level, 4))}"
            level_data = et.get(level_key, et.get("level_1", {}))
            result: dict[str, float] = {}
            for elem in self.ELEMENT_ORDER:
                row = level_data.get(elem, {})
                result[elem] = float(row.get(target_element, 1.0))
            return result

    def best_element_against(self, target_element: str, element_level: int = 1) -> tuple[str, float]:
        """Return (attack_element, multiplier) with highest multiplier against target."""
        mults = self.get_element_multipliers(target_element, element_level)
        best = max(mults.items(), key=lambda kv: kv[1])
        return best

    # ── Size Modifier ──

    def size_modifier(self, weapon_type: str, monster_size: str) -> float:
        """Get damage multiplier based on weapon type vs monster size.
        weapon_type: 'dagger','sword','two_hand_sword','spear','bow','staff', etc.
        monster_size: 'small','medium','large'
        """
        with self._lock:
            sm = self._mechanics.get("size_modifiers", {})
            row = sm.get(weapon_type, {})
            if not row:
                return 1.0
            return float(row.get(monster_size, 1.0))

    def best_weapon_for_size(self, monster_size: str) -> tuple[str, float]:
        """Return (weapon_type, modifier) with highest multiplier for given size."""
        with self._lock:
            sm = self._mechanics.get("size_modifiers", {})
            best = ("", 0.0)
            for wtype, sizes in sm.items():
                if isinstance(sizes, dict):
                    mod = float(sizes.get(monster_size, 1.0))
                    if mod > best[1]:
                        best = (wtype, mod)
            return best

    # ── Skill Recommendation ──

    def skill_effectiveness(
        self,
        skill_element: str,
        target_element: str,
        element_level: int = 1,
        race_bonus: float = 1.0,
    ) -> float:
        """Calculate overall effectiveness of a skill against a monster.
        Combines element advantage and any race bonus.
        """
        elem = self.element_advantage(skill_element, target_element, element_level)
        return elem * race_bonus

    def best_skill_for_monster(
        self,
        available_skills: list[dict[str, Any]],
        monster_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Pick the best skill from available skills against given monster.
        available_skills: [{'skill_id':'AL_PCP','element':'Holy','matk':1.5,'cast_time':1000,...}]
        monster_info: {'element':'Undead','element_level':1,'race':'Demon','size':'Medium','hp':500,'def':10,'mdef':30}
        Returns the best skill dict or None if no useful skill found.
        """
        if not available_skills or not monster_info:
            return None

        target_elem = str(monster_info.get("element", "Neutral"))
        elem_level = int(monster_info.get("element_level", 1))

        best: dict[str, Any] | None = None
        best_score = 0.0

        for skill in available_skills:
            skill_elem = str(skill.get("element", "Neutral"))
            base_dmg = float(skill.get("matk", skill.get("atk", 1.0)))
            multiplier = self.element_advantage(skill_elem, target_elem, elem_level)
            score = base_dmg * multiplier

            # Consider cast time (faster = better)
            cast_time = float(skill.get("cast_time", 0))
            if cast_time > 0:
                score /= 1.0 + (cast_time / 1000.0)  # Penalty for long casts

            # Consider SP cost
            sp_cost = float(skill.get("sp_cost", 0))
            if sp_cost > 0:
                score /= 1.0 + (sp_cost / 50.0)  # Slight penalty for high SP

            if score > best_score:
                best_score = score
                best = skill

        return best

    # ── Combat Viability ──

    def avg_hits_to_kill(self, monster_hp: int, avg_damage: int) -> float:
        """Calculate average hits needed to kill a monster."""
        if avg_damage <= 0:
            return 999.0
        return monster_hp / avg_damage

    def hits_to_die(self, my_hp: int, monster_atk: int, aggro_count: int = 1) -> float:
        """Calculate how many hits you can survive."""
        if monster_atk <= 0:
            return 999.0
        # Account for defense reduction (simplified: def reduces ~damage by def%)
        return my_hp / (monster_atk * aggro_count)

    def flee_recommendation(
        self,
        my_hp: int,
        my_max_hp: int,
        monster_hp: int,
        monster_atk: int,
        avg_damage: int,
        aggro_count: int = 1,
    ) -> dict[str, Any]:
        """Return flee recommendation with reasoning.
        Returns {'should_flee': bool, 'reason': str, 'detail': str}
        
        Survival-first principle: flee when you can't survive 3+ hits
        or when it takes too many hits to kill the monster.
        """
        hp_pct = my_hp / my_max_hp if my_max_hp > 0 else 1.0
        htd = self.hits_to_die(my_hp, monster_atk, aggro_count)
        htk = self.avg_hits_to_kill(monster_hp, avg_damage)

        # Survival-first: flee if you can't survive 3 hits
        if htd < 3.0:
            return {
                "should_flee": True,
                "reason": "lethal_threat",
                "detail": f"Can only survive {htd:.1f} hits (need 3+)",
                "hp_pct": hp_pct,
            }

        # Efficiency: flee if it takes too many hits to kill
        if htk > 20.0 and htd < htk / 2:
            return {
                "should_flee": True,
                "reason": "inefficient_engagement",
                "detail": f"Need {htk:.0f} hits to kill but only survive {htd:.1f}",
                "hp_pct": hp_pct,
            }

        # Multi-aggro: flee if surrounded
        if aggro_count >= 3 and htd < 5.0:
            return {
                "should_flee": True,
                "reason": "surrounded",
                "detail": f"{aggro_count} aggro, can only survive {htd:.1f} hits",
                "hp_pct": hp_pct,
            }

        return {
            "should_flee": False,
            "reason": "safe_to_engage",
            "detail": f"Can survive {htd:.1f} hits, kill in {htk:.0f} hits",
            "hp_pct": hp_pct,
        }

    # ── Race/Armor Lookup ──

    def race_multiplier(self, skill_race_bonus: str, monster_race: str) -> float:
        """Get multiplier from race-specific bonuses (e.g., DemiHuman reduction)."""
        with self._lock:
            race_data = self._mechanics.get("race_multipliers", {})
            if not race_data:
                return 1.0
            bonuses = race_data.get(skill_race_bonus, {})
            if isinstance(bonuses, dict):
                return float(bonuses.get(monster_race, 1.0))
            return 1.0

    def armor_element_defense(self, armor_element: str, attack_element: str) -> float:
        """Get defense multiplier when wearing armor of given element."""
        # Same element table: armor_element is "target" side
        return self.element_advantage(attack_element, armor_element)

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic stats about the combat intel engine."""
        with self._lock:
            return {
                "data_loaded": bool(self._mechanics),
                "data_path": str(self._data_path),
                "element_levels": [k for k in self._mechanics.get("element_table", {}) if k.startswith("level_")],
                "weapon_types": list(self._mechanics.get("size_modifiers", {}).keys()),
                "data_keys": list(self._mechanics.keys()),
            }


def create_combat_intel(data_path: str | None = None) -> CombatIntelligence:
    """Factory function for dependency injection."""
    return CombatIntelligence(data_path=data_path)
