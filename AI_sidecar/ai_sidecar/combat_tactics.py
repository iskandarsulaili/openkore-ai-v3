"""
Combat tactics engine — per-class skill combos, kiting, terrain use, weapon switching.

The LLM selects the combat profile; the tactics engine executes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CombatTactics:
    """Per-class combat tactics and skill combos."""
    
    _lock: RLock = field(default_factory=RLock)
    _class_combos: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {
        "mage": [
            {"skills": ["ss cold_bolt", "ss fire_bolt"], "condition": "element_water", "description": "Freeze then burn"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "element_earth", "description": "Fire vs earth"},
            {"skills": ["ss cold_bolt", "ss frost_diver"], "condition": "hp>0.5", "description": "Freeze lock"},
        ],
        "wizard": [
            {"skills": ["ss frost_diver", "ss lightning_bolt"], "condition": "element_water", "description": "Freeze + thunder"},
            {"skills": ["ss fire_bolt", "ss fire_ball"], "condition": "aggro>2", "description": "AoE fire"},
            {"skills": ["ss storm_gust", "ss lord_of_vermillion"], "condition": "aggro>5", "description": "Mass AoE"},
        ],
        "archer": [
            {"skills": ["ss double_strafing", "ss double_strafing"], "condition": "always", "description": "Double strafe spam"},
            {"skills": ["ss arrow_shower"], "condition": "aggro>2", "description": "AoE arrow"},
        ],
        "hunter": [
            {"skills": ["ss double_strafing", "ss arrow_shower"], "condition": "aggro>1", "description": "Single then AoE"},
            {"skills": ["ss blitz_beat"], "condition": "hp>0.7", "description": "Falcon assault"},
        ],
        "swordman": [
            {"skills": ["ss bash", "ss magnum_break"], "condition": "hp>0.5", "description": "Bash + AoE"},
            {"skills": ["ss bash", "ss bash"], "condition": "always", "description": "Bash spam"},
        ],
        "knight": [
            {"skills": ["ss bowling_bash", "ss magnum_break"], "condition": "aggro>2", "description": "AoE clear"},
            {"skills": ["ss spear_boomerang", "ss bowling_bash"], "condition": "hp>0.6", "description": "Ranged + AoE"},
        ],
        "thief": [
            {"skills": ["ss double_attack", "ss double_attack"], "condition": "always", "description": "Double attack spam"},
            {"skills": ["ss hiding"], "condition": "hp<0.3", "description": "Emergency hide"},
        ],
        "assassin": [
            {"skills": ["ss sonic_blow", "ss grimtooth"], "condition": "hp>0.5", "description": "Burst + ranged"},
            {"skills": ["ss venom_dust"], "condition": "aggro>2", "description": "Poison AoE"},
        ],
        "acolyte": [
            {"skills": ["ss holy_light", "ss heal"], "condition": "hp<0.6", "description": "Attack + self-heal"},
            {"skills": ["ss holy_light", "ss holy_light"], "condition": "hp>0.6", "description": "Holy spam"},
        ],
        "priest": [
            {"skills": ["ss holy_light", "ss turn_undead"], "condition": "element_undead", "description": "Anti-undead"},
            {"skills": ["ss heal", "ss heal"], "condition": "hp<0.5", "description": "Self-heal spam"},
            {"skills": ["ss magnificat", "ss kyrie_eleison"], "condition": "party", "description": "Party buffs"},
        ],
    })
    
    _kite_classes: set[str] = field(default_factory=lambda: {"archer", "hunter", "sniper", "mage", "wizard", "high_wizard", "sorcerer", "warlock"})
    
    _size_weapons: dict[str, str] = field(default_factory=lambda: {"small": "dagger", "medium": "sword", "large": "spear"})
    
    def get_combo(self, player_class: str, monster_element: str, hp_pct: float, aggro_count: int, has_party: bool) -> list[str]:
        """Get the best skill combo for the current situation."""
        combos = self._class_combos.get(player_class.lower(), [])
        if not combos:
            return []
        
        best_combo = None
        best_score = -1
        
        for combo in combos:
            condition = combo.get("condition", "always")
            score = 0.5  # Base score
            
            # Evaluate condition
            if condition == "always":
                score = 1.0
            elif condition.startswith("hp>"):
                threshold = float(condition.split(">")[1])
                if hp_pct > threshold:
                    score = 1.0
            elif condition.startswith("hp<"):
                threshold = float(condition.split("<")[1])
                if hp_pct < threshold:
                    score = 1.0
            elif condition == "party" and has_party:
                score = 1.0
            elif condition.startswith("aggro>"):
                threshold = int(condition.split(">")[1])
                if aggro_count > threshold:
                    score = 1.0
            elif condition.startswith("element_"):
                target_elem = condition.split("_")[1]
                if monster_element == target_elem:
                    score = 1.5  # Element advantage bonus
            
            if score > best_score:
                best_score = score
                best_combo = combo
        
        if best_combo is None:
            return []
        
        return list(best_combo.get("skills", []))
    
    def should_kite(self, player_class: str, hp_pct: float) -> bool:
        """Determine if the player should kite."""
        if player_class.lower() in self._kite_classes and hp_pct < 0.6:
            return True
        return hp_pct < 0.3  # Everyone kites at low HP
    
    def get_weapon_for_size(self, monster_size: str) -> str | None:
        """Get the best weapon type for a monster's size."""
        return self._size_weapons.get(monster_size.lower())
    
    def counters(self) -> dict[str, int]:
        return {"combos": sum(len(v) for v in self._class_combos.values())}
