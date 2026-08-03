"""
Combat optimization engine — kiting, MVP tactics, element-aware combat, aggro control.

Integrates with game_engine, reflex rules, and the bridge action pipeline.
All commands emitted are bridge-allowlist safe (ss, use, ai, move, attack).
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from .base_agent import BehaviorProfile

logger = logging.getLogger(__name__)

# Element chart loaded from knowledge_graph at runtime
# Fallback hardcoded chart for when knowledge_graph is unavailable
_FALLBACK_ELEMENT_CHART: dict[str, dict[str, float]] = {
    "neutral": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "holy": 1.0, "shadow": 1.0, "ghost": 1.0, "undead": 1.0},
    "water": {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.5, "wind": 1.0, "holy": 1.0, "shadow": 1.0, "ghost": 1.0, "undead": 1.0},
    "earth": {"neutral": 1.0, "water": 1.5, "earth": 0.25, "fire": 1.0, "wind": 0.75, "holy": 1.0, "shadow": 1.0, "ghost": 1.0, "undead": 1.0},
    "fire": {"neutral": 1.0, "water": 0.75, "earth": 1.5, "fire": 0.25, "wind": 1.0, "holy": 1.0, "shadow": 1.0, "ghost": 1.0, "undead": 1.0},
    "wind": {"neutral": 1.0, "water": 1.0, "earth": 0.75, "fire": 1.0, "wind": 0.25, "holy": 1.0, "shadow": 1.0, "ghost": 1.0, "undead": 1.0},
    "holy": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "holy": 0.25, "shadow": 1.5, "ghost": 1.0, "undead": 1.5},
    "shadow": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "holy": 0.75, "shadow": 0.25, "ghost": 1.0, "undead": 1.0},
    "ghost": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0, "holy": 1.0, "shadow": 1.0, "ghost": 0.5, "undead": 1.0},
    "undead": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.5, "wind": 1.0, "holy": 2.0, "shadow": 0.5, "ghost": 1.0, "undead": 0.25},
}

# MVP monsters loaded from knowledge.json at runtime
# Fallback set for when knowledge is unavailable
_FALLBACK_MVP_MONSTERS: set[str] = {
    "anglng", "detale", "deviling", "dokebi", "dragon_tail", "edga", "eow", "eyra",
    "gh_mvp", "glddm", "golden_bug", "gtb", "hatii", "inca", "jakk", "kasa", "katz",
    "kimmy", "kraken", "loli", "lord_aye", "lord_kaho", "maero", "master_bee", "maya",
    "mistress", "moonlight", "morroc", "mvp", "orc_hero", "orc_lord", "osiris",
    "pharaoh", "phreeoni", "rsx", "samurai", "seal_mvp", "sg_mvp", "smokie",
    "stormy", "tao", "thanatos", "turtle_general", "valkyrie", "vesper", "warlock",
}


@dataclass(slots=True)
class CombatProfile(BehaviorProfile):
    """Aggro management, attack priority, flee threshold — legacy CrewAI profile."""
    agent_id = "combat"
    role = "Combat Specialist"
    goal = "Eliminate threats efficiently while minimizing damage taken"
    backstory = (
        "Trained in the heat of countless battles, this agent reads the "
        "battlefield instantly — prioritizing dangerous mobs, managing aggro, "
        "and knowing exactly when to stand ground or flee."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        monsters = signals.get("monsters_around", [])
        aggro = signals.get("aggro_list", [])
        if not monsters and not aggro:
            return 0.0
        score = 0.3 * min(len(aggro), 5) / 5.0
        score += 0.4 * min(len(monsters), 10) / 10.0
        hp_pct = signals.get("hp", 1) / max(signals.get("hp_max", 1), 1)
        if hp_pct < 0.3:
            score += 0.3
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        monsters = signals.get("monsters_around", [])
        aggro = signals.get("aggro_list", [])
        hp_pct = signals.get("hp", 1) / max(signals.get("hp_max", 1), 1)
        if hp_pct < 0.2 and aggro:
            return {"kind": "flee", "command": "ai manual_flee", "confidence": 0.95, "reason": "HP critical with aggro"}
        if aggro:
            target = min(aggro, key=lambda m: m.get("hp_pct", 1)) if aggro else None
            if target:
                return {"kind": "attack", "command": f"attack {target.get('name', 'monster')}", "confidence": 0.85, "reason": f"Engaging aggro target {target.get('name', '?')}"}
        if monsters:
            target = monsters[0]
            return {"kind": "attack", "command": f"attack {target.get('name', 'monster')}", "confidence": 0.6, "reason": "Clearing nearby monsters"}
        return None


@dataclass(slots=True)
class CombatOptimizer:
    """Advanced combat tactics engine."""
    
    knowledge_path: Path | None = None
    _lock: RLock = field(default_factory=RLock)
    _monster_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    _element_skills: dict[str, list[str]] = field(default_factory=dict)
    _element_chart: dict[str, dict[str, float]] = field(default_factory=dict)
    _mvp_alerted: set[str] = field(default_factory=set)
    _kite_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if self.knowledge_path is not None and self.knowledge_path.exists():
            try:
                data = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
                # Load element chart from knowledge.json
                elements_raw = data.get("elements", {})
                if isinstance(elements_raw, dict):
                    for mode_key in ["re", "pre-re"]:
                        mode_data = elements_raw.get(mode_key, {})
                        if isinstance(mode_data, dict):
                            body = mode_data.get("Body", [])
                            if isinstance(body, list):
                                for entry in body:
                                    if isinstance(entry, dict):
                                        level = entry.get("Level", 1)
                                        if level != 1:
                                            continue
                                        for atk_ele, def_map in entry.items():
                                            if atk_ele == "Level":
                                                continue
                                            if isinstance(def_map, dict):
                                                for def_ele, dmg_pct in def_map.items():
                                                    dmg = float(dmg_pct) / 100.0
                                                    atk_key = str(atk_ele).lower()
                                                    def_key = str(def_ele).lower()
                                                    if atk_key not in self._element_chart:
                                                        self._element_chart[atk_key] = {}
                                                    self._element_chart[atk_key][def_key] = dmg
                # Load monster data
                monsters = data.get("monsters", data.get("monster_db", []))
                if isinstance(monsters, list):
                    for m in monsters:
                        name = str(m.get("Name", m.get("name", m.get("id", "")))).lower()
                        self._monster_data[name] = {
                            "element": str(m.get("Element", m.get("element", "neutral"))).lower(),
                            "race": str(m.get("Race", m.get("race", "formless"))).lower(),
                            "hp": int(m.get("Hp", m.get("hp", 0)) or 0),
                            "level": int(m.get("Level", m.get("level", 1)) or 1),
                            "mvp": bool(m.get("Modes.Mvp", False)),
                            "size": str(m.get("Size", m.get("size", "medium"))).lower(),
                            "def": int(m.get("Defense", m.get("def", 0)) or 0),
                            "mdef": int(m.get("MagicDefense", m.get("mdef", 0)) or 0),
                        }
                # Build element -> skill mapping from monster data
                self._build_element_skills()
                logger.info("combat_opt_loaded: %d monsters", len(self._monster_data))
            except Exception as e:
                logger.debug("combat_opt_load_skipped: %s", e)
    
    def _build_element_skills(self) -> None:
        """Map element names to skills that counter them."""
        self._element_skills = {
            "water": ["ss cold_bolt", "ss frost_diver"],
            "earth": ["ss stone_curse", "ss fire_bolt"],
            "fire": ["ss cold_bolt", "ss frost_diver"],
            "wind": ["ss lightning_bolt", "ss thunder_storm"],
            "holy": ["ss holy_light", "ss turn_undead"],
            "shadow": ["ss holy_light"],
            "ghost": ["ss soul_strike"],
            "undead": ["ss holy_light", "ss turn_undead"],
            "neutral": [],
        }
    
    def get_element_advantage(self, monster_element: str, player_element: str = "neutral") -> float:
        """Get elemental damage multiplier against a monster.
        
        Uses rAthena-accurate element chart from knowledge.json if available,
        falls back to hardcoded chart otherwise.
        """
        monster_element = monster_element.lower()
        player_element = player_element.lower()
        # Try knowledge.json's element chart first
        if hasattr(self, '_element_chart') and self._element_chart:
            attack_chart = self._element_chart.get(player_element, {})
            return float(attack_chart.get(monster_element, 1.0))
        attack_chart = _FALLBACK_ELEMENT_CHART.get(player_element, {})
        return float(attack_chart.get(monster_element, 1.0))
    
    def recommend_skill(self, monster_name: str, player_class: str = "novice", known_skills: list[str] | None = None) -> str | None:
        """Recommend the best skill to use against a monster based on element."""
        monster = self._monster_data.get(monster_name.lower(), {})
        element = str(monster.get("element", "neutral"))
        known = [s.lower() for s in (known_skills or [])]
        
        candidates = self._element_skills.get(element, [])
        for skill in candidates:
            # Check if player knows this skill
            skill_root = skill.split(" ")[-1] if " " in skill else skill
            if any(skill_root in ks for ks in known):
                return skill
        
        # Generic fallback: use strongest known skill
        for ks in known:
            if "bolt" in ks or "strike" in ks:
                return f"ss {ks.split()[-1]}"
        return None
    
    def is_mvp(self, monster_name: str) -> bool:
        """Check if a monster is an MVP (boss)."""
        name = monster_name.lower()
        if name in self._monster_data:
            return bool(self._monster_data[name].get("mvp", False))
        return any(mvp in name for mvp in _FALLBACK_MVP_MONSTERS)
    
    def assess_threat(self, monster_name: str, player_level: int = 1) -> float:
        """Assess threat level of a monster (0.0 = harmless, 1.0 = lethal)."""
        monster = self._monster_data.get(monster_name.lower())
        if monster is None:
            return 0.5  # unknown monster — moderate threat
        
        level = int(monster.get("level", 1))
        is_boss = bool(monster.get("mvp", False))
        atk = int(monster.get("atk", monster.get("attack", 0)) or 0)
        
        # Threat factors
        level_diff = level - player_level
        threat = 0.0
        if level_diff > 20:
            threat += 0.5
        if level_diff > 10:
            threat += 0.3
        if is_boss:
            threat += 0.3
        if atk > 100:
            threat += 0.2
        
        return min(threat, 1.0)
    
    def should_kite(self, monster_name: str, player_class: str, player_hp_pct: float) -> bool:
        """Determine if the player should kite (ranged attack while moving)."""
        monster = self._monster_data.get(monster_name.lower(), {})
        element = str(monster.get("element", "neutral"))
        
        # Always kite if the monster is high-threat
        if player_hp_pct < 0.3:
            return True
        
        # Ranged classes should kite for HP preservation
        ranged_classes = {"archer", "hunter", "sniper", "mage", "wizard", "high_wizard", "sorcerer", "warlock"}
        if player_class.lower() in ranged_classes and player_hp_pct < 0.6:
            return True
        
        return False
    
    def get_kite_command(self, bot_id: str, monster_name: str, action: str = "attack") -> str:
        """Get a kiting command sequence (move then attack)."""
        # Kiting: move back 3 cells, then attack
        # The bridge executes commands sequentially — move then attack
        with self._lock:
            state = self._kite_state.get(bot_id, {"step": "attack", "timer": 0})
            state["step"] = "move" if state["step"] == "attack" else action
            self._kite_state[bot_id] = state
        
        if state["step"] == "move":
            # Move away from monster (bridge will route)
            return "ai auto"  # Let AI auto-handle movement
            
        return f"ss attack" if action == "attack" else action
    
    def get_mvp_alert(self, monster_name: str, bot_id: str) -> list[str]:
        """Get commands to alert party members of an MVP sighting."""
        if monster_name in self._mvp_alerted:
            return []
        self._mvp_alerted.add(monster_name)
        
        # MVP tactics: party gather, coordinated attack
        return [
            f"attackAuto 0",  # Stop current AI combat (never 'ai manual' — freezes auto-attack)
            f"move prontera",  # Party up in town (simplified)
        ]
    
    def get_combo_commands(self, monster_name: str, player_class: str, hp_pct: float) -> list[str]:
        """Get a skill combo sequence for optimal damage."""
        commands: list[str] = []
        
        # No combos for novices
        if player_class == "novice":
            return []
        
        monster = self._monster_data.get(monster_name.lower(), {})
        element = str(monster.get("element", "neutral"))
        
        # Build combo based on class and element
        class_combos = {
            "mage": [("ss cold_bolt", 0.0), ("ss fire_bolt", 0.0)],
            "archer": [("ss double_strafing", 0.0), ("ss arrow_shower", 0.5)],
            "swordman": [("ss bash", 0.0), ("ss magnum_break", 0.3)],
            "thief": [("ss double_attack", 0.0), ("ss hiding", 0.5)],
            "acolyte": [("ss holy_light", 0.0), ("ss heal", 0.2)],
        }
        
        combo = class_combos.get(player_class.lower(), [])
        if not combo:
            return []
        
        # Apply element-bonus skills first if applicable
        element_skills = self._element_skills.get(element, [])
        if element_skills:
            commands.append(element_skills[0])
        
        # Add class-specific combo steps
        for skill, hp_threshold in combo:
            if hp_pct >= hp_threshold:
                commands.append(skill)
        
        return commands[:4]  # Max 4 steps in a combo
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return {"mvp_alerted": len(self._mvp_alerted)}
