"""
Combat tactics engine — per-class skill combos, kiting, terrain use, weapon switching.

The LLM selects the combat profile; the tactics engine executes it.
Fixed by Pro RO Player: correct element combos, proper kiting, real class mechanics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CombatTactics:
    """Per-class combat tactics and skill combos — fixed by Pro RO Player."""

    _lock: RLock = field(default_factory=RLock)
    _class_combos: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {
        # ── Mage ──
        # Pro mage opens with Frost Diver (freeze) then Fire Bolt for 4x damage on frozen
        # Never Cold Bolt first vs water — Cold Bolt is water, does 25% to water
        "mage": [
            {"skills": ["ss frost_diver", "ss fire_bolt"], "condition": "element_water", "description": "Freeze then fire — 4x damage on frozen water mob"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "element_earth", "description": "Fire vs earth — 175% damage"},
            {"skills": ["ss cold_bolt", "ss cold_bolt"], "condition": "element_fire", "description": "Cold Bolt vs fire — 200% damage"},
            {"skills": ["ss lightning_bolt", "ss lightning_bolt"], "condition": "element_water", "description": "Lightning vs water — 175% damage"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "element_wind", "description": "Fire vs wind — 175% damage"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "always", "description": "Fire Bolt spam — best general element"},
            {"skills": ["ss frost_diver", "ss fire_bolt"], "condition": "hp>0.5", "description": "Freeze lock + fire burst"},
        ],
        # ── Wizard ──
        # Pro wizard uses Storm Gust for AoE freeze, then Lord of Vermillion for AoE damage
        # Never uses Fire Ball as primary — Storm Gust + LoV is the meta combo
        "wizard": [
            {"skills": ["ss frost_diver", "ss lightning_bolt"], "condition": "element_water", "description": "Freeze + thunder — 4x on frozen"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "element_earth", "description": "Fire vs earth"},
            {"skills": ["ss cold_bolt", "ss cold_bolt"], "condition": "element_fire", "description": "Cold vs fire"},
            {"skills": ["ss storm_gust", "ss lord_of_vermillion"], "condition": "aggro>3", "description": "Mass AoE — freeze then shock"},
            {"skills": ["ss fire_ball", "ss fire_ball"], "condition": "aggro>2", "description": "Fire ball AoE for grouped mobs"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "always", "description": "Fire Bolt single target"},
        ],
        # ── Archer ──
        # Pro archer NEVER stands still. Double Strafe while kiting.
        # Arrow Shower for knockback + AoE
        "archer": [
            {"skills": ["ss double_strafing", "ss double_strafing"], "condition": "always", "description": "Double strafe spam while kiting"},
            {"skills": ["ss arrow_shower"], "condition": "aggro>2", "description": "Arrow Shower AoE + knockback"},
        ],
        # ── Hunter ──
        # Pro hunter uses trap + blitz beat + double strafe
        "hunter": [
            {"skills": ["ss double_strafing", "ss blitz_beat"], "condition": "hp>0.7", "description": "Double strafe + falcon"},
            {"skills": ["ss double_strafing", "ss double_strafing"], "condition": "always", "description": "Double strafe spam"},
            {"skills": ["ss arrow_shower"], "condition": "aggro>2", "description": "AoE arrow"},
        ],
        # ── Swordsman ──
        # Pro swordsman: BASH is main damage, Magnum Break for AoE
        # Bash has stun chance — use it to interrupt casts
        "swordman": [
            {"skills": ["ss bash", "ss bash"], "condition": "always", "description": "Bash spam — stun chance interrupts casts"},
            {"skills": ["ss magnum_break", "ss bash"], "condition": "aggro>2", "description": "Magnum Break AoE then bash"},
        ],
        # ── Knight ──
        # Pro knight: Bowling Bash is the main AoE, Spear Boomerang for ranged
        # Twohanded Quicken for ASPD buff before engaging
        "knight": [
            {"skills": ["ss bowling_bash", "ss bowling_bash"], "condition": "aggro>1", "description": "Bowling Bash AoE spam"},
            {"skills": ["ss spear_boomerang", "ss bowling_bash"], "condition": "hp>0.6", "description": "Ranged opener then AoE"},
            {"skills": ["ss bowling_bash", "ss magnum_break"], "condition": "aggro>3", "description": "Double AoE clear"},
        ],
        # ── Thief ──
        # Pro thief: Double Attack is passive, main attack is normal + double proc
        # Hiding for escape, then reposition
        "thief": [
            {"skills": ["ss double_attack", "ss double_attack"], "condition": "always", "description": "Double attack spam (passive double proc)"},
            {"skills": ["ss hiding"], "condition": "hp<0.3", "description": "Emergency hide — drop aggro"},
        ],
        # ── Assassin ──
        # Pro assassin: Sonic Blow is the burst finisher, NOT opener
        # Grimtooth is ranged AoE — use from distance, NOT after Sonic Blow
        # Correct combo: Grimtooth from range, then close for Sonic Blow
        "assassin": [
            {"skills": ["ss grimtooth", "ss sonic_blow"], "condition": "hp>0.5", "description": "Grimtooth ranged poke → Sonic Blow finisher"},
            {"skills": ["ss sonic_blow", "ss sonic_blow"], "condition": "hp<0.3", "description": "Desperation Sonic Blow spam"},
            {"skills": ["ss venom_dust"], "condition": "aggro>2", "description": "Poison AoE for groups"},
            {"skills": ["ss grimtooth", "ss grimtooth"], "condition": "always", "description": "Grimtooth ranged spam"},
        ],
        # ── Acolyte ──
        # Pro acolyte: Heal is both healing AND damage vs undead
        # Holy Light for non-undead, Heal for undead
        "acolyte": [
            {"skills": ["ss holy_light", "ss holy_light"], "condition": "hp>0.6", "description": "Holy Light spam"},
            {"skills": ["ss heal", "ss heal"], "condition": "hp<0.6", "description": "Self-heal + damage undead"},
            {"skills": ["ss heal", "ss holy_light"], "condition": "element_undead", "description": "Heal nukes undead, Holy Light for others"},
        ],
        # ── Priest ──
        # Pro priest: Turn Undead for instant kill on undead, Heal for sustain
        # Kyrie Eleison pre-buff, Magnificat for SP regen
        "priest": [
            {"skills": ["ss turn_undead", "ss holy_light"], "condition": "element_undead", "description": "Turn Undead (instant kill chance) then Holy Light"},
            {"skills": ["ss heal", "ss heal"], "condition": "hp<0.5", "description": "Self-heal spam"},
            {"skills": ["ss holy_light", "ss holy_light"], "condition": "always", "description": "Holy Light spam"},
        ],
    })

    _kite_classes: set[str] = field(default_factory=lambda: {
        "archer", "hunter", "sniper", "mage", "wizard", "high_wizard",
        "sorcerer", "warlock", "bard", "dancer", "clown", "gypsy",
        "gunslinger", "rebel",
    })

    _size_weapons: dict[str, str] = field(default_factory=lambda: {
        "small": "dagger", "medium": "sword", "large": "spear"
    })

    # ── Element chart (offensive: attack element → defense element multiplier) ──
    # Pre-renewal RO element chart — CORRECT values
    ELEMENT_MULT: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "neutral":  {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                     "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0},
        "water":    {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.0, "wind": 0.75,
                     "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
        "earth":    {"neutral": 1.0, "water": 1.0, "earth": 0.25, "fire": 0.75, "wind": 1.0,
                     "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
        "fire":     {"neutral": 1.0, "water": 0.5, "earth": 1.0, "fire": 0.25, "wind": 0.75,
                     "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.25},
        "wind":     {"neutral": 1.0, "water": 1.0, "earth": 0.5, "fire": 1.0, "wind": 0.25,
                     "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
        "poison":   {"neutral": 1.0, "water": 1.0, "earth": 0.75, "fire": 1.0, "wind": 1.0,
                     "poison": 0.25, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 0.5},
        "holy":     {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                     "poison": 1.0, "holy": 0.25, "dark": 2.0, "ghost": 1.0, "undead": 1.5},
        "dark":     {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                     "poison": 1.0, "holy": 0.5, "dark": 0.25, "ghost": 1.0, "undead": 1.0},
        "ghost":    {"neutral": 0.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
                     "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.5, "undead": 1.0},
        "undead":   {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.25, "wind": 1.0,
                     "poison": 0.5, "holy": 2.0, "dark": 0.5, "ghost": 1.0, "undead": 0.5},
    })

    def get_combo(self, player_class: str, monster_element: str, hp_pct: float,
                  aggro_count: int, has_party: bool) -> list[str]:
        """Get the best skill combo for the current situation."""
        combos = self._class_combos.get(player_class.lower(), [])
        if not combos:
            return []

        best_combo = None
        best_score = -1

        for combo in combos:
            condition = combo.get("condition", "always")
            score = 0.5

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
                if monster_element.lower() == target_elem:
                    score = 1.5  # Element advantage bonus

            if score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            return []

        return list(best_combo.get("skills", []))

    def should_kite(self, player_class: str, hp_pct: float) -> bool:
        """Determine if the player should kite.
        
        Pro rule: ranged classes ALWAYS kite. Melee classes kite at low HP.
        Kiting means: attack → move → attack → move (never stand still).
        """
        if player_class.lower() in self._kite_classes:
            return True  # Ranged classes always kite
        return hp_pct < 0.3  # Melee kites only at low HP

    def get_weapon_for_size(self, monster_size: str) -> str | None:
        """Get the best weapon type for a monster's size."""
        return self._size_weapons.get(monster_size.lower())

    def get_element_multiplier(self, attack_element: str, defense_element: str,
                              element_level: int = 1) -> float:
        """Get the damage multiplier for attack element vs defense element.

        Uses parsed rAthena attr_fix.yml (all 4 levels).
        A pro player knows these by heart — this just keeps them fresh.
        """
        from ai_sidecar.data.element_db import get_element_multiplier as _get_mult
        return _get_mult(attack_element, defense_element, element_level=element_level)

    def get_best_element_attack(self, player_class: str, monster_element: str,
                              element_level: int = 1) -> str:
        """Recommend the best element to use against a monster.
        
        Pro knowledge: Holy beats Undead (2x), Fire beats Undead (1.25x),
        Holy beats Dark (2x), Fire beats Earth (1.75x), etc.
        """
        monster_elem = monster_element.lower()

        # Class-specific element access
        class_elements = {
            "mage": ["fire", "water", "wind", "earth"],
            "wizard": ["fire", "water", "wind", "earth"],
            "high_wizard": ["fire", "water", "wind", "earth"],
            "sorcerer": ["fire", "water", "wind", "earth"],
            "warlock": ["fire", "water", "wind", "earth"],
            "sage": ["fire", "water", "wind", "earth"],
            "professor": ["fire", "water", "wind", "earth"],
            "acolyte": ["holy"],
            "priest": ["holy"],
            "arch_bishop": ["holy"],
            "monk": ["holy"],
            "champion": ["holy"],
            "assassin": ["poison"],
            "assassin_cross": ["poison"],
            "guillotine_cross": ["poison"],
        }

        available = class_elements.get(player_class.lower(), ["neutral"])

        best_elem = "neutral"
        best_mult = 1.0

        for elem in available:
            mult = self.get_element_multiplier(elem, monster_elem, element_level=element_level)
            # Prefer non-neutral elements when tied (neutral is always 1.0)
            if mult > best_mult or (mult == best_mult and elem != "neutral" and best_elem == "neutral"):
                best_mult = mult
                best_elem = elem

        return best_elem

    def counters(self) -> dict[str, int]:
        return {"combos": sum(len(v) for v in self._class_combos.values())}
