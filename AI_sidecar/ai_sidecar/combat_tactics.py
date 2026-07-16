"""
Combat tactics engine — per-class skill combos, kiting, terrain use, weapon switching,
card bonuses, and buff/element override awareness.

The LLM selects the combat profile; the tactics engine executes it.
Fixed by Pro RO Player: correct element combos, proper kiting, real class mechanics,
card damage multipliers, and buff-override-aware element selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CombatTactics:
    """Per-class combat tactics and skill combos — fixed by Pro RO Player.

    Now includes card damage multiplier support, buff override detection,
    and weapon size penalty awareness for "Pro RO Player" damage calculation.
    """

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
        "wizard": [
            {"skills": ["ss frost_diver", "ss lightning_bolt"], "condition": "element_water", "description": "Freeze + thunder — 4x on frozen"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "element_earth", "description": "Fire vs earth"},
            {"skills": ["ss cold_bolt", "ss cold_bolt"], "condition": "element_fire", "description": "Cold vs fire"},
            {"skills": ["ss storm_gust", "ss lord_of_vermillion"], "condition": "aggro>3", "description": "Mass AoE — freeze then shock"},
            {"skills": ["ss fire_ball", "ss fire_ball"], "condition": "aggro>2", "description": "Fire ball AoE for grouped mobs"},
            {"skills": ["ss fire_bolt", "ss fire_bolt"], "condition": "always", "description": "Fire Bolt single target"},
        ],
        # ── Archer ──
        "archer": [
            {"skills": ["ss double_strafing", "ss double_strafing"], "condition": "always", "description": "Double strafe spam while kiting"},
            {"skills": ["ss arrow_shower"], "condition": "aggro>2", "description": "Arrow Shower AoE + knockback"},
        ],
        # ── Hunter ──
        "hunter": [
            {"skills": ["ss double_strafing", "ss blitz_beat"], "condition": "hp>0.7", "description": "Double strafe + falcon"},
            {"skills": ["ss double_strafing", "ss double_strafing"], "condition": "always", "description": "Double strafe spam"},
            {"skills": ["ss arrow_shower"], "condition": "aggro>2", "description": "AoE arrow"},
        ],
        # ── Swordsman ──
        "swordman": [
            {"skills": ["ss bash", "ss bash"], "condition": "always", "description": "Bash spam — stun chance interrupts casts"},
            {"skills": ["ss magnum_break", "ss bash"], "condition": "aggro>2", "description": "Magnum Break AoE then bash"},
        ],
        # ── Knight ──
        "knight": [
            {"skills": ["ss bowling_bash", "ss bowling_bash"], "condition": "aggro>1", "description": "Bowling Bash AoE spam"},
            {"skills": ["ss spear_boomerang", "ss bowling_bash"], "condition": "hp>0.6", "description": "Ranged opener then AoE"},
            {"skills": ["ss bowling_bash", "ss magnum_break"], "condition": "aggro>3", "description": "Double AoE clear"},
        ],
        # ── Thief ──
        "thief": [
            {"skills": ["ss double_attack", "ss double_attack"], "condition": "always", "description": "Double attack spam (passive double proc)"},
            {"skills": ["ss hiding"], "condition": "hp<0.3", "description": "Emergency hide — drop aggro"},
        ],
        # ── Assassin ──
        "assassin": [
            {"skills": ["ss grimtooth", "ss sonic_blow"], "condition": "hp>0.5", "description": "Grimtooth ranged poke → Sonic Blow finisher"},
            {"skills": ["ss sonic_blow", "ss sonic_blow"], "condition": "hp<0.3", "description": "Desperation Sonic Blow spam"},
            {"skills": ["ss venom_dust"], "condition": "aggro>2", "description": "Poison AoE for groups"},
            {"skills": ["ss grimtooth", "ss grimtooth"], "condition": "always", "description": "Grimtooth ranged spam"},
        ],
        # ── Acolyte ──
        "acolyte": [
            {"skills": ["ss holy_light", "ss holy_light"], "condition": "hp>0.6", "description": "Holy Light spam"},
            {"skills": ["ss heal", "ss heal"], "condition": "hp<0.6", "description": "Self-heal + damage undead"},
            {"skills": ["ss heal", "ss holy_light"], "condition": "element_undead", "description": "Heal nukes undead, Holy Light for others"},
        ],
        # ── Priest ──
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
    # Kept for backwards compatibility; get_element_multiplier() uses the
    # parsed attr_fix.yml data which has all 4 levels.
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

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Card multiplier support
    # ══════════════════════════════════════════════════════════════════════════

    def get_card_multiplier(
        self,
        equipped_cards: list[str],
        monster_race: str,
        monster_size: str,
        monster_element: str,
    ) -> float:
        """Compute the combined card damage multiplier against the target.

        Uses the CardDatabase to compute additive stacking within bonus types
        and multiplicative stacking across types.

        Args:
            equipped_cards: List of card names (e.g. ["Hydra Card", "Hydra Card"]).
            monster_race: Target monster's race (e.g. "DemiHuman").
            monster_size: Target monster's size (e.g. "Large").
            monster_element: Target monster's element (e.g. "Water").

        Returns:
            Combined card multiplier (e.g. 1.38 for Hydra + Skeleton Worker).
            Returns 1.0 if no cards or no matching bonuses.
        """
        if not equipped_cards:
            return 1.0

        # Use the ElementalMatrix's card multiplier method
        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        mat = get_elemental_matrix()
        return mat.get_card_damage_multiplier(
            equipped_cards, monster_race, monster_size, monster_element,
        )

    def get_card_multiplier_breakdown(
        self,
        equipped_cards: list[str],
        monster_race: str,
        monster_size: str,
        monster_element: str,
    ) -> dict[str, float]:
        """Get a detailed breakdown of card multiplier components.

        Returns dict with keys: race_mult, element_mult, size_mult,
        atk_mult, phys_mult, total.
        """
        if not equipped_cards:
            return {"race_mult": 1.0, "element_mult": 1.0, "size_mult": 1.0,
                    "atk_mult": 1.0, "phys_mult": 1.0, "total": 1.0}

        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        mat = get_elemental_matrix()
        return mat.get_card_multiplier_breakdown(
            equipped_cards, monster_race, monster_size, monster_element,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Buff override detection
    # ══════════════════════════════════════════════════════════════════════════

    def get_effective_element(
        self,
        weapon_element: str,
        active_buffs: Optional[list[str]] = None,
    ) -> str:
        """Determine the effective physical attack element considering buff overrides.

        RO mechanics:
          - Elemental Converters (Fire/Water/Wind/Earth) override weapon element
          - Aspersio overrides to Holy
          - Enchant Deadly Poison overrides to Poison
          - The last-applied converter wins

        Args:
            weapon_element: The weapon's base element (usually "Neutral").
            active_buffs: List of active buff/skill names.

        Returns:
            The effective element name to use for damage calculations.
        """
        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        mat = get_elemental_matrix()
        return mat.get_effective_element(weapon_element, active_buffs or []).value

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Weapon size penalty
    # ══════════════════════════════════════════════════════════════════════════

    def get_weapon_size_multiplier(self, weapon_type: str, monster_size: str) -> float:
        """Get the size penalty/multiplier for a weapon vs a monster size.

        Examples:
          Dagger vs Large = 0.50 (50% damage penalty)
          Spear vs Large = 1.00 (full damage)
          Sword vs Medium = 1.00 (full damage)
        """
        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        mat = get_elemental_matrix()
        return mat.get_size_multiplier(weapon_type, monster_size)

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Master damage multiplier (card × size × race × element)
    # ══════════════════════════════════════════════════════════════════════════

    def get_combined_damage_multiplier(
        self,
        attack_element: str,
        weapon_type: str,
        monster_element: str,
        monster_size: str,
        monster_race: str,
        element_level: int = 1,
        equipped_cards: Optional[list[str]] = None,
        active_buffs: Optional[list[str]] = None,
        weapon_element: Optional[str] = None,
    ) -> float:
        """Compute the full "Pro RO Player" damage multiplier.

        Chains together:
          1. Elemental advantage (element × target_element)
          2. Weapon size penalty (weapon_type × target_size)
          3. Race modifier (weapon_type × target_race)
          4. Card bonuses (additive per-type, multiplicative across types)
          5. Buff element overrides (converters, aspersio, EDP)

        Args:
            attack_element: Element of the skill/spell being used.
            weapon_type: Type of weapon equipped.
            monster_element: Target's element.
            monster_size: Target's size.
            monster_race: Target's race.
            element_level: Target's ElementLevel (1-4, default 1).
            equipped_cards: List of card names on weapon (optional).
            active_buffs: List of active buffs (optional).
            weapon_element: Weapon's base element (optional; if None,
                           uses attack_element for buff override calc).

        Returns:
            Combined multiplier (float). 1.0 = 100% base damage.
        """
        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        mat = get_elemental_matrix()

        return mat.get_effective_damage_multiplier(
            attack_element=attack_element,
            weapon_type=weapon_type,
            target_element=monster_element,
            target_size=monster_size,
            target_race=monster_race,
            element_level=element_level,
            cards=equipped_cards,
            active_buffs=active_buffs,
            weapon_element=weapon_element,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Suggestion helpers for the LLM
    # ══════════════════════════════════════════════════════════════════════════

    def suggest_optimal_weapon_type(self, monster_size: str) -> str:
        """Suggest the ideal weapon type for a given monster size.

        Unlike get_weapon_for_size which returns a simple type name,
        this returns detailed reasoning about size advantages.
        """
        msize = monster_size.lower()
        if msize == "small":
            return "dagger"
        elif msize == "large":
            return "spear"
        return "sword"

    def suggest_cards_for_monster(
        self,
        monster_race: str,
        monster_size: str,
        monster_element: str,
        available_cards: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Suggest which cards would be most effective against this monster.

        Args:
            monster_race: Target's race.
            monster_size: Target's size.
            monster_element: Target's element.
            available_cards: Cards available to choose from (if None, all known).

        Returns:
            List of dicts with card name, bonus type, pct bonus, and total impact.
        """
        from ai_sidecar.combat.card_db import get_card_database
        db = get_card_database()

        if available_cards is None:
            available_cards = [c.name for c in db.list_cards()]

        suggestions: list[dict[str, Any]] = []
        for card_name in available_cards:
            card = db.get_card(card_name)
            if card is None:
                continue

            bonuses = []
            total_pct = 0

            if monster_race in card.race_bonus:
                pct = card.race_bonus[monster_race]
                bonuses.append(f"race({monster_race}: +{pct}%)")
                total_pct += pct

            if monster_element in card.element_bonus:
                pct = card.element_bonus[monster_element]
                bonuses.append(f"element({monster_element}: +{pct}%)")
                total_pct += pct

            if monster_size in card.size_bonus:
                pct = card.size_bonus[monster_size]
                bonuses.append(f"size({monster_size}: +{pct}%)")
                total_pct += pct

            if bonuses:
                suggestions.append({
                    "card": card_name,
                    "bonuses": bonuses,
                    "total_percent": total_pct,
                    "description": card.description,
                })

        # Sort by total percentage bonus descending
        suggestions.sort(key=lambda s: s["total_percent"], reverse=True)
        return suggestions

    # ══════════════════════════════════════════════════════════════════════════
    # Original methods (unchanged)
    # ══════════════════════════════════════════════════════════════════════════

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

    
    def compute_damage_multiplier(
        self,
        attack_element: str,
        defense_element: str,
        element_level: int = 1,
        weapon_type: str | None = None,
        monster_size: str | None = None,
        monster_race: str | None = None,
        cards: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute the full damage multiplier including element + card + size + race.

        Pro RO Player knows: a +10 Dagger vs Large DemiHuman with 4x Hydra Cards
        is NOT just element × card bonus. Size penalty applies FIRST,
        then element, then cards, each multiplicative.

        Args:
            attack_element: Element of the attack (e.g., "fire", "holy")
            defense_element: Monster's element
            element_level: Monster's element level (1-4)
            weapon_type: Weapon type (e.g., "dagger", "spear")
            monster_size: Monster size ("small", "medium", "large")
            monster_race: Monster race
            cards: List of equipped card names

        Returns:
            dict with breakdown: {
                "element_multiplier": float,
                "size_penalty": float,
                "card_multiplier": float,
                "race_multiplier": float,
                "total": float,
            }
        """
        result: dict[str, float] = {}

        # 1. Element multiplier
        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        em = get_elemental_matrix()
        elem_mult = em.get_elemental_multiplier(attack_element, defense_element, element_level)
        result["element_multiplier"] = elem_mult

        # 2. Size penalty
        size_penalty = 1.0
        if weapon_type and monster_size:
            try:
                from ai_sidecar.combat.elemental_matrix import WeaponType, Size
                wt_map = {
                    "dagger": WeaponType.DAGGER, "sword": WeaponType.SWORD,
                    "spear": WeaponType.SPEAR, "two_handed_sword": WeaponType.TWO_HANDED_SWORD,
                    "bow": WeaponType.BOW, "staff": WeaponType.STAFF,
                    "mace": WeaponType.MACE, "axe": WeaponType.AXE,
                    "knuckle": WeaponType.KNUCKLE, "katar": WeaponType.KATAR,
                    "instrument": WeaponType.INSTRUMENT, "whip": WeaponType.WHIP,
                    "book": WeaponType.BOOK, "claw": WeaponType.CLAW,
                    "two_handed_spear": WeaponType.TWO_HANDED_SPEAR,
                    "two_handed_axe": WeaponType.TWO_HANDED_AXE,
                    "two_handed_staff": WeaponType.TWO_HANDED_STAFF,
                }
                sz_map = {"small": Size.SMALL, "medium": Size.MEDIUM, "large": Size.LARGE}
                wt_enum = wt_map.get(weapon_type.lower().replace(" ", "_"))
                sz_enum = sz_map.get(monster_size.lower())
                if wt_enum and sz_enum:
                    size_penalty = em.get_size_penalty(wt_enum, sz_enum)
            except Exception:
                size_penalty = 1.0
        result["size_penalty"] = size_penalty

        # 3. Card multiplier
        card_mult = 1.0
        if cards:
            try:
                from ai_sidecar.combat.card_db import get_card_database
                db = get_card_database()
                if monster_race or monster_size or defense_element:
                    card_mult = db.get_card_multiplier(
                        cards,
                        target_race=monster_race or "",
                        target_size=monster_size or "",
                        target_element=defense_element,
                    )
            except Exception:
                card_mult = 1.0
        result["card_multiplier"] = card_mult

        # 4. Total = element × size × card
        total = elem_mult * size_penalty * card_mult
        result["total"] = total
        result["race_multiplier"] = 1.0

        return result

def get_element_multiplier(self, attack_element: str, defense_element: str,
                              element_level: int = 1) -> float:
        """Get the damage multiplier for attack element vs defense element.

        Uses parsed rAthena attr_fix.yml (all 4 levels).
        A pro player knows these by heart — this just keeps them fresh.
        """
        from ai_sidecar.combat.elemental_matrix import get_elemental_matrix
        return get_elemental_matrix().get_elemental_multiplier(
            attack_element, defense_element, element_level
        )

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
