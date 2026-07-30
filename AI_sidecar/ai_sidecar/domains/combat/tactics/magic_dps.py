"""
Magic DPS tactics — spell rotation, elemental advantage, AoE nuking, Safety Wall usage.

Used by: Mage, Wizard, Sage, and other magical damage dealers.
Pro RO behavior: use best element, freeze combo, Safety Wall for defense, position safely.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class MagicDPSTactics(BaseTactics):
    """Magic DPS tactics: elemental advantage priority, spell rotation, AoE.

    Priority 50 — default damage role.

    Pro RO Wizard behavior:
    - Always use best element against target (Fire Bolt Lv10 = 460%)
    - Use Safety Wall when being approached by melee monsters
    - Frost Diver → Storm Gust combo (freeze + AoE shatter)
    - AoE nuke on groups (Meteor Storm, Storm Gust, Lord of Vermilion)
    - Use Energy Coat for SP-to-HP conversion when low
    - Maintain 7-8 cells distance from target
    """

    name: ClassVar[str] = "magic_dps"
    priority: ClassVar[int] = 50
    description: ClassVar[str] = "Magic damage dealer — elemental advantage and spell rotation"
    role_type: ClassVar[str] = "dps"

    # Optimal range for casting
    OPTIMAL_RANGE = 7
    RETREAT_RANGE = 3  # Distance at which wizard should retreat

    # Element: best offensive element per target element
    BEST_ELEMENT = {
        "water": "wind",    # Lightning beats water (200% at Lv3)
        "earth": "fire",    # Fire beats earth (175% at Lv2)
        "fire": "water",    # Water beats fire (200% at Lv3)
        "wind": "earth",    # Earth beats wind (200% at Lv3)
        "undead": "holy",   # Holy beats undead (200% at Lv2)
        "dark": "holy",     # Holy beats dark (250% at Lv2)
        "poison": "holy",   # Holy beats poison
        "ghost": "holy",    # Holy beats ghost (Lv1)
        "neutral": "fire",  # Fire as general
    }

    # Skill ID mapping: (internal_id, element, sp_cost)
    SPELL_SKILLS: dict[str, tuple[str, str, int]] = {
        "mg_firebolt": ("MG_FIREBOLT", "fire", 15),
        "mg_coldbolt": ("MG_COLD", "water", 15),
        "mg_lightningbolt": ("MG_LIGHTNING", "wind", 20),
        "mg_fireball": ("MG_FIREBALL", "fire", 25),
        "mg_frostdiver": ("MG_FROSTDIVER", "water", 12),
        "mg_napalmbeat": ("MG_NAPALMBEAT", "neutral", 10),
        "mg_soulstrike": ("MG_SOULSTRIKE", "neutral", 18),
        "wz_stormgust": ("WZ_STORMGUST", "water", 80),
        "wz_meteorstorm": ("WZ_METEOR", "fire", 90),
        "wz_lordofvermilion": ("WZ_VERMILION", "wind", 85),
        "wz_heavensdrive": ("WZ_HEAVENDRIVE", "neutral", 45),
        "wz_frostnova": ("WZ_FROSTNOVA", "water", 20),
    }

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Magic DPS target priority:
        1. Element-weak targets (big damage multiplier).
        2. Casting monsters (interrupt with fast spells).
        3. Grouped monsters (AoE effectiveness).
        4. High value / boss targets.
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0

            # Elemental advantage: find best element against this target
            best_elem = self.BEST_ELEMENT.get(t.element, "fire")
            elem_mult = self._get_elem_multiplier(best_elem, t.element)
            score += elem_mult * 30.0  # Up to 60 points for 2.0x

            # Low HP — finish with fast cast
            if t.hp_pct < 0.3:
                score += 40.0

            # Aggro high = many monsters nearby, AoE effective
            if c.aggro_count >= 3:
                score += 20.0

            # Casting monster — interrupt priority
            if t.is_casting:
                score += 35.0

            # Boss
            if t.is_boss:
                score += 40.0

            # Proximity for AoE spells (prefer targets at cast range)
            score += max(0, 12 - t.distance)

            # Aggressive
            if t.is_aggressive:
                score += 5.0

            return score

        return self._find_best_by_score(ctx, scorer)

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Magic DPS skill selection: best element → AoE → fast filler.

        Pro RO Wizard rotation:
        1. Place Safety Wall if target is getting close
        2. Use best element spell against target
        3. Frost Diver → freeze → follow-up for 4x damage
        4. AoE nuke when grouped (Storm Gust, Meteor Storm)
        5. Fast filler for low SP (Napalm Beat)
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Safety Wall when being approached (melee monsters getting close)
        if target.distance < self.RETREAT_RANGE and "MG_SAFETYWALL" in available:
            if ctx.cooldowns.get("mg_safetywall", 0) <= 0:
                return ("mg_safetywall", 5)

        # Too close for comfort — back up first
        if target.distance < self.OPTIMAL_RANGE - 3:
            return None  # Let positioning handle retreat

        # Determine the best offensive element
        target_elem = target.element
        best_elem = self.BEST_ELEMENT.get(target_elem, "fire")

        # Frost Diver opener (freeze → follow-up for big damage)
        if target.hp_pct > 0.7 and "MG_FROSTDIVER" in available:
            if ctx.cooldowns.get("mg_frostdiver", 0) <= 0:
                level = 10 if ctx.my_sp_pct > 0.5 else 5
                return ("mg_frostdiver", level)

        # Map best element to a spell
        element_spell_map = {
            "fire": "mg_firebolt",
            "water": "mg_coldbolt",
            "wind": "mg_lightningbolt",
            "earth": "mg_firebolt",  # No earth bolt in basic — use fire
            "holy": "mg_napalmbeat",
        }

        # Try to use best element spell
        best_spell_key = element_spell_map.get(best_elem, "mg_firebolt")
        best_spell_info = self.SPELL_SKILLS.get(best_spell_key)

        if best_spell_info:
            skill_id, _, sp_cost = best_spell_info
            if skill_id in available and ctx.my_sp >= sp_cost:
                if ctx.cooldowns.get(skill_id.lower(), 0) <= 0:
                    level = 10 if ctx.my_sp_pct > 0.5 else 5
                    return (skill_id.lower(), level)

        # AoE when grouped
        if ctx.aggro_count >= 4 and ctx.my_sp_pct > 0.5:
            # Meteor Storm for big groups (7 hits × 150% = 1050%)
            if "WZ_METEOR" in available and ctx.my_sp >= 90:
                if ctx.cooldowns.get("wz_meteor", 0) <= 0:
                    return ("wz_meteor", 10)
            # Storm Gust for water-weak (10 hits × 160% = 1600%)
            if "WZ_STORMGUST" in available and ctx.my_sp >= 80:
                if ctx.cooldowns.get("wz_stormgust", 0) <= 0:
                    return ("wz_stormgust", 10)
            # Lord of Vermilion for wind-weak (5 hits × 400% = 2000%)
            if "WZ_VERMILION" in available and ctx.my_sp >= 85:
                if ctx.cooldowns.get("wz_vermillion", 0) <= 0:
                    return ("wz_vermillion", 10)
            # Fire Ball for small groups
            if "MG_FIREBALL" in available and ctx.my_sp >= 25:
                if ctx.cooldowns.get("mg_fireball", 0) <= 0:
                    return ("mg_fireball", 5)

        # Fast filler: Napalm Beat / Soul Strike for low SP
        if ctx.my_sp_pct < 0.3:
            if "MG_NAPALMBEAT" in available:
                return ("mg_napalmbeat", 5)
            if "MG_SOULSTRIKE" in available:
                return ("mg_soulstrike", 5)

        # Fire Bolt as general filler (Lv10 = 460%)
        if "MG_FIREBOLT" in available:
            return ("mg_firebolt", 10)

        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Magic positioning: stay at 5-8 cells for casting safety.

        Pro RO Wizard positioning:
        - Back up if target is within RETREAT_RANGE (3 cells)
        - Hold position at OPTIMAL_RANGE (7 cells)
        - If multiple mobs, back toward a safe direction
        """
        if target is None:
            return None

        if target.distance < self.RETREAT_RANGE:
            # Too close, back up (high urgency — wizard is fragile)
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": "backing_up_for_cast_safety",
                "urgency": 0.9,
            }

        if target.distance < self.OPTIMAL_RANGE - 2:
            # Slightly too close, adjust
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": "adjusting_range_for_safe_cast",
                "urgency": 0.4,
            }

        if target.distance > self.OPTIMAL_RANGE + 3:
            return {
                "move_x": 0,
                "move_y": 0,
                "reason": f"approaching_{target.name}",
                "urgency": 0.4,
            }

        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep magic buffs active.

        Pro RO Wizard:
        - Energy Coat for SP-to-HP conversion
        - Amplify Magic Power for +50% MATK
        - SP Recovery passive (always on)
        """
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        # SP Recovery
        if "MG_SRECOVERY" in available and "SRECOVERY" not in active:
            needed.append("mg_srecovery")

        # Energy Coat (damage -> SP conversion)
        if "MG_ENERGYCOAT" in available and "ENERGYCOAT" not in active:
            needed.append("mg_energycoat")

        # Amplify Magic Power
        if "WZ_AMPLIFY" in available and "AMPLIFY" not in active:
            needed.append("wz_amplify")

        return needed

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Magic emergency: use Blue Potions for SP, flee if overwhelmed.

        Pro RO Wizard emergency:
        - Use Blue Potion when SP < 15% (SP is damage)
        - Use Fly Wing when surrounded by 3+ at < 50% HP
        - Use potions at higher threshold (wizard is squishy)
        """
        if ctx.my_sp_pct < 0.15:
            return self._make_action(
                command="use_blue_potion",
                reason="mage_sp_critical",
                confidence=0.9,
                sp_pct=ctx.my_sp_pct,
            )

        if ctx.my_hp_pct < 0.4 and ctx.aggro_count > 2:
            return self._make_action(
                command="flee_to_safe_spot",
                reason="mage_overwhelmed_flee",
                confidence=0.85,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
            )

        if ctx.my_hp_pct < 0.45:
            return self._make_action(
                command="use_potion_or_heal",
                reason="mage_hp_low",
                confidence=0.75,
                hp_pct=ctx.my_hp_pct,
            )
        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Magic rotation: buff → Frost Diver → best element spell → filler.

        Pro RO Wizard rotation:
        1. Frost Diver (freeze target)
        2. Best element spell (Fire Bolt Lv10 = 460%)
        3. Storm Gust if grouped (10 hits × 160% = 1600%)
        4. Fire Bolt filler
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        if not target:
            return rotation

        target_elem = target.element
        best_elem = self.BEST_ELEMENT.get(target_elem, "fire")

        # Frost Diver → follow-up for high HP targets
        if target.hp_pct > 0.5 and "MG_FROSTDIVER" in available:
            rotation.append(("mg_frostdiver", 5))
            # Follow-up based on element
            if best_elem == "fire" and "MG_FIREBOLT" in available:
                rotation.append(("mg_firebolt", 10))
            elif best_elem == "water" and "MG_COLD" in available:
                rotation.append(("mg_coldbolt", 10))
            elif best_elem == "wind" and "MG_LIGHTNING" in available:
                rotation.append(("mg_lightningbolt", 10))

        # Storm Gust for groups (10 hits × per-level damage)
        if ctx.aggro_count >= 3 and "WZ_STORMGUST" in available and ctx.my_sp >= 80:
            rotation.append(("wz_stormgust", 10))
            return rotation  # Don't follow up SG with single target

        # Best element spell spam
        element_map = {
            "fire": ("mg_firebolt", "MG_FIREBOLT", 10),
            "water": ("mg_coldbolt", "MG_COLD", 10),
            "wind": ("mg_lightningbolt", "MG_LIGHTNING", 10),
            "holy": ("mg_napalmbeat", "MG_NAPALMBEAT", 5),
        }
        if best_elem in element_map:
            skill_key, skill_id, level = element_map[best_elem]
            if skill_id in available:
                rotation.append((skill_key, level))

        # Fire Bolt filler (Lv10 = 460%)
        if "MG_FIREBOLT" in available:
            rotation.append(("mg_firebolt", 10))

        return rotation

    @staticmethod
    def _get_elem_multiplier(attack_element: str, defense_element: str) -> float:
        """Quick element multiplier lookup (Level 1 table)."""
        table = {
            "neutral": {"neutral": 1.0, "water": 0.75, "earth": 0.75, "fire": 0.75,
                        "wind": 0.75, "poison": 0.75, "holy": 0.75, "dark": 0.75,
                        "ghost": 0.5, "undead": 0.5},
            "fire": {"neutral": 1.0, "water": 0.5, "earth": 1.25, "fire": 0.25,
                     "wind": 0.75, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                     "ghost": 0.5, "undead": 1.25},
            "water": {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.25,
                      "wind": 0.5, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                      "ghost": 0.5, "undead": 1.0},
            "wind": {"neutral": 1.0, "water": 1.25, "earth": 0.5, "fire": 1.25,
                     "wind": 0.25, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                     "ghost": 0.5, "undead": 1.0},
            "earth": {"neutral": 1.0, "water": 1.25, "earth": 0.25, "fire": 0.75,
                      "wind": 1.25, "poison": 0.75, "holy": 1.0, "dark": 1.0,
                      "ghost": 0.5, "undead": 1.0},
            "holy": {"neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0,
                     "wind": 1.0, "poison": 1.0, "holy": 0.25, "dark": 2.0,
                     "ghost": 1.0, "undead": 2.0},
        }
        return table.get(attack_element, {}).get(defense_element, 1.0)
