"""Kiting tactics for ranged classes — distance maintenance, terrain awareness.

Features:
  - Maintain 7-9 cell optimal range from target
  - Retreat/teleport when target gets within 3 cells
  - Terrain awareness: reposition for line of sight when behind a wall
  - Don't chase monsters into packs — wait for them to come back
  - Use knockback abilities to create distance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# Distance thresholds (in RO cells)
OPTIMAL_RANGE_MIN = 7       # Minimum cells to maintain from target
OPTIMAL_RANGE_MAX = 9       # Maximum cells to maintain from target
DANGER_CLOSE = 3            # When target is this close, emergency retreat
CHASE_PACK_RANGE = 5        # Don't chase if target goes into a pack this close
LOST_SIGHT_RANGE = 14       # Assume line of sight broken beyond this

# Cell types for terrain awareness
BLOCKED = 0       # Cannot walk through (wall, obstacle)
OPEN = 1          # Walkable

# Minimum cells needed for knockback distance
KNOCKBACK_MIN_DISTANCE = 3


@dataclass
class Position:
    """Position on the RO map grid."""
    x: int
    y: int


@dataclass
class KitingProfile:
    """Per-class kiting parameters."""
    optimal_range_min: int = OPTIMAL_RANGE_MIN
    optimal_range_max: int = OPTIMAL_RANGE_MAX
    danger_close: int = DANGER_CLOSE
    flee_hp_threshold: float = 0.35       # HP % below which to emergency flee
    teleport_on_overwhelm: bool = True     # Use Teleport/Fly Wing when surrounded
    overwhelm_count: int = 4              # Number of aggro mobs before teleport
    use_terrain: bool = True               # Use terrain for line-of-sight breaking
    use_knockback: bool = True             # Use knockback skills to create distance
    fallback_teleport_hp: float = 0.15    # Teleport at this HP if aggro'd


# Skills that provide knockback
_KNOCKBACK_SKILLS: dict[str, int] = {
    "ac_shower": 3,        # Arrow Shower: 3 cells knockback
    "mg_firewall": 5,      # Fire Wall: holds target at 5 cells
}

# Ranged classes and their weapon ranges
_RANGED_CLASSES: dict[str, int] = {
    "archer": 9,
    "hunter": 9,
    "sniper": 9,
    "mage": 9,
    "wizard": 9,
    "high_wizard": 9,
    "sage": 9,
    "professor": 9,
    "priest": 7,
    "high_priest": 7,
    "soul_linker": 7,
    "gunslinger": 9,
    "rebellion": 9,
    "bard": 9,
    "dancer": 9,
    "minstrel": 9,
    "wanderer": 9,
}


class KitingTactics(BaseTactics):
    """Kiting tactics — distance management and terrain-aware movement for ranged classes.

    Priority 40 (higher than default RangedDPS at 50 — kiting is a refinement
    that integrates into the existing ranged tactics pattern).

    Features:
        - Maintain 7-9 cell distance from target
        - Emergency retreat when target is within 3 cells
        - Terrain-aware repositioning (break line of sight)
        - Pack avoidance (don't chase into groups)
        - Knockback skill usage to create distance
        - Teleport/fly wing when overwhelmed
    """

    name: ClassVar[str] = "kiting"
    priority: ClassVar[int] = 40
    description: ClassVar[str] = "Kiting specialist — distance management and terrain-aware movement"
    role_type: ClassVar[str] = "kiter"

    def __init__(self) -> None:
        super().__init__()
        self._profile = KitingProfile()
        self._last_known_position: dict[int, Position] = {}  # actor_id -> Position

    def get_ranged_range(self, job_class: str) -> int:
        """Get the attack range for a job class."""
        return _RANGED_CLASSES.get(job_class.lower(), 7)

    def is_ranged_class(self, job_class: str) -> bool:
        """Check if a class benefits from kiting behavior."""
        return job_class.lower() in _RANGED_CLASSES

    # ── Target Selection ─────────────────────────────────────

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Kiting target priority:
        1. Monsters that can be kited (not in packs)
        2. Monsters at safe range (7-9 cells)
        3. Casting monsters (interrupt from safe distance)
        4. Low HP targets (finish quickly)
        5. Aggressive monsters (need to handle first)
        """
        monsters = ctx.monsters
        if not monsters:
            return None

        def scorer(t: TargetInfo, c: TacticsContext) -> float:
            score = 0.0

            # Penalize targets in packs (don't chase into groups)
            if self._monster_in_pack(t, c):
                score -= 30.0

            # Preferred distance range (7-9 cells)
            if OPTIMAL_RANGE_MIN <= t.distance <= OPTIMAL_RANGE_MAX:
                score += 25.0
            elif t.distance < DANGER_CLOSE:
                score -= 20.0  # Too close — dangerous

            # Casting monsters — safe interrupt
            if t.is_casting:
                score += 40.0

            # Low HP — rapid finish
            if t.hp_pct < 0.2:
                score += 50.0
            elif t.hp_pct < 0.4:
                score += 25.0

            # Aggressive monsters — deal with first
            if t.is_aggressive:
                score += 15.0

            # Element advantage
            elem_mult = self._get_elem_multiplier(
                ctx.my_weapon_element, t.element,
            )
            if elem_mult > 1.1:
                score += 15.0

            # Boss — careful approach
            if t.is_boss:
                score += 20.0

            return score

        return self._find_best_by_score(ctx, scorer)

    def _monster_in_pack(self, target: TargetInfo, ctx: TacticsContext) -> bool:
        """Check if a monster is within a pack (near other monsters)."""
        pack_count = 0
        for m in ctx.monsters:
            if m.get("actor_id", m.get("id", 0)) == target.actor_id:
                continue
            other_dist = abs(
                m.get("distance", 99) - target.distance
            )
            if other_dist <= CHASE_PACK_RANGE:
                pack_count += 1
                if pack_count >= 2:  # 2+ other monsters nearby = pack
                    return True
        return False

    # ── Skill Selection ──────────────────────────────────────

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Kiting skill selection:
        1. Knockback skills first when target too close
        2. Ranged damage skills
        3. Movement skills when fleeing
        """
        if not target:
            return None

        available = set(s.upper() for s in ctx.available_skills)

        # Defensive: knockback when too close
        if target.distance < DANGER_CLOSE and self._profile.use_knockback:
            for skill_name in _KNOCKBACK_SKILLS:
                if skill_name.upper() in available:
                    if ctx.cooldowns.get(skill_name, 0) <= 0:
                        return (skill_name, 10)

        # Escape: teleport when overwhelmed
        if ctx.aggro_count >= self._profile.overwhelm_count and self._profile.teleport_on_overwhelm:
            if "AL_TELEPORT" in available and ctx.cooldowns.get("al_teleport", 0) <= 0:
                return ("al_teleport", 1)

        # Primary damage: Double Strafe if archer
        if "AC_DOUBLE" in available and ctx.my_sp_pct > 0.2:
            if ctx.cooldowns.get("ac_double", 0) <= 0:
                return ("ac_double", 10)

        # Elemental bolts for mages at range
        for bolt_skill in ["MG_FIREBOLT", "MG_COLDBOLT", "MG_LIGHTNINGBOLT"]:
            if bolt_skill in available and ctx.my_sp_pct > 0.3:
                if ctx.cooldowns.get(bolt_skill.lower(), 0) <= 0:
                    return (bolt_skill, 10)

        return None

    # ── Positioning ───────────────────────────────────────────

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Advanced kiting positioning:
        1. Emergency retreat if target is danger-close
        2. Maintain ideal range (7-9 cells)
        3. Use terrain to break line of sight
        4. Don't chase into packs
        5. Reposition to kite path (circular movement)
        """
        if target is None:
            return None

        # Emergency: target is very close
        if target.distance < DANGER_CLOSE:
            # Check if we can use terrain to block
            if self._profile.use_terrain and self._can_block_with_terrain(ctx, target):
                return {
                    "move_x": 0, "move_y": 0,
                    "reason": f"terrain_block_vs_{target.name}",
                    "urgency": 0.9,
                    "tactic": "terrain_block",
                }

            # Emergency retreat
            return {
                "move_x": 0, "move_y": 0,
                "reason": f"emergency_retreat_from_{target.name}_dist_{target.distance}",
                "urgency": 0.9,
                "tactic": "retreat",
            }

        # Emergency flee if HP low and aggro'd
        if ctx.my_hp_pct < self._profile.flee_hp_threshold and ctx.aggro_count > 1:
            return {
                "move_x": 0, "move_y": 0,
                "reason": "hp_low_flee",
                "urgency": 0.85,
                "tactic": "flee",
            }

        # Maintain optimal range (7-9 cells)
        if target.distance < OPTIMAL_RANGE_MIN:
            # Too close — back up
            return {
                "move_x": 0, "move_y": 0,
                "reason": f"back_up_from_{target.name}_dist_{target.distance}_to_{OPTIMAL_RANGE_MIN}",
                "urgency": 0.6,
                "tactic": "back_up",
            }

        if target.distance > OPTIMAL_RANGE_MAX + 2:
            # Too far — approach slightly, but check for packs
            if self._monster_in_pack(target, ctx):
                return {
                    "move_x": 0, "move_y": 0,
                    "reason": f"wait_for_{target.name}_to_come_back_in_pack",
                    "urgency": 0.2,
                    "tactic": "wait",
                }

            # Approach carefully
            return {
                "move_x": 0, "move_y": 0,
                "reason": f"approach_{target.name}_dist_{target.distance}",
                "urgency": 0.3,
                "tactic": "approach",
            }

        # Terrain awareness: check line of sight
        if self._profile.use_terrain and target.distance > LOST_SIGHT_RANGE:
            # Target might be behind an obstacle — reposition
            return {
                "move_x": 0, "move_y": 0,
                "reason": f"reposition_for_los_to_{target.name}",
                "urgency": 0.5,
                "tactic": "reposition_los",
            }

        # Ideal range — no movement needed
        return None

    def _can_block_with_terrain(self, ctx: TacticsContext, target: TargetInfo) -> bool:
        """Check if we can use terrain (wall, obstacle) to block the target."""
        # In practice this uses the map's collision grid.
        # Returns True if there's an obstacle between us and the target.
        # Placeholder — full terrain analysis requires map collision data.
        return False  # Simplified: actual impl needs map data

    def _get_kite_direction(self, ctx: TacticsContext, target: TargetInfo) -> str:
        """Determine the best direction to kite in.

        Returns: 'left', 'right', 'backward', or 'circle'.
        """
        # Circular kiting: move perpendicular to the line between us and target.
        # This maintains distance while making us harder to hit.
        return "circle"

    # ── Emergency ─────────────────────────────────────────────

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Kiting emergency responses:
        1. Teleport/Fly Wing when overwhelmed by multiple aggro
        2. Emergency potion when HP very low
        3. Retreat to safe position
        """
        # Teleport when surrounded
        if ctx.aggro_count >= self._profile.overwhelm_count and self._profile.teleport_on_overwhelm:
            return self._make_action(
                command="use_fly_wing_or_teleport",
                reason=f"kiter_overwhelmed_{ctx.aggro_count}_aggro",
                confidence=0.9,
                aggro=ctx.aggro_count,
                hp_pct=ctx.my_hp_pct,
                tactic="teleport_escape",
            )

        # Emergency potion at critical HP
        if ctx.my_hp_pct < self._profile.fallback_teleport_hp:
            return self._make_action(
                command="use_emergency_potion",
                reason=f"kiter_critical_hp_{ctx.my_hp_pct:.0%}",
                confidence=0.95,
                hp_pct=ctx.my_hp_pct,
                tactic="emergency_heal",
            )

        # Flee if HP low and aggro'd
        if ctx.my_hp_pct < ctx.config.get("flee_hp", 0.3) and ctx.aggro_count > 0:
            return self._make_action(
                command="flee_to_safe_spot",
                reason=f"kiter_hp_low_flee_{ctx.my_hp_pct:.0%}",
                confidence=0.8,
                hp_pct=ctx.my_hp_pct,
                aggro=ctx.aggro_count,
                tactic="retreat",
            )

        return None

    # ── Rotation ──────────────────────────────────────────────

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Kiting rotation: knockback escape → damage at range → reposition.

        Maintains distance at all times.
        """
        rotation: list[tuple[str, int]] = []
        available = set(s.upper() for s in ctx.available_skills)

        if not target:
            return rotation

        # If target is close, prioritize knockback
        if target.distance < DANGER_CLOSE:
            if "AC_SHOWER" in available:
                rotation.append(("ac_shower", 5))
            elif "MG_FIREWALL" in available:
                rotation.append(("mg_firewall", 10))
            return rotation

        # At range: use primary ranged attack
        if ctx.my_sp_pct > 0.2 and target.distance >= OPTIMAL_RANGE_MIN:
            if "AC_DOUBLE" in available:
                rotation.append(("ac_double", 10))
            elif "MG_FIREBOLT" in available:
                # Use the elementally-strongest bolt
                rotation.append(("mg_firebolt", 10))
            elif "AL_HEAL" in available and ctx.my_hp_pct < 0.7:
                # Self-heal during downtime
                rotation.append(("al_heal", 10))

        # Basic attack fallback (handled by engine)
        return rotation

    # ── Buffs ─────────────────────────────────────────────────

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Keep movement and ASPD buffs active for effective kiting."""
        needed: list[str] = []
        available = set(s.upper() for s in ctx.available_skills)
        active = set(b.upper() for b in ctx.active_buffs)

        # Movement speed buffs
        if "AL_INCAGI" in available and "INCREASEAGI" not in active:
            needed.append("al_incagi")
        if "AC_CONCENTRATION" in available and "CONCENTRATION" not in active:
            needed.append("ac_concentration")
        if "MC_PROVOKE" in available and "PROVOKE" not in active:
            # Some classes use Provoke to pull, not as a buff
            pass

        return needed

    # ── Element Multiplier ────────────────────────────────────

    @staticmethod
    def _get_elem_multiplier(attack_element: str, defense_element: str) -> float:
        """Quick element multiplier lookup (neutral for ranged weapons)."""
        table = {
            "neutral": {"neutral": 1.0, "water": 0.75, "earth": 0.75, "fire": 0.75,
                        "wind": 0.75, "poison": 0.75, "holy": 0.75, "shadow": 0.75,
                        "ghost": 0.5, "undead": 0.5},
            "fire": {"neutral": 1.0, "water": 0.5, "earth": 1.25, "fire": 0.25,
                     "wind": 0.75, "poison": 0.75, "holy": 1.0, "shadow": 1.0,
                     "ghost": 0.5, "undead": 1.25},
            "water": {"neutral": 1.0, "water": 0.25, "earth": 0.75, "fire": 1.25,
                      "wind": 0.5, "poison": 0.75, "holy": 1.0, "shadow": 1.0,
                      "ghost": 0.5, "undead": 1.0},
            "wind": {"neutral": 1.0, "water": 1.25, "earth": 0.5, "fire": 1.25,
                     "wind": 0.25, "poison": 0.75, "holy": 1.0, "shadow": 1.0,
                     "ghost": 0.5, "undead": 1.0},
            "earth": {"neutral": 1.0, "water": 1.25, "earth": 0.25, "fire": 0.75,
                      "wind": 1.25, "poison": 0.75, "holy": 1.0, "shadow": 1.0,
                      "ghost": 0.5, "undead": 1.0},
        }
        return table.get(attack_element, {}).get(defense_element, 1.0)
