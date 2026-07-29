"""Swarm formations — party positioning patterns for coordinated combat.

Provides formation definitions (line, box, spread, protect, wedge, vanguard)
and a FormationManager that calculates positions for each bot based on
their role and the current formation type.

All positions are relative to an anchor (formation center), typically
the tank or the party leader's position.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Formation types
# ────────────────────────────────────────────────────────────────

class FormationType(StrEnum):
    """Named formation patterns the swarm can adopt."""
    LINE = "line"             # Side-by-side, tank center
    BOX = "box"               # 2×2 or 3×3 grid
    SPREAD = "spread"         # Maximum distance between members
    PROTECT = "protect"       # Surround and protect healer/backline
    WEDGE = "wedge"           # Arrowhead, tank at point
    VANGUARD = "vanguard"     # Tank front, DPS mid, healer back
    DIAMOND = "diamond"       # Four corners
    COLUMN = "column"         # Single file
    RANDOM = "random"         # Scattered (for anti-AoE)
    RETREAT = "retreat"       # Fall back, tank last


# ────────────────────────────────────────────────────────────────
#  Position and slot definitions
# ────────────────────────────────────────────────────────────────

@dataclass
class FormationSlot:
    """A slot in a formation with relative position and role preference."""
    role: str                  # Preferred role for this slot
    offset_x: int              # Relative X offset from anchor
    offset_y: int              # Relative Y offset from anchor
    priority: int = 0          # Lower = assigned first


@dataclass
class FormationDefinition:
    """Defines a formation's slot layout and metadata."""
    type: FormationType
    description: str
    slots: list[FormationSlot]
    min_bots: int = 1
    max_range: int = 9         # Max cells from formation center


# ────────────────────────────────────────────────────────────────
#  Pre-defined formations
# ────────────────────────────────────────────────────────────────

FORMATION_DEFS: dict[FormationType, FormationDefinition] = {
    FormationType.LINE: FormationDefinition(
        type=FormationType.LINE, description="Side-by-side line",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=3, priority=0),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=3, priority=1),
            FormationSlot(role="dps_ranged", offset_x=2, offset_y=3, priority=2),
            FormationSlot(role="healer", offset_x=-1, offset_y=1, priority=3),
            FormationSlot(role="dps_magic", offset_x=1, offset_y=1, priority=4),
            FormationSlot(role="buffer", offset_x=0, offset_y=0, priority=5),
            FormationSlot(role="support", offset_x=-3, offset_y=2, priority=6),
            FormationSlot(role="debuff", offset_x=3, offset_y=2, priority=6),
        ], min_bots=1,
    ),
    FormationType.BOX: FormationDefinition(
        type=FormationType.BOX, description="2×2 box formation",
        slots=[
            FormationSlot(role="tank", offset_x=-1, offset_y=3, priority=0),
            FormationSlot(role="dps_melee", offset_x=1, offset_y=3, priority=1),
            FormationSlot(role="healer", offset_x=-1, offset_y=1, priority=2),
            FormationSlot(role="dps_ranged", offset_x=1, offset_y=1, priority=3),
            FormationSlot(role="dps_magic", offset_x=-1, offset_y=-1, priority=4),
            FormationSlot(role="buffer", offset_x=1, offset_y=-1, priority=5),
            FormationSlot(role="support", offset_x=0, offset_y=2, priority=6),
        ], min_bots=2, max_range=7,
    ),
    FormationType.SPREAD: FormationDefinition(
        type=FormationType.SPREAD, description="Maximum spread for AoE avoidance",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=6, priority=0),
            FormationSlot(role="dps_ranged", offset_x=-6, offset_y=3, priority=1),
            FormationSlot(role="dps_ranged", offset_x=6, offset_y=3, priority=1),
            FormationSlot(role="dps_magic", offset_x=-4, offset_y=0, priority=2),
            FormationSlot(role="dps_magic", offset_x=4, offset_y=0, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=-3, priority=3),
            FormationSlot(role="dps_melee", offset_x=-3, offset_y=4, priority=4),
            FormationSlot(role="dps_melee", offset_x=3, offset_y=4, priority=4),
        ], min_bots=2, max_range=10,
    ),
    FormationType.PROTECT: FormationDefinition(
        type=FormationType.PROTECT, description="Protect healer/caster",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=4, priority=0),
            FormationSlot(role="dps_melee", offset_x=-3, offset_y=2, priority=1),
            FormationSlot(role="dps_melee", offset_x=3, offset_y=2, priority=1),
            FormationSlot(role="healer", offset_x=0, offset_y=0, priority=2),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=0, priority=2),
            FormationSlot(role="dps_ranged", offset_x=-4, offset_y=1, priority=3),
            FormationSlot(role="dps_ranged", offset_x=4, offset_y=1, priority=3),
            FormationSlot(role="buffer", offset_x=-2, offset_y=-1, priority=4),
        ], min_bots=2, max_range=8,
    ),
    FormationType.WEDGE: FormationDefinition(
        type=FormationType.WEDGE, description="Arrowhead formation",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=4, priority=0),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=3, priority=1),
            FormationSlot(role="dps_melee", offset_x=2, offset_y=3, priority=1),
            FormationSlot(role="dps_ranged", offset_x=-4, offset_y=2, priority=2),
            FormationSlot(role="dps_ranged", offset_x=4, offset_y=2, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=1, priority=3),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=0, priority=4),
        ], min_bots=2, max_range=9,
    ),
    FormationType.VANGUARD: FormationDefinition(
        type=FormationType.VANGUARD, description="Tank front, DPS mid, support back",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=5, priority=0),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=3, priority=1),
            FormationSlot(role="dps_melee", offset_x=2, offset_y=3, priority=1),
            FormationSlot(role="dps_ranged", offset_x=-4, offset_y=2, priority=2),
            FormationSlot(role="dps_ranged", offset_x=4, offset_y=2, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=0, priority=3),
            FormationSlot(role="dps_magic", offset_x=-2, offset_y=1, priority=4),
            FormationSlot(role="buffer", offset_x=2, offset_y=1, priority=4),
            FormationSlot(role="support", offset_x=0, offset_y=-1, priority=5),
        ], min_bots=2, max_range=9,
    ),
    FormationType.DIAMOND: FormationDefinition(
        type=FormationType.DIAMOND, description="Diamond formation",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=5, priority=0),
            FormationSlot(role="dps_melee", offset_x=-3, offset_y=2, priority=1),
            FormationSlot(role="dps_melee", offset_x=3, offset_y=2, priority=1),
            FormationSlot(role="healer", offset_x=0, offset_y=-1, priority=2),
            FormationSlot(role="dps_ranged", offset_x=-4, offset_y=1, priority=3),
            FormationSlot(role="dps_ranged", offset_x=4, offset_y=1, priority=3),
        ], min_bots=3, max_range=9,
    ),
    FormationType.COLUMN: FormationDefinition(
        type=FormationType.COLUMN, description="Single file",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=5, priority=0),
            FormationSlot(role="dps_melee", offset_x=0, offset_y=3, priority=1),
            FormationSlot(role="healer", offset_x=0, offset_y=1, priority=2),
            FormationSlot(role="dps_ranged", offset_x=0, offset_y=-1, priority=3),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=-3, priority=4),
            FormationSlot(role="buffer", offset_x=0, offset_y=-5, priority=5),
        ], min_bots=2, max_range=6,
    ),
    FormationType.RETREAT: FormationDefinition(
        type=FormationType.RETREAT, description="Ordered retreat",
        slots=[
            FormationSlot(role="healer", offset_x=0, offset_y=-5, priority=0),
            FormationSlot(role="dps_magic", offset_x=-2, offset_y=-4, priority=1),
            FormationSlot(role="dps_ranged", offset_x=2, offset_y=-4, priority=1),
            FormationSlot(role="dps_melee", offset_x=-3, offset_y=-2, priority=2),
            FormationSlot(role="dps_melee", offset_x=3, offset_y=-2, priority=2),
            FormationSlot(role="tank", offset_x=0, offset_y=0, priority=3),
        ], min_bots=2, max_range=7,
    ),
}


def get_formation(type: FormationType) -> FormationDefinition | None:
    """Get a formation definition by type."""
    return FORMATION_DEFS.get(type)


def list_formations() -> list[FormationType]:
    """List all available formation types."""
    return list(FORMATION_DEFS.keys())


# ────────────────────────────────────────────────────────────────
#  Position calculator
# ────────────────────────────────────────────────────────────────

def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ────────────────────────────────────────────────────────────────
#  FormationManager
# ────────────────────────────────────────────────────────────────

class FormationManager:
    """Assigns bots to formation positions based on their roles.

    Used by both the SwarmCoordinator (leader decides positions for all bots)
    and individual bots (to know where they should stand).
    """

    def __init__(self) -> None:
        self._current_formation: FormationType = FormationType.VANGUARD
        self._anchor_x: int = 0
        self._anchor_y: int = 0

    def assign_positions(
        self,
        bots: list[dict[str, Any]],
        formation: FormationType = FormationType.VANGUARD,
        anchor_x: int = 0,
        anchor_y: int = 0,
    ) -> dict[str, dict[str, int]]:
        """Assign formation positions to bots based on their roles.

        Args:
            bots: List of bot dicts, each with 'name' and 'role' keys.
            formation: The formation type to use.
            anchor_x, anchor_y: Formation center position.

        Returns:
            Dict mapping bot_name -> {'x': pos_x, 'y': pos_y}
        """
        self._current_formation = formation
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y

        formation_def = FORMATION_DEFS.get(formation)
        if not formation_def:
            logger.warning("Unknown formation: %s, falling back to VANGUARD", formation)
            formation_def = FORMATION_DEFS[FormationType.VANGUARD]
            self._current_formation = FormationType.VANGUARD

        if len(bots) < formation_def.min_bots:
            logger.debug(
                "Too few bots (%d) for %s formation (need %d)",
                len(bots), formation, formation_def.min_bots,
            )

        # Score each slot for each bot based on role match
        sorted_slots = sorted(formation_def.slots, key=lambda s: s.priority)
        positions: dict[str, dict[str, int]] = {}
        assigned_slot_indices: set[int] = set()

        for bot in bots:
            bot_name = bot.get("name", "")
            bot_role = bot.get("role", "idle")
            if not bot_name:
                continue

            best_slot_idx = -1
            best_score = -999

            for idx, slot in enumerate(sorted_slots):
                if idx in assigned_slot_indices:
                    continue
                score = self._role_match_score(bot_role, slot.role)
                if score > best_score:
                    best_score = score
                    best_slot_idx = idx

            if best_slot_idx >= 0:
                assigned_slot_indices.add(best_slot_idx)
                slot = sorted_slots[best_slot_idx]
                positions[bot_name] = {
                    "x": anchor_x + slot.offset_x,
                    "y": anchor_y + slot.offset_y,
                }
            else:
                # No slot left — fall back near anchor
                fallback = self._fallback_position(bot_name, bots, positions)
                positions[bot_name] = fallback

        return positions

    def _role_match_score(self, bot_role: str, slot_role: str) -> int:
        """Score how well a bot role matches a formation slot role."""
        if bot_role == slot_role:
            return 100
        # Role group compatibility
        groups: dict[str, list[str]] = {
            "tank": ["tank", "dps_melee"],
            "dps_melee": ["dps_melee", "tank", "dps_ranged", "dps_magic"],
            "dps_ranged": ["dps_ranged", "dps_melee", "dps_magic"],
            "dps_magic": ["dps_magic", "dps_ranged"],
            "healer": ["healer", "support", "buffer"],
            "support": ["support", "buffer", "healer"],
            "buffer": ["buffer", "support"],
            "debuff": ["debuff", "dps_magic", "dps_ranged"],
        }
        compatible = groups.get(bot_role, [])
        if slot_role in compatible:
            return 50
        # Broad fallback: any combat role can fill any combat slot
        combat_roles = {"tank", "dps_melee", "dps_ranged", "dps_magic", "healer", "support", "buffer", "debuff"}
        if bot_role in combat_roles and slot_role in combat_roles:
            return 20
        return 0

    def _fallback_position(
        self,
        bot_name: str,
        bots: list[dict[str, Any]],
        assigned: dict[str, dict[str, int]],
    ) -> dict[str, int]:
        """Generate a fallback position near the anchor when no slot is free."""
        # Place at increasing distance from anchor
        taken = len(assigned)
        spread = 3 + taken * 2
        angle = taken * 1.256  # Golden angle approximation
        return {
            "x": self._anchor_x + int(spread * math.cos(angle)),
            "y": self._anchor_y + int(spread * math.sin(angle)),
        }

    def positions_to_commands(
        self,
        positions: dict[str, dict[str, int]],
        bot_name: str,
    ) -> list[dict[str, Any]]:
        """Convert formation positions into move commands for this bot.

        Returns HeuristicAction-like dicts.
        """
        my_pos = positions.get(bot_name)
        if not my_pos:
            return []
        return [{
            "kind": "command",
            "command": f"move {my_pos['x']} {my_pos['y']}",
            "confidence": 0.85,
            "domain": "swarm",
            "reason": f"Move to formation position ({my_pos['x']}, {my_pos['y']})",
        }]

    def select_formation_for_situation(
        self,
        threat_level: float,
        team_hp_avg: float,
        target_count: int,
        aoe_risk: bool,
        bot_count: int,
        roles: list[str],
    ) -> FormationType:
        """Select the best formation for the current situation."""
        # Emergency: retreat on critically low HP
        if team_hp_avg < 0.25:
            return FormationType.RETREAT

        # High threat: protect healers
        if threat_level > 0.8:
            if "healer" in roles or "dps_magic" in roles:
                return FormationType.PROTECT
            return FormationType.VANGUARD

        # AoE risk: spread
        if aoe_risk:
            return FormationType.SPREAD

        # Multiple targets: diamond
        if target_count >= 3 and bot_count >= 4:
            return FormationType.DIAMOND

        # Single tough target: wedge
        if target_count <= 1 and bot_count >= 2:
            return FormationType.WEDGE

        # Default: vanguard for mixed groups, line for casters
        if any(r in roles for r in ("dps_melee", "tank")):
            return FormationType.VANGUARD
        return FormationType.LINE
