"""Party Positioning System — formation-based positioning for coordinated party combat.

Pro-botting insight: coordinated party positioning is the difference between
a group of solo players and an actual party. This module assigns positions
based on class roles and formation type to maximize party effectiveness.

Formation types:
  - LINE: All members in a horizontal line (ranged-heavy parties)
  - WEDGE: Tank front, melee flanks, ranged/magic back, support mid (standard)
  - SCATTER: Spread out to avoid AoE (no tank, all DPS)
  - PROTECT: Tight formation around a protected target (healer/support focus)

Positioning rules:
  - Tank: front-center (highest priority for aggro)
  - Melee DPS: flank (left/right of tank)
  - Ranged DPS: back row (behind melee)
  - Magic DPS: back row (max range from front)
  - Support: mid-rear (between tank and back row)
  - Healer: mid (can reach everyone)
  - Minimum 3 cells spacing between party members (AoE dodge)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────


class FormationType(str, Enum):
    """Party formation types for coordinated positioning."""
    LINE = "line"           # Horizontal line — ranged-heavy parties
    WEDGE = "wedge"         # Tank front, melee flanks, ranged back — standard
    SCATTER = "scatter"     # Spread out — no tank, all DPS
    PROTECT = "protect"     # Tight around protected target — healer focus


class BotRole(str, Enum):
    """Combat roles for party positioning."""
    TANK = "tank"
    MELEE_DPS = "melee_dps"
    RANGED_DPS = "ranged_dps"
    MAGIC_DPS = "magic_dps"
    SUPPORT = "support"
    HEALER = "healer"


# ── Role-to-class mapping ───────────────────────────────────────────────────

# Maps RO class names (lowercase) to their primary combat role
CLASS_TO_ROLE: dict[str, BotRole] = {
    # Swordsman tree
    "swordman": BotRole.MELEE_DPS,
    "knight": BotRole.TANK,
    "lord knight": BotRole.TANK,
    "crusader": BotRole.TANK,
    "paladin": BotRole.TANK,
    "baby knight": BotRole.TANK,
    "baby lord knight": BotRole.TANK,
    "baby crusader": BotRole.TANK,
    "baby paladin": BotRole.TANK,
    "rune knight": BotRole.TANK,
    "royal guard": BotRole.TANK,
    "dark knight": BotRole.TANK,
    "imperial guard": BotRole.TANK,

    # Mage tree
    "mage": BotRole.MAGIC_DPS,
    "wizard": BotRole.MAGIC_DPS,
    "high wizard": BotRole.MAGIC_DPS,
    "sage": BotRole.MAGIC_DPS,
    "professor": BotRole.MAGIC_DPS,
    "baby wizard": BotRole.MAGIC_DPS,
    "baby high wizard": BotRole.MAGIC_DPS,
    "baby sage": BotRole.MAGIC_DPS,
    "baby professor": BotRole.MAGIC_DPS,
    "warlock": BotRole.MAGIC_DPS,
    "sorcerer": BotRole.MAGIC_DPS,
    "arch mage": BotRole.MAGIC_DPS,
    "elemental master": BotRole.MAGIC_DPS,

    # Archer tree
    "archer": BotRole.RANGED_DPS,
    "hunter": BotRole.RANGED_DPS,
    "sniper": BotRole.RANGED_DPS,
    "bard": BotRole.RANGED_DPS,
    "clown": BotRole.RANGED_DPS,
    "dancer": BotRole.RANGED_DPS,
    "gypsy": BotRole.RANGED_DPS,
    "baby hunter": BotRole.RANGED_DPS,
    "baby sniper": BotRole.RANGED_DPS,
    "baby bard": BotRole.RANGED_DPS,
    "baby clown": BotRole.RANGED_DPS,
    "baby dancer": BotRole.RANGED_DPS,
    "baby gypsy": BotRole.RANGED_DPS,
    "ranger": BotRole.RANGED_DPS,
    "minstrel": BotRole.RANGED_DPS,
    "wanderer": BotRole.RANGED_DPS,

    # Acolyte tree
    "acolyte": BotRole.HEALER,
    "priest": BotRole.HEALER,
    "high priest": BotRole.HEALER,
    "monk": BotRole.MELEE_DPS,
    "champion": BotRole.MELEE_DPS,
    "baby priest": BotRole.HEALER,
    "baby high priest": BotRole.HEALER,
    "baby monk": BotRole.MELEE_DPS,
    "baby champion": BotRole.MELEE_DPS,
    "archbishop": BotRole.HEALER,
    "shura": BotRole.MELEE_DPS,
    "cardinal": BotRole.HEALER,
    "inquisitor": BotRole.MELEE_DPS,

    # Merchant tree
    "merchant": BotRole.SUPPORT,
    "blacksmith": BotRole.MELEE_DPS,
    "whitesmith": BotRole.MELEE_DPS,
    "alchemist": BotRole.SUPPORT,
    "creator": BotRole.SUPPORT,
    "baby blacksmith": BotRole.MELEE_DPS,
    "baby whitesmith": BotRole.MELEE_DPS,
    "baby alchemist": BotRole.SUPPORT,
    "baby creator": BotRole.SUPPORT,
    "mechanic": BotRole.MELEE_DPS,
    "genetic": BotRole.SUPPORT,

    # Thief tree
    "thief": BotRole.MELEE_DPS,
    "assassin": BotRole.MELEE_DPS,
    "assassin cross": BotRole.MELEE_DPS,
    "rogue": BotRole.MELEE_DPS,
    "stalker": BotRole.MELEE_DPS,
    "baby assassin": BotRole.MELEE_DPS,
    "baby assassin cross": BotRole.MELEE_DPS,
    "baby rogue": BotRole.MELEE_DPS,
    "baby stalker": BotRole.MELEE_DPS,
    "guillotine cross": BotRole.MELEE_DPS,
    "shadow chaser": BotRole.MELEE_DPS,
    "abyss chaser": BotRole.MELEE_DPS,
    "night shadow": BotRole.MELEE_DPS,

    # Taekwon tree
    "taekwon": BotRole.MELEE_DPS,
    "soul linker": BotRole.MAGIC_DPS,
    "star gladiator": BotRole.MELEE_DPS,

    # Gunslinger
    "gunslinger": BotRole.RANGED_DPS,
    "rebel": BotRole.RANGED_DPS,

    # Ninja
    "ninja": BotRole.MELEE_DPS,
    "kagerou": BotRole.MELEE_DPS,
    "oboro": BotRole.MAGIC_DPS,

    # Super Novice
    "super novice": BotRole.SUPPORT,
    "expanded super novice": BotRole.SUPPORT,

    # Doram
    "summoner": BotRole.MAGIC_DPS,
    "spirit summoner": BotRole.MAGIC_DPS,
}

# ── Position data ──────────────────────────────────────────────────────────


@dataclass
class Position:
    """A position in the party formation."""
    x: int
    y: int
    priority: int = 0
    """Lower number = higher priority (assigned first)."""

    role: str = ""
    """The role this position is intended for."""

    label: str = ""
    """Human-readable label for debugging."""


@dataclass
class PartyMember:
    """A member of the party with role and class info."""
    name: str
    class_name: str
    role: BotRole = BotRole.MELEE_DPS
    level: int = 1
    x: int = 0
    y: int = 0
    is_online: bool = True


# ── Position Assigner ────────────────────────────────────────────────────────


class PositionAssigner:
    """Assigns party positions based on class roles and formation type.

    Features:
      - 4 formation types: LINE, WEDGE, SCATTER, PROTECT
      - Automatic formation selection from party composition
      - Role-based position assignment with priority ordering
      - Minimum 3-cell spacing between party members (AoE dodge)
      - Map boundary awareness
      - Thread-safe
    """

    MIN_SPACING = 3  # Minimum cells between party members
    """Minimum distance between party members to avoid AoE overlap."""

    def __init__(self) -> None:
        self._formation_cache: dict[str, Any] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def get_position(
        self,
        bot_class: str,
        formation: FormationType,
        map_width: int = 200,
        map_height: int = 200,
        anchor_x: int = 100,
        anchor_y: int = 100,
        assigned_positions: list[Position] | None = None,
    ) -> Position:
        """Get the ideal position for a bot class in a given formation.

        Args:
            bot_class: The RO class name (e.g., 'knight', 'wizard').
            formation: The formation type to use.
            map_width: Width of the current map in cells.
            map_height: Height of the current map in cells.
            anchor_x: X coordinate of the formation anchor (usually tank or center).
            anchor_y: Y coordinate of the formation anchor.
            assigned_positions: Already-assigned positions to avoid overlap.

        Returns:
            Position with (x, y, priority) for this bot.
        """
        role = self.get_role_for_class(bot_class)
        return self._get_position_for_role(
            role=role,
            formation=formation,
            map_width=map_width,
            map_height=map_height,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            assigned_positions=assigned_positions or [],
        )

    def get_formation_from_party(
        self,
        party_members: list[PartyMember],
    ) -> FormationType:
        """Determine the best formation type from party composition.

        Rules:
          - 1 tank + 1+ DPS + 1 support -> WEDGE
          - 0 tank + 2+ DPS + 1 healer -> SCATTER
          - 1+ ranged only -> LINE
          - Default: WEDGE

        Args:
            party_members: List of party members with roles.

        Returns:
            The recommended FormationType.
        """
        if not party_members:
            return FormationType.WEDGE

        roles = [m.role for m in party_members]
        has_tank = BotRole.TANK in roles
        has_healer = BotRole.HEALER in roles
        has_support = BotRole.SUPPORT in roles
        dps_count = sum(1 for r in roles if r in (
            BotRole.MELEE_DPS, BotRole.RANGED_DPS, BotRole.MAGIC_DPS
        ))
        ranged_only = all(r in (BotRole.RANGED_DPS, BotRole.MAGIC_DPS, BotRole.HEALER) for r in roles)

        if has_tank and dps_count >= 1 and (has_support or has_healer):
            return FormationType.WEDGE

        if not has_tank and dps_count >= 2 and has_healer:
            return FormationType.SCATTER

        if ranged_only and len(party_members) >= 2:
            return FormationType.LINE

        return FormationType.WEDGE

    def get_role_for_class(self, class_name: str) -> BotRole:
        """Map an RO class name to its primary combat role.

        Args:
            class_name: The RO class name (case-insensitive).

        Returns:
            The BotRole for this class, defaulting to MELEE_DPS.
        """
        key = class_name.lower().strip()
        return CLASS_TO_ROLE.get(key, BotRole.MELEE_DPS)

    def assign_all_positions(
        self,
        party_members: list[PartyMember],
        formation: FormationType | None = None,
        map_width: int = 200,
        map_height: int = 200,
        anchor_x: int = 100,
        anchor_y: int = 100,
    ) -> dict[str, Position]:
        """Assign positions for all party members.

        Automatically determines formation if not provided.
        Assigns positions in priority order (tank first, then healer, etc.).

        Args:
            party_members: List of party members.
            formation: Formation type (auto-detected if None).
            map_width: Map width in cells.
            map_height: Map height in cells.
            anchor_x: Formation anchor X.
            anchor_y: Formation anchor Y.

        Returns:
            dict of member_name -> Position.
        """
        if not party_members:
            return {}

        if formation is None:
            formation = self.get_formation_from_party(party_members)

        # Sort by role priority: tank > healer > support > melee > ranged > magic
        role_priority = {
            BotRole.TANK: 0,
            BotRole.HEALER: 1,
            BotRole.SUPPORT: 2,
            BotRole.MELEE_DPS: 3,
            BotRole.RANGED_DPS: 4,
            BotRole.MAGIC_DPS: 5,
        }
        sorted_members = sorted(
            party_members,
            key=lambda m: role_priority.get(m.role, 99),
        )

        assigned: dict[str, Position] = {}
        assigned_positions: list[Position] = []

        for member in sorted_members:
            pos = self._get_position_for_role(
                role=member.role,
                formation=formation,
                map_width=map_width,
                map_height=map_height,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                assigned_positions=assigned_positions,
            )
            assigned[member.name] = pos
            assigned_positions.append(pos)

        return assigned

    def get_formation_spacing(self, formation: FormationType) -> int:
        """Get the recommended spacing between party members for a formation.

        Args:
            formation: The formation type.

        Returns:
            Minimum cell spacing.
        """
        if formation == FormationType.PROTECT:
            return 2  # Tighter spacing for protection
        elif formation == FormationType.SCATTER:
            return 5  # Wider spacing for AoE avoidance
        return self.MIN_SPACING

    def get_formation_description(self, formation: FormationType) -> str:
        """Get a human-readable description of a formation type."""
        descriptions = {
            FormationType.LINE: (
                "Horizontal line formation — all members spread evenly "
                "in a row. Best for ranged-heavy parties."
            ),
            FormationType.WEDGE: (
                "Wedge formation — tank at front-center, melee on flanks, "
                "ranged/magic in back, support mid-rear. Standard all-purpose."
            ),
            FormationType.SCATTER: (
                "Scatter formation — members spread wide to avoid AoE damage. "
                "Best for parties without a tank."
            ),
            FormationType.PROTECT: (
                "Protect formation — tight cluster around a protected target. "
                "Best when defending a healer or VIP."
            ),
        }
        return descriptions.get(formation, "Unknown formation")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_position_for_role(
        self,
        role: BotRole,
        formation: FormationType,
        map_width: int,
        map_height: int,
        anchor_x: int,
        anchor_y: int,
        assigned_positions: list[Position],
    ) -> Position:
        """Get the ideal position for a role in a given formation.

        Position offsets are relative to the anchor point (usually tank/center).
        Each formation has a different layout.

        Priority: lower number = assigned first (higher priority for aggro).
        """
        spacing = self.get_formation_spacing(formation)

        if formation == FormationType.LINE:
            return self._line_position(role, anchor_x, anchor_y, spacing, assigned_positions)
        elif formation == FormationType.WEDGE:
            return self._wedge_position(role, anchor_x, anchor_y, spacing, assigned_positions)
        elif formation == FormationType.SCATTER:
            return self._scatter_position(role, anchor_x, anchor_y, spacing, assigned_positions)
        elif formation == FormationType.PROTECT:
            return self._protect_position(role, anchor_x, anchor_y, spacing, assigned_positions)

        # Fallback to wedge
        return self._wedge_position(role, anchor_x, anchor_y, spacing, assigned_positions)

    def _line_position(
        self,
        role: BotRole,
        ax: int,
        ay: int,
        spacing: int,
        assigned: list[Position],
    ) -> Position:
        """LINE formation: all members in a horizontal line.

        Tank: center
        Melee DPS: left of tank
        Ranged DPS: right of tank
        Magic DPS: far right
        Support: between tank and melee
        Healer: between tank and ranged
        """
        offset = len(assigned) * spacing
        side = 1 if len(assigned) % 2 == 0 else -1
        x = ax + (side * (offset // 2 + 1) * spacing)
        y = ay

        # Role-specific adjustments
        if role == BotRole.TANK:
            x, y = ax, ay
            priority = 0
            label = "tank-center"
        elif role == BotRole.HEALER:
            x = ax - spacing
            y = ay
            priority = 1
            label = "healer-left"
        elif role == BotRole.SUPPORT:
            x = ax + spacing
            y = ay
            priority = 2
            label = "support-right"
        elif role == BotRole.MELEE_DPS:
            x = ax - (spacing * 2)
            y = ay
            priority = 3
            label = "melee-left"
        elif role == BotRole.RANGED_DPS:
            x = ax + (spacing * 2)
            y = ay
            priority = 4
            label = "ranged-right"
        elif role == BotRole.MAGIC_DPS:
            x = ax + (spacing * 3)
            y = ay
            priority = 5
            label = "magic-far-right"
        else:
            priority = 6
            label = "fill"

        return self._clamp_position(x, y, priority, role.value, label)

    def _wedge_position(
        self,
        role: BotRole,
        ax: int,
        ay: int,
        spacing: int,
        assigned: list[Position],
    ) -> Position:
        """WEDGE formation: standard party layout.

        Tank: front-center (highest priority for aggro)
        Melee DPS: flank (left/right of tank)
        Ranged DPS: back row (behind melee)
        Magic DPS: back row (max range from front)
        Support: mid-rear (between tank and back row)
        Healer: mid (can reach everyone)
        """
        if role == BotRole.TANK:
            x, y = ax, ay - (spacing * 2)  # Front
            priority = 0
            label = "tank-front"
        elif role == BotRole.HEALER:
            x, y = ax, ay  # Center
            priority = 1
            label = "healer-mid"
        elif role == BotRole.SUPPORT:
            x, y = ax, ay + spacing  # Mid-rear
            priority = 2
            label = "support-mid-rear"
        elif role == BotRole.MELEE_DPS:
            # Alternate left/right
            melee_count = sum(1 for p in assigned if p.role == BotRole.MELEE_DPS.value)
            side = -1 if melee_count % 2 == 0 else 1
            x = ax + (side * spacing)
            y = ay - spacing  # Behind tank
            priority = 3
            label = f"melee-flank-{'left' if side < 0 else 'right'}"
        elif role == BotRole.RANGED_DPS:
            ranged_count = sum(1 for p in assigned if p.role == BotRole.RANGED_DPS.value)
            side = -1 if ranged_count % 2 == 0 else 1
            x = ax + (side * spacing * 2)
            y = ay + (spacing * 2)  # Back row
            priority = 4
            label = f"ranged-back-{'left' if side < 0 else 'right'}"
        elif role == BotRole.MAGIC_DPS:
            magic_count = sum(1 for p in assigned if p.role == BotRole.MAGIC_DPS.value)
            side = -1 if magic_count % 2 == 0 else 1
            x = ax + (side * spacing * 2)
            y = ay + (spacing * 3)  # Far back
            priority = 5
            label = f"magic-far-back-{'left' if side < 0 else 'right'}"
        else:
            x, y = ax + spacing, ay + spacing
            priority = 6
            label = "fill"

        return self._clamp_position(x, y, priority, role.value, label)

    def _scatter_position(
        self,
        role: BotRole,
        ax: int,
        ay: int,
        spacing: int,
        assigned: list[Position],
    ) -> Position:
        """SCATTER formation: spread out to avoid AoE.

        No tank — all members spread in a wide arc.
        Healer: center-ish
        DPS: spread in a semicircle
        """
        total = len(assigned) + 1
        angle_step = math.pi / max(total, 2)
        start_angle = -math.pi / 2  # Start from top

        if role == BotRole.HEALER:
            # Healer stays near center
            x, y = ax, ay
            priority = 0
            label = "healer-center"
        elif role == BotRole.SUPPORT:
            x, y = ax + spacing, ay
            priority = 1
            label = "support-mid"
        else:
            # DPS spread in semicircle
            idx = len(assigned)
            angle = start_angle + (idx * angle_step)
            radius = spacing * 3
            x = ax + int(radius * math.cos(angle))
            y = ay + int(radius * math.sin(angle))
            priority = 2 + idx
            label = f"dps-scatter-{idx}"

        return self._clamp_position(x, y, priority, role.value, label)

    def _protect_position(
        self,
        role: BotRole,
        ax: int,
        ay: int,
        spacing: int,
        assigned: list[Position],
    ) -> Position:
        """PROTECT formation: tight cluster around protected target.

        Protected target: center
        Tank: front
        Melee: sides
        Ranged/Magic: behind
        Healer: very close to protected target
        """
        if role == BotRole.TANK:
            x, y = ax, ay - spacing  # Front
            priority = 0
            label = "tank-front"
        elif role == BotRole.HEALER:
            x, y = ax, ay  # Same as protected
            priority = 1
            label = "healer-protected"
        elif role == BotRole.SUPPORT:
            x, y = ax + spacing, ay
            priority = 2
            label = "support-right"
        elif role == BotRole.MELEE_DPS:
            side = -1 if len(assigned) % 2 == 0 else 1
            x = ax + (side * spacing)
            y = ay
            priority = 3
            label = f"melee-side-{'left' if side < 0 else 'right'}"
        elif role == BotRole.RANGED_DPS:
            x, y = ax, ay + spacing
            priority = 4
            label = "ranged-behind"
        elif role == BotRole.MAGIC_DPS:
            x, y = ax, ay + (spacing * 2)
            priority = 5
            label = "magic-far-behind"
        else:
            x, y = ax + spacing, ay + spacing
            priority = 6
            label = "fill"

        return self._clamp_position(x, y, priority, role.value, label)

    def _clamp_position(
        self,
        x: int,
        y: int,
        priority: int,
        role: str,
        label: str,
        map_width: int = 200,
        map_height: int = 200,
    ) -> Position:
        """Clamp position to map boundaries and ensure valid coordinates."""
        # Add some random jitter to avoid exact same positions
        x = x + random.randint(-1, 1)
        y = y + random.randint(-1, 1)

        # Clamp to map boundaries (with margin)
        margin = 5
        x = max(margin, min(map_width - margin, x))
        y = max(margin, min(map_height - margin, y))

        return Position(x=x, y=y, priority=priority, role=role, label=label)


# ── Singleton factory ───────────────────────────────────────────────────────

_position_assigner: PositionAssigner | None = None


def get_position_assigner() -> PositionAssigner:
    """Get or create the singleton PositionAssigner."""
    global _position_assigner
    if _position_assigner is None:
        _position_assigner = PositionAssigner()
    return _position_assigner
