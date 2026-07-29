"""Base class for all combat tactics modules.

Each tactics module encodes a combat role (tank, melee DPS, ranged DPS, magic DPS,
support, hybrid). Tactics are stateless strategy objects that select targets, skills,
and positioning based on the current combat context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

# Direct import to avoid triggering full import chain
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


@dataclass
class TargetInfo:
    """Scored target information."""
    actor_id: int
    name: str
    score: float
    hp_pct: float
    distance: int
    element: str = "neutral"
    size: str = "medium"
    race: str = "formless"
    is_boss: bool = False
    is_casting: bool = False
    is_aggressive: bool = True
    estimated_value: float = 0.0
    danger_level: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TacticsContext:
    """Full combat context passed to tactics modules for decision-making.

    Carries all information a tactics module needs: character state, visible
    monsters, party members, cooldowns, and environmental metadata.
    """
    # Character state
    my_hp_pct: float = 1.0
    my_sp_pct: float = 1.0
    my_hp: int = 1
    my_max_hp: int = 1
    my_sp: int = 0
    my_max_sp: int = 1
    my_job_class: str = "novice"
    my_base_level: int = 1
    my_weapon_type: str = "dagger"
    my_weapon_element: str = "neutral"

    # Combat state
    current_target_id: int = 0
    aggro_count: int = 0
    enemies_nearby: int = 0
    party_members_nearby: int = 0
    has_party: bool = False
    is_sitting: bool = False
    is_in_combat: bool = False
    map_name: str = ""

    # Monsters
    monsters: list[dict[str, Any]] = field(default_factory=list)

    # Party members (list of dicts with name, hp_pct, distance, class)
    party_members: list[dict[str, Any]] = field(default_factory=list)

    # Cooldowns: skill_name -> remaining_seconds
    cooldowns: dict[str, float] = field(default_factory=dict)

    # Active buffs
    active_buffs: list[str] = field(default_factory=list)

    # Available skills
    available_skills: list[str] = field(default_factory=list)

    # Equipped cards
    equipped_cards: list[str] = field(default_factory=list)

    # Config overrides
    config: dict[str, Any] = field(default_factory=dict)


class BaseTactics:
    """Base class for all combat tactics modules.

    Subclasses implement select_target, select_skill, evaluate_positioning,
    and assess_buffs to realize their combat role.

    ClassVars:
        name: Unique identifier for this tactics module.
        priority: Lower = higher priority (tank: 10, support: 20, dps: 50).
        description: Human-readable role description.
    """

    name: ClassVar[str] = "base"
    priority: ClassVar[int] = 100
    description: ClassVar[str] = "Base tactics — fallback behavior"

    # Role type for HeuristicAction metadata
    role_type: ClassVar[str] = "unknown"

    def select_target(self, ctx: TacticsContext) -> TargetInfo | None:
        """Select the best target from ctx.monsters.

        Args:
            ctx: Current combat context with monsters, character state, etc.

        Returns:
            TargetInfo with the best target, or None if no suitable target.
        """
        return None

    def select_skill(self, ctx: TacticsContext, target: TargetInfo | None) -> tuple[str, int] | None:
        """Select the best skill to use against the current target.

        Args:
            ctx: Current combat context.
            target: Current target info (may be None if no target).

        Returns:
            Tuple of (skill_name, skill_level) or None to basic attack.
        """
        return None

    def evaluate_positioning(self, ctx: TacticsContext, target: TargetInfo | None) -> dict[str, Any] | None:
        """Evaluate ideal positioning and return a movement intent.

        Returns dict with keys:
            - move_x, move_y: Desired position.
            - reason: Why this position.
            - urgency: 0.0 (relaxed) to 1.0 (emergency).

        Returns None if current position is acceptable.
        """
        return None

    def assess_buffs(self, ctx: TacticsContext) -> list[str]:
        """Return list of buff skill names that should be active.

        Args:
            ctx: Current combat context.

        Returns:
            List of buff skill names to cast (empty if all buffs are up).
        """
        return []

    def assess_emergency(self, ctx: TacticsContext) -> HeuristicAction | None:
        """Check for emergency actions (flee, potion, emergency heal).

        Returns a HeuristicAction if an emergency response is needed.

        Args:
            ctx: Current combat context.

        Returns:
            HeuristicAction for the emergency, or None.
        """
        return None

    def build_rotation(self, ctx: TacticsContext, target: TargetInfo | None) -> list[tuple[str, int]]:
        """Build a full skill rotation for the current situation.

        Returns list of (skill_name, skill_level) tuples to execute in order.
        Empty list means basic attack only.
        """
        skill = self.select_skill(ctx, target)
        if skill:
            return [skill]
        return []

    def _find_best_by_score(self, ctx: TacticsContext, scorer) -> TargetInfo | None:
        """Utility: score all monsters with a callable and return the best."""
        best: TargetInfo | None = None
        for m in ctx.monsters:
            info = self._monster_to_info(m)
            info.score = scorer(info, ctx)
            if best is None or info.score > best.score:
                best = info
        return best

    def _monster_to_info(self, monster: dict[str, Any]) -> TargetInfo:
        """Convert a raw monster dict to TargetInfo."""
        return TargetInfo(
            actor_id=int(monster.get("actor_id", monster.get("id", 0))),
            name=str(monster.get("name", "unknown")),
            score=0.0,
            hp_pct=float(monster.get("hp_pct", monster.get("hp_ratio", 1.0))),
            distance=int(monster.get("distance", 0)),
            element=str(monster.get("element", "neutral")).lower(),
            size=str(monster.get("size", "medium")).lower(),
            race=str(monster.get("race", "formless")).lower(),
            is_boss=bool(monster.get("is_boss", False)),
            is_casting=bool(monster.get("is_casting", False)),
            is_aggressive=bool(monster.get("is_aggressive", True)),
            metadata=monster,
        )

    def _make_action(self, command: str, reason: str, confidence: float = 0.8,
                     **metadata) -> HeuristicAction:
        """Create a HeuristicAction with the tactics domain."""
        return HeuristicAction(
            kind="command",
            command=command,
            confidence=confidence,
            domain="combat_tactics",
            reason=reason,
            metadata={
                "tactics": self.name,
                "role_type": self.role_type,
                **metadata,
            },
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.name} pri={self.priority}>"
