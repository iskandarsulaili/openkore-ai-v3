"""Target scoring system — evaluates monsters for target priority.

Provides multi-factor scoring: HP%, distance, element advantage, danger level,
loot value, and interrupt priority. Used by tactics modules to select targets.

Integrates with ro_mechanics.py for monster stats, element tables, and drop values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_sidecar.domains.combat.tactics.base import TargetInfo
from ai_sidecar.autonomy.ro_mechanics import (
    get_monster_stats, is_mvp, get_mvp_value,
    ELEMENT_TABLE, SIZE_PENALTY,
)

logger = logging.getLogger(__name__)


# ── Scoring Factors ──

@dataclass
class TargetScore:
    """Score breakdown for a single target."""
    total: float = 0.0
    hp_factor: float = 0.0
    distance_factor: float = 0.0
    element_factor: float = 0.0
    danger_factor: float = 0.0
    value_factor: float = 0.0
    interrupt_factor: float = 0.0
    aggro_factor: float = 0.0
    boss_factor: float = 0.0
    reason: str = ""


class TargetScorer:
    """Multi-factor target scoring system.

    Each factor produces a weighted score. The total is used for ranking.
    Factor weights can be tuned per tactics module.
    """

    # Default weight config
    DEFAULT_WEIGHTS: dict[str, float] = {
        "hp_factor": 20.0,          # Low HP = high score (finish kills)
        "distance_factor": 10.0,    # Closer = easier to engage
        "element_factor": 25.0,     # Element advantage
        "danger_factor": -15.0,     # Dangerous monsters penalized (or bonus for tanks)
        "value_factor": 15.0,       # Drop value / card chance
        "interrupt_factor": 30.0,   # Casting monsters
        "boss_factor": 40.0,        # Boss/MVP priority
        "aggro_factor": 5.0,        # Aggressive monsters
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def score(self, target: TargetInfo, attack_element: str = "neutral",
              party_members_nearby: int = 0, is_tank: bool = False) -> TargetScore:
        """Score a single target, returning a full breakdown.

        Args:
            target: TargetInfo for the monster.
            attack_element: Player's attack element.
            party_members_nearby: Number of party members nearby.
            is_tank: Whether the scorer is a tank (inverts danger weighting).

        Returns:
            TargetScore with breakdown.
        """
        score = TargetScore()

        # 1. HP factor: prefer low-HP targets for quick kills
        score.hp_factor = (1.0 - target.hp_pct) * 100.0
        # Bonus for nearly dead targets
        if target.hp_pct < 0.1:
            score.hp_factor += 50.0

        # 2. Distance factor: prefer closer targets
        score.distance_factor = max(0, 30 - target.distance) * 2.0

        # 3. Element factor: advantage/disadvantage
        elem_mult = self._get_elem_multiplier(attack_element, target.element)
        score.element_factor = (elem_mult - 1.0) * 50.0

        # 4. Danger factor: high ATK monsters are dangerous
        monster_stats = get_monster_stats(target.name)
        if monster_stats:
            atk = int(monster_stats.get("attack", monster_stats.get("atk", 50)))
            score.danger_factor = min(50, atk * 0.5)
        else:
            score.danger_factor = 15.0  # Unknown = somewhat dangerous

        # 5. Value factor: card/drop value
        if target.estimated_value > 0:
            score.value_factor = min(50, target.estimated_value * 0.001)
        elif is_mvp(target.name):
            score.value_factor = 40.0
        else:
            score.value_factor = 5.0  # Base value

        # 6. Interrupt factor: casting monsters
        if target.is_casting:
            score.interrupt_factor = 50.0

        # 7. Boss factor
        if target.is_boss:
            score.boss_factor = 60.0

        # 8. Aggro factor
        if target.is_aggressive:
            score.aggro_factor = 10.0

        # Compute weighted total
        weights = self.weights
        score.total = (
            score.hp_factor * (weights["hp_factor"] / 100.0)
            + score.distance_factor * (weights["distance_factor"] / 100.0)
            + score.element_factor * (weights["element_factor"] / 100.0)
            + score.danger_factor * (weights["danger_factor"] / 100.0) * (-1 if is_tank else 1)
            + score.value_factor * (weights["value_factor"] / 100.0)
            + score.interrupt_factor * (weights["interrupt_factor"] / 100.0)
            + score.boss_factor * (weights["boss_factor"] / 100.0)
            + score.aggro_factor * (weights["aggro_factor"] / 100.0)
        )

        score.reason = self._build_reason(score)
        return score

    def score_and_sort(self, targets: list[TargetInfo], attack_element: str = "neutral",
                       party_members_nearby: int = 0, is_tank: bool = False) -> list[tuple[TargetInfo, TargetScore]]:
        """Score all targets and return sorted list (highest total first)."""
        scored = []
        for t in targets:
            s = self.score(t, attack_element, party_members_nearby, is_tank)
            scored.append((t, s))
        scored.sort(key=lambda x: x[1].total, reverse=True)
        return scored

    def best_target(self, targets: list[TargetInfo], attack_element: str = "neutral",
                    party_members_nearby: int = 0, is_tank: bool = False) -> TargetInfo | None:
        """Score all targets and return the best one."""
        scored = self.score_and_sort(targets, attack_element, party_members_nearby, is_tank)
        if scored:
            return scored[0][0]
        return None

    def with_weights(self, overrides: dict[str, float]) -> TargetScorer:
        """Create a new scorer with overridden weights."""
        new_weights = dict(self.weights)
        new_weights.update(overrides)
        return TargetScorer(new_weights)

    @staticmethod
    def _build_reason(score: TargetScore) -> str:
        """Build a human-readable reason string."""
        parts = []
        if score.interrupt_factor > 0:
            parts.append("casting")
        if score.boss_factor > 0:
            parts.append("boss")
        if score.hp_factor > 50:
            parts.append("low_hp")
        if score.value_factor > 30:
            parts.append("valuable")
        return "+".join(parts) if parts else "default"

    @staticmethod
    def _get_elem_multiplier(attack_element: str, defense_element: str,
                             element_level: int = 1) -> float:
        """Get elemental damage multiplier from ro_mechanics."""
        table = ELEMENT_TABLE.get(element_level, ELEMENT_TABLE[1])
        return table.get(attack_element, {}).get(defense_element, 1.0)


# ── Convenience Factory ──

def get_target_scorer(weights: dict[str, float] | None = None) -> TargetScorer:
    """Get a TargetScorer instance with optional custom weights."""
    return TargetScorer(weights)


def get_tank_scorer() -> TargetScorer:
    """Get a TargetScorer optimized for tank target selection.

    Tanks prioritize danger (inverted penalty), proximity, and casting monsters.
    """
    return TargetScorer({
        "hp_factor": 15.0,
        "distance_factor": 15.0,
        "element_factor": 10.0,
        "danger_factor": 35.0,     # Tanks want to engage dangerous monsters
        "value_factor": 5.0,
        "interrupt_factor": 40.0,
        "boss_factor": 50.0,
        "aggro_factor": 10.0,
    })


def get_dps_scorer() -> TargetScorer:
    """Get a TargetScorer optimized for DPS target selection.

    DPS prioritize low HP, element advantage, and value.
    """
    return TargetScorer({
        "hp_factor": 30.0,
        "distance_factor": 10.0,
        "element_factor": 30.0,
        "danger_factor": -10.0,    # DPS avoids dangerous targets
        "value_factor": 20.0,
        "interrupt_factor": 25.0,
        "boss_factor": 30.0,
        "aggro_factor": 5.0,
    })


# ── Monster Enrichment ──

def enrich_target_with_stats(target: TargetInfo) -> TargetInfo:
    """Look up monster stats from ro_mechanics and enrich the target info.

    Fills in estimated_value, danger_level, and other fields from the
    monster database.
    """
    stats = get_monster_stats(target.name)
    if stats:
        # Fill in missing fields from monster DB if not provided
        if not target.element or target.element == "neutral":
            target.element = str(stats.get("element", "neutral")).lower()
        if not target.size or target.size == "medium":
            target.size = str(stats.get("size", "medium")).lower()
        if not target.race or target.race == "formless":
            target.race = str(stats.get("race", "formless")).lower()

        # Danger level based on ATK vs typical player HP
        atk = int(stats.get("attack", stats.get("atk1", 50)))
        target.danger_level = atk / 200.0  # Normalized: 0.25 (safe) to 5.0+ (deadly)

        # Estimated value from MVP check
        if is_mvp(target.name):
            target.estimated_value = float(get_mvp_value(target.name))
            target.is_boss = True

    return target


def enrich_monster_list(monsters: list[dict[str, Any]]) -> list[TargetInfo]:
    """Convert raw monster dicts to enriched TargetInfo list."""
    result = []
    for m in monsters:
        info = TargetInfo(
            actor_id=int(m.get("actor_id", m.get("id", 0))),
            name=str(m.get("name", "unknown")),
            score=0.0,
            hp_pct=float(m.get("hp_pct", m.get("hp_ratio", 1.0))),
            distance=int(m.get("distance", 0)),
            element=str(m.get("element", "neutral")).lower(),
            size=str(m.get("size", "medium")).lower(),
            race=str(m.get("race", "formless")).lower(),
            is_boss=bool(m.get("is_boss", False)),
            is_casting=bool(m.get("is_casting", False)),
            is_aggressive=bool(m.get("is_aggressive", True)),
            metadata=m,
        )
        enrich_target_with_stats(info)
        result.append(info)
    return result
TargetingSystem = TargetScorer
