"""Strategy adaptation — weighted scoring for map recommendations.

Uses learned experience data to recommend optimal maps based on:
  - Death rate (>3/hour → recommend safer map)
  - Exp/hour (<1000 → recommend better map)
  - Zeny/hour (>5000 → recommend staying)
  - Weighted scoring: exp_weight=0.4, zeny_weight=0.3, safety_weight=0.3
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_sidecar.domains.learning.experience import ExperienceTracker

logger = logging.getLogger(__name__)


# ── Global singleton ─────────────────────────────────────────────────

_adapter_instance: StrategyAdapter | None = None


def get_strategy_adapter(
    experience_tracker: ExperienceTracker | None = None,
) -> StrategyAdapter:
    """Get or create the singleton StrategyAdapter.

    Args:
        experience_tracker: Required on first call to wire the tracker.
    """
    global _adapter_instance
    if _adapter_instance is None:
        if experience_tracker is None:
            # Lazily import to break circular dependency at module level
            from ai_sidecar.domains.learning.experience import (
                get_experience_tracker,
            )
            experience_tracker = get_experience_tracker()
        _adapter_instance = StrategyAdapter(experience_tracker)
    return _adapter_instance


# ── Constants ───────────────────────────────────────────────────────

# Weights for the scoring formula
EXP_WEIGHT = 0.4
ZENY_WEIGHT = 0.3
SAFETY_WEIGHT = 0.3

# Thresholds
DEATH_RATE_DANGER = 3.0  # deaths/hour → too dangerous
MIN_EXP_RATE = 1000.0    # exp/hour → too slow
STAY_ZENY_RATE = 5000.0  # zeny/hour → good enough to stay

# Score ranges (normalized 0-100)
MAX_EXP_RATE = 50000.0   # cap for normalization
MAX_ZENY_RATE = 50000.0  # cap for normalization
MAX_DEATH_RATE = 10.0    # clamp for normalization

# Known safe maps (fallback if no data)
_DEFAULT_SAFE_MAPS = [
    "prontera", "izlude", "alberta", "morocc", "geffen",
    "payon", "aldebaran", "umbala", "comodo",
]


class StrategyAdapter:
    """Adapts bot strategy based on learned experience data.

    Provides map recommendations using a weighted scoring formula
    that balances experience gain, loot income, and safety.
    """

    def __init__(self, experience_tracker: ExperienceTracker) -> None:
        self._tracker = experience_tracker

    # ── Core scoring ─────────────────────────────────────────────────

    def score_map(
        self,
        map_name: str,
        bot_id: str = "default",
    ) -> dict[str, float]:
        """Score a map on three dimensions + composite.

        Returns:
            dict with keys: exp_score, zeny_score, safety_score, composite_score
            All scores are 0-100 (higher = better).
        """
        stats = self._tracker.get_map_stats(map_name, bot_id)
        meta = self._tracker.get_map_metadata(map_name, bot_id)

        exp_rate = stats["exp_rate"]
        death_rate = stats["death_rate"]
        loot_rate = stats["loot_rate"]

        # Exp score: normalized, capped at 100
        exp_score = min(100.0, (exp_rate / MAX_EXP_RATE) * 100.0) if exp_rate > 0 else 10.0

        # Zeny score: normalized, capped at 100
        zeny_score = min(100.0, (loot_rate / MAX_ZENY_RATE) * 100.0) if loot_rate > 0 else 10.0

        # Safety score: inverse of death rate, capped
        # Also factor in the is_safe metadata flag
        safe_meta = meta.get("is_safe", True)
        if death_rate > 0:
            clamped = min(death_rate, MAX_DEATH_RATE)
            safety_from_deaths = 100.0 * (1.0 - clamped / MAX_DEATH_RATE)
        else:
            safety_from_deaths = 100.0  # no deaths recorded → perfect safety

        # Combine: 70% from deaths, 30% from metadata safety flag
        safety_score = 0.7 * safety_from_deaths + (30.0 if safe_meta else 0.0)

        # Composite weighted score
        composite = (
            EXP_WEIGHT * exp_score
            + ZENY_WEIGHT * zeny_score
            + SAFETY_WEIGHT * safety_score
        )

        return {
            "exp_score": round(exp_score, 1),
            "zeny_score": round(zeny_score, 1),
            "safety_score": round(safety_score, 1),
            "composite_score": round(composite, 1),
        }

    def recommend_map(
        self,
        current_map: str,
        bot_id: str = "default",
    ) -> dict[str, Any]:
        """Recommend whether to stay or move, and where.

        Strategy rules:
          1. If current map death rate > 3/hour → recommend safer map
          2. If current map exp/hour < 1000 → recommend better map
          3. If current map zeny/hour > 5000 → recommend staying
          4. Otherwise → score all maps and pick the best composite

        Returns:
            dict with keys:
              - action: 'stay' | 'move'
              - command: move command if action is 'move'
              - target_map: recommended map name
              - confidence: 0.0-1.0
              - reason: human-readable explanation
              - scores: dict of per-map scores
        """
        current_stats = self._tracker.get_map_stats(current_map, bot_id)
        death_rate = current_stats["death_rate"]
        exp_rate = current_stats["exp_rate"]
        loot_rate = current_stats["loot_rate"]

        # Rule 1: Too many deaths
        if death_rate > DEATH_RATE_DANGER:
            safer = self._find_safer_map(current_map, bot_id)
            if safer:
                return {
                    "action": "move",
                    "command": f"move {safer}",
                    "target_map": safer,
                    "confidence": 0.85,
                    "reason": (
                        f"Death rate {death_rate:.1f}/h on {current_map} "
                        f"exceeds {DEATH_RATE_DANGER}/h threshold. "
                        f"Moving to safer map {safer}."
                    ),
                    "scores": {
                        current_map: self.score_map(current_map, bot_id),
                    },
                }
            else:
                return {
                    "action": "stay",
                    "command": "",
                    "target_map": current_map,
                    "confidence": 0.3,
                    "reason": (
                        f"Death rate {death_rate:.1f}/h is high but "
                        f"no safer map found."
                    ),
                    "scores": {},
                }

        # Rule 2: Not enough exp
        if exp_rate < MIN_EXP_RATE:
            better = self._find_better_exp_map(current_map, bot_id)
            if better:
                return {
                    "action": "move",
                    "command": f"move {better}",
                    "target_map": better,
                    "confidence": 0.75,
                    "reason": (
                        f"Exp rate {exp_rate:.0f}/h on {current_map} "
                        f"below {MIN_EXP_RATE:.0f}/h. "
                        f"Moving to {better}."
                    ),
                    "scores": {
                        current_map: self.score_map(current_map, bot_id),
                    },
                }
            else:
                return {
                    "action": "stay",
                    "command": "",
                    "target_map": current_map,
                    "confidence": 0.5,
                    "reason": (
                        f"Exp rate {exp_rate:.0f}/h is low but "
                        f"no better map found."
                    ),
                    "scores": {},
                }

        # Rule 3: Good loot
        if loot_rate > STAY_ZENY_RATE:
            return {
                "action": "stay",
                "command": "",
                "target_map": current_map,
                "confidence": 0.9,
                "reason": (
                    f"Zeny rate {loot_rate:.0f}/h on {current_map} "
                    f"exceeds {STAY_ZENY_RATE:.0f}/h. "
                    f"Recommend staying for farming."
                ),
                "scores": {
                    current_map: self.score_map(current_map, bot_id),
                },
            }

        # Rule 4: Compare all known maps
        all_maps = self._tracker.get_all_map_stats(bot_id)
        if all_maps:
            scored: list[tuple[float, str]] = []
            for m in all_maps:
                m_name = m["map_name"]
                composite = self.score_map(m_name, bot_id)["composite_score"]
                scored.append((composite, m_name))

            scored.sort(reverse=True)
            best_map = scored[0][1]
            current_score = self.score_map(current_map, bot_id)["composite_score"]
            best_score = scored[0][0]

            if best_map != current_map and best_score > current_score + 5:
                return {
                    "action": "move",
                    "command": f"move {best_map}",
                    "target_map": best_map,
                    "confidence": 0.6,
                    "reason": (
                        f"Map {best_map} scores {best_score:.1f} vs "
                        f"{current_map} at {current_score:.1f}. "
                        f"Recommended switch."
                    ),
                    "scores": {name: self.score_map(name, bot_id) for _, name in scored[:5]},
                }

        # Default: stay
        return {
            "action": "stay",
            "command": "",
            "target_map": current_map,
            "confidence": 0.7,
            "reason": f"Current map {current_map} is acceptable.",
            "scores": {
                current_map: self.score_map(current_map, bot_id),
            },
        }

    # ── Internal helpers ────────────────────────────────────────────

    def _find_safer_map(
        self, current_map: str, bot_id: str,
    ) -> str | None:
        """Find a map with lower death rate than current."""
        all_maps = self._tracker.get_all_map_stats(bot_id)
        current_death = self._tracker.get_death_rate_per_hour(current_map, bot_id)

        candidates: list[tuple[float, str]] = []
        for m in all_maps:
            m_name = m["map_name"]
            if m_name == current_map:
                continue
            death = m["death_rate"]
            if death < current_death and death <= DEATH_RATE_DANGER:
                score = self.score_map(m_name, bot_id)["composite_score"]
                candidates.append((score, m_name))

        candidates.sort(reverse=True)
        if candidates:
            return candidates[0][1]

        # Fallback: check safe maps list
        for safe in _DEFAULT_SAFE_MAPS:
            safe_death = self._tracker.get_death_rate_per_hour(safe, bot_id)
            if safe_death <= 1.0:
                return safe

        return None

    def _find_better_exp_map(
        self, current_map: str, bot_id: str,
    ) -> str | None:
        """Find a map with higher exp rate."""
        all_maps = self._tracker.get_all_map_stats(bot_id)
        current_exp = self._tracker.get_exp_rate_per_hour(current_map, bot_id)

        candidates: list[tuple[float, str]] = []
        for m in all_maps:
            m_name = m["map_name"]
            if m_name == current_map:
                continue
            exp = m["exp_rate"]
            if exp > current_exp:
                score = self.score_map(m_name, bot_id)["composite_score"]
                candidates.append((score, m_name))

        candidates.sort(reverse=True)
        if candidates:
            return candidates[0][1]
        return None

    def __repr__(self) -> str:
        return f"<StrategyAdapter: weights exp={EXP_WEIGHT} zeny={ZENY_WEIGHT} safety={SAFETY_WEIGHT}>"
