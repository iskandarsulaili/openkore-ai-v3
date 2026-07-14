"""Farming Target Selector — picks the most profitable monsters/maps to farm.

A pro player doesn't just kill whatever is nearby. They know:
- Which monsters drop valuable items
- Which maps have the best zeny/hour
- When to switch targets based on market prices
- Which items are worth picking up vs leaving on the ground

This module uses the ItemValueDB to make intelligent farming decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.economy.item_value_db import get_item_value_db, ItemValueDB

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Minimum expected zeny per kill to consider a target worthwhile
MIN_ZENY_PER_KILL = 10

# Minimum efficiency score (zeny per HP ratio * 1000)
MIN_EFFICIENCY = 0.1

# How often to re-evaluate targets (seconds)
REEVALUATION_INTERVAL = 300  # 5 minutes

# Maximum weight of items to pick up (heavier items left behind)
MAX_PICKUP_WEIGHT = 200

# Minimum market value to pick up an item
MIN_PICKUP_VALUE = 500

# Number of top targets to track
TOP_TARGETS_COUNT = 10


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class FarmingTarget:
    """A recommended farming target."""
    monster_name: str
    monster_level: int
    monster_hp: int
    map_name: str  # Best map to find this monster
    expected_zeny_per_kill: float
    expected_zeny_per_hour: float  # Estimated with kill speed
    efficiency_score: float  # zeny per HP ratio
    best_drop: str
    best_drop_value: float
    drop_count: int
    is_mvp: bool
    competition_risk: str  # low, medium, high
    priority: int  # 1 = highest


@dataclass
class LootFilter:
    """Filter rules for what loot to pick up."""
    min_value: int = MIN_PICKUP_VALUE
    max_weight: int = MAX_PICKUP_WEIGHT
    always_pickup_categories: set[str] = field(default_factory=lambda: {"Card", "Healing"})
    never_pickup_categories: set[str] = field(default_factory=set)
    always_pickup_items: set[str] = field(default_factory=set)
    never_pickup_items: set[str] = field(default_factory=set)


# ── Farming Target Selector ──────────────────────────────────────────────


@dataclass(slots=True)
class FarmingTargetSelector:
    """Selects the most profitable monsters/maps to farm.

    Uses ItemValueDB for item valuations and monster drop analysis.
    Thread-safe.
    """

    _lock: RLock = field(default_factory=RLock)
    _item_db: ItemValueDB = field(default_factory=get_item_value_db)
    _targets: list[FarmingTarget] = field(default_factory=list)
    _current_target: FarmingTarget | None = None
    _last_reevaluation: float = 0.0
    _loot_filter: LootFilter = field(default_factory=LootFilter)
    _stats: dict[str, int | float] = field(default_factory=lambda: {
        "evaluations": 0, "targets_found": 0, "switches": 0,
    })

    # ── Public API ─────────────────────────────────────────────────────

    def get_best_target(self, level: int, zeny: int = 0,
                         current_map: str = "") -> FarmingTarget | None:
        """Get the best farming target for the given level.

        Re-evaluates periodically. Returns cached target if within reevaluation
        interval.
        """
        with self._lock:
            now = time.time()
            if (self._current_target is not None
                    and now - self._last_reevaluation < REEVALUATION_INTERVAL):
                return self._current_target

            self._last_reevaluation = now
            self._stats["evaluations"] += 1  # type: ignore[assignment]

            # Get best farming targets from item DB
            raw_targets = self._item_db.get_best_farming_targets(
                level=level,
                max_weight=self._loot_filter.max_weight,
                top_n=TOP_TARGETS_COUNT,
            )

            if not raw_targets:
                logger.info("farming_selector: no profitable targets for level %d", level)
                return None

            # Convert to FarmingTarget objects
            targets: list[FarmingTarget] = []
            for i, rt in enumerate(raw_targets):
                # Estimate zeny per hour (assume ~600 kills/hr for low-HP, ~200 for high-HP)
                hp = rt.get("hp", 1000)
                if hp < 500:
                    kills_per_hour = 800
                elif hp < 2000:
                    kills_per_hour = 600
                elif hp < 10000:
                    kills_per_hour = 300
                else:
                    kills_per_hour = 100

                zeny_per_hour = rt["expected_zeny_per_kill"] * kills_per_hour

                # Competition risk: higher value targets attract more bots
                competition = "low"
                if rt["expected_zeny_per_kill"] > 100:
                    competition = "high"
                elif rt["expected_zeny_per_kill"] > 20:
                    competition = "medium"

                target = FarmingTarget(
                    monster_name=rt["monster"],
                    monster_level=rt["level"],
                    monster_hp=rt["hp"],
                    map_name="",  # Will be filled by map knowledge
                    expected_zeny_per_kill=rt["expected_zeny_per_kill"],
                    expected_zeny_per_hour=round(zeny_per_hour, 0),
                    efficiency_score=rt.get("adjusted_value", 0),
                    best_drop=rt["best_drop"],
                    best_drop_value=rt["best_drop_value"],
                    drop_count=rt["drop_count"],
                    is_mvp=False,
                    competition_risk=competition,
                    priority=i + 1,
                )
                targets.append(target)

            self._targets = targets
            self._stats["targets_found"] = len(targets)  # type: ignore[assignment]

            if targets:
                # Pick best non-MVP target, or best overall if nothing else
                non_mvp = [t for t in targets if not t.is_mvp]
                best = non_mvp[0] if non_mvp else targets[0]

                if (self._current_target is not None
                        and self._current_target.monster_name != best.monster_name):
                    self._stats["switches"] += 1  # type: ignore[assignment]
                    logger.info(
                        "farming_selector: switching from %s to %s "
                        "(%.0f vs %.0f zeny/kill)",
                        self._current_target.monster_name,
                        best.monster_name,
                        self._current_target.expected_zeny_per_kill,
                        best.expected_zeny_per_kill,
                    )

                self._current_target = best
                return best

            return None

    def get_all_targets(self) -> list[FarmingTarget]:
        """Get all current farming targets sorted by priority."""
        with self._lock:
            return list(self._targets)

    def should_pickup(self, item_name: str, item_value: int,
                       item_weight: int, item_category: str) -> bool:
        """Determine if an item is worth picking up.

        Uses the loot filter rules and item valuation.
        """
        # Always pickup rules
        if item_name in self._loot_filter.always_pickup_items:
            return True
        if item_category in self._loot_filter.always_pickup_categories:
            return True

        # Never pickup rules
        if item_name in self._loot_filter.never_pickup_items:
            return False
        if item_category in self._loot_filter.never_pickup_categories:
            return False

        # Value-based rules
        if item_value < self._loot_filter.min_value:
            return False
        if item_weight > self._loot_filter.max_weight:
            # Heavy items need higher value to be worth it
            value_density = item_value / max(item_weight, 1)
            if value_density < 10:  # Less than 10z per weight
                return False

        return True

    def get_loot_filter_advice(self) -> str:
        """Get formatted loot filter advice for LLM prompts."""
        with self._lock:
            lines = ["── Loot Filter ──"]
            lines.append(f"Min value: {self._loot_filter.min_value:,}z")
            lines.append(f"Max weight: {self._loot_filter.max_weight}")
            lines.append(f"Always pickup: {', '.join(self._loot_filter.always_pickup_categories)}")
            if self._current_target:
                lines.append(f"Current target: {self._current_target.monster_name}")
                lines.append(f"  Expected: {self._current_target.expected_zeny_per_kill:.0f}z/kill")
                lines.append(f"  Best drop: {self._current_target.best_drop}")
            return "\n".join(lines)

    def get_farming_summary(self) -> str:
        """Get a formatted summary of farming targets."""
        with self._lock:
            lines = ["── Farming Targets ──"]
            if not self._targets:
                lines.append("  No targets evaluated yet.")
                return "\n".join(lines)

            for t in self._targets[:5]:
                lines.append(
                    f"  #{t.priority}: {t.monster_name} (Lv{t.monster_level}) "
                    f"→ {t.expected_zeny_per_kill:.0f}z/kill "
                    f"({t.expected_zeny_per_hour:,.0f}z/hr) "
                    f"drop={t.best_drop} "
                    f"competition={t.competition_risk}"
                )

            if self._current_target:
                lines.append("")
                lines.append(f"  ▶ Current: {self._current_target.monster_name}")

            return "\n".join(lines)

    def set_loot_filter(self, loot_filter: LootFilter) -> None:
        """Update the loot filter rules."""
        with self._lock:
            self._loot_filter = loot_filter

    def counters(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_farming_selector: FarmingTargetSelector | None = None
_farming_selector_lock = RLock()


def get_farming_selector() -> FarmingTargetSelector:
    """Get the global FarmingTargetSelector singleton."""
    global _farming_selector
    with _farming_selector_lock:
        if _farming_selector is None:
            _farming_selector = FarmingTargetSelector()
        return _farming_selector
