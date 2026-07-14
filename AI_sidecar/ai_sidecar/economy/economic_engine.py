"""Economic Engine — tracks real-time profit per map and auto-switches when efficiency drops.

A top farmer doesn't just grind. They know exactly which map yields the best
zeny per hour, experience per hour, and overall efficiency. When a map's
efficiency drops below threshold, the engine recommends a switch to a better
map — using real collected data when available, and estimated data from
knowledge.json when not.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "knowledge", "knowledge.json"
)

# Default estimation parameters used when no real data exists or knowledge.json is unavailable
_DEFAULT_MONSTERS_PER_MAP = 15
_DEFAULT_KILLS_PER_HOUR = 600  # ~10 kills/min
_DEFAULT_DROP_RATE = 0.5  # 50% chance of any drop per kill
_DEFAULT_EXP_PER_KILL_FACTOR = 1.0  # multiplier for XP estimation
_DEFAULT_ZENY_PER_DROP_FACTOR = 0.3  # fraction of item buy price as drop value

# Efficiency score weights
_WEIGHT_PROFIT = 0.40
_WEIGHT_EXP = 0.25
_WEIGHT_DEATH_PENALTY = 0.20
_WEIGHT_COST = 0.15

# Switch thresholds
_SWITCH_EFFICIENCY_THRESHOLD = 30.0  # below this, consider switching
_SWITCH_IMPROVEMENT_MIN = 10.0  # minimum efficiency improvement to recommend switch
_SWITCH_COOLDOWN_SECONDS = 300.0  # don't recommend same switch more than once per 5 min

# Trend detection
_TREND_WINDOW = 5  # number of snapshots to consider for trend


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class EconomicSnapshot:
    """A single data point of economic activity on a map."""

    map_name: str
    timestamp: float
    zeny_earned: int
    zeny_spent: int  # potions, warp costs, etc.
    items_dropped: list[tuple[str, int, int]]  # (item_name, value, count)
    exp_earned: int
    time_elapsed_seconds: float
    monsters_killed: int
    deaths: int

    @property
    def profit(self) -> int:
        return self.zeny_earned - self.zeny_spent

    @property
    def hours_elapsed(self) -> float:
        return self.time_elapsed_seconds / 3600.0 if self.time_elapsed_seconds > 0 else 0.001


@dataclass
class MapEconomics:
    """Aggregated economic data for a single map."""

    map_name: str
    zeny_per_hour: float
    exp_per_hour: float
    profit_per_hour: float  # zeny_earned - zeny_spent
    cost_per_hour: float
    deaths_per_hour: float
    efficiency_score: float  # 0-100
    sample_count: int
    last_updated: float
    trend: str  # improving / declining / stable


# ── Knowledge cache ───────────────────────────────────────────────────────


class _KnowledgeCache:
    """Lazy-loaded cache for knowledge.json monster and item data."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._loaded = False
        self._monsters: list[dict[str, Any]] = []
        self._item_prices: dict[str, int] = {}
        self._monsters_by_name: dict[str, dict[str, Any]] = {}
        self._monsters_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            path = os.path.abspath(_KNOWLEDGE_PATH)
            if not os.path.exists(path):
                logger.warning("knowledge.json not found at %s — estimates will be rough", path)
                self._loaded = True
                return

            try:
                with open(path) as f:
                    data = json.load(f)

                self._monsters = data.get("monsters", [])

                # Build item price lookup from all item categories
                items_section = data.get("items", {})
                for category in ("weapons", "armors", "cards", "usable", "etc", "all"):
                    for item in items_section.get(category, []):
                        aegis = item.get("AegisName", "")
                        buy = item.get("Buy", 0)
                        if aegis and buy:
                            self._item_prices[aegis] = buy

                # Build monster lookups
                for mob in self._monsters:
                    name = mob.get("Name", "")
                    level = mob.get("Level", 0)
                    if name:
                        self._monsters_by_name[name] = mob
                    if level:
                        self._monsters_by_level[level].append(mob)

                logger.info(
                    "Loaded %d monsters, %d item prices from knowledge.json",
                    len(self._monsters),
                    len(self._item_prices),
                )
            except Exception as exc:
                logger.error("Failed to load knowledge.json: %s", exc)

            self._loaded = True

    def get_monsters_for_level_range(self, min_level: int, max_level: int) -> list[dict[str, Any]]:
        """Get monsters whose level falls within [min_level, max_level]."""
        self._ensure_loaded()
        result: list[dict[str, Any]] = []
        for level, mobs in self._monsters_by_level.items():
            if min_level <= level <= max_level:
                result.extend(mobs)
        return result

    def estimate_map_zeny_per_hour(self, map_name: str, player_level: int) -> int:
        """Estimate zeny/hour for a map based on monster data."""
        self._ensure_loaded()
        if not self._monsters:
            return _DEFAULT_KILLS_PER_HOUR * 10  # fallback

        # Find monsters near the player's level (±5)
        nearby = self.get_monsters_for_level_range(
            max(1, player_level - 5), player_level + 5
        )
        if not nearby:
            nearby = self.get_monsters_for_level_range(
                max(1, player_level - 10), player_level + 10
            )
        if not nearby:
            return _DEFAULT_KILLS_PER_HOUR * 10

        # Average drop value across nearby monsters
        total_drop_value = 0.0
        total_drops = 0
        for mob in nearby[:20]:  # sample up to 20
            drops = mob.get("Drops", [])
            for drop in drops:
                item_name = drop.get("Item", "")
                rate = drop.get("Rate", 0)
                price = self._item_prices.get(item_name, 0)
                if price > 0 and rate > 0:
                    # Expected value per kill = price * (rate / 10000)  (rate is in 0.01%)
                    total_drop_value += price * (rate / 10000.0)
                    total_drops += 1

        if total_drops == 0:
            return _DEFAULT_KILLS_PER_HOUR * 10

        avg_drop_value_per_kill = total_drop_value / max(len(nearby[:20]), 1)
        estimated_zeny_per_hour = int(
            _DEFAULT_KILLS_PER_HOUR * avg_drop_value_per_kill * _DEFAULT_ZENY_PER_DROP_FACTOR
        )
        return max(estimated_zeny_per_hour, 100)

    def estimate_map_exp_per_hour(self, map_name: str, player_level: int) -> int:
        """Estimate XP/hour for a map based on monster data."""
        self._ensure_loaded()
        if not self._monsters:
            return _DEFAULT_KILLS_PER_HOUR * 20  # fallback

        nearby = self.get_monsters_for_level_range(
            max(1, player_level - 5), player_level + 5
        )
        if not nearby:
            nearby = self.get_monsters_for_level_range(
                max(1, player_level - 10), player_level + 10
            )
        if not nearby:
            return _DEFAULT_KILLS_PER_HOUR * 20

        total_exp = 0
        count = 0
        for mob in nearby[:20]:
            base_exp = mob.get("BaseExp", 0) or 0
            job_exp = mob.get("JobExp", 0) or 0
            total_exp += base_exp + job_exp
            count += 1

        if count == 0:
            return _DEFAULT_KILLS_PER_HOUR * 20

        avg_exp_per_kill = total_exp / count
        estimated_exp_per_hour = int(
            _DEFAULT_KILLS_PER_HOUR * avg_exp_per_kill * _DEFAULT_EXP_PER_KILL_FACTOR
        )
        return max(estimated_exp_per_hour, 100)


# ── Global knowledge cache ─────────────────────────────────────────────────

_knowledge: _KnowledgeCache | None = None
_knowledge_lock = RLock()


def _get_knowledge() -> _KnowledgeCache:
    global _knowledge
    with _knowledge_lock:
        if _knowledge is None:
            _knowledge = _KnowledgeCache()
        return _knowledge


# ── Economic Engine ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class EconomicEngine:
    """Tracks real-time profit per map and recommends optimal farming locations.

    Thread-safe. Uses RLock for all internal state access.
    """

    _lock: RLock = field(default_factory=RLock)
    _snapshots: dict[str, list[EconomicSnapshot]] = field(default_factory=dict)
    _economics: dict[str, MapEconomics] = field(default_factory=dict)
    _last_switch_recommendation: dict[str, float] = field(default_factory=dict)  # map -> timestamp

    # ── Snapshot recording ──────────────────────────────────────────────

    def record_snapshot(self, snapshot: EconomicSnapshot) -> None:
        """Record a new economic snapshot and recompute map economics."""
        with self._lock:
            map_name = snapshot.map_name
            if map_name not in self._snapshots:
                self._snapshots[map_name] = []
            self._snapshots[map_name].append(snapshot)
            self._recompute_map(map_name)

    def _recompute_map(self, map_name: str) -> None:
        """Recompute aggregated economics for a map from its snapshots."""
        snapshots = self._snapshots.get(map_name, [])
        if not snapshots:
            return

        # Use the last N snapshots for trend calculation
        recent = snapshots[-_TREND_WINDOW:] if len(snapshots) >= _TREND_WINDOW else snapshots

        total_zeny_earned = sum(s.zeny_earned for s in recent)
        total_zeny_spent = sum(s.zeny_spent for s in recent)
        total_exp = sum(s.exp_earned for s in recent)
        total_time = sum(s.time_elapsed_seconds for s in recent)
        total_monsters = sum(s.monsters_killed for s in recent)
        total_deaths = sum(s.deaths for s in recent)

        hours = total_time / 3600.0 if total_time > 0 else 0.001

        zeny_per_hour = total_zeny_earned / hours
        exp_per_hour = total_exp / hours
        cost_per_hour = total_zeny_spent / hours
        profit_per_hour = (total_zeny_earned - total_zeny_spent) / hours
        deaths_per_hour = total_deaths / hours

        # Efficiency score (0-100)
        efficiency_score = self._compute_efficiency_score(
            profit_per_hour, exp_per_hour, cost_per_hour, deaths_per_hour
        )

        # Trend detection: compare first half vs second half of recent snapshots
        trend = self._detect_trend(recent)

        self._economics[map_name] = MapEconomics(
            map_name=map_name,
            zeny_per_hour=zeny_per_hour,
            exp_per_hour=exp_per_hour,
            profit_per_hour=profit_per_hour,
            cost_per_hour=cost_per_hour,
            deaths_per_hour=deaths_per_hour,
            efficiency_score=efficiency_score,
            sample_count=len(snapshots),
            last_updated=time.time(),
            trend=trend,
        )

    def _compute_efficiency_score(
        self,
        profit_per_hour: float,
        exp_per_hour: float,
        cost_per_hour: float,
        deaths_per_hour: float,
    ) -> float:
        """Compute a 0-100 efficiency score from economic metrics.

        Higher profit, higher XP, lower costs, and fewer deaths yield
        a better score.
        """
        # Normalize profit: assume 50k zeny/hr is "perfect" (score 100)
        profit_score = min(100.0, (profit_per_hour / 50000.0) * 100.0) if profit_per_hour > 0 else 0.0

        # Normalize XP: assume 500k XP/hr is "perfect"
        exp_score = min(100.0, (exp_per_hour / 500000.0) * 100.0) if exp_per_hour > 0 else 0.0

        # Death penalty: 0 deaths = 100, each death/hr reduces score
        death_score = max(0.0, 100.0 - (deaths_per_hour * 25.0))

        # Cost efficiency: lower costs relative to profit is better
        if profit_per_hour > 0 and cost_per_hour > 0:
            cost_ratio = cost_per_hour / (profit_per_hour + cost_per_hour)
            cost_score = max(0.0, 100.0 - (cost_ratio * 100.0))
        else:
            cost_score = 50.0  # neutral when no data

        score = (
            _WEIGHT_PROFIT * profit_score
            + _WEIGHT_EXP * exp_score
            + _WEIGHT_DEATH_PENALTY * death_score
            + _WEIGHT_COST * cost_score
        )
        return round(max(0.0, min(100.0, score)), 1)

    def _detect_trend(self, snapshots: list[EconomicSnapshot]) -> str:
        """Detect whether efficiency is improving, declining, or stable."""
        if len(snapshots) < 3:
            return "stable"

        mid = len(snapshots) // 2
        first_half = snapshots[:mid]
        second_half = snapshots[mid:]

        first_avg = sum(
            (s.zeny_earned - s.zeny_spent) / max(s.time_elapsed_seconds, 1)
            for s in first_half
        ) / len(first_half)

        second_avg = sum(
            (s.zeny_earned - s.zeny_spent) / max(s.time_elapsed_seconds, 1)
            for s in second_half
        ) / len(second_half)

        change_pct = ((second_avg - first_avg) / max(abs(first_avg), 1)) * 100.0

        if change_pct > 10.0:
            return "improving"
        elif change_pct < -10.0:
            return "declining"
        else:
            return "stable"

    # ── Query methods ───────────────────────────────────────────────────

    def get_map_economics(self, map_name: str) -> MapEconomics | None:
        """Get aggregated economics for a specific map."""
        with self._lock:
            return self._economics.get(map_name)

    def get_best_map(self, current_level: int, current_zeny: int) -> str | None:
        """Get the name of the best map to farm right now.

        Uses real data when available, falls back to estimates.
        """
        with self._lock:
            all_econ = self._get_all_economics_locked()
            if not all_econ:
                return None

            # Filter out maps with very low sample counts (unreliable)
            reliable = [e for e in all_econ if e.sample_count >= 3]
            if not reliable:
                reliable = all_econ

            # Score each map: efficiency + level suitability
            best_map: str | None = None
            best_score = -1.0

            for econ in reliable:
                # Penalty for maps far from player level
                level_penalty = 0.0
                if current_level > 0:
                    # Estimate map level from efficiency data
                    est_map_level = self._estimate_map_level(econ)
                    level_diff = abs(est_map_level - current_level)
                    if level_diff > 15:
                        level_penalty = 30.0
                    elif level_diff > 10:
                        level_penalty = 15.0
                    elif level_diff > 5:
                        level_penalty = 5.0

                # Bonus for maps with good profit relative to zeny
                zeny_bonus = 0.0
                if current_zeny > 0 and econ.cost_per_hour > 0:
                    if econ.cost_per_hour < current_zeny * 0.1:
                        zeny_bonus = 10.0

                score = econ.efficiency_score - level_penalty + zeny_bonus
                if score > best_score:
                    best_score = score
                    best_map = econ.map_name

            return best_map

    def get_all_economics(self) -> list[MapEconomics]:
        """Get economics for all tracked maps."""
        with self._lock:
            return self._get_all_economics_locked()

    def _get_all_economics_locked(self) -> list[MapEconomics]:
        return list(self._economics.values())

    def get_profit_rankings(self) -> list[MapEconomics]:
        """Get all maps sorted by profit_per_hour descending."""
        with self._lock:
            return sorted(
                self._economics.values(),
                key=lambda e: e.profit_per_hour,
                reverse=True,
            )

    def get_efficiency_score(self, map_name: str) -> float:
        """Get the efficiency score for a map (0-100)."""
        with self._lock:
            econ = self._economics.get(map_name)
            if econ is not None:
                return econ.efficiency_score
            # Estimate from knowledge data
            knowledge = _get_knowledge()
            est_zeny = knowledge.estimate_map_zeny_per_hour(map_name, 50)
            est_exp = knowledge.estimate_map_exp_per_hour(map_name, 50)
            score = self._compute_efficiency_score(
                profit_per_hour=float(est_zeny),
                exp_per_hour=float(est_exp),
                cost_per_hour=0.0,
                deaths_per_hour=0.0,
            )
            return score

    def should_switch_map(self, current_map: str, current_efficiency: float) -> bool:
        """Determine if we should switch away from the current map.

        Returns True when efficiency is below threshold and a better
        alternative exists.
        """
        if current_efficiency >= _SWITCH_EFFICIENCY_THRESHOLD:
            return False

        with self._lock:
            # Check if there's a better map
            for econ in self._economics.values():
                if econ.map_name == current_map:
                    continue
                if econ.efficiency_score > current_efficiency + _SWITCH_IMPROVEMENT_MIN:
                    return True

            # If no real data, check estimates
            knowledge = _get_knowledge()
            # Try a few level ranges to see if any estimate beats current
            for level in range(10, 100, 10):
                est_zeny = knowledge.estimate_map_zeny_per_hour(f"map_lv{level}", level)
                est_exp = knowledge.estimate_map_exp_per_hour(f"map_lv{level}", level)
                est_score = self._compute_efficiency_score(
                    profit_per_hour=float(est_zeny),
                    exp_per_hour=float(est_exp),
                    cost_per_hour=0.0,
                    deaths_per_hour=0.0,
                )
                if est_score > current_efficiency + _SWITCH_IMPROVEMENT_MIN:
                    return True

            return False

    def get_recommended_switch(self, map_name: str) -> str | None:
        """Get the recommended map to switch to from the current map.

        Returns None if no better map is found or if the recommendation
        was recently made (cooldown).
        """
        now = time.time()

        with self._lock:
            current = self._economics.get(map_name)
            current_score = current.efficiency_score if current else 0.0

            best_candidate: str | None = None
            best_score = -1.0

            for econ in self._economics.values():
                if econ.map_name == map_name:
                    continue
                if econ.efficiency_score <= current_score + _SWITCH_IMPROVEMENT_MIN:
                    continue

                # Check cooldown
                last_rec = self._last_switch_recommendation.get(econ.map_name, 0.0)
                if now - last_rec < _SWITCH_COOLDOWN_SECONDS:
                    continue

                if econ.efficiency_score > best_score:
                    best_score = econ.efficiency_score
                    best_candidate = econ.map_name

            if best_candidate:
                self._last_switch_recommendation[best_candidate] = now
                return best_candidate

            return None

    def get_estimated_zeny_per_hour(self, map_name: str, player_level: int) -> int:
        """Estimate zeny per hour for a map, using real data or knowledge.json."""
        with self._lock:
            econ = self._economics.get(map_name)
            if econ is not None and econ.sample_count >= 3:
                return int(econ.zeny_per_hour)

        knowledge = _get_knowledge()
        return knowledge.estimate_map_zeny_per_hour(map_name, player_level)

    def get_estimated_exp_per_hour(self, map_name: str, player_level: int) -> int:
        """Estimate experience per hour for a map."""
        with self._lock:
            econ = self._economics.get(map_name)
            if econ is not None and econ.sample_count >= 3:
                return int(econ.exp_per_hour)

        knowledge = _get_knowledge()
        return knowledge.estimate_map_exp_per_hour(map_name, player_level)

    def get_roi(self, map_name: str, potion_cost: int, warp_cost: int) -> float:
        """Return on investment for farming a map.

        ROI = (expected profit - costs) / costs
        Returns a ratio (1.0 = break-even, 2.0 = 2x return).
        """
        with self._lock:
            econ = self._economics.get(map_name)
            if econ is not None and econ.sample_count >= 3:
                profit = econ.profit_per_hour
            else:
                knowledge = _get_knowledge()
                profit = float(knowledge.estimate_map_zeny_per_hour(map_name, 50))

        total_cost = potion_cost + warp_cost
        if total_cost <= 0:
            return float("inf") if profit > 0 else 0.0

        return profit / total_cost

    def get_best_maps_for_level(
        self, level: int, min_profit: int = 0
    ) -> list[MapEconomics]:
        """Get the best maps suitable for a given player level.

        Filters by level suitability and minimum profit threshold.
        """
        with self._lock:
            candidates: list[MapEconomics] = []
            for econ in self._economics.values():
                if econ.profit_per_hour < min_profit:
                    continue
                est_level = self._estimate_map_level(econ)
                if abs(est_level - level) > 15:
                    continue
                candidates.append(econ)

            return sorted(candidates, key=lambda e: e.efficiency_score, reverse=True)

    def _estimate_map_level(self, econ: MapEconomics) -> int:
        """Estimate the recommended level for a map based on its economics."""
        # Rough heuristic: higher XP/hr maps tend to be higher level
        if econ.exp_per_hour > 500000:
            return 80
        elif econ.exp_per_hour > 200000:
            return 60
        elif econ.exp_per_hour > 50000:
            return 40
        elif econ.exp_per_hour > 10000:
            return 20
        else:
            return 10

    def reset_map_data(self, map_name: str) -> None:
        """Reset all recorded data for a specific map."""
        with self._lock:
            self._snapshots.pop(map_name, None)
            self._economics.pop(map_name, None)
            self._last_switch_recommendation.pop(map_name, None)
            logger.info("Reset economic data for map: %s", map_name)

    def get_summary(self) -> str:
        """Get a formatted summary of all map economics for LLM prompts."""
        with self._lock:
            lines = ["── Economic Engine Summary ──"]

            if not self._economics:
                lines.append("  No economic data recorded yet.")
                return "\n".join(lines)

            # Sort by efficiency score descending
            ranked = sorted(
                self._economics.values(),
                key=lambda e: e.efficiency_score,
                reverse=True,
            )

            lines.append(f"  Maps tracked: {len(ranked)}")
            lines.append("")

            for econ in ranked:
                lines.append(
                    f"  {econ.map_name}: "
                    f"eff={econ.efficiency_score:.1f} "
                    f"profit={econ.profit_per_hour:,.0f}z/hr "
                    f"exp={econ.exp_per_hour:,.0f}/hr "
                    f"cost={econ.cost_per_hour:,.0f}z/hr "
                    f"deaths={econ.deaths_per_hour:.1f}/hr "
                    f"trend={econ.trend} "
                    f"samples={econ.sample_count}"
                )

            # Top recommendation
            best = ranked[0]
            lines.append("")
            lines.append(
                f"  ▶ Best map: {best.map_name} "
                f"(eff={best.efficiency_score:.1f}, "
                f"profit={best.profit_per_hour:,.0f}z/hr)"
            )

            return "\n".join(lines)


# ── Global singleton ───────────────────────────────────────────────────────

_economic: EconomicEngine | None = None
_economic_lock = RLock()


def get_economic_engine() -> EconomicEngine:
    """Get the global EconomicEngine singleton."""
    global _economic
    with _economic_lock:
        if _economic is None:
            _economic = EconomicEngine()
        return _economic
