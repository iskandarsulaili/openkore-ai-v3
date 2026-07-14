"""
Strategy Optimizer — runs A/B tests on different builds, rotations, and maps,
tracks efficiency metrics over time, and automatically switches to the
best-performing strategy.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class StrategyTest:
    """An A/B test for a strategy."""
    name: str
    strategy_a: str = ""
    strategy_b: str = ""
    current_strategy: str = "A"
    metric: str = "zeny_per_hour"
    a_results: list[float] = field(default_factory=list)
    b_results: list[float] = field(default_factory=list)
    start_time: float = 0.0
    switch_interval_min: int = 30
    is_active: bool = False
    winner: str = ""


@dataclass
class StrategyResult:
    """The result of a strategy optimization."""
    strategy_name: str
    avg_zeny_per_hour: float = 0.0
    avg_xp_per_hour: float = 0.0
    avg_drops_per_hour: float = 0.0
    death_rate_per_hour: float = 0.0
    sample_hours: float = 0.0
    confidence: float = 0.0
    is_best: bool = False


class StrategyOptimizer:
    """Optimizes strategies through A/B testing."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._tests: list[StrategyTest] = []
        self._results: dict[str, list[StrategyResult]] = defaultdict(list)
        self._current_strategies: dict[str, str] = {}  # category -> strategy_name
        self._enqueue_fn: Callable | None = None
        self._load_default_tests()

    def _load_default_tests(self) -> None:
        """Load default A/B tests."""
        self._tests = [
            StrategyTest(
                name="farming_map_test",
                strategy_a="payon_cave",
                strategy_b="geffen_dungeon",
                metric="zeny_per_hour",
                switch_interval_min=30,
            ),
            StrategyTest(
                name="skill_rotation_test",
                strategy_a="mage_fire_combo",
                strategy_b="mage_aoe_chain",
                metric="xp_per_hour",
                switch_interval_min=20,
            ),
            StrategyTest(
                name="buff_strategy_test",
                strategy_a="all_buffs",
                strategy_b="combat_buffs_only",
                metric="zeny_per_hour",
                switch_interval_min=15,
            ),
            StrategyTest(
                name="restock_threshold_test",
                strategy_a="restock_at_20",
                strategy_b="restock_at_50",
                metric="uptime_pct",
                switch_interval_min=60,
            ),
        ]

    # ── Public API ──

    def record_result(self, category: str, strategy: str, zeny_per_hour: float = 0,
                      xp_per_hour: float = 0, drops_per_hour: float = 0,
                      deaths: int = 0, sample_hours: float = 0.5) -> None:
        """Record a strategy result."""
        with self._lock:
            result = StrategyResult(
                strategy_name=strategy,
                avg_zeny_per_hour=zeny_per_hour,
                avg_xp_per_hour=xp_per_hour,
                avg_drops_per_hour=drops_per_hour,
                death_rate_per_hour=deaths / max(sample_hours, 0.1),
                sample_hours=sample_hours,
            )
            self._results[category].append(result)
            if len(self._results[category]) > 100:
                self._results[category] = self._results[category][-100:]

    def get_best_strategy(self, category: str, metric: str = "zeny_per_hour") -> str | None:
        """Get the best strategy for a category based on results."""
        with self._lock:
            results = self._results.get(category, [])
            if not results:
                return None

            # Group by strategy name and average
            by_strategy: dict[str, list[StrategyResult]] = defaultdict(list)
            for r in results:
                by_strategy[r.strategy_name].append(r)

            best_strategy = None
            best_value = 0.0

            for name, group in by_strategy.items():
                if len(group) < 2:
                    continue
                avg = sum(getattr(r, metric, 0) for r in group) / len(group)
                if avg > best_value:
                    best_value = avg
                    best_strategy = name

            return best_strategy

    def should_switch_strategy(self, category: str, current_strategy: str) -> bool:
        """Check if we should switch to a better strategy."""
        with self._lock:
            best = self.get_best_strategy(category)
            if best and best != current_strategy:
                return True
            return False

    def get_recommended_strategy(self, category: str) -> str | None:
        """Get the recommended strategy for a category."""
        with self._lock:
            best = self.get_best_strategy(category)
            if best:
                return best
            return self._current_strategies.get(category)

    def set_strategy(self, category: str, strategy: str) -> None:
        with self._lock:
            self._current_strategies[category] = strategy

    def get_optimizer_summary(self) -> str:
        with self._lock:
            lines = [f"── Strategy Optimizer ──"]
            lines.append(f"Active tests: {len(self._tests)}")
            lines.append(f"Categories tracked: {len(self._results)}")
            for category, results in self._results.items():
                best = self.get_best_strategy(category)
                current = self._current_strategies.get(category, "none")
                lines.append(f"  {category}: current={current}, best={best or 'unknown'}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._tests.clear()
            self._results.clear()
            self._current_strategies.clear()
            self._load_default_tests()


# ── Global Singleton ──

_strategy_opt: StrategyOptimizer | None = None
_strategy_opt_lock = RLock()


def get_strategy_optimizer() -> StrategyOptimizer:
    global _strategy_opt
    with _strategy_opt_lock:
        if _strategy_opt is None:
            _strategy_opt = StrategyOptimizer()
        return _strategy_opt
