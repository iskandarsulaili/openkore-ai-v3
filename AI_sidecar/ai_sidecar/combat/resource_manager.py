"""
Resource Sustainability Model — manages potion stock, equipment durability,
weight, farming duration, and efficiency trends.

A pro player doesn't just farm until they die. They manage resources:
"I have 50 potions left. That's enough for 10 minutes. Time to restock."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ResourceState:
    """Current resource state."""
    potion_count: int = 0
    potion_type: str = "White Potion"
    potion_consumption_rate: float = 0.0  # per minute
    estimated_potion_lifetime_min: float = 0.0
    weight: int = 0
    max_weight: int = 10000
    weight_pct: float = 0.0
    equipment_durability_pct: float = 1.0
    zeny: int = 0
    farming_duration_min: float = 0.0
    efficiency_trend: str = "stable"  # improving, declining, stable
    needs_restock: bool = False
    needs_repair: bool = False
    needs_vendor: bool = False
    needs_break: bool = False
    recommended_action: str = "continue_farming"


@dataclass
class RestockPlan:
    """A plan to restock supplies."""
    items_needed: list[tuple[str, int]] = field(default_factory=list)  # (item_name, quantity)
    estimated_cost: int = 0
    estimated_time_min: int = 5
    priority: str = "normal"  # low, normal, high, critical
    reason: str = ""


class ResourceManager:
    """Manages resource sustainability."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._potion_stock: dict[str, int] = {
            "White Potion": 0,
            "Blue Potion": 0,
            "Red Potion": 0,
            "Orange Potion": 0,
            "Yggdrasil Leaf": 0,
        }
        self._potion_consumption_log: list[tuple[float, str]] = []  # (timestamp, potion_type)
        self._farming_start_time: float = time.time()
        self._last_restock_time: float = 0.0
        self._last_repair_time: float = 0.0
        self._last_vendor_time: float = 0.0
        self._break_count: int = 0
        self._total_restock_cost: int = 0
        self._enqueue_fn: Callable | None = None
        self._min_potion_stock: int = 20
        self._max_potion_stock: int = 200
        self._max_farming_duration_min: int = 240  # 4 hours before suggesting break
        self._weight_threshold_pct: float = 0.80
        self._durability_threshold_pct: float = 0.30

    # ── Public API ──

    def update_potion_stock(self, potions: dict[str, int]) -> None:
        """Update current potion stock."""
        with self._lock:
            for name, count in potions.items():
                self._potion_stock[name] = count

    def record_potion_use(self, potion_type: str = "White Potion") -> None:
        """Record a potion being consumed."""
        with self._lock:
            self._potion_consumption_log.append((time.time(), potion_type))
            if len(self._potion_consumption_log) > 1000:
                self._potion_consumption_log = self._potion_consumption_log[-500:]

    def get_resource_state(self) -> ResourceState:
        """Get the current resource state."""
        with self._lock:
            now = time.time()
            state = ResourceState()

            # Potion stock
            total_potions = sum(self._potion_stock.values())
            state.potion_count = total_potions
            state.potion_type = max(self._potion_stock, key=lambda k: self._potion_stock[k]) if self._potion_stock else "White Potion"

            # Consumption rate (potions per minute over last 10 min)
            recent = [t for t, _ in self._potion_consumption_log if now - t < 600]
            if len(recent) >= 2:
                time_span = (recent[-1] - recent[0]) / 60.0
                if time_span > 0:
                    state.potion_consumption_rate = len(recent) / time_span
            if state.potion_consumption_rate > 0 and total_potions > 0:
                state.estimated_potion_lifetime_min = total_potions / state.potion_consumption_rate

            # Weight
            state.weight_pct = state.weight / state.max_weight if state.max_weight > 0 else 0

            # Farming duration
            state.farming_duration_min = (now - self._farming_start_time) / 60.0

            # Needs assessment
            state.needs_restock = total_potions < self._min_potion_stock or state.estimated_potion_lifetime_min < 10
            state.needs_repair = state.equipment_durability_pct < self._durability_threshold_pct
            state.needs_vendor = state.weight_pct > self._weight_threshold_pct
            state.needs_break = state.farming_duration_min > self._max_farming_duration_min

            # Recommended action
            if state.needs_restock and total_potions == 0:
                state.recommended_action = "emergency_restock"
            elif state.needs_repair:
                state.recommended_action = "repair_equipment"
            elif state.needs_restock:
                state.recommended_action = "restock_potions"
            elif state.needs_vendor:
                state.recommended_action = "sell_to_vendor"
            elif state.needs_break:
                state.recommended_action = "take_break"
            else:
                state.recommended_action = "continue_farming"

            return state

    def get_restock_plan(self) -> RestockPlan:
        """Get a plan to restock supplies."""
        with self._lock:
            plan = RestockPlan()
            for name, count in self._potion_stock.items():
                if count < self._min_potion_stock:
                    needed = self._max_potion_stock - count
                    plan.items_needed.append((name, needed))
                    plan.estimated_cost += needed * self._get_item_price(name)

            if plan.items_needed:
                total_potions = sum(self._potion_stock.values())
                if total_potions == 0:
                    plan.priority = "critical"
                    plan.reason = "Out of potions!"
                elif total_potions < 10:
                    plan.priority = "high"
                    plan.reason = f"Only {total_potions} potions remaining"
                else:
                    plan.priority = "normal"
                    plan.reason = f"Restocking {len(plan.items_needed)} items"

            return plan

    def _get_item_price(self, item_name: str) -> int:
        """Get estimated price for an item."""
        prices = {
            "White Potion": 500,
            "Blue Potion": 2000,
            "Red Potion": 50,
            "Orange Potion": 200,
            "Yggdrasil Leaf": 5000,
        }
        return prices.get(item_name, 1000)

    def record_restock(self, cost: int) -> None:
        with self._lock:
            self._last_restock_time = time.time()
            self._total_restock_cost += cost
            # Reset potion stock to max
            for name in self._potion_stock:
                self._potion_stock[name] = self._max_potion_stock

    def record_repair(self) -> None:
        with self._lock:
            self._last_repair_time = time.time()

    def record_vendor(self) -> None:
        with self._lock:
            self._last_vendor_time = time.time()

    def take_break(self) -> None:
        with self._lock:
            self._break_count += 1
            self._farming_start_time = time.time()

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def get_resource_summary(self) -> str:
        with self._lock:
            state = self.get_resource_state()
            lines = [f"── Resource Summary ──"]
            lines.append(f"Potions: {state.potion_count} ({state.potion_type})")
            lines.append(f"Consumption: {state.potion_consumption_rate:.1f}/min")
            lines.append(f"Est. lifetime: {state.estimated_potion_lifetime_min:.0f} min")
            lines.append(f"Weight: {state.weight_pct*100:.0f}%")
            lines.append(f"Durability: {state.equipment_durability_pct*100:.0f}%")
            lines.append(f"Farming duration: {state.farming_duration_min:.0f} min")
            lines.append(f"Action: {state.recommended_action}")
            lines.append(f"Breaks taken: {self._break_count}")
            lines.append(f"Total restock cost: {self._total_restock_cost:,}z")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._potion_stock = {k: 0 for k in self._potion_stock}
            self._potion_consumption_log.clear()
            self._farming_start_time = time.time()
            self._last_restock_time = 0.0
            self._last_repair_time = 0.0
            self._last_vendor_time = 0.0
            self._break_count = 0


# ── Global Singleton ──

_resource_mgr: ResourceManager | None = None
_resource_mgr_lock = RLock()


def get_resource_manager() -> ResourceManager:
    global _resource_mgr
    with _resource_mgr_lock:
        if _resource_mgr is None:
            _resource_mgr = ResourceManager()
        return _resource_mgr
