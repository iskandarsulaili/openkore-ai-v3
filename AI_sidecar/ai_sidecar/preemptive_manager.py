"""
Preemptive Resource Manager — anticipates needs before they become emergencies.

The bridge reflex is the LAST resort. This system is the FIRST line of defense:
1. Tracks consumable usage rates (potions/hr, arrows/hr, fly wings/hr)
2. Predicts when resources will run out
3. Preemptively schedules restocking trips before depletion
4. Monitors HP/SP trends and adjusts behavior before hitting reflex thresholds
5. Coordinates with other bots for team-based resource sharing

This runs in the sidecar's PDCA loop and produces actions that the bridge executes.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────

# Preemptive thresholds (trigger BEFORE bridge reflex at 35% HP)
PREEMPTIVE_HEAL_HP_PCT = 0.50  # Start healing at 50% HP
PREEMPTIVE_HEAL_SP_PCT = 0.30  # Start SP regen at 30% SP
PREEMPTIVE_WEIGHT_PCT = 0.80  # Start heading to town at 80% weight
PREEMPTIVE_POTION_MIN = 5  # Restock when potions drop below 5
PREEMPTIVE_ARROW_MIN = 500  # Restock when arrows drop below 500
PREEMPTIVE_FLYWING_MIN = 5  # Restock when fly wings drop below 5

# How often to re-evaluate preemptive needs
REEVALUATION_INTERVAL = 10.0  # seconds


@dataclass
class ResourceUsage:
    """Tracked usage of a consumable resource."""
    item_name: str
    total_used: int = 0
    total_gained: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    current_stock: int = 0
    max_stock: int = 0

    @property
    def usage_rate_per_hour(self) -> float:
        """Items used per hour based on tracked data."""
        elapsed = (self.last_seen - self.first_seen) / 3600.0
        if elapsed <= 0:
            return 0.0
        return self.total_used / elapsed

    @property
    def hours_until_depletion(self) -> float:
        """Estimated hours until this resource runs out."""
        rate = self.usage_rate_per_hour
        if rate <= 0:
            return 999.0
        return self.current_stock / rate

    @property
    def needs_restock(self) -> bool:
        """Check if this resource needs restocking soon."""
        return self.hours_until_depletion < 0.5  # Less than 30 min remaining


@dataclass
class PreemptiveAction:
    """An action the sidecar should take preemptively."""
    action_type: str  # restock, heal, regen_sp, vendor, repair, etc.
    priority: int  # 1=highest, 5=lowest
    reason: str
    target_map: str = ""
    target_npc: str = ""
    items_needed: list[str] = field(default_factory=list)
    estimated_cost: int = 0


class PreemptiveResourceManager:
    """Anticipates resource needs and produces preemptive actions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._usage: dict[str, ResourceUsage] = {}
        self._last_evaluation: float = 0.0
        self._bot_state: dict[str, dict[str, Any]] = {}
        self._enqueue_fn: Callable | None = None

    def set_enqueue_fn(self, fn: Callable) -> None:
        self._enqueue_fn = fn

    def update_from_snapshot(self, bot_id: str, snapshot: Any) -> None:
        """Update resource tracking from a bot snapshot."""
        with self._lock:
            now = time.time()
            state = self._bot_state.setdefault(bot_id, {})
            state["last_seen"] = now

            # Extract vitals
            if hasattr(snapshot, "vitals"):
                v = snapshot.vitals
                state["hp_pct"] = getattr(v, "hp_ratio", 1.0)
                state["sp_pct"] = getattr(v, "sp_ratio", 1.0)
                state["hp"] = getattr(v, "hp", 0)
                state["max_hp"] = getattr(v, "hp_max", 1)
                state["sp"] = getattr(v, "sp", 0)
                state["max_sp"] = getattr(v, "sp_max", 1)
                state["weight_ratio"] = getattr(v, "weight_ratio", 0.0)
                state["base_level"] = getattr(v, "base_level", 1)
                state["job_level"] = getattr(v, "job_level", 1)
                state["job_name"] = getattr(v, "job_name", "novice")

            # Extract inventory
            if hasattr(snapshot, "inventory_items"):
                for item in (snapshot.inventory_items or []):
                    name = str(getattr(item, "name", ""))
                    amount = int(getattr(item, "amount", 0))
                    if name not in self._usage:
                        self._usage[name] = ResourceUsage(
                            item_name=name,
                            first_seen=now,
                        )
                    usage = self._usage[name]
                    usage.last_seen = now
                    if amount > usage.current_stock:
                        gained = amount - usage.current_stock
                        usage.total_gained += gained
                    elif amount < usage.current_stock:
                        used = usage.current_stock - amount
                        usage.total_used += used
                    usage.current_stock = amount
                    usage.max_stock = max(usage.max_stock, amount)

            # Extract position
            if hasattr(snapshot, "position"):
                pos = snapshot.position
                state["map"] = str(getattr(pos, "map", ""))
                state["x"] = getattr(pos, "x", 0)
                state["y"] = getattr(pos, "y", 0)

    def evaluate(self, bot_id: str) -> list[PreemptiveAction]:
        """Evaluate preemptive needs for a bot.

        Returns a list of actions sorted by priority.
        """
        with self._lock:
            now = time.time()
            if now - self._last_evaluation < REEVALUATION_INTERVAL:
                return []
            self._last_evaluation = now

            state = self._bot_state.get(bot_id, {})
            if not state:
                return []

            actions: list[PreemptiveAction] = []

            # 1. Preemptive healing (before bridge reflex at 35%)
            hp_pct = state.get("hp_pct", 1.0)
            sp_pct = state.get("sp_pct", 1.0)
            weight_ratio = state.get("weight_ratio", 0.0)

            if hp_pct < PREEMPTIVE_HEAL_HP_PCT:
                # Check if we have healing items
                heal_items = [
                    name for name, usage in self._usage.items()
                    if usage.current_stock > 0 and any(
                        kw in name.lower()
                        for kw in ["potion", "herb", "apple", "carrot", "berry"]
                    )
                ]
                if not heal_items:
                    actions.append(PreemptiveAction(
                        action_type="restock",
                        priority=1,
                        reason=f"HP at {hp_pct:.0%}, no healing items available",
                        items_needed=["White Potion"],
                        estimated_cost=500,
                    ))
                else:
                    # Use best healing item
                    best = heal_items[0]
                    actions.append(PreemptiveAction(
                        action_type="heal",
                        priority=2,
                        reason=f"HP at {hp_pct:.0%}, using {best}",
                    ))

            # 2. Preemptive SP management
            if sp_pct < PREEMPTIVE_HEAL_SP_PCT:
                actions.append(PreemptiveAction(
                    action_type="regen_sp",
                    priority=2,
                    reason=f"SP at {sp_pct:.0%}, should sit to regen",
                ))

            # 3. Preemptive weight management
            if weight_ratio > PREEMPTIVE_WEIGHT_PCT:
                actions.append(PreemptiveAction(
                    action_type="vendor",
                    priority=3,
                    reason=f"Weight at {weight_ratio:.0%}, should sell items",
                    target_map=state.get("map", "prontera"),
                ))

            # 4. Preemptive potion restock
            for name, usage in self._usage.items():
                if "potion" in name.lower() and usage.current_stock < PREEMPTIVE_POTION_MIN:
                    actions.append(PreemptiveAction(
                        action_type="restock",
                        priority=3,
                        reason=f"Only {usage.current_stock} {name} left ({usage.hours_until_depletion:.1f}h)",
                        items_needed=[name],
                        estimated_cost=usage.current_stock * 100,
                    ))

            # 5. Preemptive fly wing restock
            for name, usage in self._usage.items():
                if "fly" in name.lower() and usage.current_stock < PREEMPTIVE_FLYWING_MIN:
                    actions.append(PreemptiveAction(
                        action_type="restock",
                        priority=4,
                        reason=f"Only {usage.current_stock} {name} left",
                        items_needed=[name],
                    ))

            # Sort by priority
            actions.sort(key=lambda a: a.priority)
            return actions

    def get_summary(self, bot_id: str) -> str:
        """Get a human-readable summary of resource status."""
        with self._lock:
            state = self._bot_state.get(bot_id, {})
            lines = [f"── Preemptive Resource Status ──"]

            hp_pct = state.get("hp_pct", 1.0)
            sp_pct = state.get("sp_pct", 1.0)
            weight_ratio = state.get("weight_ratio", 0.0)

            lines.append(f"  HP: {hp_pct:.0%}  SP: {sp_pct:.0%}  Weight: {weight_ratio:.0%}")

            # Usage rates
            for name, usage in sorted(self._usage.items()):
                if usage.total_used > 0 or usage.current_stock > 0:
                    rate = usage.usage_rate_per_hour
                    depletion = usage.hours_until_depletion
                    lines.append(
                        f"  {name}: {usage.current_stock} stock, "
                        f"{rate:.1f}/hr usage, "
                        f"{depletion:.1f}h until empty"
                    )

            return "\n".join(lines)


# Global singleton
_manager: PreemptiveResourceManager | None = None
_manager_lock = RLock()


def get_preemptive_manager() -> PreemptiveResourceManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = PreemptiveResourceManager()
        return _manager
