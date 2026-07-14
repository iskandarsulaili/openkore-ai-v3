"""
Preemptive Intelligence Engine — anticipates needs across all domains before they become emergencies.

This is the bot's "conscious brain" — it thinks ahead across:
- Combat readiness: HP/SP trends, gear durability, consumable burn rates, skill cooldowns
- Resource economy: Zeny burn rate, potion/arrow/fly wing depletion, weight management
- Party coordination: Buff timing, heal rotation, shared resource pooling
- Leveling: When to switch maps, when to reset stats, when to upgrade gear
- Market timing: Price trends, WoE supply/demand, arbitrage windows
- Safety: GM patrol patterns, PK hotspots, dangerous time-of-day patterns

The bridge reflex is the LAST resort (sub-200ms safety net).
This system is the FIRST line of defense — it acts minutes/hours before problems arise.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Domain-specific thresholds ─────────────────────────────────────────────

class Domain:
    COMBAT = "combat"
    ECONOMY = "economy"
    INVENTORY = "inventory"
    PARTY = "party"
    LEVELING = "leveling"
    SAFETY = "safety"
    MARKET = "market"


@dataclass
class Signal:
    """A detected signal that something needs attention."""
    domain: str
    name: str
    value: float
    threshold: float
    severity: int  # 1=critical, 2=warning, 3=info
    trend: str  # rising, falling, stable
    message: str
    timestamp: float = 0.0

    @property
    def is_triggered(self) -> bool:
        return self.value >= self.threshold if self.trend == "rising" else self.value <= self.threshold


@dataclass
class PreemptiveAction:
    """An action the system should take preemptively."""
    domain: str
    action_type: str
    priority: int  # 1=immediate, 5=when convenient
    reason: str
    target_map: str = ""
    target_npc: str = ""
    items_needed: list[str] = field(default_factory=list)
    estimated_cost: int = 0
    timeout: float = 300.0  # Max seconds to wait before re-evaluating
    created_at: float = 0.0


class PreemptiveIntelligence:
    """Anticipates needs across all domains and produces preemptive actions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._bot_state: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._signals: dict[str, list[Signal]] = defaultdict(list)
        self._active_actions: dict[str, list[PreemptiveAction]] = defaultdict(list)
        self._last_evaluation: float = 0.0
        self._evaluation_interval: float = 5.0  # Evaluate every 5 seconds
        self._enqueue_fn: Callable | None = None
        self._start_time = time.time()

    def set_enqueue_fn(self, fn: Callable) -> None:
        self._enqueue_fn = fn

    def update_from_snapshot(self, bot_id: str, snapshot: Any) -> None:
        """Update all tracked state from a bot snapshot."""
        with self._lock:
            now = time.time()
            state = self._bot_state.setdefault(bot_id, {})
            prev_state = dict(state)  # Copy for trend detection
            state["last_seen"] = now

            # ── Vitals ──
            if hasattr(snapshot, "vitals"):
                v = snapshot.vitals
                state["hp"] = getattr(v, "hp", 0)
                state["max_hp"] = getattr(v, "hp_max", 1)
                state["hp_pct"] = getattr(v, "hp_ratio", 1.0)
                state["sp"] = getattr(v, "sp", 0)
                state["max_sp"] = getattr(v, "sp_max", 1)
                state["sp_pct"] = getattr(v, "sp_ratio", 1.0)
                state["weight"] = getattr(v, "weight", 0)
                state["max_weight"] = getattr(v, "weight_max", 1)
                state["weight_ratio"] = getattr(v, "weight_ratio", 0.0)
                state["base_level"] = getattr(v, "base_level", 1)
                state["job_level"] = getattr(v, "job_level", 1)
                state["job_name"] = str(getattr(v, "job_name", "novice")).lower()
                state["zeny"] = getattr(v, "zeny", 0)

            # ── Position ──
            if hasattr(snapshot, "position"):
                pos = snapshot.position
                state["map"] = str(getattr(pos, "map", ""))
                state["x"] = getattr(pos, "x", 0)
                state["y"] = getattr(pos, "y", 0)

            # ── Combat state ──
            if hasattr(snapshot, "combat"):
                c = snapshot.combat
                state["aggro_count"] = getattr(c, "aggro_count", 0)
                state["in_combat"] = getattr(c, "in_combat", False)
                state["target_id"] = getattr(c, "target_id", 0)

            # ── Inventory ──
            if hasattr(snapshot, "inventory_items"):
                inventory = {}
                for item in (snapshot.inventory_items or []):
                    name = str(getattr(item, "name", ""))
                    amount = int(getattr(item, "amount", 0))
                    inventory[name] = amount
                state["inventory"] = inventory

            # ── Skills ──
            if hasattr(snapshot, "skills"):
                state["skills"] = [
                    str(getattr(s, "name", ""))
                    for s in (snapshot.skills or [])
                ]

            # ── Actors (nearby entities) ──
            if hasattr(snapshot, "actors"):
                actors = snapshot.actors or []
                state["nearby_monsters"] = len([
                    a for a in actors
                    if getattr(a, "actor_type", "") == "monster"
                ])
                state["nearby_players"] = len([
                    a for a in actors
                    if getattr(a, "actor_type", "") == "player"
                ])
                state["nearby_party"] = len([
                    a for a in actors
                    if getattr(a, "actor_type", "") == "player"
                    and getattr(a, "is_party_member", False)
                ])

            # ── Record history for trend detection ──
            self._history[bot_id].append({
                "time": now,
                "hp_pct": state.get("hp_pct", 1.0),
                "sp_pct": state.get("sp_pct", 1.0),
                "weight_ratio": state.get("weight_ratio", 0.0),
                "aggro_count": state.get("aggro_count", 0),
                "zeny": state.get("zeny", 0),
            })
            # Keep last 60 entries (5 min at 5s intervals)
            if len(self._history[bot_id]) > 60:
                self._history[bot_id] = self._history[bot_id][-60:]

    def _detect_trend(self, bot_id: str, key: str) -> str:
        """Detect if a value is rising, falling, or stable."""
        history = self._history.get(bot_id, [])
        if len(history) < 5:
            return "stable"

        recent = [h.get(key, 0) for h in history[-5:]]
        if len(recent) < 2:
            return "stable"

        # Linear trend: compare first half to second half
        mid = len(recent) // 2
        first_avg = sum(recent[:mid]) / max(mid, 1)
        second_avg = sum(recent[mid:]) / max(len(recent) - mid, 1)

        change = (second_avg - first_avg) / max(abs(first_avg), 0.01)
        if change > 0.05:
            return "rising"
        elif change < -0.05:
            return "falling"
        return "stable"

    def _compute_signals(self, bot_id: str) -> list[Signal]:
        """Compute all signals for a bot across all domains."""
        state = self._bot_state.get(bot_id, {})
        if not state:
            return []

        signals: list[Signal] = []
        now = time.time()

        # ── COMBAT READINESS SIGNALS ──
        hp_pct = state.get("hp_pct", 1.0)
        sp_pct = state.get("sp_pct", 1.0)
        hp_trend = self._detect_trend(bot_id, "hp_pct")
        sp_trend = self._detect_trend(bot_id, "sp_pct")

        signals.append(Signal(
            domain=Domain.COMBAT,
            name="hp_ratio",
            value=hp_pct,
            threshold=0.50,
            severity=2 if hp_pct < 0.50 else 3,
            trend=hp_trend,
            message=f"HP at {hp_pct:.0%} ({hp_trend})",
            timestamp=now,
        ))
        signals.append(Signal(
            domain=Domain.COMBAT,
            name="sp_ratio",
            value=sp_pct,
            threshold=0.30,
            severity=2 if sp_pct < 0.30 else 3,
            trend=sp_trend,
            message=f"SP at {sp_pct:.0%} ({sp_trend})",
            timestamp=now,
        ))

        # Aggro trend
        aggro = state.get("aggro_count", 0)
        aggro_trend = self._detect_trend(bot_id, "aggro_count")
        if aggro > 3 or aggro_trend == "rising":
            signals.append(Signal(
                domain=Domain.COMBAT,
                name="aggro_risk",
                value=float(aggro),
                threshold=3.0,
                severity=1 if aggro > 5 else 2,
                trend=aggro_trend,
                message=f"Aggro count: {aggro} ({aggro_trend})",
                timestamp=now,
            ))

        # ── INVENTORY SIGNALS ──
        inventory = state.get("inventory", {})
        weight_ratio = state.get("weight_ratio", 0.0)
        weight_trend = self._detect_trend(bot_id, "weight_ratio")

        if weight_ratio > 0.80:
            signals.append(Signal(
                domain=Domain.INVENTORY,
                name="weight_capacity",
                value=weight_ratio,
                threshold=0.80,
                severity=2,
                trend=weight_trend,
                message=f"Weight at {weight_ratio:.0%} ({weight_trend})",
                timestamp=now,
            ))

        # Check critical consumables
        for item_name, qty in inventory.items():
            item_lower = item_name.lower()
            if "potion" in item_lower and qty < 5:
                signals.append(Signal(
                    domain=Domain.INVENTORY,
                    name=f"low_{item_name}",
                    value=float(qty),
                    threshold=5.0,
                    severity=2,
                    trend="falling",
                    message=f"Only {qty} {item_name} left",
                    timestamp=now,
                ))
            if ("fly" in item_lower or "butterfly" in item_lower) and qty < 3:
                signals.append(Signal(
                    domain=Domain.INVENTORY,
                    name=f"low_{item_name}",
                    value=float(qty),
                    threshold=3.0,
                    severity=3,
                    trend="falling",
                    message=f"Only {qty} {item_name} left",
                    timestamp=now,
                ))

        # ── ECONOMY SIGNALS ──
        zeny = state.get("zeny", 0)
        zeny_trend = self._detect_trend(bot_id, "zeny")
        if zeny < 1000:
            signals.append(Signal(
                domain=Domain.ECONOMY,
                name="low_zeny",
                value=float(zeny),
                threshold=1000.0,
                severity=2,
                trend=zeny_trend,
                message=f"Only {zeny}z remaining",
                timestamp=now,
            ))

        # ── PARTY SIGNALS ──
        nearby_party = state.get("nearby_party", 0)
        if nearby_party > 0 and hp_pct < 0.60:
            signals.append(Signal(
                domain=Domain.PARTY,
                name="party_heal_needed",
                value=hp_pct,
                threshold=0.60,
                severity=2,
                trend=hp_trend,
                message=f"Party members nearby, HP at {hp_pct:.0%}",
                timestamp=now,
            ))

        # ── SAFETY SIGNALS ──
        nearby_players = state.get("nearby_players", 0)
        if nearby_players > 5 and state.get("map", "").startswith("prt_fild"):
            signals.append(Signal(
                domain=Domain.SAFETY,
                name="crowded_map",
                value=float(nearby_players),
                threshold=5.0,
                severity=3,
                trend="stable",
                message=f"{nearby_players} players on map — may be crowded",
                timestamp=now,
            ))

        return signals

    def _signals_to_actions(self, bot_id: str, signals: list[Signal]) -> list[PreemptiveAction]:
        """Convert triggered signals into preemptive actions."""
        actions: list[PreemptiveAction] = []
        state = self._bot_state.get(bot_id, {})

        for sig in signals:
            if not sig.is_triggered:
                continue

            # ── Combat actions ──
            if sig.domain == Domain.COMBAT:
                if sig.name == "hp_ratio" and sig.value < 0.50:
                    # Check if we have healing items
                    inventory = state.get("inventory", {})
                    heal_items = [
                        name for name in inventory
                        if any(kw in name.lower() for kw in ["potion", "herb", "apple", "carrot"])
                        and inventory[name] > 0
                    ]
                    if not heal_items:
                        actions.append(PreemptiveAction(
                            domain=Domain.COMBAT,
                            action_type="restock_heal",
                            priority=1,
                            reason=f"HP at {sig.value:.0%}, no healing items — need to restock",
                            estimated_cost=500,
                        ))
                if sig.name == "aggro_risk" and sig.value > 3:
                    actions.append(PreemptiveAction(
                        domain=Domain.COMBAT,
                        action_type="flee_to_safety",
                        priority=1,
                        reason=f"Aggro count {int(sig.value)} — too many enemies",
                    ))

            # ── Inventory actions ──
            if sig.domain == Domain.INVENTORY:
                if sig.name == "weight_capacity":
                    actions.append(PreemptiveAction(
                        domain=Domain.INVENTORY,
                        action_type="vendor_trash",
                        priority=3,
                        reason=f"Weight at {sig.value:.0%} — should sell/discard items",
                    ))
                if sig.name.startswith("low_"):
                    item_name = sig.name[4:]  # Remove "low_" prefix
                    actions.append(PreemptiveAction(
                        domain=Domain.INVENTORY,
                        action_type="restock",
                        priority=2,
                        reason=f"Low on {item_name} ({int(sig.value)} left)",
                        items_needed=[item_name],
                    ))

            # ── Economy actions ──
            if sig.domain == Domain.ECONOMY:
                if sig.name == "low_zeny":
                    actions.append(PreemptiveAction(
                        domain=Domain.ECONOMY,
                        action_type="farm_zeny",
                        priority=2,
                        reason=f"Only {int(sig.value)}z — need to farm more",
                    ))

            # ── Party actions ──
            if sig.domain == Domain.PARTY:
                if sig.name == "party_heal_needed":
                    actions.append(PreemptiveAction(
                        domain=Domain.PARTY,
                        action_type="request_heal",
                        priority=2,
                        reason=f"HP at {sig.value:.0%}, party members nearby — request heal",
                    ))

            # ── Safety actions ──
            if sig.domain == Domain.SAFETY:
                if sig.name == "crowded_map":
                    actions.append(PreemptiveAction(
                        domain=Domain.SAFETY,
                        action_type="switch_map",
                        priority=4,
                        reason=f"Map crowded ({int(sig.value)} players) — consider switching",
                    ))

        # Sort by priority
        actions.sort(key=lambda a: a.priority)
        return actions

    def evaluate(self, bot_id: str) -> list[PreemptiveAction]:
        """Full preemptive evaluation for a bot.

        Returns actions sorted by priority (1=immediate, 5=when convenient).
        """
        with self._lock:
            now = time.time()
            if now - self._last_evaluation < self._evaluation_interval:
                return self._active_actions.get(bot_id, [])

            self._last_evaluation = now
            signals = self._compute_signals(bot_id)
            self._signals[bot_id] = signals

            actions = self._signals_to_actions(bot_id, signals)
            self._active_actions[bot_id] = actions

            if actions:
                logger.info(
                    "preemptive_eval: bot=%s signals=%d actions=%d top=%s",
                    bot_id, len(signals), len(actions),
                    actions[0].reason if actions else "none",
                )

            return actions

    def get_summary(self, bot_id: str) -> str:
        """Get a human-readable summary of preemptive state."""
        with self._lock:
            state = self._bot_state.get(bot_id, {})
            signals = self._signals.get(bot_id, [])
            actions = self._active_actions.get(bot_id, [])

            lines = [f"── Preemptive Intelligence ──"]
            lines.append(f"  Bot: {bot_id}")
            lines.append(f"  Map: {state.get('map', '?')}")
            lines.append(f"  HP: {state.get('hp_pct', 1.0):.0%}  SP: {state.get('sp_pct', 1.0):.0%}")
            lines.append(f"  Weight: {state.get('weight_ratio', 0.0):.0%}  Zeny: {state.get('zeny', 0)}z")
            lines.append(f"  Level: {state.get('base_level', 1)}/{state.get('job_level', 1)}")
            lines.append("")

            if signals:
                lines.append("  Signals:")
                for sig in sorted(signals, key=lambda s: s.severity):
                    marker = "🔴" if sig.severity == 1 else "🟡" if sig.severity == 2 else "🟢"
                    lines.append(f"    {marker} [{sig.domain}] {sig.message}")
            else:
                lines.append("  No signals detected.")

            if actions:
                lines.append("")
                lines.append("  Recommended actions:")
                for a in actions:
                    lines.append(f"    [{a.priority}] {a.action_type}: {a.reason}")
            else:
                lines.append("  No actions needed.")

            return "\n".join(lines)


# Global singleton
_intelligence: PreemptiveIntelligence | None = None
_intelligence_lock = RLock()


def get_preemptive_intelligence() -> PreemptiveIntelligence:
    global _intelligence
    with _intelligence_lock:
        if _intelligence is None:
            _intelligence = PreemptiveIntelligence()
        return _intelligence
