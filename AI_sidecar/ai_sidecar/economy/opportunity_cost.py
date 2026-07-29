"""OpportunityCostEngine — real profit per kill calculations.

A top player doesn't just ask "can I survive this fight?" They ask
"is this fight worth my time?" They calculate zeny per hour, experience
per hour, risk of death, risk of ban, and opportunity cost of not doing
something else.

This module calculates actual profit per kill accounting for potion costs,
time, and loot value. All data-driven from YAML files. Thread-safe.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Default data path relative to the project root
_DEFAULT_DATA_DIR = str(
    Path(__file__).parent.parent.parent.parent / "data"
)

# Minimum HP/damage thresholds to avoid division-by-zero
_MIN_HP = 1.0
_MIN_DAMAGE = 0.1
_MIN_ATTACK_SPEED = 0.01
_MIN_HEAL = 0.1
_MIN_POTION_COST = 0.01

# Overhead added per kill (targeting, looting, movement between spawns)
_KILL_OVERHEAD_SECONDS = 1.0

# How long break-even must be to justify a map switch (hours)
_MAX_BREAK_EVEN_HOURS = 1.0


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class ProfitResult:
    """Complete profit breakdown for a farming target."""
    gross_zeny_per_kill: float       # Expected zeny from drops per kill
    potion_cost_per_kill: float      # Zeny spent on potions per kill
    time_cost_per_kill: float        # Opportunity cost of time spent per kill
    total_costs: float               # Sum of all costs per kill
    net_zeny_per_kill: float         # gross - total_costs
    profit_margin: float             # (net/gross) * 100 as percentage
    kills_per_hour: float            # Estimated kills per hour
    net_zeny_per_hour: float         # net_zeny_per_kill * kills_per_hour
    time_to_kill_seconds: float      # Seconds needed to kill one monster

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for serialization / LLM prompts."""
        return {
            "gross_zeny_per_kill": round(self.gross_zeny_per_kill, 2),
            "potion_cost_per_kill": round(self.potion_cost_per_kill, 2),
            "time_cost_per_kill": round(self.time_cost_per_kill, 2),
            "total_costs": round(self.total_costs, 2),
            "net_zeny_per_kill": round(self.net_zeny_per_kill, 2),
            "profit_margin_pct": round(self.profit_margin, 1),
            "kills_per_hour": round(self.kills_per_hour, 1),
            "net_zeny_per_hour": round(self.net_zeny_per_hour, 1),
            "time_to_kill_seconds": round(self.time_to_kill_seconds, 2),
        }


@dataclass
class OpportunityCost:
    """Opportunity cost analysis between two farming maps."""
    current_zeny_per_hour: float
    alternative_zeny_per_hour: float
    opportunity_cost: float          # What you lose per hour by staying
    loss_per_hour: float             # negative = current is better
    is_losing_zeny: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_zeny_per_hour": round(self.current_zeny_per_hour, 1),
            "alternative_zeny_per_hour": round(self.alternative_zeny_per_hour, 1),
            "opportunity_cost": round(self.opportunity_cost, 1),
            "loss_per_hour": round(self.loss_per_hour, 1),
            "is_losing_zeny": self.is_losing_zeny,
        }


@dataclass
class Opportunity:
    """A potential activity with calculated value (legacy compatibility)."""
    activity: str
    map_name: str = ""
    estimated_zeny_per_hour: float = 0.0
    estimated_xp_per_hour: float = 0.0
    risk_score: float = 0.0
    ban_risk: float = 0.0
    travel_time_minutes: float = 0.0
    setup_time_minutes: float = 0.0
    competition_level: float = 0.0
    value_score: float = 0.0


# ── OpportunityCostEngine ────────────────────────────────────────────────


@dataclass(slots=True)
class OpportunityCostEngine:
    """Calculates real profit per kill accounting for potion costs, time, and loot value.

    All data-driven from YAML files (item_values.yaml for loot prices,
    ro_mechanics.yaml for monster stats). Thread-safe via RLock.

    Methods:
        profit_per_kill()           — Full profit breakdown for a monster
        potion_cost_per_kill()      — Zeny spent on healing per kill
        time_to_kill()              — Seconds needed to kill one monster
        opportunity_cost()          — What you lose by staying on current map
        recommend_map_switch()      — Whether switching maps is worthwhile
        analyze_farming_profit()    — Convenience: integrates with ItemValueDB
    """

    _lock: RLock = field(default_factory=RLock)
    _data_path: str = ""

    # YAML-loaded data
    _item_values: dict[str, dict[str, Any]] = field(default_factory=dict)
    _monster_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Legacy opportunities (for backward compatibility)
    _opportunities: dict[str, Opportunity] = field(default_factory=dict)
    _history: list[dict[str, Any]] = field(default_factory=list)

    # Tracked stats
    _stats: dict[str, int] = field(default_factory=lambda: {
        "calculations": 0, "recommendations": 0, "evaluations": 0,
    })

    # Loaded flag
    _loaded: bool = False

    # ── Initialization ──────────────────────────────────────────────────

    def load(self, data_path: str | None = None) -> None:
        """Load all YAML data files.

        Args:
            data_path: Path to the data directory. If None, uses default.
        """
        with self._lock:
            self._load_item_values(data_path)
            self._load_monster_stats(data_path)
            self._loaded = True
            logger.info(
                "OpportunityCostEngine loaded: %d items, %d monster stats",
                len(self._item_values), len(self._monster_stats),
            )

    def _resolve_data_path(self, data_path: str | None) -> str:
        if data_path:
            return data_path
        return _DEFAULT_DATA_DIR

    def _load_item_values(self, data_path: str | None) -> None:
        """Load item_values.yaml for loot prices."""
        path = os.path.join(self._resolve_data_path(data_path), "item_values.yaml")
        if not os.path.exists(path):
            logger.warning("item_values.yaml not found at %s", path)
            self._item_values = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                self._item_values = {}
                return

            # Filter for valid item entries (have id or classification)
            cleaned: dict[str, dict[str, Any]] = {}
            for key, val in data.items():
                if isinstance(val, dict) and ("id" in val or "classification" in val):
                    cleaned[key] = val
            self._item_values = cleaned
            logger.debug("Loaded %d items from item_values.yaml", len(self._item_values))

        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load item_values.yaml: %s", exc)
            self._item_values = {}

    def _load_monster_stats(self, data_path: str | None) -> None:
        """Load ro_mechanics.yaml for monster stats."""
        path = os.path.join(self._resolve_data_path(data_path), "ro_mechanics.yaml")
        if not os.path.exists(path):
            logger.warning("ro_mechanics.yaml not found at %s", path)
            self._monster_stats = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                self._monster_stats = {}
                return

            # Extract monster stats if present; the file is primarily
            # mechanics data (element table, skills), but may have monster entries.
            monsters = data.get("monsters", data.get("mob_db", {}))
            if isinstance(monsters, dict):
                self._monster_stats = monsters
            else:
                self._monster_stats = {}
            logger.debug("Loaded %d monster stats from ro_mechanics.yaml", len(self._monster_stats))

        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load ro_mechanics.yaml: %s", exc)
            self._monster_stats = {}

    # ── Core Profit Calculations ────────────────────────────────────────

    # Requirement 3
    def time_to_kill(
        self,
        monster_hp: float,
        avg_damage: float,
        attack_speed: float = 1.0,
    ) -> float:
        """Calculate seconds needed to kill a monster.

        Args:
            monster_hp: Monster's total HP.
            avg_damage: Average damage per attack.
            attack_speed: Attacks per second (default 1.0).
                Examples: 2.0 = 2 attacks/sec (fast), 0.5 = 1 attack per 2 sec (slow).

        Returns:
            Seconds to kill the monster. Returns float('inf') if inputs are invalid.
        """
        if monster_hp <= 0 or avg_damage <= _MIN_DAMAGE or attack_speed <= _MIN_ATTACK_SPEED:
            return float("inf")

        hits_needed = monster_hp / avg_damage
        return hits_needed / attack_speed

    # Requirement 2
    def potion_cost_per_kill(
        self,
        damage_taken_per_kill: float,
        heal_amount: float,
        potion_cost: float,
    ) -> float:
        """Calculate zeny spent on potions per kill.

        Args:
            damage_taken_per_kill: HP lost per kill (damage received from the monster).
            heal_amount: HP restored per potion/consumable.
            potion_cost: Zeny cost per potion (from NPC or market).

        Returns:
            Zeny cost per kill from potion usage.
        """
        if damage_taken_per_kill <= 0:
            return 0.0
        if heal_amount < _MIN_HEAL or potion_cost < _MIN_POTION_COST:
            # Can't heal effectively — assume full HP cost at 1 zeny per HP
            return damage_taken_per_kill

        potions_needed = damage_taken_per_kill / heal_amount
        return potions_needed * potion_cost

    def _get_monster_loot_value(self, monster_loot_drops: list[dict[str, Any]]) -> float:
        """Calculate expected loot value per kill from monster drops.

        Uses market prices from item_values.yaml when available.
        Falls back to 'value' field in the drop dict.

        Args:
            monster_loot_drops: List of drop dicts, each with at minimum:
                {'name': str, 'rate': float, 'value': float (optional)}.
                rate is typically in 0.01% units (RO data convention).

        Returns:
            Expected zeny per kill from all drops.
        """
        total = 0.0

        for drop in monster_loot_drops:
            item_name = str(drop.get("name", drop.get("item", drop.get("Item", ""))))
            raw_rate = float(drop.get("rate", drop.get("Rate", 0)))
            fallback_value = float(drop.get("value", drop.get("Value", 0)))

            # Normalize drop rate
            # RO data uses 0.01% units (e.g. 5000 = 50%, 1 = 0.01%)
            # If rate > 1, treat as 0.01% units
            if raw_rate > 1.0:
                drop_rate = raw_rate / 10000.0
            else:
                drop_rate = raw_rate

            # Try to get actual value from item_values.yaml data
            yaml_data = self._item_values.get(item_name)
            if yaml_data is None:
                # Try normalized name
                normalized = item_name.lower().replace(" ", "_").replace("-", "_")
                for key, val in self._item_values.items():
                    if key.lower() == normalized:
                        yaml_data = val
                        break

            if yaml_data:
                npc_sell = float(yaml_data.get("npc_sell", 0) or 0)
                market_price = float(yaml_data.get("market_price", 0) or 0)
                item_value = max(npc_sell, market_price)
            else:
                # Fall back to provided value
                item_value = float(fallback_value)

            total += item_value * drop_rate

        return total

    # Requirement 1
    def profit_per_kill(
        self,
        monster_hp: float,
        avg_damage: float,
        monster_loot_drops: list[dict[str, Any]],
        potion_cost_per_heal: float | None = None,
        kills_per_hour: float | None = None,
        *,
        damage_taken_per_kill: float = 0.0,
        heal_amount: float = 0.0,
        potion_cost: float = 0.0,
        attack_speed: float = 1.0,
        alternative_zeny_per_hour: float = 0.0,
    ) -> ProfitResult:
        """Calculate actual profit per kill accounting for all costs.

        Core method — returns a full ProfitResult with gross, costs, net, margin.

        Args:
            monster_hp: Monster's total HP.
            avg_damage: Average damage per attack.
            monster_loot_drops: List of drop dicts with name, rate, value.
            potion_cost_per_heal: Direct potion cost per kill (overrides
                damage_taken_per_kill + heal_amount + potion_cost).
            kills_per_hour: Direct kills-per-hour override. If None, calculated
                from time_to_kill + overhead.
            damage_taken_per_kill: HP lost per kill (for potion cost calc).
            heal_amount: HP restored per potion.
            potion_cost: Zeny cost per potion.
            attack_speed: Attacks per second.
            alternative_zeny_per_hour: The zeny/hr you could earn in the best
                alternative. Used to compute time opportunity cost.

        Returns:
            ProfitResult with full breakdown.
        """
        with self._lock:
            self._stats["calculations"] += 1

        # 1) Gross income from loot drops
        gross = self._get_monster_loot_value(monster_loot_drops)

        # 2) Potion cost per kill
        if potion_cost_per_heal is not None:
            potion_cost_kill = potion_cost_per_heal
        else:
            potion_cost_kill = self.potion_cost_per_kill(
                damage_taken_per_kill, heal_amount, potion_cost,
            )

        # 3) Time to kill
        ttk = self.time_to_kill(monster_hp, avg_damage, attack_speed)

        # 4) Kills per hour
        if kills_per_hour is not None:
            kph = kills_per_hour
        elif ttk != float("inf") and ttk > 0:
            kill_cycle = ttk + _KILL_OVERHEAD_SECONDS
            kph = 3600.0 / kill_cycle
        else:
            kph = 0.0

        # 5) Time opportunity cost
        # If we have an alternative, calculate what each second of kill time costs us
        if alternative_zeny_per_hour > 0 and kph > 0:
            alternative_per_second = alternative_zeny_per_hour / 3600.0
            time_cost_kill = alternative_per_second * (ttk if ttk != float("inf") else 0)
        else:
            time_cost_kill = 0.0

        # 6) Net
        total_costs = potion_cost_kill + time_cost_kill
        net_per_kill = gross - total_costs

        # 7) Profit margin
        if gross > 0:
            margin = (net_per_kill / gross) * 100.0
        elif net_per_kill < 0:
            margin = -100.0  # Negative profit (pure loss)
        else:
            margin = 0.0

        # 8) Net per hour
        net_per_hour = net_per_kill * kph

        return ProfitResult(
            gross_zeny_per_kill=gross,
            potion_cost_per_kill=potion_cost_kill,
            time_cost_per_kill=time_cost_kill,
            total_costs=total_costs,
            net_zeny_per_kill=net_per_kill,
            profit_margin=margin,
            kills_per_hour=kph,
            net_zeny_per_hour=net_per_hour,
            time_to_kill_seconds=ttk if ttk != float("inf") else -1.0,
        )

    # Requirement 4
    def opportunity_cost(
        self,
        current_map_earnings: float,
        alternative_map_earnings: float,
    ) -> dict[str, Any]:
        """Calculate what you're losing by staying on the current map.

        Args:
            current_map_earnings: Net zeny per hour on the current map.
            alternative_map_earnings: Net zeny per hour on the best alternative.

        Returns:
            Dict with keys:
                current_zeny_per_hour
                alternative_zeny_per_hour
                opportunity_cost        — max(0, alternative - current)
                loss_per_hour           — alternative - current (negative if current is better)
                is_losing_zeny          — True if alternative > current
        """
        loss_per_hour = alternative_map_earnings - current_map_earnings

        return {
            "current_zeny_per_hour": round(current_map_earnings, 1),
            "alternative_zeny_per_hour": round(alternative_map_earnings, 1),
            "opportunity_cost": round(max(0.0, loss_per_hour), 1),
            "loss_per_hour": round(loss_per_hour, 1),
            "is_losing_zeny": loss_per_hour > 0,
        }

    # Requirement 5
    def recommend_map_switch(
        self,
        current_net: float,
        alternative_net: float,
        switch_cost: float = 0.0,
    ) -> bool:
        """Determine if switching maps is worthwhile.

        Compares net earnings and accounts for the one-time cost of switching
        (warp fees, travel time, lost farming while moving).

        Args:
            current_net: Net zeny per hour on the current map.
            alternative_net: Net zeny per hour on the alternative map.
            switch_cost: One-time cost (in zeny equivalent) of switching.
                Includes warp fees, lost farming time, etc.

        Returns:
            True if switching maps is recommended (break-even within threshold).
        """
        with self._lock:
            self._stats["recommendations"] += 1

        gain_per_hour = alternative_net - current_net
        if gain_per_hour <= 0:
            # Switching would earn same or less — not worth it
            return False

        if switch_cost <= 0:
            # No switching cost — always worth it
            return True

        # Only recommend if break-even is within acceptable threshold
        break_even_hours = switch_cost / gain_per_hour
        return break_even_hours < _MAX_BREAK_EVEN_HOURS

    # ── YAML Data Lookups ───────────────────────────────────────────────

    def get_item_value(self, item_name: str) -> dict[str, Any] | None:
        """Look up an item's value data from item_values.yaml.

        Args:
            item_name: Name of the item.

        Returns:
            Item data dict (npc_sell, market_price, etc.) or None if not found.
        """
        with self._lock:
            if not self._loaded:
                return None

            # Exact match
            result = self._item_values.get(item_name)
            if result:
                return result

            # Normalized match
            normalized = item_name.lower().replace(" ", "_").replace("-", "_")
            for key, val in self._item_values.items():
                if key.lower() == normalized:
                    return val

            return None

    def get_monster_stat(self, monster_name: str) -> dict[str, Any] | None:
        """Look up a monster's stats from ro_mechanics.yaml.

        Args:
            monster_name: Name of the monster.

        Returns:
            Monster stat dict (HP, level, etc.) or None if not found.
        """
        with self._lock:
            if not self._loaded:
                return None

            # Exact match
            result = self._monster_stats.get(monster_name)
            if result:
                return result

            # Case-insensitive
            lower = monster_name.lower()
            for key, val in self._monster_stats.items():
                if key.lower() == lower:
                    return val

            return None

    def analyze_farming_profit(
        self,
        monster_name: str,
        monster_hp: float,
        avg_damage: float,
        *,
        attack_speed: float = 1.0,
        damage_taken_per_kill: float = 0.0,
        heal_amount: float = 0.0,
        potion_cost: float = 0.0,
        kills_per_hour: float | None = None,
        alternative_zeny_per_hour: float = 0.0,
    ) -> ProfitResult:
        """Convenience: analyze profit for a monster, integrating with ItemValueDB.

        Automatically looks up the monster's drops from the item_value_db
        (knowledge.json) and calculates real profit per kill.

        Args:
            monster_name: Name of the monster to analyze.
            monster_hp: Monster HP (used as fallback if not in DB).
            avg_damage: Average damage per attack.
            attack_speed: Attacks per second.
            damage_taken_per_kill: HP lost per kill.
            heal_amount: HP restored per potion.
            potion_cost: Zeny per potion.
            kills_per_hour: Override kills per hour.
            alternative_zeny_per_hour: Best alternative zeny/hr for time cost.

        Returns:
            ProfitResult with full profit breakdown.
        """
        # Try to get loot data from the ItemValueDB
        monster_loot_drops: list[dict[str, Any]] = []
        try:
            from ai_sidecar.economy.item_value_db import get_item_value_db
            item_db = get_item_value_db()
            mdv = item_db.get_monster_drop_value(monster_name)
            if mdv and mdv.drops:
                monster_loot_drops = [
                    {
                        "name": d.get("item", ""),
                        "rate": d.get("rate", 0),
                        "value": d.get("expected_value", 0),
                    }
                    for d in mdv.drops
                ]
        except Exception:
            logger.debug("item_value_db unavailable for %s", monster_name)

        # If no drops found, use monster HP as weak signal (base zeny drop)
        if not monster_loot_drops:
            monster_loot_drops = [{"name": "zeny", "rate": 100, "value": monster_hp * 0.01}]

        return self.profit_per_kill(
            monster_hp=monster_hp,
            avg_damage=avg_damage,
            monster_loot_drops=monster_loot_drops,
            damage_taken_per_kill=damage_taken_per_kill,
            heal_amount=heal_amount,
            potion_cost=potion_cost,
            attack_speed=attack_speed,
            kills_per_hour=kills_per_hour,
            alternative_zeny_per_hour=alternative_zeny_per_hour,
        )

    # ── Legacy Opportunity Evaluation ──────────────────────────────────

    def evaluate_opportunity(
        self,
        activity: str,
        *,
        map_name: str = "",
        estimated_zeny_per_hour: float = 0.0,
        estimated_xp_per_hour: float = 0.0,
        risk_score: float = 0.0,
        ban_risk: float = 0.0,
        travel_time_minutes: float = 0.0,
        setup_time_minutes: float = 0.0,
        competition_level: float = 0.0,
    ) -> Opportunity:
        """Evaluate an opportunity and compute its value score (legacy API).

        Combines zeny, XP, risk, time overhead, and competition into a
        single value score.

        Returns:
            Opportunity with computed value_score.
        """
        with self._lock:
            self._stats["evaluations"] += 1

        # Base value: zeny + XP (converted to zeny equivalent)
        xp_value = estimated_xp_per_hour * 0.01  # 1 XP = 0.01 zeny (rough)
        base_value = estimated_zeny_per_hour + xp_value

        # Time penalty: travel + setup
        total_overhead = travel_time_minutes + setup_time_minutes
        time_penalty = total_overhead * (base_value / 60) * 0.5

        # Risk penalty
        risk_penalty = base_value * risk_score * 2.0
        ban_penalty = base_value * ban_risk * 10.0

        # Competition penalty
        competition_penalty = base_value * competition_level * 0.3

        # Final value score
        value_score = base_value - time_penalty - risk_penalty - ban_penalty - competition_penalty

        opp = Opportunity(
            activity=activity,
            map_name=map_name,
            estimated_zeny_per_hour=estimated_zeny_per_hour,
            estimated_xp_per_hour=estimated_xp_per_hour,
            risk_score=risk_score,
            ban_risk=ban_risk,
            travel_time_minutes=travel_time_minutes,
            setup_time_minutes=setup_time_minutes,
            competition_level=competition_level,
            value_score=max(0, value_score),
        )

        with self._lock:
            self._opportunities[activity] = opp

        return opp

    def get_best_opportunity(self, min_value: float = 0.0) -> Opportunity | None:
        """Get the highest-value opportunity (legacy API)."""
        with self._lock:
            valid = [o for o in self._opportunities.values() if o.value_score >= min_value]
            if not valid:
                return None
            return max(valid, key=lambda o: o.value_score)

    # ── History & Reporting ─────────────────────────────────────────────

    def record_outcome(
        self,
        activity: str,
        actual_zeny: float = 0.0,
        actual_xp: float = 0.0,
        died: bool = False,
        *,
        map_name: str = "",
        kills: int = 0,
    ) -> None:
        """Record the actual outcome of a farming session for learning.

        Args:
            activity: Activity or map name.
            actual_zeny: Zeny earned.
            actual_xp: XP earned.
            died: Whether the character died.
            map_name: Map name (alternative to activity).
            kills: Number of kills.
        """
        with self._lock:
            self._history.append({
                "activity": activity,
                "map_name": map_name or activity,
                "actual_zeny": actual_zeny,
                "actual_xp": actual_xp,
                "died": died,
                "kills": kills,
                "timestamp": time.time(),
            })

    def get_profit_context(
        self,
        results: list[ProfitResult] | None = None,
        top_n: int = 5,
    ) -> str:
        """Get formatted profit analysis for LLM prompts / console output.

        Args:
            results: List of ProfitResult to display. If None, uses legacy
                     opportunity data.
            top_n: Number of results to show.

        Returns:
            Formatted multi-line string.
        """
        lines: list[str] = []

        if results:
            lines.append("── Profit Analysis ──")
            for i, r in enumerate(results[:top_n]):
                lines.append(
                    f"  #{i + 1}: gross={r.gross_zeny_per_kill:.1f}z "
                    f"potion={r.potion_cost_per_kill:.1f}z "
                    f"time={r.time_cost_per_kill:.1f}z "
                    f"net={r.net_zeny_per_kill:.1f}z/kill "
                    f"({r.net_zeny_per_hour:.0f}z/hr) "
                    f"margin={r.profit_margin:.1f}%"
                )
        else:
            # Legacy context
            with self._lock:
                opportunities = sorted(
                    self._opportunities.values(),
                    key=lambda o: -o.value_score,
                )[:top_n]

            if opportunities:
                lines.append("── Opportunity Cost Analysis ──")
                for opp in opportunities:
                    lines.append(
                        f"  {opp.activity} ({opp.map_name}): "
                        f"value={opp.value_score:.0f} "
                        f"zeny/hr={opp.estimated_zeny_per_hour:.0f} "
                        f"risk={opp.risk_score:.1f} "
                        f"ban={opp.ban_risk:.1f}"
                    )

        return "\n".join(lines)

    def get_opportunity_context(self) -> str:
        """Legacy alias for get_profit_context()."""
        return self.get_profit_context(results=None)

    def counters(self) -> dict[str, int]:
        """Get engine usage counters."""
        with self._lock:
            return dict(self._stats)


# ── Factory / Singleton ──────────────────────────────────────────────────


_engine: OpportunityCostEngine | None = None
_engine_lock = RLock()


def get_opportunity_cost_engine() -> OpportunityCostEngine:
    """Get the global OpportunityCostEngine singleton (factory function).

    Creates the engine on first call, loads YAML data, and caches it.
    Thread-safe.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = OpportunityCostEngine()
            _engine.load()
        return _engine


# Legacy alias for backward compatibility
def get_opportunity() -> OpportunityCostEngine:
    """Legacy alias for get_opportunity_cost_engine()."""
    return get_opportunity_cost_engine()
