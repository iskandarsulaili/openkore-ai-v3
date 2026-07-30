"""
Farming Profitability Calculator — Market-aware farming with session tracking.

Given a map's monster spawns and the player's capabilities, calculates:
  - Expected zeny/hour = sum((drop_rate × item_value) for each drop)
  - Weighted by monster spawn density and kill speed
  - Compares across maps to recommend best farm spot
  - Tracks actual zeny/hour and adjusts estimates over time
  - Server economy integration: fetch current market prices
  - Kill target priority based on real-time profit potential
  - Potion cost tracking, weight-adjusted profit, diminishing returns detection
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.economy.database import ItemValueDB

logger = logging.getLogger(__name__)

# ── rAthena drop rate constants ──
COMMON_DROP_RATE = 0.55
UNCOMMON_DROP_RATE = 0.20
RARE_DROP_RATE = 0.05
CARD_DROP_RATE = 0.0001
MVP_CARD_DROP_RATE = 0.00001

# ── Monster drop tables ──
_MONSTER_DROPS: dict[str, list[tuple[str, float, int]]] = {
    "poring": [
        ("jellopy", COMMON_DROP_RATE, 12),
        ("apple", UNCOMMON_DROP_RATE, 25),
        ("clover", RARE_DROP_RATE, 25),
        ("poring_card", CARD_DROP_RATE, 50000),
    ],
    "lunatic": [
        ("clover", COMMON_DROP_RATE, 25),
        ("carrot", UNCOMMON_DROP_RATE, 5),
        ("lunatic_card", CARD_DROP_RATE, 30000),
    ],
    "pupa": [
        ("sticky_mucus", COMMON_DROP_RATE, 40),
        ("pupa_card", CARD_DROP_RATE, 20000),
    ],
    "familiar": [
        ("bat", COMMON_DROP_RATE, 30),
        ("immortal_heart", UNCOMMON_DROP_RATE, 300),
        ("familiar_card", CARD_DROP_RATE, 25000),
    ],
    "zombie": [
        ("decayed_nail", COMMON_DROP_RATE, 60),
        ("immortal_heart", UNCOMMON_DROP_RATE, 300),
        ("rotten_bandage", RARE_DROP_RATE, 200),
        ("zombie_card", CARD_DROP_RATE, 40000),
    ],
    "skeleton": [
        ("bone", COMMON_DROP_RATE, 30),
        ("skull", UNCOMMON_DROP_RATE, 50),
        ("skeleton_card", CARD_DROP_RATE, 35000),
    ],
    "orc_warrior": [
        ("orcs_eye", COMMON_DROP_RATE, 100),
        ("orcish_voucher", UNCOMMON_DROP_RATE, 250),
        ("iron", RARE_DROP_RATE, 200),
        ("orc_warrior_card", CARD_DROP_RATE, 80000),
    ],
    "poporing": [
        ("sticky_mucus", COMMON_DROP_RATE, 40),
        ("poison_spore", UNCOMMON_DROP_RATE, 250),
        ("green_herb", UNCOMMON_DROP_RATE, 50),
        ("poporing_card", CARD_DROP_RATE, 60000),
    ],
    "creamy": [
        ("scell", COMMON_DROP_RATE, 15),
        ("feather", UNCOMMON_DROP_RATE, 10),
        ("creamy_card", CARD_DROP_RATE, 25000),
    ],
    "rocker": [
        ("scell", COMMON_DROP_RATE, 15),
        ("coal", RARE_DROP_RATE, 150),
        ("rocker_card", CARD_DROP_RATE, 20000),
    ],
    "spore": [
        ("mushroom_spore", COMMON_DROP_RATE, 50),
        ("scell", UNCOMMON_DROP_RATE, 15),
        ("spore_card", CARD_DROP_RATE, 25000),
    ],
    "drainliar": [
        ("drainliar_card", CARD_DROP_RATE, 30000),
        ("little_bug_horn", COMMON_DROP_RATE, 50),
    ],
    "hunter_fly": [
        ("hunter_fly_card", CARD_DROP_RATE, 80000),
        ("feather", COMMON_DROP_RATE, 10),
    ],
    "thief_bug": [
        ("thief_bug_card", CARD_DROP_RATE, 20000),
        ("shell", COMMON_DROP_RATE, 17),
    ],
    "munak": [
        ("munak_card", CARD_DROP_RATE, 50000),
        ("bone", COMMON_DROP_RATE, 30),
        ("immortal_heart", UNCOMMON_DROP_RATE, 300),
    ],
    "bongun": [
        ("bongun_card", CARD_DROP_RATE, 50000),
        ("bone", COMMON_DROP_RATE, 30),
        ("immortal_heart", UNCOMMON_DROP_RATE, 300),
    ],
    "ghoul": [
        ("ghoul_card", CARD_DROP_RATE, 70000),
        ("decayed_nail", COMMON_DROP_RATE, 60),
        ("immortal_heart", UNCOMMON_DROP_RATE, 300),
    ],
    "marine_sphere": [
        ("shell", COMMON_DROP_RATE, 17),
        ("marine_sphere_card", CARD_DROP_RATE, 40000),
    ],
    "hydra": [
        ("hydra_card", CARD_DROP_RATE, 120000),
        ("sticky_mucus", COMMON_DROP_RATE, 40),
    ],
    "kukre": [
        ("kukre_card", CARD_DROP_RATE, 30000),
        ("shell", COMMON_DROP_RATE, 17),
    ],
    "vadon": [
        ("vadon_card", CARD_DROP_RATE, 60000),
        ("shell", COMMON_DROP_RATE, 17),
    ],
    "orc_zombie": [
        ("decayed_nail", COMMON_DROP_RATE, 60),
        ("orcish_voucher", UNCOMMON_DROP_RATE, 250),
        ("immortal_heart", UNCOMMON_DROP_RATE, 300),
    ],
    "steel_chonchon": [
        ("iron", RARE_DROP_RATE, 200),
        ("steel", RARE_DROP_RATE, 750),
    ],
    "plankton": [
        ("shell", COMMON_DROP_RATE, 17),
    ],
    "marina": [
        ("shell", COMMON_DROP_RATE, 17),
    ],
    "soldier_skeleton": [
        ("bone", COMMON_DROP_RATE, 30),
        ("soldier_skeleton_card", CARD_DROP_RATE, 150000),
    ],
}

# ── Map spawn data ──
_MAP_SPAWNS: dict[str, list[tuple[str, int, int]]] = {
    "pay_dun00": [("Familiar", 15, 0), ("Zombie", 20, 0), ("Skeleton", 35, 0), ("Poporing", 15, 0)],
    "pay_dun01": [("Munak", 20, 0), ("Bongun", 15, 0), ("Ghoul", 10, 0), ("Skeleton", 30, 0)],
    "gef_dun00": [("Hunter Fly", 30, 60000), ("Poporing", 15, 0), ("Poison Spore", 25, 0)],
    "orcsdun01": [("Steel Chonchon", 10, 0), ("Familiar", 15, 0), ("Drainliar", 5, 0), ("Orc Zombie", 80, 60000)],
    "iz_dun00": [("Plankton", 65, 0), ("Marina", 45, 0), ("Kukre", 15, 0), ("Hydra", 15, 0), ("Vadon", 15, 0)],
    "prt_fild05": [("Poring", 70, 0), ("Thief Bug Egg", 20, 0), ("Lunatic", 30, 0), ("Pupa", 30, 0), ("Thief Bug", 10, 0)],
    "prt_fild04": [("Rocker", 70, 0), ("Creamy", 40, 0), ("Pupa", 10, 0), ("Poring", 30, 0)],
}

# ── Potion costs ──
_POTION_COSTS: dict[str, int] = {
    "red_potion": 50,
    "orange_potion": 200,
    "yellow_potion": 500,
    "white_potion": 1200,
    "blue_potion": 800,
    "concentration_potion": 2000,
    "awakening_potion": 3000,
    "berserk_potion": 5000,
}

# ── Weight constants ──
_MAX_WEIGHT_BASE = 2000  # Base max weight for most classes
_WEIGHT_PER_STR = 30     # Additional weight per STR point


@dataclass
class SessionMetrics:
    """Per-session farming metrics."""
    zeny_gained: int = 0
    items_gained: int = 0
    kills: int = 0
    deaths: int = 0
    potion_cost: int = 0
    start_time: float = 0.0
    map_name: str = ""
    returns_to_town: int = 0

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600.0

    @property
    def zeny_per_hour(self) -> float:
        h = self.elapsed_hours
        return self.zeny_gained / h if h > 0 else 0.0

    @property
    def kills_per_hour(self) -> float:
        h = self.elapsed_hours
        return self.kills / h if h > 0 else 0.0

    @property
    def deaths_per_hour(self) -> float:
        h = self.elapsed_hours
        return self.deaths / h if h > 0 else 0.0

    @property
    def potion_cost_per_hour(self) -> float:
        h = self.elapsed_hours
        return self.potion_cost / h if h > 0 else 0.0

    @property
    def net_zeny_per_hour(self) -> float:
        """Gross zeny/hour minus potion cost/hour."""
        return self.zeny_per_hour - self.potion_cost_per_hour

    @property
    def items_per_hour(self) -> float:
        h = self.elapsed_hours
        return self.items_gained / h if h > 0 else 0.0


@dataclass
class MapProfitability:
    """Profitability assessment for a single map."""
    map_name: str
    zeny_per_hour: float
    expected_zeny_per_hour: float
    net_zeny_per_hour: float
    zeny_per_kill: float
    kills_per_hour: float
    potion_cost_per_hour: float
    monsters: list[dict[str, Any]]
    confidence: float
    risks: list[str]
    recommended: bool = False


@dataclass
class KillTargetPriority:
    """Priority score for a monster as a farming target."""
    monster_name: str
    profit_score: float
    zeny_per_kill: float
    card_value: int
    drop_rate: float
    kill_time_s: float
    kills_per_hour: float


@dataclass
class FarmingTracker:
    """Tracks actual farming performance over time with historical comparison."""

    total_zeny_gained: int = 0
    total_kills: int = 0
    total_deaths: int = 0
    total_items: int = 0
    total_potion_cost: int = 0
    total_time_seconds: float = 0.0
    session_start_time: float = 0.0
    zeny_history: list[tuple[float, int]] = field(default_factory=list)
    map_performance: dict[str, dict[str, float]] = field(default_factory=dict)
    session_history: list[SessionMetrics] = field(default_factory=list)
    current_session: SessionMetrics = field(default_factory=SessionMetrics)

    def start_session(self, map_name: str = "") -> None:
        self.session_start_time = time.time()
        self.total_zeny_gained = 0
        self.total_kills = 0
        self.total_deaths = 0
        self.total_items = 0
        self.total_potion_cost = 0
        self.current_session = SessionMetrics(
            start_time=time.time(),
            map_name=map_name,
        )

    def record_kill(self, zeny_from_drops: int = 0) -> None:
        self.total_kills += 1
        self.total_zeny_gained += zeny_from_drops
        self.current_session.kills += 1
        self.current_session.zeny_gained += zeny_from_drops

    def record_death(self) -> None:
        self.total_deaths += 1
        self.current_session.deaths += 1

    def record_item(self, count: int = 1) -> None:
        self.total_items += count
        self.current_session.items_gained += count

    def record_potion_cost(self, amount: int) -> None:
        self.total_potion_cost += amount
        self.current_session.potion_cost += amount

    def record_zeny(self, amount: int) -> None:
        self.total_zeny_gained += amount
        self.current_session.zeny_gained += amount
        self.zeny_history.append((time.time(), amount))

    def record_return_to_town(self) -> None:
        self.current_session.returns_to_town += 1

    def end_session(self) -> SessionMetrics:
        """End the current session and store it in history."""
        metrics = self.current_session
        self.session_history.append(metrics)
        # Update map performance
        if metrics.map_name:
            self._update_map_performance(metrics)
        return metrics

    def _update_map_performance(self, metrics: SessionMetrics) -> None:
        """Update rolling average for a map."""
        mn = metrics.map_name.lower()
        if mn not in self.map_performance:
            self.map_performance[mn] = {
                "net_zeny_per_hour": 0.0,
                "kills_per_hour": 0.0,
                "deaths_per_hour": 0.0,
                "sessions": 0,
            }
        perf = self.map_performance[mn]
        sessions = perf["sessions"]
        perf["net_zeny_per_hour"] = (
            perf["net_zeny_per_hour"] * sessions + metrics.net_zeny_per_hour
        ) / (sessions + 1)
        perf["kills_per_hour"] = (
            perf["kills_per_hour"] * sessions + metrics.kills_per_hour
        ) / (sessions + 1)
        perf["deaths_per_hour"] = (
            perf["deaths_per_hour"] * sessions + metrics.deaths_per_hour
        ) / (sessions + 1)
        perf["sessions"] = sessions + 1

    def get_historical_average(self, map_name: str = "") -> dict[str, float]:
        """Get historical average metrics for a map or overall."""
        if map_name:
            return self.map_performance.get(map_name.lower(), {})
        # Overall average across all maps
        if not self.session_history:
            return {}
        total_zeny = sum(s.zeny_per_hour for s in self.session_history)
        total_kills = sum(s.kills_per_hour for s in self.session_history)
        total_deaths = sum(s.deaths_per_hour for s in self.session_history)
        count = len(self.session_history)
        return {
            "avg_zeny_per_hour": total_zeny / count,
            "avg_kills_per_hour": total_kills / count,
            "avg_deaths_per_hour": total_deaths / count,
            "sessions": count,
        }

    def detect_diminishing_returns(self, current_net_zeny_hour: float, map_name: str = "") -> bool:
        """Detect if profit has dropped >30% from historical average."""
        hist = self.get_historical_average(map_name)
        avg_zeny = hist.get("avg_zeny_per_hour", 0) or hist.get("net_zeny_per_hour", 0)
        if avg_zeny <= 0 or hist.get("sessions", 0) < 2:
            return False
        return current_net_zeny_hour < avg_zeny * 0.7

    def report(self, current_map: str) -> dict[str, Any]:
        """Get current session's farming report with full metrics."""
        elapsed = time.time() - self.session_start_time
        if elapsed < 1:
            return {"zeny_per_hour": 0, "kills_per_hour": 0, "active": False}

        zeny_per_hour = (self.total_zeny_gained / elapsed) * 3600
        kills_per_hour = (self.total_kills / elapsed) * 3600
        deaths_per_hour = (self.total_deaths / elapsed) * 3600
        potion_per_hour = (self.total_potion_cost / elapsed) * 3600
        items_per_hour = (self.total_items / elapsed) * 3600
        net_zeny = zeny_per_hour - potion_per_hour

        # Diminishing returns check
        diminishing = self.detect_diminishing_returns(net_zeny, current_map)

        return {
            "map": current_map,
            "zeny_per_hour": round(zeny_per_hour),
            "net_zeny_per_hour": round(net_zeny),
            "kills_per_hour": round(kills_per_hour),
            "deaths_per_hour": round(deaths_per_hour),
            "potion_cost_per_hour": round(potion_per_hour),
            "items_per_hour": round(items_per_hour),
            "total_zeny": self.total_zeny_gained,
            "total_kills": self.total_kills,
            "total_deaths": self.total_deaths,
            "total_potion_cost": self.total_potion_cost,
            "elapsed_minutes": round(elapsed / 60, 1),
            "diminishing_returns": diminishing,
            "active": True,
        }


class ProfitabilityCalculator:
    """Calculates and compares farming profitability across maps.

    Features:
      - Market-aware pricing via ItemValueDB
      - Session tracking with full metrics
      - Kill target priority based on profit potential
      - Map comparison with top-5 ranking
      - Weight-adjusted profit accounting
      - Potion cost tracking
      - Historical session comparison
      - Diminishing returns detection
    """

    def __init__(self, db: ItemValueDB | None = None) -> None:
        self._db = db or ItemValueDB()
        self._tracker = FarmingTracker()
        self._custom_spawns: dict[str, list[tuple[str, int, int]]] = {}
        self._market_prices: dict[str, int] = {}  # Cached market prices
        self._last_market_fetch: float = 0.0
        self._market_fetch_interval: float = 300.0  # 5 min cache

    # ── Market price integration ──────────────────────────────────────

    def fetch_market_prices(self) -> dict[str, int]:
        """Fetch current market prices (NPC buy/sell, player vending).

        Caches for 5 minutes to avoid excessive queries.
        """
        now = time.time()
        if now - self._last_market_fetch < self._market_fetch_interval:
            return self._market_prices

        prices: dict[str, int] = {}
        try:
            # Try to get prices from the item value DB
            for monster_name, drops in _MONSTER_DROPS.items():
                for item_name, _, _ in drops:
                    if item_name not in prices:
                        db_price = self._db.get_best_price(item_name)
                        if db_price > 0:
                            prices[item_name] = db_price

            # Add potion costs
            for potion_name, cost in _POTION_COSTS.items():
                prices[potion_name] = cost

        except Exception as exc:
            logger.warning("Failed to fetch market prices: %s", exc)

        self._market_prices = prices
        self._last_market_fetch = now
        logger.debug("Fetched %d market prices", len(prices))
        return prices

    def get_item_market_price(self, item_name: str) -> int:
        """Get the current market price for an item."""
        prices = self.fetch_market_prices()
        if item_name in prices:
            return prices[item_name]
        db_price = self._db.get_best_price(item_name)
        if db_price > 0:
            return db_price
        return 0

    # ── Map spawn management ──────────────────────────────────────

    def set_spawns(self, map_name: str, spawns: list[tuple[str, int, int]]) -> None:
        self._custom_spawns[map_name.lower()] = spawns

    def get_spawns(self, map_name: str) -> list[tuple[str, int, int]]:
        m = map_name.lower()
        if m in self._custom_spawns:
            return self._custom_spawns[m]
        return _MAP_SPAWNS.get(m, [])

    # ── Kill target priority ──────────────────────────────────────────

    def calculate_kill_target_priority(
        self,
        monster_name: str,
        kill_time_s: float = 5.0,
    ) -> KillTargetPriority | None:
        """Calculate profit priority for a monster.

        Priority = (card_drop_value * drop_rate) / kill_time
        Higher score = better target.
        """
        mn = monster_name.lower().strip()
        drops = _MONSTER_DROPS.get(mn, [])
        if not drops:
            return None

        # Find card drop
        card_value = 0
        card_drop_rate = 0.0
        zeny_per_kill = 0.0
        for item_name, drop_rate, est_value in drops:
            if "card" in item_name:
                card_value = self.get_item_market_price(item_name) or est_value
                card_drop_rate = drop_rate
            else:
                item_price = self.get_item_market_price(item_name) or est_value
                zeny_per_kill += drop_rate * item_price

        # Profit score = expected card value per kill / kill time
        expected_card_value = card_value * card_drop_rate
        profit_score = (zeny_per_kill + expected_card_value) / max(0.1, kill_time_s)
        kills_per_hour = 3600.0 / max(0.1, kill_time_s)

        return KillTargetPriority(
            monster_name=monster_name,
            profit_score=round(profit_score, 2),
            zeny_per_kill=round(zeny_per_kill),
            card_value=card_value,
            drop_rate=card_drop_rate,
            kill_time_s=kill_time_s,
            kills_per_hour=round(kills_per_hour),
        )

    def rank_kill_targets(
        self,
        monster_names: list[str],
        base_kill_time_s: float = 5.0,
    ) -> list[KillTargetPriority]:
        """Rank monsters by profit priority."""
        priorities: list[KillTargetPriority] = []
        for name in monster_names:
            priority = self.calculate_kill_target_priority(name, base_kill_time_s)
            if priority:
                priorities.append(priority)
        priorities.sort(key=lambda p: -p.profit_score)
        return priorities

    # ── Core calculation ──────────────────────────────────────────

    def calculate_map_profitability(
        self,
        map_name: str,
        kills_per_minute: float = 15.0,
        base_level: int = 1,
        job_name: str = "novice",
        str_stat: int = 1,
        potion_cost_per_minute: float = 0.0,
    ) -> MapProfitability:
        """Calculate expected zeny/hour for a given map with weight-adjusted profit.

        Args:
            map_name: Map identifier.
            kills_per_minute: Estimated kills per minute.
            base_level: Player's base level.
            job_name: Player's class.
            str_stat: STR stat for weight calculation.
            potion_cost_per_minute: Estimated potion cost per minute.

        Returns:
            MapProfitability dataclass.
        """
        spawns = self.get_spawns(map_name)
        if not spawns:
            return MapProfitability(
                map_name=map_name,
                zeny_per_hour=0.0,
                expected_zeny_per_hour=0.0,
                net_zeny_per_hour=0.0,
                zeny_per_kill=0.0,
                kills_per_hour=0.0,
                potion_cost_per_hour=0.0,
                monsters=[],
                confidence=0.0,
                risks=["No spawn data for this map"],
            )

        kills_per_hour = kills_per_minute * 60.0
        potion_cost_per_hour = potion_cost_per_minute * 60.0

        total_zeny_per_hour = 0.0
        total_kills_per_hour = 0.0
        monster_breakdown: list[dict[str, Any]] = []
        risks: list[str] = []

        # Calculate per-monster contribution
        for monster_name, spawn_count, respawn_ms in spawns:
            mn = monster_name.lower().strip()
            drops = _MONSTER_DROPS.get(mn, [])

            if not drops:
                zeny_per_kill = estimate_base_zeny_drop(mn)
            else:
                zeny_per_kill = self._calc_expected_drop_value(mn, drops)

            total_spawns = sum(s[1] for s in spawns)
            spawn_proportion = spawn_count / max(1, total_spawns)
            monster_kph = kills_per_hour * spawn_proportion

            if respawn_ms > 0:
                max_kph = spawn_count * (3600.0 / max(1, respawn_ms / 1000.0))
                monster_kph = min(monster_kph, max_kph)

            monster_zeny_hour = zeny_per_kill * monster_kph
            total_zeny_per_hour += monster_zeny_hour
            total_kills_per_hour += monster_kph

            monster_breakdown.append({
                "monster": monster_name,
                "spawn_count": spawn_count,
                "zeny_per_kill": round(zeny_per_kill),
                "kills_per_hour": round(monster_kph),
                "zeny_per_hour": round(monster_zeny_hour),
                "spawn_proportion": round(spawn_proportion, 2),
            })

        # Weight-adjusted profit: account for returns to town
        max_weight = _MAX_WEIGHT_BASE + str_stat * _WEIGHT_PER_STR
        # Estimate average item weight (rough: 10-50 per item)
        avg_item_weight = 30
        items_per_hour = total_kills_per_hour * 0.5  # ~50% drop rate
        weight_full_time_minutes = (max_weight / max(1, avg_item_weight * items_per_hour / 60.0))
        returns_per_hour = 60.0 / max(1, weight_full_time_minutes)
        return_cost_per_trip = 200  # Estimated warp/fly wing cost
        return_cost_per_hour = returns_per_hour * return_cost_per_trip

        # Net profit
        net_zeny_per_hour = total_zeny_per_hour - potion_cost_per_hour - return_cost_per_hour

        # Historical adjustment
        historical = self._tracker.map_performance.get(map_name.lower(), {})
        historical_sessions = historical.get("sessions", 0)
        if historical_sessions > 0:
            actual_net_zeny = historical.get("net_zeny_per_hour", 0)
            blend_weight = min(0.7, 0.3 + historical_sessions * 0.1)
            adjusted = net_zeny_per_hour * (1 - blend_weight) + actual_net_zeny * blend_weight
            expected_zeny_per_hour = adjusted
            confidence = min(1.0, 0.5 + historical_sessions * 0.1)
        else:
            expected_zeny_per_hour = net_zeny_per_hour
            confidence = 0.5

        # Risk assessment
        if kills_per_minute < 10:
            risks.append("Low kill rate — consider switching to easier mobs")
        if confidence < 0.6:
            risks.append("Limited data — estimate may be inaccurate")
        if potion_cost_per_hour > total_zeny_per_hour * 0.5:
            risks.append("High potion cost — profit margin is thin")
        if returns_per_hour > 3:
            risks.append("Frequent returns to town — consider lighter gear or higher STR")

        return MapProfitability(
            map_name=map_name,
            zeny_per_hour=round(total_zeny_per_hour),
            expected_zeny_per_hour=round(expected_zeny_per_hour),
            net_zeny_per_hour=round(net_zeny_per_hour),
            zeny_per_kill=round(total_zeny_per_hour / max(1, total_kills_per_hour)),
            kills_per_hour=round(total_kills_per_hour),
            potion_cost_per_hour=round(potion_cost_per_hour),
            monsters=monster_breakdown,
            confidence=round(confidence, 2),
            risks=risks,
        )

    def _calc_expected_drop_value(
        self,
        monster_name: str,
        drops: list[tuple[str, float, int]],
    ) -> float:
        """Calculate expected zeny per kill from drops using market prices."""
        total = 0.0
        seen_items: set[str] = set()

        for item_name, drop_rate, est_value in drops:
            if item_name in seen_items:
                continue
            seen_items.add(item_name)

            # Use market price if available, else estimated value
            market_price = self.get_item_market_price(item_name)
            item_value = market_price if market_price > 0 else est_value
            expected = drop_rate * item_value
            total += expected

        return total

    # ── Map comparison ─────────────────────────────────────────────

    def compare_maps(
        self,
        map_names: list[str],
        kills_per_minute: float = 15.0,
        base_level: int = 1,
        job_name: str = "novice",
        str_stat: int = 1,
        potion_cost_per_minute: float = 0.0,
    ) -> list[MapProfitability]:
        """Compare profitability across multiple maps.

        Returns:
            Maps sorted by net_zeny_per_hour descending, top 5.
        """
        results: list[MapProfitability] = []
        for map_name in map_names:
            try:
                result = self.calculate_map_profitability(
                    map_name, kills_per_minute, base_level, job_name,
                    str_stat, potion_cost_per_minute,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Failed to calculate profitability for %s: %s", map_name, exc)

        results.sort(key=lambda r: r.net_zeny_per_hour, reverse=True)

        if results:
            results[0].recommended = True

        # Return top 5
        return results[:5]

    def recommend_best_map(
        self,
        map_names: list[str],
        kills_per_minute: float = 15.0,
        base_level: int = 1,
        job_name: str = "novice",
        str_stat: int = 1,
        potion_cost_per_minute: float = 0.0,
        min_confidence: float = 0.3,
    ) -> MapProfitability | None:
        """Get the single best map recommendation."""
        results = self.compare_maps(
            map_names, kills_per_minute, base_level, job_name,
            str_stat, potion_cost_per_minute,
        )
        qualified = [r for r in results if r.confidence >= min_confidence]
        if not qualified:
            return results[0] if results else None
        return qualified[0]

    # ── Farming tracker ───────────────────────────────────────────

    def start_tracking(self, map_name: str = "") -> None:
        """Start a farming session tracker."""
        self._tracker.start_session(map_name)

    def record_kill(self, zeny_from_drops: int = 0) -> None:
        self._tracker.record_kill(zeny_from_drops)

    def record_death(self) -> None:
        self._tracker.record_death()

    def record_item(self, count: int = 1) -> None:
        self._tracker.record_item(count)

    def record_potion_cost(self, amount: int) -> None:
        self._tracker.record_potion_cost(amount)

    def record_zeny(self, amount: int) -> None:
        self._tracker.record_zeny(amount)

    def record_return_to_town(self) -> None:
        self._tracker.record_return_to_town()

    def end_session(self) -> SessionMetrics:
        """End the current farming session and return metrics."""
        return self._tracker.end_session()

    def get_tracking_report(self, current_map: str = "") -> dict[str, Any]:
        """Get current session's farming report."""
        return self._tracker.report(current_map)

    def get_session_comparison(self, current_map: str = "") -> dict[str, Any]:
        """Compare current session metrics to historical average."""
        current = self._tracker.current_session
        hist = self._tracker.get_historical_average(current_map)

        if not hist or hist.get("sessions", 0) < 1:
            return {"comparison_available": False}

        current_zeny = current.net_zeny_per_hour
        avg_zeny = hist.get("avg_zeny_per_hour", 0) or hist.get("net_zeny_per_hour", 0)

        return {
            "comparison_available": True,
            "current_zeny_per_hour": round(current_zeny),
            "historical_avg_zeny_per_hour": round(avg_zeny),
            "difference_pct": round(
                ((current_zeny - avg_zeny) / max(1, avg_zeny)) * 100, 1
            ),
            "current_kills_per_hour": round(current.kills_per_hour),
            "historical_avg_kills_per_hour": round(
                hist.get("avg_kills_per_hour", 0), 1
            ),
            "current_deaths_per_hour": round(current.deaths_per_hour, 2),
            "historical_avg_deaths_per_hour": round(
                hist.get("avg_deaths_per_hour", 0), 2
            ),
            "diminishing_returns": self._tracker.detect_diminishing_returns(
                current_zeny, current_map
            ),
            "sessions_tracked": hist.get("sessions", 0),
        }

    def get_diminishing_returns_suggestion(self, current_map: str) -> str | None:
        """If diminishing returns detected, suggest a map change."""
        current = self._tracker.current_session
        if self._tracker.detect_diminishing_returns(current.net_zeny_per_hour, current_map):
            return (
                f"Profit on {current_map} has dropped >30% from historical average. "
                f"Consider switching maps."
            )
        return None


def estimate_base_zeny_drop(monster_name: str) -> int:
    """Estimate the base zeny drop for a monster."""
    tier_map: dict[str, tuple[int, int]] = {
        "poring": (10, 20),
        "lunatic": (10, 20),
        "pupa": (5, 10),
        "thief_bug": (15, 25),
        "thief_bug_egg": (5, 10),
        "familiar": (20, 40),
        "zombie": (30, 50),
        "skeleton": (25, 45),
        "poporing": (20, 40),
        "creamy": (15, 30),
        "rocker": (20, 35),
        "spore": (15, 30),
        "drainliar": (30, 50),
        "hunter_fly": (40, 70),
        "munak": (40, 80),
        "bongun": (40, 80),
        "ghoul": (60, 120),
        "orc_warrior": (50, 100),
        "orc_zombie": (40, 75),
        "steel_chonchon": (30, 60),
        "plankton": (5, 15),
        "marina": (10, 25),
        "kukre": (20, 40),
        "hydra": (30, 50),
        "vadon": (40, 80),
        "marine_sphere": (20, 40),
        "soldier_skeleton": (50, 100),
    }

    mn = monster_name.lower().strip()
    if mn in tier_map:
        low, high = tier_map[mn]
        return (low + high) // 2

    return 30
