"""
Farming Profitability Calculator — find the best zeny/hour farming spots.

Given a map's monster spawns and the player's capabilities, calculates:
  - Expected zeny/hour = sum((drop_rate × item_value) for each drop)
  - Weighted by monster spawn density and kill speed
  - Compares across maps to recommend best farm spot
  - Tracks actual zeny/hour and adjusts estimates over time
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.economy.database import ItemValueDB

logger = logging.getLogger(__name__)

# ── rAthena drop rate constants ──
# Pre-renewal (classic) drop rates
COMMON_DROP_RATE = 0.55     # 55% — common drops (Jellopy, etc.)
UNCOMMON_DROP_RATE = 0.20   # 20% — uncommon drops
RARE_DROP_RATE = 0.05       # 5%  — rare drops (cards are 0.01%)
CARD_DROP_RATE = 0.0001     # 0.01% — cards
MVP_CARD_DROP_RATE = 0.00001  # 0.001% — MVP cards

# ── Monster drop tables (keyed by monster name) ──
# Structure: {monster: [(item_name, drop_rate, estimated_value)]}
# Based on rAthena pre-re mob_db data
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

# ── Map spawn data (monster, count, respawn_ms) ──
# From heuristic_service.py AdaptiveDataStore.map_spawns
_MAP_SPAWNS: dict[str, list[tuple[str, int, int]]] = {
    "pay_dun00": [("Familiar", 15, 0), ("Zombie", 20, 0), ("Skeleton", 35, 0), ("Poporing", 15, 0)],
    "pay_dun01": [("Munak", 20, 0), ("Bongun", 15, 0), ("Ghoul", 10, 0), ("Skeleton", 30, 0)],
    "gef_dun00": [("Hunter Fly", 30, 60000), ("Poporing", 15, 0), ("Poison Spore", 25, 0)],
    "orcsdun01": [("Steel Chonchon", 10, 0), ("Familiar", 15, 0), ("Drainliar", 5, 0), ("Orc Zombie", 80, 60000)],
    "iz_dun00": [("Plankton", 65, 0), ("Marina", 45, 0), ("Kukre", 15, 0), ("Hydra", 15, 0), ("Vadon", 15, 0)],
    "prt_fild05": [("Poring", 70, 0), ("Thief Bug Egg", 20, 0), ("Lunatic", 30, 0), ("Pupa", 30, 0), ("Thief Bug", 10, 0)],
    "prt_fild04": [("Rocker", 70, 0), ("Creamy", 40, 0), ("Pupa", 10, 0), ("Poring", 30, 0)],
}


@dataclass
class MapProfitability:
    """Profitability assessment for a single map."""
    map_name: str
    zeny_per_hour: float
    expected_zeny_per_hour: float
    zeny_per_kill: float
    kills_per_hour: float
    monsters: list[dict[str, Any]]  # per-monster breakdown
    confidence: float  # 0.0 (low) to 1.0 (high)
    risks: list[str]
    recommended: bool = False


@dataclass
class FarmingTracker:
    """Tracks actual farming performance over time."""
    total_zeny_gained: int = 0
    total_kills: int = 0
    total_time_seconds: float = 0.0
    session_start_time: float = 0.0
    zeny_history: list[tuple[float, int]] = field(default_factory=list)  # (timestamp, zeny_delta)
    map_performance: dict[str, dict[str, float]] = field(default_factory=dict)

    def start_session(self) -> None:
        self.session_start_time = time.time()
        self.total_zeny_gained = 0
        self.total_kills = 0

    def record_kill(self, zeny_from_drops: int = 0) -> None:
        self.total_kills += 1
        self.total_zeny_gained += zeny_from_drops

    def record_zeny(self, amount: int) -> None:
        self.total_zeny_gained += amount
        self.zeny_history.append((time.time(), amount))

    def report(self, current_map: str) -> dict[str, Any]:
        elapsed = time.time() - self.session_start_time
        if elapsed < 1:
            return {"zeny_per_hour": 0, "kills_per_hour": 0, "active": False}

        zeny_per_hour = (self.total_zeny_gained / elapsed) * 3600
        kills_per_hour = (self.total_kills / elapsed) * 3600

        # Update map performance tracking
        if current_map not in self.map_performance:
            self.map_performance[current_map] = {
                "zeny_per_hour": 0,
                "kills_per_hour": 0,
                "sessions": 0,
            }
        perf = self.map_performance[current_map]
        # Rolling average
        sessions = perf["sessions"]
        perf["zeny_per_hour"] = (perf["zeny_per_hour"] * sessions + zeny_per_hour) / (sessions + 1)
        perf["kills_per_hour"] = (perf["kills_per_hour"] * sessions + kills_per_hour) / (sessions + 1)
        perf["sessions"] = sessions + 1

        return {
            "map": current_map,
            "zeny_per_hour": round(zeny_per_hour),
            "kills_per_hour": round(kills_per_hour),
            "total_zeny": self.total_zeny_gained,
            "total_kills": self.total_kills,
            "elapsed_minutes": round(elapsed / 60, 1),
            "active": True,
        }


class ProfitabilityCalculator:
    """Calculates and compares farming profitability across maps.

    Recommends the best map for zeny farming based on:
      - Monster drops × spawn density
      - Kill speed (player DPS vs monster HP/def)
      - Potion/food cost per kill
      - Card drop expected value
      - Historical performance tracking
    """

    def __init__(self, db: ItemValueDB | None = None) -> None:
        self._db = db or ItemValueDB()
        self._tracker = FarmingTracker()
        # Custom spawn data (can override defaults)
        self._custom_spawns: dict[str, list[tuple[str, int, int]]] = {}

    # ── Map spawn management ──────────────────────────────────────

    def set_spawns(self, map_name: str, spawns: list[tuple[str, int, int]]) -> None:
        """Override spawn data for a map."""
        self._custom_spawns[map_name.lower()] = spawns

    def get_spawns(self, map_name: str) -> list[tuple[str, int, int]]:
        """Get spawn data for a map (custom > default > empty)."""
        m = map_name.lower()
        if m in self._custom_spawns:
            return self._custom_spawns[m]
        return _MAP_SPAWNS.get(m, [])

    # ── Core calculation ──────────────────────────────────────────

    def calculate_map_profitability(
        self,
        map_name: str,
        kills_per_minute: float = 15.0,
        base_level: int = 1,
        job_name: str = "novice",
    ) -> MapProfitability:
        """Calculate expected zeny/hour for a given map.

        Args:
            map_name: Map identifier (e.g. 'pay_dun00').
            kills_per_minute: Estimated kills per minute based on player DPS.
            base_level: Player's base level (affects drop rate bonuses).
            job_name: Player's class.

        Returns:
            MapProfitability dataclass.
        """
        spawns = self.get_spawns(map_name)
        if not spawns:
            return MapProfitability(
                map_name=map_name,
                zeny_per_hour=0.0,
                expected_zeny_per_hour=0.0,
                zeny_per_kill=0.0,
                kills_per_hour=0.0,
                monsters=[],
                confidence=0.0,
                risks=["No spawn data for this map"],
            )

        # Effective kills per hour (accounting for respawn + travel)
        kills_per_hour = kills_per_minute * 60.0

        total_zeny_per_hour = 0.0
        total_kills_per_hour = 0.0
        monster_breakdown: list[dict[str, Any]] = []
        risks: list[str] = []

        # Calculate per-monster contribution
        for monster_name, spawn_count, respawn_ms in spawns:
            mn = monster_name.lower().strip()
            drops = _MONSTER_DROPS.get(mn, [])

            if not drops:
                # Monster with no drop data — estimate base zeny drop
                zeny_per_kill = estimate_base_zeny_drop(mn)
            else:
                zeny_per_kill = self._calc_expected_drop_value(mn, drops)

            # Spawn proportion: how many of this monster's kills per hour
            total_spawns = sum(s[1] for s in spawns)
            spawn_proportion = spawn_count / max(1, total_spawns)
            monster_kph = kills_per_hour * spawn_proportion

            # Account for respawn time limiting factor
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

        # Historical adjustment: if we have tracked data for this map,
        # blend with prediction (weight actual more as sessions increase)
        historical = self._tracker.map_performance.get(map_name.lower(), {})
        historical_sessions = historical.get("sessions", 0)
        if historical_sessions > 0:
            actual_zeny_hour = historical.get("zeny_per_hour", 0)
            # Blend: 30% actual for 1 session, increasing to 70% actual
            blend_weight = min(0.7, 0.3 + historical_sessions * 0.1)
            adjusted = total_zeny_per_hour * (1 - blend_weight) + actual_zeny_hour * blend_weight
            expected_zeny_per_hour = adjusted
            confidence = min(1.0, 0.5 + historical_sessions * 0.1)
        else:
            expected_zeny_per_hour = total_zeny_per_hour
            confidence = 0.5  # moderate confidence without historical data

        # Risk assessment
        if kills_per_minute < 10:
            risks.append("Low kill rate — consider switching to easier mobs")
        if confidence < 0.6:
            risks.append("Limited data — estimate may be inaccurate")

        return MapProfitability(
            map_name=map_name,
            zeny_per_hour=round(total_zeny_per_hour),
            expected_zeny_per_hour=round(expected_zeny_per_hour),
            zeny_per_kill=round(total_zeny_per_hour / max(1, total_kills_per_hour)),
            kills_per_hour=round(total_kills_per_hour),
            monsters=monster_breakdown,
            confidence=round(confidence, 2),
            risks=risks,
        )

    def _calc_expected_drop_value(
        self,
        monster_name: str,
        drops: list[tuple[str, float, int]],
    ) -> float:
        """Calculate expected zeny per kill from drops.

        Sums (drop_rate × item_value) for all possible drops.
        """
        total = 0.0
        seen_items: set[str] = set()

        for item_name, drop_rate, _ in drops:
            if item_name in seen_items:
                continue
            seen_items.add(item_name)

            # Get actual item value from DB if available
            db_value = self._db.get_best_price(item_name)
            if db_value > 0:
                item_value = db_value
            else:
                # Use the estimated value from the drop table
                item_value = 0
                for _in, _dr, _iv in drops:
                    if _in == item_name:
                        item_value = _iv
                        break

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
    ) -> list[MapProfitability]:
        """Compare profitability across multiple maps.

        Returns:
            Maps sorted by expected_zeny_per_hour descending.
        """
        results: list[MapProfitability] = []
        for map_name in map_names:
            try:
                result = self.calculate_map_profitability(
                    map_name, kills_per_minute, base_level, job_name,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Failed to calculate profitability for %s: %s", map_name, exc)

        # Sort by expected zeny/hour descending
        results.sort(key=lambda r: r.expected_zeny_per_hour, reverse=True)

        # Mark the best one
        if results:
            results[0].recommended = True

        return results

    def recommend_best_map(
        self,
        map_names: list[str],
        kills_per_minute: float = 15.0,
        base_level: int = 1,
        job_name: str = "novice",
        min_confidence: float = 0.3,
    ) -> MapProfitability | None:
        """Get the single best map recommendation.

        Args:
            map_names: Candidate maps.
            kills_per_minute: Estimated kills per minute.
            base_level: Player base level.
            job_name: Player class.
            min_confidence: Minimum confidence threshold.

        Returns:
            Best MapProfitability or None if none meet threshold.
        """
        results = self.compare_maps(map_names, kills_per_minute, base_level, job_name)
        qualified = [r for r in results if r.confidence >= min_confidence]
        if not qualified:
            return results[0] if results else None
        return qualified[0]

    # ── Farming tracker ───────────────────────────────────────────

    def start_tracking(self) -> None:
        """Start a farming session tracker."""
        self._tracker.start_session()

    def record_kill(self, zeny_from_drops: int = 0) -> None:
        """Record one kill with estimated drop value."""
        self._tracker.record_kill(zeny_from_drops)

    def record_zeny(self, amount: int) -> None:
        """Record zeny gained (from selling, drops, etc.)."""
        self._tracker.record_zeny(amount)

    def get_tracking_report(self, current_map: str = "") -> dict[str, Any]:
        """Get current session's farming report."""
        return self._tracker.report(current_map)


def estimate_base_zeny_drop(monster_name: str) -> int:
    """Estimate the base zeny drop for a monster.

    Uses monster level tier to approximate:
      - < 20: 10-30z
      - 20-40: 30-80z
      - 40-60: 80-200z
      - 60-80: 200-500z
      - 80+: 500-2000z
    """
    # Rough mapping based on common RO monsters
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

    return 30  # default for unknown monsters
