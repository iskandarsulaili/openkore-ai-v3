"""Server Adaptation Module — detects EXP/drop rates, server type, and adjusts strategy.

Every rAthena server is different: different EXP rates, drop rates, mechanics
(renewal vs pre-renewal), custom items, custom WoE schedules. The bot needs
to detect these automatically and adapt.

This module provides:

- EXPObserver:     Records EXP gained per kill, smooths over multiple kills,
                   returns estimated EXP multiplier vs rAthena reference.

- DropObserver:    Records observed drops over time, compares to expected
                   drop probabilities (from rAthena DB), estimates server
                   drop multiplier, tracks per-monster rates.

- ServerAdapter:   High-level facade that uses EXPObserver + DropObserver
                   to estimate server rates, detect pre-renewal vs renewal
                   (via DEF formula matching), flag custom mechanics, and
                   expose confidence-weighted strategy guidance.

- StrategyAdjuster:Takes detected rates and adjusts domain modules:
                   cold-start, item valuation, combat formulas, etc.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Constants — rAthena reference values
# ═══════════════════════════════════════════════════════════════

# Reference base EXP formula for a monster of a given level.
# rAthena's default mob_db values vary per monster, but we approximate
# the typical "named monster" EXP curve for detection purposes.
# Returns expected base EXP for a non-boss monster of level L on a 1x server.
def reference_base_exp(monster_level: int) -> int:
    """Expected base EXP for a level L monster on a 1x pre-renewal server.

    Approximates rAthena mob_db default curve (fits within ~20% of most
    standard monsters between Lv1-99). Used only for rate detection, so
    absolute accuracy per monster is less important than stable averaging.
    """
    # Quadratic fit to typical rAthena EXP curve: ~L²/3 + 8L
    return int(monster_level * monster_level / 3 + 8 * monster_level + 5)


# Reference base drop rates for common categories on a 1x server.
# Values from rAthena db: (drop_rate1 .. drop_rateN in mob_db).
REFERENCE_DROP_RATES: dict[str, float] = {
    "common":    0.55,   # ~55% common drop (stems, feathers, etc.)
    "medium":    0.35,   # ~35% moderate drop (ores, elemental stones)
    "rare":      0.05,   # ~5%  rare (equipment, cards-like at 0.01% actually)
    "card":      0.0001, # 0.01% for MVP cards, 0.01%-0.1% for normal cards
    "equipment": 0.02,   # 2% for equipment drops
}


# ═══════════════════════════════════════════════════════════════
# Server Type
# ═══════════════════════════════════════════════════════════════

class ServerType(Enum):
    """Detected server mechanics type."""
    UNKNOWN = auto()
    PRE_RENEWAL = auto()
    RENEWAL = auto()

    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")


class ServerRateCategory(Enum):
    """Categorical rate tier for strategy adjustment."""
    UNKNOWN = auto()
    LOW_RATE = auto()        # 1x-5x
    MEDIUM_RATE = auto()     # 5x-25x
    HIGH_RATE = auto()       # 25x-100x
    EXTREME_RATE = auto()    # 100x+

    def is_high(self) -> bool:
        return self in (self.HIGH_RATE, self.EXTREME_RATE)

    def is_low(self) -> bool:
        return self == self.LOW_RATE

    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")


# ═══════════════════════════════════════════════════════════════
# Damage formula helpers (pre-renewal vs renewal)
# ═══════════════════════════════════════════════════════════════

def pre_renewal_damage_taken(defence: int) -> float:
    """Fraction of damage that gets through pre-renewal DEF.

    Pre-renewal formula:  damage_mult = (DEF × 0.5 + 2) / (DEF + 2)
    - At DEF=0:  1.000
    - At DEF=20: 0.545
    - At DEF=50: 0.519
    - Capped fast — DEF beyond ~20 has very diminishing returns.
    """
    return (defence * 0.5 + 2.0) / (defence + 2.0)


def renewal_damage_taken(defence: int) -> float:
    """Fraction of damage that gets through renewal DEF.

    Renewal formula:  damage_mult = DEF / (DEF + 300)
    - At DEF=0:   0.000  (100% blocked → damage = 0?) 
      No — renewal uses a different meaning. For comparison we treat this as
      the fraction of damage that is _reduced_, so the intake is:
        damage_mult = 300 / (DEF + 300)
    - At DEF=0:   1.000
    - At DEF=300: 0.500
    - At DEF=600: 0.333
    """
    return 300.0 / (defence + 300.0)


# ═══════════════════════════════════════════════════════════════
# EXPObserver
# ═══════════════════════════════════════════════════════════════

@dataclass
class EXPObserver:
    """Observes EXP gained per kill and estimates the server EXP multiplier.

    Each kill reports:
      - monster_level
      - exp_gained (base_exp only — job exp is ignored for rate detection)

    The observer:
      1. Divides exp_gained by reference_base_exp(monster_level) → per-kill mult
      2. Smooths with an exponential moving average for noise rejection
      3. Returns estimated multiplier and confidence level

    Confidence grows as sqrt(num_observations), capped at 0.95.
    """

    # Exponential moving average weight (higher = more weight on recent kills)
    smoothing_alpha: float = 0.15
    # Minimum observations before returning a non-default estimate
    min_observations: int = 5

    # ── internal state ──
    _mult_estimate: float = 1.0   # current EMA estimate
    _observation_count: int = 0
    _total_raw_mult: float = 0.0  # simple sum for alternative avg
    _recent_multipliers: list[float] = field(default_factory=list, repr=False)

    # Per-level tracking for accuracy diagnostics
    _per_level: dict[int, list[float]] = field(default_factory=dict, repr=False)

    def observe_kill(self, monster_level: int, base_exp_gained: int) -> None:
        """Record a kill observation.

        Args:
            monster_level: The monster's level (from monster DB lookup).
            base_exp_gained: The base experience actually received.
        """
        expected = reference_base_exp(monster_level)
        if expected <= 0:
            logger.warning("Expected EXP <= 0 for Lv%d monster, skipping", monster_level)
            return
        if base_exp_gained <= 0:
            return

        raw_mult = base_exp_gained / expected
        self._observation_count += 1
        self._total_raw_mult += raw_mult

        # EMA update
        if self._observation_count == 1:
            self._mult_estimate = raw_mult
        else:
            self._mult_estimate = (
                self.smoothing_alpha * raw_mult
                + (1 - self.smoothing_alpha) * self._mult_estimate
            )

        # Track per-level
        self._per_level.setdefault(monster_level, []).append(raw_mult)

        # Keep recent sample for stddev
        self._recent_multipliers.append(raw_mult)
        if len(self._recent_multipliers) > 100:
            self._recent_multipliers.pop(0)

    def get_mult_estimate(self) -> float:
        """Current best estimate of the server's base EXP multiplier."""
        return self._mult_estimate

    def confidence(self) -> float:
        """Confidence in the estimate (0.0 — 1.0).

        Grows with observations. Accounts for per-level consistency:
        if observations across many levels agree, confidence is higher.
        """
        if self._observation_count < self.min_observations:
            return 0.0

        # Base from count
        count_factor = math.sqrt(self._observation_count / 20.0)

        # Consistency bonus: lower stddev = higher confidence
        if len(self._recent_multipliers) >= 5:
            mean = sum(self._recent_multipliers) / len(self._recent_multipliers)
            variance = sum((x - mean) ** 2 for x in self._recent_multipliers) / len(self._recent_multipliers)
            stddev = math.sqrt(variance)
            # Normalize: coefficient of variation < 0.3 is very consistent
            cv = stddev / max(mean, 0.01)
            consistency = max(0.0, 1.0 - cv)
        else:
            consistency = 0.5

        confidence = min(0.95, count_factor * 0.7 + consistency * 0.3)
        return max(0.0, confidence)

    def observation_count(self) -> int:
        return self._observation_count

    def standard_deviation(self) -> float:
        """Stddev of recent multipliers — lower means more stable estimate."""
        if len(self._recent_multipliers) < 2:
            return 0.0
        mean = sum(self._recent_multipliers) / len(self._recent_multipliers)
        variance = sum((x - mean) ** 2 for x in self._recent_multipliers) / len(self._recent_multipliers)
        return math.sqrt(variance)

    def per_level_multipliers(self) -> dict[int, float]:
        """Average multiplier per monster level (for diagnostics)."""
        return {
            lv: sum(vals) / len(vals)
            for lv, vals in self._per_level.items()
        }

    def summary(self) -> dict[str, Any]:
        return {
            "estimated_mult": round(self._mult_estimate, 2),
            "confidence": round(self.confidence(), 3),
            "observations": self._observation_count,
            "stddev": round(self.standard_deviation(), 3),
            "levels_seen": sorted(self._per_level.keys()),
        }


# ═══════════════════════════════════════════════════════════════
# DropObserver
# ═══════════════════════════════════════════════════════════════

@dataclass
class DropObservation:
    """A single drop observation.

    - monster_name:    name of the monster killed
    - item_name:       name of the item dropped
    - category:        rarity category ('common', 'medium', 'rare', 'card', 'equipment')
    - expected_rate:   the 1x reference drop rate for this item
    """

    monster_name: str
    item_name: str
    category: str = "common"
    expected_rate: float = 0.55


@dataclass
class DropObserver:
    """Records observed drops and estimates the server drop multiplier.

    For each monster type tracked:
      - kills: number killed
      - drops: dict of item_name → count dropped

    The observer estimates the per-monster and aggregate drop multiplier
    by comparing observed drop rates to the reference 1x rates.
    """

    # ── state ──
    _monster_kills: dict[str, int] = field(default_factory=dict)
    _monster_drops: dict[str, dict[str, int]] = field(default_factory=dict)
    # Maps monster_name -> item_name -> expected_rate for rate tracking
    _monster_item_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    _drop_mult_estimate: float = 1.0
    _observation_count: int = 0
    _recent_multipliers: list[float] = field(default_factory=list, repr=False)

    def record_kill(self, monster_name: str) -> None:
        """Record that a monster was killed (no drops)."""
        self._monster_kills[monster_name] = self._monster_kills.get(monster_name, 0) + 1

    def record_drop(self, monster_name: str, item_name: str,
                    category: str = "common",
                    expected_rate: float | None = None) -> None:
        """Record a drop from a monster kill.

        IMPORTANT: record_kill() must be called first for each kill event.
        This method only records the item drop itself.

        Args:
            monster_name: Monster name.
            item_name: Item that dropped.
            category: Rarity category key (from REFERENCE_DROP_RATES).
            expected_rate: Override the 1x reference rate explicitly.
        """
        if expected_rate is None:
            expected_rate = REFERENCE_DROP_RATES.get(category, 0.55)

        if monster_name not in self._monster_drops:
            self._monster_drops[monster_name] = {}
            self._monster_item_rates[monster_name] = {}
        drops = self._monster_drops[monster_name]
        drops[item_name] = drops.get(item_name, 0) + 1
        # Store expected rate for this item (first registration wins to avoid
        # rate inflation from counting the same item multiple times)
        if item_name not in self._monster_item_rates[monster_name]:
            self._monster_item_rates[monster_name][item_name] = expected_rate

    def per_monster_drop_multiplier(self, monster_name: str) -> float | None:
        """Estimated drop multiplier for a single monster type.

        Uses the most-observed item per monster for stability.
        Returns None if insufficient data.
        """
        kills = self._monster_kills.get(monster_name, 0)
        drops = self._monster_drops.get(monster_name, {})
        rates = self._monster_item_rates.get(monster_name, {})

        if kills < 3 or not drops:
            return None

        # Use the item with the most observed drops
        best_item = max(drops, key=lambda k: drops[k])
        drop_count = drops[best_item]
        obs_rate = drop_count / kills

        expected_rate = rates.get(best_item, REFERENCE_DROP_RATES.get("common", 0.55))
        return obs_rate / expected_rate

    def update_aggregate_estimate(self) -> float:
        """Recompute the aggregate drop multiplier across all observations.

        Uses weighted average expected rate from all observed items.
        Returns the updated estimate.
        """
        total_kills = sum(self._monster_kills.values())
        total_drops = sum(
            sum(drops.values()) for drops in self._monster_drops.values()
        )
        if total_kills == 0:
            return self._drop_mult_estimate

        # Compute weighted average expected rate across all observed drops
        total_expected_weight = 0.0
        for monster_name, drops in self._monster_drops.items():
            rates = self._monster_item_rates.get(monster_name, {})
            for item_name, count in drops.items():
                expected_rate = rates.get(item_name, REFERENCE_DROP_RATES.get("common", 0.55))
                total_expected_weight += expected_rate * count

        # Average observed drop rate (items per kill)
        obs_rate = total_drops / total_kills
        # Weighted average expected rate
        avg_expected_rate = total_expected_weight / max(total_drops, 1)

        # Estimate multiplier
        mult = obs_rate / max(avg_expected_rate, 0.0001)

        # EMA
        if self._observation_count == 0:
            self._drop_mult_estimate = mult
        else:
            alpha = 0.15
            self._drop_mult_estimate = alpha * mult + (1 - alpha) * self._drop_mult_estimate

        self._observation_count = total_kills
        return self._drop_mult_estimate

    def get_mult_estimate(self) -> float:
        """Current best estimate of the server's drop multiplier."""
        self.update_aggregate_estimate()
        return self._drop_mult_estimate

    def confidence(self) -> float:
        """Confidence in the drop rate estimate (0.0 — 1.0)."""
        total_kills = sum(self._monster_kills.values())
        if total_kills < 10:
            return 0.0
        count_factor = math.sqrt(total_kills / 50.0)
        return min(0.95, count_factor)

    def summary(self) -> dict[str, Any]:
        total_kills = sum(self._monster_kills.values())
        total_drops = sum(
            sum(drops.values()) for drops in self._monster_drops.values()
        )
        return {
            "estimated_mult": round(self._drop_mult_estimate, 2),
            "confidence": round(self.confidence(), 3),
            "total_kills": total_kills,
            "total_drops": total_drops,
            "monsters_tracked": len(self._monster_kills),
            "per_monster": {
                m: {
                    "kills": self._monster_kills[m],
                    "drops": len(self._monster_drops.get(m, {})),
                    "estimated_mult": self.per_monster_drop_multiplier(m),
                }
                for m in sorted(self._monster_kills.keys())
            },
        }


# ═══════════════════════════════════════════════════════════════
# ServerAdapter
# ═══════════════════════════════════════════════════════════════

@dataclass
class ServerProfile:
    """Detected server configuration."""
    server_type: ServerType = ServerType.UNKNOWN
    exp_multiplier: float = 1.0
    drop_multiplier: float = 1.0
    exp_confidence: float = 0.0
    drop_confidence: float = 0.0
    type_confidence: float = 0.0
    has_custom_items: bool = False
    has_custom_npcs: bool = False
    has_custom_warps: bool = False
    observation_count: int = 0

    @property
    def rate_category(self) -> ServerRateCategory:
        exp = self.exp_multiplier
        if self.exp_confidence < 0.3:
            return ServerRateCategory.UNKNOWN
        if exp < 5.0:
            return ServerRateCategory.LOW_RATE
        if exp < 25.0:
            return ServerRateCategory.MEDIUM_RATE
        if exp < 100.0:
            return ServerRateCategory.HIGH_RATE
        return ServerRateCategory.EXTREME_RATE

    def is_high_rate(self) -> bool:
        return self.rate_category.is_high()

    def is_low_rate(self) -> bool:
        return self.rate_category.is_low()

    def summary(self) -> dict[str, Any]:
        return {
            "server_type": str(self.server_type),
            "exp_multiplier": round(self.exp_multiplier, 2),
            "drop_multiplier": round(self.drop_multiplier, 2),
            "rate_category": str(self.rate_category),
            "confidence": {
                "exp": round(self.exp_confidence, 3),
                "drop": round(self.drop_confidence, 3),
                "type": round(self.type_confidence, 3),
            },
            "custom": {
                "items": self.has_custom_items,
                "npcs": self.has_custom_npcs,
                "warps": self.has_custom_warps,
            },
            "observations": self.observation_count,
        }


@dataclass
class ServerAdapter:
    """High-level server adaptation — detects rates, type, and custom mechanics.

    Usage::

        adapter = ServerAdapter()
        # As kills happen:
        adapter.observe_exp_kill(monster_level=45, base_exp_gained=1200)
        adapter.observe_drop("Poring", "Apple", category="common")
        adapter.observe_damage(defence=48, observed_damage=142)

        profile = adapter.get_profile()
        if profile.is_high_rate():
            # skip low-level farming
            pass
    """

    exp_observer: EXPObserver = field(default_factory=EXPObserver)
    drop_observer: DropObserver = field(default_factory=DropObserver)

    # Server type detection via DEF formula fitting
    _damage_observations: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _pre_renewal_residuals: list[float] = field(default_factory=list, repr=False)
    _renewal_residuals: list[float] = field(default_factory=list, repr=False)

    # Custom mechanics flags
    _standard_items_checked: set[str] = field(default_factory=set)
    _missing_standard_items: set[str] = field(default_factory=set)
    _custom_npc_hints: int = 0
    _custom_warp_hints: int = 0

    # Final cached profile
    _profile: ServerProfile | None = None

    # ── EXP observation ──

    def observe_exp_kill(self, monster_level: int, base_exp_gained: int) -> None:
        """Record EXP gained from killing a monster."""
        self.exp_observer.observe_kill(monster_level, base_exp_gained)
        self._profile = None  # invalidate cache

    def get_exp_multiplier(self) -> float:
        """Current best estimate of the server's base EXP multiplier."""
        return self.exp_observer.get_mult_estimate()

    def exp_confidence(self) -> float:
        return self.exp_observer.confidence()

    # ── Drop observation ──

    def observe_drop(self, monster_name: str, item_name: str,
                     category: str = "common",
                     expected_rate: float | None = None) -> None:
        """Record a drop from a monster kill."""
        self.drop_observer.record_drop(monster_name, item_name, category, expected_rate)
        self._profile = None

    def observe_kill_no_drop(self, monster_name: str) -> None:
        """Record a kill that produced no notable drops."""
        self.drop_observer.record_kill(monster_name)
        self._profile = None

    def get_drop_multiplier(self) -> float:
        return self.drop_observer.get_mult_estimate()

    def drop_confidence(self) -> float:
        return self.drop_observer.confidence()

    # ── Server type detection (DEF formula fitting) ──

    def observe_damage(self, monster_defence: int, observed_damage: int,
                       estimated_base_damage: int = 100) -> None:
        """Record a damage observation for server type detection.

        This compares observed damage to what would be expected under
        pre-renewal vs renewal DEF formulas.

        Args:
            monster_defence: Monster's DEF value (from monster DB).
            observed_damage: Actual damage dealt.
            estimated_base_damage: Estimated damage before DEF reduction.
                If unknown, pass total observed damage and the function
                uses relative comparisons.
        """
        obs = {
            "def": monster_defence,
            "observed": observed_damage,
            "base_est": estimated_base_damage,
        }
        self._damage_observations.append(obs)

        # Compute expected damage under each formula
        pre_mult = pre_renewal_damage_taken(monster_defence)
        ren_mult = renewal_damage_taken(monster_defence)

        pre_expected = estimated_base_damage * pre_mult
        ren_expected = estimated_base_damage * ren_mult

        self._pre_renewal_residuals.append(observed_damage - pre_expected)
        self._renewal_residuals.append(observed_damage - ren_expected)

        self._profile = None

    def _compute_server_type(self) -> tuple[ServerType, float]:
        """Determine server type by comparing DEF formula residuals.

        Returns (server_type, confidence).
        Uses a chi-squared-like comparison of the two residual sets.
        """
        if len(self._damage_observations) < 3:
            return ServerType.UNKNOWN, 0.0

        pre_sq = sum(r * r for r in self._pre_renewal_residuals)
        ren_sq = sum(r * r for r in self._renewal_residuals)

        # Avoid division by zero
        total = pre_sq + ren_sq
        if total < 0.001:
            return ServerType.PRE_RENEWAL, 0.5  # both formulas match perfectly

        # Pre-renewal wins if its residuals are smaller
        pre_fraction = pre_sq / total
        ren_fraction = ren_sq / total

        if pre_fraction < ren_fraction:
            confidence = 1.0 - pre_fraction
            return ServerType.PRE_RENEWAL, max(0.5, min(0.95, confidence))
        else:
            confidence = 1.0 - ren_fraction
            return ServerType.RENEWAL, max(0.5, min(0.95, confidence))

    # ── Custom mechanics detection ──

    def check_standard_item(self, item_name: str) -> None:
        """Flag a standard rAthena item as 'checked' (believed to exist)."""
        self._standard_items_checked.add(item_name)

    def report_missing_item(self, item_name: str) -> None:
        """Report that a standard rAthena item does not exist on this server."""
        self._missing_standard_items.add(item_name)
        self._profile = None

    def report_custom_npc(self) -> None:
        """Report encountering a custom NPC location."""
        self._custom_npc_hints += 1
        self._profile = None

    def report_custom_warp(self) -> None:
        """Report encountering a custom warp point."""
        self._custom_warp_hints += 1
        self._profile = None

    # ── Profile assembly ──

    def get_profile(self) -> ServerProfile:
        """Assemble and cache the current best-estimate server profile."""
        if self._profile is not None:
            return self._profile

        exp_mult = self.exp_observer.get_mult_estimate()
        drop_mult = self.drop_observer.get_mult_estimate()
        exp_conf = self.exp_observer.confidence()
        drop_conf = self.drop_observer.confidence()

        server_type, type_conf = self._compute_server_type()

        has_custom_items = len(self._missing_standard_items) > 0
        has_custom_npcs = self._custom_npc_hints > 0
        has_custom_warps = self._custom_warp_hints > 0

        self._profile = ServerProfile(
            server_type=server_type,
            exp_multiplier=exp_mult,
            drop_multiplier=drop_mult,
            exp_confidence=exp_conf,
            drop_confidence=drop_conf,
            type_confidence=type_conf,
            has_custom_items=has_custom_items,
            has_custom_npcs=has_custom_npcs,
            has_custom_warps=has_custom_warps,
            observation_count=self.exp_observer.observation_count(),
        )
        return self._profile

    def summary(self) -> dict[str, Any]:
        """Full diagnostic summary."""
        profile = self.get_profile()
        return {
            "profile": profile.summary(),
            "exp_observer": self.exp_observer.summary(),
            "drop_observer": self.drop_observer.summary(),
            "damage_observations": len(self._damage_observations),
            "server_type_test": (
                None if not self._damage_observations
                else {
                    "pre_renewal_rss": round(sum(r*r for r in self._pre_renewal_residuals), 2),
                    "renewal_rss": round(sum(r*r for r in self._renewal_residuals), 2),
                }
            ),
            "custom_mechanics": {
                "standard_items_checked": len(self._standard_items_checked),
                "missing_standard_items": sorted(self._missing_standard_items),
                "custom_npc_hints": self._custom_npc_hints,
                "custom_warp_hints": self._custom_warp_hints,
            },
        }


# ═══════════════════════════════════════════════════════════════
# StrategyAdjuster
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategyAdjustment:
    """Strategy adjustments derived from the server profile.

    Each field describes how the bot's behaviour should change.
    """
    # Cold-start / progression
    skip_levels_under: int = 0           # Don't bother with monsters below this level
    buy_equipment_at_level: int = 1      # Start buying gear at this level
    target_level_before_gear: int = 70   # Level to reach before gear investment
    farm_frugally: bool = True           # Keep everything, sell to NPC
    allow_vending: bool = False          # Spend time vending items to players

    # Combat
    damage_formula: str = "pre_renewal"  # 'pre_renewal' or 'renewal'
    favor_elemental_damage: bool = False # On high-rate, element advantage matters less

    # Economy
    npc_price_tolerance: float = 1.0     # How much above NPC price to pay at player market
    keep_minimum_items: int = 5          # Min stack of consumables to keep
    item_valuation_bias: str = "npc"     # 'npc' (frugal) or 'market' (speculative)

    # General
    grind_efficiency_mode: bool = False  # If True, optimize for XP/hr over zeny/hr
    priority_stat: str = "str"           # Which stat to prioritize


class StrategyAdjuster:
    """Translates a ServerProfile into concrete strategy adjustments.

    The adjuster updates other domain modules by producing HeuristicActions
    and changing configuration values that downstream systems consume.
    """

    def __init__(self) -> None:
        self._current_adjustment: StrategyAdjustment = StrategyAdjustment()
        self._last_profile: ServerProfile | None = None

    def adjust(self, profile: ServerProfile) -> StrategyAdjustment:
        """Compute strategy adjustments from the given server profile.

        This is re-entrant: calling it multiple times with the same profile
        returns the cached adjustment. Call again with a new profile to re-compute.
        """
        if self._last_profile is profile:
            return self._current_adjustment

        adj = StrategyAdjustment()

        # ── EXP-rate based adjustments ──
        if profile.exp_confidence >= 0.3:
            rate_cat = profile.rate_category

            if rate_cat == ServerRateCategory.LOW_RATE:
                # 1x-5x: farm frugally, keep everything
                adj.farm_frugally = True
                adj.skip_levels_under = 0
                adj.buy_equipment_at_level = 40
                adj.target_level_before_gear = 70
                adj.allow_vending = True
                adj.keep_minimum_items = 10
                adj.npc_price_tolerance = 1.0
                adj.item_valuation_bias = "npc"
                adj.grind_efficiency_mode = False
                adj.priority_stat = "str"

            elif rate_cat == ServerRateCategory.MEDIUM_RATE:
                # 5x-25x: balanced
                adj.farm_frugally = True
                adj.skip_levels_under = 5
                adj.buy_equipment_at_level = 25
                adj.target_level_before_gear = 70
                adj.allow_vending = False
                adj.keep_minimum_items = 5
                adj.npc_price_tolerance = 1.5
                adj.item_valuation_bias = "npc"
                adj.grind_efficiency_mode = False
                adj.priority_stat = "str"

            elif rate_cat == ServerRateCategory.HIGH_RATE:
                # 25x-100x: skip low-level, buy gear
                adj.farm_frugally = False
                adj.skip_levels_under = 15
                adj.buy_equipment_at_level = 1
                adj.target_level_before_gear = 99
                adj.allow_vending = False
                adj.keep_minimum_items = 3
                adj.npc_price_tolerance = 2.0
                adj.item_valuation_bias = "market"
                adj.grind_efficiency_mode = True
                adj.priority_stat = "agi"  # High rate: AGI for fast kills

            elif rate_cat == ServerRateCategory.EXTREME_RATE:
                # 100x+: skip everything, max level ASAP
                adj.farm_frugally = False
                adj.skip_levels_under = 40
                adj.buy_equipment_at_level = 1
                adj.target_level_before_gear = 99
                adj.allow_vending = False
                adj.keep_minimum_items = 1
                adj.npc_price_tolerance = 5.0
                adj.item_valuation_bias = "market"
                adj.grind_efficiency_mode = True
                adj.priority_stat = "agi"

        # ── Server type based adjustments ──
        if profile.type_confidence >= 0.5:
            if profile.server_type == ServerType.RENEWAL:
                adj.damage_formula = "renewal"
                adj.favor_elemental_damage = True
            else:
                adj.damage_formula = "pre_renewal"
                adj.favor_elemental_damage = False

        # ── Drop-rate based adjustments ──
        if profile.drop_confidence >= 0.3:
            if profile.drop_multiplier >= 5.0:
                # High drop rate: be more selective about what to pick up
                adj.item_valuation_bias = "market"
            else:
                # Low drop rate: keep everything
                adj.item_valuation_bias = "npc"

        # ── Custom mechanics ──
        if profile.has_custom_items:
            logger.info("Server has custom items — reverting to generic strategies")
            adj.item_valuation_bias = "npc"

        self._current_adjustment = adj
        self._last_profile = profile
        return adj

    def get_adjustment(self) -> StrategyAdjustment:
        return self._current_adjustment

    def produce_actions(self, profile: ServerProfile,
                        actions: list[Any] | None = None) -> list[Any]:
        """Produce HeuristicAction-like dicts for downstream domain modules.

        Args:
            profile: The current server profile.
            actions: Optional list to append to (mimics domain module pattern).

        Returns:
            List of action dicts (or the appended-to list).
        """
        from ai_sidecar.actions import HeuristicAction

        if actions is None:
            actions = []

        adj = self.adjust(profile)

        # ── Cold start adjustment ──
        if adj.skip_levels_under > 0:
            actions.append(HeuristicAction(
                kind="command",
                command=f"cold_start skip_under={adj.skip_levels_under}",
                confidence=profile.exp_confidence,
                reason=f"Server rate {profile.rate_category}: skip levels < {adj.skip_levels_under}",
                domain="server_adapter",
                metadata={
                    "rate_category": str(profile.rate_category),
                    "exp_mult": round(profile.exp_multiplier, 2),
                    "skip_levels_under": adj.skip_levels_under,
                },
            ))

        # ── Equipment buying threshold ──
        actions.append(HeuristicAction(
            kind="command",
            command=f"equipment start_buying_at={adj.buy_equipment_at_level} "
                    f"target_level={adj.target_level_before_gear}",
            confidence=profile.exp_confidence,
            reason=f"Adjusted gear acquisition for {profile.rate_category} rate",
            domain="server_adapter",
            metadata={
                "buy_at_level": adj.buy_equipment_at_level,
                "target_level": adj.target_level_before_gear,
            },
        ))

        # ── Combat formula ──
        if profile.type_confidence >= 0.5:
            actions.append(HeuristicAction(
                kind="command",
                command=f"combat formula={adj.damage_formula}",
                confidence=profile.type_confidence,
                reason=f"Server type detected: {profile.server_type}",
                domain="server_adapter",
                metadata={
                    "server_type": str(profile.server_type),
                    "damage_formula": adj.damage_formula,
                },
            ))

        # ── Economy adjustment ──
        actions.append(HeuristicAction(
            kind="command",
            command=f"economy bias={adj.item_valuation_bias} "
                    f"keep_min={adj.keep_minimum_items} "
                    f"price_tolerance={adj.npc_price_tolerance}",
            confidence=max(profile.exp_confidence, profile.drop_confidence),
            reason=f"Economy adjusted for rate={profile.rate_category} "
                   f"drop_mult={profile.drop_multiplier:.1f}x",
            domain="server_adapter",
            metadata={
                "valuation_bias": adj.item_valuation_bias,
                "keep_minimum": adj.keep_minimum_items,
                "price_tolerance": adj.npc_price_tolerance,
            },
        ))

        return actions
