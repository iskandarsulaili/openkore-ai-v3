"""Danger predictor and safety evaluator for proactive death prevention.

Replaces reactive death handling (walk to Prontera at 20% HP) with
predictive danger assessment before HP drops.

Core components:
  - DangerPredictor: Per-map danger rating, proactive escape decisions
  - SafetyEvaluator: Safe tolerance calculation, potion consumption monitoring
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Danger threshold: if rating > 0.25 (monster hits for >25% of HP), leave map
_DEFAULT_DANGER_THRESHOLD = 0.25

# HP loss threshold: if average HP loss per kill > 15%, equip more defensive gear
_DEFAULT_HP_LOSS_THRESHOLD = 0.15

# Safe tolerance threshold: if would die to < 3 hits at current HP, consider fleeing
_DEFAULT_SAFE_TOLERANCE_MIN = 3

# Potion consumption threshold: if using > 5 pots per minute, map too dangerous
_DEFAULT_POTION_RATE_THRESHOLD = 5.0  # pots per minute

# Emergency HP threshold: if no potions AND no wings AND HP < 50%, walk to town NOW
_DEFAULT_EMERGENCY_HP_THRESHOLD = 0.50


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class DangerAssessment:
    """Assessment of danger for a specific map and player state."""
    map_name: str
    danger_rating: float            # 0-1: overall map danger
    is_dangerous: bool              # True if rating > threshold
    max_monster_damage: int         # Highest monster hit observed
    player_max_hp: int              # Player's max HP
    danger_vs_hp_ratio: float       # max_monster_damage / player_max_hp
    monster_density: float          # 0-1
    reason: str = ""


@dataclass
class SafetyRecommendation:
    """A safety action recommended by the evaluator."""
    action_type: str                # "flee", "flywing", "butterfly_wing", "retreat", "defensive_gear"
    priority: int                   # 0=critical, 1=urgent, 2=advisory
    command: str                    # OpenKore command
    reason: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KillRecord:
    """Record of a single kill for HP loss tracking."""
    monster_name: str
    hp_before: int
    hp_after: int
    hp_loss: int
    hp_loss_pct: float              # loss / max_hp
    timestamp: float
    potions_used: int = 0


# ═══════════════════════════════════════════════════════════════
# DangerPredictor
# ═══════════════════════════════════════════════════════════════

class DangerPredictor:
    """Predicts danger level per map and recommends proactive escape.

    Danger model:
      - Per-map danger rating = (max_monster_damage / player_max_hp) × monster_density
      - If rating > 0.25 (monster hits for >25% of HP), leave map
      - If average HP loss per kill > 15%, equip more defensive gear
      - Escape priority: Butterfly Wing > Fly Wing > Walk to nearest town
      - Emergency consumable check: if no potions AND no wings AND HP < 50% → walk to town NOW
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-map monster stats: map_name -> [(damage_per_hit, monster_name)]
        self._map_monster_stats: dict[str, list[tuple[int, str]]] = {}

        # Per-bot kill records: bot_id -> deque of KillRecord
        self._kill_history: dict[str, deque[KillRecord]] = {}

        # Per-bot: map danger cache
        self._map_danger_cache: dict[str, float] = {}

        # Thresholds
        self.danger_threshold: float = _DEFAULT_DANGER_THRESHOLD
        self.hp_loss_threshold: float = _DEFAULT_HP_LOSS_THRESHOLD

        # Track current map for each bot
        self._current_map: dict[str, str] = {}

    # ── Public API ──

    def set_thresholds(
        self,
        danger_threshold: float | None = None,
        hp_loss_threshold: float | None = None,
    ) -> None:
        """Override danger thresholds."""
        with self._lock:
            if danger_threshold is not None:
                self.danger_threshold = max(0.05, min(1.0, danger_threshold))
            if hp_loss_threshold is not None:
                self.hp_loss_threshold = max(0.05, min(1.0, hp_loss_threshold))

    def record_monster_encounter(
        self,
        map_name: str,
        monster_name: str,
        monster_damage: int,
    ) -> None:
        """Record a monster's damage per hit for map danger calculation."""
        with self._lock:
            if map_name not in self._map_monster_stats:
                self._map_monster_stats[map_name] = []
            # Keep top 10 highest-damage monsters per map
            stats = self._map_monster_stats[map_name]
            stats.append((monster_damage, monster_name))
            stats.sort(key=lambda x: -x[0])
            self._map_monster_stats[map_name] = stats[:10]
            # Invalidate cache
            self._map_danger_cache.pop(map_name, None)

    def record_kill(
        self,
        bot_id: str,
        monster_name: str,
        hp_before: int,
        hp_after: int,
        player_max_hp: int,
        potions_used: int = 0,
    ) -> None:
        """Record a kill with HP state for loss tracking."""
        with self._lock:
            if bot_id not in self._kill_history:
                self._kill_history[bot_id] = deque(maxlen=100)

            hp_loss = max(0, hp_before - hp_after)
            hp_loss_pct = hp_loss / max(player_max_hp, 1)

            record = KillRecord(
                monster_name=monster_name,
                hp_before=hp_before,
                hp_after=hp_after,
                hp_loss=hp_loss,
                hp_loss_pct=hp_loss_pct,
                timestamp=time.time(),
                potions_used=potions_used,
            )
            self._kill_history[bot_id].append(record)

    def record_potion_use(self, bot_id: str, count: int = 1) -> None:
        """Record potion usage for consumption rate tracking."""
        with self._lock:
            if bot_id not in self._kill_history:
                self._kill_history[bot_id] = deque(maxlen=100)
            # Add a synthetic record to track potion use
            record = KillRecord(
                monster_name="__potion__",
                hp_before=0,
                hp_after=0,
                hp_loss=0,
                hp_loss_pct=0.0,
                timestamp=time.time(),
                potions_used=count,
            )
            self._kill_history[bot_id].append(record)

    def assess_map_danger(
        self,
        map_name: str,
        player_max_hp: int,
        monster_density: float = 0.5,
    ) -> DangerAssessment:
        """Assess the danger level of a map for the current player.

        Args:
            map_name: The map to assess.
            player_max_hp: Player's maximum HP.
            monster_density: Estimated monster density (0-1), from map data.

        Returns:
            DangerAssessment with rating and recommendation.
        """
        with self._lock:
            # Check cache
            cache_key = f"{map_name}:{player_max_hp}"
            cached = self._map_danger_cache.get(cache_key)
            if cached is not None and isinstance(cached, (int, float)):
                cached_rating = cached
            else:
                # Calculate from monster stats
                stats = self._map_monster_stats.get(map_name, [])
                max_damage = max((d for d, _ in stats), default=0)
                density = monster_density if stats else monster_density * 0.5

                if player_max_hp > 0:
                    ratio = max_damage / player_max_hp
                else:
                    ratio = 0.0

                danger_rating = ratio * density
                self._map_danger_cache[cache_key] = danger_rating
                cached_rating = danger_rating

            # Calculate values for the assessment
            stats = self._map_monster_stats.get(map_name, [])
            max_damage = max((d for d, _ in stats), default=0)
            ratio = max_damage / max(player_max_hp, 1)
            danger_rating = cached_rating

            is_dangerous = ratio > self.danger_threshold
            reason_parts = []
            if is_dangerous:
                reason_parts.append(
                    f"Monster hits for {ratio:.1%} of HP (threshold: {self.danger_threshold:.0%})"
                )
            if not stats:
                reason_parts.append("No monster data for this map yet")

            danger_result = DangerAssessment(
                map_name=map_name,
                danger_rating=danger_rating,
                is_dangerous=is_dangerous,
                max_monster_damage=max_damage,
                player_max_hp=player_max_hp,
                danger_vs_hp_ratio=ratio,
                monster_density=monster_density,
                reason="; ".join(reason_parts) if reason_parts else "Safe map",
            )
            return danger_result

    def get_avg_hp_loss_per_kill(
        self,
        bot_id: str,
        window_seconds: float = 300.0,
    ) -> float:
        """Get average HP loss per kill over the recent window.

        Returns fraction of max HP lost per kill (0-1).
        """
        with self._lock:
            history = self._kill_history.get(bot_id)
            if not history:
                return 0.0

            now = time.time()
            recent = [
                r for r in history
                if r.timestamp > now - window_seconds
                and r.monster_name != "__potion__"
            ]
            if not recent:
                return 0.0

            return sum(r.hp_loss_pct for r in recent) / len(recent)

    def get_potion_rate(
        self,
        bot_id: str,
        window_seconds: float = 60.0,
    ) -> float:
        """Get potion consumption rate (pots per minute) over the recent window."""
        with self._lock:
            history = self._kill_history.get(bot_id)
            if not history:
                return 0.0

            now = time.time()
            recent = [
                r for r in history
                if r.timestamp > now - window_seconds
                and r.monster_name == "__potion__"
            ]
            total_pots = sum(r.potions_used for r in recent)

            window_min = max(window_seconds / 60.0, 0.1)
            return total_pots / window_min

    def assess_safety(
        self,
        bot_id: str,
        signals: dict[str, Any],
        monster_density: float = 0.5,
        has_fly_wing: bool = False,
        has_butterfly_wing: bool = False,
        has_potions: bool = False,
        inventory: list[Any] | None = None,
    ) -> list[SafetyRecommendation]:
        """Full safety assessment — produce recommendations.

        Checks:
          1. Per-map danger: if monster hits for >25% HP → leave map
          2. Average HP loss per kill > 15% → defensive gear
          3. Emergency: no pots + no wings + HP < 50% → walk to town NOW

        Args:
            bot_id: Bot identifier.
            signals: Current bot state signals.
            monster_density: Estimated monster density for current map.
            has_fly_wing: Whether bot has Fly Wings.
            has_butterfly_wing: Whether bot has Butterfly Wings.
            has_potions: Whether bot has potions.
            inventory: Full inventory item list (for item detection).

        Returns:
            List of SafetyRecommendation (sorted by priority).
        """
        recommendations: list[SafetyRecommendation] = []

        current_map = signals.get("map", "").lower()
        hp = signals.get("hp", signals.get("hp_ratio", 1.0))
        hp_ratio = signals.get("hp_ratio", 1.0)
        if isinstance(hp, (int, float)) and hp > 1:
            # hp is absolute value, need to also get hp_max
            hp_max = signals.get("hp_max", 1) or 1
            hp_ratio = hp / max(hp_max, 1)
        else:
            hp_ratio = float(hp) if isinstance(hp, (int, float)) else 1.0

        player_max_hp = signals.get("hp_max", signals.get("maxHp", 1)) or 1

        # Detect consumables from inventory if not explicitly provided
        if inventory is not None:
            item_names = [str(item).lower() for item in inventory]
            if not has_fly_wing:
                has_fly_wing = any("fly wing" in name or "flywing" in name for name in item_names)
            if not has_butterfly_wing:
                has_butterfly_wing = any(
                    "butterfly wing" in name or "butterflywing" in name
                    for name in item_names
                )
            if not has_potions:
                has_potions = any("potion" in name or "red" in name for name in item_names)

        # ── 1. Per-map danger check ──
        danger = self.assess_map_danger(current_map, player_max_hp, monster_density)
        if danger.is_dangerous:
            # Determine escape method priority: Butterfly Wing > Fly Wing > Walk
            if has_butterfly_wing:
                recommendations.append(SafetyRecommendation(
                    action_type="butterfly_wing",
                    priority=0,
                    command="ai manual; use_butterfly_wing",
                    reason=(
                        f"DANGER: {danger.reason}. "
                        f"Using Butterfly Wing to escape {current_map}"
                    ),
                    confidence=0.95,
                    metadata=danger.__dict__,
                ))
            elif has_fly_wing:
                recommendations.append(SafetyRecommendation(
                    action_type="flywing",
                    priority=0,
                    command="ai manual; use_fly_wing",
                    reason=(
                        f"DANGER: {danger.reason}. "
                        f"Using Fly Wing to escape {current_map}"
                    ),
                    confidence=0.90,
                    metadata=danger.__dict__,
                ))
            else:
                # Walk to nearest town
                target_town = self._guess_nearest_town(current_map)
                recommendations.append(SafetyRecommendation(
                    action_type="flee",
                    priority=0,
                    command=f"ai manual; move {target_town}",
                    reason=(
                        f"DANGER: {danger.reason}. "
                        f"No wings — walking to {target_town}"
                    ),
                    confidence=0.85,
                    metadata={"target_town": target_town, **danger.__dict__},
                ))

        # ── 2. Average HP loss per kill > 15% → defensive gear ──
        avg_loss = self.get_avg_hp_loss_per_kill(bot_id)
        if avg_loss > self.hp_loss_threshold:
            recommendations.append(SafetyRecommendation(
                action_type="defensive_gear",
                priority=1,
                command="equip_defensive",
                reason=(
                    f"Average HP loss per kill {avg_loss:.1%} exceeds "
                    f"threshold ({self.hp_loss_threshold:.0%}) — "
                    f"equipping more defensive gear"
                ),
                confidence=0.80,
                metadata={
                    "avg_hp_loss_pct": round(avg_loss, 3),
                    "threshold": self.hp_loss_threshold,
                },
            ))

        # ── 3. Emergency: no pots + no wings + HP < 50% → walk to town NOW ──
        no_consumables = not has_potions and not has_fly_wing and not has_butterfly_wing
        if no_consumables and hp_ratio < _DEFAULT_EMERGENCY_HP_THRESHOLD:
            target_town = self._guess_nearest_town(current_map)
            recommendations.append(SafetyRecommendation(
                action_type="flee",
                priority=0,  # Critical
                command=f"ai manual; move {target_town}",
                reason=(
                    f"EMERGENCY: HP={hp_ratio:.0%}, no potions, no wings. "
                    f"Walking to {target_town} immediately"
                ),
                confidence=0.99,
                metadata={
                    "target_town": target_town,
                    "hp_ratio": hp_ratio,
                },
            ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)
        return recommendations

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Full assessment entry point for the heuristic loop.

        Converts SafetyRecommendations into HeuristicActions.
        """
        _bot_id = bot_id or signals.get("bot_id", "default")
        current_map = signals.get("map", "").lower()

        # Track current map
        self._current_map[_bot_id] = current_map

        # Get inventory items for consumable detection
        inventory = signals.get("inventory_items", signals.get("inventory", []))

        # Run full safety assessment
        safety_recs = self.assess_safety(
            _bot_id, signals,
            monster_density=0.5,  # Default; evaluate_monster_data updates this
            inventory=inventory,
        )

        for rec in safety_recs:
            actions.append(HeuristicAction(
                kind="command",
                command=rec.command,
                confidence=rec.confidence,
                reason=rec.reason,
                domain="combat",
                metadata={
                    "safety_action_type": rec.action_type,
                    "priority": rec.priority,
                    **rec.metadata,
                },
            ))

    # ── Internal helpers ──

    @staticmethod
    def _guess_nearest_town(map_name: str) -> str:
        """Guess the nearest town from a map name."""
        map_lower = map_name.lower()
        town_map: dict[str, str] = {
            "prt_": "prontera",
            "pay_": "payon",
            "gef_": "geffen",
            "moc_": "morocc",
            "iz_": "izlude",
            "alde_": "aldebaran",
            "comodo": "comodo",
            "umbala": "umbala",
            "niflheim": "niflheim",
            "rachel": "rachel",
            "veins": "veins",
            "ein_": "einbroch",
            "lhz_": "lighthalzen",
            "yuno": "yuno",
            "hugel": "hugel",
            "ama_": "amatsu",
            "gon_": "gonryun",
            "lou_": "louyang",
            "ayo_": "ayothaya",
            "mjolnir": "prontera",
        }
        for prefix, town in town_map.items():
            if prefix in map_lower:
                return town
        return "prontera"  # Default fallback


# ═══════════════════════════════════════════════════════════════
# SafetyEvaluator
# ═══════════════════════════════════════════════════════════════

class SafetyEvaluator:
    """Evaluates combat safety in real-time.

    Calculates:
      - safe_tolerance: how many consecutive hits a bot can survive at current HP
      - Potion consumption rate monitoring
      - Retreat/flee recommendations based on danger analysis

    Designed to work alongside DangerPredictor for comprehensive safety.
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-bot: deque of potion timestamps for rate calculation
        self._potion_timestamps: dict[str, deque[float]] = {}

        # Per-bot: average monster damage observed
        self._avg_monster_damage: dict[str, float] = {}

        # Thresholds
        self.safe_tolerance_min: int = _DEFAULT_SAFE_TOLERANCE_MIN
        self.potion_rate_threshold: float = _DEFAULT_POTION_RATE_THRESHOLD

    # ── Public API ──

    def set_thresholds(
        self,
        safe_tolerance_min: int | None = None,
        potion_rate_threshold: float | None = None,
    ) -> None:
        """Override safety thresholds."""
        with self._lock:
            if safe_tolerance_min is not None:
                self.safe_tolerance_min = max(1, min(20, safe_tolerance_min))
            if potion_rate_threshold is not None:
                self.potion_rate_threshold = max(0.5, min(100.0, potion_rate_threshold))

    def record_potion_use(self, bot_id: str) -> None:
        """Record that a potion was used."""
        with self._lock:
            if bot_id not in self._potion_timestamps:
                self._potion_timestamps[bot_id] = deque(maxlen=200)
            self._potion_timestamps[bot_id].append(time.time())

    def record_monster_damage(self, bot_id: str, damage: int) -> None:
        """Record damage taken from a monster hit, updating moving average."""
        with self._lock:
            current = self._avg_monster_damage.get(bot_id, 0.0)
            if current == 0.0:
                self._avg_monster_damage[bot_id] = float(damage)
            else:
                # EMA smoothing
                self._avg_monster_damage[bot_id] = (
                    0.3 * damage + 0.7 * current
                )

    def calculate_safe_tolerance(
        self,
        bot_id: str,
        current_hp: int,
        player_max_hp: int,
    ) -> float:
        """Calculate how many consecutive hits the bot can survive.

        Uses the average monster damage observed. Higher = safer.
        If tolerance < 3, the bot is at risk.

        Args:
            bot_id: Bot identifier.
            current_hp: Current HP value.
            player_max_hp: Maximum HP.

        Returns:
            Number of hits the bot can survive at current HP.
        """
        with self._lock:
            avg_damage = self._avg_monster_damage.get(bot_id, 0.0)
            if avg_damage <= 0:
                # No damage data yet — estimate from max HP
                avg_damage = player_max_hp * 0.15  # Assume 15% per hit

            if avg_damage <= 0:
                return float("inf")

            tolerance = current_hp / avg_damage
            return max(0.0, tolerance)

    def get_potion_rate_per_minute(self, bot_id: str) -> float:
        """Get potion consumption rate (pots per minute) over the last 60s."""
        with self._lock:
            timestamps = self._potion_timestamps.get(bot_id)
            if not timestamps:
                return 0.0

            now = time.time()
            recent = [t for t in timestamps if t > now - 60.0]
            if not recent:
                return 0.0

            return len(recent) / 1.0  # Per minute

    def assess_safety(
        self,
        bot_id: str,
        current_hp: int,
        player_max_hp: int,
        current_map: str,
    ) -> list[SafetyRecommendation]:
        """Run full safety evaluation.

        Args:
            bot_id: Bot identifier.
            current_hp: Current HP value.
            player_max_hp: Maximum HP.
            current_map: Current map name.

        Returns:
            List of SafetyRecommendation (sorted by priority).
        """
        recommendations: list[SafetyRecommendation] = []

        hp_ratio = current_hp / max(player_max_hp, 1)

        # ── 1. Safe tolerance check ──
        tolerance = self.calculate_safe_tolerance(bot_id, current_hp, player_max_hp)
        if tolerance < self.safe_tolerance_min:
            # Would die too fast — recommend retreat
            target = DangerPredictor._guess_nearest_town(current_map)
            recommendations.append(SafetyRecommendation(
                action_type="retreat",
                priority=0,
                command=f"ai manual; move {target}",
                reason=(
                    f"Safe tolerance {tolerance:.1f} hits < {self.safe_tolerance_min}. "
                    f"Would die to {self.safe_tolerance_min} hits at current HP "
                    f"({hp_ratio:.0%}). Retreating to {target}"
                ),
                confidence=0.90,
                metadata={
                    "safe_tolerance": round(tolerance, 1),
                    "threshold": self.safe_tolerance_min,
                    "hp_ratio": round(hp_ratio, 2),
                    "avg_monster_damage": round(self._avg_monster_damage.get(bot_id, 0), 0),
                },
            ))

        # ── 2. Potion consumption rate check ──
        potion_rate = self.get_potion_rate_per_minute(bot_id)
        if potion_rate > self.potion_rate_threshold:
            recommendations.append(SafetyRecommendation(
                action_type="retreat",
                priority=1,
                command=f"ai manual; move {DangerPredictor._guess_nearest_town(current_map)}",
                reason=(
                    f"Potion consumption {potion_rate:.1f}/min exceeds "
                    f"threshold ({self.potion_rate_threshold:.0f}/min). "
                    f"Map {current_map} too dangerous, retreating"
                ),
                confidence=0.80,
                metadata={
                    "potion_rate": round(potion_rate, 1),
                    "threshold": self.potion_rate_threshold,
                    "map": current_map,
                },
            ))

        # ── 3. Low HP + no safe tolerance → immediate fly wing ──
        if tolerance < 1.0 and hp_ratio < 0.3:
            recommendations.append(SafetyRecommendation(
                action_type="flywing",
                priority=0,
                command="ai manual; use_fly_wing",
                reason=(
                    f"CRITICAL: HP={hp_ratio:.0%}, safe_tolerance={tolerance:.1f}. "
                    f"Would die to 1 hit. Using Fly Wing."
                ),
                confidence=0.99,
                metadata={
                    "safe_tolerance": round(tolerance, 1),
                    "hp_ratio": round(hp_ratio, 2),
                },
            ))

        recommendations.sort(key=lambda r: r.priority)
        return recommendations

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Full assessment entry point for the heuristic loop."""
        _bot_id = bot_id or signals.get("bot_id", "default")
        current_map = signals.get("map", "").lower()

        # Parse HP values from signals
        hp = signals.get("hp", 0)
        hp_max = signals.get("hp_max", signals.get("maxHp", 1)) or 1
        if isinstance(hp, float) and hp <= 1.0:
            # hp as ratio
            hp_ratio = hp
            hp = int(hp_ratio * hp_max)
        else:
            hp = int(hp) if hp else 0
            hp_ratio = hp / max(hp_max, 1)

        # Record damage if combat data available
        combat_data = signals.get("combat", {})
        if isinstance(combat_data, dict):
            last_damage = combat_data.get("last_damage_taken", 0)
            if last_damage:
                self.record_monster_damage(_bot_id, int(last_damage))

        # Run safety evaluation
        safety_recs = self.assess_safety(
            _bot_id, hp, hp_max, current_map,
        )

        for rec in safety_recs:
            actions.append(HeuristicAction(
                kind="command",
                command=rec.command,
                confidence=rec.confidence,
                reason=rec.reason,
                domain="combat",
                metadata={
                    "safety_action_type": rec.action_type,
                    "priority": rec.priority,
                    "evaluator": "SafetyEvaluator",
                    **rec.metadata,
                },
            ))


# ═══════════════════════════════════════════════════════════════
# Combined SafetyDomain (wires both predictors into heuristics)
# ═══════════════════════════════════════════════════════════════

class SafetyDomain:
    """Combined safety domain that runs both DangerPredictor and SafetyEvaluator.

    Designed to be called from the heuristic assess() loop.
    Wires into the existing architecture via assess() method matching BaseDomain.
    """

    name = "safety"
    priority = 10  # Runs early (survival-level priority)

    def __init__(self) -> None:
        self.predictor = DangerPredictor()
        self.evaluator = SafetyEvaluator()
        self._initialized = False

    def initialize(self) -> None:
        """Called once during domain registration."""
        self._initialized = True

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run full safety assessment.

        Runs DangerPredictor (map-level danger, HP loss, emergency) first,
        then SafetyEvaluator (safe tolerance, potion rate) for fine-grained checks.

        Both append safety actions to the shared actions list.
        """
        # Run DangerPredictor first (coarse-grained)
        self.predictor.assess(signals, actions, bot_id)

        # Then run SafetyEvaluator (fine-grained)
        self.evaluator.assess(signals, actions, bot_id)

    def counters(self) -> dict[str, Any]:
        """Return diagnostic counters."""
        return {
            "tracked_maps": len(self.predictor._map_monster_stats),
            "tracked_bots": len(self.predictor._kill_history),
            "potion_tracked_bots": len(self.evaluator._potion_timestamps),
        }
