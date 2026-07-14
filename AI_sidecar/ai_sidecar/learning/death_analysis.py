"""
Death Analysis / Adaptive Learning module.

Performs post-mortem analysis on deaths and adjusts behavior
based on learned patterns. Thread-safe via RLock.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Final, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DeathRecord:
    """A single death event captured for post-mortem analysis."""

    timestamp: float
    map_name: str
    position: Tuple[float, float]
    monster_name: str
    monster_id: int
    hp_before_death: int
    max_hp: int
    aggro_count: int
    had_potions: bool
    was_casting: bool
    buffs_active: List[str]
    seconds_since_last_heal: float
    cause_of_death: str
    lesson_learned: str


@dataclass
class BehaviorAdjustment:
    """A suggested or applied adjustment to bot behavior parameters."""

    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: float


# ---------------------------------------------------------------------------
# Cause-of-death classifier
# ---------------------------------------------------------------------------

_CAUSE_RULES: Final[List[Tuple[str, str]]] = [
    ("overpulled", "Died with high aggro count and no potions available."),
    ("no_potions", "Had no potions available when taking fatal damage."),
    ("boss_skill", "Killed by a boss/mvp-level monster, likely a special skill."),
    ("ambush", "Died very shortly after last heal with no buffs active."),
    ("heal_starvation", "Long time since last heal while HP was critically low."),
    ("cast_lock", "Was casting a skill when killed, unable to flee or heal."),
    ("buff_drop", "Buffs expired or were absent before death."),
    ("unknown", "Could not determine a specific cause pattern."),
]


def _classify_cause(record: DeathRecord) -> str:
    """Heuristic classification of death cause based on record fields."""
    if record.aggro_count >= 5 and not record.had_potions:
        return "overpulled"
    if not record.had_potions and record.hp_before_death < record.max_hp * 0.3:
        return "no_potions"
    if record.monster_name and any(
        kw in record.monster_name.lower() for kw in ("mvp", "boss", "mini", "lord", "queen", "king")
    ):
        return "boss_skill"
    if record.seconds_since_last_heal < 1.5 and not record.buffs_active:
        return "ambush"
    if record.seconds_since_last_heal > 8.0 and record.hp_before_death < record.max_hp * 0.4:
        return "heal_starvation"
    if record.was_casting and record.seconds_since_last_heal > 3.0:
        return "cast_lock"
    if not record.buffs_active and record.hp_before_death < record.max_hp * 0.5:
        return "buff_drop"
    return "unknown"


def _generate_lesson(record: DeathRecord, cause: str) -> str:
    """Generate a human-readable lesson from a death record and its cause."""
    lessons = {
        "overpulled": (
            f"Aggro count was {record.aggro_count} with no potions. "
            f"Reduce max_aggro or ensure potion supply before engaging multiple targets."
        ),
        "no_potions": (
            f"HP was {record.hp_before_death}/{record.max_hp} with no potions available. "
            f"Maintain a minimum potion stock and increase heal_threshold."
        ),
        "boss_skill": (
            f"Killed by {record.monster_name} (ID {record.monster_id}). "
            f"Consider adding a boss-specific flee or teleport trigger."
        ),
        "ambush": (
            f"Took fatal damage within {record.seconds_since_last_heal:.1f}s of last heal "
            f"with no buffs active. Increase situational awareness radius."
        ),
        "heal_starvation": (
            f"Last heal was {record.seconds_since_last_heal:.1f}s ago at {record.hp_before_death}/{record.max_hp} HP. "
            f"Lower heal_threshold or reduce flee_hp_pct."
        ),
        "cast_lock": (
            f"Was casting when killed — unable to react. "
            f"Consider interrupt-on-danger or faster cast times."
        ),
        "buff_drop": (
            f"No buffs active at death with HP at {record.hp_before_death}/{record.max_hp}. "
            f"Ensure auto-buff triggers before HP drops below 50%."
        ),
        "unknown": (
            f"Death by {record.monster_name} on {record.map_name} at {record.position}. "
            f"No clear pattern identified."
        ),
    }
    return lessons.get(cause, lessons["unknown"])


# ---------------------------------------------------------------------------
# Adjustment suggestions
# ---------------------------------------------------------------------------

_ADJUSTMENT_RULES: Final[dict] = {
    "overpulled": [
        ("max_aggro", 0.75),
        ("flee_hp_pct", 0.05),
    ],
    "no_potions": [
        ("heal_threshold", 0.10),
        ("min_potion_stock", 0.0),
    ],
    "boss_skill": [
        ("flee_hp_pct", 0.10),
        ("max_aggro", 0.0),
    ],
    "ambush": [
        ("flee_hp_pct", 0.08),
        ("heal_threshold", 0.05),
    ],
    "heal_starvation": [
        ("heal_threshold", 0.10),
        ("flee_hp_pct", 0.05),
    ],
    "cast_lock": [
        ("flee_hp_pct", 0.05),
        ("max_aggro", 0.0),
    ],
    "buff_drop": [
        ("heal_threshold", 0.05),
        ("max_aggro", 0.0),
    ],
    "unknown": [],
}

_DEFAULT_PARAM_VALUES: Final[dict] = {
    "max_aggro": 3.0,
    "heal_threshold": 0.6,
    "flee_hp_pct": 0.3,
    "min_potion_stock": 10.0,
}


# ---------------------------------------------------------------------------
# DeathAnalyzer
# ---------------------------------------------------------------------------


class DeathAnalyzer:
    """Thread-safe death analysis and adaptive learning engine.

    Maintains a history of death events, classifies their causes, and
    produces behavior adjustments to reduce future deaths.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._deaths: List[DeathRecord] = []
        self._adjustments: List[BehaviorAdjustment] = []
        self._start_time: float = time.time()

    # -- Recording ---------------------------------------------------------

    def record_death(self, record: DeathRecord) -> None:
        """Record a death event and automatically classify it."""
        with self._lock:
            if record.cause_of_death == "unknown" or not record.cause_of_death:
                cause = _classify_cause(record)
                object.__setattr__(record, "cause_of_death", cause)
            if not record.lesson_learned:
                lesson = _generate_lesson(record, record.cause_of_death)
                object.__setattr__(record, "lesson_learned", lesson)
            self._deaths.append(record)

    # -- Analysis ----------------------------------------------------------

    def analyze_death(self, record: DeathRecord) -> str:
        """Analyze a single death record and return the cause analysis string."""
        cause = _classify_cause(record)
        lesson = _generate_lesson(record, cause)
        return (
            f"Cause: {cause}\n"
            f"Monster: {record.monster_name} (ID {record.monster_id})\n"
            f"Map: {record.map_name} at {record.position}\n"
            f"HP: {record.hp_before_death}/{record.max_hp}  "
            f"Aggro: {record.aggro_count}  "
            f"Potions: {record.had_potions}  "
            f"Casting: {record.was_casting}\n"
            f"Last heal: {record.seconds_since_last_heal:.1f}s ago  "
            f"Buffs: {len(record.buffs_active)} active\n"
            f"Lesson: {lesson}"
        )

    # -- Queries -----------------------------------------------------------

    def get_adjustments(self) -> List[BehaviorAdjustment]:
        """Return all recorded behavior adjustments."""
        with self._lock:
            return list(self._adjustments)

    def get_death_history(self, limit: int = 20) -> List[DeathRecord]:
        """Return the most recent death records, newest first."""
        with self._lock:
            return list(reversed(self._deaths[-limit:]))

    def get_deaths_by_map(self, map_name: str) -> List[DeathRecord]:
        """Return all death records for a specific map."""
        with self._lock:
            return [d for d in self._deaths if d.map_name == map_name]

    def get_deaths_by_monster(self, monster_name: str) -> List[DeathRecord]:
        """Return all death records for a specific monster (case-insensitive)."""
        with self._lock:
            return [d for d in self._deaths if d.monster_name.lower() == monster_name.lower()]

    def get_death_rate_per_hour(self) -> float:
        """Calculate deaths per hour based on elapsed time since first record or start."""
        with self._lock:
            if not self._deaths:
                return 0.0
            elapsed = time.time() - self._start_time
            if elapsed <= 0:
                return 0.0
            return len(self._deaths) / (elapsed / 3600.0)

    def get_most_common_cause(self) -> str:
        """Return the most frequently occurring cause of death."""
        with self._lock:
            if not self._deaths:
                return "none"
            counter: Counter[str] = Counter(d.cause_of_death for d in self._deaths)
            return counter.most_common(1)[0][0]

    def get_most_dangerous_monster(self) -> str:
        """Return the monster name that has killed us the most."""
        with self._lock:
            if not self._deaths:
                return "none"
            counter: Counter[str] = Counter(d.monster_name for d in self._deaths)
            return counter.most_common(1)[0][0]

    def get_most_dangerous_map(self) -> str:
        """Return the map name where we have died the most."""
        with self._lock:
            if not self._deaths:
                return "none"
            counter: Counter[str] = Counter(d.map_name for d in self._deaths)
            return counter.most_common(1)[0][0]

    # -- Adaptive adjustments ---------------------------------------------

    def get_suggested_adjustments(self) -> List[BehaviorAdjustment]:
        """Generate behavior adjustments based on recent death patterns.

        Analyzes the last 10 deaths and suggests parameter changes
        to mitigate the most common causes.
        """
        with self._lock:
            if not self._deaths:
                return []

            recent = self._deaths[-10:]
            cause_counts: Counter[str] = Counter(d.cause_of_death for d in recent)
            dominant_cause, _ = cause_counts.most_common(1)[0]

            param_deltas: dict[str, float] = {}
            for cause, adjustments in _ADJUSTMENT_RULES.items():
                if cause_counts.get(cause, 0) > 0:
                    for param, delta in adjustments:
                        param_deltas[param] = max(
                            param_deltas.get(param, 0.0), delta
                        )

            suggestions: List[BehaviorAdjustment] = []
            now = time.time()
            for param, delta in param_deltas.items():
                old_val = _DEFAULT_PARAM_VALUES.get(param, 0.0)
                new_val = max(0.0, old_val - delta)
                if abs(new_val - old_val) < 0.01:
                    continue
                suggestions.append(
                    BehaviorAdjustment(
                        parameter=param,
                        old_value=old_val,
                        new_value=new_val,
                        reason=(
                            f"Dominant death cause: {dominant_cause} "
                            f"({cause_counts[dominant_cause]} of last {len(recent)} deaths)"
                        ),
                        timestamp=now,
                    )
                )

            return suggestions

    def get_learning_summary(self) -> str:
        """Return a human-readable summary of all learned death patterns."""
        with self._lock:
            if not self._deaths:
                return "No deaths recorded yet."

            total = len(self._deaths)
            rate = self.get_death_rate_per_hour()
            common_cause = self.get_most_common_cause()
            dangerous_monster = self.get_most_dangerous_monster()
            dangerous_map = self.get_most_dangerous_map()

            cause_counts = Counter(d.cause_of_death for d in self._deaths)
            monster_counts = Counter(d.monster_name for d in self._deaths)
            map_counts = Counter(d.map_name for d in self._deaths)

            lines = [
                f"Death Analysis Summary",
                f"{'=' * 50}",
                f"Total deaths: {total}",
                f"Death rate: {rate:.2f}/hour",
                f"",
                f"Most common cause: {common_cause}",
                f"Most dangerous monster: {dangerous_monster}",
                f"Most dangerous map: {dangerous_map}",
                f"",
                f"Cause breakdown:",
            ]
            for cause, count in cause_counts.most_common():
                pct = count / total * 100
                lines.append(f"  {cause}: {count} ({pct:.1f}%)")

            lines.append("")
            lines.append("Top monsters:")
            for monster, count in monster_counts.most_common(5):
                lines.append(f"  {monster}: {count}")

            lines.append("")
            lines.append("Top maps:")
            for map_name, count in map_counts.most_common(5):
                lines.append(f"  {map_name}: {count}")

            if self._adjustments:
                lines.append("")
                lines.append("Applied adjustments:")
                for adj in self._adjustments[-5:]:
                    lines.append(
                        f"  {adj.parameter}: {adj.old_value} -> {adj.new_value} "
                        f"({adj.reason})"
                    )

            return "\n".join(lines)

    # -- Reset -------------------------------------------------------------

    def reset(self) -> None:
        """Clear all death records and adjustments."""
        with self._lock:
            self._deaths.clear()
            self._adjustments.clear()
            self._start_time = time.time()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_analyzer: Optional[DeathAnalyzer] = None
_global_analyzer_lock = RLock()


def get_death_analyzer() -> DeathAnalyzer:
    """Return the global DeathAnalyzer singleton (thread-safe)."""
    global _global_analyzer  # noqa: PLW0603
    if _global_analyzer is None:
        with _global_analyzer_lock:
            if _global_analyzer is None:
                _global_analyzer = DeathAnalyzer()
    return _global_analyzer
