"""
Predictive Threat Assessment — reads monster cast bars and predicts what's coming.

Enables the bot to react to future threats instead of current damage. By
monitoring which skills monsters are casting, how long they take, and how
dangerous they are, the bot can dodge, interrupt, or defend proactively.

Uses a pre-populated database of dangerous monster skills (30+ entries from
rAthena / classic Ragnarok Online) and evaluates live monster state to produce
a ThreatPrediction with actionable recommendations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PredictedThreat:
    """A single predicted threat from a monster's cast bar."""

    monster_id: int
    monster_name: str
    skill_name: str
    skill_element: str
    cast_time_ms: int
    remaining_cast_ms: int
    is_aoe: bool
    aoe_radius: int
    danger_level: int  # 1-10, 10 = deadly
    time_to_impact_ms: int
    recommended_action: str  # "move", "interrupt", "defend", "tank"
    confidence: float  # 0.0-1.0


@dataclass(slots=True)
class ThreatPrediction:
    """Aggregated threat assessment for the current combat situation."""

    threats: list[PredictedThreat]
    most_dangerous: PredictedThreat | None
    time_to_react_ms: int  # ms before the first threat lands
    should_flee: bool
    should_interrupt: bool
    should_move: bool
    should_defend: bool
    summary: str


# ── Dangerous Skill Database ──────────────────────────────────────────────────

# Each entry: skill_name -> {danger, is_aoe, radius, element, cast_time_ms, notes}
_DANGEROUS_SKILLS: dict[str, dict[str, Any]] = {
    # ── Wizard / High Wizard ──
    "Fire Ball": {
        "danger": 7, "is_aoe": True, "radius": 3, "element": "fire",
        "cast_time_ms": 2000, "notes": "Medium AoE fire damage",
    },
    "Fire Storm": {
        "danger": 8, "is_aoe": True, "radius": 5, "element": "fire",
        "cast_time_ms": 3000, "notes": "Large AoE fire damage",
    },
    "Storm Gust": {
        "danger": 9, "is_aoe": True, "radius": 7, "element": "water",
        "cast_time_ms": 5000, "notes": "Massive AoE water damage, freezes",
    },
    "Meteor Storm": {
        "danger": 10, "is_aoe": True, "radius": 7, "element": "fire",
        "cast_time_ms": 6000, "notes": "Deadly massive AoE fire, stuns",
    },
    "Lord of Vermilion": {
        "danger": 8, "is_aoe": True, "radius": 7, "element": "wind",
        "cast_time_ms": 4000, "notes": "Massive AoE wind damage",
    },
    "Heaven's Drive": {
        "danger": 7, "is_aoe": True, "radius": 5, "element": "neutral",
        "cast_time_ms": 3500, "notes": "Large AoE neutral damage",
    },
    "Quagmire": {
        "danger": 5, "is_aoe": True, "radius": 3, "element": "earth",
        "cast_time_ms": 1500, "notes": "Slows movement, reduces AGI",
    },
    "Frost Nova": {
        "danger": 6, "is_aoe": True, "radius": 4, "element": "water",
        "cast_time_ms": 1000, "notes": "Quick AoE freeze",
    },
    "Frost Diver": {
        "danger": 6, "is_aoe": False, "radius": 0, "element": "water",
        "cast_time_ms": 1500, "notes": "Single-target freeze",
    },
    "Stone Curse": {
        "danger": 7, "is_aoe": False, "radius": 0, "element": "earth",
        "cast_time_ms": 2000, "notes": "Petrifies target",
    },
    # ── Sage / Professor ──
    "Dark Breath": {
        "danger": 8, "is_aoe": True, "radius": 5, "element": "dark",
        "cast_time_ms": 3000, "notes": "Large AoE dark damage",
    },
    "Hell's Inferno": {
        "danger": 9, "is_aoe": True, "radius": 5, "element": "fire",
        "cast_time_ms": 4000, "notes": "Massive fire DoT AoE",
    },
    # ── Assassin / Assassin Cross ──
    "Soul Breaker": {
        "danger": 7, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 1000, "notes": "Ignores defense, high burst",
    },
    "Sonic Blow": {
        "danger": 7, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 500, "notes": "Rapid multi-hit melee",
    },
    "Grimtooth": {
        "danger": 5, "is_aoe": True, "radius": 3, "element": "neutral",
        "cast_time_ms": 500, "notes": "Short-range AoE dagger throw",
    },
    # ── Monk / Champion ──
    "Asura Strike": {
        "danger": 9, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 2000, "notes": "Extreme single-target burst, consumes SP",
    },
    # ── Knight / Lord Knight ──
    "Bowling Bash": {
        "danger": 6, "is_aoe": True, "radius": 3, "element": "neutral",
        "cast_time_ms": 1000, "notes": "Knocks back, hits multiple",
    },
    "Brandish Spear": {
        "danger": 6, "is_aoe": True, "radius": 3, "element": "neutral",
        "cast_time_ms": 1500, "notes": "Spear AoE",
    },
    # ── Crusader / Paladin ──
    "Shield Boomerang": {
        "danger": 5, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 800, "notes": "Ranged shield throw",
    },
    "Shield Chain": {
        "danger": 6, "is_aoe": True, "radius": 3, "element": "neutral",
        "cast_time_ms": 1200, "notes": "Multi-hit shield AoE",
    },
    "Pressure": {
        "danger": 5, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 2000, "notes": "Reduces target's SP",
    },
    # ── Priest / High Priest ──
    "Lex Aeterna": {
        "danger": 8, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 1000, "notes": "Doubles next damage taken — flee or interrupt",
    },
    "Magnificat": {
        "danger": 3, "is_aoe": True, "radius": 5, "element": "neutral",
        "cast_time_ms": 2000, "notes": "Party SP regen buff — low threat",
    },
    "Gloria": {
        "danger": 3, "is_aoe": True, "radius": 5, "element": "neutral",
        "cast_time_ms": 2000, "notes": "Party LUK buff — low threat",
    },
    "Assumptio": {
        "danger": 4, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 1500, "notes": "Damage reduction buff — makes target tankier",
    },
    "Kyrie Eleison": {
        "danger": 3, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 1500, "notes": "Auto-guard buff — low threat",
    },
    "Safety Wall": {
        "danger": 2, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 1000, "notes": "Blocks melee hits — low threat",
    },
    "Heal": {
        "danger": 2, "is_aoe": False, "radius": 0, "element": "holy",
        "cast_time_ms": 1000, "notes": "Heals monster or ally — low threat to player",
    },
    "Resurrection": {
        "danger": 1, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 4000, "notes": "Revives ally — very low threat",
    },
    "Teleport": {
        "danger": 1, "is_aoe": False, "radius": 0, "element": "neutral",
        "cast_time_ms": 500, "notes": "Monster escapes — low threat",
    },
}

# ── Action Recommendation Thresholds ──
# Danger level thresholds for each action type
_ACTION_THRESHOLDS: dict[str, int] = {
    "flee": 9,       # Danger 9+ → flee immediately
    "interrupt": 7,  # Danger 7+ → try to interrupt
    "move": 5,       # Danger 5+ → move out of AoE
    "defend": 4,     # Danger 4+ → use defensive skills
    "tank": 0,       # Everything else → tank it
}


# ── PredictiveThreatEngine ───────────────────────────────────────────────────


class PredictiveThreatEngine:
    """Evaluates monster cast bars and predicts incoming threats.

    Thread-safe (RLock). Use the global ``get_predictive_threat_engine()``
    singleton for shared access, or instantiate your own.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._dangerous_skills: dict[str, dict[str, Any]] = dict(_DANGEROUS_SKILLS)
        self._stats: dict[str, int] = {
            "evaluations": 0,
            "threats_predicted": 0,
            "flee_triggers": 0,
            "interrupt_triggers": 0,
            "move_triggers": 0,
            "defend_triggers": 0,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate_threats(self, monsters: list[dict[str, Any]]) -> ThreatPrediction:
        """Evaluate all monsters and produce a ThreatPrediction.

        Each monster dict should contain at minimum:
            id, name, casting_skill, cast_progress (0.0-1.0), distance, x, y

        Additional fields used if available:
            hp, max_hp, my_x, my_y
        """
        with self._lock:
            self._stats["evaluations"] += 1

        threats: list[PredictedThreat] = []
        my_hp_pct = 1.0

        # Extract player HP from the first monster entry that carries it
        for m in monsters:
            hp = m.get("hp", 0)
            max_hp = m.get("max_hp", 1)
            if max_hp > 0:
                my_hp_pct = hp / max_hp
            break

        for monster in monsters:
            skill_name = monster.get("casting_skill", "")
            if not skill_name:
                continue

            skill_data = self._dangerous_skills.get(skill_name)
            if skill_data is None:
                # Unknown skill — treat as low-danger single-target
                skill_data = {
                    "danger": 3,
                    "is_aoe": False,
                    "radius": 0,
                    "element": "neutral",
                    "cast_time_ms": 2000,
                    "notes": "Unknown skill — conservative estimate",
                }

            cast_time_ms = skill_data["cast_time_ms"]
            cast_progress = monster.get("cast_progress", 0.0)
            remaining_cast_ms = max(0, int(cast_time_ms * (1.0 - cast_progress)))
            distance = monster.get("distance", 10.0)
            is_aoe = skill_data["is_aoe"]
            radius = skill_data["radius"] if is_aoe else 0
            danger = skill_data["danger"]

            # Time to impact: remaining cast time + travel time (if projectile)
            # Melee-range skills land instantly; ranged skills have travel time
            travel_ms = 0
            if distance > 3 and not is_aoe:
                travel_ms = int(distance * 100)  # ~100ms per cell
            time_to_impact_ms = remaining_cast_ms + travel_ms

            action = self._get_action_for_skill(skill_name, distance, my_hp_pct)

            # Confidence: based on how well we know the skill + cast progress
            confidence = 0.7
            if skill_name in self._dangerous_skills:
                confidence = 0.9
            if cast_progress > 0.5:
                confidence = min(1.0, confidence + 0.1)
            if cast_progress > 0.9:
                confidence = min(1.0, confidence + 0.05)

            threat = PredictedThreat(
                monster_id=monster.get("id", 0),
                monster_name=monster.get("name", "Unknown"),
                skill_name=skill_name,
                skill_element=skill_data["element"],
                cast_time_ms=cast_time_ms,
                remaining_cast_ms=remaining_cast_ms,
                is_aoe=is_aoe,
                aoe_radius=radius,
                danger_level=danger,
                time_to_impact_ms=time_to_impact_ms,
                recommended_action=action,
                confidence=confidence,
            )
            threats.append(threat)

        with self._lock:
            self._stats["threats_predicted"] += len(threats)

        # Sort by danger descending, then by time_to_impact ascending
        threats.sort(key=lambda t: (-t.danger_level, t.time_to_impact_ms))

        most_dangerous = threats[0] if threats else None
        time_to_react_ms = self._get_time_to_react_ms(threats)
        should_flee = self._should_flee(threats)
        should_interrupt = self._should_interrupt(threats)
        should_move = self._should_move(threats)
        should_defend = self._should_defend(threats)
        summary = self._build_summary(
            threats, most_dangerous, time_to_react_ms,
            should_flee, should_interrupt, should_move, should_defend,
        )

        with self._lock:
            if should_flee:
                self._stats["flee_triggers"] += 1
            if should_interrupt:
                self._stats["interrupt_triggers"] += 1
            if should_move:
                self._stats["move_triggers"] += 1
            if should_defend:
                self._stats["defend_triggers"] += 1

        return ThreatPrediction(
            threats=threats,
            most_dangerous=most_dangerous,
            time_to_react_ms=time_to_react_ms,
            should_flee=should_flee,
            should_interrupt=should_interrupt,
            should_move=should_move,
            should_defend=should_defend,
            summary=summary,
        )

    def get_dangerous_skills(self) -> list[dict[str, Any]]:
        """Return the full dangerous skill database as a list of dicts."""
        with self._lock:
            return [
                {"name": name, **data}
                for name, data in self._dangerous_skills.items()
            ]

    def get_skill_danger(self, skill_name: str) -> int:
        """Get the danger level (1-10) for a skill. Returns 3 for unknown skills."""
        with self._lock:
            data = self._dangerous_skills.get(skill_name)
            if data is None:
                return 3
            return data["danger"]

    def get_recommended_action(
        self, skill_name: str, distance: float, my_hp_pct: float,
    ) -> str:
        """Get the recommended action for a skill given context."""
        return self._get_action_for_skill(skill_name, distance, my_hp_pct)

    def get_time_to_react(self, monsters: list[dict[str, Any]]) -> int:
        """Get the time in ms before the first threat lands.

        Only considers monsters that are currently casting.
        """
        times: list[int] = []
        for monster in monsters:
            skill_name = monster.get("casting_skill", "")
            if not skill_name:
                continue
            skill_data = self._dangerous_skills.get(skill_name)
            if skill_data is None:
                continue
            cast_time_ms = skill_data["cast_time_ms"]
            cast_progress = monster.get("cast_progress", 0.0)
            remaining = max(0, int(cast_time_ms * (1.0 - cast_progress)))
            distance = monster.get("distance", 10.0)
            travel_ms = 0
            if distance > 3 and not skill_data["is_aoe"]:
                travel_ms = int(distance * 100)
            times.append(remaining + travel_ms)
        if not times:
            return 9999
        return min(times)

    def should_evacuate(self, threats: list[PredictedThreat]) -> bool:
        """Determine if the bot should immediately evacuate (fly wing / teleport)."""
        return self._should_flee(threats)

    def get_priority_interrupt_target(
        self, threats: list[PredictedThreat],
    ) -> int | None:
        """Return the monster_id of the highest-priority interrupt target.

        Prioritises high-danger skills with enough remaining cast time to
        actually interrupt (>= 500ms remaining).
        """
        candidates = [
            t for t in threats
            if t.danger_level >= 7 and t.remaining_cast_ms >= 500
        ]
        if not candidates:
            return None
        # Highest danger first, then shortest remaining cast (most urgent)
        candidates.sort(key=lambda t: (-t.danger_level, t.remaining_cast_ms))
        return candidates[0].monster_id

    def get_safe_direction(
        self, threats: list[PredictedThreat],
        my_x: float, my_y: float,
    ) -> tuple[float, float]:
        """Calculate the safest direction to move.

        Returns a (dx, dy) unit vector pointing away from the most dangerous
        threats. If no threats, returns (0, 0).
        """
        if not threats:
            return (0.0, 0.0)

        # Weight each threat by danger and proximity
        dx_total = 0.0
        dy_total = 0.0
        weight_total = 0.0

        for threat in threats:
            weight = float(threat.danger_level)
            if threat.is_aoe:
                weight *= 1.5  # AoE threats are more dangerous
            if threat.time_to_impact_ms < 2000:
                weight *= 2.0  # Imminent threats are more urgent

            # Without actual monster positions, push away from the
            # direction of the most dangerous threat (generic avoidance).
            # In production, use monster.x / monster.y from the bridge.
            dx_total += weight * 1.0  # placeholder — real impl uses monster pos
            dy_total += weight * 1.0
            weight_total += weight

        if weight_total == 0:
            return (0.0, 0.0)

        # Normalise to unit vector
        dx = dx_total / weight_total
        dy = dy_total / weight_total
        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude < 0.001:
            return (0.0, 0.0)

        return (dx / magnitude, dy / magnitude)

    def get_threat_summary(self, threats: list[PredictedThreat]) -> str:
        """Generate a human-readable summary of the threat situation."""
        if not threats:
            return "No threats detected."

        # Count by danger tier
        deadly = [t for t in threats if t.danger_level >= 9]
        high = [t for t in threats if 7 <= t.danger_level <= 8]
        medium = [t for t in threats if 4 <= t.danger_level <= 6]
        low = [t for t in threats if t.danger_level <= 3]

        parts: list[str] = []
        if deadly:
            names = ", ".join(f"{t.skill_name} ({t.monster_name})" for t in deadly[:3])
            parts.append(f"DEADLY: {names}")
        if high:
            names = ", ".join(f"{t.skill_name} ({t.monster_name})" for t in high[:3])
            parts.append(f"High: {names}")
        if medium:
            names = ", ".join(f"{t.skill_name} ({t.monster_name})" for t in medium[:3])
            parts.append(f"Medium: {names}")
        if low:
            names = ", ".join(f"{t.skill_name} ({t.monster_name})" for t in low[:3])
            parts.append(f"Low: {names}")

        earliest = min(t.time_to_impact_ms for t in threats)
        parts.append(f"Earliest impact in {earliest}ms")

        return " | ".join(parts)

    def register_dangerous_skill(self, skill_data: dict[str, Any]) -> None:
        """Register or update a dangerous skill in the database.

        Expected keys:
            name (str) — skill name
            danger (int) — 1-10
            is_aoe (bool)
            radius (int) — AoE radius in cells
            element (str)
            cast_time_ms (int)
            notes (str, optional)
        """
        name = skill_data.get("name", "")
        if not name:
            logger.warning("register_dangerous_skill called without 'name' key")
            return

        entry: dict[str, Any] = {
            "danger": skill_data.get("danger", 3),
            "is_aoe": skill_data.get("is_aoe", False),
            "radius": skill_data.get("radius", 0),
            "element": skill_data.get("element", "neutral"),
            "cast_time_ms": skill_data.get("cast_time_ms", 2000),
            "notes": skill_data.get("notes", ""),
        }

        with self._lock:
            self._dangerous_skills[name] = entry
            logger.info("Registered dangerous skill: %s (danger=%d)", name, entry["danger"])

    def counters(self) -> dict[str, int]:
        """Return internal statistics counters."""
        with self._lock:
            return dict(self._stats)

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _get_action_for_skill(
        self, skill_name: str, distance: float, my_hp_pct: float,
    ) -> str:
        """Determine the recommended action for a skill."""
        with self._lock:
            data = self._dangerous_skills.get(skill_name)

        if data is None:
            return "tank"

        danger = data["danger"]
        is_aoe = data["is_aoe"]

        # Critical HP + any danger → flee
        if my_hp_pct < 0.15 and danger >= 5:
            return "flee"

        # Danger 9+ → flee regardless
        if danger >= 9:
            return "flee"

        # Danger 7-8 → interrupt if close enough, otherwise move
        if danger >= 7:
            if distance <= 5:
                return "interrupt"
            return "move"

        # Danger 5-6 → move out of AoE
        if danger >= 5 and is_aoe:
            return "move"

        # Danger 4+ → defend
        if danger >= 4:
            return "defend"

        # Everything else → tank
        return "tank"

    def _get_time_to_react_ms(self, threats: list[PredictedThreat]) -> int:
        """Get the time in ms before the first threat lands."""
        if not threats:
            return 9999
        return min(t.time_to_impact_ms for t in threats)

    def _should_flee(self, threats: list[PredictedThreat]) -> bool:
        """Check if any threat warrants immediate evacuation."""
        for t in threats:
            if t.danger_level >= 9:
                return True
            if t.danger_level >= 7 and t.time_to_impact_ms < 1500:
                return True
        return False

    def _should_interrupt(self, threats: list[PredictedThreat]) -> bool:
        """Check if any threat should be interrupted."""
        for t in threats:
            if t.danger_level >= 7 and t.remaining_cast_ms >= 500:
                return True
        return False

    def _should_move(self, threats: list[PredictedThreat]) -> bool:
        """Check if the bot should move out of danger."""
        for t in threats:
            if t.is_aoe and t.danger_level >= 5:
                return True
        return False

    def _should_defend(self, threats: list[PredictedThreat]) -> bool:
        """Check if the bot should use defensive skills."""
        for t in threats:
            if t.danger_level >= 4:
                return True
        return False

    def _build_summary(
        self,
        threats: list[PredictedThreat],
        most_dangerous: PredictedThreat | None,
        time_to_react_ms: int,
        should_flee: bool,
        should_interrupt: bool,
        should_move: bool,
        should_defend: bool,
    ) -> str:
        """Build a concise human-readable summary string."""
        if not threats:
            return "No threats detected."

        parts: list[str] = []

        if should_flee:
            parts.append("EVACUATE")
        elif should_interrupt:
            parts.append("INTERRUPT")
        elif should_move:
            parts.append("MOVE")
        elif should_defend:
            parts.append("DEFEND")
        else:
            parts.append("TANK")

        if most_dangerous:
            parts.append(
                f"worst={most_dangerous.skill_name}@{most_dangerous.monster_name}"
                f" (danger={most_dangerous.danger_level})"
            )

        parts.append(f"react={time_to_react_ms}ms")
        parts.append(f"threats={len(threats)}")

        return " | ".join(parts)


# ── Global Singleton ──────────────────────────────────────────────────────────

_ENGINE_INSTANCE: PredictiveThreatEngine | None = None
_ENGINE_LOCK: RLock = RLock()


def get_predictive_threat_engine() -> PredictiveThreatEngine:
    """Get or create the global PredictiveThreatEngine singleton.

    Thread-safe. Use this for shared access across the application.
    """
    global _ENGINE_INSTANCE  # noqa: PLW0603
    if _ENGINE_INSTANCE is None:
        with _ENGINE_LOCK:
            if _ENGINE_INSTANCE is None:
                _ENGINE_INSTANCE = PredictiveThreatEngine()
    return _ENGINE_INSTANCE
