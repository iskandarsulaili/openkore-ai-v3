"""
Combat instinct engine — reads combat context, not just HP numbers.

A pro player FEELS combat. They know WHY the HP dropped:
- Was it a skill cast? (read the cast bar)
- Was it an AoE? (check position)
- Was it a crit? (check damage spike pattern)
- Was it a DoT? (check debuff)
- Was it a multi-hit? (check attack speed)

This module analyzes combat events to determine the CAUSE of damage,
not just the fact that damage occurred.

Fixed by Pro RO Player: added monster skill awareness, multi-hit detection,
element tracking, and proper threat escalation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CombatEvent:
    timestamp: float
    event_type: str  # damage_taken, damage_dealt, skill_cast, skill_hit, debuff_applied, heal_received
    source: str  # monster name or skill name
    value: int
    element: str = "neutral"
    is_aoe: bool = False
    is_crit: bool = False
    is_dot: bool = False


# ── Known dangerous monster skills (pre-renewal) ──
# A pro player knows which skills to dodge and which to tank.
DANGEROUS_MONSTER_SKILLS: dict[str, dict[str, Any]] = {
    # AoE skills — MUST dodge
    "WZ_STORMGUST": {"name": "Storm Gust", "element": "water", "aoe": True, "danger": "critical", "action": "dodge"},
    "WZ_METEORSTORM": {"name": "Meteor Storm", "element": "fire", "aoe": True, "danger": "critical", "action": "dodge"},
    "WZ_HEAVENDRIVE": {"name": "Heaven's Drive", "element": "holy", "aoe": True, "danger": "high", "action": "dodge"},
    "MG_THUNDERSTORM": {"name": "Thunderstorm", "element": "wind", "aoe": True, "danger": "high", "action": "dodge"},
    "WZ_VERMILION": {"name": "Lord of Vermillion", "element": "wind", "aoe": True, "danger": "critical", "action": "dodge"},
    "WZ_QUAGMIRE": {"name": "Quagmire", "element": "earth", "aoe": True, "danger": "medium", "action": "dodge"},
    "WZ_FIREPILLAR": {"name": "Fire Pillar", "element": "fire", "aoe": True, "danger": "high", "action": "dodge"},
    "WZ_ICEWALL": {"name": "Ice Wall", "element": "water", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_HELLJUDGEMENT": {"name": "Hell's Judgement", "element": "dark", "aoe": True, "danger": "critical", "action": "dodge"},
    "NPC_WIDEWEB": {"name": "Wide Web", "element": "neutral", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_WIDECURSE": {"name": "Wide Curse", "element": "neutral", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_WIDESTUN": {"name": "Wide Stun", "element": "neutral", "aoe": True, "danger": "high", "action": "dodge"},
    "NPC_WIDESLEEP": {"name": "Wide Sleep", "element": "neutral", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_WIDECONFUSE": {"name": "Wide Confuse", "element": "neutral", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_WIDEFREEZE": {"name": "Wide Freeze", "element": "water", "aoe": True, "danger": "high", "action": "dodge"},
    "NPC_WIDESILENCE": {"name": "Wide Silence", "element": "neutral", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_WIDEBLEEDING": {"name": "Wide Bleeding", "element": "neutral", "aoe": True, "danger": "medium", "action": "dodge"},
    "NPC_PULSESTRIKE": {"name": "Pulse Strike", "element": "neutral", "aoe": True, "danger": "critical", "action": "dodge"},
    "NPC_EARTHQUAKE": {"name": "Earthquake", "element": "earth", "aoe": True, "danger": "critical", "action": "dodge"},
    "NPC_DARKBREATH": {"name": "Dark Breath", "element": "dark", "aoe": True, "danger": "critical", "action": "dodge"},
    "NPC_FIREBREATH": {"name": "Fire Breath", "element": "fire", "aoe": True, "danger": "high", "action": "dodge"},
    "NPC_THUNDERBREATH": {"name": "Thunder Breath", "element": "wind", "aoe": True, "danger": "high", "action": "dodge"},
    "NPC_ACIDBREATH": {"name": "Acid Breath", "element": "poison", "aoe": True, "danger": "high", "action": "dodge"},
    # Single-target skills — pot up
    "MG_FIREBOLT": {"name": "Fire Bolt", "element": "fire", "aoe": False, "danger": "medium", "action": "pot"},
    "MG_COLDBOLT": {"name": "Cold Bolt", "element": "water", "aoe": False, "danger": "medium", "action": "pot"},
    "MG_LIGHTNINGBOLT": {"name": "Lightning Bolt", "element": "wind", "aoe": False, "danger": "medium", "action": "pot"},
    "MG_NAPARMBEAT": {"name": "Napalm Beat", "element": "neutral", "aoe": False, "danger": "low", "action": "pot"},
    "MG_SOULSTRIKE": {"name": "Soul Strike", "element": "neutral", "aoe": False, "danger": "low", "action": "pot"},
    "NPC_POISON": {"name": "Poison Attack", "element": "poison", "aoe": False, "danger": "medium", "action": "cure"},
    "NPC_BLIND": {"name": "Blind Attack", "element": "neutral", "aoe": False, "danger": "low", "action": "cure"},
    "NPC_SILENCE": {"name": "Silence Attack", "element": "neutral", "aoe": False, "danger": "medium", "action": "cure"},
    "NPC_CURSE": {"name": "Curse Attack", "element": "neutral", "aoe": False, "danger": "medium", "action": "cure"},
    "NPC_STUN": {"name": "Stun Attack", "element": "neutral", "aoe": False, "danger": "high", "action": "cure"},
    "NPC_SLEEP": {"name": "Sleep Attack", "element": "neutral", "aoe": False, "danger": "low", "action": "cure"},
    "NPC_CONFUSION": {"name": "Confusion Attack", "element": "neutral", "aoe": False, "danger": "low", "action": "cure"},
    "NPC_FREEZE": {"name": "Freeze Attack", "element": "water", "aoe": False, "danger": "high", "action": "cure"},
}


@dataclass(slots=True)
class CombatInstinctEngine:
    """Reads combat context to determine WHY damage occurred."""

    _lock: RLock = field(default_factory=RLock)
    _event_history: dict[str, deque[CombatEvent]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=50)))
    _monster_skill_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _last_cast_seen: dict[str, str] = field(default_factory=dict)  # bot_id -> last skill cast by monster
    _stats: dict[str, int] = field(default_factory=lambda: {"events_processed": 0, "instinct_triggers": 0})

    def record_event(self, bot_id: str, event: CombatEvent) -> None:
        with self._lock:
            self._event_history[bot_id].append(event)
            self._stats["events_processed"] += 1

    def analyze_damage(self, bot_id: str, hp_drop: int, current_hp: int, max_hp: int,
                       nearby_monsters: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze WHY damage occurred and recommend response."""
        with self._lock:
            events = list(self._event_history.get(bot_id, []))

        result = {
            "cause": "unknown",
            "element": "neutral",
            "is_aoe": False,
            "is_crit": False,
            "is_dot": False,
            "is_multi_hit": False,
            "threat_level": "low",
            "recommendation": "continue",
            "evasive_action": None,
        }

        # Check recent events for context
        recent = [e for e in events[-10:] if time.time() - e.timestamp < 3.0]

        # Detect multi-hit (rapid consecutive damage events)
        damage_events = [e for e in recent if e.event_type == "damage_taken"]
        if len(damage_events) >= 3:
            time_span = damage_events[-1].timestamp - damage_events[0].timestamp
            if time_span < 1.0:
                result["is_multi_hit"] = True
                result["threat_level"] = "high"
                result["recommendation"] = "flee"
                result["evasive_action"] = "fly_wing"

        for event in reversed(recent):
            if event.event_type == "skill_cast":
                # Check if this is a known dangerous skill
                skill_info = DANGEROUS_MONSTER_SKILLS.get(event.source, {})
                if skill_info:
                    result["cause"] = f"skill:{event.source}"
                    result["element"] = skill_info.get("element", event.element)
                    result["is_aoe"] = skill_info.get("aoe", event.is_aoe)
                    danger = skill_info.get("danger", "medium")
                    result["threat_level"] = danger
                    action = skill_info.get("action", "pot")
                    if action == "dodge":
                        result["recommendation"] = "dodge"
                        result["evasive_action"] = "move"
                    elif action == "cure":
                        result["recommendation"] = "cure"
                        result["evasive_action"] = "use_green_potion"
                    else:
                        result["recommendation"] = "pot"
                        result["evasive_action"] = "use_potion"
                else:
                    # Unknown skill — treat as medium threat
                    result["cause"] = f"skill:{event.source}"
                    result["element"] = event.element
                    result["is_aoe"] = event.is_aoe
                    result["threat_level"] = "medium"
                    result["recommendation"] = "pot"
                    result["evasive_action"] = "use_potion"
                self._stats["instinct_triggers"] += 1
                break
            elif event.event_type == "debuff_applied":
                result["cause"] = f"debuff:{event.source}"
                result["recommendation"] = "cure"
                result["evasive_action"] = "use_green_potion"
                self._stats["instinct_triggers"] += 1
                break

        # Check if damage is lethal
        hp_pct = current_hp / max(max_hp, 1)
        if hp_pct < 0.2 and result["recommendation"] != "dodge":
            result["recommendation"] = "flee"
            result["evasive_action"] = "fly_wing"
            result["threat_level"] = "critical"

        return result

    def should_interrupt(self, bot_id: str, current_action: str,
                         monster_casting: str | None) -> bool:
        """Should the bot interrupt its current action to dodge?"""
        if monster_casting:
            skill_info = DANGEROUS_MONSTER_SKILLS.get(monster_casting, {})
            if skill_info.get("aoe") and skill_info.get("danger") in ("high", "critical"):
                return True
        return False

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
