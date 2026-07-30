"""Emperium Mechanics — Emperium breaking logic, damage calculations, class-specific break times.

The Emperium is the heart of WoE — a crystal that must be destroyed to claim a castle.
Key mechanics:
- Emperium is Holy element level 4
- Needs Neutral level 4 or Holy weapon to damage it
- Wizards break in ~5s with Heaven's Drive
- Assassins break in ~15s with auto-attack
- Champions break in ~3s with Asura Strike
- Paladins can't damage it (no neutral/holy weapon)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Emperium stats (standard rAthena)
EMPERIUM_HP: int = 2000000       # 2M HP on most servers
EMPERIUM_DEF: int = 40           # Defense
EMPERIUM_MDEF: int = 30          # Magic defense
EMPERIUM_ELEMENT: str = "holy"   # Holy element level 4
EMPERIUM_SIZE: str = "large"
EMPERIUM_RACE: str = "demon"

# Classes that CAN damage the Emperium
EMPERIUM_BREAKER_CLASSES: set[str] = {
    "assassin", "assassin_cross", "guillotine_cross",
    "rogue", "stalker", "shadow_chaser",
    "monk", "champion", "sura",
    "wizard", "high_wizard", "warlock",
    "sage", "professor", "sorcerer",
    "lord_knight", "rune_knight",
    "whitesmith", "mechanic",
    "sniper", "ranger",
    "clown", "minstrel",
    "gypsy", "wanderer",
    "gunslinger", "rebellion",
    "ninja", "kagerou", "oboro",
    "soul_linker",
}

# Classes that CANNOT damage the Emperium (no neutral/holy weapon)
EMPERIUM_NON_BREAKER_CLASSES: set[str] = {
    "priest", "high_priest", "arch_bishop",
    "paladin", "royal_guard",
    "crusader",
    "alchemist", "creator", "genetic",
    "blacksmith",
    "taekwon",
    "super_novice",
}

# Class-specific DPS estimates against Emperium (damage per second)
# Based on standard rAthena gear (no MVP gear)
CLASS_EMP_DPS: dict[str, float] = {
    "assassin": 2200.0,          # Double Attack + high ASPD
    "assassin_cross": 2800.0,   # Soul Destroyer + high ASPD
    "guillotine_cross": 3500.0, # Cross Impact + high ASPD
    "rogue": 1800.0,            # Backstab on EMP? No, but decent auto
    "stalker": 2200.0,
    "shadow_chaser": 2600.0,
    "monk": 1500.0,             # Auto-attack (Asura is burst)
    "champion": 2000.0,         # Auto-attack
    "sura": 2500.0,             # Auto-attack
    "wizard": 8000.0,           # Heaven's Drive (AoE, hits EMP 3x)
    "high_wizard": 10000.0,     # Heaven's Drive max level
    "warlock": 12000.0,         # Crimson Rock / Heaven's Drive
    "sage": 5000.0,             # Auto-attack + bolts
    "professor": 6000.0,
    "sorcerer": 8000.0,
    "lord_knight": 3000.0,     # Bowling Bash / auto
    "rune_knight": 4000.0,
    "whitesmith": 2500.0,
    "mechanic": 3500.0,
    "sniper": 3000.0,           # Double Strafe / auto
    "ranger": 4000.0,
    "clown": 2000.0,
    "minstrel": 2500.0,
    "gypsy": 2000.0,
    "wanderer": 2500.0,
    "gunslinger": 3500.0,      # Full Buster / auto
    "rebellion": 4000.0,
    "ninja": 3000.0,
    "kagerou": 3500.0,
    "oboro": 3500.0,
    "soul_linker": 2000.0,
}

# Class-specific burst skills against Emperium
CLASS_EMP_BURST_SKILLS: dict[str, list[dict[str, Any]]] = {
    "assassin": [{"skill": "soul_destroyer", "damage": 50000, "sp_cost": 30, "cast_time": 1.0}],
    "assassin_cross": [{"skill": "soul_destroyer", "damage": 80000, "sp_cost": 35, "cast_time": 1.0}],
    "guillotine_cross": [{"skill": "cross_impact", "damage": 120000, "sp_cost": 40, "cast_time": 1.5}],
    "monk": [{"skill": "asura_strike", "damage": 300000, "sp_cost": 100, "cast_time": 2.0}],
    "champion": [{"skill": "asura_strike", "damage": 500000, "sp_cost": 120, "cast_time": 2.0}],
    "sura": [{"skill": "asura_strike", "damage": 700000, "sp_cost": 150, "cast_time": 2.0}],
    "wizard": [{"skill": "heavens_drive", "damage": 15000, "sp_cost": 40, "cast_time": 3.0, "aoe": True}],
    "high_wizard": [{"skill": "heavens_drive", "damage": 25000, "sp_cost": 50, "cast_time": 3.0, "aoe": True}],
    "warlock": [{"skill": "crimson_rock", "damage": 40000, "sp_cost": 60, "cast_time": 4.0, "aoe": True}],
    "lord_knight": [{"skill": "bowling_bash", "damage": 20000, "sp_cost": 20, "cast_time": 1.0}],
    "rune_knight": [{"skill": "sonic_wave", "damage": 35000, "sp_cost": 30, "cast_time": 1.5}],
    "sniper": [{"skill": "sharp_shooting", "damage": 25000, "sp_cost": 25, "cast_time": 1.5}],
    "ranger": [{"skill": "aimed_bolt", "damage": 40000, "sp_cost": 35, "cast_time": 2.0}],
    "gunslinger": [{"skill": "full_buster", "damage": 30000, "sp_cost": 30, "cast_time": 1.5}],
    "rebellion": [{"skill": "fire_rain", "damage": 45000, "sp_cost": 40, "cast_time": 2.0}],
}

# Best skill for EMP breaking per class
CLASS_EMP_BEST_SKILL: dict[str, str] = {
    "assassin": "soul_destroyer",
    "assassin_cross": "soul_destroyer",
    "guillotine_cross": "cross_impact",
    "monk": "asura_strike",
    "champion": "asura_strike",
    "sura": "asura_strike",
    "wizard": "heavens_drive",
    "high_wizard": "heavens_drive",
    "warlock": "crimson_rock",
    "lord_knight": "bowling_bash",
    "rune_knight": "sonic_wave",
    "sniper": "sharp_shooting",
    "ranger": "aimed_bolt",
    "gunslinger": "full_buster",
    "rebellion": "fire_rain",
}


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class EmperiumBreakEstimate:
    """Estimated time to break the Emperium."""
    class_name: str = ""
    dps: float = 0.0
    estimated_seconds: float = 0.0
    estimated_seconds_formatted: str = ""
    can_break: bool = True
    best_skill: str = ""
    burst_damage: int = 0
    burst_sp_cost: int = 0
    strategy: str = ""


@dataclass
class EmperiumState:
    """Current Emperium state during WoE."""
    alive: bool = True
    hp: int = EMPERIUM_HP
    max_hp: int = EMPERIUM_HP
    hp_pct: float = 1.0
    last_update: float = 0.0
    attackers_nearby: int = 0
    defenders_nearby: int = 0
    damage_dealt_by_us: int = 0
    damage_dealt_by_enemy: int = 0
    last_threshold_called: float = 1.0


# ── Emperium Mechanics Engine ─────────────────────────────────────────────

class EmperiumMechanics:
    """Emperium breaking logic — damage calculations, class-specific break times.

    Features:
      - Calculate break time for any class
      - Best skill recommendation per class
      - Burst damage calculations
      - Emperium state tracking
      - HP threshold callouts
      - Class-specific break strategies
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._state: EmperiumState = EmperiumState()

    # ── Public API ────────────────────────────────────────────────────

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Assess Emperium state and emit break actions."""
        if not signals:
            return

        with self._lock:
            # Update Emperium state from signals
            self._update_state(signals)

            if not self._state.alive:
                return

            my_job = str(signals.get("job_name", "novice") or "novice").lower()

            # Check if we can break
            estimate = self.calculate_break_time(my_job)
            if not estimate.can_break:
                actions.append(HeuristicAction(
                    kind="log",
                    command="emperium_cannot_break",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"{my_job} cannot damage Emperium (needs Neutral 4 or Holy weapon)",
                    metadata={"class": my_job, "can_break": False},
                ))
                return

            # Emit break strategy
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack_emperium {estimate.best_skill}",
                confidence=0.95,
                domain="pvp",
                reason=f"Breaking EMP: {estimate.estimated_seconds_formatted} with {estimate.best_skill} ({estimate.dps:.0f} DPS)",
                metadata={
                    "class": my_job,
                    "dps": estimate.dps,
                    "estimated_seconds": estimate.estimated_seconds,
                    "best_skill": estimate.best_skill,
                    "strategy": estimate.strategy,
                },
            ))

            # Check HP thresholds
            self._check_thresholds(actions, bot_id)

    def calculate_break_time(self, class_name: str) -> EmperiumBreakEstimate:
        """Calculate estimated time to break Emperium for a given class."""
        class_lower = class_name.lower()

        # Check if class can break
        if class_lower in EMPERIUM_NON_BREAKER_CLASSES:
            return EmperiumBreakEstimate(
                class_name=class_name,
                can_break=False,
                strategy="Cannot damage Emperium — support role only",
            )

        # Get DPS
        dps = CLASS_EMP_DPS.get(class_lower, 1000.0)
        if dps <= 0:
            dps = 1000.0

        # Calculate time
        current_hp = self._state.hp
        estimated_seconds = current_hp / dps if dps > 0 else float("inf")

        # Format time
        if estimated_seconds < 60:
            formatted = f"{estimated_seconds:.0f}s"
        else:
            formatted = f"{estimated_seconds / 60:.1f}min"

        # Get best skill
        best_skill = CLASS_EMP_BEST_SKILL.get(class_lower, "auto_attack")

        # Get burst info
        burst_skills = CLASS_EMP_BURST_SKILLS.get(class_lower, [])
        burst_damage = burst_skills[0]["damage"] if burst_skills else 0
        burst_sp_cost = burst_skills[0]["sp_cost"] if burst_skills else 0

        # Strategy
        if estimated_seconds < 10:
            strategy = "Fast break — spam best skill"
        elif estimated_seconds < 30:
            strategy = "Medium break — use burst skills, manage SP"
        elif estimated_seconds < 60:
            strategy = "Slow break — sustain DPS, watch for defenders"
        else:
            strategy = "Very slow break — need more DPS or help"

        return EmperiumBreakEstimate(
            class_name=class_name,
            dps=dps,
            estimated_seconds=estimated_seconds,
            estimated_seconds_formatted=formatted,
            can_break=True,
            best_skill=best_skill,
            burst_damage=burst_damage,
            burst_sp_cost=burst_sp_cost,
            strategy=strategy,
        )

    def get_break_time_for_party(
        self, party_classes: list[str]
    ) -> dict[str, Any]:
        """Calculate combined break time for a party."""
        total_dps = 0.0
        breakers = 0
        non_breakers = 0

        for cls in party_classes:
            estimate = self.calculate_break_time(cls)
            if estimate.can_break:
                total_dps += estimate.dps
                breakers += 1
            else:
                non_breakers += 1

        current_hp = self._state.hp
        combined_time = current_hp / total_dps if total_dps > 0 else float("inf")

        return {
            "total_dps": total_dps,
            "breakers": breakers,
            "non_breakers": non_breakers,
            "combined_break_time_s": combined_time,
            "combined_break_time": f"{combined_time:.0f}s" if combined_time < 60 else f"{combined_time / 60:.1f}min",
            "emperium_hp": current_hp,
            "emperium_hp_pct": self._state.hp_pct,
        }

    def get_state(self) -> EmperiumState:
        """Get current Emperium state."""
        with self._lock:
            return self._state

    def update_emperium_hp(self, hp: int, max_hp: int) -> None:
        """Update Emperium HP from external signal."""
        with self._lock:
            self._state.hp = hp
            self._state.max_hp = max_hp
            self._state.hp_pct = hp / max_hp if max_hp > 0 else 1.0
            self._state.alive = hp > 0
            self._state.last_update = time.time()

    # ── Internal ─────────────────────────────────────────────────────

    def _update_state(self, signals: dict[str, Any]) -> None:
        """Update Emperium state from signals."""
        emp_hp = signals.get("emperium_hp")
        emp_max_hp = signals.get("emperium_max_hp")

        if emp_hp is not None and emp_max_hp and emp_max_hp > 0:
            self._state.hp = int(emp_hp)
            self._state.max_hp = int(emp_max_hp)
            self._state.hp_pct = emp_hp / emp_max_hp
            self._state.alive = emp_hp > 0
            self._state.last_update = time.time()

        # Track nearby players
        players = signals.get("players", []) or []
        my_guild = str(signals.get("guild_name", "") or "")
        attackers = 0
        defenders = 0
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
                if pg and pg != my_guild:
                    attackers += 1
                elif pg == my_guild:
                    defenders += 1
        self._state.attackers_nearby = attackers
        self._state.defenders_nearby = defenders

    def _check_thresholds(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Check Emperium HP thresholds and emit callouts."""
        thresholds = [
            (0.75, "EMP at 75% — keep pushing!"),
            (0.50, "EMP at 50% — focus damage!"),
            (0.25, "EMP at 25% — burn it down!"),
            (0.10, "EMP at 10% — finish it!"),
            (0.05, "EMP at 5% — almost dead!"),
        ]

        for threshold, message in sorted(thresholds, reverse=True):
            if self._state.hp_pct <= threshold and self._state.last_threshold_called > threshold:
                self._state.last_threshold_called = threshold
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"party_message {message}",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"EMP HP threshold: {message}",
                    metadata={"emp_hp_pct": self._state.hp_pct, "threshold": threshold},
                ))
                break


# ── Singleton factory ─────────────────────────────────────────────────────

_emperium_mechanics: EmperiumMechanics | None = None


def get_emperium_mechanics() -> EmperiumMechanics:
    """Get or create the singleton EmperiumMechanics."""
    global _emperium_mechanics
    if _emperium_mechanics is None:
        _emperium_mechanics = EmperiumMechanics()
    return _emperium_mechanics
