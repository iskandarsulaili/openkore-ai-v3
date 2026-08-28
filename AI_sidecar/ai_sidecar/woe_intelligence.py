"""War of Emperium intelligence — guild wars, castle defense, emperium breaking.

A pro player lives for WoE. This module handles:
- WoE schedule awareness with pre-WoE preparation
- Castle defense/attack tactics with barricade management
- Emperium breaking strategy with class-specific DPS
- Guild coordination and chemistry tracking
- 50+ player battlefield awareness
- Escape scroll economy (Fly Wing vs Butterfly Wing vs Escape Scroll vs die)
- WoE consumable preparation
- Castle ownership tracking
- Class-vs-class counter strategies
- Battlefield threat assessment
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.woe.emperium_mechanics import (
    EmperiumMechanics,
    get_emperium_mechanics,
    EMPERIUM_BREAKER_CLASSES,
    EMPERIUM_NON_BREAKER_CLASSES,
    CLASS_EMP_DPS,
    CLASS_EMP_BEST_SKILL,
)
from ai_sidecar.woe.battlefield_awareness import (
    BattlefieldAwareness,
    get_battlefield_awareness,
    ThreatPriority,
    CLASS_THREAT_PRIORITY,
    RUN_THRESHOLD,
    FLY_WING_ID,
    BUTTERFLY_WING_ID,
)
from ai_sidecar.woe.castle_intel import (
    CastleIntelligence,
    get_castle_intelligence,
    WOE_CASTLES,
    WOE_CONSUMABLES,
    WOE_EQUIPMENT,
    BARRICADE_POSITIONS,
    BARRICADE_BYPASS,
)

logger = logging.getLogger(__name__)


# ── WoE constants ──────────────────────────────────────────────────────────

WOE_WAR_HOURS: tuple[int, ...] = (20, 21, 22)
WOE_WAR_DAYS: tuple[int, ...] = (3, 5, 6)  # Thursday, Saturday, Sunday

CASTLE_PREFIXES: tuple[str, ...] = (
    "gld_dun", "gld_castle", "gld_dun01", "gld_dun02",
    "gld_dun03", "gld_dun04",
)

EMPERIUM_MAP_FRAGMENTS: tuple[str, ...] = (
    "gld_dun04", "gld_dun03_",
    "aldeba_dun04", "ayotha_dun04",
    "gefg_dun04", "payg_dun04",
    "prtg_dun04",
)

# WoE map chokepoints and defensive positions
WOE_CHOKEPOINTS: dict[str, list[tuple[int, int]]] = {
    "gld_dun01": [(50, 50), (100, 100)],
    "gld_dun02": [(30, 30), (120, 80)],
    "gld_dun03": [(60, 60), (90, 90)],
    "gld_dun04": [(40, 40), (80, 80), (110, 110)],
    "prtg_cas01": [(50, 50), (100, 100), (150, 150)],
    "prtg_cas02": [(30, 30), (80, 80), (120, 120)],
    "prtg_cas03": [(40, 40), (90, 90), (140, 140)],
    "prtg_cas04": [(60, 60), (110, 110), (160, 160)],
    "prtg_cas05": [(20, 20), (70, 70), (130, 130)],
    "gefg_cas01": [(45, 45), (95, 95), (145, 145)],
    "gefg_cas02": [(35, 35), (85, 85), (135, 135)],
    "gefg_cas03": [(55, 55), (105, 105), (155, 155)],
    "gefg_cas04": [(25, 25), (75, 75), (125, 125)],
    "gefg_cas05": [(65, 65), (115, 115), (165, 165)],
    "payg_cas01": [(40, 40), (90, 90), (140, 140)],
    "payg_cas02": [(30, 30), (80, 80), (130, 130)],
    "payg_cas03": [(50, 50), (100, 100), (150, 150)],
    "payg_cas04": [(20, 20), (70, 70), (120, 120)],
    "payg_cas05": [(60, 60), (110, 110), (160, 160)],
    "aldeba_cas01": [(50, 50), (100, 100), (150, 150)],
    "aldeba_cas02": [(30, 30), (80, 80), (130, 130)],
    "aldeba_cas03": [(40, 40), (90, 90), (140, 140)],
    "aldeba_cas04": [(60, 60), (110, 110), (160, 160)],
    "aldeba_cas05": [(20, 20), (70, 70), (120, 120)],
}

WOE_ENTRANCES: dict[str, list[tuple[int, int]]] = {
    "gld_dun01": [(10, 10), (150, 10)],
    "gld_dun02": [(5, 5), (140, 5)],
    "gld_dun03": [(15, 15), (130, 15)],
    "gld_dun04": [(8, 8), (145, 8)],
    "prtg_cas01": [(5, 5), (200, 5)],
    "prtg_cas02": [(5, 5), (180, 5)],
    "prtg_cas03": [(5, 5), (190, 5)],
    "prtg_cas04": [(5, 5), (210, 5)],
    "prtg_cas05": [(5, 5), (170, 5)],
    "gefg_cas01": [(5, 5), (195, 5)],
    "gefg_cas02": [(5, 5), (185, 5)],
    "gefg_cas03": [(5, 5), (205, 5)],
    "gefg_cas04": [(5, 5), (175, 5)],
    "gefg_cas05": [(5, 5), (215, 5)],
    "payg_cas01": [(5, 5), (190, 5)],
    "payg_cas02": [(5, 5), (180, 5)],
    "payg_cas03": [(5, 5), (200, 5)],
    "payg_cas04": [(5, 5), (170, 5)],
    "payg_cas05": [(5, 5), (210, 5)],
    "aldeba_cas01": [(5, 5), (200, 5)],
    "aldeba_cas02": [(5, 5), (180, 5)],
    "aldeba_cas03": [(5, 5), (190, 5)],
    "aldeba_cas04": [(5, 5), (210, 5)],
    "aldeba_cas05": [(5, 5), (170, 5)],
}

WOE_DEFENSIVE_POSITIONS: dict[str, list[tuple[int, int]]] = {
    "gld_dun01": [(55, 55), (95, 95)],
    "gld_dun02": [(35, 35), (115, 85)],
    "gld_dun03": [(65, 65), (85, 85)],
    "gld_dun04": [(45, 45), (75, 75), (105, 105)],
    "prtg_cas01": [(55, 55), (105, 105), (155, 155)],
    "prtg_cas02": [(35, 35), (85, 85), (125, 125)],
    "prtg_cas03": [(45, 45), (95, 95), (145, 145)],
    "prtg_cas04": [(65, 65), (115, 115), (165, 165)],
    "prtg_cas05": [(25, 25), (75, 75), (135, 135)],
    "gefg_cas01": [(50, 50), (100, 100), (150, 150)],
    "gefg_cas02": [(40, 40), (90, 90), (140, 140)],
    "gefg_cas03": [(60, 60), (110, 110), (160, 160)],
    "gefg_cas04": [(30, 30), (80, 80), (130, 130)],
    "gefg_cas05": [(70, 70), (120, 120), (170, 170)],
    "payg_cas01": [(45, 45), (95, 95), (145, 145)],
    "payg_cas02": [(35, 35), (85, 85), (135, 135)],
    "payg_cas03": [(55, 55), (105, 105), (155, 155)],
    "payg_cas04": [(25, 25), (75, 75), (125, 125)],
    "payg_cas05": [(65, 65), (115, 115), (165, 165)],
    "aldeba_cas01": [(55, 55), (105, 105), (155, 155)],
    "aldeba_cas02": [(35, 35), (85, 85), (135, 135)],
    "aldeba_cas03": [(45, 45), (95, 95), (145, 145)],
    "aldeba_cas04": [(65, 65), (115, 115), (165, 165)],
    "aldeba_cas05": [(25, 25), (75, 75), (135, 135)],
}

# EMP HP thresholds for damage focus calls
EMP_HP_THRESHOLDS: list[tuple[float, str]] = [
    (0.75, "EMP at 75% — keep pushing!"),
    (0.50, "EMP at 50% — focus damage!"),
    (0.25, "EMP at 25% — burn it down!"),
    (0.10, "EMP at 10% — finish it!"),
    (0.05, "EMP at 5% — almost dead!"),
]

# High-value caster priority list
HIGH_VALUE_CASTERS: set[str] = {
    "sura", "champion", "wizard", "high_wizard", "creator", "genetic",
    "soul_linker", "warlock", "sorcerer",
}

# Class-vs-class counter strategies for WoE
CLASS_COUNTERS: dict[str, dict[str, str]] = {
    "alchemist": {
        "paladin": "Acid Demonstration bypasses Paladin's high DEF and Guard",
        "crusader": "Acid Demonstration ignores defense — focus fire",
    },
    "wizard": {
        "champion": "Storm Gust freezes Champion before Asura Strike",
        "monk": "Frost Diver stops Monk charge",
        "paladin": "Heaven's Drive deals neutral damage through Guard",
    },
    "assassin": {
        "wizard": "Cloak + Backstab — Wizards can't see cloaked Assassins",
        "priest": "Soul Destroyer interrupts heal cast",
    },
    "hunter": {
        "wizard": "Ankle Snare stops Wizard movement, ranged attack from safety",
        "priest": "Sharp Shooting from outside dispel range",
    },
    "priest": {
        "assassin": "Kyrie Eleison reflects backstab damage",
        "champion": "Lex Aeterna + turn undead if applicable",
    },
    "paladin": {
        "assassin": "Shield Reflect counters backstab",
        "hunter": "Defending Aura reduces ranged damage",
    },
    "champion": {
        "paladin": "Asura Strike bypasses Guard (neutral property)",
        "wizard": "Mental Strength + charge through AoE",
    },
}


# ── Data models ──────────────────────────────────────────────────────────

@dataclass(slots=True)
class WoEIntelligence:
    """War of Emperium strategy engine — comprehensive WoE intelligence.

    Features:
      - WoE schedule awareness (Sat/Sun 20:00-22:00)
      - Pre-WoE preparation (30 min before)
      - Castle defense/attack tactics
      - Emperium breaking strategy with class-specific DPS
      - Guild coordination and chemistry tracking
      - 50+ player battlefield awareness
      - Escape scroll economy
      - WoE consumable preparation
      - Castle ownership tracking
      - Class-vs-class counter strategies
      - Barricade management and bypass methods
      - Battlefield threat assessment
    """

    _lock: RLock = field(default_factory=RLock)
    _woe_schedule: dict[str, list[int]] = field(default_factory=lambda: {
        "monday": [], "tuesday": [], "wednesday": [], "thursday": [],
        "friday": [], "saturday": [20, 22], "sunday": [20, 22],
    })
    _guild_info: dict[str, Any] = field(default_factory=dict)
    _castle_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "woe_participations": 0, "emperium_breaks": 0, "deaths_in_woe": 0,
        "enemies_killed": 0, "castles_defended": 0, "castles_captured": 0,
    })
    _prep_started: bool = False
    _last_prep_check: float = 0.0
    _woe_active: bool = False
    _last_schedule_check: float = 0.0

    # Sub-engines
    _emperium: EmperiumMechanics = field(default_factory=EmperiumMechanics)
    _battlefield: BattlefieldAwareness = field(default_factory=BattlefieldAwareness)
    _castle_intel: CastleIntelligence = field(default_factory=CastleIntelligence)

    # ── Public API ────────────────────────────────────────────────────

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run full WoE intelligence assessment."""
        if not signals:
            return

        with self._lock:
            map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
            players = signals.get("players", []) or []
            guild_name = str(signals.get("guild_name", "") or "")
            job_name = str(signals.get("job_name", "novice") or "novice").lower()
            my_hp_pct = signals.get("hp_ratio", 1.0) or 1.0

            # 1. Check WoE schedule
            self._check_schedule(actions, bot_id, signals)

            # 2. Update guild info
            if guild_name:
                self._guild_info["name"] = guild_name
            self._guild_info["has_castle"] = bool(self._castle_intel.get_owned_castles())

            # 3. Run sub-engines
            self._emperium.assess(signals, actions, bot_id)
            self._battlefield.assess(signals, actions, bot_id)
            self._castle_intel.assess(signals, actions, bot_id)

            # 4. Update castle state
            self._update_castle_state(map_name, players, guild_name)

            # 5. Emit WoE-specific actions
            if self.is_woe_active():
                self._emit_woe_actions(actions, bot_id, map_name, signals, job_name, my_hp_pct)

            # 6. Update stats
            self._update_stats(signals)

    def is_woe_active(self) -> bool:
        """Check if WoE is currently active."""
        now = time.localtime()
        day_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][now.tm_wday]
        hour = now.tm_hour

        for start_hour in self._woe_schedule.get(day_name, []):
            if start_hour <= hour < start_hour + 2:
                return True
        return False

    def get_woe_status(self) -> dict[str, Any]:
        """Get current WoE status and recommendations."""
        active = self.is_woe_active()
        now = time.localtime()
        day_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][now.tm_wday]

        if not active:
            next_woe = None
            for start_hour in self._woe_schedule.get(day_name, []):
                if start_hour > now.tm_hour:
                    next_woe = start_hour
                    break

            # Check if prep should start
            should_prep = next_woe and next_woe - now.tm_hour <= 0.5  # 30 min

            return {
                "active": False,
                "next_woe_hour": next_woe,
                "recommendation": "prepare" if should_prep else "ignore",
                "prep_needed": should_prep,
                "current_hour": now.tm_hour,
                "current_day": day_name,
            }

        return {
            "active": True,
            "recommendation": "join_guild_war",
            "strategy": "defend_castle" if self._guild_info.get("has_castle") else "attack_castle",
            "priority": "emperium" if self._guild_info.get("has_castle") else "survival",
            "time_remaining": f"{22 - now.tm_hour}h",
        }

    def recommend_equipment(self) -> dict[str, str]:
        """Recommend equipment for WoE."""
        if not self.is_woe_active():
            return {"armor": "normal", "weapon": "normal", "shield": "normal", "consumables": "normal"}

        return {
            "armor": "valkyrie_armor",
            "weapon": "holy_weapon",
            "shield": "valkyrie_shield",
            "consumables": "woe_potions",
            "headgear": "pithy_crown",
            "garment": "valkyrie_manteau",
            "shoes": "greaves",
            "accessory1": "ring_of_resistance",
            "accessory2": "ring_of_resistance",
        }

    def should_engage(self, enemy_count: int, ally_count: int,
                      has_emperium: bool = False) -> dict[str, Any]:
        """Should we engage in PvP during WoE?"""
        if not self.is_woe_active():
            return {"engage": False, "reason": "not_woe_time"}

        if has_emperium:
            return {"engage": True, "reason": "emperium_break_priority", "target": "emperium"}

        if ally_count >= enemy_count:
            return {"engage": True, "reason": "numerical_advantage"}

        if enemy_count >= RUN_THRESHOLD:
            return {"engage": False, "reason": "outnumbered", "action": "retreat"}

        if ally_count < enemy_count * 0.5:
            return {"engage": False, "reason": "outnumbered", "action": "retreat"}

        return {"engage": True, "reason": "even_fight"}

    def get_class_counter(self, attacker_class: str, defender_class: str) -> str | None:
        """Get counter strategy for class-vs-class in WoE."""
        return CLASS_COUNTERS.get(attacker_class.lower(), {}).get(defender_class.lower())

    def get_emperium_break_time(self, class_name: str) -> dict[str, Any]:
        """Calculate Emperium break time for a class."""
        estimate = self._emperium.calculate_break_time(class_name)
        return {
            "class": estimate.class_name,
            "dps": estimate.dps,
            "estimated_seconds": estimate.estimated_seconds,
            "estimated_time": estimate.estimated_seconds_formatted,
            "can_break": estimate.can_break,
            "best_skill": estimate.best_skill,
            "strategy": estimate.strategy,
        }

    def get_woe_prep_checklist(self, class_name: str) -> dict[str, Any]:
        """Get WoE preparation checklist."""
        return self._castle_intel.get_woe_prep_checklist(class_name)

    def get_castle_intel(self) -> CastleIntelligence:
        """Get the castle intelligence sub-engine."""
        return self._castle_intel

    def get_battlefield(self) -> BattlefieldAwareness:
        """Get the battlefield awareness sub-engine."""
        return self._battlefield

    def get_emperium_mechanics(self) -> EmperiumMechanics:
        """Get the Emperium mechanics sub-engine."""
        return self._emperium

    def counters(self) -> dict[str, int]:
        """Get WoE statistics counters."""
        with self._lock:
            return dict(self._stats)

    # ── Internal ────────────────────────────────────────────────────

    def _check_schedule(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Check WoE schedule and emit prep/start/end actions."""
        now = time.time()
        if now - self._last_schedule_check < 30:  # Check every 30s
            return
        self._last_schedule_check = now

        active = self.is_woe_active()

        # WoE just started
        if active and not self._woe_active:
            self._woe_active = True
            self._prep_started = False
            actions.append(HeuristicAction(
                kind="command",
                command="woe_start",
                confidence=0.99,
                domain="pvp",
                reason="WoE has started! Moving to castle",
                metadata={"action": "woe_start"},
            ))

        # WoE just ended
        if not active and self._woe_active:
            self._woe_active = False
            actions.append(HeuristicAction(
                kind="command",
                command="woe_end",
                confidence=0.99,
                domain="pvp",
                reason="WoE has ended — returning to normal operations",
                metadata={"action": "woe_end"},
            ))

        # Pre-WoE preparation
        if not active and not self._prep_started:
            now_t = time.localtime()
            day_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][now_t.tm_wday]
            for start_hour in self._woe_schedule.get(day_name, []):
                if 0 < start_hour - now_t.tm_hour <= 0.5:  # Within 30 min
                    self._prep_started = True
                    my_class = str(signals.get("job_name", "novice") or "novice").lower()
                    checklist = self.get_woe_prep_checklist(my_class)

                    actions.append(HeuristicAction(
                        kind="command",
                        command="woe_prep_start",
                        confidence=0.95,
                        domain="economy",
                        reason=f"WoE in {(start_hour - now_t.tm_hour) * 60:.0f} min — starting preparation",
                        metadata={
                            "action": "woe_prep",
                            "minutes_to_woe": (start_hour - now_t.tm_hour) * 60,
                            "checklist": checklist,
                        },
                    ))

                    # Emit consumable buy actions
                    for item_name, item_info in WOE_CONSUMABLES.items():
                        actions.append(HeuristicAction(
                            kind="command",
                            command=f"buy {item_info['id']} {item_info['min_count']}",
                            confidence=0.85,
                            domain="economy",
                            reason=f"WoE prep: buy {item_name} x{item_info['min_count']}",
                            metadata={"item": item_name, "count": item_info["min_count"]},
                        ))
                    break

    def _update_castle_state(
        self,
        map_name: str,
        players: list[Any],
        guild_name: str,
    ) -> None:
        """Update castle state from current map."""
        if map_name not in self._castle_state:
            self._castle_state[map_name] = {
                "name": map_name,
                "guild_owner": "",
                "emperium_alive": True,
                "emperium_hp_pct": 1.0,
                "allies_nearby": 0,
                "enemies_nearby": 0,
                "last_seen": time.time(),
                "in_emperium_room": self._is_emperium_room(map_name),
            }

        castle = self._castle_state[map_name]
        castle["last_seen"] = time.time()
        castle["allies_nearby"] = 0
        castle["enemies_nearby"] = 0

        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
            else:
                pg = ""
            if pg == guild_name:
                castle["allies_nearby"] += 1
            elif pg:
                castle["enemies_nearby"] += 1

        if guild_name:
            castle["guild_owner"] = guild_name

    def _is_emperium_room(self, map_name: str) -> bool:
        """Check if current map is an Emperium room."""
        return any(frag in map_name for frag in EMPERIUM_MAP_FRAGMENTS)

    def _emit_woe_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        job_name: str,
        my_hp_pct: float,
    ) -> None:
        """Emit WoE-specific actions based on role and situation."""
        # Determine role
        role = self._resolve_role(job_name)
        in_emp_room = self._is_emperium_room(map_name)

        # HP management
        if my_hp_pct < 0.30:
            actions.append(HeuristicAction(
                kind="command",
                command="use potion",  # RULE.md: generic heal item (OpenKore resolves from tables)
                confidence=0.99,
                domain="pvp",
                reason=f"WoE HP low ({my_hp_pct:.0%}) — using potion",
                metadata={"hp_pct": my_hp_pct, "action": "heal"},
            ))
            return

        # Role-based actions
        if role == "breaker" and in_emp_room:
            self._emit_emperium_attack(actions, bot_id, map_name, signals, job_name)
        elif role == "attacker":
            self._emit_push_actions(actions, bot_id, map_name, signals)
        elif role == "defender":
            self._emit_defend_actions(actions, bot_id, map_name, signals)
        elif role == "support":
            self._emit_support_actions(actions, bot_id, map_name, signals, job_name)
        elif role == "scout":
            self._emit_scout_actions(actions, bot_id, map_name)

        # Cross-role actions
        self._emit_class_specific_actions(actions, bot_id, map_name, signals, job_name, role)
        self._emit_caster_interrupt(actions, signals, job_name)

    def _resolve_role(self, job_name: str) -> str:
        """Resolve WoE role based on job class."""
        job_lower = job_name.lower()

        if any(s in job_lower for s in ["priest", "arch_bishop", "acolyte"]):
            return "support"
        if any(s in job_lower for s in ["knight", "lord_knight", "rune_knight",
                                         "crusader", "paladin", "royal_guard",
                                         "swordman"]):
            return "defender"
        if any(s in job_lower for s in ["assassin", "guillotine_cross",
                                         "rogue", "stalker", "shadow_chaser",
                                         "monk", "champion", "sura"]):
            return "breaker"
        if any(s in job_lower for s in ["hunter", "sniper", "ranger",
                                         "bard", "clown", "minstrel",
                                         "dancer", "gypsy", "wanderer"]):
            return "scout"
        return "attacker"

    def _emit_emperium_attack(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        job_name: str,
    ) -> None:
        """Focus all DPS on the Emperium with class-appropriate skills."""
        logger.info("[WoE] %s: BREAKING emperium on %s!", bot_id, map_name)

        # Get best skill for this class
        best_skill = CLASS_EMP_BEST_SKILL.get(job_name, "attack")
        estimate = self._emperium.calculate_break_time(job_name)

        actions.append(HeuristicAction(
            kind="command",
            command=f"attack_emperium {best_skill}",
            confidence=0.99,
            domain="pvp",
            reason=f"WoE breaker: killing emperium on {map_name} ({estimate.estimated_seconds_formatted})",
            metadata={
                "map": map_name,
                "target": "emperium",
                "woe_role": "breaker",
                "best_skill": best_skill,
                "estimated_time": estimate.estimated_seconds_formatted,
                "dps": estimate.dps,
            },
        ))

        # Max attack for EMP DPS
        actions.append(HeuristicAction(
            kind="command",
            command="set attackAuto 3",
            confidence=0.95,
            domain="pvp",
            reason="Max attack for emperium DPS",
        ))

        # Class-specific EMP attacks
        if "assassin" in job_name or "rogue" in job_name:
            actions.append(HeuristicAction(
                kind="command",
                command=f"use_skill {best_skill} Emperium",
                confidence=0.90,
                domain="pvp",
                reason=f"Assassin: {best_skill} EMP for max damage",
            ))
        elif "champion" in job_name or "monk" in job_name or "sura" in job_name:
            # Asura timing when EMP HP is low
            emp_state = self._emperium.get_state()
            if emp_state.hp_pct < 0.30:
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill Asura Strike Emperium",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"Champion: Asura Strike on low HP EMP ({emp_state.hp_pct:.0%})",
                ))
        elif "wizard" in job_name or "sage" in job_name:
            actions.append(HeuristicAction(
                kind="command",
                command=f"use_skill {best_skill} Emperium",
                confidence=0.85,
                domain="pvp",
                reason=f"Wizard: {best_skill} on EMP position",
            ))

    def _emit_push_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
    ) -> None:
        """Push into the castle and engage enemies."""
        my_hp = int(signals.get("hp", 1) or 1)
        my_max_hp = int(signals.get("max_hp", 100) or 100)
        hp_ratio = my_hp / max(my_max_hp, 1)

        if hp_ratio < 0.30:
            actions.append(HeuristicAction(
                kind="command",
                command="use potion",
                confidence=0.95,
                domain="pvp",
                reason=f"WoE HP low ({hp_ratio:.0%}) — healing",
            ))
            return

        # Navigate through chokepoints
        chokepoints = WOE_CHOKEPOINTS.get(map_name, [])
        castle = self._castle_state.get(map_name)
        in_emp_room = castle.get("in_emperium_room", False) if castle else False

        if chokepoints and not in_emp_room:
            target_cp = chokepoints[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {target_cp[0]} {target_cp[1]}",
                confidence=0.80,
                domain="pvp",
                reason=f"WoE push: moving to chokepoint on {map_name}",
            ))

        # Find and engage enemies
        enemies = self._find_enemies(signals)
        if enemies:
            # Prioritize high-value casters
            high_value = self._find_high_value_casters(signals)
            target = high_value[0] if high_value else enemies[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target}",
                confidence=0.90,
                domain="pvp",
                reason=f"WoE attack: pushing on {map_name} — engaging {target}",
                metadata={"map": map_name, "woe_role": "attacker"},
            ))

        if castle and castle.get("in_emperium_room"):
            self._emit_emperium_attack(actions, bot_id, map_name, signals,
                                       str(signals.get("job_name", "novice") or "novice"))

    def _emit_defend_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
    ) -> None:
        """Hold choke points — intercept enemies before they reach the emperium."""
        enemies = self._find_enemies(signals)

        if not enemies:
            # Move to defensive position
            def_positions = WOE_DEFENSIVE_POSITIONS.get(map_name, [])
            if def_positions:
                pos = def_positions[0]
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"move {pos[0]} {pos[1]}",
                    confidence=0.80,
                    domain="pvp",
                    reason=f"WoE defend: moving to defensive position on {map_name}",
                ))
            return

        # Intercept enemies at chokepoints
        chokepoints = WOE_CHOKEPOINTS.get(map_name, [])
        if chokepoints:
            target_cp = chokepoints[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {target_cp[0]} {target_cp[1]}",
                confidence=0.85,
                domain="pvp",
                reason=f"WoE defend: intercepting at chokepoint on {map_name}",
            ))

        # Engage nearest enemy
        target = enemies[0]
        actions.append(HeuristicAction(
            kind="command",
            command=f"attack {target}",
            confidence=0.90,
            domain="pvp",
            reason=f"WoE defend: engaging {target} on {map_name}",
            metadata={"map": map_name, "woe_role": "defender"},
        ))

    def _emit_support_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        job_name: str,
    ) -> None:
        """Support role: heal allies, cast buffs, stay behind tanks."""
        # Heal lowest HP ally
        allies = self._find_allies(signals)
        if allies:
            lowest_hp_ally = min(allies, key=lambda a: a.get("hp_ratio", 1.0) if isinstance(a, dict) else 1.0)
            if isinstance(lowest_hp_ally, dict) and lowest_hp_ally.get("hp_ratio", 1.0) < 0.50:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"heal {lowest_hp_ally.get('name', '')}",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"WoE support: healing {lowest_hp_ally.get('name', '')} (HP {lowest_hp_ally.get('hp_ratio', 1.0):.0%})",
                ))

        # Stay behind defensive position
        def_positions = WOE_DEFENSIVE_POSITIONS.get(map_name, [])
        if def_positions:
            pos = def_positions[-1]  # Furthest back position
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {pos[0]} {pos[1]}",
                confidence=0.80,
                domain="pvp",
                reason=f"WoE support: staying behind tanks at defensive position",
            ))

        # Cast party buffs
        actions.append(HeuristicAction(
            kind="command",
            command="use_skill_assumptio tank",
            confidence=0.85,
            domain="pvp",
            reason="WoE support: casting Assumptio on tank",
        ))

    def _emit_scout_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
    ) -> None:
        """Scout role: report enemy positions, avoid direct combat."""
        entrances = WOE_ENTRANCES.get(map_name, [])
        if entrances:
            # Move to entrance to scout
            pos = entrances[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {pos[0]} {pos[1]}",
                confidence=0.80,
                domain="pvp",
                reason=f"WoE scout: moving to entrance on {map_name}",
            ))

        actions.append(HeuristicAction(
            kind="command",
            command="scout_report",
            confidence=0.70,
            domain="pvp",
            reason=f"WoE scout: reporting enemy positions on {map_name}",
        ))

    def _emit_class_specific_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        job_name: str,
        role: str,
    ) -> None:
        """Emit class-specific WoE actions."""
        job_lower = job_name.lower()

        # Wizard: use Heaven's Drive on chokepoints
        if "wizard" in job_lower or "sage" in job_lower:
            chokepoints = WOE_CHOKEPOINTS.get(map_name, [])
            if chokepoints and role != "breaker":
                cp = chokepoints[0]
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill heavens_drive {cp[0]} {cp[1]}",
                    confidence=0.80,
                    domain="pvp",
                    reason=f"Wizard: Heaven's Drive on chokepoint ({cp[0]}, {cp[1]})",
                ))

        # Assassin: use cloaking to bypass barricades
        if "assassin" in job_lower or "rogue" in job_lower:
            bypass_methods = BARRICADE_BYPASS.get(job_lower, [])
            if "cloaking" in bypass_methods:
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill cloaking",
                    confidence=0.85,
                    domain="pvp",
                    reason="Assassin: cloaking to bypass barricades",
                ))

        # Priest: cast Kyrie Eleison on party
        if "priest" in job_lower:
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill kyrie_eleison party",
                confidence=0.85,
                domain="pvp",
                reason="Priest: Kyrie Eleison on party for WoE",
            ))

        # Paladin: use Shield Reflect
        if "paladin" in job_lower or "crusader" in job_lower:
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill shield_reflect",
                confidence=0.85,
                domain="pvp",
                reason="Paladin: Shield Reflect for WoE defense",
            ))

    def _emit_caster_interrupt(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        job_name: str,
    ) -> None:
        """Interrupt enemy casters."""
        enemies = signals.get("players", []) or []
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            e_job = str(enemy.get("job", "") or "").lower()
            if e_job in HIGH_VALUE_CASTERS:
                e_name = str(enemy.get("name", "") or "")
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"interrupt {e_name}",
                    confidence=0.85,
                    domain="pvp",
                    reason=f"Interrupting high-value caster: {e_name} ({e_job})",
                    metadata={"target": e_name, "target_class": e_job},
                ))
                break

    def _find_enemies(self, signals: dict[str, Any]) -> list[str]:
        """Find enemy player names on the map."""
        players = signals.get("players", []) or []
        my_guild = str(signals.get("guild_name", "") or "")
        enemies = []
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
                if pg and pg != my_guild:
                    enemies.append(str(p.get("name", "") or ""))
        return enemies

    def _find_allies(self, signals: dict[str, Any]) -> list[dict[str, Any]]:
        """Find ally player info on the map."""
        players = signals.get("players", []) or []
        my_guild = str(signals.get("guild_name", "") or "")
        allies = []
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
                if pg == my_guild:
                    allies.append(p)
        return allies

    def _find_high_value_casters(self, signals: dict[str, Any]) -> list[str]:
        """Find high-value caster targets."""
        players = signals.get("players", []) or []
        my_guild = str(signals.get("guild_name", "") or "")
        casters = []
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
                pj = str(p.get("job", "") or "").lower()
                if pg and pg != my_guild and pj in HIGH_VALUE_CASTERS:
                    casters.append(str(p.get("name", "") or ""))
        return casters

    def _update_stats(self, signals: dict[str, Any]) -> None:
        """Update WoE statistics."""
        with self._lock:
            if signals.get("woe_participation", False):
                self._stats["woe_participations"] += 1
