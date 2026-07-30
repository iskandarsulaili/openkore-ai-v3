"""Castle Intelligence — castle ownership tracking, WoE schedule, pre-WoE preparation.

A pro WoE player knows:
- Which guild owns which castle
- When owned castles are under attack
- WoE schedule (Sat/Sun 20:00-22:00)
- Pre-WoE prep checklist (30 min before)
- Barricade positions and bypass methods per castle
- Castle-specific strategies
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

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# WoE schedule (server time)
WOE_DAYS: list[int] = [5, 6]  # Saturday=5, Sunday=6
WOE_START_HOUR: int = 20       # 8 PM
WOE_END_HOUR: int = 22         # 10 PM
WOE_DURATION_HOURS: int = 2
PRE_WOE_PREP_MINUTES: int = 30  # Start prep 30 min before

# All WoE castles with their map names
WOE_CASTLES: dict[str, dict[str, Any]] = {
    # Prontera castles
    "prtg_cas01": {"name": "Prtg Castle 01", "map": "prtg_cas01", "city": "prontera", "barricades": 3},
    "prtg_cas02": {"name": "Prtg Castle 02", "map": "prtg_cas02", "city": "prontera", "barricades": 3},
    "prtg_cas03": {"name": "Prtg Castle 03", "map": "prtg_cas03", "city": "prontera", "barricades": 3},
    "prtg_cas04": {"name": "Prtg Castle 04", "map": "prtg_cas04", "city": "prontera", "barricades": 3},
    "prtg_cas05": {"name": "Prtg Castle 05", "map": "prtg_cas05", "city": "prontera", "barricades": 3},
    # Geffen castles
    "gefg_cas01": {"name": "Gefg Castle 01", "map": "gefg_cas01", "city": "geffen", "barricades": 2},
    "gefg_cas02": {"name": "Gefg Castle 02", "map": "gefg_cas02", "city": "geffen", "barricades": 2},
    "gefg_cas03": {"name": "Gefg Castle 03", "map": "gefg_cas03", "city": "geffen", "barricades": 2},
    "gefg_cas04": {"name": "Gefg Castle 04", "map": "gefg_cas04", "city": "geffen", "barricades": 2},
    "gefg_cas05": {"name": "Gefg Castle 05", "map": "gefg_cas05", "city": "geffen", "barricades": 2},
    # Payon castles
    "payg_cas01": {"name": "Payg Castle 01", "map": "payg_cas01", "city": "payon", "barricades": 2},
    "payg_cas02": {"name": "Payg Castle 02", "map": "payg_cas02", "city": "payon", "barricades": 2},
    "payg_cas03": {"name": "Payg Castle 03", "map": "payg_cas03", "city": "payon", "barricades": 2},
    "payg_cas04": {"name": "Payg Castle 04", "map": "payg_cas04", "city": "payon", "barricades": 2},
    "payg_cas05": {"name": "Payg Castle 05", "map": "payg_cas05", "city": "payon", "barricades": 2},
    # Aldebaran castles
    "aldeba_cas01": {"name": "Aldeba Castle 01", "map": "aldeba_cas01", "city": "aldebaran", "barricades": 3},
    "aldeba_cas02": {"name": "Aldeba Castle 02", "map": "aldeba_cas02", "city": "aldebaran", "barricades": 3},
    "aldeba_cas03": {"name": "Aldeba Castle 03", "map": "aldeba_cas03", "city": "aldebaran", "barricades": 3},
    "aldeba_cas04": {"name": "Aldeba Castle 04", "map": "aldeba_cas04", "city": "aldebaran", "barricades": 3},
    "aldeba_cas05": {"name": "Aldeba Castle 05", "map": "aldeba_cas05", "city": "aldebaran", "barricades": 3},
}

# Barricade positions per castle (map_name -> [(x, y, type)])
# type: "wall" = standard barricade, "gate" = gate, "emp_room" = emperium room entrance
BARRICADE_POSITIONS: dict[str, list[tuple[int, int, str]]] = {
    "prtg_cas01": [(50, 50, "wall"), (100, 100, "gate"), (150, 150, "emp_room")],
    "prtg_cas02": [(30, 30, "wall"), (80, 80, "gate"), (120, 120, "emp_room")],
    "prtg_cas03": [(40, 40, "wall"), (90, 90, "gate"), (140, 140, "emp_room")],
    "prtg_cas04": [(60, 60, "wall"), (110, 110, "gate"), (160, 160, "emp_room")],
    "prtg_cas05": [(20, 20, "wall"), (70, 70, "gate"), (130, 130, "emp_room")],
    "gefg_cas01": [(45, 45, "wall"), (95, 95, "gate"), (145, 145, "emp_room")],
    "gefg_cas02": [(35, 35, "wall"), (85, 85, "gate"), (135, 135, "emp_room")],
    "gefg_cas03": [(55, 55, "wall"), (105, 105, "gate"), (155, 155, "emp_room")],
    "gefg_cas04": [(25, 25, "wall"), (75, 75, "gate"), (125, 125, "emp_room")],
    "gefg_cas05": [(65, 65, "wall"), (115, 115, "gate"), (165, 165, "emp_room")],
    "payg_cas01": [(40, 40, "wall"), (90, 90, "gate"), (140, 140, "emp_room")],
    "payg_cas02": [(30, 30, "wall"), (80, 80, "gate"), (130, 130, "emp_room")],
    "payg_cas03": [(50, 50, "wall"), (100, 100, "gate"), (150, 150, "emp_room")],
    "payg_cas04": [(20, 20, "wall"), (70, 70, "gate"), (120, 120, "emp_room")],
    "payg_cas05": [(60, 60, "wall"), (110, 110, "gate"), (160, 160, "emp_room")],
    "aldeba_cas01": [(50, 50, "wall"), (100, 100, "gate"), (150, 150, "emp_room")],
    "aldeba_cas02": [(30, 30, "wall"), (80, 80, "gate"), (130, 130, "emp_room")],
    "aldeba_cas03": [(40, 40, "wall"), (90, 90, "gate"), (140, 140, "emp_room")],
    "aldeba_cas04": [(60, 60, "wall"), (110, 110, "gate"), (160, 160, "emp_room")],
    "aldeba_cas05": [(20, 20, "wall"), (70, 70, "gate"), (120, 120, "emp_room")],
}

# Barricade bypass methods per class
BARRICADE_BYPASS: dict[str, list[str]] = {
    "soul_linker": ["teleport", "soul_link_pass"],
    "assassin": ["cloaking", "hiding"],
    "assassin_cross": ["cloaking", "hiding", "dark_illusion"],
    "guillotine_cross": ["cloaking", "hiding", "dark_illusion"],
    "rogue": ["cloaking", "hiding"],
    "stalker": ["cloaking", "hiding"],
    "shadow_chaser": ["cloaking", "hiding", "masquerade"],
    "wizard": ["safety_wall_break"],
    "high_wizard": ["safety_wall_break", "ganbantein"],
    "warlock": ["ganbantein", "teleport"],
    "ninja": ["shadow_jump", "teleport"],
    "kagerou": ["shadow_jump", "teleport"],
    "oboro": ["shadow_jump", "teleport"],
}

# Pre-WoE consumable checklist
WOE_CONSUMABLES: dict[str, dict[str, Any]] = {
    "white_potion": {"id": 504, "min_count": 50, "cost_multiplier": 1.5, "priority": 1},
    "blue_potion": {"id": 505, "min_count": 20, "cost_multiplier": 1.5, "priority": 2},
    "fly_wing": {"id": 601, "min_count": 30, "cost_multiplier": 1.0, "priority": 3},
    "butterfly_wing": {"id": 602, "min_count": 5, "cost_multiplier": 1.0, "priority": 4},
    "awakening_potion": {"id": 506, "min_count": 5, "cost_multiplier": 2.0, "priority": 5},
    "concentration_potion": {"id": 507, "min_count": 5, "cost_multiplier": 2.0, "priority": 6},
    "elemental_converter": {"id": 10000, "min_count": 10, "cost_multiplier": 3.0, "priority": 7},
    "resistance_potion": {"id": 10001, "min_count": 5, "cost_multiplier": 2.0, "priority": 8},
}

# WoE equipment recommendations per class
WOE_EQUIPMENT: dict[str, dict[str, str]] = {
    "default": {
        "armor": "valkyrie_armor",
        "shield": "valkyrie_shield",
        "weapon": "holy_weapon",
        "headgear": "pithy_crown",
        "garment": "valkyrie_manteau",
        "shoes": "greaves",
        "accessory1": "ring_of_resistance",
        "accessory2": "ring_of_resistance",
    },
    "wizard": {
        "armor": "valkyrie_armor",
        "shield": "valkyrie_shield",
        "weapon": "holy_staff",
        "headgear": "crown_of_ancient",
        "garment": "valkyrie_manteau",
        "shoes": "greaves",
        "accessory1": "orb_of_concentration",
        "accessory2": "orb_of_concentration",
    },
    "assassin": {
        "armor": "valkyrie_armor",
        "shield": "valkyrie_shield",
        "weapon": "holy_katar",
        "headgear": "assassin_mask",
        "garment": "valkyrie_manteau",
        "shoes": "greaves",
        "accessory1": "ring_of_resistance",
        "accessory2": "ring_of_resistance",
    },
    "priest": {
        "armor": "valkyrie_armor",
        "shield": "valkyrie_shield",
        "weapon": "holy_mace",
        "headgear": "mitre",
        "garment": "valkyrie_manteau",
        "shoes": "greaves",
        "accessory1": "orb_of_healing",
        "accessory2": "orb_of_healing",
    },
}


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class CastleInfo:
    """Information about a WoE castle."""
    castle_id: str = ""
    name: str = ""
    map_name: str = ""
    city: str = ""
    guild_owner: str = ""
    barricade_count: int = 0
    under_attack: bool = False
    last_attack_time: float = 0.0
    last_seen: float = 0.0
    emperium_alive: bool = True
    emperium_hp_pct: float = 1.0
    allies_nearby: int = 0
    enemies_nearby: int = 0

    @property
    def is_owned(self) -> bool:
        return bool(self.guild_owner)

    @property
    def age(self) -> float:
        return time.time() - self.last_seen if self.last_seen > 0 else float("inf")


@dataclass
class WoESchedule:
    """WoE schedule information."""
    is_woe_day: bool = False
    is_woe_time: bool = False
    minutes_to_woe: int = 0
    minutes_since_woe_start: int = 0
    should_prep: bool = False
    next_woe_day: str = ""
    next_woe_time: str = ""


# ── Castle Intelligence Engine ───────────────────────────────────────────

class CastleIntelligence:
    """Castle ownership tracking and WoE schedule management.

    Features:
      - Track which guild owns which castle
      - Detect when owned castles are under attack
      - WoE schedule with pre-WoE preparation
      - Barricade positions and bypass methods per castle
      - WoE consumable preparation checklist
      - Equipment recommendations per class
      - Castle-specific strategies
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._castles: dict[str, CastleInfo] = {}
        self._my_guild: str = ""
        self._last_schedule_check: float = 0.0
        self._prep_started: bool = False
        self._woe_active_this_session: bool = False

        # Initialize castle info
        for cid, info in WOE_CASTLES.items():
            self._castles[cid] = CastleInfo(
                castle_id=cid,
                name=info["name"],
                map_name=info["map"],
                city=info["city"],
                barricade_count=info["barricades"],
            )

    # ── Public API ────────────────────────────────────────────────────

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run castle intelligence assessment."""
        if not signals:
            return

        with self._lock:
            map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
            my_guild = str(signals.get("guild_name", "") or "")
            if my_guild:
                self._my_guild = my_guild

            # 1. Update castle state from current map
            self._update_castle_state(signals, map_name)

            # 2. Check WoE schedule
            schedule = self.get_woe_schedule()
            self._handle_schedule(schedule, actions, bot_id, signals)

            # 3. Check if owned castle is under attack
            self._check_owned_castles(actions, bot_id)

            # 4. Emit castle intelligence
            if map_name in self._castles:
                castle = self._castles[map_name]
                self._emit_castle_intel(castle, actions, bot_id)

    def get_castle(self, castle_id: str) -> CastleInfo | None:
        """Get info for a specific castle."""
        with self._lock:
            return self._castles.get(castle_id)

    def get_castle_by_map(self, map_name: str) -> CastleInfo | None:
        """Get castle info by map name."""
        with self._lock:
            for castle in self._castles.values():
                if castle.map_name == map_name:
                    return castle
            return None

    def get_owned_castles(self) -> list[CastleInfo]:
        """Get all castles owned by our guild."""
        with self._lock:
            return [c for c in self._castles.values() if c.guild_owner == self._my_guild]

    def get_castles_under_attack(self) -> list[CastleInfo]:
        """Get all owned castles currently under attack."""
        with self._lock:
            return [c for c in self._castles.values()
                    if c.guild_owner == self._my_guild and c.under_attack]

    def get_woe_schedule(self) -> WoESchedule:
        """Get current WoE schedule information."""
        now = datetime.now()
        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        minute = now.minute

        is_woe_day = weekday in WOE_DAYS
        is_woe_time = is_woe_day and WOE_START_HOUR <= hour < WOE_END_HOUR

        # Calculate minutes to next WoE
        minutes_to_woe = 0
        if is_woe_day and hour < WOE_START_HOUR:
            minutes_to_woe = (WOE_START_HOUR - hour) * 60 - minute
        elif is_woe_day and hour >= WOE_END_HOUR:
            # Next WoE is next week
            minutes_to_woe = (7 - weekday + WOE_DAYS[0]) * 24 * 60 + WOE_START_HOUR * 60
        elif not is_woe_day:
            # Find next WoE day
            days_until = min((d - weekday) % 7 for d in WOE_DAYS)
            if days_until == 0:
                days_until = 7
            minutes_to_woe = days_until * 24 * 60 + WOE_START_HOUR * 60 - hour * 60 - minute

        minutes_since_start = 0
        if is_woe_time:
            minutes_since_start = (hour - WOE_START_HOUR) * 60 + minute

        should_prep = is_woe_day and not is_woe_time and minutes_to_woe <= PRE_WOE_PREP_MINUTES

        return WoESchedule(
            is_woe_day=is_woe_day,
            is_woe_time=is_woe_time,
            minutes_to_woe=minutes_to_woe,
            minutes_since_woe_start=minutes_since_start,
            should_prep=should_prep,
            next_woe_day="Saturday" if WOE_DAYS[0] == 5 else "Sunday",
            next_woe_time=f"{WOE_START_HOUR}:00 - {WOE_END_HOUR}:00",
        )

    def get_barricade_positions(self, castle_id: str) -> list[tuple[int, int, str]]:
        """Get barricade positions for a castle."""
        return BARRICADE_POSITIONS.get(castle_id, [])

    def get_barricade_bypass(self, class_name: str) -> list[str]:
        """Get barricade bypass methods for a class."""
        return BARRICADE_BYPASS.get(class_name.lower(), [])

    def get_woe_prep_checklist(self, class_name: str) -> dict[str, Any]:
        """Get WoE preparation checklist."""
        class_lower = class_name.lower()
        equipment = WOE_EQUIPMENT.get(class_lower, WOE_EQUIPMENT["default"])

        return {
            "consumables": WOE_CONSUMABLES,
            "equipment": equipment,
            "estimated_cost": self._estimate_prep_cost(),
            "prep_time_minutes": PRE_WOE_PREP_MINUTES,
        }

    def update_castle_owner(self, castle_id: str, guild_name: str) -> None:
        """Update the owner of a castle."""
        with self._lock:
            if castle_id in self._castles:
                self._castles[castle_id].guild_owner = guild_name
                logger.info("[CastleIntel] %s now owned by %s", castle_id, guild_name)

    # ── Internal ─────────────────────────────────────────────────────

    def _update_castle_state(
        self,
        signals: dict[str, Any],
        map_name: str,
    ) -> None:
        """Update castle state from signals."""
        castle = self.get_castle_by_map(map_name)
        if not castle:
            return

        castle.last_seen = time.time()

        # Update guild owner from players on map
        players = signals.get("players", []) or []
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
                if pg:
                    castle.guild_owner = pg
                    break

        # Count allies and enemies
        allies = 0
        enemies = 0
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
                if pg == self._my_guild:
                    allies += 1
                elif pg:
                    enemies += 1
        castle.allies_nearby = allies
        castle.enemies_nearby = enemies

        # Detect if under attack
        if enemies > 2 and castle.guild_owner == self._my_guild:
            castle.under_attack = True
            castle.last_attack_time = time.time()
        else:
            castle.under_attack = False

        # Update Emperium state
        emp_hp = signals.get("emperium_hp")
        emp_max_hp = signals.get("emperium_max_hp")
        if emp_hp is not None and emp_max_hp and emp_max_hp > 0:
            castle.emperium_alive = emp_hp > 0
            castle.emperium_hp_pct = emp_hp / emp_max_hp

    def _handle_schedule(
        self,
        schedule: WoESchedule,
        actions: list[HeuristicAction],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Handle WoE schedule events."""
        if schedule.is_woe_time and not self._woe_active_this_session:
            self._woe_active_this_session = True
            actions.append(HeuristicAction(
                kind="command",
                command="woe_start",
                confidence=0.99,
                domain="pvp",
                reason="WoE has started! Moving to castle",
                metadata={"action": "woe_start"},
            ))

        if not schedule.is_woe_time and self._woe_active_this_session:
            self._woe_active_this_session = False
            actions.append(HeuristicAction(
                kind="command",
                command="woe_end",
                confidence=0.99,
                domain="pvp",
                reason="WoE has ended — returning to normal operations",
                metadata={"action": "woe_end"},
            ))

        if schedule.should_prep and not self._prep_started:
            self._prep_started = True
            my_class = str(signals.get("job_name", "novice") or "novice").lower()
            checklist = self.get_woe_prep_checklist(my_class)

            actions.append(HeuristicAction(
                kind="command",
                command="woe_prep_start",
                confidence=0.95,
                domain="economy",
                reason=f"WoE in {schedule.minutes_to_woe} min — starting preparation",
                metadata={
                    "action": "woe_prep",
                    "minutes_to_woe": schedule.minutes_to_woe,
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
                    reason=f"WoE prep: buy {item_name} x{item_info['min_count']} (priority {item_info['priority']})",
                    metadata={"item": item_name, "count": item_info["min_count"]},
                ))

    def _check_owned_castles(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Check if any owned castles are under attack."""
        under_attack = self.get_castles_under_attack()
        if under_attack:
            for castle in under_attack:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"defend {castle.castle_id}",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"Castle {castle.name} under attack! Defending!",
                    metadata={
                        "castle_id": castle.castle_id,
                        "castle_name": castle.name,
                        "action": "defend",
                        "enemies": castle.enemies_nearby,
                    },
                ))

    def _emit_castle_intel(
        self,
        castle: CastleInfo,
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Emit castle intelligence information."""
        actions.append(HeuristicAction(
            kind="log",
            command="castle_intel",
            confidence=0.80,
            domain="pvp",
            reason=f"Castle {castle.name}: owned by {castle.guild_owner or 'none'}, "
                   f"EMP {'alive' if castle.emperium_alive else 'dead'} "
                   f"({castle.emperium_hp_pct:.0%}), "
                   f"allies={castle.allies_nearby}, enemies={castle.enemies_nearby}",
            metadata={
                "castle_id": castle.castle_id,
                "owner": castle.guild_owner,
                "emperium_alive": castle.emperium_alive,
                "emperium_hp_pct": castle.emperium_hp_pct,
                "allies": castle.allies_nearby,
                "enemies": castle.enemies_nearby,
                "under_attack": castle.under_attack,
            },
        ))

    def _estimate_prep_cost(self) -> int:
        """Estimate total cost for WoE preparation."""
        total = 0
        for item_name, item_info in WOE_CONSUMABLES.items():
            base_cost = 100  # Approximate base cost
            cost = int(base_cost * item_info["cost_multiplier"] * item_info["min_count"])
            total += cost
        return total


# ── Singleton factory ─────────────────────────────────────────────────────

_castle_intelligence: CastleIntelligence | None = None


def get_castle_intelligence() -> CastleIntelligence:
    """Get or create the singleton CastleIntelligence."""
    global _castle_intelligence
    if _castle_intelligence is None:
        _castle_intelligence = CastleIntelligence()
    return _castle_intelligence
