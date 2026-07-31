"""
WoE Combat AI — real-time navigation to tactical positions, guardian kill
sequencing, emperium break automation, retreat logic, dispel awareness,
guild coordination, and defensive positioning.

Features:
  - WOE time detection from game time (Wed/Sat 20:00-22:00, 21:00-23:00)
  - Emperium targeting with HP tracking
  - Dispel awareness (detect and react to Lex Aeterna, dispel skills)
  - Guild coordination (formation, chokepoint holding)
  - Defensive positioning (barricade, chokepoint)
  - Escape reflex for WOE (teleport when outnumbered or dispelled)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class WoEPhase:
    """Current phase of WoE combat."""
    phase: str = "idle"  # idle, approach, clear_guardians, break_emperium, defend, retreat, escape
    target_position: tuple[int, int] = (0, 0)
    target_monster_id: int = 0
    started_at: float = 0.0
    enemies_nearby: int = 0
    allies_nearby: int = 0
    emperium_hp_pct: float = 1.0
    guardians_alive: int = 0
    is_dispelled: bool = False
    dispelled_at: float = 0.0
    last_dispel_source: str = ""
    formation: str = "scatter"  # scatter, line, wedge, protect
    chokepoint_position: tuple[int, int] = (0, 0)
    barricade_active: bool = False
    escape_triggered: bool = False
    guild_leader_bot_id: str = ""
    castle_name: str = ""


@dataclass
class WoEEnemy:
    """Tracked enemy player in WoE."""
    name: str
    job: str = ""
    guild: str = ""
    last_seen: float = 0.0
    last_position: tuple[int, int] = (0, 0)
    threat_level: int = 50  # 0-100
    has_dispel: bool = False
    has_stun: bool = False
    has_freeze: bool = False


class WoECombatAI:
    """Executes WoE combat tactics in real-time."""

    # WOE schedule: Wed/Sat 20:00-22:00 (SE: 21:00-23:00)
    WOE_DAYS = [2, 5]  # Wednesday=2, Saturday=5 (Monday=0)
    WOE_START_HOUR = 20
    WOE_END_HOUR = 22
    WOE_SE_START_HOUR = 21
    WOE_SE_END_HOUR = 23

    # Known WOE castles per map
    WOE_CASTLES = {
        "prtg_cas01": "Prontera Castle 1",
        "prtg_cas02": "Prontera Castle 2",
        "prtg_cas03": "Prontera Castle 3",
        "prtg_cas04": "Prontera Castle 4",
        "prtg_cas05": "Prontera Castle 5",
        "gefg_cas01": "Geffen Castle 1",
        "gefg_cas02": "Geffen Castle 2",
        "gefg_cas03": "Geffen Castle 3",
        "gefg_cas04": "Geffen Castle 4",
        "gefg_cas05": "Geffen Castle 5",
        "payg_cas01": "Payon Castle 1",
        "payg_cas02": "Payon Castle 2",
        "payg_cas03": "Payon Castle 3",
        "payg_cas04": "Payon Castle 4",
        "payg_cas05": "Payon Castle 5",
        "alde_gld": "Aldebaran Guild Hall",
        "schg_cas01": "Schwaltzvald Castle 1",
        "schg_cas02": "Schwaltzvald Castle 2",
        "schg_cas03": "Schwaltzvald Castle 3",
        "schg_cas04": "Schwaltzvald Castle 4",
        "schg_cas05": "Schwaltzvald Castle 5",
    }

    # Emperium stats
    EMPERIUM_HP = 1000000  # Base HP
    EMPERIUM_DEF = 60
    EMPERIUM_MDEF = 40
    EMPERIUM_ELEMENT = "Neutral"
    EMPERIUM_ELEMENT_LEVEL = 1
    EMPERIUM_SIZE = "Large"
    EMPERIUM_RACE = "Formless"

    # Guardian stats
    GUARDIAN_HP = 500000
    GUARDIAN_DEF = 40
    GUARDIAN_MDEF = 20
    GUARDIAN_ELEMENT = "Neutral"
    GUARDIAN_ELEMENT_LEVEL = 1
    GUARDIAN_SIZE = "Large"
    GUARDIAN_RACE = "Formless"

    # Dispel skills to watch for
    DISPEL_SKILLS = ["lex_aeterna", "lex_divina", "dispel", "magic_rod", "nullify"]
    STUN_SKILLS = ["bash", "sonic_blow", "stun", "bash_stun"]
    FREEZE_SKILLS = ["frost_diver", "storm_gust", "frost_nova"]

    def __init__(self) -> None:
        self._lock = RLock()
        self._phase: WoEPhase = WoEPhase()
        self._castle_map: str = ""
        self._enqueue_fn: Callable | None = None
        self._last_action: float = 0.0
        self._action_cooldown: float = 1.0
        self._tracked_enemies: dict[str, WoEEnemy] = {}
        self._guild_members: list[str] = []
        self._guild_leader: str = ""
        self._woe_active: bool = False
        self._escape_cooldown: float = 0.0
        self._last_dispel_check: float = 0.0

    # ── WOE Time Detection ──

    def is_woe_time(self, game_time: dict | None = None) -> bool:
        """Check if it's currently WOE time.

        Args:
            game_time: Optional game time dict with 'hour', 'minute', 'day_of_week'
                       If None, uses system time.

        Returns:
            True if currently in WOE hours
        """
        if game_time:
            day = game_time.get("day_of_week", datetime.now(timezone.utc).weekday())
            hour = game_time.get("hour", datetime.now(timezone.utc).hour)
        else:
            now = datetime.now(timezone.utc)
            day = now.weekday()
            hour = now.hour

        if day not in self.WOE_DAYS:
            return False

        # Check WOE hours (20:00-22:00)
        if self.WOE_START_HOUR <= hour < self.WOE_END_HOUR:
            return True
        # Check SE WOE hours (21:00-23:00)
        if self.WOE_SE_START_HOUR <= hour < self.WOE_SE_END_HOUR:
            return True

        return False

    def get_woe_time_remaining(self) -> int:
        """Get minutes remaining in current WOE session."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute

        if self.WOE_START_HOUR <= hour < self.WOE_END_HOUR:
            return (self.WOE_END_HOUR - hour) * 60 - minute
        if self.WOE_SE_START_HOUR <= hour < self.WOE_SE_END_HOUR:
            return (self.WOE_SE_END_HOUR - hour) * 60 - minute
        return 0

    # ── Guild Coordination ──

    def set_guild_members(self, members: list[str], leader: str = "") -> None:
        """Set guild member bot IDs for coordination."""
        with self._lock:
            self._guild_members = members
            self._guild_leader = leader or (members[0] if members else "")
            self._phase.guild_leader_bot_id = self._guild_leader
            logger.info("woe_guild_set: %d members, leader=%s", len(members), self._guild_leader)

    def get_guild_members(self) -> list[str]:
        """Get guild member bot IDs."""
        with self._lock:
            return list(self._guild_members)

    def get_formation_positions(self, formation: str, center: tuple[int, int]) -> dict[str, tuple[int, int]]:
        """Get formation positions for guild members.

        Args:
            formation: 'scatter', 'line', 'wedge', 'protect'
            center: Center position (x, y)

        Returns:
            Dict mapping bot_id to (x, y) position
        """
        with self._lock:
            members = list(self._guild_members)
            if not members:
                return {}

            positions: dict[str, tuple[int, int]] = {}
            cx, cy = center

            if formation == "line":
                # Horizontal line
                for i, bot_id in enumerate(members):
                    offset = (i - len(members) / 2) * 3
                    positions[bot_id] = (int(cx + offset), cy)
            elif formation == "wedge":
                # V-formation
                for i, bot_id in enumerate(members):
                    offset = (i + 1) * 2
                    positions[bot_id] = (int(cx - offset), int(cy - offset))
            elif formation == "protect":
                # Circle around center (protect healer)
                import math
                for i, bot_id in enumerate(members):
                    angle = (2 * math.pi / len(members)) * i
                    positions[bot_id] = (
                        int(cx + 5 * math.cos(angle)),
                        int(cy + 5 * math.sin(angle)),
                    )
            else:
                # Scatter: spread out
                for i, bot_id in enumerate(members):
                    positions[bot_id] = (int(cx + (i % 3 - 1) * 4), int(cy + (i // 3 - 1) * 4))

            return positions

    # ── Dispel Awareness ──

    def check_dispel(self, snapshot: dict) -> bool:
        """Check if the bot has been dispelled.

        Scans snapshot for status effects indicating dispel.

        Args:
            snapshot: Bot state snapshot

        Returns:
            True if dispelled
        """
        now = time.time()
        if now - self._last_dispel_check < 1.0:
            return self._phase.is_dispelled
        self._last_dispel_check = now

        with self._lock:
            # Check status effects
            statuses = snapshot.get("statuses", snapshot.get("status", []))
            if isinstance(statuses, dict):
                statuses = list(statuses.keys())

            # Check for dispel indicators
            dispel_indicators = ["lex_aeterna", "dispel", "magic_rod", "silence", "curse"]
            for status in statuses:
                status_lower = str(status).lower()
                for indicator in dispel_indicators:
                    if indicator in status_lower:
                        self._phase.is_dispelled = True
                        self._phase.dispelled_at = now
                        self._phase.last_dispel_source = status
                        logger.warning("woe_dispel_detected: %s", status)
                        return True

            # Check for buffs missing (indicates dispel happened)
            buffs = snapshot.get("buffs", [])
            essential_buffs = ["blessing", "increase_agility", "kyrie_eleison", "impositio_manus"]
            if buffs and isinstance(buffs, list):
                active_buffs = [b.lower() if isinstance(b, str) else b.get("name", "").lower() for b in buffs]
                missing = [b for b in essential_buffs if b not in str(active_buffs)]
                if len(missing) >= 2 and self._phase.phase in ("break_emperium", "clear_guardians"):
                    # Multiple essential buffs missing in combat = likely dispelled
                    self._phase.is_dispelled = True
                    self._phase.dispelled_at = now
                    logger.warning("woe_dispel_suspected: missing buffs %s", missing)
                    return True

            self._phase.is_dispelled = False
            return False

    def handle_dispel_reaction(self) -> str | None:
        """Get action to take when dispelled.

        Returns:
            Action string or None
        """
        with self._lock:
            if not self._phase.is_dispelled:
                return None

            # Escape if dispelled in combat
            if self._phase.enemies_nearby > 0:
                self._phase.escape_triggered = True
                logger.warning("woe_dispel_escape: dispelled with %d enemies nearby", self._phase.enemies_nearby)
                return "escape_and_rebuff"

            # Re-buff if safe
            return "rebuff"

    # ── Emperium Targeting ──

    def get_emperium_info(self) -> dict:
        """Get emperium stats for damage calculation."""
        return {
            "hp": self.EMPERIUM_HP,
            "def": self.EMPERIUM_DEF,
            "mdef": self.EMPERIUM_MDEF,
            "element": self.EMPERIUM_ELEMENT,
            "element_level": self.EMPERIUM_ELEMENT_LEVEL,
            "size": self.EMPERIUM_SIZE,
            "race": self.EMPERIUM_RACE,
        }

    def get_guardian_info(self) -> dict:
        """Get guardian stats for damage calculation."""
        return {
            "hp": self.GUARDIAN_HP,
            "def": self.GUARDIAN_DEF,
            "mdef": self.GUARDIAN_MDEF,
            "element": self.GUARDIAN_ELEMENT,
            "element_level": self.GUARDIAN_ELEMENT_LEVEL,
            "size": self.GUARDIAN_SIZE,
            "race": self.GUARDIAN_RACE,
        }

    def estimate_emperium_break_time(self, dps: float, party_dps: float = 0) -> float:
        """Estimate time to break emperium in seconds.

        Args:
            dps: Player's DPS
            party_dps: Additional party DPS

        Returns:
            Estimated seconds to break
        """
        total_dps = max(1, dps + party_dps)
        return self.EMPERIUM_HP / total_dps

    # ── Defensive Positioning ──

    def get_chokepoint_position(self, castle_map: str) -> tuple[int, int]:
        """Get the chokepoint position for a castle.

        Returns (x, y) position to hold.
        """
        # Known chokepoints for common castles
        chokepoints = {
            "prtg_cas01": (120, 120),
            "prtg_cas02": (100, 150),
            "prtg_cas03": (140, 100),
            "prtg_cas04": (80, 130),
            "prtg_cas05": (110, 110),
            "gefg_cas01": (90, 90),
            "gefg_cas02": (130, 130),
            "payg_cas01": (100, 100),
            "alde_gld": (150, 150),
        }
        return chokepoints.get(castle_map, (100, 100))

    def get_barricade_position(self, castle_map: str) -> tuple[int, int]:
        """Get the barricade/defense position for a castle."""
        barricades = {
            "prtg_cas01": (115, 115),
            "prtg_cas02": (95, 145),
            "prtg_cas03": (135, 95),
            "gefg_cas01": (85, 85),
        }
        return barricades.get(castle_map, (95, 95))

    # ── Escape Reflex ──

    def should_escape(self, enemies: int, allies: int, hp_pct: float) -> bool:
        """Check if the bot should escape.

        Escape conditions:
          - Outnumbered 3:1 or worse
          - HP below 30%
          - Dispel + enemies nearby
          - Emperium is dead and enemies are near

        Args:
            enemies: Number of nearby enemies
            allies: Number of nearby allies
            hp_pct: Current HP percentage

        Returns:
            True if should escape
        """
        now = time.time()

        # Cooldown on escape to prevent ping-pong
        if now < self._escape_cooldown:
            return False

        with self._lock:
            # Outnumbered 3:1
            if allies <= 0 and enemies >= 3:
                self._escape_cooldown = now + 10
                return True
            if allies > 0 and enemies / max(allies, 1) >= 3:
                self._escape_cooldown = now + 10
                return True

            # Low HP
            if hp_pct < 0.3 and enemies > 0:
                self._escape_cooldown = now + 15
                return True

            # Dispel + enemies
            if self._phase.is_dispelled and enemies > 0:
                self._escape_cooldown = now + 20
                return True

            return False

    # ── Public API ──

    def set_castle(self, map_name: str) -> None:
        with self._lock:
            self._castle_map = map_name
            castle_name = self.WOE_CASTLES.get(map_name, map_name)
            self._phase = WoEPhase(phase="approach", started_at=time.time(), castle_name=castle_name)
            self._woe_active = True
            logger.info("woe_castle_set: %s (%s)", castle_name, map_name)

    def update_battlefield(self, enemies: int = 0, allies: int = 0,
                           guardians: int = 0, emperium_hp: float = 1.0) -> None:
        """Update battlefield state and adjust phase."""
        with self._lock:
            self._phase.enemies_nearby = enemies
            self._phase.allies_nearby = allies
            self._phase.guardians_alive = guardians
            self._phase.emperium_hp_pct = emperium_hp

            # Phase transitions
            if self._phase.phase == "approach" and guardians > 0:
                self._phase.phase = "clear_guardians"
                self._phase.formation = "wedge"
                logger.info("woe_phase: clear_guardians (%d alive)", guardians)
            elif self._phase.phase == "clear_guardians" and guardians == 0:
                self._phase.phase = "break_emperium"
                self._phase.formation = "line"
                logger.info("woe_phase: break_emperium")
            elif self._phase.phase == "break_emperium" and emperium_hp <= 0:
                self._phase.phase = "defend"
                self._phase.formation = "protect"
                logger.info("woe_phase: defend (emperium broken)")
            elif enemies > allies * 2 and self._phase.phase not in ("retreat", "escape"):
                self._phase.phase = "retreat"
                self._phase.formation = "scatter"
                logger.info("woe_phase: retreat (outnumbered %d vs %d)", enemies, allies)

    def get_action(self) -> str | None:
        """Get the next action based on current phase."""
        with self._lock:
            now = time.time()
            if now - self._last_action < self._action_cooldown:
                return None
            self._last_action = now

            # Check dispel reaction first (highest priority)
            if self._phase.is_dispelled:
                if self._phase.enemies_nearby > 0:
                    return "escape_and_rebuff"
                return "rebuff"

            if self._phase.phase == "approach":
                return "move_to_castle_entrance"
            elif self._phase.phase == "clear_guardians":
                return "attack_nearest_guardian"
            elif self._phase.phase == "break_emperium":
                return "attack_emperium"
            elif self._phase.phase == "defend":
                return "hold_chokepoint"
            elif self._phase.phase == "retreat":
                return "retreat_to_safe_zone"
            elif self._phase.phase == "escape":
                return "emergency_teleport"
            return None

    def get_phase(self) -> WoEPhase:
        with self._lock:
            return self._phase

    def get_woe_summary(self) -> str:
        with self._lock:
            status = "ACTIVE" if self._woe_active else "INACTIVE"
            time_left = self.get_woe_time_remaining()
            time_str = f"{time_left}m remaining" if time_left > 0 else "Not WOE time"

            return (
                f"── WoE Combat ──\n"
                f"Status: {status} ({time_str})\n"
                f"Phase: {self._phase.phase}\n"
                f"Castle: {self._phase.castle_name or self._castle_map}\n"
                f"Enemies: {self._phase.enemies_nearby} | Allies: {self._phase.allies_nearby}\n"
                f"Guardians: {self._phase.guardians_alive} | Emperium: {self._phase.emperium_hp_pct:.0%}\n"
                f"Formation: {self._phase.formation}\n"
                f"Dispelled: {'YES' if self._phase.is_dispelled else 'no'}\n"
                f"Guild members: {len(self._guild_members)}"
            )

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._phase = WoEPhase()
            self._castle_map = ""
            self._woe_active = False
            self._tracked_enemies.clear()
            self._escape_cooldown = 0.0


# ── Global Singleton ──

_woe_combat: WoECombatAI | None = None
_woe_combat_lock = RLock()


def get_woe_combat_ai() -> WoECombatAI:
    global _woe_combat
    with _woe_combat_lock:
        if _woe_combat is None:
            _woe_combat = WoECombatAI()
        return _woe_combat
