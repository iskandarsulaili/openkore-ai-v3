"""
Real-Time Party Coordination — sub-second coordination via shared memory.

A pro party doesn't just follow a leader. They execute coordinated tactics:
- Tank pulls -> DPS attacks -> Healer heals -> Scout watches
- Real-time target sharing (everyone attacks the same target)
- Real-time position sharing (formation awareness)
- Real-time cooldown sharing (combo coordination)
- Emergency response (sub-second reaction to party member in danger)

Uses SharedMemoryIPC for ultra-low-latency communication.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from ai_sidecar.domains.social.swarm.shm_ipc import (
    SharedMemoryIPC,
    SharedMemoryCoordination,
    SHM_STATE,
    SHM_ALERTS,
    SHM_COORD,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

PARTY_STATE_TTL = 0.5  # Party state expires after 500ms
TARGET_SHARE_INTERVAL = 0.1  # Share target every 100ms
POSITION_SHARE_INTERVAL = 0.1  # Share position every 100ms
COOLDOWN_SHARE_INTERVAL = 0.5  # Share cooldowns every 500ms
EMERGENCY_RESPONSE_TIMEOUT = 1.0  # Respond to emergency within 1s

# ── Enums ──────────────────────────────────────────────────────────


class PartyRole(Enum):
    """Roles in a coordinated party."""
    TANK = "tank"
    DPS = "dps"
    HEALER = "healer"
    SCOUT = "scout"
    SUPPORT = "support"
    LEADER = "leader"


class PartyTactic(Enum):
    """Tactics the party can execute."""
    FOCUS_FIRE = "focus_fire"  # Everyone attacks the same target
    TANK_PULL = "tank_pull"    # Tank pulls, DPS waits, Healer watches
    KITE = "kite"              # Ranged kites, melee intercepts
    RETREAT = "retreat"        # Everyone falls back
    FORMATION = "formation"    # Hold formation position
    SCATTER = "scatter"        # Spread out (AoE avoidance)
    AMBUSH = "ambush"          # Surround and destroy
    PROTECT = "protect"        # Protect a specific member


class EmergencyType(Enum):
    """Types of emergencies that need immediate response."""
    UNDER_ATTACK = "under_attack"  # Being attacked by multiple mobs
    LOW_HP = "low_hp"              # HP below 30%
    STUNNED = "stunned"            # Stunned/status effect
    TRAPPED = "trapped"            # Surrounded/cornered
    DEAD = "dead"                  # Party member died


# ── Data Models ────────────────────────────────────────────────────


@dataclass
class PartyMemberState:
    """Real-time state of a party member."""
    bot_id: str
    role: PartyRole
    map_name: str = ""
    x: int = 0
    y: int = 0
    hp_pct: float = 1.0
    sp_pct: float = 1.0
    target_id: str = ""
    target_hp_pct: float = 0.0
    is_attacking: bool = False
    is_casting: bool = False
    current_skill: str = ""
    cooldowns: dict[str, float] = field(default_factory=dict)  # skill_id -> remaining_seconds
    aggro_count: int = 0
    status_effects: list[str] = field(default_factory=list)
    last_updated: float = 0.0


@dataclass
class PartyTacticState:
    """Current tactical state of the party."""
    current_tactic: PartyTactic = PartyTactic.FOCUS_FIRE
    target_bot_id: str = ""  # Which bot is the current focus target
    target_monster_id: str = ""  # Which monster is the current focus target
    formation_center_x: int = 0
    formation_center_y: int = 0
    formation_spread: int = 3  # How far apart members should be
    active_combos: list[dict] = field(default_factory=list)  # Active combo chains
    last_tactic_change: float = 0.0
    tactic_cooldown: float = 2.0  # Can't change tactics more than every 2s


# ── RealTimePartyCoordinator ──────────────────────────────────────


class RealTimePartyCoordinator:
    """Sub-second party coordination via shared memory.

    Provides:
    - Tank pulls -> DPS attacks -> Healer heals -> Scout watches
    - Real-time target sharing
    - Real-time position sharing
    - Real-time cooldown sharing
    - Emergency response
    - Tactic switching
    """

    def __init__(self, bot_id: str = "", role: PartyRole = PartyRole.DPS):
        self._lock = RLock()
        self._bot_id = bot_id
        self._role = role
        self._party_members: dict[str, PartyMemberState] = {}
        self._tactic_state = PartyTacticState()
        self._last_target_share: float = 0.0
        self._last_position_share: float = 0.0
        self._last_cooldown_share: float = 0.0
        self._last_emergency_check: float = 0.0
        self._stats: dict[str, int] = {
            "targets_shared": 0,
            "positions_shared": 0,
            "cooldowns_shared": 0,
            "emergencies_handled": 0,
            "tactic_changes": 0,
            "combos_executed": 0,
        }

    # ── State Sharing ──

    def share_target(self, monster_id: str, monster_hp_pct: float) -> None:
        """Share current target with party (sub-second)."""
        now = time.time()
        if now - self._last_target_share < TARGET_SHARE_INTERVAL:
            return

        SharedMemoryIPC.send_state(self._bot_id, "target", {
            "monster_id": monster_id,
            "hp_pct": monster_hp_pct,
            "role": self._role.value,
        })
        self._last_target_share = now
        self._stats["targets_shared"] += 1

    def share_position(self, map_name: str, x: int, y: int, hp_pct: float = 1.0, sp_pct: float = 1.0) -> None:
        """Share current position with party (sub-second)."""
        now = time.time()
        if now - self._last_position_share < POSITION_SHARE_INTERVAL:
            return

        SharedMemoryCoordination.share_position(self._bot_id, map_name, x, y, hp_pct)
        self._last_position_share = now
        self._stats["positions_shared"] += 1

    def share_cooldowns(self, cooldowns: dict[str, float]) -> None:
        """Share skill cooldowns with party."""
        now = time.time()
        if now - self._last_cooldown_share < COOLDOWN_SHARE_INTERVAL:
            return

        SharedMemoryIPC.send_state(self._bot_id, "cooldowns", {
            "cooldowns": cooldowns,
            "role": self._role.value,
        })
        self._last_cooldown_share = now
        self._stats["cooldowns_shared"] += 1

    # ── Party State Reading ──

    def get_party_member_state(self, bot_id: str) -> PartyMemberState | None:
        """Get the latest state of a party member."""
        state = SharedMemoryIPC.read_state(bot_id, "position", max_age=PARTY_STATE_TTL)
        if state is None:
            return None

        data = state.get("data", {})
        target = SharedMemoryIPC.read_state(bot_id, "target", max_age=PARTY_STATE_TTL)
        target_data = target.get("data", {}) if target else {}

        return PartyMemberState(
            bot_id=bot_id,
            role=PartyRole(data.get("role", "dps")),
            map_name=data.get("map", ""),
            x=data.get("x", 0),
            y=data.get("y", 0),
            hp_pct=data.get("hp_pct", 1.0),
            target_id=target_data.get("monster_id", ""),
            target_hp_pct=target_data.get("hp_pct", 0.0),
            last_updated=state.get("ts", 0),
        )

    def get_all_party_states(self) -> list[PartyMemberState]:
        """Get the latest state of all party members."""
        members = []
        for bot_id in list(self._party_members.keys()):
            state = self.get_party_member_state(bot_id)
            if state:
                members.append(state)
        return members

    # ── Tactic Management ──

    def set_tactic(self, tactic: PartyTactic) -> None:
        """Set the current party tactic."""
        now = time.time()
        if now - self._tactic_state.last_tactic_change < self._tactic_state.tactic_cooldown:
            return  # Too soon to change tactic

        with self._lock:
            self._tactic_state.current_tactic = tactic
            self._tactic_state.last_tactic_change = now
            self._stats["tactic_changes"] += 1

        # Broadcast tactic change
        SharedMemoryIPC.send_state("party", "tactic", {
            "tactic": tactic.value,
            "changed_by": self._bot_id,
            "timestamp": now,
        })
        logger.info("party_tactic_changed: %s by %s", tactic.value, self._bot_id)

    def get_current_tactic(self) -> PartyTactic:
        """Get the current party tactic."""
        # Check for tactic updates from other bots
        tactic_state = SharedMemoryIPC.read_state("party", "tactic", max_age=5.0)
        if tactic_state:
            data = tactic_state.get("data", {})
            tactic_str = data.get("tactic", "")
            if tactic_str:
                try:
                    return PartyTactic(tactic_str)
                except ValueError:
                    pass
        return self._tactic_state.current_tactic

    # ── Role-Specific Actions ──

    def get_tank_target(self) -> str | None:
        """Get the tank's current target (for DPS focus fire)."""
        for bot_id, member in self._party_members.items():
            if member.role == PartyRole.TANK and member.target_id:
                return member.target_id
        return None

    def get_healer_target(self) -> str | None:
        """Get the member that needs healing most."""
        lowest_hp = 1.0
        target = None
        for bot_id, member in self._party_members.items():
            if member.hp_pct < lowest_hp and member.hp_pct < 0.7:
                lowest_hp = member.hp_pct
                target = bot_id
        return target

    def get_scout_report(self) -> dict[str, Any]:
        """Get scout's report on nearby threats."""
        for bot_id, member in self._party_members.items():
            if member.role == PartyRole.SCOUT:
                return {
                    "bot_id": bot_id,
                    "aggro_count": member.aggro_count,
                    "map": member.map_name,
                    "position": (member.x, member.y),
                }
        return {}

    # ── Emergency Response ──

    def check_emergencies(self) -> list[dict]:
        """Check for emergencies from party members."""
        now = time.time()
        if now - self._last_emergency_check < 0.2:
            return []  # Check max 5 times per second

        self._last_emergency_check = now
        emergencies = []

        # Check shared memory for alerts
        alerts = SharedMemoryIPC.get_alerts(self._bot_id, since_ts=now - 2.0)
        for alert in alerts:
            if alert.get("type") == "emergency":
                emergencies.append({
                    "bot_id": alert.get("bot_id", ""),
                    "message": alert.get("message", ""),
                    "urgency": alert.get("urgency", 5),
                    "timestamp": alert.get("ts", 0),
                })
                self._stats["emergencies_handled"] += 1

        # Check party member states for low HP
        for bot_id, member in self._party_members.items():
            if member.hp_pct < 0.3 and member.hp_pct > 0:
                emergencies.append({
                    "bot_id": bot_id,
                    "type": "low_hp",
                    "hp_pct": member.hp_pct,
                    "timestamp": now,
                })

        return emergencies

    def send_emergency(self, message: str, urgency: int = 10) -> None:
        """Send an emergency alert to the party."""
        SharedMemoryCoordination.emergency_alert(self._bot_id, message)
        self._stats["emergencies_handled"] += 1

    # ── Combo Coordination ──

    def request_combo(self, target_bot: str, combo_type: str, skill_id: str) -> None:
        """Request a combo execution from another party member."""
        SharedMemoryCoordination.request_combo(self._bot_id, target_bot, combo_type, skill_id)
        self._stats["combos_executed"] += 1

    def check_pending_combos(self) -> list[dict]:
        """Check for pending combo requests."""
        combos = []
        for f in SHM_COORD.glob(f"combo_*_*.json"):
            try:
                data = json.loads(f.read_text())
                age = time.time() - data.get("ts", 0)
                if age < 5.0 and data.get("target") == self._bot_id:
                    combos.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return combos

    # ── Party Member Registration ──

    def register_member(self, bot_id: str, role: PartyRole) -> None:
        """Register a party member for tracking."""
        with self._lock:
            self._party_members[bot_id] = PartyMemberState(
                bot_id=bot_id,
                role=role,
            )
        logger.info("party_member_registered: %s as %s", bot_id, role.value)

    def unregister_member(self, bot_id: str) -> None:
        """Unregister a party member."""
        with self._lock:
            self._party_members.pop(bot_id, None)

    def set_bot_id(self, bot_id: str) -> None:
        self._bot_id = bot_id

    def set_role(self, role: PartyRole) -> None:
        self._role = role

    # ── Tactical Decision Making ──

    def decide_tactic(self, party_states: list[PartyMemberState]) -> PartyTactic:
        """Decide the best tactic based on party state."""
        if not party_states:
            return PartyTactic.FOCUS_FIRE

        # Count roles
        tanks = sum(1 for m in party_states if m.role == PartyRole.TANK)
        healers = sum(1 for m in party_states if m.role == PartyRole.HEALER)
        dps = sum(1 for m in party_states if m.role == PartyRole.DPS)

        # Check for emergencies
        low_hp_members = [m for m in party_states if m.hp_pct < 0.3]
        if low_hp_members:
            return PartyTactic.RETREAT

        # Check for high aggro
        high_aggro = [m for m in party_states if m.aggro_count > 3]
        if high_aggro:
            return PartyTactic.PROTECT

        # Tank present -> tank pull
        if tanks > 0:
            return PartyTactic.TANK_PULL

        # Multiple DPS -> focus fire
        if dps > 1:
            return PartyTactic.FOCUS_FIRE

        # Default
        return PartyTactic.FORMATION

    def get_tactical_context(self) -> str:
        """Get formatted tactical context for LLM prompts."""
        with self._lock:
            lines = ["── Real-Time Party Coordination ──"]
            lines.append(f"Role: {self._role.value}")
            lines.append(f"Current tactic: {self._tactic_state.current_tactic.value}")
            lines.append(f"Party members: {len(self._party_members)}")

            for bot_id, member in self._party_members.items():
                lines.append(
                    f"  {bot_id}: {member.role.value} "
                    f"HP={member.hp_pct:.0%} "
                    f"target={member.target_id or 'none'}"
                )

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ──

_party_coordinator: RealTimePartyCoordinator | None = None
_party_coordinator_lock = RLock()


def get_party_coordinator() -> RealTimePartyCoordinator:
    global _party_coordinator
    with _party_coordinator_lock:
        if _party_coordinator is None:
            _party_coordinator = RealTimePartyCoordinator()
        return _party_coordinator
