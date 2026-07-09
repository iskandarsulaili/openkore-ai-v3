"""FleetCoordinatorService — in-process multi-bot state management & role assignment."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    TANK = "tank"
    HEALER = "healer"
    DPS_MELEE = "dps_melee"
    DPS_RANGED = "dps_ranged"
    DPS_MAGIC = "dps_magic"
    SUPPORT = "support"
    BUFFER = "buffer"
    DEBUFFER = "debuff"
    CRAFTER = "crafter"
    MERCHANT = "merchant"
    REFINER = "refiner"
    FARMER = "farmer"
    LOOTER = "looter"
    QUESTER = "quester"
    MVP_HUNTER = "mvp_hunter"
    PVP_ATTACKER = "pvp_attacker"
    PVP_DEFENDER = "pvp_defender"
    GVG_FRONTLINE = "gvg_frontline"
    GVG_SIEGE = "gvg_siege"
    GVG_SUPPORT = "gvg_support"
    SCOUT = "scout"
    IDLE = "idle"


@dataclass(slots=True)
class RoleMetrics:
    role: str
    total_assignments: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    total_damage_dealt: float = 0.0
    total_healing_done: float = 0.0
    total_zeny_earned: float = 0.0
    total_xp_gained: float = 0.0
    deaths: int = 0
    avg_response_time_s: float = 0.0
    last_assigned_at: float = 0.0
    score: float = 0.0

    def success_rate(self) -> float:
        total = self.successful_actions + self.failed_actions
        return (self.successful_actions / total) if total > 0 else 0.5

    def compute_score(self) -> float:
        sr = self.success_rate()
        eff = 0.0
        if self.role in ("dps_melee", "dps_ranged", "dps_magic", "mvp_hunter", "pvp_attacker"):
            eff = min(1.0, self.total_damage_dealt / max(1.0, self.total_assignments * 1000))
        elif self.role in ("healer", "support", "buffer"):
            eff = min(1.0, self.total_healing_done / max(1.0, self.total_assignments * 500))
        elif self.role in ("merchant", "crafter", "refiner"):
            eff = min(1.0, self.total_zeny_earned / max(1.0, max(1, self.total_assignments) * 10000))
        death_penalty = max(0.0, 1.0 - (self.deaths / max(1, self.total_assignments)) * 2)
        self.score = sr * 0.4 + eff * 0.3 + death_penalty * 0.3
        return self.score


@dataclass(slots=True)
class BotFleetState:
    bot_id: str
    position: tuple[int, int] = (0, 0)
    map_name: str = ""
    hp: int = 1
    hp_max: int = 1
    sp: int = 0
    sp_max: int = 1
    level: int = 1
    job_level: int = 1
    zeny: int = 0
    weight: int = 0
    max_weight: int = 10000
    party_id: str = ""
    guild_id: str = ""
    current_role: str = RoleType.IDLE.value
    available_roles: list[str] = field(default_factory=list)
    role_metrics: dict[str, RoleMetrics] = field(default_factory=dict)
    is_online: bool = False
    last_seen_at: float = field(default_factory=time.time)
    active_objective: str = ""
    status_message: str = ""

    def hp_pct(self) -> float:
        return self.hp / max(1, self.hp_max)

    def sp_pct(self) -> float:
        return self.sp / max(1, self.sp_max)

    def weight_pct(self) -> float:
        return self.weight / max(1, self.max_weight)


@dataclass(slots=True)
class FleetMessage:
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    sent_at: float = field(default_factory=time.time)
    ttl_seconds: int = 60


class FleetCoordinatorService:
    """In-process fleet state manager. Thread-safe singleton per sidecar runtime."""

    def __init__(self, max_bots: int = 256, role_rotation_cooldown_s: int = 120):
        self._lock = threading.RLock()
        self._max_bots = max_bots
        self._role_rotation_cooldown = role_rotation_cooldown_s
        self._bots: dict[str, BotFleetState] = {}
        self._messages: dict[str, FleetMessage] = {}
        self._parties: dict[str, list[str]] = {}
        self._objectives: dict[str, dict[str, Any]] = {}
        self._mvp_spawns: dict[str, dict[str, Any]] = {}
        self._shared_inventory: dict[str, int] = defaultdict(int)
        self._shared_zeny: int = 0

    def register_bot(self, bot_id: str, available_roles: list[str] | None = None) -> BotFleetState:
        with self._lock:
            if bot_id in self._bots:
                existing = self._bots[bot_id]
                existing.is_online = True
                existing.last_seen_at = time.time()
                if available_roles:
                    existing.available_roles = list(set(existing.available_roles) | set(available_roles))
                return existing
            if len(self._bots) >= self._max_bots:
                raise RuntimeError(f"Fleet max bots ({self._max_bots}) reached")
            state = BotFleetState(bot_id=bot_id, available_roles=list(available_roles or []), is_online=True)
            self._bots[bot_id] = state
            return state

    def unregister_bot(self, bot_id: str) -> None:
        with self._lock:
            self._bots.pop(bot_id, None)
            for party_id in list(self._parties.keys()):
                members = self._parties.get(party_id, [])
                if bot_id in members:
                    members.remove(bot_id)
                    if not members:
                        del self._parties[party_id]
                    else:
                        self._parties[party_id] = members

    def update_bot_state(self, bot_id: str, **kwargs: Any) -> BotFleetState | None:
        with self._lock:
            state = self._bots.get(bot_id)
            if state is None:
                return None
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.last_seen_at = time.time()
            return state

    def get_bot(self, bot_id: str) -> BotFleetState | None:
        with self._lock:
            return self._bots.get(bot_id)

    def list_bots(self, online_only: bool = True) -> list[BotFleetState]:
        with self._lock:
            raw = list(self._bots.values())
            now = time.time()
            return [b for b in raw if (not online_only) or (b.is_online and (now - b.last_seen_at) < 120)]

    def _metrics_for(self, bot_id: str, role: str) -> RoleMetrics:
        bot = self._bots.get(bot_id)
        if bot is None:
            return RoleMetrics(role=role)
        if role not in bot.role_metrics:
            bot.role_metrics[role] = RoleMetrics(role=role)
        return bot.role_metrics[role]

    def record_role_action(self, bot_id: str, role: str, success: bool,
                           damage: float = 0.0, healing: float = 0.0,
                           zeny: float = 0.0, xp: float = 0.0,
                           death: bool = False, response_time_s: float = 0.0) -> None:
        with self._lock:
            m = self._metrics_for(bot_id, role)
            m.total_assignments += 1
            if success:
                m.successful_actions += 1
            else:
                m.failed_actions += 1
            m.total_damage_dealt += damage
            m.total_healing_done += healing
            m.total_zeny_earned += zeny
            m.total_xp_gained += xp
            if death:
                m.deaths += 1
            if response_time_s > 0:
                prev = m.avg_response_time_s
                m.avg_response_time_s = (prev * (m.total_assignments - 1) + response_time_s) / m.total_assignments
            m.last_assigned_at = time.time()
            m.compute_score()

    def assign_role(self, bot_id: str, target_role: str, reason: str = "coordinator") -> str | None:
        with self._lock:
            bot = self._bots.get(bot_id)
            if bot is None:
                return None
            if target_role not in bot.available_roles:
                target_role = self._find_nearest_role(target_role, bot.available_roles)
                if target_role is None:
                    return None
            prev_role = bot.current_role
            if prev_role == target_role:
                return target_role
            m = bot.role_metrics.get(target_role)
            if m and m.last_assigned_at > 0:
                elapsed = time.time() - m.last_assigned_at
                if elapsed < self._role_rotation_cooldown:
                    pass  # Still assign anyway, just warn
            bot.current_role = target_role
            return target_role

    def recommend_role_change(self, bot_id: str) -> dict[str, Any]:
        with self._lock:
            bot = self._bots.get(bot_id)
            if bot is None or len(bot.available_roles) <= 1:
                return {"should_change": False, "reason": "insufficient_roles"}
            current_metrics = bot.role_metrics.get(bot.current_role)
            if current_metrics is None:
                return {"should_change": False, "reason": "no_metrics"}
            current_score = current_metrics.compute_score()
            best_role = bot.current_role
            best_score = current_score
            for role in bot.available_roles:
                if role == bot.current_role:
                    continue
                m = bot.role_metrics.get(role)
                if m and m.total_assignments >= 5:
                    score = m.compute_score()
                    if score > best_score + 0.15:
                        best_score = score
                        best_role = role
            return {
                "should_change": best_role != bot.current_role,
                "current_role": bot.current_role,
                "recommended_role": best_role,
                "current_score": current_score,
                "recommended_score": best_score,
                "improvement": best_score - current_score,
                "reason": "better_performance" if best_role != bot.current_role else "current_role_optimal",
            }

    def _find_nearest_role(self, target: str, available: list[str]) -> str | None:
        if not available:
            return None
        if target in available:
            return target
        groups = [
            ({"tank", "gvg_frontline", "pvp_defender"}, ["tank", "gvg_frontline", "pvp_defender"]),
            ({"dps_melee", "dps_ranged", "dps_magic", "mvp_hunter", "pvp_attacker", "gvg_siege"},
             ["dps_melee", "dps_ranged", "dps_magic", "mvp_hunter", "pvp_attacker", "gvg_siege"]),
            ({"healer", "support", "buffer", "gvg_support"}, ["healer", "support", "buffer", "gvg_support"]),
            ({"merchant", "crafter", "refiner"}, ["merchant", "crafter", "refiner"]),
            ({"farmer", "looter"}, ["farmer", "looter"]),
        ]
        for group_roles, members in groups:
            if target in group_roles:
                for member in members:
                    if member in available:
                        return member
        return available[0]

    def send_message(self, message: FleetMessage) -> None:
        with self._lock:
            self._messages[message.message_id] = message
            now = time.time()
            stale = [mid for mid, msg in self._messages.items() if now - msg.sent_at > msg.ttl_seconds]
            for mid in stale:
                del self._messages[mid]

    def get_messages_for(self, bot_id: str, since: float = 0.0) -> list[FleetMessage]:
        with self._lock:
            now = time.time()
            return [
                msg for msg in self._messages.values()
                if now - msg.sent_at <= msg.ttl_seconds and msg.sent_at > since
                and (msg.recipient_id in ("*", bot_id) or msg.sender_id == bot_id)
            ]

    def create_party(self, party_id: str, leader_id: str, member_ids: list[str] | None = None) -> bool:
        with self._lock:
            if party_id in self._parties:
                return False
            members = [leader_id] + (member_ids or [])
            self._parties[party_id] = members
            for mid in members:
                bot = self._bots.get(mid)
                if bot:
                    bot.party_id = party_id
            return True

    def disband_party(self, party_id: str) -> None:
        with self._lock:
            members = self._parties.pop(party_id, [])
            for mid in members:
                bot = self._bots.get(mid)
                if bot:
                    bot.party_id = ""

    def party_members(self, party_id: str) -> list[BotFleetState]:
        with self._lock:
            return [s for b in self._parties.get(party_id, []) if (s := self._bots.get(b)) is not None]

    def suggest_party_composition(self, objective: str, available_bots: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            candidates = [self._bots[b] for b in available_bots if b in self._bots] if available_bots else list(self._bots.values())
            candidates = [b for b in candidates if b.is_online]
            role_requirements = self._party_requirements(objective)
            remaining_roles = dict(role_requirements)
            assignment: dict[str, str] = {}
            for bot in candidates:
                best_role, best_score = "", 0.0
                for role, needed in remaining_roles.items():
                    if needed <= 0 or role not in bot.available_roles:
                        continue
                    m = bot.role_metrics.get(role)
                    score = m.compute_score() if m else 0.5
                    if score > best_score:
                        best_score, best_role = score, role
                if best_role:
                    assignment[bot.bot_id] = best_role
                    remaining_roles[best_role] -= 1
            return {
                "objective": objective,
                "required_roles": role_requirements,
                "filled_roles": dict(remaining_roles),
                "assignment": assignment,
                "complete": all(v <= 0 for v in remaining_roles.values()),
            }

    def _party_requirements(self, objective: str) -> dict[str, int]:
        return {
            "farming": {"dps_melee": 1, "dps_ranged": 1, "looter": 1},
            "questing": {"dps_melee": 1, "support": 1, "quester": 1},
            "mvp": {"tank": 1, "healer": 1, "dps_melee": 2, "dps_ranged": 1, "buffer": 1},
            "pvp": {"pvp_attacker": 2, "pvp_defender": 1, "healer": 1, "debuff": 1},
            "gvg": {"gvg_frontline": 2, "gvg_siege": 2, "gvg_support": 1, "healer": 1},
            "trade": {"merchant": 1, "crafter": 1, "farmer": 2},
            "refine": {"refiner": 1, "merchant": 1, "farmer": 1},
            "leveling": {"dps_melee": 1, "healer": 1, "buffer": 1},
        }.get(objective, {"dps_melee": 1})

    def report_mvp_spawn(self, mvp_name: str, map_name: str, position: tuple[int, int],
                         hp: int = 0, hp_max: int = 0, reported_by: str = "") -> None:
        with self._lock:
            self._mvp_spawns[mvp_name] = {
                "mvp_name": mvp_name, "map_name": map_name, "position": position,
                "hp": hp, "hp_max": hp_max, "reported_by": reported_by,
                "reported_at": time.time(), "status": "active",
            }

    def get_active_mvps(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.time()
            return [i for i in self._mvp_spawns.values() if i.get("status") == "active" and (now - i.get("reported_at", 0)) < 600]

    def add_to_shared_zeny(self, amount: int) -> int:
        with self._lock:
            self._shared_zeny += amount
            return self._shared_zeny

    def take_from_shared_zeny(self, amount: int) -> int:
        with self._lock:
            taken = min(amount, self._shared_zeny)
            self._shared_zeny -= taken
            return taken

    def shared_zeny_balance(self) -> int:
        with self._lock:
            return self._shared_zeny

    def add_to_shared_inventory(self, item_name: str, quantity: int) -> int:
        with self._lock:
            self._shared_inventory[item_name] += quantity
            return self._shared_inventory[item_name]

    def take_from_shared_inventory(self, item_name: str, quantity: int) -> int:
        with self._lock:
            available = self._shared_inventory.get(item_name, 0)
            taken = min(quantity, available)
            self._shared_inventory[item_name] -= taken
            if self._shared_inventory[item_name] <= 0:
                del self._shared_inventory[item_name]
            return taken

    def fleet_status(self) -> dict[str, Any]:
        bots = self.list_bots(online_only=False)
        with self._lock:
            parties_data = dict(self._parties)
        return {
            "total_bots": len(self._bots),
            "online_bots": len([b for b in bots if b.is_online]),
            "bots": [{
                "bot_id": b.bot_id, "map_name": b.map_name, "position": list(b.position),
                "level": b.level, "job_level": b.job_level, "hp_pct": b.hp_pct(),
                "sp_pct": b.sp_pct(), "current_role": b.current_role,
                "available_roles": b.available_roles, "party_id": b.party_id,
                "is_online": b.is_online, "zeny": b.zeny, "weight_pct": b.weight_pct(),
                "active_objective": b.active_objective, "status_message": b.status_message,
                "role_scores": {r: m.compute_score() for r, m in b.role_metrics.items()},
            } for b in bots],
            "parties": [{"party_id": pid, "members": mids, "member_count": len(mids)}
                        for pid, mids in parties_data.items()],
            "active_mvps": self.get_active_mvps(),
            "shared_zeny": self._shared_zeny,
            "shared_inventory": dict(self._shared_inventory),
        }

    def blackboard_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bots": {bid: {"map": b.map_name, "role": b.current_role, "hp_pct": b.hp_pct(),
                               "online": b.is_online, "level": b.level} for bid, b in self._bots.items()},
                "parties": dict(self._parties),
                "mvp_spawns": self.get_active_mvps(),
                "shared_zeny": self._shared_zeny,
                "shared_inventory": dict(self._shared_inventory),
                "objectives": dict(self._objectives),
            }
