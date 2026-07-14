"""
Multi-account synergy — coordinates multiple bots as a team.

A top player with 3 accounts doesn't have 3 bots. They have a team.
Each account has a role. They complement each other. They cover each
other's weaknesses. This module coordinates multi-account operations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TeamRole:
    """A role in the multi-account team."""
    name: str
    primary_task: str  # farmer, buffer, merchant, scout, tank, healer
    secondary_task: str = ""
    bot_id: str = ""
    level: int = 0
    class_name: str = ""
    map: str = ""
    status: str = "idle"  # idle, farming, trading, scouting, returning, dead
    last_updated: float = 0.0


@dataclass
class TeamOrder:
    """An order for a team member."""
    bot_id: str
    order_type: str  # move, attack, buff, trade, return, defend
    target: str = ""
    priority: int = 5
    issued_at: float = 0.0
    completed: bool = False


@dataclass(slots=True)
class MultiAccountSynergy:
    """Coordinates multiple bots as a coordinated team."""
    
    _lock: RLock = field(default_factory=RLock)
    _roles: dict[str, TeamRole] = field(default_factory=dict)
    _orders: list[TeamOrder] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "orders_issued": 0, "orders_completed": 0, "synergies": 0,
    })
    _enqueue_fn: Callable | None = None
    
    def assign_role(self, bot_id: str, primary: str, secondary: str = "", 
                    level: int = 0, class_name: str = "") -> None:
        """Assign a role to a bot."""
        with self._lock:
            role = self._roles.setdefault(bot_id, TeamRole(
                name=bot_id,
                primary_task=primary,
                bot_id=bot_id,
            ))
            role.primary_task = primary
            role.secondary_task = secondary
            role.level = level or role.level
            role.class_name = class_name or role.class_name
            role.last_updated = time.time()
            logger.info("team_role_assigned: %s → %s (secondary: %s)", bot_id, primary, secondary)
    
    def update_status(self, bot_id: str, status: str, map: str = "") -> None:
        """Update a bot's status."""
        with self._lock:
            role = self._roles.get(bot_id)
            if role:
                role.status = status
                if map:
                    role.map = map
                role.last_updated = time.time()
    
    def issue_order(self, bot_id: str, order_type: str, target: str = "", priority: int = 5) -> bool:
        """Issue an order to a team member."""
        with self._lock:
            order = TeamOrder(
                bot_id=bot_id,
                order_type=order_type,
                target=target,
                priority=priority,
                issued_at=time.time(),
            )
            self._orders.append(order)
            self._stats["orders_issued"] += 1
            
            # Execute via enqueue
            if self._enqueue_fn:
                cmd = ""
                if order_type == "move":
                    cmd = f"move {target}"
                elif order_type == "attack":
                    cmd = f"attack {target}"
                elif order_type == "return":
                    cmd = "move prontera"
                elif order_type == "defend":
                    cmd = f"move {target}"
                elif order_type == "buff":
                    cmd = f"use {target}"
                elif order_type == "trade":
                    cmd = f"chat anyone selling {target}?"
                
                if cmd:
                    self._enqueue_fn(bot_id, cmd)
                    logger.info("team_order: %s → %s (%s)", bot_id, order_type, target)
                    return True
            return False
    
    def get_synergy_context(self) -> str:
        """Get formatted team context for LLM prompts."""
        with self._lock:
            lines = ["── Multi-Account Team ──"]
            active = [r for r in self._roles.values() if r.status != "dead"]
            
            if not active:
                lines.append("  No team members assigned.")
                return "\n".join(lines)
            
            lines.append(f"  Active members: {len(active)}")
            for role in active:
                lines.append(f"    {role.bot_id}: {role.primary_task} ({role.status}) on {role.map}")
            
            # Check for synergy opportunities
            farmers = [r for r in active if r.primary_task == "farmer"]
            buffers = [r for r in active if r.primary_task == "buffer"]
            merchants = [r for r in active if r.primary_task == "merchant"]
            
            if farmers and buffers:
                lines.append(f"  Synergy: {len(farmers)} farmer(s) + {len(buffers)} buffer(s) available")
            if merchants:
                lines.append(f"  Merchant available for selling")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_synergy: MultiAccountSynergy | None = None
_synergy_lock = RLock()


def get_multi_account_synergy() -> MultiAccountSynergy:
    global _synergy
    with _synergy_lock:
        if _synergy is None:
            _synergy = MultiAccountSynergy()
        return _synergy
