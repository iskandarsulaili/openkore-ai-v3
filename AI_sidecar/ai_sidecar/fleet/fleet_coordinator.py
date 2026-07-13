"""
Fleet coordination system — multi-bot party management and synchronization.

A top player runs 5+ accounts simultaneously. One farms, one buffs,
one merchants, one scouts, one PvPs. They coordinate like a well-oiled
machine. This module makes our bots work as a team.

Key capabilities:
- Party formation and management
- Buff coordination (Priest, Sage, etc.)
- Shared threat detection
- Coordinated retreat/attack
- Resource sharing (zeny, items)
- Role assignment (farmer, buffer, merchant, scout, WoE alt)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class BotRole:
    """Role assignment for a bot in the fleet."""
    bot_id: str
    role: str  # farmer, buffer, merchant, scout, woe_alt, crafter
    priority: int = 5  # 1-10, higher = more important
    class_name: str = ""
    level: int = 1
    map: str = ""
    status: str = "idle"  # idle, farming, following, trading, scouting, fighting
    last_seen: float = 0.0


@dataclass
class FleetOrder:
    """An order issued to a specific bot."""
    bot_id: str
    command: str
    reason: str
    issued_at: float = 0.0
    expires_at: float = 0.0
    priority: int = 5


@dataclass(slots=True)
class FleetCoordinator:
    """Coordinates multiple bots as a team."""
    
    _lock: RLock = field(default_factory=RLock)
    _bots: dict[str, BotRole] = field(default_factory=dict)
    _orders: list[FleetOrder] = field(default_factory=list)
    _party_id: str = ""
    _shared_threats: list[dict[str, Any]] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "orders_issued": 0, "orders_completed": 0, "threats_shared": 0,
        "party_formed": 0, "buffs_cast": 0,
    })
    _enqueue_fn: Callable | None = None
    
    def register_bot(self, bot_id: str, role: str, class_name: str = "", level: int = 1) -> None:
        """Register a bot with the fleet."""
        with self._lock:
            self._bots[bot_id] = BotRole(
                bot_id=bot_id,
                role=role,
                class_name=class_name,
                level=level,
                last_seen=time.time(),
            )
            logger.info("fleet_bot_registered: bot=%s role=%s class=%s", bot_id, role, class_name)
    
    def update_bot_status(self, bot_id: str, *, map: str = "", status: str = "", level: int = 0) -> None:
        """Update a bot's status."""
        with self._lock:
            bot = self._bots.get(bot_id)
            if bot:
                if map:
                    bot.map = map
                if status:
                    bot.status = status
                if level:
                    bot.level = level
                bot.last_seen = time.time()
    
    def get_bots_by_role(self, role: str) -> list[BotRole]:
        """Get all bots with a specific role."""
        with self._lock:
            return [b for b in self._bots.values() if b.role == role]
    
    def get_bots_by_map(self, map_name: str) -> list[BotRole]:
        """Get all bots on a specific map."""
        with self._lock:
            return [b for b in self._bots.values() if b.map == map_name]
    
    def get_farmer(self) -> BotRole | None:
        """Get the primary farmer bot."""
        farmers = self.get_bots_by_role("farmer")
        return max(farmers, key=lambda b: b.level) if farmers else None
    
    def get_buffer(self) -> BotRole | None:
        """Get the best buffer bot."""
        buffers = self.get_bots_by_role("buffer")
        return max(buffers, key=lambda b: b.level) if buffers else None
    
    def issue_order(self, bot_id: str, command: str, reason: str, priority: int = 5, ttl_seconds: int = 30) -> bool:
        """Issue an order to a specific bot."""
        now = time.time()
        order = FleetOrder(
            bot_id=bot_id,
            command=command,
            reason=reason,
            issued_at=now,
            expires_at=now + ttl_seconds,
            priority=priority,
        )
        with self._lock:
            self._orders.append(order)
            self._stats["orders_issued"] += 1
        logger.info("fleet_order_issued: bot=%s cmd=%s reason=%s", bot_id, command, reason)
        
        # If we have an enqueue function, push the action
        if self._enqueue_fn is not None:
            try:
                self._enqueue_fn(bot_id, command)
                return True
            except Exception:
                pass
        return True
    
    def share_threat(self, threat: dict[str, Any]) -> None:
        """Share a threat detection with all bots."""
        with self._lock:
            threat["shared_at"] = time.time()
            self._shared_threats.append(threat)
            self._stats["threats_shared"] += 1
        logger.info("fleet_threat_shared: type=%s map=%s", threat.get("type"), threat.get("map"))
    
    def get_active_threats(self, max_age_seconds: int = 300) -> list[dict[str, Any]]:
        """Get recent threats."""
        now = time.time()
        with self._lock:
            return [t for t in self._shared_threats if now - t.get("shared_at", 0) < max_age_seconds]
    
    def form_party(self) -> bool:
        """Form a party with all registered bots."""
        with self._lock:
            bot_ids = list(self._bots.keys())
            if len(bot_ids) < 2:
                return False
            self._party_id = f"fleet-{int(time.time())}"
            self._stats["party_formed"] += 1
        logger.info("fleet_party_formed: bots=%s", bot_ids)
        return True
    
    def get_party_summary(self) -> str:
        """Get a human-readable summary of the fleet state."""
        with self._lock:
            if not self._bots:
                return "No bots registered in fleet."
            
            lines = [f"── Fleet Party ({len(self._bots)} bots) ──"]
            for bot_id, bot in sorted(self._bots.items()):
                lines.append(
                    f"  {bot_id}: {bot.role} ({bot.class_name} Lv.{bot.level}) "
                    f"on {bot.map} — {bot.status}"
                )
            
            threats = self.get_active_threats()
            if threats:
                lines.append(f"  Active threats: {len(threats)}")
                for t in threats[:3]:
                    lines.append(f"    {t.get('type')} on {t.get('map')}")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_coordinator: FleetCoordinator | None = None
_coordinator_lock = RLock()


def get_fleet() -> FleetCoordinator:
    """Get or create the global fleet coordinator."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = FleetCoordinator()
        return _coordinator
