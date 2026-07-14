"""
Cross-Bot Resource Manager — tracks inventory across all bot instances,
automatically transfers items and zeny between bots, and optimizes
resource allocation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class BotInventory:
    """Inventory snapshot for a single bot."""
    bot_id: str
    zeny: int = 0
    items: dict[str, int] = field(default_factory=dict)  # item_name -> quantity
    potion_count: int = 0
    weight_pct: float = 0.0
    last_updated: float = 0.0


@dataclass
class TransferRequest:
    """A request to transfer items between bots."""
    from_bot: str
    to_bot: str
    item_name: str
    quantity: int = 1
    priority: int = 50
    reason: str = ""
    is_complete: bool = False
    created_at: float = 0.0


class CrossBotResourceManager:
    """Manages resources across all bot instances."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._inventories: dict[str, BotInventory] = {}
        self._transfer_requests: list[TransferRequest] = []
        self._total_transfers: int = 0
        self._total_zeny_transferred: int = 0
        self._enqueue_fn: Callable | None = None

    # ── Public API ──

    def update_inventory(self, bot_id: str, zeny: int = 0, items: dict[str, int] | None = None,
                         potion_count: int = 0, weight_pct: float = 0.0) -> None:
        """Update inventory for a bot."""
        with self._lock:
            self._inventories[bot_id] = BotInventory(
                bot_id=bot_id,
                zeny=zeny,
                items=items or {},
                potion_count=potion_count,
                weight_pct=weight_pct,
                last_updated=time.time(),
            )
            self._check_imbalances()

    def _check_imbalances(self) -> None:
        """Check for resource imbalances between bots."""
        if len(self._inventories) < 2:
            return

        bots = list(self._inventories.values())
        avg_zeny = sum(b.zeny for b in bots) / len(bots)
        avg_potions = sum(b.potion_count for b in bots) / len(bots)

        for bot in bots:
            # Check zeny imbalance
            if bot.zeny > avg_zeny * 2:
                # This bot has too much zeny, transfer to poorer bots
                for other in bots:
                    if other.zeny < avg_zeny * 0.5 and other.bot_id != bot.bot_id:
                        transfer = int((bot.zeny - avg_zeny) * 0.5)
                        if transfer > 10000:
                            self._transfer_requests.append(TransferRequest(
                                from_bot=bot.bot_id,
                                to_bot=other.bot_id,
                                item_name="zeny",
                                quantity=transfer,
                                priority=70,
                                reason=f"Balance zeny: {bot.bot_id} has {bot.zeny}, {other.bot_id} has {other.zeny}",
                            ))

            # Check potion imbalance
            if bot.potion_count > avg_potions * 2:
                for other in bots:
                    if other.potion_count < avg_potions * 0.5 and other.bot_id != bot.bot_id:
                        transfer = int((bot.potion_count - avg_potions) * 0.5)
                        if transfer > 10:
                            self._transfer_requests.append(TransferRequest(
                                from_bot=bot.bot_id,
                                to_bot=other.bot_id,
                                item_name="White Potion",
                                quantity=transfer,
                                priority=60,
                                reason=f"Balance potions: {bot.bot_id} has {bot.potion_count}, {other.bot_id} has {other.potion_count}",
                            ))

    def get_pending_transfers(self) -> list[TransferRequest]:
        with self._lock:
            return [t for t in self._transfer_requests if not t.is_complete]

    def mark_transfer_complete(self, from_bot: str, to_bot: str, item_name: str) -> None:
        with self._lock:
            for req in self._transfer_requests:
                if not req.is_complete and req.from_bot == from_bot and req.to_bot == to_bot and req.item_name == item_name:
                    req.is_complete = True
                    self._total_transfers += 1
                    if item_name == "zeny":
                        self._total_zeny_transferred += req.quantity
                    break

    def get_bot_inventory(self, bot_id: str) -> BotInventory | None:
        with self._lock:
            return self._inventories.get(bot_id)

    def get_resource_summary(self) -> str:
        with self._lock:
            lines = [f"── Cross-Bot Resource Manager ──"]
            lines.append(f"Bots tracked: {len(self._inventories)}")
            lines.append(f"Total transfers: {self._total_transfers} ({self._total_zeny_transferred:,}z)")
            pending = self.get_pending_transfers()
            if pending:
                lines.append(f"Pending transfers: {len(pending)}")
                for p in pending[:3]:
                    lines.append(f"  {p.from_bot} -> {p.to_bot}: {p.quantity}x {p.item_name}")
            for bot_id, inv in self._inventories.items():
                lines.append(f"  {bot_id}: {inv.zeny:,}z, {inv.potion_count} potions, {inv.weight_pct*100:.0f}% weight")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._inventories.clear()
            self._transfer_requests.clear()
            self._total_transfers = 0
            self._total_zeny_transferred = 0


# ── Global Singleton ──

_cross_bot_mgr: CrossBotResourceManager | None = None
_cross_bot_mgr_lock = RLock()


def get_cross_bot_resource_manager() -> CrossBotResourceManager:
    global _cross_bot_mgr
    with _cross_bot_mgr_lock:
        if _cross_bot_mgr is None:
            _cross_bot_mgr = CrossBotResourceManager()
        return _cross_bot_mgr
