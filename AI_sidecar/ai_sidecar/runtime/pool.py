"""Resource Pooling — shared bank, pooled zeny/inventory across bots.

Three bots with three separate economies is wasteful. This module:
- Tracks total pooled resources (sum of all bots' zeny + item value)
- Recommends resource transfers (Bot2 has 20k zeny, Bot1 needs potions)
- Tracks shared equipment (the +7 Staff goes to whoever needs it most)
"""
from __future__ import annotations
from typing import Any
import logging
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class ResourcePool:
    """Cross-bot resource pooling.
    
    Instead of each bot having its own economy, track:
    - Total zeny across all bots
    - Redistribute from rich bots to poor bots
    - Shared equipment pool
    """
    
    def __init__(self):
        self._pool: dict[str, float] = {}  # bot_id -> total assets
    
    def update_bot_assets(self, bot_id: str, zeny: int, item_value: int = 0) -> None:
        self._pool[bot_id] = float(zeny) + float(item_value) * 0.5  # items at 50% liquidation
    
    def get_total_assets(self) -> float:
        return sum(self._pool.values())
    
    def get_poorest_bot(self, exclude: str = "") -> str | None:
        """Get the bot with the least assets."""
        candidates = {k: v for k, v in self._pool.items() if k != exclude}
        if not candidates:
            return None
        return min(candidates, key=candidates.get)
    
    def get_richest_bot(self, exclude: str = "") -> str | None:
        """Get the bot with the most assets."""
        candidates = {k: v for k, v in self._pool.items() if k != exclude}
        if not candidates:
            return None
        return max(candidates, key=candidates.get)
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Check if resource redistribution is needed."""
        zeny = int(signals.get("zeny", 0) or 0)
        self.update_bot_assets(bot_id, zeny)
        
        # If this bot is the poorest and has < 1000 zeny, request transfer
        poorest = self.get_poorest_bot()
        if poorest == bot_id and zeny < 1000:
            richest = self.get_richest_bot(exclude=bot_id)
            if richest and self._pool.get(richest, 0) > 5000:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"transfer_zeny {richest} 1000",
                    confidence=0.6,
                    reason=f"Resource pool: requesting 1000z from {richest} (has {self._pool.get(richest, 0):.0f}z)",
                    domain="economy",
                ))
        
        # Log pool status
        actions.append(HeuristicAction(
            kind="log",
            command=f"resource_pool bot={bot_id} zeny={zeny} total={self.get_total_assets():.0f}",
            confidence=0.5,
            reason=f"Resource pool: {self.get_total_assets():.0f}z across {len(self._pool)} bots",
            domain="economy",
        ))
