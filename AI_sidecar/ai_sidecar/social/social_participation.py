"""
Social participation — plays with strangers, trades, joins guilds.

Ragnarok Online is an MMO. Your bots should participate: party with
strangers, trade with players, join guilds, attend events. This module
provides the hooks for social MMO behaviors.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class GuildInfo:
    """Information about a guild."""
    name: str
    members: int = 0
    level: int = 1
    is_accepting: bool = False
    requirement: str = ""
    observed_at: float = 0.0


@dataclass
class TradeOffer:
    """A trade offer from another player."""
    player: str
    item_wanted: str = ""
    item_offered: str = ""
    zeny_offered: int = 0
    zeny_wanted: int = 0
    observed_at: float = 0.0


@dataclass(slots=True)
class SocialParticipation:
    """Participates in MMO social activities."""
    
    _lock: RLock = field(default_factory=RLock)
    _guilds: dict[str, GuildInfo] = field(default_factory=dict)
    _trade_offers: list[TradeOffer] = field(default_factory=list)
    _party_invites: int = 0
    _party_accepts: int = 0
    _stats: dict[str, int] = field(default_factory=lambda: {
        "parties": 0, "trades": 0, "guild_applications": 0, "events": 0,
    })
    _enqueue_fn: Callable | None = None
    
    def record_guild(self, name: str, members: int = 0, level: int = 1, accepting: bool = False) -> None:
        with self._lock:
            guild = self._guilds.setdefault(name, GuildInfo(name=name))
            guild.members = members or guild.members
            guild.level = level or guild.level
            guild.is_accepting = accepting
            guild.observed_at = time.time()
    
    def accept_party_invite(self, leader_name: str) -> bool:
        """Accept a party invitation."""
        with self._lock:
            self._party_invites += 1
            self._party_accepts += 1
            self._stats["parties"] += 1
        logger.info("social_party_joined: leader=%s", leader_name)
        if self._enqueue_fn:
            self._enqueue_fn("default", f"chat thanks for party invite {leader_name}")
        return True
    
    def apply_to_guild(self, guild_name: str) -> bool:
        """Apply to join a guild."""
        with self._lock:
            self._stats["guild_applications"] += 1
        logger.info("social_guild_applied: %s", guild_name)
        if self._enqueue_fn:
            self._enqueue_fn("default", f"chat can i join {guild_name}?")
        return True
    
    def respond_to_trade(self, offer: TradeOffer, accept: bool = False) -> bool:
        """Respond to a trade offer."""
        with self._lock:
            self._stats["trades"] += 1
        if accept:
            logger.info("social_trade_accepted: %s wants %s", offer.player, offer.item_wanted)
        else:
            logger.info("social_trade_declined: %s", offer.player)
        return accept
    
    def attend_event(self, event_name: str) -> bool:
        """Attend a server event."""
        with self._lock:
            self._stats["events"] += 1
        logger.info("social_event_attending: %s", event_name)
        if self._enqueue_fn:
            self._enqueue_fn("default", f"chat attending {event_name}")
        return True
    
    def get_participation_context(self) -> str:
        """Get formatted participation context for LLM prompts."""
        with self._lock:
            lines = ["── Social Participation ──"]
            lines.append(f"  Guilds observed: {len(self._guilds)}")
            lines.append(f"  Parties joined: {self._party_accepts}/{max(self._party_invites, 1)}")
            lines.append(f"  Trades: {self._stats['trades']}")
            lines.append(f"  Events attended: {self._stats['events']}")
            
            open_guilds = [g for g in self._guilds.values() if g.is_accepting]
            if open_guilds:
                lines.append(f"  Guilds accepting members: {len(open_guilds)}")
                for g in open_guilds[:3]:
                    lines.append(f"    {g.name} ({g.members} members)")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_participation: SocialParticipation | None = None
_participation_lock = RLock()


def get_social_participation() -> SocialParticipation:
    global _participation
    with _participation_lock:
        if _participation is None:
            _participation = SocialParticipation()
        return _participation
