"""
Social Intelligence Module — party networking, trading, information sharing.

The real efficiency gains in RO come from party play, trading, and information
networks. A solo bot is a capped bot. This module enables social interaction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PartyMember:
    """A party member."""
    name: str
    job_class: str = "unknown"
    level: int = 1
    hp_pct: float = 1.0
    sp_pct: float = 1.0
    distance: float = 0.0
    is_online: bool = True
    role: str = "unknown"  # tank, healer, dps, support
    last_seen: float = 0.0


@dataclass
class TradeOffer:
    """A trade offer."""
    item_name: str
    quantity: int = 1
    price_per_unit: int = 0
    offer_type: str = "sell"  # buy or sell
    trader_name: str = ""
    timestamp: float = 0.0
    is_active: bool = True


@dataclass
class SocialOpportunity:
    """A social opportunity."""
    opportunity_type: str  # party, trade, info, alliance
    target_name: str = ""
    description: str = ""
    value_estimate: float = 0.0
    risk_level: str = "low"  # low, medium, high
    priority: int = 50
    timestamp: float = 0.0


class SocialIntelligence:
    """Manages social interactions — parties, trading, information sharing."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._party_members: dict[str, PartyMember] = {}
        self._trade_offers: list[TradeOffer] = []
        self._social_opportunities: list[SocialOpportunity] = []
        self._known_players: dict[str, dict] = {}  # player_name -> reputation data
        self._party_invites_pending: list[str] = []
        self._max_trade_offers: int = 50
        self._max_opportunities: int = 50
        self._party_leader: bool = False
        self._party_name: str = ""
        self._guild_name: str = ""
        self._is_in_party: bool = False
        self._is_in_guild: bool = False
        self._chat_enabled: bool = True
        self._auto_invite_enabled: bool = False
        self._auto_trade_enabled: bool = False
        self._enqueue_fn: Callable | None = None

    # ── Party Management ──

    def update_party_members(self, members: list[dict]) -> None:
        with self._lock:
            now = time.time()
            seen: set[str] = set()
            for m in members:
                name = str(m.get("name", ""))
                if not name:
                    continue
                seen.add(name)
                if name in self._party_members:
                    pm = self._party_members[name]
                    pm.hp_pct = float(m.get("hp_pct", 1.0))
                    pm.sp_pct = float(m.get("sp_pct", 1.0))
                    pm.distance = float(m.get("distance", 0))
                    pm.is_online = True
                    pm.last_seen = now
                else:
                    self._party_members[name] = PartyMember(
                        name=name,
                        job_class=str(m.get("job_class", "unknown")),
                        level=int(m.get("level", 1)),
                        hp_pct=float(m.get("hp_pct", 1.0)),
                        sp_pct=float(m.get("sp_pct", 1.0)),
                        distance=float(m.get("distance", 0)),
                        is_online=True,
                        role=str(m.get("role", "unknown")),
                        last_seen=now,
                    )

            # Mark missing members as offline
            for name, pm in self._party_members.items():
                if name not in seen:
                    pm.is_online = False

    def get_party_members(self) -> list[PartyMember]:
        with self._lock:
            return list(self._party_members.values())

    def get_online_party_members(self) -> list[PartyMember]:
        with self._lock:
            return [m for m in self._party_members.values() if m.is_online]

    def get_party_size(self) -> int:
        with self._lock:
            return len([m for m in self._party_members.values() if m.is_online])

    def get_party_composition(self) -> dict[str, int]:
        with self._lock:
            comp: dict[str, int] = {}
            for m in self._party_members.values():
                if m.is_online:
                    comp[m.job_class] = comp.get(m.job_class, 0) + 1
            return comp

    def get_party_hp_status(self) -> str:
        with self._lock:
            low_hp = [m.name for m in self._party_members.values() if m.is_online and m.hp_pct < 0.5]
            if low_hp:
                return f"Low HP: {', '.join(low_hp)}"
            return "Party HP OK"

    def get_party_bonus_xp(self) -> float:
        size = self.get_party_size()
        if size <= 1:
            return 1.0
        if size == 2:
            return 1.2
        if size == 3:
            return 1.3
        if size == 4:
            return 1.4
        return 1.5  # 5+ members

    def set_party_leader(self, is_leader: bool) -> None:
        with self._lock:
            self._party_leader = is_leader

    def is_party_leader(self) -> bool:
        with self._lock:
            return self._party_leader

    def set_in_party(self, in_party: bool, party_name: str = "") -> None:
        with self._lock:
            self._is_in_party = in_party
            self._party_name = party_name

    def is_in_party(self) -> bool:
        with self._lock:
            return self._is_in_party

    # ── Guild Management ──

    def set_guild(self, guild_name: str) -> None:
        with self._lock:
            self._guild_name = guild_name
            self._is_in_guild = bool(guild_name)

    def get_guild_name(self) -> str:
        with self._lock:
            return self._guild_name

    def is_in_guild(self) -> bool:
        with self._lock:
            return self._is_in_guild

    # ── Trade Management ──

    def add_trade_offer(self, offer: TradeOffer) -> None:
        with self._lock:
            self._trade_offers.append(offer)
            if len(self._trade_offers) > self._max_trade_offers:
                self._trade_offers.pop(0)

    def get_trade_offers(self, offer_type: str | None = None) -> list[TradeOffer]:
        with self._lock:
            if offer_type:
                return [o for o in self._trade_offers if o.offer_type == offer_type and o.is_active]
            return [o for o in self._trade_offers if o.is_active]

    def get_best_trade_opportunity(self, item_name: str, max_price: int = 0) -> TradeOffer | None:
        with self._lock:
            candidates = [o for o in self._trade_offers if o.is_active and o.item_name == item_name]
            if not candidates:
                return None
            if max_price > 0:
                candidates = [o for o in candidates if o.price_per_unit <= max_price]
            if not candidates:
                return None
            candidates.sort(key=lambda o: o.price_per_unit)
            return candidates[0]

    def expire_old_offers(self, max_age_s: float = 3600) -> None:
        with self._lock:
            now = time.time()
            for offer in self._trade_offers:
                if offer.is_active and now - offer.timestamp > max_age_s:
                    offer.is_active = False

    # ── Player Reputation ──

    def record_player_interaction(self, player_name: str, interaction_type: str, positive: bool = True) -> None:
        with self._lock:
            if player_name not in self._known_players:
                self._known_players[player_name] = {
                    "interactions": [],
                    "reputation": 0.0,
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                }
            p = self._known_players[player_name]
            p["interactions"].append({
                "type": interaction_type,
                "positive": positive,
                "timestamp": time.time(),
            })
            p["last_seen"] = time.time()
            # Update reputation: +1 for positive, -2 for negative
            p["reputation"] += 1.0 if positive else -2.0
            p["reputation"] = max(-10.0, min(10.0, p["reputation"]))

    def get_player_reputation(self, player_name: str) -> float:
        with self._lock:
            p = self._known_players.get(player_name)
            return p["reputation"] if p else 0.0

    def get_trusted_players(self, min_reputation: float = 3.0) -> list[str]:
        with self._lock:
            return [name for name, data in self._known_players.items() if data["reputation"] >= min_reputation]

    def get_avoid_players(self, max_reputation: float = -3.0) -> list[str]:
        with self._lock:
            return [name for name, data in self._known_players.items() if data["reputation"] <= max_reputation]

    # ── Social Opportunities ──

    def add_opportunity(self, opportunity: SocialOpportunity) -> None:
        with self._lock:
            self._social_opportunities.append(opportunity)
            if len(self._social_opportunities) > self._max_opportunities:
                self._social_opportunities.pop(0)

    def get_best_opportunity(self, opportunity_type: str | None = None) -> SocialOpportunity | None:
        with self._lock:
            candidates = self._social_opportunities
            if opportunity_type:
                candidates = [o for o in candidates if o.opportunity_type == opportunity_type]
            if not candidates:
                return None
            candidates.sort(key=lambda o: -o.priority)
            return candidates[0]

    def get_opportunities(self, opportunity_type: str | None = None) -> list[SocialOpportunity]:
        with self._lock:
            if opportunity_type:
                return [o for o in self._social_opportunities if o.opportunity_type == opportunity_type]
            return list(self._social_opportunities)

    def get_party_opportunities(self) -> list[SocialOpportunity]:
        return self.get_opportunities("party")

    def get_trade_opportunities(self) -> list[SocialOpportunity]:
        return self.get_opportunities("trade")

    # ── Chat / Communication ──

    def send_chat(self, message: str) -> None:
        """Send a chat message via the enqueue function."""
        with self._lock:
            if self._enqueue_fn and self._chat_enabled:
                self._enqueue_fn("self", f"c {message}")

    def send_party_chat(self, message: str) -> None:
        """Send a party chat message."""
        with self._lock:
            if self._enqueue_fn and self._chat_enabled:
                self._enqueue_fn("self", f"p {message}")

    def send_guild_chat(self, message: str) -> None:
        """Send a guild chat message."""
        with self._lock:
            if self._enqueue_fn and self._chat_enabled:
                self._enqueue_fn("self", f"g {message}")

    def request_party(self, player_name: str) -> None:
        """Request to party a player."""
        with self._lock:
            if self._enqueue_fn:
                self._enqueue_fn("self", f"p {player_name}")

    def accept_party_invite(self, player_name: str) -> None:
        """Accept a party invite."""
        with self._lock:
            if self._enqueue_fn:
                self._enqueue_fn("self", f"p {player_name}")

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def set_chat_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._chat_enabled = enabled

    def set_auto_invite(self, enabled: bool) -> None:
        with self._lock:
            self._auto_invite_enabled = enabled

    def set_auto_trade(self, enabled: bool) -> None:
        with self._lock:
            self._auto_trade_enabled = enabled

    # ── Summary ──

    def get_social_summary(self) -> str:
        with self._lock:
            lines = [f"── Social Summary ──"]
            lines.append(f"In party: {self._is_in_party} ({self._party_name})")
            lines.append(f"In guild: {self._is_in_guild} ({self._guild_name})")
            lines.append(f"Party size: {self.get_party_size()}")
            lines.append(f"Party XP bonus: +{int((self.get_party_bonus_xp() - 1) * 100)}%")
            lines.append(f"Known players: {len(self._known_players)}")
            lines.append(f"Active trade offers: {len(self.get_trade_offers())}")
            lines.append(f"Social opportunities: {len(self._social_opportunities)}")
            trusted = self.get_trusted_players()
            if trusted:
                lines.append(f"Trusted players: {', '.join(trusted[:5])}")
            avoid = self.get_avoid_players()
            if avoid:
                lines.append(f"Avoid players: {', '.join(avoid[:5])}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._party_members.clear()
            self._trade_offers.clear()
            self._social_opportunities.clear()
            self._known_players.clear()
            self._party_invites_pending.clear()
            self._is_in_party = False
            self._is_in_guild = False
            self._party_name = ""
            self._guild_name = ""


# ── Global Singleton ──

_social_intel: SocialIntelligence | None = None
_social_intel_lock = RLock()


def get_social_intelligence() -> SocialIntelligence:
    global _social_intel
    with _social_intel_lock:
        if _social_intel is None:
            _social_intel = SocialIntelligence()
        return _social_intel
