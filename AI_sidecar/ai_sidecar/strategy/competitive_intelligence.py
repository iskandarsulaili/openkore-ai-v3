"""
Competitive Intelligence — tracks every player's activities, builds, trades, and threats.

A top player doesn't just farm. They know:
- Who is farming what, where, when (competition analysis)
- Who is selling what, at what price, volume (market intelligence)
- Who is leveling what class, what build (meta tracking)
- Guild memberships, alliances, rivalries (social intelligence)
- Who is a threat, who is an ally, who is irrelevant (threat assessment)

This module wires into player_profiler.py for player tracking and
p2p_knowledge.py for sharing intelligence across the fleet.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FarmingActivity:
    """What a player is farming, where, and when."""
    player_name: str
    map_name: str
    monster_name: str = ""
    estimated_zeny_per_hour: int = 0
    estimated_xp_per_hour: int = 0
    first_observed: float = 0.0
    last_observed: float = 0.0
    observation_count: int = 0
    competition_level: int = 0  # 0-10, how much they compete with us


@dataclass
class MarketActivity:
    """What a player is selling, at what price, and volume."""
    player_name: str
    item_name: str
    price: int = 0
    quantity: int = 0
    shop_location: str = ""
    first_observed: float = 0.0
    last_observed: float = 0.0
    total_volume: int = 0
    price_trend: str = "stable"  # rising, falling, stable


@dataclass
class PlayerBuild:
    """A player's class, build, and level progression."""
    player_name: str
    job_class: str = ""
    base_level: int = 0
    job_level: int = 0
    build_type: str = ""  # agi, int, str, vit, hybrid
    guild_name: str = ""
    alliance_tag: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    level_up_rate: float = 0.0  # levels per day observed
    is_competitor: bool = False
    is_threat: bool = False
    is_ally: bool = False
    threat_score: int = 0  # 0-100


@dataclass
class GuildIntel:
    """Intelligence about a guild."""
    guild_name: str
    member_count: int = 0
    members: list[str] = field(default_factory=list)
    alliance_with: list[str] = field(default_factory=list)
    enemy_of: list[str] = field(default_factory=list)
    territory: list[str] = field(default_factory=list)
    strength_rating: int = 0  # 0-100
    first_seen: float = 0.0
    last_seen: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# CompetitiveIntelligence
# ---------------------------------------------------------------------------


class CompetitiveIntelligence:
    """Tracks competitive intelligence about players, markets, and guilds.

    Wires into:
      - player_profiler.py: enriches player profiles with CI data
      - p2p_knowledge.py: shares intelligence across the fleet
      - fleet_coordinator.py: informs fleet decisions
    """

    def __init__(
        self,
        player_profiler: Any = None,
        p2p_node: Any = None,
        fleet_coordinator: Any = None,
    ) -> None:
        self._lock = RLock()
        self._player_profiler = player_profiler
        self._p2p_node = p2p_node
        self._fleet_coordinator = fleet_coordinator

        # Core intelligence stores
        self._farming_activities: dict[str, list[FarmingActivity]] = defaultdict(list)
        self._market_activities: dict[str, list[MarketActivity]] = defaultdict(list)
        self._player_builds: dict[str, PlayerBuild] = {}
        self._guild_intel: dict[str, GuildIntel] = {}

        # Derived assessments
        self._threat_list: list[str] = []
        self._ally_list: list[str] = []
        self._competitor_list: list[str] = []

        # Stats
        self._stats: dict[str, int] = {
            "farming_observations": 0,
            "market_observations": 0,
            "build_tracked": 0,
            "guilds_tracked": 0,
            "threats_identified": 0,
            "allies_identified": 0,
            "intel_shares": 0,
        }

    # ── Farming Intelligence ──────────────────────────────────────

    def observe_farming(
        self,
        player_name: str,
        map_name: str,
        monster_name: str = "",
        estimated_zeny_per_hour: int = 0,
        estimated_xp_per_hour: int = 0,
    ) -> None:
        """Record a farming observation for a player."""
        with self._lock:
            now = time.time()
            activities = self._farming_activities[player_name]

            # Find existing activity on this map
            found = False
            for act in activities:
                if act.map_name == map_name:
                    act.last_observed = now
                    act.observation_count += 1
                    if monster_name:
                        act.monster_name = monster_name
                    if estimated_zeny_per_hour:
                        act.estimated_zeny_per_hour = int(
                            act.estimated_zeny_per_hour * 0.7
                            + estimated_zeny_per_hour * 0.3
                        )
                    if estimated_xp_per_hour:
                        act.estimated_xp_per_hour = int(
                            act.estimated_xp_per_hour * 0.7
                            + estimated_xp_per_hour * 0.3
                        )
                    found = True
                    break

            if not found:
                activities.append(FarmingActivity(
                    player_name=player_name,
                    map_name=map_name,
                    monster_name=monster_name,
                    estimated_zeny_per_hour=estimated_zeny_per_hour,
                    estimated_xp_per_hour=estimated_xp_per_hour,
                    first_observed=now,
                    last_observed=now,
                    observation_count=1,
                ))

            self._stats["farming_observations"] += 1

            # Update player profiler
            self._update_profiler_from_farming(player_name, map_name)

    def _update_profiler_from_farming(self, player_name: str, map_name: str) -> None:
        """Update the player profiler with farming intelligence."""
        if self._player_profiler is None:
            try:
                from ai_sidecar.player_profiler import get_player_profiler
                self._player_profiler = get_player_profiler()
            except Exception:
                return
        try:
            self._player_profiler.observe_player(
                name=player_name,
                map_name=map_name,
            )
            # Mark as competitor if they're farming the same maps
            profile = self._player_profiler.get_player(player_name)
            if profile and profile.category == "unknown":
                profile.category = "farmer"
        except Exception:
            pass

    def get_farming_competition(self, map_name: str) -> list[FarmingActivity]:
        """Get all players farming on a specific map."""
        with self._lock:
            result = []
            for activities in self._farming_activities.values():
                for act in activities:
                    if act.map_name == map_name:
                        result.append(act)
            return sorted(result, key=lambda a: a.observation_count, reverse=True)

    def get_most_competitive_maps(self, top_n: int = 5) -> list[tuple[str, int]]:
        """Get the maps with the most competition."""
        with self._lock:
            map_counts: dict[str, int] = defaultdict(int)
            for activities in self._farming_activities.values():
                for act in activities:
                    map_counts[act.map_name] += 1
            return sorted(map_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ── Market Intelligence ───────────────────────────────────────

    def observe_market(
        self,
        player_name: str,
        item_name: str,
        price: int,
        quantity: int = 1,
        shop_location: str = "",
    ) -> None:
        """Record a market observation (what someone is selling)."""
        with self._lock:
            now = time.time()
            activities = self._market_activities[player_name]

            found = False
            for act in activities:
                if act.item_name == item_name:
                    act.last_observed = now
                    act.total_volume += quantity
                    # Track price trend
                    if price > act.price:
                        act.price_trend = "rising"
                    elif price < act.price:
                        act.price_trend = "falling"
                    else:
                        act.price_trend = "stable"
                    # Weighted average price
                    act.price = int(act.price * 0.7 + price * 0.3)
                    act.quantity = quantity
                    found = True
                    break

            if not found:
                activities.append(MarketActivity(
                    player_name=player_name,
                    item_name=item_name,
                    price=price,
                    quantity=quantity,
                    shop_location=shop_location,
                    first_observed=now,
                    last_observed=now,
                    total_volume=quantity,
                ))

            self._stats["market_observations"] += 1

    def get_market_prices(self, item_name: str) -> list[MarketActivity]:
        """Get all observed prices for an item."""
        with self._lock:
            result = []
            for activities in self._market_activities.values():
                for act in activities:
                    if act.item_name == item_name:
                        result.append(act)
            return sorted(result, key=lambda a: a.price)

    def get_best_price(self, item_name: str) -> int | None:
        """Get the best (lowest) observed price for an item."""
        prices = self.get_market_prices(item_name)
        if prices:
            return min(a.price for a in prices)
        return None

    def get_top_sellers(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get the top sellers by total volume."""
        with self._lock:
            seller_volume: dict[str, int] = defaultdict(int)
            for player, activities in self._market_activities.items():
                for act in activities:
                    seller_volume[player] += act.total_volume
            return sorted(seller_volume.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ── Build / Meta Tracking ─────────────────────────────────────

    def track_player_build(
        self,
        player_name: str,
        job_class: str = "",
        base_level: int = 0,
        job_level: int = 0,
        build_type: str = "",
        guild_name: str = "",
    ) -> None:
        """Track a player's class, build, and level progression."""
        with self._lock:
            now = time.time()
            existing = self._player_builds.get(player_name)

            if existing is None:
                self._player_builds[player_name] = PlayerBuild(
                    player_name=player_name,
                    job_class=job_class,
                    base_level=base_level,
                    job_level=job_level,
                    build_type=build_type,
                    guild_name=guild_name,
                    first_seen=now,
                    last_seen=now,
                )
                self._stats["build_tracked"] += 1
            else:
                # Calculate level-up rate
                if base_level > existing.base_level and existing.last_seen > existing.first_seen:
                    days_elapsed = (now - existing.last_seen) / 86400.0
                    if days_elapsed > 0:
                        levels_gained = base_level - existing.base_level
                        existing.level_up_rate = (
                            existing.level_up_rate * 0.7
                            + (levels_gained / days_elapsed) * 0.3
                        )

                existing.job_class = job_class or existing.job_class
                existing.base_level = base_level or existing.base_level
                existing.job_level = job_level or existing.job_level
                existing.build_type = build_type or existing.build_type
                existing.guild_name = guild_name or existing.guild_name
                existing.last_seen = now

            # Track guild
            if guild_name:
                self._track_guild(guild_name, player_name)

    def _track_guild(self, guild_name: str, member_name: str) -> None:
        """Track a guild and its members."""
        with self._lock:
            guild = self._guild_intel.get(guild_name)
            if guild is None:
                guild = GuildIntel(
                    guild_name=guild_name,
                    first_seen=time.time(),
                    last_seen=time.time(),
                )
                self._guild_intel[guild_name] = guild
                self._stats["guilds_tracked"] += 1
            else:
                guild.last_seen = time.time()

            if member_name not in guild.members:
                guild.members.append(member_name)
                guild.member_count = len(guild.members)

    def get_player_build(self, player_name: str) -> PlayerBuild | None:
        """Get the tracked build for a player."""
        with self._lock:
            return self._player_builds.get(player_name)

    def get_players_by_class(self, job_class: str) -> list[PlayerBuild]:
        """Get all players with a specific class."""
        with self._lock:
            return [p for p in self._player_builds.values() if p.job_class == job_class]

    def get_players_by_guild(self, guild_name: str) -> list[PlayerBuild]:
        """Get all players in a specific guild."""
        with self._lock:
            return [p for p in self._player_builds.values() if p.guild_name == guild_name]

    # ── Threat Assessment ─────────────────────────────────────────

    def assess_threat(
        self,
        player_name: str,
        is_attacking: bool = False,
        is_following: bool = False,
        is_ks: bool = False,
        is_gm: bool = False,
    ) -> int:
        """Assess and update a player's threat score. Returns the new score."""
        with self._lock:
            build = self._player_builds.get(player_name)
            if build is None:
                build = PlayerBuild(player_name=player_name)
                self._player_builds[player_name] = build

            score = build.threat_score

            if is_attacking:
                score = min(100, score + 30)
                build.is_threat = True
            if is_following:
                score = min(100, score + 15)
            if is_ks:
                score = min(100, score + 20)
                build.is_competitor = True
            if is_gm:
                score = min(100, score + 50)
                build.is_threat = True

            # Decay over time
            if not is_attacking and not is_following and not is_ks:
                score = max(0, score - 1)

            build.threat_score = score

            # Update lists
            if score >= 50 and player_name not in self._threat_list:
                self._threat_list.append(player_name)
                self._stats["threats_identified"] += 1
            elif score < 30 and player_name in self._threat_list:
                self._threat_list.remove(player_name)

            return score

    def mark_ally(self, player_name: str) -> None:
        """Mark a player as an ally."""
        with self._lock:
            build = self._player_builds.get(player_name)
            if build is None:
                build = PlayerBuild(player_name=player_name)
                self._player_builds[player_name] = build
            build.is_ally = True
            build.is_threat = False
            build.threat_score = 0

            if player_name not in self._ally_list:
                self._ally_list.append(player_name)
                self._stats["allies_identified"] += 1
            if player_name in self._threat_list:
                self._threat_list.remove(player_name)

    def get_threats(self, min_score: int = 30) -> list[PlayerBuild]:
        """Get all players above a threat threshold."""
        with self._lock:
            return [p for p in self._player_builds.values() if p.threat_score >= min_score]

    def get_allies(self) -> list[PlayerBuild]:
        """Get all marked allies."""
        with self._lock:
            return [p for p in self._player_builds.values() if p.is_ally]

    def get_competitors(self) -> list[PlayerBuild]:
        """Get all marked competitors."""
        with self._lock:
            return [p for p in self._player_builds.values() if p.is_competitor]

    # ── Guild Intelligence ────────────────────────────────────────

    def get_guild_intel(self, guild_name: str) -> GuildIntel | None:
        """Get intelligence about a guild."""
        with self._lock:
            return self._guild_intel.get(guild_name)

    def get_all_guilds(self) -> list[GuildIntel]:
        """Get all tracked guilds."""
        with self._lock:
            return list(self._guild_intel.values())

    def set_guild_alliance(self, guild_a: str, guild_b: str) -> None:
        """Record an alliance between two guilds."""
        with self._lock:
            for g in (guild_a, guild_b):
                guild = self._guild_intel.get(g)
                if guild is None:
                    guild = GuildIntel(guild_name=g)
                    self._guild_intel[g] = guild
                other = guild_b if g == guild_a else guild_a
                if other not in guild.alliance_with:
                    guild.alliance_with.append(other)

    def set_guild_enmity(self, guild_a: str, guild_b: str) -> None:
        """Record enmity between two guilds."""
        with self._lock:
            for g in (guild_a, guild_b):
                guild = self._guild_intel.get(g)
                if guild is None:
                    guild = GuildIntel(guild_name=g)
                    self._guild_intel[g] = guild
                other = guild_b if g == guild_a else guild_a
                if other not in guild.enemy_of:
                    guild.enemy_of.append(other)

    # ── P2P Sharing ────────────────────────────────────────────────

    def share_intel(self) -> None:
        """Share competitive intelligence with the P2P network."""
        if self._p2p_node is None:
            try:
                from ai_sidecar.p2p_knowledge import P2PKnowledgeNode
                # Can't get singleton easily, skip if not wired
                return
            except Exception:
                return

        try:
            # Share top threats
            threats = self.get_threats(min_score=50)
            for threat in threats[:5]:
                self._p2p_node.broadcast_message(
                    msg_type="competitive_intel",
                    payload={
                        "player": threat.player_name,
                        "class": threat.job_class,
                        "level": threat.base_level,
                        "guild": threat.guild_name,
                        "threat_score": threat.threat_score,
                        "timestamp": time.time(),
                    },
                )
                self._stats["intel_shares"] += 1

            # Share market prices for high-value items
            for player, activities in list(self._market_activities.items())[:3]:
                for act in activities[:3]:
                    if act.price > 10000:  # Only share high-value intel
                        self._p2p_node.broadcast_message(
                            msg_type="market_intel",
                            payload={
                                "seller": player,
                                "item": act.item_name,
                                "price": act.price,
                                "location": act.shop_location,
                                "timestamp": time.time(),
                            },
                        )
                        self._stats["intel_shares"] += 1
        except Exception:
            pass

    # ── Summary / Context ────────────────────────────────────────

    def get_intelligence_summary(self) -> str:
        """Get a formatted summary of competitive intelligence."""
        with self._lock:
            lines = ["── Competitive Intelligence ──"]

            # Threats
            threats = self.get_threats(min_score=30)
            if threats:
                lines.append(f"  Threats ({len(threats)}):")
                for t in sorted(threats, key=lambda x: x.threat_score, reverse=True)[:5]:
                    lines.append(
                        f"    {t.player_name} ({t.job_class} Lv.{t.base_level}) "
                        f"threat={t.threat_score} guild={t.guild_name}"
                    )
            else:
                lines.append("  No threats detected.")

            # Allies
            allies = self.get_allies()
            if allies:
                lines.append(f"  Allies ({len(allies)}):")
                for a in allies[:5]:
                    lines.append(f"    {a.player_name} ({a.job_class})")

            # Competitors
            competitors = self.get_competitors()
            if competitors:
                lines.append(f"  Competitors ({len(competitors)}):")
                for c in competitors[:5]:
                    lines.append(f"    {c.player_name} ({c.job_class} Lv.{c.base_level})")

            # Hot maps
            hot_maps = self.get_most_competitive_maps(3)
            if hot_maps:
                lines.append("  Hot maps:")
                for map_name, count in hot_maps:
                    lines.append(f"    {map_name}: {count} farmers")

            # Guilds
            guilds = self.get_all_guilds()
            if guilds:
                lines.append(f"  Guilds tracked: {len(guilds)}")
                for g in guilds[:3]:
                    lines.append(
                        f"    {g.guild_name}: {g.member_count} members, "
                        f"strength={g.strength_rating}"
                    )

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ──

_ci: CompetitiveIntelligence | None = None
_ci_lock = RLock()


def get_competitive_intelligence() -> CompetitiveIntelligence:
    global _ci
    with _ci_lock:
        if _ci is None:
            _ci = CompetitiveIntelligence()
        return _ci
