"""War of Emporium (WoE) tactics — castle detection, emperium break, guild warfare.

Provides:
  - CastleState: tracked state for one castle
  - WoERole: roles a bot can take during WoE
  - WoETactics: heuristic assessment for WoE castle maps
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = __import__("logging").getLogger(__name__)


# ── WoE constants ────────────────────────────────────────────────────────

# WoE schedule — hours when war is active (server time, 24h range)
WOE_WAR_HOURS: tuple[int, ...] = (
    20, 21, 22,  # Night WoE (20:00–22:59)
)

# WoE days (0=Mon … 6=Sun)
WOE_WAR_DAYS: tuple[int, ...] = (
    3, 5, 6,    # Wed, Fri, Sat
)

# Notable WoE map fragments
CASTLE_PREFIXES: tuple[str, ...] = (
    "gld_dun", "gld_castle", "gld_dun01", "gld_dun02",
    "gld_dun03", "gld_dun04",
)

EMPERIUM_MAP_FRAGMENTS: tuple[str, ...] = (
    "gld_dun04", "gld_dun03_",
    "aldeba_dun04", "ayotha_dun04",
    "gefg_dun04", "payg_dun04",
    "prtg_dun04",
)


# ── Data models ──────────────────────────────────────────────────────────

class WoERole(str, Enum):
    """Role a bot plays during WoE."""
    ATTACKER = "attacker"        # Push into castle, kill emperium
    DEFENDER = "defender"        # Hold choke points, protect emperium
    SUPPORT = "support"          # Heal / buff allies
    SCOUT = "scout"              # Recon enemy movements
    BREAKER = "breaker"          # Focus entirely on emperium


@dataclass
class CastleState:
    """Tracked state for a single WoE castle."""
    name: str
    map_name: str
    guild_owner: str = ""
    emperium_alive: bool = True
    allies_nearby: int = 0
    enemies_nearby: int = 0
    last_seen: float = 0.0
    in_emperium_room: bool = False

    @property
    def age(self) -> float:
        return time.time() - self.last_seen

    @property
    def is_contested(self) -> bool:
        return self.enemies_nearby > 2 and self.allies_nearby > 0


# ── WoE tactics engine ───────────────────────────────────────────────────

class WoETactics:
    """WoE decision-making: castle map detection, emperium break, and
    role-appropriate behaviour (defence / offence).

    Connects to the PvPDomain.assess() loop when on gld_* maps.
    """

    def __init__(self) -> None:
        self._castles: dict[str, CastleState] = {}
        self._role: WoERole = WoERole.ATTACKER
        self._last_check: float = 0.0
        self._breach_attempted: bool = False

    # ── Public API ────────────────────────────────────────────────────

    @staticmethod
    def is_war_time() -> bool:
        """Return True if WoE is currently active (server-local heuristic)."""
        now = time.localtime()
        if now.tm_wday not in WOE_WAR_DAYS:
            return False
        return now.tm_hour in WOE_WAR_HOURS

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Evaluate WoE state and emit role-appropriate actions."""
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        players = signals.get("players", []) or []
        guild_name = str(signals.get("guild_name", "") or "")

        self._update_castle_state(map_name, players, guild_name)

        # Detect emperium room
        in_emp_room = self._is_emperium_room(map_name)
        current = self._castles.get(map_name)

        if current is not None:
            current.in_emperium_room = in_emp_room

        # ── Role-based behaviour ──
        role = self._resolve_role(signals)

        if role == WoERole.BREAKER and in_emp_room:
            self._emit_emperium_attack(actions, bot_id, map_name)
        elif role in (WoERole.ATTACKER, WoERole.BREAKER):
            self._emit_push_actions(actions, bot_id, map_name, signals, current)
        elif role == WoERole.DEFENDER:
            self._emit_defend_actions(actions, bot_id, map_name, signals, current)
        elif role == WoERole.SUPPORT:
            self._emit_support_actions(actions, bot_id, map_name, signals)
        elif role == WoERole.SCOUT:
            self._emit_scout_actions(actions, bot_id, map_name, current)

    # ── War-time helpers ──────────────────────────────────────────────

    def set_role(self, role: WoERole) -> None:
        """Override the WoE role for this bot."""
        self._role = role
        logger.info("[WoE] Role set to %s", role.value)

    def get_role(self) -> WoERole:
        return self._role

    def get_castle(self, map_name: str) -> CastleState | None:
        return self._castles.get(map_name)

    # ── Internal ──────────────────────────────────────────────────────

    def _resolve_role(self, signals: dict[str, Any]) -> WoERole:
        """Determine the bot's WoE role based on job/signals.

        Default: attacker.  Can be overridden by calling ``set_role()``.
        """
        if self._role != WoERole.ATTACKER:
            return self._role

        job_name = str(signals.get("job_name", "novice") or "novice").lower()

        # Support roles
        if any(s in job_name for s in ["priest", "arch bishop", "acolyte"]):
            return WoERole.SUPPORT

        # Defender roles
        if any(s in job_name for s in ["knight", "lord knight", "rune knight",
                                        "crusader", "paladin", "royal guard",
                                        "swordman"]):
            return WoERole.DEFENDER

        # Breaker (high DPS melee)
        if any(s in job_name for s in ["assassin", "guillotine cross",
                                        "rogue", "stalker", "shadow chaser",
                                        "monk", "champion", "sura"]):
            return WoERole.BREAKER

        # Attacker (ranged/magic DPS)
        return WoERole.ATTACKER

    def _update_castle_state(
        self,
        map_name: str,
        players: list[Any],
        guild_name: str,
    ) -> None:
        """Refresh castle state from the player list."""
        if map_name not in self._castles:
            self._castles[map_name] = CastleState(
                name=map_name,
                map_name=map_name,
            )

        castle = self._castles[map_name]
        castle.last_seen = time.time()
        castle.allies_nearby = 0
        castle.enemies_nearby = 0

        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
            else:
                pg = ""
            if pg == guild_name:
                castle.allies_nearby += 1
            elif pg:
                castle.enemies_nearby += 1

        if guild_name:
            castle.guild_owner = guild_name

    def _is_emperium_room(self, map_name: str) -> bool:
        """Heuristic check: are we in the emperium (innermost) room?"""
        return any(frag in map_name for frag in EMPERIUM_MAP_FRAGMENTS)

    # ── Action emitters ───────────────────────────────────────────────

    def _emit_emperium_attack(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
    ) -> None:
        """Focus all DPS on the emperium."""
        logger.info("[WoE] %s: BREAKING emperium on %s!", bot_id, map_name)
        actions.append(HeuristicAction(
            kind="command",
            command="attack Emperium",
            confidence=0.99,
            domain="pvp",
            reason=f"WoE breaker: killing emperium on {map_name}",
            metadata={"map": map_name, "target": "emperium", "woe_role": "breaker"},
        ))
        actions.append(HeuristicAction(
            kind="command",
            command="set attackAuto 3",
            confidence=0.95,
            domain="pvp",
            reason="Max attack for emperium DPS",
        ))

    def _emit_push_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        castle: CastleState | None,
    ) -> None:
        """Push into the castle and engage enemies."""
        my_hp = int(signals.get("hp", 1) or 1)
        my_max_hp = int(signals.get("max_hp", 100) or 100)
        hp_ratio = my_hp / max(my_max_hp, 1)

        if hp_ratio < 0.30:
            # Retreat for healing
            actions.append(HeuristicAction(
                kind="command",
                command="use 501",  # Red Potion
                confidence=0.95,
                domain="pvp",
                reason=f"WoE HP low ({hp_ratio:.0%}) — healing",
            ))
            return

        # Find nearest enemy
        enemies = self._find_enemies(signals)
        if enemies:
            target = enemies[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target}",
                confidence=0.90,
                domain="pvp",
                reason=f"WoE attack: pushing on {map_name} — engaging {target}",
                metadata={"map": map_name, "woe_role": "attacker"},
            ))

        if castle and castle.in_emperium_room:
            self._emit_emperium_attack(actions, bot_id, map_name)

    def _emit_defend_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        castle: CastleState | None,
    ) -> None:
        """Hold choke points — intercept enemies before they reach the emperium."""
        enemies = self._find_enemies(signals)

        if not enemies:
            # Stand guard near emperium entrance
            actions.append(HeuristicAction(
                kind="command",
                command="sit",
                confidence=0.70,
                domain="pvp",
                reason=f"WoE defend: guarding {map_name} — no enemies in sight",
            ))
            return

        # Engage the first enemy
        target = enemies[0]
        logger.info("[WoE] %s: DEFENDING {map_name} vs {target}", bot_id, target)
        actions.append(HeuristicAction(
            kind="command",
            command=f"attack {target}",
            confidence=0.95,
            domain="pvp",
            reason=f"WoE defend: intercepting {target} on {map_name}",
            metadata={"map": map_name, "target": target, "woe_role": "defender"},
        ))

    def _emit_support_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
    ) -> None:
        """Heal and buff nearby allies."""
        players = signals.get("players", []) or []
        guild = str(signals.get("guild_name", "") or "")
        my_name = str(signals.get("name", "") or "")

        low_hp_allies: list[str] = []
        for p in players:
            if not isinstance(p, dict):
                continue
            p_name = str(p.get("name", "") or "")
            pg = str(p.get("guild_name", "") or "")
            p_hp = float(p.get("hp_ratio", 1.0) or 1.0)
            if p_name and p_name != my_name and pg == guild and p_hp < 0.50:
                low_hp_allies.append(p_name)

        if low_hp_allies:
            target = low_hp_allies[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"use_skill Heal {target}",
                confidence=0.95,
                domain="pvp",
                reason=f"WoE support: healing {target} on {map_name}",
                metadata={"map": map_name, "target": target, "woe_role": "support"},
            ))
        else:
            # Idle buff / standby
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill Blessing",
                confidence=0.70,
                domain="pvp",
                reason=f"WoE support: maintaining buffs on {map_name}",
            ))

    def _emit_scout_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        castle: CastleState | None,
    ) -> None:
        """Scout the castle perimeter for enemy movement."""
        logger.info("[WoE] %s: scouting %s", bot_id, map_name)
        actions.append(HeuristicAction(
            kind="log",
            command="",
            confidence=0.90,
            domain="pvp",
            reason=f"[WoE scout] {map_name}: guild={castle.guild_owner if castle else '?'} "
                   f"enemies={castle.enemies_nearby if castle else 0} "
                   f"allies={castle.allies_nearby if castle else 0}",
            metadata={"map": map_name, "woe_role": "scout"},
        ))

    # ── Utility ───────────────────────────────────────────────────────

    @staticmethod
    def _find_enemies(signals: dict[str, Any]) -> list[str]:
        """Return names of enemy players (non-guild) in the current map."""
        players = signals.get("players", []) or []
        guild = str(signals.get("guild_name", "") or "")
        enemies: list[str] = []
        for p in players:
            if isinstance(p, dict):
                p_name = str(p.get("name", "") or "")
                pg = str(p.get("guild_name", "") or "")
            else:
                p_name = str(p)
                pg = ""
            if p_name and pg != guild:
                enemies.append(p_name)
        return enemies
