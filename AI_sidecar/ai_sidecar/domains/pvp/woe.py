"""War of Emporium (WoE) tactics — castle detection, emperium break, guild warfare.

Provides:
  - CastleState: tracked state for one castle
  - WoERole: roles a bot can take during WoE
  - WoETactics: full tactical assessment for WoE castle maps
  - Class-specific role behaviors
  - EMP damage race tracking
  - Guild chemistry (alliance/neutral/hostile)
  - Gate timing coordination
  - Weather tracking (guild skills)
  - Battlefield threat assessment
  - Escape scroll economy
  - Barricade management
  - Class-vs-class counter strategies
  - WoE consumable preparation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.woe.emperium_mechanics import (
    EmperiumMechanics,
    get_emperium_mechanics,
    EMPERIUM_BREAKER_CLASSES,
    EMPERIUM_NON_BREAKER_CLASSES,
    CLASS_EMP_DPS,
    CLASS_EMP_BEST_SKILL,
    CLASS_EMP_BURST_SKILLS,
)
from ai_sidecar.woe.battlefield_awareness import (
    BattlefieldAwareness,
    get_battlefield_awareness,
    ThreatPriority,
    CLASS_THREAT_PRIORITY,
    RUN_THRESHOLD,
    FLY_WING_ID,
    BUTTERFLY_WING_ID,
)
from ai_sidecar.woe.castle_intel import (
    CastleIntelligence,
    get_castle_intelligence,
    WOE_CASTLES,
    WOE_CONSUMABLES,
    WOE_EQUIPMENT,
    BARRICADE_POSITIONS,
    BARRICADE_BYPASS,
)

logger = __import__("logging").getLogger(__name__)


# ── WoE constants ────────────────────────────────────────────────────────

WOE_WAR_HOURS: tuple[int, ...] = (20, 21, 22)
WOE_WAR_DAYS: tuple[int, ...] = (3, 5, 6)

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

# WoE map chokepoints and defensive positions (map_name -> list of (x, y) positions)
WOE_CHOKEPOINTS: dict[str, list[tuple[int, int]]] = {
    "gld_dun01": [(50, 50), (100, 100)],
    "gld_dun02": [(30, 30), (120, 80)],
    "gld_dun03": [(60, 60), (90, 90)],
    "gld_dun04": [(40, 40), (80, 80), (110, 110)],
}

WOE_ENTRANCES: dict[str, list[tuple[int, int]]] = {
    "gld_dun01": [(10, 10), (150, 10)],
    "gld_dun02": [(5, 5), (140, 5)],
    "gld_dun03": [(15, 15), (130, 15)],
    "gld_dun04": [(8, 8), (145, 8)],
}

WOE_DEFENSIVE_POSITIONS: dict[str, list[tuple[int, int]]] = {
    "gld_dun01": [(55, 55), (95, 95)],
    "gld_dun02": [(35, 35), (115, 85)],
    "gld_dun03": [(65, 65), (85, 85)],
    "gld_dun04": [(45, 45), (75, 75), (105, 105)],
}

# EMP HP thresholds for damage focus calls
EMP_HP_THRESHOLDS: list[tuple[float, str]] = [
    (0.50, "EMP at 50% — focus damage!"),
    (0.25, "EMP at 25% — burn it down!"),
    (0.10, "EMP at 10% — finish it!"),
]

# High-value caster priority list
HIGH_VALUE_CASTERS: set[str] = {
    "sura", "champion", "wizard", "high_wizard", "creator", "genetic",
    "soul_linker", "warlock", "sorcerer",
}


# ── Data models ──────────────────────────────────────────────────────────

class WoERole(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"
    SUPPORT = "support"
    SCOUT = "scout"
    BREAKER = "breaker"


class GuildRelation(str, Enum):
    ALLIANCE = "alliance"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"


@dataclass
class CastleState:
    """Tracked state for a single WoE castle."""
    name: str
    map_name: str
    guild_owner: str = ""
    emperium_alive: bool = True
    emperium_hp_pct: float = 1.0
    allies_nearby: int = 0
    enemies_nearby: int = 0
    last_seen: float = 0.0
    in_emperium_room: bool = False
    gate_open: bool = False
    weather_effect: str = ""
    last_weather_change: float = 0.0
    mass_dispel_detected: bool = False
    last_mass_dispel_time: float = 0.0

    @property
    def age(self) -> float:
        return time.time() - self.last_seen

    @property
    def is_contested(self) -> bool:
        return self.enemies_nearby > 2 and self.allies_nearby > 0


@dataclass
class GuildChemistry:
    """Tracked guild relationships."""
    guild_name: str
    relation: GuildRelation = GuildRelation.NEUTRAL
    last_encounter: float = 0.0
    aggression_score: float = 0.5  # 0.0 (passive) to 1.0 (aggressive)


@dataclass
class EmpDamageRace:
    """Track EMP damage race state."""
    emp_hp_pct: float = 1.0
    last_threshold_called: float = 1.0  # Highest threshold already called
    damage_dealt: int = 0
    damage_taken: int = 0
    last_update: float = 0.0

    def check_threshold(self) -> str | None:
        """Check if a new threshold has been crossed. Returns call message or None."""
        for threshold, message in sorted(EMP_HP_THRESHOLDS, reverse=True):
            if self.emp_hp_pct <= threshold and self.last_threshold_called > threshold:
                self.last_threshold_called = threshold
                return message
        return None


# ── WoE tactics engine ───────────────────────────────────────────────────

class WoETactics:
    """WoE decision-making: castle map detection, emperium break, and
    role-appropriate behaviour with class-specific tactics.

    Features:
      - Base defense/offense positioning with chokepoint awareness
      - EMP targeting with damage race tracking
      - Class-specific roles (Wizard, Priest, Assassin, Paladin, Champion)
      - Mass dispel detection and repositioning
      - Enemy caster priority interruption
      - EMP damage race with threshold calls
      - Guild chemistry tracking
      - Gate timing coordination
      - Weather tracking (guild skills)
    """

    def __init__(self) -> None:
        self._castles: dict[str, CastleState] = {}
        self._role: WoERole = WoERole.ATTACKER
        self._last_check: float = 0.0
        self._breach_attempted: bool = False
        self._guild_chemistry: dict[str, GuildChemistry] = {}
        self._emp_damage_race: EmpDamageRace = EmpDamageRace()
        self._last_gate_check: float = 0.0
        self._gate_cooldown: float = 0.0

        # Sub-engines
        self._emperium: EmperiumMechanics = get_emperium_mechanics()
        self._battlefield: BattlefieldAwareness = get_battlefield_awareness()
        self._castle_intel: CastleIntelligence = get_castle_intelligence()

    # ── Public API ────────────────────────────────────────────────────

    @staticmethod
    def is_war_time() -> bool:
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
        job_name = str(signals.get("job_name", "novice") or "novice").lower()

        self._update_castle_state(map_name, players, guild_name)
        self._update_guild_chemistry(players, guild_name)
        self._update_emp_state(signals)
        self._update_weather(signals, map_name)
        self._detect_mass_dispel(signals, map_name)

        in_emp_room = self._is_emperium_room(map_name)
        current = self._castles.get(map_name)
        if current is not None:
            current.in_emperium_room = in_emp_room

        role = self._resolve_role(signals)

        # ── Role-based behaviour ──
        if role == WoERole.BREAKER and in_emp_room:
            self._emit_emperium_attack(actions, bot_id, map_name, signals)
        elif role in (WoERole.ATTACKER, WoERole.BREAKER):
            self._emit_push_actions(actions, bot_id, map_name, signals, current)
        elif role == WoERole.DEFENDER:
            self._emit_defend_actions(actions, bot_id, map_name, signals, current)
        elif role == WoERole.SUPPORT:
            self._emit_support_actions(actions, bot_id, map_name, signals, job_name)
        elif role == WoERole.SCOUT:
            self._emit_scout_actions(actions, bot_id, map_name, current)

        # ── Cross-role actions ──
        self._emit_class_specific_actions(actions, bot_id, map_name, signals, job_name, role)
        self._emit_emp_damage_call(actions, bot_id, map_name)
        self._emit_caster_interrupt(actions, signals, job_name)
        self._emit_gate_timing(actions, bot_id, map_name, signals)
        self._emit_weather_reaction(actions, bot_id, map_name, current)

    # ── War-time helpers ──────────────────────────────────────────────

    def set_role(self, role: WoERole) -> None:
        self._role = role
        logger.info("[WoE] Role set to %s", role.value)

    def get_role(self) -> WoERole:
        return self._role

    def get_castle(self, map_name: str) -> CastleState | None:
        return self._castles.get(map_name)

    def set_guild_relation(self, guild_name: str, relation: GuildRelation) -> None:
        """Manually set a guild's relation."""
        if guild_name not in self._guild_chemistry:
            self._guild_chemistry[guild_name] = GuildChemistry(guild_name=guild_name)
        self._guild_chemistry[guild_name].relation = relation
        logger.info("[WoE] Guild %s set to %s", guild_name, relation.value)

    # ── Internal ──────────────────────────────────────────────────────

    def _resolve_role(self, signals: dict[str, Any]) -> WoERole:
        if self._role != WoERole.ATTACKER:
            return self._role

        job_name = str(signals.get("job_name", "novice") or "novice").lower()

        if any(s in job_name for s in ["priest", "arch bishop", "acolyte"]):
            return WoERole.SUPPORT
        if any(s in job_name for s in ["knight", "lord knight", "rune knight",
                                        "crusader", "paladin", "royal guard",
                                        "swordman"]):
            return WoERole.DEFENDER
        if any(s in job_name for s in ["assassin", "guillotine cross",
                                        "rogue", "stalker", "shadow chaser",
                                        "monk", "champion", "sura"]):
            return WoERole.BREAKER
        return WoERole.ATTACKER

    def _update_castle_state(
        self,
        map_name: str,
        players: list[Any],
        guild_name: str,
    ) -> None:
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

    def _update_guild_chemistry(self, players: list[Any], my_guild: str) -> None:
        """Track guild relationships based on encounters."""
        now = time.time()
        for p in players:
            if isinstance(p, dict):
                pg = str(p.get("guild_name", "") or "")
            else:
                pg = ""
            if not pg or pg == my_guild:
                continue

            if pg not in self._guild_chemistry:
                self._guild_chemistry[pg] = GuildChemistry(guild_name=pg)

            chem = self._guild_chemistry[pg]
            chem.last_encounter = now

            # Escalate hostility on repeated encounters
            if chem.relation == GuildRelation.NEUTRAL:
                chem.relation = GuildRelation.HOSTILE
                chem.aggression_score = min(1.0, chem.aggression_score + 0.1)

    def _update_emp_state(self, signals: dict[str, Any]) -> None:
        """Update EMP damage race state from signals."""
        emp_hp = signals.get("emperium_hp")
        emp_max_hp = signals.get("emperium_max_hp")
        if emp_hp is not None and emp_max_hp and emp_max_hp > 0:
            self._emp_damage_race.emp_hp_pct = emp_hp / emp_max_hp
            self._emp_damage_race.last_update = time.time()

    def _update_weather(self, signals: dict[str, Any], map_name: str) -> None:
        """Track guild weather skills (flare, chaos, etc.)."""
        weather = signals.get("weather", signals.get("map_weather", ""))
        if weather and map_name in self._castles:
            castle = self._castles[map_name]
            if weather != castle.weather_effect:
                castle.weather_effect = weather
                castle.last_weather_change = time.time()
                logger.info("[WoE] Weather on %s changed to: %s", map_name, weather)

    def _detect_mass_dispel(self, signals: dict[str, Any], map_name: str) -> None:
        """Detect mass dispel (Lex Aeterna, Darkness) and react."""
        now = time.time()
        # Check for debuff signals
        debuffs = signals.get("debuffs", []) or []
        mass_dispel_skills = {"lex_aeterna", "darkness", "dispel", "clearance"}
        for debuff in debuffs:
            if isinstance(debuff, dict):
                skill = str(debuff.get("skill", "") or "").lower()
            elif isinstance(debuff, str):
                skill = debuff.lower()
            else:
                continue
            if skill in mass_dispel_skills and map_name in self._castles:
                self._castles[map_name].mass_dispel_detected = True
                self._castles[map_name].last_mass_dispel_time = now
                logger.info("[WoE] Mass dispel detected on %s: %s", map_name, skill)

    def _is_emperium_room(self, map_name: str) -> bool:
        return any(frag in map_name for frag in EMPERIUM_MAP_FRAGMENTS)

    def _get_chokepoints(self, map_name: str) -> list[tuple[int, int]]:
        """Get chokepoint positions for a WoE map."""
        return WOE_CHOKEPOINTS.get(map_name, [])

    def _get_entrances(self, map_name: str) -> list[tuple[int, int]]:
        """Get entrance positions for a WoE map."""
        return WOE_ENTRANCES.get(map_name, [])

    def _get_defensive_positions(self, map_name: str) -> list[tuple[int, int]]:
        """Get defensive positions for a WoE map."""
        return WOE_DEFENSIVE_POSITIONS.get(map_name, [])

    # ── Action emitters ───────────────────────────────────────────────

    def _emit_emperium_attack(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
    ) -> None:
        """Focus all DPS on the emperium with class-appropriate skills."""
        logger.info("[WoE] %s: BREAKING emperium on %s!", bot_id, map_name)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()

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

        # Class-specific EMP attacks
        if "assassin" in job_name or "rogue" in job_name:
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill Backstab Emperium",
                confidence=0.90,
                domain="pvp",
                reason="Assassin: backstab EMP for max damage",
            ))
        elif "champion" in job_name or "monk" in job_name:
            # Asura timing when EMP HP is low
            if self._emp_damage_race.emp_hp_pct < 0.30:
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill Asura Strike Emperium",
                    confidence=0.95,
                    domain="pvp",
                    reason="Champion: Asura Strike on low HP EMP",
                ))
        elif "wizard" in job_name or "sage" in job_name:
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill Storm Gust Emperium",
                confidence=0.85,
                domain="pvp",
                reason="Wizard: AoE on EMP position",
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
            actions.append(HeuristicAction(
                kind="command",
                command="use 501",
                confidence=0.95,
                domain="pvp",
                reason=f"WoE HP low ({hp_ratio:.0%}) — healing",
            ))
            return

        # Navigate through chokepoints
        chokepoints = self._get_chokepoints(map_name)
        if chokepoints and (castle is None or not castle.in_emperium_room):
            target_cp = chokepoints[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {target_cp[0]} {target_cp[1]}",
                confidence=0.80,
                domain="pvp",
                reason=f"WoE push: moving to chokepoint on {map_name}",
            ))

        # Find and engage enemies
        enemies = self._find_enemies(signals)
        if enemies:
            # Prioritize high-value casters
            high_value = self._find_high_value_casters(signals)
            target = high_value[0] if high_value else enemies[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target}",
                confidence=0.90,
                domain="pvp",
                reason=f"WoE attack: pushing on {map_name} — engaging {target}",
                metadata={"map": map_name, "woe_role": "attacker"},
            ))

        if castle and castle.in_emperium_room:
            self._emit_emperium_attack(actions, bot_id, map_name, signals)

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
            # Move to defensive position
            def_positions = self._get_defensive_positions(map_name)
            if def_positions:
                pos = def_positions[0]
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"move {pos[0]} {pos[1]}",
                    confidence=0.80,
                    domain="pvp",
                    reason=f"WoE defend: moving to defensive position on {map_name}",
                ))
            else:
                actions.append(HeuristicAction(
                    kind="command",
                    command="sit",
                    confidence=0.70,
                    domain="pvp",
                    reason=f"WoE defend: guarding {map_name} — no enemies in sight",
                ))
            return

        # Intercept at chokepoints
        chokepoints = self._get_chokepoints(map_name)
        if chokepoints:
            cp = chokepoints[0]
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {cp[0]} {cp[1]}",
                confidence=0.85,
                domain="pvp",
                reason=f"WoE defend: holding chokepoint on {map_name}",
            ))

        # Engage the first enemy
        target = enemies[0]
        logger.info("[WoE] %s: DEFENDING %s vs %s", bot_id, map_name, target)
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
        job_name: str,
    ) -> None:
        """Heal, buff, and support allies with class-specific skills."""
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

            # Priest-specific: party healing range
            if "priest" in job_name or "high_priest" in job_name:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill Heal {target}",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"WoE support: healing {target} on {map_name}",
                    metadata={"map": map_name, "target": target, "woe_role": "support"},
                ))
                # Lex Aeterna on EMP if in range
                if self._is_emperium_room(map_name):
                    actions.append(HeuristicAction(
                        kind="command",
                        command="use_skill Lex Aeterna Emperium",
                        confidence=0.80,
                        domain="pvp",
                        reason="Priest: Lex Aeterna on EMP for 2x magic damage",
                    ))
                # Status removal
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill Status Removal",
                    confidence=0.70,
                    domain="pvp",
                    reason="Priest: removing status effects from party",
                ))
            else:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill Heal {target}",
                    confidence=0.95,
                    domain="pvp",
                    reason=f"WoE support: healing {target} on {map_name}",
                    metadata={"map": map_name, "target": target, "woe_role": "support"},
                ))
        else:
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

    # ── Class-specific actions ─────────────────────────────────────────

    def _emit_class_specific_actions(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
        job_name: str,
        role: WoERole,
    ) -> None:
        """Emit class-specific tactical actions based on job."""
        # ── Wizard: AoE placement, Safety Wall timing, SG at entrances ──
        if "wizard" in job_name or "high_wizard" in job_name:
            entrances = self._get_entrances(map_name)
            if entrances and role in (WoERole.DEFENDER, WoERole.ATTACKER):
                entrance = entrances[0]
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill Storm Gust {entrance[0]} {entrance[1]}",
                    confidence=0.80,
                    domain="pvp",
                    reason=f"Wizard: Storm Gust at entrance on {map_name}",
                ))
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill Safety Wall {bot_id}",
                    confidence=0.75,
                    domain="pvp",
                    reason="Wizard: Safety Wall for self-protection",
                ))

        # ── Priest: party healing range, Lex Aeterna, status removal ──
        elif "priest" in job_name or "high_priest" in job_name:
            if role == WoERole.SUPPORT:
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill Party Heal",
                    confidence=0.85,
                    domain="pvp",
                    reason="Priest: party-wide heal",
                ))

        # ── Assassin: cloaked EMP approach, backstab timing ──
        elif "assassin" in job_name or "guillotine" in job_name:
            if self._is_emperium_room(map_name):
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill Cloaking",
                    confidence=0.90,
                    domain="pvp",
                    reason="Assassin: cloaking for EMP approach",
                ))

        # ── Paladin: Devotion target selection, shield reflect ──
        elif "paladin" in job_name or "crusader" in job_name:
            # Find lowest HP ally for Devotion
            players = signals.get("players", []) or []
            guild = str(signals.get("guild_name", "") or "")
            lowest_hp_ally = ""
            lowest_hp = 1.0
            for p in players:
                if not isinstance(p, dict):
                    continue
                pg = str(p.get("guild_name", "") or "")
                p_hp = float(p.get("hp_ratio", 1.0) or 1.0)
                if pg == guild and p_hp < lowest_hp:
                    lowest_hp = p_hp
                    lowest_hp_ally = str(p.get("name", "") or "")
            if lowest_hp_ally:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill Devotion {lowest_hp_ally}",
                    confidence=0.85,
                    domain="pvp",
                    reason=f"Paladin: Devotion on {lowest_hp_ally} (HP {lowest_hp:.0%})",
                ))
            # Shield reflect
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill Shield Reflect",
                confidence=0.75,
                domain="pvp",
                reason="Paladin: activating Shield Reflect",
            ))

        # ── Champion: Asura timing when EMP HP is low ──
        elif "champion" in job_name or "monk" in job_name:
            if self._emp_damage_race.emp_hp_pct < 0.30:
                actions.append(HeuristicAction(
                    kind="command",
                    command="use_skill Asura Strike Emperium",
                    confidence=0.95,
                    domain="pvp",
                    reason="Champion: Asura Strike on low HP EMP",
                ))

    # ── EMP damage race ────────────────────────────────────────────────

    def _emit_emp_damage_call(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
    ) -> None:
        """Check EMP HP thresholds and call for damage focus."""
        message = self._emp_damage_race.check_threshold()
        if message:
            actions.append(HeuristicAction(
                kind="command",
                command=f"p '[WoE] {message}'",
                confidence=0.95,
                domain="pvp",
                reason=f"EMP damage race: {message}",
            ))

    # ── Enemy caster priority ──────────────────────────────────────────

    def _find_high_value_casters(self, signals: dict[str, Any]) -> list[str]:
        """Find high-value enemy casters to interrupt."""
        players = signals.get("players", []) or []
        guild = str(signals.get("guild_name", "") or "")
        casters: list[str] = []
        for p in players:
            if not isinstance(p, dict):
                continue
            p_name = str(p.get("name", "") or "")
            pg = str(p.get("guild_name", "") or "")
            p_job = str(p.get("job_name", "") or "").lower()
            if p_name and pg != guild and p_job in HIGH_VALUE_CASTERS:
                casters.append(p_name)
        return casters

    def _emit_caster_interrupt(
        self,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
        job_name: str,
    ) -> None:
        """Interrupt high-value enemy casters."""
        high_value = self._find_high_value_casters(signals)
        if not high_value:
            return

        target = high_value[0]

        # Melee classes: attack to interrupt
        if any(cls in job_name for cls in ["assassin", "rogue", "knight", "paladin",
                                            "champion", "monk", "swordman"]):
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target}",
                confidence=0.90,
                domain="pvp",
                reason=f"Interrupting high-value caster: {target}",
            ))

        # Ranged/magic: use interrupt skills
        if "wizard" in job_name or "sage" in job_name:
            actions.append(HeuristicAction(
                kind="command",
                command=f"use_skill Fire Bolt {target}",
                confidence=0.80,
                domain="pvp",
                reason=f"Interrupting caster {target} with Fire Bolt",
            ))

    # ── Gate timing ─────────────────────────────────────────────────────

    def _emit_gate_timing(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        signals: dict[str, Any],
    ) -> None:
        """Coordinate with guild for castle entry timing."""
        now = time.time()
        if now - self._last_gate_check < 5.0:
            return
        self._last_gate_check = now

        # Check if gate is open
        gate_open = signals.get("gate_open", signals.get("castle_gate_open", False))
        if map_name in self._castles:
            self._castles[map_name].gate_open = bool(gate_open)

        if not gate_open and now > self._gate_cooldown:
            # Request gate opening
            actions.append(HeuristicAction(
                kind="command",
                command="p '[WoE] Requesting gate open for castle entry'",
                confidence=0.70,
                domain="pvp",
                reason=f"Gate timing: requesting gate open on {map_name}",
            ))
            self._gate_cooldown = now + 30  # Don't spam

    # ── Weather tracking ───────────────────────────────────────────────

    def _emit_weather_reaction(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        castle: CastleState | None,
    ) -> None:
        """React to guild weather skills (flare, chaos, etc.)."""
        if not castle or not castle.weather_effect:
            return

        weather = castle.weather_effect.lower()

        # Flare: reduces accuracy — use AoE or reposition
        if "flare" in weather:
            actions.append(HeuristicAction(
                kind="command",
                command="move 50 50",
                confidence=0.70,
                domain="pvp",
                reason=f"Weather reaction: repositioning away from Flare on {map_name}",
            ))

        # Chaos: randomizes targets — use AoE skills
        elif "chaos" in weather:
            actions.append(HeuristicAction(
                kind="command",
                command="use_skill AoE Attack",
                confidence=0.75,
                domain="pvp",
                reason=f"Weather reaction: Chaos active — using AoE on {map_name}",
            ))

    # ── Mass dispel reaction ────────────────────────────────────────────

    def _emit_mass_dispel_reaction(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        map_name: str,
        castle: CastleState | None,
    ) -> None:
        """React to mass dispel by repositioning and rebuffing."""
        if not castle or not castle.mass_dispel_detected:
            return

        now = time.time()
        if now - castle.last_mass_dispel_time > 10.0:
            castle.mass_dispel_detected = False
            return

        # Reposition
        actions.append(HeuristicAction(
            kind="command",
            command="move 30 30",
            confidence=0.80,
            domain="pvp",
            reason=f"Mass dispel detected — repositioning on {map_name}",
        ))

        # Rebuff
        actions.append(HeuristicAction(
            kind="command",
            command="use_skill Blessing",
            confidence=0.85,
            domain="pvp",
            reason="Mass dispel — rebuffing",
        ))

    # ── Utility ───────────────────────────────────────────────────────

    @staticmethod
    def _find_enemies(signals: dict[str, Any]) -> list[str]:
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

    # ══════════════════════════════════════════════════════════════════
    #  New methods: Battlefield awareness, escape economy, barricade mgmt
    # ══════════════════════════════════════════════════════════════════

    def assess_battlefield_threat(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Assess battlefield threat and emit retreat/engage actions."""
        self._battlefield.assess(signals, actions, bot_id)

    def get_emperium_break_time(self, class_name: str) -> dict[str, Any]:
        """Calculate Emperium break time for a class."""
        estimate = self._emperium.calculate_break_time(class_name)
        return {
            "class": estimate.class_name,
            "dps": estimate.dps,
            "estimated_seconds": estimate.estimated_seconds,
            "estimated_time": estimate.estimated_seconds_formatted,
            "can_break": estimate.can_break,
            "best_skill": estimate.best_skill,
            "strategy": estimate.strategy,
        }

    def get_class_counter(self, attacker_class: str, defender_class: str) -> str | None:
        """Get counter strategy for class-vs-class in WoE."""
        from ai_sidecar.domains.social.class_combos import get_class_vs_class_counter
        return get_class_vs_class_counter(attacker_class, defender_class)

    def get_woe_prep_checklist(self, class_name: str) -> dict[str, Any]:
        """Get WoE preparation checklist."""
        return self._castle_intel.get_woe_prep_checklist(class_name)

    def recommend_escape_item(
        self,
        hp_pct: float,
        has_fly_wings: bool,
        has_butterfly_wings: bool,
        has_escape_scrolls: bool,
        zeny: int,
    ) -> tuple[str, str]:
        """Recommend escape item based on situation."""
        return self._battlefield.recommend_escape_item(
            hp_pct, has_fly_wings, has_butterfly_wings,
            has_escape_scrolls, zeny, in_woe=True,
        )

    def get_barricade_bypass(self, class_name: str) -> list[str]:
        """Get barricade bypass methods for a class."""
        return BARRICADE_BYPASS.get(class_name.lower(), [])

    def get_castle_under_attack(self) -> list[str]:
        """Get list of owned castles currently under attack."""
        castles = self._castle_intel.get_castles_under_attack()
        return [c.castle_id for c in castles]

    def update_guild_intel(self, guild_name: str, relation: str = "neutral") -> None:
        """Update guild intelligence."""
        rel = GuildRelation.NEUTRAL
        if relation == "alliance":
            rel = GuildRelation.ALLIANCE
        elif relation == "hostile":
            rel = GuildRelation.HOSTILE
        self.set_guild_relation(guild_name, rel)

