"""
PartyEngine — real RO party system with skill combos, positioning,
EXP share optimization, role assignment, and loot distribution.

A pro party in RO doesn't just stand next to each other. It:
  - Chains skills for multiplicative damage (Aspersio → Holy DPS)
  - Positions every member optimally (tank in front, priest mid, ranged back)
  - Tracks EXP share range so nobody misses the 160% bonus
  - Assigns roles based on job class and enforces position discipline
  - Distributes loot intelligently (cards/valuables → tank, rest → anyone)

Usage:
    from ai_sidecar.domains.social.party import PartyEngine, get_party_engine

    engine = get_party_engine()
    engine.assess(signals, actions, bot_id)
    positions = engine.get_formation_positions("kicapmasin")
    combos = engine.get_skill_combos(party_members)
    in_range = engine.is_within_share_range("kicapmasin", "kicapmasin2")
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Enums & data classes
# ──────────────────────────────────────────────


class PartyRole(StrEnum):
    """Roles the party engine assigns based on job class."""
    TANK = "Tank"
    DD_MELEE = "DD_Melee"
    DD_RANGED = "DD_Ranged"
    MAGIC = "Magic"
    SUPPORT = "Support"
    UNKNOWN = "Unknown"


class FormationType(StrEnum):
    """Named formations the party can adopt."""
    LINE = "line"           # Side-by-side, tank center-front
    WEDGE = "wedge"         # Arrowhead, tank at point
    CLUSTER = "cluster"     # Tight group (safe passages)
    SPREAD = "spread"       # Max distance (AoE avoidance)


class ComboState(StrEnum):
    """State of a skill combo in progress."""
    PENDING = "pending"
    PREP_CAST = "prep_cast"
    READY = "ready"
    MAIN_CAST = "main_cast"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SkillCombo:
    """A skill synergy that two or more party members can execute."""
    name: str
    prep_skill: str
    main_skill: str
    prep_class: str
    main_class: str
    prep_time_s: float = 2.0       # How long prep takes
    window_s: float = 5.0          # Window to follow up after prep lands
    description: str = ""
    min_level: int = 1
    target_required: bool = False   # Whether a specific target ID is needed

    # Active tracking
    state: ComboState = ComboState.PENDING
    started_at: float = 0.0
    prep_caster: str = ""
    main_caster: str = ""
    target_id: int = 0
    target_x: int = 0
    target_y: int = 0


@dataclass
class ComboOpportunity:
    """A detected combo opportunity between two party members."""
    combo: SkillCombo
    prep_member_name: str
    main_member_name: str
    readiness: float  # 0.0-1.0, how ready/viable this combo is right now
    target_id: int = 0
    target_x: int = 0
    target_y: int = 0


@dataclass
class PartyMemberInfo:
    """Runtime info about a party member for the engine."""
    name: str
    job: str = "novice"
    job_class: str = "novice"
    base_level: int = 1
    role: PartyRole = PartyRole.UNKNOWN
    x: int = 0
    y: int = 0
    map: str = ""
    online: bool = True
    hp: int = 1
    hp_max: int = 1
    sp: int = 1
    sp_max: int = 1
    distance_to_leader: float = 0.0
    distance_to_target: float = 0.0
    last_update: float = 0.0


@dataclass
class LootRule:
    """Rules for who should pick up what type of loot."""
    item_category: str          # "card", "equipment", "consumable", "valuable", "any"
    priority_role: PartyRole    # Which role gets first pick
    shared: bool = True         # Whether anyone can pick up if primary misses it
    announce: bool = False       # Whether to announce in party chat


# ──────────────────────────────────────────────
#  Skill combo definitions (RO-accurate)
# ──────────────────────────────────────────────

SKILL_COMBOS: list[SkillCombo] = [
    SkillCombo(
        name="Aspersio → Holy DPS",
        prep_skill="aspersio",
        main_skill="holy_attack",
        prep_class="priest",
        main_class="dd_melee",
        prep_time_s=2.0,
        window_s=30.0,  # Aspersio lasts 30s
        description="Priest blesses weapon with holy element — all attacks deal bonus holy damage vs undead/dark",
        target_required=True,
    ),
    SkillCombo(
        name="Storm Gust → Bowling Bash",
        prep_skill="storm_gust",
        main_skill="bowling_bash",
        prep_class="wizard",
        main_class="knight",
        prep_time_s=4.0,
        window_s=3.0,
        description="Wizard freezes/slows mobs in AoE, Knight Bowling Bash pushes them back through the storm",
        target_required=True,
    ),
    SkillCombo(
        name="Ankle Snare → Fire Wall",
        prep_skill="ankle_snare",
        main_skill="fire_wall",
        prep_class="hunter",
        main_class="wizard",
        prep_time_s=1.5,
        window_s=8.0,
        description="Hunter traps a monster, Wizard layers Fire Wall on top — trapped target takes repeated fire ticks",
        target_required=True,
    ),
    SkillCombo(
        name="Gloria → Crit Party",
        prep_skill="gloria",
        main_skill="crit_attack",
        prep_class="priest",
        main_class="dd_melee",
        prep_time_s=2.0,
        window_s=120.0,  # Gloria lasts 2min
        description="Priest casts Gloria — +20% crit rate for entire party, huge for Assassins and Hunters",
    ),
    SkillCombo(
        name="Frost Diver → Ranged Bonus",
        prep_skill="frost_diver",
        main_skill="ranged_attack",
        prep_class="mage",
        main_class="dd_ranged",
        prep_time_s=1.5,
        window_s=4.0,
        description="Mage freezes target solid, ranged attacker shatters it for bonus frost damage",
        target_required=True,
    ),
    SkillCombo(
        name="Lex Aeterna → Soul Strike",
        prep_skill="lex_aeterna",
        main_skill="soul_strike",
        prep_class="priest",
        main_class="wizard",
        prep_time_s=1.5,
        window_s=3.0,
        description="Priest marks target with Lex Aeterna (2x magic damage), Wizard follows with Soul Strike",
        target_required=True,
    ),
    SkillCombo(
        name="Magnificat → Any SP-heavy skill",
        prep_skill="magnificat",
        main_skill="any_sp_skill",
        prep_class="priest",
        main_class="any",
        prep_time_s=2.0,
        window_s=300.0,
        description="Priest casts Magnificat — +80% SP regen for 5min, enables sustained casting",
    ),
    SkillCombo(
        name="Safety Wall → Ranged PvP",
        prep_skill="safety_wall",
        main_skill="ranged_attack",
        prep_class="priest",
        main_class="dd_ranged",
        prep_time_s=1.0,
        window_s=10.0,
        description="Priest drops Safety Wall under the ranged DPS — 10 hits of melee immunity while they free-fire",
        target_required=False,
    ),
    SkillCombo(
        name="Impositio Manus → Heal Bomb",
        prep_skill="impositio_manus",
        main_skill="heal",
        prep_class="priest",
        main_class="priest",
        prep_time_s=1.0,
        window_s=60.0,
        description="Priest buffs own INT with Impositio Manus then delivers massive Heal-bomb damage to undead",
    ),
]

# ──────────────────────────────────────────────
#  Job → role mapping (RO-accurate)
# ──────────────────────────────────────────────

# Each entry: list of job names (lowercase) -> (role, priority)
# Priority determines which role wins if multi-class
JOB_TO_ROLE: dict[str, tuple[PartyRole, int]] = {
    # Novice / first classes
    "novice": (PartyRole.DD_MELEE, 5),
    "swordman": (PartyRole.TANK, 100),
    "mage": (PartyRole.MAGIC, 100),
    "archer": (PartyRole.DD_RANGED, 100),
    "acolyte": (PartyRole.SUPPORT, 100),
    "thief": (PartyRole.DD_MELEE, 100),
    "merchant": (PartyRole.DD_MELEE, 50),
    # 2-1 classes (transcendent preferred if matched)
    "knight": (PartyRole.TANK, 100),
    "crusader": (PartyRole.TANK, 100),
    "wizard": (PartyRole.MAGIC, 100),
    "sage": (PartyRole.MAGIC, 80),
    "hunter": (PartyRole.DD_RANGED, 100),
    "bard": (PartyRole.DD_RANGED, 70),
    "dancer": (PartyRole.DD_RANGED, 70),
    "priest": (PartyRole.SUPPORT, 100),
    "monk": (PartyRole.TANK, 60),  # Monk can off-tank or DPS
    "assassin": (PartyRole.DD_MELEE, 100),
    "rogue": (PartyRole.DD_MELEE, 90),
    "blacksmith": (PartyRole.DD_MELEE, 60),
    "alchemist": (PartyRole.SUPPORT, 50),
    # Transcendent 2-1
    "lord_knight": (PartyRole.TANK, 100),
    "paladin": (PartyRole.TANK, 100),
    "high_wizard": (PartyRole.MAGIC, 100),
    "professor": (PartyRole.MAGIC, 80),
    "sniper": (PartyRole.DD_RANGED, 100),
    "clown": (PartyRole.DD_RANGED, 70),
    "gypsy": (PartyRole.DD_RANGED, 70),
    "high_priest": (PartyRole.SUPPORT, 100),
    "champion": (PartyRole.TANK, 70),
    "assassin_cross": (PartyRole.DD_MELEE, 100),
    "stalker": (PartyRole.DD_MELEE, 90),
    "whitesmith": (PartyRole.DD_MELEE, 60),
    "creator": (PartyRole.SUPPORT, 50),
    # 2-2 classes
    "super_novice": (PartyRole.SUPPORT, 30),
    "taekwon": (PartyRole.DD_MELEE, 50),
    "soul_linker": (PartyRole.SUPPORT, 60),
    "ninja": (PartyRole.DD_MELEE, 80),
    "gunslinger": (PartyRole.DD_RANGED, 90),
}

# ──────────────────────────────────────────────
#  Positioning rules (cells)
# ──────────────────────────────────────────────

# Optimal distance ranges by role
ROLE_POSITION_RULES: dict[PartyRole, dict[str, int | float]] = {
    PartyRole.TANK: {
        "min_dist_from_target": 1,
        "max_dist_from_target": 2,
        "ideal_dist_from_anchor": 0,
        "min_dist_from_priest": 3,
    },
    PartyRole.DD_MELEE: {
        "min_dist_from_target": 1,
        "max_dist_from_target": 3,
        "ideal_dist_from_anchor": 2,
        "min_dist_from_priest": 2,
    },
    PartyRole.DD_RANGED: {
        "min_dist_from_target": 7,
        "max_dist_from_target": 9,
        "ideal_dist_from_anchor": 4,
        "min_dist_from_priest": 3,
    },
    PartyRole.MAGIC: {
        "min_dist_from_target": 5,
        "max_dist_from_target": 9,
        "ideal_dist_from_anchor": 4,
        "min_dist_from_priest": 2,
    },
    PartyRole.SUPPORT: {
        "min_dist_from_target": 8,
        "max_dist_from_target": 10,
        "ideal_dist_from_anchor": 3,
        "min_dist_from_priest": 0,
    },
    PartyRole.UNKNOWN: {
        "min_dist_from_target": 3,
        "max_dist_from_target": 7,
        "ideal_dist_from_anchor": 3,
        "min_dist_from_priest": 2,
    },
}

# EXP share range (RO hard cap)
EXP_SHARE_RANGE: int = 14

# Base EXP multipliers by party size
EXP_MULTIPLIERS: dict[int, float] = {
    1: 1.0,
    2: 1.2,
    3: 1.3,
    4: 1.4,
    5: 1.5,
    6: 1.6,
    7: 1.7,
    8: 1.8,
}

# ──────────────────────────────────────────────
#  Loot distribution rules
# ──────────────────────────────────────────────

LOOT_RULES: list[LootRule] = [
    LootRule(item_category="card", priority_role=PartyRole.TANK, shared=True, announce=True),
    LootRule(item_category="valuable", priority_role=PartyRole.TANK, shared=True, announce=True),
    LootRule(item_category="equipment", priority_role=PartyRole.TANK, shared=True, announce=False),
    LootRule(item_category="consumable", priority_role=PartyRole.SUPPORT, shared=True, announce=False),
    LootRule(item_category="any", priority_role=PartyRole.SUPPORT, shared=True, announce=False),
]


# ──────────────────────────────────────────────
#  Position helpers
# ──────────────────────────────────────────────


def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Euclidean distance between two points on the RO grid."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


# ──────────────────────────────────────────────
#  Engine
# ──────────────────────────────────────────────


class PartyEngine:
    """Real RO party system — skill combos, positioning, EXP share, roles, loot.

    Runs on every bot in the party. Each bot:
      1. Assesses party state from signals
      2. Detects skill combo opportunities
      3. Calculates optimal positions
      4. Tracks EXP share range
      5. Assigns roles
      6. Manages loot distribution
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Member tracking
        self._members: dict[str, PartyMemberInfo] = {}

        # Party metadata
        self._leader_name: str = ""
        self._party_name: str = ""
        self._in_party: bool = False
        self._share_exp: bool = True
        self._share_item: bool = False

        # Current formation
        self._current_formation: FormationType = FormationType.WEDGE
        self._formation_anchor_x: int = 0
        self._formation_anchor_y: int = 0

        # Active combos
        self._active_combos: list[SkillCombo] = []
        self._combo_cooldowns: dict[str, float] = {}  # combo_name -> time when ready again

        # Loot tracking
        self._loot_announced: set[str] = set()  # item names already announced

        # EXP stats
        self._exp_stats: dict[str, Any] = {
            "multiplier": 1.0,
            "members_in_range": 0,
            "total_members": 0,
            "warning": "",
        }

        # Leader cache
        self._is_leader: bool = False

    # ══════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[Any],
        bot_id: str,
    ) -> None:
        """Main assess method — run every PDCA cycle.

        Evaluates party state, manages combos, positions, EXP sharing,
        roles, and loot. Appends HeuristicAction-compatible dicts to
        *actions*.

        Args:
            signals: Bridge snapshot signals dict.
            actions: List to append generated actions to.
            bot_id: This bot's identifier.
        """
        if not signals or not bot_id:
            return

        with self._lock:
            self._sync_members(signals)
            self._update_party_meta(signals, bot_id)

            # 1. Role assignment
            self._assign_roles()

            # 2. Positioning
            self._evaluate_positioning(actions, bot_id, signals)

            # 3. EXP share optimization
            self._evaluate_exp_share(actions, bot_id)

            # 4. Skill combo detection & execution
            self._process_combos(actions, bot_id, signals)

            # 5. Loot distribution
            self._process_loot(actions, bot_id, signals)

            # 6. Party response commands
            self._handle_party_commands(actions, bot_id, signals)

    def get_formation_positions(self, bot_id: str) -> list[dict[str, int]]:
        """Get the formation position(s) for a bot.

        Returns a list of (x, y) dicts for this bot's role in the
        current formation. Usually one position, but some formations
        may have multiple valid slots.

        Args:
            bot_id: Bot identifier.

        Returns:
            List of dicts with 'x' and 'y' keys.
        """
        with self._lock:
            member = self._get_member(bot_id)
            if not member:
                return []

            role = member.role
            anchor_x = self._formation_anchor_x or member.x
            anchor_y = self._formation_anchor_y or member.y

            offsets = self._get_role_offsets(role, len(self._online_members()))
            target_x = anchor_x + offsets["dx"]
            target_y = anchor_y + offsets["dy"]

            return [{"x": target_x, "y": target_y}]

    def get_skill_combos(
        self,
        party_members: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Check what skill combos are available given current party composition.

        Args:
            party_members: List of party member dicts, each with at least
                          'name' and 'job' or 'job_class' keys.

        Returns:
            List of combo opportunity dicts with keys:
                combo_name, description, prep_member, main_member,
                prep_skill, main_skill, window_s, readiness.
        """
        with self._lock:
            opportunities: list[dict[str, Any]] = []

            # Parse member classes into a lookup
            member_jobs: dict[str, str] = {}
            for m in party_members:
                name = m.get("name", "") or m.get("bot_id", "")
                job = (m.get("job_class", "") or m.get("job", "") or "").lower()
                if name:
                    member_jobs[name] = job

            for combo in SKILL_COMBOS:
                prep_candidates = []
                main_candidates = []

                for name, job in member_jobs.items():
                    if self._job_matches_class(job, combo.prep_class):
                        prep_candidates.append(name)
                    if (
                        combo.main_class == "any"
                        or self._job_matches_class(job, combo.main_class)
                    ):
                        main_candidates.append(name)

                # Need at least one prep and one main (can be same person for self-combos)
                if prep_candidates and main_candidates:
                    for prep_name in prep_candidates:
                        for main_name in main_candidates:
                            readiness = self._calc_readiness(
                                combo, prep_name, main_name, member_jobs,
                            )
                            opportunities.append({
                                "combo_name": combo.name,
                                "description": combo.description,
                                "prep_member": prep_name,
                                "main_member": main_name,
                                "prep_skill": combo.prep_skill,
                                "main_skill": combo.main_skill,
                                "window_s": combo.window_s,
                                "readiness": round(readiness, 2),
                            })

            return opportunities

    def is_within_share_range(
        self,
        bot_id: str,
        other_bot_id: str,
    ) -> bool:
        """Check if two party members are within EXP share range (≤14 cells).

        Args:
            bot_id: First bot identifier.
            other_bot_id: Second bot identifier.

        Returns:
            True if both are on the same map and within 14 cells.
        """
        with self._lock:
            a = self._get_member(bot_id)
            b = self._get_member(other_bot_id)
            if not a or not b or not a.online or not b.online:
                return False
            if a.map != b.map:
                return False
            dist = _distance(a.x, a.y, b.x, b.y)
            return dist <= EXP_SHARE_RANGE

    # ── Getters ──────────────────────────────

    def get_member(self, name: str) -> PartyMemberInfo | None:
        with self._lock:
            return self._members.get(name)

    def get_online_members(self) -> list[PartyMemberInfo]:
        with self._lock:
            return self._online_members()

    def get_role(self, name: str) -> PartyRole:
        with self._lock:
            member = self._members.get(name)
            return member.role if member else PartyRole.UNKNOWN

    def get_exp_multiplier(self) -> float:
        with self._lock:
            return self._exp_stats.get("multiplier", 1.0)

    def get_party_summary(self) -> str:
        """Return a human-readable party status overview."""
        with self._lock:
            lines = [f"── Party Engine ──"]
            lines.append(f"Party: {self._party_name or 'none'}")
            lines.append(f"Leader: {self._leader_name or 'none'}")
            lines.append(f"Members: {len(self._members)} online, "
                         f"{sum(1 for m in self._members.values() if m.online)}")
            lines.append(f"Formation: {self._current_formation}")
            exp = self._exp_stats
            lines.append(f"EXP mult: {exp.get('multiplier', 1.0):.1f}x "
                         f"({exp.get('members_in_range', 0)}/{exp.get('total_members', 0)} in range)")
            if exp.get("warning"):
                lines.append(f"⚠  {exp['warning']}")
            lines.append("")
            for name, member in self._members.items():
                dist = member.distance_to_leader
                lines.append(
                    f"  {name:20s} Lv{member.base_level:<3d} "
                    f"{member.role.value:<10s} "
                    f"{member.job:<15s} "
                    f"dist={dist:.0f}c"
                )
            active = [c for c in self._active_combos if c.state not in (ComboState.COMPLETED, ComboState.FAILED)]
            if active:
                lines.append("")
                lines.append("Active combos:")
                for c in active:
                    lines.append(f"  [{c.state.value}] {c.name} ({c.prep_caster}→{c.main_caster})")
            return "\n".join(lines)

    def reset(self) -> None:
        """Reset all party state."""
        with self._lock:
            self._members.clear()
            self._active_combos.clear()
            self._combo_cooldowns.clear()
            self._loot_announced.clear()
            self._exp_stats = {"multiplier": 1.0, "members_in_range": 0,
                               "total_members": 0, "warning": ""}
            self._leader_name = ""
            self._party_name = ""
            self._in_party = False

    # ══════════════════════════════════════════
    #  Internal: member sync
    # ══════════════════════════════════════════

    def _sync_members(self, signals: dict[str, Any]) -> None:
        """Sync party member data from signals."""
        now = time.time()
        raw_members: list[dict[str, Any]] = list(
            signals.get("party_members", signals.get("party", [])) or []
        )
        all_bots: list[str] = list(signals.get("all_bots", []) or [])
        leader_name = signals.get("party_leader") or signals.get("leader_name", "")
        self_x = int(signals.get("x", 0))
        self_y = int(signals.get("y", 0))
        current_map = str(signals.get("map", "") or "")

        # If signals has my own info, ensure we're tracked
        my_name = signals.get("name", "")
        if my_name and my_name not in self._members:
            self._members[my_name] = PartyMemberInfo(name=my_name)

        # Process each raw member
        seen: set[str] = set()
        for m in raw_members:
            if isinstance(m, str):
                name = m
                if name not in self._members:
                    self._members[name] = PartyMemberInfo(name=name)
                seen.add(name)
            elif isinstance(m, dict):
                name = str(m.get("name", ""))
                if not name:
                    continue
                seen.add(name)
                if name not in self._members:
                    self._members[name] = PartyMemberInfo(name=name)
                pm = self._members[name]
                pm.job = str(m.get("job", m.get("job_class", pm.job)))
                pm.job_class = str(m.get("job_class", m.get("job", pm.job_class)))
                pm.base_level = int(m.get("level", m.get("base_level", pm.base_level)))
                pm.x = int(m.get("x", m.get("pos_x", pm.x)))
                pm.y = int(m.get("y", m.get("pos_y", pm.y)))
                pm.map = str(m.get("map", m.get("map_name", pm.map or current_map)))
                pm.hp = int(m.get("hp", pm.hp))
                pm.hp_max = int(m.get("hp_max", m.get("max_hp", pm.hp_max)))
                pm.sp = int(m.get("sp", pm.sp))
                pm.sp_max = int(m.get("sp_max", m.get("max_sp", pm.sp_max)))
                pm.online = bool(m.get("online", True))
                pm.last_update = now

        # Also check signals for direct position data
        self_x = int(signals.get("x", self_x))
        self_y = int(signals.get("y", self_y))
        if my_name and my_name in self._members:
            self._members[my_name].x = self_x
            self._members[my_name].y = self_y
            self._members[my_name].map = current_map
            self._members[my_name].last_update = now
            self._members[my_name].online = True

        # Mark unseen members as offline
        for name in self._members:
            if name not in seen and name != my_name:
                if now - self._members[name].last_update > 30:
                    self._members[name].online = False

        # Set anchor to leader position
        leader_name = leader_name or self._leader_name
        if leader_name and leader_name in self._members:
            self._formation_anchor_x = self._members[leader_name].x
            self._formation_anchor_y = self._members[leader_name].y

    def _update_party_meta(self, signals: dict[str, Any], bot_id: str) -> None:
        """Update party metadata from signals."""
        self._in_party = bool(signals.get("in_party", False))
        self._party_name = str(signals.get("party_name", self._party_name or ""))
        self._leader_name = signals.get("party_leader") or signals.get("leader_name", self._leader_name)
        self._is_leader = bool(signals.get("is_leader", False))
        self._share_exp = bool(signals.get("party_share_exp", True))
        self._share_item = bool(signals.get("party_share_item", False))

    def _get_member(self, name: str) -> PartyMemberInfo | None:
        return self._members.get(name)

    def _online_members(self) -> list[PartyMemberInfo]:
        return [m for m in self._members.values() if m.online]

    # ══════════════════════════════════════════
    #  Internal: role assignment
    # ══════════════════════════════════════════

    def _assign_roles(self) -> None:
        """Auto-assign PartyRole to every member based on job class."""
        for name, member in self._members.items():
            if not member.online:
                continue
            member.role = self._job_to_role(member.job.lower())

    @staticmethod
    def _job_to_role(job: str) -> PartyRole:
        """Map a job class string to a PartyRole.

        Uses class family matching (e.g. 'champion' matches as tank).
        Falls back to role for base classes if 2-1 class not found.
        """
        # Direct lookup
        if job in JOB_TO_ROLE:
            return JOB_TO_ROLE[job][0]

        # Partial matching: check if any known job is a substring or starts with
        # e.g. "high_wizard" -> check "high_wizard", "wizard"
        for known_job, (role, _priority) in JOB_TO_ROLE.items():
            if known_job in job or job in known_job:
                return role

        # Family matching: detect based on class keywords
        job_lower = job.lower().replace("_", " ").replace("-", " ")
        tank_keywords = {"swordman", "knight", "crusader", "paladin", "tank"}
        melee_keywords = {"thief", "assassin", "rogue", "dagger", "melee", "taekwon", "ninja"}
        ranged_keywords = {"archer", "hunter", "bow", "ranged", "gunslinger", "bard", "dancer",
                           "clown", "gypsy"}
        magic_keywords = {"mage", "wizard", "sage", "magic", "professor", "soul"}
        support_keywords = {"acolyte", "priest", "monk", "heal", "support", "alchemist", "creator"}

        words = set(job_lower.split())
        if words & tank_keywords:
            return PartyRole.TANK
        if words & melee_keywords:
            return PartyRole.DD_MELEE
        if words & ranged_keywords:
            return PartyRole.DD_RANGED
        if words & magic_keywords:
            return PartyRole.MAGIC
        if words & support_keywords:
            return PartyRole.SUPPORT

        return PartyRole.DD_MELEE  # Default to melee DPS

    # ══════════════════════════════════════════
    #  Internal: positioning
    # ══════════════════════════════════════════

    @staticmethod
    def _get_role_offsets(role: PartyRole, member_count: int) -> dict[str, int]:
        """Get formation position offsets for a given role.

        Produces a staggered formation with tank in front, support behind,
        melee flanking, ranged/magic at range.
        """
        rules = ROLE_POSITION_RULES.get(role, ROLE_POSITION_RULES[PartyRole.UNKNOWN])
        ideal_dist = int(rules["ideal_dist_from_anchor"])

        # Different formations produce different offsets
        # WEDGE: tank at point, melee on flanks, range/magic back, support rear
        base_dx = 0
        base_dy = ideal_dist

        # Add staggering based on role to create a natural formation
        stagger: dict[PartyRole, tuple[int, int]] = {
            PartyRole.TANK: (0, 0),        # Center-front
            PartyRole.DD_MELEE: (-2, -1),   # Left flank (slightly behind tank)
            PartyRole.DD_RANGED: (-4, -2),   # Left-back (range)
            PartyRole.MAGIC: (4, -2),        # Right-back (range)
            PartyRole.SUPPORT: (0, -4),     # Center-rear
            PartyRole.UNKNOWN: (-3, -2),
        }

        dx_off, dy_off = stagger.get(role, (0, 0))

        # If multiple members have the same role, distribute them
        return {"dx": base_dx + dx_off, "dy": base_dy + dy_off}

    def _evaluate_positioning(
        self,
        actions: list[Any],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Check positioning and issue move commands if needed."""
        member = self._get_member(bot_id)
        if not member:
            return

        leader = self._get_member(self._leader_name)
        anchor_x = self._formation_anchor_x
        anchor_y = self._formation_anchor_y

        # Update distance tracking
        for name, m in self._members.items():
            if name == bot_id:
                if leader:
                    m.distance_to_leader = _distance(m.x, m.y, leader.x, leader.y)
            else:
                if member:
                    m.distance_to_leader = _distance(m.x, m.y, anchor_x, anchor_y)

        # What's the current target position?
        target_positions = self.get_formation_positions(bot_id)
        if not target_positions:
            return
        target = target_positions[0]
        tx, ty = target["x"], target["y"]

        # How far is the bot from its ideal position?
        dist = _distance(member.x, member.y, tx, ty)

        # Only move if we're significantly out of position
        if dist > 3.0:
            self._append_action(
                actions,
                command=f"move {tx} {ty}",
                confidence=0.85,
                reason=f"Party position: move to formation slot ({tx},{ty}) "
                       f"(currently {int(dist)}c away)",
            )

        # Tank positioning check: must be closest to target
        if member.role == PartyRole.TANK:
            monsters: list[dict[str, Any]] = list(signals.get("monsters", []) or [])
            if monsters:
                closest_monster = monsters[0]
                mx = int(closest_monster.get("x", 0))
                my = int(closest_monster.get("y", 0))
                tank_dist = _distance(member.x, member.y, mx, my)

                for name, other in self._members.items():
                    if name == bot_id or not other.online:
                        continue
                    other_dist = _distance(other.x, other.y, mx, my)
                    if other_dist < tank_dist - 2 and other.role != PartyRole.TANK:
                        # Tank is not in front — someone else is closer
                        self._append_action(
                            actions,
                            command=f"move {mx} {my}",
                            confidence=0.90,
                            reason=f"Party tank: move ahead — {name} is closer to target",
                        )
                        break

    # ══════════════════════════════════════════
    #  Internal: EXP share optimization
    # ══════════════════════════════════════════

    def _evaluate_exp_share(
        self,
        actions: list[Any],
        bot_id: str,
    ) -> None:
        """Evaluate EXP share status and recommend actions.

        In RO:
        - Party of Priest + Tank + DD = 160% total XP (1.6x base multi)
        - All members must be within 14 cells to share
        - Out-of-range members lose the bonus for everyone
        """
        online = self._online_members()
        total = len(online)
        self._exp_stats["total_members"] = total

        if total <= 1 or not self._in_party:
            self._exp_stats["multiplier"] = 1.0
            self._exp_stats["members_in_range"] = total
            self._exp_stats["warning"] = ""
            return

        # Find members in share range
        leader = self._get_member(self._leader_name)
        if not leader:
            leader = self._get_member(bot_id)
        if not leader:
            return

        in_range = 0
        out_of_range_names: list[str] = []
        for m in online:
            dist = _distance(m.x, m.y, leader.x, leader.y)
            m.distance_to_leader = dist
            if dist <= EXP_SHARE_RANGE:
                in_range += 1
            else:
                out_of_range_names.append(m.name)

        self._exp_stats["members_in_range"] = in_range

        # Compute multiplier
        base_mult = EXP_MULTIPLIERS.get(total, 1.5)
        if in_range < total:
            # Not everyone is sharing — reduced multiplier
            # RO: only members in range get the share; members out of range
            # get solo XP. The party still gets the bonus for those in range.
            effective = in_range
            self._exp_stats["multiplier"] = EXP_MULTIPLIERS.get(effective, 1.0)
            self._exp_stats["warning"] = (
                f"{total - in_range} member(s) out of EXP range "
                f"(>{EXP_SHARE_RANGE}c). "
                f"Effective multi: {self._exp_stats['multiplier']:.1f}x instead of {base_mult:.1f}x"
            )
        else:
            self._exp_stats["multiplier"] = base_mult
            self._exp_stats["warning"] = ""

        # Recommend regroup if someone is too far
        if out_of_range_names and self._is_leader:
            self._append_action(
                actions,
                command="party regroup",
                confidence=0.80,
                reason=f"EXP share: {', '.join(out_of_range_names)} out of range "
                       f"(>{EXP_SHARE_RANGE}c). Regroup to restore "
                       f"{base_mult:.1f}x multiplier",
            )

    # ══════════════════════════════════════════
    #  Internal: skill combos
    # ══════════════════════════════════════════

    def _process_combos(
        self,
        actions: list[Any],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Detect combo opportunities and manage active combos."""
        now = time.time()

        # 1. Clean up expired/old combos
        self._active_combos = [
            c for c in self._active_combos
            if c.state not in (ComboState.COMPLETED, ComboState.FAILED)
            and (c.started_at == 0 or now - c.started_at < 60)
        ]

        # 2. Find new combo opportunities
        member_names: dict[str, str] = {}
        for m in self._online_members():
            member_names[m.name] = m.job.lower()

        if not member_names:
            return

        for combo in SKILL_COMBOS:
            # Skip if on cooldown
            cd_key = f"{combo.name}:{bot_id}"
            if cd_key in self._combo_cooldowns and now < self._combo_cooldowns[cd_key]:
                continue

            # Check if already active
            if any(c.name == combo.name for c in self._active_combos):
                continue

            # Find prep caster
            if not self._job_matches_class(
                member_names.get(bot_id, ""), combo.prep_class
            ):
                continue

            # Find main caster among party
            for m_name, m_job in member_names.items():
                if m_name == bot_id and combo.prep_class == combo.main_class:
                    # Self-combo (e.g. Priest Impositio Manus → Heal)
                    pass
                if m_name == bot_id:
                    continue
                if combo.main_class == "any" or self._job_matches_class(m_job, combo.main_class):
                    opportunity = self._check_combo_viability(
                        combo, bot_id, m_name, signals,
                    )
                    if opportunity and opportunity.readiness > 0.5:
                        self._start_combo(opportunity, bot_id)
                        self._append_action(
                            actions,
                            command=f"party skill_combo {combo.prep_skill} {m_name}",
                            confidence=0.85,
                            reason=f"Combo: {combo.name} — {bot_id} casts "
                                   f"{combo.prep_skill} → {m_name} follows with "
                                   f"{combo.main_skill}",
                        )
                        break

        # 3. Progress active combos
        for combo in self._active_combos:
            self._progress_combo(combo, actions, bot_id, now)

    def _check_combo_viability(
        self,
        combo: SkillCombo,
        prep_caster: str,
        main_caster: str,
        signals: dict[str, Any],
    ) -> ComboOpportunity | None:
        """Check if a combo is viable right now."""
        prep_member = self._get_member(prep_caster)
        main_member = self._get_member(main_caster)

        if not prep_member or not main_member or not prep_member.online or not main_member.online:
            return None

        # Check SP availability
        if prep_member.sp < prep_member.sp_max * 0.2:
            return None  # Not enough SP to cast

        # Check distance (prep and main must be within combo range)
        dist = _distance(prep_member.x, prep_member.y, main_member.x, main_member.y)

        # Basic readiness calculation
        readiness = 1.0

        # Distance penalty
        if dist > 14:
            readiness -= 0.3
        if dist > 20:
            readiness -= 0.3

        # Level penalty if below min
        if prep_member.base_level < combo.min_level:
            readiness -= 0.5
        if main_member.base_level < combo.min_level:
            readiness -= 0.3

        # HP penalty: don't combo if about to die
        hp_pct = prep_member.hp / max(1, prep_member.hp_max)
        if hp_pct < 0.3:
            readiness -= 0.5
        elif hp_pct < 0.5:
            readiness -= 0.2

        # Find target if required
        target_id = 0
        target_x = 0
        target_y = 0
        if combo.target_required:
            monsters: list[dict[str, Any]] = list(signals.get("monsters", []) or [])
            if monsters:
                closest = monsters[0]
                target_id = int(closest.get("id", 0))
                target_x = int(closest.get("x", 0))
                target_y = int(closest.get("y", 0))
            else:
                readiness -= 0.5  # No target available

        return ComboOpportunity(
            combo=combo,
            prep_member_name=prep_caster,
            main_member_name=main_caster,
            readiness=max(0.0, min(1.0, readiness)),
            target_id=target_id,
            target_x=target_x,
            target_y=target_y,
        )

    def _start_combo(self, opportunity: ComboOpportunity, bot_id: str) -> None:
        """Begin tracking a new combo execution."""
        now = time.time()
        combo = SkillCombo(
            name=opportunity.combo.name,
            prep_skill=opportunity.combo.prep_skill,
            main_skill=opportunity.combo.main_skill,
            prep_class=opportunity.combo.prep_class,
            main_class=opportunity.combo.main_class,
            prep_time_s=opportunity.combo.prep_time_s,
            window_s=opportunity.combo.window_s,
            description=opportunity.combo.description,
            min_level=opportunity.combo.min_level,
            target_required=opportunity.combo.target_required,
            state=ComboState.PREP_CAST,
            started_at=now,
            prep_caster=opportunity.prep_member_name,
            main_caster=opportunity.main_member_name,
            target_id=opportunity.target_id,
            target_x=opportunity.target_x,
            target_y=opportunity.target_y,
        )
        self._active_combos.append(combo)

    def _progress_combo(
        self,
        combo: SkillCombo,
        actions: list[Any],
        bot_id: str,
        now: float,
    ) -> None:
        """Move a combo through its lifecycle."""
        elapsed = now - combo.started_at

        if combo.state == ComboState.PREP_CAST:
            if elapsed >= combo.prep_time_s:
                combo.state = ComboState.READY
                # Notify main caster
                if bot_id == combo.prep_caster or bot_id == combo.main_caster:
                    self._append_action(
                        actions,
                        command=f"p 'Combo ready: {combo.name} — follow up!'",
                        confidence=0.90,
                        reason=f"Combo ready: {combo.name} prep complete",
                        kind="chat",
                    )

        elif combo.state == ComboState.READY:
            # Check if window is closing
            remaining = combo.window_s - elapsed
            if remaining <= 0:
                combo.state = ComboState.FAILED
                cd_key = f"{combo.name}:{combo.prep_caster}"
                self._combo_cooldowns[cd_key] = now + 10
                if bot_id == combo.main_caster or bot_id == combo.prep_caster:
                    self._append_action(
                        actions,
                        command=f"p 'Combo missed: {combo.name} window expired'",
                        confidence=0.70,
                        reason=f"Combo failed: {combo.name} window expired",
                        kind="chat",
                    )
            elif combo.main_caster == bot_id and elapsed >= combo.prep_time_s:
                # Signal main caster to execute
                combo.state = ComboState.MAIN_CAST
                target_str = ""
                if combo.target_id:
                    target_str = f" on target {combo.target_id}"
                self._append_action(
                    actions,
                    command=f"ss {combo.main_skill}{target_str}",
                    confidence=0.75,
                    reason=f"Combo execute: {combo.main_skill} "
                           f"(follow-up to {combo.prep_caster}'s {combo.prep_skill})",
                )

        elif combo.state == ComboState.MAIN_CAST:
            if elapsed >= combo.prep_time_s + 2.0:
                combo.state = ComboState.COMPLETED
                cd_key = f"{combo.name}:{combo.prep_caster}"
                self._combo_cooldowns[cd_key] = now + 15
                if bot_id == combo.prep_caster:
                    self._append_action(
                        actions,
                        command=f"p '{combo.name} landed!'",
                        confidence=0.85,
                        reason=f"Combo completed: {combo.name}",
                        kind="chat",
                    )

    # ══════════════════════════════════════════
    #  Internal: loot distribution
    # ══════════════════════════════════════════

    def _process_loot(
        self,
        actions: list[Any],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Evaluate loot on the ground and assign pickup responsibilities."""
        ground_items: list[dict[str, Any]] = list(
            signals.get("items", signals.get("ground_items", [])) or []
        )
        if not ground_items:
            return

        member = self._get_member(bot_id)
        if not member:
            return

        for item in ground_items:
            item_name = str(item.get("name", item.get("identifiedDisplayName", "")))
            item_id = str(item.get("id", item.get("item_id", 0)))
            item_x = int(item.get("x", item.get("pos_x", 0)))
            item_y = int(item.get("y", item.get("pos_y", 0)))

            if not item_name or not item_id:
                continue

            # Determine item category
            category = self._classify_item(item)

            # Find matching loot rule
            rule = self._find_loot_rule(category)

            # Check if this bot should pick it up
            should_pickup = False
            role = member.role

            if rule:
                if role == rule.priority_role:
                    should_pickup = True
                elif rule.shared:
                    # Anyone can pick up, but check proximity
                    dist = _distance(member.x, member.y, item_x, item_y)
                    should_pickup = dist < 5

            if should_pickup:
                # Announce valuable drops
                if rule and rule.announce and item_name not in self._loot_announced:
                    self._loot_announced.add(item_name)
                    self._append_action(
                        actions,
                        command=f"p '🎒 {item_name} dropped — I\'ll grab it'",
                        confidence=0.95,
                        reason=f"Loot: picking up {item_name} (role: {role.value})",
                        kind="chat",
                    )

                self._append_action(
                    actions,
                    command=f"take {item_id}",
                    confidence=0.90,
                    reason=f"Loot: {item_name} at ({item_x},{item_y})",
                )

    @staticmethod
    def _classify_item(item: dict[str, Any]) -> str:
        """Classify a ground item into a loot category."""
        name = str(item.get("name", item.get("identifiedDisplayName", ""))).lower()
        item_type = str(item.get("type", "")).lower()

        # Card detection
        if "card" in name or item_type == "card":
            return "card"

        # Equipment detection
        equipment_types = {"weapon", "armor", "shield", "shoe", "garment",
                           "accessory", "headgear", "helm", "manteau", "boots"}
        if item_type in equipment_types:
            return "equipment"
        if any(kw in name for kw in ["_helm", "_shield", "_armor", "_boot",
                                     "muffler", "cloak", "coat", "robe",
                                     "suit", "plate", "greaves", "bracers"]):
            return "equipment"

        # Valuable detection (ores, gems, hard-to-find materials)
        valuable_keywords = {"rough_elunium", "rough_oridecon", "elunium", "oridecon",
                             "emerald", "ruby", "sapphire", "diamond", "topaz",
                             "amethyst", "opal", "garnet", "zircon", "jade",
                             "pearl", "aquamarine", "carnelian", "moonstone",
                             "gold", "mithril", "mythril", "star_crumb",
                             "bubble_gum", "life_insurance", "insurance"}
        if name in valuable_keywords:
            return "valuable"
        if any(kw in name for kw in ["rough_", "jewel", "gem", "pearl", "diamond"]):
            return "valuable"

        # Potions/consumables
        consumable_keywords = {"potion", "fruit", "food", "fish", "meat",
                               "bread", "cake", "cookie", "candy", "herb",
                               "mushroom", "flower", "leaf", "sprout",
                               "bottle", "concentrated", "mixture", "pill"}
        if item_type == "potion" or item_type == "consumable" or any(kw in name for kw in consumable_keywords):
            return "consumable"

        return "any"

    @staticmethod
    def _find_loot_rule(category: str) -> LootRule | None:
        """Find the best loot rule for a category."""
        exact_rules = [r for r in LOOT_RULES if r.item_category == category]
        if exact_rules:
            return exact_rules[0]
        # Fallback to "any"
        any_rules = [r for r in LOOT_RULES if r.item_category == "any"]
        return any_rules[0] if any_rules else None

    # ══════════════════════════════════════════
    #  Internal: party commands
    # ══════════════════════════════════════════

    def _handle_party_commands(
        self,
        actions: list[Any],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Handle party response commands from signals/bot input."""
        cmd = signals.get("party_command", signals.get("command", ""))
        if not cmd or not isinstance(cmd, str):
            return

        cmd = cmd.strip().lower()

        if cmd.startswith("party position"):
            # Format: party position <formation_type>
            parts = cmd.split()
            if len(parts) >= 3:
                formation_str = parts[2]
                try:
                    self._current_formation = FormationType(formation_str)
                except ValueError:
                    logger.warning("Unknown formation: %s", formation_str)
            self._append_action(
                actions,
                command=f"p 'Formation set to {self._current_formation.value}'",
                confidence=0.95,
                reason=f"Party command: formation → {self._current_formation}",
                kind="chat",
            )

        elif cmd == "party regroup":
            # Everyone moves to leader's position
            leader = self._get_member(self._leader_name)
            if leader:
                self._append_action(
                    actions,
                    command=f"move {leader.x} {leader.y}",
                    confidence=0.95,
                    reason=f"Party regroup to leader @ ({leader.x},{leader.y})",
                )
                self._append_action(
                    actions,
                    command="p 'Regrouping to leader position'",
                    confidence=0.90,
                    reason="Party regroup announcement",
                    kind="chat",
                )

        elif cmd.startswith("party skill_combo"):
            # Format: party skill_combo <skill_id> <target_id>
            parts = cmd.split()
            if len(parts) >= 3:
                skill_id = parts[2]
                target_id = parts[3] if len(parts) >= 4 else "self"
                self._append_action(
                    actions,
                    command=f"ss {skill_id}",
                    confidence=0.80,
                    reason=f"Party command: combo skill {skill_id} on {target_id}",
                )

        elif cmd.startswith("follow"):
            # Format: follow <member>
            parts = cmd.split()
            if len(parts) >= 2:
                target_name = parts[1]
                target = self._get_member(target_name)
                if target:
                    # Store follow target for the PDCA loop
                    self._exp_stats["follow_target"] = target_name
                    self._append_action(
                        actions,
                        command=f"move {target.x} {target.y}",
                        confidence=0.90,
                        reason=f"Following party member {target_name}",
                    )

    # ══════════════════════════════════════════
    #  Internal: helpers
    # ══════════════════════════════════════════

    @staticmethod
    def _job_matches_class(job: str, class_name: str) -> bool:
        """Check if a job name matches a combo class requirement."""
        job_norm = job.lower().replace("_", " ").replace("-", " ").strip()
        cls_norm = class_name.lower().replace("_", " ").replace("-", " ").strip()

        if cls_norm == "any":
            return True

        # Direct match
        if cls_norm == job_norm:
            return True

        # Class family matching
        class_families: dict[str, list[str]] = {
            "priest": ["acolyte", "priest", "high priest", "monk", "champion"],
            "wizard": ["mage", "wizard", "high wizard", "sage", "professor"],
            "knight": ["swordman", "knight", "lord knight", "crusader", "paladin"],
            "hunter": ["archer", "hunter", "sniper", "bard", "dancer", "clown", "gypsy"],
            "mage": ["mage", "wizard", "high wizard", "sage", "professor"],
            "dd melee": ["thief", "assassin", "assassin cross", "rogue", "stalker",
                         "swordman", "knight", "lord knight", "monk", "champion",
                         "ninja", "taekwon", "merchant", "blacksmith", "whitesmith"],
            "dd ranged": ["archer", "hunter", "sniper", "bard", "dancer",
                          "clown", "gypsy", "gunslinger"],
        }

        if cls_norm in class_families:
            return any(
                job_norm == fam_job
                or job_norm.startswith(fam_job)
                or fam_job.startswith(job_norm)
                for fam_job in class_families[cls_norm]
            )

        # Substring match
        if cls_norm in job_norm or job_norm in cls_norm:
            return True

        return False

    @staticmethod
    def _calc_readiness(
        combo: SkillCombo,
        prep_name: str,
        main_name: str,
        member_jobs: dict[str, str],
    ) -> float:
        """Calculate how ready a combo is (simplified, no position data)."""
        readiness = 0.85  # Base: pretty ready
        if prep_name == main_name:
            readiness -= 0.1  # Self-combo slightly harder to coordinate
        return max(0.0, min(1.0, readiness))

    def _append_action(
        self,
        actions: list[Any],
        command: str,
        confidence: float,
        reason: str,
        kind: str = "command",
    ) -> None:
        """Append an action safely, matching the HeuristicAction pattern."""
        actions.append({
            "kind": kind,
            "command": command,
            "confidence": confidence,
            "domain": "party",
            "reason": f"[Party] {reason}",
        })


# ──────────────────────────────────────────────
#  Global singleton
# ──────────────────────────────────────────────

_engine: PartyEngine | None = None
_engine_lock = RLock()


def get_party_engine() -> PartyEngine:
    """Get or create the global PartyEngine singleton."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = PartyEngine()
        return _engine
