"""
Swarm AI — Tactical Formations, Skill Combos, Combat Timing, Role Discovery
===========================================================================
Complete multi-bot team play system for 3-bot party.

Key capabilities:
- Tactical Formations: V-formation, line, spread, surround, protect-caster
- Skill Combos: Chain skills between bots (Lex Aeterna → Asura, etc.)
- Combat Timing: Sync attacks, heal rotation, aggro management
- Role Discovery: AI reads skills/equipment, assigns optimal roles
- Adaptive Tactics: LLM-driven formation/strategy selection
- Position Orchestrator: Maintains formation during combat
- Swarm Reflex: Instant coordinated responses to threats
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. FORMATIONS
# ═══════════════════════════════════════════════════════════════

class FormationType(StrEnum):
    VANGUARD = "vanguard"        # Tank front, DPS mid, healer back
    WEDGE = "wedge"              # Arrowhead - all push forward
    LINE = "line"                # Side-by-side
    SPREAD = "spread"            # Max distance, avoid AoE
    SURROUND = "surround"        # Circle around target
    PROTECT = "protect"          # Protect caster/healer
    COLUMN = "column"            # Single file
    DIAMOND = "diamond"          # Diamond formation
    FLANK = "flank"              # Flank target from sides
    RETREAT = "retreat"          # Fall back in formation


@dataclass(slots=True)
class FormationSlot:
    """A position in a formation relative to anchor."""
    role: str
    offset_x: int  # Relative to anchor (formation center)
    offset_y: int
    priority: int = 0  # Lower = assigned first


@dataclass(slots=True)
class Formation:
    """A tactical formation definition."""
    type: FormationType
    description: str
    slots: list[FormationSlot]
    min_bots: int = 1
    max_bots: int = 99
    ideal_range: tuple[int, int] = (1, 9)  # Cell range from target
    requires_melee: bool = False
    requires_ranged: bool = False


# Pre-defined formations
FORMATIONS: dict[FormationType, Formation] = {
    FormationType.VANGUARD: Formation(
        type=FormationType.VANGUARD, description="Tank front, DPS mid, support back",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=3, priority=0),
            FormationSlot(role="tank", offset_x=0, offset_y=4, priority=1),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=2, priority=2),
            FormationSlot(role="dps_melee", offset_x=2, offset_y=2, priority=2),
            FormationSlot(role="dps_ranged", offset_x=-4, offset_y=1, priority=3),
            FormationSlot(role="dps_ranged", offset_x=4, offset_y=1, priority=3),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=0, priority=4),
            FormationSlot(role="healer", offset_x=0, offset_y=-2, priority=4),
            FormationSlot(role="support", offset_x=-2, offset_y=-1, priority=5),
            FormationSlot(role="buffer", offset_x=2, offset_y=-1, priority=5),
        ], min_bots=2, requires_melee=True,
    ),
    FormationType.WEDGE: Formation(
        type=FormationType.WEDGE, description="Arrowhead push",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=3, priority=0),
            FormationSlot(role="dps_melee", offset_x=-1, offset_y=2, priority=1),
            FormationSlot(role="dps_melee", offset_x=1, offset_y=2, priority=1),
            FormationSlot(role="dps_ranged", offset_x=-2, offset_y=1, priority=2),
            FormationSlot(role="dps_ranged", offset_x=2, offset_y=1, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=0, priority=3),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=-1, priority=4),
        ], min_bots=2,
    ),
    FormationType.LINE: Formation(
        type=FormationType.LINE, description="Side-by-side advance",
        slots=[
            FormationSlot(role="tank", offset_x=-3, offset_y=3, priority=0),
            FormationSlot(role="dps_melee", offset_x=-1, offset_y=3, priority=1),
            FormationSlot(role="dps_melee", offset_x=1, offset_y=3, priority=1),
            FormationSlot(role="dps_ranged", offset_x=3, offset_y=3, priority=2),
            FormationSlot(role="healer", offset_x=-2, offset_y=1, priority=3),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=1, priority=4),
            FormationSlot(role="support", offset_x=2, offset_y=1, priority=5),
        ], min_bots=2,
    ),
    FormationType.SPREAD: Formation(
        type=FormationType.SPREAD, description="Max distance to avoid AoE",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=5, priority=0),
            FormationSlot(role="dps_ranged", offset_x=-5, offset_y=3, priority=1),
            FormationSlot(role="dps_ranged", offset_x=5, offset_y=3, priority=1),
            FormationSlot(role="dps_magic", offset_x=-3, offset_y=0, priority=2),
            FormationSlot(role="dps_magic", offset_x=3, offset_y=0, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=-3, priority=3),
        ], min_bots=2, requires_ranged=True,
    ),
    FormationType.SURROUND: Formation(
        type=FormationType.SURROUND, description="Circle around target",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=3, priority=0),
            FormationSlot(role="dps_melee", offset_x=3, offset_y=0, priority=1),
            FormationSlot(role="dps_melee", offset_x=-3, offset_y=0, priority=1),
            FormationSlot(role="dps_melee", offset_x=0, offset_y=-3, priority=2),
            FormationSlot(role="dps_ranged", offset_x=4, offset_y=3, priority=3),
            FormationSlot(role="dps_ranged", offset_x=-4, offset_y=-3, priority=3),
        ], min_bots=3,
    ),
    FormationType.PROTECT: Formation(
        type=FormationType.PROTECT, description="Protect caster/healer",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=3, priority=0),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=2, priority=1),
            FormationSlot(role="dps_melee", offset_x=2, offset_y=2, priority=1),
            FormationSlot(role="healer", offset_x=0, offset_y=0, priority=2),
            FormationSlot(role="dps_magic", offset_x=0, offset_y=0, priority=2),
            FormationSlot(role="dps_ranged", offset_x=-3, offset_y=1, priority=3),
            FormationSlot(role="dps_ranged", offset_x=3, offset_y=1, priority=3),
        ], min_bots=2,
    ),
    FormationType.DIAMOND: Formation(
        type=FormationType.DIAMOND, description="Diamond formation",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=4, priority=0),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=2, priority=1),
            FormationSlot(role="dps_melee", offset_x=2, offset_y=2, priority=1),
            FormationSlot(role="healer", offset_x=0, offset_y=0, priority=2),
            FormationSlot(role="dps_ranged", offset_x=-3, offset_y=-1, priority=3),
            FormationSlot(role="dps_ranged", offset_x=3, offset_y=-1, priority=3),
        ], min_bots=3,
    ),
    FormationType.FLANK: Formation(
        type=FormationType.FLANK, description="Flank from sides",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=4, priority=0),
            FormationSlot(role="dps_melee", offset_x=-4, offset_y=2, priority=1),
            FormationSlot(role="dps_melee", offset_x=4, offset_y=2, priority=1),
            FormationSlot(role="dps_ranged", offset_x=-5, offset_y=0, priority=2),
            FormationSlot(role="dps_ranged", offset_x=5, offset_y=0, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=0, priority=3),
        ], min_bots=2,
    ),
    FormationType.RETREAT: Formation(
        type=FormationType.RETREAT, description="Ordered retreat",
        slots=[
            FormationSlot(role="tank", offset_x=0, offset_y=0, priority=0),
            FormationSlot(role="dps_melee", offset_x=-2, offset_y=-1, priority=1),
            FormationSlot(role="dps_melee", offset_x=2, offset_y=-1, priority=1),
            FormationSlot(role="dps_ranged", offset_x=-3, offset_y=-2, priority=2),
            FormationSlot(role="dps_ranged", offset_x=3, offset_y=-2, priority=2),
            FormationSlot(role="healer", offset_x=0, offset_y=-3, priority=3),
        ], min_bots=2,
    ),
}


# ═══════════════════════════════════════════════════════════════
# 2. SKILL COMBOS
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SkillComboStep:
    """A single step in a skill combo chain."""
    role: str
    skill: str
    target: str  # "enemy" | "self" | "teammate:role" | "previous_target"
    delay_ms: int = 0  # Delay after previous step
    condition: str = ""  # Optional condition (e.g., "target_hp > 50%")
    description: str = ""


@dataclass(slots=True)
class SkillCombo:
    """A coordinated skill chain between multiple bots."""
    name: str
    description: str
    steps: list[SkillComboStep]
    min_party_size: int = 2
    cooldown_s: int = 30
    damage_multiplier: float = 1.0
    tags: list[str] = field(default_factory=list)


# Pre-defined skill combos for various party compositions
SKILL_COMBOS: dict[str, SkillCombo] = {
    "lex_asura": SkillCombo(
        name="Lex Aeterna → Asura Strike",
        description="Priest casts Lex Aeterna, then Monk does Asura Strike (double damage)",
        steps=[
            SkillComboStep(role="support", skill="LEX_AETERNA", target="enemy", delay_ms=0, description="Priest casts Lex Aeterna on target"),
            SkillComboStep(role="dps_melee", skill="ASURA_STRIKE", target="enemy", delay_ms=500, description="Monk follows up with Asura Strike"),
        ], min_party_size=2, cooldown_s=60, damage_multiplier=2.0, tags=["boss", "mvp", "burst"],
    ),
    "magnus_freeze": SkillCombo(
        name="Storm Gust → Freeze → Magnus",
        description="Wizard casts Storm Gust to freeze, then Priest casts Magnus Exorcismus",
        steps=[
            SkillComboStep(role="dps_magic", skill="STORM_GUST", target="enemy", delay_ms=0, description="Wizard casts Storm Gust to freeze"),
            SkillComboStep(role="support", skill="MAGNUS_EXORCISMUS", target="enemy", delay_ms=2000, description="Priest Magnus on frozen targets"),
        ], min_party_size=2, cooldown_s=45, damage_multiplier=1.5, tags=["aoe", "undead", "boss"],
    ),
    "safety_wall_asura": SkillCombo(
        name="Safety Wall → Asura",
        description="Priest casts Safety Wall, Monk stands in it and Asuras",
        steps=[
            SkillComboStep(role="support", skill="SAFETY_WALL", target="teammate:dps_melee", delay_ms=0, description="Safety Wall on Monk"),
            SkillComboStep(role="dps_melee", skill="ASURA_STRIKE", target="enemy", delay_ms=1000, description="Monk Asuras from Safety Wall"),
        ], min_party_size=2, cooldown_s=60, tags=["boss", "mvp"],
    ),
    "combo_attack_chain": SkillCombo(
        name="Full Combo Chain",
        description="Full party combo: tank taunt → buffer buff → DPS burst",
        steps=[
            SkillComboStep(role="tank", skill="TAUNT", target="enemy", delay_ms=0, description="Tank taunts for aggro"),
            SkillComboStep(role="buffer", skill="ASSUMPTIO", target="teammate:tank", delay_ms=500, description="Buffer puts Assumptio on tank"),
            SkillComboStep(role="buffer", skill="IMPOSITIO_MANUS", target="teammate:dps_melee", delay_ms=1000, description="Impositio on melee DPS"),
            SkillComboStep(role="buffer", skill="BENEDICTIO", target="self", delay_ms=500, description="Benedictio for party buff"),
            SkillComboStep(role="dps_ranged", skill="ARROW_VULCAN", target="enemy", delay_ms=1500, description="Ranged DPS burst"),
            SkillComboStep(role="dps_melee", skill="ASURA_STRIKE", target="enemy", delay_ms=1000, description="Melee DPS finisher"),
            SkillComboStep(role="dps_magic", skill="JUPITEL_THUNDER", target="enemy", delay_ms=500, description="Magic DPS finisher"),
        ], min_party_size=3, cooldown_s=90, damage_multiplier=2.5, tags=["boss", "mvp", "full_combo"],
    ),
    "heal_rotation": SkillCombo(
        name="Party Heal Rotation",
        description="Coordinated healing to maximize efficiency",
        steps=[
            SkillComboStep(role="healer", skill="HEAL", target="teammate:tank", delay_ms=0, description="Healer heals tank"),
            SkillComboStep(role="support", skill="HEAL", target="teammate:dps_melee", delay_ms=2000, description="Support heals melee DPS"),
            SkillComboStep(role="healer", skill="HEAL", target="teammate:dps_ranged", delay_ms=2000, description="Healer heals ranged DPS"),
        ], min_party_size=2, cooldown_s=10, tags=["heal", "sustain"],
    ),
    "buff_rotation": SkillCombo(
        name="Party Buff Rotation",
        description="Buff party members in optimal order",
        steps=[
            SkillComboStep(role="buffer", skill="BENEDICTIO", target="self", delay_ms=0, description="Party-wide blessing"),
            SkillComboStep(role="buffer", skill="IMPOSITIO_MANUS", target="teammate:dps_melee", delay_ms=500, description="Weapon blessing on melee"),
            SkillComboStep(role="buffer", skill="ASSUMPTIO", target="teammate:tank", delay_ms=500, description="Assumptio on tank"),
            SkillComboStep(role="buffer", skill="GLORIA", target="self", delay_ms=500, description="Gloria for LUK buff"),
        ], min_party_size=2, cooldown_s=120, tags=["buff", "utility"],
    ),
    "gank_combo": SkillCombo(
        name="PVP Gank Combo",
        description="Coordinated PVP gank - stun, debuff, burst",
        steps=[
            SkillComboStep(role="debuff", skill="STUN", target="enemy", delay_ms=0, description="Debuffer stuns target"),
            SkillComboStep(role="debuff", skill="LEX_AETERNA", target="enemy", delay_ms=200, description="Debuffer doubles damage"),
            SkillComboStep(role="dps_melee", skill="ASURA_STRIKE", target="enemy", delay_ms=500, description="Melee burst"),
            SkillComboStep(role="dps_magic", skill="SOUL_DRAIN", target="enemy", delay_ms=500, description="Magic follow-up"),
        ], min_party_size=2, cooldown_s=45, tags=["pvp", "gank"],
    ),
}


# ═══════════════════════════════════════════════════════════════
# 3. ROLE DISCOVERY
# ═══════════════════════════════════════════════════════════════

class RoleDiscoveryEngine:
    """Discovers bot roles by reading skills, equipment, stats from game state.

    No hardcoded class mappings — reads the actual game data and infers roles.
    """

    ROLE_SKILL_KEYWORDS: dict[str, list[str]] = {
        "tank": ["provoke", "taunt", "endure", "defense", "defender", "shield",
                 "reflect", "armor", "guard", "protection"],
        "healer": ["heal", "cure", "recovery", "restore", "sanctuary", "pneuma",
                   "blessing", "high_heal"],
        "dps_melee": ["bash", "bash", "spiral", "spear", "strike", "asura",
                      "combo", "fist", "sonic_blow", "backstab", "riposte"],
        "dps_ranged": ["arrow", "bow", "shoot", "falcon", "trap", "grenade",
                       "bullet", "sniping", "target"],
        "dps_magic": ["bolt", "ball", "storm", "meteor", "tunder", "fire",
                      "frost", "soul", "magic", "wizard", "sage"],
        "support": ["magnus", "turn_undead", "ressurection", "pneuma",
                    "safety_wall", "lex_aeterna", "kyrie"],
        "buffer": ["blessing", "agi_up", "impositio", "assumptio", "benedictio",
                   "gloria", "magnificat", "suffragium"],
        "debuff": ["lex_aeterna", "stun", "curse", "poison", "blind",
                   "silence", "slow", "halt", "freeze"],
        "merchant": ["discount", "overcharge", "cart", "merchant", "pushcart"],
        "crafter": ["smith", "forge", "enchant", "craft", "refine", "repair"],
        "farmer": ["steal", "pick", "loot", "harvest", "gather", "collect"],
    }

    @classmethod
    def discover_roles(cls, snapshot: dict[str, Any] | None) -> list[str]:
        """Discover available roles from game state snapshot."""
        if not snapshot:
            return ["idle"]

        roles = set()
        skills = cls._extract_skills(snapshot)

        # Check skills
        for role, keywords in cls.ROLE_SKILL_KEYWORDS.items():
            for skill_name in skills:
                skill_lower = skill_name.lower().replace("_", " ").replace("-", " ")
                if any(kw in skill_lower for kw in keywords):
                    roles.add(role)
                    break

        # Check equipment
        equipment = cls._extract_equipment(snapshot)
        for equip in equipment:
            equip_lower = equip.lower()
            if any(w in equip_lower for w in ["shield", "armor", "plate", "guard"]):
                roles.add("tank")
            if any(w in equip_lower for w in ["bow", "arrow", "quiver"]):
                roles.add("dps_ranged")
            if any(w in equip_lower for w in ["staff", "rod", "wand", "book"]):
                roles.add("dps_magic")
            if any(w in equip_lower for w in ["knife", "dagger", "katar", "claw"]):
                roles.add("dps_melee")

        # Check stats for role inference
        stats = cls._extract_stats(snapshot)
        if stats:
            # High VIT = tank
            if stats.get("vit", 0) > stats.get("str", 0) + 10:
                roles.add("tank")
            # High INT = magic
            if stats.get("int", 0) > stats.get("str", 0) + 10:
                roles.add("dps_magic")
            # High DEX = ranged
            if stats.get("dex", 0) > stats.get("str", 0) + 10:
                roles.add("dps_ranged")

        if not roles:
            roles.add("idle")

        return list(roles)

    @classmethod
    def _extract_skills(cls, snapshot: dict[str, Any] | None) -> list[str]:
        """Extract skill names from snapshot."""
        if not snapshot:
            return []
        skills = []
        skills_data = snapshot.get("skills") or snapshot.get("known_skills") or []
        if isinstance(skills_data, dict):
            skills = list(skills_data.keys())
        elif isinstance(skills_data, list):
            for s in skills_data:
                if isinstance(s, dict):
                    skills.append(str(s.get("name") or s.get("skill") or ""))
                elif isinstance(s, str):
                    skills.append(s)
        return skills

    @classmethod
    def _extract_equipment(cls, snapshot: dict[str, Any] | None) -> list[str]:
        """Extract equipment names from snapshot."""
        if not snapshot:
            return []
        equip = []
        equip_data = snapshot.get("equipment") or snapshot.get("items_equipped") or []
        if isinstance(equip_data, dict):
            equip = list(equip_data.keys())
        elif isinstance(equip_data, list):
            for e in equip_data:
                if isinstance(e, dict):
                    equip.append(str(e.get("name") or e.get("item") or ""))
                elif isinstance(e, str):
                    equip.append(e)
        return equip

    @classmethod
    def _extract_stats(cls, snapshot: dict[str, Any] | None) -> dict[str, int]:
        """Extract stats from snapshot."""
        if not snapshot:
            return {}
        stats = snapshot.get("stats") or {}
        if isinstance(stats, dict):
            return {k.lower(): int(v) for k, v in stats.items() if isinstance(v, (int, float))}
        return {}


# ═══════════════════════════════════════════════════════════════
# 4. SWARM TACTICS ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SwarmTacticalState:
    """Current tactical state of the swarm."""
    formation: FormationType = FormationType.VANGUARD
    target_id: str = ""
    target_position: tuple[int, int] = (0, 0)
    anchor_position: tuple[int, int] = (0, 0)  # Formation center
    combat_active: bool = False
    threat_level: float = 0.0  # 0.0 = safe, 1.0 = critical
    active_combo: str = ""  # Currently executing combo name
    combo_step: int = 0
    combo_started_at: float = 0.0
    party_hp_avg: float = 1.0
    active_formation_changed_at: float = 0.0
    last_aggro_swap: float = 0.0
    last_heal_rotation: float = 0.0
    bot_positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    bot_roles: dict[str, str] = field(default_factory=dict)
    bot_hp: dict[str, float] = field(default_factory=dict)
    bot_mp: dict[str, float] = field(default_factory=dict)


class SwarmTacticsEngine:
    """The tactical brain of the swarm.

    Selects formations, initiates skill combos, manages aggro,
    and coordinates multi-bot positioning.
    """

    def __init__(self):
        self._lock = RLock()
        self._state: dict[str, SwarmTacticalState] = {}  # party_id -> state
        self._party_roles: dict[str, dict[str, str]] = {}  # party_id -> {bot_id: role}
        self._cooldowns: dict[str, float] = defaultdict(float)
        self._combo_lock: dict[str, bool] = defaultdict(bool)  # party_id -> is executing combo

    def get_or_create_state(self, party_id: str) -> SwarmTacticalState:
        if party_id not in self._state:
            self._state[party_id] = SwarmTacticalState()
        return self._state[party_id]

    def select_formation(self, *, party_id: str, bots: list[dict[str, Any]],
                         target_count: int = 1, threat_level: float = 0.0,
                         aoe_risk: bool = False, team_hp_avg: float = 1.0) -> FormationType:
        """Select the best formation based on situation."""
        state = self.get_or_create_state(party_id)
        roles = [b.get("role", "idle") for b in bots]
        bot_count = len(bots)
        now = time.time()

        # Emergency: retreat if critically low HP
        if team_hp_avg < 0.25:
            return FormationType.RETREAT

        # High threat: protect caster/healer
        if threat_level > 0.8:
            if "tank" in roles and ("healer" in roles or "dps_magic" in roles):
                return FormationType.PROTECT
            return FormationType.VANGUARD

        # AoE risk: spread out
        if aoe_risk and any(r in roles for r in ["dps_ranged", "dps_magic"]):
            return FormationType.SPREAD

        # Multiple targets: surround
        if target_count >= 3 and bot_count >= 3:
            return FormationType.SURROUND

        # Single target, full party: diamond
        if bot_count >= 3 and target_count == 1:
            return FormationType.DIAMOND

        # Default: vanguard
        if any(r in roles for r in ["tank", "dps_melee"]):
            if bot_count >= 2:
                return FormationType.WEDGE
            return FormationType.VANGUARD
        return FormationType.LINE

    def get_formation_positions(self, *, formation: FormationType,
                                 bots: list[dict[str, Any]],
                                 anchor_x: int, anchor_y: int) -> dict[str, tuple[int, int]]:
        """Assign formation positions to bots based on their roles."""
        formation_def = FORMATIONS.get(formation)
        if not formation_def:
            return {}

        # Sort slots by priority
        sorted_slots = sorted(formation_def.slots, key=lambda s: s.priority)

        # Match bots to slots
        assigned: dict[str, tuple[int, int]] = {}
        used_slots: set[int] = set()

        for bot in bots:
            role = bot.get("role", "idle")
            bot_id = bot.get("bot_id", "")
            # Find best slot for this role
            for slot in sorted_slots:
                slot_idx = sorted_slots.index(slot)
                if slot_idx in used_slots:
                    continue
                if slot.role == role or (slot.role.startswith("dps") and role.startswith("dps")):
                    assigned[bot_id] = (anchor_x + slot.offset_x, anchor_y + slot.offset_y)
                    used_slots.add(slot_idx)
                    break
            else:
                # No matching slot — assign to any unassigned slot
                for slot in sorted_slots:
                    slot_idx = sorted_slots.index(slot)
                    if slot_idx not in used_slots:
                        assigned[bot_id] = (anchor_x + slot.offset_x, anchor_y + slot.offset_y)
                        used_slots.add(slot_idx)
                        break

        return assigned

    def select_combo(self, *, party_id: str, bots: list[dict[str, Any]],
                     target_type: str = "normal") -> SkillCombo | None:
        """Select the best skill combo for the current situation."""
        now = time.time()
        roles = [b.get("role", "idle") for b in bots]
        bot_count = len(bots)

        # Check cooldown
        if now - self._cooldowns.get(f"combo_{party_id}", 0) < 30:
            return None

        # Check if combo is locked
        if self._combo_lock[party_id]:
            return None

        # Select combo based on party composition and target type
        candidates = []
        for name, combo in SKILL_COMBOS.items():
            if bot_count < combo.min_party_size:
                continue
            if now - self._cooldowns.get(f"combo_{name}_{party_id}", 0) < combo.cooldown_s:
                continue
            # Check if party has required roles
            required_roles = set(s.role for s in combo.steps)
            available_roles = set(roles)
            if required_roles.issubset(available_roles):
                # Score based on relevance
                score = 1.0
                if target_type == "boss" and "boss" in combo.tags:
                    score += 2.0
                if target_type == "mvp" and "mvp" in combo.tags:
                    score += 3.0
                if target_type == "pvp" and "pvp" in combo.tags:
                    score += 2.0
                if "heal" in combo.tags and any(b.get("hp_pct", 1.0) < 0.5 for b in bots):
                    score += 2.0
                candidates.append((score, name, combo))

        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][2]

    def execute_combo_step(self, *, party_id: str, combo: SkillCombo,
                           step_index: int, bots: dict[str, Any]) -> str | None:
        """Generate the command for the current combo step."""
        if step_index >= len(combo.steps):
            return None

        step = combo.steps[step_index]
        now = time.time()

        # Find the bot for this role
        bot_id = None
        for bid, bdata in bots.items():
            if bdata.get("role") == step.role:
                bot_id = bid
                break

        if not bot_id:
            return None

        # Build command based on target type
        if step.target == "enemy":
            target_id = "target"
            return f"use_skill {step.skill} {target_id}"
        elif step.target == "self":
            return f"use_skill_self {step.skill}"
        elif step.target.startswith("teammate:"):
            target_role = step.target.split(":")[1]
            for bid, bdata in bots.items():
                if bdata.get("role") == target_role:
                    return f"use_skill_teammate {step.skill} {bid}"
            return None
        elif step.target == "previous_target":
            return f"use_skill {step.skill} target"

        # Update state
        state = self.get_or_create_state(party_id)
        state.active_combo = combo.name
        state.combo_step = step_index
        state.combo_started_at = now

        return f"use_skill {step.skill} target"

    def is_combo_complete(self, *, party_id: str, combo: SkillCombo) -> bool:
        """Check if all steps of the combo have been executed."""
        state = self.get_or_create_state(party_id)
        return state.combo_step >= len(combo.steps) - 1

    def complete_combo(self, *, party_id: str, combo: SkillCombo) -> None:
        """Mark combo as complete and set cooldown."""
        now = time.time()
        self._cooldowns[f"combo_{party_id}"] = now
        self._cooldowns[f"combo_{combo.name}_{party_id}"] = now
        self._combo_lock[party_id] = False

        state = self.get_or_create_state(party_id)
        state.active_combo = ""
        state.combo_step = 0

    def manage_aggro(self, *, party_id: str, bots: list[dict[str, Any]],
                     tank_id: str | None) -> str | None:
        """Generate aggro management commands."""
        now = time.time()
        state = self.get_or_create_state(party_id)

        if not tank_id:
            return None

        # Check if tank needs to taunt
        tank = next((b for b in bots if b.get("bot_id") == tank_id), None)
        if tank and tank.get("hp_pct", 1.0) < 0.5:
            return f"use_skill taunt target"

        # Check if any non-tank has aggro
        for bot in bots:
            if bot.get("bot_id") == tank_id:
                continue
            if bot.get("has_aggro", False) and bot.get("hp_pct", 1.0) < 0.6:
                return f"use_skill taunt target"

        return None

    def suggest_heal_target(self, *, party_id: str, bots: list[dict[str, Any]]) -> str | None:
        """Suggest which bot needs healing most."""
        lowest = None
        lowest_hp = 1.0
        for bot in bots:
            hp = bot.get("hp_pct", 1.0)
            if hp < lowest_hp:
                lowest_hp = hp
                lowest = bot.get("bot_id")
        if lowest_hp < 0.7:
            return lowest
        return None

    def suggest_movement(self, *, party_id: str, bot_id: str,
                         formation: FormationType, positions: dict[str, tuple[int, int]]) -> str | None:
        """Generate a movement command to maintain formation position."""
        target_pos = positions.get(bot_id)
        if not target_pos:
            return None
        return f"move {target_pos[0]} {target_pos[1]}"


# ═══════════════════════════════════════════════════════════════
# 5. SWARM REFLEX SYSTEM
# ═══════════════════════════════════════════════════════════════

class SwarmReflexSystem:
    """Instant coordinated responses to threats and opportunities.

    Fires within 100ms of detecting a trigger condition.
    """

    def __init__(self):
        self._lock = RLock()
        self._cooldowns: dict[str, float] = defaultdict(float)

    def assess(self, *, party_id: str, bots: list[dict[str, Any]],
               signals: dict[str, Any]) -> dict[str, Any] | None:
        """Assess if a swarm reflex action is needed."""
        now = time.time()
        results = {}

        # 1. Party member death — immediate response
        dead = [b for b in bots if b.get("is_dead", False)]
        if dead and now - self._cooldowns.get(f"death_{party_id}", 0) > 5:
            self._cooldowns[f"death_{party_id}"] = now
            results["response"] = "party_member_dead"
            results["action"] = "retreat_and_cover"
            results["target"] = dead[0].get("bot_id", "")
            return results

        # 2. MVP spotted — swarm alert
        mvp = signals.get("mvp_spotted", "")
        if mvp and now - self._cooldowns.get(f"mvp_{party_id}", 0) > 30:
            self._cooldowns[f"mvp_{party_id}"] = now
            results["response"] = "mvp_spotted"
            results["action"] = "form_up_for_mvp"
            results["mvp_name"] = mvp
            return results

        # 3. Multiple party members low HP — emergency heal
        low_hp = [b for b in bots if b.get("hp_pct", 1.0) < 0.3]
        if len(low_hp) >= 2 and now - self._cooldowns.get(f"emergency_heal_{party_id}", 0) > 10:
            self._cooldowns[f"emergency_heal_{party_id}"] = now
            results["response"] = "multiple_low_hp"
            results["action"] = "emergency_heal_rotation"
            results["targets"] = [b.get("bot_id", "") for b in low_hp]
            return results

        # 4. Overwhelming force — retreat
        hostiles = signals.get("nearby_hostiles", 0)
        if hostiles > 5 and now - self._cooldowns.get(f"retreat_{party_id}", 0) > 15:
            self._cooldowns[f"retreat_{party_id}"] = now
            results["response"] = "overwhelmed"
            results["action"] = "retreat"
            results["reason"] = f"{hostiles} nearby hostiles"
            return results

        # 5. Party member PVP attacked — counter
        pvp_attack = signals.get("pvp_attacked", "")
        if pvp_attack and now - self._cooldowns.get(f"pvp_counter_{party_id}", 0) > 10:
            self._cooldowns[f"pvp_counter_{party_id}"] = now
            results["response"] = "pvp_attacked"
            results["action"] = "counter_attack"
            results["aggressor"] = pvp_attack
            return results

        return None


# ═══════════════════════════════════════════════════════════════
# 6. SWARM TELEMETRY
# ═══════════════════════════════════════════════════════════════

def swarm_telemetry(state: SwarmTacticalState, party_id: str) -> dict[str, Any]:
    """Produce telemetry snapshot for the swarm."""
    return {
        "party_id": party_id,
        "formation": state.formation.value,
        "combat_active": state.combat_active,
        "threat_level": state.threat_level,
        "active_combo": state.active_combo,
        "combo_step": state.combo_step,
        "party_hp_avg": state.party_hp_avg,
        "bot_count": len(state.bot_positions),
        "bot_roles": dict(state.bot_roles),
        "bot_hp": {k: round(v, 2) for k, v in state.bot_hp.items()},
    }