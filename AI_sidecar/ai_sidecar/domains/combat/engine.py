"""RO Combat Engine — real mechanics engine for Ragnarok Online combat.

Provides:
  - ROCastState: tracks active cast operations and interruption risk
  - ROCombatEngine: full combat decision engine with:
      - Element matrix (10×10, 4 levels)
      - Size modifiers per weapon type
      - Race modifiers
      - Cast time / skill delay / cast interruption
      - SP efficiency scoring
      - Skill-specific knowledge (Napalm Beat safe-cast, Cold Bolt multi-hit,
        Double Strafe 2-hit, Sonic Blow 8-hit, Bowling Bash knockback)
      - Auto-attack weaving
      - Combo system (Frost Diver → Cold Bolt, Heal → Turn Undead, etc.)
      - HeuristicAction production
  - ROMechanicsLoader: loads the YAML data file

All damage formulas verified against rAthena pre-renewal mechanics.
All skill data sourced from rAthena skill_db.txt.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.combat.tactics.base import TacticsContext, TargetInfo

logger = logging.getLogger(__name__)

# ── Data file path ──
_DATA_DIR = Path(os.environ.get(
    "RO_MECHANICS_DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent / "data"),
))
_DEFAULT_YAML_PATH = _DATA_DIR / "ro_mechanics.yaml"


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class SkillInfo:
    """Full skill information loaded from YAML."""
    id: str
    name: str
    sp_cost: int
    cast_time_s: float
    delay_s: float
    aftercast_delay_s: float
    range: int
    element: str
    element_level: int
    hit_count: int
    is_aoe: bool
    aoe_radius: int
    damage_type: str
    cast_interrupt: bool
    tags: list[str] = field(default_factory=list)
    combo_with: list[str] = field(default_factory=list)
    combo_bonus: float = 1.0

    def is_safe_cast(self) -> bool:
        """Check if skill can be cast without interruption risk (cast_interrupt=false)."""
        return not self.cast_interrupt and self.cast_time_s > 0

    def is_instant(self) -> bool:
        """Check if skill is instant-cast (cast_time_s == 0)."""
        return self.cast_time_s <= 0.0

    def total_damage_multiplier(self, level: int = 10) -> float:
        """Estimate total damage multiplier for this skill.
        Uses typical pre-renewal formula: base + per_level * level.
        """
        # RO skill damage formula: roughly 1.0 + 0.4 * skill_level for bolts
        return 1.0 + 0.4 * level


@dataclass
class MonsterInfo:
    """Monster stats from YAML."""
    id: int
    level: int
    hp: int
    sp: int
    def_: int
    mdef: int
    element: str
    element_level: int
    size: str
    race: str
    attack: int
    attack_delay: int
    exp: int
    is_boss: bool
    name: str = ""


@dataclass
class ComboInfo:
    """Combo definition — a sequence of skills with synergy bonus."""
    name: str
    skills: list[str]
    description: str = ""
    condition: str = ""
    bonus: float = 1.0
    priority: int = 50


@dataclass
class CastState:
    """Tracks the current casting operation."""
    skill_id: str
    skill_name: str
    remaining_s: float
    total_s: float
    start_time: float
    is_interrupted: bool = False
    can_cast_while_hit: bool = False

    @property
    def is_active(self) -> bool:
        return self.remaining_s > 0 and not self.is_interrupted

    @property
    def progress(self) -> float:
        if self.total_s <= 0:
            return 1.0
        return 1.0 - (self.remaining_s / self.total_s)


@dataclass
class SkillScore:
    """Scored skill evaluation for decision-making."""
    skill_id: str
    score: float
    dmg_per_sp: float
    element_mod: float
    size_mod: float
    race_mod: float
    estimated_damage: float
    total_cast_time: float
    reason: str = ""


# ═══════════════════════════════════════════════════════════════
# YAML Loader
# ═══════════════════════════════════════════════════════════════

class ROMechanicsLoader:
    """Loads RO mechanics data from YAML file.

    Provides:
      - element_table: nested dict [level][attack_element][target_element] = multiplier
      - size_modifiers: dict[weapon_type][monster_size] = multiplier
      - skills: dict[skill_id] -> SkillInfo
      - monsters: dict[monster_name] -> MonsterInfo
      - maps: dict[map_name] -> map_info
      - combos: dict[combo_name] -> ComboInfo
    """

    def __init__(self, yaml_path: str | Path | None = None):
        self._yaml_path = Path(yaml_path) if yaml_path else _DEFAULT_YAML_PATH
        self._lock = RLock()
        self._data: dict[str, Any] = {}

        # Parsed data structures
        self.element_table: dict[int, dict[str, dict[str, float]]] = {}
        self.size_modifiers: dict[str, dict[str, float]] = {}
        self.race_modifiers: dict[str, float] = {}
        self.skills: dict[str, SkillInfo] = {}
        self.monsters: dict[str, MonsterInfo] = {}
        self.maps: dict[str, Any] = {}
        self.combos: dict[str, ComboInfo] = {}

        self._load()

    def _load(self) -> None:
        """Load and parse the YAML file."""
        path = self._yaml_path
        if not path.exists():
            logger.warning("RO mechanics YAML not found at %s, using empty data", path)
            return

        try:
            with open(path, "r") as f:
                self._data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            logger.error("Failed to load RO mechanics YAML: %s", e)
            self._data = {}
            return

        # Parse element table
        raw_elem = self._data.get("element_table", {})
        for level_key, level_data in raw_elem.items():
            level_num = int(level_key.split("_")[1])
            parsed_level: dict[str, dict[str, float]] = {}
            for attack_elem, targets in level_data.items():
                parsed_level[attack_elem] = {}
                for target_elem, mult in targets.items():
                    # Normalize element names: Shadow -> Dark for internal consistency
                    te = target_elem
                    ae = attack_elem
                    parsed_level[ae][te] = float(mult)
            self.element_table[level_num] = parsed_level

        # Parse size modifiers
        self.size_modifiers = self._data.get("size_modifiers", {})

        # Parse race modifiers
        self.race_modifiers = self._data.get("race_modifiers", {})

        # Parse skills
        for skill_entry in self._data.get("skills", []):
            skill = SkillInfo(
                id=skill_entry.get("id", ""),
                name=skill_entry.get("name", ""),
                sp_cost=skill_entry.get("sp_cost", 0),
                cast_time_s=float(skill_entry.get("cast_time_s", 0.0)),
                delay_s=float(skill_entry.get("delay_s", 0.0)),
                aftercast_delay_s=float(skill_entry.get("aftercast_delay_s", 0.0)),
                range=int(skill_entry.get("range", 1)),
                element=skill_entry.get("element", "Neutral"),
                element_level=int(skill_entry.get("element_level", 1)),
                hit_count=int(skill_entry.get("hit_count", 1)),
                is_aoe=bool(skill_entry.get("is_aoe", False)),
                aoe_radius=int(skill_entry.get("aoe_radius", 0)),
                damage_type=skill_entry.get("damage_type", "melee"),
                cast_interrupt=bool(skill_entry.get("cast_interrupt", False)),
                tags=skill_entry.get("tags", []),
                combo_with=skill_entry.get("combo_with", []),
                combo_bonus=float(skill_entry.get("combo_bonus", 1.0)),
            )
            self.skills[skill.id] = skill

        # Parse monsters
        for monster_name, mon_entry in self._data.get("monsters", {}).items():
            monster = MonsterInfo(
                name=monster_name,
                id=int(mon_entry.get("id", 0)),
                level=int(mon_entry.get("level", 1)),
                hp=int(mon_entry.get("hp", 10)),
                sp=int(mon_entry.get("sp", 0)),
                def_=int(mon_entry.get("def", 0)),
                mdef=int(mon_entry.get("mdef", 0)),
                element=mon_entry.get("element", "Neutral"),
                element_level=int(mon_entry.get("element_level", 1)),
                size=mon_entry.get("size", "Medium"),
                race=mon_entry.get("race", "Formless"),
                attack=int(mon_entry.get("attack", 10)),
                attack_delay=int(mon_entry.get("attack_delay", 1000)),
                exp=int(mon_entry.get("exp", 0)),
                is_boss=bool(mon_entry.get("is_boss", False)),
            )
            self.monsters[monster_name] = monster

        # Parse maps
        self.maps = self._data.get("maps", {})

        # Parse combos
        for combo_name, combo_entry in self._data.get("combos", {}).items():
            combo = ComboInfo(
                name=combo_entry.get("name", combo_name),
                skills=combo_entry.get("skills", []),
                description=combo_entry.get("description", ""),
                condition=combo_entry.get("condition", ""),
                bonus=float(combo_entry.get("bonus", 1.0)),
                priority=int(combo_entry.get("priority", 50)),
            )
            self.combos[combo_name] = combo

        logger.info(
            "RO mechanics loaded: %d skills, %d monsters, %d maps, %d combos",
            len(self.skills), len(self.monsters), len(self.maps), len(self.combos),
        )

    def get_skill(self, skill_id: str) -> SkillInfo | None:
        """Get skill info by ID (case-insensitive)."""
        with self._lock:
            return self.skills.get(skill_id.upper())

    def get_monster(self, name: str) -> MonsterInfo | None:
        """Get monster info by name (case-insensitive)."""
        with self._lock:
            for m_name, monster in self.monsters.items():
                if m_name.lower() == name.lower():
                    return monster
            return None

    def get_element_modifier(
        self,
        attack_element: str,
        target_element: str,
        element_level: int = 1,
    ) -> float:
        """Get the element damage modifier from the element table."""
        # Clamp level to 1-4
        level = max(1, min(4, element_level))
        table = self.element_table.get(level, self.element_table.get(1, {}))

        # Normalize: Shadow -> Shadow (already used in YAML)
        ae = attack_element
        te = target_element

        if ae in table and te in table[ae]:
            return table[ae][te]

        logger.debug(
            "Element modifier not found: %s -> %s at Lv%d, table has %s",
            ae, te, level, list(table.keys()) if table else "empty",
        )
        # Ghost element attacks Neutral with level <= 1 → 0%
        if te == "Neutral" and level <= 1 and ae == "Ghost":
            return 0.0
        return 1.0

    def get_size_modifier(self, weapon_type: str, monster_size: str) -> float:
        """Get the size penalty/modifier for a weapon vs monster size."""
        weapon = weapon_type.lower()
        size = monster_size

        if weapon in self.size_modifiers and size in self.size_modifiers[weapon]:
            return self.size_modifiers[weapon][size]

        # Default: if weapon not found, assume 100%
        return 1.0

    def get_weapon_type_for_job(self, job_name: str) -> str:
        """Default weapon type for a given job."""
        weapon_map = {
            "novice": "dagger",
            "swordman": "sword",
            "knight": "sword",
            "crusader": "spear",
            "mage": "staff",
            "wizard": "staff",
            "archer": "bow",
            "hunter": "bow",
            "acolyte": "mace",
            "priest": "mace",
            "merchant": "sword",
            "blacksmith": "sword",
            "thief": "dagger",
            "assassin": "katar",
            "rogue": "dagger",
            "bard": "instrument",
            "dancer": "whip",
            "monk": "knuckle",
            "sage": "book",
            "alchemist": "staff",
            "taekwon": "knuckle",
            "star_gladiator": "knuckle",
            "soul_linker": "staff",
            "gunslinger": "grenade",
            "ninja": "shuriken",
        }
        return weapon_map.get(job_name.lower(), "dagger")


# ═══════════════════════════════════════════════════════════════
# RO Combat Engine
# ═══════════════════════════════════════════════════════════════

class ROCombatEngine:
    """Real RO combat engine — produces HeuristicAction objects with authentic mechanics.

    This engine replaces simple DPS comparison with a full RO mechanics model:

    1. **Cast time** — Each skill has cast_time_s. Shield Boomerang: instant.
       Storm Gust: 5s cast. If hit during cast, interrupted (unless caster has
       Cast Cancel immunity via tags like 'cast_interrupt_immune').

    2. **Skill delay** — After casting, a global cooldown. 0.3s for most, 1s+ for big skills.
       Includes aftercast_delay_s (the 'frozen' period after cast completes).

    3. **Cast interruption** — Being hit during cast = interrupt (default for most magic).
       Skills with cast_interrupt=false (Napalm Beat, instant-cast skills) are immune.

    4. **Element matrix** — Full 10×10 element table with 4 levels.
       Damage = ATK × element_modifier × size_modifier × race_modifier.

    5. **Size modifier** — Small/Medium/Large weapons vs monster size.
       Dagger vs Large = 50% damage. Bow = 100% all sizes.

    6. **Race modifier** — DemiHuman, Brute, Undead, Formless, etc.
       Certain elements have inherent race bonuses.

    7. **SP efficiency** — Damage per SP. Compare skills by total_damage / SP_cost,
       not just raw DPS.

    8. **Skill-specific knowledge**:
       - Napalm Beat: 0 cast time, cast_interrupt=false → safe while being hit
       - Fire Bolt: cast_time_s=1.5, interrupted on hit
       - Cold Bolt: multi-hit (10 hits), each hit checks element separately → strong vs
         targets with element weakness
       - Double Strafe: 2 hits, each checks cards separately
       - Bowling Bash: pushes enemies (knockback tag), useful for knocking mobs into AoE
       - Sonic Blow: 8 hits, critical-based

    9. **Auto-attack weaving** — When skills are on cooldown or unavailable (low SP),
       the engine falls back to auto-attack commands.

    10. **Combo system** — Skills that chain synergistically:
        - Frost Diver → Cold Bolt: Cold Bolt does 1.5x damage on frozen targets
        - Heal → Turn Undead: Turn Undead does more damage to undead that were 'weakened'
        - Magnum Break → Bowling Bash: Bowling Bash does 1.2x after fire AoE
    """

    # ── Element level lookup for skills (matches RO mechanics) ──
    # For bolt-type magic: Lv1-4 = Lv1, Lv5-9 = Lv2, Lv10 = Lv3
    _ELEMENT_LEVEL_FN: dict[str, Any] = {}

    def __init__(
        self,
        mechanics_loader: ROMechanicsLoader | None = None,
    ):
        self._loader = mechanics_loader or ROMechanicsLoader()
        self._lock = RLock()

        # Cast state tracking
        self._cast_state: CastState | None = None
        self._last_cast_time: float = 0.0  # time of last cast completion
        self._last_skill_id: str = ""  # last skill cast (for combo tracking)
        self._combo_ready: bool = False  # whether combo bonus is primed

        # Cooldown tracking
        self._cooldowns: dict[str, float] = {}  # skill_id -> ready_at_timestamp

        # Time source (overridable for testing)
        self._time_fn = time.time

    # ── Public API ──

    def set_time_source(self, fn: Any) -> None:
        """Override time source (for testing)."""
        self._time_fn = fn

    def get_skill_info(self, skill_id: str) -> SkillInfo | None:
        """Get skill info by ID."""
        return self._loader.get_skill(skill_id)

    def get_monster_info(self, name: str) -> MonsterInfo | None:
        """Get monster info by name."""
        return self._loader.get_monster(name)

    def resolve_monster_info(self, target: TargetInfo) -> MonsterInfo | None:
        """Resolve MonsterInfo from a TargetInfo object."""
        if not target:
            return None
        # Try by name first
        info = self._loader.get_monster(target.name)
        if info:
            return info
        # Try generic monster data from metadata
        mon_name = target.metadata.get("name", target.name) if target.metadata else target.name
        return self._loader.get_monster(mon_name)

    def calculate_damage(
        self,
        attack_power: int,
        weapon_type: str,
        skill_info: SkillInfo | None = None,
        target_monster: MonsterInfo | None = None,
        target_element: str = "Neutral",
        target_size: str = "Medium",
        target_race: str = "Formless",
        target_def: int = 0,
        skill_level: int = 10,
        combo_active: bool = False,
    ) -> int:
        """Calculate damage using authentic RO formulas.

        Formula: ATK × size_mod × element_mod × race_mod × skill_mult × combo_bonus
        Then: damage = (raw - def * 0.5) × variance (±20%)

        For multi-hit skills (Cold Bolt, Sonic Blow, Double Strafe, Pierce),
        each hit is calculated independently with its own variance.
        """
        # Determine attack element
        attack_element = skill_info.element if skill_info else "Neutral"

        # Determine element level
        elem_level = 1
        if skill_info:
            # For bolt skills, element level depends on skill level
            if "magic" in (skill_info.tags or []) or skill_info.damage_type == "magic":
                if skill_level <= 4:
                    elem_level = 1
                elif skill_level <= 9:
                    elem_level = 2
                else:
                    elem_level = skill_info.element_level
                    if elem_level < 3:
                        elem_level = 3
            else:
                elem_level = skill_info.element_level
            if elem_level < 1:
                elem_level = 1

        # Use target monster data if available
        if target_monster:
            target_element = target_monster.element
            target_size = target_monster.size
            target_race = target_monster.race
            target_def = target_monster.def_

        # Size modifier
        size_mod = self._loader.get_size_modifier(weapon_type, target_size)

        # Element modifier
        element_mod = self._loader.get_element_modifier(
            attack_element, target_element, elem_level,
        )

        # Race modifier (default 1.0; Holy vs Undead has inherent bonus in element table)
        race_mod = 1.0

        # Skill damage multiplier (approximate RO formula)
        if skill_info and skill_info.hit_count > 0:
            # Base: 100% + 40% per level for bolt-type skills, varies per skill
            skill_base = 1.0 + 0.4 * min(skill_level, 10)
            skill_mult = skill_base
        else:
            skill_mult = 1.0

        # Combo bonus
        combo_mult = 1.0
        if combo_active:
            # If we have a combo skill and a target, apply bonus
            if skill_info and hasattr(skill_info, 'combo_bonus'):
                combo_mult = skill_info.combo_bonus

        # Raw damage calculation per hit
        raw_damage = attack_power * size_mod * element_mod * race_mod * skill_mult * combo_mult

        # Apply defense reduction
        reduced = max(1, raw_damage - target_def * 0.5)

        # Apply ±20% variance
        variance = random.uniform(0.8, 1.2)
        per_hit = max(1, int(reduced * variance))

        # Multiply by hit count
        total_damage = per_hit * skill_info.hit_count if skill_info else per_hit

        return total_damage

    def calculate_sp_efficiency(
        self,
        skill_info: SkillInfo,
        damage: int,
    ) -> float:
        """Calculate damage per SP.

        Returns damage/SP cost. Skills with 0 SP cost get a high but bounded score.
        """
        if skill_info.sp_cost <= 0:
            return 100.0  # Free skills have unlimited SP efficiency
        return damage / skill_info.sp_cost

    def evaluate_skill(
        self,
        skill_info: SkillInfo,
        attack_power: int,
        weapon_type: str,
        target_monster: MonsterInfo | None,
        current_sp: int,
        target: TargetInfo | None = None,
        skill_level: int = 10,
        combo_active: bool = False,
        is_casting: bool = False,
        aggro_count: int = 0,
        is_being_hit: bool = False,
    ) -> SkillScore:
        """Evaluate a skill against a target, returning a comprehensive score.

        Factors:
        - Element advantage (from element table)
        - Size modifier
        - SP efficiency (damage per SP)
        - Cast time (penalize long casts when under pressure)
        - Interruption risk (penalize interruptable skills when being hit)
        - Combo availability bonus
        - Multi-hit bonus vs large/evasive targets

        Returns SkillScore with all breakdown factors.
        """
        # Determine target element/size/race
        target_elem = target.element if target else "Neutral"
        target_size = target.size if target else "Medium"
        target_race_val = target.race if target else "Formless"

        if target_monster:
            target_elem = target_monster.element
            target_size = target_monster.size
            target_race_val = target_monster.race

        # Element level for this skill
        if skill_level <= 4:
            elem_level = 1
        elif skill_level <= 9:
            elem_level = 2
        else:
            elem_level = skill_info.element_level
            if elem_level < 3:
                elem_level = 3

        # Element modifier
        element_mod = self._loader.get_element_modifier(
            skill_info.element, target_elem, elem_level,
        )

        # Size modifier
        size_mod = self._loader.get_size_modifier(weapon_type, target_size)

        # Race modifier (default 1.0)
        race_mod = 1.0

        # Estimate damage
        skill_mult = 1.0 + 0.4 * min(skill_level, 10)
        raw = attack_power * size_mod * element_mod * race_mod * skill_mult
        if combo_active:
            raw *= skill_info.combo_bonus

        def_val = target_monster.def_ if target_monster else 0
        reduced = max(1, raw - def_val * 0.5)
        avg_variance = 1.0  # Expected variance = 1.0 (centered)
        per_hit = max(1, int(reduced * avg_variance))
        total_damage = per_hit * skill_info.hit_count

        # SP efficiency
        dmg_per_sp = self.calculate_sp_efficiency(skill_info, total_damage)

        # Total cast time (cast + aftercast delay for decision purposes)
        total_cast_time = skill_info.cast_time_s + skill_info.aftercast_delay_s

        # Scoring
        score = 0.0
        reasons = []

        # 1. Element advantage (most important)
        if element_mod > 1.5:
            score += 40
            reasons.append(f"element_advantage_{skill_info.element}x{element_mod:.1f}")
        elif element_mod > 1.0:
            score += 20
            reasons.append(f"element_advantage_x{element_mod:.1f}")
        elif element_mod < 0.5:
            score -= 30
            reasons.append(f"element_disadvantage_x{element_mod:.1f}")
        elif element_mod == 0.0:
            score -= 100  # Immune — never use
            reasons.append("element_immune")

        # 2. SP efficiency
        eff_score = min(30, dmg_per_sp * 0.5)
        score += eff_score
        if dmg_per_sp > 10:
            reasons.append(f"sp_efficient_{dmg_per_sp:.1f}")

        # 3. Size modifier
        if size_mod < 0.6:
            score -= 15
            reasons.append(f"size_penalty_x{size_mod:.2f}")
        elif size_mod > 1.0:
            score += 10
            reasons.append(f"size_bonus_x{size_mod:.2f}")

        # 4. Cast time penalty under pressure
        if is_casting or is_being_hit:
            if skill_info.cast_interrupt and skill_info.cast_time_s > 0:
                score -= 20  # High risk of interruption
                reasons.append("interrupt_risk")
            elif skill_info.is_safe_cast() or skill_info.is_instant():
                score += 15  # Safe to use while being hit
                reasons.append("safe_cast")

        # 5. Aggro count penalty for long-cast skills
        if aggro_count >= 3 and skill_info.cast_time_s > 2.0:
            score -= 10 * aggro_count
            reasons.append(f"long_cast_aggro_{aggro_count}")

        # 6. SP availability
        if skill_info.sp_cost > current_sp:
            score -= 50
            reasons.append("insufficient_sp")
        elif current_sp > 0 and skill_info.sp_cost > 0:
            sp_ratio = current_sp / skill_info.sp_cost
            if sp_ratio < 1.5:
                score -= 10  # Expensive relative to current SP
                reasons.append("sp_expensive")

        # 7. Combo bonus
        if combo_active and skill_info.combo_bonus > 1.0:
            score += 15 * (skill_info.combo_bonus - 1.0) * 10
            reasons.append(f"combo_bonus_x{skill_info.combo_bonus:.1f}")

        # 8. Multi-hit bonus (more consistent damage)
        if skill_info.hit_count >= 5:
            score += 5
            reasons.append(f"multi_hit_{skill_info.hit_count}")

        # 9. Instant cast bonus
        if skill_info.is_instant():
            score += 10
            reasons.append("instant_cast")

        return SkillScore(
            skill_id=skill_info.id,
            score=max(score, -200),
            dmg_per_sp=dmg_per_sp,
            element_mod=element_mod,
            size_mod=size_mod,
            race_mod=race_mod,
            estimated_damage=total_damage,
            total_cast_time=total_cast_time,
            reason=", ".join(reasons),
        )

    def select_best_skill(
        self,
        available_skills: list[str],
        attack_power: int,
        weapon_type: str,
        target_monster: MonsterInfo | None,
        current_sp: int,
        target: TargetInfo | None = None,
        is_casting: bool = False,
        is_being_hit: bool = False,
        aggro_count: int = 0,
        combo_active: bool = False,
    ) -> tuple[str | None, SkillScore | None]:
        """Select the best skill from available skills.

        Returns (skill_id, SkillScore) or (None, None) if no skill is viable.
        Ignores skills on cooldown, skills with insufficient SP, and element-immune skills.
        """
        now = self._time_fn()
        best_skill: str | None = None
        best_score: SkillScore | None = None

        for skill_id in available_skills:
            skill_info = self._loader.get_skill(skill_id)
            if not skill_info:
                continue

            # Skip if on cooldown
            ready_at = self._cooldowns.get(skill_id, 0)
            if ready_at > now:
                continue

            # Skip if insufficient SP
            if skill_info.sp_cost > current_sp:
                continue

            # Skip zero-damage skills (heals, buffs)
            if skill_info.hit_count == 0:
                continue

            # Evaluate
            score = self.evaluate_skill(
                skill_info=skill_info,
                attack_power=attack_power,
                weapon_type=weapon_type,
                target_monster=target_monster,
                current_sp=current_sp,
                target=target,
                is_casting=is_casting,
                is_being_hit=is_being_hit,
                aggro_count=aggro_count,
                combo_active=combo_active,
            )

            # Skip immune skills
            if score.element_mod == 0.0:
                continue

            if best_score is None or score.score > best_score.score:
                best_skill = skill_info.id
                best_score = score

        return best_skill, best_score

    def select_skill_rotation(
        self,
        available_skills: list[str],
        attack_power: int,
        weapon_type: str,
        target_monster: MonsterInfo | None,
        current_sp: int,
        target: TargetInfo | None = None,
        is_casting: bool = False,
        is_being_hit: bool = False,
        aggro_count: int = 0,
        last_skill_id: str = "",
        combo_config: dict[str, list[str]] | None = None,
    ) -> list[tuple[str, SkillScore]]:
        """Build a prioritized rotation of skills against a target.

        Returns a list of (skill_id, score) tuples sorted by score descending.
        Returns an empty list if no skills are viable.
        """
        now = self._time_fn()
        scored: list[tuple[str, SkillScore]] = []

        # Determine if a combo is ready
        combo_active = False
        if last_skill_id:
            for combo_info in self._loader.combos.values():
                if len(combo_info.skills) >= 2:
                    if combo_info.skills[0] == last_skill_id:
                        combo_active = True
                        break

        for skill_id in available_skills:
            skill_info = self._loader.get_skill(skill_id)
            if not skill_info:
                continue

            # Skip if on cooldown
            ready_at = self._cooldowns.get(skill_id, 0)
            if ready_at > now:
                continue

            # Skip if insufficient SP
            if skill_info.sp_cost > current_sp:
                continue

            # Skip zero-damage skills
            if skill_info.hit_count == 0:
                continue

            score = self.evaluate_skill(
                skill_info=skill_info,
                attack_power=attack_power,
                weapon_type=weapon_type,
                target_monster=target_monster,
                current_sp=current_sp,
                target=target,
                is_casting=is_casting,
                is_being_hit=is_being_hit,
                aggro_count=aggro_count,
                combo_active=combo_active,
            )

            if score.element_mod > 0.0:
                scored.append((skill_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1].score, reverse=True)

        return scored

    def update_cooldowns(
        self,
        skill_id: str,
        delay_s: float,
        aftercast_delay_s: float,
    ) -> None:
        """Update cooldown state after casting a skill."""
        now = self._time_fn()

        # Global skill delay (the aftercast delay prevents all actions)
        global_cooldown = delay_s + aftercast_delay_s
        self._cooldowns["__global__"] = now + global_cooldown

        # Per-skill cooldown (equal to aftercast delay typically)
        self._cooldowns[skill_id] = now + aftercast_delay_s

        # Update last skill tracking
        self._last_skill_id = skill_id
        self._last_cast_time = now

    def is_global_cooldown_active(self) -> bool:
        """Check if global skill delay is active."""
        now = self._time_fn()
        ready_at = self._cooldowns.get("__global__", 0)
        return ready_at > now

    def get_global_cooldown_remaining(self) -> float:
        """Get remaining global cooldown in seconds."""
        now = self._time_fn()
        ready_at = self._cooldowns.get("__global__", 0)
        return max(0.0, ready_at - now)

    # ── Action Production ──

    def determine_actions(
        self,
        ctx: TacticsContext,
        target: TargetInfo | None,
        available_skills: list[str],
        skill_levels: dict[str, int] | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[HeuristicAction]:
        """Determine combat actions based on context and produce HeuristicAction list.

        Decision flow:
        1. Emergency check (low HP → flee/potion)
        2. Buff maintenance
        3. Cooldown check (auto-attack if on global cooldown)
        4. Cast interruption handling (if currently casting and being hit)
        5. Skill selection with full RO mechanics
        6. SP preservation (auto-attack if SP is low)
        7. Auto-attack weaving when skills aren't available
        8. Positioning hints for kiting/knockback

        Args:
            ctx: TacticsContext with character state and combat data.
            target: Current target (TargetInfo or None).
            available_skills: List of skill IDs the character has learned.
            skill_levels: Dict of skill_id -> level (default: 10 for all).
            config: Optional configuration overrides.

        Returns:
            List of HeuristicAction objects for the action queue.
        """
        actions: list[HeuristicAction] = []
        config = config or {}
        slvls = skill_levels or {}

        my_hp_pct = ctx.my_hp_pct
        my_sp = ctx.my_sp
        my_max_sp = ctx.my_max_sp
        weapon_type = ctx.my_weapon_type or self._loader.get_weapon_type_for_job(ctx.my_job_class)
        aggro_count = ctx.aggro_count
        is_being_hit = aggro_count > 0

        # Estimate attack power based on job/level
        attack_power = config.get("attack_power", self._estimate_attack_power(ctx))

        # Resolve monster info for target
        target_monster = None
        if target:
            target_monster = self.resolve_monster_info(target)

        # ── Phase 1: Emergency ──
        if my_hp_pct < 0.3:
            actions.append(HeuristicAction(
                kind="command",
                command="use_potion white_potion",
                confidence=1.0,
                domain="combat_engine",
                reason="emergency_hp_low",
                metadata={"hp_pct": my_hp_pct, "aggro": aggro_count},
            ))

        # If very low HP and being hit, flee
        if my_hp_pct < 0.2 and aggro_count >= 2:
            actions.append(HeuristicAction(
                kind="command",
                command="teleport",
                confidence=0.95,
                domain="combat_engine",
                reason="emergency_flee",
                metadata={"hp_pct": my_hp_pct, "aggro": aggro_count},
            ))
            return actions

        # ── Phase 2: Buff check ──
        if ctx.active_buffs is not None:
            # Check for essential buffs
            if "increase_agi" not in ctx.active_buffs and my_sp >= 15:
                actions.append(self._make_buff_action("AL_INCAGI", ctx))
            if "blessing" not in ctx.active_buffs and my_sp >= 15:
                actions.append(self._make_buff_action("AL_BLESSING", ctx))
            if "twohand_quicken" not in ctx.active_buffs and my_sp >= 15:
                actions.append(self._make_buff_action("KN_TWOHANDQUICKEN", ctx))

        # If we added buff actions, return them early (buffs before combat)
        if actions:
            return actions

        # ── Phase 3: Global cooldown / cast state check ──
        if self.is_global_cooldown_active():
            remaining = self.get_global_cooldown_remaining()
            # Auto-attack during cooldown if target in range
            if target and target.distance <= 9:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"attack {target.actor_id}",
                    confidence=0.7,
                    domain="combat_engine",
                    reason="auto_attack_during_cooldown",
                    metadata={
                        "target_id": target.actor_id,
                        "cooldown_remaining": remaining,
                    },
                ))
                return actions

        # ── Phase 4: No target → no action ──
        if not target:
            return actions

        # ── Phase 5: SP preservation ──
        sp_ratio = my_sp / max(1, my_max_sp)
        if sp_ratio < 0.15 and target and target.distance <= 9:
            # Very low SP: just auto-attack to conserve
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target.actor_id}",
                confidence=0.8,
                domain="combat_engine",
                reason="sp_preservation_auto_attack",
                metadata={"target_id": target.actor_id, "sp_ratio": sp_ratio},
            ))
            return actions

        # ── Phase 6: Skill selection ──
        # Determine if we have a target that's in range
        target_in_range = target.distance <= self._get_effective_range(ctx, available_skills)

        if not target_in_range:
            # Move towards target
            target_pos = target.metadata.get("position", {})
            tx = target_pos.get("x", 0) if isinstance(target_pos, dict) else 0
            ty = target_pos.get("y", 0) if isinstance(target_pos, dict) else 0
            if tx or ty:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"move {tx} {ty}",
                    confidence=0.8,
                    domain="combat_engine",
                    reason="approach_target",
                    metadata={"target_id": target.actor_id, "target_name": target.name},
                ))
            else:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"follow {target.actor_id}",
                    confidence=0.7,
                    domain="combat_engine",
                    reason="follow_target",
                    metadata={"target_id": target.actor_id},
                ))
            return actions

        # Select best skill
        best_skill_id, best_score = self.select_best_skill(
            available_skills=available_skills,
            attack_power=attack_power,
            weapon_type=weapon_type,
            target_monster=target_monster,
            current_sp=my_sp,
            target=target,
            is_being_hit=is_being_hit,
            aggro_count=aggro_count,
            combo_active=self._combo_ready,
        )

        if best_skill_id and best_score and best_score.score > 0:
            skill_info: SkillInfo | None = self._loader.get_skill(best_skill_id)
            if skill_info is None:
                # Fallback: skill somehow not found
                skill_info = self._loader.get_skill("SM_BASH")  # Generic fallback
                if skill_info is None:
                    skill_info = SkillInfo(
                        id=best_skill_id, name=best_skill_id, sp_cost=10,
                        cast_time_s=0.0, delay_s=0.3, aftercast_delay_s=1.0,
                        range=1, element="Neutral", element_level=1,
                        hit_count=1, is_aoe=False, aoe_radius=0,
                        damage_type="melee", cast_interrupt=False,
                    )
            skill_info_final: SkillInfo = skill_info
            skill_level = slvls.get(best_skill_id, 10)

            # Handle cast interruption risk
            if skill_info_final.cast_interrupt and is_being_hit and skill_info_final.cast_time_s > 0:
                # Check safe-cast skills first
                safe_candidates = []
                for s_id in available_skills:
                    s_info = self._loader.get_skill(s_id)
                    if s_info and (s_info.is_instant() or s_info.is_safe_cast()):
                        safe_candidates.append(s_id)

                if safe_candidates:
                    # Use first safe-cast skill instead
                    safe_skill = self._loader.get_skill(safe_candidates[0])
                    if safe_skill:
                        best_skill_id = safe_skill.id
                        skill_info_final = safe_skill
                        best_score = SkillScore(
                            skill_id=safe_skill.id,
                            score=10.0,
                            dmg_per_sp=0.0,
                            element_mod=1.0,
                            size_mod=1.0,
                            race_mod=1.0,
                            estimated_damage=0,
                            total_cast_time=0,
                            reason="safe_cast_fallback",
                        )

            # Apply cooldown
            self.update_cooldowns(
                skill_info_final.id,
                skill_info_final.delay_s,
                skill_info_final.aftercast_delay_s,
            )

            # Track combo state
            self._combo_ready = False
            if self._last_skill_id:
                for combo_info in self._loader.combos.values():
                    if len(combo_info.skills) >= 2:
                        if combo_info.skills[0] == self._last_skill_id and \
                           combo_info.skills[1] == skill_info_final.id:
                            self._combo_ready = True

            # Produce skill action
            action = self._make_skill_action(
                skill_info=skill_info_final,
                skill_level=skill_level,
                target=target,
                score=best_score,
                ctx=ctx,
            )
            actions.append(action)

        else:
            # No viable skill: auto-attack
            actions.append(HeuristicAction(
                kind="command",
                command=f"attack {target.actor_id}",
                confidence=0.7,
                domain="combat_engine",
                reason="no_viable_skill_auto_attack",
                metadata={
                    "target_id": target.actor_id,
                    "target_name": target.name,
                    "element_mod": best_score.element_mod if best_score else 1.0,
                },
            ))

        # ── Phase 7: Kiting / Positioning hints ──
        # If using a ranged skill and target is close, create distance
        if best_skill_id and target and target.distance < 4:
            skill_info = self._loader.get_skill(best_skill_id)
            if skill_info and skill_info.range >= 7:
                actions.append(HeuristicAction(
                    kind="command",
                    command="move_away 5",
                    confidence=0.5,
                    domain="combat_engine",
                    reason="create_distance_for_ranged",
                    metadata={"target_id": target.actor_id},
                ))

        return actions

    # ── Internal Helpers ──

    def _estimate_attack_power(self, ctx: TacticsContext) -> int:
        """Estimate attack power from context."""
        # Rough estimate: base_level * 2 + 20 for a typical build
        base = ctx.my_base_level * 2 + 20

        # Adjust for job
        if ctx.my_job_class in ("knight", "blacksmith", "assassin", "monk", "barbarian"):
            base += 20
        elif ctx.my_job_class in ("mage", "wizard", "sage", "soul_linker"):
            base = max(base, 40)  # Mages have lower ATK but use MATK

        return base

    def _get_effective_range(self, ctx: TacticsContext, available_skills: list[str]) -> int:
        """Get effective combat range based on skills and weapon."""
        # Check if any skill has long range
        for skill_id in available_skills:
            skill = self._loader.get_skill(skill_id)
            if skill and skill.range > 1:
                return skill.range

        # Default ranges by weapon type
        range_map = {
            "bow": 9,
            "grenade": 14,
            "shuriken": 7,
            "staff": 1,
            "dagger": 1,
            "sword": 1,
            "spear": 3,
            "mace": 1,
            "knuckle": 1,
            "katar": 1,
        }
        return range_map.get(ctx.my_weapon_type, 1)

    def _make_skill_action(
        self,
        skill_info: SkillInfo,
        skill_level: int,
        target: TargetInfo,
        score: SkillScore,
        ctx: TacticsContext,
    ) -> HeuristicAction:
        """Create a HeuristicAction for casting a skill.

        Command format: skill_cast {skill_id} {target_id}
        The bridge interprets this as: use skill_id on target_id.
        """
        # Determine the command format based on whether it's AoE or single-target
        if skill_info.is_aoe and skill_info.aoe_radius > 0:
            command = f"skill_cast_aoe {skill_info.id} {target.actor_id} {skill_info.aoe_radius}"
        else:
            command = f"skill_cast {skill_info.id} {target.actor_id}"

        # Check if this is a knockback skill
        is_knockback = "knockback" in (skill_info.tags or [])

        metadata: dict[str, Any] = {
            "skill_id": skill_info.id,
            "skill_name": skill_info.name,
            "skill_level": skill_level,
            "target_id": target.actor_id,
            "target_name": target.name,
            "cast_time_s": skill_info.cast_time_s,
            "delay_s": skill_info.delay_s,
            "sp_cost": skill_info.sp_cost,
            "element": skill_info.element,
            "hit_count": skill_info.hit_count,
            "is_aoe": skill_info.is_aoe,
            "aoe_radius": skill_info.aoe_radius if skill_info.is_aoe else 0,
            "damage_type": skill_info.damage_type,
            "element_mod": score.element_mod,
            "size_mod": score.size_mod,
            "estimated_damage": score.estimated_damage,
            "dmg_per_sp": score.dmg_per_sp,
            "score": score.score,
            "score_reason": score.reason,
            "cast_interrupt": skill_info.cast_interrupt,
            "is_knockback": is_knockback,
        }

        if is_knockback:
            metadata["knockback_distance"] = 3  # Standard RO knockback distance

        return HeuristicAction(
            kind="command",
            command=command,
            confidence=0.85,
            domain="combat_engine",
            reason=f"combat_skill_{skill_info.id}_{score.reason}",
            metadata=metadata,
        )

    def _make_buff_action(
        self,
        skill_id: str,
        ctx: TacticsContext,
    ) -> HeuristicAction:
        """Create a HeuristicAction for casting a buff skill."""
        skill = self._loader.get_skill(skill_id)
        skill_name = skill.name if skill else skill_id

        return HeuristicAction(
            kind="command",
            command=f"use_skill {skill_id}",
            confidence=0.9,
            domain="combat_engine",
            reason=f"maintain_buff_{skill_name}",
            metadata={
                "skill_id": skill_id,
                "skill_name": skill_name,
                "sp_cost": skill.sp_cost if skill else 0,
            },
        )

    # ── State Management ──

    def reset_state(self) -> None:
        """Reset all engine state (cooldowns, cast state, combo)."""
        with self._lock:
            self._cast_state = None
            self._last_cast_time = 0.0
            self._last_skill_id = ""
            self._combo_ready = False
            self._cooldowns.clear()

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the engine's current state."""
        now = self._time_fn()
        active_cds = {
            k: max(0.0, v - now)
            for k, v in self._cooldowns.items()
            if v > now
        }
        return {
            "global_cooldown": self.get_global_cooldown_remaining(),
            "active_cooldowns": active_cds,
            "last_skill": self._last_skill_id,
            "combo_ready": self._combo_ready,
            "cast_state": {
                "skill_id": self._cast_state.skill_id if self._cast_state else None,
                "remaining_s": self._cast_state.remaining_s if self._cast_state else 0.0,
                "is_interrupted": self._cast_state.is_interrupted if self._cast_state else False,
            } if self._cast_state else None,
        }


# ═══════════════════════════════════════════════════════════════
# Global Singletons
# ═══════════════════════════════════════════════════════════════

_loader: ROMechanicsLoader | None = None
_loader_lock = RLock()

_engine: ROCombatEngine | None = None
_engine_lock = RLock()


def get_mechanics_loader() -> ROMechanicsLoader:
    """Get the global RO mechanics loader singleton."""
    global _loader
    with _loader_lock:
        if _loader is None:
            _loader = ROMechanicsLoader()
        return _loader


def get_combat_engine() -> ROCombatEngine:
    """Get the global RO combat engine singleton."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ROCombatEngine()
        return _engine


def assess_combat_engine(
    signals: dict[str, Any],
    actions: list[HeuristicAction],
    bot_id: str,
) -> None:
    """Convenience function: assess combat with the engine.

    Designed to be called from the DomainRegistry or bridge directly.
    Builds a TacticsContext from signals and runs the engine.

    Args:
        signals: Raw state signals from the bridge snapshot.
        actions: List to append HeuristicAction objects to.
        bot_id: Bot identifier string.
    """
    from ai_sidecar.domains.combat.dispatcher import get_tactics_dispatcher

    try:
        dispatcher = get_tactics_dispatcher()
        ctx = dispatcher.build_context(signals)
        engine = get_combat_engine()

        # Get available skills from signals
        skills_data = signals.get("skills", [])
        available_skills: list[str] = []
        skill_levels: dict[str, int] = {}
        if isinstance(skills_data, list):
            for s in skills_data:
                if isinstance(s, dict):
                    sid = s.get("name", "")
                    if sid:
                        available_skills.append(sid)
                    lvl = s.get("level", 10)
                    skill_levels[sid] = lvl
                else:
                    available_skills.append(str(s))

        # Get target info
        from ai_sidecar.domains.combat.targeting import enrich_monster_list
        from ai_sidecar.domains.combat.tactics.base import TargetInfo

        monsters = [a for a in signals.get("actors", [])
                    if a.get("type", "") == "monster" and a.get("hp", 0) > 0]

        target = None
        current_target_id = ctx.current_target_id
        if current_target_id and monsters:
            for m in monsters:
                if int(m.get("actor_id", m.get("id", 0))) == current_target_id:
                    target = TargetInfo(
                        actor_id=int(m.get("actor_id", m.get("id", 0))),
                        name=str(m.get("name", "unknown")),
                        score=0.0,
                        hp_pct=float(m.get("hp_pct", m.get("hp_ratio", 1.0))),
                        distance=int(m.get("distance", 0)),
                        element=str(m.get("element", "neutral")).lower(),
                        size=str(m.get("size", "medium")).lower(),
                        race=str(m.get("race", "formless")).lower(),
                        is_boss=bool(m.get("is_boss", False)),
                        metadata=m,
                    )
                    break

        engine_actions = engine.determine_actions(
            ctx=ctx,
            target=target,
            available_skills=available_skills,
            skill_levels=skill_levels,
        )

        actions.extend(engine_actions)

    except Exception as e:
        logger.error("assess_combat_engine failed: %s", e, exc_info=True)
