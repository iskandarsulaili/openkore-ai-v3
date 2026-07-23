"""
MVP encounter knowledge system — data-driven per-MVP encounter templates.

Each MVP has a structured encounter template covering:
- Mechanics (AoE patterns, status effects, special skills)
- Gimmicks (knockback immunity, silence protection, DPS checks)
- Positioning strategy (wall at back, range, Line-of-Sight)
- Gear requirements (elemental armor, status immunity)
- Pre-engage checklist (HP threshold, buffs, inventory)
"""

from dataclasses import dataclass, field
from typing import Any
import enum


class MVPSkill(enum.Enum):
    """Known MVP special skills."""
    HELLS_JUDGEMENT = "hells_judgement"  # Full-map AoE (Baphomet)
    SILENCE_ATTACK = "silence_attack"     # Silences target (Mistress)
    KNOCKBACK_ATTACK = "knockback"        # Knocks target back (Orc Hero)
    SELF_HEAL = "self_heal"               # Heals self (Moonlight Flower)
    TELEPORT = "teleport"                 # Teleports away
    STUN_ATTACK = "stun_attack"           # Stuns target
    CURSE_ATTACK = "curse_attack"         # Curses target
    PETRIFY = "petrify"                   # Petrifies target
    RANDOM_TELEPORT = "random_teleport"   # Random teleport on hit
    AREA_SILENCE = "area_silence"         # AoE silence
    AREA_STUN = "area_stun"               # AoE Stun
    EARTHQUAKE = "earthquake"             # AoE damage (Maya)
    FULLMAP_COLD = "fullmap_cold"         # Full-map freeze effect
    SUMMON = "summon"                     # Summons minions
    BERSERK = "berserk"                   # Enrages at low HP
    INVINCIBLE = "invincible"             # Temporary invincibility
    FIRE_BREATH = "fire_breath"           # AoE fire damage
    POISON_MIST = "poison_mist"           # Ground-targeted poison


class EncounterPhase(enum.Enum):
    PRE_ENGAGE = "pre_engage"         # Before engaging — prepare
    ENGAGE = "engage"                 # Initial engagement
    BERSERK = "berserk"               # MVP enraged at low HP
    EMERGENCY = "emergency"           # Emergency flee/retreat
    POST_KILL = "post_kill"           # After MVP killed — loot


@dataclass
class MVPTemplate:
    """Single MVP encounter template."""
    name: str                         # Monster name
    aliases: list[str] = field(default_factory=list)  # Alternative names
    
    # Mechanics
    special_skills: list[MVPSkill] = field(default_factory=list)
    aoE_pattern: str = "none"         # "none", "circle", "fullmap", "line"
    aoE_radius_tiles: int = 0
    aoE_interval_ms: int = 0          # Time between AoE pulses
    status_effects: list[str] = field(default_factory=list)
    
    # Gimmicks
    requires_silence_protection: bool = False
    requires_knockback_immunity: bool = False
    requires_curse_protection: bool = False
    requires_stun_protection: bool = False
    requires_LOS: bool = False        # Line-of-sight needed (pillar kiting)
    has_dps_check: bool = False       # Must out-DPS healing/regeneration
    has_enrage: bool = False          # Becomes more dangerous at low HP
    enrage_hp_pct: float = 0.20      # HP % at which enrage triggers
    spawns_adds: bool = False         # Spawns minions
    add_names: list[str] = field(default_factory=list)
    
    # Positioning
    recommended_range: str = "melee"  # "melee", "ranged", "max_range"
    recommended_formation: str = "standard"  # "standard", "wall", "kite", "los_pillar"
    needs_wall_at_back: bool = False  # Position with wall to avoid knockback
    avoid_walls: bool = False         # Don't get cornered
    
    # Gear
    recommended_armor_element: str = "neutral"
    recommended_weapon_element: str = "neutral"
    immune_to_elements: list[str] = field(default_factory=list)
    weak_to_elements: list[str] = field(default_factory=list)
    minimum_accuracy: int = 0         # Minimum HIT needed
    
    # Pre-engage
    min_hp_pct: float = 0.80         # Minimum HP before engaging
    required_buffs: list[str] = field(default_factory=list)
    required_items: list[str] = field(default_factory=list)
    
    # Phase commands
    engage_command: str = ""
    berserk_command: str = ""
    emergency_command: str = "teleport auto"
    post_kill_command: str = "sit; loot"


# ── Master MVP template registry ──
MVP_TEMPLATES: dict[str, MVPTemplate] = {
    "Baphomet": MVPTemplate(
        name="Baphomet",
        aliases=["Baphomet Jr"],
        special_skills=[MVPSkill.HELLS_JUDGEMENT, MVPSkill.SUMMON],
        aoE_pattern="fullmap",
        aoE_interval_ms=8000,  # Hell's Judgement every ~8s
        status_effects=["curse"],
        requires_curse_protection=True,
        has_dps_check=False,
        has_enrage=True,
        enrage_hp_pct=0.15,
        spawns_adds=True,
        add_names=["Injustice", "Myst"],
        recommended_range="melee",
        recommended_formation="standard",
        recommended_armor_element="neutral",
        recommended_weapon_element="holy",
        immune_to_elements=["dark"],
        weak_to_elements=["holy"],
        min_hp_pct=0.85,
        required_buffs=["Increase AGI", "Blessing", "Assumptio"],
        required_items=["Holy Water", "Panacea"],
        engage_command="attack Baphomet",
        berserk_command="teleport auto; sit; heal; buff; attack Baphomet",
        emergency_command="teleport auto",
    ),
    "Mistress": MVPTemplate(
        name="Mistress",
        aliases=["Mistress"],
        special_skills=[MVPSkill.SILENCE_ATTACK, MVPSkill.AREA_SILENCE, MVPSkill.TELEPORT],
        aoE_pattern="none",
        status_effects=["silence"],
        requires_silence_protection=True,
        has_dps_check=False,
        has_enrage=False,
        spawns_adds=False,
        recommended_range="ranged",
        recommended_formation="kite",
        recommended_armor_element="wind",
        recommended_weapon_element="fire",
        immune_to_elements=["wind"],
        weak_to_elements=["fire"],
        min_hp_pct=0.80,
        required_buffs=["Increase AGI", "Blessing"],
        required_items=["Green Potion", "Panacea"],
        engage_command="attack Mistress",
        emergency_command="teleport auto",
    ),
    "Orc Hero": MVPTemplate(
        name="Orc Hero",
        aliases=["Orc Hero"],
        special_skills=[MVPSkill.KNOCKBACK_ATTACK, MVPSkill.STUN_ATTACK, MVPSkill.BERSERK],
        aoE_pattern="none",
        status_effects=["stun"],
        requires_knockback_immunity=True,
        requires_stun_protection=True,
        has_enrage=True,
        enrage_hp_pct=0.25,
        spawns_adds=False,
        recommended_range="melee",
        recommended_formation="wall",
        needs_wall_at_back=True,
        recommended_armor_element="earth",
        recommended_weapon_element="water",
        immune_to_elements=["earth"],
        weak_to_elements=["water"],
        min_hp_pct=0.85,
        required_buffs=["Increase AGI", "Blessing"],
        required_items=["Panacea"],
        engage_command="attack Orc Hero",
        berserk_command="teleport auto; heal; rebuff; attack Orc Hero",
        emergency_command="teleport auto",
    ),
    "Moonlight Flower": MVPTemplate(
        name="Moonlight Flower",
        aliases=["Moonlight"],
        special_skills=[MVPSkill.SELF_HEAL, MVPSkill.TELEPORT, MVPSkill.SUMMON],
        aoE_pattern="none",
        status_effects=[],
        has_dps_check=True,  # Must out-DPS her healing
        spawns_adds=True,
        add_names=["Spring Rabbit"],
        recommended_range="melee",
        recommended_formation="standard",
        recommended_armor_element="neutral",
        recommended_weapon_element="holy",
        immune_to_elements=["dark", "ghost"],
        weak_to_elements=["holy"],
        min_hp_pct=0.90,
        required_buffs=["Increase AGI", "Blessing", "Assumptio"],
        required_items=["Convex Mirror"],
        engage_command="attack Moonlight",
        emergency_command="teleport auto",
    ),
    "Phreeoni": MVPTemplate(
        name="Phreeoni",
        aliases=["Phreeoni"],
        special_skills=[MVPSkill.STUN_ATTACK, MVPSkill.KNOCKBACK_ATTACK],
        aoE_pattern="none",
        status_effects=["stun"],
        requires_stun_protection=True,
        recommended_range="ranged",
        recommended_formation="kite",
        avoid_walls=True,
        recommended_armor_element="neutral",
        recommended_weapon_element="wind",
        immune_to_elements=["neutral"],  # Physical immune
        weak_to_elements=["wind"],
        min_hp_pct=0.85,
        required_buffs=["Increase AGI", "Blessing"],
        required_items=["Panacea"],
        minimum_accuracy=200,
        engage_command="attack Phreeoni",
        emergency_command="teleport auto",
    ),
    "Drake": MVPTemplate(
        name="Drake",
        aliases=["Drake"],
        special_skills=[MVPSkill.KNOCKBACK_ATTACK, MVPSkill.FIRE_BREATH, MVPSkill.RANDOM_TELEPORT],
        aoE_pattern="circle",
        aoE_radius_tiles=5,
        status_effects=["burn"],
        requires_knockback_immunity=True,
        recommended_range="melee",
        recommended_formation="wall",
        needs_wall_at_back=True,
        recommended_armor_element="fire",
        recommended_weapon_element="wind",
        immune_to_elements=["fire"],
        weak_to_elements=["wind"],
        min_hp_pct=0.80,
        required_buffs=["Increase AGI", "Blessing"],
        required_items=["Fire Armor"],
        engage_command="attack Drake",
        emergency_command="teleport auto",
    ),
    "Osiris": MVPTemplate(
        name="Osiris",
        aliases=["Osiris"],
        special_skills=[MVPSkill.CURSE_ATTACK, MVPSkill.SUMMON, MVPSkill.SELF_HEAL],
        aoE_pattern="none",
        status_effects=["curse"],
        requires_curse_protection=True,
        has_enrage=True,
        enrage_hp_pct=0.20,
        spawns_adds=True,
        add_names=["Mummy", "Mummy"],
        recommended_range="melee",
        recommended_formation="standard",
        recommended_armor_element="undead",
        recommended_weapon_element="holy",
        immune_to_elements=["undead", "dark"],
        weak_to_elements=["holy", "fire"],
        min_hp_pct=0.85,
        required_buffs=["Increase AGI", "Blessing", "Assumptio"],
        required_items=["Holy Water", "Panacea", "Convex Mirror"],
        engage_command="attack Osiris",
        berserk_command="teleport auto; heal; rebuff; attack Osiris",
        emergency_command="teleport auto",
    ),
    "Maya": MVPTemplate(
        name="Maya",
        aliases=["Maya"],
        special_skills=[MVPSkill.EARTHQUAKE, MVPSkill.STUN_ATTACK, MVPSkill.BERSERK],
        aoE_pattern="circle",
        aoE_radius_tiles=7,
        aoE_interval_ms=5000,
        status_effects=["stun"],
        requires_stun_protection=True,
        has_enrage=True,
        enrage_hp_pct=0.20,
        spawns_adds=False,
        recommended_range="ranged",
        recommended_formation="kite",
        avoid_walls=True,
        recommended_armor_element="earth",
        recommended_weapon_element="holy",
        immune_to_elements=["earth"],
        weak_to_elements=["holy", "fire"],
        min_hp_pct=0.90,
        required_buffs=["Increase AGI", "Blessing", "Assumptio"],
        required_items=["Panacea"],
        minimum_accuracy=250,
        engage_command="attack Maya",
        berserk_command="teleport auto; heal; rebuff; ranged attack Maya",
        emergency_command="teleport auto",
    ),
}


def get_mvp_template(name: str) -> MVPTemplate | None:
    """Look up MVP template by name (exact or alias)."""
    name_lower = name.lower().replace("_", " ").strip()
    for mvp_name, template in MVP_TEMPLATES.items():
        if mvp_name.lower() == name_lower:
            return template
        for alias in template.aliases:
            if alias.lower() == name_lower:
                return template
    return None


def get_encounter_checklist(template: MVPTemplate) -> list[str]:
    """Get a pre-engage checklist for an MVP encounter."""
    checklist = []
    checklist.append(f"HP > {template.min_hp_pct:.0%}")
    if template.minimum_accuracy > 0:
        checklist.append(f"HIT > {template.minimum_accuracy}")
    for buff in template.required_buffs:
        checklist.append(f"Buff: {buff}")
    for item in template.required_items:
        checklist.append(f"Item: {item}")
    if template.needs_wall_at_back:
        checklist.append("Position: wall at back")
    if template.recommended_range == "ranged":
        checklist.append("Position: maintain range")
    if template.requires_silence_protection:
        checklist.append("Status: silence protection ON")
    if template.requires_knockback_immunity:
        checklist.append("Status: knockback immunity ON")
    if template.requires_stun_protection:
        checklist.append("Status: stun protection ON")
    return checklist


def assess_engagement_safety(
    template: MVPTemplate,
    hp_pct: float,
    has_buffs: list[str],
    has_items: list[str],
    current_hit: int,
) -> tuple[bool, list[str]]:
    """Assess whether it's safe to engage this MVP. Returns (safe, reasons)."""
    reasons = []
    safe = True
    if hp_pct < template.min_hp_pct:
        reasons.append(f"HP too low: {hp_pct:.0%} < {template.min_hp_pct:.0%}")
        safe = False
    if template.minimum_accuracy > 0 and current_hit < template.minimum_accuracy:
        reasons.append(f"HIT too low: {current_hit} < {template.minimum_accuracy}")
        safe = False
    for buff in template.required_buffs:
        if buff not in has_buffs:
            reasons.append(f"Missing buff: {buff}")
            safe = False
    for item in template.required_items:
        if item not in has_items:
            reasons.append(f"Missing item: {item}")
            safe = False
    return safe, reasons


def get_phase_command(
    template: MVPTemplate,
    phase: EncounterPhase,
    mvp_hp_pct: float,
) -> str:
    """Get the appropriate command for the current encounter phase."""
    if phase == EncounterPhase.BERSERK and template.has_enrage and mvp_hp_pct < template.enrage_hp_pct:
        return template.berserk_command
    if phase == EncounterPhase.EMERGENCY:
        return template.emergency_command
    if phase == EncounterPhase.PRE_ENGAGE:
        return "buff; check_prepare"
    if phase == EncounterPhase.ENGAGE:
        return template.engage_command or "attack"
    if phase == EncounterPhase.POST_KILL:
        return template.post_kill_command or "sit; loot"
    return "attack"
