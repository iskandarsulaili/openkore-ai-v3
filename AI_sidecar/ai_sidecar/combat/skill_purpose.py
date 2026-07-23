"""
Skill purpose classification — classifies skills by combat purpose, not just DPS.

Every RO skill serves a PURPOSE beyond dealing damage:
- ZONING: Controls where monsters can move (Fire Wall, LoV, Quagmire)
- DENIAL: Prevents monster actions (Safety Wall, Cyclone, Lex Aeterna)
- SETUP: Enables follow-up damage (Cold Bolt for wet status, Provoke for def down)
- CLEANUP: Finishes low-HP mobs efficiently (Napalm Beat, normal attack)
- SURVIVAL: Keeps player alive (Heal, Increase AGI, Blessing, Assumptio)
- MOBILITY: Changes position (Teleport, Fly Wing, Pneuma)
- SUSTAIN: Manages resources (Soul Drain, Energy Coat)
- BURST: High damage in short window (Storm Gust, Lord of Vermilion)
- DOT: Damage over time (Poison, Fire Pillar)
- UTILITY: Everything else (Detect, Sight, NPC interaction)
"""

from enum import Enum
from dataclasses import dataclass, field


class SkillPurpose(Enum):
    ZONING = "zoning"           # Controls monster movement/pathing
    DENIAL = "denial"           # Prevents monster actions
    SETUP = "setup"             # Enables follow-up damage/combo
    CLEANUP = "cleanup"         # Finishes low-HP mobs efficiently
    SURVIVAL = "survival"       # Keeps player alive (heals, buffs)
    MOBILITY = "mobility"       # Changes position
    SUSTAIN = "sustain"         # Manages resources
    BURST = "burst"             # High damage in short window
    DOT = "dot"                 # Damage over time
    UTILITY = "utility"         # Everything else


class SkillCategory(Enum):
    PHYSICAL = "physical"
    MAGICAL = "magical"
    BUFF = "buff"
    HEAL = "heal"
    PASSIVE = "passive"
    MISCELLANEOUS = "misc"


@dataclass
class SkillClass:
    """Purpose-classified skill entry."""
    name: str
    purpose: SkillPurpose
    category: SkillCategory
    element: str = "neutral"
    
    # Combo fields
    combo_with: list[str] = field(default_factory=list)  # Skills this combos with
    combo_description: str = ""
    
    # Efficiency fields
    sp_efficiency: float = 1.0      # Damage per SP (relative to baseline)
    cast_time_s: float = 0.0
    after_cast_delay_s: float = 0.0
    
    # Targeting
    targets_self: bool = False
    targets_ground: bool = False
    targets_enemy: bool = True
    
    # Level notes
    level_notes: str = ""  # e.g., "Level 1 better than 10 for zoning"


# ── Skill purpose classification registry ──

# --- ZONING skills ---
ZONING_SKILLS = [
    SkillClass("Fire Wall", SkillPurpose.ZONING, SkillCategory.MAGICAL, "fire",
               combo_with=["Storm Gust", "Lord of Vermilion"],
               combo_description="Wall holds mobs in AoE",
               sp_efficiency=0.5, targets_ground=True,
               level_notes="Level 1 is BETTER than 10 (same wall time, less cast time)"),
    SkillClass("Quagmire", SkillPurpose.ZONING, SkillCategory.MAGICAL, "earth",
               combo_description="Slows mobs for kiting",
               sp_efficiency=0.3, cast_time_s=1.0, targets_ground=True,
               level_notes="Level 5 for max slow"),
    SkillClass("Ice Wall", SkillPurpose.ZONING, SkillCategory.MAGICAL, "water",
               combo_description="Blocks pathing completely",
               sp_efficiency=0.4, targets_ground=True,
               level_notes="Level 1-3 for blocking"),
    SkillClass("Safety Wall", SkillPurpose.ZONING, SkillCategory.MAGICAL, "neutral",
               combo_with=["Storm Gust"],
               combo_description="Tank while casting AoE",
               sp_efficiency=1.5, targets_ground=True,
               level_notes="Level 10 for max HP"),
]

# --- DENIAL skills ---
DENIAL_SKILLS = [
    SkillClass("Lex Aeterna", SkillPurpose.DENIAL, SkillCategory.MAGICAL, "neutral",
               combo_with=["Storm Gust", "Lord of Vermilion", "Soul Strike"],
               combo_description="Doubles magic damage on next hit",
               sp_efficiency=10.0, cast_time_s=0.5,
               level_notes="Level 1 only (no level scaling)"),
    SkillClass("Provoke", SkillPurpose.DENIAL, SkillCategory.PHYSICAL, "neutral",
               combo_with=["Bash", "Hammer Fall"],
               combo_description="Reduces DEF for physical follow-up",
               sp_efficiency=2.0, targets_self=False,
               level_notes="Level 5 for -25% DEF, Level 10 for aggro only"),
    SkillClass("Cyclone", SkillPurpose.DENIAL, SkillCategory.PHYSICAL, "wind",
               combo_with=["Whirlwind", "Wind Blade"],
               combo_description="Knocks back and damages",
               sp_efficiency=0.8, cast_time_s=0.8),
    SkillClass("Stone Curse", SkillPurpose.DENIAL, SkillCategory.MAGICAL, "earth",
               combo_description="Petrifies target (stops all action)",
               sp_efficiency=0.1, cast_time_s=1.5,
               level_notes="Level 1 for quick petrify"),
]

# --- SETUP skills ---
SETUP_SKILLS = [
    SkillClass("Cold Bolt", SkillPurpose.SETUP, SkillCategory.MAGICAL, "water",
               combo_with=["Fire Bolt", "Fire Ball"],
               combo_description="Wets target → Fire does 1.5x on wet in some versions",
               sp_efficiency=0.9, cast_time_s=1.0,
               level_notes="Level 5-10 for wet duration"),
    SkillClass("Mental Freeze", SkillPurpose.SETUP, SkillCategory.MAGICAL, "neutral",
               combo_with=["Storm Gust"],
               combo_description="Prevents teleport, locks target in place",
               sp_efficiency=0.5, cast_time_s=0.3),
    SkillClass("Enchant Poison", SkillPurpose.SETUP, SkillCategory.BUFF, "poison",
               combo_with=["Poison React", "Venom Dust"],
               combo_description="Enables poison weapon effects",
               sp_efficiency=0.0, targets_self=True),
    SkillClass("Enchant Fire", SkillPurpose.SETUP, SkillCategory.BUFF, "fire",
               combo_with=["Double Bolt Combo"],
               combo_description="Weapon enchant for element matching",
               sp_efficiency=0.0, targets_self=True),
]

# --- CLEANUP skills ---
CLEANUP_SKILLS = [
    SkillClass("Napalm Beat", SkillPurpose.CLEANUP, SkillCategory.MAGICAL, "neutral",
               combo_description="Finishes ghost-type mobs, low SP cost",
               sp_efficiency=2.0, targets_enemy=True,
               level_notes="Level 5 for efficiency"),
    SkillClass("Normal Attack", SkillPurpose.CLEANUP, SkillCategory.PHYSICAL, "neutral",
               combo_description="Zero SP cost, always available",
               sp_efficiency=float('inf'),  # Unlimited
               targets_enemy=True),
]

# --- SURVIVAL skills ---
SURVIVAL_SKILLS = [
    SkillClass("Heal", SkillPurpose.SURVIVAL, SkillCategory.HEAL, "neutral",
               combo_with=["Increase AGI", "Blessing", "Teleport"],
               combo_description="Sustain rotation",
               sp_efficiency=0.5, cast_time_s=0.8, targets_self=True,
               level_notes="Level 10 for efficiency"),
    SkillClass("Increase AGI", SkillPurpose.SURVIVAL, SkillCategory.BUFF, "neutral",
               combo_description="Boosts flee and attack speed",
               sp_efficiency=0.0, targets_self=True,
               level_notes="Level 10 for max AGI bonus"),
    SkillClass("Blessing", SkillPurpose.SURVIVAL, SkillCategory.BUFF, "neutral",
               combo_description="Boosts DEX, INT, LUK and ATK",
               sp_efficiency=0.0, targets_self=True,
               level_notes="Level 10 for +10 DEX/INT"),
    SkillClass("Assumptio", SkillPurpose.SURVIVAL, SkillCategory.BUFF, "neutral",
               combo_description="Reduces damage taken by ~40%",
               sp_efficiency=0.0, targets_self=True,
               level_notes="Level 5 for full effect"),
    SkillClass("Impositio Manus", SkillPurpose.SURVIVAL, SkillCategory.BUFF, "neutral",
               combo_description="Increases weapon ATK",
               sp_efficiency=0.0, targets_self=True,
               level_notes="Level 5 for +25 ATK"),
    SkillClass("Teleport", SkillPurpose.SURVIVAL, SkillCategory.MISCELLANEOUS, "neutral",
               combo_with=["Teleport", "Fly Wing"],
               combo_description="Emergency escape",
               sp_efficiency=0.0, targets_self=True),
    SkillClass("Pneuma", SkillPurpose.SURVIVAL, SkillCategory.MISCELLANEOUS, "neutral",
               combo_description="Makes party immune to ranged attacks",
               sp_efficiency=0.0, targets_ground=True),
    SkillClass("Safe Wall", SkillPurpose.SURVIVAL, SkillCategory.MAGICAL, "neutral",
               combo_description="Immune to physical damage on tile",
               sp_efficiency=0.0, targets_ground=True),
    SkillClass("Endure", SkillPurpose.SURVIVAL, SkillCategory.BUFF, "neutral",
               combo_description="HP won't drop below 1 for short time",
               sp_efficiency=0.0, targets_self=True),
]

# --- MOBILITY skills ---
MOBILITY_SKILLS = [
    SkillClass("Teleport", SkillPurpose.MOBILITY, SkillCategory.MISCELLANEOUS, "neutral",
               combo_description="Instant movement to random location",
               sp_efficiency=0.0, targets_self=True),
    SkillClass("Warp Portal", SkillPurpose.MOBILITY, SkillCategory.MISCELLANEOUS, "neutral",
               combo_description="Opens portal to saved location",
               sp_efficiency=0.0, targets_ground=True),
]

# --- BURST skills ---
BURST_SKILLS = [
    SkillClass("Storm Gust", SkillPurpose.BURST, SkillCategory.MAGICAL, "water",
               combo_with=["Safety Wall", "Lex Aeterna"],
               combo_description="Highest AoE damage, freezes enemies",
               sp_efficiency=1.5, cast_time_s=6.0, after_cast_delay_s=3.0, targets_ground=True,
               level_notes="Level 10 for max damage, cast time is long"),
    SkillClass("Lord of Vermilion", SkillPurpose.BURST, SkillCategory.MAGICAL, "wind",
               combo_with=["Safety Wall"],
               combo_description="Strong AoE, no freeze",
               sp_efficiency=1.3, cast_time_s=4.0, targets_ground=True,
               level_notes="Level 10 for max damage"),
    SkillClass("Meteor Storm", SkillPurpose.BURST, SkillCategory.MAGICAL, "fire",
               combo_with=["Safety Wall"],
               combo_description="Highest fire AoE, stuns enemies",
               sp_efficiency=1.2, cast_time_s=8.0, targets_ground=True,
               level_notes="Level 10 for max damage and stun"),
    SkillClass("Soul Strike", SkillPurpose.BURST, SkillCategory.MAGICAL, "neutral",
               combo_with=["Lex Aeterna"],
               combo_description="High single-target damage",
               sp_efficiency=1.1, cast_time_s=1.5, targets_enemy=True,
               level_notes="Level 10 for max damage"),
    SkillClass("Frost Diver", SkillPurpose.BURST, SkillCategory.MAGICAL, "water",
               combo_with=["Soul Strike"],
               combo_description="Single-target freeze + damage",
               sp_efficiency=0.8, cast_time_s=1.0, targets_enemy=True),
    SkillClass("Bowling Bash", SkillPurpose.BURST, SkillCategory.PHYSICAL, "neutral",
               combo_description="Knocks back all surrounding enemies",
               sp_efficiency=1.0, cast_time_s=0.5, targets_enemy=True,
               level_notes="Level 10 for max damage"),
    SkillClass("Spear Boomerang", SkillPurpose.BURST, SkillCategory.PHYSICAL, "neutral",
               combo_description="Ranged attack for knights",
               sp_efficiency=0.9, cast_time_s=0.3, targets_enemy=True),
]

# --- DOT skills ---
DOT_SKILLS = [
    SkillClass("Poison", SkillPurpose.DOT, SkillCategory.MAGICAL, "poison",
               combo_description="Poisons target for DoT over time",
               sp_efficiency=1.0, cast_time_s=0.5, targets_enemy=True),
    SkillClass("Fire Pillar", SkillPurpose.DOT, SkillCategory.MAGICAL, "fire",
               combo_description="Ground-targeted fire DoT",
               sp_efficiency=0.7, cast_time_s=1.0, targets_ground=True),
]

# ── Combined registry ──
ALL_CLASSIFIED_SKILLS: dict[str, SkillClass] = {}

for _skills in [ZONING_SKILLS, DENIAL_SKILLS, SETUP_SKILLS, CLEANUP_SKILLS,
                SURVIVAL_SKILLS, MOBILITY_SKILLS, BURST_SKILLS, DOT_SKILLS]:
    for _s in _skills:
        ALL_CLASSIFIED_SKILLS[_s.name.lower()] = _s


def get_skill_purpose(skill_name: str) -> SkillPurpose | None:
    """Get the purpose of a skill by name."""
    key = skill_name.lower().replace("_", " ")
    if key in ALL_CLASSIFIED_SKILLS:
        return ALL_CLASSIFIED_SKILLS[key].purpose
    # Fallback: classify by element/category heuristics
    return None


def get_skills_by_purpose(purpose: SkillPurpose) -> list[SkillClass]:
    """Get all skills with a given purpose."""
    return [s for s in ALL_CLASSIFIED_SKILLS.values() if s.purpose == purpose]


def recommend_rotation(
    available_skills: list[str],
    target_element: str,
    target_hp_pct: float,
) -> list[dict]:
    """
    Recommend a skill rotation based on available skills and target state.
    
    Priority:
    1. SURVIVAL first (heal, buff) — keep player alive
    2. ZONING/SETUP if needed — control the fight
    3. BURST for high-HP targets
    4. DOT for durable targets
    5. CLEANUP for finishing low-HP targets
    """
    rotation = []
    classified = []
    
    for skill_name in available_skills:
        key = skill_name.lower().replace("_", " ")
        if key in ALL_CLASSIFIED_SKILLS:
            classified.append(ALL_CLASSIFIED_SKILLS[key])
        else:
            # Unknown skill — treat as generic damage
            classified.append(SkillClass(skill_name, SkillPurpose.BURST, SkillCategory.PHYSICAL))
    
    # Group by purpose
    by_purpose: dict[SkillPurpose, list[SkillClass]] = {}
    for s in classified:
        by_purpose.setdefault(s.purpose, []).append(s)
    
    # 1. Survival first
    for s in by_purpose.get(SkillPurpose.SURVIVAL, []):
        rotation.append({"skill": s.name, "reason": "survival"})
    
    # 2. Zoning for area control
    for s in by_purpose.get(SkillPurpose.ZONING, []):
        rotation.append({"skill": s.name, "reason": "zone:" + s.combo_description})
    
    # 3. Setup skills (elemental advantage)
    for s in by_purpose.get(SkillPurpose.SETUP, []):
        if s.element == target_element or s.element == "neutral":
            rotation.append({"skill": s.name, "reason": "setup:" + s.combo_description})
    
    # 4. Denial if target is dangerous
    for s in by_purpose.get(SkillPurpose.DENIAL, []):
        rotation.append({"skill": s.name, "reason": "deny:" + s.combo_description})
    
    # 5. Burst for high-HP targets
    if target_hp_pct > 0.30:
        for s in by_purpose.get(SkillPurpose.BURST, []):
            if s.element == target_element or s.element == "neutral":
                rotation.append({"skill": s.name, "reason": f"burst:{s.combo_description}"})
    
    # 6. DOT for durable targets
    if target_hp_pct > 0.50:
        for s in by_purpose.get(SkillPurpose.DOT, []):
            rotation.append({"skill": s.name, "reason": "dot:" + s.combo_description})
    
    # 7. Cleanup for finishing
    if target_hp_pct < 0.30:
        for s in by_purpose.get(SkillPurpose.CLEANUP, []):
            rotation.append({"skill": s.name, "reason": "finish"})
    
    return rotation


def get_skill_combo(primary: str, secondary: str) -> str | None:
    """Check if two skills combo and return the combo description."""
    p_key = primary.lower().replace("_", " ")
    s_key = secondary.lower().replace("_", " ")
    
    p_skill = ALL_CLASSIFIED_SKILLS.get(p_key)
    if p_skill and s_key in [c.lower() for c in p_skill.combo_with]:
        return p_skill.combo_description
    
    s_skill = ALL_CLASSIFIED_SKILLS.get(s_key)
    if s_skill and p_key in [c.lower() for c in s_skill.combo_with]:
        return s_skill.combo_description
    
    return None
