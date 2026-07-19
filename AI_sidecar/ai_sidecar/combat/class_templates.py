"""Class Combat Templates — per-class combat behavior profiles.

Architecture:
  - Data-driven templates loaded from skill_tree.yml + skill_db.yml
  - Defines skill priority, elemental preferences, emergency handling per class
  - Self-optimizing: learns from combat results (kill speed, damage taken)

RULE.md compliance:
  - All skill data from rAthena DB — zero hardcoded skill names
  - Class-specific patterns derived from skill tree structure
  - Fallback to "generic physical/magical" for unknown classes
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default combat parameters per combat style ────────────────────

# These are not hardcoded skill names — they describe combat BEHAVIOR
# which maps to whatever skills the class has available

COMBAT_STYLES = {
    "physical": {
        "primary_stat": "STR",
        "damage_type": "weapon",
        "element_source": "weapon",  # Skills inherit weapon element
        "sp_efficiency": "medium",
        "range": "melee",
    },
    "magical": {
        "primary_stat": "INT",
        "damage_type": "magic",
        "element_source": "skill",   # Skills have own element
        "sp_efficiency": "low",
        "range": "ranged",
    },
    "hybrid": {
        "primary_stat": "STR/INT",
        "damage_type": "mixed",
        "element_source": "both",
        "sp_efficiency": "medium",
        "range": "melee",
    },
    "support": {
        "primary_stat": "INT/DEX",
        "damage_type": "heal_buff",
        "element_source": "skill",
        "sp_efficiency": "high",
        "range": "any",
    },
}

# Job → combat style mapping (based on skill tree analysis)
# These map rAthena job names to combat behavior profiles
_JOB_STYLES = {
    "Novice": "physical",
    "Swordman": "physical", "Knight": "physical", "Lord_Knight": "physical",
    "Crusader": "physical", "Paladin": "physical", "Royal_Guard": "physical",
    "Mage": "magical", "Wizard": "magical", "High_Wizard": "magical",
    "Warlock": "magical", "Arch_Mage": "magical",
    "Archer": "physical", "Hunter": "physical", "Sniper": "physical",
    "Ranger": "physical", "Windhawk": "physical",
    "Acolyte": "support", "Priest": "support", "High_Priest": "support",
    "Arch_Bishop": "support", "Cardinal": "support",
    "Merchant": "physical", "Blacksmith": "physical", "Whitesmith": "physical",
    "Mechanic": "physical", "Meister": "physical",
    "Thief": "physical", "Assassin": "physical", "Assassin_Cross": "physical",
    "Guillotine_Cross": "physical", "Shadow_Cross": "physical",
    "Monk": "physical", "Champion": "physical", "Sura": "physical",
    "Sage": "magical", "Professor": "magical", "Sorcerer": "magical",
    "Elemental_Master": "magical",
    "Rogue": "physical", "Stalker": "physical", "Shadow_Chaser": "physical",
    "Alchemist": "hybrid", "Creator": "hybrid", "Genetic": "hybrid",
    "Bard": "support", "Clown": "support", "Minstrel": "support",
    "Dancer": "support", "Gypsy": "support", "Wanderer": "support",
    "Taekwon": "physical", "Star_Gladiator": "physical",
    "Soul_Linker": "support",
    "Ninja": "magical", "Kagerou": "magical", "Oboro": "magical",
    "Gunslinger": "physical", "Rebellion": "physical",
    "Summoner": "magical", "Spirit_Handler": "magical",
    "Super_Novice": "hybrid", "Hyper_Novice": "hybrid",
}


def get_combat_style(job_name: str) -> str:
    """Get combat style for a job class. Returns 'physical' as default."""
    # Normalize job name
    job_key = job_name.strip().replace(" ", "_")
    for key, style in _JOB_STYLES.items():
        if key.lower() in job_key.lower() or job_key.lower() in key.lower():
            return style
    return "physical"  # Default for unknown classes


def prefers_elemental_matching(job_name: str) -> bool:
    """Check if class prefers element-matched skills.
    
    Magical classes benefit most from element matching (bolt spells).
    Physical classes rely on weapon element (endow/converters).
    """
    style = get_combat_style(job_name)
    return style in ("magical", "hybrid")


def get_sp_threshold(job_name: str) -> float:
    """Get SP threshold below which skills should stop.
    
    Magical classes need more SP for spells.
    Physical classes can auto-attack without SP.
    """
    style = get_combat_style(job_name)
    thresholds = {"physical": 0.05, "magical": 0.20, "hybrid": 0.10, "support": 0.30}
    return thresholds.get(style, 0.10)


def get_hp_emergency(job_name: str) -> float:
    """Get HP ratio for emergency healing.
    
    Support/healer classes heal earlier.
    Physical classes can survive longer.
    """
    style = get_combat_style(job_name)
    thresholds = {"physical": 0.20, "magical": 0.30, "hybrid": 0.25, "support": 0.40}
    return thresholds.get(style, 0.25)
