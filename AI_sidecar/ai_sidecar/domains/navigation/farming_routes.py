"""Farming Routes — pragmatic leveling path system for RO.

Maps character level to optimal farming zones with monster data,
EXP efficiency estimates, and party requirements.

Each FarmingRoute is a structured definition of one map zone:
  - Which monsters spawn there
  - What level range it's optimal for
  - Estimated zeny/hour (conservative, pre-renewal ballpark)
  - Whether it requires a party
  - Recommended build type for the zone

The system provides:
  - route_for_level(level): returns the best route for a given level
  - all_routes_for_level(level): returns all viable routes
  - suggest_route_for_player(level, class_type): class-specific recommendation
  - suggest_next_routes(level): what to move to next

All EXP/zeny estimates are conservative pre-renewal iRO/private-server
ballpark figures. Actual rates vary based on server rates, class,
gear, and competition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Route definition ────────────────────────────────────────────────


@dataclass
class FarmingRoute:
    """A farming route (map zone) for a specific level range.

    Attributes:
        map_name: RO internal map name (e.g. \"prontera\").
        display_name: Human-readable name (e.g. \"Prontera Fields\").
        min_level: Minimum recommended base level.
        max_level: Maximum recommended base level (diminishing returns above).
        monsters: List of monster names that spawn here.
        zeny_per_hour_estimate: Rough net zeny/hour (after potion cost).
            Conservative pre-renewal public-server estimates.
        requires_party: True if soloing is impractical.
        danger_level: 1 (very safe) to 5 (deadly without caution).
        recommended_builds: List of class archetypes that do well here.
            Values: \"melee\", \"ranged\", \"magic\", \"dagger\", \"bow\", \"staff\", \"tank\".
        best_element: Element that deals the most damage here.
        notes: Additional strategy notes.
    """
    map_name: str
    display_name: str
    min_level: int
    max_level: int
    monsters: list[str] = field(default_factory=list)
    zeny_per_hour_estimate: int = 0
    requires_party: bool = False
    danger_level: int = 1
    recommended_builds: list[str] = field(default_factory=list)
    best_element: str = "neutral"
    notes: str = ""

    def is_in_range(self, level: int) -> bool:
        """Check if a level falls within this route's optimal range."""
        return self.min_level <= level <= self.max_level

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "name": self.display_name,
            "level_range": f"{self.min_level}-{self.max_level}",
            "monsters": self.monsters,
            "zeny_per_hour": self.zeny_per_hour_estimate,
            "requires_party": self.requires_party,
            "danger_level": self.danger_level,
            "recommended_builds": self.recommended_builds,
            "best_element": self.best_element,
            "notes": self.notes,
        }


# ── EXP bracket definitions ─────────────────────────────────────────

EXP_BRACKETS: list[dict[str, Any]] = [
    {"range": (1, 10),   "label": "Novice",      "total_exp": 1_000,     "best_map": "prontera"},
    {"range": (10, 20),  "label": "Early",        "total_exp": 50_000,   "best_map": "payon"},
    {"range": (20, 30),  "label": "Apprentice",   "total_exp": 500_000,  "best_map": "orcs"},
    {"range": (30, 40),  "label": "Intermediate", "total_exp": 3_000_000, "best_map": "orcs_2"},
    {"range": (40, 50),  "label": "Skilled",      "total_exp": 12_000_000, "best_map": "geographers"},
    {"range": (50, 60),  "label": "Veteran",      "total_exp": 40_000_000, "best_map": "alarms"},
    {"range": (60, 70),  "label": "Expert",       "total_exp": 100_000_000, "best_map": "juperos"},
    {"range": (70, 80),  "label": "Master",       "total_exp": 250_000_000, "best_map": "juperos_2"},
    {"range": (80, 90),  "label": "Grand Master", "total_exp": 500_000_000, "best_map": "glast_heim"},
    {"range": (90, 99),  "label": "Transcendent", "total_exp": 1_000_000_000, "best_map": "thanatos"},
]

# Cumulative experience needed to reach each level (pre-renewal).
# Key = level, Value = cumulative EXP from level 1.
# Useful for estimating time-to-next-level.
CUMULATIVE_EXP: dict[int, int] = {
    1: 0, 2: 54, 3: 222, 4: 573, 5: 1176, 6: 2100, 7: 3414, 8: 5187, 9: 7488, 10: 10386,
    11: 13950, 12: 18249, 13: 23352, 14: 29328, 15: 36246, 16: 44175, 17: 53184, 18: 63342,
    19: 74718, 20: 87381, 21: 101400, 22: 116844, 23: 133782, 24: 152283, 25: 172416,
    26: 194250, 27: 217854, 28: 243297, 29: 270648, 30: 299976, 31: 331350, 32: 364839,
    33: 400512, 34: 438438, 35: 478686, 36: 521325, 37: 566424, 38: 614052, 39: 664278,
    40: 717171, 41: 772800, 42: 831234, 43: 892542, 44: 956793, 45: 1024056, 46: 1094400,
    47: 1167894, 48: 1244607, 49: 1324608, 50: 1407966, 51: 1494750, 52: 1585029,
    53: 1678872, 54: 1776348, 55: 1877526, 56: 1982475, 57: 2091264, 58: 2203962,
    59: 2320638, 60: 2441361, 61: 2566200, 62: 2695224, 63: 2828502, 64: 2966103,
    65: 3108096, 66: 3254550, 67: 3405534, 68: 3561117, 69: 3721368, 70: 3886356,
    71: 4056150, 72: 4230819, 73: 4410432, 74: 4595058, 75: 4784766, 76: 4979625,
    77: 5179704, 78: 5385072, 79: 5595798, 80: 5811951, 81: 6033600, 82: 6260814,
    83: 6493662, 84: 6732213, 85: 6976536, 86: 7226700, 87: 7482774, 88: 7744827,
    89: 8012928, 90: 8287146, 91: 8567550, 92: 8854209, 93: 9147192, 94: 9446568,
    95: 9752406, 96: 10064775, 97: 10383644, 98: 10709082, 99: 11041158,
}


# ── Route database ──────────────────────────────────────────────────

FARMING_ROUTES: list[FarmingRoute] = [
    # ── Tier 1: 1-10  (Novice) ────────────────────────────────────
    FarmingRoute(
        map_name="prontera",
        display_name="Prontera Fields (prt_fild00-11)",
        min_level=1, max_level=12,
        monsters=["Poring", "Lunatic", "Fabre", "Picky", "Chonchon"],
        zeny_per_hour_estimate=500,
        danger_level=1,
        recommended_builds=["melee", "ranged", "magic"],
        notes="Safe starter zones. Porings everywhere, easy EXP.",
    ),
    FarmingRoute(
        map_name="mjolnir_01",
        display_name="Mjolnir Dead Fild 01 (moc_fild01)",
        min_level=5, max_level=15,
        monsters=["Poring", "Fabre", "Pupa", "Condor", "Wolf"],
        zeny_per_hour_estimate=800,
        danger_level=1,
        recommended_builds=["melee", "ranged"],
        notes="Wolves hit a bit harder but still safe.",
    ),

    # ── Tier 2: 10-20 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="payon_cave_01",
        display_name="Payon Cave 1F (pay_dun00)",
        min_level=10, max_level=22,
        monsters=["Skeleton", "Familiar", "Orc Zombie", "Bathory"],
        zeny_per_hour_estimate=2_000,
        danger_level=2,
        recommended_builds=["melee", "ranged"],
        best_element="holy",
        notes="Undead-heavy. Holy element weapons/spells double damage. Watch for Bathory.",
    ),
    FarmingRoute(
        map_name="payon_cave_02",
        display_name="Payon Cave 2F (pay_dun01)",
        min_level=14, max_level=24,
        monsters=["Familiar", "Skeleton", "Orc Zombie", "Werewolf", "Munak"],
        zeny_per_hour_estimate=3_000,
        danger_level=2,
        recommended_builds=["melee", "ranged", "magic"],
        best_element="holy",
        notes="Darker level, more spawns. Good holy-grind spot.",
    ),
    FarmingRoute(
        map_name="mjolnir_02",
        display_name="Mjolnir Dead Fild 02 (mjo_dun01)",
        min_level=10, max_level=20,
        monsters=["Wolf", "Vadon", "Thief Bug", "Hornet"],
        zeny_per_hour_estimate=1_500,
        danger_level=1,
        recommended_builds=["melee", "ranged"],
        notes="Alternative to Payon Cave. Thief Bugs drop Jellopy (vendor).",
    ),

    # ── Tier 3: 20-30 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="orcs_01",
        display_name="Orc Dungeon 1F (orcsdun01)",
        min_level=18, max_level=30,
        monsters=["Orc Warrior", "Orc Zombie", "Orc Skeleton"],
        zeny_per_hour_estimate=8_000,
        danger_level=2,
        recommended_builds=["melee", "ranged"],
        best_element="fire",
        notes="Classic leveling spot. Orc Warriors drop Orc Claw (vendor). High spawn density.",
    ),
    FarmingRoute(
        map_name="geographers_01",
        display_name="Geographers (pay_dun00)",
        min_level=20, max_level=32,
        monsters=["Geographer", "Drainliar", "Metaller", "Plankton"],
        zeny_per_hour_estimate=6_000,
        danger_level=2,
        recommended_builds=["magic", "ranged"],
        best_element="fire",
        notes="Geographers are fire-weak plants. Good for mages with Fire Bolt.",
    ),

    # ── Tier 4: 30-40 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="orcs_02",
        display_name="Orc Dungeon 2F (orcsdun02)",
        min_level=30, max_level=42,
        monsters=["Orc Archer", "Orc Warrior", "Orc Lady", "Orc Skeleton"],
        zeny_per_hour_estimate=15_000,
        danger_level=3,
        recommended_builds=["melee", "ranged"],
        best_element="fire",
        notes="Orc Archers hurt — bring fire armor or ranged counter. High zeny from Orc Lady drops.",
    ),
    FarmingRoute(
        map_name="eggra",
        display_name="Eggra (gef_fild14)",
        min_level=28, max_level=40,
        monsters=["Eggra", "Caramel", "Rideword"],
        zeny_per_hour_estimate=12_000,
        danger_level=2,
        recommended_builds=["magic", "ranged"],
        notes="Eggra are good EXP. Rideword drops are valuable. Holy damage works well.",
    ),

    # ── Tier 5: 40-50 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="geographers_02",
        display_name="Geographers Deep (pay_dun01)",
        min_level=38, max_level=50,
        monsters=["Geographer", "Stem Worm", "Drainliar"],
        zeny_per_hour_estimate=20_000,
        danger_level=2,
        recommended_builds=["magic", "ranged"],
        best_element="fire",
        notes="Fire mage paradise. Stem Worms hit back — keep distance.",
    ),
    FarmingRoute(
        map_name="anaconda",
        display_name="Anaconda (moc_fild13)",
        min_level=40, max_level=52,
        monsters=["Anaconda", "Snake", "Argiope", "Side Winder"],
        zeny_per_hour_estimate=18_000,
        danger_level=2,
        recommended_builds=["melee", "ranged"],
        notes="Snake skins vendor well. Argiope drop is decent.",
    ),

    # ── Tier 6: 50-60 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="alarms",
        display_name="Alarm (alde_dun01)",
        min_level=50, max_level=62,
        monsters=["Alarm", "Clock", "Grand Peco"],
        zeny_per_hour_estimate=35_000,
        danger_level=3,
        recommended_builds=["magic", "ranged"],
        best_element="water",
        notes="Alarms are magic-resistant but water-weak. Alarms drop valuable loot.",
    ),
    FarmingRoute(
        map_name="skeleton_workers",
        display_name="Skeleton Workers (moc_fild01)",
        min_level=50, max_level=60,
        monsters=["Skeleton Worker", "Skeleton"],
        zeny_per_hour_estimate=30_000,
        danger_level=2,
        recommended_builds=["melee", "ranged", "magic"],
        best_element="holy",
        notes="Skeleton Workers drop Steel (valuable). Holy element doubles damage.",
    ),

    # ── Tier 7: 60-70 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="juperos_01",
        display_name="Juperos 1F (juperos_01)",
        min_level=60, max_level=72,
        monsters=["Apocalype", "Dragon", "Scorpion"],
        zeny_per_hour_estimate=50_000,
        danger_level=3,
        recommended_builds=["magic", "ranged"],
        best_element="wind",
        notes="Dragon drops valuable scales. Apocalype hits hard — bring elemental armor.",
    ),
    FarmingRoute(
        map_name="stings",
        display_name="Sting (ra_san01)",
        min_level=60, max_level=70,
        monsters=["Sting", "Raydric", "Marionette"],
        zeny_per_hour_estimate=45_000,
        danger_level=3,
        recommended_builds=["magic", "ranged"],
        best_element="fire",
        notes="Stings are earth-element. Fire magic deals 175%. Raydric drops Immortal Heart.",
    ),
    FarmingRoute(
        map_name="magmarings",
        display_name="Magmarings (mag_dun01)",
        min_level=62, max_level=72,
        monsters=["Magmaring", "Kaho", "Lava Golem"],
        zeny_per_hour_estimate=48_000,
        danger_level=3,
        recommended_builds=["magic"],
        best_element="water",
        notes="Fire-element zone. Water magic deals 175%. Bring fire armor or you'll get melted.",
    ),

    # ── Tier 8: 70-80 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="juperos_02",
        display_name="Juperos 2F (juperos_02)",
        min_level=70, max_level=82,
        monsters=["Apocalype", "Dragon", "Scorpion King"],
        zeny_per_hour_estimate=70_000,
        danger_level=4,
        recommended_builds=["magic", "ranged", "tank"],
        best_element="wind",
        notes="Higher density than 1F. Scorpion King is mini-boss. Great zeny.",
    ),
    FarmingRoute(
        map_name="abyss_lake",
        display_name="Abyss Lake (abyss_01)",
        min_level=72, max_level=85,
        monsters=["Medusa", "Strouf", "Mermaid"],
        zeny_per_hour_estimate=65_000,
        danger_level=4,
        requires_party=True,
        recommended_builds=["magic", "tank"],
        best_element="wind",
        notes="Medusa petrifies — bring Green Herbs or Vit-based build. Party recommended.",
    ),

    # ── Tier 9: 80-90 ─────────────────────────────────────────────
    FarmingRoute(
        map_name="culs_de_sac",
        display_name="Culs-de-sac (c_tower1-4)",
        min_level=80, max_level=92,
        monsters=["Zombie", "Ghoul", "Wraith", "Evil Druid"],
        zeny_per_hour_estimate=90_000,
        danger_level=4,
        recommended_builds=["magic", "ranged"],
        best_element="holy",
        notes="Undead tower. Holy magic destroys everything. Evil Druid cards are jackpot.",
    ),
    FarmingRoute(
        map_name="glast_heim",
        display_name="Glast Heim — Bathory (gl_dun01)",
        min_level=80, max_level=90,
        monsters=["Bathory", "Injustice", "Wraith", "Evil Druid"],
        zeny_per_hour_estimate=85_000,
        danger_level=4,
        recommended_builds=["magic", "ranged"],
        best_element="holy",
        notes="Bathory drops are decent. Watch for Injustice — they hit hard.",
    ),

    # ── Tier 10: 90-99 ────────────────────────────────────────────
    FarmingRoute(
        map_name="thanatos_01",
        display_name="Thanatos Tower 1F (tha_t01)",
        min_level=90, max_level=99,
        monsters=["Thanatos", "Nightmare Terror", "Dullahan"],
        zeny_per_hour_estimate=120_000,
        danger_level=5,
        requires_party=False,
        recommended_builds=["magic", "ranged", "tank"],
        best_element="holy",
        notes="Thanatos Tower is the premier 90+ spot. High EXP, high zeny. Holy element essential.",
    ),
    FarmingRoute(
        map_name="thanatos_04",
        display_name="Thanatos Tower 4F (tha_t04)",
        min_level=93, max_level=99,
        monsters=["Thanatos", "Banshee", "Dullahan", "False Angel"],
        zeny_per_hour_estimate=150_000,
        danger_level=5,
        requires_party=True,
        recommended_builds=["magic", "ranged", "tank"],
        best_element="holy",
        notes="Elite floor. Party mandatory. Best EXP in the game before transcending.",
    ),
    FarmingRoute(
        map_name="endless_tower",
        display_name="Endless Tower (e_tower)",
        min_level=90, max_level=99,
        monsters=["Various mini-bosses and MVP's"],
        zeny_per_hour_estimate=200_000,
        danger_level=5,
        requires_party=True,
        recommended_builds=["magic", "ranged", "tank"],
        notes="MVPs and mini-bosses. Requires coordinated party. Best loot in the game.",
    ),
]


# ── Route lookup methods ────────────────────────────────────────────


def route_for_level(level: int) -> FarmingRoute | None:
    """Get the single best farming route for a given level.

    Uses a simple heuristic: prefers higher danger (higher EXP) routes
    within the player's level range, prioritizing routes with the best
    zeny estimate for their bracket.
    """
    viable = [r for r in FARMING_ROUTES if r.is_in_range(level)]
    if not viable:
        # No exact match — find closest
        viable = sorted(FARMING_ROUTES, key=lambda r: abs(r.min_level - level))
        if viable:
            return viable[0]
        return None

    # Best = highest zeny/hour within range
    viable.sort(key=lambda r: r.zeny_per_hour_estimate, reverse=True)
    return viable[0]


def all_routes_for_level(level: int) -> list[FarmingRoute]:
    """Get all farming routes that cover a given level, sorted by zeny/hour."""
    viable = [r for r in FARMING_ROUTES if r.is_in_range(level)]
    viable.sort(key=lambda r: r.zeny_per_hour_estimate, reverse=True)
    return viable


def suggest_route_for_player(
    level: int,
    class_type: str | None = None,
) -> list[dict[str, Any]]:
    """Get class-aware route suggestions for a player.

    Args:
        level: Player's current base level.
        class_type: Optional class archetype filter ("melee", "magic", "ranged", "tank").

    Returns:
        List of dicts with full route info, sorted by suitability.
    """
    viable = all_routes_for_level(level)

    if class_type:
        # Prefer routes that recommend the player's build type
        def sort_key(r: FarmingRoute) -> float:
            build_match = 1.0 if class_type in r.recommended_builds else 0.5
            return r.zeny_per_hour_estimate * build_match
        viable.sort(key=sort_key, reverse=True)
    else:
        viable.sort(key=lambda r: r.zeny_per_hour_estimate, reverse=True)

    return [r.to_dict() for r in viable]


def suggest_next_routes(level: int, count: int = 3) -> list[dict[str, Any]]:
    """Suggest the next routes coming up after the current level.

    Returns routes whose min_level is above the given level, sorted
    by min_level ascending.

    Useful for forward-planning: 'I'm level 32, what maps should I
    be heading toward?'
    """
    upcoming = [r for r in FARMING_ROUTES if r.min_level > level]
    upcoming.sort(key=lambda r: r.min_level)
    return [r.to_dict() for r in upcoming[:count]]


def get_exp_bracket(level: int) -> dict[str, Any] | None:
    """Get the EXP bracket that contains a given level."""
    for bracket in EXP_BRACKETS:
        lo, hi = bracket["range"]
        if lo <= level <= hi:
            return bracket
    return None


def exp_to_next_level(current_level: int) -> int:
    """Get the EXP needed to reach the next level.

    Returns 0 if already at max level.
    """
    if current_level >= 99:
        return 0
    current_cumulative = CUMULATIVE_EXP.get(current_level, 0)
    next_cumulative = CUMULATIVE_EXP.get(current_level + 1, 0)
    return next_cumulative - current_cumulative


def get_best_element_for_map(map_name: str) -> str:
    """Get the recommended attack element for a map by name."""
    for route in FARMING_ROUTES:
        if route.map_name == map_name or route.display_name.lower().startswith(map_name.lower()):
            return route.best_element
    return "neutral"


def get_route_by_map(map_name: str) -> FarmingRoute | None:
    """Look up a route by its map_name."""
    for route in FARMING_ROUTES:
        if route.map_name == map_name:
            return route
    return None


def get_routes_summary() -> dict[str, Any]:
    """Get a summary of all routes for the combat intel / frontend."""
    return {
        "total_routes": len(FARMING_ROUTES),
        "tiers": len(EXP_BRACKETS),
        "routes": [r.to_dict() for r in FARMING_ROUTES],
        "exp_brackets": EXP_BRACKETS,
    }
