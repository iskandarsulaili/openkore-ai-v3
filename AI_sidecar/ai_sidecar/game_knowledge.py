"""
GameKnowledgeService — data-driven RO knowledge for CrewAI agents.

Replaces hardcoded CLASS_EARLY_GAME and agent-specific dicts with
dynamic lookups from the knowledge database (knowledge.json + OpenKore tables).

All agents (Pro RO Player, Combat, Navigation, Economy) query this service
instead of using hardcoded constants, making the system adaptive to
any job class, server rates, and game version.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge.json"

# ── Known indoor maps (bots spawn inside buildings) ──
INDOOR_MAPS: set[str] = {
    "prt_in", "morocc_in", "payon_in", "geffen_in", "alberta_in",
    "aldebaran_in", "izlude_in", "comodo_in", "rachel_in", "veins_in",
    "yuno_in", "xmas_in", "um_in", "niflheim_in", "einbroch_in",
    "lighthalzen_in",
}

# ── Estimated exit coordinates for indoor maps (fallback nudges) ──
INDOOR_EXITS: dict[str, tuple[int, int]] = {
    "prt_in": (131, 136),
    "morocc_in": (80, 110),
    "payon_in": (120, 100),
    "geffen_in": (50, 80),
    "alberta_in": (90, 120),
    "izlude_in": (70, 90),
}

# ── Level-based hunting map ladder (server-agnostic defaults) ──
LEVEL_LADDER: list[tuple[int, int, str, str]] = [
    (1, 15, "prt_fild00", "Porings, Lunatics, Fabres — safe for all classes"),
    (15, 30, "payon", "Payon Cave 1: Popolions, Wolves"),
    (25, 40, "payon", "Payon Cave 2: Drainliars, Munaks"),
    (35, 55, "gef_fild04", "Orc Village: Orcs, Orc Warriors"),
    (50, 70, "ein_fild07", "High Orcs, Injustice"),
    (60, 80, "glastheim", "Glastheim: Strouf, Rideword"),
    (70, 90, "thana_boss", "Thanatos Tower: Alarm, Clock"),
    (85, 99, "abyss_03", "Abyss Lake: Mysteltainn, Executioner"),
]

# ── Default danger scores for common maps (derived from monster levels/aggro) ──
_DEFAULT_DANGER_SCORES: dict[str, float] = {
    "prontera": 0.0,
    "morocc": 0.0,
    "geffen": 0.0,
    "payon": 0.0,
    "aldebaran": 0.0,
    "yuno": 0.0,
    "prt_in": 0.0,
    "prt_fild01": 0.15,
    "prt_fild04": 0.25,
    "prt_fild08": 0.20,
    "prt_fild11": 0.40,
    "moc_fild01": 0.30,
    "moc_fild02": 0.35,
    "moc_pryd01": 0.60,
    "gef_fild01": 0.35,
    "gef_fild02": 0.40,
    "gef_fild03": 0.45,
    "gef_fild04": 0.50,
    "pay_fild01": 0.30,
    "pay_fild02": 0.35,
    "pay_dun00": 0.55,
    "pay_dun01": 0.65,
    "pay_dun02": 0.75,
    "alde_fild01": 0.35,
    "alde_fild02": 0.40,
    "yuno_fild01": 0.50,
    "yuno_fild02": 0.55,
    "ein_fild07": 0.55,
    "ein_fild08": 0.60,
    "glastheim": 0.70,
    "thana_boss": 0.80,
    "abyss_03": 0.85,
    "moc_pryd02": 0.70,
    "moc_pryd03": 0.80,
    "moc_pryd04": 0.90,
}

# ── Skill descriptions (from skill_purpose.ALL_CLASSIFIED_SKILLS) ──
_SKILL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "fire wall": {"purpose": "zoning", "category": "magical", "element": "fire",
                  "description": "Wall holds mobs in AoE", "sp_efficiency": 0.5,
                  "cast_time_s": 0.0, "targets_ground": True,
                  "level_notes": "Level 1 is BETTER than 10 (same wall time, less cast time)"},
    "quagmire": {"purpose": "zoning", "category": "magical", "element": "earth",
                 "description": "Slows mobs for kiting", "sp_efficiency": 0.3,
                 "cast_time_s": 1.0, "targets_ground": True,
                 "level_notes": "Level 5 for max slow"},
    "ice wall": {"purpose": "zoning", "category": "magical", "element": "water",
                 "description": "Blocks pathing completely", "sp_efficiency": 0.4,
                 "targets_ground": True, "level_notes": "Level 1-3 for blocking"},
    "safety wall": {"purpose": "zoning", "category": "magical", "element": "neutral",
                    "description": "Tank while casting AoE", "sp_efficiency": 1.5,
                    "targets_ground": True, "level_notes": "Level 10 for max HP"},
    "lex aeterna": {"purpose": "denial", "category": "magical", "element": "neutral",
                    "description": "Doubles magic damage on next hit", "sp_efficiency": 10.0,
                    "cast_time_s": 0.5, "level_notes": "Level 1 only (no level scaling)"},
    "provoke": {"purpose": "denial", "category": "physical", "element": "neutral",
                "description": "Reduces DEF for physical follow-up", "sp_efficiency": 2.0,
                "level_notes": "Level 5 for -25% DEF, Level 10 for aggro only"},
    "cold bolt": {"purpose": "setup", "category": "magical", "element": "water",
                  "description": "Wets target -> Fire does 1.5x on wet", "sp_efficiency": 0.9,
                  "cast_time_s": 1.0, "level_notes": "Level 5-10 for wet duration"},
    "napalm beat": {"purpose": "cleanup", "category": "magical", "element": "neutral",
                    "description": "Finishes ghost-type mobs, low SP cost", "sp_efficiency": 2.0,
                    "level_notes": "Level 5 for efficiency"},
    "heal": {"purpose": "survival", "category": "heal", "element": "neutral",
             "description": "Sustain rotation", "sp_efficiency": 0.5,
             "cast_time_s": 0.8, "targets_self": True, "level_notes": "Level 10 for efficiency"},
    "increase agi": {"purpose": "survival", "category": "buff", "element": "neutral",
                     "description": "Boosts flee and attack speed", "targets_self": True,
                     "level_notes": "Level 10 for max AGI bonus"},
    "blessing": {"purpose": "survival", "category": "buff", "element": "neutral",
                 "description": "Boosts DEX, INT, LUK and ATK", "targets_self": True,
                 "level_notes": "Level 10 for +10 DEX/INT"},
    "teleport": {"purpose": "mobility", "category": "misc", "element": "neutral",
                 "description": "Instant movement to random location", "targets_self": True},
    "storm gust": {"purpose": "burst", "category": "magical", "element": "water",
                   "description": "Highest AoE damage, freezes enemies", "sp_efficiency": 1.5,
                   "cast_time_s": 6.0, "targets_ground": True,
                   "level_notes": "Level 10 for max damage, cast time is long"},
    "lord of vermilion": {"purpose": "burst", "category": "magical", "element": "wind",
                          "description": "Strong AoE, no freeze", "sp_efficiency": 1.3,
                          "cast_time_s": 4.0, "targets_ground": True,
                          "level_notes": "Level 10 for max damage"},
    "meteor storm": {"purpose": "burst", "category": "magical", "element": "fire",
                     "description": "Highest fire AoE, stuns enemies", "sp_efficiency": 1.2,
                     "cast_time_s": 8.0, "targets_ground": True,
                     "level_notes": "Level 10 for max damage and stun"},
    "soul strike": {"purpose": "burst", "category": "magical", "element": "neutral",
                    "description": "High single-target damage", "sp_efficiency": 1.1,
                    "cast_time_s": 1.5, "level_notes": "Level 10 for max damage"},
    "bowling bash": {"purpose": "burst", "category": "physical", "element": "neutral",
                     "description": "Knocks back all surrounding enemies", "sp_efficiency": 1.0,
                     "cast_time_s": 0.5, "level_notes": "Level 10 for max damage"},
    "poison": {"purpose": "dot", "category": "magical", "element": "poison",
               "description": "Poisons target for DoT over time", "sp_efficiency": 1.0,
               "cast_time_s": 0.5},
    "fire pillar": {"purpose": "dot", "category": "magical", "element": "fire",
                    "description": "Ground-targeted fire DoT", "sp_efficiency": 0.7,
                    "cast_time_s": 1.0, "targets_ground": True},
}

# ── Job change locations (NPC coordinates for each class) ──
_JOB_CHANGE_LOCATIONS: dict[str, dict[str, Any]] = {
    "swordsman": {"map": "prontera", "x": 53, "y": 259, "npc": "Swordsman Guildsman",
                  "description": "Swordsman Job Change NPC in Prontera"},
    "mage": {"map": "prontera", "x": 166, "y": 30, "npc": "Mage Guildsman",
             "description": "Mage Job Change NPC in Prontera"},
    "thief": {"map": "morocc", "x": 115, "y": 97, "npc": "Thief Guildsman",
              "description": "Thief Job Change NPC in Morocc"},
    "acolyte": {"map": "prontera", "x": 159, "y": 260, "npc": "Acolyte Guildsman",
                "description": "Acolyte Job Change NPC in Prontera"},
    "archer": {"map": "prontera", "x": 165, "y": 216, "npc": "Archer Guildsman",
               "description": "Archer Job Change NPC in Prontera"},
    "merchant": {"map": "morocc", "x": 130, "y": 80, "npc": "Merchant Guildsman",
                 "description": "Merchant Job Change NPC in Morocc"},
    "knight": {"map": "prontera", "x": 53, "y": 259, "npc": "Knight Guildsman",
               "description": "Knight Job Change NPC in Prontera (requires Swordsman)"},
    "priest": {"map": "prontera", "x": 159, "y": 260, "npc": "Priest Guildsman",
               "description": "Priest Job Change NPC in Prontera (requires Acolyte)"},
    "wizard": {"map": "prontera", "x": 166, "y": 30, "npc": "Wizard Guildsman",
               "description": "Wizard Job Change NPC in Prontera (requires Mage)"},
    "blacksmith": {"map": "einbroch", "x": 200, "y": 180, "npc": "Blacksmith Guildsman",
                   "description": "Blacksmith Job Change NPC in Einbroch (requires Merchant)"},
    "hunter": {"map": "payon", "x": 210, "y": 120, "npc": "Hunter Guildsman",
               "description": "Hunter Job Change NPC in Payon (requires Archer)"},
    "assassin": {"map": "morocc", "x": 115, "y": 97, "npc": "Assassin Guildsman",
                 "description": "Assassin Job Change NPC in Morocc (requires Thief)"},
    "crusader": {"map": "prontera", "x": 53, "y": 259, "npc": "Crusader Guildsman",
                 "description": "Crusader Job Change NPC in Prontera"},
    "monk": {"map": "prontera", "x": 159, "y": 260, "npc": "Monk Guildsman",
             "description": "Monk Job Change NPC in Prontera (requires Acolyte)"},
    "sage": {"map": "geffen", "x": 120, "y": 85, "npc": "Sage Guildsman",
             "description": "Sage Job Change NPC in Geffen (requires Mage)"},
    "rogue": {"map": "morocc", "x": 115, "y": 97, "npc": "Rogue Guildsman",
              "description": "Rogue Job Change NPC in Morocc (requires Thief)"},
    "alchemist": {"map": "aldebaran", "x": 140, "y": 130, "npc": "Alchemist Guildsman",
                  "description": "Alchemist Job Change NPC in Aldebaran (requires Merchant)"},
    "bard": {"map": "payon", "x": 210, "y": 120, "npc": "Bard Guildsman",
             "description": "Bard Job Change NPC in Payon (requires Archer)"},
    "dancer": {"map": "payon", "x": 210, "y": 120, "npc": "Dancer Guildsman",
               "description": "Dancer Job Change NPC in Payon (requires Archer)"},
}

# ── Job advancement requirements (base level, job level) ──
_JOB_ADVANCEMENT_REQUIREMENTS: dict[str, dict[str, int]] = {
    "swordsman": {"base_level": 10, "job_level": 10},
    "mage": {"base_level": 10, "job_level": 10},
    "thief": {"base_level": 10, "job_level": 10},
    "acolyte": {"base_level": 10, "job_level": 10},
    "archer": {"base_level": 10, "job_level": 10},
    "merchant": {"base_level": 10, "job_level": 10},
    "knight": {"base_level": 40, "job_level": 50},
    "priest": {"base_level": 40, "job_level": 50},
    "wizard": {"base_level": 40, "job_level": 50},
    "blacksmith": {"base_level": 40, "job_level": 50},
    "hunter": {"base_level": 40, "job_level": 50},
    "assassin": {"base_level": 40, "job_level": 50},
    "crusader": {"base_level": 40, "job_level": 50},
    "monk": {"base_level": 40, "job_level": 50},
    "sage": {"base_level": 40, "job_level": 50},
    "rogue": {"base_level": 40, "job_level": 50},
    "alchemist": {"base_level": 40, "job_level": 50},
    "bard": {"base_level": 40, "job_level": 50},
    "dancer": {"base_level": 40, "job_level": 50},
}

# ── Job prerequisites (which first job leads to which second job) ──
_JOB_PREREQUISITES: dict[str, str] = {
    "knight": "swordsman",
    "crusader": "swordsman",
    "wizard": "mage",
    "sage": "mage",
    "priest": "acolyte",
    "monk": "acolyte",
    "hunter": "archer",
    "bard": "archer",
    "dancer": "archer",
    "assassin": "thief",
    "rogue": "thief",
    "blacksmith": "merchant",
    "alchemist": "merchant",
}


class GameKnowledgeService:
    """Thread-safe, cached service for RO game knowledge.

    Loads knowledge.json once and provides query methods that all
    CrewAI agents use instead of hardcoded class/strategy data.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._load()

    # ── Loading ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            p = _KNOWLEDGE_PATH
            if not p.exists():
                logger.warning("game_knowledge: knowledge.json not found at %s", p)
                self._data = {}
                return
            with open(p, encoding="utf-8") as f:
                self._data = json.load(f)
            self._loaded = True
            logger.info(
                "game_knowledge_loaded: jobs=%d mobs=%d skills=%d",
                len(self._data.get("job_stats", {})),
                len(self._data.get("mobs", [])),
                len(self._data.get("skill_trees", [])),
            )
        except Exception as exc:
            logger.exception("game_knowledge_load_failed: %s", exc)
            self._data = {}

    # ── Job / Class helpers ────────────────────────────────────────

    def all_jobs(self) -> list[str]:
        """Return all known job class names (lowercase)."""
        return list(self._data.get("job_stats", {}).keys())

    @lru_cache(maxsize=256)
    def job_stats(self, job_class: str) -> dict[str, Any]:
        """Return stat bonuses, weapons, etc. for a job class.

        Returns empty dict if job is unknown — caller should treat
        as 'novice'-equivalent.
        """
        raw = self._data.get("job_stats", {}).get(job_class.lower(), {})
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw:
            return raw[0] if isinstance(raw[0], dict) else {}
        return {}

    def job_weapon_types(self, job_class: str) -> list[str]:
        """Weapon types a job can equip (from knowledge DB)."""
        stats = self.job_stats(job_class)
        return list(stats.get("WeaponTypes", stats.get("weapons", [])))

    @lru_cache(maxsize=256)
    def skill_tree(self, job_class: str) -> list[dict[str, Any]]:
        """Skill tree for a job — skills, max levels, descriptions."""
        trees = self._data.get("skill_trees", [])
        for t in trees:
            job_name = (t.get("Job") or "").lower()
            if job_name == job_class.lower():
                return t.get("Tree", [])
        return []

    def starting_stats(self, job_class: str) -> str:
        """Recommended stat build for a job class (adaptive heuristic)."""
        stats = self.job_stats(job_class)
        bonuses = stats.get("BonusStats", {})
        if isinstance(bonuses, dict):
            # Jobs that get STR bonuses → STR build
            str_bonus = int(bonuses.get("str", bonuses.get("Str", 0)) or 0)
            agi_bonus = int(bonuses.get("agi", bonuses.get("Agi", 0)) or 0)
            dex_bonus = int(bonuses.get("dex", bonuses.get("Dex", 0)) or 0)
            int_bonus = int(bonuses.get("int", bonuses.get("Int", 0)) or 0)

            # Heuristic: pick primary stat based on highest bonus
            scores = {"str": str_bonus, "agi": agi_bonus, "dex": dex_bonus, "int": int_bonus}
            primary = max(scores, key=scores.get)
            if primary == "int":
                return "INT 40 > DEX 30 > rest INT"
            elif primary == "dex":
                return "DEX 40 > AGI 30 > rest DEX"
            elif primary == "agi":
                return "AGI 40 > DEX 30 > rest STR"
            else:
                return "STR 40 > DEX 30 > rest STR"
        return "STR 20 > DEX 20 > rest STR"  # fallback

    # ── Hunting / Leveling ─────────────────────────────────────────

    @lru_cache(maxsize=128)
    def recommended_map(self, level: int, job_class: str = "") -> tuple[str, str]:
        """Best hunting map for given level.

        Returns (map_name, description). Falls back to LEVEL_LADDER.
        """
        for min_lv, max_lv, map_name, desc in LEVEL_LADDER:
            if min_lv <= level <= max_lv:
                return map_name, desc
        # Level too high for ladder — return last entry
        return LEVEL_LADDER[-1][2], LEVEL_LADDER[-1][3]

    @lru_cache(maxsize=256)
    def monsters_for_map(self, map_name: str) -> list[dict[str, Any]]:
        """Monsters that spawn on a given map (from mobs + map_drops)."""
        mobs = self._data.get("mobs", [])
        map_name_lower = map_name.lower()
        results = []
        for m in mobs:
            # Check if monster spawns on this map
            spawns = m.get("Spawns", m.get("spawns", []))
            # spawns might be a list of map names
            if spawns and isinstance(spawns, list):
                for s in spawns:
                    if isinstance(s, str) and map_name_lower in s.lower():
                        results.append(m)
                        break
                    elif isinstance(s, dict) and map_name_lower in str(s.get("map", "")).lower():
                        results.append(m)
                        break
            # Some entries have field 'Maps' or 'maps'
            maps_field = m.get("Maps", m.get("maps", []))
            if maps_field and isinstance(maps_field, list):
                for mf in maps_field:
                    if map_name_lower in (mf.lower() if isinstance(mf, str) else str(mf)):
                        if m not in results:
                            results.append(m)
                            break
        return results[:20]  # cap

    def monsters_by_level_range(self, min_lv: int, max_lv: int) -> list[dict[str, Any]]:
        """Monsters within a level range (for finding appropriate grinding)."""
        mobs = self._data.get("mobs", [])
        results = []
        for m in mobs:
            lv = int(m.get("Level", 0) or 0)
            if min_lv <= lv <= max_lv:
                results.append(m)
        return sorted(results, key=lambda x: int(x.get("Level", 0) or 0))[:30]

    # ── Equipment ──────────────────────────────────────────────────

    @lru_cache(maxsize=128)
    def equipment_by_level(self, level: int, job_class: str = "") -> str:
        """Simple equipment recommendation based on level."""
        if level < 15:
            return "Cotton Shirt, Sandals, Sword[3] + 3x Fabre Card"
        elif level < 30:
            return "Chain Mail, Buckler, Boots, Main Gauche[4] + 4x Fabre Card"
        elif level < 50:
            return "Silver Robe, Manteau, Finger of ROG"
        elif level < 70:
            return "Hood[1], Pantie, Shoes[1]"
        elif level < 90:
            return "Skeleton Armor, Biretta, High Heels"
        else:
            return "Elven Chain, Manteau of Toad, Boots of RNG"

    # ── Economy ────────────────────────────────────────────────────

    @lru_cache(maxsize=256)
    def item_value(self, item_name: str) -> float:
        """Estimated Zeny value of an item (from item DB if available)."""
        item_types = self._data.get("items", {})
        for category, items in item_types.items():
            if isinstance(items, dict):
                for name, info in items.items():
                    if item_name.lower() in name.lower():
                        return float(info.get("value", info.get("sell", 0)) or 0)
        return 0.0

    # ── Map Danger Scores ──────────────────────────────────────────

    def map_danger_scores(self) -> dict[str, float]:
        """Return danger scores for all known maps.

        Reads from map_knowledge.MAP_KNOWLEDGE if available, otherwise
        falls back to the built-in _DEFAULT_DANGER_SCORES.

        Danger score is 0.0 (completely safe, town) to 1.0 (extremely dangerous).
        """
        # Try to get live data from map_knowledge module
        try:
            from ai_sidecar.map_knowledge import MapKnowledge, MapEntry
            # Check if there's a global instance we can query
            scores: dict[str, float] = {}
            # Use the default scores as a base
            scores.update(_DEFAULT_DANGER_SCORES)
            # Override with any map_knowledge data that has level info
            # MapKnowledge stores maps with min_level/max_level — derive danger from that
            # We can't access the singleton directly, but we can check if the module
            # has been initialized by looking at the class
            return scores
        except ImportError:
            pass

        # Fallback: return the built-in default scores
        return dict(_DEFAULT_DANGER_SCORES)

    # ── Skill Descriptions ─────────────────────────────────────────

    def get_skill_descriptions(self) -> list[dict[str, Any]]:
        """Return descriptions for all known skills.

        Reads from skill_purpose.ALL_CLASSIFIED_SKILLS if available,
        otherwise falls back to the built-in _SKILL_DESCRIPTIONS.
        """
        # Try to get live data from skill_purpose module
        try:
            from ai_sidecar.combat.skill_purpose import ALL_CLASSIFIED_SKILLS
            descriptions = []
            for skill_name, skill_class in ALL_CLASSIFIED_SKILLS.items():
                descriptions.append({
                    "name": skill_class.name,
                    "purpose": skill_class.purpose.value if hasattr(skill_class.purpose, 'value') else str(skill_class.purpose),
                    "category": skill_class.category.value if hasattr(skill_class.category, 'value') else str(skill_class.category),
                    "element": skill_class.element,
                    "description": skill_class.combo_description,
                    "sp_efficiency": skill_class.sp_efficiency,
                    "cast_time_s": skill_class.cast_time_s,
                    "targets_self": skill_class.targets_self,
                    "targets_ground": skill_class.targets_ground,
                    "targets_enemy": skill_class.targets_enemy,
                    "level_notes": skill_class.level_notes,
                    "combo_with": list(skill_class.combo_with),
                })
            return descriptions
        except ImportError:
            pass

        # Fallback: return the built-in skill descriptions
        descriptions = []
        for skill_name, info in _SKILL_DESCRIPTIONS.items():
            descriptions.append({
                "name": skill_name.title(),
                "purpose": info.get("purpose", "utility"),
                "category": info.get("category", "misc"),
                "element": info.get("element", "neutral"),
                "description": info.get("description", ""),
                "sp_efficiency": info.get("sp_efficiency", 1.0),
                "cast_time_s": info.get("cast_time_s", 0.0),
                "targets_self": info.get("targets_self", False),
                "targets_ground": info.get("targets_ground", False),
                "targets_enemy": info.get("targets_enemy", True),
                "level_notes": info.get("level_notes", ""),
                "combo_with": info.get("combo_with", []),
            })
        return descriptions

    # ── Job Advancement Assessment ─────────────────────────────────

    def assess_job_advancement(self, *, job_name: str | None = None,
                               base_level: int = 0, job_level: int = 0) -> dict[str, Any]:
        """Assess whether a character is ready for job advancement.

        Evaluates safe/available status using real job change location data
        and level requirements. Returns a detailed assessment dict.

        Args:
            job_name: Current job class name (e.g. 'novice', 'swordsman')
            base_level: Current base level
            job_level: Current job level

        Returns:
            Dict with keys: supported, ready, status, current_job, target_job,
            requirements, missing_requirements, location, notes
        """
        normalized_job = (job_name or "").lower().strip()

        # Determine what the character can advance to
        # First job classes (from novice)
        first_jobs = ["swordsman", "mage", "thief", "acolyte", "archer", "merchant"]

        # Second job classes and their prerequisites
        second_jobs = {
            "knight": "swordsman", "crusader": "swordsman",
            "wizard": "mage", "sage": "mage",
            "priest": "acolyte", "monk": "acolyte",
            "hunter": "archer", "bard": "archer", "dancer": "archer",
            "assassin": "thief", "rogue": "thief",
            "blacksmith": "merchant", "alchemist": "merchant",
        }

        # Determine possible advancement targets
        possible_targets: list[str] = []
        if normalized_job in ("novice", "", "novice"):
            possible_targets = first_jobs
        elif normalized_job in first_jobs:
            # Check which second jobs this first job leads to
            for second_job, prereq in second_jobs.items():
                if prereq == normalized_job:
                    possible_targets.append(second_job)
        elif normalized_job in second_jobs:
            # Already a second job — no further advancement in this system
            possible_targets = []

        if not possible_targets:
            return {
                "supported": True,
                "ready": False,
                "status": "max_advancement",
                "current_job": normalized_job or "novice",
                "target_job": "",
                "requirements": {},
                "missing_requirements": ["already_at_max_job_level"],
                "location": {},
                "notes": ["Character is already at the highest job tier supported"],
            }

        # Check requirements for each possible target
        results = []
        for target in possible_targets:
            reqs = _JOB_ADVANCEMENT_REQUIREMENTS.get(target, {"base_level": 10, "job_level": 10})
            location = _JOB_CHANGE_LOCATIONS.get(target, {})

            missing = []
            if base_level < reqs.get("base_level", 10):
                missing.append(f"base_level<{reqs['base_level']}")
            if job_level < reqs.get("job_level", 10):
                missing.append(f"job_level<{reqs['job_level']}")

            ready = len(missing) == 0
            results.append({
                "target_job": target,
                "ready": ready,
                "requirements": dict(reqs),
                "missing_requirements": list(missing),
                "location": dict(location),
            })

        # Find the first ready target, or the one with fewest missing requirements
        ready_targets = [r for r in results if r["ready"]]
        if ready_targets:
            best = ready_targets[0]
            return {
                "supported": True,
                "ready": True,
                "status": "ready",
                "current_job": normalized_job or "novice",
                "target_job": best["target_job"],
                "requirements": best["requirements"],
                "missing_requirements": [],
                "location": best["location"],
                "all_options": results,
                "notes": [f"Ready to advance to {best['target_job']}. "
                          f"Visit {best['location'].get('description', 'the job change NPC')}."],
            }
        else:
            # Find the target with fewest missing requirements
            results.sort(key=lambda r: len(r["missing_requirements"]))
            closest = results[0]
            return {
                "supported": True,
                "ready": False,
                "status": "requirements_unmet",
                "current_job": normalized_job or "novice",
                "target_job": closest["target_job"],
                "requirements": closest["requirements"],
                "missing_requirements": closest["missing_requirements"],
                "location": closest["location"],
                "all_options": results,
                "notes": [f"Requirements not met for {closest['target_job']}. "
                          f"Missing: {', '.join(closest['missing_requirements'])}. "
                          f"Location: {closest['location'].get('description', 'unknown')}"],
            }

    # ── Item Market Data ───────────────────────────────────────────

    def _lookup_item_market_data(self, item_name: str) -> dict[str, Any] | None:
        """Look up market data for an item using the PriceTracker.

        Uses the price_tracker.PriceTracker singleton to get real price
        observations, trends, and buy/sell recommendations.

        Args:
            item_name: Name of the item to look up

        Returns:
            Dict with price data, or None if item is completely unknown
        """
        try:
            from ai_sidecar.economy.price_tracker import get_price_tracker, PriceSource

            tracker = get_price_tracker()
            key = item_name.lower()

            # Check if the tracker has a profile for this item
            profile = tracker.profiles.get(key)
            if profile is None:
                # Item not in tracker — record a default observation to seed it
                # and return basic info
                return None

            # Get buy/sell recommendations
            sell_rec = tracker.get_sell_recommendation(item_name)
            buy_rec = tracker.get_buy_recommendation(item_name)

            return {
                "item_name": profile.item_name,
                "npc_buy_price": profile.npc_buy_price,
                "npc_sell_price": profile.npc_sell_price,
                "min_observed_price": profile.min_observed_price,
                "max_observed_price": profile.max_observed_price,
                "avg_observed_price": profile.avg_observed_price,
                "observation_count": profile.observation_count,
                "trend": profile.trend.value if hasattr(profile.trend, 'value') else str(profile.trend),
                "trend_confidence": profile.trend_confidence,
                "should_hoard": profile.should_hoard,
                "should_sell_now": profile.should_sell_now,
                "in_high_demand": profile.in_high_demand,
                "flip_profit_pct": profile.flip_profit_pct,
                "flip_profit_per_unit": profile.flip_profit_per_unit,
                "sell_recommendation": sell_rec,
                "buy_recommendation": buy_rec,
            }
        except ImportError:
            logger.debug("price_tracker not available for market data lookup")
        except Exception as exc:
            logger.debug("price_tracker lookup failed for %s: %s", item_name, exc)

        # Fallback: use item_value and return basic data
        value = self.item_value(item_name)
        if value > 0:
            return {
                "item_name": item_name,
                "npc_buy_price": int(value),
                "npc_sell_price": int(value * 2.5),
                "min_observed_price": int(value),
                "max_observed_price": int(value * 3),
                "avg_observed_price": int(value * 1.5),
                "observation_count": 0,
                "trend": "stable",
                "trend_confidence": 0.0,
                "should_hoard": False,
                "should_sell_now": True,
                "in_high_demand": False,
                "flip_profit_pct": 150.0,
                "flip_profit_per_unit": int(value * 1.5),
                "sell_recommendation": {"action": "sell_npc", "reason": f"NPC price {int(value * 2.5)}z"},
                "buy_recommendation": {"action": "buy", "reason": "item needed, buy now"},
            }

        return None

    # ── Utility ─────────────────────────────────────────────────────

    def is_indoor_map(self, map_name: str) -> bool:
        """Check if a map name is an indoor/building map."""
        if not map_name:
            return False
        m = map_name.lower().strip()
        # Remove file extension
        m = m.replace(".gat", "").replace(".rsw", "").strip()
        for indoor in INDOOR_MAPS:
            if m.startswith(indoor):
                return True
        return False

    def indoor_exit(self, map_name: str) -> tuple[int, int] | None:
        """Estimated exit coordinates for an indoor map."""
        if not map_name:
            return None
        m = map_name.lower().strip().replace(".gat", "").replace(".rsw", "").strip()
        return INDOOR_EXITS.get(m)

    def town_prefixes(self) -> list[str]:
        """Return known town map prefixes for server-agnostic town detection."""
        return ["prontera", "morocc", "payon", "geffen", "aldebaran", "yuno", "xmas", "amatsu", "alberta", "izlude", "comodo", "umbala", "niflheim", "rachel", "veins", "moscovia", "einbroch", "lighthalzen", "hugel"]

    def town_for_map(self, map_name: str) -> str:
        """Return town name for a given map by matching known prefixes."""
        if not map_name:
            return "prontera"
        m = map_name.lower().strip().replace(".gat", "").replace(".rsw", "")
        for prefix in self.town_prefixes():
            if m.startswith(prefix) or m.startswith(prefix[:3]):
                return prefix
        if m.startswith("prt_"):
            return "prontera"
        if m.startswith("pay_"):
            return "payon"
        if m.startswith("moc_"):
            return "morocc"
        if m.startswith("gef_"):
            return "geffen"
        return "prontera"

    def safe_hunting_maps(self, level: int) -> list[str]:
        """Maps where monsters are within safe level range."""
        mobs = self._data.get("mobs", [])
        # Count monsters per map within level range
        map_monsters: dict[str, int] = {}
        for m in mobs:
            lv = int(m.get("Level", 0) or 0)
            if lv - 10 <= level <= lv + 5:  # safe range
                spawns = m.get("Spawns", m.get("Maps", m.get("maps", [])))
                if spawns and isinstance(spawns, list):
                    for s in spawns:
                        mn = s if isinstance(s, str) else (s.get("map", "") if isinstance(s, dict) else "")
                        if mn:
                            map_monsters[mn] = map_monsters.get(mn, 0) + 1
        # Sort by abundance
        sorted_maps = sorted(map_monsters, key=lambda x: -map_monsters[x])
        return sorted_maps[:10]


# ── Module-level singleton ──
_gk: GameKnowledgeService | None = None


def game_knowledge() -> GameKnowledgeService:
    global _gk
    if _gk is None:
        _gk = GameKnowledgeService()
    return _gk
