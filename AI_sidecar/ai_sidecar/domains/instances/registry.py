"""Instance definitions — entrance NPC, requirements, rewards."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Known instance dungeons
_INSTANCE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "edda_biolab": {
        "name": "Edda Biolab",
        "abbreviation": "edda_biolab",
        "entrance_map": "lhz_dun03",
        "entrance_npc": "Researcher",
        "requirements": {
            "min_level": 70,
            "max_level": 99,
            "party_size": (1, 6),
            "zeny_cost": 10000,
            "quest_required": "",
        },
        "stages": [
            {"name": "Floor 1", "monsters": ["Ragged Zombie", "Cramp", "Bloody Knight"]},
            {"name": "Floor 2", "monsters": ["Maya", "Phendark", "Kiel"]},
            {"name": "Floor 3", "monsters": ["Eremes", "Seyren", "Katro"]},
            {"name": "Boss Floor", "monsters": ["Biolab Master"]},
        ],
        "rewards": {
            "base_exp": 5000000,
            "job_exp": 3000000,
            "items": ["Biolab Card", "Memory Fragment", "Elunium"],
        },
        "cooldown_hours": 24,
        "max_entries_per_day": 1,
    },
    "thanatos_tower": {
        "name": "Thanatos Tower",
        "abbreviation": "thanatos",
        "entrance_map": "lhz_dun01",
        "entrance_npc": "Lighthalzen Guard",
        "requirements": {
            "min_level": 55,
            "max_level": 99,
            "party_size": (1, 12),
            "zeny_cost": 5000,
            "quest_required": "",
        },
        "stages": [
            {"name": "Floor 1-3", "monsters": ["Rideword", "Pasana", "Deviruchi"]},
            {"name": "Floor 4-6", "monsters": ["Incubus", "Succubus", "Necromancer"]},
            {"name": "Floor 7-9", "monsters": ["Dark Illusion", "Bloody Murderer"]},
            {"name": "Boss Floor", "monsters": ["Thanatos Lord"]},
        ],
        "rewards": {
            "base_exp": 3000000,
            "job_exp": 2000000,
            "items": ["Thanatos Card", "Hallow Ring", "Blue Potion"],
        },
        "cooldown_hours": 24,
        "max_entries_per_day": 1,
    },
    "endless_tower": {
        "name": "Endless Tower",
        "abbreviation": "et",
        "entrance_map": "mid_camp",
        "entrance_npc": "Tower Guard",
        "requirements": {
            "min_level": 40,
            "max_level": 99,
            "party_size": (1, 12),
            "zeny_cost": 0,
            "quest_required": "",
        },
        "stages": [
            {"name": "Floors 1-25", "monsters": ["Poring", "Lunatic", "Fabre"]},
            {"name": "Floors 26-50", "monsters": ["Wolf", "Drainliar", "Argiope"]},
            {"name": "Floors 51-75", "monsters": ["Mummy", "Raydric", "Firelock Soldier"]},
            {"name": "Floors 76-100", "monsters": ["MVP Mixed", "Mini-Boss Wave"]},
        ],
        "rewards": {
            "base_exp": 10000000,
            "job_exp": 5000000,
            "items": ["Tower Card", "Ancient Elunium", "Old Purple Box"],
        },
        "cooldown_hours": 48,
        "max_entries_per_day": 1,
    },
    "ghost_palace": {
        "name": "Ghost Palace",
        "abbreviation": "ghost_palace",
        "entrance_map": "niflheim",
        "entrance_npc": "Gatekeeper",
        "requirements": {
            "min_level": 50,
            "max_level": 99,
            "party_size": (1, 6),
            "zeny_cost": 2000,
            "quest_required": "",
        },
        "stages": [
            {"name": "Courtyard", "monsters": ["Ghost", "Wraith", "Wraith Dead"]},
            {"name": "Main Hall", "monsters": ["Nightmare", "Maya Purple", "Angeling"]},
            {"name": "Boss Chamber", "monsters": ["Garm", "Lady of the Ghost"]},
        ],
        "rewards": {
            "base_exp": 2000000,
            "job_exp": 1500000,
            "items": ["Ghost Palace Card", "Cursed Seal", "Old Blue Box"],
        },
        "cooldown_hours": 24,
        "max_entries_per_day": 2,
    },
}


class InstanceRegistry:
    """Store instance definitions and provide lookups."""

    def __init__(self, db: Any = None) -> None:
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def get_available_instances(self, base_level: int) -> list[dict]:
        """Get instance dungeons available at the bot's level."""
        available = []
        for inst_id, inst_def in _INSTANCE_DEFINITIONS.items():
            req = inst_def["requirements"]
            if req["min_level"] <= base_level <= req.get("max_level", 99):
                available.append({
                    "instance_id": inst_id,
                    "name": inst_def["name"],
                    "entrance_map": inst_def["entrance_map"],
                    "entrance_npc": inst_def["entrance_npc"],
                    "requirements": req,
                    "stages": len(inst_def["stages"]),
                    "cooldown_hours": inst_def["cooldown_hours"],
                })
        return available

    def get_instance(self, instance_id: str) -> dict | None:
        """Get the full definition for a specific instance."""
        return _INSTANCE_DEFINITIONS.get(instance_id)

    def get_stages(self, instance_id: str) -> list[dict]:
        """Get stage definitions for an instance."""
        inst = _INSTANCE_DEFINITIONS.get(instance_id)
        return inst["stages"] if inst else []

    def get_rewards(self, instance_id: str) -> dict:
        """Get reward info for an instance."""
        inst = _INSTANCE_DEFINITIONS.get(instance_id)
        return inst["rewards"] if inst else {}

    def meets_requirements(
        self,
        instance_id: str,
        base_level: int,
        party_size: int = 1,
    ) -> tuple[bool, list[str]]:
        """Check if bot meets instance requirements.

        Returns (meets, reasons).
        """
        inst = _INSTANCE_DEFINITIONS.get(instance_id)
        if not inst:
            return False, ["Instance not found"]

        reasons: list[str] = []
        req = inst["requirements"]

        if base_level < req["min_level"]:
            reasons.append(f"Need level {req['min_level']}+ (have {base_level})")
        if req.get("max_level", 99) and base_level > req["max_level"]:
            reasons.append(f"Max level is {req['max_level']} (have {base_level})")
        if party_size < req["party_size"][0]:
            reasons.append(f"Need at least {req['party_size'][0]} party members")

        return len(reasons) == 0, reasons

    def get_cooldown_status(self, instance_id: str, last_run: float) -> dict:
        """Get cooldown info for an instance."""
        import time
        inst = _INSTANCE_DEFINITIONS.get(instance_id)
        if not inst:
            return {"is_available": True, "time_until_available": 0, "cooldown_hours": 0}

        cooldown_seconds = inst["cooldown_hours"] * 3600
        elapsed = time.time() - last_run
        remaining = cooldown_seconds - elapsed

        return {
            "is_available": remaining <= 0,
            "time_until_available": max(0, remaining),
            "cooldown_hours": inst["cooldown_hours"],
        }

    def get_entry_cost(self, instance_id: str) -> int:
        """Get zeny cost to enter an instance."""
        inst = _INSTANCE_DEFINITIONS.get(instance_id)
        if inst:
            return inst["requirements"].get("zeny_cost", 0)
        return 0

    def instance_ids(self) -> list[str]:
        """Get all registered instance IDs."""
        return list(_INSTANCE_DEFINITIONS.keys())
