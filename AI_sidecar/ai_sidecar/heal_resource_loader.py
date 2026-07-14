"""
Heal Resource Loader — computes available healing items/skills from knowledge DB
and pushes them to the bridge as config. No hardcoded lists.

The bridge reads from config keys:
  aiSidecar_healItems  = "aegis1:qty:heal_hp:heal_sp:weight,aegis2:..."
  aiSidecar_healSkills = "skill1:level:sp_cost:heal_hp:heal_sp,skill2:..."
  aiSidecar_healThreshold = 0.35

Each item entry includes quantity, heal amount, SP heal, and weight so the
bridge can predict and estimate without querying the database at reflex time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "knowledge", "knowledge.json"
)
_RATHENA_DB = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "knowledge", "rathena_db", "db", "re"
)


def _parse_heal_script(script: str) -> tuple[int, int]:
    """Parse a rAthena item script to extract HP and SP heal values.

    Handles formats like:
      itemheal rand(45,65),0;
      itemheal 45,0;
      heal 45,0;
      percentheal 10,0;
    """
    hp_heal = 0
    sp_heal = 0

    # itemheal <hp>,<sp>;
    m = re.search(r"itemheal\s+rand\((\d+),(\d+)\)\s*,\s*(\d+)", script)
    if m:
        hp_heal = (int(m.group(1)) + int(m.group(2))) // 2
        sp_heal = int(m.group(3))
    else:
        m = re.search(r"itemheal\s+(\d+)\s*,\s*(\d+)", script)
        if m:
            hp_heal = int(m.group(1))
            sp_heal = int(m.group(2))

    # percentheal <hp%>,<sp%>;
    m = re.search(r"percentheal\s+(\d+)\s*,\s*(\d+)", script)
    if m:
        hp_heal = max(hp_heal, int(m.group(1)) * 100)  # Store as % * 100
        sp_heal = max(sp_heal, int(m.group(2)) * 100)

    # heal <hp>,<sp>;
    m = re.search(r"heal\s+(\d+)\s*,\s*(\d+)", script)
    if m:
        hp_heal = max(hp_heal, int(m.group(1)))
        sp_heal = max(sp_heal, int(m.group(2)))

    return hp_heal, sp_heal


class HealResourceLoader:
    """Computes healing resources from knowledge DB and pushes to bridge config."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._last_push: dict[str, str] = {}
        self._last_push_time: float = 0.0
        self._heal_items_db: list[dict[str, Any]] = []
        self._heal_skills_db: list[dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        """Load healing items and skills from knowledge database."""
        with self._lock:
            if self._loaded:
                return
            self._heal_items_db = self._load_heal_items()
            self._heal_skills_db = self._load_heal_skills()
            self._loaded = True
            logger.info(
                "heal_resources_loaded: %d items, %d skills",
                len(self._heal_items_db),
                len(self._heal_skills_db),
            )

    def _load_heal_items(self) -> list[dict[str, Any]]:
        """Load healing items from rAthena item_db_usable.yml."""
        items: list[dict[str, Any]] = []
        yml_path = os.path.join(_RATHENA_DB, "item_db_usable.yml")
        if not os.path.exists(yml_path):
            logger.warning("item_db_usable.yml not found at %s", yml_path)
            return items

        try:
            import yaml
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            body = data.get("Body", [])
            for entry in body:
                script = str(entry.get("Script", ""))
                if "heal" in script.lower() and (
                    "hp" in script.lower() or "hpheal" in script.lower()
                ):
                    hp_heal, sp_heal = _parse_heal_script(script)
                    items.append(
                        {
                            "aegis": entry.get("AegisName", ""),
                            "name": entry.get("Name", ""),
                            "weight": entry.get("Weight", 0),
                            "buy": entry.get("Buy", 0),
                            "hp_heal": hp_heal,
                            "sp_heal": sp_heal,
                            "is_percent": hp_heal > 1000 or sp_heal > 1000,
                        }
                    )
        except Exception as e:
            logger.warning("Failed to load item_db_usable.yml: %s", e)

        # Sort by heal amount descending (best first)
        items.sort(key=lambda x: -x["hp_heal"])
        return items

    def _load_heal_skills(self) -> list[dict[str, Any]]:
        """Load healing skills from rAthena skill_tree.yml."""
        skills: list[dict[str, Any]] = []
        yml_path = os.path.join(_RATHENA_DB, "skill_tree.yml")
        if not os.path.exists(yml_path):
            logger.warning("skill_tree.yml not found at %s", yml_path)
            return skills

        try:
            import yaml
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            body = data.get("Body", [])
            for job_entry in body:
                job = job_entry.get("Job", "")
                tree = job_entry.get("Tree", [])
                for skill in tree:
                    name = skill.get("Name", "")
                    max_lv = skill.get("MaxLevel", 0)
                    if any(
                        kw in name.lower()
                        for kw in ["heal", "cure", "recovery", "firstaid"]
                    ):
                        skills.append(
                            {
                                "name": name,
                                "job": job,
                                "max_level": max_lv,
                                "base_hp_heal": 45 if "heal" in name.lower() else 5,
                                "base_sp_cost": 10 if "heal" in name.lower() else 5,
                            }
                        )
        except Exception as e:
            logger.warning("Failed to load skill_tree.yml: %s", e)

        return skills

    def get_config_push(
        self, bot_id: str, snapshot: Any
    ) -> dict[str, str] | None:
        """Compute config push for a bot based on its snapshot.

        Returns a dict of config key->value to push to the bridge, or None
        if nothing changed.
        """
        self.load()

        # Extract inventory and skills from snapshot
        inventory_items: list[dict[str, Any]] = []
        skills_known: list[str] = []
        job_class = "novice"
        max_hp = 100
        max_sp = 50

        if hasattr(snapshot, "inventory_items"):
            inventory_items = [
                {"name": item.name, "amount": item.amount}
                for item in (snapshot.inventory_items or [])
            ]
        if hasattr(snapshot, "skills"):
            skills_known = [s.name for s in (snapshot.skills or [])]
        if hasattr(snapshot, "vitals"):
            job_class = str(
                getattr(snapshot.vitals, "job_name", "novice")
            ).lower()
            max_hp = getattr(snapshot.vitals, "hp_max", 100) or 100
            max_sp = getattr(snapshot.vitals, "sp_max", 50) or 50

        # Build item entries: "aegis:qty:heal_hp:heal_sp:weight"
        item_entries: list[str] = []
        inv_lookup = {}
        for item in inventory_items:
            name = str(item.get("name", "")).lower()
            inv_lookup[name] = int(item.get("amount", 0))

        for db_item in self._heal_items_db:
            aegis = db_item["aegis"]
            # Match by aegis name (underscore-separated) or display name
            match_key = aegis.lower().replace("_", " ")
            qty = inv_lookup.get(match_key, 0) or inv_lookup.get(aegis.lower(), 0)
            if qty > 0:
                hp = db_item["hp_heal"]
                sp = db_item["sp_heal"]
                # Convert percent heals to estimated absolute values
                if db_item["is_percent"]:
                    hp = int(max_hp * (hp / 10000.0))
                    sp = int(max_sp * (sp / 10000.0))
                item_entries.append(f"{aegis}:{qty}:{hp}:{sp}:{db_item['weight']}")

        # Build skill entries: "skill:level:sp_cost:heal_hp:heal_sp"
        skill_entries: list[str] = []
        skill_lookup = {s.lower(): s for s in skills_known}

        for db_skill in self._heal_skills_db:
            name = db_skill["name"]
            if name.lower() in skill_lookup:
                max_lv = db_skill["max_level"]
                sp_cost = db_skill["base_sp_cost"]
                hp_heal = db_skill["base_hp_heal"] * max_lv
                skill_entries.append(f"{name}:{max_lv}:{sp_cost}:{hp_heal}:0")

        push = {
            "aiSidecar_healItems": ",".join(item_entries),
            "aiSidecar_healSkills": ",".join(skill_entries),
            "aiSidecar_healThreshold": "0.35",
        }

        # Only push if changed
        key = f"{bot_id}:{push['aiSidecar_healItems']}:{push['aiSidecar_healSkills']}"
        now = time.time()
        if key == self._last_push.get(bot_id) and now - self._last_push_time < 30:
            return None

        self._last_push[bot_id] = key
        self._last_push_time = now
        return push


# Global singleton
_loader: HealResourceLoader | None = None
_loader_lock = RLock()


def get_heal_resource_loader() -> HealResourceLoader:
    global _loader
    with _loader_lock:
        if _loader is None:
            _loader = HealResourceLoader()
        return _loader
