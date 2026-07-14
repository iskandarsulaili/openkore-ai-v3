"""
Knowledge Database Loader — loads data from the unified knowledge.json
database instead of hardcoding. All modules should use this instead of
hardcoded data.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH: str = ""
_KNOWLEDGE_DATA: dict[str, Any] = {}
_KNOWLEDGE_LOCK = threading.RLock()
_KNOWLEDGE_LOADED = False


def _resolve_knowledge_path() -> str:
    """Resolve the path to knowledge.json."""
    global _KNOWLEDGE_PATH
    if _KNOWLEDGE_PATH:
        return _KNOWLEDGE_PATH

    # Check env var first
    env_path = os.environ.get("KNOWLEDGE_OUTPUT", "")
    if env_path and os.path.exists(env_path):
        _KNOWLEDGE_PATH = env_path
        return _KNOWLEDGE_PATH

    # Check relative to this file
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "knowledge", "knowledge.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge", "knowledge.json"),
        "/home/lot399/openkore-ai-v3/knowledge/knowledge.json",
    ]
    for c in candidates:
        resolved = os.path.abspath(c)
        if os.path.exists(resolved):
            _KNOWLEDGE_PATH = resolved
            return _KNOWLEDGE_PATH

    logger.warning("knowledge.json not found, checked: %s", candidates)
    return ""


def load_knowledge(force_reload: bool = False) -> dict[str, Any]:
    """Load the knowledge database. Thread-safe with lazy loading.
    
    Args:
        force_reload: If True, re-read the file even if already loaded.
                      Use this when the database may have been updated by
                      another bot instance.
    """
    global _KNOWLEDGE_DATA, _KNOWLEDGE_LOADED
    if _KNOWLEDGE_LOADED and not force_reload:
        return _KNOWLEDGE_DATA

    with _KNOWLEDGE_LOCK:
        if _KNOWLEDGE_LOADED and not force_reload:
            return _KNOWLEDGE_DATA

        path = _resolve_knowledge_path()
        if not path:
            logger.error("knowledge.json not found")
            _KNOWLEDGE_DATA = {}
            _KNOWLEDGE_LOADED = True
            return _KNOWLEDGE_DATA

        try:
            with open(path, "r") as f:
                _KNOWLEDGE_DATA = json.load(f)
            _KNOWLEDGE_LOADED = True
            logger.info("knowledge_loaded: %s (%d keys)", path, len(_KNOWLEDGE_DATA))
        except Exception as e:
            logger.error("knowledge_load_failed: %s", e)
            _KNOWLEDGE_DATA = {}

        return _KNOWLEDGE_DATA


def reload_knowledge() -> dict[str, Any]:
    """Force-reload the knowledge database. Call this when the DB file
    may have been updated by another bot instance or by the bot itself."""
    return load_knowledge(force_reload=True)


def get_items() -> list[dict]:
    """Get all items from the knowledge database."""
    data = load_knowledge()
    return data.get("items", {}).get("all", [])


def get_weapons() -> list[dict]:
    data = load_knowledge()
    return data.get("items", {}).get("weapons", [])


def get_armors() -> list[dict]:
    data = load_knowledge()
    return data.get("items", {}).get("armors", [])


def get_cards() -> list[dict]:
    data = load_knowledge()
    return data.get("items", {}).get("cards", [])


def get_mobs() -> list[dict]:
    data = load_knowledge()
    return data.get("mobs", [])


def get_mvps() -> list[dict]:
    """Get all MVP monsters (those with MvpExp > 0 or Mvp flag)."""
    mobs = get_mobs()
    return [m for m in mobs if m.get("MvpExp", 0) > 0 or m.get("Mvp", False)]


def get_quests() -> list[dict]:
    data = load_knowledge()
    return data.get("quests", [])


def get_guild_skills() -> list[dict]:
    data = load_knowledge()
    guild = data.get("guild", {})
    # Guild skills are in the skill_trees section
    skill_trees = data.get("skill_trees", [])
    guild_skills = [s for s in skill_trees if "Guild" in str(s.get("Job", ""))]
    return guild_skills


def get_refine_data() -> dict:
    data = load_knowledge()
    return data.get("refine", {})


def get_item_by_name(name: str) -> dict | None:
    """Find an item by name in the knowledge database."""
    items = get_items()
    for item in items:
        if item.get("Name", "").lower() == name.lower():
            return item
        if item.get("AegisName", "").lower() == name.lower():
            return item
    return None


def get_mob_by_name(name: str) -> dict | None:
    """Find a mob by name."""
    mobs = get_mobs()
    for mob in mobs:
        if mob.get("Name", "").lower() == name.lower():
            return mob
        if mob.get("AegisName", "").lower() == name.lower():
            return mob
    return None


def get_mob_by_id(mob_id: int) -> dict | None:
    """Find a mob by ID."""
    mobs = get_mobs()
    for mob in mobs:
        if mob.get("Id") == mob_id:
            return mob
    return None


def get_quest_by_id(quest_id: int) -> dict | None:
    """Find a quest by ID."""
    quests = get_quests()
    for q in quests:
        if q.get("Id") == quest_id:
            return q
    return None


def get_item_price(item_name: str) -> int:
    """Get the buy price of an item from the knowledge database."""
    item = get_item_by_name(item_name)
    if item:
        return item.get("Buy", 0)
    return 0


def get_item_slots(item_name: str) -> int:
    """Get the number of card slots for an item."""
    item = get_item_by_name(item_name)
    if item:
        return item.get("Slots", 0)
    return 0


def get_item_type(item_name: str) -> str:
    """Get the type of an item."""
    item = get_item_by_name(item_name)
    if item:
        return item.get("Type", "")
    return ""


def get_item_subtype(item_name: str) -> str:
    """Get the subtype of an item."""
    item = get_item_by_name(item_name)
    if item:
        return item.get("SubType", "")
    return ""


def get_mob_drops(mob_name: str) -> list[dict]:
    """Get the drops of a mob."""
    mob = get_mob_by_name(mob_name)
    if mob:
        drops = []
        for i in range(1, 11):
            drop_key = f"Drop{i}Id"
            rate_key = f"Drop{i}Rate"
            if drop_key in mob:
                drops.append({"item_id": mob[drop_key], "rate": mob.get(rate_key, 0)})
        return drops
    return []


def get_mob_stats(mob_name: str) -> dict:
    """Get stats of a mob."""
    mob = get_mob_by_name(mob_name)
    if mob:
        return {
            "level": mob.get("Level", 0),
            "hp": mob.get("Hp", 0),
            "attack": mob.get("Attack", 0),
            "defense": mob.get("Defense", 0),
            "race": mob.get("Race", ""),
            "element": mob.get("Element", ""),
            "size": mob.get("Size", ""),
        }
    return {}


def get_knowledge_summary() -> str:
    """Get a summary of the loaded knowledge database."""
    data = load_knowledge()
    if not data:
        return "Knowledge database not loaded"
    lines = [f"── Knowledge Database ──"]
    lines.append(f"Items: {len(get_items())}")
    lines.append(f"Weapons: {len(get_weapons())}")
    lines.append(f"Armors: {len(get_armors())}")
    lines.append(f"Cards: {len(get_cards())}")
    lines.append(f"Mobs: {len(get_mobs())}")
    lines.append(f"MVPs: {len(get_mvps())}")
    lines.append(f"Quests: {len(get_quests())}")
    return "\n".join(lines)
