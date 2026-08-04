"""Achievement knowledge — load the server's achievement DB for progression tracking.

The server defines hundreds of achievements (achievement_db.yml: Adventure/Battle/
Taming/Job_Change/Goal_Level groups, with Score, Condition, Rewards, Targets). This
lets the AI track which achievements are available/completable as part of its
progression knowledge (titles, achievement points, quest unlocks), so a high-level
character can pursue worthwhile achievement rewards. Server-agnostic: loads from the
server DB, falls back to the bundled knowledge, and no-ops if neither exists.
"""
from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _knowledge_base() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent  # AI_sidecar/


def _server_achievement_paths() -> list[Path]:
    base = _knowledge_base()
    return [
        Path.home() / "rathena-AI-world" / "db" / "re" / "achievement_db.yml",
        Path.home() / "rathena-AI-world" / "db" / "pre-re" / "achievement_db.yml",
        base / "knowledge" / "rathena_db" / "db" / "re" / "achievement_db.yml",
        base / "knowledge" / "rathena_db" / "db" / "pre-re" / "achievement_db.yml",
    ]


_ACHIEVEMENTS: list[dict[str, Any]] = []
_ACHIEVEMENTS_LOADED = False


def load_achievements() -> list[dict[str, Any]]:
    """Load the achievement list (id/group/name/score/condition/rewards/targets).

    Prefers the live server's achievement_db.yml; falls back to the bundled
    knowledge.json achievements. Cached after load.
    """
    global _ACHIEVEMENTS, _ACHIEVEMENTS_LOADED
    if _ACHIEVEMENTS_LOADED:
        return list(_ACHIEVEMENTS)

    # Prefer the live server DB.
    path = next((p for p in _server_achievement_paths() if p.is_file()), None)
    if path is not None:
        try:
            import yaml
            data = yaml.safe_load(open(path, errors="replace"))
            body = data.get("Body", []) or [] if isinstance(data, dict) else []
            for entry in body:
                if not isinstance(entry, dict) or "Id" not in entry:
                    continue
                _ACHIEVEMENTS.append({
                    "id": int(entry["Id"]),
                    "group": str(entry.get("Group", "") or ""),
                    "name": str(entry.get("Name", "") or ""),
                    "score": int(entry.get("Score", 0) or 0),
                    "condition": str(entry.get("Condition", "") or ""),
                    "rewards": entry.get("Rewards", []) if isinstance(entry.get("Rewards"), list) else [],
                    "targets": entry.get("Targets", []) if isinstance(entry.get("Targets"), list) else [],
                    "title_id": entry.get("TitleId", entry.get("TitleID", 0) or 0),
                })
            logger.info("achievements: loaded %d from %s", len(_ACHIEVEMENTS), path)
            _ACHIEVEMENTS_LOADED = True
            return list(_ACHIEVEMENTS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("achievements: server db load failed: %s", exc)

    # Fall back to bundled knowledge.json achievements.
    try:
        kpath = _knowledge_base() / "knowledge" / "knowledge.json"
        if kpath.is_file():
            data = json.load(open(kpath))
            _ACHIEVEMENTS = list(data.get("achievements", {}).get("achievements", []))
            _ACHIEVEMENTS_LOADED = True
            logger.info("achievements: loaded %d from bundled knowledge.json", len(_ACHIEVEMENTS))
    except Exception as exc:  # noqa: BLE001
        logger.debug("achievements: bundled load failed: %s", exc)

    return list(_ACHIEVEMENTS)


def get_achievement_by_id(achievement_id: int) -> dict[str, Any] | None:
    """Return an achievement by its server ID, or None if unknown."""
    for a in load_achievements():
        if int(a.get("id", -1)) == achievement_id:
            return a
    return None


def get_achievements_by_group(group: str) -> list[dict[str, Any]]:
    """Return all achievements in a group (case-insensitive)."""
    g = group.lower()
    return [a for a in load_achievements() if str(a.get("group", "")).lower() == g]


def achievement_groups() -> list[str]:
    """Distinct achievement group names on this server."""
    groups = {str(a.get("group", "")) for a in load_achievements()}
    return sorted(g for g in groups if g)


def total_achievement_score() -> int:
    """Sum of all achievement scores (total obtainable achievement points)."""
    return sum(int(a.get("score", 0) or 0) for a in load_achievements())


def achievement_count() -> int:
    """Number of achievements defined on this server."""
    return len(load_achievements())
