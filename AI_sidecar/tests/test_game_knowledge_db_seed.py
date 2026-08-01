"""Regression test: GameKnowledgeDB must self-create schema and seed data.

The DB-backed knowledge layer was a read-only facade over tables that
NOTHING ever created or populated — every query returned empty and
consumers (heal strategy, crafting NPC lookups, hunting-zone advice)
silently fell back to hardcoded paths. _ensure_seeded() now creates the
full schema on first use and seeds zone_ladder (LEVEL_LADDER + map_drops),
skill_builds (skill trees), and job_paths (job stats).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ai_sidecar.game_knowledge_db import GameKnowledgeDB


def _fresh_db(tmp_path: Path) -> GameKnowledgeDB:
    return GameKnowledgeDB(db_path=str(tmp_path / "bot_knowledge.db"))


def test_schema_created_and_seeded(tmp_path: Path) -> None:
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(str(tmp_path / "bot_knowledge.db"))
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    for t in ["zone_ladder", "skill_builds", "job_paths", "npc_interactions",
              "stat_builds", "player_memory", "exp_efficiency"]:
        assert t in tables, f"missing table {t}"


def test_zone_ladder_covers_low_and_high_levels(tmp_path: Path) -> None:
    db = _fresh_db(tmp_path)
    z_low = db.get_hunting_zone(5)
    assert z_low is not None, "level 5 must resolve a zone"
    z_mid = db.get_hunting_zone(60)
    assert z_mid is not None, "level 60 must resolve a zone"
    assert z_mid["map_name"] != z_low["map_name"], "different levels -> different zones"


def test_skill_builds_seeded_for_base_jobs(tmp_path: Path) -> None:
    db = _fresh_db(tmp_path)
    for job in ("novice", "swordman", "mage", "archer"):
        build = db.get_skill_build(job)
        assert build is not None, f"skill build missing for {job}"
        assert "skill_order" in build


def test_job_paths_connect_base_to_advanced(tmp_path: Path) -> None:
    db = _fresh_db(tmp_path)
    for base in ("swordman", "mage", "archer", "thief"):
        path = db.get_job_path(base)
        assert path is not None, f"job path missing for {base}"
        assert path.get("to_job"), f"to_job empty for {base}"


def test_seed_idempotent(tmp_path: Path) -> None:
    db1 = _fresh_db(tmp_path)
    n1 = db1.get_zone_ladder()
    db2 = GameKnowledgeDB(db_path=str(tmp_path / "bot_knowledge.db"))
    n2 = db2.get_zone_ladder()
    assert len(n1) == len(n2), "re-instantiation must not duplicate seed rows"
    assert len(n1) >= 20, "ladder should have 20+ zones (LEVEL_LADDER + map_drops)"
