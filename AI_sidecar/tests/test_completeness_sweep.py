"""Regression tests for completeness-batch fixes (dead/stub code wired).

Covers:
- GameKnowledgeDB.find_npc_for_task uses the real `task_type` column
  (was `interaction_type` -> "no such column" error) + list_npcs_on_map.
- npc/services.get_npcs_on_map returns DB rows (was a `return []` stub).
- quests/tracker.get_available_for_level returns tracked quests (was []).
- skills_curator._run_consolidation performs deterministic dedupe (was a
  "Phase 2 / LLM not implemented" placeholder).
- progression/lifecycle get_config returns real config (was a None stub).
"""
from __future__ import annotations

import sqlite3

from ai_sidecar.domains.npc.services import NPCService
from ai_sidecar.domains.progression.lifecycle import LifecycleStateMachine
from ai_sidecar.domains.quests.tracker import QuestTracker
from ai_sidecar.game_knowledge_db import GameKnowledgeDB


# ── S2: GameKnowledgeDB NPC queries ────────────────────────────────────


def _seed_npc(db_path, npc_name="Kafra", map_name="prontera", task_type="buy"):
    # Ensure the GameKnowledgeDB schema exists first (the class creates it
    # on construction), then seed a row.
    GameKnowledgeDB(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO npc_interactions (npc_name, map_name, task_type, steps_json) "
        "VALUES (?, ?, ?, ?)",
        (npc_name, map_name, task_type, "[]"),
    )
    conn.commit()
    conn.close()


def test_find_npc_for_task_uses_task_type_column(tmp_path) -> None:
    """find_npc_for_task must query the real `task_type` column (not the old
    nonexistent `interaction_type`), else it throws 'no such column'."""
    db_path = str(tmp_path / "gk.sqlite")
    db = GameKnowledgeDB(db_path)
    _seed_npc(db_path, npc_name="PKafra", map_name="prontera", task_type="buy")
    # Must not raise; returns the row matching task_type=buy on prontera.
    row = db.find_npc_for_task("buy", "prontera")
    assert row is not None
    assert row["npc_name"] == "PKafra"
    assert row["task_type"] == "buy"


def test_list_npcs_on_map_returns_rows(tmp_path) -> None:
    db_path = str(tmp_path / "gk.sqlite")
    db = GameKnowledgeDB(db_path)
    _seed_npc(db_path, npc_name="A", map_name="prontera", task_type="buy")
    _seed_npc(db_path, npc_name="B", map_name="prontera", task_type="sell")
    _seed_npc(db_path, npc_name="C", map_name="izlude", task_type="heal")
    rows = db.list_npcs_on_map("prontera")
    assert {r["npc_name"] for r in rows} == {"A", "B"}


def test_get_npcs_on_map_service_no_longer_stub(tmp_path) -> None:
    db_path = str(tmp_path / "gk.sqlite")
    _seed_npc(db_path, npc_name="Shop", map_name="prt_fild08", task_type="buy")
    gk = GameKnowledgeDB(db_path)
    svc = NPCService(gk)
    rows = svc.get_npcs_on_map("prt_fild08")
    assert any(r["npc_name"] == "Shop" for r in rows)


# ── S3: QuestTracker.get_available_for_level ───────────────────────────


def test_get_available_for_level_returns_tracked(tmp_path) -> None:
    tracker = QuestTracker(str(tmp_path / "quests.sqlite"))
    from ai_sidecar.domains.quests.tracker import QuestState

    tracker.track_quest("bot:x", QuestState(
        quest_id="q1", quest_name="Academy Basics", status="active",
        npc_start="Receptionist", npc_complete="Receptionist",
        objectives=[], rewards={}, started_at=0.0, completed_at=0.0,
    ))
    avail = tracker.get_available_for_level(base_level=1, map_name="iz_ac01")
    assert any(q.get("quest_name") == "Academy Basics" for q in avail)


# ── S4: skills_curator deterministic consolidation ─────────────────────


def test_run_consolidation_deduplicates(tmp_path, monkeypatch) -> None:
    import ai_sidecar.skills_curator as sc
    from ai_sidecar import skills_usage

    usage = {
        "combat_attack": {"state": "active"},
        "combat_attack_old": {"state": "archived"},
        "party_heal": {"active": "active"},  # unrelated active
    }
    monkeypatch.setattr(skills_usage, "list_skills", lambda: usage)
    monkeypatch.setattr(skills_usage, "set_state", lambda *a, **k: True)
    monkeypatch.setattr(skills_usage, "remove_skill", lambda *a, **k: True)

    result = sc._run_consolidation()
    # combat_attack_old is a prefixed near-duplicate of active combat_attack
    assert "combat_attack_old" in result["merged"] or "combat_attack_old" in result["deleted"]
    assert "Deterministic consolidation" in result["note"]


# ── S5: progression get_config ─────────────────────────────────────────


def test_lifecycle_get_config_returns_real_config() -> None:
    sm = LifecycleStateMachine()
    cfg = sm.get_config("bot:x", job_name="novice")
    assert cfg["phase"] in ("DISCONNECTED", "disconnected")
    assert cfg["job_change_at_level"] == 10
    assert isinstance(cfg["state_timeouts"], dict)
    assert "map_loaded" in cfg["state_timeouts"]
    assert isinstance(cfg["backoff"], dict)
