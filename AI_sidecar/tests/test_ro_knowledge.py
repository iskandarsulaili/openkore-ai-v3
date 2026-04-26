from __future__ import annotations

from pathlib import Path

from ai_sidecar.autonomy.ro_knowledge import load_ro_knowledge


def _repo_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "AI_sidecar" / "ai_sidecar" / "autonomy" / "data"
    tables_dir = root / "tables"
    return data_dir, tables_dir


def test_load_ro_knowledge_parses_profiles_playbooks_and_tables() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    assert knowledge.version == "stage3-ro-progression-v1"
    assert "novice" in knowledge.profile_lookup
    assert "swordman" in knowledge.playbooks_by_job
    assert "knight" in knowledge.job_change_locations
    assert "swordman" in knowledge.job_builds


def test_playbook_lookup_ready_and_unsupported_routes() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    ready = knowledge.assess_job_advancement(job_name="Novice", base_level=10, job_level=10)
    assert ready["supported"] is True
    assert ready["ready"] is True
    assert ready["route_id"] == "novice_to_swordman"
    assert ready["target_job"] == "Swordman"
    assert isinstance(ready["location"], dict)
    assert ready["location"].get("map") == "izlude_in"

    unsupported = knowledge.assess_job_advancement(job_name="Merchant", base_level=30, job_level=30)
    assert unsupported["supported"] is False
    assert unsupported["status"] == "unsupported_job_route"


def test_progression_recommendation_uses_level_band() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    recommendation = knowledge.recommend_leveling(job_name="Mage", base_level=28)
    assert recommendation["knowledge_loaded"] is True
    assert recommendation["profile_key"] == "mage_path"
    assert recommendation["band_label"] == "mage_job_window"
    assert recommendation["target_maps"]


def test_stage4_opportunistic_upgrade_actionable_with_complete_evidence() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    assessment = knowledge.assess_opportunistic_upgrades(
        job_name="Novice",
        base_level=9,
        zeny=8000,
        inventory_items=[
            {
                "item_id": "sword_2",
                "name": "Sword [2]",
                "equipped": True,
                "category": "weapon",
                "metadata": {"slot": "weapon"},
            }
        ],
        market_listings=[
            {
                "item_id": "sword_3",
                "item_name": "Sword [3]",
                "buy_price": 5500,
                "source": "npc_shop",
            }
        ],
    )

    assert assessment["supported"] is True
    assert assessment["actionable"] is True
    assert assessment["status"] == "actionable"
    assert assessment["opportunities"]
    top = assessment["opportunities"][0]
    assert top["rule_id"] == "novice_weapon_sword_2_to_3"
    assert top["execution_mode"] == "direct"
    assert top["score_delta"] > 0
    assert top["buy_price"] == 5500


def test_stage4_opportunistic_upgrade_non_actionable_when_market_is_ambiguous() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    assessment = knowledge.assess_opportunistic_upgrades(
        job_name="Novice",
        base_level=9,
        zeny=9000,
        inventory_items=[
            {
                "item_id": "sword_2",
                "name": "Sword [2]",
                "equipped": True,
                "category": "weapon",
                "metadata": {"slot": "weapon"},
            }
        ],
        market_listings=[
            {
                "item_id": "sword_3",
                "item_name": "Sword [3]",
                "buy_price": 5600,
                "source": "mystery_source",
            }
        ],
    )

    assert assessment["supported"] is True
    assert assessment["actionable"] is False
    assert assessment["status"] == "non_actionable"
    assert assessment["opportunities"] == []
    reasons = [str(item) for item in assessment.get("non_actionable_reasons", [])]
    assert any("market_source_unsupported" in item for item in reasons)
