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


def test_wave_a1_signal_domains_actionable_with_intent_signals() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    assessment = knowledge.assess_opportunistic_upgrades(
        job_name="Novice",
        base_level=20,
        zeny=3000,
        inventory_items=[],
        market_listings=[
            {
                "item_id": "red_potion",
                "item_name": "Red Potion",
                "buy_price": 45,
                "source": "npc_shop",
            }
        ],
        signals={
            "in_combat": False,
            "map_known": True,
            "nearby_hostiles": 1,
            "inventory_pressure": False,
            "market_listing_count": 1,
            "vendor_exposure": 2,
            "exploration_intent": True,
            "card_gear_farming_intent": True,
            "mercenary_homunculus_intent": True,
            "companion_available": True,
            "vending_intent": True,
            "overweight_ratio": 0.5,
        },
    )

    assert assessment["supported"] is True
    assert assessment["actionable"] is True
    opportunities = assessment["opportunities"]
    assert isinstance(opportunities, list) and opportunities
    domains = {str(item.get("domain") or "") for item in opportunities if isinstance(item, dict)}
    assert "exploration" in domains
    assert "card_gear_farming" in domains
    assert "mercenary_homunculus" in domains
    assert "vending" in domains


def test_wave_a1_signal_domain_non_actionable_when_signal_missing() -> None:
    data_dir, tables_dir = _repo_paths()
    knowledge = load_ro_knowledge(data_dir=data_dir, tables_dir=tables_dir)

    assessment = knowledge.assess_opportunistic_upgrades(
        job_name="Novice",
        base_level=20,
        zeny=1000,
        inventory_items=[],
        market_listings=[],
        signals={
            "in_combat": False,
            "map_known": True,
            "nearby_hostiles": 0,
            "inventory_pressure": False,
            "market_listing_count": 0,
            "vendor_exposure": 0,
            "exploration_intent": False,
            "card_gear_farming_intent": False,
            "mercenary_homunculus_intent": False,
            "companion_available": False,
            "vending_intent": False,
            "overweight_ratio": 0.4,
        },
    )

    assert assessment["supported"] is True
    assert assessment["actionable"] is False
    reasons = [str(item) for item in assessment.get("non_actionable_reasons", [])]
    assert any("exploration_route_refresh_posture:signal<exploration_intent>_mismatch" in item for item in reasons)
    assert any("card_gear_farm_window:signal<card_gear_farming_intent>_mismatch" in item for item in reasons)
    assert any("mercenary_homunculus_support_posture:signal<mercenary_homunculus_intent>_mismatch" in item for item in reasons)
    assert any("vending_cycle_posture:signal<vending_intent>_mismatch" in item for item in reasons)
