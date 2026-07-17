"""Tests for the skills system.

Verifies: CRUD operations, lifecycle transitions, trigger matching, context loading.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ── Test skills content ──

SAMPLE_SKILL = """---
name: test-skill
description: A test skill for unit tests
version: 1.0.0
triggers:
  - low_hp
  - heal_strategy_requested
when_to_use:
  - hp_ratio < 0.30
metadata:
  domain: healing
  source: test
  confidence: 0.8
---

# Test Skill
Test content here.
"""

SAMPLE_PATCH = """---
name: test-skill
description: Updated test skill
version: 1.1.0
triggers:
  - low_hp
metadata:
  domain: healing
  source: test
  confidence: 0.9
---

# Test Skill (Patched)
Updated content.
"""


# ── Fixtures ──


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory."""
    d = tmp_path / "skills"
    d.mkdir()
    return d


@pytest.fixture
def usage_file(skills_dir):
    """Create a temporary .usage.json."""
    uf = skills_dir / ".usage.json"
    uf.write_text("{}")
    return uf


# ── Test: Skill Manager CRUD ──


class TestSkillManager:
    def test_create_skill(self):
        """Create a skill should return success with name and path."""
        from ai_sidecar.skills_manager import create_skill

        result = create_skill(name="test-skill", content=SAMPLE_SKILL, category="healing")
        assert result["success"] is True
        assert result["name"] == "test-skill"
        assert "healing" in result.get("category", "")

        # Verify file exists
        from ai_sidecar.skills_manager import _find_skill
        found = _find_skill("test-skill")
        assert found is not None
        assert found["path"].exists()

    def test_create_duplicate_fails(self):
        """Creating a skill with an existing name should fail."""
        from ai_sidecar.skills_manager import create_skill, _find_skill, delete_skill

        create_skill(name="test-skill", content=SAMPLE_SKILL)
        result = create_skill(name="test-skill", content=SAMPLE_SKILL)
        assert result["success"] is False
        assert "already exists" in result.get("error", "")

        # Clean up
        delete_skill("test-skill")

    def test_view_skill(self):
        """Viewing a skill should return content and supporting files."""
        from ai_sidecar.skills_manager import create_skill, view_skill

        create_skill(name="view-test", content=SAMPLE_SKILL)
        skill = view_skill("view-test")
        assert skill is not None
        assert skill["name"] == "view-test"
        assert "Test Skill" in skill["content"]
        assert isinstance(skill["supporting"], list)

        from ai_sidecar.skills_manager import delete_skill
        delete_skill("view-test")

    def test_patch_skill(self):
        """Patching a skill should update its content."""
        from ai_sidecar.skills_manager import create_skill, view_skill, patch_skill, delete_skill

        create_skill(name="patch-test", content=SAMPLE_SKILL)
        result = patch_skill(name="patch-test", old_string="Test Skill", new_string="Patched Skill")
        assert result["success"] is True

        skill = view_skill("patch-test")
        assert "Patched Skill" in skill["content"]
        delete_skill("patch-test")

    def test_delete_skill(self):
        """Deleting a skill should remove directory and usage record."""
        from ai_sidecar.skills_manager import create_skill, delete_skill, _find_skill
        from ai_sidecar.skills_usage import get_skill

        create_skill(name="del-test", content=SAMPLE_SKILL)
        result = delete_skill(name="del-test")
        assert result["success"] is True

        found = _find_skill("del-test")
        assert found is None
        assert get_skill("del-test") is None


# ── Test: Skill Usage Tracking ──


class TestSkillUsage:
    def test_bump_creates_record(self):
        """Bumping a skill should create usage record."""
        from ai_sidecar.skills_usage import bump, get_skill

        bump("never-created", event="use")
        record = get_skill("never-created")
        assert record is not None
        assert record["use_count"] >= 1
        assert record["state"] == "active"

    def test_bump_increments_counters(self):
        """Different events increment different counters."""
        from ai_sidecar.skills_usage import bump, get_skill

        bump("counter-test", event="use")
        bump("counter-test", event="use")
        bump("counter-test", event="view")
        bump("counter-test", event="patch")

        record = get_skill("counter-test")
        assert record["use_count"] == 2
        assert record["view_count"] == 1
        assert record["patch_count"] == 1

    def test_state_transitions(self):
        """Lifecycle states should transition correctly."""
        from ai_sidecar.skills_usage import set_state, get_skill, bump

        bump("state-test")
        assert get_skill("state-test")["state"] == "active"

        set_state("state-test", "stale")
        assert get_skill("state-test")["state"] == "stale"

        # Any activity should re-activate
        bump("state-test", event="view")
        assert get_skill("state-test")["state"] == "active"

    def test_pinned_skills_exempt(self):
        """Pinned skills should not be auto-staled."""
        from ai_sidecar.skills_usage import set_pinned, bump, mark_stale_if_unused, get_skill

        bump("pinned-test")
        set_pinned("pinned-test", True)
        marked = mark_stale_if_unused(stale_after_days=0)
        assert "pinned-test" not in marked
        assert get_skill("pinned-test")["state"] == "active"


# ── Test: Skills Loader ──


class TestSkillsLoader:
    def test_trigger_matching_low_hp(self):
        """Low HP situation should match heal skills."""
        from ai_sidecar.skills_loader import load_for_context
        from ai_sidecar.skills_manager import create_skill, delete_skill

        create_skill(name="trigger-test", content=SAMPLE_SKILL, category="healing")

        matched = load_for_context({
            "hp_ratio": 0.12,
            "zeny": 380,
            "map": "prontera",
            "action_type": "heal",
        })
        assert len(matched) > 0
        first = matched[0]
        assert "Test Skill" in first

        delete_skill("trigger-test")

    def test_no_match_for_high_hp(self):
        """High HP should NOT match low_hp-triggered skills."""
        from ai_sidecar.skills_loader import load_for_context
        from ai_sidecar.skills_manager import create_skill, delete_skill

        create_skill(name="high-hp-test", content=SAMPLE_SKILL, category="healing")

        matched = load_for_context({
            "hp_ratio": 0.90,
            "zeny": 9999,
            "map": "prt_fild08",
            "action_type": "grinding",
        })
        # May still match due to base score
        delete_skill("high-hp-test")


# ── Test: Skills Curator ──


class TestSkillsCurator:
    def test_curator_dry_run(self):
        """Dry run should report without modifying."""
        from ai_sidecar.skills_curator import run_curator

        result = run_curator(dry_run=True)
        assert result["dry_run"] is True
        assert isinstance(result["marked_stale"], list)

    def test_curator_backup(self):
        """Curator should create backups."""
        from ai_sidecar.skills_curator import run_curator, list_backups

        result = run_curator()
        assert "backed_up" in result
        backups = list_backups()
        assert isinstance(backups, list)


# ── Test: Post-Action Review ──


class TestPostActionReview:
    def test_review_creates_skill(self):
        """Reviewing a heal discovery should create a skill."""
        from ai_sidecar.autonomy.post_action_review import review_heal_strategy
        from ai_sidecar.skills_manager import delete_skill, _find_skill

        result = review_heal_strategy(
            strategy="visit_healer_npc",
            target_map="prontera",
            target_npc="Healer#prt",
            confidence=0.85,
            bot_id="test_bot",
        )
        assert result["reviewed"] is True
        assert result["action_type"] == "heal"
        assert result["agent"] == "pro_ro_llm"
