"""
OnboardingService — gets a brand-new character from spawn to hunting.

Handles:
1. Stat point allocation (from DB-backed build plans)
2. Skill training (Basic Skill first for weight)
3. First NPC interactions (Novice NPC for free stuff)
4. Route to first hunting zone (appropriate for level)
5. Everything is data-driven from GameKnowledgeDB — zero hardcoding
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from ai_sidecar.game_knowledge_db import GameKnowledgeDB

logger = logging.getLogger(__name__)

# -- Steps a new character goes through (in order) --
_STEPS = [
    "allocate_stats",      # spend all stat points
    "train_skills",        # train Basic Skill and build-essential skills
    "route_to_hunt",       # move to level-appropriate hunting zone
]


class OnboardingService:
    """Manages the new-character setup flow. Adaptive per-server via DB."""

    def __init__(self, db: GameKnowledgeDB | None = None) -> None:
        self.db = db or GameKnowledgeDB()
        self._completed_steps: dict[str, set[str]] = {}  # bot_id -> completed steps

    def get_completed(self, bot_id: str) -> set[str]:
        return self._completed_steps.get(bot_id, set())

    def mark_completed(self, bot_id: str, step: str) -> None:
        if bot_id not in self._completed_steps:
            self._completed_steps[bot_id] = set()
        self._completed_steps[bot_id].add(step)
        logger.info("onboarding_completed_step: bot=%s step=%s", bot_id, step)

    def is_complete(self, bot_id: str) -> bool:
        return self.get_completed(bot_id) >= set(_STEPS)

    def next_step(self, bot_id: str) -> str | None:
        """Return the next uncompleted step, or None if fully onboarded."""
        done = self.get_completed(bot_id)
        for step in _STEPS:
            if step not in done:
                return step
        return None

    def evaluate(self, bot_id: str, snapshot: Any) -> list[dict]:
        """
        Check what the new bot needs to do next.

        Returns a list of action dicts (commands to execute).
        """
        if not snapshot:
            return []

        base_level = getattr(snapshot, "base_level", 1) or 1
        job_name = getattr(snapshot, "job_name", "novice") or "novice"
        current_stats = {
            "str": getattr(snapshot, "str", 1) or 1,
            "agi": getattr(snapshot, "agi", 1) or 1,
            "vit": getattr(snapshot, "vit", 1) or 1,
            "int": getattr(snapshot, "int", 1) or 1,
            "dex": getattr(snapshot, "dex", 1) or 1,
            "luk": getattr(snapshot, "luk", 1) or 1,
        }
        stat_points = getattr(snapshot, "stat_points", 0) or 0
        skill_points = getattr(snapshot, "skill_points", 0) or 0
        current_skills = getattr(snapshot, "skills", {}) or {}
        hp_pct = getattr(snapshot, "hp_pct", 100) or 100
        map_name = getattr(snapshot, "map_name", "prt_in") or "prt_in"
        zeny = getattr(snapshot, "zeny", 0) or 0
        weight_pct = getattr(snapshot, "weight_pct", 0) or 0

        actions = []
        next_step = self.next_step(bot_id)

        # ── Step 1: Allocate Stats ──
        if next_step == "allocate_stats" and stat_points > 0:
            allocation = self.db.allocate_stats(base_level, current_stats, job_name)
            if allocation:
                for stat, amount in allocation.items():
                    if amount > 0:
                        # stat: str, agi, vit, int, dex, luk
                        actions.append({
                            "action": "stat_add",
                            "stat": stat.upper(),
                            "points": min(amount, stat_points),
                            "reason": f"allocate {amount} to {stat} per build plan",
                            "priority": 1,
                        })
            if not actions:
                # No stats to allocate — mark done
                self.mark_completed(bot_id, "allocate_stats")

        # ── Step 2: Train Skills ──
        if next_step == "train_skills" and skill_points > 0:
            next_skill = self.db.get_next_skill(job_name, current_skills)
            if next_skill:
                skill_id, target_lv = next_skill
                actions.append({
                    "action": "skill_add",
                    "skill_id": skill_id,
                    "reason": f"train {skill_id} to level {target_lv} per build plan",
                    "priority": 2,
                })
            if not actions:
                self.mark_completed(bot_id, "train_skills")

        # ── Step 3: Route to Hunting Zone ──
        if next_step == "route_to_hunt":
            # Use DB-backed optimization for best map
            target_map = self.db.optimize_hunting_map(bot_id, base_level, {map_name})
            if target_map and target_map != map_name:
                actions.append({
                    "action": "move",
                    "target_map": target_map,
                    "reason": f"Route to {target_map} for level {base_level} grinding",
                    "priority": 5,
                    "confidence": 0.95,
                })
            elif target_map == map_name:
                # Already on correct map — onboarding is complete
                self.mark_completed(bot_id, "route_to_hunt")
            else:
                # No hunting zone found — skip
                self.mark_completed(bot_id, "route_to_hunt")

        return actions

    def should_cold_start(self, bot_id: str, snapshot: Any) -> bool:
        """Check if the bot is still in onboarding (pre-cold-start) state."""
        if not snapshot or not bot_id:
            return True
        return not self.is_complete(bot_id)


# ── Fallback hardcoded defaults (server-agnostic, only if DB misses) ──
# These should never be needed if DB has proper seed data.
FALLBACK_NOVICE_NPC = {
    "map": "prontera", "x": 243, "y": 124,
    "name": "Novice NPC",
    "steps": ["c", "r0", "c", "r1", "c", "r1", "c", "r0"]
}

FALLBACK_HEALER = {
    "map": "prontera", "x": 157, "y": 195,
    "name": "Healer NPC",
    "steps": ["c", "r0", "c"]
}
