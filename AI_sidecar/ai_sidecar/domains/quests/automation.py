"""Auto-accept and auto-complete quests based on level and location."""
from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.domains.quests.tracker import QuestTracker, QuestState, QuestObjective

logger = logging.getLogger(__name__)

# Quest level brackets — which quest lines unlock at which levels
_QUEST_LINES: dict[str, dict[str, Any]] = {
    "novice_grounds": {
        "min_level": 1,
        "max_level": 15,
        "quests": [
            {"id": "novice_1", "name": "Novice Training", "npc": "Novice Guide"},
            {"id": "novice_2", "name": "Guild Apprentice", "npc": "Guild Member"},
        ],
    },
    "early_adventure": {
        "min_level": 10,
        "max_level": 30,
        "quests": [
            {"id": "weapon_quest_1", "name": "First Weapon", "npc": "Weapons Dealer"},
            {"id": "potions_101", "name": "Potion Basics", "npc": "Apothecary"},
        ],
    },
    "mid_game": {
        "min_level": 30,
        "max_level": 60,
        "quests": [
            {"id": "job_advancement", "name": "Job Advancement", "npc": "Job Master"},
            {"id": "dungeon_intro", "name": "Dungeon Explorer", "npc": "Adventurer"},
        ],
    },
    "end_game": {
        "min_level": 60,
        "max_level": 99,
        "quests": [
            {"id": "heroic_trial", "name": "Heroic Trial", "npc": "Hero"},
            {"id": "mvp_hunt", "name": "MVP Hunt", "npc": "Hunt Master"},
        ],
    },
}


class QuestAutomation:
    """Automatically accept and complete quests."""

    def __init__(self, tracker: QuestTracker | None = None, db: Any = None) -> None:
        self.tracker = tracker or QuestTracker()
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()
        self._accepted_quest_ids: set[str] = set()

    def assess_quest_opportunities(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if there are quests to pick up or complete.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        base_level = int(signals.get("base_level", 1) or 1)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        inventory = signals.get("inventory", []) or []
        zeny = int(signals.get("zeny", 0) or 0)

        # Active quests we can progress
        active = self.tracker.get_active_quests(bot_id)

        # Check for quest completion (objectives met, haven't turned in)
        completable = self._find_completable_quests(active, inventory)
        for qs in completable:
            actions.append({
                "type": "complete_quest",
                "priority": "high",
                "reason": f"Quest '{qs.quest_name}' objectives met — turn in at {qs.npc_complete}",
                "quest_id": qs.quest_id,
                "npc": qs.npc_complete,
            })

        # Check for new quests to accept
        available = self._find_available_quests(base_level, bot_id)
        for quest_info in available:
            if quest_info["id"] not in self._accepted_quest_ids:
                actions.append({
                    "type": "accept_quest",
                    "priority": "medium",
                    "reason": f"Quest '{quest_info['name']}' available at level {base_level}",
                    "quest_id": quest_info["id"],
                    "npc": quest_info.get("npc", ""),
                })

        # Check if we should focus on questing vs grinding
        if active and not completable:
            # Progress quest objectives
            for qs in active:
                incomplete = [o for o in qs.objectives if not o.completed]
                if incomplete:
                    obj = incomplete[0]
                    actions.append({
                        "type": "progress_quest",
                        "priority": "low",
                        "reason": f"Quest '{qs.quest_name}': {obj.description} ({obj.current}/{obj.required})",
                        "quest_id": qs.quest_id,
                        "objective": {
                            "target": obj.target,
                            "target_type": obj.target_type,
                            "current": obj.current,
                            "required": obj.required,
                        },
                    })

        return actions

    def accept_quest(self, quest_id: str, bot_id: str) -> tuple[str, str] | None:
        """Generate commands to accept a quest from an NPC.

        Returns (talk_sequence_command, npc_name) or None if unknown.
        """
        # Find the quest info
        for line_name, line_info in _QUEST_LINES.items():
            for q in line_info["quests"]:
                if q["id"] == quest_id:
                    # Start conversation with NPC
                    self._accepted_quest_ids.add(quest_id)
                    qs = QuestState(
                        quest_id=q["id"],
                        quest_name=q["name"],
                        status="active",
                        npc_start=q["npc"],
                        started_at=__import__("time").time(),
                    )
                    self.tracker.track_quest(bot_id, qs)
                    logger.info("[quests] %s: accepted quest '%s' from %s", bot_id, q["name"], q["npc"])
                    return (f"talk @{q['npc']}@", q["npc"])
        return None

    def complete_quest(self, quest_id: str, bot_id: str) -> str | None:
        """Generate command to turn in a completed quest.

        Returns the talk command to complete the quest, or None.
        """
        qs = self.tracker.get_quest(bot_id, quest_id)
        if not qs:
            logger.warning("[quests] %s: no quest state for %s", bot_id, quest_id)
            return None

        self.tracker.complete_quest(bot_id, quest_id)
        return f"talk @{qs.npc_complete}@" if qs.npc_complete else None

    def get_quest_npc(self, quest_id: str) -> str:
        """Get the NPC name associated with a quest."""
        for line_name, line_info in _QUEST_LINES.items():
            for q in line_info["quests"]:
                if q["id"] == quest_id:
                    return q.get("npc", "")
        return ""

    def _find_completable_quests(
        self,
        active_quests: list[QuestState],
        inventory: list[dict],
    ) -> list[QuestState]:
        """Find quests where all objectives are met and can be turned in."""
        completable: list[QuestState] = []
        inventory_names = set()
        for item in inventory:
            name = (item.get("name", "") or "").lower()
            inventory_names.add(name)

        for qs in active_quests:
            if not qs.objectives:
                continue
            all_done = True
            for obj in qs.objectives:
                if not obj.completed:
                    # Check if objective is "collect X item" and we have it
                    if obj.target_type == "collect" and obj.target.lower() in inventory_names:
                        obj.completed = True
                    else:
                        all_done = False
            if all_done:
                completable.append(qs)
        return completable

    def _find_available_quests(self, base_level: int, bot_id: str) -> list[dict]:
        """Find quests available at this level that haven't been accepted yet."""
        available: list[dict] = []
        active = self.tracker.get_active_quests(bot_id)
        active_ids = {qs.quest_id for qs in active}

        for line_name, line_info in _QUEST_LINES.items():
            if line_info["min_level"] <= base_level <= line_info["max_level"]:
                for q in line_info["quests"]:
                    if q["id"] not in active_ids and q["id"] not in self._accepted_quest_ids:
                        available.append(q)

        return available[:3]  # Limit to 3 suggestions

    def should_quest_vs_grind(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> str:
        """Decide whether bot should quest or grind.

        Returns 'quest' or 'grind'.
        """
        active = self.tracker.get_active_quests(bot_id)
        if not active:
            # No quests -> go quest if we're in town
            return "quest"
        completable = self._find_completable_quests(
            active, signals.get("inventory", []) or [],
        )
        if completable:
            return "quest"
        return "grind"

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove quest automation state for a bot."""
        self._accepted_quest_ids.difference_update(
            qs.quest_id for qs in self.tracker.get_active_quests(bot_id)
        )
        self.tracker.cleanup_bot(bot_id)
