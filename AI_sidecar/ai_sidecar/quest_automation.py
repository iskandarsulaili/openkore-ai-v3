"""
Quest Automation Pipeline — prioritizes quests by reward value,
executes quest steps automatically, and tracks quest completion status.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class QuestStep:
    """A single step in a quest."""
    step_id: int = 0
    description: str = ""
    action: str = ""  # talk, kill, collect, deliver, move
    target: str = ""  # npc_name, monster_name, item_name, map_name
    quantity: int = 1
    completed: bool = False


@dataclass
class Quest:
    """A quest with its steps and rewards."""
    quest_id: int = 0
    name: str = ""
    quest_type: str = ""  # main, side, daily, repeatable
    min_level: int = 1
    max_level: int = 99
    steps: list[QuestStep] = field(default_factory=list)
    rewards: dict[str, int] = field(default_factory=dict)  # stat -> value
    zeny_reward: int = 0
    xp_reward: int = 0
    item_rewards: list[str] = field(default_factory=list)
    priority: int = 50
    is_completed: bool = False
    is_active: bool = False
    notes: str = ""


class QuestAutomation:
    """Automates quest execution."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._quests: dict[int, Quest] = {}
        self._active_quest: Quest | None = None
        self._completed_quests: set[int] = set()
        self._enqueue_fn: Callable | None = None
        self._load_important_quests()

    def _load_important_quests(self) -> None:
        """Load quests from the knowledge database."""
        try:
            from ai_sidecar.knowledge_loader import get_quests
            db_quests = get_quests()
            for q in db_quests:
                qid = q.get("Id", 0)
                title = q.get("Title", f"Quest_{qid}")
                level = q.get("Level", 1)
                if qid and title:
                    self._quests[qid] = Quest(
                        quest_id=qid,
                        name=title,
                        quest_type="side",
                        min_level=max(1, level - 10),
                        max_level=level + 10,
                        priority=50,
                        notes=f"From knowledge DB: {title}",
                    )
            logger.info("quests_loaded_from_db: %d quests", len(self._quests))
        except Exception as e:
            logger.warning("quests_db_load_failed: %s (DB is the source of truth)", e)

    # ── Public API ──

    def get_quest(self, quest_id: int) -> Quest | None:
        with self._lock:
            return self._quests.get(quest_id)

    def get_available_quests(self, level: int) -> list[Quest]:
        """Get quests available for a given level."""
        with self._lock:
            return [
                q for q in self._quests.values()
                if q.min_level <= level <= q.max_level
                and q.quest_id not in self._completed_quests
                and not q.is_completed
            ]

    def get_best_quest(self, level: int) -> Quest | None:
        """Get the best quest to do right now."""
        with self._lock:
            available = self.get_available_quests(level)
            if not available:
                return None
            available.sort(key=lambda q: -q.priority)
            return available[0]

    def start_quest(self, quest_id: int) -> bool:
        """Start a quest."""
        with self._lock:
            quest = self._quests.get(quest_id)
            if not quest or quest.is_completed:
                return False
            quest.is_active = True
            self._active_quest = quest
            logger.info("quest_started: %s", quest.name)
            return True

    def complete_step(self, quest_id: int, step_id: int) -> None:
        """Mark a quest step as completed."""
        with self._lock:
            quest = self._quests.get(quest_id)
            if not quest:
                return
            for step in quest.steps:
                if step.step_id == step_id:
                    step.completed = True
                    break
            # Check if all steps are completed
            if all(s.completed for s in quest.steps):
                quest.is_completed = True
                quest.is_active = False
                self._completed_quests.add(quest_id)
                self._active_quest = None
                logger.info("quest_completed: %s", quest.name)

    def get_quest_summary(self) -> str:
        with self._lock:
            lines = [f"── Quest Automation ──"]
            lines.append(f"Total quests: {len(self._quests)}")
            lines.append(f"Completed: {len(self._completed_quests)}")
            if self._active_quest:
                q = self._active_quest
                lines.append(f"Active: {q.name}")
                completed = sum(1 for s in q.steps if s.completed)
                lines.append(f"Progress: {completed}/{len(q.steps)} steps")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._quests.clear()
            self._active_quest = None
            self._completed_quests.clear()
            self._load_important_quests()


# ── Global Singleton ──

_quest_auto: QuestAutomation | None = None
_quest_auto_lock = RLock()


def get_quest_automation() -> QuestAutomation:
    global _quest_auto
    with _quest_auto_lock:
        if _quest_auto is None:
            _quest_auto = QuestAutomation()
        return _quest_auto
