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
        """Load important quests with permanent stat rewards."""
        quests = [
            # Eden Group quests
            Quest(1, "Eden Group Registration", "main", 1, 99, priority=90,
                  rewards={"all_stats": 5}, zeny_reward=10000, xp_reward=50000,
                  notes="Permanent stat boost, do first",
                  steps=[QuestStep(1, "Talk to Eden Group NPC", "talk", "eden_recruiter")]),
            Quest(2, "Eden Group Equipment Quest", "main", 12, 99, priority=85,
                  rewards={"all_stats": 3}, zeny_reward=5000,
                  notes="Free equipment set",
                  steps=[QuestStep(1, "Talk to Eden Equipment NPC", "talk", "eden_equipment")]),

            # Ice Cave quests
            Quest(10, "Ice Cave Investigation", "side", 50, 99, priority=80,
                  rewards={"int": 5}, zeny_reward=50000, xp_reward=200000,
                  notes="+5 INT permanent",
                  steps=[
                      QuestStep(1, "Talk to researcher in Lutie", "talk", "ice_researcher"),
                      QuestStep(2, "Kill Ice Titans", "kill", "Ice Titan", 30),
                      QuestStep(3, "Report back", "talk", "ice_researcher"),
                  ]),

            # Thanatos Tower quests
            Quest(20, "Thanatos Tower Access", "side", 70, 99, priority=75,
                  rewards={"str": 3, "int": 3}, zeny_reward=100000, xp_reward=500000,
                  notes="MVP access + stat bonuses",
                  steps=[
                      QuestStep(1, "Talk to researcher in Rachel", "talk", "thanatos_researcher"),
                      QuestStep(2, "Collect Thanatos fragments", "collect", "Thanatos Fragment", 10),
                      QuestStep(3, "Report back", "talk", "thanatos_researcher"),
                  ]),

            # Bio Labs quests
            Quest(30, "Bio Labs Access", "side", 75, 99, priority=80,
                  rewards={"all_stats": 5}, zeny_reward=200000, xp_reward=1000000,
                  notes="Equipment + stat bonuses",
                  steps=[
                      QuestStep(1, "Talk to scientist in Lighthalzen", "talk", "bio_scientist"),
                      QuestStep(2, "Kill Bio monsters", "kill", "Bio Monster", 50),
                      QuestStep(3, "Collect samples", "collect", "Bio Sample", 20),
                      QuestStep(4, "Report back", "talk", "bio_scientist"),
                  ]),

            # Amatsu quests
            Quest(40, "Amatsu Ninja Training", "side", 60, 99, priority=70,
                  rewards={"agi": 3, "dex": 3}, zeny_reward=80000, xp_reward=300000,
                  notes="Stat bonuses + ninja skills",
                  steps=[
                      QuestStep(1, "Talk to ninja master in Amatsu", "talk", "ninja_master"),
                      QuestStep(2, "Complete training", "kill", "Training Dummy", 20),
                      QuestStep(3, "Report back", "talk", "ninja_master"),
                  ]),

            # Kunlun quests
            Quest(50, "Kunlun Martial Arts", "side", 55, 99, priority=70,
                  rewards={"str": 3, "vit": 3}, zeny_reward=60000, xp_reward=250000,
                  notes="Stat bonuses",
                  steps=[
                      QuestStep(1, "Talk to martial arts master", "talk", "kunlun_master"),
                      QuestStep(2, "Complete trials", "kill", "Martial Artist", 30),
                      QuestStep(3, "Report back", "talk", "kunlun_master"),
                  ]),

            # Repeatable daily quests
            Quest(100, "Eden Daily: Monster Hunting", "daily", 40, 99, priority=60,
                  rewards={"all_stats": 1}, zeny_reward=30000, xp_reward=100000,
                  notes="Repeatable daily, good XP",
                  steps=[
                      QuestStep(1, "Accept daily from Eden", "talk", "eden_daily"),
                      QuestStep(2, "Kill target monsters", "kill", "Daily Target", 100),
                      QuestStep(3, "Report back", "talk", "eden_daily"),
                  ]),
            Quest(101, "Eden Daily: Material Collection", "daily", 40, 99, priority=55,
                  zeny_reward=20000, xp_reward=80000,
                  notes="Repeatable daily, easy materials",
                  steps=[
                      QuestStep(1, "Accept daily from Eden", "talk", "eden_daily"),
                      QuestStep(2, "Collect materials", "collect", "Daily Material", 50),
                      QuestStep(3, "Report back", "talk", "eden_daily"),
                  ]),
        ]

        for q in quests:
            self._quests[q.quest_id] = q

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
