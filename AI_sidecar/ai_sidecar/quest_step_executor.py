"""
Quest Step Executor — reads quest steps, navigates to required NPCs/maps,
interacts with NPCs, kills target monsters, collects items, and returns
for rewards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class QuestExecutionState:
    """Current state of quest execution."""
    quest_id: int = 0
    quest_name: str = ""
    current_step: int = 0
    total_steps: int = 0
    is_active: bool = False
    started_at: float = 0.0
    last_action: str = ""
    progress_pct: float = 0.0


class QuestStepExecutor:
    """Executes quest steps automatically."""

    # NPC locations for quest-related NPCs
    NPC_LOCATIONS: dict[str, tuple[str, int, int]] = {
        "eden_recruiter": ("izlude", 120, 80),
        "eden_equipment": ("izlude", 125, 85),
        "ice_researcher": ("lute", 50, 50),
        "thanatos_researcher": ("rachel", 100, 100),
        "bio_scientist": ("lighthalzen", 150, 150),
        "ninja_master": ("amatsu", 80, 80),
        "kunlun_master": ("kunlun", 60, 60),
        "eden_daily": ("izlude", 130, 90),
    }

    # Monster locations for quest targets
    MONSTER_LOCATIONS: dict[str, tuple[str, int, int]] = {
        "Ice Titan": ("ice_dun01", 100, 100),
        "Thanatos Fragment": ("thanatos_dun01", 50, 50),
        "Bio Monster": ("bio_dun01", 100, 100),
        "Training Dummy": ("amatsu_dun01", 50, 50),
        "Martial Artist": ("kunlun_dun01", 50, 50),
        "Daily Target": ("payon_dun01", 100, 100),
        "Daily Material": ("geffen_dun01", 100, 100),
    }

    def __init__(self) -> None:
        self._lock = RLock()
        self._state: QuestExecutionState = QuestExecutionState()
        self._enqueue_fn: Callable | None = None

    # ── Public API ──

    def start_quest(self, quest_id: int, quest_name: str, total_steps: int) -> None:
        """Start executing a quest."""
        with self._lock:
            self._state = QuestExecutionState(
                quest_id=quest_id,
                quest_name=quest_name,
                current_step=0,
                total_steps=total_steps,
                is_active=True,
                started_at=time.time(),
            )
            logger.info("quest_execution_started: %s (%d steps)", quest_name, total_steps)

    def get_next_action(self, step_description: str, step_action: str, step_target: str) -> str | None:
        """Get the next action for the current quest step."""
        with self._lock:
            if not self._state.is_active:
                return None

            if step_action == "talk":
                loc = self.NPC_LOCATIONS.get(step_target)
                if loc:
                    self._state.last_action = f"move_to_npc_{step_target}"
                    return f"move {loc[0]} {loc[1]} {loc[2]}"
                return f"talk {step_target}"

            elif step_action == "kill":
                loc = self.MONSTER_LOCATIONS.get(step_target)
                if loc:
                    self._state.last_action = f"move_to_kill_{step_target}"
                    return f"move {loc[0]} {loc[1]} {loc[2]}"
                return f"attack {step_target}"

            elif step_action == "collect":
                loc = self.MONSTER_LOCATIONS.get(step_target)
                if loc:
                    self._state.last_action = f"move_to_collect_{step_target}"
                    return f"move {loc[0]} {loc[1]} {loc[2]}"
                return f"loot {step_target}"

            elif step_action == "deliver":
                loc = self.NPC_LOCATIONS.get(step_target)
                if loc:
                    self._state.last_action = f"move_to_deliver_{step_target}"
                    return f"move {loc[0]} {loc[1]} {loc[2]}"
                return f"talk {step_target}"

            elif step_action == "move":
                return f"move {step_target}"

            return None

    def advance_step(self) -> None:
        """Advance to the next quest step."""
        with self._lock:
            self._state.current_step += 1
            if self._state.total_steps > 0:
                self._state.progress_pct = (self._state.current_step / self._state.total_steps) * 100
            if self._state.current_step >= self._state.total_steps:
                self._state.is_active = False
                logger.info("quest_execution_completed: %s", self._state.quest_name)

    def get_state(self) -> QuestExecutionState:
        with self._lock:
            return self._state

    def is_active(self) -> bool:
        with self._lock:
            return self._state.is_active

    def get_execution_summary(self) -> str:
        with self._lock:
            if not self._state.is_active:
                return "No active quest execution"
            return (
                f"── Quest Execution ──\n"
                f"Quest: {self._state.quest_name}\n"
                f"Step: {self._state.current_step}/{self._state.total_steps}\n"
                f"Progress: {self._state.progress_pct:.0f}%\n"
                f"Last action: {self._state.last_action}"
            )

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._state = QuestExecutionState()


# ── Global Singleton ──

_quest_exec: QuestStepExecutor | None = None
_quest_exec_lock = RLock()


def get_quest_step_executor() -> QuestStepExecutor:
    global _quest_exec
    with _quest_exec_lock:
        if _quest_exec is None:
            _quest_exec = QuestStepExecutor()
        return _quest_exec
