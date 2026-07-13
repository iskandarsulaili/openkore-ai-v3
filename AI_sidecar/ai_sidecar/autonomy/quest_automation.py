"""
Quest & job change automation — multi-step NPC dialog chains.

Supports all first-class job change quests with full NPC interaction
sequences. Integrates with NPCDialogEngine and NPC discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QuestStep:
    """A single step in a multi-step quest or job change chain."""
    npc_name: str
    npc_map: str
    npc_coords: tuple[int, int]
    dialog_sequence: list[str]  # Response options to select
    requirement: str = ""  # e.g. "level>=10", "item=1000zeny"
    reward: str = ""


@dataclass(slots=True)
class QuestChain:
    """A complete multi-step quest or job change path."""
    name: str
    steps: list[QuestStep]
    job_change_to: str = ""  # Empty for quests, set for job change
    required_base_level: int = 1
    required_job_level: int = 0
    required_items: dict[str, int] = field(default_factory=dict)
    required_zeny: int = 0
    reward: str = ""


# ── JOB CHANGE CHAINS ──
# NPC names, maps, coords, and dialog sequences for first-class job changes.
# Coordinates are approximate for standard rAthena/loki servers.

JOB_CHANGE_CHAINS: dict[str, QuestChain] = {
    "swordman": QuestChain(
        name="Swordman Job Change",
        job_change_to="swordman",
        required_base_level=10,
        required_job_level=10,
        steps=[
            QuestStep("Swordman Guild Member", "prontera", (165, 95), ["1", "1", "1"]),
            QuestStep("Swordman Guild Member", "prontera", (165, 95), ["1", "2", "1", "1"]),
            QuestStep("Swordman Guild Member", "prontera", (165, 95), ["1", "2"]),
            QuestStep("Swordman Guild Member", "prontera", (165, 95), ["1"]),
        ],
    ),
    "mage": QuestChain(
        name="Mage Job Change",
        job_change_to="mage",
        required_base_level=10,
        required_job_level=10,
        steps=[
            QuestStep("Mage Guild Member", "geffen", (120, 85), ["1", "1", "1"]),
            QuestStep("Mage Guild Member", "geffen", (120, 85), ["1", "2", "1", "1"]),
            QuestStep("Mage Guild Member", "geffen", (120, 85), ["1", "2"]),
            QuestStep("Mage Guild Member", "geffen", (120, 85), ["1"]),
        ],
    ),
    "archer": QuestChain(
        name="Archer Job Change",
        job_change_to="archer",
        required_base_level=10,
        required_job_level=10,
        steps=[
            QuestStep("Archer Guild Member", "payon", (210, 120), ["1", "1", "1"]),
            QuestStep("Archer Guild Member", "payon", (210, 120), ["1", "2", "1"]),
            QuestStep("Archer Guild Member", "payon", (210, 120), ["1"]),
        ],
    ),
    "acolyte": QuestChain(
        name="Acolyte Job Change",
        job_change_to="acolyte",
        required_base_level=10,
        required_job_level=10,
        steps=[
            QuestStep("Acolyte Guild Member", "prontera", (255, 280), ["1", "1", "1"]),
            QuestStep("Acolyte Guild Member", "prontera", (255, 280), ["1", "2", "1"]),
            QuestStep("Acolyte Guild Member", "prontera", (255, 280), ["1", "2"]),
            QuestStep("Acolyte Guild Member", "prontera", (255, 280), ["1"]),
        ],
    ),
    "thief": QuestChain(
        name="Thief Job Change",
        job_change_to="thief",
        required_base_level=10,
        required_job_level=10,
        steps=[
            QuestStep("Thief Guild Member", "morocc", (150, 95), ["1", "1", "1"]),
            QuestStep("Thief Guild Member", "morocc", (150, 95), ["1", "2", "1"]),
            QuestStep("Thief Guild Member", "morocc", (150, 95), ["1"]),
        ],
    ),
    "merchant": QuestChain(
        name="Merchant Job Change",
        job_change_to="merchant",
        required_base_level=10,
        required_job_level=10,
        steps=[
            QuestStep("Merchant Guild Member", "aldebaran", (140, 130), ["1", "1", "1"]),
            QuestStep("Merchant Guild Member", "aldebaran", (140, 130), ["1", "2", "1"]),
            QuestStep("Merchant Guild Member", "aldebaran", (140, 130), ["1"]),
        ],
    ),
}


# ── QUEST CHAINS ──
QUEST_CHAINS: dict[str, QuestChain] = {
    "poring_quest": QuestChain(
        name="Poring Hunt Quest",
        steps=[
            QuestStep("Poring Collector", "prontera", (170, 230), ["1", "1", "1"]),
            QuestStep("Poring Collector", "prontera", (170, 230), ["1", "2"]),
        ],
        reward="poring_card_box",
    ),
}


class QuestAutomation:
    """Handles multi-step quest chains and job change automation."""

    def __init__(self):
        self._active_quests: dict[str, str] = {}  # bot_id -> quest_name
        self._quest_progress: dict[str, int] = {}  # bot_id -> step_index
        self._completed_jobs: set[str] = set()  # bot_id that completed job change

    def get_available_quests(self, bot_id: str, base_level: int, job_level: int, current_job: str, zeny: int) -> list[dict[str, Any]]:
        """Get quests/job changes the bot qualifies for."""
        available: list[dict[str, Any]] = []

        # Check job changes
        for job_name, chain in JOB_CHANGE_CHAINS.items():
            if current_job != "novice":
                continue
            if base_level < chain.required_base_level:
                continue
            if job_level < chain.required_job_level:
                continue
            if chain.required_zeny > zeny:
                continue
            available.append({
                "kind": "job_change",
                "target": chain.job_change_to,
                "name": chain.name,
                "quest_name": f"job_{job_name}",
                "steps": len(chain.steps),
            })

        # Check other quests
        for qname, chain in QUEST_CHAINS.items():
            if qname not in self._active_quests.get(bot_id, ""):
                available.append({
                    "kind": "quest",
                    "target": chain.name,
                    "name": chain.name,
                    "quest_name": qname,
                    "steps": len(chain.steps),
                })

        return available

    def start_quest(self, bot_id: str, quest_name: str) -> str | None:
        """Start a quest or job change. Returns the first command to execute."""
        if quest_name.startswith("job_"):
            job = quest_name.replace("job_", "")
            chain = JOB_CHANGE_CHAINS.get(job)
        else:
            chain = QUEST_CHAINS.get(quest_name)

        if chain is None:
            return None

        self._active_quests[bot_id] = quest_name
        self._quest_progress[bot_id] = 0

        # First step: move to the NPC
        step = chain.steps[0]
        return f"move {step.npc_map}"

    def get_next_command(self, bot_id: str, current_map: str) -> str | None:
        """Get the next command for an active quest."""
        quest_name = self._active_quests.get(bot_id)
        if quest_name is None:
            return None

        chain = QUEST_CHAINS if not quest_name.startswith("job_") else JOB_CHANGE_CHAINS
        if quest_name.startswith("job_"):
            job = quest_name.replace("job_", "")
            chain = JOB_CHANGE_CHAINS.get(job)
        else:
            chain = QUEST_CHAINS.get(quest_name)

        if chain is None:
            self.end_quest(bot_id)
            return None

        step_idx = self._quest_progress.get(bot_id, 0)
        if step_idx >= len(chain.steps):
            self.end_quest(bot_id)
            return None

        step = chain.steps[step_idx]

        # If not on the right map, move there
        if current_map != step.npc_map:
            return f"move {step.npc_map}"

        # Walk to NPC coordinates
        return f"move {step.npc_map} {step.npc_coords[0]} {step.npc_coords[1]}"

    def get_dialog_commands(self, bot_id: str) -> list[str] | None:
        """Get the dialog sequence for the current step."""
        quest_name = self._active_quests.get(bot_id)
        if quest_name is None:
            return None

        chain = QUEST_CHAINS if not quest_name.startswith("job_") else JOB_CHANGE_CHAINS
        if quest_name.startswith("job_"):
            job = quest_name.replace("job_", "")
            chain = JOB_CHANGE_CHAINS.get(job)
        else:
            chain = QUEST_CHAINS.get(quest_name)

        if chain is None:
            return None

        step_idx = self._quest_progress.get(bot_id, 0)
        if step_idx >= len(chain.steps):
            return None

        return chain.steps[step_idx].dialog_sequence

    def advance_step(self, bot_id: str) -> bool:
        """Advance to the next step. Returns True if quest is complete."""
        quest_name = self._active_quests.get(bot_id)
        if quest_name is None:
            return True

        step_idx = self._quest_progress.get(bot_id, 0) + 1

        chain = QUEST_CHAINS if not quest_name.startswith("job_") else JOB_CHANGE_CHAINS
        if quest_name.startswith("job_"):
            job = quest_name.replace("job_", "")
            chain = JOB_CHANGE_CHAINS.get(job)
        else:
            chain = QUEST_CHAINS.get(quest_name)

        if chain is None or step_idx >= len(chain.steps):
            self._completed_jobs.add(bot_id)
            self.end_quest(bot_id)
            return True

        self._quest_progress[bot_id] = step_idx
        return False

    def end_quest(self, bot_id: str) -> None:
        """End the current quest."""
        self._active_quests.pop(bot_id, None)
        self._quest_progress.pop(bot_id, None)

    def has_completed_job_change(self, bot_id: str) -> bool:
        return bot_id in self._completed_jobs
