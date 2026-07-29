"""Quest automation domain."""
from __future__ import annotations

from ai_sidecar.domains.quests.tracker import QuestTracker
from ai_sidecar.domains.quests.automation import QuestAutomation
from ai_sidecar.domains.quests.daily import DailyQuestManager

__all__ = [
    "QuestDomain",
    "QuestTracker",
    "QuestAutomation",
    "DailyQuestManager",
]


class QuestDomain:
    """Aggregate domain for all quest tracking and automation."""

    name = "quests"
    priority = 50

    def __init__(self) -> None:
        self.tracker = QuestTracker()
        self.automation = QuestAutomation(tracker=self.tracker)
        self.daily = DailyQuestManager()
