"""QuestState — active quests, progress, objectives."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QuestObjective(BaseModel):
    """A single objective within a quest."""

    model_config = ConfigDict(extra="ignore")

    objective_id: str = ""
    description: str = ""
    status: str = "unknown"  # inactive | active | completed
    current: int = 0
    target: int = 0

    @property
    def is_complete(self) -> bool:
        return self.target > 0 and self.current >= self.target

    @property
    def progress_pct(self) -> float:
        if self.target <= 0:
            return 0.0
        return min(1.0, self.current / self.target)


class QuestEntry(BaseModel):
    """An active or completed quest."""

    model_config = ConfigDict(extra="ignore")

    quest_id: str = ""
    title: str = ""
    state: str = "active"  # active | inactive | completed | inactive
    level: int = 0
    npc_name: str | None = None
    objectives: list[QuestObjective] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return bool(
            self.state == "completed"
            or (self.objectives and all(o.is_complete for o in self.objectives))
        )

    @property
    def is_active(self) -> bool:
        return self.state in ("active",) and not self.is_complete


class QuestState(BaseModel):
    """Current quest state — all active and completed quests."""

    model_config = ConfigDict(extra="ignore")

    active_quests: list[QuestEntry] = Field(default_factory=list)
    completed_quests: list[QuestEntry] = Field(default_factory=list)
    total_active: int = 0
    total_completed: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_quests(signals: dict[str, Any]) -> QuestState:
    """Parse quest state from the bridge signal dict.

    Handles:
      - ``signals['quests']`` — list of quest dicts
      - ``signals['quests.active']`` / ``signals['quests.completed']`` — separated
    """
    raw_quests: list[dict] = list(signals.get("quests", []) or [])

    active: list[QuestEntry] = []
    completed: list[QuestEntry] = []

    for raw in raw_quests:
        if not isinstance(raw, dict):
            continue

        state = str(raw.get("state", "active"))
        objectives_raw: list[dict] = list(raw.get("objectives", []) or [])
        objectives = [
            QuestObjective(
                objective_id=str(o.get("objective_id", o.get("id", ""))),
                description=str(o.get("description", "")),
                status=str(o.get("status", "unknown")),
                current=int(o.get("current", 0)),
                target=int(o.get("target", 0)),
            )
            for o in objectives_raw if isinstance(o, dict)
        ]

        entry = QuestEntry(
            quest_id=str(raw.get("quest_id", raw.get("id", ""))),
            title=str(raw.get("title", raw.get("name", ""))),
            state=state,
            level=int(raw.get("level", 0)),
            npc_name=str(raw.get("npc_name", raw.get("npc", ""))) or None,
            objectives=objectives,
        )

        if entry.is_complete or state == "completed":
            completed.append(entry)
        else:
            active.append(entry)

    # Also check structured quest dict
    quests_dict: dict = signals.get("quests_data") or {}
    if quests_dict:
        for raw in quests_dict.get("active", []):
            if isinstance(raw, dict) and not any(q.quest_id == raw.get("quest_id", raw.get("id", "")) for q in active):
                q = QuestEntry(
                    quest_id=str(raw.get("quest_id", raw.get("id", ""))),
                    title=str(raw.get("title", raw.get("name", ""))),
                    state="active",
                    level=int(raw.get("level", 0)),
                )
                active.append(q)

    return QuestState(
        active_quests=active,
        completed_quests=completed,
        total_active=len(active),
        total_completed=len(completed),
    )
