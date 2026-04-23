from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent
from ai_sidecar.contracts.state_graph import QuestState


@dataclass(slots=True)
class _QuestEntry:
    quest_id: str
    state: str = "unknown"
    npc: str | None = None
    title: str | None = None
    objectives: list[dict[str, object]] = field(default_factory=list)
    updated_at: datetime | None = None


@dataclass(slots=True)
class _QuestWindow:
    entries: dict[str, _QuestEntry] = field(default_factory=dict)
    completed_order: list[str] = field(default_factory=list)
    last_npc: str | None = None


class QuestProgressTracker:
    """Tracks quest transitions and objective progression for compact planner context."""

    def __init__(self) -> None:
        self._by_bot: dict[str, _QuestWindow] = {}

    def observe_snapshot(self, *, bot_id: str, payload: dict[str, object], observed_at: datetime) -> None:
        window = self._by_bot.setdefault(bot_id, _QuestWindow())
        quests = payload.get("quests") if isinstance(payload.get("quests"), list) else []
        for item in quests:
            if not isinstance(item, dict):
                continue
            quest_id = str(item.get("quest_id") or "").strip()
            if not quest_id:
                continue
            entry = window.entries.setdefault(quest_id, _QuestEntry(quest_id=quest_id))
            state = str(item.get("state") or entry.state or "unknown").strip().lower() or "unknown"
            entry.state = state
            npc = str(item.get("npc") or "").strip()
            if npc:
                entry.npc = npc
                window.last_npc = npc
            title = str(item.get("title") or "").strip()
            if title:
                entry.title = title
            objectives = item.get("objectives") if isinstance(item.get("objectives"), list) else []
            entry.objectives = [obj for obj in objectives if isinstance(obj, dict)][:16]
            entry.updated_at = observed_at
            self._sync_completion(window, quest_id, state)

    def observe_event(self, event: NormalizedEvent) -> None:
        if event.event_family != EventFamily.quest:
            return

        bot_id = event.meta.bot_id
        now = _normalize_dt(event.observed_at)
        window = self._by_bot.setdefault(bot_id, _QuestWindow())

        payload = event.payload
        if event.event_type == "quest.active_set":
            active = payload.get("active_quests") if isinstance(payload.get("active_quests"), list) else []
            active_set = {str(item).strip() for item in active if str(item).strip()}
            for quest_id in active_set:
                entry = window.entries.setdefault(quest_id, _QuestEntry(quest_id=quest_id))
                if entry.state in {"unknown", "completed", "abandoned"}:
                    entry.state = "active"
                entry.updated_at = now
            for quest_id, entry in window.entries.items():
                if quest_id not in active_set and entry.state == "active":
                    entry.state = "inactive"
            return

        quest_id = str(payload.get("quest_id") or event.tags.get("quest_id") or "").strip()
        if not quest_id:
            return

        entry = window.entries.setdefault(quest_id, _QuestEntry(quest_id=quest_id))
        state_from = str(payload.get("state_from") or event.tags.get("state_from") or "").strip().lower()
        state_to = str(payload.get("state_to") or event.tags.get("state_to") or "").strip().lower()
        if state_to:
            entry.state = state_to
        elif state_from and entry.state == "unknown":
            entry.state = state_from

        npc = str(payload.get("npc") or event.tags.get("npc") or "").strip()
        if npc:
            entry.npc = npc
            window.last_npc = npc

        title = str(payload.get("title") or "").strip()
        if title:
            entry.title = title

        objectives = payload.get("objectives") if isinstance(payload.get("objectives"), list) else []
        if objectives:
            entry.objectives = [obj for obj in objectives if isinstance(obj, dict)][:16]

        entry.updated_at = now
        self._sync_completion(window, quest_id, entry.state)

    def export(self, *, bot_id: str, observed_at: datetime) -> QuestState:
        window = self._by_bot.get(bot_id)
        if window is None:
            return QuestState(updated_at=observed_at)

        quest_status: dict[str, str] = {}
        quest_titles: dict[str, str] = {}
        quest_objectives: dict[str, list[dict[str, object]]] = {}
        active_quests: list[str] = []

        objective_total = 0
        objective_done = 0

        for quest_id, entry in sorted(window.entries.items(), key=lambda item: item[0]):
            state = entry.state or "unknown"
            quest_status[quest_id] = state
            if entry.title:
                quest_titles[quest_id] = entry.title
            if entry.objectives:
                compact_objs: list[dict[str, object]] = []
                for obj in entry.objectives[:16]:
                    compact = {
                        "objective_id": str(obj.get("objective_id") or ""),
                        "description": str(obj.get("description") or "")[:160],
                        "status": str(obj.get("status") or "unknown").lower(),
                        "current": _int_or_none(obj.get("current")),
                        "target": _int_or_none(obj.get("target")),
                    }
                    compact_objs.append(compact)
                    objective_total += 1
                    if compact["status"] in {"done", "completed", "complete"}:
                        objective_done += 1
                quest_objectives[quest_id] = compact_objs
            if state in {"active", "in_progress", "ongoing", "started"}:
                active_quests.append(quest_id)

        ratio = float(objective_done) / float(objective_total) if objective_total > 0 else 0.0

        completed = list(window.completed_order[-64:])
        return QuestState(
            active_quests=active_quests,
            quest_status=quest_status,
            quest_titles=quest_titles,
            quest_objectives=quest_objectives,
            completed_quests=completed,
            active_objective_count=objective_total,
            objective_completion_ratio=max(0.0, min(1.0, ratio)),
            last_npc=window.last_npc,
            updated_at=observed_at,
            raw={
                "quest_count": len(window.entries),
                "completed_count": len(completed),
            },
        )

    def _sync_completion(self, window: _QuestWindow, quest_id: str, state: str) -> None:
        if state in {"completed", "done", "finished", "abandoned"}:
            if quest_id not in window.completed_order:
                window.completed_order.append(quest_id)
            if len(window.completed_order) > 128:
                del window.completed_order[: len(window.completed_order) - 128]


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

