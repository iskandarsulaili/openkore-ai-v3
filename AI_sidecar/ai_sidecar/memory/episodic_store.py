from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ai_sidecar.persistence.models import MemoryEpisodeRecord
from ai_sidecar.persistence.repositories import MemoryRepository


class EpisodicMemoryStore:
    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def add_episode(
        self,
        *,
        bot_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        memory_id = uuid4().hex
        self._repository.add_episode(
            MemoryEpisodeRecord(
                id=memory_id,
                bot_id=bot_id,
                event_type=event_type,
                content=content,
                metadata=dict(metadata or {}),
                created_at=datetime.now(UTC),
            )
        )
        return memory_id

    def recent(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        records = self._repository.recent_episodes(bot_id=bot_id, limit=limit)
        return [
            {
                "id": item.id,
                "bot_id": item.bot_id,
                "event_type": item.event_type,
                "content": item.content,
                "metadata": item.metadata,
                "created_at": item.created_at,
            }
            for item in records
        ]

    def count(self, *, bot_id: str) -> int:
        return self._repository.count_episodes(bot_id=bot_id)

