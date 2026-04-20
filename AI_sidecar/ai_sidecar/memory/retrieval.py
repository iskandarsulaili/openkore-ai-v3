from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from ai_sidecar.memory.embeddings import LocalSemanticEmbedder
from ai_sidecar.memory.episodic_store import EpisodicMemoryStore
from ai_sidecar.memory.semantic_store import SemanticMemoryStore

logger = logging.getLogger(__name__)


def _resolve_value(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


class MemoryProvider:
    def add_episode(
        self,
        *,
        bot_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        raise NotImplementedError

    def add_semantic(
        self,
        *,
        bot_id: str,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        raise NotImplementedError

    def search_semantic(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        raise NotImplementedError

    def recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        raise NotImplementedError

    def stats(self, *, bot_id: str) -> dict[str, int]:
        raise NotImplementedError


class SQLiteMemoryProvider(MemoryProvider):
    def __init__(self, *, episodic: EpisodicMemoryStore, semantic: SemanticMemoryStore) -> None:
        self._episodic = episodic
        self._semantic = semantic

    def add_episode(
        self,
        *,
        bot_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        return self._episodic.add_episode(
            bot_id=bot_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def add_semantic(
        self,
        *,
        bot_id: str,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        return self._semantic.add(
            bot_id=bot_id,
            source=source,
            content=content,
            metadata=metadata,
        )

    def search_semantic(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return self._semantic.search(bot_id=bot_id, query=query, limit=limit)

    def recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return self._episodic.recent(bot_id=bot_id, limit=limit)

    def stats(self, *, bot_id: str) -> dict[str, int]:
        return {
            "episodes": self._episodic.count(bot_id=bot_id),
            "semantic_records": self._semantic.count(bot_id=bot_id),
        }


class OpenMemoryProvider(MemoryProvider):
    def __init__(
        self,
        *,
        sqlite_fallback: MemoryProvider,
        mode: str,
        path: str,
    ) -> None:
        self._sqlite_fallback = sqlite_fallback
        self._mode = mode
        self._path = path
        self._memory_client: object | None = None
        self._enabled = False
        self._init_error = ""
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openmemory.client import Memory  # type: ignore

            self._memory_client = Memory(mode=self._mode, path=self._path)
            self._enabled = True
        except Exception as exc:
            self._enabled = False
            self._init_error = str(exc)
            logger.warning(
                "openmemory_init_failed",
                extra={"event": "openmemory_init_failed", "reason": str(exc)},
            )

    def add_episode(
        self,
        *,
        bot_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        if self._enabled and self._memory_client is not None:
            meta = dict(metadata or {})
            meta["event_type"] = event_type
            result = _resolve_value(self._memory_client.add(content, user_id=bot_id, meta=meta))
            if isinstance(result, dict) and isinstance(result.get("id"), str):
                return result["id"]
        return self._sqlite_fallback.add_episode(
            bot_id=bot_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def add_semantic(
        self,
        *,
        bot_id: str,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        if self._enabled and self._memory_client is not None:
            meta = dict(metadata or {})
            meta["source"] = source
            result = _resolve_value(self._memory_client.add(content, user_id=bot_id, meta=meta))
            if isinstance(result, dict) and isinstance(result.get("id"), str):
                return result["id"]
        return self._sqlite_fallback.add_semantic(
            bot_id=bot_id,
            source=source,
            content=content,
            metadata=metadata,
        )

    def search_semantic(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        if self._enabled and self._memory_client is not None:
            result = _resolve_value(self._memory_client.search(query, user_id=bot_id, limit=limit))
            if isinstance(result, dict) and isinstance(result.get("matches"), list):
                rows: list[dict[str, object]] = []
                for idx, item in enumerate(result["matches"]):
                    if not isinstance(item, dict):
                        continue
                    rows.append(
                        {
                            "id": str(item.get("id") or f"openmemory-{idx}"),
                            "bot_id": bot_id,
                            "source": str(item.get("primary_sector") or "openmemory"),
                            "content": str(item.get("content") or ""),
                            "metadata": {"provider": "openmemory"},
                            "created_at": datetime.now(UTC),
                            "score": float(item.get("score") or 0.0),
                        }
                    )
                return rows[:limit]
            if isinstance(result, list):
                rows = []
                for idx, item in enumerate(result):
                    if not isinstance(item, dict):
                        continue
                    rows.append(
                        {
                            "id": str(item.get("id") or f"openmemory-{idx}"),
                            "bot_id": bot_id,
                            "source": str(item.get("primary_sector") or "openmemory"),
                            "content": str(item.get("content") or ""),
                            "metadata": {"provider": "openmemory"},
                            "created_at": datetime.now(UTC),
                            "score": float(item.get("score") or 0.0),
                        }
                    )
                return rows[:limit]
        return self._sqlite_fallback.search_semantic(bot_id=bot_id, query=query, limit=limit)

    def recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        if self._enabled and self._memory_client is not None:
            result = _resolve_value(self._memory_client.history(bot_id))
            rows: list[dict[str, object]] = []
            if isinstance(result, list):
                for idx, item in enumerate(result[:limit]):
                    if not isinstance(item, dict):
                        continue
                    rows.append(
                        {
                            "id": str(item.get("id") or f"openmemory-history-{idx}"),
                            "bot_id": bot_id,
                            "event_type": "history",
                            "content": str(item.get("content") or ""),
                            "metadata": {"provider": "openmemory"},
                            "created_at": datetime.now(UTC),
                        }
                    )
                return rows
        return self._sqlite_fallback.recent_episodes(bot_id=bot_id, limit=limit)

    def stats(self, *, bot_id: str) -> dict[str, int]:
        return self._sqlite_fallback.stats(bot_id=bot_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def init_error(self) -> str:
        return self._init_error


class InMemoryMemoryProvider(MemoryProvider):
    def __init__(self, *, dimensions: int) -> None:
        self._embedder = LocalSemanticEmbedder(dimensions)
        self._lock = RLock()
        self._episodes: dict[str, list[dict[str, object]]] = {}
        self._semantic: dict[str, list[dict[str, object]]] = {}

    def add_episode(
        self,
        *,
        bot_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        memory_id = uuid4().hex
        row = {
            "id": memory_id,
            "bot_id": bot_id,
            "event_type": event_type,
            "content": content,
            "metadata": dict(metadata or {}),
            "created_at": datetime.now(UTC),
        }
        with self._lock:
            self._episodes.setdefault(bot_id, []).append(row)
            self._episodes[bot_id] = self._episodes[bot_id][-5000:]
        return memory_id

    def add_semantic(
        self,
        *,
        bot_id: str,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        memory_id = uuid4().hex
        vector, norm, _ = self._embedder.embed(content)
        row = {
            "id": memory_id,
            "bot_id": bot_id,
            "source": source,
            "content": content,
            "metadata": dict(metadata or {}),
            "created_at": datetime.now(UTC),
            "vector": vector,
            "norm": norm,
        }
        with self._lock:
            self._semantic.setdefault(bot_id, []).append(row)
            self._semantic[bot_id] = self._semantic[bot_id][-10000:]
        return memory_id

    def search_semantic(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        query_vector, query_norm, _ = self._embedder.embed(query)
        with self._lock:
            rows = list(self._semantic.get(bot_id, []))
        scored: list[dict[str, object]] = []
        for row in rows:
            score = self._embedder.cosine(
                query_vector,
                row["vector"],
                lhs_norm=query_norm,
                rhs_norm=row["norm"],
            )
            scored.append(
                {
                    "id": row["id"],
                    "bot_id": bot_id,
                    "source": row["source"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"],
                    "score": score,
                }
            )
        scored.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return scored[:limit]

    def recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._episodes.get(bot_id, []))
        rows.sort(key=lambda item: item["created_at"], reverse=True)
        return rows[:limit]

    def stats(self, *, bot_id: str) -> dict[str, int]:
        with self._lock:
            return {
                "episodes": len(self._episodes.get(bot_id, [])),
                "semantic_records": len(self._semantic.get(bot_id, [])),
            }


@dataclass(slots=True)
class MemoryRetrievalService:
    provider: MemoryProvider

    def capture_snapshot(self, *, bot_id: str, tick_id: str, summary: str, payload: dict[str, object]) -> None:
        self.provider.add_episode(
            bot_id=bot_id,
            event_type="snapshot",
            content=summary,
            metadata={"tick_id": tick_id, **payload},
        )
        self.provider.add_semantic(
            bot_id=bot_id,
            source="snapshot",
            content=summary,
            metadata={"tick_id": tick_id},
        )

    def capture_action(self, *, bot_id: str, action_id: str, kind: str, message: str, metadata: dict[str, object]) -> None:
        content = f"action {action_id} ({kind}): {message}"
        self.provider.add_episode(
            bot_id=bot_id,
            event_type="action",
            content=content,
            metadata={"action_id": action_id, **metadata},
        )
        self.provider.add_semantic(
            bot_id=bot_id,
            source="action",
            content=content,
            metadata={"action_id": action_id},
        )

    def search_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return self.provider.search_semantic(bot_id=bot_id, query=query, limit=limit)

    def recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return self.provider.recent_episodes(bot_id=bot_id, limit=limit)

    def stats(self, *, bot_id: str) -> dict[str, int]:
        return self.provider.stats(bot_id=bot_id)
