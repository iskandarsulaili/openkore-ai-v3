"""Memory retrieval — episodic & semantic memory with cosine similarity search.

Provides in-memory storage for episodic and semantic memories with
configurable memory limits, pruning, and cosine similarity search.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from ai_sidecar.memory.embeddings import LocalSemanticEmbedder, SemanticEmbedder
from ai_sidecar.memory.episodic_store import EpisodicMemoryStore
from ai_sidecar.memory.semantic_store import SemanticMemoryStore

logger = logging.getLogger(__name__)


def _resolve_value(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


# ── Memory models ───────────────────────────────────────────────────────

@dataclass
class EpisodicMemory:
    """A single episodic memory: something that happened to a bot."""
    memory_id: str = field(default_factory=lambda: uuid4().hex)
    bot_id: str = ""
    event_type: str = ""     # e.g. "combat", "death", "level_up", "trade"
    context: str = ""        # description of the situation
    outcome: str = ""        # what happened as a result
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SemanticMemory:
    """A single semantic memory: a learned concept or fact."""
    memory_id: str = field(default_factory=lambda: uuid4().hex)
    bot_id: str = ""
    concept: str = ""        # the concept learned (e.g. "zone_difficulty")
    value: str = ""          # the value associated (e.g. "high")
    confidence: float = 0.5  # how confident we are in this knowledge [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)
    norm: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── Abstract Provider ───────────────────────────────────────────────────

class MemoryProvider:
    """Abstract base for memory providers.

    Each provider stores episodic and semantic memories for multiple bots.
    """

    def add_episode(
        self,
        *,
        bot_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Store an episodic memory. Returns the memory ID."""
        return self._add_episode_impl(bot_id, event_type, content, metadata or {})

    def add_semantic(
        self,
        *,
        bot_id: str,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Store a semantic memory. Returns the memory ID."""
        return self._add_semantic_impl(bot_id, source, content, metadata or {})

    def search_semantic(
        self, *, bot_id: str, query: str, limit: int
    ) -> list[dict[str, object]]:
        """Search semantic memories by relevance to a query."""
        return self._search_semantic_impl(bot_id, query, limit)

    def recent_episodes(
        self, *, bot_id: str, limit: int
    ) -> list[dict[str, object]]:
        """Return the most recent episodic memories."""
        return self._recent_episodes_impl(bot_id, limit)

    def query_episodic(
        self,
        *,
        bot_id: str,
        event_type: str | None = None,
        context_query: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Query episodic memories by event_type and/or context substring."""
        return self._query_episodic_impl(bot_id, event_type, context_query, limit)

    def query_semantic(
        self,
        *,
        bot_id: str,
        concept: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Query semantic memories by concept name and confidence threshold."""
        return self._query_semantic_impl(bot_id, concept, min_confidence, limit)

    def search_by_relevance(
        self,
        *,
        bot_id: str,
        query: str,
        limit: int = 20,
        include_episodic: bool = True,
        include_semantic: bool = True,
    ) -> list[dict[str, object]]:
        """Combined search across episodic and semantic memories.

        Semantic memories are scored by cosine similarity.
        Episodic memories are matched by keyword relevance.
        Results are interleaved by score.
        """
        return self._search_by_relevance_impl(
            bot_id, query, limit, include_episodic, include_semantic,
        )

    def stats(self, *, bot_id: str) -> dict[str, int]:
        """Return memory counts for a bot."""
        return self._stats_impl(bot_id)

    # ── Internal implementation hooks (override in subclasses) ──────────

    def _add_episode_impl(
        self, bot_id: str, event_type: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        raise NotImplementedError

    def _add_semantic_impl(
        self, bot_id: str, source: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        raise NotImplementedError

    def _search_semantic_impl(
        self, bot_id: str, query: str, limit: int,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    def _recent_episodes_impl(
        self, bot_id: str, limit: int,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    def _query_episodic_impl(
        self, bot_id: str, event_type: str | None,
        context_query: str | None, limit: int,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    def _query_semantic_impl(
        self, bot_id: str, concept: str | None,
        min_confidence: float, limit: int,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    def _search_by_relevance_impl(
        self, bot_id: str, query: str, limit: int,
        include_episodic: bool, include_semantic: bool,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    def _stats_impl(self, bot_id: str) -> dict[str, int]:
        raise NotImplementedError


# ── In-Memory Provider (primary) ────────────────────────────────────────

class InMemoryMemoryProvider(MemoryProvider):
    """In-memory memory provider with configurable limits and pruning.

    Stores episodic and semantic memories per bot in dicts.
    Supports cosine similarity search via embeddings for semantic memories.
    Automatically prunes oldest memories when limits are exceeded.
    """

    def __init__(
        self,
        *,
        dimensions: int = 128,
        embedder: SemanticEmbedder | None = None,
        max_episodes_per_bot: int = 5000,
        max_semantic_per_bot: int = 10000,
    ):
        self._embedder = embedder or LocalSemanticEmbedder(dimensions)
        self._lock = RLock()
        self._max_episodes = max_episodes_per_bot
        self._max_semantic = max_semantic_per_bot

        # Per-bot storage: bot_id -> list of EpisodicMemory / SemanticMemory
        self._episodes: dict[str, list[EpisodicMemory]] = {}
        self._semantic: dict[str, list[SemanticMemory]] = {}

    # ── Internal implementations ────────────────────────────────────────

    def _add_episode_impl(
        self, bot_id: str, event_type: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        memory = EpisodicMemory(
            bot_id=bot_id,
            event_type=event_type,
            context=content,
            outcome=metadata.get("outcome", ""),
            metadata=dict(metadata),
        )
        with self._lock:
            episodes = self._episodes.setdefault(bot_id, [])
            episodes.append(memory)
            # Prune oldest if over limit
            if len(episodes) > self._max_episodes:
                excess = len(episodes) - self._max_episodes
                self._episodes[bot_id] = episodes[excess:]
        return memory.memory_id

    def _add_semantic_impl(
        self, bot_id: str, source: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        vector, norm, _ = self._embedder.embed(content)
        memory = SemanticMemory(
            bot_id=bot_id,
            concept=source,           # map source -> concept field
            value=content,
            confidence=float(metadata.get("confidence", 0.8)),
            metadata=dict(metadata),
            vector=vector,
            norm=norm,
        )
        with self._lock:
            sem = self._semantic.setdefault(bot_id, [])
            sem.append(memory)
            # Prune oldest if over limit
            if len(sem) > self._max_semantic:
                excess = len(sem) - self._max_semantic
                self._semantic[bot_id] = sem[excess:]
        return memory.memory_id

    def _search_semantic_impl(
        self, bot_id: str, query: str, limit: int,
    ) -> list[dict[str, object]]:
        query_vector, query_norm, _ = self._embedder.embed(query)
        with self._lock:
            rows = list(self._semantic.get(bot_id, []))

        scored: list[dict[str, object]] = []
        for m in rows:
            score = self._embedder.cosine(
                query_vector, m.vector,
                lhs_norm=query_norm, rhs_norm=m.norm,
            )
            scored.append({
                "id": m.memory_id,
                "bot_id": bot_id,
                "source": m.concept,
                "content": m.value,
                "metadata": m.metadata,
                "created_at": m.created_at,
                "score": score,
            })
        scored.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return scored[:limit]

    def _recent_episodes_impl(
        self, bot_id: str, limit: int,
    ) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._episodes.get(bot_id, []))
        rows.sort(key=lambda m: m.created_at, reverse=True)
        return [
            {
                "id": m.memory_id,
                "bot_id": bot_id,
                "event_type": m.event_type,
                "content": m.context,
                "outcome": m.outcome,
                "metadata": m.metadata,
                "created_at": m.created_at,
            }
            for m in rows[:limit]
        ]

    def _query_episodic_impl(
        self, bot_id: str, event_type: str | None,
        context_query: str | None, limit: int,
    ) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._episodes.get(bot_id, []))

        # Filter by event_type
        if event_type:
            rows = [m for m in rows if m.event_type == event_type]

        # Filter by context substring (case-insensitive)
        if context_query:
            q = context_query.lower()
            rows = [m for m in rows if q in m.context.lower()]

        # Sort newest first
        rows.sort(key=lambda m: m.created_at, reverse=True)

        return [
            {
                "id": m.memory_id,
                "bot_id": bot_id,
                "event_type": m.event_type,
                "content": m.context,
                "outcome": m.outcome,
                "metadata": m.metadata,
                "created_at": m.created_at,
            }
            for m in rows[:limit]
        ]

    def _query_semantic_impl(
        self, bot_id: str, concept: str | None,
        min_confidence: float, limit: int,
    ) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._semantic.get(bot_id, []))

        # Filter by concept
        if concept:
            q = concept.lower()
            rows = [m for m in rows if q in m.concept.lower()]

        # Filter by confidence
        if min_confidence > 0.0:
            rows = [m for m in rows if m.confidence >= min_confidence]

        # Sort by confidence descending, then newest
        rows.sort(key=lambda m: (m.confidence, m.created_at), reverse=True)

        return [
            {
                "id": m.memory_id,
                "bot_id": bot_id,
                "concept": m.concept,
                "value": m.value,
                "confidence": m.confidence,
                "metadata": m.metadata,
                "created_at": m.created_at,
            }
            for m in rows[:limit]
        ]

    def _search_by_relevance_impl(
        self, bot_id: str, query: str, limit: int,
        include_episodic: bool, include_semantic: bool,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []

        if include_semantic:
            # Semantic: cosine similarity search
            semantic_results = self._search_semantic_impl(bot_id, query, limit)
            for r in semantic_results:
                r["_source"] = "semantic"
                r["_relevance"] = r.get("score", 0.0)
            results.extend(semantic_results)

        if include_episodic:
            # Episodic: keyword-based relevance scoring
            q = query.lower()
            q_words = set(q.split())
            with self._lock:
                episodes = list(self._episodes.get(bot_id, []))

            for m in episodes:
                text = f"{m.event_type} {m.context} {m.outcome}".lower()
                # Simple keyword match score
                word_matches = sum(1 for w in q_words if w in text)
                sub_matches = text.count(q) * 2
                relevance = (word_matches + sub_matches) / max(1, len(q_words) + 1)
                if relevance > 0:
                    results.append({
                        "id": m.memory_id,
                        "bot_id": bot_id,
                        "event_type": m.event_type,
                        "content": m.context,
                        "outcome": m.outcome,
                        "metadata": m.metadata,
                        "created_at": m.created_at,
                        "_source": "episodic",
                        "_relevance": min(1.0, relevance),
                    })

        # Sort by relevance descending, then by date descending
        results.sort(
            key=lambda r: (r.get("_relevance", 0.0), r.get("created_at", datetime.min)),
            reverse=True,
        )
        return results[:limit]

    def _stats_impl(self, bot_id: str) -> dict[str, int]:
        with self._lock:
            return {
                "episodes": len(self._episodes.get(bot_id, [])),
                "semantic_records": len(self._semantic.get(bot_id, [])),
            }


# ── SQLite Memory Provider ──────────────────────────────────────────────

class SQLiteMemoryProvider(MemoryProvider):
    """SQLite-backed memory provider that delegates to store classes."""

    def __init__(
        self,
        *,
        episodic: EpisodicMemoryStore,
        semantic: SemanticMemoryStore,
        max_episodes_per_bot: int = 5000,
        max_semantic_per_bot: int = 10000,
    ):
        self._episodic = episodic
        self._semantic = semantic
        self._max_episodes = max_episodes_per_bot
        self._max_semantic = max_semantic_per_bot

    def _add_episode_impl(
        self, bot_id: str, event_type: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        return self._episodic.add_episode(
            bot_id=bot_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def _add_semantic_impl(
        self, bot_id: str, source: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        return self._semantic.add(
            bot_id=bot_id,
            source=source,
            content=content,
            metadata=metadata,
        )

    def _search_semantic_impl(
        self, bot_id: str, query: str, limit: int,
    ) -> list[dict[str, object]]:
        return self._semantic.search(bot_id=bot_id, query=query, limit=limit)

    def _recent_episodes_impl(
        self, bot_id: str, limit: int,
    ) -> list[dict[str, object]]:
        return self._episodic.recent(bot_id=bot_id, limit=limit)

    def _query_episodic_impl(
        self, bot_id: str, event_type: str | None,
        context_query: str | None, limit: int,
    ) -> list[dict[str, object]]:
        recent = self._episodic.recent(bot_id=bot_id, limit=10000)
        results = []
        for r in recent:
            if event_type and r.get("event_type") != event_type:
                continue
            if context_query:
                content = str(r.get("content", "")).lower()
                if context_query.lower() not in content:
                    continue
            results.append(r)
            if len(results) >= limit:
                break
        return results

    def _query_semantic_impl(
        self, bot_id: str, concept: str | None,
        min_confidence: float, limit: int,
    ) -> list[dict[str, object]]:
        # Use semantic search with empty query to get all, then filter
        all_sem = self._semantic.search(bot_id=bot_id, query="", limit=10000)
        results = []
        for r in all_sem:
            if concept:
                src = str(r.get("source", "")).lower()
                if concept.lower() not in src:
                    continue
            if min_confidence > 0.0:
                conf = float(r.get("metadata", {}).get("confidence", 0.0))
                if conf < min_confidence:
                    continue
            results.append(r)
            if len(results) >= limit:
                break
        return results

    def _search_by_relevance_impl(
        self, bot_id: str, query: str, limit: int,
        include_episodic: bool, include_semantic: bool,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []

        if include_semantic:
            sem_results = self._semantic.search(
                bot_id=bot_id, query=query, limit=limit,
            )
            for r in sem_results:
                r["_source"] = "semantic"
                r["_relevance"] = r.get("score", 0.0)
            results.extend(sem_results)

        if include_episodic:
            q = query.lower()
            q_words = set(q.split())
            recent = self._episodic.recent(bot_id=bot_id, limit=5000)
            for r in recent:
                text = f"{r.get('event_type', '')} {r.get('content', '')}".lower()
                word_matches = sum(1 for w in q_words if w in text)
                sub_matches = text.count(q) * 2
                relevance = (word_matches + sub_matches) / max(1, len(q_words) + 1)
                if relevance > 0:
                    r["_source"] = "episodic"
                    r["_relevance"] = min(1.0, relevance)
                    results.append(r)

        results.sort(
            key=lambda r: (r.get("_relevance", 0.0), r.get("created_at", datetime.min)),
            reverse=True,
        )
        return results[:limit]

    def _stats_impl(self, bot_id: str) -> dict[str, int]:
        return {
            "episodes": self._episodic.count(bot_id=bot_id),
            "semantic_records": self._semantic.count(bot_id=bot_id),
        }


# ── Open Memory Provider ────────────────────────────────────────────────

class OpenMemoryProvider(MemoryProvider):
    """OpenMemory-backed provider with in-memory fallback."""

    def __init__(
        self,
        *,
        sqlite_fallback: MemoryProvider,
        mode: str,
        path: str,
        max_episodes_per_bot: int = 5000,
        max_semantic_per_bot: int = 10000,
    ):
        self._sqlite_fallback = sqlite_fallback
        self._mode = mode
        self._path = path
        self._max_episodes = max_episodes_per_bot
        self._max_semantic = max_semantic_per_bot
        self._memory_client: object | None = None
        self._enabled = False
        self._init_error = ""
        self._init_client()

    def _init_client(self) -> None:
        try:
            if self._path.strip():
                from pathlib import Path as _Path
                _Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            from openmemory.client import Memory  # type: ignore
            self._memory_client = Memory(mode=self._mode, path=self._path)
            self._enabled = True
        except Exception as exc:
            self._enabled = False
            self._init_error = str(exc)
            logger.warning(
                "OpenMemoryProvider: init failed: %s", exc,
            )

    def _add_episode_impl(
        self, bot_id: str, event_type: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        if self._enabled and self._memory_client is not None:
            meta = dict(metadata)
            meta["event_type"] = event_type
            result = _resolve_value(
                self._memory_client.add(content, user_id=bot_id, meta=meta)
            )
            if isinstance(result, dict) and isinstance(result.get("id"), str):
                return result["id"]
        return self._sqlite_fallback.add_episode(
            bot_id=bot_id, event_type=event_type, content=content,
            metadata=metadata,
        )

    def _add_semantic_impl(
        self, bot_id: str, source: str, content: str,
        metadata: dict[str, object],
    ) -> str:
        if self._enabled and self._memory_client is not None:
            meta = dict(metadata)
            meta["source"] = source
            result = _resolve_value(
                self._memory_client.add(content, user_id=bot_id, meta=meta)
            )
            if isinstance(result, dict) and isinstance(result.get("id"), str):
                return result["id"]
        return self._sqlite_fallback.add_semantic(
            bot_id=bot_id, source=source, content=content,
            metadata=metadata,
        )

    def _search_semantic_impl(
        self, bot_id: str, query: str, limit: int,
    ) -> list[dict[str, object]]:
        if self._enabled and self._memory_client is not None:
            result = _resolve_value(
                self._memory_client.search(query, user_id=bot_id, limit=limit)
            )
            if isinstance(result, dict) and isinstance(result.get("matches"), list):
                rows: list[dict[str, object]] = []
                for idx, item in enumerate(result["matches"]):
                    if not isinstance(item, dict):
                        continue
                    rows.append({
                        "id": str(item.get("id") or f"openmemory-{idx}"),
                        "bot_id": bot_id,
                        "source": str(item.get("primary_sector") or "openmemory"),
                        "content": str(item.get("content") or ""),
                        "metadata": {"provider": "openmemory"},
                        "created_at": datetime.now(UTC),
                        "score": float(item.get("score") or 0.0),
                    })
                return rows[:limit]
            if isinstance(result, list):
                rows = []
                for idx, item in enumerate(result):
                    if not isinstance(item, dict):
                        continue
                    rows.append({
                        "id": str(item.get("id") or f"openmemory-{idx}"),
                        "bot_id": bot_id,
                        "source": str(item.get("primary_sector") or "openmemory"),
                        "content": str(item.get("content") or ""),
                        "metadata": {"provider": "openmemory"},
                        "created_at": datetime.now(UTC),
                        "score": float(item.get("score") or 0.0),
                    })
                return rows[:limit]
        return self._sqlite_fallback.search_semantic(
            bot_id=bot_id, query=query, limit=limit,
        )

    def _recent_episodes_impl(
        self, bot_id: str, limit: int,
    ) -> list[dict[str, object]]:
        if self._enabled and self._memory_client is not None:
            result = _resolve_value(self._memory_client.history(bot_id))
            rows: list[dict[str, object]] = []
            if isinstance(result, list):
                for idx, item in enumerate(result[:limit]):
                    if not isinstance(item, dict):
                        continue
                    rows.append({
                        "id": str(item.get("id") or f"openmemory-history-{idx}"),
                        "bot_id": bot_id,
                        "event_type": "history",
                        "content": str(item.get("content") or ""),
                        "outcome": str(item.get("outcome", "")),
                        "metadata": {"provider": "openmemory"},
                        "created_at": datetime.now(UTC),
                    })
                return rows
        return self._sqlite_fallback.recent_episodes(
            bot_id=bot_id, limit=limit,
        )

    def _query_episodic_impl(
        self, bot_id: str, event_type: str | None,
        context_query: str | None, limit: int,
    ) -> list[dict[str, object]]:
        # Fallback to SQLite for filtered queries
        return self._sqlite_fallback.query_episodic(
            bot_id=bot_id, event_type=event_type,
            context_query=context_query, limit=limit,
        )

    def _query_semantic_impl(
        self, bot_id: str, concept: str | None,
        min_confidence: float, limit: int,
    ) -> list[dict[str, object]]:
        return self._sqlite_fallback.query_semantic(
            bot_id=bot_id, concept=concept,
            min_confidence=min_confidence, limit=limit,
        )

    def _search_by_relevance_impl(
        self, bot_id: str, query: str, limit: int,
        include_episodic: bool, include_semantic: bool,
    ) -> list[dict[str, object]]:
        return self._sqlite_fallback.search_by_relevance(
            bot_id=bot_id, query=query, limit=limit,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
        )

    def _stats_impl(self, bot_id: str) -> dict[str, int]:
        return self._sqlite_fallback.stats(bot_id=bot_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def init_error(self) -> str:
        return self._init_error


# ── Memory Retrieval Service ────────────────────────────────────────────

@dataclass(slots=True)
class MemoryRetrievalService:
    """High-level service exposing memory operations to the rest of the system."""

    provider: MemoryProvider

    def capture_snapshot(
        self, *, bot_id: str, tick_id: str, summary: str,
        payload: dict[str, object],
    ) -> None:
        """Store a periodic snapshot as both episodic and semantic memory."""
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

    def capture_action(
        self, *, bot_id: str, action_id: str, kind: str,
        message: str, metadata: dict[str, object],
    ) -> None:
        """Store an action event as both episodic and semantic memory."""
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

    def search_context(
        self, *, bot_id: str, query: str, limit: int,
    ) -> list[dict[str, object]]:
        """Search semantic memory for context relevant to a query."""
        return self.provider.search_semantic(
            bot_id=bot_id, query=query, limit=limit,
        )

    def recent_episodes(
        self, *, bot_id: str, limit: int,
    ) -> list[dict[str, object]]:
        """Return the most recent episodic memories."""
        return self.provider.recent_episodes(
            bot_id=bot_id, limit=limit,
        )

    def query_episodic(
        self, *, bot_id: str, event_type: str | None = None,
        context_query: str | None = None, limit: int = 50,
    ) -> list[dict[str, object]]:
        """Query episodic memories by event type and/or context."""
        return self.provider.query_episodic(
            bot_id=bot_id, event_type=event_type,
            context_query=context_query, limit=limit,
        )

    def query_semantic(
        self, *, bot_id: str, concept: str | None = None,
        min_confidence: float = 0.0, limit: int = 50,
    ) -> list[dict[str, object]]:
        """Query semantic memories by concept and confidence."""
        return self.provider.query_semantic(
            bot_id=bot_id, concept=concept,
            min_confidence=min_confidence, limit=limit,
        )

    def search_by_relevance(
        self, *, bot_id: str, query: str, limit: int = 20,
        include_episodic: bool = True, include_semantic: bool = True,
    ) -> list[dict[str, object]]:
        """Combined search across episodic and semantic memories."""
        return self.provider.search_by_relevance(
            bot_id=bot_id, query=query, limit=limit,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
        )

    def stats(self, *, bot_id: str) -> dict[str, int]:
        """Return memory counts for a bot."""
        return self.provider.stats(bot_id=bot_id)
