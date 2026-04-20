from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ai_sidecar.memory.embeddings import LocalSemanticEmbedder
from ai_sidecar.persistence.models import MemorySemanticRecord
from ai_sidecar.persistence.repositories import MemoryRepository


class SemanticMemoryStore:
    def __init__(self, *, repository: MemoryRepository, embedder: LocalSemanticEmbedder, candidates: int) -> None:
        self._repository = repository
        self._embedder = embedder
        self._candidates = candidates

    def add(
        self,
        *,
        bot_id: str,
        source: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        vector, norm, lexical_signature = self._embedder.embed(content)
        memory_id = uuid4().hex
        self._repository.add_semantic(
            MemorySemanticRecord(
                id=memory_id,
                bot_id=bot_id,
                source=source,
                content=content,
                lexical_signature=lexical_signature,
                metadata=dict(metadata or {}),
                created_at=datetime.now(UTC),
                dimensions=self._embedder.dimensions,
                vector=vector,
                norm=norm,
            )
        )
        return memory_id

    def search(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        query_vector, query_norm, _ = self._embedder.embed(query)
        lexical_tokens = self._embedder.tokenize(query)
        candidates = self._repository.semantic_candidates(
            bot_id=bot_id,
            lexical_tokens=lexical_tokens,
            limit=max(limit, self._candidates),
        )

        scored: list[dict[str, object]] = []
        for item in candidates:
            score = self._embedder.cosine(
                query_vector,
                item.vector,
                lhs_norm=query_norm,
                rhs_norm=item.norm,
            )
            scored.append(
                {
                    "id": item.id,
                    "bot_id": item.bot_id,
                    "source": item.source,
                    "content": item.content,
                    "metadata": item.metadata,
                    "created_at": item.created_at,
                    "score": score,
                }
            )

        scored.sort(key=lambda rec: (rec["score"], rec["created_at"]), reverse=True)
        return scored[:limit]

    def count(self, *, bot_id: str) -> int:
        return self._repository.count_semantic(bot_id=bot_id)

