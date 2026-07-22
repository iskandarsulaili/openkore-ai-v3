"""
Long-term memory system using OpenMemory.

Stores and retrieves persistent knowledge about:
- Server patterns (GM patrols, ban waves, economy cycles)
- Farming spot quality (zeny/hr, xp/hr, danger level)
- Player interactions (who to trust, who to avoid)
- WoE intelligence (guild patterns, castle defense)
- Meta evolution (builds, prices, popular spots)
- Personal history (what worked, what got us killed)

Uses OpenMemory (https://github.com/CaviraOSS/OpenMemory) as the backend.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Memory Categories ──────────────────────────────────────────────

MEMORY_CATEGORIES = {
    "server_pattern": "Server behavior patterns (GM patrols, ban waves, economy)",
    "farming_spot": "Farming location quality and safety data",
    "player_profile": "Information about other players (trust, behavior, guild)",
    "woe_intel": "WoE intelligence (guild movements, castle defenses, timing)",
    "economy_trend": "Market trends, price movements, arbitrage opportunities",
    "meta_shift": "Meta changes, new builds, popular strategies",
    "personal_history": "Personal experiences, successes, failures, lessons",
    "timing_pattern": "Time-based patterns (peak hours, safe hours, event schedules)",
    "guild_intel": "Guild information (alliances, enemies, territory, strength)",
    "danger_zone": "Areas with high risk (PKers, GMs, dangerous mobs)",
}


@dataclass(slots=True)
class LongTermMemory:
    """Persistent memory system using OpenMemory backend.
    
    Stores structured memories with categories, tags, and importance scores.
    Retrieves relevant memories based on context similarity.
    """
    
    _lock: RLock = field(default_factory=RLock)
    _memory: Any = None
    _initialized: bool = False
    _stats: dict[str, int] = field(default_factory=lambda: {
        "stores": 0, "retrievals": 0, "deletes": 0, "errors": 0,
    })
    
    def __post_init__(self):
        self._init_memory()
    
    def _init_memory(self) -> bool:
        """Initialize OpenMemory backend."""
        if self._initialized:
            return True
        try:
            from openmemory import Memory
            self._memory = Memory()
            self._initialized = True
            logger.info("long_term_memory_initialized: backend=openmemory")
            return True
        except ImportError:
            logger.warning("long_term_memory_no_openmemory: falling back to local JSON")
            self._memory = None
            self._initialized = True
            return False
        except Exception as e:
            logger.warning("long_term_memory_init_failed: %s", e)
            self._memory = None
            self._initialized = True
            return False
    
    def store(
        self,
        *,
        category: str,
        content: str,
        tags: list[str] | None = None,
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a memory.
        
        Args:
            category: One of MEMORY_CATEGORIES keys
            content: The memory content (what happened, what we learned)
            tags: Optional tags for retrieval
            importance: 1-10, how important this memory is
            metadata: Optional structured data
        """
        with self._lock:
            self._stats["stores"] += 1
        
        if not self._initialized:
            self._init_memory()
        
        if self._memory is not None:
            try:
                payload = {
                    "category": category,
                    "content": content,
                    "tags": tags or [],
                    "importance": importance,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata or {},
                }
                import asyncio; asyncio.run(self._memory.add(json.dumps(payload)))
                return True
            except Exception as e:
                logger.warning("long_term_memory_store_failed: %s", e)
                with self._lock:
                    self._stats["errors"] += 1
                return False
        else:
            # Fallback: local JSON file
            return self._store_local(category, content, tags, importance, metadata)
    
    def search(
        self,
        *,
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_importance: int = 1,
    ) -> list[dict[str, Any]]:
        """Search memories by query, category, and tags.
        
        Returns list of matching memories sorted by relevance.
        """
        with self._lock:
            self._stats["retrievals"] += 1
        
        if not self._initialized:
            self._init_memory()
        
        if self._memory is not None:
            try:
                results = self._memory.search(query)
                # Handle both sync and async (coroutine) returns
                import asyncio
                if asyncio.iscoroutine(results):
                    results = asyncio.run(results)
                memories = []
                for r in results:
                    try:
                        data = json.loads(r) if isinstance(r, str) else r
                    except (json.JSONDecodeError, TypeError):
                        data = {"content": str(r)}
                    
                    # Filter by category
                    if category and data.get("category") != category:
                        continue
                    # Filter by tags
                    if tags and not all(t in data.get("tags", []) for t in tags):
                        continue
                    # Filter by importance
                    if data.get("importance", 0) < min_importance:
                        continue
                    
                    memories.append(data)
                
                return memories[:limit]
            except Exception as e:
                logger.warning("long_term_memory_search_failed: %s", e)
                with self._lock:
                    self._stats["errors"] += 1
                return []
        else:
            return self._search_local(query, category, tags, limit, min_importance)
    
    def get_relevant_context(
        self,
        *,
        current_map: str = "",
        current_time: str = "",
        current_activity: str = "",
        limit: int = 5,
    ) -> str:
        """Get a formatted context string of relevant memories for the LLM prompt."""
        queries = []
        if current_map:
            queries.append(f"map:{current_map}")
        if current_time:
            queries.append(f"time:{current_time}")
        if current_activity:
            queries.append(f"activity:{current_activity}")
        
        if not queries:
            queries = ["recent", "important"]
        
        all_memories = []
        for q in queries:
            results = self.search(query=q, limit=limit)
            all_memories.extend(results)
        
        # Deduplicate and sort by importance
        seen = set()
        unique = []
        for m in all_memories:
            key = m.get("content", "")[:50]
            if key not in seen:
                seen.add(key)
                unique.append(m)
        unique.sort(key=lambda x: -x.get("importance", 0))
        
        if not unique:
            return ""
        
        lines = ["── Long-Term Memory Context ──"]
        for m in unique[:limit]:
            cat = m.get("category", "unknown")
            content = m.get("content", "")[:200]
            imp = m.get("importance", 0)
            lines.append(f"  [{cat}] (imp={imp}) {content}")
        
        return "\n".join(lines)
    
    def forget(self, *, category: str | None = None, older_than_days: int = 30) -> int:
        """Delete old or category-specific memories."""
        with self._lock:
            self._stats["deletes"] += 1
        
        if self._memory is not None:
            try:
                self._memory.delete_all()
                return 0
            except Exception as e:
                logger.warning("long_term_memory_forget_failed: %s", e)
                return 0
        return 0
    
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
    
    # ── Local JSON fallback ──
    
    _local_path: Path = Path("/tmp/openkore_memory.json")
    
    def _store_local(self, category, content, tags, importance, metadata) -> bool:
        try:
            memories = []
            if self._local_path.exists():
                with open(self._local_path) as f:
                    memories = json.load(f)
            
            memories.append({
                "category": category,
                "content": content,
                "tags": tags or [],
                "importance": importance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            })
            
            # Keep only last 1000
            if len(memories) > 1000:
                memories = memories[-1000:]
            
            with open(self._local_path, "w") as f:
                json.dump(memories, f)
            return True
        except Exception as e:
            logger.warning("long_term_memory_local_store_failed: %s", e)
            return False
    
    def _search_local(self, query, category, tags, limit, min_importance) -> list[dict]:
        try:
            if not self._local_path.exists():
                return []
            with open(self._local_path) as f:
                memories = json.load(f)
            
            results = []
            query_lower = query.lower()
            for m in memories:
                if category and m.get("category") != category:
                    continue
                if tags and not all(t in m.get("tags", []) for t in tags):
                    continue
                if int(m.get("importance", 0) or 0) < min_importance:
                    continue
                if query_lower in m.get("content", "").lower():
                    results.append(m)
            
            results.sort(key=lambda x: -x.get("importance", 0))
            return results[:limit]
        except Exception:
            return []


# ── Global instance ──

_memory: LongTermMemory | None = None
_memory_lock = RLock()


def get_memory() -> LongTermMemory:
    """Get or create the global long-term memory instance."""
    global _memory
    with _memory_lock:
        if _memory is None:
            _memory = LongTermMemory()
        return _memory


def store_memory(
    category: str,
    content: str,
    tags: list[str] | None = None,
    importance: int = 5,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to store a memory."""
    return get_memory().store(
        category=category,
        content=content,
        tags=tags,
        importance=importance,
        metadata=metadata,
    )


def search_memory(
    query: str,
    category: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Convenience function to search memories."""
    return get_memory().search(query=query, category=category, tags=tags, limit=limit)


def get_memory_context(
    current_map: str = "",
    current_time: str = "",
    current_activity: str = "",
) -> str:
    """Get formatted memory context for LLM prompts."""
    return get_memory().get_relevant_context(
        current_map=current_map,
        current_time=current_time,
        current_activity=current_activity,
    )
