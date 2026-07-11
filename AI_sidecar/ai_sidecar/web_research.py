"""Web research engine — sidecar researches problems it can't solve and saves knowledge."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

RESEARCH_TERMS: dict[str, list[str]] = {
    "stuck_map": [
        "Ragnarok Online {map} hunting guide",
        "OpenKore {map} macro lockMap route",
        "RO {map} monster spawn level range",
        "Ragnarok Online bot stuck no monsters fix",
    ],
    "job_change": [
        "Ragnarok Online {job} job change quest guide",
        "RO {job} job change NPC location",
        "OpenKore job change macro",
    ],
    "stats": [
        "Ragnarok Online {job} stat build guide",
        "RO {job} stat allocation leveling",
    ],
    "skills": [
        "Ragnarok Online {job} skill build leveling",
        "RO {job} best skills early level",
    ],
    "equipment": [
        "Ragnarok Online {job} equipment guide early level",
        "RO {job} weapon armor leveling",
    ],
    "cards": [
        "Ragnarok Online {map} card drops",
        "RO {map} monster card guide",
    ],
    "refine": [
        "Ragnarok Online refine system guide safe level",
        "RO refining成功率 安全強化",
    ],
    "mvp": [
        "Ragnarok Online {map} MVP spawn time",
        "RO MVP {name} strategy party",
    ],
    "quest": [
        "Ragnarok Online {quest} walkthrough",
        "RO {quest} guide step by step",
    ],
}


@dataclass
class ResearchResult:
    topic: str
    query: str
    summary: str
    source_url: str = ""
    applied: bool = False
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


class WebResearchEngine:
    """Researches problems via web search and stores results in knowledge base."""

    def __init__(self, experience_db=None):
        self._exp_db = experience_db
        self._cooldown: dict[str, float] = {}  # bot_id -> last research time
        self._results: list[ResearchResult] = []
        self._search_func = None  # Will be set from outside (web_search)
        self._extract_func = None  # Will be set from outside (web_extract)

    def set_search_tools(self, search_fn, extract_fn):
        """Set web search and extract functions for research."""
        self._search_func = search_fn
        self._extract_func = extract_fn

    def needs_research(self, bot_id: str, problem: str, cooldown_s: int = 300) -> bool:
        """Check if research is needed (not on cooldown, problem is known)."""
        key = f"{bot_id}:{problem}"
        last = self._cooldown.get(key, 0)
        if time.time() - last < cooldown_s:
            return False
        self._cooldown[key] = time.time()
        return True

    async def research(self, topic: str, context: dict[str, Any] | None = None) -> ResearchResult | None:
        """Execute web research on a topic. Returns summary or None on failure."""
        queries = self._build_queries(topic, context or {})
        for query in queries:
            logger.info("web_research: query=%s", query)
            result = await self._execute_search(query)
            if result:
                self._results.append(result)
                self._save_to_knowledge(result)
                return result
        return None

    def _build_queries(self, topic: str, context: dict[str, Any]) -> list[str]:
        """Build concrete search queries from topic + context."""
        if topic in RESEARCH_TERMS:
            templates = RESEARCH_TERMS[topic]
            return [t.format(**context) for t in templates]
        # Generic fallback
        return [
            f"Ragnarok Online {topic} guide",
            f"OpenKore {topic} configuration",
            f"RO bot {topic} fix",
        ]

    async def _execute_search(self, query: str) -> ResearchResult | None:
        """Execute a web search using local SearXNG instance."""
        if self._search_func is not None:
            return await self._search_func(query)
        
        # Default: use local SearXNG
        try:
            import httpx
            url = "http://127.0.0.1:8080/search"
            params = {"q": query, "format": "json", "language": "en", "categories": "general"}
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    return None
            
            best = results[0]
            summary = best.get("content", "")[:1000]
            return ResearchResult(
                topic=query[:60],
                query=query,
                summary=summary or "Result found",
                source_url=best.get("url", ""),
            )
        except Exception as exc:
            logger.exception("web_research_search_failed: %s", exc)
            return None

    def _save_to_knowledge(self, result: ResearchResult) -> None:
        """Save research result to ExperienceDatabase for cross-bot learning."""
        if self._exp_db is None:
            return
        try:
            # Store as shared knowledge entry
            from ai_sidecar.experience_db import ExperienceEntry
            import time
            knowledge_entry = ExperienceEntry(
                bot_id="web_research",
                timestamp=time.time(),
                context_type="web_research",
                map_name="",
                monster_name="",
                role="",
                action_taken=f"research:{result.topic}",
                success=True,
                reward=0.0,
                details={
                    "topic": result.topic,
                    "query": result.query,
                    "summary": result.summary[:500],
                    "source": result.source_url,
                    "timestamp": result.timestamp,
                },
            )
            # The ExperienceDatabase accepts record() for storing observations
            if hasattr(self._exp_db, "record"):
                self._exp_db.record(knowledge_entry)
            logger.info("web_research_knowledge_saved: topic=%s", result.topic)
        except Exception as exc:
            logger.exception("web_research_save_failed: %s", exc)

    def get_research(self, topic: str) -> list[ResearchResult]:
        """Get past research results for a topic."""
        return [r for r in self._results if topic in r.topic or topic in r.query]
