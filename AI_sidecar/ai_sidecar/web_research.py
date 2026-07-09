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
        """Execute a web search and extract content."""
        if self._search_func is None:
            logger.warning("web_research: no search function available")
            return None
        try:
            search_results = await self._search_func(query, limit=3)
            if not search_results or not search_results.get("data", {}).get("web"):
                return None
            
            best_url = None
            for item in search_results["data"]["web"][:2]:
                url = item.get("url", "")
                if url and not url.startswith("https://github.com") and not url.startswith("https://forums.openkore"):
                    best_url = url
                    break
            if not best_url:
                best_url = search_results["data"]["web"][0].get("url", "")
            
            summary = ""
            if best_url and self._extract_func:
                extract_result = await self._extract_func([best_url], char_limit=3000)
                if extract_result and extract_result.get("results"):
                    content = extract_result["results"][0].get("content", "")
                    summary = content[:1000] if content else ""
            
            return ResearchResult(
                topic=query[:60],
                query=query,
                summary=summary or "Result found (no extract)",
                source_url=best_url or "",
            )
        except Exception as exc:
            logger.exception("web_research_failed: %s", exc)
            return None

    def _save_to_knowledge(self, result: ResearchResult) -> None:
        """Save research result to ExperienceDatabase for cross-bot learning."""
        if self._exp_db is None:
            return
        try:
            # Store as shared knowledge entry
            knowledge_entry = {
                "type": "web_research",
                "topic": result.topic,
                "query": result.query,
                "summary": result.summary[:500],
                "source": result.source_url,
                "timestamp": result.timestamp,
            }
            # The ExperienceDatabase accepts record() for storing observations
            if hasattr(self._exp_db, "record"):
                self._exp_db.record(knowledge_entry)
            logger.info("web_research_knowledge_saved: topic=%s", result.topic)
        except Exception as exc:
            logger.exception("web_research_save_failed: %s", exc)

    def get_research(self, topic: str) -> list[ResearchResult]:
        """Get past research results for a topic."""
        return [r for r in self._results if topic in r.topic or topic in r.query]
