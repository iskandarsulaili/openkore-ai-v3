"""
Patch adapter — reads patch notes and adapts to new mechanics.

When a new patch drops, everything changes. New skills, maps, items,
mechanics. A top player reads patch notes, tests new mechanics, and
finds the new meta within 24 hours. This module monitors for patches
and adapts the bot's strategy.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PatchNote:
    """A patch note entry."""
    version: str
    date: str
    changes: list[str] = field(default_factory=list)
    new_maps: list[str] = field(default_factory=list)
    new_items: list[str] = field(default_factory=list)
    new_skills: list[str] = field(default_factory=list)
    balance_changes: list[str] = field(default_factory=list)
    processed: bool = False
    adaptation_plan: str = ""


@dataclass(slots=True)
class PatchAdapter:
    """Adapts to game patches and updates."""
    
    _lock: RLock = field(default_factory=RLock)
    _patches: list[PatchNote] = field(default_factory=list)
    _current_version: str = ""
    _stats: dict[str, int] = field(default_factory=lambda: {"patches_detected": 0, "adaptations": 0})
    _enqueue_fn: Callable | None = None
    
    def detect_patch(self, version: str, date: str, changes: list[str] | None = None) -> PatchNote:
        """Detect and record a new patch."""
        with self._lock:
            # Check if we already know about this version
            for p in self._patches:
                if p.version == version:
                    return p
            
            patch = PatchNote(
                version=version,
                date=date,
                changes=changes or [],
            )
            self._patches.append(patch)
            self._current_version = version
            self._stats["patches_detected"] += 1
            logger.info("patch_detected: %s (%s) — %d changes", version, date, len(patch.changes))
            return patch
    
    def add_new_map(self, version: str, map_name: str) -> None:
        """Record a new map from a patch."""
        with self._lock:
            for p in self._patches:
                if p.version == version and map_name not in p.new_maps:
                    p.new_maps.append(map_name)
                    logger.info("patch_new_map: %s → %s", version, map_name)
    
    def add_new_item(self, version: str, item_name: str) -> None:
        """Record a new item from a patch."""
        with self._lock:
            for p in self._patches:
                if p.version == version and item_name not in p.new_items:
                    p.new_items.append(item_name)
    
    def get_adaptation_context(self) -> str:
        """Get formatted patch adaptation context for LLM prompts."""
        with self._lock:
            unprocessed = [p for p in self._patches if not p.processed]
            if not unprocessed:
                return ""
            
            lines = ["── Recent Patches ──"]
            for p in unprocessed[-3:]:
                lines.append(f"  v{p.version} ({p.date}):")
                if p.new_maps:
                    lines.append(f"    New maps: {', '.join(p.new_maps[:3])}")
                if p.new_items:
                    lines.append(f"    New items: {', '.join(p.new_items[:3])}")
                if p.balance_changes:
                    lines.append(f"    Balance: {', '.join(p.balance_changes[:3])}")
                if p.changes:
                    lines.append(f"    Changes: {len(p.changes)} total")
                if p.adaptation_plan:
                    lines.append(f"    Plan: {p.adaptation_plan}")
            
            return "\n".join(lines)
    
    def mark_processed(self, version: str) -> None:
        """Mark a patch as processed."""
        with self._lock:
            for p in self._patches:
                if p.version == version:
                    p.processed = True
                    self._stats["adaptations"] += 1
                    logger.info("patch_adapted: %s", version)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# Global instance
_patch: PatchAdapter | None = None
_patch_lock = RLock()


def get_patch_adapter() -> PatchAdapter:
    global _patch
    with _patch_lock:
        if _patch is None:
            _patch = PatchAdapter()
        return _patch
