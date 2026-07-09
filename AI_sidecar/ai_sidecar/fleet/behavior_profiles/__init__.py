"""BehaviorProfile — base class for all RO mechanic behavior profiles."""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_sidecar.experience_db import ExperienceDatabase, ExperienceEntry

logger = logging.getLogger(__name__)


class BehaviorProfile:
    """Base class for behavior profiles that self-improve via ExperienceDatabase."""

    def __init__(self, bot_id: str, experience_db: ExperienceDatabase | None = None):
        self.bot_id = bot_id
        self._db = experience_db or ExperienceDatabase()
        self._signals: dict[str, Any] = {}

    def update_signals(self, signals: dict[str, Any]) -> None:
        self._signals.update(signals)

    def _record_experience(self, context_type: str, action_taken: str, success: bool,
                           reward: float = 0.0, **details: Any) -> None:
        self._db.record(ExperienceEntry(
            bot_id=self.bot_id,
            timestamp=time.time(),
            context_type=context_type,
            map_name=self._signals.get("map_name", ""),
            monster_name=self._signals.get("monster_name", ""),
            role=self._signals.get("role", ""),
            action_taken=action_taken,
            success=success,
            reward=reward,
            details=details,
        ))

    def best_action(self, context_type: str, **filters: Any) -> tuple[str, float]:
        return self._db.best_action(context_type=context_type,
                                    map_name=filters.get("map_name", ""),
                                    monster_name=filters.get("monster_name", ""))
