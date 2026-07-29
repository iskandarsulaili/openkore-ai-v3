"""Learning domain — experience tracking, strategy adaptation.

Extracted from heuristic_service.py AdaptiveDataStore methods and
kill-tracking logic (lines 556-893, 1283-1304).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class LearningDomain(BaseDomain):
    name: str = "learning"
    priority: int = 80

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate learning/adaptation decisions.

        Records kill/death data in the adaptive store, tracks
        kills/hour, and escalates if zero kills for extended period.
        """
        bot_id = service._resolve_bot_id(signals)
        map_name = str(signals.get("map", "") or "").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        atk_power = int(signals.get("attack_power", 25) or 25)

        # ── RECORD VISIT ──
        service._adaptive.record_visit(map_name)

        # ── RECORD KILLS ──
        _monster_kill = signals.get("last_monster_kill", 0) or 0
        _exp_gained = signals.get("exp", 0) or 0
        if _monster_kill > 0:
            _x = int(signals.get("x", 0) or 0)
            _y = int(signals.get("y", 0) or 0)
            _monster_name = str(signals.get("monster_name", "") or "")
            service._adaptive.record_kill(
                map_name, float(_exp_gained), _x, _y, _monster_name,
            )

        # ── TRACK KILLS/HOUR ──
        self._track_kills_per_hour(signals, bot_id, service)

        # ── RECORD DEATHS ──
        _hp = float(signals.get("hp_ratio", 1.0) or 1.0)
        if _hp <= 0:
            service._adaptive.record_death(map_name)
            service._bot_deaths[bot_id] = service._bot_deaths.get(bot_id, 0) + 1

        # ── OPTIMAL MAP SELECTION (logged for routing domain) ──
        _optimal_map, _reason = service._adaptive.get_optimal_hunting_map(
            job_name, base_level, atk_power,
        )
        _survivability = service._adaptive.estimate_survivability(
            _optimal_map, base_level, atk_power,
        )
        logger.debug(
            "[learning] %s: optimal map=%s (survivability=%.2f) — %s",
            bot_id, _optimal_map, _survivability, _reason,
        )

    def _track_kills_per_hour(
        self,
        signals: dict[str, Any],
        bot_id: str,
        service: Any,
    ) -> None:
        """Track kills/hour and log warning if 0 for 30+ minutes."""
        _now = __import__("time").time()
        _kills = int(signals.get("last_monster_kill", 0) or 0)
        _last_kills = service._last_kills_count.get(bot_id, 0)
        _last_time = service._last_kills_time.get(bot_id, _now)
        _elapsed = _now - _last_time

        if _elapsed < 60:
            return

        _gained = _kills - _last_kills
        _rate = (_gained / _elapsed) * 3600

        service._last_kills_count[bot_id] = _kills
        service._last_kills_time[bot_id] = _now

        # Log every 5 minutes
        _last_log = service._last_kills_log.get(bot_id, 0)
        if _now - _last_log > 300:
            service._last_kills_log[bot_id] = _now
            logger.info(
                "[kills_hour] %s: %d kills in %.0fs = %.1f/hour",
                bot_id, _gained, _elapsed, _rate,
            )

        # Escalate if 0 kills for 30+ minutes
        if _gained == 0 and _elapsed > 1800:
            logger.warning(
                "[kills_hour] %s: ZERO kills in %.0fs! Escalating.",
                bot_id, _elapsed,
            )


def create_domain() -> LearningDomain:
    return LearningDomain()
