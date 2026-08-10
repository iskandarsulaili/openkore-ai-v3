"""Quests domain — quest tracking and automation awareness.

Wired to the REAL QuestTracker (ai_sidecar.domains.quests.tracker) so the
observe-only legacy domain layer performs genuine quest analysis instead of
being a pass-stub:

  - feeds quest data from bridge signals into the tracker (parse_quest_info)
  - surfaces near-completion quests (log-kind intent)
  - surfaces level-appropriate available quests (log-kind intent)
  - surfaces active-quest count per bot (log-kind intent)

OBSERVE-ONLY: every action this domain emits is kind="log" (observable intent,
never executed) — the modern per-manager wiring in heuristic_service owns actual
quest command emission (get_quests_near_completion log + QuestAutomation NPC
chains). This module's analysis runs live to prove the quest layer is exercised.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.autonomy.domains import BaseDomain
from ai_sidecar.autonomy.heuristic_service import HeuristicAction

logger = logging.getLogger(__name__)


class QuestsDomain(BaseDomain):
    name: str = "quests"
    priority: int = 55

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        service: Any,
    ) -> None:
        """Evaluate quest tracking/automation decisions.

        Uses the service's live QuestTracker (initialized in
        heuristic_service._init_new_domains). Emits log-kind intents only —
        observe-only layer, the modern wiring owns command emission.
        """
        tracker = getattr(service, "_quest_tracker", None)
        if tracker is None:
            return
        bot_id = service._resolve_bot_id(signals) if hasattr(service, "_resolve_bot_id") else str(signals.get("bot_id", "default"))

        # Feed quest data the bridge signals may carry (active_quests,
        # quest_window) so the tracker's persisted state stays current.
        try:
            tracker.parse_quest_info(signals, bot_id)
        except Exception as exc:  # never let quest analysis break the cycle
            logger.debug("quests: parse_quest_info skipped: %s", exc)

        # Near-completion quests — the most actionable quest signal (a bot
        # should head to the turn-in NPC next).
        try:
            near = tracker.get_quests_near_completion(bot_id)
            if near:
                names = ", ".join(q.quest_name or q.quest_id for q in near[:3])
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"quests_near_complete={len(near)}",
                    confidence=0.6,
                    reason=f"Quest(s) near completion: {names}",
                    domain="quests",
                ))
        except Exception as exc:
            logger.debug("quests: near-completion check skipped: %s", exc)

        # Active-quest awareness (log-only observation for the conscious tier).
        try:
            active = tracker.get_active_quests(bot_id)
            if active:
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"quests_active={len(active)}",
                    confidence=0.4,
                    reason=f"{len(active)} active quest(s) tracked",
                    domain="quests",
                ))
        except Exception as exc:
            logger.debug("quests: active-quest check skipped: %s", exc)

        # Level-appropriate available quests (log-only; the modern automation
        # layer decides whether to start them).
        try:
            base_level = int(signals.get("base_level", 1) or 1)
            available = tracker.get_available_for_level(base_level, limit=5)
            if available:
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"quests_available={len(available)}",
                    confidence=0.4,
                    reason=f"{len(available)} quest(s) available at level {base_level}",
                    domain="quests",
                ))
        except Exception as exc:
            logger.debug("quests: available-quest check skipped: %s", exc)


def create_domain() -> QuestsDomain:
    return QuestsDomain()
