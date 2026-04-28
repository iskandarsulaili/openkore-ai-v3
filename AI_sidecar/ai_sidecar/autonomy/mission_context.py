from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ai_sidecar.autonomy.goal_stack import summarize_goal_stack
from ai_sidecar.contracts.autonomy import AutonomyMissionContext, GoalStackState
from ai_sidecar.contracts.common import ContractMeta

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AutonomyMissionContextAssembler:
    runtime: Any

    def assemble(
        self,
        *,
        meta: ContractMeta,
        horizon: str,
        objective_input: str,
        goal_state: GoalStackState,
        trigger_reasons: list[str],
        event_limit: int = 48,
        memory_limit: int = 8,
    ) -> AutonomyMissionContext:
        bot_id = meta.bot_id

        snapshot_payload = self._compact_payload(self._snapshot_payload(bot_id=bot_id), max_depth=4)
        enriched_payload = self._compact_payload(self._enriched_payload(bot_id=bot_id), max_depth=4)

        queue_depth = self._queue_depth(bot_id=bot_id)
        latency_avg_ms = self._latency_avg_ms()
        startup_gate = self._startup_gate_status(bot_id=bot_id)
        planner_status = self._planner_status(bot_id=bot_id)

        fleet_constraints = self._compact_payload(self._fleet_constraints(bot_id=bot_id), max_depth=4)
        fleet_blackboard = self._compact_payload(self._fleet_blackboard(bot_id=bot_id), max_depth=3)

        recent_events = self._compact_records(self._recent_events(bot_id=bot_id, limit=event_limit), limit=event_limit)
        memory_matches = self._compact_records(
            self._memory_context(bot_id=bot_id, query=objective_input, limit=memory_limit),
            limit=max(1, min(memory_limit, 8)),
        )
        recent_episodes = self._compact_records(
            self._memory_episodes(bot_id=bot_id, limit=min(20, memory_limit * 2)),
            limit=max(1, min(memory_limit * 2, 10)),
        )

        return AutonomyMissionContext(
            bot_id=bot_id,
            horizon=str(horizon),
            objective_input=str(objective_input)[:512],
            assessment=goal_state.assessment,
            selected_goal=goal_state.selected_goal,
            deterministic_goal_stack=list(goal_state.goal_stack),
            deterministic_goal_summary=summarize_goal_stack(state=goal_state),
            replan_reasons=list(goal_state.assessment.replan_reasons),
            trigger_reasons=[str(item) for item in trigger_reasons if str(item).strip()],
            queue={
                "pending_actions": queue_depth,
                "latency_avg_ms": latency_avg_ms,
            },
            startup_gate=startup_gate,
            planner_status=planner_status,
            snapshot=snapshot_payload,
            enriched_state=enriched_payload,
            fleet_constraints=fleet_constraints,
            fleet_blackboard=fleet_blackboard,
            recent_events=recent_events,
            memory_matches=memory_matches,
            recent_episodes=recent_episodes,
        )

    def _snapshot_payload(self, *, bot_id: str) -> dict[str, object]:
        cache = getattr(self.runtime, "snapshot_cache", None)
        if cache is None or not hasattr(cache, "get"):
            return {}
        try:
            snapshot = cache.get(bot_id)
        except Exception:
            logger.exception(
                "autonomy_mission_context_snapshot_failed",
                extra={"event": "autonomy_mission_context_snapshot_failed", "bot_id": bot_id},
            )
            return {}
        if snapshot is None:
            return {}
        if hasattr(snapshot, "model_dump"):
            payload = snapshot.model_dump(mode="json")
            if isinstance(payload, dict):
                return payload
            return {}
        if isinstance(snapshot, dict):
            return dict(snapshot)
        return {}

    def _enriched_payload(self, *, bot_id: str) -> dict[str, object]:
        fn = getattr(self.runtime, "enriched_state", None)
        if not callable(fn):
            return {}
        try:
            state = fn(bot_id=bot_id)
        except Exception:
            logger.exception(
                "autonomy_mission_context_enriched_failed",
                extra={"event": "autonomy_mission_context_enriched_failed", "bot_id": bot_id},
            )
            return {}
        if hasattr(state, "model_dump"):
            payload = state.model_dump(mode="json")
            if isinstance(payload, dict):
                return payload
            return {}
        if isinstance(state, dict):
            return dict(state)
        return {}

    def _queue_depth(self, *, bot_id: str) -> int:
        queue = getattr(self.runtime, "action_queue", None)
        if queue is None or not hasattr(queue, "count"):
            return 0
        try:
            return int(queue.count(bot_id))
        except Exception:
            return 0

    def _latency_avg_ms(self) -> float:
        latency = getattr(self.runtime, "latency_router", None)
        if latency is None or not hasattr(latency, "average_ms"):
            return 0.0
        try:
            return float(latency.average_ms())
        except Exception:
            return 0.0

    def _startup_gate_status(self, *, bot_id: str) -> dict[str, object]:
        fn = getattr(self.runtime, "startup_gate_status", None)
        if not callable(fn):
            return {}
        try:
            payload = fn(bot_id=bot_id)
        except Exception:
            logger.exception(
                "autonomy_mission_context_startup_gate_failed",
                extra={"event": "autonomy_mission_context_startup_gate_failed", "bot_id": bot_id},
            )
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _planner_status(self, *, bot_id: str) -> dict[str, object]:
        fn = getattr(self.runtime, "planner_status", None)
        if not callable(fn):
            return {}
        try:
            payload = fn(bot_id=bot_id)
        except Exception:
            logger.exception(
                "autonomy_mission_context_planner_status_failed",
                extra={"event": "autonomy_mission_context_planner_status_failed", "bot_id": bot_id},
            )
            return {}
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump(mode="json")
            return dumped if isinstance(dumped, dict) else {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _fleet_constraints(self, *, bot_id: str) -> dict[str, object]:
        fn = getattr(self.runtime, "fleet_constraints", None)
        if not callable(fn):
            return {}
        try:
            response = fn(bot_id=bot_id)
        except Exception:
            logger.exception(
                "autonomy_mission_context_fleet_constraints_failed",
                extra={"event": "autonomy_mission_context_fleet_constraints_failed", "bot_id": bot_id},
            )
            return {}
        if hasattr(response, "model_dump"):
            dumped = response.model_dump(mode="json")
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(response, dict):
            return dict(response)
        return {}

    def _fleet_blackboard(self, *, bot_id: str) -> dict[str, object]:
        fn = getattr(self.runtime, "fleet_blackboard", None)
        if not callable(fn):
            return {}
        try:
            response = fn(bot_id=bot_id)
        except Exception:
            logger.exception(
                "autonomy_mission_context_fleet_blackboard_failed",
                extra={"event": "autonomy_mission_context_fleet_blackboard_failed", "bot_id": bot_id},
            )
            return {}
        if hasattr(response, "model_dump"):
            dumped = response.model_dump(mode="json")
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(response, dict):
            return dict(response)
        return {}

    def _recent_events(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        fn = getattr(self.runtime, "recent_ingest_events", None)
        if not callable(fn):
            return []
        try:
            rows = fn(bot_id=bot_id, limit=int(limit))
        except Exception:
            logger.exception(
                "autonomy_mission_context_recent_events_failed",
                extra={"event": "autonomy_mission_context_recent_events_failed", "bot_id": bot_id},
            )
            return []
        out: list[dict[str, object]] = []
        if not isinstance(rows, list):
            return out
        for item in rows:
            if isinstance(item, dict):
                out.append(dict(item))
        return out

    def _memory_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        fn = getattr(self.runtime, "memory_context", None)
        if not callable(fn):
            return []
        try:
            rows = fn(bot_id=bot_id, query=query, limit=int(limit))
        except Exception:
            logger.exception(
                "autonomy_mission_context_memory_context_failed",
                extra={"event": "autonomy_mission_context_memory_context_failed", "bot_id": bot_id},
            )
            return []
        out: list[dict[str, object]] = []
        if not isinstance(rows, list):
            return out
        for item in rows:
            if isinstance(item, dict):
                out.append(dict(item))
        return out

    def _memory_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        fn = getattr(self.runtime, "memory_recent_episodes", None)
        if not callable(fn):
            return []
        try:
            rows = fn(bot_id=bot_id, limit=int(limit))
        except Exception:
            logger.exception(
                "autonomy_mission_context_memory_episodes_failed",
                extra={"event": "autonomy_mission_context_memory_episodes_failed", "bot_id": bot_id},
            )
            return []
        out: list[dict[str, object]] = []
        if not isinstance(rows, list):
            return out
        for item in rows:
            if isinstance(item, dict):
                out.append(dict(item))
        return out

    def _compact_records(self, rows: list[dict[str, object]], *, limit: int) -> list[dict[str, object]]:
        compact: list[dict[str, object]] = []
        for item in rows[: max(0, int(limit))]:
            compacted = self._compact_payload(item, max_depth=3)
            if isinstance(compacted, dict):
                compact.append(compacted)
        return compact

    def _compact_payload(self, value: object, *, max_depth: int) -> object:
        if isinstance(value, str):
            return value[:220]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            if max_depth <= 0:
                return [self._compact_scalar(item) for item in value[:12]]
            return [self._compact_payload(item, max_depth=max_depth - 1) for item in value[:12]]
        if isinstance(value, dict):
            keys = list(value.keys())
            if max_depth <= 0:
                return {str(key): self._compact_scalar(value.get(key)) for key in keys[:24]}
            return {
                str(key): self._compact_payload(value.get(key), max_depth=max_depth - 1)
                for key in keys[:24]
            }
        return str(value)[:220]

    def _compact_scalar(self, value: object) -> object:
        if isinstance(value, str):
            return value[:180]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {"_type": "dict", "_len": len(value)}
        if isinstance(value, list):
            return {"_type": "list", "_len": len(value)}
        return str(value)[:180]

