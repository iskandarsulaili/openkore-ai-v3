from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.planner.schemas import PlanHorizon, PlannerContext


@dataclass(slots=True)
class PlannerContextAssembler:
    runtime: Any

    def assemble(
        self,
        *,
        meta: ContractMeta,
        objective: str,
        horizon: PlanHorizon,
        event_limit: int = 64,
        memory_limit: int = 8,
    ) -> PlannerContext:
        bot_id = meta.bot_id
        state = self.runtime.enriched_state(bot_id=bot_id)
        state_payload = state.model_dump(mode="json")

        recent_events = self.runtime.recent_ingest_events(bot_id=bot_id, limit=event_limit)
        memory_matches = self.runtime.memory_context(bot_id=bot_id, query=objective, limit=memory_limit)
        episodes = self.runtime.memory_recent_episodes(bot_id=bot_id, limit=min(20, memory_limit * 2))

        doctrine: dict[str, object] = {
            "doctrine_version": state_payload.get("fleet_intent", {}).get("constraints", {}).get("config.doctrine_version"),
            "constraints": state_payload.get("fleet_intent", {}).get("constraints", {}),
        }

        fleet_coordination: dict[str, object] = {
            "mode": "local",
            "doctrine_version": doctrine.get("doctrine_version") or "local",
            "constraints": {},
            "blackboard": {},
        }
        try:
            central_constraints = self.runtime.fleet_constraints(bot_id=bot_id)
            fleet_coordination["mode"] = central_constraints.mode
            fleet_coordination["doctrine_version"] = central_constraints.doctrine_version
            fleet_coordination["constraints"] = dict(central_constraints.constraints)
            doctrine["doctrine_version"] = central_constraints.doctrine_version
            doctrine["constraints"] = dict(central_constraints.constraints)
        except Exception:
            pass

        try:
            blackboard_view = self.runtime.fleet_blackboard(bot_id=bot_id)
            fleet_coordination["mode"] = blackboard_view.mode
            fleet_coordination["blackboard"] = dict(blackboard_view.blackboard)
        except Exception:
            pass

        queue_depth = self.runtime.action_queue.count(bot_id)
        queue_info = {
            "pending_actions": queue_depth,
            "latency_avg_ms": round(self.runtime.latency_router.average_ms(), 3),
        }

        macros_info: dict[str, object] = {
            "latest_publication": self.runtime.latest_macro_publication(bot_id=bot_id),
        }

        fleet_constraints = {
            "role": state_payload.get("fleet_intent", {}).get("role"),
            "assignment": state_payload.get("fleet_intent", {}).get("assignment"),
            "objective": state_payload.get("fleet_intent", {}).get("objective"),
            "constraints": state_payload.get("fleet_intent", {}).get("constraints", {}),
            "coordination": fleet_coordination,
        }

        return PlannerContext(
            bot_id=bot_id,
            objective=objective,
            horizon=horizon,
            state=state_payload,
            recent_events=recent_events,
            memory_matches=memory_matches,
            episodes=episodes,
            doctrine=doctrine,
            fleet_constraints=fleet_constraints,
            queue=queue_info,
            macros=macros_info,
        )
