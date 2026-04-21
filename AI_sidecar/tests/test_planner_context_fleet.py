from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.fleet_v2 import FleetBlackboardLocalResponse, FleetConstraintResponse
from ai_sidecar.planner.context_assembler import PlannerContextAssembler
from ai_sidecar.planner.schemas import PlanHorizon


@dataclass(slots=True)
class _State:
    bot_id: str

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {
            "fleet_intent": {
                "role": "grinder",
                "assignment": "prt_fild08",
                "objective": "farm",
                "constraints": {"config.doctrine_version": "legacy-v1"},
            }
        }


@dataclass(slots=True)
class _Queue:
    depth: int = 2

    def count(self, _bot_id: str) -> int:
        return self.depth


class _Latency:
    def average_ms(self) -> float:
        return 1.75


class _Runtime:
    def __init__(self) -> None:
        self.action_queue = _Queue(depth=2)
        self.latency_router = _Latency()

    def enriched_state(self, *, bot_id: str) -> _State:
        return _State(bot_id=bot_id)

    def recent_ingest_events(self, *, bot_id: str, limit: int = 100) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "event_type": "snapshot.compact", "limit": limit}]

    def memory_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "query": query, "limit": limit}]

    def memory_recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return [{"bot_id": bot_id, "episode_id": "ep-1", "limit": limit}]

    def latest_macro_publication(self, *, bot_id: str) -> dict[str, object]:
        return {"bot_id": bot_id, "version": "macro-v1"}

    def fleet_constraints(self, *, bot_id: str) -> FleetConstraintResponse:
        return FleetConstraintResponse(
            ok=True,
            bot_id=bot_id,
            mode="central",
            doctrine_version="doctrine-v3",
            constraints={
                "avoid": [{"conflict_key": "quest.alpha", "type": "quest_collision"}],
                "required": [{"keep": "route.prt_fild08", "type": "territory"}],
                "sources": ["conflict_resolution"],
                "policy": {
                    "step_1_detect_conflict": True,
                    "step_2_compare_priority_and_lease": True,
                    "step_3_apply_doctrine": True,
                    "step_4_emit_constraints": True,
                    "step_5_rearbitrate_pending_strategic": True,
                },
            },
        )

    def fleet_blackboard(self, *, bot_id: str) -> FleetBlackboardLocalResponse:
        return FleetBlackboardLocalResponse(
            ok=True,
            bot_id=bot_id,
            mode="central",
            constraints={"avoid": [], "required": [], "sources": ["fleet-central"]},
            blackboard={"doctrine": {"version": "doctrine-v3"}, "patterns": {"quest_swarm": {"slice_count": 3}}},
            local_summary={"queue_depth": 2, "outcome_backlog": 0},
        )


def test_planner_context_assembler_prefers_fleet_v2_constraints() -> None:
    runtime = _Runtime()
    assembler = PlannerContextAssembler(runtime=runtime)
    context = assembler.assemble(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:p1", trace_id="trace-p1"),
        objective="farm safely",
        horizon=PlanHorizon.strategic,
        event_limit=8,
        memory_limit=4,
    )

    assert context.doctrine["doctrine_version"] == "doctrine-v3"
    assert context.doctrine["constraints"]["required"][0]["keep"] == "route.prt_fild08"
    assert context.fleet_constraints["coordination"]["mode"] == "central"
    assert context.fleet_constraints["coordination"]["blackboard"]["doctrine"]["version"] == "doctrine-v3"

