from __future__ import annotations

from dataclasses import dataclass
import json

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
    assert context.reflex["active_rules"] >= 0
    assert "categories" in context.reflex


@dataclass(slots=True)
class _LargeState:
    bot_id: str
    raw_payload: dict[str, object]

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return self.raw_payload


class _LargeRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self._raw_state = {
            "generated_at": "2026-01-01T00:00:00Z",
            "operational": {
                "map": "prt_fild08",
                "x": 100,
                "y": 100,
                "hp": 900,
                "hp_max": 1000,
                "sp": 110,
                "sp_max": 200,
                "in_combat": False,
                "target_id": None,
                "ai_sequence": "route",
                "base_level": 45,
                "job_level": 20,
                "job_name": "knight",
                "skill_points": 3,
                "stat_points": 5,
                "skill_count": 40,
                "skills": {f"skill_{i}": i for i in range(300)},
            },
            "encounter": {
                "in_encounter": False,
                "target_id": None,
                "nearby_hostiles": 1,
                "nearby_allies": 2,
                "risk_score": 0.3,
            },
            "navigation": {
                "map": "prt_fild08",
                "x": 100,
                "y": 100,
                "route_status": "moving",
                "destination_map": "prt_fild09",
                "destination_x": 80,
                "destination_y": 77,
            },
            "inventory": {
                "zeny": 100000,
                "item_count": 300,
                "weight": 2000,
                "weight_max": 8000,
                "overweight_ratio": 0.25,
                "consumables": {f"cons_{i}": i for i in range(100)},
            },
            "economy": {
                "zeny": 100000,
                "zeny_delta_1m": 200,
                "zeny_delta_10m": 1000,
                "vendor_exposure": 99,
                "transaction_count_10m": 44,
                "inventory_value_estimate": 555555,
                "price_signal_index": 1.25,
                "market_listings": [
                    {
                        "item_id": f"item_{i}",
                        "item_name": f"item_name_{i}",
                        "buy_price": 100 + i,
                        "sell_price": 50 + i,
                        "quantity": 500 + i,
                        "source": "market",
                        "metadata": {"noise": "x" * 80},
                    }
                    for i in range(200)
                ],
            },
            "quest": {
                "active_quests": [f"quest_{i}" for i in range(120)],
                "completed_quests": [f"quest_done_{i}" for i in range(120)],
                "quest_status": {f"quest_{i}": "active" for i in range(200)},
                "quest_objectives": {
                    f"quest_{i}": [
                        {
                            "objective_id": f"obj_{i}_{j}",
                            "description": "long objective text " + ("x" * 100),
                            "status": "in_progress",
                            "current": j,
                            "target": 10,
                        }
                        for j in range(20)
                    ]
                    for i in range(40)
                },
                "active_objective_count": 120,
                "objective_completion_ratio": 0.2,
                "last_npc": "Tool Dealer",
            },
            "npc": {
                "last_interacted_npc": "Tool Dealer",
                "total_known_npcs": 70,
                "interaction_count_10m": 25,
                "relationships": [
                    {
                        "npc_id": f"npc_{i}",
                        "npc_name": f"NPC {i}",
                        "relation": "quest",
                        "affinity_score": 0.1,
                        "trust_score": 0.2,
                        "interaction_count": 5,
                    }
                    for i in range(200)
                ],
            },
            "social": {
                "recent_chat_count": 40,
                "private_messages_5m": 10,
                "party_messages_5m": 10,
                "guild_messages_5m": 10,
                "last_interaction_at": "2026-01-01T00:00:00Z",
            },
            "risk": {
                "danger_score": 0.5,
                "death_risk_score": 0.1,
                "pvp_risk_score": 0.1,
                "anomaly_flags": [f"flag_{i}" for i in range(100)],
            },
            "fleet_intent": {
                "role": "grinder",
                "assignment": "prt_fild08",
                "objective": "farm",
                "constraints": {"config.doctrine_version": "legacy-v1"},
            },
            "features": {
                "values": {f"f_{i}": float(i) for i in range(2000)},
                "labels": {"map": "prt_fild08", "job_name": "knight"},
            },
            "entities": [
                {
                    "entity_id": f"ent_{i}",
                    "entity_type": "monster",
                    "name": f"Mob {i}",
                    "map": "prt_fild08",
                    "x": i,
                    "y": i,
                    "relation": "hostile",
                }
                for i in range(400)
            ],
            "recent_event_ids": [f"evt_{i}" for i in range(300)],
        }

    def enriched_state(self, *, bot_id: str) -> _LargeState:
        return _LargeState(bot_id=bot_id, raw_payload=self._raw_state)

    def recent_ingest_events(self, *, bot_id: str, limit: int = 100) -> list[dict[str, object]]:
        return [
            {
                "event_id": f"evt_{i}",
                "event_family": "snapshot",
                "event_type": "snapshot.compact",
                "severity": "info",
                "observed_at": "2026-01-01T00:00:00Z",
                "text": "snapshot" + ("x" * 200),
                "tags": {f"k{j}": f"v{j}" for j in range(20)},
                "numeric": {f"n{j}": float(j) for j in range(20)},
                "payload": {
                    "actor_id": f"a{i}",
                    "quest_id": f"q{i}",
                    "state_to": "active",
                    "npc": "Tool Dealer",
                    "revision": f"tick-{i}",
                    "large_blob": "x" * 2000,
                },
            }
            for i in range(limit)
        ]

    def memory_context(self, *, bot_id: str, query: str, limit: int) -> list[dict[str, object]]:
        return [
            {
                "bot_id": bot_id,
                "query": query,
                "memory_id": f"mem_{i}",
                "text": "memory:" + ("x" * 1200),
                "source": "openmemory",
                "topic": "grinding",
                "score": 0.91,
                "importance": 0.77,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:01Z",
                "labels": [f"label_{j}" for j in range(32)],
                "entities": [{"id": f"e_{j}", "name": "poring", "blob": "x" * 120} for j in range(20)],
                "metadata": {
                    f"k_{j}": {
                        "value": f"v_{j}",
                        "nested": {"signal": "x" * 300, "weight": j},
                    }
                    for j in range(24)
                },
                "trace": {
                    "origin": "planner",
                    "path": [f"hop_{j}" for j in range(30)],
                },
                "overflow_1": "y" * 300,
                "overflow_2": "z" * 300,
            }
            for i in range(limit)
        ]

    def memory_recent_episodes(self, *, bot_id: str, limit: int) -> list[dict[str, object]]:
        return [
            {
                "bot_id": bot_id,
                "episode_id": f"ep_{i}",
                "objective": "farm safely",
                "summary": "episode:" + ("q" * 1400),
                "result": "success",
                "score": 0.66,
                "duration_s": 123.4,
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:02:00Z",
                "maps": [f"map_{j}" for j in range(28)],
                "actions": [{"kind": "travel", "detail": "x" * 140} for _ in range(24)],
                "rewards": [{"kind": "loot", "value": j} for j in range(24)],
                "metrics": {f"m_{j}": float(j) for j in range(30)},
                "notes": {f"n_{j}": "x" * 160 for j in range(20)},
                "overflow_1": "w" * 300,
                "overflow_2": "v" * 300,
            }
            for i in range(limit)
        ]


def test_planner_context_assembler_compacts_state_for_prompt_density() -> None:
    runtime = _LargeRuntime()
    assembler = PlannerContextAssembler(runtime=runtime)

    raw_state_bytes = len(json.dumps(runtime._raw_state, ensure_ascii=False, default=str))
    raw_events_bytes = len(json.dumps(runtime.recent_ingest_events(bot_id="bot:p2", limit=64), ensure_ascii=False, default=str))
    raw_memory_bytes = len(
        json.dumps(runtime.memory_context(bot_id="bot:p2", query="farm safely", limit=4), ensure_ascii=False, default=str)
    )
    raw_episodes_bytes = len(json.dumps(runtime.memory_recent_episodes(bot_id="bot:p2", limit=8), ensure_ascii=False, default=str))

    context = assembler.assemble(
        meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:p2", trace_id="trace-p2"),
        objective="farm safely",
        horizon=PlanHorizon.strategic,
        event_limit=64,
        memory_limit=4,
    )

    compact_state_bytes = len(json.dumps(context.state, ensure_ascii=False, default=str))
    compact_events_bytes = len(json.dumps(context.recent_events, ensure_ascii=False, default=str))
    compact_memory_bytes = len(json.dumps(context.memory_matches, ensure_ascii=False, default=str))
    compact_episodes_bytes = len(json.dumps(context.episodes, ensure_ascii=False, default=str))

    assert compact_state_bytes < raw_state_bytes
    assert compact_events_bytes < raw_events_bytes
    assert compact_memory_bytes < raw_memory_bytes
    assert compact_episodes_bytes < raw_episodes_bytes

    assert context.queue["context_state_bytes"] == compact_state_bytes
    assert context.queue["context_events_bytes"] == compact_events_bytes
    assert context.queue["context_memory_bytes"] == compact_memory_bytes
    assert context.queue["context_episodes_bytes"] == compact_episodes_bytes
    assert len(context.state["entities"]) <= 24
    assert len(context.state["features"]["values"]) <= 128
    assert len(context.recent_events) == 64
    assert len(context.recent_events[0]["tags"]) <= 8
    assert len(context.recent_events[0]["numeric"]) <= 8
    assert len(context.memory_matches) <= 4
    assert len(context.episodes) <= 8
    assert all(len(item) <= 14 for item in context.memory_matches)
    assert all(len(item) <= 14 for item in context.episodes)
    assert all(len(str(item.get("text", ""))) <= 220 for item in context.memory_matches)
    assert all(len(str(item.get("summary", ""))) <= 220 for item in context.episodes)
    assert "reflex" in context.model_dump(mode="json")
