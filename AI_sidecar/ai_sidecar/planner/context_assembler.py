from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.planner.schemas import PlanHorizon, PlannerContext

logger = logging.getLogger(__name__)


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
        state_payload_full = state.model_dump(mode="json")
        state_payload = self._compact_state(state_payload_full)

        recent_events_raw = self.runtime.recent_ingest_events(bot_id=bot_id, limit=event_limit)
        recent_events = self._compact_events(recent_events_raw, limit=event_limit)
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
            "degraded": False,
            "degradation_reasons": [],
        }
        try:
            central_constraints = self.runtime.fleet_constraints(bot_id=bot_id)
            fleet_coordination["mode"] = central_constraints.mode
            fleet_coordination["doctrine_version"] = central_constraints.doctrine_version
            fleet_coordination["constraints"] = dict(central_constraints.constraints)
            doctrine["doctrine_version"] = central_constraints.doctrine_version
            doctrine["constraints"] = dict(central_constraints.constraints)
        except Exception as exc:
            reason = f"fleet_constraints_unavailable:{type(exc).__name__}"
            fleet_coordination["degraded"] = True
            fleet_coordination["degradation_reasons"] = [
                *list(fleet_coordination.get("degradation_reasons") or []),
                reason,
            ]
            logger.exception(
                "planner_context_fleet_constraints_failed",
                extra={
                    "event": "planner_context_fleet_constraints_failed",
                    "bot_id": bot_id,
                    "objective": objective,
                    "reason": reason,
                },
            )

        try:
            blackboard_view = self.runtime.fleet_blackboard(bot_id=bot_id)
            fleet_coordination["mode"] = blackboard_view.mode
            fleet_coordination["blackboard"] = dict(blackboard_view.blackboard)
        except Exception as exc:
            reason = f"fleet_blackboard_unavailable:{type(exc).__name__}"
            fleet_coordination["degraded"] = True
            fleet_coordination["degradation_reasons"] = [
                *list(fleet_coordination.get("degradation_reasons") or []),
                reason,
            ]
            logger.exception(
                "planner_context_fleet_blackboard_failed",
                extra={
                    "event": "planner_context_fleet_blackboard_failed",
                    "bot_id": bot_id,
                    "objective": objective,
                    "reason": reason,
                },
            )

        queue_depth = self.runtime.action_queue.count(bot_id)
        state_bytes = len(json.dumps(state_payload, ensure_ascii=False, default=str))
        events_bytes = len(json.dumps(recent_events, ensure_ascii=False, default=str))
        queue_info = {
            "pending_actions": queue_depth,
            "latency_avg_ms": round(self.runtime.latency_router.average_ms(), 3),
            "context_state_bytes": state_bytes,
            "context_events_bytes": events_bytes,
        }

        horizon_budget_ms = 15000 if horizon == PlanHorizon.tactical else 30000
        latency_headroom = {
            "horizon_budget_ms": horizon_budget_ms,
            "observed_avg_ms": queue_info["latency_avg_ms"],
            "remaining_ms": max(0.0, float(horizon_budget_ms) - float(queue_info["latency_avg_ms"])),
        }

        operational = state_payload.get("operational") if isinstance(state_payload.get("operational"), dict) else {}
        quest_state = state_payload.get("quest") if isinstance(state_payload.get("quest"), dict) else {}
        npc_state = state_payload.get("npc") if isinstance(state_payload.get("npc"), dict) else {}
        economy_state = state_payload.get("economy") if isinstance(state_payload.get("economy"), dict) else {}
        inventory_state = state_payload.get("inventory") if isinstance(state_payload.get("inventory"), dict) else {}

        job_progression: dict[str, object] = {
            "job_name": operational.get("job_name"),
            "base_level": operational.get("base_level"),
            "job_level": operational.get("job_level"),
            "skill_points": operational.get("skill_points"),
            "stat_points": operational.get("stat_points"),
            "skill_count": operational.get("skill_count"),
            "skills": operational.get("skills", {}),
        }

        economy_context: dict[str, object] = {
            "zeny": economy_state.get("zeny") if economy_state.get("zeny") is not None else inventory_state.get("zeny"),
            "zeny_delta_1m": economy_state.get("zeny_delta_1m"),
            "zeny_delta_10m": economy_state.get("zeny_delta_10m"),
            "inventory_value_estimate": economy_state.get("inventory_value_estimate"),
            "price_signal_index": economy_state.get("price_signal_index"),
            "market_listings": economy_state.get("market_listings", []),
            "overweight_ratio": inventory_state.get("overweight_ratio"),
        }

        quest_context: dict[str, object] = {
            "active_quests": quest_state.get("active_quests", []),
            "completed_quests": quest_state.get("completed_quests", []),
            "quest_status": quest_state.get("quest_status", {}),
            "quest_objectives": quest_state.get("quest_objectives", {}),
            "active_objective_count": quest_state.get("active_objective_count"),
            "objective_completion_ratio": quest_state.get("objective_completion_ratio"),
            "last_npc": quest_state.get("last_npc"),
        }

        npc_context: dict[str, object] = {
            "last_interacted_npc": npc_state.get("last_interacted_npc"),
            "total_known_npcs": npc_state.get("total_known_npcs"),
            "interaction_count_10m": npc_state.get("interaction_count_10m"),
            "relationships": npc_state.get("relationships", []),
        }

        macros_info: dict[str, object] = {
            "latest_publication": self.runtime.latest_macro_publication(bot_id=bot_id),
        }

        reflex_info: dict[str, object] = {
            "active_rules": 0,
            "override_rules": 0,
            "complement_rules": 0,
            "categories": {},
            "open_breakers": 0,
            "latest_trigger_outcome": "",
            "latest_trigger_rule": "",
        }
        try:
            rules = self.runtime.list_reflex_rules(bot_id=bot_id)
            enabled_rules = [item for item in rules if bool(getattr(item, "enabled", False))]
            by_category: dict[str, int] = {}
            override_rules = 0
            complement_rules = 0
            for item in enabled_rules:
                category = str(getattr(getattr(item, "category", None), "value", getattr(item, "category", "")) or "unknown")
                by_category[category] = by_category.get(category, 0) + 1
                planner_interop = str(
                    getattr(getattr(item, "planner_interop", None), "value", getattr(item, "planner_interop", "")) or "override"
                )
                if planner_interop == "complement":
                    complement_rules += 1
                else:
                    override_rules += 1

            reflex_info["active_rules"] = len(enabled_rules)
            reflex_info["override_rules"] = override_rules
            reflex_info["complement_rules"] = complement_rules
            reflex_info["categories"] = by_category
        except Exception as exc:
            logger.exception(
                "planner_context_reflex_rules_failed",
                extra={
                    "event": "planner_context_reflex_rules_failed",
                    "bot_id": bot_id,
                    "objective": objective,
                    "reason": type(exc).__name__,
                },
            )

        try:
            breakers = self.runtime.reflex_breakers(bot_id=bot_id)
            reflex_info["open_breakers"] = sum(1 for item in breakers if str((item.state or "")).lower() != "closed")
        except Exception as exc:
            logger.exception(
                "planner_context_reflex_breakers_failed",
                extra={
                    "event": "planner_context_reflex_breakers_failed",
                    "bot_id": bot_id,
                    "objective": objective,
                    "reason": type(exc).__name__,
                },
            )

        try:
            recent = self.runtime.recent_reflex_triggers(bot_id=bot_id, limit=1)
            if recent:
                reflex_info["latest_trigger_outcome"] = str(recent[0].outcome or "")
                reflex_info["latest_trigger_rule"] = str(recent[0].rule_id or "")
        except Exception as exc:
            logger.exception(
                "planner_context_reflex_recent_failed",
                extra={
                    "event": "planner_context_reflex_recent_failed",
                    "bot_id": bot_id,
                    "objective": objective,
                    "reason": type(exc).__name__,
                },
            )

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
            reflex=reflex_info,
            job_progression=job_progression,
            economy_context=economy_context,
            quest_context=quest_context,
            npc_context=npc_context,
            latency_headroom=latency_headroom,
        )

    def _compact_state(self, state_payload: dict[str, object]) -> dict[str, object]:
        def _safe_dict(value: object) -> dict[str, object]:
            return value if isinstance(value, dict) else {}

        def _top_list(value: object, *, limit: int) -> list[dict[str, object]]:
            if not isinstance(value, list):
                return []
            out: list[dict[str, object]] = []
            for item in value[:limit]:
                if isinstance(item, dict):
                    out.append(item)
            return out

        def _top_dict_items(value: object, *, limit: int) -> dict[str, object]:
            data = _safe_dict(value)
            return {key: item for key, item in list(data.items())[:limit]}

        operational = _safe_dict(state_payload.get("operational"))
        encounter = _safe_dict(state_payload.get("encounter"))
        navigation = _safe_dict(state_payload.get("navigation"))
        inventory = _safe_dict(state_payload.get("inventory"))
        economy = _safe_dict(state_payload.get("economy"))
        quest = _safe_dict(state_payload.get("quest"))
        npc = _safe_dict(state_payload.get("npc"))
        social = _safe_dict(state_payload.get("social"))
        risk = _safe_dict(state_payload.get("risk"))
        fleet = _safe_dict(state_payload.get("fleet_intent"))
        features = _safe_dict(state_payload.get("features"))
        entities = _top_list(state_payload.get("entities"), limit=24)

        return {
            "generated_at": state_payload.get("generated_at"),
            "operational": {
                "map": operational.get("map"),
                "x": operational.get("x"),
                "y": operational.get("y"),
                "hp": operational.get("hp"),
                "hp_max": operational.get("hp_max"),
                "sp": operational.get("sp"),
                "sp_max": operational.get("sp_max"),
                "in_combat": operational.get("in_combat"),
                "target_id": operational.get("target_id"),
                "ai_sequence": operational.get("ai_sequence"),
                "base_level": operational.get("base_level"),
                "job_level": operational.get("job_level"),
                "job_name": operational.get("job_name"),
                "skill_points": operational.get("skill_points"),
                "stat_points": operational.get("stat_points"),
                "skill_count": operational.get("skill_count"),
                "skills": _top_dict_items(operational.get("skills"), limit=16),
            },
            "encounter": {
                "in_encounter": encounter.get("in_encounter"),
                "target_id": encounter.get("target_id"),
                "nearby_hostiles": encounter.get("nearby_hostiles"),
                "nearby_allies": encounter.get("nearby_allies"),
                "risk_score": encounter.get("risk_score"),
            },
            "navigation": {
                "map": navigation.get("map"),
                "x": navigation.get("x"),
                "y": navigation.get("y"),
                "route_status": navigation.get("route_status"),
                "destination_map": navigation.get("destination_map"),
                "destination_x": navigation.get("destination_x"),
                "destination_y": navigation.get("destination_y"),
            },
            "inventory": {
                "zeny": inventory.get("zeny"),
                "item_count": inventory.get("item_count"),
                "weight": inventory.get("weight"),
                "weight_max": inventory.get("weight_max"),
                "overweight_ratio": inventory.get("overweight_ratio"),
                "consumables": _top_dict_items(inventory.get("consumables"), limit=20),
            },
            "economy": {
                "zeny": economy.get("zeny"),
                "zeny_delta_1m": economy.get("zeny_delta_1m"),
                "zeny_delta_10m": economy.get("zeny_delta_10m"),
                "vendor_exposure": economy.get("vendor_exposure"),
                "transaction_count_10m": economy.get("transaction_count_10m"),
                "inventory_value_estimate": economy.get("inventory_value_estimate"),
                "price_signal_index": economy.get("price_signal_index"),
                "market_listings": _top_list(economy.get("market_listings"), limit=12),
            },
            "quest": {
                "active_quests": list((quest.get("active_quests") or [])[:16]),
                "completed_quests": list((quest.get("completed_quests") or [])[:16]),
                "quest_status": _top_dict_items(quest.get("quest_status"), limit=32),
                "quest_objectives": {
                    key: (value[:8] if isinstance(value, list) else value)
                    for key, value in list(_safe_dict(quest.get("quest_objectives")).items())[:16]
                },
                "active_objective_count": quest.get("active_objective_count"),
                "objective_completion_ratio": quest.get("objective_completion_ratio"),
                "last_npc": quest.get("last_npc"),
            },
            "npc": {
                "last_interacted_npc": npc.get("last_interacted_npc"),
                "total_known_npcs": npc.get("total_known_npcs"),
                "interaction_count_10m": npc.get("interaction_count_10m"),
                "relationships": _top_list(npc.get("relationships"), limit=16),
            },
            "social": {
                "recent_chat_count": social.get("recent_chat_count"),
                "private_messages_5m": social.get("private_messages_5m"),
                "party_messages_5m": social.get("party_messages_5m"),
                "guild_messages_5m": social.get("guild_messages_5m"),
                "last_interaction_at": social.get("last_interaction_at"),
            },
            "risk": {
                "danger_score": risk.get("danger_score"),
                "death_risk_score": risk.get("death_risk_score"),
                "pvp_risk_score": risk.get("pvp_risk_score"),
                "anomaly_flags": list((risk.get("anomaly_flags") or [])[:24]),
            },
            "fleet_intent": {
                "role": fleet.get("role"),
                "assignment": fleet.get("assignment"),
                "objective": fleet.get("objective"),
                "constraints": fleet.get("constraints", {}),
            },
            "features": {
                "values": {
                    key: value for key, value in sorted(_safe_dict(features.get("values")).items())[:128]
                },
                "labels": features.get("labels", {}),
            },
            "entities": entities,
            "recent_event_ids": list((state_payload.get("recent_event_ids") or [])[:32]),
        }

    def _compact_events(self, events: list[dict[str, object]], *, limit: int) -> list[dict[str, object]]:
        compact: list[dict[str, object]] = []
        for item in events[:limit]:
            tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
            numeric = item.get("numeric") if isinstance(item.get("numeric"), dict) else {}
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            compact.append(
                {
                    "event_id": item.get("event_id"),
                    "event_family": item.get("event_family"),
                    "event_type": item.get("event_type"),
                    "severity": item.get("severity"),
                    "observed_at": item.get("observed_at"),
                    "text": str(item.get("text") or "")[:180],
                    "tags": {k: tags[k] for k in list(tags.keys())[:8]},
                    "numeric": {k: numeric[k] for k in list(numeric.keys())[:8]},
                    "payload": {
                        "actor_id": payload.get("actor_id"),
                        "actor_type": payload.get("actor_type"),
                        "quest_id": payload.get("quest_id"),
                        "state_to": payload.get("state_to"),
                        "npc": payload.get("npc"),
                        "revision": payload.get("revision"),
                    },
                }
            )
        return compact
