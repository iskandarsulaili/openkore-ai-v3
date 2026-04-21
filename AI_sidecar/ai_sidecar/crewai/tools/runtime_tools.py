from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any
from uuid import uuid4

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.macros import EventAutomacro, MacroRoutine
from ai_sidecar.contracts.ml_subconscious import MLPredictRequest, ModelFamily


@dataclass(slots=True)
class CrewToolFacade:
    runtime: Any
    _lock: RLock = field(default_factory=RLock)

    def get_bot_state(self, *, bot_id: str) -> dict[str, object]:
        state = self.runtime.enriched_state(bot_id=bot_id)
        return {
            "ok": True,
            "bot_id": bot_id,
            "state": state.model_dump(mode="json"),
            "queue_depth": self.runtime.action_queue.count(bot_id),
        }

    def query_memory(self, *, bot_id: str, query: str, limit: int = 10) -> dict[str, object]:
        limit = max(1, min(int(limit), 50))
        matches = self.runtime.memory_context(bot_id=bot_id, query=query, limit=limit)
        episodes = self.runtime.memory_recent_episodes(bot_id=bot_id, limit=min(limit * 2, 20))
        return {
            "ok": True,
            "bot_id": bot_id,
            "query": query,
            "limit": limit,
            "matches": matches,
            "episodes": episodes,
            "stats": self.runtime.memory_stats(bot_id=bot_id),
        }

    def check_reflex_rules(self, *, bot_id: str, trigger_type: str) -> dict[str, object]:
        rules = self.runtime.list_reflex_rules(bot_id=bot_id)
        active = [item for item in rules if item.enabled]
        trigger = trigger_type.strip()
        if trigger:
            filtered = []
            for rule in active:
                predicates = [*rule.trigger.all, *rule.trigger.any]
                if any((pred.fact == "event.event_type" and str(pred.value or "") == trigger) for pred in predicates):
                    filtered.append(rule)
            active = filtered
        return {
            "ok": True,
            "bot_id": bot_id,
            "trigger_type": trigger_type,
            "total_active": len(active),
            "rules": [item.model_dump(mode="json") for item in active],
        }

    def generate_macro_template(self, *, bot_id: str, action_sequence: list[str]) -> dict[str, object]:
        normalized = [str(item).strip() for item in action_sequence if str(item).strip()]
        name = f"crew_auto_{uuid4().hex[:10]}"
        macro = MacroRoutine(name=name, lines=normalized)
        automacro = EventAutomacro(
            name=f"on_{name}",
            conditions=["OnCharLogIn"],
            call=name,
            parameters={"bot_id": bot_id},
        )
        return {
            "ok": True,
            "bot_id": bot_id,
            "macro": macro.model_dump(mode="json"),
            "automacro": automacro.model_dump(mode="json"),
            "step_count": len(normalized),
        }

    def evaluate_plan_feasibility(self, *, bot_id: str, plan: dict[str, object]) -> dict[str, object]:
        queue_depth = self.runtime.action_queue.count(bot_id)
        max_queue = int(settings.action_max_queue_per_bot)
        risk_score = float(plan.get("risk_score") or 0.0)
        blockers: list[str] = []
        warnings: list[str] = []
        if queue_depth >= max_queue:
            blockers.append("queue_full")
        elif queue_depth >= int(max_queue * 0.8):
            warnings.append("queue_pressure_high")
        if risk_score >= 0.85:
            blockers.append("risk_score_too_high")
        elif risk_score >= 0.6:
            warnings.append("risk_score_elevated")

        feasible = not blockers
        return {
            "ok": True,
            "bot_id": bot_id,
            "feasible": feasible,
            "queue_depth": queue_depth,
            "max_queue": max_queue,
            "risk_score": risk_score,
            "blockers": blockers,
            "warnings": warnings,
            "recommended": "continue" if feasible else "replan_safe_mode",
        }

    def coordinate_with_fleet(self, *, bot_id: str, action: str, target_bots: list[str]) -> dict[str, object]:
        now = datetime.now(UTC)
        targets = [item for item in target_bots if item]
        if not targets:
            return {
                "ok": True,
                "bot_id": bot_id,
                "action": action,
                "coordinated": [],
                "message": "no_target_bots",
            }

        rows: list[dict[str, object]] = []
        for target in targets:
            proposal = ActionProposal(
                action_id=f"fleet-{uuid4().hex[:20]}",
                kind="command",
                command=str(action)[:256],
                priority_tier=ActionPriorityTier.strategic,
                conflict_key="fleet.coordination",
                created_at=now,
                expires_at=now + timedelta(seconds=settings.action_default_ttl_seconds),
                idempotency_key=f"fleet:{target}:{action}"[:128],
                metadata={"source": "crewai", "coordinator_bot_id": bot_id},
            )
            accepted, status, action_id, reason = self.runtime.queue_action(proposal, bot_id=target)
            rows.append(
                {
                    "target_bot_id": target,
                    "accepted": accepted,
                    "status": status.value,
                    "action_id": action_id,
                    "reason": reason,
                }
            )
        return {
            "ok": True,
            "bot_id": bot_id,
            "action": action,
            "count": len(rows),
            "coordinated": rows,
        }

    def ml_shadow_predict(
        self,
        *,
        bot_id: str,
        model_family: str,
        objective: str = "",
        planner_choice: dict[str, object] | None = None,
    ) -> dict[str, object]:
        planner_payload = dict(planner_choice or {})
        try:
            family = ModelFamily(model_family)
        except Exception:
            return {
                "ok": False,
                "bot_id": bot_id,
                "message": f"invalid_model_family:{model_family}",
                "allowed_families": [item.value for item in ModelFamily],
            }

        request = MLPredictRequest(
            meta=ContractMeta(contract_version=settings.contract_version, source="crewai_tool", bot_id=bot_id),
            model_family=family,
            state_features={},
            context={"objective": objective},
            planner_choice=planner_payload,
        )
        response = self.runtime.ml_predict(request)
        return {
            "ok": response.ok,
            "bot_id": bot_id,
            "family": family.value,
            "model_version": response.model_version,
            "recommendation": response.recommendation,
            "confidence": response.confidence,
            "shadow": response.shadow,
            "message": response.message,
        }

    def execute(self, *, bot_id: str, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        tool = tool_name.strip()
        args = dict(arguments)
        if tool == "get_bot_state":
            return self.get_bot_state(bot_id=str(args.get("bot_id") or bot_id))
        if tool == "query_memory":
            return self.query_memory(
                bot_id=str(args.get("bot_id") or bot_id),
                query=str(args.get("query") or ""),
                limit=int(args.get("limit") or 10),
            )
        if tool == "check_reflex_rules":
            return self.check_reflex_rules(
                bot_id=str(args.get("bot_id") or bot_id),
                trigger_type=str(args.get("trigger_type") or ""),
            )
        if tool == "generate_macro_template":
            seq = args.get("action_sequence")
            values = list(seq) if isinstance(seq, list) else []
            return self.generate_macro_template(bot_id=str(args.get("bot_id") or bot_id), action_sequence=values)
        if tool == "evaluate_plan_feasibility":
            plan = args.get("plan")
            return self.evaluate_plan_feasibility(
                bot_id=str(args.get("bot_id") or bot_id),
                plan=dict(plan) if isinstance(plan, dict) else {},
            )
        if tool == "coordinate_with_fleet":
            target_bots = args.get("target_bots")
            return self.coordinate_with_fleet(
                bot_id=str(args.get("bot_id") or bot_id),
                action=str(args.get("action") or ""),
                target_bots=list(target_bots) if isinstance(target_bots, list) else [],
            )
        if tool == "ml_shadow_predict":
            planner_choice = args.get("planner_choice")
            return self.ml_shadow_predict(
                bot_id=str(args.get("bot_id") or bot_id),
                model_family=str(args.get("model_family") or ""),
                objective=str(args.get("objective") or ""),
                planner_choice=dict(planner_choice) if isinstance(planner_choice, dict) else {},
            )
        return {"ok": False, "bot_id": bot_id, "message": f"unknown_tool:{tool}"}
