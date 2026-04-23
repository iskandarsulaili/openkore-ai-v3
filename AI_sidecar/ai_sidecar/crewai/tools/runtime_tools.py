from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import logging
from threading import RLock
from typing import Any
from uuid import uuid4

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.macros import EventAutomacro, MacroPublishRequest, MacroRoutine
from ai_sidecar.contracts.ml_subconscious import MLPredictRequest, ModelFamily

logger = logging.getLogger(__name__)


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

    def get_enriched_state(self, *, bot_id: str) -> dict[str, object]:
        state = self.runtime.enriched_state(bot_id=bot_id)
        logger.info(
            "crewai_tool_get_enriched_state",
            extra={"event": "crewai_tool_get_enriched_state", "bot_id": bot_id},
        )
        return {
            "ok": True,
            "bot_id": bot_id,
            "state": state.model_dump(mode="json"),
        }

    def list_active_macros(self, *, bot_id: str) -> dict[str, object]:
        publication = self.runtime.latest_macro_publication(bot_id=bot_id)
        manifest_payload: dict[str, object] | None = None
        manifest_error = ""
        try:
            publisher = getattr(self.runtime, "macro_publisher", None)
            manifest_path = publisher.manifest_file if publisher is not None else None
            if manifest_path is not None and manifest_path.exists():
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            manifest_error = str(exc)

        return {
            "ok": True,
            "bot_id": bot_id,
            "latest_publication": publication,
            "manifest": manifest_payload,
            "manifest_error": manifest_error,
        }

    def propose_actions(self, *, bot_id: str, intents: list[dict[str, object]]) -> dict[str, object]:
        now = datetime.now(UTC)
        results: list[dict[str, object]] = []
        accepted_total = 0
        rejected_total = 0
        logger.info(
            "crewai_tool_propose_actions",
            extra={"event": "crewai_tool_propose_actions", "bot_id": bot_id, "intent_count": len(intents)},
        )

        for idx, intent in enumerate(intents):
            if not isinstance(intent, dict):
                rejected_total += 1
                results.append({"intent_index": idx, "accepted": False, "reason": "invalid_intent_format"})
                continue

            command = str(intent.get("command") or intent.get("action") or intent.get("intent") or "").strip()
            if not command:
                rejected_total += 1
                results.append({"intent_index": idx, "accepted": False, "reason": "missing_command"})
                continue

            priority_raw = str(intent.get("priority_tier") or intent.get("priority") or "tactical").strip().lower()
            try:
                priority_tier = ActionPriorityTier(priority_raw)
            except Exception:
                priority_tier = ActionPriorityTier.tactical

            ttl_seconds = int(intent.get("expires_in_seconds") or settings.action_default_ttl_seconds)
            ttl_seconds = max(5, min(ttl_seconds, int(settings.action_default_ttl_seconds) * 6))
            action_id = str(intent.get("action_id") or f"crew-{uuid4().hex[:20]}")
            idempotency_key = str(intent.get("idempotency_key") or f"crew:{bot_id}:{command}"[:128])
            conflict_key = intent.get("conflict_key")
            metadata = dict(intent.get("metadata") or {})
            metadata.setdefault("source", "crewai")
            metadata.setdefault("intent_index", idx)

            proposal = ActionProposal(
                action_id=action_id,
                kind=str(intent.get("kind") or "command")[:64],
                command=command[:256],
                priority_tier=priority_tier,
                conflict_key=None if conflict_key is None else str(conflict_key)[:128],
                created_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
                idempotency_key=idempotency_key[:128],
                metadata=metadata,
            )
            accepted, status, queued_action_id, reason = self.runtime.queue_action(proposal, bot_id=bot_id)
            if accepted:
                accepted_total += 1
            else:
                rejected_total += 1

            results.append(
                {
                    "intent_index": idx,
                    "accepted": accepted,
                    "status": status.value,
                    "action_id": queued_action_id,
                    "reason": reason,
                }
            )

        return {
            "ok": True,
            "bot_id": bot_id,
            "accepted": accepted_total,
            "rejected": rejected_total,
            "results": results,
        }

    def publish_macro(self, *, bot_id: str, macro_bundle: dict[str, object]) -> dict[str, object]:
        macros_raw = macro_bundle.get("macros") if isinstance(macro_bundle, dict) else []
        event_raw = macro_bundle.get("event_macros") if isinstance(macro_bundle, dict) else []
        automacros_raw = macro_bundle.get("automacros") if isinstance(macro_bundle, dict) else []
        errors: list[str] = []

        macros: list[MacroRoutine] = []
        for item in macros_raw if isinstance(macros_raw, list) else []:
            try:
                macros.append(item if isinstance(item, MacroRoutine) else MacroRoutine.model_validate(item))
            except Exception as exc:
                errors.append(f"macro_invalid:{exc}")

        event_macros: list[MacroRoutine] = []
        for item in event_raw if isinstance(event_raw, list) else []:
            try:
                event_macros.append(item if isinstance(item, MacroRoutine) else MacroRoutine.model_validate(item))
            except Exception as exc:
                errors.append(f"event_macro_invalid:{exc}")

        automacros: list[EventAutomacro] = []
        for item in automacros_raw if isinstance(automacros_raw, list) else []:
            try:
                automacros.append(item if isinstance(item, EventAutomacro) else EventAutomacro.model_validate(item))
            except Exception as exc:
                errors.append(f"automacro_invalid:{exc}")

        if errors:
            return {
                "ok": False,
                "bot_id": bot_id,
                "message": "invalid_macro_bundle",
                "errors": errors,
            }

        enqueue_reload = bool(macro_bundle.get("enqueue_reload", True)) if isinstance(macro_bundle, dict) else True
        reload_conflict_key = str(macro_bundle.get("reload_conflict_key") or "macro_reload")
        macro_plugin = macro_bundle.get("macro_plugin")
        event_macro_plugin = macro_bundle.get("event_macro_plugin")
        request = MacroPublishRequest(
            meta=ContractMeta(contract_version=settings.contract_version, source="crewai_tool", bot_id=bot_id),
            target_bot_id=str(macro_bundle.get("target_bot_id") or bot_id) if isinstance(macro_bundle, dict) else bot_id,
            macros=macros,
            event_macros=event_macros,
            automacros=automacros,
            enqueue_reload=enqueue_reload,
            reload_conflict_key=reload_conflict_key,
            macro_plugin=None if macro_plugin is None else str(macro_plugin),
            event_macro_plugin=None if event_macro_plugin is None else str(event_macro_plugin),
        )
        logger.info(
            "crewai_tool_publish_macro",
            extra={
                "event": "crewai_tool_publish_macro",
                "bot_id": bot_id,
                "macro_count": len(macros),
                "event_macro_count": len(event_macros),
                "automacro_count": len(automacros),
            },
        )
        ok, publication_info, message = self.runtime.publish_macros(request)
        return {
            "ok": ok,
            "bot_id": bot_id,
            "message": message,
            "publication": publication_info,
        }

    def get_fleet_constraints(self, *, bot_id: str) -> dict[str, object]:
        response = self.runtime.fleet_constraints(bot_id=bot_id)
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        return {
            "ok": True,
            "bot_id": bot_id,
            "constraints": payload,
        }

    def write_reflection(self, *, bot_id: str, episode: dict[str, object]) -> dict[str, object]:
        summary = str(episode.get("summary") or episode.get("content") or "").strip()
        if not summary:
            return {"ok": False, "bot_id": bot_id, "message": "missing_summary"}
        event_type = str(episode.get("event_type") or "reflection")
        metadata = dict(episode.get("metadata") or {})
        tags = episode.get("tags")
        if isinstance(tags, list):
            metadata.setdefault("tags", tags)
        reflection_id = uuid4().hex
        logger.info(
            "crewai_tool_write_reflection",
            extra={"event": "crewai_tool_write_reflection", "bot_id": bot_id, "event_type": event_type},
        )
        try:
            self.runtime.memory.provider.add_episode(
                bot_id=bot_id,
                event_type=event_type,
                content=summary,
                metadata={"reflection_id": reflection_id, **metadata},
            )
            self.runtime.memory.provider.add_semantic(
                bot_id=bot_id,
                source="reflection",
                content=summary,
                metadata={"reflection_id": reflection_id},
            )
        except Exception as exc:
            logger.exception(
                "crewai_tool_write_reflection_failed",
                extra={"event": "crewai_tool_write_reflection_failed", "bot_id": bot_id},
            )
            return {"ok": False, "bot_id": bot_id, "message": f"reflection_write_failed:{exc}"}

        return {
            "ok": True,
            "bot_id": bot_id,
            "reflection_id": reflection_id,
            "event_type": event_type,
            "summary": summary,
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
        if tool == "get_enriched_state":
            return self.get_enriched_state(bot_id=str(args.get("bot_id") or bot_id))
        if tool == "list_active_macros":
            return self.list_active_macros(bot_id=str(args.get("bot_id") or bot_id))
        if tool == "propose_actions":
            intents = args.get("intents")
            return self.propose_actions(
                bot_id=str(args.get("bot_id") or bot_id),
                intents=list(intents) if isinstance(intents, list) else [],
            )
        if tool == "publish_macro":
            bundle = args.get("macro_bundle")
            return self.publish_macro(
                bot_id=str(args.get("bot_id") or bot_id),
                macro_bundle=dict(bundle) if isinstance(bundle, dict) else {},
            )
        if tool == "get_fleet_constraints":
            return self.get_fleet_constraints(bot_id=str(args.get("bot_id") or bot_id))
        if tool == "write_reflection":
            episode = args.get("episode")
            return self.write_reflection(
                bot_id=str(args.get("bot_id") or bot_id),
                episode=dict(episode) if isinstance(episode, dict) else {},
            )
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
