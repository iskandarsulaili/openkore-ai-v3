from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
import logging
from threading import Thread
from typing import Any

from ai_sidecar.autonomy.goal_stack import compute_goal_stack
from ai_sidecar.autonomy.ro_knowledge import ROKnowledgeBundle, load_ro_knowledge
from ai_sidecar.contracts.autonomy import GoalStackState, SituationalAssessment
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import (
    CrewAutonomyRefinementRequest,
    CrewAutonomyRefinementResponse,
    CrewAutonomyDecisionOutput,
    CrewAutonomyDecisionContext,
)
from ai_sidecar.contracts.events import EventFamily, EventSeverity

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DecisionService:
    runtime: Any
    decision_version: str = "stage1-deterministic-v1"
    refinement_decision_version: str = "stage2-autonomy-crewai-refinement-v1"
    autonomy_task_hint: str = "autonomous_decision_intelligence"
    autonomy_required_agents: tuple[str, ...] = (
        "state_assessor",
        "progression_planner",
        "opportunistic_trader",
        "command_emitter",
    )
    ro_knowledge: ROKnowledgeBundle | None = None

    def __post_init__(self) -> None:
        if self.ro_knowledge is not None:
            return
        try:
            self.ro_knowledge = load_ro_knowledge()
            logger.info(
                "autonomy_stage3_ro_knowledge_loaded",
                extra={
                    "event": "autonomy_stage3_ro_knowledge_loaded",
                    "version": self.ro_knowledge.version,
                    "profiles": len(self.ro_knowledge.profile_lookup),
                    "playbook_jobs": len(self.ro_knowledge.playbooks_by_job),
                },
            )
        except Exception:
            self.ro_knowledge = None
            logger.exception(
                "autonomy_stage3_ro_knowledge_load_failed",
                extra={"event": "autonomy_stage3_ro_knowledge_load_failed"},
            )

    def decide(
        self,
        *,
        meta: ContractMeta,
        horizon: str,
        replan_reasons: list[str] | None = None,
    ) -> GoalStackState:
        snapshot = self.runtime.snapshot_cache.get(meta.bot_id)
        enriched = self.runtime.enriched_state(bot_id=meta.bot_id)
        assessment = self._build_assessment(
            bot_id=meta.bot_id,
            snapshot=snapshot,
            enriched=enriched,
            replan_reasons=replan_reasons or [],
        )
        computed = compute_goal_stack(assessment=assessment, horizon=horizon)

        advancement = assessment.job_advancement if isinstance(assessment.job_advancement, dict) else {}
        if (
            computed.selected_goal.goal_key.value == "job_advancement"
            and bool(advancement.get("supported"))
            and bool(advancement.get("ready"))
        ):
            logger.info(
                "autonomy_stage3_goal_transition_job_advancement",
                extra={
                    "event": "autonomy_stage3_goal_transition_job_advancement",
                    "bot_id": meta.bot_id,
                    "route_id": str(advancement.get("route_id") or ""),
                    "target_job": str(advancement.get("target_job") or ""),
                    "status": str(advancement.get("status") or ""),
                },
            )
        if (
            computed.selected_goal.goal_key.value == "leveling"
            and not bool(advancement.get("supported"))
            and str(advancement.get("status") or "") == "unsupported_job_route"
        ):
            logger.info(
                "autonomy_stage3_goal_transition_leveling_fallback",
                extra={
                    "event": "autonomy_stage3_goal_transition_leveling_fallback",
                    "bot_id": meta.bot_id,
                    "job_name": str(assessment.job_name or "unknown"),
                    "job_advancement_status": str(advancement.get("status") or "unknown"),
                },
            )

        goal_state = GoalStackState(
            bot_id=meta.bot_id,
            tick_id=assessment.tick_id,
            horizon=str(horizon),
            decision_version=self.decision_version,
            assessment=assessment,
            goal_stack=computed.goal_stack,
            selected_goal=computed.selected_goal,
        )

        refined_goal_state, refinement_errors = self._apply_crewai_refinement(
            meta=meta,
            goal_state=goal_state,
        )
        goal_state = refined_goal_state

        self.runtime.persist_goal_state(bot_id=meta.bot_id, state=goal_state)

        self.runtime._audit(
            level="info",
            event_type="autonomy_goal_decision",
            summary="deterministic goal selected",
            bot_id=meta.bot_id,
            payload={
                "decision_version": goal_state.decision_version,
                "horizon": goal_state.horizon,
                "tick_id": goal_state.tick_id,
                "selected_goal": goal_state.selected_goal.goal_key.value,
                "selected_objective": goal_state.selected_goal.objective,
                "refinement_applied": goal_state.decision_version == self.refinement_decision_version,
                "refinement_errors": list(refinement_errors),
                "stack": [
                    {
                        "goal_key": item.goal_key.value,
                        "priority_rank": item.priority_rank,
                        "active": item.active,
                    }
                    for item in goal_state.goal_stack
                ],
            },
        )
        self.runtime._emit_runtime_event(
            bot_id=meta.bot_id,
            event_family=EventFamily.system,
            event_type="autonomy.goal.decision",
            severity=EventSeverity.info,
            text=(
                f"goal decision selected={goal_state.selected_goal.goal_key.value} "
                f"horizon={goal_state.horizon} tick={goal_state.tick_id or ''}"
            ),
            numeric={
                "goal_priority": float(goal_state.selected_goal.priority_rank),
                "danger_score": float(goal_state.assessment.danger_score),
                "death_risk_score": float(goal_state.assessment.death_risk_score),
                "overweight_ratio": float(goal_state.assessment.overweight_ratio),
            },
            payload={
                "decision_version": goal_state.decision_version,
                "selected_goal": goal_state.selected_goal.goal_key.value,
                "selected_objective": goal_state.selected_goal.objective,
                "replan_reasons": list(goal_state.assessment.replan_reasons),
                "refinement_applied": goal_state.decision_version == self.refinement_decision_version,
                "refinement_errors": list(refinement_errors),
            },
            tags={"horizon": goal_state.horizon},
        )
        logger.info(
            "autonomy_goal_decision",
            extra={
                "event": "autonomy_goal_decision",
                "bot_id": meta.bot_id,
                "horizon": goal_state.horizon,
                "selected_goal": goal_state.selected_goal.goal_key.value,
                "selected_objective": goal_state.selected_goal.objective,
                "decision_version": goal_state.decision_version,
                "refinement_applied": goal_state.decision_version == self.refinement_decision_version,
                "refinement_errors": list(refinement_errors),
            },
        )
        return goal_state

    def _apply_crewai_refinement(
        self,
        *,
        meta: ContractMeta,
        goal_state: GoalStackState,
    ) -> tuple[GoalStackState, list[str]]:
        refinement_call = getattr(self.runtime, "crewai_autonomy_refine_decision", None)
        if not callable(refinement_call):
            logger.info(
                "autonomy_crewai_refine_skipped",
                extra={
                    "event": "autonomy_crewai_refine_skipped",
                    "bot_id": meta.bot_id,
                    "reason": "crewai_refinement_unavailable",
                },
            )
            return goal_state, ["crewai_refinement_unavailable"]

        context = CrewAutonomyDecisionContext(
            horizon=goal_state.horizon,
            assessment=goal_state.assessment,
            selected_goal=goal_state.selected_goal,
            goal_stack=list(goal_state.goal_stack),
            deterministic_priority_order=[item.goal_key.value for item in goal_state.goal_stack],
            replan_reasons=list(goal_state.assessment.replan_reasons),
            task_hint=self.autonomy_task_hint,
            required_agents=list(self.autonomy_required_agents),
        )
        request = CrewAutonomyRefinementRequest(
            meta=meta,
            task_hint=self.autonomy_task_hint,
            required_agents=list(self.autonomy_required_agents),
            decision_context=context,
            objective=goal_state.selected_goal.objective,
            metadata={"deterministic_decision_version": goal_state.decision_version},
        )
        logger.info(
            "autonomy_crewai_refine_handoff",
            extra={
                "event": "autonomy_crewai_refine_handoff",
                "bot_id": meta.bot_id,
                "task_hint": self.autonomy_task_hint,
                "required_agents": list(self.autonomy_required_agents),
                "selected_goal": goal_state.selected_goal.goal_key.value,
            },
        )

        try:
            raw_response = refinement_call(request)
            response = self._resolve_async_value(raw_response)
        except Exception as exc:
            logger.exception(
                "autonomy_crewai_refine_failed",
                extra={
                    "event": "autonomy_crewai_refine_failed",
                    "bot_id": meta.bot_id,
                    "error": type(exc).__name__,
                },
            )
            return goal_state, [f"crewai_refine_exception:{type(exc).__name__}"]

        if not isinstance(response, CrewAutonomyRefinementResponse):
            logger.warning(
                "autonomy_crewai_refine_unusable",
                extra={
                    "event": "autonomy_crewai_refine_unusable",
                    "bot_id": meta.bot_id,
                    "reason": "unexpected_response_type",
                    "response_type": type(response).__name__,
                },
            )
            return goal_state, ["crewai_refine_unexpected_response_type"]

        if not response.ok or response.decision_output is None:
            logger.info(
                "autonomy_crewai_refine_degraded",
                extra={
                    "event": "autonomy_crewai_refine_degraded",
                    "bot_id": meta.bot_id,
                    "ok": response.ok,
                    "errors": list(response.errors),
                },
            )
            return goal_state, [*list(response.errors), "crewai_refine_unusable_output"]

        decision_output = response.decision_output
        selected_goal = goal_state.selected_goal.goal_key.value
        requested_goal = str(decision_output.selected_goal_key or "").strip()
        if requested_goal and requested_goal != selected_goal:
            logger.warning(
                "autonomy_crewai_refine_goal_override_rejected",
                extra={
                    "event": "autonomy_crewai_refine_goal_override_rejected",
                    "bot_id": meta.bot_id,
                    "selected_goal": selected_goal,
                    "requested_goal": requested_goal,
                },
            )

        patched_state = self._merge_refinement_output(goal_state=goal_state, decision_output=decision_output)
        logger.info(
            "autonomy_crewai_refine_applied",
            extra={
                "event": "autonomy_crewai_refine_applied",
                "bot_id": meta.bot_id,
                "selected_goal": selected_goal,
                "objective": patched_state.selected_goal.objective,
            },
        )
        return patched_state, []

    def _merge_refinement_output(
        self,
        *,
        goal_state: GoalStackState,
        decision_output: CrewAutonomyDecisionOutput,
    ) -> GoalStackState:
        refined_objective = str(decision_output.refined_objective or "").strip()
        if not refined_objective:
            return goal_state

        existing_metadata = dict(goal_state.selected_goal.metadata)
        existing_annotations = existing_metadata.get("annotations") if isinstance(existing_metadata.get("annotations"), dict) else {}
        merged_annotations = {**dict(existing_annotations), **dict(decision_output.annotations)}
        refinement_metadata = {
            "task_hint": self.autonomy_task_hint,
            "required_agents": list(self.autonomy_required_agents),
            "situational_report": str(decision_output.situational_report or "")[:2048],
            "execution_translation": [str(item) for item in decision_output.execution_translation if str(item).strip()],
            "confidence": float(decision_output.confidence),
            "annotations": merged_annotations,
        }

        selected_goal = goal_state.selected_goal.model_copy(
            update={
                "objective": refined_objective[:512],
                "rationale": self._merge_rationale(
                    deterministic=goal_state.selected_goal.rationale,
                    refined=str(decision_output.rationale or ""),
                ),
                "metadata": {
                    **existing_metadata,
                    "stage2_refinement": refinement_metadata,
                },
            }
        )

        goal_stack = [
            selected_goal if item.goal_key == selected_goal.goal_key else item
            for item in goal_state.goal_stack
        ]
        return goal_state.model_copy(
            update={
                "decision_version": self.refinement_decision_version,
                "goal_stack": goal_stack,
                "selected_goal": selected_goal,
            }
        )

    def _merge_rationale(self, *, deterministic: str, refined: str) -> str:
        left = str(deterministic or "").strip()
        right = str(refined or "").strip()
        if not right:
            return left[:1024]
        merged = f"{left}; crewai_refinement:{right}" if left else f"crewai_refinement:{right}"
        return merged[:1024]

    def _resolve_async_value(self, value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)

        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                result["value"] = asyncio.run(value)
            except Exception as exc:  # pragma: no cover - passthrough to caller
                result["error"] = exc

        thread = Thread(target=worker, daemon=True)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _build_assessment(
        self,
        *,
        bot_id: str,
        snapshot: Any,
        enriched: Any,
        replan_reasons: list[str],
    ) -> SituationalAssessment:
        observed_at = self._snapshot_observed_at(snapshot)
        map_name = self._first_non_empty(
            self._safe_get(snapshot, "position", "map"),
            self._safe_get(enriched, "operational", "map"),
        )

        hp = self._to_int(self._first_non_empty(self._safe_get(snapshot, "vitals", "hp"), self._safe_get(enriched, "operational", "hp")))
        hp_max = self._to_int(
            self._first_non_empty(self._safe_get(snapshot, "vitals", "hp_max"), self._safe_get(enriched, "operational", "hp_max"))
        )

        skill_points = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "skill_points"),
                self._safe_get(enriched, "operational", "skill_points"),
            )
        )
        stat_points = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "stat_points"),
                self._safe_get(enriched, "operational", "stat_points"),
            )
        )

        base_exp = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "base_exp"),
                self._safe_get(enriched, "operational", "base_exp"),
            )
        )
        base_exp_max = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "base_exp_max"),
                self._safe_get(enriched, "operational", "base_exp_max"),
            )
        )
        job_exp = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "job_exp"),
                self._safe_get(enriched, "operational", "job_exp"),
            )
        )
        job_exp_max = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "job_exp_max"),
                self._safe_get(enriched, "operational", "job_exp_max"),
            )
        )

        active_quest_count = self._to_int(
            self._first_non_empty(
                self._safe_get(enriched, "quest", "active_objective_count"),
                self._len_or_zero(self._safe_get(snapshot, "quests")),
            )
        )
        objective_completion_ratio = self._to_float(self._safe_get(enriched, "quest", "objective_completion_ratio"))

        overweight_ratio = self._to_float(
            self._first_non_empty(
                self._safe_get(enriched, "inventory", "overweight_ratio"),
                self._calc_ratio(
                    numerator=self._safe_get(snapshot, "vitals", "weight"),
                    denominator=self._safe_get(snapshot, "vitals", "weight_max"),
                ),
            )
        )

        item_count = self._to_int(self._first_non_empty(self._safe_get(snapshot, "inventory", "item_count"), self._safe_get(enriched, "inventory", "item_count")))
        zeny = self._to_int(self._first_non_empty(self._safe_get(snapshot, "inventory", "zeny"), self._safe_get(enriched, "inventory", "zeny")))
        vendor_exposure = self._to_int(self._safe_get(enriched, "economy", "vendor_exposure"))

        is_disconnected = self._snapshot_disconnected(snapshot)
        reconnect_age_s = self._snapshot_reconnect_age_s(snapshot)
        in_combat = bool(self._first_non_empty(self._safe_get(snapshot, "combat", "is_in_combat"), self._safe_get(enriched, "operational", "in_combat"), False))

        danger_score = self._to_float(self._safe_get(enriched, "risk", "danger_score"))
        death_risk_score = self._to_float(self._safe_get(enriched, "risk", "death_risk_score"))

        base_level = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "base_level"),
                self._safe_get(enriched, "operational", "base_level"),
            )
        )
        job_level = self._to_int(
            self._first_non_empty(
                self._safe_get(snapshot, "progression", "job_level"),
                self._safe_get(enriched, "operational", "job_level"),
            )
        )
        job_name = self._first_non_empty(
            self._safe_get(snapshot, "progression", "job_name"),
            self._safe_get(enriched, "operational", "job_name"),
        )
        progression_recommendation = self._build_progression_recommendation(
            job_name=job_name,
            base_level=base_level,
        )
        job_advancement = self._build_job_advancement_assessment(
            job_name=job_name,
            base_level=base_level,
            job_level=job_level,
        )
        inventory_items = self._extract_inventory_items(snapshot)
        market_listings = self._extract_market_listings(snapshot=snapshot, enriched=enriched)
        opportunistic_signals = self._build_opportunistic_signals(
            snapshot=snapshot,
            enriched=enriched,
            replan_reasons=replan_reasons,
            map_name=map_name,
            in_combat=in_combat,
            item_count=item_count,
            overweight_ratio=overweight_ratio,
            vendor_exposure=vendor_exposure,
            market_listings=market_listings,
        )
        opportunistic_upgrades = self._build_opportunistic_upgrade_assessment(
            job_name=job_name,
            base_level=base_level,
            zeny=zeny,
            inventory_items=inventory_items,
            market_listings=market_listings,
            signals=opportunistic_signals,
        )

        trigger_flags: list[str] = []
        if self._to_ratio(hp, hp_max) <= 0.35:
            trigger_flags.append("hp_low")
        if danger_score >= 0.75:
            trigger_flags.append("danger_high")
        if death_risk_score >= 0.75:
            trigger_flags.append("death_risk_high")
        if is_disconnected:
            trigger_flags.append("disconnected")
        if reconnect_age_s is not None and reconnect_age_s >= 20.0:
            trigger_flags.append("reconnect_stale")
        if skill_points > 0 or stat_points > 0:
            trigger_flags.append("job_points_available")
        if overweight_ratio >= 0.85 or item_count >= 95:
            trigger_flags.append("upgrade_window_inventory_pressure")
        if bool(opportunistic_upgrades.get("actionable")):
            trigger_flags.append("upgrade_window_actionable")

        placeholders: list[str] = []
        if not bool(opportunistic_upgrades.get("knowledge_loaded")):
            placeholders.append("stage4_placeholder:opportunistic_knowledge_unavailable")
        elif not bool(opportunistic_upgrades.get("supported")):
            placeholders.append("stage4_placeholder:opportunistic_rules_unsupported")
        elif not bool(opportunistic_upgrades.get("actionable")):
            placeholders.append("stage4_placeholder:opportunistic_non_actionable")
        if not bool(progression_recommendation.get("knowledge_loaded")):
            placeholders.append("stage3_placeholder:progression_knowledge_unavailable")

        return SituationalAssessment(
            bot_id=bot_id,
            tick_id=str(self._safe_get(snapshot, "tick_id") or "") or None,
            observed_at=observed_at,
            map_name=map_name,
            in_combat=in_combat,
            hp_ratio=self._to_ratio(hp, hp_max),
            danger_score=danger_score,
            death_risk_score=death_risk_score,
            reconnect_age_s=reconnect_age_s,
            is_dead=hp <= 0,
            is_disconnected=is_disconnected,
            skill_points=skill_points,
            stat_points=stat_points,
            base_level=base_level,
            job_level=job_level,
            job_name=str(job_name or "") or None,
            base_exp_ratio=self._to_ratio(base_exp, base_exp_max),
            job_exp_ratio=self._to_ratio(job_exp, job_exp_max),
            active_quest_count=active_quest_count,
            objective_completion_ratio=max(0.0, min(1.0, objective_completion_ratio)),
            overweight_ratio=max(0.0, min(2.0, overweight_ratio)),
            item_count=item_count,
            zeny=zeny,
            vendor_exposure=vendor_exposure,
            replan_reasons=[str(item) for item in replan_reasons if str(item).strip()],
            trigger_flags=trigger_flags,
            placeholders=placeholders,
            progression_recommendation=progression_recommendation,
            job_advancement=job_advancement,
            opportunistic_upgrades=opportunistic_upgrades,
            raw_signals={
                "horizon_inputs": {"replan_reasons": [str(item) for item in replan_reasons]},
                "snapshot_present": snapshot is not None,
                "enriched_present": enriched is not None,
                "knowledge_version": self.ro_knowledge.version if self.ro_knowledge is not None else "unavailable",
                "opportunistic_signals": opportunistic_signals,
            },
        )

    def _build_opportunistic_upgrade_assessment(
        self,
        *,
        job_name: Any,
        base_level: int,
        zeny: int,
        inventory_items: list[dict[str, object]],
        market_listings: list[dict[str, object]],
        signals: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_job = str(job_name or "")
        signal_map = dict(signals or {})
        if self.ro_knowledge is None:
            return {
                "knowledge_loaded": False,
                "supported": False,
                "actionable": False,
                "status": "knowledge_unavailable",
                "job_name": normalized_job,
                "base_level": int(base_level),
                "zeny": int(max(0, int(zeny))),
                "signals": signal_map,
                "known_rule_ids": [],
                "opportunities": [],
                "non_actionable_reasons": ["stage4 ro knowledge unavailable"],
                "execution_hints": [],
                "recommended_opportunity": {},
            }

        try:
            assessment = self.ro_knowledge.assess_opportunistic_upgrades(
                job_name=normalized_job,
                base_level=base_level,
                zeny=zeny,
                inventory_items=inventory_items,
                market_listings=market_listings,
                signals=signal_map,
            )
        except Exception:
            logger.exception(
                "autonomy_stage4_opportunistic_assessment_failed",
                extra={
                    "event": "autonomy_stage4_opportunistic_assessment_failed",
                    "job_name": normalized_job,
                    "base_level": int(base_level),
                },
            )
            return {
                "knowledge_loaded": False,
                "supported": False,
                "actionable": False,
                "status": "knowledge_error",
                "job_name": normalized_job,
                "base_level": int(base_level),
                "zeny": int(max(0, int(zeny))),
                "signals": signal_map,
                "known_rule_ids": [],
                "opportunities": [],
                "non_actionable_reasons": ["stage4 opportunistic assessment failed"],
                "execution_hints": [],
                "recommended_opportunity": {},
            }

        opportunities = assessment.get("opportunities") if isinstance(assessment.get("opportunities"), list) else []
        top_opportunity = opportunities[0] if opportunities and isinstance(opportunities[0], dict) else {}
        execution_hints: list[dict[str, object]] = []
        if isinstance(top_opportunity, dict):
            hint = self._derive_execution_hint_from_opportunity(top_opportunity)
            if hint:
                execution_hints.append(hint)

        return {
            **assessment,
            "execution_hints": execution_hints,
            "recommended_opportunity": dict(top_opportunity) if isinstance(top_opportunity, dict) else {},
        }

    def _derive_execution_hint_from_opportunity(self, opportunity: dict[str, object]) -> dict[str, object]:
        execution_mode = str(opportunity.get("execution_mode") or "").strip().lower()
        payload = opportunity.get("execution_payload") if isinstance(opportunity.get("execution_payload"), dict) else {}
        rule_id = str(opportunity.get("rule_id") or "").strip()
        if execution_mode == "direct":
            command = str(payload.get("direct_command") or "").strip()
            if not command:
                return {}
            preconditions = [str(item) for item in payload.get("preconditions", []) if str(item).strip()]
            conflict_key = str(payload.get("conflict_key") or "opportunistic.upgrade")
            return {
                "execution_mode": "direct",
                "tool": "propose_actions",
                "rule_id": rule_id,
                "intents": [
                    {
                        "kind": "command",
                        "command": command,
                        "conflict_key": conflict_key,
                        "priority_tier": "tactical",
                        "metadata": {
                            "source": "autonomy_stage4",
                            "rule_id": rule_id,
                            "preconditions": preconditions,
                        },
                    }
                ],
            }
        if execution_mode == "config":
            return {
                "execution_mode": "config",
                "tool": "plan_control_change",
                "rule_id": rule_id,
                "request": dict(payload),
            }
        if execution_mode == "macro":
            return {
                "execution_mode": "macro",
                "tool": "publish_macro",
                "rule_id": rule_id,
                "macro_bundle": dict(payload),
            }
        return {}

    def _extract_inventory_items(self, snapshot: Any) -> list[dict[str, object]]:
        rows = self._safe_get(snapshot, "inventory_items")
        if not isinstance(rows, list):
            return []
        out: list[dict[str, object]] = []
        for item in rows:
            if isinstance(item, dict):
                row = dict(item)
            elif hasattr(item, "model_dump"):
                row = dict(item.model_dump(mode="json"))
            else:
                continue
            out.append(row)
        return out

    def _extract_market_listings(self, *, snapshot: Any, enriched: Any) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        snapshot_market = self._safe_get(snapshot, "market", "listings")
        if isinstance(snapshot_market, list):
            for item in snapshot_market:
                if isinstance(item, dict):
                    rows.append(dict(item))
                elif hasattr(item, "model_dump"):
                    rows.append(dict(item.model_dump(mode="json")))

        enriched_market = self._safe_get(enriched, "economy", "market_listings")
        if isinstance(enriched_market, list):
            for item in enriched_market:
                if isinstance(item, dict):
                    rows.append(dict(item))

        deduped: dict[tuple[str, str], dict[str, object]] = {}
        for item in rows:
            item_id = str(item.get("item_id") or "").strip().lower()
            if not item_id:
                continue
            source = str(item.get("source") or "").strip().lower()
            key = (item_id, source)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = item
                continue
            current_price = self._to_int(item.get("buy_price"))
            existing_price = self._to_int(existing.get("buy_price"))
            if current_price > 0 and (existing_price <= 0 or current_price < existing_price):
                deduped[key] = item

        return list(deduped.values())

    def _build_opportunistic_signals(
        self,
        *,
        snapshot: Any,
        enriched: Any,
        replan_reasons: list[str],
        map_name: str | None,
        in_combat: bool,
        item_count: int,
        overweight_ratio: float,
        vendor_exposure: int,
        market_listings: list[dict[str, object]],
    ) -> dict[str, object]:
        reason_text = " ".join(str(item).strip().lower() for item in replan_reasons if str(item).strip())
        fleet_objective = str(self._safe_get(enriched, "fleet_intent", "objective") or "").strip().lower()
        fleet_target_item = str(self._safe_get(enriched, "fleet_intent", "target_item") or "").strip().lower()
        fleet_target_mob = str(self._safe_get(enriched, "fleet_intent", "target_mob") or "").strip().lower()
        target_id = self._first_non_empty(
            self._safe_get(snapshot, "combat", "target_id"),
            self._safe_get(enriched, "encounter", "target_id"),
        )

        nearby_hostiles = self._to_int(
            self._first_non_empty(
                self._safe_get(enriched, "encounter", "nearby_hostiles"),
                self._safe_get(enriched, "operational", "nearby_hostiles"),
                0,
            )
        )
        actors = self._safe_get(snapshot, "actors")
        actor_hostiles = 0
        if isinstance(actors, list):
            for actor in actors:
                relation = ""
                if isinstance(actor, dict):
                    relation = str(actor.get("relation") or "").strip().lower()
                else:
                    relation = str(getattr(actor, "relation", "") or "").strip().lower()
                if relation in {"hostile", "enemy"}:
                    actor_hostiles += 1
        nearby_hostiles = max(nearby_hostiles, actor_hostiles)

        raw = self._safe_get(snapshot, "raw")
        raw_dict = raw if isinstance(raw, dict) else {}
        companion_raw = self._first_non_empty(
            self._safe_get(enriched, "operational", "companion_available"),
            self._safe_get(enriched, "social", "companion_available"),
            raw_dict.get("companion_available"),
            raw_dict.get("mercenary_active"),
            raw_dict.get("homunculus_active"),
            raw_dict.get("mercenary"),
            raw_dict.get("homunculus"),
        )
        if isinstance(companion_raw, (dict, list, tuple, set)):
            companion_available = bool(companion_raw)
        else:
            companion_available = self._to_bool(companion_raw)

        vending_active = self._to_bool(
            self._first_non_empty(
                self._safe_get(enriched, "economy", "vending_active"),
                raw_dict.get("vending_active"),
                raw_dict.get("is_vending"),
            )
        )

        map_known = bool(str(map_name or "").strip() and str(map_name or "").strip().lower() != "unknown")
        inventory_pressure = bool(float(overweight_ratio) >= 0.90 or int(item_count) >= 95)

        exploration_intent = bool(
            fleet_objective in {"explore", "exploration"}
            or any(token in reason_text for token in ("explore", "navigation", "route", "seek"))
        )
        card_gear_farming_intent = bool(
            fleet_objective in {"farm", "grind", "loot"}
            or bool(fleet_target_item)
            or bool(fleet_target_mob)
            or any(token in reason_text for token in ("farm", "grind", "card", "gear", "loot"))
        )
        mercenary_homunculus_intent = bool(
            companion_available
            or any(token in reason_text for token in ("mercenary", "homunculus", "companion"))
        )
        vending_intent = bool(
            vending_active
            or int(vendor_exposure) > 0
            or fleet_objective in {"trade", "vending"}
            or any(token in reason_text for token in ("vending", "vendor", "trade", "shop"))
        )

        signals = {
            "in_combat": bool(in_combat),
            "map_known": map_known,
            "nearby_hostiles": max(0, int(nearby_hostiles)),
            "market_listing_count": len(market_listings),
            "overweight_ratio": max(0.0, min(2.0, float(overweight_ratio))),
            "inventory_pressure": inventory_pressure,
            "vendor_exposure": max(0, int(vendor_exposure)),
            "targets_present": bool(target_id) or max(0, int(nearby_hostiles)) > 0,
            "exploration_intent": exploration_intent,
            "card_gear_farming_intent": card_gear_farming_intent,
            "mercenary_homunculus_intent": mercenary_homunculus_intent,
            "companion_available": companion_available,
            "vending_intent": vending_intent,
        }
        logger.info(
            "autonomy_stage4_opportunistic_signals_computed",
            extra={
                "event": "autonomy_stage4_opportunistic_signals_computed",
                "signals": dict(signals),
            },
        )
        return signals

    def _build_progression_recommendation(self, *, job_name: Any, base_level: int) -> dict[str, object]:
        normalized_job = str(job_name or "")
        if self.ro_knowledge is None:
            return {
                "knowledge_loaded": False,
                "job_name": normalized_job,
                "base_level": int(base_level),
                "profile_key": "unknown",
                "band_label": "unknown",
                "recommended_focus": "safe_grind",
                "target_maps": [],
                "objective_template": "continue deterministic leveling progression safely",
            }
        try:
            return self.ro_knowledge.recommend_leveling(job_name=normalized_job, base_level=base_level)
        except Exception:
            logger.exception(
                "autonomy_stage3_progression_recommendation_failed",
                extra={
                    "event": "autonomy_stage3_progression_recommendation_failed",
                    "job_name": normalized_job,
                    "base_level": int(base_level),
                },
            )
            return {
                "knowledge_loaded": False,
                "job_name": normalized_job,
                "base_level": int(base_level),
                "profile_key": "unknown",
                "band_label": "unknown",
                "recommended_focus": "safe_grind",
                "target_maps": [],
                "objective_template": "continue deterministic leveling progression safely",
            }

    def _build_job_advancement_assessment(
        self,
        *,
        job_name: Any,
        base_level: int,
        job_level: int,
    ) -> dict[str, object]:
        normalized_job = str(job_name or "")
        if self.ro_knowledge is None:
            return {
                "supported": False,
                "ready": False,
                "status": "knowledge_unavailable",
                "current_job": normalized_job or "unknown",
                "route_id": "",
                "target_job": "",
                "requirements": {},
                "missing_requirements": [],
                "known_routes": [],
                "playbook_steps": [],
                "location": {},
                "build_recommendation": {},
                "notes": ["stage-3 ro knowledge unavailable"],
            }
        try:
            return self.ro_knowledge.assess_job_advancement(
                job_name=normalized_job,
                base_level=base_level,
                job_level=job_level,
            )
        except Exception:
            logger.exception(
                "autonomy_stage3_job_advancement_assessment_failed",
                extra={
                    "event": "autonomy_stage3_job_advancement_assessment_failed",
                    "job_name": normalized_job,
                    "base_level": int(base_level),
                    "job_level": int(job_level),
                },
            )
            return {
                "supported": False,
                "ready": False,
                "status": "knowledge_error",
                "current_job": normalized_job or "unknown",
                "route_id": "",
                "target_job": "",
                "requirements": {},
                "missing_requirements": [],
                "known_routes": [],
                "playbook_steps": [],
                "location": {},
                "build_recommendation": {},
                "notes": ["stage-3 job advancement assessment failed"],
            }

    def _safe_get(self, obj: Any, *path: str) -> Any:
        current = obj
        for key in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
                continue
            current = getattr(current, key, None)
        return current

    def _first_non_empty(self, *values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    def _to_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _to_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enabled", "active"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disabled", "inactive", ""}:
            return False
        return bool(text)

    def _to_ratio(self, numerator: Any, denominator: Any) -> float:
        num = self._to_float(numerator)
        den = self._to_float(denominator)
        if den <= 0.0:
            return 0.0
        return max(0.0, min(2.0, num / den))

    def _len_or_zero(self, value: Any) -> int:
        if isinstance(value, (list, tuple, set)):
            return len(value)
        return 0

    def _calc_ratio(self, *, numerator: Any, denominator: Any) -> float:
        return self._to_ratio(numerator, denominator)

    def _snapshot_observed_at(self, snapshot: Any) -> datetime:
        observed_at = self._safe_get(snapshot, "observed_at")
        if isinstance(observed_at, datetime):
            return observed_at
        return datetime.now(UTC)

    def _snapshot_disconnected(self, snapshot: Any) -> bool:
        raw = self._safe_get(snapshot, "raw")
        if not isinstance(raw, dict):
            return False
        if raw.get("in_game") is False:
            return True
        status = str(raw.get("status") or raw.get("state") or raw.get("net_state") or "").strip().lower()
        return status in {
            "offline",
            "disconnected",
            "disconnect",
            "reconnecting",
            "connecting",
            "not_connected",
        }

    def _snapshot_reconnect_age_s(self, snapshot: Any) -> float | None:
        raw = self._safe_get(snapshot, "raw")
        if not isinstance(raw, dict):
            return None
        for key in ("reconnect_age_s", "disconnect_age_s", "offline_age_s"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None
