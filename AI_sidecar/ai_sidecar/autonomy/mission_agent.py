from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from pydantic import ValidationError

from ai_sidecar.contracts.autonomy import (
    AutonomyMissionDecision,
    AutonomyMissionDecisionRequest,
    AutonomyMissionDecisionResponse,
)
from ai_sidecar.providers.base import PlannerModelRequest

logger = logging.getLogger(__name__)


def _mission_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision"],
        "properties": {
            "decision": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "decision_class",
                    "selected_goal_key",
                    "mission_objective",
                    "mission_rationale",
                    "confidence",
                    "execution_hints",
                    "planner_handoff",
                    "annotations",
                ],
                "properties": {
                    "decision_class": {
                        "type": "string",
                        "enum": ["maintain", "adjust_objective", "replan"],
                    },
                    "selected_goal_key": {"type": "string", "minLength": 1, "maxLength": 64},
                    "mission_objective": {"type": "string", "minLength": 1, "maxLength": 512},
                    "mission_rationale": {"type": "string", "maxLength": 2048},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "execution_hints": {"type": "array", "items": {"type": "object"}},
                    "planner_handoff": {"type": "object"},
                    "annotations": {"type": "object"},
                },
            }
        },
    }


@dataclass(slots=True)
class MissionAgentService:
    model_router: Any
    timeout_seconds: float = 35.0
    max_retries: int = 1
    max_prompt_chars: int = 26000
    workload: str = "autonomy_mission_decision"

    async def decide(self, payload: AutonomyMissionDecisionRequest) -> AutonomyMissionDecisionResponse:
        user_prompt = self._user_prompt(payload=payload)
        request = PlannerModelRequest(
            bot_id=payload.meta.bot_id,
            trace_id=payload.meta.trace_id,
            task=self.workload,
            model="",
            system_prompt=self._system_prompt(),
            user_prompt=user_prompt,
            schema=_mission_schema(),
            timeout_seconds=float(self.timeout_seconds),
            max_retries=max(0, int(self.max_retries)),
            metadata={
                "source": "autonomy_mission_agent",
                "horizon": payload.context.horizon,
                "trigger_reasons": list(payload.trigger_reasons),
            },
        )

        response, route = await self.model_router.generate_with_fallback(request=request)
        if not bool(getattr(response, "ok", False)):
            error = str(getattr(response, "error", "") or "provider_route_failed")
            logger.warning(
                "autonomy_mission_agent_provider_failed",
                extra={
                    "event": "autonomy_mission_agent_provider_failed",
                    "bot_id": payload.meta.bot_id,
                    "trace_id": payload.meta.trace_id,
                    "workload": self.workload,
                    "error": error,
                },
            )
            return AutonomyMissionDecisionResponse(
                ok=False,
                message="provider_failed",
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                decision=None,
                errors=[error],
                provider=str(getattr(response, "provider", "") or ""),
                model=str(getattr(response, "model", "") or ""),
                latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
                route={
                    "workload": route.workload,
                    "selected_provider": route.selected_provider,
                    "selected_model": route.selected_model,
                    "planned_provider": route.planned_provider,
                    "planned_model": route.planned_model,
                    "attempted_providers": list(route.attempted_providers),
                    "fallback_used": bool(route.fallback_used),
                },
            )

        content = response.content if isinstance(response.content, dict) else {}
        candidate = content.get("decision") if isinstance(content.get("decision"), dict) else content
        try:
            decision = AutonomyMissionDecision.model_validate(candidate)
        except ValidationError as exc:
            logger.warning(
                "autonomy_mission_agent_validation_failed",
                extra={
                    "event": "autonomy_mission_agent_validation_failed",
                    "bot_id": payload.meta.bot_id,
                    "trace_id": payload.meta.trace_id,
                    "error_count": len(exc.errors()),
                },
            )
            return AutonomyMissionDecisionResponse(
                ok=False,
                message="invalid_decision_schema",
                trace_id=payload.meta.trace_id,
                bot_id=payload.meta.bot_id,
                decision=None,
                errors=["invalid_decision_schema", *[str(item.get("type") or "validation_error") for item in exc.errors()[:8]]],
                provider=str(getattr(response, "provider", "") or ""),
                model=str(getattr(response, "model", "") or ""),
                latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
                route={
                    "workload": route.workload,
                    "selected_provider": route.selected_provider,
                    "selected_model": route.selected_model,
                    "planned_provider": route.planned_provider,
                    "planned_model": route.planned_model,
                    "attempted_providers": list(route.attempted_providers),
                    "fallback_used": bool(route.fallback_used),
                },
            )

        return AutonomyMissionDecisionResponse(
            ok=True,
            message="ok",
            trace_id=payload.meta.trace_id,
            bot_id=payload.meta.bot_id,
            decision=decision,
            errors=[],
            provider=str(getattr(response, "provider", "") or ""),
            model=str(getattr(response, "model", "") or ""),
            latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
            route={
                "workload": route.workload,
                "selected_provider": route.selected_provider,
                "selected_model": route.selected_model,
                "planned_provider": route.planned_provider,
                "planned_model": route.planned_model,
                "attempted_providers": list(route.attempted_providers),
                "fallback_used": bool(route.fallback_used),
            },
        )

    def _system_prompt(self) -> str:
        return (
            "You are the authoritative mission-decision layer between deterministic state assembly and deterministic execution. "
            "Return JSON that matches schema exactly and never add top-level keys outside {decision}. "
            "Treat context.invariants.reasoning_protocol as mandatory eight-phase reasoning order and enforce it internally before writing output. "
            "Ground every claim in provided context only (assessment, deterministic_goal_stack, snapshot, enriched_state, runtime_facts, knowledge_summary, invariants). "
            "Do not fabricate formulas, drop rates, NPC scripts, map mechanics, or server-specific details that are not present in context. "
            "When facts are insufficient, abstain explicitly: keep selected_goal_key deterministic, set a conservative decision_class, list missing facts, and lower confidence. "
            "selected_goal_key must remain the deterministic selected goal key from context.selected_goal unless the key is absent/invalid in payload. "
            "execution_hints must use capability-bounded modes only: direct | config | macro | unsupported. "
            "For direct mode use tool=propose_actions and only bridge-safe command roots declared in context.invariants.capability_truth.direct.allowed_roots. "
            "For config mode use tool=plan_control_change and emit a planning request, not an apply action. "
            "For macro mode use tool=publish_macro and provide macro_bundle payload only. "
            "Use unsupported mode when no safe capability path exists and include missing_facts plus abstention reason. "
            "mission_objective must be concrete, rAthena-grounded, and executable by downstream planner. "
            "planner_handoff must summarize risk posture, constraints, and concrete next planning focus. "
            "annotations should include evidence_used and abstention metadata when applicable."
        )

    def _user_prompt(self, *, payload: AutonomyMissionDecisionRequest) -> str:
        body = {
            "meta": {
                "trace_id": payload.meta.trace_id,
                "bot_id": payload.meta.bot_id,
                "source": payload.meta.source,
            },
            "workload": payload.workload,
            "trigger_reasons": list(payload.trigger_reasons),
            "context": payload.context.model_dump(mode="json"),
            "metadata": dict(payload.metadata),
        }
        text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= self.max_prompt_chars:
            return text
        marker = "...<truncated_for_budget>..."
        tail_budget = max(0, int(self.max_prompt_chars * 0.7) - len(marker))
        head_budget = max(0, self.max_prompt_chars - tail_budget - len(marker))
        if head_budget <= 0:
            return text[-self.max_prompt_chars :]
        return f"{text[:head_budget]}{marker}{text[-tail_budget:]}"
