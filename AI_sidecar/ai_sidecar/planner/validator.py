from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.planner.schemas import PlanHorizon, PlannerStepKind, StrategicPlan


@dataclass(slots=True)
class PlanValidationVerdict:
    ok: bool
    issues: list[str]
    warnings: list[str]
    normalized: StrategicPlan


@dataclass(slots=True)
class PlanValidator:
    tactical_budget_ms: int
    strategic_budget_ms: int

    def validate(self, *, plan: StrategicPlan, latency_ms: float) -> PlanValidationVerdict:
        issues: list[str] = []
        warnings: list[str] = []

        normalized_plan = plan.model_copy(deep=True)
        normalized_steps = []
        for idx, step in enumerate(normalized_plan.steps):
            kind = self._normalize_step_kind(step.kind)
            if step.kind != kind:
                warnings.append(f"step_kind_normalized:{step.step_id}:{step.kind}->{kind}")
            normalized_steps.append(step.model_copy(update={"kind": kind}))

            if idx >= 64:
                issues.append("step_count_exceeds_schema_limit")
                break

        normalized_plan = normalized_plan.model_copy(update={"steps": normalized_steps[:64]})

        horizon_budget_ms = self.tactical_budget_ms if normalized_plan.horizon == PlanHorizon.tactical else self.strategic_budget_ms
        if float(latency_ms) > float(horizon_budget_ms):
            issues.append(f"latency_budget_exceeded:{float(latency_ms):.3f}>{horizon_budget_ms}")

        max_steps_allowed = 12 if normalized_plan.horizon == PlanHorizon.tactical else 64
        if len(normalized_plan.steps) > max_steps_allowed:
            issues.append(f"step_limit_exceeded:{len(normalized_plan.steps)}>{max_steps_allowed}")

        if not normalized_plan.steps:
            issues.append("empty_plan_steps")

        if not normalized_plan.recommended_actions:
            warnings.append("no_recommended_actions")

        return PlanValidationVerdict(
            ok=not issues,
            issues=issues,
            warnings=warnings,
            normalized=normalized_plan,
        )

    def _normalize_step_kind(self, kind: PlannerStepKind | str) -> PlannerStepKind:
        value = str(kind or "").strip().lower()
        aliases = {
            "move": PlannerStepKind.travel,
            "route": PlannerStepKind.travel,
            "fight": PlannerStepKind.combat,
            "attack": PlannerStepKind.combat,
            "collect": PlannerStepKind.loot,
            "heal": PlannerStepKind.rest,
            "recover": PlannerStepKind.rest,
            "economy": PlannerStepKind.econ,
            "skill": PlannerStepKind.skill_up,
        }
        if value in aliases:
            return aliases[value]
        for item in PlannerStepKind:
            if item.value == value:
                return item
        return PlannerStepKind.task
