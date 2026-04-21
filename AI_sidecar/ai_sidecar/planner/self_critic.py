from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.planner.schemas import StrategicPlan


@dataclass(slots=True)
class CriticVerdict:
    ok: bool
    issues: list[str]
    risk_score: float


@dataclass(slots=True)
class SelfCritic:
    tactical_budget_ms: int
    strategic_budget_ms: int

    def evaluate(self, *, plan: StrategicPlan) -> CriticVerdict:
        issues: list[str] = []
        risk = float(plan.risk_score)

        if not plan.steps:
            issues.append("empty_steps")
        if len(plan.steps) > 64:
            issues.append("too_many_steps")

        forbidden = ("gm", "botcheck", "exploit")
        objective_lc = plan.objective.lower()
        for token in forbidden:
            if token in objective_lc:
                issues.append(f"forbidden_objective_token:{token}")

        for step in plan.steps:
            kind = (step.kind or "").strip().lower()
            if not kind:
                issues.append(f"invalid_step_kind:{step.step_id}")

        for action in plan.recommended_actions:
            cmd = (action.command or "").strip().lower()
            if "eval " in cmd or "system(" in cmd:
                issues.append(f"unsafe_action_command:{action.action_id}")

        if plan.horizon.value == "tactical":
            if len(plan.steps) > 12:
                issues.append("tactical_plan_too_long")
            if risk > 0.85:
                issues.append("tactical_risk_too_high")
        else:
            if risk > 0.95:
                issues.append("strategic_risk_too_high")

        return CriticVerdict(ok=not issues, issues=issues, risk_score=risk)

