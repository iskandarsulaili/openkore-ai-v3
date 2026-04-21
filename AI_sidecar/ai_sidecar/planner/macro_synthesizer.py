from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.contracts.macros import EventAutomacro, MacroRoutine
from ai_sidecar.planner.schemas import MacroSynthesisProposal, StrategicPlan


@dataclass(slots=True)
class MacroSynthesizer:
    def synthesize(self, *, plan: StrategicPlan, min_repeat: int) -> MacroSynthesisProposal | None:
        if len(plan.steps) < max(2, min_repeat):
            return None

        travel_steps = [step for step in plan.steps if (step.kind or "").lower() in {"travel", "move", "route"}]
        combat_steps = [step for step in plan.steps if (step.kind or "").lower() in {"combat", "fight", "attack"}]
        if not travel_steps and not combat_steps:
            return None

        macro_name = f"plan_macro_{plan.plan_id[:18]}"
        lines: list[str] = []
        for step in plan.steps:
            kind = (step.kind or "").lower()
            target = (step.target or "").strip()
            if kind in {"travel", "move", "route"} and target:
                lines.append(f"do move {target}")
            elif kind in {"combat", "fight", "attack"}:
                lines.append("do attack")
            elif kind in {"loot", "collect"}:
                lines.append("do ai manual")
                lines.append("do ai auto")

        if not lines:
            return None

        routine = MacroRoutine(name=macro_name, lines=lines)
        automacro = EventAutomacro(
            name=f"auto_{macro_name}",
            conditions=["BaseLevel >= 1"],
            call=macro_name,
            parameters={"priority": "normal"},
        )
        return MacroSynthesisProposal(
            proposal_id=f"macro-{plan.plan_id[:20]}",
            bot_id=plan.bot_id,
            rationale="repeated stable sequence promoted from strategic plan",
            confidence=0.65,
            macros=[routine],
            event_macros=[],
            automacros=[automacro],
        )

