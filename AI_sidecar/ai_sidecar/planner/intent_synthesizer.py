from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.planner.schemas import PlannerContext, TacticalIntent


@dataclass(slots=True)
class IntentSynthesizer:
    def synthesize(self, *, context: PlannerContext) -> list[TacticalIntent]:
        intents: list[TacticalIntent] = []
        risk = float((context.state.get("risk") or {}).get("danger_score") or 0.0)
        in_combat = bool((context.state.get("operational") or {}).get("in_combat") or False)

        if risk >= 0.7:
            intents.append(
                TacticalIntent(
                    intent_id="intent-safety-1",
                    objective="Stabilize safety posture and reduce immediate risk",
                    priority=5,
                    constraints=["avoid_high_risk_maps", "prefer_escape_or_recover"],
                    expected_latency_ms=500,
                )
            )

        if in_combat:
            intents.append(
                TacticalIntent(
                    intent_id="intent-combat-1",
                    objective="Resolve active combat safely with retreat threshold",
                    priority=20,
                    constraints=["maintain_hp_threshold", "avoid_overpull"],
                    expected_latency_ms=1200,
                )
            )

        intents.append(
            TacticalIntent(
                intent_id="intent-objective-1",
                objective=context.objective,
                priority=50,
                constraints=[f"horizon={context.horizon.value}"],
                expected_latency_ms=1800,
            )
        )
        return intents

