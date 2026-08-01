"""Regression test: observability-only reflex rules must not leak their
intent label into macro bodies (would fire Unknown command spam)."""
from __future__ import annotations

from ai_sidecar.contracts.actions import ActionPriorityTier
from ai_sidecar.contracts.reflex import (
    ReflexActionTemplate,
    ReflexCategory,
    ReflexFactOp,
    ReflexPlannerInterop,
    ReflexPredicate,
    ReflexRule,
    ReflexTriggerClause,
)
from ai_sidecar.reflex.micro_macro_generator import MicroMacroGenerator


def _rule(command: str, kind: str = "command", observability: bool = False) -> ReflexRule:
    metadata: dict[str, object] = {}
    if observability:
        metadata["observability_only"] = True
        metadata["bridge_compat"] = {
            "status": "observability_only",
            "original_command": "sit",
            "reason": "bridge_root_not_allowed",
        }
    return ReflexRule(
        rule_id="test_macro_safety",
        enabled=True,
        priority=80,
        trigger=ReflexTriggerClause(
            all=[ReflexPredicate(fact="event.event_type", op=ReflexFactOp.eq, value="snapshot.compact")]
        ),
        guards=[],
        action_template=ReflexActionTemplate(
            kind=kind,
            command=command,
            priority_tier=ActionPriorityTier.reflex,
            conflict_key="test",
            metadata=metadata,
        ),
        fallback_macro="reflex_test_fallback",
        category=ReflexCategory.survival,
        planner_interop=ReflexPlannerInterop.override,
    )


def test_observability_only_rule_macro_has_no_command_line() -> None:
    gen = MicroMacroGenerator()
    rule = _rule("extreme_overweight_alert", kind="log", observability=True)
    macro = gen.build_micro_macro(rule)
    assert macro is not None
    assert macro.name == "reflex_test_fallback"
    # The intent label must NOT appear as an executable line.
    assert "extreme_overweight_alert" not in macro.lines
    assert macro.lines[-1] == "stop"


def test_command_rule_macro_keeps_command_line() -> None:
    gen = MicroMacroGenerator()
    rule = _rule("use red_potion")
    macro = gen.build_micro_macro(rule)
    assert macro is not None
    assert "use red_potion" in macro.lines
    assert macro.lines[-1] == "stop"


def test_observability_rule_name_still_uses_fallback() -> None:
    gen = MicroMacroGenerator()
    rule = _rule("planner_rest_pending", kind="log", observability=True)
    assert gen.macro_name_for_rule(rule) == "reflex_test_fallback"
