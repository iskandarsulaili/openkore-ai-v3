from __future__ import annotations

from ai_sidecar.contracts.macros import EventAutomacro, MacroRoutine
from ai_sidecar.contracts.reflex import ReflexRule


class MicroMacroGenerator:
    def macro_name_for_rule(self, rule: ReflexRule) -> str:
        if rule.fallback_macro:
            return rule.fallback_macro
        return f"reflex_{rule.rule_id}".replace("-", "_")

    def event_automacro_name_for_rule(self, rule: ReflexRule) -> str:
        return f"reflex_auto_{rule.rule_id}".replace("-", "_")

    def build_micro_macro(self, rule: ReflexRule) -> MacroRoutine | None:
        name = self.macro_name_for_rule(rule)
        command = (rule.action_template.command or "").strip()
        lines: list[str] = [
            f"log [reflex] executing fallback macro for {rule.rule_id}",
        ]
        if command:
            lines.append(command)
        lines.append("stop")
        return MacroRoutine(name=name, lines=lines)

    def build_event_automacro(self, rule: ReflexRule) -> EventAutomacro | None:
        if not rule.event_macro_conditions:
            return None

        call_name = self.macro_name_for_rule(rule)

        return EventAutomacro(
            name=self.event_automacro_name_for_rule(rule),
            conditions=[item.strip() for item in rule.event_macro_conditions if item.strip()],
            call=call_name,
            parameters={"priority": "0"},
        )
