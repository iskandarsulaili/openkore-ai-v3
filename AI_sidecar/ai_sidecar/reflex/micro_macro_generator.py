from __future__ import annotations

from ai_sidecar.contracts.macros import EventAutomacro, MacroRoutine
from ai_sidecar.contracts.reflex import ReflexRule


class MicroMacroGenerator:
    def _resolve_carried_heal(self) -> str:
        """Best heal the bot ACTUALLY carries, else "" (never invent an item).

        Reads the sidecar's live snapshot inventory (same source the SELL/heal
        paths use). Prefers potions, then herbs/apple — mirrors HighFreqReflex.
        """
        try:
            import json as _json
            import os as _os
            # 1) character status snapshot written by the bridge (authoritative).
            for _p in (
                "/home/lot399/openkore-ai-v3/data/charstatus/charstatus_Local_rAthena_AI_World_testbot99.json",
            ):
                if not _os.path.isfile(_p):
                    continue
                _d = _json.load(open(_p, encoding="utf-8", errors="replace"))
                _items = (
                    (_d.get("inventory") or {}).get("items")
                    or _d.get("inventory_items")
                    or []
                )
                names = []
                for _it in _items:
                    _n = (_it.get("name") if isinstance(_it, dict) else str(_it)) or ""
                    if _n:
                        names.append(_n.strip().lower())
                for pref in ("white potion", "orange potion", "red potion",
                             "novice potion", "apple", "green herb", "red herb"):
                    for _n in names:
                        if pref in _n:
                            return f"use {_n.title()}"
        except Exception:
            return ""
        return ""

    def macro_name_for_rule(self, rule: ReflexRule) -> str:
        if rule.fallback_macro:
            return rule.fallback_macro
        return f"reflex_{rule.rule_id}".replace("-", "_")

    def event_automacro_name_for_rule(self, rule: ReflexRule) -> str:
        return f"reflex_auto_{rule.rule_id}".replace("-", "_")

    def build_micro_macro(self, rule: ReflexRule) -> MacroRoutine | None:
        name = self.macro_name_for_rule(rule)
        command = (rule.action_template.command or "").strip()
        # INVENTORY-AWARE HEAL MACRO (2026-09-14): a heal rule carrying a hardcoded
        # `use Red Potion` (reflex_rules.yaml / _default_rules) must NOT bake that
        # into the macro — a broke bot carrying only herbs/Apple does not own a Red
        # Potion, so the fallback macro silently failed while the bot died (live:
        # macro reflex_survival_heal -> `use Red Potion`, bot at HP 10/280 with 15
        # hostile mobs). Resolve to the best CARRIED heal at macro-build time; if
        # nothing is carried, emit a safe no-op log instead of a failing use.
        try:
            from ai_sidecar.knowledge_loader import get_items  # noqa: F401
            _low = command.lower()
            if _low.startswith("use ") and any(
                k in _low for k in ("potion", "herb", "apple", "berry")
            ):
                _carried = self._resolve_carried_heal()
                command = _carried if _carried else ""
                if not _carried:
                    pass  # no carried heal -> macro becomes a log-only no-op
        except Exception:
            pass
        lines: list[str] = [
            f"log reflex executing fallback macro for {rule.rule_id}",
        ]
        # Include the actual command in the macro body so the fallback
        # actually does something useful (e.g. "sit", "use red_potion").
        # EXCEPT observability-only rules: their command text is an intent
        # label (e.g. "extreme_overweight_alert"), NOT an executable root —
        # emitting it would fire "Unknown command" spam. Their recovery is
        # handled by the pdca action paths, so the macro is a no-op log.
        is_observability = (
            (rule.action_template.kind or "command").strip().lower() != "command"
            or bool(dict(rule.action_template.metadata or {}).get("observability_only"))
        )
        if command and not is_observability:
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
