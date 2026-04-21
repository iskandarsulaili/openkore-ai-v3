from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.macros import EventAutomacro, MacroPublishRequest, MacroRoutine
from ai_sidecar.contracts.reflex import ReflexRule
from ai_sidecar.reflex.micro_macro_generator import MicroMacroGenerator


class EmitOutcome:
    def __init__(
        self,
        *,
        emitted: bool,
        execution_target: str,
        action_id: str | None,
        reason: str,
    ) -> None:
        self.emitted = emitted
        self.execution_target = execution_target
        self.action_id = action_id
        self.reason = reason


class ActionEmitter:
    _AUTOMACRO_PARAMETER_KEYS = {
        "priority",
        "exclusive",
        "run-once",
        "disabled",
        "timeout",
        "overrideAI",
        "delay",
    }

    def __init__(
        self,
        *,
        workspace_root: Path,
        contract_version: str,
        action_ttl_seconds: int,
    ) -> None:
        self._workspace_root = workspace_root
        self._contract_version = contract_version
        self._action_ttl_seconds = max(1, int(action_ttl_seconds))
        self._macro_generator = MicroMacroGenerator()
        self._lock = RLock()
        self._assets_prepared: set[tuple[str, str]] = set()

    def emit_chain(
        self,
        *,
        bot_id: str,
        rule: ReflexRule,
        trigger_id: str,
        queue_action: callable,
        publish_macros: callable,
    ) -> EmitOutcome:
        direct = self._emit_direct_queue_action(
            bot_id=bot_id,
            rule=rule,
            trigger_id=trigger_id,
            queue_action=queue_action,
        )
        if direct.emitted:
            return direct

        micro = self._emit_micro_macro(
            bot_id=bot_id,
            rule=rule,
            trigger_id=trigger_id,
            queue_action=queue_action,
            publish_macros=publish_macros,
        )
        if micro.emitted:
            return micro

        event_macro = self._emit_event_macro_trigger(
            bot_id=bot_id,
            rule=rule,
            trigger_id=trigger_id,
            queue_action=queue_action,
            publish_macros=publish_macros,
        )
        if event_macro.emitted:
            return event_macro

        return EmitOutcome(
            emitted=False,
            execution_target="none",
            action_id=None,
            reason=f"direct={direct.reason}|micro={micro.reason}|event={event_macro.reason}",
        )

    def prepare_rule_assets(self, *, bot_id: str, rule: ReflexRule, publish_macros: callable) -> tuple[bool, str]:
        needs_micro = bool(rule.fallback_macro)
        needs_event = bool(rule.event_macro_conditions)
        if not needs_micro and not needs_event:
            return True, "assets_not_required"

        key = (bot_id, rule.rule_id)
        with self._lock:
            if key in self._assets_prepared:
                return True, "assets_cached"

        existing_macros, existing_event_macros, existing_automacros = self._load_existing_generated_assets()

        macro_by_name: dict[str, MacroRoutine] = {item.name: item for item in existing_macros}
        event_macro_by_name: dict[str, MacroRoutine] = {item.name: item for item in existing_event_macros}
        automacro_by_name: dict[str, EventAutomacro] = {item.name: item for item in existing_automacros}

        micro_macro = self._macro_generator.build_micro_macro(rule)
        if micro_macro is not None:
            macro_by_name[micro_macro.name] = micro_macro

        event_auto = self._macro_generator.build_event_automacro(rule)
        if event_auto is not None:
            automacro_by_name[event_auto.name] = event_auto

        request = MacroPublishRequest(
            meta=ContractMeta(
                contract_version=self._contract_version,
                source="sidecar-reflex",
                bot_id=bot_id,
            ),
            target_bot_id=bot_id,
            macros=[macro_by_name[name] for name in sorted(macro_by_name)],
            event_macros=[event_macro_by_name[name] for name in sorted(event_macro_by_name)],
            automacros=[automacro_by_name[name] for name in sorted(automacro_by_name)],
            enqueue_reload=True,
            reload_conflict_key="reflex.macro_reload",
        )

        ok, _, message = publish_macros(request)
        if not ok:
            return False, f"asset_publish_failed:{message}"

        with self._lock:
            self._assets_prepared.add(key)
        return True, "assets_prepared"

    def _emit_direct_queue_action(
        self,
        *,
        bot_id: str,
        rule: ReflexRule,
        trigger_id: str,
        queue_action: callable,
    ) -> EmitOutcome:
        command = (rule.action_template.command or "").strip()
        kind = (rule.action_template.kind or "command").strip().lower()
        if not command:
            return EmitOutcome(emitted=False, execution_target="direct_queue_action", action_id=None, reason="empty_command")
        if kind not in {"command", "macro_reload"}:
            return EmitOutcome(
                emitted=False,
                execution_target="direct_queue_action",
                action_id=None,
                reason=f"unsupported_direct_kind:{kind}",
            )

        proposal = self._build_proposal(
            bot_id=bot_id,
            rule=rule,
            trigger_id=trigger_id,
            execution_target="direct_queue_action",
            kind=kind,
            command=command,
        )
        accepted, status, action_id, reason = queue_action(proposal, bot_id)
        if accepted:
            return EmitOutcome(emitted=True, execution_target="direct_queue_action", action_id=action_id, reason="action_queued")

        return EmitOutcome(
            emitted=False,
            execution_target="direct_queue_action",
            action_id=action_id,
            reason=f"queue_rejected:{status.value}:{reason}",
        )

    def _emit_micro_macro(
        self,
        *,
        bot_id: str,
        rule: ReflexRule,
        trigger_id: str,
        queue_action: callable,
        publish_macros: callable,
    ) -> EmitOutcome:
        if not rule.fallback_macro:
            return EmitOutcome(
                emitted=False,
                execution_target="published_micro_macro",
                action_id=None,
                reason="fallback_macro_missing",
            )

        prepared, prep_reason = self.prepare_rule_assets(bot_id=bot_id, rule=rule, publish_macros=publish_macros)
        if not prepared:
            return EmitOutcome(
                emitted=False,
                execution_target="published_micro_macro",
                action_id=None,
                reason=prep_reason,
            )

        macro_name = self._macro_generator.macro_name_for_rule(rule)
        proposal = self._build_proposal(
            bot_id=bot_id,
            rule=rule,
            trigger_id=trigger_id,
            execution_target="published_micro_macro",
            kind="command",
            command=f"macro {macro_name}",
        )
        accepted, status, action_id, reason = queue_action(proposal, bot_id)
        if accepted:
            return EmitOutcome(emitted=True, execution_target="published_micro_macro", action_id=action_id, reason="action_queued")

        return EmitOutcome(
            emitted=False,
            execution_target="published_micro_macro",
            action_id=action_id,
            reason=f"queue_rejected:{status.value}:{reason}",
        )

    def _emit_event_macro_trigger(
        self,
        *,
        bot_id: str,
        rule: ReflexRule,
        trigger_id: str,
        queue_action: callable,
        publish_macros: callable,
    ) -> EmitOutcome:
        if not rule.event_macro_conditions:
            return EmitOutcome(
                emitted=False,
                execution_target="eventmacro_trigger",
                action_id=None,
                reason="event_macro_conditions_missing",
            )

        prepared, prep_reason = self.prepare_rule_assets(bot_id=bot_id, rule=rule, publish_macros=publish_macros)
        if not prepared:
            return EmitOutcome(
                emitted=False,
                execution_target="eventmacro_trigger",
                action_id=None,
                reason=prep_reason,
            )

        automacro_name = self._macro_generator.event_automacro_name_for_rule(rule)
        proposal = self._build_proposal(
            bot_id=bot_id,
            rule=rule,
            trigger_id=trigger_id,
            execution_target="eventmacro_trigger",
            kind="command",
            command=f"eventMacro {automacro_name}",
        )
        accepted, status, action_id, reason = queue_action(proposal, bot_id)
        if accepted:
            return EmitOutcome(emitted=True, execution_target="eventmacro_trigger", action_id=action_id, reason="action_queued")

        return EmitOutcome(
            emitted=False,
            execution_target="eventmacro_trigger",
            action_id=action_id,
            reason=f"queue_rejected:{status.value}:{reason}",
        )

    def _build_proposal(
        self,
        *,
        bot_id: str,
        rule: ReflexRule,
        trigger_id: str,
        execution_target: str,
        kind: str,
        command: str,
    ) -> ActionProposal:
        now = datetime.now(UTC)
        action_id = f"reflex-{rule.rule_id}-{uuid4().hex[:16]}"
        conflict_key = rule.action_template.conflict_key or f"reflex.{rule.rule_id}"
        return ActionProposal(
            action_id=action_id,
            kind=kind,
            command=command,
            priority_tier=ActionPriorityTier.reflex,
            conflict_key=conflict_key,
            created_at=now,
            expires_at=now + timedelta(seconds=self._action_ttl_seconds),
            idempotency_key=f"reflex:{rule.rule_id}:{trigger_id}:{execution_target}",
            metadata={
                "source": "reflex",
                "latency_budget_ms": 100,
                "preconditions": [],
                "postconditions": [],
                "rollback_hint": "teleport_to_save",
                "reflex_rule_id": rule.rule_id,
                "reflex_trigger_id": trigger_id,
                "execution_target": execution_target,
                **dict(rule.action_template.metadata),
            },
        )

    def _load_existing_generated_assets(self) -> tuple[list[MacroRoutine], list[MacroRoutine], list[EventAutomacro]]:
        macro_file = self._workspace_root / "control" / "ai_sidecar_generated_macros.txt"
        event_file = self._workspace_root / "control" / "ai_sidecar_generated_eventmacros.txt"
        macros = self._parse_macro_routines(macro_file)
        event_macros, automacros = self._parse_event_macro_file(event_file)
        return macros, event_macros, automacros

    def _parse_macro_routines(self, path: Path) -> list[MacroRoutine]:
        if not path.exists():
            return []
        return self._extract_macro_blocks(path.read_text(encoding="utf-8"))

    def _parse_event_macro_file(self, path: Path) -> tuple[list[MacroRoutine], list[EventAutomacro]]:
        if not path.exists():
            return [], []
        content = path.read_text(encoding="utf-8")
        macros = self._extract_macro_blocks(content)
        automacros = self._extract_automacro_blocks(content)
        return macros, automacros

    def _extract_macro_blocks(self, content: str) -> list[MacroRoutine]:
        lines = content.splitlines()
        idx = 0
        out: list[MacroRoutine] = []
        while idx < len(lines):
            line = lines[idx].strip()
            if not (line.startswith("macro ") and line.endswith("{")):
                idx += 1
                continue
            name = line[len("macro ") : -1].strip()
            idx += 1
            body: list[str] = []
            while idx < len(lines):
                current = lines[idx].rstrip("\n")
                if current.strip() == "}":
                    break
                body.append(current.strip())
                idx += 1
            out.append(MacroRoutine(name=name, lines=[item for item in body if item]))
            idx += 1
        return out

    def _extract_automacro_blocks(self, content: str) -> list[EventAutomacro]:
        lines = content.splitlines()
        idx = 0
        out: list[EventAutomacro] = []
        while idx < len(lines):
            line = lines[idx].strip()
            if not (line.startswith("automacro ") and line.endswith("{")):
                idx += 1
                continue
            name = line[len("automacro ") : -1].strip()
            idx += 1
            call = ""
            conditions: list[str] = []
            parameters: dict[str, str] = {}
            while idx < len(lines):
                current = lines[idx].strip()
                if current == "}":
                    break
                if not current:
                    idx += 1
                    continue
                if current.startswith("call "):
                    call = current[len("call ") :].strip()
                    idx += 1
                    continue
                parts = current.split(maxsplit=1)
                if len(parts) == 2 and parts[0] in self._AUTOMACRO_PARAMETER_KEYS:
                    parameters[parts[0]] = parts[1]
                else:
                    conditions.append(current)
                idx += 1

            if call:
                out.append(
                    EventAutomacro(
                        name=name,
                        conditions=conditions,
                        call=call,
                        parameters=parameters,
                    )
                )
            idx += 1
        return out

