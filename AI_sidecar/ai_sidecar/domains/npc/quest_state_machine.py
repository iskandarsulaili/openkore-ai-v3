"""Quest State Machine — branching dialogue for complex RO quests.

Replaces simple "h r1 c r0" dialogue sequences with full state machines
that handle multi-step quests with conditional branches, item collection,
zeny payment, and NPC chains across maps.

Each quest is a directed graph of states with transitions driven by
NPC responses, inventory checks, or player actions. The engine produces
HeuristicAction commands that the bot executes in sequence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_QUEST_DB: Any = None  # Initialised lazily — holds dict after first load


def _load_quest_data() -> dict[str, Any]:
    """Load quest state machine definitions from YAML (cached)."""
    global _QUEST_DB
    if _QUEST_DB is not None:
        return _QUEST_DB
    if yaml is None:
        _QUEST_DB = {}
        return _QUEST_DB
    path = _DATA_DIR / "quest_state_machines.yaml"
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
            _QUEST_DB = raw.get("quest_state_machines", {})
    else:
        _QUEST_DB = {}
    return _QUEST_DB


# ── Internal state storage ──────────────────────────────────────────

_ACTIVE_STATES: dict[str, dict[str, str]] = {}
"""bot_id -> {quest_name_fqn: state_name}

Fully-qualified quest name is typically ``{source_job}_to_{target_job}``
— the same key used in the YAML file.
"""


def _fqn(quest_name: str) -> str:
    """Normalise a quest name to its fully-qualified key.

    Accepts either the short display name (eg "Knight Identity Quest")
    or the FQN key (eg "swordman_to_knight").
    """
    data = _load_quest_data()
    if quest_name in data:
        return quest_name
    for key, qdef in data.items():
        if qdef.get("name", "") == quest_name:
            return key
    return quest_name


# ====================================================================
# Public API
# ====================================================================


class QuestStateMachine:
    """State-machine engine for branching RO trans-class quests.

    Usage::

        qsm = QuestStateMachine()
        actions = qsm.start_quest("bot_1", "swordman_to_knight")
        # -> [HeuristicAction(kind="command", command="quest_talk prt_castle 31 208 h c c", ...)]

        # On next assessment, pass current signals:
        actions = qsm.assess(bot_id="bot_1", signals={...})
    """

    def __init__(self) -> None:
        self._data = _load_quest_data()
        self._state: dict[str, dict[str, str]] = {}
        """bot_id -> {quest_fqn: current_state_name}"""

        self._quest_status: dict[str, dict[str, str]] = {}
        """bot_id -> {quest_fqn: status} — one of active | completed | failed"""

        # Track dialogue branch choices for choose states
        self._pending_choices: dict[str, dict[str, dict]] = {}
        """bot_id -> {quest_fqn: {answer_text, transitions, next_state_on_complete}}"""

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def get_available_quests(self, current_job: str, base_level: int) -> list[dict]:
        """Return quest definitions that match the bot's current job + level."""
        results: list[dict] = []
        for key, qdef in self._data.items():
            if qdef.get("source_job", "") == current_job:
                if base_level >= qdef.get("level_req", 1):
                    entry = dict(qdef)
                    entry["fqn"] = key
                    results.append(entry)
        return results

    def get_quest_def(self, quest_name: str) -> dict | None:
        """Return the full definition dict for a quest (by FQN or display name)."""
        fqn = _fqn(quest_name)
        return self._data.get(fqn)

    def get_current_state(self, bot_id: str, quest_name: str) -> str | None:
        """Return the current state name for a quest, or None if not started."""
        fqn = _fqn(quest_name)
        return self._state.get(bot_id, {}).get(fqn)

    def get_status(self, bot_id: str, quest_name: str) -> str:
        """Return 'active', 'completed', 'failed', or 'not_started'."""
        fqn = _fqn(quest_name)
        st = self._quest_status.get(bot_id, {}).get(fqn, "")
        return st if st else "not_started"

    def list_active_quests(self, bot_id: str) -> list[dict]:
        """Return all active quests for a bot with current state info."""
        results: list[dict] = []
        bot_states = self._state.get(bot_id, {})
        for fqn, state_name in bot_states.items():
            if self._quest_status.get(bot_id, {}).get(fqn) == "completed":
                continue
            qdef = self._data.get(fqn, {})
            state_def = qdef.get("states", {}).get(state_name, {})
            results.append({
                "fqn": fqn,
                "name": qdef.get("name", fqn),
                "state": state_name,
                "state_type": state_def.get("type", "unknown"),
                "level_req": qdef.get("level_req", 1),
            })
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_quest(self, bot_id: str, quest_name: str) -> list[HeuristicAction]:
        """Begin a quest from its start state.

        Returns a list of HeuristicAction commands.
        """
        fqn = _fqn(quest_name)
        qdef = self._data.get(fqn)
        if not qdef:
            logger.warning("[quest_sm] %s: unknown quest '%s'", bot_id, quest_name)
            return []

        if self.get_status(bot_id, fqn) == "completed":
            logger.info("[quest_sm] %s: quest '%s' already completed", bot_id, fqn)
            return []

        start_state_name = "start"
        start_state = qdef.get("states", {}).get(start_state_name)
        if not start_state:
            logger.warning("[quest_sm] %s: quest '%s' has no 'start' state", bot_id, fqn)
            return []

        self._state.setdefault(bot_id, {})[fqn] = start_state_name
        self._quest_status.setdefault(bot_id, {})[fqn] = "active"

        logger.info("[quest_sm] %s: started quest '%s'", bot_id, qdef.get("name", fqn))
        return self._emit_actions_for_state(bot_id, fqn, start_state_name, start_state)

    def advance(self, bot_id: str, quest_name: str, next_state_name: str) -> list[HeuristicAction]:
        """Move to a named state and emit its actions.

        Call this when a transition condition is met (eg item collected,
        NPC dialogue finished, payment made).
        """
        fqn = _fqn(quest_name)
        qdef = self._data.get(fqn)
        if not qdef:
            return []

        state_def = qdef.get("states", {}).get(next_state_name)
        if not state_def:
            logger.warning("[quest_sm] %s: unknown state '%s' in quest '%s'",
                           bot_id, next_state_name, fqn)
            return []

        self._state.setdefault(bot_id, {})[fqn] = next_state_name

        # Handle terminal states
        if state_def.get("type") == "complete":
            self._quest_status.setdefault(bot_id, {})[fqn] = "completed"
            logger.info("[quest_sm] %s: completed quest '%s'", bot_id, qdef.get("name", fqn))

        return self._emit_actions_for_state(bot_id, fqn, next_state_name, state_def)

    def fail_quest(self, bot_id: str, quest_name: str) -> list[HeuristicAction]:
        """Mark a quest as failed and return empty actions."""
        fqn = _fqn(quest_name)
        self._quest_status.setdefault(bot_id, {})[fqn] = "failed"
        logger.warning("[quest_sm] %s: failed quest '%s'", bot_id, fqn)
        return []

    # ------------------------------------------------------------------
    # Main assessment — called periodically by the heuristic engine
    # ------------------------------------------------------------------

    def assess(self, bot_id: str, signals: dict[str, Any]) -> list[HeuristicAction]:
        """Examine current quest state and produce next actions.

        ``signals`` contains current bot state: map, inventory, zeny,
        monster kills, NPC dialogue text, etc.

        Returns a list of HeuristicAction commands to execute.
        """
        actions: list[HeuristicAction] = []

        bot_states = dict(self._state.get(bot_id, {}))
        for fqn, state_name in bot_states.items():
            if self._quest_status.get(bot_id, {}).get(fqn) in ("completed", "failed"):
                continue

            qdef = self._data.get(fqn)
            if not qdef:
                continue

            state_def = qdef.get("states", {}).get(state_name)
            if not state_def:
                continue

            actions.extend(
                self._evaluate_state(bot_id, fqn, state_name, state_def, signals)
            )

        return actions

    # ------------------------------------------------------------------
    # Handle NPC dialogue response (for choose states)
    # ------------------------------------------------------------------

    def choose_answer(self, bot_id: str, quest_name: str, answer_text: str) -> list[HeuristicAction]:
        """Select a dialogue branch answer by text label.

        Must be called when a ``choose`` state's NPC is talking and the bot
        needs to pick an answer.
        """
        fqn = _fqn(quest_name)
        qdef = self._data.get(fqn)
        if not qdef:
            return []

        state_name = self._state.get(bot_id, {}).get(fqn)
        if not state_name:
            return self.start_quest(bot_id, fqn)

        state_def = qdef.get("states", {}).get(state_name)
        if not state_def or state_def.get("type") != "choose":
            logger.warning("[quest_sm] %s: state '%s' is not a choose state", bot_id, state_name)
            return []

        answers = state_def.get("answers", {})
        answer_def = answers.get(answer_text)
        if not answer_def:
            # Case-insensitive fallback
            for key, val in answers.items():
                if key.lower() == answer_text.lower():
                    answer_def = val
                    break
        if not answer_def:
            logger.warning("[quest_sm] %s: no answer '%s' in state '%s'",
                           bot_id, answer_text, state_name)
            return []

        next_state = answer_def.get("next_state")
        dialogue = answer_def.get("dialogue", ["c"])

        actions: list[HeuristicAction] = []

        # Emit talk command with chosen dialogue sequence
        npc = state_def.get("npc", {})
        if npc:
            talk_cmd = _build_talk_cmd(npc, dialogue)
            actions.append(HeuristicAction(
                kind="command",
                command=talk_cmd,
                confidence=0.95,
                reason=f"[quest_sm] {fqn}: chose '{answer_text}' → {next_state}",
                domain="quest",
                metadata={"quest": fqn, "state": state_name, "answer": answer_text},
            ))

        # Schedule a follow-up advance on next assess()
        # We store the pending transition so assess() can pick it up
        transitions = state_def.get("transitions", {})
        next_on_complete = transitions.get("next_state_on_complete", "")
        self._pending_choices.setdefault(bot_id, {})[fqn] = {
            "next_state_correct": next_state,
            "next_state_on_complete": next_on_complete,
            "answered": True,
        }

        return actions

    # ------------------------------------------------------------------
    # Reset / cleanup
    # ------------------------------------------------------------------

    def reset_quest(self, bot_id: str, quest_name: str) -> None:
        """Reset a quest to unstarted state."""
        fqn = _fqn(quest_name)
        self._state.get(bot_id, {}).pop(fqn, None)
        self._quest_status.get(bot_id, {}).pop(fqn, None)
        self._pending_choices.get(bot_id, {}).pop(fqn, None)

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove all quest state for a bot."""
        self._state.pop(bot_id, None)
        self._quest_status.pop(bot_id, None)
        self._pending_choices.pop(bot_id, None)

    # ================================================================
    # Internal helpers
    # ================================================================

    def _evaluate_state(
        self,
        bot_id: str,
        fqn: str,
        state_name: str,
        state_def: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[HeuristicAction]:
        """Decide what actions to emit based on state type + signals."""
        stype = state_def.get("type", "talk")

        # Check for pending choice transition (from choose_answer)
        pending = self._pending_choices.get(bot_id, {}).pop(fqn, None)
        if pending and pending.get("answered"):
            next_state = pending.get("next_state_correct", "")
            if not next_state and pending.get("next_state_on_complete"):
                next_state = pending["next_state_on_complete"]
            if next_state:
                return self.advance(bot_id, fqn, next_state)
            return []

        if stype == "collect":
            return self._evaluate_collect(bot_id, fqn, state_name, state_def, signals)
        elif stype == "pay":
            return self._evaluate_pay(bot_id, fqn, state_name, state_def, signals)
        elif stype == "choose":
            # The bot is waiting for the LLM or heuristic to pick an answer
            return self._emit_actions_for_state(bot_id, fqn, state_name, state_def)
        elif stype == "complete":
            return self._emit_actions_for_state(bot_id, fqn, state_name, state_def)
        else:
            # talk / default — emit dialogue commands, then auto-advance
            actions = self._emit_actions_for_state(bot_id, fqn, state_name, state_def)
            transitions = state_def.get("transitions", {})
            next_state = transitions.get("next_state", "")
            if next_state:
                logger.debug("[quest_sm] %s: talk auto-advance %s → %s",
                             bot_id, state_name, next_state)
                advance_actions = self.advance(bot_id, fqn, next_state)
                actions.extend(advance_actions)
            return actions

    def _evaluate_collect(
        self,
        bot_id: str,
        fqn: str,
        state_name: str,
        state_def: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[HeuristicAction]:
        """Check inventory for required items; if met, advance."""
        item_name = state_def.get("item", "")
        required = state_def.get("quantity", 1)

        inventory = signals.get("inventory", []) or []
        have = sum(
            entry.get("amount", 1) or 1
            for entry in inventory
            if (entry.get("name", "") or "").lower() == item_name.lower()
        )

        if have >= required:
            # Items collected — advance to next state
            npc = state_def.get("npc_for_turnin", state_def.get("npc", {}))
            transitions = state_def.get("transitions", {})
            next_state = transitions.get("next_state", "")
            if next_state:
                logger.info("[quest_sm] %s: collected %d/%d %s → %s",
                            bot_id, have, required, item_name, next_state)
                return self.advance(bot_id, fqn, next_state)
            return []

        # Need more items
        drop_map = state_def.get("drop_map", "")
        drop_monster = state_def.get("drop_monster", "")
        buy_npc = state_def.get("buy_from_npc", {})

        actions: list[HeuristicAction] = []

        if buy_npc:
            buy_price = state_def.get("buy_price", 100)
            zeny = int(signals.get("zeny", 0) or 0)
            needed = required - have
            cost = needed * buy_price
            if zeny >= cost:
                cmd = _build_talk_cmd(buy_npc, ["h", "c"])
                actions.append(HeuristicAction(
                    kind="command",
                    command=cmd,
                    confidence=0.90,
                    reason=f"[quest_sm] {fqn}: buy {needed}x {item_name} from shop",
                    domain="quest",
                    metadata={"quest": fqn, "state": state_name},
                ))
            else:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"quest_collect {item_name} {required - have}",
                    confidence=0.85,
                    reason=f"[quest_sm] {fqn}: need {required - have} more {item_name} — farm zeny first",
                    domain="quest",
                    metadata={"quest": fqn, "state": state_name},
                ))
        elif drop_monster:
            actions.append(HeuristicAction(
                kind="command",
                command=f"quest_collect {item_name} {required - have}",
                confidence=0.85,
                reason=f"[quest_sm] {fqn}: farm {drop_monster} for {item_name} ({have}/{required})",
                domain="quest",
                metadata={
                    "quest": fqn, "state": state_name,
                    "drop_monster": drop_monster,
                    "drop_map": drop_map,
                    "drop_rate": state_def.get("drop_rate", 0),
                },
            ))
        else:
            actions.append(HeuristicAction(
                kind="log",
                command="",
                confidence=1.0,
                reason=f"[quest_sm] {fqn}: need {required - have} more {item_name} (no source info)",
                domain="quest",
            ))

        return actions

    def _evaluate_pay(
        self,
        bot_id: str,
        fqn: str,
        state_name: str,
        state_def: dict[str, Any],
        signals: dict[str, Any],
    ) -> list[HeuristicAction]:
        """Check zeny; if enough, pay and advance."""
        amount = state_def.get("amount", 0)
        zeny = int(signals.get("zeny", 0) or 0)

        if zeny >= amount:
            transitions = state_def.get("transitions", {})
            next_state = transitions.get("next_state", "")
            if next_state:
                logger.info("[quest_sm] %s: paying %dz for %s → %s",
                            bot_id, amount, fqn, next_state)
                return self.advance(bot_id, fqn, next_state)

        # Not enough zeny
        actions: list[HeuristicAction] = []
        actions.append(HeuristicAction(
            kind="log",
            command="",
            confidence=1.0,
            reason=f"[quest_sm] {fqn}: need {amount}z (have {zeny}z) — farm before returning",
            domain="quest",
        ))
        return actions

    def _emit_actions_for_state(
        self,
        bot_id: str,
        fqn: str,
        state_name: str,
        state_def: dict[str, Any],
    ) -> list[HeuristicAction]:
        """Generate actions for entering a state (talk, complete, choose prompt)."""
        actions: list[HeuristicAction] = []

        stype = state_def.get("type", "talk")
        npc = state_def.get("npc", {})
        dialogue = state_def.get("dialogue", ["h", "c"])
        on_enter = state_def.get("on_enter", [])

        # Emit on_enter log/status commands
        for cmd_text in on_enter:
            actions.append(HeuristicAction(
                kind="command",
                command=cmd_text,
                confidence=1.0,
                reason=f"[quest_sm] {fqn}: {cmd_text}",
                domain="quest",
                metadata={"quest": fqn, "state": state_name},
            ))

        # For choose states, emit the NPC response text and available answers
        if stype == "choose":
            npc_response = state_def.get("npc_response", "")
            answers = state_def.get("answers", {})

            if npc:
                talk_cmd = _build_talk_cmd(npc, [state_def.get("dialogue_prompt", "h")])
                actions.append(HeuristicAction(
                    kind="command",
                    command=talk_cmd,
                    confidence=0.95,
                    reason=f"[quest_sm] {fqn}: approach NPC at {npc.get('map', '?')}",
                    domain="quest",
                    metadata={"quest": fqn, "state": state_name},
                ))

            # Attach answer options as metadata so the LLM/heuristic can choose
            actions.append(HeuristicAction(
                kind="command",
                command="quest_status",
                confidence=1.0,
                reason=f"[quest_sm] {fqn} question: '{npc_response}' — "
                       f"choose from: {', '.join(answers.keys())}",
                domain="quest",
                metadata={
                    "quest": fqn,
                    "state": state_name,
                    "state_type": "choose",
                    "npc_response": npc_response,
                    "answers": list(answers.keys()),
                    "answer_map": {k: v.get("next_state", "?") for k, v in answers.items()},
                },
            ))

        elif stype == "complete":
            # Final dialogue sequence — job change completes
            if npc:
                talk_cmd = _build_talk_cmd(npc, dialogue)
                actions.append(HeuristicAction(
                    kind="command",
                    command=talk_cmd,
                    confidence=0.95,
                    reason=f"[quest_sm] {fqn}: complete — job change dialogue",
                    domain="quest",
                    metadata={"quest": fqn, "state": state_name},
                ))

        else:
            # talk / default
            if npc:
                talk_cmd = _build_talk_cmd(npc, dialogue)
                actions.append(HeuristicAction(
                    kind="command",
                    command=talk_cmd,
                    confidence=0.95,
                    reason=f"[quest_sm] {fqn}: talk to NPC at {npc.get('map', '?')}",
                    domain="quest",
                    metadata={"quest": fqn, "state": state_name},
                ))

        return actions

    def __repr__(self) -> str:
        return f"<QuestStateMachine {len(self._data)} quests loaded>"


# ====================================================================
# Module-level helpers
# ====================================================================


def _build_talk_cmd(npc: dict[str, Any], dialogue: list[str]) -> str:
    """Build an OpenKore 'talknpc map x y seq1 seq2 ...' command."""
    npc_map = npc.get("map", "")
    x = npc.get("x", 0)
    y = npc.get("y", 0)
    parts = ["talknpc", str(npc_map), str(x), str(y)]
    parts.extend(dialogue)
    return " ".join(parts)


# ====================================================================
# Singleton factory
# ====================================================================

_INSTANCE: QuestStateMachine | None = None


def get_quest_sm() -> QuestStateMachine:
    """Return a shared QuestStateMachine singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = QuestStateMachine()
    return _INSTANCE
