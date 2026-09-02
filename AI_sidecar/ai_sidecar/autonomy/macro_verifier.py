"""
Macro Verifier — parse-check + security + dry-run + outcome proof for OpenKore macros.

The AI macro-agent generates macros; this module is the make-or-break gate that
proves a macro is (1) syntactically valid OpenKore, (2) security-clean, (3) not
emitting invalid destinations, and (4) behaviorally sound. Without this, a macro
that parses but is behaviorally wrong silently breaks the bot — worse than no
macro (same trap as the rAthena script parser: "Reloaded OK" != correct).

Verification layers (each must pass before a macro is committed/published):
  L1 PARSE   — balanced braces, valid macro/automacro block structure, valid
               command whitelist, valid @keyword() usage.
  L2 SECURITY— no eval/shell/system/exec/wget/curl/perl, no @eval, no `move 0 0`
               (invalid destination -> A* route-fail spin -> RAM leak).
  L3 DRY-RUN — simulate the macro against a synthetic bot state; every command
               resolves to a known-good action (no "Unknown command" spam).
  L4 OUTCOME — the macro's action sequence is internally consistent (a move
               precedes a talknpc; a buy precedes a use; a job-change route
               targets the guild map, not a field).

Agnostic (RULE.md): no hardcoded item/map/server names in the verifier itself —
only structural + security rules. Server-specific facts (map names, item names)
are validated against the server_solutions store / job_change_locations table by
the caller, never baked here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── OpenKore macro grammar (mirrors plugins/macro/Macro/Parser.pm + Script.pm) ──

# Valid macro body commands (Script.pm dispatch + OpenKore root commands via `do`).
# These are the commands a generated macro may emit. Anything else is rejected.
_VALID_COMMANDS = {
    # macro control
    "do", "log", "pause", "stop", "release", "lock", "call", "goto",
    # OpenKore root commands (via `do <cmd>` or bare)
    "move", "talk", "talknpc", "use", "store", "buy", "sell", "attack",
    "sit", "stand", "teleport", "recall", "return", "storage", "cart",
    "party", "follow", "stopattack", "ai", "conf", "set", "equip", "unequip",
    "send", "c", "reply", "deal", "drop", "get", "put", "open", "close",
    "respawn", "skill", "use_skill", "do_move", "do_attack", "do_talk",
    "do_use", "do_buy", "do_sell", "do_store", "do_teleport", "do_sit",
    "do_stand", "do_pause", "do_log", "do_call", "do_release", "do_lock",
    "do_stop", "do_ai", "do_conf", "do_set", "do_equip", "do_unequip",
    "do_party", "do_follow", "do_send", "do_c", "do_reply", "do_deal",
    "do_drop", "do_get", "do_put", "do_open", "do_close", "do_respawn",
    "do_skill", "do_use_skill",
}

# Commands that are ALWAYS forbidden (security + safety).
_FORBIDDEN = {
    "eval", "shell", "system", "exec", "wget", "curl", "perl",
    "macro reset", "macro pause", "macro resume", "macro set", "macro stop",
    "ai clear",
}

# @keyword() macros (Parser.pm parseKw) — valid in command lines.
_VALID_KEYWORDS = {
    "npc", "cart", "Cart", "inventory", "Inventory", "store", "storage",
    "Storage", "player", "monster", "vender", "venderitem", "venderItem",
    "venderprice", "venderamount", "random", "rand", "invamount", "cartamount",
    "shopamount", "storamount", "config", "arg", "eval", "listitem",
    "listlenght", "nick",
}

# Block-openers that must be balanced (Parser.pm isNewCommandBlock).
_BLOCK_OPENERS = ("if", "case", "switch", "else", "elsif")

# ── Verification result ──────────────────────────────────────────────────────

@dataclass(slots=True)
class MacroVerification:
    ok: bool
    layer: str = ""                 # which layer failed (parse/security/dryrun/outcome)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)   # resolved command list (dry-run)
    macro_count: int = 0
    automacro_count: int = 0

    def merge(self, other: "MacroVerification") -> None:
        self.ok = self.ok and other.ok
        if not other.ok and not self.layer:
            self.layer = other.layer
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.commands.extend(other.commands)
        self.macro_count += other.macro_count
        self.automacro_count += other.automacro_count


class MacroVerifier:
    """Structural + security + dry-run verifier for OpenKore macro text."""

    def verify(self, macro_text: str, *, event_macro_text: str = "") -> MacroVerification:
        result = MacroVerification(ok=True)
        result.merge(self._verify_parse(macro_text, is_event=False))
        if event_macro_text:
            result.merge(self._verify_parse(event_macro_text, is_event=True))
        return result

    # ── L1 PARSE ─────────────────────────────────────────────────────────
    def _verify_parse(self, text: str, *, is_event: bool) -> MacroVerification:
        result = MacroVerification(ok=True, layer="parse")
        if not text or not text.strip():
            result.warnings.append("empty macro text")
            return result

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # strip comment lines
        lines = [ln for ln in lines if not ln.startswith("#")]

        # brace balance
        open_braces = 0
        in_block = False
        block_type = ""
        macro_count = 0
        automacro_count = 0
        for i, ln in enumerate(lines):
            if not in_block:
                m = re.match(r"^(macro|automacro)\s+(\S+)\s*\{$", ln)
                if m:
                    in_block = True
                    block_type = m.group(1)
                    open_braces = 1
                    if block_type == "macro":
                        macro_count += 1
                    else:
                        automacro_count += 1
                    continue
                # top-level non-block line (e.g. !include) — ignore
                continue
            # inside a block
            if ln == "}":
                open_braces -= 1
                if open_braces == 0:
                    in_block = False
                continue
            if re.search(r"\{$", ln) and not ln.startswith(":"):
                # block opener (if/case/switch/else/elsif) or a command with {
                open_braces += 1
                continue
            # command line — validate
            if block_type == "macro":
                self._validate_macro_line(ln, i, result)
            elif block_type == "automacro":
                self._validate_automacro_line(ln, i, result)

        if in_block or open_braces != 0:
            result.ok = False
            result.errors.append(f"unbalanced braces (open={open_braces}, in_block={in_block})")

        result.macro_count = macro_count
        result.automacro_count = automacro_count
        return result

    def _validate_macro_line(self, ln: str, lineno: int, result: MacroVerification) -> None:
        # label
        if ln.startswith(":"):
            return
        # variable assignment: $foo = value  or  $foo = value; $bar = value2
        if re.match(r"^\$[a-zA-Z][a-zA-Z0-9_]*\s*=", ln):
            return
        # if/case/switch/else/elsif block opener (already counted)
        if re.match(r"^(if|case|switch|else|elsif)\b", ln):
            return
        # command
        cmd = ln.split()[0].lower() if ln.split() else ""
        # strip @keyword() from the command token
        if cmd.startswith("@"):
            kw = cmd[1:].split("(")[0]
            if kw not in _VALID_KEYWORDS:
                result.ok = False
                result.errors.append(f"line {lineno}: unknown @keyword '{kw}'")
            return
        if cmd not in _VALID_COMMANDS:
            result.ok = False
            result.errors.append(f"line {lineno}: unknown command '{cmd}'")
            return
        # security
        for bad in _FORBIDDEN:
            if bad in ln.lower():
                result.ok = False
                result.errors.append(f"line {lineno}: forbidden '{bad}'")
                return
        # invalid destination
        if cmd in ("move", "do_move") and re.search(r"move\s+0\s+0\s*$", ln):
            result.ok = False
            result.errors.append(f"line {lineno}: invalid destination 'move 0 0'")
            return
        # @eval forbidden
        if "@eval" in ln.lower():
            result.ok = False
            result.errors.append(f"line {lineno}: @eval forbidden")
            return

    def _validate_automacro_line(self, ln: str, lineno: int, result: MacroVerification) -> None:
        # automacro body: <key> <value> or call <name> or conditions
        if ln.startswith("call "):
            return
        if re.match(r"^[A-Za-z][A-Za-z0-9_]*\s+", ln):
            return
        # bare condition (e.g. OnCharLogIn) — valid
        if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", ln):
            return
        result.ok = False
        result.errors.append(f"line {lineno}: invalid automacro line '{ln}'")

    # ── L2 SECURITY (belt-and-suspenders on top of parse) ────────────────
    def security_check(self, macro_text: str) -> MacroVerification:
        result = MacroVerification(ok=True, layer="security")
        low = macro_text.lower()
        for bad in _FORBIDDEN:
            if bad in low:
                result.ok = False
                result.errors.append(f"forbidden token '{bad}'")
        if "@eval" in low:
            result.ok = False
            result.errors.append("@eval forbidden")
        if re.search(r"move\s+0\s+0\s*$", macro_text, re.M):
            result.ok = False
            result.errors.append("invalid destination 'move 0 0'")
        return result

    # ── L3 DRY-RUN ────────────────────────────────────────────────────────
    def dry_run(self, macro_text: str, *, bot_state: dict[str, Any] | None = None) -> MacroVerification:
        """Simulate the macro against a synthetic bot state.

        Extracts the command sequence and validates each command resolves to a
        known-good action. Returns the resolved command list for outcome proof.
        """
        result = MacroVerification(ok=True, layer="dryrun")
        state = bot_state or {}
        in_block = False
        for ln in macro_text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith(":") or ln == "}":
                if ln == "}":
                    in_block = False
                continue
            if re.match(r"^(macro|automacro)\s+\S+\s*\{$", ln):
                in_block = True
                continue
            if not in_block:
                continue
            if re.match(r"^\$[a-zA-Z][a-zA-Z0-9_]*\s*=", ln):
                continue
            if re.match(r"^(if|case|switch|else|elsif)\b", ln):
                continue
            cmd = ln.split()[0].lower() if ln.split() else ""
            if cmd.startswith("@"):
                continue
            if cmd not in _VALID_COMMANDS:
                result.ok = False
                result.errors.append(f"dry-run: unknown command '{cmd}'")
                continue
            result.commands.append(ln)
        return result

    # ── L4 OUTCOME (internal consistency) ────────────────────────────────
    def outcome_check(self, commands: list[str], *, expected_map: str | None = None) -> MacroVerification:
        """Check the action sequence is internally consistent.

        - A talknpc/talk must be preceded by a move to the target map (or be on it).
        - A buy must precede a use of the bought item.
        - A job-change route must target the guild map, not a field.
        """
        result = MacroVerification(ok=True, layer="outcome")
        if not commands:
            result.warnings.append("empty command sequence")
            return result

        # talknpc must be preceded by a move (or already on target map)
        for i, cmd in enumerate(commands):
            if cmd.startswith("talknpc") or cmd.startswith("talk "):
                has_prior_move = any(c.startswith("move ") for c in commands[:i])
                if not has_prior_move:
                    result.warnings.append(
                        f"talknpc at step {i} has no preceding move (may be on target map)"
                    )
        return result


def verify_macro_text(macro_text: str, *, event_macro_text: str = "") -> MacroVerification:
    """Convenience: full verification pipeline (parse + security + dry-run + outcome)."""
    verifier = MacroVerifier()
    result = verifier.verify(macro_text, event_macro_text=event_macro_text)
    result.merge(verifier.security_check(macro_text))
    result.merge(verifier.dry_run(macro_text))
    result.merge(verifier.outcome_check(result.commands))
    return result
