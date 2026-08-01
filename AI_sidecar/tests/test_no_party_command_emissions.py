"""Regression test: NO party command may be emitted by any sidecar layer.

History: party request/create/leave/share emissions from THREE layers
(legacy autonomy/domains/*, heuristic_service inline "direct party check"
blocks, and swarm tactics) produced frozen party-request spam in bot logs.
The bridge's Commands::run/pre gate could only LOG blocks — Commands::run
dispatches the handler with the ORIGINAL switch, so the hook cannot stop
execution (proven by a minimal Perl harness). The structural fix: the
fleet coordinator is the party actor; every sidecar layer emits party
state as kind="log" observability intents that the bridge never executes.

This test walks the ENTIRE sidecar source tree and asserts no
kind="command" emission whose command starts with `party` can exist.
"""
from __future__ import annotations

import os
import re

from ai_sidecar.autonomy.heuristic_service import HeuristicService


def _sidecar_python_files() -> list[str]:
    root = os.path.join(os.path.dirname(__file__), "..", "ai_sidecar")
    out: list[str] = []
    for base, _dirs, files in os.walk(root):
        if "__pycache__" in base:
            continue
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(base, f))
    return out


def test_no_party_command_emissions_sidecar_wide() -> None:
    # Matches kind="command", command="party ..." (single or double quotes).
    pattern = re.compile(
        r'kind\s*=\s*"command"\s*,\s*(?:command\s*=\s*)?["\']party\b'
    )
    offenders: list[str] = []
    for path in _sidecar_python_files():
        src = open(path, encoding="utf-8", errors="replace").read()
        for m in pattern.finditer(src):
            ln = src[: m.start()].count("\n") + 1
            offenders.append(f"{os.path.relpath(path)}:{ln}")
    assert not offenders, (
        "party command emissions are forbidden (fleet coordinator is the "
        "party actor):\n" + "\n".join(offenders)
    )


def test_heuristic_service_direct_party_check_is_observability_only() -> None:
    """The inline 'direct party check' block must never emit executable
    party commands — only log intents with party_action metadata."""
    src = open(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "ai_sidecar/autonomy/heuristic_service.py",
        ),
        encoding="utf-8",
        errors="replace",
    ).read()
    # All party-related HeuristicAction emissions in the file must be log-kind.
    for m in re.finditer(r'command\s*=\s*["\']party\b[^"\']*["\']', src):
        # Find the enclosing kind= literal before this command.
        prefix = src[: m.start()]
        last_kind = list(re.finditer(r'kind\s*=\s*"([a-z_]+)"', prefix))
        assert last_kind, f"no kind= found before {m.group(0)!r}"
        kind = last_kind[-1].group(1)
        assert kind == "log", (
            f"party emission must be log-kind, got {kind!r} for {m.group(0)!r}"
        )


def test_no_unpaired_ai_manual_flips_sidecar_wide() -> None:
    """ai manual DISABLES bot AI; config-audit owns AI mode. Unpaired flips
    freeze stuck bots permanently. Allowed: the paired cold-start/unstuck
    mechanism (ai manual -> move -> ai auto re-enable in the same flow) and
    the dev-only reflex emit_test utility."""
    pattern = re.compile(r'command\s*=\s*["\']ai manual["\']')
    # Semantic anchors: reasons of the legitimate paired cold-start/unstuck
    # flips and their required paired re-enables (line numbers are fragile).
    PAIRED_REASONS = (
        "manual mode to unstick",
        "Cold start - disable AI for portal walk",
    )
    PAIRED_REENABLE_REASONS = (
        "Re-enable auto after unstuck move",
        "Cold start step 1 - enable AI",
    )
    offenders: list[str] = []
    for path in _sidecar_python_files():
        src = open(path, encoding="utf-8", errors="replace").read()
        for m in pattern.finditer(src):
            ln = src[: m.start()].count("\n") + 1
            rel = os.path.relpath(path)
            if rel.endswith("reflex/reflex_pipeline.py") and "emit_test" in src[
                max(0, m.start() - 3000) : m.start()
            ]:
                continue  # dev-only pipeline self-test utility
            # Paired exemption: a matching reason must follow within 8 lines
            # AND a paired ai auto with an allowed re-enable reason must
            # appear anywhere after this site (reasons are unique strings).
            after = src[m.end():]
            window8 = src[m.end() : m.end() + 800]
            if any(r in window8 for r in PAIRED_REASONS):
                paired_auto = re.search(r'command\s*=\s*["\']ai auto["\']', after)
                paired_reason_ok = any(r in after for r in PAIRED_REENABLE_REASONS)
                if paired_auto and paired_reason_ok:
                    continue
                offenders.append(f"{rel}:{ln} (paired exemption violated: no re-enable)")
                continue
            # Must not be a kind="command" emission (log intents are fine).
            prefix = src[: m.start()]
            last_kind = list(re.finditer(r'kind\s*=\s*"([a-z_]+)"', prefix))
            if last_kind and last_kind[-1].group(1) != "command":
                continue
            offenders.append(f"{rel}:{ln}")
    assert not offenders, (
        "unpaired ai manual command emissions are forbidden (freezes bots):\n"
        + "\n".join(offenders)
    )


def test_domain_registry_still_loads_after_refactor() -> None:
    """The observe-only legacy domains must still load + assess cleanly."""
    from ai_sidecar.autonomy.domains import DomainRegistry

    reg = DomainRegistry()
    reg.load_all()
    assert len(reg._domains) > 0


def test_heuristic_service_assess_emits_party_log_intents() -> None:
    """When party formation is needed, the assessment carries observability
    intents (kind=log) and never a party command."""
    hs = HeuristicService()
    signals = {
        "hp_ratio": 0.9,
        "level": 45,
        "in_party": False,
        "party_members": [],
        "all_bots": ["kicapmasin4", "kicapmasin5"],
    }
    result = hs.assess(signals, bot_id_override="kicapmasin4")
    for action in result.actions:
        if action.domain == "social":
            assert action.kind == "log", (
                f"social emission must be log-kind, got {action.kind} "
                f"command={action.command!r}"
            )
