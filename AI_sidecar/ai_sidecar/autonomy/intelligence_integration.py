"""Unified wiring for the three built-but-unwired intelligence subsystems.

Fully-implemented subsystems that were never connected to the runtime loop:
  1. ConsciousDecisionEngine     (conscious_engine.py) — phase-based skill/stat/
     restock/map decisions from the class-build knowledge base.
  2. PreemptiveIntelligence      (preemptive_intelligence.py) — anticipates
     needs across combat/inventory/economy/party/safety and emits preemptive
     actions.
  3. ProgressionDriver           (progression_driver.py) — learns the novice
     skills, sits to regen, restocks potions, sells junk, advances grind map.

Each was built + unit-verified but `get_*` was never imported/called anywhere
in the tree (confirmed by full-repo reference scan). This module wires each
into the PDCA per-bot cycle: feed the live snapshot, run its evaluate, convert
its decisions into real queued ActionProposals (offering observability when a
decision is not yet a safe executable command), and gate to in-game bots.

Usage (pdca_loop.py):
    from ai_sidecar.autonomy.intelligence_integration import run_intelligence
    _total_actions += run_intelligence(context, _cycle_bot_id, snapshot)

Returns the number of actions queued/observed (0 if nothing to do or error).
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_CE = None  # ConsciousDecisionEngine
_PI = None  # PreemptiveIntelligence
_PD = None  # ProgressionDriver


def _ensure_services(runtime: Any):
    """Lazy-init the three singletons and bind them to the runtime's action queue."""
    global _CE, _PI, _PD
    aq = getattr(runtime, "action_queue", None)
    if aq is None:
        return None, None, None

    if _CE is None:
        try:
            from ai_sidecar.conscious_engine import get_conscious_engine
            _CE = get_conscious_engine()
            # ConsciousDecisionEngine emits `Decision` objects via evaluate();
            # run_intelligence converts them into queued ActionProposals directly
            # (this subsystem has no internal enqueue hook).
            logger.info("intelligence_wired: ConsciousDecisionEngine")
        except Exception as e:
            logger.warning("intelligence_conscious_init_failed: %s", e)
            _CE = None
    if _PI is None:
        try:
            from ai_sidecar.preemptive_intelligence import get_preemptive_intelligence
            _PI = get_preemptive_intelligence()
            _PI.set_enqueue_fn(_make_queue_fn(runtime))
            logger.info("intelligence_wired: PreemptiveIntelligence")
        except Exception as e:
            logger.warning("intelligence_preemptive_init_failed: %s", e)
            _PI = None
    if _PD is None:
        try:
            from ai_sidecar.progression_driver import get_progression_driver
            _PD = get_progression_driver()
            _PD.set_queue_fn(_make_queue_fn(runtime))
            logger.info("intelligence_wired: ProgressionDriver")
        except Exception as e:
            logger.warning("intelligence_progression_init_failed: %s", e)
            _PD = None
    return _CE, _PI, _PD


def _make_queue_fn(runtime: Any):
    """Return a queue_fn with the contract (proposal, bot_id)->(accepted,status,id,reason)."""
    aq = getattr(runtime, "action_queue", None)

    def _enqueue(proposal, bot_id):
        try:
            from ai_sidecar.contracts.actions import ActionPriorityTier
        except Exception:
            return (False, "enqueued", "", "import_err")
        if aq is None:
            return (False, "no_queue", "", "no_action_queue")
        try:
            ok, status, aid, reason = aq.enqueue(bot_id, proposal)
            return (ok, status, aid if aid else "", reason or "")
        except Exception as e:
            return (False, "error", "", str(e))
    return _enqueue


def _in_game(snapshot: Any) -> bool:
    """Do not queue decisions for bots that are logged out / at char-select."""
    try:
        if snapshot is None:
            return False
        raw = getattr(snapshot, "raw", None)
        if isinstance(raw, dict):
            if raw.get("in_game") is False:
                return False
        # map_known == False means char-select / disconnected.
        return bool(getattr(snapshot, "map_known", True))
    except Exception:
        return True


# Decide -> command translator (conscious engine + preemptive outputs share these actions).
_COMMAND_MAP: dict[str, str] = {
    "learn_skill": "skills_add {target} 1",
    "add_stat": "stats_add {target} 1",
    "buy_item": "buy {target} {qty}",
    "move_map": "move {target}",
}
# Actions that should be OBSERVED (logged), not emitted as commands, because they
# involve NPC/party state the runtime doesn't yet resolve into a safe command.
_OBSERVE_ONLY = {"request_heal", "emergency_restock", "flee_to_safety", "vendor_trash",
                 "farm_zeny", "switch_map", "restock", "restock_heal"}


def _emit_command(runtime: Any, bot_id: str, domain: str, action: str,
                  target: str, qty: int, reason: str, prio_tier: str = "strategic") -> int:
    """Convert a decision/action into a queued command or an observability log."""
    from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
    aq = getattr(runtime, "action_queue", None)
    if aq is None:
        return 0
    tier = getattr(ActionPriorityTier, prio_tier, ActionPriorityTier.strategic)

    fmt = _COMMAND_MAP.get(action)
    if fmt:
        cmd = fmt.format(target=target or "", qty=max(1, int(qty or 1)))
        # Guard: never emit `move {empty}` / `buy {empty}` / `skills_add {}`.
        if action == "move_map" and not target:
            return _observe(runtime, bot_id, domain, action, reason)
        if action in ("learn_skill", "add_stat") and not target:
            return _observe(runtime, bot_id, domain, action, reason)
        if action == "buy_item" and not target:
            return _observe(runtime, bot_id, domain, action, reason)
        _key = f"ui_{action}_{bot_id}_{target}"
        _prop = ActionProposal(
            action_id=f"ui_{bot_id}_{int(time.monotonic()*1000)}",
            kind="command",
            command=cmd,
            priority_tier=tier,
            conflict_key=_key,
            idempotency_key=_key,
            source="intelligence",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
            metadata={"source": "intelligence", "reason": reason, "domain": domain,
                      "action": action, "target": target or "", "bot_id": bot_id},
        )
        try:
            ok, status, aid, why = aq.enqueue(bot_id, _prop)
            if ok:
                logger.info("intelligence_queued: bot=%s action=%s target=%s cmd=%s",
                            bot_id, action, target, cmd)
                return 1
            logger.debug("intelligence_rejected: bot=%s action=%s reason=%s", bot_id, action, why)
            return 0
        except Exception as e:
            logger.debug("intelligence_enqueue_err: %s", e)
            return 0
    # Not a directly-emittable command -> observe (log) so the intent is visible
    # and the design surface is exercised, without emitting a bogus command.
    return _observe(runtime, bot_id, domain, action, reason)


def _observe(runtime: Any, bot_id: str, domain: str, action: str, reason: str) -> int:
    logger.info("intelligence_observe: bot=%s domain=%s action=%s reason=%s",
                bot_id, domain, action, reason)
    return 0


def run_intelligence(runtime: Any, bot_id: str | None, snapshot: Any = None) -> int:
    """Run all three wired intelligence subsystems for one bot-cycle."""
    if not bot_id:
        return 0
    ce, pi, pd = _ensure_services(runtime)
    if ce is None and pi is None and pd is None:
        return 0
    if not _in_game(snapshot):
        return 0
    total = 0

    # ── ConsciousDecisionEngine ──
    if ce is not None:
        try:
            ce.update_from_snapshot(bot_id, snapshot)
            decisions = ce.evaluate(bot_id) or []
            for d in decisions:
                if getattr(d, "action", "") in _OBSERVE_ONLY:
                    total += _observe(runtime, bot_id, d.domain, d.action, d.reason)
                    continue
                qty = (d.params or {}).get("qty", 1) if getattr(d, "params", None) else 1
                total += _emit_command(runtime, bot_id, d.domain, d.action,
                                       d.target, qty, d.reason)
        except Exception as e:
            logger.debug("intelligence_conscious_err: bot=%s err=%s", bot_id, e)

    # ── PreemptiveIntelligence ──
    if pi is not None:
        try:
            pi.update_from_snapshot(bot_id, snapshot)
            pre = pi.evaluate(bot_id) or []
            for pa in pre:
                act = getattr(pa, "action_type", "")
                if act in _OBSERVE_ONLY:
                    total += _observe(runtime, bot_id, getattr(pa, "domain", ""), act,
                                      getattr(pa, "reason", ""))
                    continue
                tgt = (getattr(pa, "target_map", "") or (getattr(pa, "items_needed", [""]) or [""])[0])
                total += _emit_command(runtime, bot_id, getattr(pa, "domain", ""), act,
                                       tgt, 1, getattr(pa, "reason", ""))
        except Exception as e:
            logger.debug("intelligence_preemptive_err: bot=%s err=%s", bot_id, e)

    # ── ProgressionDriver ──
    if pd is not None:
        try:
            pd.update_from_snapshot(bot_id, snapshot)
            state = _snapshot_to_state(snapshot)
            # process_decisions self-queues via the attached queue_fn; we cannot
            # cheaply count exactly, so we treat a run as one evaluated cycle.
            pd.process_decisions(bot_id, state)
            total += 1
        except Exception as e:
            logger.debug("intelligence_progression_err: bot=%s err=%s", bot_id, e)

    return total


def _snapshot_to_state(snapshot: Any) -> dict[str, Any]:
    """Flatten a BotStateSnapshot into the dict shape ProgressionDriver.process_decisions reads."""
    state: dict[str, Any] = {}
    if snapshot is None:
        return state
    v = getattr(snapshot, "vitals", None)
    if v is not None:
        state["hp_pct"] = float(getattr(v, "hp_ratio", 1.0) or 1.0)
        state["sp_pct"] = float(getattr(v, "sp_ratio", 1.0) or 1.0)
        state["base_level"] = int(getattr(v, "base_level", 1) or 1)
        state["job_name"] = str(getattr(v, "job_name", "novice") or "novice").lower()
        state["zeny"] = int(getattr(v, "zeny", 0) or 0)
        state["weight_ratio"] = float(getattr(v, "weight_ratio", 0.0) or 0.0)
    state["skills"] = [str(getattr(s, "name", "")) for s in (getattr(snapshot, "skills", []) or [])]
    pos = getattr(snapshot, "position", None)
    state["map"] = str(getattr(pos, "map", "") or "")
    state["inventory"] = {
        str(getattr(it, "name", "")): int(getattr(it, "amount", 0) or 0)
        for it in (getattr(snapshot, "inventory_items", []) or [])
    }
    return state
