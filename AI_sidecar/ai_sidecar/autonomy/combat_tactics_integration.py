"""Wire the dormant CombatTactics class into the per-bot combat cycle.

`CombatTactics` (combat_tactics.py) holds the Pro-RO guntiter skill-combo
knowledge base (per-class skill combos, kiting rules, weapon-for-size, element
advice) but was only CONSTRUCTED onto the runtime (pdca_loop:2963). None of its
methods were ever called (full-repo reference scan: get_combo/should_kite/
suggest_cards_for_monster/... have 0 callers) -> a fully-implemented
god-tier-combat layer left dormant.

This module drives it each bot-cycle:
  - Resolve the bot's job name + monster element + combat pressure from the
    live snapshot.
  - `CombatTactics.get_combo(...)` -> the best skill sequence for this class vs
    the target's element; emit each skill as an `ss <skill>` command when the
    bot can execute it (SP/HP gate via CombatTactics.can_execute_skill) and is
    actually in combat (aggro>0).
  - `CombatTactics.should_kite(...)` -> emit a combat reposition intent when the
    class/HP rule says kite.
  - Non-executable intents are observed (logged) rather than emitted as bogus
    commands; the bot must be in-game to receive anything.

Usage (pdca_loop.py, per-bot cycle):
    from ai_sidecar.autonomy.combat_tactics_integration import run_combat_tactics
    _total_actions += run_combat_tactics(context, _cycle_bot_id, snapshot)
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_CT = None  # CombatTactics singleton
_LAST_CAST: dict[str, float] = {}  # bot_id -> last skill-cast epoch (per-bot throttle)


def _get_ct() -> Any:
    global _CT
    if _CT is None:
        try:
            from ai_sidecar.combat_tactics import CombatTactics
            _CT = CombatTactics()
            logger.info("combat_tactics_integration_initialized")
        except Exception as e:
            logger.warning("combat_tactics_integration_init_failed: %s", e)
            _CT = None
    return _CT


def _in_game(snapshot: Any) -> bool:
    try:
        if snapshot is None:
            return False
        raw = getattr(snapshot, "raw", None)
        if isinstance(raw, dict) and raw.get("in_game") is False:
            return False
        return bool(getattr(snapshot, "map_known", True))
    except Exception:
        return True


def _snap_vitals(snapshot: Any) -> tuple[float, int, int, int, str]:
    """(hp_pct, current_sp, current_hp, max_hp, job_name) with safe defaults."""
    hp_pct = 1.0
    sp = 0
    hp = 100
    max_hp = 100
    job = "novice"
    v = getattr(snapshot, "vitals", None)
    if v is not None:
        hp_pct = float(getattr(v, "hp_ratio", 1.0) or 1.0)
        sp = int(getattr(v, "sp", 0) or 0)
        hp = int(getattr(v, "hp", 100) or 100)
        max_hp = int(getattr(v, "hp_max", 100) or 100)
        job = str(getattr(v, "job_name", "novice") or "novice").lower()
    return hp_pct, sp, hp, max_hp, job


def _target_element(snapshot: Any) -> str:
    """Best-effort current target monster element from the snapshot's combat state."""
    try:
        combat = getattr(snapshot, "combat", None)
        if combat is not None:
            elem = getattr(combat, "target_element", None) or getattr(combat, "monster_element", "")
            if elem:
                return str(elem).lower()
        # fallback: any actor marked as target
        target = getattr(snapshot, "target", None)
        if target is not None:
            return str(getattr(target, "element", "neutral") or "neutral").lower()
    except Exception:
        pass
    return "neutral"


def _aggro(snapshot: Any) -> int:
    try:
        combat = getattr(snapshot, "combat", None)
        if combat is not None:
            return int(getattr(combat, "aggro_count", 0) or 0)
    except Exception:
        pass
    return 0


def _emit(runtime: Any, bot_id: str, command: str, reason: str,
          domain: str, conflict_key: str, obs: bool = False) -> int:
    aq = getattr(runtime, "action_queue", None)
    if aq is None:
        return 0
    if obs:
        logger.info("combat_tactics_observe: bot=%s command=%s reason=%s", bot_id, command, reason)
        return 0
    try:
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        _key = conflict_key or f"ct_{bot_id}"
        prop = ActionProposal(
            action_id=f"ct_{bot_id}_{int(time.monotonic()*1000)}",
            kind="command",
            command=command,
            priority_tier=ActionPriorityTier.tactical,
            conflict_key=_key,
            idempotency_key=_key,
            source="combat_tactics",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            metadata={"source": "combat_tactics", "reason": reason, "domain": domain,
                      "bot_id": bot_id},
        )
        ok, status, aid, why = aq.enqueue(bot_id, prop)
        if ok:
            logger.info("combat_tactics_queued: bot=%s cmd=%s", bot_id, command)
            return 1
        logger.debug("combat_tactics_rejected: bot=%s reason=%s", bot_id, why)
        return 0
    except Exception as e:
        logger.debug("combat_tactics_enqueue_err: %s", e)
        return 0


def run_combat_tactics(runtime: Any, bot_id: str | None, snapshot: Any = None) -> int:
    """Evaluate the dormant CombatTactics class for one bot-cycle and queue skills."""
    if not bot_id:
        return 0
    ct = _get_ct()
    if ct is None:
        return 0
    if not _in_game(snapshot):
        return 0

    hp_pct, current_sp, current_hp, _max_hp, job = _snap_vitals(snapshot)
    element = _target_element(snapshot)
    aggro = _aggro(snapshot)
    total = 0

    # Kiting rule (ranged always kite; melee at low HP).
    try:
        if ct.should_kite(job, hp_pct):
            total += _emit(runtime, bot_id, "move (kite_back)", "kite", "combat_tactics",
                           f"ct_kite_{bot_id}", obs=True)
    except Exception as e:
        logger.debug("combat_tactics_kite_err: %s", e)

    # Only bother computing a skill combo when actively in combat.
    if aggro <= 0:
        return total
    try:
        combo = ct.get_combo(job, element, hp_pct, aggro, False) or []
    except Exception as e:
        logger.debug("combat_tactics_get_combo_err: %s", e)
        combo = []
    for skill in combo[:3]:
        sname = str(skill).strip()
        if not sname:
            continue
        # turn `ss frost_diver` -> skill name `frost_diver` for the SP check
        sk = sname.split()[-1] if sname.startswith("ss ") else sname
        sp_cost = _sp_cost(sk)
        # Gate by SP/HP so we never cast a skill we can't afford (bogus command).
        try:
            can, why = _gate_execute(ct, sk, sp_cost, current_sp, current_hp)
            if not can:
                logger.debug("combat_tactics_skill_ungated: %s (%s)", sname, why)
                continue
        except Exception:
            pass
        # Emit the real skill command (throttled per bot to avoid spam).
        now = time.time()
        if now - _LAST_CAST.get(bot_id, 0) < 3.0:
            break
        _LAST_CAST[bot_id] = now
        total += _emit(runtime, bot_id, sname, f"combo {element}", "combat_tactics",
                       f"ct_skill_{bot_id}_{sk}")
    return total


def _sp_cost(skill: str) -> int:
    if skill in ("frost_diver", "fire_bolt", "cold_bolt", "lightning_bolt"):
        return 10
    if skill in ("double_strafing", "arrow_shower"):
        return 8
    if skill in ("improve_concentration",):
        return 10
    return 15


def _gate_execute(ct: Any, skill_name: str, sp_cost: int, current_sp: int, current_hp: int) -> tuple[bool, str]:
    """Use CombatTactics.can_execute_skill to gate on SP/HP (default HP cost 0)."""
    try:
        from ai_sidecar.combat_tactics import can_execute_skill
        return can_execute_skill(skill_name, sp_cost, current_sp, current_hp)
    except Exception:
        return True, "ok"
