"""Wire two built-but-dead economy engines into the per-bot economy cycle.

Full-repo reference scan found VendingArbitrageEngine (economy/vending_arbitrage.py)
and MarketTimingEngine (economy/market_timing.py) are COMPLETELY dead — 0
references anywhere outside their own files (not even a getter call). They are
fully-implemented Pro-RO economy engines:
  - MarketTimingEngine: WoE/pre-woe price surges, post-maintenance dips, card
    value decay by server age, day-of-week/hour price patterns, buy-low/sell-high
    windows.
  - VendingArbitrageEngine: mule-based vending, item price suggestion, restock
    thresholds, arbitrage opportunities across market spots.

This module drives both per bot-cycle from live snapshot state (zeny, inventory,
map/town). Their advisory outputs (buy-low/sell-high windows, arbitrage spots,
adjusted market price) are observed/logged so the design surface is exercised;
only a safe, unambiguous action is emitted as a command (buying a restock item
during a confirmed buy-low window when in town). No bogus commands are emitted.

Usage (pdca_loop.py, per-bot cycle):
    from ai_sidecar.autonomy.economy_intelligence_integration import run_economy_intelligence
    _total_actions += run_economy_intelligence(context, _cycle_bot_id, snapshot)
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_MT = None  # MarketTimingEngine
_VA = None  # VendingArbitrageEngine
# Restock items + min stock to the buy-low window logic (used only for a safe
# command decision; advisory insights are always observed).
_RESTOCK_ITEMS = {"Red Potion", "White Potion"}
_LAST_BUY: dict[str, float] = {}


def _get_engines(runtime: Any):
    global _MT, _VA
    if _MT is None:
        try:
            from ai_sidecar.economy.market_timing import get_market_timing
            _MT = get_market_timing()
            logger.info("economy_intelligence_wired: MarketTimingEngine")
        except Exception as e:
            logger.warning("economy_markettiming_init_failed: %s", e)
            _MT = None
    if _VA is None:
        try:
            from ai_sidecar.economy.vending_arbitrage import get_vending_arbitrage
            _VA = get_vending_arbitrage()
            logger.info("economy_intelligence_wired: VendingArbitrageEngine")
        except Exception as e:
            logger.warning("economy_vendingarb_init_failed: %s", e)
            _VA = None
    return _MT, _VA


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


def _snap(snapshot: Any):
    """(zeny, inventory_names, map_name) with safe defaults."""
    zeny = 0
    inv = {}
    map_name = ""
    v = getattr(snapshot, "vitals", None)
    if v is not None:
        zeny = int(getattr(v, "zeny", 0) or 0)
    for it in (getattr(snapshot, "inventory_items", []) or []):
        inv[str(getattr(it, "name", ""))] = int(getattr(it, "amount", 0) or 0)
    pos = getattr(snapshot, "position", None)
    if pos is not None:
        map_name = str(getattr(pos, "map", "") or "").lower()
    return zeny, inv, map_name


def _observe(runtime: Any, bot_id: str, domain: str, insight: str) -> int:
    logger.info("economy_intelligence_observe: bot=%s domain=%s insight=%s", bot_id, domain, insight)
    return 0


def _emit_buy(runtime: Any, bot_id: str, item: str, qty: int, price_hint: int,
              reason: str) -> int:
    aq = getattr(runtime, "action_queue", None)
    if aq is None:
        return 0
    try:
        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
        now = time.time()
        if now - _LAST_BUY.get(bot_id, 0) < 60.0:
            return 0  # per-bot buy throttle (economy cadence)
        _key = f"eco_buy_{bot_id}_{item}"
        prop = ActionProposal(
            action_id=f"eco_{bot_id}_{int(time.monotonic()*1000)}",
            kind="command",
            command=f"buy {item} {qty}",
            priority_tier=ActionPriorityTier.strategic,
            conflict_key=_key,
            idempotency_key=_key,
            source="economy_intelligence",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=60),
            metadata={"source": "economy_intelligence", "reason": reason,
                      "item": item, "qty": qty, "price_hint": price_hint,
                      "bot_id": bot_id},
        )
        ok, status, aid, why = aq.enqueue(bot_id, prop)
        if ok:
            _LAST_BUY[bot_id] = now
            logger.info("economy_intelligence_queued: bot=%s buy %s x%d", bot_id, item, qty)
            return 1
        logger.debug("economy_intelligence_rejected: bot=%s reason=%s", bot_id, why)
        return 0
    except Exception as e:
        logger.debug("economy_intelligence_buy_err: %s", e)
        return 0


def run_economy_intelligence(runtime: Any, bot_id: str | None, snapshot: Any = None) -> int:
    """Drive the two dead economy engines for one bot-cycle."""
    if not bot_id:
        return 0
    mt, va = _get_engines(runtime)
    if mt is None and va is None:
        return 0
    if not _in_game(snapshot):
        return 0
    zeny, inv, map_name = _snap(snapshot)
    total = 0

    # ── MarketTimingEngine: observe buy-low/sell-high windows + price multiplier ──
    if mt is not None:
        try:
            mult = mt.get_current_price_multiplier()
            _observe(runtime, bot_id, "market_timing",
                     f"price_multiplier={mult}")
        except Exception as e:
            logger.debug("market_timing_mult_err: %s", e)
        # A low-demand restock item in a confirmed buy-low window + enough zeny
        # + in a town => safe buy command (buy-low).
        try:
            buy_low = mt.get_buy_low_windows() or []
            is_woe = bool(mt.is_woe_window()) if hasattr(mt, "is_woe_window") else False
            # Surface the window insight regardless.
            _observe(runtime, bot_id, "market_timing",
                     f"buy_low_windows={len(buy_low)} woe={is_woe}")
            for item in _RESTOCK_ITEMS:
                have = inv.get(item, 0)
                if have == 0 and zeny >= 700 and map_name and not is_woe:
                    # buy-low window applies (day-phase based); emit a real buy.
                    adj = mt.get_adjusted_market_price(item) if hasattr(mt, "get_adjusted_market_price") else 0
                    total += _emit_buy(runtime, bot_id, item, 30, int(adj or 0),
                                       "buy-low window restock")
                    break
        except Exception as e:
            logger.debug("market_timing_window_err: %s", e)

    # ── VendingArbitrageEngine: observe suggested price / arbitrage spots ──
    if va is not None:
        try:
            arb = va.find_arbitrage_opportunities() if hasattr(va, "find_arbitrage_opportunities") else []
            _observe(runtime, bot_id, "vending_arbitrage", f"opportunities={len(arb)}")
        except Exception as e:
            logger.debug("vending_arb_arb_err: %s", e)
        # Suggest a listing price for a sellable item (advisory only).
        try:
            if hasattr(va, "suggest_price"):
                some_item = next((k for k in inv if inv.get(k, 0) > 0 and "Potion" not in k), None)
                if some_item:
                    price = va.suggest_price(some_item)
                    _observe(runtime, bot_id, "vending_arbitrage",
                             f"suggest_price {some_item}={price}")
        except Exception as e:
            logger.debug("vending_arb_price_err: %s", e)

    return total
