from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent
from ai_sidecar.contracts.state_graph import EconomyState


@dataclass(slots=True)
class _EconomyWindow:
    zeny_points: deque[tuple[datetime, int]] = field(default_factory=deque)
    transactions: deque[datetime] = field(default_factory=deque)
    listings: list[dict[str, object]] = field(default_factory=list)
    inventory_value_estimate: int = 0


class EconomyTracker:
    """Tracks economy and market signals for compact enriched state."""

    def __init__(self) -> None:
        self._by_bot: dict[str, _EconomyWindow] = {}

    def observe_snapshot(self, *, bot_id: str, payload: dict[str, object], observed_at: datetime) -> None:
        window = self._by_bot.setdefault(bot_id, _EconomyWindow())
        inventory = payload.get("inventory") if isinstance(payload.get("inventory"), dict) else {}
        zeny = _int_or_default(inventory.get("zeny"), 0)
        window.zeny_points.append((observed_at, zeny))

        market = payload.get("market") if isinstance(payload.get("market"), dict) else {}
        listings = market.get("listings") if isinstance(market.get("listings"), list) else []
        compact_listings: list[dict[str, object]] = []
        for item in listings[:40]:
            if not isinstance(item, dict):
                continue
            compact_listings.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "item_name": str(item.get("item_name") or "")[:64],
                    "buy_price": _int_or_none(item.get("buy_price")),
                    "sell_price": _int_or_none(item.get("sell_price")),
                    "quantity": _int_or_none(item.get("quantity")),
                    "source": str(item.get("source") or "")[:32],
                }
            )
        if compact_listings:
            window.listings = compact_listings

        inventory_items_payload = payload.get("inventory_items")
        if isinstance(inventory_items_payload, list):
            estimate = 0
            for item in inventory_items_payload:
                if not isinstance(item, dict):
                    continue
                qty = max(0, _int_or_default(item.get("quantity"), 0))
                sell = max(0, _int_or_default(item.get("sell_price"), 0))
                buy = max(0, _int_or_default(item.get("buy_price"), 0))
                estimate += qty * max(sell, buy // 2)
            window.inventory_value_estimate = estimate

        self._trim(window, observed_at)

    def observe_event(self, event: NormalizedEvent) -> None:
        bot_id = event.meta.bot_id
        now = _normalize_dt(event.observed_at)
        window = self._by_bot.setdefault(bot_id, _EconomyWindow())

        zeny = _int_or_none(event.payload.get("zeny"))
        if zeny is None:
            numeric_zeny = event.numeric.get("zeny")
            if isinstance(numeric_zeny, float):
                zeny = int(numeric_zeny)
        if zeny is not None:
            window.zeny_points.append((now, zeny))

        if event.event_family in {EventFamily.snapshot, EventFamily.action, EventFamily.macro}:
            if any(token in event.event_type for token in {"buy", "sell", "vendor", "deal", "trade", "market"}):
                window.transactions.append(now)

        if event.event_family == EventFamily.snapshot:
            market = event.payload.get("market") if isinstance(event.payload.get("market"), dict) else {}
            listings = market.get("listings") if isinstance(market.get("listings"), list) else []
            if listings:
                compact_listings: list[dict[str, object]] = []
                for item in listings[:40]:
                    if not isinstance(item, dict):
                        continue
                    compact_listings.append(
                        {
                            "item_id": str(item.get("item_id") or ""),
                            "item_name": str(item.get("item_name") or "")[:64],
                            "buy_price": _int_or_none(item.get("buy_price")),
                            "sell_price": _int_or_none(item.get("sell_price")),
                            "quantity": _int_or_none(item.get("quantity")),
                            "source": str(item.get("source") or "")[:32],
                        }
                    )
                if compact_listings:
                    window.listings = compact_listings

        self._trim(window, now)

    def export(self, *, bot_id: str, zeny: int | None, observed_at: datetime) -> EconomyState:
        window = self._by_bot.get(bot_id)
        if window is None:
            return EconomyState(zeny=zeny, updated_at=observed_at)

        self._trim(window, observed_at)
        zeny_delta_1m = _delta(window.zeny_points, observed_at - timedelta(minutes=1))
        zeny_delta_10m = _delta(window.zeny_points, observed_at - timedelta(minutes=10))

        price_samples = [
            float(item.get("sell_price") or item.get("buy_price") or 0.0)
            for item in window.listings
            if isinstance(item, dict)
        ]
        price_signal_index = 0.0
        if price_samples:
            mean = sum(price_samples) / float(len(price_samples))
            price_signal_index = round(mean / 1000.0, 4)

        return EconomyState(
            zeny=zeny,
            zeny_delta_1m=zeny_delta_1m,
            zeny_delta_10m=zeny_delta_10m,
            vendor_exposure=len(window.listings),
            transaction_count_10m=len(window.transactions),
            inventory_value_estimate=window.inventory_value_estimate,
            price_signal_index=price_signal_index,
            market_listings=list(window.listings[:20]),
            updated_at=observed_at,
            raw={
                "zeny_points": len(window.zeny_points),
                "transactions_window": len(window.transactions),
            },
        )

    def _trim(self, window: _EconomyWindow, observed_at: datetime) -> None:
        zeny_threshold = observed_at - timedelta(minutes=12)
        while window.zeny_points and window.zeny_points[0][0] < zeny_threshold:
            window.zeny_points.popleft()

        txn_threshold = observed_at - timedelta(minutes=10)
        while window.transactions and window.transactions[0] < txn_threshold:
            window.transactions.popleft()


def _delta(points: deque[tuple[datetime, int]], floor: datetime) -> int:
    if not points:
        return 0
    baseline = points[0][1]
    for ts, value in points:
        if ts >= floor:
            baseline = value
            break
    return points[-1][1] - baseline


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
