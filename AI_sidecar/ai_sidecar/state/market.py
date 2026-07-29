"""MarketState — vending, buying store, player shop interaction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ShopItem(BaseModel):
    """An item listed in a player shop or vending stall."""

    model_config = ConfigDict(extra="ignore")

    item_id: int | None = None
    item_name: str = ""
    quantity: int = 1
    price: int = 0
    slots: int = 0
    cards: list[int] = Field(default_factory=list)


class MarketState(BaseModel):
    """Market interaction state — vending, buying, player shops."""

    model_config = ConfigDict(extra="ignore")

    is_vending: bool = False  # Currently vending (selling to others)
    is_buying_store: bool = False  # Currently buying (buying from others)
    shop_title: str | None = None  # Title of our vending/buying store
    shop_items: list[ShopItem] = Field(default_factory=list)  # Items in our shop
    shop_zeny: int = 0  # Zeny earned by shop while AFK
    shop_transactions: int = 0  # Number of transactions since opening

    # Nearby player shops
    nearby_shops: list[dict[str, Any]] = Field(default_factory=list)
    nearby_vendors: int = 0

    # OpenKore buying store
    buying_item: str | None = None  # Item we're currently buying from Shop
    buying_price: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_market(signals: dict[str, Any]) -> MarketState:
    """Parse market/vending state from the bridge signal dict.

    Handles:
      - ``signals['market']`` — dict with market info
      - ``signals['is_vending']``, ``signals['shop_title']`` — flat keys
      - ``signals['shop_items']`` — list of item dicts in our shop
      - ``signals['nearby_shops']`` — list of nearby player shop dicts
    """
    m_dict: dict[str, Any] = signals.get("market") or {}

    is_vending = bool(signals.get("is_vending", m_dict.get("is_vending", False)))
    is_buying = bool(signals.get("is_buying_store", m_dict.get("is_buying_store", False)))

    if not is_vending and not is_buying:
        # Check flat keys
        shop_title = signals.get("shop_title") or m_dict.get("shop_title")
        if not shop_title:
            return MarketState()

    # Parse shop items
    items_raw: list[dict] = list(
        signals.get("shop_items", m_dict.get("shop_items", [])) or []
    )
    shop_items = [
        ShopItem(
            item_id=int(i.get("item_id", 0)) or None,
            item_name=str(i.get("item_name", i.get("name", ""))),
            quantity=int(i.get("quantity", 1)),
            price=int(i.get("price", 0)),
            slots=int(i.get("slots", 0)),
            cards=[int(c) for c in (i.get("cards") or []) if c],
        )
        for i in items_raw if isinstance(i, dict)
    ]

    # Nearby shops
    nearby_raw: list[dict] = list(
        signals.get("nearby_shops", m_dict.get("nearby_shops", [])) or []
    )
    nearby_shops: list[dict] = []
    for s in nearby_raw:
        if isinstance(s, dict):
            nearby_shops.append(s)

    return MarketState(
        is_vending=is_vending,
        is_buying_store=is_buying,
        shop_title=str(signals.get("shop_title", m_dict.get("shop_title", ""))) or None,
        shop_items=shop_items,
        shop_zeny=int(signals.get("shop_zeny", m_dict.get("shop_zeny", 0))),
        shop_transactions=int(signals.get("shop_transactions", m_dict.get("shop_transactions", 0))),
        nearby_shops=nearby_shops,
        nearby_vendors=len(nearby_shops),
        buying_item=str(signals.get("buying_item", m_dict.get("buying_item", ""))) or None,
        buying_price=int(signals.get("buying_price", m_dict.get("buying_price", 0))),
    )
