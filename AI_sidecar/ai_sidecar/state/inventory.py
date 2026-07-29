"""InventoryState — items list, equipped gear, weight tracking."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InventoryItem(BaseModel):
    """A single inventory item as reported by the bridge."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    name_id: int | None = None
    quantity: int = 0
    equipped: bool = False
    slot: int | None = None  # Equipment slot index if equipped
    type_: str | None = Field(default=None, alias="type")
    identified: bool = True
    broken: bool = False
    cards: list[int] = Field(default_factory=list)
    bind_on_equip: bool = False
    weight: int | None = None
    buy_price: int | None = None
    sell_price: int | None = None


class EquipmentSlot(BaseModel):
    """An equipped item in a specific slot."""

    model_config = ConfigDict(extra="ignore")

    slot_name: str = ""
    item_name: str = ""
    item_id: int | None = None
    refine: int = 0
    cards: list[int] = Field(default_factory=list)


class InventoryState(BaseModel):
    """Full inventory state: items list, equipped gear, and weight info."""

    model_config = ConfigDict(extra="ignore")

    items: list[InventoryItem] = Field(default_factory=list)
    equipped: dict[str, EquipmentSlot] = Field(default_factory=dict)
    item_count: int = 0
    weight: int = 0
    weight_max: int = 2000
    weight_ratio: float = 0.0
    zeny: int = 0
    arrows: int = 0
    potions: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


# ── Known equipment slot names (OpenKore slot indices) ──
_EQUIP_SLOTS: list[str] = [
    "upper_head", "mid_head", "lower_head",
    "armor", "weapon", "shield",
    "garment", "shoes",
    "accessory_1", "accessory_2",
    "ammo",
    "costume_top", "costume_mid", "costume_bottom",
    "costume_robe",
]


def collect_inventory(signals: dict[str, Any]) -> InventoryState:
    """Parse inventory and equipment from the bridge signal dict.

    Handles:
      - ``signals['inventory_items']`` — list of item dicts
      - ``signals['inventory']`` — dict with weight/zeny metadata
      - ``signals['equipment']`` — dict of equipped slots
      - Flat keys: ``weight``, ``weight_max``, ``weight_ratio``, ``zeny``
    """
    raw_items: list[dict] = list(signals.get("inventory_items") or [])
    inv_meta: dict = signals.get("inventory") or {}
    equip_dict: dict = signals.get("equipment") or {}

    # Parse inventory items
    items: list[InventoryItem] = []
    for raw in raw_items:
        if isinstance(raw, str):
            # Bridge sometimes sends item names as plain strings
            items.append(InventoryItem(name=str(raw), quantity=1))
        elif isinstance(raw, dict):
            items.append(InventoryItem(**{k: v for k, v in raw.items() if k in InventoryItem.model_fields}))
        else:
            items.append(InventoryItem(name=str(raw), quantity=1))

    # Parse equipped gear
    equipped: dict[str, EquipmentSlot] = {}
    for slot_name in _EQUIP_SLOTS:
        slot_data = equip_dict.get(slot_name) or {}
        if isinstance(slot_data, dict) and slot_data.get("item_name"):
            equipped[slot_name] = EquipmentSlot(slot_name=slot_name, **slot_data)
        elif isinstance(slot_data, str):
            equipped[slot_name] = EquipmentSlot(slot_name=slot_name, item_name=slot_data)

    # Count potions and arrows
    potion_count = 0
    arrow_count = 0
    for item in items:
        name_lower = item.name.lower()
        if "potion" in name_lower or "red" in name_lower or "white" in name_lower or "orange" in name_lower:
            potion_count += item.quantity
        if "arrow" in name_lower or "shell" in name_lower or "bullet" in name_lower:
            arrow_count += item.quantity

    weight = int(signals.get("weight", inv_meta.get("weight", 0)))
    weight_max = int(signals.get("weight_max", inv_meta.get("weight_max", 2000)))
    weight_ratio = float(signals.get("weight_ratio", inv_meta.get("weight_ratio", 0.0)))

    return InventoryState(
        items=items,
        equipped=equipped,
        item_count=len(items),
        weight=weight,
        weight_max=weight_max or 2000,
        weight_ratio=weight_ratio or (weight / max(weight_max, 1) if weight_max > 0 else 0.0),
        zeny=int(signals.get("zeny", inv_meta.get("zeny", 0))),
        arrows=arrow_count,
        potions=potion_count,
    )
