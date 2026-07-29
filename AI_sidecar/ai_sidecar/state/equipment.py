"""EquipmentState — all equipped gear slots with refine and cards."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EquippedItem(BaseModel):
    """An item equipped in a specific slot."""

    model_config = ConfigDict(extra="ignore")

    slot: str = ""
    item_id: int | None = None
    item_name: str = ""
    refine: int = 0
    cards: list[int] = Field(default_factory=list)
    position: int = 0  # Original position/slot index
    quantity: int = 1


class EquipmentState(BaseModel):
    """Complete equipment state — all equipped slots."""

    model_config = ConfigDict(extra="ignore")

    slots: dict[str, EquippedItem] = Field(default_factory=dict)
    weapon: EquippedItem | None = None
    shield: EquippedItem | None = None
    armor: EquippedItem | None = None
    head_top: EquippedItem | None = None
    head_mid: EquippedItem | None = None
    head_bottom: EquippedItem | None = None
    garment: EquippedItem | None = None
    shoes: EquippedItem | None = None
    accessory_1: EquippedItem | None = None
    accessory_2: EquippedItem | None = None
    ammo: EquippedItem | None = None
    attack_power: int = 0
    defence: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


# ── Slot name mappings ──
_SLOT_KEYS: dict[str, str] = {
    "weapon": "weapon",
    "shield": "shield",
    "armor": "armor",
    "upper_head": "head_top",
    "mid_head": "head_mid",
    "lower_head": "head_bottom",
    "head_top": "head_top",
    "head_mid": "head_mid",
    "head_bottom": "head_bottom",
    "garment": "garment",
    "robe": "garment",
    "shoes": "shoes",
    "boots": "shoes",
    "accessory_1": "accessory_1",
    "accessory_2": "accessory_2",
    "right_accessory": "accessory_1",
    "left_accessory": "accessory_2",
    "ammo": "ammo",
    "arrow": "ammo",
}


def _build_equipped_item(slot_name: str, raw: dict[str, Any] | str) -> EquippedItem:
    """Convert a raw equipment entry into an EquippedItem."""
    if isinstance(raw, str):
        return EquippedItem(slot=slot_name, item_name=raw)
    return EquippedItem(
        slot=slot_name,
        item_id=int(raw.get("item_id", 0)) or None,
        item_name=str(raw.get("item_name", raw.get("name", ""))),
        refine=int(raw.get("refine", raw.get("refines", 0))),
        cards=[int(c) for c in (raw.get("cards") or []) if c],
        position=int(raw.get("position", 0)),
        quantity=int(raw.get("quantity", 1)),
    )


def collect_equipment(signals: dict[str, Any]) -> EquipmentState:
    """Parse equipment state from the bridge signal dict.

    Handles:
      - ``signals['equipment']`` — dict of slot_name -> item dict
      - ``signals['equipment_slots']`` — alternative slot dict
      - ``signals['attack_power']``, ``signals['defence']`` — combat stats
    """
    equip_dict: dict[str, Any] = signals.get("equipment", signals.get("equipment_slots", {}))
    if not isinstance(equip_dict, dict):
        equip_dict = {}

    attack_power = int(signals.get("attack_power", 0))
    defence = int(signals.get("defence", signals.get("defense", 0)))

    slots: dict[str, EquippedItem] = {}
    slot_refs: dict[str, EquippedItem | None] = {}

    for raw_slot, raw_item in equip_dict.items():
        normalized = _SLOT_KEYS.get(raw_slot, raw_slot)
        if isinstance(raw_item, (dict, str)) and raw_item:
            item = _build_equipped_item(normalized, raw_item)
            slots[raw_slot] = item
            slot_refs[normalized] = item

    return EquipmentState(
        slots=slots,
        weapon=slot_refs.get("weapon"),
        shield=slot_refs.get("shield"),
        armor=slot_refs.get("armor"),
        head_top=slot_refs.get("head_top"),
        head_mid=slot_refs.get("head_mid"),
        head_bottom=slot_refs.get("head_bottom"),
        garment=slot_refs.get("garment"),
        shoes=slot_refs.get("shoes"),
        accessory_1=slot_refs.get("accessory_1"),
        accessory_2=slot_refs.get("accessory_2"),
        ammo=slot_refs.get("ammo"),
        attack_power=attack_power,
        defence=defence,
    )
