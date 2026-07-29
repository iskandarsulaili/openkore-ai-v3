"""PetState — pet name, intimacy, hunger, stats."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PetState(BaseModel):
    """Active pet information."""

    model_config = ConfigDict(extra="ignore")

    active: bool = False
    name: str | None = None
    pet_id: int | None = None
    level: int | None = None
    intimacy: int = 0  # 0-1000 intimacy scale
    hunger: int = 0  # 0-100 hunger scale
    friendly: int | None = None  # Friendly rating (some servers)
    rename_flag: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def intimacy_rating(self) -> str:
        """Return a human-readable intimacy level."""
        if self.intimacy >= 900:
            return "loyal"
        elif self.intimacy >= 750:
            return "very_intimate"
        elif self.intimacy >= 500:
            return "intimate"
        elif self.intimacy >= 250:
            return "neutral"
        else:
            return "awkward"

    @property
    def is_hungry(self) -> bool:
        return self.hunger < 30

    @property
    def is_starving(self) -> bool:
        return self.hunger < 10


def collect_pets(signals: dict[str, Any]) -> PetState:
    """Parse pet information from the bridge signal dict.

    Handles:
      - ``signals['pet']`` — dict with pet info (name, intimacy, hunger, level)
      - ``signals['pet_name']``, ``signals['pet_intimacy']``, etc. — flat keys
      - ``signals['has_pet']`` — boolean indicator
    """
    pet_dict: dict[str, Any] = signals.get("pet") or {}
    if isinstance(pet_dict, str):
        # Bridge may send just the pet name
        return PetState(active=True, name=pet_dict)

    has_pet = bool(signals.get("has_pet", False) or pet_dict)
    if not has_pet and not pet_dict:
        # Check flat keys
        pet_name = signals.get("pet_name")
        if not pet_name:
            return PetState(active=False)
        return PetState(
            active=True,
            name=str(pet_name),
            intimacy=int(signals.get("pet_intimacy", 0)),
            hunger=int(signals.get("pet_hunger", 0)),
            level=int(signals.get("pet_level", 0)) or None,
            pet_id=int(signals.get("pet_id", 0)) or None,
        )

    return PetState(
        active=True,
        name=str(pet_dict.get("name", pet_dict.get("pet_name", ""))) or None,
        pet_id=int(pet_dict.get("pet_id", 0)) or None,
        level=int(pet_dict.get("level", 0)) or None,
        intimacy=int(pet_dict.get("intimacy", pet_dict.get("intimate", 0))),
        hunger=int(pet_dict.get("hunger", pet_dict.get("hungry", 0))),
        friendly=int(pet_dict.get("friendly", 0)) or None,
        rename_flag=bool(pet_dict.get("rename_flag", False)),
    )
