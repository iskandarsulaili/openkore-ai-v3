"""CompanionState — aggregates pet + homunculus + mercenary + mount."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.state.pets import PetState, collect_pets
from ai_sidecar.state.homunculus import HomunculusState, collect_homunculus
from ai_sidecar.state.mercenary import MercenaryState, collect_mercenary
from ai_sidecar.state.mount import MountState, collect_mount


class CompanionState(BaseModel):
    """Aggregated companion info — pet, homunculus, mercenary, and mount."""

    model_config = ConfigDict(extra="ignore")

    has_active_companion: bool = False
    companion_type: str | None = None  # "pet" | "homunculus" | "mercenary" | "mount"
    companion_name: str | None = None

    pet: PetState = Field(default_factory=PetState)
    homunculus: HomunculusState = Field(default_factory=HomunculusState)
    mercenary: MercenaryState = Field(default_factory=MercenaryState)
    mount: MountState = Field(default_factory=MountState)

    raw: dict[str, Any] = Field(default_factory=dict)


def collect_companions(signals: dict[str, Any]) -> CompanionState:
    """Parse all companion types and determine the active companion.

    Each sub-collector handles its own parsing.
    We then determine which companion is active (priority: pet > homunculus > mercenary > mount).
    """
    pet = collect_pets(signals)
    homunculus = collect_homunculus(signals)
    mercenary = collect_mercenary(signals)
    mount = collect_mount(signals)

    # Determine active companion (first one active in priority order)
    companion_type: str | None = None
    companion_name: str | None = None

    if pet.active:
        companion_type = "pet"
        companion_name = pet.name
    elif homunculus.active:
        companion_type = "homunculus"
        companion_name = homunculus.name
    elif mercenary.active:
        companion_type = "mercenary"
        companion_name = mercenary.name
    elif mount.active:
        companion_type = "mount"
        companion_name = mount.name

    return CompanionState(
        has_active_companion=companion_type is not None,
        companion_type=companion_type,
        companion_name=companion_name,
        pet=pet,
        homunculus=homunculus,
        mercenary=mercenary,
        mount=mount,
    )
