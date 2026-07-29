"""StateCollector — aggregates all 17 state collectors into a structured GameState."""

from __future__ import annotations

import time as _time_module
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_sidecar.state.character import CharacterState, collect_character
from ai_sidecar.state.inventory import InventoryState, collect_inventory
from ai_sidecar.state.map_state import MapState, collect_map
from ai_sidecar.state.party import PartyState, collect_party
from ai_sidecar.state.guild import GuildState, collect_guild
from ai_sidecar.state.buffs import BuffState, collect_buffs
from ai_sidecar.state.pets import PetState, collect_pets
from ai_sidecar.state.homunculus import HomunculusState, collect_homunculus
from ai_sidecar.state.mercenary import MercenaryState, collect_mercenary
from ai_sidecar.state.mount import MountState, collect_mount
from ai_sidecar.state.equipment import EquipmentState, collect_equipment
from ai_sidecar.state.dialogue import DialogueState, collect_dialogue
from ai_sidecar.state.quests import QuestState, collect_quests
from ai_sidecar.state.market import MarketState, collect_market
from ai_sidecar.state.environment import EnvironmentState, collect_environment
from ai_sidecar.state.instances import InstanceState, collect_instances
from ai_sidecar.state.companions import CompanionState, collect_companions


class GameState(BaseModel):
    """Complete structured game state — all 17 sub-states aggregated.

    This is the top-level output of ``StateCollector.collect()``.
    Every field is a fully-parsed Pydantic model with sensible defaults,
    so downstream consumers never have to handle raw dicts.
    """

    model_config = ConfigDict(extra="ignore")

    # ── Metadata ──
    bot_id: str = "default"
    bot_name: str = ""
    collected_at: float = 0.0  # Unix timestamp
    horizon: str = "short_term"

    # ── 17 Specialized sub-states ──
    character: CharacterState = Field(default_factory=CharacterState)
    inventory: InventoryState = Field(default_factory=InventoryState)
    map_state: MapState = Field(default_factory=MapState)
    party: PartyState = Field(default_factory=PartyState)
    guild: GuildState = Field(default_factory=GuildState)
    buffs: BuffState = Field(default_factory=BuffState)
    pets: PetState = Field(default_factory=PetState)
    homunculus: HomunculusState = Field(default_factory=HomunculusState)
    mercenary: MercenaryState = Field(default_factory=MercenaryState)
    mount: MountState = Field(default_factory=MountState)
    equipment: EquipmentState = Field(default_factory=EquipmentState)
    dialogue: DialogueState = Field(default_factory=DialogueState)
    quests: QuestState = Field(default_factory=QuestState)
    market: MarketState = Field(default_factory=MarketState)
    environment: EnvironmentState = Field(default_factory=EnvironmentState)
    instances: InstanceState = Field(default_factory=InstanceState)
    companions: CompanionState = Field(default_factory=CompanionState)

    # ── Raw signals (unparsed) ──
    raw: dict[str, Any] = Field(default_factory=dict)


class StateCollector:
    """Collects structured game state from bridge signal dicts.

    Usage::

        collector = StateCollector()
        game_state = collector.collect(bridge_signals)
        print(game_state.character.job_name)
        print(game_state.inventory.potions)
    """

    def collect(self, bridge_signals: dict[str, Any]) -> GameState:
        """Parse a flat bridge signal dictionary into a structured GameState.

        Each sub-state collector handles its own parsing with graceful defaults
        for missing fields. The bridge signal format is a flat dict with keys
        like ``hp_ratio``, ``map``, ``inventory_items``, etc.
        """
        collected_at = _time_module.time()

        bot_id = str(bridge_signals.get("bot_id", "default"))
        bot_name = str(bridge_signals.get("bot_name", bridge_signals.get("char_name", bot_id)))

        return GameState(
            bot_id=bot_id,
            bot_name=bot_name,
            collected_at=collected_at,
            horizon=str(bridge_signals.get("horizon", "short_term")),
            character=collect_character(bridge_signals),
            inventory=collect_inventory(bridge_signals),
            map_state=collect_map(bridge_signals),
            party=collect_party(bridge_signals),
            guild=collect_guild(bridge_signals),
            buffs=collect_buffs(bridge_signals),
            pets=collect_pets(bridge_signals),
            homunculus=collect_homunculus(bridge_signals),
            mercenary=collect_mercenary(bridge_signals),
            mount=collect_mount(bridge_signals),
            equipment=collect_equipment(bridge_signals),
            dialogue=collect_dialogue(bridge_signals),
            quests=collect_quests(bridge_signals),
            market=collect_market(bridge_signals),
            environment=collect_environment(bridge_signals),
            instances=collect_instances(bridge_signals),
            companions=collect_companions(bridge_signals),
            raw=dict(bridge_signals),
        )
