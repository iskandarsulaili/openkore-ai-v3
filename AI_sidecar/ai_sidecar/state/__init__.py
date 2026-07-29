"""State module — 17 specialized state collectors producing Pydantic models.

Every public model and collector function is re-exported from here so
consumers can import from a single location::

    from ai_sidecar.state import (
        GameState, StateCollector,
        CharacterState, InventoryState, MapState,
        PartyState, GuildState, BuffState,
        PetState, HomunculusState, MercenaryState,
        MountState, EquipmentState, DialogueState,
        QuestState, MarketState, EnvironmentState,
        InstanceState, CompanionState,
        collect_character, collect_inventory, collect_map,
        collect_party, collect_guild, collect_buffs,
        collect_pets, collect_homunculus, collect_mercenary,
        collect_mount, collect_equipment, collect_dialogue,
        collect_quests, collect_market, collect_environment,
        collect_instances, collect_companions,
    )
"""

from __future__ import annotations

from ai_sidecar.state.collector import GameState, StateCollector

from ai_sidecar.state.character import CharacterState, collect_character
from ai_sidecar.state.inventory import InventoryState, InventoryItem, EquipmentSlot, collect_inventory
from ai_sidecar.state.map_state import MapState, PortalInfo, MonsterSpawn, collect_map
from ai_sidecar.state.party import PartyState, PartyMember, collect_party
from ai_sidecar.state.guild import GuildState, GuildMember, GuildSkill, collect_guild
from ai_sidecar.state.buffs import BuffState, ActiveBuff, collect_buffs
from ai_sidecar.state.pets import PetState, collect_pets
from ai_sidecar.state.homunculus import HomunculusState, HomunculusSkill, collect_homunculus
from ai_sidecar.state.mercenary import MercenaryState, collect_mercenary
from ai_sidecar.state.mount import MountState, collect_mount
from ai_sidecar.state.equipment import EquipmentState, EquippedItem, collect_equipment
from ai_sidecar.state.dialogue import DialogueState, DialogueOption, collect_dialogue
from ai_sidecar.state.quests import QuestState, QuestEntry, QuestObjective, collect_quests
from ai_sidecar.state.market import MarketState, ShopItem, collect_market
from ai_sidecar.state.environment import EnvironmentState, collect_environment
from ai_sidecar.state.instances import InstanceState, InstanceEntry, collect_instances
from ai_sidecar.state.companions import CompanionState, collect_companions

__all__ = [
    # Top-level
    "GameState",
    "StateCollector",
    # Character
    "CharacterState",
    "collect_character",
    # Inventory
    "InventoryState",
    "InventoryItem",
    "EquipmentSlot",
    "collect_inventory",
    # Map
    "MapState",
    "PortalInfo",
    "MonsterSpawn",
    "collect_map",
    # Party
    "PartyState",
    "PartyMember",
    "collect_party",
    # Guild
    "GuildState",
    "GuildMember",
    "GuildSkill",
    "collect_guild",
    # Buffs
    "BuffState",
    "ActiveBuff",
    "collect_buffs",
    # Pets
    "PetState",
    "collect_pets",
    # Homunculus
    "HomunculusState",
    "HomunculusSkill",
    "collect_homunculus",
    # Mercenary
    "MercenaryState",
    "collect_mercenary",
    # Mount
    "MountState",
    "collect_mount",
    # Equipment
    "EquipmentState",
    "EquippedItem",
    "collect_equipment",
    # Dialogue
    "DialogueState",
    "DialogueOption",
    "collect_dialogue",
    # Quests
    "QuestState",
    "QuestEntry",
    "QuestObjective",
    "collect_quests",
    # Market
    "MarketState",
    "ShopItem",
    "collect_market",
    # Environment
    "EnvironmentState",
    "collect_environment",
    # Instances
    "InstanceState",
    "InstanceEntry",
    "collect_instances",
    # Companions
    "CompanionState",
    "collect_companions",
]
