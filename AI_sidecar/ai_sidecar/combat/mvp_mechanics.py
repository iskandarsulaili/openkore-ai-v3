"""
MVP Mechanics Knowledge — Ragnarok Online boss monster mechanics database.

Provides detailed knowledge about MVP (boss) monster mechanics including skills,
phases, elemental weaknesses, spawn info, and counter strategies.  Thread-safe
via RLock.  Falls back to knowledge.json for MVPs not in the curated list.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Final, Optional

# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MvpSkill:
    """A single skill an MVP can use."""

    name: str
    description: str
    hp_threshold: int = 100  # HP% below which this skill becomes active (100 = always)
    danger_level: int = 3  # 1 (low) – 5 (lethal)
    counter_strategy: str = ""


@dataclass
class MvpPhase:
    """A behaviour phase the MVP enters at a given HP threshold."""

    hp_threshold: int  # HP% below which this phase activates
    behavior_change: str
    warning: str = ""


@dataclass
class MvpMechanic:
    """Complete mechanical profile for a single MVP monster."""

    monster_id: int
    name: str
    level: int
    hp: int
    element: str
    size: str
    race: str
    skills: list[MvpSkill] = field(default_factory=list)
    phases: list[MvpPhase] = field(default_factory=list)
    dangerous_skills: list[str] = field(default_factory=list)
    recommended_element: str = "Neutral"
    recommended_weapon_type: str = ""
    min_party_size: int = 1
    spawn_map: str = ""
    spawn_timer_minutes: int = 0
    drops: list[str] = field(default_factory=list)
    strategy_summary: str = ""


# ──────────────────────────────────────────────────────────────────────
# Curated MVP data  (30+ major MVPs)
# ──────────────────────────────────────────────────────────────────────

# Each entry is a dict that can be passed as **kwargs to MvpMechanic.
# Skills and phases are expanded inline for readability.

_MVP_DATA: dict[int, dict[str, Any]] = {
    # ── Osiris ────────────────────────────────────────────────────────
    1038: {
        "monster_id": 1038,
        "name": "Osiris",
        "level": 68,
        "hp": 1_175_840,
        "element": "Undead",
        "size": "Medium",
        "race": "Undead",
        "skills": [
            MvpSkill("Meteor Storm", "Massive AoE fire damage around self", 100, 5,
                      "Spread out to avoid AoE; use Fire armor or Field Manual"),
            MvpSkill("Dark Breath", "Dark-element attack that can curse", 100, 4,
                      "Bring Panacea / Holy Water; equip Undead-element reduction gear"),
            MvpSkill("Power Up", "Increases ATK significantly", 30, 3,
                      "Dispel or wait it out; avoid melee during buff"),
            MvpSkill("Agility Up", "Increases ASPD and flee", 30, 2,
                      "Use high-hit-rate attacks; Dispel if available"),
            MvpSkill("Stone Curse", "Turns target to stone (petrify)", 100, 4,
                      "Bring Stone Curse remedy / Green Potion; keep distance"),
            MvpSkill("Summon Slave", "Calls Mummy and Verit adds", 100, 3,
                      "Kill adds quickly or AoE them down"),
            MvpSkill("Full Heal", "Heals self for large amount when damaged heavily", 100, 4,
                      "Burst damage to outpace healing; stop damage during heal"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks; avoid rude-attack skills"),
        ],
        "phases": [
            MvpPhase(50, "Starts using Power Up + Agi Up more frequently", "Buffs incoming!"),
            MvpPhase(30, "Full Heal becomes very likely; aggressive summon spam", "Heal phase — burn hard!"),
            MvpPhase(10, "Enraged — rapid skill spam, high damage output", "Final burst — kill or be killed!"),
        ],
        "dangerous_skills": ["Meteor Storm", "Stone Curse", "Full Heal", "Dark Breath"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Mace",
        "min_party_size": 4,
        "spawn_map": "egypt",
        "spawn_timer_minutes": 120,
        "drops": ["Osiris Card", "Turtle General Card", "Yggdrasilberry", "Osiris Doll", "Seed of Yggdrasil"],
        "strategy_summary": (
            "Osiris is an Undead 4 MVP that uses heavy AoE (Meteor Storm), curses, "
            "and summons adds.  Use Holy-element weapons (Mace) for 200% damage.  "
            "Bring Fire armor, Panacea, and Holy Water.  Spread out to avoid Meteor "
            "Storm overlap.  Burst through the Full Heal phase at 30% HP."
        ),
    },
    # ── Drake ────────────────────────────────────────────────────────
    1112: {
        "monster_id": 1112,
        "name": "Drake",
        "level": 91,
        "hp": 804_500,
        "element": "Undead",
        "size": "Medium",
        "race": "Undead",
        "skills": [
            MvpSkill("Water Ball", "High-damage single-target Water attack", 100, 4,
                      "Use Water-resist gear; stay at range"),
            MvpSkill("Guided Attack", "Homing attack that ignores flee", 100, 4,
                      "Use cover/LoD; tank with high DEF/VIT"),
            MvpSkill("Armor Break", "Destroys target's armor", 100, 3,
                      "Bring extra armor or repair kits"),
            MvpSkill("Maximize Power", "Doubles ATK for a period", 100, 3,
                      "Dispel or kite until it expires"),
            MvpSkill("Agility Up", "Increases flee and ASPD", 30, 2,
                      "Use high-hit attacks"),
            MvpSkill("Summon Slave", "Calls Zombie and Ghoul adds", 100, 2,
                      "AoE the adds quickly"),
            MvpSkill("Decrease Agility", "Slows target's ASPD and movement", 100, 2,
                      "Bring Agi-up potions or Dispel"),
        ],
        "phases": [
            MvpPhase(50, "Increases Water Ball and Guided Attack frequency", "Ranged pressure increases!"),
            MvpPhase(25, "Enrages — Maximize Power + Armor Break spam", "Beware the armor break!"),
        ],
        "dangerous_skills": ["Water Ball", "Guided Attack", "Armor Break"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Mace",
        "min_party_size": 3,
        "spawn_map": "treasure02",
        "spawn_timer_minutes": 120,
        "drops": ["Drake Card", "Violet Jewel", "White Potion", "Old Blue Box"],
        "strategy_summary": (
            "Drake is an Undead 1 MVP with powerful Water attacks and homing "
            "Guided Attack.  Holy-element weapons deal 200% damage.  Bring "
            "Water-resist gear and spare armor for Armor Break.  Stay spread "
            "to minimize Water Ball splash."
        ),
    },
    # ── Moonlight Flower ─────────────────────────────────────────────
    1150: {
        "monster_id": 1150,
        "name": "Moonlight Flower",
        "level": 79,
        "hp": 324_000,
        "element": "Fire",
        "size": "Medium",
        "race": "Demon",
        "skills": [
            MvpSkill("Fire Attack", "Fire-element damage on target", 100, 3,
                      "Use Fire-resist gear"),
            MvpSkill("Soul Strike", "Dark-element magic attack", 100, 3,
                      "Use Dark-resist gear or GTEF card"),
            MvpSkill("Teleport", "Teleports when attacked", 100, 2,
                      "Use stun-lock or high-DPS to prevent escape"),
            MvpSkill("Summon Slave", "Calls Miyabi Doll adds", 100, 2,
                      "AoE adds quickly"),
        ],
        "phases": [
            MvpPhase(40, "Teleports more frequently; becomes evasive", "Lock it down!"),
            MvpPhase(15, "Enrages — increased ATK and ASPD", "Finish quickly!"),
        ],
        "dangerous_skills": ["Soul Strike", "Fire Attack"],
        "recommended_element": "Water",
        "recommended_weapon_type": "Dagger",
        "min_party_size": 3,
        "spawn_map": "pay_d04_i",
        "spawn_timer_minutes": 120,
        "drops": ["Moonlight Flower Card", "Golden Jewel", "Fox Tail", "White Potion"],
        "strategy_summary": (
            "Moonlight Flower is a Fire 3 Demon MVP.  Use Water-element weapons "
            "for 175% damage.  She teleports frequently — use stun-lock or high "
            "burst DPS.  Bring Fire-resist gear and Dark-resist gear."
        ),
    },
    # ── Eddga ─────────────────────────────────────────────────────────
    1115: {
        "monster_id": 1115,
        "name": "Eddga",
        "level": 65,
        "hp": 947_500,
        "element": "Fire",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Fire Breath", "Cone AoE Fire damage", 100, 4,
                      "Stay behind or to the side; use Fire-resist"),
            MvpSkill("Stun Attack", "Stuns target on hit", 100, 3,
                      "Bring Stun-resist gear or Green Potion"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel or kite"),
            MvpSkill("Summon Slave", "Calls Savage adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Fire Breath becomes more frequent", "Watch the cone!"),
            MvpPhase(20, "Enrages — massive ATK increase", "Tank with high DEF/VIT"),
        ],
        "dangerous_skills": ["Fire Breath", "Stun Attack"],
        "recommended_element": "Water",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "moc_pryd06",
        "spawn_timer_minutes": 120,
        "drops": ["Eddga Card", "Flame Heart", "Tiger's Skin", "Tiger Footskin"],
        "strategy_summary": (
            "Eddga is a Fire 1 Brute MVP.  Water-element weapons deal 175% damage.  "
            "Avoid the Fire Breath cone by staying behind him.  Bring Stun-resist "
            "gear.  Large-size weapons (Spear, Two-Handed Sword) deal full damage."
        ),
    },
    # ── Maya ──────────────────────────────────────────────────────────
    1147: {
        "monster_id": 1147,
        "name": "Maya",
        "level": 55,
        "hp": 380_000,
        "element": "Earth",
        "size": "Large",
        "race": "Insect",
        "skills": [
            MvpSkill("Earthquake", "AoE Earth damage that stuns", 100, 5,
                      "Use Earth-resist gear; stay spread; bring Stun-resist"),
            MvpSkill("Stone Curse", "Petrifies target", 100, 4,
                      "Bring Stone Curse remedy / Green Potion"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel or avoid melee"),
            MvpSkill("Summon Slave", "Calls Hornet adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(50, "Earthquake frequency increases", "Spread out!"),
            MvpPhase(20, "Enrages — rapid skill spam", "Final push!"),
        ],
        "dangerous_skills": ["Earthquake", "Stone Curse"],
        "recommended_element": "Fire",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "anthell02",
        "spawn_timer_minutes": 120,
        "drops": ["Maya Card", "Crystal Jewel", "Old Violet Box", "Yggdrasilberry"],
        "strategy_summary": (
            "Maya is an Earth 4 Insect MVP.  Fire-element weapons deal 200% damage.  "
            "Earthquake is lethal — spread out and bring Stun-resist.  Large-size "
            "weapons recommended.  Stone Curse requires immediate curing."
        ),
    },
    # ── Phreeoni ─────────────────────────────────────────────────────
    1159: {
        "monster_id": 1159,
        "name": "Phreeoni",
        "level": 71,
        "hp": 300_000,
        "element": "Neutral",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Sight Blast", "Long-range magic attack", 100, 3,
                      "Use ranged attacks; stay at max range"),
            MvpSkill("Teleport", "Teleports when attacked", 100, 2,
                      "Use stun-lock or high burst"),
            MvpSkill("Summon Slave", "Calls adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Teleports more frequently", "Lock it down!"),
            MvpPhase(15, "Enrages — increased damage", "Finish quickly!"),
        ],
        "dangerous_skills": ["Sight Blast"],
        "recommended_element": "Neutral",
        "recommended_weapon_type": "Spear",
        "min_party_size": 2,
        "spawn_map": "moc_fild17",
        "spawn_timer_minutes": 120,
        "drops": ["Phreeoni Card", "Frozen Heart", "Star Crumb", "Crystal Jewel"],
        "strategy_summary": (
            "Phreeoni is a Neutral 3 Brute MVP.  No elemental weakness — use "
            "high-refine Neutral weapons or race/element cards.  Large-size "
            "weapons deal full damage.  Bring stun-lock to prevent teleporting."
        ),
    },
    # ── Mistress ──────────────────────────────────────────────────────
    1059: {
        "monster_id": 1059,
        "name": "Mistress",
        "level": 78,
        "hp": 378_000,
        "element": "Wind",
        "size": "Small",
        "race": "Insect",
        "skills": [
            MvpSkill("Thunder Storm", "AoE Wind damage", 100, 4,
                      "Use Wind-resist gear; spread out"),
            MvpSkill("Jupitel Thunder", "High-damage single-target Wind", 100, 4,
                      "Use Wind-resist gear; tank with high MDEF"),
            MvpSkill("Teleport", "Teleports when attacked", 100, 2,
                      "Use stun-lock"),
            MvpSkill("Summon Slave", "Calls adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Thunder Storm spam increases", "Spread out!"),
            MvpPhase(15, "Enrages — rapid casting", "Final burst!"),
        ],
        "dangerous_skills": ["Thunder Storm", "Jupitel Thunder"],
        "recommended_element": "Earth",
        "recommended_weapon_type": "Dagger",
        "min_party_size": 3,
        "spawn_map": "mjolnir_04",
        "spawn_timer_minutes": 120,
        "drops": ["Mistress Card", "Royal Jelly", "Scarlet Jewel", "Rough Wind"],
        "strategy_summary": (
            "Mistress is a Wind 4 Insect MVP.  Earth-element weapons deal 200% "
            "damage.  Small size means Daggers and Bows deal full damage.  "
            "Bring high MDEF and Wind-resist gear.  Spread out to avoid Thunder "
            "Storm overlap."
        ),
    },
    # ── Baphomet ──────────────────────────────────────────────────────
    1399: {
        "monster_id": 1399,
        "name": "Baphomet",
        "level": 68,
        "hp": 1_264_000,
        "element": "Dark",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Hell's Judgement", "Massive AoE Dark damage around self", 100, 5,
                      "Stay at range; use Ghost-element armor or Deviling card"),
            MvpSkill("Curse Attack", "Curses target on hit", 100, 3,
                      "Bring Panacea; use Curse-resist gear"),
            MvpSkill("Power Up", "Increases ATK significantly", 30, 4,
                      "Dispel immediately; avoid melee"),
            MvpSkill("Summon Slave", "Calls Incubus/Succubus adds", 100, 3,
                      "AoE adds quickly; they curse too"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks"),
        ],
        "phases": [
            MvpPhase(50, "Hell's Judgement frequency increases", "Stay spread!"),
            MvpPhase(30, "Power Up becomes more frequent", "Dispel ready!"),
            MvpPhase(10, "Enraged — constant skill spam", "Kill or be killed!"),
        ],
        "dangerous_skills": ["Hell's Judgement", "Power Up", "Curse Attack"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 5,
        "spawn_map": "gl_church02",
        "spawn_timer_minutes": 120,
        "drops": ["Baphomet Card", "Yggdrasilberry", "Evil Horn", "Baphomet Doll", "Royal Jelly"],
        "strategy_summary": (
            "Baphomet is a Dark 3 Demon MVP — one of the most dangerous classic "
            "MVPs.  Holy-element weapons deal 175% damage.  Hell's Judgement is "
            "lethal AoE — stay spread and use Ghost-element armor.  Bring Panacea "
            "for Curse.  Large-size weapons recommended.  Minimum 5 players."
        ),
    },
    # ── Orc Hero ─────────────────────────────────────────────────────
    1087: {
        "monster_id": 1087,
        "name": "Orc Hero",
        "level": 77,
        "hp": 585_700,
        "element": "Earth",
        "size": "Large",
        "race": "Demihuman",
        "skills": [
            MvpSkill("Bash", "Stuns target on hit", 100, 3,
                      "Use Stun-resist gear; tank with high VIT"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel or kite"),
            MvpSkill("Summon Slave", "Calls Orc Warrior adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Bash spam increases", "Stun-resist essential!"),
            MvpPhase(20, "Enrages — massive ATK increase", "Tank carefully!"),
        ],
        "dangerous_skills": ["Bash"],
        "recommended_element": "Fire",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "gef_fild14",
        "spawn_timer_minutes": 120,
        "drops": ["Orc Hero Card", "Red Jewel", "Steel", "Yggdrasilberry"],
        "strategy_summary": (
            "Orc Hero is an Earth 2 Demihuman MVP.  Fire-element weapons deal "
            "175% damage.  Bring Stun-resist gear for Bash.  Large-size weapons "
            "deal full damage.  Demihuman race cards (Hydra) work well here."
        ),
    },
    # ── Orc Lord ─────────────────────────────────────────────────────
    1190: {
        "monster_id": 1190,
        "name": "Orc Lord",
        "level": 74,
        "hp": 783_000,
        "element": "Earth",
        "size": "Large",
        "race": "Demihuman",
        "skills": [
            MvpSkill("Earthquake", "AoE Earth damage that stuns", 100, 5,
                      "Spread out; use Earth-resist; Stun-resist gear"),
            MvpSkill("Bash", "Stuns target on hit", 100, 3,
                      "Stun-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Orc Skeleton adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(50, "Earthquake becomes frequent", "Spread out NOW!"),
            MvpPhase(25, "Enrages — Earthquake + Bash spam", "Critical phase!"),
        ],
        "dangerous_skills": ["Earthquake", "Bash"],
        "recommended_element": "Fire",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "gef_fild10",
        "spawn_timer_minutes": 120,
        "drops": ["Orc Lord Card", "Voucher of Orcish Hero", "Old Violet Box", "Yggdrasilberry"],
        "strategy_summary": (
            "Orc Lord is an Earth 4 Demihuman MVP.  Fire-element weapons deal "
            "200% damage.  Earthquake is lethal AoE — spread out and bring "
            "Stun-resist.  Large-size weapons recommended.  Demihuman race cards "
            "are effective."
        ),
    },
    # ── Doppelganger ──────────────────────────────────────────────────
    1046: {
        "monster_id": 1046,
        "name": "Doppelganger",
        "level": 77,
        "hp": 380_000,
        "element": "Dark",
        "size": "Medium",
        "race": "Demon",
        "skills": [
            MvpSkill("Power Up", "Increases ATK significantly", 100, 4,
                      "Dispel immediately; avoid melee when buffed"),
            MvpSkill("Agility Up", "Increases ASPD and flee", 100, 3,
                      "Use high-hit-rate attacks"),
            MvpSkill("Full Strip", "Removes all equipment effects temporarily", 100, 4,
                      "Re-equip after strip; bring spare gear"),
            MvpSkill("Summon Slave", "Calls Doppelganger adds", 100, 3,
                      "Kill adds quickly"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks"),
        ],
        "phases": [
            MvpPhase(50, "Power Up + Full Strip spam increases", "Dispel ready!"),
            MvpPhase(25, "Enrages — constant buffing and stripping", "Burst through!"),
            MvpPhase(10, "Desperate — rapid skill spam", "Final push!"),
        ],
        "dangerous_skills": ["Full Strip", "Power Up"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Mace",
        "min_party_size": 3,
        "spawn_map": "gl_chyard",
        "spawn_timer_minutes": 120,
        "drops": ["Doppelganger Card", "Cardinal Jewel", "Blue Potion", "Yggdrasilberry"],
        "strategy_summary": (
            "Doppelganger is a Dark 3 Demon MVP.  Holy-element weapons deal 175% "
            "damage.  Full Strip removes all equipment — bring spare gear.  "
            "Dispel Power Up immediately.  Medium size means most weapons work."
        ),
    },
    # ── Golden Thief Bug ──────────────────────────────────────────────
    1088: {
        "monster_id": 1088,
        "name": "Golden Thief Bug",
        "level": 65,
        "hp": 222_750,
        "element": "Fire",
        "size": "Large",
        "race": "Insect",
        "skills": [
            MvpSkill("Fire Attack", "Fire-element damage", 100, 2,
                      "Fire-resist gear"),
            MvpSkill("Summon Slave", "Calls Thief Bug adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(30, "Summon spam increases", "AoE ready!"),
            MvpPhase(10, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": [],
        "recommended_element": "Water",
        "recommended_weapon_type": "Spear",
        "min_party_size": 2,
        "spawn_map": "prt_sewb4",
        "spawn_timer_minutes": 120,
        "drops": ["Golden Thief Bug Card", "Gold Ring", "Ora Ora", "Yggdrasilberry"],
        "strategy_summary": (
            "Golden Thief Bug is a Fire 2 Insect MVP.  Water-element weapons deal "
            "175% damage.  Relatively easy MVP — main challenge is the add spam.  "
            "Large-size weapons recommended."
        ),
    },
    # ── Stormy Knight ─────────────────────────────────────────────────
    1251: {
        "monster_id": 1251,
        "name": "Stormy Knight",
        "level": 92,
        "hp": 630_500,
        "element": "Wind",
        "size": "Large",
        "race": "Formless",
        "skills": [
            MvpSkill("Thunder Storm", "AoE Wind damage", 100, 4,
                      "Wind-resist gear; spread out"),
            MvpSkill("Jupitel Thunder", "High-damage single-target Wind", 100, 4,
                      "High MDEF tank"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Stormy Knight adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(50, "Thunder Storm spam increases", "Spread out!"),
            MvpPhase(25, "Enrages — rapid casting", "Burst through!"),
        ],
        "dangerous_skills": ["Thunder Storm", "Jupitel Thunder"],
        "recommended_element": "Earth",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "xmas_fild01",
        "spawn_timer_minutes": 120,
        "drops": ["Stormy Knight Card", "Skyblue Jewel", "Mistic Frozen", "Boots"],
        "strategy_summary": (
            "Stormy Knight is a Wind 4 Formless MVP.  Earth-element weapons deal "
            "200% damage.  Bring high MDEF and Wind-resist gear.  Spread out to "
            "avoid Thunder Storm.  Large-size weapons recommended."
        ),
    },
    # ── Dark Lord ────────────────────────────────────────────────────
    1272: {
        "monster_id": 1272,
        "name": "Dark Lord",
        "level": 96,
        "hp": 1_190_900,
        "element": "Undead",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Hell's Judgement", "Massive AoE Dark damage", 100, 5,
                      "Ghost-element armor; spread out"),
            MvpSkill("Dark Breath", "Dark-element attack that can curse", 100, 4,
                      "Panacea; Dark-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 4,
                      "Dispel immediately"),
            MvpSkill("Summon Slave", "Calls Banshee and Zombie adds", 100, 3,
                      "AoE adds quickly"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through the heal"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks"),
        ],
        "phases": [
            MvpPhase(60, "Hell's Judgement becomes frequent", "Stay spread!"),
            MvpPhase(35, "Full Heal + Power Up spam", "Burst through heal!"),
            MvpPhase(15, "Enraged — constant skill spam", "Final push!"),
        ],
        "dangerous_skills": ["Hell's Judgement", "Dark Breath", "Full Heal"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 5,
        "spawn_map": "gl_sew04",
        "spawn_timer_minutes": 120,
        "drops": ["Dark Lord Card", "Skull", "Blue Coif", "Yggdrasilberry", "Old Violet Box"],
        "strategy_summary": (
            "Dark Lord is an Undead 4 Demon MVP.  Holy-element weapons deal 200% "
            "damage.  Hell's Judgement is lethal AoE — use Ghost-element armor.  "
            "Bring Panacea for Curse.  Burst through Full Heal at 35% HP.  "
            "Minimum 5 players recommended."
        ),
    },
    # ── Lord of the Dead ──────────────────────────────────────────────
    1373: {
        "monster_id": 1373,
        "name": "Lord of the Dead",
        "level": 94,
        "hp": 603_883,
        "element": "Dark",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Dark Breath", "Dark-element attack", 100, 3,
                      "Dark-resist gear"),
            MvpSkill("Summon Slave", "Calls undead adds", 100, 3,
                      "AoE adds; they drain HP"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
        ],
        "phases": [
            MvpPhase(40, "Summon spam increases", "AoE ready!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Dark Breath"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "gl_dun02",
        "spawn_timer_minutes": 120,
        "drops": ["Lord of the Dead Card", "Crystal Jewel", "Yggdrasilberry", "Old Violet Box"],
        "strategy_summary": (
            "Lord of the Dead is a Dark 3 Demon MVP.  Holy-element weapons deal "
            "175% damage.  Main challenge is the add spam — bring strong AoE.  "
            "Large-size weapons recommended."
        ),
    },
    # ── Turtle General ───────────────────────────────────────────────
    1312: {
        "monster_id": 1312,
        "name": "Turtle General",
        "level": 110,
        "hp": 1_442_000,
        "element": "Earth",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Earthquake", "AoE Earth damage that stuns", 100, 5,
                      "Earth-resist; Stun-resist; spread out"),
            MvpSkill("Power Up", "Increases ATK", 30, 4,
                      "Dispel immediately"),
            MvpSkill("Agility Up", "Increases ASPD", 30, 3,
                      "High-hit-rate attacks"),
            MvpSkill("Summon Slave", "Calls Turtle adds", 100, 2,
                      "AoE adds"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
        ],
        "phases": [
            MvpPhase(60, "Earthquake becomes frequent", "Spread out!"),
            MvpPhase(35, "Full Heal + Power Up spam", "Burst through heal!"),
            MvpPhase(15, "Enraged — constant skill spam", "Final push!"),
        ],
        "dangerous_skills": ["Earthquake", "Full Heal", "Power Up"],
        "recommended_element": "Fire",
        "recommended_weapon_type": "Spear",
        "min_party_size": 5,
        "spawn_map": "tur_dun04",
        "spawn_timer_minutes": 120,
        "drops": ["Turtle General Card", "Turtle Shell", "Yggdrasilberry", "Old Violet Box"],
        "strategy_summary": (
            "Turtle General is an Earth 2 Brute MVP.  Fire-element weapons deal "
            "175% damage.  Earthquake is lethal — spread out and bring Stun-resist.  "
            "Burst through Full Heal at 35% HP.  Large-size weapons recommended.  "
            "Minimum 5 players."
        ),
    },
    # ── Amon Ra ──────────────────────────────────────────────────────
    1511: {
        "monster_id": 1511,
        "name": "Amon Ra",
        "level": 88,
        "hp": 1_214_138,
        "element": "Earth",
        "size": "Large",
        "race": "Demihuman",
        "skills": [
            MvpSkill("Earthquake", "AoE Earth damage that stuns", 100, 5,
                      "Earth-resist; Stun-resist; spread out"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Mummy adds", 100, 2,
                      "AoE adds"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
        ],
        "phases": [
            MvpPhase(50, "Earthquake frequency increases", "Spread out!"),
            MvpPhase(30, "Full Heal becomes likely", "Burst through!"),
            MvpPhase(15, "Enrages", "Final push!"),
        ],
        "dangerous_skills": ["Earthquake", "Full Heal"],
        "recommended_element": "Fire",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "in_sphinx05",
        "spawn_timer_minutes": 120,
        "drops": ["Amon Ra Card", "Seed of Yggdrasil", "Crystal Jewel", "Yggdrasilberry"],
        "strategy_summary": (
            "Amon Ra is an Earth 3 Demihuman MVP.  Fire-element weapons deal "
            "175% damage.  Earthquake is dangerous — spread out.  Burst through "
            "Full Heal.  Demihuman race cards (Hydra) are effective."
        ),
    },
    # ── Hatii ────────────────────────────────────────────────────────
    1252: {
        "monster_id": 1252,
        "name": "Hatii",
        "level": 98,
        "hp": 1_275_500,
        "element": "Water",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Water Ball", "High-damage single-target Water", 100, 4,
                      "Water-resist gear; high MDEF tank"),
            MvpSkill("Storm Gust", "AoE Water damage that freezes", 100, 5,
                      "Water-resist; Freeze-resist; spread out"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Snowier adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(50, "Storm Gust becomes frequent", "Freeze-resist essential!"),
            MvpPhase(25, "Enrages — Water Ball + Storm Gust spam", "Critical phase!"),
        ],
        "dangerous_skills": ["Storm Gust", "Water Ball"],
        "recommended_element": "Wind",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "xmas_dun02",
        "spawn_timer_minutes": 120,
        "drops": ["Hatii Card", "Fang of Garm", "Mistic Frozen", "Old Blue Box"],
        "strategy_summary": (
            "Hatii is a Water 4 Brute MVP.  Wind-element weapons deal 200% damage.  "
            "Storm Gust freezes — bring Freeze-resist gear.  Water-resist gear "
            "essential.  Large-size weapons recommended."
        ),
    },
    # ── Ifrit ─────────────────────────────────────────────────────────
    1832: {
        "monster_id": 1832,
        "name": "Ifrit",
        "level": 146,
        "hp": 6_935_000,
        "element": "Fire",
        "size": "Large",
        "race": "Formless",
        "skills": [
            MvpSkill("Fire Breath", "Cone AoE Fire damage", 100, 5,
                      "Stay behind; Fire-resist gear; high HP tank"),
            MvpSkill("Meteor Storm", "Massive AoE Fire damage", 90, 5,
                      "Spread out; Fire-resist; Ghost-element armor"),
            MvpSkill("Earthquake", "AoE Earth damage that stuns", 40, 5,
                      "Earth-resist; Stun-resist; spread out"),
            MvpSkill("Full Strip", "Removes all equipment effects", 100, 5,
                      "Re-equip; bring spare gear"),
            MvpSkill("Power Up", "Increases ATK significantly", 30, 4,
                      "Dispel immediately"),
            MvpSkill("Sonic Blow", "High-damage single-target attack", 100, 4,
                      "High DEF/VIT tank"),
            MvpSkill("Summon Slave", "Calls Ifrit adds", 100, 3,
                      "AoE adds quickly"),
            MvpSkill("Pulse Strike", "Massive AoE around self", 60, 5,
                      "Run out of range immediately!"),
        ],
        "phases": [
            MvpPhase(80, "Fire Breath + Meteor Storm spam begins", "Fire-resist essential!"),
            MvpPhase(60, "Pulse Strike becomes active — massive AoE", "Run from Pulse Strike!"),
            MvpPhase(40, "Earthquake + Full Strip spam", "Stun-resist + spare gear!"),
            MvpPhase(20, "Enraged — all skills on rapid cooldown", "Final desperate push!"),
        ],
        "dangerous_skills": ["Meteor Storm", "Fire Breath", "Pulse Strike", "Full Strip", "Earthquake"],
        "recommended_element": "Water",
        "recommended_weapon_type": "Spear",
        "min_party_size": 8,
        "spawn_map": "thor_v03",
        "spawn_timer_minutes": 120,
        "drops": ["Ifrit Card", "Carnium", "Old Violet Box", "Yggdrasilberry"],
        "strategy_summary": (
            "Ifrit is a Fire 4 Formless MVP — one of the hardest in the game.  "
            "Water-element weapons deal 200% damage.  Requires full party with "
            "Fire-resist, Earth-resist, Stun-resist, and spare gear for Full Strip.  "
            "Pulse Strike at 60% HP is a wipe mechanic — run out of range.  "
            "Minimum 8 players.  Dedicated tank with high HP and DEF essential."
        ),
    },
    # ── Ktullanux ─────────────────────────────────────────────────────
    1779: {
        "monster_id": 1779,
        "name": "Ktullanux",
        "level": 98,
        "hp": 4_417_000,
        "element": "Water",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Storm Gust", "AoE Water damage that freezes", 100, 5,
                      "Freeze-resist; Water-resist; spread out"),
            MvpSkill("Water Ball", "High-damage single-target Water", 100, 4,
                      "Water-resist gear; high MDEF tank"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Ice Titan adds", 100, 3,
                      "AoE adds; they also freeze"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
        ],
        "phases": [
            MvpPhase(60, "Storm Gust becomes frequent", "Freeze-resist essential!"),
            MvpPhase(35, "Full Heal + Water Ball spam", "Burst through heal!"),
            MvpPhase(15, "Enraged — constant skill spam", "Final push!"),
        ],
        "dangerous_skills": ["Storm Gust", "Water Ball", "Full Heal"],
        "recommended_element": "Wind",
        "recommended_weapon_type": "Spear",
        "min_party_size": 6,
        "spawn_map": "ice_dun03",
        "spawn_timer_minutes": 120,
        "drops": ["Ktullanux Card", "Yggdrasilberry", "Old Violet Box", "Mistic Frozen"],
        "strategy_summary": (
            "Ktullanux is a Water 4 Brute MVP.  Wind-element weapons deal 200% "
            "damage.  Storm Gust freezes — Freeze-resist is mandatory.  Burst "
            "through Full Heal at 35% HP.  Large-size weapons recommended.  "
            "Minimum 6 players."
        ),
    },
    # ── Gloom Under Night ────────────────────────────────────────────
    1768: {
        "monster_id": 1768,
        "name": "Gloom Under Night",
        "level": 139,
        "hp": 3_005_000,
        "element": "Ghost",
        "size": "Large",
        "race": "Formless",
        "skills": [
            MvpSkill("Ghost Attack", "Ghost-element damage", 100, 4,
                      "Use Neutral-element armor; Ghost-resist gear"),
            MvpSkill("Teleport", "Teleports frequently", 100, 3,
                      "Stun-lock or high burst DPS"),
            MvpSkill("Summon Slave", "Calls Ghost adds", 100, 3,
                      "AoE adds; they also use Ghost attacks"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
        ],
        "phases": [
            MvpPhase(50, "Teleport spam increases", "Lock it down!"),
            MvpPhase(25, "Enrages — increased damage output", "Finish quickly!"),
        ],
        "dangerous_skills": ["Ghost Attack"],
        "recommended_element": "Ghost",
        "recommended_weapon_type": "Spear",
        "min_party_size": 5,
        "spawn_map": "ra_fild01",
        "spawn_timer_minutes": 120,
        "drops": ["Gloom Under Night Card", "Yggdrasilberry", "Old Violet Box"],
        "strategy_summary": (
            "Gloom Under Night is a Ghost 3 Formless MVP.  Ghost-element weapons "
            "deal 175% damage.  Neutral-element armor is best against Ghost attacks.  "
            "Teleports frequently — use stun-lock.  Large-size weapons recommended."
        ),
    },
    # ── Beelzebub ─────────────────────────────────────────────────────
    1874: {
        "monster_id": 1874,
        "name": "Beelzebub",
        "level": 98,
        "hp": 6_666_666,
        "element": "Ghost",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Hell's Judgement", "Massive AoE Dark damage", 100, 5,
                      "Ghost-element armor; spread out"),
            MvpSkill("Ghost Attack", "Ghost-element damage", 100, 4,
                      "Neutral-element armor"),
            MvpSkill("Power Up", "Increases ATK significantly", 30, 4,
                      "Dispel immediately"),
            MvpSkill("Full Strip", "Removes all equipment effects", 100, 5,
                      "Spare gear; re-equip quickly"),
            MvpSkill("Summon Slave", "Calls Demon adds", 100, 3,
                      "AoE adds"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
        ],
        "phases": [
            MvpPhase(70, "Hell's Judgement becomes frequent", "Ghost armor essential!"),
            MvpPhase(45, "Full Strip + Full Heal spam", "Spare gear ready!"),
            MvpPhase(25, "Enraged — all skills on rapid cooldown", "Final push!"),
        ],
        "dangerous_skills": ["Hell's Judgement", "Full Strip", "Full Heal", "Ghost Attack"],
        "recommended_element": "Ghost",
        "recommended_weapon_type": "Spear",
        "min_party_size": 8,
        "spawn_map": "abyss_03",
        "spawn_timer_minutes": 120,
        "drops": ["Beelzebub Card", "Yggdrasilberry", "Old Violet Box", "Seed of Yggdrasil"],
        "strategy_summary": (
            "Beelzebub is a Ghost 4 Demon MVP — an extremely difficult endgame boss.  "
            "Ghost-element weapons deal 200% damage.  Hell's Judgement requires "
            "Ghost-element armor.  Full Strip + Full Heal at 45% is deadly.  "
            "Minimum 8 players with coordinated burst phases."
        ),
    },
    # ── Valkyrie Randgris ─────────────────────────────────────────────
    1751: {
        "monster_id": 1751,
        "name": "Valkyrie Randgris",
        "level": 141,
        "hp": 3_205_000,
        "element": "Holy",
        "size": "Large",
        "race": "Angel",
        "skills": [
            MvpSkill("Holy Light", "Holy-element magic attack", 100, 4,
                      "Use Dark-element armor; high MDEF"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Angel adds", 100, 3,
                      "AoE adds"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks"),
        ],
        "phases": [
            MvpPhase(50, "Holy Light spam increases", "Dark armor essential!"),
            MvpPhase(30, "Full Heal becomes likely", "Burst through!"),
            MvpPhase(15, "Enrages — increased damage", "Final push!"),
        ],
        "dangerous_skills": ["Holy Light", "Full Heal"],
        "recommended_element": "Dark",
        "recommended_weapon_type": "Spear",
        "min_party_size": 6,
        "spawn_map": "odin_tem03",
        "spawn_timer_minutes": 120,
        "drops": ["Valkyrie Randgris Card", "Old Violet Box", "Old Blue Box", "Old Card Album"],
        "strategy_summary": (
            "Valkyrie Randgris is a Holy 4 Angel MVP.  Dark-element weapons deal "
            "200% damage.  Use Dark-element armor to resist Holy Light.  Burst "
            "through Full Heal at 30% HP.  Large-size weapons recommended.  "
            "Minimum 6 players."
        ),
    },
    # ── Pharaoh ──────────────────────────────────────────────────────
    1157: {
        "monster_id": 1157,
        "name": "Pharaoh",
        "level": 85,
        "hp": 900_000,
        "element": "Dark",
        "size": "Large",
        "race": "Demihuman",
        "skills": [
            MvpSkill("Dark Attack", "Dark-element damage", 100, 3,
                      "Dark-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Mummy adds", 100, 2,
                      "AoE adds"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks"),
        ],
        "phases": [
            MvpPhase(40, "Summon spam increases", "AoE ready!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Dark Attack"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "in_sphinx04",
        "spawn_timer_minutes": 120,
        "drops": ["Pharaoh Card", "Royal Jelly", "Crystal Jewel", "Yggdrasilberry"],
        "strategy_summary": (
            "Pharaoh is a Dark 3 Demihuman MVP.  Holy-element weapons deal 175% "
            "damage.  Demihuman race cards (Hydra) are effective.  Large-size "
            "weapons recommended."
        ),
    },
    # ── Evil Snake Lord ───────────────────────────────────────────────
    1418: {
        "monster_id": 1418,
        "name": "Evil Snake Lord",
        "level": 105,
        "hp": 1_101_000,
        "element": "Ghost",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Ghost Attack", "Ghost-element damage", 100, 3,
                      "Neutral-element armor"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Snake adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Ghost Attack spam increases", "Neutral armor!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Ghost Attack"],
        "recommended_element": "Ghost",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "ama_dun03",
        "spawn_timer_minutes": 120,
        "drops": ["Evil Snake Lord Card", "Elunium", "Old Violet Box", "Yggdrasilberry"],
        "strategy_summary": (
            "Evil Snake Lord is a Ghost 3 Brute MVP.  Ghost-element weapons deal "
            "175% damage.  Use Neutral-element armor to resist Ghost attacks.  "
            "Large-size weapons recommended."
        ),
    },
    # ── Samurai Specter ───────────────────────────────────────────────
    1492: {
        "monster_id": 1492,
        "name": "Samurai Specter",
        "level": 100,
        "hp": 901_000,
        "element": "Dark",
        "size": "Large",
        "race": "Demihuman",
        "skills": [
            MvpSkill("Bash", "Stuns target on hit", 100, 3,
                      "Stun-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Ninja adds", 100, 2,
                      "AoE adds"),
            MvpSkill("Teleport", "Teleports when rude-attacked", 100, 2,
                      "Use normal attacks"),
        ],
        "phases": [
            MvpPhase(40, "Bash spam increases", "Stun-resist essential!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Bash"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "ama_dun02",
        "spawn_timer_minutes": 120,
        "drops": ["Samurai Specter Card", "Seed of Yggdrasil", "Elunium", "Yggdrasilberry"],
        "strategy_summary": (
            "Samurai Specter is a Dark 3 Demihuman MVP.  Holy-element weapons deal "
            "175% damage.  Bring Stun-resist for Bash.  Demihuman race cards "
            "(Hydra) are effective."
        ),
    },
    # ── Tao Gunka ────────────────────────────────────────────────────
    1583: {
        "monster_id": 1583,
        "name": "Tao Gunka",
        "level": 110,
        "hp": 1_252_000,
        "element": "Neutral",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Power Up", "Increases ATK significantly", 30, 4,
                      "Dispel immediately"),
            MvpSkill("Agility Up", "Increases ASPD", 30, 3,
                      "High-hit-rate attacks"),
            MvpSkill("Summon Slave", "Calls Demon adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(50, "Power Up becomes frequent", "Dispel ready!"),
            MvpPhase(25, "Enrages — massive ATK increase", "Tank carefully!"),
        ],
        "dangerous_skills": ["Power Up"],
        "recommended_element": "Neutral",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "abyss_01",
        "spawn_timer_minutes": 120,
        "drops": ["Tao Gunka Card", "Oridecon", "Old Violet Box", "Blue Potion"],
        "strategy_summary": (
            "Tao Gunka is a Neutral 3 Demon MVP.  No elemental weakness — use "
            "high-refine weapons with race/element cards.  Dispel Power Up "
            "immediately.  Large-size weapons recommended."
        ),
    },
    # ── RSX-0806 ──────────────────────────────────────────────────────
    1623: {
        "monster_id": 1623,
        "name": "RSX-0806",
        "level": 100,
        "hp": 1_001_000,
        "element": "Neutral",
        "size": "Large",
        "race": "Formless",
        "skills": [
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Alarm adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Summon spam increases", "AoE ready!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": [],
        "recommended_element": "Neutral",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "ein_dun02",
        "spawn_timer_minutes": 120,
        "drops": ["RSX-0806 Card", "Dark Blindfold", "Crystal Jewel", "Yggdrasilberry"],
        "strategy_summary": (
            "RSX-0806 is a Neutral 3 Formless MVP.  No elemental weakness — use "
            "high-refine weapons.  Relatively straightforward MVP.  Large-size "
            "weapons recommended."
        ),
    },
    # ── White Lady ────────────────────────────────────────────────────
    1630: {
        "monster_id": 1630,
        "name": "White Lady",
        "level": 97,
        "hp": 720_500,
        "element": "Wind",
        "size": "Large",
        "race": "Demihuman",
        "skills": [
            MvpSkill("Thunder Storm", "AoE Wind damage", 100, 4,
                      "Wind-resist gear; spread out"),
            MvpSkill("Jupitel Thunder", "High-damage single-target Wind", 100, 4,
                      "High MDEF tank"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Thunder Storm spam increases", "Spread out!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Thunder Storm", "Jupitel Thunder"],
        "recommended_element": "Earth",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "lhz_dun02",
        "spawn_timer_minutes": 120,
        "drops": ["White Lady Card", "Celestial Robe", "Old Violet Box", "Yggdrasilberry"],
        "strategy_summary": (
            "White Lady is a Wind 3 Demihuman MVP.  Earth-element weapons deal "
            "175% damage.  Bring high MDEF and Wind-resist gear.  Demihuman race "
            "cards (Hydra) are effective."
        ),
    },
    # ── Detardeurus ───────────────────────────────────────────────────
    1719: {
        "monster_id": 1719,
        "name": "Detardeurus",
        "level": 135,
        "hp": 6_005_000,
        "element": "Dark",
        "size": "Large",
        "race": "Dragon",
        "skills": [
            MvpSkill("Dark Breath", "Dark-element cone attack", 100, 5,
                      "Dark-resist gear; stay behind"),
            MvpSkill("Power Up", "Increases ATK significantly", 30, 4,
                      "Dispel immediately"),
            MvpSkill("Summon Slave", "Calls Dragon adds", 100, 3,
                      "AoE adds"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
        ],
        "phases": [
            MvpPhase(60, "Dark Breath becomes frequent", "Stay behind!"),
            MvpPhase(35, "Full Heal + Power Up spam", "Burst through heal!"),
            MvpPhase(15, "Enraged — constant skill spam", "Final push!"),
        ],
        "dangerous_skills": ["Dark Breath", "Full Heal"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 6,
        "spawn_map": "abyss_02",
        "spawn_timer_minutes": 120,
        "drops": ["Detardeurus Card", "Old Violet Box", "Crystal Jewel", "Yggdrasilberry"],
        "strategy_summary": (
            "Detardeurus is a Dark 3 Dragon MVP.  Holy-element weapons deal 175% "
            "damage.  Dark Breath is a cone — stay behind.  Burst through Full "
            "Heal at 35% HP.  Dragon race cards (Dragon Killer) are effective."
        ),
    },
    # ── Kiel D-01 ────────────────────────────────────────────────────
    1734: {
        "monster_id": 1734,
        "name": "Kiel D-01",
        "level": 125,
        "hp": 2_502_000,
        "element": "Dark",
        "size": "Medium",
        "race": "Formless",
        "skills": [
            MvpSkill("Dark Attack", "Dark-element damage", 100, 3,
                      "Dark-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Clock adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Summon spam increases", "AoE ready!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Dark Attack"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Mace",
        "min_party_size": 4,
        "spawn_map": "kiel_dun02",
        "spawn_timer_minutes": 120,
        "drops": ["Kiel D-01 Card", "Old Violet Box", "Old Card Album", "Yggdrasilberry"],
        "strategy_summary": (
            "Kiel D-01 is a Dark 2 Formless MVP.  Holy-element weapons deal 175% "
            "damage.  Medium size means most weapon types work.  Bring Dark-resist gear."
        ),
    },
    # ── Fallen Bishop Hibram ──────────────────────────────────────────
    1871: {
        "monster_id": 1871,
        "name": "Fallen Bishop Hibram",
        "level": 138,
        "hp": 5_655_000,
        "element": "Dark",
        "size": "Medium",
        "race": "Demon",
        "skills": [
            MvpSkill("Dark Attack", "Dark-element damage", 100, 3,
                      "Dark-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Demon adds", 100, 3,
                      "AoE adds"),
            MvpSkill("Full Heal", "Heals self for large amount", 100, 4,
                      "Burst through heal"),
        ],
        "phases": [
            MvpPhase(50, "Summon spam increases", "AoE ready!"),
            MvpPhase(30, "Full Heal becomes likely", "Burst through!"),
            MvpPhase(15, "Enrages", "Final push!"),
        ],
        "dangerous_skills": ["Full Heal", "Dark Attack"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Mace",
        "min_party_size": 5,
        "spawn_map": "ra_san05",
        "spawn_timer_minutes": 120,
        "drops": ["Fallen Bishop Hibram Card", "Seed of Yggdrasil", "Yggdrasilberry", "Old Violet Box"],
        "strategy_summary": (
            "Fallen Bishop Hibram is a Dark 2 Demon MVP.  Holy-element weapons "
            "deal 175% damage.  Burst through Full Heal at 30% HP.  Medium size "
            "means most weapon types work."
        ),
    },
    # ── Vesper ────────────────────────────────────────────────────────
    1685: {
        "monster_id": 1685,
        "name": "Vesper",
        "level": 128,
        "hp": 3_802_000,
        "element": "Holy",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Holy Light", "Holy-element magic attack", 100, 4,
                      "Dark-element armor; high MDEF"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Holy Light spam increases", "Dark armor!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Holy Light"],
        "recommended_element": "Dark",
        "recommended_weapon_type": "Spear",
        "min_party_size": 5,
        "spawn_map": "ra_fild04",
        "spawn_timer_minutes": 120,
        "drops": ["Vesper Card", "Old Violet Box", "Crystal Jewel", "Yggdrasilberry"],
        "strategy_summary": (
            "Vesper is a Holy 2 Brute MVP.  Dark-element weapons deal 175% damage.  "
            "Use Dark-element armor to resist Holy Light.  Large-size weapons "
            "recommended."
        ),
    },
    # ── Lady Tanee ───────────────────────────────────────────────────
    1688: {
        "monster_id": 1688,
        "name": "Lady Tanee",
        "level": 80,
        "hp": 360_000,
        "element": "Wind",
        "size": "Large",
        "race": "Plant",
        "skills": [
            MvpSkill("Thunder Storm", "AoE Wind damage", 100, 3,
                      "Wind-resist gear"),
            MvpSkill("Summon Slave", "Calls Plant adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(30, "Summon spam increases", "AoE ready!"),
            MvpPhase(10, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Thunder Storm"],
        "recommended_element": "Earth",
        "recommended_weapon_type": "Spear",
        "min_party_size": 2,
        "spawn_map": "lhz_fild03",
        "spawn_timer_minutes": 120,
        "drops": ["Lady Tanee Card", "Dex Dish10", "Crystal Jewel", "Old Violet Box"],
        "strategy_summary": (
            "Lady Tanee is a Wind 3 Plant MVP.  Earth-element weapons deal 175% "
            "damage.  Relatively easy MVP.  Plant race cards (Mandragora) are "
            "effective."
        ),
    },
    # ── Thanatos Phantom ───────────────────────────────────────────────
    1708: {
        "monster_id": 1708,
        "name": "Thanatos Phantom",
        "level": 99,
        "hp": 1_445_660,
        "element": "Ghost",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Ghost Attack", "Ghost-element damage", 100, 4,
                      "Neutral-element armor"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Ghost adds", 100, 3,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(50, "Ghost Attack spam increases", "Neutral armor!"),
            MvpPhase(25, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Ghost Attack"],
        "recommended_element": "Ghost",
        "recommended_weapon_type": "Spear",
        "min_party_size": 4,
        "spawn_map": "tha_t10",
        "spawn_timer_minutes": 120,
        "drops": ["Thanatos Card", "Old Blue Box", "Crystal Jewel", "Yggdrasilberry"],
        "strategy_summary": (
            "Thanatos Phantom is a Ghost 4 Demon MVP.  Ghost-element weapons deal "
            "200% damage.  Use Neutral-element armor.  Large-size weapons "
            "recommended."
        ),
    },
    # ── Dracula ───────────────────────────────────────────────────────
    1389: {
        "monster_id": 1389,
        "name": "Dracula",
        "level": 75,
        "hp": 350_000,
        "element": "Dark",
        "size": "Large",
        "race": "Demon",
        "skills": [
            MvpSkill("Dark Attack", "Dark-element damage", 100, 3,
                      "Dark-resist gear"),
            MvpSkill("Power Up", "Increases ATK", 30, 3,
                      "Dispel"),
            MvpSkill("Summon Slave", "Calls Bat adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(40, "Summon spam increases", "AoE ready!"),
            MvpPhase(20, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Dark Attack"],
        "recommended_element": "Holy",
        "recommended_weapon_type": "Spear",
        "min_party_size": 3,
        "spawn_map": "gef_dun00",
        "spawn_timer_minutes": 120,
        "drops": ["Dracula Card", "Crystal Jewel", "Fruit of Mastela", "Yggdrasilberry"],
        "strategy_summary": (
            "Dracula is a Dark 4 Demon MVP.  Holy-element weapons deal 200% damage.  "
            "Bring Dark-resist gear.  Large-size weapons recommended."
        ),
    },
    # ── Garm (Hatii is the MVP name; Garm is the mini-boss variant) ──
    1313: {
        "monster_id": 1313,
        "name": "Garm",
        "level": 85,
        "hp": 450_000,
        "element": "Water",
        "size": "Large",
        "race": "Brute",
        "skills": [
            MvpSkill("Water Ball", "Water-element damage", 100, 3,
                      "Water-resist gear"),
            MvpSkill("Summon Slave", "Calls Snowier adds", 100, 2,
                      "AoE adds"),
        ],
        "phases": [
            MvpPhase(30, "Summon spam increases", "AoE ready!"),
            MvpPhase(10, "Enrages", "Finish quickly!"),
        ],
        "dangerous_skills": ["Water Ball"],
        "recommended_element": "Wind",
        "recommended_weapon_type": "Spear",
        "min_party_size": 2,
        "spawn_map": "xmas_dun01",
        "spawn_timer_minutes": 60,
        "drops": ["Garm Card", "Mistic Frozen", "Old Blue Box"],
        "strategy_summary": (
            "Garm is a Water-element mini-boss.  Wind-element weapons deal 175% "
            "damage.  Relatively easy — bring Water-resist gear."
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────
# Knowledge JSON fallback loader
# ──────────────────────────────────────────────────────────────────────

_KNOWLEDGE_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "knowledge", "knowledge.json"
)

_MONSTER_LOOKUP: dict[int, dict[str, Any]] = {}
_LOADED: bool = False
_LOAD_LOCK: threading.Lock = threading.Lock()


def _ensure_knowledge_loaded() -> None:
    """Load monster data from knowledge.json (once, thread-safe)."""
    global _MONSTER_LOOKUP, _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        path = os.path.abspath(_KNOWLEDGE_PATH)
        if not os.path.isfile(path):
            _LOADED = True
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            _LOADED = True
            return
        monsters = data.get("monsters", [])
        for m in monsters:
            mid = m.get("Id")
            if mid is not None:
                # Keep the highest-HP entry per ID (preferring the "real" version)
                if mid in _MONSTER_LOOKUP:
                    existing = _MONSTER_LOOKUP[mid]
                    if (m.get("Hp", 0) or 0) > (existing.get("Hp", 0) or 0):
                        _MONSTER_LOOKUP[mid] = m
                else:
                    _MONSTER_LOOKUP[mid] = m
        _LOADED = True


def _get_monster_from_knowledge(monster_id: int) -> Optional[dict[str, Any]]:
    """Look up a monster's basic stats from knowledge.json."""
    _ensure_knowledge_loaded()
    return _MONSTER_LOOKUP.get(monster_id)


# ──────────────────────────────────────────────────────────────────────
# MvpMechanicsDatabase  (thread-safe singleton)
# ──────────────────────────────────────────────────────────────────────


class MvpMechanicsDatabase:
    """Thread-safe database of MVP mechanics with knowledge.json fallback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_id: dict[int, MvpMechanic] = {}
        self._by_name: dict[str, MvpMechanic] = {}
        self._by_map: dict[str, list[MvpMechanic]] = {}
        self._initialized = False

    # ── Initialisation ────────────────────────────────────────────────

    def _build(self) -> None:
        """Build internal indexes from the curated MVP data."""
        for mid, data in _MVP_DATA.items():
            mech = MvpMechanic(**data)
            self._by_id[mid] = mech
            self._by_name[mech.name.lower()] = mech
            if mech.spawn_map:
                self._by_map.setdefault(mech.spawn_map, []).append(mech)
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._build()

    # ── Derive basic info for unknown MVPs ────────────────────────────

    def _derive_from_knowledge(self, monster_id: int) -> Optional[MvpMechanic]:
        """Create a basic MvpMechanic from knowledge.json for an unknown MVP."""
        raw = _get_monster_from_knowledge(monster_id)
        if raw is None:
            return None
        name = raw.get("Name", f"Unknown MVP {monster_id}")
        drops_raw = raw.get("MvpDrops", []) or raw.get("Drops", [])
        drops: list[str] = []
        for d in drops_raw:
            if isinstance(d, dict):
                item_name = d.get("Item", "")
                if item_name:
                    drops.append(item_name)
            elif isinstance(d, str):
                drops.append(d)

        return MvpMechanic(
            monster_id=monster_id,
            name=name,
            level=raw.get("Level", 1),
            hp=raw.get("Hp", 0),
            element=raw.get("Element", "Neutral"),
            size=raw.get("Size", "Medium"),
            race=raw.get("Race", "Formless"),
            drops=drops,
            strategy_summary=(
                f"{name} is a {raw.get('Element', 'Neutral')} {raw.get('ElementLevel', 1)} "
                f"{raw.get('Race', 'Formless')} MVP.  No curated strategy data available — "
                f"use elemental matrix for damage optimization."
            ),
        )

    # ── Public query methods ─────────────────────────────────────────

    def get_mvp_mechanics(
        self, monster_id_or_name: int | str
    ) -> Optional[MvpMechanic]:
        """Look up an MVP by numeric ID or name string."""
        self._ensure_initialized()
        with self._lock:
            if isinstance(monster_id_or_name, int):
                mech = self._by_id.get(monster_id_or_name)
                if mech is not None:
                    return mech
                return self._derive_from_knowledge(monster_id_or_name)
            else:
                key = monster_id_or_name.lower().strip()
                mech = self._by_name.get(key)
                if mech is not None:
                    return mech
                # Try partial name match
                for name, m in self._by_name.items():
                    if key in name or name in key:
                        return m
                return None

    def get_dangerous_skills(self, monster_id: int) -> list[str]:
        """Return the list of dangerous skill names for an MVP."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return []
        return list(mech.dangerous_skills)

    def get_counter_strategy(
        self, monster_id: int, current_hp_pct: float
    ) -> str:
        """Return a strategy hint based on the MVP's current HP percentage."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return "No strategy data available for this MVP."

        with self._lock:
            # Check phases in descending HP threshold order
            active_phases: list[MvpPhase] = []
            for phase in sorted(mech.phases, key=lambda p: p.hp_threshold, reverse=True):
                if current_hp_pct <= phase.hp_threshold:
                    active_phases.append(phase)

            # Check which dangerous skills are active based on HP thresholds
            active_skills: list[str] = []
            for skill in mech.skills:
                if current_hp_pct <= skill.hp_threshold:
                    active_skills.append(skill.name)

            parts: list[str] = []
            if active_phases:
                phase_warnings = [p.warning for p in active_phases if p.warning]
                if phase_warnings:
                    parts.append("⚠ " + " | ".join(phase_warnings))
                behavior = [p.behavior_change for p in active_phases if p.behavior_change]
                if behavior:
                    parts.append("Behavior: " + "; ".join(behavior))

            if active_skills:
                parts.append(f"Active dangerous skills: {', '.join(active_skills)}")

            if not parts:
                parts.append("Standard phase — follow general strategy.")

            parts.append(f"Recommended element: {mech.recommended_element}")
            parts.append(f"Min party size: {mech.min_party_size}")

            return " | ".join(parts)

    def get_recommended_element(self, monster_id: int) -> str:
        """Return the recommended attack element for an MVP."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return "Neutral"
        return mech.recommended_element

    def get_recommended_party_size(self, monster_id: int) -> int:
        """Return the minimum recommended party size for an MVP."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return 1
        return mech.min_party_size

    def is_dangerous_phase(
        self, monster_id: int, current_hp_pct: float
    ) -> bool:
        """Check if the MVP is in a dangerous phase at the given HP %."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return False
        with self._lock:
            for phase in mech.phases:
                if current_hp_pct <= phase.hp_threshold:
                    return True
            return False

    def get_mvp_spawn_info(self, monster_id: int) -> dict[str, Any]:
        """Return spawn map and timer info for an MVP."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return {"spawn_map": "Unknown", "spawn_timer_minutes": 0}
        return {
            "spawn_map": mech.spawn_map,
            "spawn_timer_minutes": mech.spawn_timer_minutes,
        }

    def get_all_mvps(self) -> list[MvpMechanic]:
        """Return all curated MVPs."""
        self._ensure_initialized()
        with self._lock:
            return list(self._by_id.values())

    def get_mvps_on_map(self, map_name: str) -> list[MvpMechanic]:
        """Return all MVPs that spawn on a given map."""
        self._ensure_initialized()
        with self._lock:
            return list(self._by_map.get(map_name, []))

    def get_mvp_strategy_summary(self, monster_id: int) -> str:
        """Return the full strategy summary for an MVP."""
        mech = self.get_mvp_mechanics(monster_id)
        if mech is None:
            return "No strategy data available for this MVP."
        return mech.strategy_summary


# ──────────────────────────────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────────────────────────────

_INSTANCE: Optional[MvpMechanicsDatabase] = None
_INSTANCE_LOCK: threading.Lock = threading.Lock()


def get_mvp_mechanics_db() -> MvpMechanicsDatabase:
    """Return the global MvpMechanicsDatabase singleton (thread-safe)."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = MvpMechanicsDatabase()
    return _INSTANCE
