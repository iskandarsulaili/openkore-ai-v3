"""
RO damage formulas — pre-renewal (classic) accurate.

Covers:
- Hard DEF (VIT-based) vs Soft DEF (equipment-based)
- MDEF formula: (MDEF * 0.5 + MDEF * 0.5 * INT/100) with hard cap
- Race/Element/Size modifiers
- Level-based damage penalty
- Refinement bonus
- Card slot effects
- ASPD-based attack interval
- Cast time (variable + fixed, modified by DEX)
- Skill delay (modified by DEX)
"""

import datetime
import math
from dataclasses import dataclass, field
from typing import Any

# ── Element chart (attacker row vs defender col) ──
# Rows: attacker element (Neutral, Water, Earth, Fire, Wind, Poison, Holy, Dark, Ghost, Undead)
# Cols: defender element (same order)
# Values: damage multiplier (1.0 = 100%)
ELEMENT_CHART: list[list[float]] = [
    # Neutral Water Earth Fire Wind Poison Holy Dark Ghost Undead
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.75, 0.75],  # Neutral
    [1.0,  0.25, 0.75, 1.5,  0.75, 1.0,  1.0,  1.0,  1.0,  1.0 ],  # Water
    [1.0,  1.5,  0.25, 0.75, 0.75, 1.0,  1.0,  1.0,  1.0,  1.0 ],  # Earth
    [1.0,  0.75, 1.5,  0.25, 0.75, 1.0,  1.0,  1.0,  1.0,  1.0 ],  # Fire
    [1.0,  0.75, 0.75, 1.5,  0.25, 1.0,  1.0,  1.0,  1.0,  1.0 ],  # Wind
    [1.0,  1.0,  0.75, 1.0,  1.0,  0.25, 1.0,  1.0,  1.0,  1.0 ],  # Poison
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.25, 1.0,  1.25],  # Holy
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.25, 1.0,  1.0,  1.25],  # Dark
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.75, 1.0 ],  # Ghost
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0 ],  # Undead
]

# ── Size chart (weapon type vs monster size) ──
# Rows: weapon type (Dagger, 1H Sword, 2H Sword, Spear, Mace, Axe, Staff, Bow, Knuckle, Instrument, Whip, Katar)
# Cols: monster size (Small, Medium, Large)
SIZE_CHART: dict[str, list[float]] = {
    "Dagger":       [1.0, 0.75, 0.5],
    "1H_Sword":     [0.75, 1.0, 0.75],
    "2H_Sword":     [0.75, 0.75, 1.0],
    "Spear":        [0.75, 0.75, 1.0],
    "Mace":         [0.75, 1.0, 0.75],
    "Axe":          [0.5, 0.75, 1.0],
    "Staff":        [1.0, 1.0, 1.0],
    "Bow":          [1.0, 1.0, 1.0],
    "Knuckle":      [1.0, 0.75, 0.5],
    "Instrument":   [0.75, 1.0, 0.75],
    "Whip":         [0.75, 1.0, 0.75],
    "Katar":        [1.0, 0.75, 0.5],
}

# ── Race chart (attacker weapon vs defender race) ──
# Races: Angel, Brute, DemiHuman, Demon, Dragon, Fish, Formless, Insect, Plant, Undead
RACE_CHART: dict[str, dict[str, float]] = {
    "Dagger":       {"DemiHuman": 1.0, "Brute": 1.0, "Dragon": 0.75, "Undead": 0.75, "Formless": 0.75},
    "Mace":         {"Undead": 1.5, "DemiHuman": 1.0, "Dragon": 0.75, "Formless": 0.75},
    "Bow":          {"Brute": 1.0, "DemiHuman": 1.0, "Dragon": 0.75, "Formless": 0.75},
}

# ── Race-to-element mapping for common monsters ──
RACE_ELEMENTS: dict[str, str] = {
    "Poring": "Water", "Drops": "Water", "Poporing": "Poison",
    "Lunatic": "Neutral", "Picky": "Fire", "Picky_": "Fire",
    "Fabre": "Earth", "Chonchon": "Wind", "Hornet": "Wind",
    "Thief_Bug": "Earth", "Thief_Bug_Egg": "Neutral",
    "Savage_Babe": "Earth", "Savage": "Earth",
    "Familiar": "Dark", "Orc_Warrior": "Earth", "Orc_Archer": "Earth",
    "Orc_Skeleton": "Undead", "Orc_Zombie": "Undead",
    "Skeleton": "Undead", "Skeleton_Archer": "Undead",
    "Zombie": "Undead", "Ghoul": "Undead",
    "Mummy": "Undead", "Mummy_": "Undead",
    "Wolf": "Neutral", "High_Orc": "Earth",
    "Anacondaq": "Poison", "Snake": "Poison",
    "Spore": "Water", "Mushroom": "Earth",
    "Elder_Willow": "Earth", "Willow": "Earth",
    "Creamy": "Wind", "Dustiness": "Wind",
    "Metaller": "Neutral", "Plankton": "Water",
    "Marina": "Water", "Kukre": "Water",
    "Vadon": "Water", "Hydra": "Water",
    "Pirate_Skeleton": "Undead", "Coco": "Earth",
    "Deniro": "Earth", "Peco_Peco": "Neutral",
    "Peco_Peco_Egg": "Neutral", "Smokie": "Earth",
    "Yoyo": "Earth", "Steel_Chonchon": "Wind",
    "Hunter_Fly": "Wind", "Mantis": "Earth",
    "Muka": "Earth", "Rocker": "Wind",
    "Stainer": "Poison", "Worm_Tail": "Earth",
    "Scorpion": "Fire", "Swordfish": "Water",
    "Caramel": "Earth", "Savage_Bebe": "Earth",
    "Golem": "Neutral", "Beetle_King": "Earth",
    "Myst_Case": "Ghost", "Nightmare": "Ghost",
    "Punk": "Wind", "Bongun": "Undead",
    "Munak": "Undead", "Nine_Tail": "Fire",
    "Sohee": "Water", "Dokebi": "Fire",
    "Deviruchi": "Dark", "Isis": "Dark",
    "Petite": "Wind", "Petite_": "Wind",
    "Gargoyle": "Wind", "Rideword": "Ghost",
    "Neraid": "Water", "Phendark": "Ghost",
    "Strouf": "Water", "Kraken": "Water",
    "Maya": "Earth", "Maya_Pupa": "Earth",
    "Phreeoni": "Neutral", "Moonlight": "Fire",
    "Osiris": "Undead", "Baphomet": "Dark",
    "Dracula": "Dark", "Doppelganger": "Dark",
    "Golden_Thief_Bug": "Earth", "Eddga": "Fire",
    "Orc_Hero": "Earth", "Orc_Lord": "Earth",
    "Mistress": "Wind", "Stormy_Knight": "Wind",
    "Lord_of_Death": "Dark", "Turtle_General": "Water",
    "Atroce": "Brute", "Kiel": "DemiHuman",
    "Valkyrie": "Holy", "Randgris": "Holy",
    "Gloom": "Dark", "Ktullanux": "Water",
    "Ifrit": "Fire", "Beelzebub": "Dark",
    "Tao_Gunka": "Neutral", "RSX_0806": "Neutral",
    "Detard": "Dark", "Egnigem": "Fire",
    "Biolab_Monster": "DemiHuman",
}

# ── Weapon type mapping ──
WEAPON_TYPES: dict[str, str] = {
    "Dagger": "Dagger", "Knife": "Dagger", "Main_Gauche": "Dagger",
    "Sword": "1H_Sword", "Blade": "1H_Sword", "Edge": "1H_Sword",
    "Two_Handed_Sword": "2H_Sword", "Broadsword": "2H_Sword", "Bastard_Sword": "2H_Sword",
    "Spear": "Spear", "Lance": "Spear", "Pike": "Spear",
    "Mace": "Mace", "Morning_Star": "Mace", "Flail": "Mace",
    "Axe": "Axe", "Battle_Axe": "Axe", "Great_Axe": "Axe",
    "Staff": "Staff", "Rod": "Staff", "Wand": "Staff",
    "Bow": "Bow", "Crossbow": "Bow", "Composite_Bow": "Bow",
    "Knuckle": "Knuckle", "Fist": "Knuckle",
    "Instrument": "Instrument", "Guitar": "Instrument", "Harp": "Instrument",
    "Whip": "Whip", "Rope": "Whip", "Wire": "Whip",
    "Katar": "Katar", "Claw": "Katar", "Fist_Blade": "Katar",
}

# ── Skill data: cast time, delay, cooldown, range ──
# Format: (variable_cast_ms, fixed_cast_ms, after_cast_delay_ms, cooldown_ms, range_cells)
# Source: rAthena skill_db
SKILL_DATA: dict[str, tuple[int, int, int, int, int]] = {
    # ── Archer ──
    "Double Strafe":    (0, 0, 100, 0, 9),
    "Arrow Shower":     (0, 0, 500, 0, 9),
    "Improve Concentration": (0, 0, 0, 0, 0),
    "Owl's Eye":        (0, 0, 0, 0, 0),
    "Vulture's Eye":    (0, 0, 0, 0, 0),
    "Anklesnare":       (500, 0, 1000, 0, 9),
    "Blitz Beat":       (0, 0, 0, 0, 9),
    "Detecting":        (0, 0, 0, 0, 0),
    "Falconry Mastery": (0, 0, 0, 0, 0),
    "Steel Crow":       (0, 0, 0, 0, 9),
    # ── Mage ──
    "Fire Bolt":        (3200, 0, 1000, 0, 9),
    "Cold Bolt":        (3200, 0, 1000, 0, 9),
    "Lightning Bolt":   (3200, 0, 1000, 0, 9),
    "Earth Spike":      (3200, 0, 1000, 0, 9),
    "Fire Ball":        (3500, 0, 1500, 0, 9),
    "Fire Wall":        (2000, 0, 1000, 0, 9),
    "Frost Diver":      (1500, 0, 1000, 0, 9),
    "Frost Nova":       (3000, 0, 2000, 0, 9),
    "Soul Strike":      (2000, 0, 1000, 0, 9),
    "Napalm Beat":      (1000, 0, 500, 0, 9),
    "Safety Wall":      (2000, 0, 1000, 0, 9),
    "Stone Curse":      (2000, 0, 1000, 0, 9),
    "Energy Coat":      (0, 0, 0, 0, 0),
    # ── Wizard ──
    "Storm Gust":       (5000, 5000, 5000, 0, 9),
    "Meteor Storm":     (6000, 3000, 5000, 0, 9),
    "Lord of Vermilion":(4000, 2000, 3000, 0, 9),
    "Heaven's Drive":   (4000, 2000, 3000, 0, 9),
    "Quagmire":         (2000, 0, 1000, 0, 9),
    "Ice Wall":         (2000, 0, 1000, 0, 9),
    "Water Ball":       (3000, 0, 2000, 0, 9),
    "Sight":            (0, 0, 0, 0, 0),
    # ── Acolyte / Priest ──
    "Heal":             (1000, 0, 500, 0, 9),
    "Blessing":         (2000, 0, 1000, 0, 9),
    "Increase Agility": (2000, 0, 1000, 0, 9),
    "Teleport":         (0, 0, 0, 0, 0),
    "Warp Portal":      (3000, 0, 2000, 0, 9),
    "Holy Light":       (1000, 0, 500, 0, 9),
    "Turn Undead":      (1000, 0, 1000, 0, 9),
    "Magnificat":       (2000, 0, 1000, 0, 0),
    "Impositio Manus":  (2000, 0, 1000, 0, 9),
    "Aspersio":         (2000, 0, 1000, 0, 9),
    "Kyrie Eleison":    (2000, 0, 1000, 0, 9),
    "Gloria":           (2000, 0, 1000, 0, 0),
    "Lex Aeterna":      (1000, 0, 500, 0, 9),
    "Lex Divina":       (1000, 0, 500, 0, 9),
    # ── Thief / Assassin ──
    "Double Attack":    (0, 0, 0, 0, 1),
    "Sonic Blow":       (0, 0, 1000, 0, 1),
    "Grimtooth":        (0, 0, 500, 0, 9),
    "Hide":             (0, 0, 0, 0, 0),
    "Cloaking":         (0, 0, 0, 0, 0),
    "Envenom":          (0, 0, 500, 0, 1),
    "Poison React":     (0, 0, 0, 0, 0),
    "Katar Mastery":    (0, 0, 0, 0, 0),
    "Right Hand Mastery": (0, 0, 0, 0, 0),
    "Left Hand Mastery": (0, 0, 0, 0, 0),
    "Soul Breaker":     (0, 0, 1000, 0, 9),
    "Meteor Assault":   (0, 0, 1000, 0, 1),
    # ── Swordsman / Knight ──
    "Bash":             (0, 0, 500, 0, 1),
    "Magnum Break":     (0, 0, 1000, 0, 1),
    "Endure":           (0, 0, 0, 0, 0),
    "Provoke":          (0, 0, 500, 0, 9),
    "Two-Handed Sword Mastery": (0, 0, 0, 0, 0),
    "Spear Mastery":    (0, 0, 0, 0, 0),
    "Spear Boomerang":  (0, 0, 500, 0, 9),
    "Pierce":           (0, 0, 500, 0, 1),
    "Brandish Spear":   (0, 0, 1000, 0, 1),
    "Bowling Bash":     (0, 0, 1000, 0, 1),
    "Riding":           (0, 0, 0, 0, 0),
    "Cavalry Mastery":  (0, 0, 0, 0, 0),
    # ── Merchant / Blacksmith ──
    "Mammonite":        (0, 0, 1000, 0, 1),
    "Cart Revolution":  (0, 0, 1000, 0, 1),
    "Change Cart":      (0, 0, 0, 0, 0),
    "Crazy Uproar":     (0, 0, 0, 0, 0),
    "Weaponry Research": (0, 0, 0, 0, 0),
    "Adrenaline Rush":  (0, 0, 0, 0, 0),
    "Weapon Perfection": (0, 0, 0, 0, 0),
    "Over Thrust":      (0, 0, 0, 0, 0),
    "Maximum Over Thrust": (0, 0, 0, 0, 0),
    # ── Acolyte / Monk ──
    "Iron Fists":       (0, 0, 0, 0, 0),
    "Flee":             (0, 0, 0, 0, 0),
    "Spirit Recovery":  (0, 0, 0, 0, 0),
    "Call Spirits":     (0, 0, 0, 0, 0),
    "Asura Strike":     (0, 0, 2000, 0, 1),
    "Finger Offensive": (0, 0, 500, 0, 9),
    "Triple Attack":    (0, 0, 0, 0, 1),
    "Guillotine Fist":  (0, 0, 2000, 0, 1),
    "Raging Trifecta":  (0, 0, 1000, 0, 1),
    "Chain Combo":      (0, 0, 0, 0, 1),
    # ── General ──
    "Basic Attack":     (0, 0, 0, 0, 1),
}

# ── Skill element mapping ──
SKILL_ELEMENTS: dict[str, str] = {
    "Fire Bolt": "Fire", "Fire Ball": "Fire", "Fire Wall": "Fire",
    "Meteor Storm": "Fire", "Mammonite": "Neutral",
    "Cold Bolt": "Water", "Frost Diver": "Water", "Frost Nova": "Water",
    "Storm Gust": "Water", "Water Ball": "Water",
    "Lightning Bolt": "Wind", "Lord of Vermilion": "Wind",
    "Earth Spike": "Earth", "Heaven's Drive": "Earth", "Quagmire": "Earth",
    "Soul Strike": "Ghost", "Napalm Beat": "Ghost",
    "Holy Light": "Holy", "Turn Undead": "Holy",
    "Bash": "Neutral", "Magnum Break": "Fire",
    "Sonic Blow": "Neutral", "Grimtooth": "Neutral",
    "Double Strafe": "Neutral", "Arrow Shower": "Neutral",
    "Heal": "Holy", "Bowling Bash": "Neutral",
    "Pierce": "Neutral", "Spear Boomerang": "Neutral",
    "Brandish Spear": "Neutral",
    "Asura Strike": "Neutral", "Finger Offensive": "Neutral",
    "Triple Attack": "Neutral",
    "Basic Attack": "Neutral",
}

# ── Monster size mapping ──
MONSTER_SIZES: dict[str, str] = {
    "Poring": "Small", "Drops": "Small", "Poporing": "Small",
    "Lunatic": "Small", "Picky": "Small", "Picky_": "Small",
    "Fabre": "Small", "Chonchon": "Small", "Hornet": "Small",
    "Thief_Bug": "Small", "Thief_Bug_Egg": "Small",
    "Savage_Babe": "Medium", "Savage": "Medium",
    "Familiar": "Small", "Orc_Warrior": "Medium", "Orc_Archer": "Medium",
    "Orc_Skeleton": "Medium", "Orc_Zombie": "Medium",
    "Skeleton": "Medium", "Skeleton_Archer": "Medium",
    "Zombie": "Medium", "Ghoul": "Medium",
    "Mummy": "Medium", "Mummy_": "Medium",
    "Wolf": "Medium", "High_Orc": "Large",
    "Anacondaq": "Medium", "Snake": "Small",
    "Spore": "Small", "Mushroom": "Small",
    "Elder_Willow": "Large", "Willow": "Medium",
    "Creamy": "Small", "Dustiness": "Small",
    "Metaller": "Small", "Plankton": "Small",
    "Marina": "Small", "Kukre": "Small",
    "Vadon": "Small", "Hydra": "Small",
    "Pirate_Skeleton": "Medium", "Coco": "Small",
    "Deniro": "Small", "Peco_Peco": "Large",
    "Peco_Peco_Egg": "Small", "Smokie": "Small",
    "Yoyo": "Small", "Steel_Chonchon": "Small",
    "Hunter_Fly": "Medium", "Mantis": "Medium",
    "Muka": "Medium", "Rocker": "Medium",
    "Stainer": "Small", "Worm_Tail": "Small",
    "Scorpion": "Small", "Swordfish": "Medium",
    "Caramel": "Small", "Savage_Bebe": "Small",
    "Golem": "Large", "Beetle_King": "Medium",
    "Myst_Case": "Medium", "Nightmare": "Medium",
    "Punk": "Small", "Bongun": "Medium",
    "Munak": "Medium", "Nine_Tail": "Medium",
    "Sohee": "Small", "Dokebi": "Small",
    "Deviruchi": "Small", "Isis": "Medium",
    "Petite": "Medium", "Petite_": "Medium",
    "Gargoyle": "Medium", "Rideword": "Small",
    "Neraid": "Small", "Phendark": "Small",
    "Strouf": "Large", "Kraken": "Large",
    "Maya": "Large", "Maya_Pupa": "Large",
    "Phreeoni": "Large", "Moonlight": "Large",
    "Osiris": "Large", "Baphomet": "Large",
    "Dracula": "Large", "Doppelganger": "Large",
    "Golden_Thief_Bug": "Large", "Eddga": "Large",
    "Orc_Hero": "Large", "Orc_Lord": "Large",
    "Mistress": "Medium", "Stormy_Knight": "Large",
    "Lord_of_Death": "Large", "Turtle_General": "Large",
    "Atroce": "Large", "Kiel": "Medium",
    "Valkyrie": "Large", "Randgris": "Large",
    "Gloom": "Large", "Ktullanux": "Large",
    "Ifrit": "Large", "Beelzebub": "Large",
    "Tao_Gunka": "Large", "RSX_0806": "Large",
    "Detard": "Large", "Egnigem": "Medium",
    "Biolab_Monster": "Medium",
}

# ── Monster race mapping ──
MONSTER_RACES: dict[str, str] = {
    "Poring": "Brute", "Drops": "Brute", "Poporing": "Brute",
    "Lunatic": "Brute", "Picky": "Brute", "Picky_": "Brute",
    "Fabre": "Insect", "Chonchon": "Insect", "Hornet": "Insect",
    "Thief_Bug": "Insect", "Thief_Bug_Egg": "Formless",
    "Savage_Babe": "Brute", "Savage": "Brute",
    "Familiar": "Brute", "Orc_Warrior": "DemiHuman", "Orc_Archer": "DemiHuman",
    "Orc_Skeleton": "Undead", "Orc_Zombie": "Undead",
    "Skeleton": "Undead", "Skeleton_Archer": "Undead",
    "Zombie": "Undead", "Ghoul": "Undead",
    "Mummy": "Undead", "Mummy_": "Undead",
    "Wolf": "Brute", "High_Orc": "DemiHuman",
    "Anacondaq": "Brute", "Snake": "Brute",
    "Spore": "Plant", "Mushroom": "Plant",
    "Elder_Willow": "Plant", "Willow": "Plant",
    "Creamy": "Insect", "Dustiness": "Insect",
    "Metaller": "Formless", "Plankton": "Fish",
    "Marina": "Fish", "Kukre": "Fish",
    "Vadon": "Fish", "Hydra": "Plant",
    "Pirate_Skeleton": "Undead", "Coco": "Brute",
    "Deniro": "Insect", "Peco_Peco": "Brute",
    "Peco_Peco_Egg": "Brute", "Smokie": "Brute",
    "Yoyo": "Brute", "Steel_Chonchon": "Insect",
    "Hunter_Fly": "Insect", "Mantis": "Insect",
    "Muka": "Plant", "Rocker": "Brute",
    "Stainer": "Insect", "Worm_Tail": "Insect",
    "Scorpion": "Insect", "Swordfish": "Fish",
    "Caramel": "Brute", "Savage_Bebe": "Brute",
    "Golem": "Formless", "Beetle_King": "Insect",
    "Myst_Case": "Formless", "Nightmare": "Demon",
    "Punk": "Brute", "Bongun": "Undead",
    "Munak": "Undead", "Nine_Tail": "Brute",
    "Sohee": "Demon", "Dokebi": "Demon",
    "Deviruchi": "Demon", "Isis": "Demon",
    "Petite": "Dragon", "Petite_": "Dragon",
    "Gargoyle": "Demon", "Rideword": "Formless",
    "Neraid": "Demon", "Phendark": "Demon",
    "Strouf": "Fish", "Kraken": "Fish",
    "Maya": "Insect", "Maya_Pupa": "Insect",
    "Phreeoni": "Brute", "Moonlight": "Demon",
    "Osiris": "Undead", "Baphomet": "Demon",
    "Dracula": "Demon", "Doppelganger": "Demon",
    "Golden_Thief_Bug": "Insect", "Eddga": "Brute",
    "Orc_Hero": "DemiHuman", "Orc_Lord": "DemiHuman",
    "Mistress": "Insect", "Stormy_Knight": "Formless",
    "Lord_of_Death": "Demon", "Turtle_General": "Brute",
    "Atroce": "Brute", "Kiel": "DemiHuman",
    "Valkyrie": "Angel", "Randgris": "Angel",
    "Gloom": "Demon", "Ktullanux": "Brute",
    "Ifrit": "Formless", "Beelzebub": "Demon",
    "Tao_Gunka": "Brute", "RSX_0806": "Formless",
    "Detard": "Demon", "Egnigem": "Demon",
    "Biolab_Monster": "DemiHuman",
}

# ── Monster DEF/MDEF data (from rAthena mob_db) ──
MONSTER_DEF: dict[str, dict[str, int]] = {
    "Poring": {"def": 2, "mdef": 5, "vit": 1, "int": 0},
    "Drops": {"def": 2, "mdef": 5, "vit": 1, "int": 0},
    "Poporing": {"def": 10, "mdef": 15, "vit": 10, "int": 5},
    "Lunatic": {"def": 0, "mdef": 0, "vit": 1, "int": 0},
    "Picky": {"def": 2, "mdef": 0, "vit": 5, "int": 0},
    "Fabre": {"def": 2, "mdef": 0, "vit": 1, "int": 0},
    "Chonchon": {"def": 5, "mdef": 10, "vit": 5, "int": 5},
    "Hornet": {"def": 5, "mdef": 10, "vit": 5, "int": 5},
    "Thief_Bug": {"def": 10, "mdef": 5, "vit": 10, "int": 0},
    "Savage_Babe": {"def": 5, "mdef": 0, "vit": 5, "int": 0},
    "Savage": {"def": 20, "mdef": 5, "vit": 20, "int": 5},
    "Familiar": {"def": 5, "mdef": 10, "vit": 5, "int": 5},
    "Orc_Warrior": {"def": 20, "mdef": 0, "vit": 20, "int": 0},
    "Orc_Archer": {"def": 10, "mdef": 5, "vit": 10, "int": 5},
    "Orc_Skeleton": {"def": 15, "mdef": 10, "vit": 15, "int": 0},
    "Orc_Zombie": {"def": 20, "mdef": 5, "vit": 20, "int": 0},
    "Skeleton": {"def": 15, "mdef": 10, "vit": 15, "int": 0},
    "Zombie": {"def": 10, "mdef": 15, "vit": 10, "int": 5},
    "Wolf": {"def": 10, "mdef": 0, "vit": 10, "int": 0},
    "High_Orc": {"def": 30, "mdef": 5, "vit": 30, "int": 5},
    "Golem": {"def": 40, "mdef": 20, "vit": 40, "int": 10},
    "Myst_Case": {"def": 0, "mdef": 30, "vit": 1, "int": 30},
    "Nightmare": {"def": 5, "mdef": 20, "vit": 5, "int": 20},
    "Baphomet": {"def": 40, "mdef": 30, "vit": 50, "int": 30},
    "Osiris": {"def": 30, "mdef": 40, "vit": 40, "int": 40},
    "Orc_Hero": {"def": 40, "mdef": 20, "vit": 50, "int": 20},
    "Orc_Lord": {"def": 50, "mdef": 25, "vit": 60, "int": 25},
    "Mistress": {"def": 20, "mdef": 40, "vit": 20, "int": 50},
    "Stormy_Knight": {"def": 30, "mdef": 30, "vit": 30, "int": 30},
    "Dracula": {"def": 25, "mdef": 35, "vit": 30, "int": 40},
    "Doppelganger": {"def": 35, "mdef": 25, "vit": 40, "int": 30},
    "Moonlight": {"def": 15, "mdef": 30, "vit": 20, "int": 35},
    "Phreeoni": {"def": 25, "mdef": 20, "vit": 30, "int": 20},
    "Maya": {"def": 35, "mdef": 30, "vit": 40, "int": 30},
    "Eddga": {"def": 30, "mdef": 20, "vit": 35, "int": 20},
    "Golden_Thief_Bug": {"def": 40, "mdef": 35, "vit": 45, "int": 35},
    "Lord_of_Death": {"def": 45, "mdef": 40, "vit": 50, "int": 40},
    "Turtle_General": {"def": 35, "mdef": 30, "vit": 40, "int": 30},
    "Atroce": {"def": 40, "mdef": 25, "vit": 45, "int": 25},
    "Kiel": {"def": 30, "mdef": 35, "vit": 35, "int": 40},
    "Valkyrie": {"def": 40, "mdef": 50, "vit": 45, "int": 50},
    "Randgris": {"def": 45, "mdef": 55, "vit": 50, "int": 55},
    "Gloom": {"def": 35, "mdef": 40, "vit": 40, "int": 45},
    "Ktullanux": {"def": 40, "mdef": 35, "vit": 45, "int": 40},
    "Ifrit": {"def": 50, "mdef": 40, "vit": 55, "int": 45},
    "Beelzebub": {"def": 55, "mdef": 50, "vit": 60, "int": 50},
    "Tao_Gunka": {"def": 30, "mdef": 20, "vit": 35, "int": 20},
    "RSX_0806": {"def": 45, "mdef": 30, "vit": 50, "int": 30},
    "Detard": {"def": 40, "mdef": 45, "vit": 45, "int": 50},
    "Egnigem": {"def": 35, "mdef": 30, "vit": 40, "int": 35},
    "Biolab_Monster": {"def": 25, "mdef": 20, "vit": 25, "int": 20},
}


def get_element_multiplier(attacker_element: str, defender_element: str) -> float:
    """Get damage multiplier from element chart."""
    elements = ["Neutral", "Water", "Earth", "Fire", "Wind", "Poison", "Holy", "Dark", "Ghost", "Undead"]
    try:
        a_idx = elements.index(attacker_element)
        d_idx = elements.index(defender_element)
        return ELEMENT_CHART[a_idx][d_idx]
    except (ValueError, IndexError):
        return 1.0


def get_size_multiplier(weapon_type: str, monster_size: str) -> float:
    """Get damage multiplier from size chart."""
    size_idx = {"Small": 0, "Medium": 1, "Large": 2}
    wtype = WEAPON_TYPES.get(weapon_type, weapon_type)
    if wtype in SIZE_CHART:
        sidx = size_idx.get(monster_size, 1)
        return SIZE_CHART[wtype][sidx]
    return 1.0


def get_race_multiplier(weapon_type: str, monster_race: str) -> float:
    """Get race-based damage modifier."""
    wtype = WEAPON_TYPES.get(weapon_type, weapon_type)
    if wtype in RACE_CHART:
        return RACE_CHART[wtype].get(monster_race, 1.0)
    return 1.0


def get_level_penalty(attacker_level: int, monster_level: int) -> float:
    """Level-based damage penalty.
    If attacker is >25 levels above monster, damage is reduced.
    If attacker is >25 levels below monster, damage is reduced.
    """
    diff = attacker_level - monster_level
    if diff > 25:
        # 25+ levels above: 1% damage per level over 25
        return max(0.01, 1.0 - (diff - 25) * 0.01)
    if diff < -25:
        # 25+ levels below: 1% damage per level under 25
        return max(0.01, 1.0 - (abs(diff) - 25) * 0.01)
    return 1.0


def calculate_hard_def(vit: int, soft_def: int = 0) -> float:
    """Hard DEF from VIT: VIT * 0.5 + soft_def * 0.5 (rough approximation).
    Actual formula: hard_def = VIT * 0.5 + soft_def * 0.5
    """
    return vit * 0.5 + soft_def * 0.5


def calculate_soft_def(def_stat: int, refinement_level: int = 0) -> float:
    """Soft DEF from equipment + refinement.
    Each refine adds: 0.7 * (refine_level + 1) for armor, 0.5 * (refine_level + 1) for weapons.
    """
    refine_bonus = refinement_level * 0.7
    return def_stat + refine_bonus


def calculate_mdef(mdef_stat: int, int_stat: int) -> float:
    """MDEF formula: (MDEF * 0.5 + MDEF * 0.5 * INT/100) with hard cap at 99%.
    """
    raw = mdef_stat * 0.5 + mdef_stat * 0.5 * (int_stat / 100.0)
    return min(99.0, raw)


def calculate_damage(
    raw_damage: float,
    attacker_level: int = 1,
    monster_name: str = "",
    weapon_type: str = "Dagger",
    attacker_element: str = "Neutral",
    is_physical: bool = True,
    monster_def: int = 0,
    monster_mdef: int = 0,
    monster_vit: int = 0,
    monster_int: int = 0,
    monster_level: int = 1,
    monster_size: str = "Medium",
    monster_race: str = "Brute",
    monster_element: str = "Neutral",
    card_bonus_vs_race: float = 1.0,
    card_bonus_vs_size: float = 1.0,
    card_bonus_vs_element: float = 1.0,
    refinement_level: int = 0,
) -> float:
    """Full RO damage calculation with all modifiers.

    Returns actual damage after all reductions and bonuses.
    """
    # 1. Element multiplier
    element_mult = get_element_multiplier(attacker_element, monster_element)

    # 2. Size multiplier
    size_mult = get_size_multiplier(weapon_type, monster_size)

    # 3. Race multiplier
    race_mult = get_race_multiplier(weapon_type, monster_race)

    # 4. Card bonuses
    card_mult = card_bonus_vs_race * card_bonus_vs_size * card_bonus_vs_element

    # 5. Level penalty
    level_pen = get_level_penalty(attacker_level, monster_level)

    # 6. Apply multipliers to raw damage
    modified_damage = raw_damage * element_mult * size_mult * race_mult * card_mult * level_pen

    # 7. Apply DEF/MDEF reduction
    if is_physical:
        hard_def = calculate_hard_def(monster_vit, monster_def)
        soft_def = calculate_soft_def(monster_def, refinement_level)
        total_def = hard_def + soft_def
        reduction = total_def / (total_def + 100)
        return max(1, int(modified_damage * (1.0 - reduction)))
    else:
        mdef_val = calculate_mdef(monster_mdef, monster_int)
        reduction = mdef_val / 100.0
        return max(1, int(modified_damage * (1.0 - reduction)))


def calculate_cast_time(skill_name: str, dex: int = 0, is_dual: bool = False) -> float:
    """Calculate actual cast time in seconds.
    Variable cast = base_variable_cast * (1 - DEX/150)
    Fixed cast is not reduced by DEX.
    Total = variable + fixed.
    """
    skill_info = SKILL_DATA.get(skill_name)
    if skill_info is None:
        return 0.0
    var_cast_ms, fixed_cast_ms, _, _, _ = skill_info

    # Variable cast reduction from DEX
    dex_factor = max(0.0, 1.0 - dex / 150.0)
    var_cast_actual = var_cast_ms * dex_factor

    total_ms = var_cast_actual + fixed_cast_ms
    return total_ms / 1000.0


def calculate_after_cast_delay(skill_name: str, dex: int = 0) -> float:
    """Calculate after-cast delay in seconds.
    Delay = base_delay * (1 - DEX/150)
    """
    skill_info = SKILL_DATA.get(skill_name)
    if skill_info is None:
        return 0.0
    _, _, delay_ms, _, _ = skill_info
    dex_factor = max(0.0, 1.0 - dex / 150.0)
    return (delay_ms * dex_factor) / 1000.0


def calculate_aspd_interval(aspd: int) -> float:
    """Calculate attack interval in seconds from ASPD.
    ASPD ranges from 100 (slowest) to 190 (fastest).
    Interval = (200 - ASPD) * 0.02 seconds
    """
    return (200 - aspd) * 0.02


def get_skill_cooldown(skill_name: str) -> float:
    """Get skill cooldown in seconds from skill data."""
    skill_info = SKILL_DATA.get(skill_name)
    if skill_info is None:
        return 0.0
    _, _, _, cd_ms, _ = skill_info
    return cd_ms / 1000.0


def get_skill_range(skill_name: str) -> int:
    """Get skill range in cells."""
    skill_info = SKILL_DATA.get(skill_name)
    if skill_info is None:
        return 1
    _, _, _, _, range_cells = skill_info
    return range_cells


def get_skill_element(skill_name: str) -> str:
    """Get the element of a skill."""
    return SKILL_ELEMENTS.get(skill_name, "Neutral")


def get_monster_element(monster_name: str) -> str:
    """Get the element of a monster."""
    return RACE_ELEMENTS.get(monster_name, "Neutral")


def get_monster_size(monster_name: str) -> str:
    """Get the size of a monster."""
    return MONSTER_SIZES.get(monster_name, "Medium")


def get_monster_race(monster_name: str) -> str:
    """Get the race of a monster."""
    return MONSTER_RACES.get(monster_name, "Formless")


def get_monster_def_data(monster_name: str) -> dict[str, int]:
    """Get DEF/MDEF/VIT/INT data for a monster."""
    return MONSTER_DEF.get(monster_name, {"def": 0, "mdef": 0, "vit": 0, "int": 0})


def estimate_hits_to_kill(
    raw_damage_per_hit: float,
    monster_hp: int,
    monster_name: str = "",
    weapon_type: str = "Dagger",
    attacker_element: str = "Neutral",
    attacker_level: int = 1,
    is_physical: bool = True,
) -> int:
    """Estimate how many hits to kill a monster with full damage calculation."""
    def_data = get_monster_def_data(monster_name)
    actual_damage = calculate_damage(
        raw_damage=raw_damage_per_hit,
        attacker_level=attacker_level,
        monster_name=monster_name,
        weapon_type=weapon_type,
        attacker_element=attacker_element,
        is_physical=is_physical,
        monster_def=def_data.get("def", 0),
        monster_mdef=def_data.get("mdef", 0),
        monster_vit=def_data.get("vit", 0),
        monster_int=def_data.get("int", 0),
        monster_element=get_monster_element(monster_name),
        monster_size=get_monster_size(monster_name),
        monster_race=get_monster_race(monster_name),
    )
    if actual_damage <= 0:
        return 999
    return max(1, math.ceil(monster_hp / actual_damage))


class SkillCooldownTracker:
    """Track per-skill cooldowns using RO-accurate cooldown data.
    Also tracks cast time and after-cast delay for proper combat timing.
    """

    def __init__(self):
        self._last_used: dict[str, datetime.datetime] = {}
        self._last_cast_start: dict[str, datetime.datetime] = {}
        self._dex: int = 0

    def set_dex(self, dex: int) -> None:
        self._dex = dex

    def set_cooldown(self, skill_name: str, cooldown_seconds: float) -> None:
        """Override default cooldown from skill data."""
        self._cooldowns[skill_name] = cooldown_seconds

    def record_use(self, skill_name: str) -> None:
        """Record that a skill was used (for cooldown tracking)."""
        self._last_used[skill_name] = datetime.datetime.now(datetime.timezone.utc)

    def record_cast_start(self, skill_name: str) -> None:
        """Record when a skill cast started (for cast time tracking)."""
        self._last_cast_start[skill_name] = datetime.datetime.now(datetime.timezone.utc)

    def is_available(self, skill_name: str) -> bool:
        """Check if a skill is available (cooldown expired)."""
        if skill_name not in self._last_used:
            return True
        cd = get_skill_cooldown(skill_name)
        if cd <= 0:
            return True
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - self._last_used[skill_name]).total_seconds()
        return elapsed >= cd

    def seconds_until_available(self, skill_name: str) -> float:
        """Seconds until skill is available."""
        if skill_name not in self._last_used:
            return 0.0
        cd = get_skill_cooldown(skill_name)
        if cd <= 0:
            return 0.0
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - self._last_used[skill_name]).total_seconds()
        return max(0.0, cd - elapsed)

    def cast_time_remaining(self, skill_name: str) -> float:
        """Seconds remaining in current cast."""
        if skill_name not in self._last_cast_start:
            return 0.0
        cast_time = calculate_cast_time(skill_name, self._dex)
        if cast_time <= 0:
            return 0.0
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - self._last_cast_start[skill_name]).total_seconds()
        return max(0.0, cast_time - elapsed)

    def is_casting(self, skill_name: str) -> bool:
        """Check if a skill is currently being cast."""
        return self.cast_time_remaining(skill_name) > 0.0

    def total_action_time(self, skill_name: str) -> float:
        """Total time from cast start to being able to act again.
        = cast_time + after_cast_delay
        """
        cast_time = calculate_cast_time(skill_name, self._dex)
        delay = calculate_after_cast_delay(skill_name, self._dex)
        return cast_time + delay
